"""Independent Earth MERRA-2 training entry point for the official DLinear model.

This script deliberately does not touch any Mars data path: it reads a verified
daily or three-hour release through the dataset registry, builds its dataset
profile's windows with training-only normalization, trains the official
DLinear, evaluates validation and test in DU and writes one versioned checkpoint.

The parent process passes everything through ``ASTRAATMOS_EARTH_TRAINING_SPEC``
rather than the command line, so dataset identity, the release fingerprint and
the snapshot never become CLI arguments. Only ``--output_path`` is parsed here.

Progress output keeps the existing parent-side format::

    Epoch 1/10 Batch 1/45 Loss=0.123456
    Epoch 1/10 Loss=0.110000 Val Loss=0.120000
    Earth best epoch: 1
    Earth metrics unit: DU
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_registry import DatasetRegistry  # noqa: E402
from services.dataset_identity import DatasetRequestError, is_earth_three_hourly_dataset_id  # noqa: E402
from services.earth_dataset import (  # noqa: E402
    EarthOzoneWindows, EarthThreeHourlyWindows, build_threehour_training_cache,
    fit_threehour_normalization,
    threehour_global_cell_area_weights,
)
from training_backbones.earth_3hourly_uploaded_contract import (  # noqa: E402
    FULL_GRID_CONTRACT_SCHEMA, contract_profile,
)
from services.earth_training_artifact import (  # noqa: E402
    CELL_AREA_WEIGHTED_AGGREGATION,
    GRID_UNIFORM_AGGREGATION,
    EarthArtifactError,
    ErrorAccumulator,
    assert_release_matches_binding,
    build_checkpoint_payload,
    build_earth_model_from_checkpoint,
    build_metrics_block,
    load_earth_training_artifact,
    normalization_from_release,
    save_earth_artifact_atomic,
)
from services.earth_training_contract import (  # noqa: E402
    EARTH_HORIZON,
    EARTH_TRAINING_SPEC_SCHEMA,
    EARTH_WINDOW,
    normalize_earth_training_hyperparameters,
    earth_spec_metrics_contract,
    EARTH_METRICS_SCHEMA_KEY,
    EARTH_METRICS_AGGREGATION_KEY,
    earth_training_profile,
    split_window_counts,
)
from services.earth_model_source import (  # noqa: E402
    MODEL_SOURCE_OFFICIAL,
    MODEL_SOURCE_UPLOADED,
    EarthModelBuildError,
    ModelSourcePlan,
    build_earth_model_for_plan,
    earth_forward_for_model,
)

SPEC_ENV_VAR = "ASTRAATMOS_EARTH_TRAINING_SPEC"
PROGRESS_EVERY_BATCHES = 20
GRID_HORIZON_INDEX = 1
THREE_HOUR_TILE_SHAPE = (24, 48)


class EarthTrainingError(RuntimeError):
    """A training failure that must not publish a completed artifact."""


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Pick the execution device; tests may pin it explicitly."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def configure_cuda_acceleration(device: torch.device) -> dict[str, Any]:
    """Enable safe fixed-shape CUDA acceleration without changing the model."""
    if device.type != "cuda":
        return {"amp": False, "amp_dtype": None, "tf32": False, "cudnn_benchmark": False}
    # TF32 affects matrix/conv math while preserving the model and checkpoint
    # structure. It is a good fit for the Earth training workload on Ampere+.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass
    enabled = os.getenv("ASTRAATMOS_EARTH_AMP", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }
    dtype = "bfloat16" if enabled and torch.cuda.is_bf16_supported() else "float16" if enabled else None
    return {"amp": enabled, "amp_dtype": dtype, "tf32": True, "cudnn_benchmark": True}


def autocast_context(acceleration: dict[str, Any]):
    if not acceleration.get("amp"):
        return nullcontext()
    dtype = torch.bfloat16 if acceleration.get("amp_dtype") == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(acceleration: dict[str, Any]):
    enabled = bool(acceleration.get("amp")) and acceleration.get("amp_dtype") != "bfloat16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError, RuntimeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _cuda_math_state():
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision(),
            torch.backends.cudnn.deterministic)


def _restore_cuda_math(state):
    matmul_tf32, cudnn_tf32, benchmark, precision, deterministic = state
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


class _PrefetchLoader:
    """Overlap one CPU batch with GPU compute without pickling protected memmaps."""

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        sentinel = object()
        iterator = iter(self.loader)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="earth-batch") as executor:
            future = executor.submit(next, iterator, sentinel)
            while True:
                batch = future.result()
                if batch is sentinel:
                    break
                future = executor.submit(next, iterator, sentinel)
                yield batch


def _prefetch_loader(loader, source, device):
    enabled = os.getenv("ASTRAATMOS_EARTH_PREFETCH", "1").strip().lower() not in {"0", "false", "no", "off"}
    # Bound the extra resident batch to 64 MiB. Large native full-grid windows
    # retain synchronous loading rather than staging another multi-GiB batch.
    shape = (len(getattr(source, "lat", ())), len(getattr(source, "lon", ())))
    if isinstance(loader.dataset, _SpatialTileDataset):
        shape = THREE_HOUR_TILE_SHAPE
    elements = (source.window * len(source.input_channels) + source.horizon) * np.prod(shape)
    if (enabled and device is not None and device.type == "cuda"
            and getattr(source, "_normalized_map", None) is not None
            and elements * 4 * loader.batch_size <= 64 * 1024**2):
        return _PrefetchLoader(loader)
    return loader


def _uses_full_grid(spec: dict) -> bool:
    reference = spec.get('uploaded_model') or {}
    full_grid = reference.get('contract_schema') == FULL_GRID_CONTRACT_SCHEMA
    if spec['dataset_binding']['dataset_id'] == 'earth_merra2_3hourly_5deg_v1':
        return True
    if full_grid and (not is_earth_three_hourly_dataset_id(spec['dataset_binding']['dataset_id'])
                      or spec['hyperparameters'].get('model_source') != MODEL_SOURCE_UPLOADED):
        raise EarthTrainingError('The full-grid contract requires an uploaded three-hour Earth model')
    return full_grid


def _build_loaders(
    spec: dict,
    registry: DatasetRegistry,
    batch_size: int,
    seed: int,
    cache_root: Optional[Path] = None,
    device: Optional[torch.device] = None,
):
    """Build the three split datasets sharing one training-fitted normalization."""
    binding = spec["dataset_binding"]
    metrics_contract = earth_spec_metrics_contract(spec)
    full_grid = _uses_full_grid(spec)
    max_batch_size = contract_profile(FULL_GRID_CONTRACT_SCHEMA)['max_batch_size']
    if full_grid and (type(batch_size) is not int or not 1 <= batch_size <= max_batch_size):
        raise EarthTrainingError(f'Full-grid Earth training batch_size must be an integer between 1 and {max_batch_size}')
    preparation_started = time.perf_counter()
    print('Earth preparation: loading verified dataset and checking task binding', flush=True)
    hyperparameters = normalize_earth_training_hyperparameters(
        spec["hyperparameters"], dataset_id=binding["dataset_id"]
    )
    profile = earth_training_profile(binding["dataset_id"], hyperparameters)
    validation_started = time.perf_counter()
    release = registry.get_earth_snapshot(binding["dataset_id"])
    assert_release_matches_binding(release, binding)
    topography_asset = None
    if spec.get("uploaded_model") is not None:
        from services.earth_derived_inputs import topography_for_reference
        topography_asset = topography_for_reference(
            spec["uploaded_model"], release, spec.get("topography"))
    validation_seconds = time.perf_counter() - validation_started
    print(f"Earth preparation: data validation {'hit' if release.verification_cached else 'verified'} "
          f"elapsed={validation_seconds:.3f}s", flush=True)

    selected = hyperparameters["selected_channels"]
    # Fit on the frozen task train interval (manifest for legacy tasks), then
    # independently build windows inside each partition.
    task_split = spec.get("task_split")
    if "task_split" in spec:
        from services.earth_task_split import validate_earth_task_split
        validate_earth_task_split(task_split, release.dates, profile["window"], profile["horizon"], hyperparameters)
    window_type = EarthThreeHourlyWindows if is_earth_three_hourly_dataset_id(binding["dataset_id"]) else EarthOzoneWindows
    normalization_started = time.perf_counter()
    if window_type is EarthThreeHourlyWindows:
        from config import EARTH_TRAINING_CACHE_DIR
        effective_cache_root = cache_root if cache_root is not None else EARTH_TRAINING_CACHE_DIR
        normalization = fit_threehour_normalization(
            release, ["TO3", *selected], task_split=task_split,
            cache_root=effective_cache_root,
            progress=lambda message: print(message, flush=True),
        )
    else:
        normalization = normalization_from_release(release, ["TO3", *selected], task_split=task_split)
    normalization_seconds = time.perf_counter() - normalization_started
    print(f'Earth preparation: normalization stage elapsed={normalization_seconds:.3f}s', flush=True)
    splits = {
        name: window_type.from_release(
            release,
            split=name,
            window=profile["window"],
            horizon=profile["horizon"],
            selected_channels=selected,
            normalization=normalization,
            **({"task_split": task_split} if window_type is EarthThreeHourlyWindows else {}),
        )
        for name in ("train", "validation", "test")
    }
    counts = {name: len(dataset) for name, dataset in splits.items()}
    cache_path = None
    cache_started = time.perf_counter()
    if window_type is EarthThreeHourlyWindows:
        cache_path = build_threehour_training_cache(
            release, ["TO3", *selected], normalization,
            effective_cache_root,
            full_grid=full_grid, progress=lambda message: print(message, flush=True),
        )
        for dataset in splits.values():
            dataset.use_training_cache(cache_path, progress=lambda message: print(message, flush=True))
    cache_seconds = time.perf_counter() - cache_started
    if window_type is EarthThreeHourlyWindows:
        print(f'Earth preparation: cache stage (including integrity/attach) elapsed={cache_seconds:.3f}s', flush=True)
    split_ranges = {
        name: {
            "date_start": (np.datetime_as_string(dataset.dates[0], unit="s") + "Z"
                           if window_type is EarthThreeHourlyWindows else str(dataset.dates[0])),
            "date_end": (np.datetime_as_string(dataset.dates[-1], unit="s") + "Z"
                         if window_type is EarthThreeHourlyWindows else str(dataset.dates[-1])),
            "window_count": int(len(dataset)),
        }
        for name, dataset in splits.items()
    }
    if task_split is not None:
        split_ranges = task_split["ranges"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    area_weights = (threehour_global_cell_area_weights(release)
                    if metrics_contract is not None and metrics_contract["aggregation"] == CELL_AREA_WEIGHTED_AGGREGATION
                    else None)
    wrapper = _SpatialTileDataset if window_type is EarthThreeHourlyWindows and not full_grid else _ArrayDataset
    pin_memory = device is not None and device.type == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        wrapper(splits["train"], area_weights=area_weights, topography_asset=topography_asset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        wrapper(splits["validation"], area_weights=area_weights, topography_asset=topography_asset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        wrapper(splits["test"], area_weights=area_weights, topography_asset=topography_asset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    train_loader = _prefetch_loader(train_loader, splits["train"], device)
    validation_loader = _prefetch_loader(validation_loader, splits["validation"], device)
    test_loader = _prefetch_loader(test_loader, splits["test"], device)
    spatial_mode = 'spatial_tile' if wrapper is _SpatialTileDataset else 'full_grid'
    print(
        f'Earth preparation complete: spatial mode={spatial_mode}, '
        f'train samples={len(train_loader.dataset)}, batches={len(train_loader)}, '
        f'elapsed={time.perf_counter() - preparation_started:.3f}s',
        flush=True,
    )
    return {
        "release": release,
        "profile": profile,
        "normalization": normalization,
        "input_channel_order": ["TO3", *selected],
        "splits": splits,
        "counts": counts,
        "split_ranges": split_ranges,
        "task_split": task_split,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
        "full_grid": full_grid,
        "area_weights": area_weights,
        "cache_path": cache_path,
        "topography_asset": topography_asset,
        "preparation_timings": {'validation_seconds': validation_seconds,
                                'normalization_seconds': normalization_seconds,
                                'cache_seconds': cache_seconds,
                                'total_seconds': time.perf_counter() - preparation_started},
    }


class _ArrayDataset(torch.utils.data.Dataset):
    """Convert only the requested window; never expand all overlapping windows.

    Samples are returned as a mapping. The seasonal coordinate is always produced
    when the source can derive it; whether a model *receives* it is decided by the
    model's own declaration, never by the loader. A mapping also keeps the
    ``[B, window]`` coordinate distinguishable from the ``[H, W]`` area weights,
    which a positional tuple cannot do.
    """

    def __init__(self, source: EarthOzoneWindows, area_weights=None, topography_asset=None):
        self.source = source
        self.area_weights = area_weights
        self.topography_asset = topography_asset
        # Area weights are normalized over the complete global grid. A full-grid
        # sample therefore represents one unit of spatial mass. Tile datasets
        # override this with one global tile's share below.
        self.area_weight_denominator = 1.0

    def __len__(self) -> int:
        return len(self.source)

    def _sample(self, index, inputs, targets, spatial_slices=None,
                *, longitude=None, topography=None):
        input_tensor = inputs if isinstance(inputs, torch.Tensor) else torch.from_numpy(inputs)
        target_tensor = targets if isinstance(targets, torch.Tensor) else torch.from_numpy(targets)
        sample = {"inputs": input_tensor, "targets": target_tensor}
        if topography is not None:
            sample["topography"] = topography
        elif self.topography_asset is not None:
            from services.earth_derived_inputs import topography_kwargs
            sample["topography"] = topography_kwargs(
                self.topography_asset, input_tensor.unsqueeze(0),
                spatial_slices=None if spatial_slices is None else [spatial_slices],
            )["topography"][0]
        longitude = self._longitude(index) if longitude is None else longitude
        if longitude is not None:
            sample["longitude"] = longitude
        if self.area_weights is not None:
            sample["area_weights"] = torch.from_numpy(np.asarray(self.area_weights, dtype='float32'))
        return sample

    def _batch_samples(self, entries):
        """Batch derived-input preparation so each terrain is copied once per batch."""
        if not entries:
            return []
        # Only the shape is needed by terrain preparation. Do not materialize
        # another full history batch before DataLoader's own collation.
        input_tensor = torch.from_numpy(entries[0][1][0]).unsqueeze(0).expand(len(entries), -1, -1, -1, -1)
        terrain_batch = None
        if self.topography_asset is not None:
            from services.earth_derived_inputs import topography_kwargs
            slices = [item[2] for item in entries]
            terrain_batch = topography_kwargs(
                self.topography_asset, input_tensor,
                spatial_slices=None if all(value is None for value in slices) else slices,
            )["topography"]
        samples = []
        for offset, (index, (inputs, targets), spatial_slices) in enumerate(entries):
            samples.append(self._sample(
                index, inputs, targets, spatial_slices,
                longitude=self._longitude(index),
                topography=None if terrain_batch is None else terrain_batch[offset],
            ))
        return samples

    def _longitude(self, index):
        if not hasattr(self.source, 'window_longitude'):
            return None
        return torch.from_numpy(np.asarray(self.source.window_longitude(index), dtype='float32'))

    def __getitem__(self, index):
        inputs, targets = self.source[index]
        return self._sample(index, inputs, targets)

    def __getitems__(self, indices):
        if not hasattr(self.source, 'read_windows'):
            return [self[index] for index in indices]
        requests = [(index, slice(None), slice(None)) for index in indices]
        windows = self.source.read_windows(requests)
        return self._batch_samples([
            (index, window, None) for index, window in zip(indices, windows)
        ])


class _SpatialTileDataset(_ArrayDataset):
    """Read disjoint tiles for the shared, spatially independent DLinear weights.

    A batch is a batch of spatial tiles. Temporal sample counts still refer to
    forecast windows; evaluation visits every cell of every window exactly once.
    """

    def __init__(self, source, area_weights=None, topography_asset=None):
        super().__init__(source, area_weights=area_weights, topography_asset=topography_asset)
        self.tiles = [
            (slice(lat, lat + THREE_HOUR_TILE_SHAPE[0]),
             slice(lon, lon + THREE_HOUR_TILE_SHAPE[1]))
            for lat in range(0, 240, THREE_HOUR_TILE_SHAPE[0])
            for lon in range(0, 480, THREE_HOUR_TILE_SHAPE[1])
        ]
        self.area_weight_denominator = 1.0 / len(self.tiles)

    def __len__(self):
        return len(self.source) * len(self.tiles)

    def __getitem__(self, index):
        window_index, tile_index = divmod(index, len(self.tiles))
        latitude, longitude = self.tiles[tile_index]
        inputs, targets = self.source.read_window(
            window_index, lat_slice=latitude, lon_slice=longitude
        )
        sample = self._sample(window_index, inputs, targets, (latitude, longitude))
        if self.area_weights is not None:
            sample["area_weights"] = torch.from_numpy(
                np.asarray(self.area_weights[latitude, longitude], dtype='float32'))
        return sample

    def __getitems__(self, indices):
        if not hasattr(self.source, 'read_windows'):
            return [self[index] for index in indices]
        requests = []
        for index in indices:
            window, tile = divmod(index, len(self.tiles))
            latitude, longitude = self.tiles[tile]
            requests.append((window, latitude, longitude))
        windows = self.source.read_windows(requests)
        samples = self._batch_samples([
            (window, values, (latitude, longitude))
            for (window, latitude, longitude), values in zip(requests, windows)
        ])
        for sample, (window, latitude, longitude) in zip(samples, requests):
            if self.area_weights is not None:
                sample["area_weights"] = torch.from_numpy(
                    np.asarray(self.area_weights[latitude, longitude], dtype='float32'))
        return samples


def _require_finite_gradients(model):
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.grad is not None]
    if not parameters:
        return
    flags = torch.stack([torch.isfinite(parameter.grad).all() for _, parameter in parameters])
    if not flags.all():
        invalid = flags.detach().cpu().tolist().index(False)
        raise EarthTrainingError(f"Non-finite gradient for {parameters[invalid][0]}")


def _forward_batch(model, inputs, *, model_source, horizon, longitude=None, topography=None):
    # Test doubles from the legacy runner contract expose a boolean marker and
    # implement the tensor call directly; real uploaded models carry a schema id.
    if model_source == MODEL_SOURCE_UPLOADED and getattr(model, "_astraatmos_earth_3hourly_contract", None) is True:
        return model(inputs)
    # The coordinate is only ever passed when one exists, so a model that declares
    # nothing keeps the exact call it had before derived inputs existed.
    extra = {} if longitude is None else {"longitude": longitude}
    if topography is not None:
        extra["topography"] = topography.to(inputs.device)
    return earth_forward_for_model(
        model, inputs, model_source=model_source, horizon=horizon,
        height=inputs.shape[-2], width=inputs.shape[-1], **extra,
    )


def _weighted_mse(output, targets, spatial_weights=None, sample_weight_denominator=None):
    """Return normalized MSE and its denominator for stable streamed logging."""
    squared = torch.square(output.float() - targets.float())
    if spatial_weights is None:
        denominator = torch.as_tensor(squared.numel(), dtype=squared.dtype, device=squared.device)
        return squared.mean(), denominator
    weights = spatial_weights.to(device=output.device, dtype=torch.float32)
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)
    if weights.ndim != 3 or tuple(weights.shape[-2:]) != tuple(output.shape[-2:]):
        raise EarthTrainingError('Area weights must match each batch tile/global grid')
    expanded = weights[:, None, None, :, :].expand_as(output)
    # ``spatial_weights`` are global normalized cell weights. For a tile
    # sample, dividing by its local sum would make every tile contribute
    # equally to SGD. Use the fixed global share of one sample instead.
    if sample_weight_denominator is None:
        denominator = expanded.sum()
    else:
        if not np.isfinite(sample_weight_denominator) or sample_weight_denominator <= 0:
            raise EarthTrainingError('Area-weighted sample denominator must be finite and positive')
        denominator = torch.as_tensor(
            float(output.shape[0] * output.shape[1] * sample_weight_denominator),
            dtype=squared.dtype, device=output.device,
        )
    if not torch.isfinite(denominator) or denominator <= 0:
        raise EarthTrainingError('Area-weighted MSE denominator must be finite and positive')
    return (squared * expanded).sum() / denominator, denominator


def _split_batch(batch):
    """Split a loader batch into ``(inputs, longitude, targets, area_weights)``.

    This runner's own datasets return mappings, so the seasonal coordinate is
    explicit. Positional tuples stay supported for loaders built elsewhere:
    ``(inputs, targets)`` and the legacy ``(inputs, targets, area weights)``.
    A coordinate is never inferred from a tuple position, because an ``[H, W]``
    weight tensor and a small ``[B, window]`` coordinate have the same rank.
    """
    if isinstance(batch, dict):
        for key in ("inputs", "targets"):
            if key not in batch:
                raise EarthTrainingError(f"Earth loader batch is missing {key}")
        return batch["inputs"], batch.get("longitude"), batch["targets"], batch.get("area_weights")
    if len(batch) == 2:
        return batch[0], None, batch[1], None
    if len(batch) == 3:
        return batch[0], None, batch[1], batch[2]
    raise EarthTrainingError(
        'Earth loader batches must be mappings or (inputs, targets[, area weights]) tuples')


def _evaluate_split(
    model: nn.Module,
    loader,
    dataset: EarthOzoneWindows,
    device: torch.device,
    *,
    loss_function=None,
    model_source: str = MODEL_SOURCE_OFFICIAL,
    acceleration: Optional[dict[str, Any]] = None,
) -> tuple[dict, float]:
    """Return DU metrics plus the mean normalized MSE for a whole partition."""
    model.eval()
    dataset_id = (getattr(getattr(dataset, "_release", None), "metadata", {}) or {}).get("dataset_id")
    horizon = getattr(dataset, "horizon", EARTH_HORIZON)
    accumulator = ErrorAccumulator(dataset_id=dataset_id, horizon=horizon)
    total_loss = 0.0
    total_loss_denominator = 0.0
    sample_weight_denominator = getattr(getattr(loader, 'dataset', None),
                                        'area_weight_denominator', None)
    with torch.no_grad():
        for batch in loader:
            inputs, longitude, targets, spatial_weights = _split_batch(batch)
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
            targets = targets.to(device, non_blocking=device.type == 'cuda')
            if longitude is not None:
                longitude = longitude.to(device, non_blocking=device.type == 'cuda')
            with autocast_context(acceleration or {}):
                output = _forward_batch(model, inputs, model_source=model_source, horizon=horizon,
                                        longitude=longitude, topography=(batch.get("topography") if isinstance(batch, dict) else None))
                loss, denominator = _weighted_mse(
                    output, targets, spatial_weights, sample_weight_denominator,
                )
            if not torch.isfinite(output).all():
                raise EarthTrainingError("Non-finite model output during evaluation")
            if not torch.isfinite(loss):
                raise EarthTrainingError("Non-finite evaluation loss")
            total_loss += float(loss.item()) * float(denominator.item())
            total_loss_denominator += float(denominator.item())
            prediction_du = dataset.denormalize_ozone(output[:, :, 0]).detach().cpu().numpy()
            target_du = dataset.denormalize_ozone(targets[:, :, 0]).detach().cpu().numpy()
            weight_array = spatial_weights.detach().cpu().numpy() if spatial_weights is not None else None
            accumulator.update(prediction_du[:, :, None], target_du[:, :, None],
                               spatial_weights=weight_array)
    if total_loss_denominator == 0:
        raise EarthTrainingError("Evaluation loader produced no batches")
    return accumulator.result(), total_loss / total_loss_denominator


def _optimizer_step(model, optimizer, loss, scaler):
    """Skip only recoverable FP16 scaling overflows; other non-finite gradients fail."""
    if not scaler.is_enabled():
        loss.backward()
        _require_finite_gradients(model)
        optimizer.step()
        return True
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    previous_scale = scaler.get_scale()
    # GradScaler checks every unscaled gradient and suppresses optimizer.step
    # on overflow. Reject repeated overflows in the caller instead of publishing
    # an untrained checkpoint, while permitting normal dynamic-scale warmup.
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= previous_scale


def run_training(spec, output_path, registry, device=None, cache_root=None):
    """Keep CUDA math configuration local to this run, including failure paths."""
    baseline_math = _cuda_math_state()
    try:
        return _run_training(spec, output_path, registry, device=device,
                             cache_root=cache_root, baseline_math=baseline_math)
    finally:
        _restore_cuda_math(baseline_math)


def _run_training(
    spec: dict,
    output_path: Path,
    registry: DatasetRegistry,
    device: Optional[str] = None,
    cache_root: Optional[Path] = None,
    baseline_math=None,
) -> dict:
    """Train, evaluate and publish one Earth checkpoint.

    Returns a small result summary. Raises on any failure; the caller treats a
    non-zero exit code as a failed task and no artifact is published.
    """
    if not isinstance(spec, dict):
        raise EarthTrainingError("Training spec must be an object")
    if spec.get("schema") != EARTH_TRAINING_SPEC_SCHEMA:
        raise EarthTrainingError("Unsupported Earth training spec schema")
    task_id = spec.get("task_id")
    if not isinstance(task_id, int) or task_id < 1:
        raise EarthTrainingError("Training spec has no valid task id")
    binding = spec.get("dataset_binding")
    if not isinstance(binding, dict):
        raise EarthTrainingError("Training spec has no dataset binding")
    metrics_contract = earth_spec_metrics_contract(spec)

    hyperparameters = normalize_earth_training_hyperparameters(
        spec["hyperparameters"], dataset_id=binding["dataset_id"]
    )
    profile = earth_training_profile(binding["dataset_id"], hyperparameters)
    if spec.get("training_profile") is not None and spec["training_profile"] != profile:
        raise EarthTrainingError("Training spec profile disagrees with the dataset")
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]
    learning_rate = hyperparameters["learning_rate"]
    seed = hyperparameters["seed"]
    patience = hyperparameters["early_stopping_patience"]
    hidden_layers = hyperparameters["linear_hidden_layers"]

    # The server pinned the model source and, for an uploaded model, its exact
    # verified reference. Nothing here re-reads the user's current upload, so a new
    # version published after this task was created cannot change the trained code.
    raw_source = str(spec.get("hyperparameters", {}).get("model_source") or "official").strip().lower()
    model_source = MODEL_SOURCE_UPLOADED if raw_source == "uploaded" else MODEL_SOURCE_OFFICIAL
    uploaded_reference = spec.get("uploaded_model")
    if model_source == MODEL_SOURCE_UPLOADED and not isinstance(uploaded_reference, dict):
        raise EarthTrainingError(
            "This Earth task was pinned to an uploaded model but the training spec has "
            "no model reference"
        )
    if model_source == MODEL_SOURCE_UPLOADED:
        # The pinned reference carries the parameters that were validated at task
        # creation. Request-level values only fill gaps, so a later request can never
        # retroactively change what a pinned reference already fixed.
        uploaded_reference = dict(uploaded_reference)
        request_params = spec.get("hyperparameters", {}).get("custom_model_params")
        if isinstance(request_params, dict):
            merged = dict(request_params)
            merged.update(uploaded_reference.get("custom_model_params") or {})
            uploaded_reference["custom_model_params"] = merged

    seed_everything(seed)
    resolved_device = resolve_device(device)
    acceleration = configure_cuda_acceleration(resolved_device)
    scaler = make_grad_scaler(acceleration)
    prepared = _build_loaders(spec, registry, batch_size, seed, cache_root=cache_root, device=resolved_device)
    order = prepared["input_channel_order"]
    dataset_by_name = prepared["splits"]
    train_dataset = dataset_by_name["train"]

    plan = ModelSourcePlan(
        model_source=model_source,
        input_channel_order=order,
        linear_hidden_layers=hidden_layers,
        uploaded_model=uploaded_reference if model_source == MODEL_SOURCE_UPLOADED else None,
        dataset_id=binding["dataset_id"],
        window=profile['window'], horizon=profile['horizon'],
    )
    try:
        model, model_build_config, model_warnings = build_earth_model_for_plan(plan)
    except EarthModelBuildError as exc:
        raise EarthTrainingError(str(exc)) from exc
    model = model.to(resolved_device)
    for warning in model_warnings:
        print(f"Earth model warning: {warning}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    area_weighted = metrics_contract is not None and metrics_contract["aggregation"] == CELL_AREA_WEIGHTED_AGGREGATION

    model_label = (
        f"uploaded:{uploaded_reference.get('display_name')}"
        f"@v{uploaded_reference.get('version')}"
        if model_source == MODEL_SOURCE_UPLOADED
        else "official:dlinear"
    )
    print(
        f"Training Device: {resolved_device}; AMP={acceleration['amp']}; "
        f"TF32={acceleration['tf32']}; cuDNN benchmark={acceleration['cudnn_benchmark']}",
        flush=True,
    )
    print(
        f"EarthModel={model_label}, Dataset={binding['dataset_id']}@{binding['dataset_version']}, "
        f"Channels={','.join(order)}, Window={profile['window']}, Horizon={profile['horizon']}",
        flush=True,
    )
    if model_source == MODEL_SOURCE_UPLOADED:
        print(
            f"Earth uploaded model source sha256="
            f"{str(uploaded_reference.get('content_hash'))[:12]}… "
            f"(pinned at task creation)",
            flush=True,
        )
    print(
        "Earth split windows: "
        + ", ".join(f"{name}={count}" for name, count in prepared["counts"].items()),
        flush=True,
    )
    print(f"Earth normalization fitted on: {train_dataset.normalization['fit_date_start']} "
          f"to {train_dataset.normalization['fit_date_end']}", flush=True)

    history = {"train": [], "val": []}
    best_validation_loss: Optional[float] = None
    best_epoch = 0
    best_state: Optional[dict] = None
    epochs_completed = 0
    stopped_early = False
    epochs_without_improvement = 0
    start_time = time.time()
    skipped_amp_steps = 0
    consecutive_overflows = 0
    sample_weight_denominator = getattr(
        getattr(prepared["train_loader"], 'dataset', None),
        'area_weight_denominator', None,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loss_elements = 0
        total_batches = len(prepared["train_loader"])
        optimizer_steps = 0
        for batch_index, batch in enumerate(prepared["train_loader"], start=1):
            inputs, longitude, targets, spatial_weights = _split_batch(batch)
            inputs = inputs.to(resolved_device, non_blocking=resolved_device.type == 'cuda')
            targets = targets.to(resolved_device, non_blocking=resolved_device.type == 'cuda')
            if longitude is not None:
                longitude = longitude.to(resolved_device, non_blocking=resolved_device.type == 'cuda')
            if spatial_weights is not None:
                spatial_weights = spatial_weights.to(resolved_device, non_blocking=resolved_device.type == 'cuda')
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(acceleration):
                output = _forward_batch(model, inputs, model_source=model_source, horizon=profile["horizon"],
                                        longitude=longitude, topography=(batch.get("topography") if isinstance(batch, dict) else None))
                loss, denominator = _weighted_mse(
                    output, targets, spatial_weights, sample_weight_denominator,
                )
            if not torch.isfinite(output).all():
                raise EarthTrainingError("Non-finite model output during training")
            if not torch.isfinite(loss):
                raise EarthTrainingError("Non-finite training loss")
            if not _optimizer_step(model, optimizer, loss, scaler):
                skipped_amp_steps += 1
                consecutive_overflows += 1
                print(f"Earth AMP: skipped overflowing optimizer step; scale={scaler.get_scale():g}", flush=True)
                if consecutive_overflows >= 8:
                    raise EarthTrainingError("Repeated FP16 gradient overflows; disable ASTRAATMOS_EARTH_AMP")
            else:
                consecutive_overflows = 0
                optimizer_steps += 1
            denominator_value = float(denominator.item())
            running_loss += float(loss.item()) * denominator_value
            loss_elements += denominator_value
            if batch_index % PROGRESS_EVERY_BATCHES == 0 or batch_index == total_batches:
                loss_name = "area-weighted normalized MSE" if area_weighted else "normalized MSE"
                print(
                    f"Epoch {epoch}/{epochs} Batch {batch_index}/{total_batches} "
                    f"Loss={loss.item():.6f} ({loss_name})",
                    flush=True,
                )

        if optimizer_steps == 0:
            raise EarthTrainingError("No finite optimizer update was completed in this epoch")
        training_loss = running_loss / max(loss_elements, 1)
        validation_metrics, validation_loss = _evaluate_split(
            model,
            prepared["validation_loader"],
            dataset_by_name["validation"],
            resolved_device,
            model_source=model_source,
            acceleration=acceleration,
        )
        epochs_completed = epoch
        loss_name = "area-weighted normalized MSE" if area_weighted else "normalized MSE"
        print(f"Epoch {epoch}/{epochs} Loss={training_loss:.6f} ({loss_name}) "
              f"Val Loss={validation_loss:.6f} ({loss_name})", flush=True)
        if len(history["train"]) < epoch:
            history["train"].append(training_loss)
            history["val"].append(validation_loss)

        improved = best_validation_loss is None or validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().to("cpu").clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"[Early Stopping] Val loss did not improve for {patience} epochs. "
                    f"Stopped at epoch {epoch}.",
                    flush=True,
                )
                stopped_early = True
                break

    if best_state is None:
        raise EarthTrainingError("Training produced no best weights")
    print(f"Earth best epoch: {best_epoch}", flush=True)

    if baseline_math is not None:
        _restore_cuda_math(baseline_math)
    if resolved_device.type == 'cuda':
        # Final metrics and the strict reload proof must use reproducible conv
        # kernels: even two eval calls on identical CUDA weights may differ.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model.load_state_dict(best_state, strict=True)
    model.to(resolved_device)
    validation_metrics, validation_loss = _evaluate_split(
        model,
        prepared["validation_loader"],
        dataset_by_name["validation"],
        resolved_device,
        model_source=model_source,
    )
    test_metrics, test_loss = _evaluate_split(
        model,
        prepared["test_loader"],
        dataset_by_name["test"],
        resolved_device,
        model_source=model_source,
    )
    print("Earth metrics unit: DU", flush=True)
    print(
        f"Earth validation RMSE={validation_metrics['overall']['rmse']:.6f} "
        f"MAE={validation_metrics['overall']['mae']:.6f}",
        flush=True,
    )
    print(
        f"Earth test RMSE={test_metrics['overall']['rmse']:.6f} "
        f"MAE={test_metrics['overall']['mae']:.6f}",
        flush=True,
    )
    for row in test_metrics["by_lead"]:
        print(
            f"Earth test lead {row.get('lead_hours', row.get('lead_day'))} "
            f"{'hours' if 'lead_hours' in row else 'day'}: RMSE={row['rmse']:.6f} "
            f"MAE={row['mae']:.6f}",
            flush=True,
        )
    for row in test_metrics.get("by_horizon", []):
        print(f"Earth test cumulative {row['horizon_hours']} hours: "
              f"RMSE={row['rmse']:.6f} MAE={row['mae']:.6f}", flush=True)

    metrics = build_metrics_block(
        validation=validation_metrics,
        test=test_metrics,
        validation_window_count=prepared["counts"]["validation"],
        test_window_count=prepared["counts"]["test"],
        dataset_id=binding["dataset_id"],
        aggregation=(CELL_AREA_WEIGHTED_AGGREGATION if area_weighted else GRID_UNIFORM_AGGREGATION),
        horizon=profile['horizon'],
    )
    if metrics_contract is not None and any(metrics.get(key) != value for key, value in metrics_contract.items()):
        raise EarthTrainingError("Generated metrics disagree with the frozen execution contract")
    run = {
        "task_id": task_id,
        "seed": seed,
        "optimizer": "Adam",
        "loss": "normalized_mse_cell_area_weighted" if area_weighted else "normalized_mse_grid_uniform",
        "hyperparameters": hyperparameters,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "best_validation_loss": float(best_validation_loss),
        "final_validation_loss": float(validation_loss),
        "final_test_loss": float(test_loss),
        "stopped_early": bool(stopped_early),
        "run_complete": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": str(resolved_device),
        "acceleration": dict(acceleration),
        "amp_skipped_optimizer_steps": skipped_amp_steps,
        "saved_metrics_precision": "float32",
        "saved_metrics_cudnn_deterministic": resolved_device.type == 'cuda',
        "duration_seconds": round(time.time() - start_time, 3),
    }
    if is_earth_three_hourly_dataset_id(binding["dataset_id"]):
        if prepared["full_grid"]:
            run["memory_strategy"] = "normalized_memmap_full_grid"
            run["spatial_grid_shape"] = list(prepared["profile"]["grid_shape"])
            run["batch_unit"] = "global_time_window"
        else:
            run["memory_strategy"] = "normalized_memmap_spatial_tiles"
            run["spatial_tile_shape"] = list(THREE_HOUR_TILE_SHAPE)
            run["batch_unit"] = "spatial_tile"
        run["missing_value_policy"] = "reject_selected_channel_missing_values"
    uploaded_block = None
    if model_source == MODEL_SOURCE_UPLOADED:
        uploaded_block = {
            "package_id": uploaded_reference.get("package_id"),
            "display_name": uploaded_reference.get("display_name"),
            "version": uploaded_reference.get("version"),
            "content_hash": uploaded_reference.get("content_hash"),
            "source_path": uploaded_reference.get("source_path"),
            "source_text": uploaded_reference.get("source_text"),
            "param_schema": uploaded_reference.get("param_schema") or {},
            "custom_model_params": uploaded_reference.get("custom_model_params") or {},
            "build_config": model_build_config,
            "contract_schema": uploaded_reference.get("contract_schema"),
            "input_channel_order": order,
        }
    payload = build_checkpoint_payload(
        model=model,
        input_channel_order=order,
        linear_hidden_layers=hidden_layers,
        dataset_binding=binding,
        normalization=prepared["normalization"],
        run=run,
        metrics=metrics,
        split_window_counts=prepared["counts"],
        split_ratios={
            "train_ratio": hyperparameters["train_ratio"],
            "validation_ratio": hyperparameters["validation_ratio"],
            "test_ratio": hyperparameters["test_ratio"],
        },
        split_ranges=prepared["split_ranges"],
        task_split=prepared["task_split"],
        task_id=task_id,
        model_source=model_source,
        uploaded_model=uploaded_block,
        **({"topography_identity": prepared["topography_asset"].identity,
            "topography_grid": {name: getattr(prepared["topography_asset"], name).tolist()
                                for name in ("latitude", "longitude", "lat_bounds", "lon_bounds")}}
           if prepared["topography_asset"] is not None else {}),
    )
    saved_path = save_earth_artifact_atomic(payload, output_path, strict_reload=True)
    # Final proof on the published file: reload it strictly and reproduce the
    # first test batch, so a corrupt write can never be reported as success.
    checkpoint = load_earth_training_artifact(
        saved_path, expected_binding=binding, expected_task_id=task_id,
        expected_hyperparameters=(None if metrics_contract is None else {
            **hyperparameters,
            **({"_earth_topography": spec["topography"]} if "topography" in spec else {}),
            **({"_earth_task_split": prepared["task_split"],
                EARTH_METRICS_SCHEMA_KEY: metrics_contract["schema"],
                EARTH_METRICS_AGGREGATION_KEY: metrics_contract["aggregation"]}
               if prepared["task_split"] is not None else {}),
        }),
    )
    if metrics_contract is not None and any(checkpoint.metrics.get(key) != value for key, value in metrics_contract.items()):
        raise EarthTrainingError("Published checkpoint metrics disagree with the frozen execution contract")
    reloaded, reload_warnings = build_earth_model_from_checkpoint(checkpoint)
    for warning in reload_warnings:
        print(f"Earth reload warning: {warning}", flush=True)
    reloaded.to(resolved_device)
    sample_batch = next(iter(prepared["test_loader"]))
    sample_inputs, sample_longitude, _, _ = _split_batch(sample_batch)
    sample_topography = sample_batch.get("topography") if isinstance(sample_batch, dict) else None
    sample_inputs = sample_inputs.to(resolved_device)
    if sample_longitude is not None:
        sample_longitude = sample_longitude.to(resolved_device)
    with torch.no_grad():
        before = _forward_batch(model, sample_inputs, model_source=model_source,
                                horizon=profile["horizon"], longitude=sample_longitude, topography=sample_topography)
        after = _forward_batch(reloaded, sample_inputs, model_source=model_source,
                               horizon=profile["horizon"], longitude=sample_longitude, topography=sample_topography)
    if not torch.equal(before, after):
        raise EarthTrainingError("The published checkpoint does not reproduce the trained model")
    print(f"Model saved: {saved_path}", flush=True)
    print("Earth training completed", flush=True)
    return {
        "output_path": saved_path,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "split_window_counts": prepared["counts"],
        "metrics": checkpoint.metrics,
        "model_source": model_source,
        "model_identity": checkpoint.model_identity(),
        "warnings": list(model_warnings) + list(reload_warnings),
    }


def _load_spec_from_env() -> dict:
    raw = os.environ.get(SPEC_ENV_VAR, "")
    if not raw.strip():
        raise EarthTrainingError(
            f"Missing {SPEC_ENV_VAR}; Earth training must be started by the training service"
        )
    try:
        spec = json.loads(raw)
    except ValueError as exc:
        raise EarthTrainingError(f"{SPEC_ENV_VAR} is not valid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise EarthTrainingError(f"{SPEC_ENV_VAR} must contain a JSON object")
    return spec


def _build_registry(spec: Optional[dict] = None) -> DatasetRegistry:
    from config import (EARTH_MERRA2_DIR, EARTH_MERRA2_V1_DIR,
                        EARTH_MERRA2_3HOURLY_DIR, EARTH_MERRA2_3HOURLY_5DEG_DIR, EARTH_TRAINING_CACHE_DIR)

    threehour_dir = EARTH_MERRA2_3HOURLY_DIR
    lowres_dir = EARTH_MERRA2_3HOURLY_5DEG_DIR
    if is_earth_three_hourly_dataset_id((spec or {}).get("dataset_binding", {}).get("dataset_id")):
        data_path = (spec or {}).get("data_path")
        if data_path is not None:
            path = Path(data_path)
            if path.name != "earth_merra2_3hourly.nc" or not path.is_absolute():
                raise EarthTrainingError("The server three-hour data path is invalid")
            if spec["dataset_binding"]["dataset_id"] == "earth_merra2_3hourly_5deg_v1":
                lowres_dir = path.parent
            else:
                threehour_dir = path.parent
    return DatasetRegistry(
        EARTH_MERRA2_DIR,
        earth_dataset_id="earth_merra2_daily_v2",
        legacy_earth_package_dir=EARTH_MERRA2_V1_DIR,
        earth_3hourly_package_dir=threehour_dir,
        earth_3hourly_5deg_package_dir=lowres_dir,
        verification_cache_dir=EARTH_TRAINING_CACHE_DIR,
        progress=lambda message: print(message, flush=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Earth MERRA-2 DLinear training")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    try:
        spec = _load_spec_from_env()
        run_training(spec, Path(args.output_path), _build_registry(spec))
    except (EarthTrainingError, EarthArtifactError, DatasetRequestError, ValueError) as exc:
        print(f"Earth training failed: {exc}", flush=True)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected, surfaced verbatim
        print(f"Earth training failed unexpectedly: {type(exc).__name__}: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
