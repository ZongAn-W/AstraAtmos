"""Earth historical prediction from a task's verified release and checkpoint.

This service is the Earth counterpart of the Mars inference path. It exists so an
Earth task can be predicted from **its own** verified release and checkpoint:
nothing here reads OpenMARS, MCD, MY/Ls, Mars channels or the Mars prediction
cache, and an Earth task is never routed through Mars data preparation.

Daily tasks retain date-only 7-to-3 windows. Three-hour tasks use a UTC interval
center, 56 inputs ending at that origin and 24 subsequent reference timestamps.
Every returned field is TO3 in DU on the task's published grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Mapping, Optional

import numpy as np
import torch

from services.dataset_identity import (
    DatasetRequestError, is_earth_training_task, EARTH_DATASET_3HOURLY_ID, EARTH_THREE_HOURLY_DATASET_IDS,
    require_active_dataset, training_task_dataset_id,
)
from services.dataset_registry import DatasetRegistry
from services.earth_dataset import (
    TARGET_CHANNEL,
    TARGET_CHANNEL_INDEX,
    normalize_cube,
    reference_ozone,
    release_date_index,
    stack_release_cube,
    validate_normalization,
    _threehour_path, _threehour_values, threehour_release_split_codes,
    threehour_global_cell_area_weights,
)
from services.earth_training_artifact import (
    AREA_WEIGHTED_METRIC_POLICY,
    CELL_AREA_WEIGHTED_AGGREGATION,
    GRID_UNIFORM_AGGREGATION,
    EarthArtifactError,
    ErrorAccumulator,
    available_origin_range,
    build_earth_model_from_checkpoint,
    forecast_dates,
    load_earth_training_artifact,
    origin_split,
    validate_checkpoint_payload,
    threehour_available_origin_range, threehour_origin_index,
    threehour_forecast_timestamps, threehour_origin_split, has_earth_task_split,
)
from services.earth_training_contract import (
    EARTH_HORIZON,
    EARTH_TARGET_UNIT,
    EARTH_WINDOW,
)
from services.earth_model_source import (
    MODEL_SOURCE_OFFICIAL,
    MODEL_SOURCE_UPLOADED,
    EarthModelBuildError,
    earth_forward_for_model,
)
from services.uploaded_model_source import (
    UploadedModelSourceError,
    verify_reference_source,
)
from services.earth_dataset_metadata import file_sha256
from services.earth_prediction_cache import (
    EARTH_PREDICTION_CACHE, TASK_SPLIT_BOUNDARY_VERSION, build_earth_prediction_cache_key,
)
from services.netcdf_read_lock import netcdf_read_lock
from services.earth_derived_inputs import topography_for_checkpoint, topography_kwargs
import xarray as xr

ORIGIN_OUT_OF_RANGE = "earth_prediction_origin_out_of_range"
ORIGIN_INVALID = "invalid_earth_prediction_origin"
TASK_NOT_COMPLETED = "earth_prediction_task_not_completed"

VARIABLE_LABELS = {
    "TO3": "Total column ozone",
}


@dataclass(frozen=True)
class EarthPredictionContext:
    """Everything the prediction page needs before a run is requested."""

    task_id: int
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    target: str
    target_unit: str
    horizon: int
    window: int
    input_channel_order: list[str]
    input_units: list[str]
    latitude: list[float]
    longitude: list[float]
    grid_shape: list[int]
    origins: dict
    metrics: dict
    run: dict
    training_split_end: Optional[str] = field(default=None)
    #: Identity of the trained model (official DLinear or a pinned uploaded model).
    model: dict = field(default_factory=dict)
    #: Warnings raised while rebuilding the model, e.g. the uploaded file is gone.
    warnings: list[str] = field(default_factory=list)
    temporal: dict = field(default_factory=dict)


def _task_binding(task: Any) -> dict:
    binding = {
        "dataset_id": getattr(task, "dataset_id", None),
        "dataset_version": getattr(task, "dataset_version", None),
        "dataset_fingerprint": getattr(task, "dataset_fingerprint", None),
        "dataset_identity_status": getattr(task, "dataset_identity_status", None),
        "dataset_snapshot": getattr(task, "dataset_snapshot", None),
    }
    if not binding["dataset_id"] or not binding["dataset_fingerprint"]:
        # A task without a verified Earth binding cannot be predicted: there is no
        # release to recover the reference field from.
        raise DatasetRequestError(
            "invalid_earth_training_artifact",
            "This Earth task has no verified dataset binding",
            status_code=409,
        )
    if binding["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS and (
            binding["dataset_version"] != "v1" or binding["dataset_identity_status"] != "verified"
            or not binding["dataset_snapshot"]):
        raise DatasetRequestError("invalid_earth_training_artifact",
                                  "The three-hour task has no complete verified identity", status_code=409)
    return binding


def _require_completed_earth_task(task: Any) -> None:
    if not is_earth_training_task(task):
        raise DatasetRequestError(
            "dataset_prediction_not_supported",
            "Earth historical prediction requires an Earth training task",
            status_code=409,
        )
    require_active_dataset(training_task_dataset_id(task))
    status = getattr(task, "status", None)
    if status != "completed":
        raise DatasetRequestError(
            TASK_NOT_COMPLETED,
            "The Earth training task has not completed successfully",
            status_code=409,
        )
    output_path = getattr(task, "output_model_path", None)
    if not output_path:
        raise DatasetRequestError(
            TASK_NOT_COMPLETED,
            "The Earth training task has no model artifact",
            status_code=409,
        )


def _load_checkpoint(task: Any):
    binding = _task_binding(task)
    snapshot = binding.get("dataset_snapshot")
    if isinstance(snapshot, str):
        try:
            binding["dataset_snapshot"] = json.loads(snapshot)
        except ValueError as exc:
            raise DatasetRequestError(
                "invalid_earth_training_artifact",
                "The task dataset snapshot is damaged",
                status_code=409,
            ) from exc
    try:
        expected_hypers = None
        if binding["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS:
            expected_hypers = json.loads(getattr(task, "hyperparameters", None) or "{}")
            if not isinstance(expected_hypers, dict):
                raise EarthArtifactError("The task hyperparameters are damaged")
        checkpoint = load_earth_training_artifact(
            getattr(task, "output_model_path"),
            expected_binding=binding,
            expected_task_id=getattr(task, "id", None),
            expected_hyperparameters=expected_hypers,
        )
        if binding["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS:
            build_earth_model_from_checkpoint(checkpoint)  # Strict state-dict shape check for context/cache hits too.
        if binding["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS:
            expected_snapshot = binding.get("dataset_snapshot")
            actual_snapshot = checkpoint.dataset_binding["dataset_snapshot"]
            if not isinstance(expected_snapshot, dict) or any(
                    expected_snapshot.get(key) != actual_snapshot.get(key)
                    for key in ("dataset_id", "dataset_version", "dataset_fingerprint", "data_sha256",
                                "time", "grid", "splits", "frequency_hours", "step_unit", "step")):
                raise EarthArtifactError("The task snapshot does not match its checkpoint")
        return checkpoint
    except DatasetRequestError:
        raise
    except Exception as exc:
        # Checkpoint loaders can raise filesystem, deserialization, and model
        # construction exceptions. Keep their details out of the HTTP contract:
        # they may include an absolute path or an internal traceback fragment.
        code = getattr(exc, "code", "invalid_earth_training_artifact")
        message = (
            "The Earth dataset release changed since this checkpoint was created"
            if code == "dataset_version_changed"
            else "The Earth model artifact/checkpoint is unavailable or invalid"
        )
        raise DatasetRequestError(code, message, status_code=409) from exc


def _sync_release(registry: DatasetRegistry, binding: Mapping[str, Any]):
    from services.earth_training_contract import earth_training_profile
    """Return the release the task was bound to, or a stable version error."""
    try:
        release = registry.get_earth_snapshot(binding["dataset_id"])
    except DatasetRequestError as exc:
        if (binding["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS
                and exc.code == "dataset_unavailable"
                and exc.availability_reason != "package_unreadable"):
            raise DatasetRequestError("dataset_version_changed",
                                      "The bound three-hour Earth package is no longer verifiable", status_code=409) from exc
        raise
    metadata = release.metadata
    if binding.get("dataset_fingerprint") != metadata.get("dataset_fingerprint"):
        raise DatasetRequestError(
            "dataset_version_changed",
            "The dataset release changed since this task was trained; retrain before predicting",
            status_code=409,
        )
    if binding.get("dataset_id") in EARTH_THREE_HOURLY_DATASET_IDS:
        snapshot = binding.get("dataset_snapshot") or {}
        if (metadata.get("dataset_id") != binding["dataset_id"]
                or metadata.get("dataset_version") != binding.get("dataset_version")
                or metadata.get("grid") != snapshot.get("grid")
                or metadata.get("time") != snapshot.get("time")
                or metadata.get("splits") != snapshot.get("splits")
                or any(metadata.get(key) != snapshot.get(key) for key in (
                    "schema", "data_sha256", "variables", "channel_order"))):
            raise DatasetRequestError("dataset_version_changed",
                                      "The bound Earth release metadata changed", status_code=409)
        try:
            threehour_release_split_codes(release.dates, metadata)
            # Window availability belongs to the checkpoint's configured
            # window/horizon below. A valid short release need not support the
            # default 56->24 configuration to serve a shorter trained model.
            height, width = earth_training_profile(binding["dataset_id"])["grid_shape"]
            if (not np.array_equal(release.latitude, -90 + (np.arange(height) + .5) * (180 / height))
                    or not np.array_equal(release.longitude, -180 + (np.arange(width) + .5) * (360 / width))):
                raise ValueError("The three-hour grid coordinates changed")
            _threehour_path(release)
        except (EarthArtifactError, ValueError) as exc:
            raise DatasetRequestError("dataset_version_changed", str(exc), status_code=409) from exc
    return release


def _validate_checkpoint_split_ranges(checkpoint: Any, release: Any) -> None:
    """Reject a checkpoint whose published split contract no longer matches."""
    topography_for_checkpoint(checkpoint, release)
    contract = checkpoint.training_contract
    if checkpoint.dataset_binding.get("dataset_id") in EARTH_THREE_HOURLY_DATASET_IDS:
        from services.earth_task_split import EARTH_TASK_SPLIT_POLICY, validate_earth_task_split
        if contract.get("split_policy") == EARTH_TASK_SPLIT_POLICY:
            try:
                validate_earth_task_split(contract.get("task_split"), release.dates,
                                          contract["window"], contract["horizon"], contract["split_ratios"])
            except ValueError as exc:
                raise DatasetRequestError("invalid_earth_training_artifact", str(exc), status_code=409) from exc
            return
        # Legacy checkpoint ranges continue to describe the published manifest.
        for name in ("train", "validation", "test"):
            expected = contract["split_ranges"][name]
            current = (release.metadata.get("splits") or {})[name]
            first = np.datetime64(expected["date_start"].removesuffix("Z"), "ns")
            last = np.datetime64(expected["date_end"].removesuffix("Z"), "ns")
            if (str(first.astype("datetime64[D]")) != current["start"]
                    or str(last.astype("datetime64[D]")) != current["end"]
                    or expected["window_count"] != current["steps"] - contract['window'] - contract['horizon'] + 1):
                raise DatasetRequestError("dataset_version_changed", "The Earth release split changed", status_code=409)
        return
    if contract.get("split_policy") != "published_manifest_splits":
        # Historical Earth artifacts predate persisted manifest ranges. They
        # remain readable; newly trained artifacts always use the strict policy.
        return None
    expected = contract.get("split_ranges") or {}
    published = release.metadata.get("splits") or {}
    for name in ("train", "validation", "test"):
        actual = expected.get(name) or {}
        current = published.get(name) or {}
        if (
            str(actual.get("date_start")) != str(current.get("start"))
            or str(actual.get("date_end")) != str(current.get("end"))
        ):
            raise DatasetRequestError(
                "dataset_version_changed",
                "The Earth release split dates changed since this task was trained",
                status_code=409,
            )
        days = int(current.get("days", 0))
        expected_count = days - int(contract.get("window", EARTH_WINDOW)) - int(contract.get("horizon", EARTH_HORIZON)) + 1
        if int(actual.get("window_count", -1)) != expected_count:
            raise DatasetRequestError(
                "dataset_version_changed",
                "The Earth release split window counts changed since this task was trained",
                status_code=409,
            )
    return None


def build_prediction_context(task: Any, registry: DatasetRegistry) -> EarthPredictionContext:
    """Return the selectable origins, grid, units, model identity and metrics."""
    _require_completed_earth_task(task)
    checkpoint = _load_checkpoint(task)
    release = _sync_release(registry, checkpoint.dataset_binding)
    _validate_checkpoint_split_ranges(checkpoint, release)
    contract = checkpoint.training_contract
    task_split = contract.get("task_split")
    strict_task_split = has_earth_task_split(contract)
    # Preserve the published-manifest restriction for old uploaded artifacts,
    # while applying the frozen task split to both model sources.
    strict_origins = strict_task_split or checkpoint.model_source == MODEL_SOURCE_UPLOADED
    origins = threehour_available_origin_range(release, window=contract['window'], horizon=contract['horizon'],
        strict_split=strict_origins, task_split=task_split) if checkpoint.dataset_binding["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS else available_origin_range(
        release,
        window=int(contract.get("window", EARTH_WINDOW)),
        horizon=int(contract.get("horizon", EARTH_HORIZON)),
    )
    splits = checkpoint.metrics.get("splits") or {}
    run = checkpoint.run
    snapshot = checkpoint.dataset_binding.get("dataset_snapshot") or {}
    published_splits = snapshot.get("splits") or {}
    training_end = (published_splits.get("train") or {}).get("end")
    if checkpoint.dataset_binding["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS:
        training_end = contract["split_ranges"]["train"]["date_end"]
    # Report the uploaded model's file state without failing the context call: the
    # page must be able to explain that the original file is gone even though the
    # checkpoint can still rebuild the model from its embedded copy.
    warnings = _model_source_warnings(checkpoint)
    return EarthPredictionContext(
        task_id=int(getattr(task, "id", 0) or 0),
        dataset_id=checkpoint.dataset_binding["dataset_id"],
        dataset_version=str(checkpoint.dataset_binding.get("dataset_version")),
        dataset_fingerprint=checkpoint.dataset_binding["dataset_fingerprint"],
        target=contract.get("target", TARGET_CHANNEL),
        target_unit=contract.get("target_unit", EARTH_TARGET_UNIT),
        horizon=int(contract.get("horizon", EARTH_HORIZON)),
        window=int(contract.get("window", EARTH_WINDOW)),
        input_channel_order=list(contract.get("input_channel_order") or []),
        input_units=list(contract.get("input_units") or []),
        latitude=[float(value) for value in release.latitude],
        longitude=[float(value) for value in release.longitude],
        grid_shape=[int(release.latitude.size), int(release.longitude.size)],
        origins=origins,
        metrics={
            "schema": checkpoint.metrics.get("schema"),
            "unit": checkpoint.metrics.get("unit"),
            "aggregation": checkpoint.metrics.get("aggregation"),
            "splits": splits,
            **{key: checkpoint.metrics[key] for key in ("metric_units", "metric_policy") if key in checkpoint.metrics},
            "split_policy": contract.get("split_policy"),
            "task_split": contract.get("task_split"),
            "split_ratios": contract.get("split_ratios"),
            "split_ranges": contract.get("split_ranges"),
        },
        run={
            "best_epoch": run.get("best_epoch"),
            "epochs_completed": run.get("epochs_completed"),
            "seed": run.get("seed"),
            "device": run.get("device"),
            "created_utc": run.get("created_utc"),
            "linear_hidden_layers": (checkpoint.model_config.get("architecture_params") or {}).get(
                "linear_hidden_layers"
            ),
        },
        training_split_end=str(training_end) if training_end else None,
        model=checkpoint.model_identity(),
        warnings=warnings,
        temporal=({"frequency_hours": 3, "step_unit": "hour", "step": 3,
                   "time_zone": "UTC", "timestamp_rule": "interval_center"}
                  if checkpoint.dataset_binding["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS else {}),
    )


def _model_source_state(checkpoint: Any) -> tuple[str, list[str]]:
    """Return current source-file state and UI warnings without rebuilding code."""
    if checkpoint.model_source != MODEL_SOURCE_UPLOADED:
        return "official", []
    reference = checkpoint.uploaded_model or {}
    report = verify_reference_source(reference)
    if report["status"] == "available":
        return "available", []
    if report["status"] == "missing":
        return "missing", [
            "the original uploaded model file is unavailable; prediction uses the "
            "verified copy stored inside the checkpoint"
        ]
    if report["status"] == "tampered":
        return "tampered", [
            f"the uploaded model file no longer matches the trained version ({report['detail']}); "
            "prediction will use the verified copy stored inside the checkpoint"
        ]
    return "unknown", [f"the uploaded model file state is unknown: {report['detail']}"]


def _model_source_warnings(checkpoint: Any) -> list[str]:
    return _model_source_state(checkpoint)[1]


def _origin_index(release, origin: Any, window: int, horizon: int) -> int:
    if not isinstance(origin, str) or not origin.strip():
        raise DatasetRequestError(
            ORIGIN_INVALID,
            "A forecast origin date is required",
            status_code=422,
        )
    text = origin.strip()
    if len(text) > 10:
        raise DatasetRequestError(
            ORIGIN_INVALID, "Daily Earth tasks require a date-only forecast origin", status_code=422,
        )
    try:
        index = release_date_index(release, text)
    except (KeyError, ValueError) as exc:
        raise DatasetRequestError(
            ORIGIN_OUT_OF_RANGE,
            "The forecast origin is not a date in the published dataset",
            status_code=422,
        ) from exc
    dates = np.asarray(release.dates, dtype="datetime64[D]")
    if index < window - 1 or index + horizon >= len(dates):
        raise DatasetRequestError(
            ORIGIN_OUT_OF_RANGE,
            "The forecast origin needs a complete input window before it and "
            "published reference days after it",
            status_code=422,
        )
    return index


def run_earth_prediction(
    task: Any,
    origin: Any,
    registry: DatasetRegistry,
    device: Optional[str] = None,
    cache=None,
) -> dict:
    """Forecast the three days after ``origin`` and score them against the release."""
    _require_completed_earth_task(task)
    if getattr(task, "dataset_id", None) in EARTH_THREE_HOURLY_DATASET_IDS:
        return _run_threehour_prediction(task, origin, registry, device=device, cache=cache)
    checkpoint = _load_checkpoint(task)
    try:
        validate_checkpoint_payload(checkpoint.payload, expected_task_id=getattr(task, "id", None))
    except EarthArtifactError as exc:
        raise DatasetRequestError(
            getattr(exc, "code", "invalid_earth_training_artifact"),
            str(exc),
            status_code=409,
        ) from exc
    release = _sync_release(registry, checkpoint.dataset_binding)
    _validate_checkpoint_split_ranges(checkpoint, release)

    contract = checkpoint.training_contract
    window = int(contract.get("window", EARTH_WINDOW))
    horizon = int(contract.get("horizon", EARTH_HORIZON))
    order = list(contract.get("input_channel_order") or [])
    index = _origin_index(release, origin, window, horizon)
    dates = np.asarray(release.dates, dtype="datetime64[D]")
    origin_date = str(dates[index])

    mean, scale = validate_normalization(checkpoint.normalization, order)
    cube = stack_release_cube(release)
    # Input window is ``origin - (window - 1) .. origin`` inclusive.
    block = cube[index - window + 1:index + 1]
    inputs = normalize_cube(block, order, mean, scale)[None, ...]

    resolved_device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    try:
        model, model_warnings = build_earth_model_from_checkpoint(checkpoint)
    except (EarthArtifactError, EarthModelBuildError, UploadedModelSourceError) as exc:
        raise DatasetRequestError(
            getattr(exc, "code", "invalid_earth_training_artifact"),
            str(exc),
            status_code=409,
        ) from exc
    model = model.to(resolved_device)
    with torch.no_grad():
        output = earth_forward_for_model(
            model,
            torch.from_numpy(inputs).to(resolved_device),
            model_source=checkpoint.model_source,
            horizon=horizon,
        )
    if not torch.isfinite(output).all():
        raise DatasetRequestError(
            "invalid_earth_training_artifact",
            "The Earth model produced non-finite values",
            status_code=409,
        )
    prediction_scaled = output[0, :, 0].detach().cpu().numpy()
    target_mean = float(checkpoint.normalization["mean"][TARGET_CHANNEL_INDEX])
    target_scale = float(checkpoint.normalization["scale"][TARGET_CHANNEL_INDEX])
    prediction_du = prediction_scaled * target_scale + target_mean

    target_dates = forecast_dates(release, origin_date, horizon)
    reference_du = reference_ozone(release, target_dates[0], horizon)
    if prediction_du.shape != reference_du.shape:
        raise DatasetRequestError(
            "invalid_earth_training_artifact",
            "Prediction and reference shapes disagree",
            status_code=409,
        )
    residual_du = prediction_du - reference_du

    accumulator = ErrorAccumulator()
    # ErrorAccumulator expects [N, horizon, 1, lat, lon]; one origin, one channel.
    accumulator.update(prediction_du[None, :, None], reference_du[None, :, None])
    metrics = accumulator.result()
    metrics.update({
        "unit": EARTH_TARGET_UNIT,
        "target": TARGET_CHANNEL,
        "aggregation": "user_forecast_origin_lead_grid_uniform",
        "reference_available": True,
    })

    snapshot = checkpoint.dataset_binding.get("dataset_snapshot") or {}
    published_grid = snapshot.get("grid") or {}
    identity = checkpoint.model_identity()
    return {
        "planet": "earth",
        "task_id": int(getattr(task, "id", 0) or 0),
        "dataset_id": checkpoint.dataset_binding["dataset_id"],
        "dataset_version": str(checkpoint.dataset_binding.get("dataset_version")),
        "dataset_fingerprint": checkpoint.dataset_binding["dataset_fingerprint"],
        "target": TARGET_CHANNEL,
        "target_unit": EARTH_TARGET_UNIT,
        "model_architecture": identity.get("model_architecture"),
        "model_source": identity.get("model_source"),
        "model": identity,
        "warnings": list(model_warnings),
        "input_channel_order": order,
        "input_units": list(contract.get("input_units") or []),
        "forecast_origin": origin_date,
        "origin_split": origin_split(release, origin_date),
        "input_dates": [
            str(value) for value in dates[index - window + 1:index + 1]
        ],
        "target_dates": target_dates,
        "horizon": horizon,
        "window": window,
        "grid": {
            "shape": [int(release.latitude.size), int(release.longitude.size)],
            "latitude": [float(value) for value in release.latitude],
            "longitude": [float(value) for value in release.longitude],
            "latitude_range": published_grid.get("latitude_range"),
            "longitude_range": published_grid.get("longitude_range"),
            "latitude_step": published_grid.get("latitude_step"),
            "longitude_step": published_grid.get("longitude_step"),
            "coverage": published_grid.get("coverage"),
            "wrap_longitude": published_grid.get("wrap_longitude"),
        },
        "variables": [
            {
                "id": TARGET_CHANNEL,
                "label": VARIABLE_LABELS.get(TARGET_CHANNEL, TARGET_CHANNEL),
                "units": EARTH_TARGET_UNIT,
                "role": "target_and_input",
            }
        ],
        "prediction": _field_list(prediction_du),
        "reference": _field_list(reference_du),
        "residual": _field_list(residual_du),
        "metrics": metrics,
    }


def _threehour_checkpoint_sha(task):
    try:
        return file_sha256(Path(task.output_model_path))
    except (OSError, TypeError, ValueError) as exc:
        raise DatasetRequestError("invalid_earth_training_artifact",
                                  "The Earth checkpoint cannot be read", status_code=409) from exc


def _read_threehour_forecast_block(release, index, order, normalization, *, window=56, horizon=24):
    """Read the task's input/reference span outside the tile loop."""
    mean, scale = validate_normalization(normalization, order)
    inputs = np.empty((window, len(order), len(release.latitude), len(release.longitude)), dtype="float32")
    reference = np.empty((horizon, len(release.latitude), len(release.longitude)), dtype="float32")
    try:
        path = _threehour_path(release)
        first, forecast, stop = index - window + 1, index + 1, index + horizon + 1
        with netcdf_read_lock(), xr.open_dataset(path, engine="netcdf4", mask_and_scale=False) as ds:
            if (not np.array_equal(ds.time.isel(time=slice(first, stop)).values, release.dates[first:stop])
                    or not np.array_equal(ds.lat.values, release.latitude)
                    or not np.array_equal(ds.lon.values, release.longitude)):
                raise DatasetRequestError("dataset_version_changed",
                                          "Prediction coordinates or timestamps changed", status_code=409)
            for channel_index, channel in enumerate(order):
                for start in range(first, forecast, 8):
                    end = min(start + 8, forecast)
                    values = _threehour_values(ds, channel, start, end)
                    inputs[start - first:end - first, channel_index] = (
                        (values.astype("float64") - mean[channel_index]) / scale[channel_index]
                    ).astype("float32")
            for start in range(forecast, stop, 8):
                end = min(start + 8, stop)
                reference[start - forecast:end - forecast] = _threehour_values(ds, "TO3", start, end)
        _threehour_path(release)
    except DatasetRequestError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        try:
            _threehour_path(release)
        except ValueError as changed:
            raise DatasetRequestError("dataset_version_changed", str(changed), status_code=409) from changed
        raise DatasetRequestError("earth_prediction_data_unavailable",
                                  "The selected Earth input or reference data are unavailable", status_code=503) from exc
    return inputs, reference


def _threehour_window_longitude(release, checkpoint, index, window):
    """Return the input window's seasonal coordinate, or ``None`` when undeclared.

    Built from the release's own timestamps at exactly the indices the input block
    is read from, so the coordinate and the data can never describe different
    windows.
    """
    if "ls" not in (checkpoint.model_config.get("derived_inputs") or []):
        return None
    from services.earth_season import solar_longitude
    dates = np.asarray(release.dates, dtype="datetime64[ns]")[index - window + 1:index + 1]
    if len(dates) != int(window):
        raise DatasetRequestError(
            "dataset_version_changed",
            "The forecast window does not fit the published timeline", status_code=409,
        )
    # One forecast window is one batch element: the contract fixes [batch, window].
    return torch.from_numpy(np.ascontiguousarray(solar_longitude(dates)[None, :], dtype="float32"))


def _serialize_threehour_result(result):
    public = {key: value for key, value in result.items() if not key.startswith("_")}
    public["prediction"] = _field_list(result["_prediction_du"])
    public["reference"] = _field_list(result["_reference_du"])
    public["residual"] = _field_list(result["_residual_du"])
    return public


def _run_threehour_prediction(task, origin, registry, *, device=None, cache=None):
    checkpoint_sha = _threehour_checkpoint_sha(task)
    checkpoint = _load_checkpoint(task)
    if _threehour_checkpoint_sha(task) != checkpoint_sha:
        raise DatasetRequestError("invalid_earth_training_artifact",
                                  "The checkpoint changed during loading", status_code=409)
    release = _sync_release(registry, checkpoint.dataset_binding)
    topography_asset = topography_for_checkpoint(checkpoint, release)
    _validate_checkpoint_split_ranges(checkpoint, release)
    contract = checkpoint.training_contract
    window, horizon = contract['window'], contract['horizon']
    try:
        task_split = contract.get("task_split")
        strict_task_split = has_earth_task_split(contract)
        strict_origins = strict_task_split or checkpoint.model_source == MODEL_SOURCE_UPLOADED
        index = threehour_origin_index(release, origin, window=window, horizon=horizon,
            strict_split=strict_origins, task_split=task_split)
        timestamps = threehour_forecast_timestamps(release, origin, window=window, horizon=horizon,
            strict_split=strict_origins, task_split=task_split)
        split_name = threehour_origin_split(release, origin, window=window, horizon=horizon,
            strict_split=strict_origins, task_split=task_split)
    except EarthArtifactError as exc:
        raise DatasetRequestError(exc.code, str(exc), status_code=422 if exc.code in (
            ORIGIN_INVALID, ORIGIN_OUT_OF_RANGE) else 409) from exc
    iso = lambda value: np.datetime_as_string(value, unit="s") + "Z"
    normalized_origin = iso(release.dates[index])
    key = build_earth_prediction_cache_key(
        planet="earth", dataset_id=checkpoint.dataset_binding["dataset_id"],
        dataset_version=checkpoint.dataset_binding["dataset_version"],
        dataset_fingerprint=checkpoint.dataset_binding["dataset_fingerprint"],
        task_id=task.id, forecast_origin=normalized_origin, target_timestamps=timestamps,
        checkpoint_sha256=checkpoint_sha,
        split_boundary_policy=(TASK_SPLIT_BOUNDARY_VERSION if strict_task_split
                               else "legacy_compatibility"),
        **({"topography_identity": topography_asset.identity} if topography_asset is not None else {}),
    )
    cache = EARTH_PREDICTION_CACHE if cache is None else cache
    cached = cache.get(key)
    if cached is not None:
        if _threehour_checkpoint_sha(task) != checkpoint_sha:
            raise DatasetRequestError("invalid_earth_training_artifact",
                                      "The checkpoint changed during prediction", status_code=409)
        source_status, source_warnings = _model_source_state(checkpoint)
        cached["warnings"] = list(source_warnings)
        cached["model_source_status"] = source_status
        return _serialize_threehour_result(cached)
    order = checkpoint.training_contract["input_channel_order"]
    inputs, reference = _read_threehour_forecast_block(release, index, order, checkpoint.normalization,
                                                     window=window, horizon=horizon)
    longitude = _threehour_window_longitude(release, checkpoint, index, window)
    try:
        model, warnings = build_earth_model_from_checkpoint(checkpoint)
        resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(resolved_device).eval()
        prediction = np.empty_like(reference)
        mean, scale = checkpoint.normalization["mean"][0], checkpoint.normalization["scale"][0]
        if longitude is not None:
            longitude = longitude.to(resolved_device)
        extra = {} if longitude is None else {"longitude": longitude}
        with torch.no_grad():
            full_grid = checkpoint.dataset_binding["dataset_id"] == "earth_merra2_3hourly_5deg_v1"
            if checkpoint.model_source == MODEL_SOURCE_UPLOADED:
                from training_backbones.earth_3hourly_uploaded_contract import contract_profile
                full_grid = contract_profile(checkpoint.model_config["contract_schema"])["full_grid"]
            if full_grid:
                tensor = torch.from_numpy(np.ascontiguousarray(inputs[None])).to(resolved_device)
                output = earth_forward_for_model(
                    model, tensor,
                    model_source=checkpoint.model_source, horizon=horizon, height=len(release.latitude), width=len(release.longitude),
                    **extra, **topography_kwargs(topography_asset, tensor),
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    prediction[:] = output[0, :, 0].cpu().numpy() * scale + mean
            else:
                for lat in range(0, 240, 24):
                    for lon in range(0, 480, 48):
                        tile = np.ascontiguousarray(inputs[None, :, :, lat:lat + 24, lon:lon + 48])
                        tensor = torch.from_numpy(tile).to(resolved_device)
                        output = earth_forward_for_model(
                            model, tensor, model_source=checkpoint.model_source,
                            horizon=horizon, height=24, width=48,
                            **extra, **topography_kwargs(topography_asset, tensor,
                                spatial_slices=[(slice(lat, lat + 24), slice(lon, lon + 48))]),
                        )
                        with np.errstate(over="ignore", invalid="ignore"):
                            prediction[:, lat:lat + 24, lon:lon + 48] = output[0, :, 0].cpu().numpy() * scale + mean
        if not np.isfinite(prediction).all():
            raise EarthArtifactError("The Earth model produced non-finite DU predictions")
    except (EarthArtifactError, EarthModelBuildError, UploadedModelSourceError, ValueError, RuntimeError) as exc:
        raise DatasetRequestError("invalid_earth_training_artifact", str(exc), status_code=409) from exc
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            residual = prediction - reference
        if not np.isfinite(residual).all():
            raise EarthArtifactError("The Earth model produced non-finite residuals")
        checkpoint_metrics = getattr(checkpoint, "metrics", None) or {}
        aggregation = checkpoint_metrics.get("aggregation", GRID_UNIFORM_AGGREGATION)
        area_weighted = aggregation == CELL_AREA_WEIGHTED_AGGREGATION
        spatial_weights = threehour_global_cell_area_weights(release) if area_weighted else None
        accumulator = ErrorAccumulator(dataset_id=EARTH_DATASET_3HOURLY_ID, horizon=horizon)
        accumulator.update(prediction[None, :, None], reference[None, :, None],
                           spatial_weights=spatial_weights)
        from services.earth_training_artifact import METRIC_UNITS, METRIC_POLICY
        from services.earth_training_contract import EARTH_3HOURLY_METRICS_SCHEMA_V2, EARTH_3HOURLY_METRICS_SCHEMA_V3
        metrics = {**accumulator.result(), "unit": "DU", "target": "TO3",
                   "schema": checkpoint_metrics.get("schema", EARTH_3HOURLY_METRICS_SCHEMA_V3 if area_weighted else EARTH_3HOURLY_METRICS_SCHEMA_V2),
                   "metric_units": dict(METRIC_UNITS),
                   "metric_policy": dict(AREA_WEIGHTED_METRIC_POLICY if area_weighted else METRIC_POLICY),
                   "aggregation": ("user_forecast_origin_lead_cell_area_weighted" if area_weighted
                                   else "user_forecast_origin_lead_grid_uniform"),
                   "reference_available": True}
    except EarthArtifactError as exc:
        raise DatasetRequestError("invalid_earth_training_artifact", str(exc), status_code=409) from exc
    published_grid = checkpoint.dataset_binding["dataset_snapshot"]["grid"]
    identity = checkpoint.model_identity()
    source_status, source_warnings = _model_source_state(checkpoint)
    result = {
        "planet": "earth", "task_id": task.id,
        "dataset_id": checkpoint.dataset_binding["dataset_id"],
        "dataset_version": checkpoint.dataset_binding["dataset_version"],
        "dataset_fingerprint": checkpoint.dataset_binding["dataset_fingerprint"],
        "target": "TO3", "target_unit": "DU", "window": window, "horizon": horizon,
        "frequency_hours": 3, "step_unit": "hour", "step": 3, "time_zone": "UTC",
        "timestamp_rule": "interval_center", "forecast_origin": normalized_origin,
        "origin_split": split_name, "input_channel_order": list(order),
        "input_units": checkpoint.training_contract["input_units"],
        "input_dates": [iso(value) for value in release.dates[index - window + 1:index + 1]],
        "input_timestamps": [iso(value) for value in release.dates[index - window + 1:index + 1]],
        "target_dates": timestamps, "target_timestamps": timestamps,
        "model_architecture": identity.get("model_architecture"), "model_source": identity.get("model_source"),
        "model": identity, "model_source_status": source_status,
        "warnings": list(dict.fromkeys([*warnings, *source_warnings])), "metrics": metrics, "cache_key": key,
        "grid": {"shape": [len(release.latitude), len(release.longitude)], "latitude": release.latitude.tolist(), "longitude": release.longitude.tolist(),
                 **{name: published_grid.get(name) for name in (
                     "latitude_range", "longitude_range", "latitude_step", "longitude_step", "coverage", "wrap_longitude")}},
        "variables": [{"id": "TO3", "label": VARIABLE_LABELS["TO3"], "units": "DU", "role": "target_and_input"}],
        "_prediction_du": prediction, "_reference_du": reference, "_residual_du": residual,
    }
    # Check the bound package again after inference, before publishing/cache insertion.
    _sync_release(registry, checkpoint.dataset_binding)
    topography_for_checkpoint(checkpoint, release)
    if _threehour_checkpoint_sha(task) != checkpoint_sha:
        raise DatasetRequestError("invalid_earth_training_artifact", "The checkpoint changed during prediction", status_code=409)
    cache.put(key, result)
    return _serialize_threehour_result(result)


def compare_earth_test_metrics(tasks: list[Any], registry: DatasetRegistry) -> dict:
    """Compare verified, persisted full-test metrics without rerunning inference."""
    if len(tasks) < 2 or len({task.id for task in tasks}) != len(tasks):
        raise DatasetRequestError("invalid_earth_comparison", "Select at least two distinct Earth tasks", status_code=422)
    items, guards, signature = [], [], None
    for task in tasks:
        _require_completed_earth_task(task)
        sha = _threehour_checkpoint_sha(task)
        checkpoint = _load_checkpoint(task)
        release = _sync_release(registry, checkpoint.dataset_binding)
        _validate_checkpoint_split_ranges(checkpoint, release)
        contract = checkpoint.training_contract
        test = checkpoint.metrics["splits"]["test"]
        current = {
            "dataset_id": checkpoint.dataset_binding["dataset_id"],
            "version": checkpoint.dataset_binding["dataset_version"],
            "fingerprint": checkpoint.dataset_binding["dataset_fingerprint"],
            "window": contract["window"], "horizon": contract["horizon"],
            "target": contract["target"], "unit": checkpoint.metrics["unit"],
            "aggregation": checkpoint.metrics["aggregation"],
            "test_range": {key: contract["split_ranges"]["test"][key]
                           for key in ("date_start", "date_end", "window_count")},
            "window_count": test["window_count"],
        }
        if signature is not None and current != signature:
            raise DatasetRequestError("earth_comparison_incompatible",
                                      "Models have different test time ranges, input/output windows or metric policies; select tasks with the same test partition", status_code=409)
        signature = current
        identity = checkpoint.model_identity()
        raw = getattr(task, "hyperparameters", None) or {}
        hypers = json.loads(raw) if isinstance(raw, str) else dict(raw)
        metrics = {
            "schema": checkpoint.metrics["schema"],
            "overall": dict(test["overall"]),
            "per_step": [{"step": row["lead_step"], **row} for row in test["by_lead"]],
            "by_horizon": test["by_horizon"], "unit": "DU",
            "aggregation": checkpoint.metrics["aggregation"],
            "split_meta": {"source": contract["split_policy"], "test_range": current["test_range"],
                           "window_count": test["window_count"]},
        }
        for key in ("metric_units", "metric_policy"):
            if key in checkpoint.metrics:
                metrics[key] = checkpoint.metrics[key]
        items.append({"task_id": task.id, "model_name": task.custom_model_name or f"Task #{task.id}",
                      "planet": "earth", "dataset_id": current["dataset_id"],
                      "window": contract['window'], "horizon": contract['horizon'],
                      "dataset_version": current["version"], "dataset_fingerprint": current["fingerprint"],
                      "model_source": checkpoint.model_source, "architecture": identity.get("model_architecture"),
                      "selected_channels": list(contract["input_channel_order"]), "hyperparameters": hypers,
                      "model": identity, "metrics": metrics})
        guards.append((task, sha, checkpoint.dataset_binding))
    for task, sha, binding in guards:
        _sync_release(registry, binding)
        if _threehour_checkpoint_sha(task) != sha:
            raise DatasetRequestError("invalid_earth_training_artifact", "The checkpoint changed during comparison", status_code=409)
    return {"planet": "earth", "metric_source": "verified_checkpoint_test_metrics", "items": items}


def _field_list(values: np.ndarray) -> list[dict]:
    """Serialize ``[horizon, lat, lon]`` into finite per-day field payloads."""
    fields = []
    for day in np.asarray(values, dtype="float64"):
        finite = np.isfinite(day)
        fields.append({
            "field": [[float(cell) for cell in row] for row in day],
            "minVal": float(day.min()),
            "maxVal": float(day.max()),
            "valid_cells": int(finite.sum()),
        })
    return fields


__all__ = [
    "compare_earth_test_metrics",
    "EarthPredictionContext",
    "ORIGIN_INVALID",
    "ORIGIN_OUT_OF_RANGE",
    "TASK_NOT_COMPLETED",
    "build_prediction_context",
    "run_earth_prediction",
]
