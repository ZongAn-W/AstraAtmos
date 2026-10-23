"""Versioned Mars forecast checkpoint contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from services.mars_dataset_identity import normalize_mars_dataset_identity, volume_dataset_identity
from services.training_split import STRICT_TIMELINE_SPLIT_POLICY

MARS_CHECKPOINT_SCHEMA = "astraatmos_mars_forecast_checkpoint_v1"
OPENMARS_WINDOW_POLICY = "openmars_ls_year_segments_v2"
MCD_WINDOW_POLICY = "mcd_merged_timeline_v2"
MARS_AUXILIARY_CHANNEL_ORDER = ("U", "V", "D", "S", "T")
MARS_LEGACY_SPLIT_POLICY = "legacy_compatibility"


class MarsCheckpointIdentityError(ValueError):
    """A versioned checkpoint has incomplete or inconsistent data identity."""


def mars_input_channel_order(selected_channels: list[str]) -> list[str]:
    """Keep O3 first; selected_channels contains only auxiliary inputs."""
    if (
        not isinstance(selected_channels, list)
        or any(not isinstance(channel, str) or channel == "O3" for channel in selected_channels)
        or selected_channels != [channel for channel in MARS_AUXILIARY_CHANNEL_ORDER if channel in selected_channels]
    ):
        raise ValueError("Mars selected_channels must contain distinct auxiliary channels in canonical order")
    return ["O3", *selected_channels]


def validate_mars_normalization(
    normalization: Any,
    *,
    selected_channels: list[str],
    grid_shape: tuple[int, int],
) -> dict[str, Any]:
    """Validate statistics in exactly the order of the model input tensor."""
    expected_order = mars_input_channel_order(selected_channels)
    if not isinstance(normalization, dict):
        raise ValueError("Mars checkpoint input normalization metadata is invalid")
    if normalization.get("input_channel_order") != expected_order:
        raise ValueError("Mars checkpoint input_channel_order does not match O3 and selected_channels")
    if len(grid_shape) != 2 or any(type(size) is not int or size <= 0 for size in grid_shape):
        raise ValueError("Mars checkpoint grid shape is invalid")
    means = normalization.get("input_mean")
    scales = normalization.get("input_scale")
    mask = normalization.get("constant_channel_mask")
    if normalization.get("method") != "spatial_standard" or any(
        not isinstance(values, list) or len(values) != len(expected_order)
        for values in (means, scales, mask)
    ):
        raise ValueError("Mars checkpoint input normalization metadata is invalid")
    if any(type(value) is not bool for value in mask):
        raise ValueError("Mars checkpoint constant_channel_mask must contain booleans")
    for mean, scale in zip(means, scales):
        try:
            mean_array = np.asarray(mean, dtype=np.float32)
            scale_array = np.asarray(scale, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Mars checkpoint input normalization arrays must be numeric") from exc
        if mean_array.shape != grid_shape or scale_array.shape != grid_shape:
            raise ValueError("Mars checkpoint input normalization shape is invalid")
        if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(scale_array)):
            raise ValueError("Mars checkpoint input normalization must be finite")
        if np.any(scale_array <= 0):
            raise ValueError("Mars checkpoint input normalization scale must be positive")
    target_mean = normalization.get("target_mean")
    target_scale = normalization.get("target_scale")
    if any(type(value) not in (int, float) or not np.isfinite(value) for value in (target_mean, target_scale)):
        raise ValueError("Mars checkpoint target normalization metadata is invalid")
    # A constant O3 target has std=0; the existing transform adds 1e-6.
    if target_scale < 0:
        raise ValueError("Mars checkpoint target normalization scale must be non-negative")
    return normalization


def _json_array(value: Any) -> list:
    return np.asarray(value, dtype=np.float32).tolist()


def build_mars_checkpoint(
    *,
    model_state_dict: dict,
    volume: Any,
    selected_channels: list[str],
    window: int,
    horizon: int,
    split_ratios: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if not hasattr(volume, "input_means"):
        # Unit-test doubles and legacy runner integrations may only return the
        # historical tuple. They are kept readable by the legacy path; real
        # Mars training always supplies a ScaledVolume with these statistics.
        raise ValueError("Mars checkpoint requires persisted input normalization statistics")
    input_mean = [_json_array(value) for value in volume.input_means]
    input_scale = [_json_array(value) for value in volume.input_stds]
    grid_shape = [int(volume.height), int(volume.width)]
    identity = volume_dataset_identity(volume)
    payload = {
        "schema": MARS_CHECKPOINT_SCHEMA,
        "model_state_dict": model_state_dict,
        "normalization": {
            "method": "spatial_standard",
            "fit_split": "train",
            "input_channel_order": mars_input_channel_order(selected_channels),
            "input_mean": input_mean,
            "input_scale": input_scale,
            "target_mean": float(volume.y_mean),
            "target_scale": float(volume.y_std),
            "split_idx": int(getattr(volume, "split_idx", 0)),
            "constant_channel_mask": [
                bool(np.all(np.asarray(scale) <= 1e-6)) for scale in input_scale
            ],
        },
        "training_contract": {
            "training_dataset": str(volume.dataset_id),
            "selected_channels": list(selected_channels),
            "window": int(window),
            "horizon": int(horizon),
            "train_ratio": float(split_ratios["train_ratio"]),
            "validation_ratio": float(split_ratios["validation_ratio"]),
            "test_ratio": float(split_ratios["test_ratio"]),
            "grid_shape": grid_shape,
            "window_policy": OPENMARS_WINDOW_POLICY if volume.dataset_id == "openmars_mcd" else MCD_WINDOW_POLICY,
            "split_policy": str(getattr(volume, "split_policy", MARS_LEGACY_SPLIT_POLICY)),
            "split_window_counts": {
                name: int(len(starts))
                for name, starts in (getattr(volume, "split_window_starts", {}) or {}).items()
            },
            "split_ranges": {
                name: dict(value)
                for name, value in (getattr(volume, "split_ranges", {}) or {}).items()
            },
        },
        "data_binding": {
            **identity,
            "data_source_type": str(volume.source_type),
            "file_fingerprint": identity["dataset_fingerprint"],
            "year_blocks": [
                {
                    "mars_year": int(block.mars_year),
                    "start": int(block.start),
                    "end": int(block.end),
                    "source_files": list(block.source_files),
                    "segments": [segment.to_metadata() for segment in getattr(block, "segments", ())],
                    "window_count": int(np.sum(
                        (np.asarray(getattr(volume, "sample_starts", ())) >= block.start)
                        & (np.asarray(getattr(volume, "sample_starts", ())) < block.end)
                    )),
                }
                for block in getattr(volume, "blocks", ())
            ],
        },
        "run": {
            "seed": int(seed),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    return validate_mars_checkpoint(payload)


def checkpoint_state_dict(payload: Any) -> dict | None:
    if not isinstance(payload, dict):
        return None
    state = payload.get("model_state_dict")
    return state if isinstance(state, dict) else None


def checkpoint_normalization(payload: Any) -> dict | None:
    if not isinstance(payload, dict):
        return None
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        return None
    return normalization


def validate_mars_checkpoint(
    payload: Any,
    *,
    selected_channels: list[str] | None = None,
    window: int | None = None,
    horizon: int | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != MARS_CHECKPOINT_SCHEMA:
        raise ValueError("Mars checkpoint schema is missing or unsupported")
    state = checkpoint_state_dict(payload)
    normalization = checkpoint_normalization(payload)
    contract = payload.get("training_contract")
    binding = payload.get("data_binding")
    if state is None or normalization is None or not isinstance(contract, dict) or not isinstance(binding, dict):
        raise ValueError("Mars checkpoint is missing model, normalization, contract, or data binding metadata")
    channels = contract.get("selected_channels")
    mars_input_channel_order(channels)
    if selected_channels is not None and channels != list(selected_channels):
        raise ValueError("Mars checkpoint channel order does not match the training task")
    if window is not None and int(contract.get("window", -1)) != int(window):
        raise ValueError("Mars checkpoint window does not match the training task")
    if horizon is not None and int(contract.get("horizon", -1)) != int(horizon):
        raise ValueError("Mars checkpoint horizon does not match the training task")
    actual_shape = tuple(contract.get("grid_shape") or ())
    if grid_shape is not None and actual_shape != tuple(grid_shape):
        raise ValueError("Mars checkpoint grid shape does not match the loaded data")
    validate_mars_normalization(normalization, selected_channels=channels, grid_shape=actual_shape)
    split_policy = contract.get("split_policy", MARS_LEGACY_SPLIT_POLICY)
    if split_policy == STRICT_TIMELINE_SPLIT_POLICY:
        ranges = contract.get("split_ranges")
        counts = contract.get("split_window_counts")
        if not isinstance(ranges, dict) or not isinstance(counts, dict):
            raise ValueError("Strict Mars checkpoint split metadata is missing")
        for name in ("train", "validation", "test"):
            entry = ranges.get(name)
            if not isinstance(entry, dict):
                raise ValueError(f"Strict Mars checkpoint has no {name} split range")
            if int(entry.get("window_count", -1)) != int(counts.get(name, -1)):
                raise ValueError(f"Mars checkpoint {name} window count disagrees with its split range")
            if int(entry.get("window_count", 0)) == 0 and name == "validation":
                if entry.get("window_ranges") != []:
                    raise ValueError("Mars checkpoint empty validation split has invalid window boundaries")
                continue
            if not isinstance(entry.get("window_ranges"), list) or not entry["window_ranges"]:
                raise ValueError(f"Strict Mars checkpoint has no {name} window boundaries")
        # Adjacent partitions may have a gap, but their input and target time
        # ranges must never overlap.
        previous = []
        for name in ("train", "validation", "test"):
            entry = ranges[name]
            input_start, input_end = entry.get("input_time_start"), entry.get("input_time_end")
            target_start, target_end = entry.get("target_time_start"), entry.get("target_time_end")
            if int(entry.get("window_count", 0)) == 0:
                continue
            if None in (input_start, input_end, target_start, target_end):
                raise ValueError(f"Strict Mars checkpoint {name} time boundaries are incomplete")
            current = set(range(int(input_start), int(input_end))) | set(range(int(target_start), int(target_end)))
            if any(current & other for other in previous):
                raise ValueError("Strict Mars checkpoint split input/target ranges overlap")
            previous.append(current)
    elif split_policy != MARS_LEGACY_SPLIT_POLICY:
        raise ValueError("Mars checkpoint split policy is unsupported")
    if any(key not in binding for key in ("data_directories", "file_manifest", "file_fingerprint")):
        raise MarsCheckpointIdentityError("Mars checkpoint data identity metadata is incomplete")
    if contract.get("training_dataset") not in ("mcd_overview", "openmars_mcd"):
        raise MarsCheckpointIdentityError("Mars checkpoint training dataset is missing or unsupported")
    try:
        normalize_mars_dataset_identity(binding, dataset_id=contract.get("training_dataset"))
    except (ValueError, TypeError) as exc:
        raise MarsCheckpointIdentityError(f"Mars checkpoint data identity metadata is invalid: {exc}") from exc
    return payload


def mars_checkpoint_identity_snapshot(payload: Any) -> dict[str, Any]:
    """Rebuild the same task snapshot fields as identity_snapshot()."""
    validate_mars_checkpoint(payload)
    contract = payload["training_contract"]
    identity = normalize_mars_dataset_identity(payload["data_binding"], dataset_id=contract["training_dataset"])
    normalization = payload["normalization"]
    return {
        **identity, "manifest": identity["file_manifest"],
        "selected_channels": list(contract["selected_channels"]),
        "window": contract["window"], "horizon": contract["horizon"],
        "split_ratios": {key: contract.get(key) for key in ("train_ratio", "validation_ratio", "test_ratio")},
        "version_status": "verified", "window_policy": contract.get("window_policy"),
        "normalization": {
            "input_means": normalization["input_mean"], "input_stds": normalization["input_scale"],
            "y_mean": normalization["target_mean"], "y_std": normalization["target_scale"],
            "split_idx": normalization.get("split_idx"),
        },
        "year_blocks": list(payload["data_binding"].get("year_blocks", ())),
        "split_policy": contract.get("split_policy", MARS_LEGACY_SPLIT_POLICY),
        "split_window_counts": dict(contract.get("split_window_counts") or {}),
        "split_ranges": dict(contract.get("split_ranges") or {}),
    }


__all__ = [
    "MARS_AUXILIARY_CHANNEL_ORDER",
    "MARS_CHECKPOINT_SCHEMA",
    "MARS_LEGACY_SPLIT_POLICY",
    "MCD_WINDOW_POLICY",
    "MarsCheckpointIdentityError",
    "OPENMARS_WINDOW_POLICY",
    "build_mars_checkpoint",
    "checkpoint_normalization",
    "checkpoint_state_dict",
    "mars_input_channel_order",
    "mars_checkpoint_identity_snapshot",
    "validate_mars_checkpoint",
    "validate_mars_normalization",
]
