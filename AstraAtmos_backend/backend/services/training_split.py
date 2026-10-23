"""Chronological train/validation/test split contract for training tasks."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


DEFAULT_SPLIT_RATIOS = {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}
LEGACY_SPLIT_RATIOS = {"train_ratio": 0.8, "validation_ratio": 0.0, "test_ratio": 0.2}
STRICT_TIMELINE_SPLIT_POLICY = "strict_timeline_windows_v1"
_RATIO_KEYS = tuple(DEFAULT_SPLIT_RATIOS)
_SUM_TOLERANCE = Decimal("0.000001")


class TrainingSplitError(ValueError):
    """Raised when a split request cannot produce a valid chronological split."""


def normalize_split_ratios(
    raw: Mapping[str, Any] | None,
    *,
    legacy_default: bool = False,
) -> dict[str, float]:
    """Validate and normalize train/validation/test ratios.

    New tasks default to 70/20/10. ``legacy_default`` explicitly selects the
    historical 80/20 train/test contract for readers of old task metadata.
    Ratios are not silently scaled: callers must provide a sum of one.
    """
    source = LEGACY_SPLIT_RATIOS if legacy_default and not raw else DEFAULT_SPLIT_RATIOS
    values: dict[str, Decimal] = {}
    for key in _RATIO_KEYS:
        value = (raw or {}).get(key, source[key])
        if isinstance(value, bool):
            raise TrainingSplitError(f"{key} must be a finite number")
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            raise TrainingSplitError(f"{key} must be a finite number") from None
        if not parsed.is_finite() or parsed < 0 or (parsed == 0 and key != "validation_ratio"):
            raise TrainingSplitError(f"{key} must be finite and non-negative")
        values[key] = parsed
    if values["validation_ratio"] < 0:
        raise TrainingSplitError("validation_ratio must be non-negative")
    if abs(sum(values.values()) - Decimal("1")) > _SUM_TOLERANCE:
        raise TrainingSplitError("train_ratio, validation_ratio and test_ratio must sum to 1")
    if values["train_ratio"] <= 0 or values["test_ratio"] <= 0:
        raise TrainingSplitError("train_ratio and test_ratio must be positive")
    return {key: float(values[key]) for key in _RATIO_KEYS}


def split_sample_ranges(sample_count: int, ratios: Mapping[str, float]) -> dict[str, tuple[int, int]]:
    """Allocate contiguous half-open sample ranges using largest remainders."""
    count = int(sample_count)
    if count < 3:
        raise TrainingSplitError("at least three samples are required")
    normalized = normalize_split_ratios(ratios)
    exact = [Decimal(str(normalized[key])) * count for key in _RATIO_KEYS]
    sizes = [int(value) for value in exact]
    remainder = count - sum(sizes)
    order = sorted(range(3), key=lambda index: (exact[index] - sizes[index], -index), reverse=True)
    for index in order[:remainder]:
        sizes[index] += 1
    if sizes[0] < 1 or sizes[2] < 1 or (normalized["validation_ratio"] > 0 and sizes[1] < 1):
        raise TrainingSplitError("train and test splits must contain samples; validation may be empty")
    ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for key, size in zip(_RATIO_KEYS, sizes):
        ranges[key.removesuffix("_ratio")] = (start, start + size)
        start += size
    return ranges


def split_time_boundary(sample_count: int, window: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Return raw timeline boundaries for normalization fitting.

    The returned ``train_end`` is the target-sample boundary plus ``window``;
    validation and test boundaries are target indices and remain chronological.
    """
    try:
        ranges = split_sample_ranges(sample_count, ratios)
    except TrainingSplitError:
        # Preparation and inference callers may need normalization metadata for a
        # tiny legacy dataset; full three-way validity is enforced by the runner.
        count = int(sample_count)
        if count < 1:
            raise
        normalized = normalize_split_ratios(ratios)
        train_end = max(1, min(count, int(count * normalized["train_ratio"])))
        return {
            "train_start": 0,
            "train_end": train_end + int(window),
            "validation_start": train_end,
            "validation_end": train_end,
            "test_start": train_end,
            "test_end": count,
        }
    return {
        "train_start": ranges["train"][0],
        "train_end": ranges["train"][1] + int(window),
        "validation_start": ranges["validation"][0],
        "validation_end": ranges["validation"][1],
        "test_start": ranges["test"][0],
        "test_end": ranges["test"][1],
    }


def _allocate_counts(count: int, ratios: Mapping[str, float], minimum: int = 0) -> list[int]:
    """Allocate chronological raw-time counts while reserving a minimum per split."""
    normalized = normalize_split_ratios(ratios)
    count = int(count)
    minimum = int(minimum)
    if count < 3 * minimum:
        raise TrainingSplitError(
            f"at least {3 * minimum} raw time points are required for strict windows"
        )
    remaining = count - 3 * minimum
    exact = [Decimal(str(normalized[key])) * remaining for key in _RATIO_KEYS]
    sizes = [minimum + int(value) for value in exact]
    remainder = count - sum(sizes)
    order = sorted(
        range(3),
        key=lambda index: (exact[index] - int(exact[index]), -index),
        reverse=True,
    )
    for index in order[:remainder]:
        sizes[index] += 1
    return sizes


def split_timeline_ranges(
    time_count: int,
    ratios: Mapping[str, float],
    *,
    minimum_segment: int = 0,
) -> dict[str, tuple[int, int]]:
    """Return half-open raw-time ranges before any sliding windows are built.

    ``minimum_segment`` is used by strict window preparation to reserve enough
    raw points for one complete input plus target sequence in every split.
    """
    count = int(time_count)
    if count < 1:
        raise TrainingSplitError("at least one raw time point is required")
    sizes = _allocate_counts(count, ratios, int(minimum_segment))
    ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for key, size in zip(_RATIO_KEYS, sizes):
        name = key.removesuffix("_ratio")
        ranges[name] = (start, start + int(size))
        start += int(size)
    return ranges


def strict_window_split_metadata(
    time_count: int,
    window: int,
    horizon: int,
    ratios: Mapping[str, float],
    *,
    blocks: Any = None,
) -> dict[str, Any]:
    """Build chronologically ordered, non-overlapping windows by requested ratio.

    Ratios apply to the retained window count, rather than to raw time points.
    The ``window + horizon - 1`` legal starts immediately around each split
    boundary are discarded so an input/target sequence cannot overlap another
    split. Windows use the complete chronological timeline and may cross Mars
    year boundaries. The returned ``window_ranges`` contain half-open runs of
    legal starts; callers can persist the metadata and reconstruct exactly the
    same train, validation and test windows for metrics and inference.
    """
    window = int(window)
    horizon = int(horizon)
    if window <= 0 or horizon <= 0:
        raise TrainingSplitError("window and horizon must be positive")
    # Generate every legal start first. Allocating from this list makes the
    # configured ratios apply to actual training examples. The optional
    # ``blocks`` argument is retained for checkpoint/API compatibility; it no
    # longer restricts starts because a forecast window may cross an MY boundary.
    legal_starts = list(range(0, int(time_count) - window - horizon + 1))
    if not legal_starts:
        raise TrainingSplitError("no complete windows are available")

    normalized = normalize_split_ratios(ratios)
    active_names = ["train"]
    if normalized["validation_ratio"] > 0:
        active_names.append("validation")
    active_names.append("test")
    boundary_gap = window + horizon - 1
    usable_count = len(legal_starts) - (len(active_names) - 1) * boundary_gap
    if usable_count < len(active_names):
        raise TrainingSplitError(
            "not enough non-overlapping windows for the requested strict splits"
        )
    if len(active_names) == 3:
        sample_ranges = split_sample_ranges(usable_count, normalized)
        counts = {name: int(right - left) for name, (left, right) in sample_ranges.items()}
    else:
        exact = {
            name: Decimal(str(normalized[f"{name}_ratio"] if name != "validation" else 0.0)) * usable_count
            for name in active_names
        }
        counts = {name: int(value) for name, value in exact.items()}
        remainder = usable_count - sum(counts.values())
        order = sorted(
            active_names,
            key=lambda name: (exact[name] - counts[name], -active_names.index(name)),
            reverse=True,
        )
        for name in order[:remainder]:
            counts[name] += 1
        if any(counts[name] < 1 for name in active_names):
            raise TrainingSplitError("train and test splits must contain windows")
        counts["validation"] = 0
    selected: dict[str, list[int]] = {}
    cursor = 0
    for index, name in enumerate(("train", "validation", "test")):
        count = counts.get(name, 0)
        selected[name] = legal_starts[cursor:cursor + count]
        cursor += count
        if name in active_names and name != active_names[-1]:
            cursor += boundary_gap
    if cursor > len(legal_starts) or any(not selected[name] for name in active_names):
        raise TrainingSplitError(
            "requested ratios cannot produce non-overlapping splits"
        )

    metadata: dict[str, Any] = {"policy": STRICT_TIMELINE_SPLIT_POLICY, "window": window, "horizon": horizon}
    for name in ("train", "validation", "test"):
        starts = selected[name]
        raw_start = min(starts) if starts else None
        raw_end = max(starts) + window + horizon if starts else None
        runs: list[list[int]] = []
        for start in starts:
            if runs and start == runs[-1][1]:
                runs[-1][1] = start + 1
            else:
                runs.append([start, start + 1])
        metadata[name] = {
            "raw_start": int(raw_start) if raw_start is not None else None,
            "raw_end": int(raw_end) if raw_end is not None else None,
            "window_count": int(len(starts)),
            "window_ranges": runs,
            "input_time_start": int(min(starts)) if starts else None,
            "input_time_end": int(max(starts) + window) if starts else None,
            "target_time_start": int(min(starts) + window) if starts else None,
            "target_time_end": int(max(starts) + window + horizon) if starts else None,
        }
        if name in active_names and not starts:
            raise TrainingSplitError(f"strict {name} split has no complete windows")
    return metadata


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "LEGACY_SPLIT_RATIOS",
    "STRICT_TIMELINE_SPLIT_POLICY",
    "TrainingSplitError",
    "normalize_split_ratios",
    "split_sample_ranges",
    "split_time_boundary",
    "split_timeline_ranges",
    "strict_window_split_metadata",
]
