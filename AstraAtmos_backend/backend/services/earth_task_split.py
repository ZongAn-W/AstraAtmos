"""Versioned task partitions, independent of the published release identity."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Mapping

import numpy as np

from services.training_split import TrainingSplitError, normalize_split_ratios

EARTH_TASK_SPLIT_POLICY = "earth_raw_utc_timeline_v1"
SPLIT_NAMES = ("train", "validation", "test")
TASK_SPLIT_KEY = "_earth_task_split"
TASK_SPLIT_POLICY_KEY = "_earth_task_split_policy"


def required_task_split(hyperparameters):
    """Recognize new task evidence before allowing the legacy manifest fallback."""
    raw = {} if hyperparameters is None else hyperparameters
    if not isinstance(raw, Mapping):
        raise TrainingSplitError("Earth task hyperparameters must be an object")
    if (TASK_SPLIT_POLICY_KEY in raw
            and raw[TASK_SPLIT_POLICY_KEY] != EARTH_TASK_SPLIT_POLICY):
        raise TrainingSplitError("Earth task split policy marker is damaged or unsupported")
    from services.earth_training_contract import earth_task_metrics_contract
    try:
        earth_task_metrics_contract(raw, executable=True)
    except ValueError as exc:
        raise TrainingSplitError(str(exc)) from exc
    requires_split = any(key in raw for key in (
        TASK_SPLIT_KEY, TASK_SPLIT_POLICY_KEY, "_earth_metrics_schema", "_earth_metrics_aggregation",
    ))
    split = raw.get(TASK_SPLIT_KEY)
    if requires_split and not isinstance(split, dict):
        raise TrainingSplitError("Earth task has lost its required frozen partition metadata")
    return split


def earth_split_ratios(raw):
    # Earth is strict about JSON types; the shared Mars reader also accepts strings.
    for name in SPLIT_NAMES:
        key = name + "_ratio"
        if key in (raw or {}) and (type(raw[key]) not in (int, float)):
            raise TrainingSplitError(f"{key} must be a finite number")
    ratios = normalize_split_ratios(raw)
    if ratios["validation_ratio"] <= 0:
        raise TrainingSplitError("Earth validation_ratio must be greater than 0")
    return ratios


def utc(value):
    return np.datetime_as_string(np.datetime64(value, "s"), unit="s") + "Z"


def timeline_from_snapshot(snapshot):
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    time = snapshot.get("time") if isinstance(snapshot, Mapping) else None
    if not isinstance(time, Mapping) or type(time.get("count")) is not int or time["count"] < 1:
        raise TrainingSplitError("Earth release snapshot has no complete UTC timeline")
    try:
        if any(not isinstance(time.get(key), str) or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", time[key]) is None
               for key in ("start", "end")):
            raise ValueError("UTC timestamps are required")
        start = np.datetime64(time["start"].removesuffix("Z"), "ns")
        end = np.datetime64(time["end"].removesuffix("Z"), "ns")
        dates = start + np.arange(time["count"]) * np.timedelta64(3, "h")
        if (np.isnat(start) or dates[-1] != end
                or (start - start.astype("datetime64[D]")) % np.timedelta64(3, "h") != np.timedelta64(90, "m")):
            raise ValueError("invalid time coverage")
        return dates
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise TrainingSplitError("Earth release snapshot has an invalid UTC timeline") from exc


def build_earth_task_split(dates, window, horizon, ratios):
    dates = np.asarray(dates, dtype="datetime64[ns]")
    if (dates.ndim != 1 or len(dates) < 3 or np.isnat(dates).any()
            or not np.all(np.diff(dates) == np.timedelta64(3, "h"))):
        raise TrainingSplitError("Earth task splits require a complete continuous three-hour UTC timeline")
    if type(window) is not int or type(horizon) is not int or window < 1 or horizon < 1:
        raise TrainingSplitError("Earth window and horizon must be positive integers")
    ratios = earth_split_ratios(ratios)
    weights = [Decimal(str(ratios[name + "_ratio"])) for name in SPLIT_NAMES]
    # Divide by the sum only to absorb the shared validator's floating-point tolerance.
    exact = [weight * len(dates) / sum(weights) for weight in weights]
    sizes = [int(value) for value in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))
    for i in order[:len(dates) - sum(sizes)]:
        sizes[i] += 1
    ranges, cursor = {}, 0
    for name, steps in zip(SPLIT_NAMES, sizes):
        if steps < window + horizon:
            raise TrainingSplitError(
                f"Earth {name} partition has {steps} time steps; window={window} + "
                f"horizon={horizon} requires at least {window + horizon}. Adjust the ratios or windows."
            )
        stop = cursor + steps
        ranges[name] = {
            "raw_start": cursor, "raw_end": stop,
            "date_start": utc(dates[cursor]), "date_end": utc(dates[stop - 1]),
            "step_count": steps, "window_count": steps - window - horizon + 1,
        }
        cursor = stop
    return {"policy": EARTH_TASK_SPLIT_POLICY, "ratios": ratios,
            "time_count": len(dates), "window": window, "horizon": horizon, "ranges": ranges}


def validate_earth_task_split(split, dates, window, horizon, ratios):
    if not isinstance(split, dict) or split.get("policy") != EARTH_TASK_SPLIT_POLICY:
        raise TrainingSplitError("Earth task split strategy is missing, damaged or unsupported")
    expected = build_earth_task_split(dates, window, horizon, ratios)
    # JSON comparison distinguishes bool/int and float/int in index and count fields.
    try:
        actual_json = json.dumps(split, sort_keys=True, allow_nan=False)
        expected_json = json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TrainingSplitError("Earth task split metadata is damaged or inconsistent") from exc
    if actual_json != expected_json:
        raise TrainingSplitError("Earth task split boundaries, ratios or window counts are inconsistent")
    return expected


def task_split_codes(dates, metadata, split=None):
    # Always retain the release continuity and published partition validation.
    from services.earth_dataset import threehour_release_split_codes
    published = threehour_release_split_codes(dates, metadata)
    if split is None:
        return published
    if not isinstance(split, dict):
        raise TrainingSplitError("Earth task split metadata must be an object")
    validate_earth_task_split(split, dates, split.get("window"), split.get("horizon"), split.get("ratios"))
    codes = np.empty(len(dates), dtype="int8")
    for code, name in enumerate(SPLIT_NAMES):
        entry = split["ranges"][name]
        codes[entry["raw_start"]:entry["raw_end"]] = code
    return codes
