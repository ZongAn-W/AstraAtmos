"""Optional dataset-capability declaration for uploaded user models.

Uploaded models are single ``.py`` files exporting ``MODEL_SPEC`` and
``build_model(config)``. Historically a spec only described its parameter schema
and (optionally) Mars auxiliary inputs, and every accepted model was assumed to
be usable for Mars data. That implicit assumption is wrong for Earth: an Earth
sample is a different tensor shape on a different grid with no Ls and no MOLA
topography, so "works on Mars" must never be read as "works on Earth".

This module adds an **optional** ``MODEL_SPEC.datasets`` block:

.. code-block:: python

    MODEL_SPEC = {
        "name": "MyEarthModel",
        "parameters": {...},
        "datasets": {
            "earth_merra2": {
                # heights/widths the model accepts; absent means "any"
                "grid": [[36, 72]],
                "window": [7],
                "horizon": [3],
                # auxiliary inputs the model needs *in addition* to x; the Earth
                # feed provides none, so any declared input makes it incompatible.
                "auxiliary_inputs": [],
                # optional: declare how many leading input channels must be TO3
                "target_leading_channels": 1,
            }
        },
    }

Backward compatibility rules:

* No ``datasets`` block -> the spec keeps the historical meaning: usable for
  Mars, and **not** usable for Earth.
* ``datasets`` present but without an Earth feed -> not usable for Earth.
* ``datasets`` present -> the spec must still describe every dataset it claims
  consistently, so a malformed block is a validation error rather than a silent
  downgrade.
"""

from __future__ import annotations

from typing import Any
from training_backbones.earth_3hourly_uploaded_contract import FEED, FULL_GRID_FEED, FULL_GRID_CONTRACT_SCHEMA, LOWRES_FEED, LOWRES_DATASET_ID, LOWRES_CONTRACT_SCHEMA

#: Keys of the independent Earth feeds inside ``MODEL_SPEC.datasets``.
EARTH_FEED_KEY = "earth_merra2"
EARTH_3HOURLY_FEED_KEY = "earth_merra2_3hourly_v1"

EARTH_FEED_FIELDS = frozenset({
    "grid",
    "window",
    "horizon",
    "auxiliary_inputs",
    "target_leading_channels",
})
EARTH_3HOURLY_FEED_FIELDS = frozenset(FEED)

#: Auxiliary inputs that exist only for Mars. The Earth feed never provides them.
MARS_ONLY_AUXILIARY_INPUTS = ("ls", "topography")


class DatasetCapabilityError(ValueError):
    """A malformed ``MODEL_SPEC.datasets`` declaration."""


def _normalize_int_pairs(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise DatasetCapabilityError(f"{label} must be a non-empty list of [height, width] pairs")
    normalized: list[list[int]] = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise DatasetCapabilityError(f"{label} entries must be [height, width] pairs")
        height, width = entry
        for size in (height, width):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise DatasetCapabilityError(f"{label} entries must use positive integers")
        normalized.append([int(height), int(width)])
    return normalized


def _normalize_positive_ints(value: Any, label: str) -> list[int]:
    if isinstance(value, bool) or not isinstance(value, (list, tuple)) or not value:
        raise DatasetCapabilityError(f"{label} must be a non-empty list of positive integers")
    normalized: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise DatasetCapabilityError(f"{label} entries must be positive integers")
        normalized.append(int(item))
    return normalized


def normalize_earth_feed(feed: Any) -> dict[str, Any]:
    """Validate one ``datasets.earth_merra2`` block and return its canonical form."""
    if not isinstance(feed, dict):
        raise DatasetCapabilityError(
            f"MODEL_SPEC.datasets.{EARTH_FEED_KEY} must be a dict, got {type(feed).__name__}"
        )
    unexpected = sorted(set(feed) - EARTH_FEED_FIELDS)
    if unexpected:
        raise DatasetCapabilityError(
            f"MODEL_SPEC.datasets.{EARTH_FEED_KEY} unexpected fields: {unexpected}"
        )

    normalized: dict[str, Any] = {}

    if "grid" in feed:
        normalized["grid"] = _normalize_int_pairs(
            feed["grid"], f"MODEL_SPEC.datasets.{EARTH_FEED_KEY}.grid"
        )
    if "window" in feed:
        normalized["window"] = _normalize_positive_ints(
            feed["window"], f"MODEL_SPEC.datasets.{EARTH_FEED_KEY}.window"
        )
    if "horizon" in feed:
        normalized["horizon"] = _normalize_positive_ints(
            feed["horizon"], f"MODEL_SPEC.datasets.{EARTH_FEED_KEY}.horizon"
        )

    auxiliary = feed.get("auxiliary_inputs", [])
    if isinstance(auxiliary, str) or not isinstance(auxiliary, (list, tuple)):
        raise DatasetCapabilityError(
            f"MODEL_SPEC.datasets.{EARTH_FEED_KEY}.auxiliary_inputs must be a list"
        )
    names = []
    for item in auxiliary:
        if not isinstance(item, str) or not item.strip():
            raise DatasetCapabilityError(
                f"MODEL_SPEC.datasets.{EARTH_FEED_KEY}.auxiliary_inputs must contain names"
            )
        names.append(item.strip())
    normalized["auxiliary_inputs"] = names

    if "target_leading_channels" in feed:
        value = feed["target_leading_channels"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DatasetCapabilityError(
                f"MODEL_SPEC.datasets.{EARTH_FEED_KEY}.target_leading_channels "
                "must be a non-negative integer"
            )
        normalized["target_leading_channels"] = int(value)

    return normalized


def normalize_earth_3hourly_feed(feed: Any, dataset_id=EARTH_3HOURLY_FEED_KEY) -> dict[str, Any]:
    """Validate the versioned Earth three-hour uploaded-model contract."""
    if not isinstance(feed, dict):
        raise DatasetCapabilityError(f"MODEL_SPEC.datasets.{EARTH_3HOURLY_FEED_KEY} must be a dict")
    unexpected = sorted(set(feed) - EARTH_3HOURLY_FEED_FIELDS)
    if unexpected:
        raise DatasetCapabilityError(
            f"MODEL_SPEC.datasets.{EARTH_3HOURLY_FEED_KEY} unexpected fields: {unexpected}"
        )
    def exact(actual, expected):
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, list):
            return len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
        return actual == expected

    expected_feed = LOWRES_FEED if dataset_id == LOWRES_DATASET_ID else FULL_GRID_FEED if feed.get("schema") == FULL_GRID_CONTRACT_SCHEMA else FEED
    for key, value in expected_feed.items():
        actual = feed.get(key)
        if key in ('window', 'horizon'):
            values = _normalize_positive_ints(actual, f'{EARTH_3HOURLY_FEED_KEY}.{key}')
            if len(values) != len(set(values)) or any(v > 240 for v in values):
                raise DatasetCapabilityError(f'{key} must contain unique integers between 1 and 240')
            continue
        if not exact(actual, value):
            raise DatasetCapabilityError(
                f"MODEL_SPEC.datasets.{EARTH_3HOURLY_FEED_KEY}.{key} must equal {value!r}"
            )
    return dict(feed)


def normalize_dataset_declarations(model_spec: Any) -> dict[str, dict[str, Any]]:
    """Return the normalized ``datasets`` block, or ``{}`` when it is absent."""
    if not isinstance(model_spec, dict):
        raise DatasetCapabilityError("MODEL_SPEC must be a dict")
    datasets = model_spec.get("datasets")
    if datasets is None:
        return {}
    if not isinstance(datasets, dict):
        raise DatasetCapabilityError("MODEL_SPEC.datasets must be a dict")
    if any(key in datasets for key in (EARTH_3HOURLY_FEED_KEY, LOWRES_DATASET_ID)):
        if all(key in datasets for key in (EARTH_3HOURLY_FEED_KEY, LOWRES_DATASET_ID)):
            raise DatasetCapabilityError("Declare one three-hour spatial feed per uploaded file")
        # ``derived_inputs`` declares tensors the server derives for the model;
        # ``auxiliary_inputs`` is the Mars block, accepted here only so the model
        # build can refuse it with a message that names the Earth equivalent.
        extra = set(model_spec) - {"name", "description", "parameters", "datasets",
                                   "auxiliary_inputs", "derived_inputs"}
        if extra:
            raise DatasetCapabilityError(f"Earth three-hour MODEL_SPEC unexpected fields: {sorted(extra)}")

    normalized: dict[str, dict[str, Any]] = {}
    for name, feed in datasets.items():
        if not isinstance(name, str) or not name.strip():
            raise DatasetCapabilityError("MODEL_SPEC.datasets keys must be non-empty strings")
        if name.strip() == EARTH_FEED_KEY:
            normalized[EARTH_FEED_KEY] = normalize_earth_feed(feed)
        elif name.strip() in (EARTH_3HOURLY_FEED_KEY, LOWRES_DATASET_ID):
            normalized[name.strip()] = normalize_earth_3hourly_feed(feed, dataset_id=name.strip())
        else:
            raise DatasetCapabilityError(
                "MODEL_SPEC.datasets supports only earth_merra2 and "
                f"{EARTH_3HOURLY_FEED_KEY} or {LOWRES_DATASET_ID}, got unknown feed {name!r}"
            )
    return normalized


def declares_earth_feed(model_spec: Any) -> bool:
    """Whether the spec declares an Earth feed at all (malformed counts as declared)."""
    return isinstance(model_spec, dict) and isinstance(model_spec.get("datasets"), dict) \
        and (EARTH_FEED_KEY in model_spec["datasets"] or EARTH_3HOURLY_FEED_KEY in model_spec["datasets"] or LOWRES_DATASET_ID in model_spec["datasets"])


def earth_feed_from_spec(model_spec: Any) -> dict[str, Any] | None:
    """Return the normalized Earth feed, or ``None`` when it is not declared."""
    declarations = normalize_dataset_declarations(model_spec)
    return declarations.get(EARTH_FEED_KEY)


def earth_3hourly_feed_from_spec(model_spec: Any, dataset_id=EARTH_3HOURLY_FEED_KEY) -> dict[str, Any] | None:
    return normalize_dataset_declarations(model_spec).get(dataset_id)


def earth_incompatibility_reasons(
    model_spec: Any,
    auxiliary_inputs: dict[str, Any] | None,
    *,
    height: int,
    width: int,
    channel_count: int,
    window: int,
    horizon: int,
) -> list[str]:
    """Return human-readable reasons why this spec cannot serve the Earth feed.

    An empty list means the declaration is compatible. The channel count is the
    number of model input channels the platform would feed (``TO3`` plus the
    selected auxiliary variables), so a spec that hard-codes a channel count in
    its ``parameters`` is still judged on what it actually receives.
    """
    reasons: list[str] = []
    if earth_feed_from_spec(model_spec) is None:
        reasons.append(
            "MODEL_SPEC does not declare the earth_merra2 dataset feed; "
            "an uploaded model must opt in before it can be trained on Earth data"
        )
        return reasons

    feed = earth_feed_from_spec(model_spec) or {}

    declared_auxiliary = list(feed.get("auxiliary_inputs") or [])
    provided_auxiliary = sorted((auxiliary_inputs or {}).keys())
    mars_only = sorted(set(declared_auxiliary).intersection(MARS_ONLY_AUXILIARY_INPUTS))
    if mars_only:
        reasons.append(
            "the Earth feed provides no auxiliary inputs, but the model declares: "
            f"{', '.join(mars_only)}"
        )
    extra = sorted(set(declared_auxiliary) - set(provided_auxiliary) - set(mars_only))
    if extra:
        reasons.append(
            f"the Earth feed does not provide declared auxiliary inputs: {', '.join(extra)}"
        )

    grid = feed.get("grid")
    if grid is not None and [int(height), int(width)] not in grid:
        reasons.append(
            f"the declared grid does not include Earth's {int(height)}x{int(width)} grid "
            f"(declared: {grid})"
        )
    windows = feed.get("window")
    if windows is not None and int(window) not in windows:
        reasons.append(f"the declared window does not include {int(window)} (declared: {windows})")
    horizons = feed.get("horizon")
    if horizons is not None and int(horizon) not in horizons:
        reasons.append(f"the declared horizon does not include {int(horizon)} (declared: {horizons})")

    leading = feed.get("target_leading_channels")
    if leading is not None and int(leading) > int(channel_count):
        reasons.append(
            "the declared target_leading_channels "
            f"({int(leading)}) exceeds the model input channels ({int(channel_count)})"
        )
    return reasons


__all__ = [
    "DatasetCapabilityError",
    "EARTH_FEED_KEY",
    "EARTH_3HOURLY_FEED_KEY",
    "MARS_ONLY_AUXILIARY_INPUTS",
    "declares_earth_feed",
    "earth_feed_from_spec",
    "earth_3hourly_feed_from_spec",
    "earth_incompatibility_reasons",
    "normalize_dataset_declarations",
    "normalize_earth_feed",
    "normalize_earth_3hourly_feed",
]
