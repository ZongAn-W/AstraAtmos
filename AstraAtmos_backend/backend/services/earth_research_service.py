"""Earth research / scientific-analysis layer over the verified v2 global release.

Everything here reads the immutable release snapshot handed out by
:class:`DatasetRegistry` once per request. The arrays are never copied into
mutable storage, never normalized and never rewritten: a non-finite sample is a
publish defect and raises ``invalid_dataset`` instead of being replaced by a
sentinel. Daily series always keep the real calendar dates, so a leap day is
carried explicitly and a missing day is an error rather than a gap.

The module level helpers are pure functions over NumPy arrays: they are the
unit-tested numeric core, while :class:`EarthResearchService` only wires them to
a release and shapes JSON-safe responses.
"""

from __future__ import annotations

import datetime as dt
import math
import threading
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from services.dataset_registry import DatasetRegistry
from services.dataset_identity import DatasetRequestError
from services.dataset_identity import EARTH_DATASET_3HOURLY_ID
from services.earth_dataset_metadata import VARIABLE_LABELS, VerifiedEarthRelease
from services.earth_overview_service import (
    GLOBAL_AGGREGATION,
    VARIABLE_IDS,
    WIND_VARIABLE_IDS,
    EarthOverviewError,
    EarthOverviewService,
)
from services.earth_scales import color_range as dataset_color_range

# ── public vocabulary ──────────────────────────────────────────────────

PLANET = "earth"
SOURCE_LABEL = "NASA MERRA-2"
CADENCE = "daily mean"
POLAR_MIN_ABS_LATITUDE = 60.0
POLAR_SAMPLING = "daily_mean"
POLAR_NOTE = "diurnal_variation_not_resolvable"
DIURNAL_UNAVAILABLE_REASON = "daily_data_has_no_diurnal_samples"
THREE_HOUR_DAILY_AGGREGATION = "utc_daily_mean_of_8_three_hour_means"

SEASONAL_AGGREGATION = "equal_longitude_mean"
REGIONAL_AGGREGATION = GLOBAL_AGGREGATION  # spherical_cell_area_mean
BAND_AGGREGATION = "cell_area_weighted_band_mean"
SPATIAL_REFERENCE = "annual_mean_minus_equal_longitude_mean"

INSIGHT_MODEL_BUILTIN = "builtin-digest"
INSIGHT_LOCALES = ("zh", "en")

# Marker in front of every prefix the request may carry. Only the five grid
# bands are addressable; anything else is a caller error.
BAND_IDS = ("south_polar", "south_mid", "tropics", "north_mid", "north_polar")

# Lower bound of every band, inclusive; the next band starts at its ``upper``.
# ``upper`` of the last band is infinity so no sampled row can fall through.
_BAND_RULES: tuple[tuple[str, Optional[str], float, float, bool], ...] = (
    ("south_polar", "south", -math.inf, -60.0, False),
    ("south_mid", "south", -60.0, -30.0, False),
    ("tropics", None, -30.0, 30.0, True),
    ("north_mid", "north", 30.0, 60.0, True),
    ("north_polar", "north", 60.0, math.inf, True),
)

RESEARCH_LIMITATIONS = [
    "Values are UTC daily means; the package carries no diurnal cycle",
    "Spatial scope is the published global 5 degree grid or one named latitude band",
    "Reported correlations are associations, not evidence of causation",
    "Ozone is total column Dobson units; the release is a reanalysis product, not observations",
]


def _finite_array(values: Any, *, name: str, ndim: int) -> np.ndarray:
    """Return ``values`` as a finite float64 array with an exact rank check."""
    array = np.asarray(values, dtype="float64")
    if array.ndim != ndim:
        raise EarthOverviewError(
            "invalid_dataset", f"{name} must be a {ndim}D array"
        )
    if array.size and not np.isfinite(array).all():
        # A missing sample is a publish defect, never something to skip.
        raise EarthOverviewError("invalid_dataset", f"{name} contains invalid values")
    return array


# ── pure numeric helpers ───────────────────────────────────────────────

def latitude_cell_edges(
    latitude: np.ndarray, bounds: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Latitude cell edges for the sampled grid, west-to-east in index order.

    Explicit ``bounds`` (the package ``lat_bounds`` extent) win; otherwise the
    edges are derived from the cell centres. Edges are always ascending, and a
    descending centre axis is reversed to match.
    """
    centres = np.asarray(latitude, dtype="float64")
    if centres.ndim != 1 or centres.size < 2:
        raise EarthOverviewError("invalid_dataset", "Latitude axis is invalid")
    if not np.isfinite(centres).all():
        raise EarthOverviewError("invalid_dataset", "Latitude contains invalid values")
    if centres[0] > centres[-1]:
        centres = centres[::-1]
    step = float(np.diff(centres)[0])
    if bounds is not None:
        lower, upper = float(bounds[0]), float(bounds[1])
        return np.linspace(min(lower, upper), max(lower, upper), centres.size + 1)
    return np.concatenate((
        [float(centres[0]) - step / 2.0],
        (centres[:-1] + centres[1:]) / 2.0,
        [float(centres[-1]) + step / 2.0],
    ))


def latitude_band_weights(edges: np.ndarray) -> np.ndarray:
    """Spherical area weight of each latitude cell: ``diff(sin(radians(edge)))``."""
    values = np.asarray(edges, dtype="float64")
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise EarthOverviewError("invalid_dataset", "Latitude cell edges are invalid")
    if np.any(np.diff(values) <= 0):
        raise EarthOverviewError("invalid_dataset", "Latitude cell edges are not increasing")
    weights = np.diff(np.sin(np.deg2rad(values)))
    if np.any(weights <= 0):
        raise EarthOverviewError("invalid_dataset", "Latitude cell weights are invalid")
    return weights


def _band_membership(latitude: np.ndarray) -> dict[str, list[int]]:
    values = np.asarray(latitude, dtype="float64")
    members: dict[str, list[int]] = {}
    for band_id, _hemisphere, lower, upper, lower_inclusive in _BAND_RULES:
        low_ok = values >= lower if lower_inclusive else values > lower
        members[band_id] = [int(index) for index in np.flatnonzero(low_ok & (values < upper))]
    return members


def band_definitions(latitude: np.ndarray) -> list[dict]:
    """Describe the five latitude bands of a sampled grid.

    Each band reports the *actual sampled latitude extremes* of its own rows
    (never the threshold), the sampled latitude values and the row count. The
    bands are disjoint and every grid row belongs to exactly one of them. An
    empty band reports ``None`` extremes instead of a fabricated range.
    """
    values = np.asarray(latitude, dtype="float64")
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise EarthOverviewError("invalid_dataset", "Latitude axis is invalid")
    membership = _band_membership(values)
    definitions = []
    for band_id, hemisphere, _lower, _upper, _inclusive in _BAND_RULES:
        indexes = membership[band_id]
        if indexes:
            sampled = values[indexes]
            min_latitude: Optional[float] = float(sampled.min())
            max_latitude: Optional[float] = float(sampled.max())
            latitude_values = [float(value) for value in sampled]
        else:
            min_latitude = max_latitude = None
            latitude_values = []
        definitions.append({
            "id": band_id,
            "hemisphere": hemisphere,
            "min_latitude": min_latitude,
            "max_latitude": max_latitude,
            "latitude_values": latitude_values,
            "grid_point_count": len(indexes),
        })
    return definitions


def equal_longitude_mean(field_tij: np.ndarray) -> np.ndarray:
    """Mean over longitude for every (day, latitude row): ``[T, H]``.

    All longitude cells carry the same area, so a plain arithmetic mean over
    the column axis is the equal-area zonal mean.
    """
    values = _finite_array(field_tij, name="Field", ndim=3)
    if values.shape[2] < 1:
        raise EarthOverviewError("invalid_dataset", "Field has no longitude cells")
    return values.mean(axis=2, dtype="float64")


def cell_area_mean_series(
    field_tij: np.ndarray, latitude: np.ndarray, bounds: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Global spherical cell-area daily mean: ``[T]``.

    Weights are ``diff(sin(deg2rad(lat_cell_edges)))``, so cells near the poles
    contribute proportionally less area than the naive ``cos(lat)`` sample mean.
    """
    values = _finite_array(field_tij, name="Field", ndim=3)
    if values.shape[1] != len(np.asarray(latitude)):
        raise EarthOverviewError("invalid_dataset", "Latitude axis does not match the field")
    weights = latitude_band_weights(latitude_cell_edges(latitude, bounds))
    return np.einsum("tij,i->t", values, weights) / (values.shape[2] * weights.sum())


def band_area_mean_series(
    field_tij: np.ndarray,
    latitude: np.ndarray,
    lat_indexes: Iterable[int],
    bounds: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Area mean restricted to a latitude band: ``[T]``.

    The same spherical cell weights are reused; only the selected rows enter
    numerator and denominator, so a band mean is never diluted by the rest of
    the grid.
    """
    values = _finite_array(field_tij, name="Field", ndim=3)
    latitude_values = np.asarray(latitude, dtype="float64")
    if values.shape[1] != latitude_values.size:
        raise EarthOverviewError("invalid_dataset", "Latitude axis does not match the field")
    indexes = np.asarray(sorted({int(index) for index in lat_indexes}), dtype="intp")
    if indexes.size == 0:
        raise EarthOverviewError("invalid_dataset", "Latitude band is empty")
    if indexes[0] < 0 or indexes[-1] >= latitude_values.size:
        raise EarthOverviewError("invalid_dataset", "Latitude band indexes are out of range")
    weights = latitude_band_weights(latitude_cell_edges(latitude_values, bounds))[indexes]
    selected = values[:, indexes, :]
    return np.einsum("tij,i->t", selected, weights) / (selected.shape[2] * weights.sum())


def seasonal_extremes(values: np.ndarray, dates: Sequence[str]) -> dict:
    """Warmest/coldest day of a daily series, ties resolved to the earliest date."""
    series = np.asarray(values, dtype="float64")
    labels = [str(value) for value in dates]
    if series.ndim != 1:
        raise EarthOverviewError("invalid_dataset", "Series must be one dimensional")
    if series.size != len(labels):
        raise EarthOverviewError("invalid_dataset", "Series and dates do not match")
    if series.size == 0:
        raise EarthOverviewError("invalid_dataset", "Series is empty")
    if not np.isfinite(series).all():
        raise EarthOverviewError("invalid_dataset", "Series contains invalid values")
    # ``argmax``/``argmin`` return the first occurrence, which is the earliest
    # date for a tie.
    max_index = int(np.argmax(series))
    min_index = int(np.argmin(series))
    return {
        "max_value": float(series[max_index]),
        "max_date": labels[max_index],
        "min_value": float(series[min_index]),
        "min_date": labels[min_index],
        "peak_to_peak": float(series[max_index] - series[min_index]),
    }


def pearson_summary(x: Any, y: Any) -> dict:
    """Pearson correlation with an explicit reason instead of a fake ``0.0``.

    Fewer than three paired samples is ``insufficient_samples``; a series with
    zero variance is ``constant_series``. In both cases ``r`` is ``None``.
    """
    left = np.asarray(x, dtype="float64").reshape(-1)
    right = np.asarray(y, dtype="float64").reshape(-1)
    if left.size != right.size:
        raise EarthOverviewError("invalid_dataset", "Paired series must have equal length")
    if left.size and (not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise EarthOverviewError("invalid_dataset", "Paired series contains invalid values")
    count = int(left.size)
    if count < 3:
        return {"r": None, "n": count, "reason": "insufficient_samples"}
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_norm = float(np.sqrt(np.dot(left_centered, left_centered)))
    right_norm = float(np.sqrt(np.dot(right_centered, right_centered)))
    if left_norm == 0.0 or right_norm == 0.0:
        # A constant series has no linear relationship at all: never report 0.0,
        # which would read as "uncorrelated".
        return {"r": None, "n": count, "reason": "constant_series"}
    r = float(np.dot(left_centered, right_centered) / (left_norm * right_norm))
    return {"r": float(np.clip(r, -1.0, 1.0)), "n": count, "reason": None}


def lagged_correlation(driver: Any, target: Any, max_lag: int = 30) -> list[dict]:
    """Correlate ``driver[t]`` with ``target[t + k]`` for ``k`` in ``-max_lag..max_lag``.

    A positive lag therefore means the driver *leads* the target by that many
    days. Only genuinely overlapping samples are paired: the series are never
    wrapped and never padded with fabricated values, so ``n`` shrinks by
    ``abs(k)`` and reaches ``insufficient_samples`` for short overlaps.
    """
    driver_values = np.asarray(driver, dtype="float64").reshape(-1)
    target_values = np.asarray(target, dtype="float64").reshape(-1)
    if driver_values.size != target_values.size:
        raise EarthOverviewError("invalid_dataset", "Paired series must have equal length")
    if driver_values.size and (
        not np.isfinite(driver_values).all() or not np.isfinite(target_values).all()
    ):
        raise EarthOverviewError("invalid_dataset", "Paired series contains invalid values")
    step = int(max_lag)
    if step < 0:
        raise EarthOverviewError("invalid_dataset", "max_lag must not be negative")
    size = int(driver_values.size)
    results = []
    for k in range(-step, step + 1):
        first = max(0, -k)
        last = min(size, size - k)
        if last <= first:
            pair_driver = driver_values[0:0]
            pair_target = target_values[0:0]
        else:
            pair_driver = driver_values[first:last]
            pair_target = target_values[first + k:last + k]
        summary = pearson_summary(pair_driver, pair_target)
        results.append({
            "lag_days": int(k),
            "r": summary["r"],
            "n": summary["n"],
            "reason": summary["reason"],
        })
    return results


def linear_regression(x: Any, y: Any) -> Optional[dict]:
    """Ordinary least squares slope/intercept, or ``None`` when undefined.

    ``None`` is returned for fewer than three samples and for a constant
    predictor; the slope is then genuinely undetermined rather than zero.
    """
    predictor = np.asarray(x, dtype="float64").reshape(-1)
    response = np.asarray(y, dtype="float64").reshape(-1)
    if predictor.size != response.size:
        raise EarthOverviewError("invalid_dataset", "Paired series must have equal length")
    if predictor.size < 3:
        return None
    if not np.isfinite(predictor).all() or not np.isfinite(response).all():
        raise EarthOverviewError("invalid_dataset", "Paired series contains invalid values")
    mean_x = float(predictor.mean())
    mean_y = float(response.mean())
    centered_x = predictor - mean_x
    denominator = float(np.dot(centered_x, centered_x))
    if denominator == 0.0:
        return None
    slope = float(np.dot(centered_x, response - mean_y) / denominator)
    return {"slope": slope, "intercept": float(mean_y - slope * mean_x)}


def zscore_series(values: Any) -> dict:
    """Population z-score (``ddof=0``) with an explicit constant-series reason."""
    series = np.asarray(values, dtype="float64").reshape(-1)
    if series.size and not np.isfinite(series).all():
        raise EarthOverviewError("invalid_dataset", "Series contains invalid values")
    if series.size == 0:
        return {"values": [], "reason": "insufficient_samples"}
    std = float(series.std())
    if std == 0.0:
        # Never emit a division by zero or a fabricated zero series.
        return {"values": [None for _ in range(int(series.size))], "reason": "constant_series"}
    return {"values": [float(value) for value in (series - series.mean()) / std], "reason": None}


def compute_spatial_diagnostics(
    field_tij: np.ndarray, latitude: np.ndarray, bounds: Optional[Sequence[float]] = None
) -> dict:
    """Annual anomaly field plus its area-weighted band statistics.

    The reference for every grid point is the equal-longitude mean of its own
    row, so ``sum_j anomaly[i, j]`` is zero for every latitude row by
    construction. ``rms`` uses the spherical cell weights restricted to the
    band rows.
    """
    values = _finite_array(field_tij, name="Field", ndim=3)
    latitude_values = np.asarray(latitude, dtype="float64")
    if values.shape[1] != latitude_values.size:
        raise EarthOverviewError("invalid_dataset", "Latitude axis does not match the field")
    annual_mean = values.mean(axis=0, dtype="float64")
    if annual_mean.size and not np.isfinite(annual_mean).all():
        raise EarthOverviewError("invalid_dataset", "Annual mean contains invalid values")
    zonal_mean = annual_mean.mean(axis=1, keepdims=True)
    anomaly = annual_mean - zonal_mean
    if anomaly.size and not np.isfinite(anomaly).all():
        raise EarthOverviewError("invalid_dataset", "Anomaly contains invalid values")

    membership = _band_membership(latitude_values)
    band_weights = latitude_band_weights(latitude_cell_edges(latitude_values, bounds))
    bands = []
    for definition in band_definitions(latitude_values):
        band_id = definition["id"]
        indexes = np.asarray(membership[band_id], dtype="intp")
        if indexes.size == 0:
            bands.append({
                "id": band_id,
                "rms": None,
                "peak_to_peak": None,
                "grid_point_count": 0,
            })
            continue
        selected = anomaly[indexes, :]
        weights = band_weights[indexes][:, None]
        rms = float(np.sqrt(np.sum(weights * selected ** 2) / np.sum(weights)))
        bands.append({
            "id": band_id,
            "rms": rms,
            "peak_to_peak": float(selected.max() - selected.min()),
            "grid_point_count": int(indexes.size),
        })
    return {"anomaly": anomaly, "bands": bands}


# ── JSON shaping ───────────────────────────────────────────────────────

def _finite_or_none(value: Any) -> Optional[float]:
    """Plain float, or ``None``. ``NaN``/``Inf`` are never serialized."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_list(values: Any) -> list:
    result = []
    for value in np.asarray(values, dtype="float64").reshape(-1):
        result.append(_finite_or_none(value))
    return result


def _finite_matrix(values: Any) -> list[list]:
    matrix = np.atleast_2d(np.asarray(values, dtype="float64"))
    return [_finite_list(row) for row in matrix]


def _iso_dates(dates: np.ndarray) -> list[str]:
    return [str(value) for value in dates]


def _reason_from_summary(summary: Mapping[str, Any]) -> Optional[str]:
    return summary["reason"]


def _processing_text(processing: Any) -> Optional[str]:
    """Flatten the manifest ``processing`` block into one short description.

    Later manifests publish it as a mapping of method names; the public context
    reports a single string, so the pairs are joined in a stable order. No value
    here is a path: the manifest may only name the fixed data file.
    """
    if processing is None:
        return None
    if isinstance(processing, str):
        return processing or None
    if isinstance(processing, Mapping):
        parts = [
            f"{key}={value}"
            for key, value in processing.items()
            if isinstance(key, str) and isinstance(value, (str, int, float))
        ]
        return "; ".join(parts) or None
    return None


class EarthResearchService:
    """Read-only scientific analysis over one verified Earth release."""

    def __init__(self, registry: DatasetRegistry):
        self._registry = registry
        # Re-entrant: derived accessors compose (the context builder calls the
        # band and colour-range accessors, which take the same lock).
        self._lock = threading.RLock()
        self._cached_fingerprint: Optional[str] = None
        # Derived per-year aggregates keyed by (year, variable) and by year.
        self._variable_cache: dict[tuple[int, str], dict] = {}
        self._year_cache: dict[int, dict] = {}

    # ── public API ─────────────────────────────────────────────────────
    def get_context(self, dataset_id: str, expected_fingerprint: Optional[str] = None) -> dict:
        """Identity, geometry and declared capabilities of the release."""
        release = self._release(dataset_id, expected_fingerprint)
        return self._context(release)

    def get_research_suite(
        self, dataset_id: str, year: int, expected_fingerprint: Optional[str] = None
    ) -> dict:
        """Seasonal, regional, band and relationship aggregate series for one year."""
        release = self._release(dataset_id, expected_fingerprint)
        year = self._require_year(year)
        dates = self._year_dates(release, year)
        day_count = len(dates)

        seasonal: dict[str, dict] = {}
        regional_series: dict[str, list] = {}
        band_series: dict[str, dict] = {band_id: {} for band_id in BAND_IDS}
        extremes: dict[str, dict] = {band_id: {} for band_id in BAND_IDS}
        zscore: dict[str, dict] = {}
        global_series: dict[str, np.ndarray] = {}
        per_band: dict[str, dict[str, np.ndarray]] = {}

        for variable in VARIABLE_IDS:
            data = self._variable_data(release, year, variable)
            units = data["units"]
            global_series[variable] = data["global"]
            per_band[variable] = data["bands"]
            regional_series[variable] = _finite_list(data["global"])
            seasonal[variable] = {
                # z[i][t]: latitude rows in ascending order, then the daily series.
                "z": _finite_matrix(self._seasonal_display(release, data["zonal"]).T),
                "units": units,
                "aggregation": SEASONAL_AGGREGATION,
            }
            score = zscore_series(data["global"])
            zscore[variable] = {"values": score["values"], "reason": score["reason"]}
            for band_id in BAND_IDS:
                series = data["bands"][band_id]
                band_series[band_id][variable] = (
                    _finite_list(series) if series is not None else None
                )
                extremes[band_id][variable] = (
                    seasonal_extremes(series, dates)
                    if series is not None
                    else {
                        "max_value": None, "max_date": None,
                        "min_value": None, "min_date": None,
                        "peak_to_peak": None,
                    }
                )

        relationships = {}
        for scope_id in ("global",) + BAND_IDS:
            relationships[scope_id] = self._relationships(
                global_series, per_band, scope_id
            )

        return {
            **self._identity(release),
            "year": int(year),
            "start": dates[0],
            "end": dates[-1],
            "day_count": day_count,
            "dates": dates,
            "latitude": [float(value) for value in (
                release.latitude.reshape(-1, 4).mean(axis=1) if self._is_threehour(release) else release.latitude)],
            "bands": self._bands(release),
            "variables": [
                {"id": str(declared["id"]), "units": str(declared["units"])}
                for declared in release.metadata["variables"]
            ],
            "aggregation": {
                "regional": REGIONAL_AGGREGATION,
                "seasonal": SEASONAL_AGGREGATION,
                "band": BAND_AGGREGATION,
                "scope": "global",
            },
            "seasonal": seasonal,
            "regional_series": regional_series,
            "band_series": band_series,
            "extremes": extremes,
            "zscore": zscore,
            "relationships": relationships,
        }

    def get_spatial_diagnostics(
        self,
        dataset_id: str,
        year: int,
        variable: str,
        expected_fingerprint: Optional[str] = None,
    ) -> dict:
        """Annual anomaly field and per-band area-weighted RMS for one variable."""
        release = self._release(dataset_id, expected_fingerprint)
        year = self._require_year(year)
        variable_id, units = self._require_variable(release, variable)
        if self._is_threehour(release):
            field = self._variable_data(release, year, variable_id)["annual"][None]
        else:
            first, last = self._year_bounds(release, year)
            field = np.asarray(release.fields[variable_id], dtype="float64")[first:last + 1]
        diagnostics = compute_spatial_diagnostics(
            field, release.latitude, self._latitude_bounds(release)
        )
        anomaly = diagnostics["anomaly"]
        latitude, longitude = release.latitude, release.longitude
        render_meta = {}
        if self._is_threehour(release):
            from services.earth_3hourly_overview import render_block_mean
            anomaly = render_block_mean(anomaly, release.latitude)
            latitude = release.latitude.reshape(-1, 4).mean(axis=1)
            longitude = release.longitude.reshape(-1, 4).mean(axis=1)
            render_meta = {"source_grid_shape": [240, 480], "render_grid_shape": [60, 120],
                           "render_method": "spherical_cell_area_mean_4x4"}
        bound = _finite_or_none(np.abs(anomaly).max()) if anomaly.size else None
        if bound is None or bound == 0.0:
            color_range = {"min": 0.0, "max": 0.0, "centered_on_zero": True}
        else:
            color_range = {"min": -bound, "max": bound, "centered_on_zero": True}
        return {
            **self._identity(release),
            "year": int(year),
            "variable": variable_id,
            "units": units,
            "lat": [float(value) for value in latitude],
            "lon": [float(value) for value in longitude],
            "anomaly": _finite_matrix(anomaly),
            "reference": SPATIAL_REFERENCE,
            "color_range": color_range,
            "bands": diagnostics["bands"],
            **render_meta,
        }

    def get_polar_dynamics(
        self, dataset_id: str, year: int, expected_fingerprint: Optional[str] = None
    ) -> dict:
        """Daily polar-band series plus the south/north hemisphere contrast."""
        release = self._release(dataset_id, expected_fingerprint)
        year = self._require_year(year)
        dates = self._year_dates(release, year)
        day_count = len(dates)
        definitions = {item["id"]: item for item in self._bands(release)}

        bands = []
        for band_id in ("south_polar", "north_polar"):
            definition = definitions[band_id]
            variables: dict[str, dict] = {}
            empty = definition["grid_point_count"] == 0
            for variable in VARIABLE_IDS:
                if empty:
                    # An empty band has no statistics at all; never zeros.
                    variables[variable] = {
                        "units": self._units(release, variable),
                        "mean": None, "min_value": None, "min_date": None,
                        "max_value": None, "max_date": None, "peak_to_peak": None,
                        "series": [],
                        "reason": "band_empty",
                        "days": 0,
                    }
                    continue
                data = self._variable_data(release, year, variable)
                series = data["bands"][band_id]
                extremes = seasonal_extremes(series, dates)
                variables[variable] = {
                    "units": data["units"],
                    "mean": _finite_or_none(series.mean(dtype="float64")),
                    "min_value": extremes["min_value"],
                    "min_date": extremes["min_date"],
                    "max_value": extremes["max_value"],
                    "max_date": extremes["max_date"],
                    "peak_to_peak": extremes["peak_to_peak"],
                    "series": _finite_list(series),
                    "reason": None,
                    "days": day_count,
                }
            bands.append({**definition, "days": day_count, "variables": variables})

        hemisphere_contrast = {}
        for variable in VARIABLE_IDS:
            south = self._variable_data(release, year, variable)["bands"]["south_polar"]
            north = self._variable_data(release, year, variable)["bands"]["north_polar"]
            if south is None or north is None:
                hemisphere_contrast[variable] = {
                    "n": 0, "r": None, "reason": "band_empty", "mean_difference": None,
                }
                continue
            summary = pearson_summary(south, north)
            hemisphere_contrast[variable] = {
                "n": summary["n"],
                "r": summary["r"],
                "reason": summary["reason"],
                "mean_difference": _finite_or_none(
                    north.mean(dtype="float64") - south.mean(dtype="float64")
                ),
            }

        return {
            **self._identity(release),
            "year": int(year),
            "start": dates[0],
            "end": dates[-1],
            "day_count": day_count,
            "dates": dates,
            "polar_scope": self._polar_scope(release),
            "bands": bands,
            "hemisphere_contrast": hemisphere_contrast,
        }

    def build_insight_summary(
        self,
        dataset_id: str,
        payload: Mapping[str, Any],
        expected_fingerprint: Optional[str] = None,
    ) -> dict:
        """Deterministic statistics digest for the AI insight endpoint.

        The digest is computed from the *real* release for the requested year,
        variable and scope, so the model never has to trust numbers sent by the
        client. Client supplied ``cards`` become supplementary notes only.
        """
        release = self._release(dataset_id, expected_fingerprint)
        year = self._require_year(payload.get("year"))
        variable_id, units = self._require_variable(release, payload.get("variable"))
        scope = payload.get("scope") or "global"
        if not isinstance(scope, str) or scope not in ("global",) + BAND_IDS:
            raise EarthOverviewError("unsupported_scope", "Unsupported analysis scope")
        date = payload.get("date")
        dates = self._year_dates(release, year)
        if date is not None:
            index = self._require_date_in_year(date, dates)
        else:
            index = len(dates) // 2

        data = self._variable_data(release, year, variable_id)
        series = data["global"] if scope == "global" else data["bands"][scope]
        if series is None:
            raise EarthOverviewError(
                "unsupported_scope", "The requested latitude band has no grid rows"
            )
        band_definition = {item["id"]: item for item in self._bands(release)}.get(scope)
        scope_label = (
            "global"
            if scope == "global"
            else str(band_definition.get("label") or scope.replace("_", " "))
        )

        extremes = seasonal_extremes(series, dates)
        cards = [{
            "card": "series",
            "values": {
                "scope": scope,
                "day_count": len(dates),
                "mean": _finite_or_none(series.mean(dtype="float64")),
                "min_value": extremes["min_value"],
                "min_date": extremes["min_date"],
                "max_value": extremes["max_value"],
                "max_date": extremes["max_date"],
                "peak_to_peak": extremes["peak_to_peak"],
                "value_on_date": _finite_or_none(series[index]),
            },
            "notes": [],
        }]
        relationships = self._relationships(
            {variable_id: data["global"]},
            {variable_id: data["bands"]},
            scope,
            targets=tuple(name for name in VARIABLE_IDS if name != variable_id),
        )
        for target, summary in relationships["correlation"]["pairs"].items():
            cards.append({
                "card": "correlation",
                "values": {
                    "variable": variable_id,
                    "against": target,
                    "r": summary["r"],
                    "n": summary["n"],
                    "reason": summary["reason"],
                    "scope": scope,
                },
                "notes": [],
            })
        if scope != "global" and band_definition is not None:
            cards.append({
                "card": "band",
                "values": {
                    "band_id": scope,
                    "min_latitude": band_definition["min_latitude"],
                    "max_latitude": band_definition["max_latitude"],
                    "grid_point_count": band_definition["grid_point_count"],
                },
                "notes": [],
            })
        for note in self._client_notes(payload.get("summary")):
            cards.append({"card": "client_note", "values": {}, "notes": [note]})

        return {
            "planet": PLANET,
            "dataset_id": release.metadata["dataset_id"],
            "dataset_version": release.metadata.get("dataset_version"),
            "dataset_fingerprint": release.metadata["dataset_fingerprint"],
            "source": SOURCE_LABEL,
            "cadence": ("UTC daily means aggregated from 3-hour means" if self._is_threehour(release) else CADENCE),
            "calendar": release.metadata["time"]["calendar"],
            "year": int(year),
            "date_range": {"start": dates[0], "end": dates[-1]},
            "variable": variable_id,
            "units": units,
            "scope": scope,
            "scope_label": scope_label,
            "day_count": len(dates),
            "date": dates[index],
            "requested_date": date if isinstance(date, str) else None,
            "engine": "deterministic_release_digest",
            "cards": cards,
            "units_note": (
                f"{variable_id} in {units}; paired series are same-day "
                f"{REGIONAL_AGGREGATION if scope == 'global' else BAND_AGGREGATION} aggregates"
            ),
        }

    # ── release plumbing ───────────────────────────────────────────────
    def _release(
        self, dataset_id: str, expected_fingerprint: Optional[str]
    ) -> VerifiedEarthRelease:
        """Take the verified release exactly once for this request."""
        fingerprint = expected_fingerprint
        if fingerprint is None:
            # Default to the registry's current verified fingerprint for this id.
            descriptor = self._registry.get_dataset(dataset_id)
            fingerprint = descriptor.get("dataset_fingerprint")
        return self._registry.get_earth_overview_snapshot(dataset_id, fingerprint)

    def _require_variable(self, release: VerifiedEarthRelease, variable: Any) -> tuple[str, str]:
        if not isinstance(variable, str) or variable not in VARIABLE_IDS:
            raise EarthOverviewError("unsupported_variable", "Unsupported variable")
        for declared in release.metadata["variables"]:
            if declared["id"] == variable:
                return variable, str(declared["units"])
        raise EarthOverviewError("invalid_dataset", "Variable is missing from the release")

    def _require_year(self, year: Any) -> int:
        if isinstance(year, bool) or not isinstance(year, int):
            raise EarthOverviewError("year_out_of_range", "year must be an integer", 422)
        return int(year)

    def _year_mask(self, release: VerifiedEarthRelease, year: int) -> np.ndarray:
        labels = np.asarray([str(value) for value in release.dates])
        prefix = f"{year:04d}-"
        mask = np.char.startswith(labels.astype(str), prefix)
        if not mask.any():
            raise EarthOverviewError(
                "year_out_of_range", "Year is outside the available range", 422
            )
        return mask

    def _year_bounds(self, release: VerifiedEarthRelease, year: int) -> tuple[int, int]:
        indexes = np.flatnonzero(self._year_mask(release, year))
        return int(indexes[0]), int(indexes[-1])

    def _year_dates(self, release: VerifiedEarthRelease, year: int) -> list[str]:
        first, last = self._year_bounds(release, year)
        if self._is_threehour(release):
            return _iso_dates(np.unique(release.dates[first:last + 1].astype("datetime64[D]")))
        return _iso_dates(release.dates[first:last + 1])

    def _require_date_in_year(self, date: Any, dates: Sequence[str]) -> int:
        if not isinstance(date, str) or date not in set(dates):
            raise EarthOverviewError(
                "date_out_of_range", "Date is outside the requested year"
            )
        return list(dates).index(date)

    def _latitude_bounds(self, release: VerifiedEarthRelease) -> Optional[Sequence[float]]:
        grid = release.metadata.get("grid") or {}
        bounds = (grid.get("cell_bounds") or {}).get("latitude")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            return bounds
        return None

    def _units(self, release: VerifiedEarthRelease, variable: str) -> str:
        for declared in release.metadata["variables"]:
            if declared["id"] == variable:
                return str(declared["units"])
        raise EarthOverviewError("invalid_dataset", "Variable is missing from the release")

    # ── cached per-year aggregates ─────────────────────────────────────
    def _invalidate(self, release: VerifiedEarthRelease) -> None:
        fingerprint = (release.metadata["dataset_id"], release.metadata.get("dataset_version"),
                       release.metadata["dataset_fingerprint"])
        if self._cached_fingerprint != fingerprint:
            self._variable_cache = {}
            self._year_cache = {}
            self._cached_fingerprint = fingerprint

    def _variable_data(
        self, release: VerifiedEarthRelease, year: int, variable: str
    ) -> dict:
        """Daily aggregate series of one variable for one calendar year.

        ``global`` is the published spherical cell-area mean, ``bands`` holds the
        same mean restricted to each latitude band (``None`` when a band has no
        rows) and ``zonal`` is the equal-longitude mean ``[H, T]``.
        """
        key = (int(year), variable)
        with self._lock:
            self._invalidate(release)
            cached = self._variable_cache.get(key)
            if cached is not None:
                return cached

            if self._is_threehour(release):
                cached = self._threehour_variable_data(release, year, variable)
                self._registry.get_earth_overview_snapshot(release.metadata["dataset_id"], release.metadata["dataset_fingerprint"])
                self._variable_cache[key] = cached
                return cached

            first, last = self._year_bounds(release, year)
            field = np.asarray(release.fields[variable], dtype="float64")[first:last + 1]
            latitude = release.latitude
            bounds = self._latitude_bounds(release)
            zonal = equal_longitude_mean(field)
            global_series = cell_area_mean_series(field, latitude, bounds)
            bands: dict[str, Optional[np.ndarray]] = {}
            membership = _band_membership(latitude)
            for band_id in BAND_IDS:
                indexes = membership[band_id]
                bands[band_id] = (
                    band_area_mean_series(field, latitude, indexes, bounds)
                    if indexes else None
                )
            cached = {
                "units": self._units(release, variable),
                "global": global_series,
                "bands": bands,
                "zonal": zonal,
            }
            self._variable_cache[key] = cached
            return cached

    @staticmethod
    def _is_threehour(release):
        return release.metadata["dataset_id"] == EARTH_DATASET_3HOURLY_ID

    def _threehour_variable_data(self, release, year, variable):
        """Retain daily scalar/zonal statistics and one annual field, never a yearly cube."""
        from services.earth_3hourly_overview import iter_threehour_chunks
        first, last = self._year_bounds(release, year)
        times = release.dates[first:last + 1]
        days, counts = np.unique(times.astype("datetime64[D]"), return_counts=True)
        if (np.any(counts != 8) or np.any(np.diff(times) != np.timedelta64(3, "h"))
                or np.any(times[::8] - days != np.timedelta64(90, "m"))):
            raise EarthOverviewError("incomplete_daily_aggregate", "Daily analysis requires eight consecutive UTC samples per day", 503)
        latitude = release.latitude
        bounds = self._latitude_bounds(release)
        membership = _band_membership(latitude)
        global_values, zonal_values = [], []
        bands = {band: [] if indexes else None for band, indexes in membership.items()}
        annual = np.zeros((240, 480), dtype="float64")
        for _, values in iter_threehour_chunks(release, variable, first, last + 1):
            if len(values) != 8 or not np.isfinite(values).all():
                raise EarthOverviewError("incomplete_daily_aggregate", "Daily analysis requires eight valid samples in every source cell", 503)
            daily = values.mean(axis=0, dtype="float64")
            annual += daily
            global_values.append(cell_area_mean_series(daily[None], latitude, bounds)[0])
            zonal_values.append(equal_longitude_mean(daily[None])[0])
            for band, indexes in membership.items():
                if indexes:
                    bands[band].append(band_area_mean_series(daily[None], latitude, indexes, bounds)[0])
        return {"units": self._units(release, variable), "global": np.asarray(global_values),
                "bands": {band: np.asarray(values) if values is not None else None for band, values in bands.items()},
                "zonal": np.asarray(zonal_values), "annual": annual / len(days)}

    def _seasonal_display(self, release, zonal):
        if not self._is_threehour(release):
            return zonal
        weights = latitude_band_weights(latitude_cell_edges(release.latitude, self._latitude_bounds(release)))
        return np.sum(zonal.reshape(-1, 60, 4) * weights.reshape(1, 60, 4), axis=2) / weights.reshape(60, 4).sum(axis=1)

    def _bands(self, release: VerifiedEarthRelease) -> list[dict]:
        """The five latitude bands of this release, with real sampled extremes."""
        with self._lock:
            self._invalidate(release)
            cached = self._year_cache.get("bands")
            if cached is None:
                cached = []
                for definition in band_definitions(release.latitude):
                    label = str(definition["id"]).replace("_", " ")
                    if definition["min_latitude"] is not None:
                        label = (
                            f"{label} ({definition['min_latitude']:.1f} to "
                            f"{definition['max_latitude']:.1f} deg)"
                        )
                    cached.append({**definition, "label": label})
                self._year_cache["bands"] = cached
            return [
                {**item, "latitude_values": list(item["latitude_values"])}
                for item in cached
            ]

    def _polar_scope(self, release: VerifiedEarthRelease) -> dict:
        """Only the bands whose grid rows sit at or beyond the polar threshold.

        That is ``south_polar`` and ``north_polar``: the mid-latitude bands are
        not part of the polar scope even though they carry a hemisphere label.
        """
        polar = [
            item for item in self._bands(release)
            if item["min_latitude"] is not None
            and min(abs(item["min_latitude"]), abs(item["max_latitude"]))
            >= POLAR_MIN_ABS_LATITUDE
        ]
        return {
            "min_abs_latitude": POLAR_MIN_ABS_LATITUDE,
            "bands": polar,
            "sampling": POLAR_SAMPLING,
            "note": POLAR_NOTE,
        }

    # ── response shaping ───────────────────────────────────────────────
    def _identity(self, release: VerifiedEarthRelease) -> dict:
        metadata = release.metadata
        return {
            "planet": PLANET,
            "dataset_id": metadata["dataset_id"],
            "dataset_version": metadata.get("dataset_version"),
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "schema": metadata.get("schema"),
            **({"frequency_hours": 3, "time_zone": "UTC", "time_aggregation": THREE_HOUR_DAILY_AGGREGATION}
               if self._is_threehour(release) else {}),
        }

    def _variables(self, release: VerifiedEarthRelease) -> list[dict]:
        color_ranges = self._color_ranges(release)
        variables = []
        for declared in release.metadata["variables"]:
            variable = str(declared["id"])
            variables.append({
                "id": variable,
                "label": str(declared.get("label") or VARIABLE_LABELS.get(variable, variable)),
                "units": str(declared["units"]),
                "color_range": color_ranges.get(variable),
            })
        return variables

    def _color_ranges(self, release: VerifiedEarthRelease) -> dict:
        with self._lock:
            self._invalidate(release)
            cached = self._year_cache.get("color_ranges")
            if cached is None:
                cached = self._threehour_color_ranges(release) if self._is_threehour(release) else {
                    name: dataset_color_range(
                        np.asarray(release.fields[name], dtype="float32"), name
                    )
                    for name in VARIABLE_IDS
                }
                self._year_cache["color_ranges"] = cached
            return {key: dict(value) for key, value in cached.items()}

    def _threehour_color_ranges(self, release):
        from services.earth_3hourly_overview import iter_threehour_chunks
        declarations = release.metadata.get("manifest", {}).get("variables", {})
        result = {}
        for name in VARIABLE_IDS:
            entry = declarations.get(name, {})
            low, high = entry.get("min"), entry.get("max")
            if low is None or high is None:
                # Trusted fixtures can omit the manifest. Published releases
                # use already-verified global extrema and need no volume scan.
                low, high = np.inf, -np.inf
                for _, values in iter_threehour_chunks(release, name):
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        low, high = min(low, float(finite.min())), max(high, float(finite.max()))
                if not np.isfinite(low) or not np.isfinite(high):
                    low, high = 0., 1.
            if name in WIND_VARIABLE_IDS:
                bound = max(abs(low), abs(high)); low, high = -bound, bound
            result[name] = {"min": float(low), "max": float(high), "scope": "dataset", "centered_on_zero": name in WIND_VARIABLE_IDS}
        return result

    def _source_meta(self, release: VerifiedEarthRelease) -> dict:
        metadata = release.metadata
        manifest = metadata.get("manifest") if isinstance(metadata.get("manifest"), dict) else {}
        source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        files_count = source.get("daily_file_count")
        return {
            "source": SOURCE_LABEL,
            "cadence": ("3-hour mean (UTC)" if self._is_threehour(release) else CADENCE),
            "schema": metadata.get("schema"),
            "dataset_version": metadata.get("dataset_version"),
            "source_files_count": int(files_count) if isinstance(files_count, int) else None,
            "source_sha256": manifest.get("source_sha256"),
            "processing": _processing_text(manifest.get("processing")),
            "calendar": metadata["time"].get("calendar", "proleptic_gregorian"),
            **({"frequency_hours": 3, "time_zone": "UTC", "analysis_aggregation": THREE_HOUR_DAILY_AGGREGATION}
               if self._is_threehour(release) else {}),
        }

    def _time_model(self, release: VerifiedEarthRelease) -> dict:
        time = release.metadata["time"]
        calendar = time.get("calendar", "proleptic_gregorian")
        years: list[int] = []
        for value in release.dates:
            year = int(str(value)[:4])
            if year not in years:
                years.append(year)
        return {
            "kind": "iso-datetime" if self._is_threehour(release) else "iso-date",
            "calendar": calendar,
            "start": (np.datetime_as_string(release.dates[0], unit="s") + "Z") if self._is_threehour(release) else str(time["start"]),
            "end": (np.datetime_as_string(release.dates[-1], unit="s") + "Z") if self._is_threehour(release) else str(time["end"]),
            "count": len(release.dates) if self._is_threehour(release) else int(time["count"]),
            "step": 3 if self._is_threehour(release) else int(time["step"]),
            "step_unit": "hour" if self._is_threehour(release) else "day",
            "years": years,
            **({"frequency_hours": 3, "time_zone": "UTC", "timestamp_rule": "interval_center"}
               if self._is_threehour(release) else {}),
        }

    def _geometry(self, release: VerifiedEarthRelease) -> dict:
        grid = release.metadata["grid"]
        cell_bounds = grid.get("cell_bounds") or {}
        return {
            "shape": [int(grid["shape"][0]), int(grid["shape"][1])],
            "dimension_order": ["lat", "lon"],
            "lat_centers": [float(value) for value in release.latitude],
            "lon_centers": [float(value) for value in release.longitude],
            "lat_bounds": [float(value) for value in grid["latitude_range"]],
            "lon_bounds": [float(value) for value in grid["longitude_range"]],
            "cell_bounds": {
                "latitude": [float(value) for value in cell_bounds.get("latitude", [])],
                "longitude": [float(value) for value in cell_bounds.get("longitude", [])],
            },
            "latitude_step": float(grid["latitude_step"]),
            "longitude_step": float(grid["longitude_step"]),
            "wrap_longitude": bool(grid["wrap_longitude"]),
            "coverage": str(grid["coverage"]),
        }

    def _context(self, release: VerifiedEarthRelease) -> dict:
        with self._lock:
            self._invalidate(release)
            cached = self._year_cache.get("context")
            if cached is None:
                cached = {
                    **self._identity(release),
                    "source_meta": self._source_meta(release),
                    "time": self._time_model(release),
                    "variables": self._variables(release),
                    "geometry": self._geometry(release),
                    "capabilities": {
                        "field": True,
                        "playback": True,
                        "pointProbe": True,
                        "polar": True,
                        "diurnal": False,
                        "researchSuite": True,
                        "spatialDiagnostics": True,
                        "aiInsight": True,
                    },
                    "unavailable": {"diurnal": "three_hourly_diurnal_analysis_not_implemented" if self._is_threehour(release) else DIURNAL_UNAVAILABLE_REASON},
                    "polar_scope": self._polar_scope(release),
                    "limitations": list(release.metadata.get("limitations", [])) + ([
                        "Observation fields and point series are 3-hour UTC means on the native 0.75 degree grid",
                        "Annual analyses aggregate eight consecutive three-hour samples into UTC daily means",
                        "Diurnal analysis is not implemented; daily analysis aggregates discard within-day variation",
                        *RESEARCH_LIMITATIONS[2:],
                    ] if self._is_threehour(release) else RESEARCH_LIMITATIONS),
                }
                self._year_cache["context"] = cached
            # Deep copy the mutable containers so callers cannot edit the cache.
            return {
                **cached,
                "source_meta": dict(cached["source_meta"]),
                "time": {**cached["time"], "years": list(cached["time"]["years"])},
                "variables": [
                    {**item, "color_range": dict(item["color_range"])}
                    for item in cached["variables"]
                ],
                "geometry": {
                    **cached["geometry"],
                    "lat_centers": list(cached["geometry"]["lat_centers"]),
                    "lon_centers": list(cached["geometry"]["lon_centers"]),
                    "lat_bounds": list(cached["geometry"]["lat_bounds"]),
                    "lon_bounds": list(cached["geometry"]["lon_bounds"]),
                    "cell_bounds": {
                        key: list(value)
                        for key, value in cached["geometry"]["cell_bounds"].items()
                    },
                },
                "capabilities": dict(cached["capabilities"]),
                "unavailable": dict(cached["unavailable"]),
                "polar_scope": {
                    **cached["polar_scope"],
                    "bands": [dict(item) for item in cached["polar_scope"]["bands"]],
                },
                "limitations": list(cached["limitations"]),
            }

    def _relationships(
        self,
        global_series: Mapping[str, np.ndarray],
        per_band: Mapping[str, Mapping[str, Optional[np.ndarray]]],
        scope_id: str,
        targets: Optional[Sequence[str]] = None,
    ) -> dict:
        """Correlation matrix plus the solar/temperature ozone relationships."""
        selected = {}
        for variable in VARIABLE_IDS:
            if scope_id == "global":
                series = global_series.get(variable)
            else:
                band = per_band.get(variable) or {}
                series = band.get(scope_id)
            if series is not None:
                selected[variable] = np.asarray(series, dtype="float64")

        matrix_r: list[list] = []
        matrix_n: list[list] = []
        matrix_reason: list[list] = []
        for left in VARIABLE_IDS:
            row_r: list = []
            row_n: list = []
            row_reason: list = []
            for right in VARIABLE_IDS:
                if left in selected and right in selected:
                    summary = pearson_summary(selected[left], selected[right])
                else:
                    summary = {"r": None, "n": 0, "reason": "band_empty"}
                row_r.append(summary["r"])
                row_n.append(summary["n"])
                row_reason.append(_reason_from_summary(summary))
            matrix_r.append(row_r)
            matrix_n.append(row_n)
            matrix_reason.append(row_reason)

        pairs = {}
        for target in (targets if targets is not None else VARIABLE_IDS):
            if target == "TO3":
                continue
            if "TO3" in selected and target in selected:
                summary = pearson_summary(selected[target], selected["TO3"])
                pairs[target] = {
                    "r": summary["r"], "n": summary["n"], "reason": summary["reason"],
                }

        scope_block = {
            "kind": "global" if scope_id == "global" else "band",
            "band_id": None if scope_id == "global" else scope_id,
            "units_note": (
                "Paired series are same-day spherical cell-area daily means in "
                "their own units; ozone is DU."
                if scope_id == "global"
                else (
                    "Paired series are same-day cell-area weighted means of the "
                    "named latitude band in their own units; ozone is DU."
                )
            ),
        }
        return {
            "scope": scope_block,
            "correlation": {
                "variable_order": list(VARIABLE_IDS),
                "r": matrix_r,
                "n": matrix_n,
                "reason": matrix_reason,
                "pairs": pairs,
            },
            "solar_ozone": self._driver_relationship(
                "SWGDN", selected, slope_units="DU/(W m-2)"
            ),
            "temperature_ozone": self._driver_relationship(
                "T2M", selected, slope_units="DU/K"
            ),
        }

    def _driver_relationship(
        self, driver: str, selected: Mapping[str, np.ndarray], *, slope_units: str
    ) -> dict:
        if driver not in selected or "TO3" not in selected:
            return {
                "driver": driver, "target": "TO3", "slope_units": slope_units,
                "r": None, "n": 0, "reason": "band_empty",
                "regression": None, "lag": [],
            }
        x = np.asarray(selected[driver], dtype="float64")
        y = np.asarray(selected["TO3"], dtype="float64")
        summary = pearson_summary(x, y)
        return {
            "driver": driver,
            "target": "TO3",
            "slope_units": slope_units,
            "r": summary["r"],
            "n": summary["n"],
            "reason": summary["reason"],
            "regression": linear_regression(x, y),
            "lag": lagged_correlation(x, y),
        }

    @staticmethod
    def _client_notes(summary: Any) -> list[str]:
        """Flatten client supplied card notes into short supplementary strings."""
        if not isinstance(summary, dict):
            return []
        notes: list[str] = []
        cards = summary.get("cards")
        if not isinstance(cards, list):
            return []
        for card in cards:
            if not isinstance(card, dict):
                continue
            raw = card.get("notes")
            if not isinstance(raw, list):
                continue
            for item in raw:
                text = str(item).strip()
                if text:
                    notes.append(text[:280])
        return notes[:20]


__all__ = [
    "BAND_IDS",
    "DIURNAL_UNAVAILABLE_REASON",
    "EarthResearchService",
    "POLAR_MIN_ABS_LATITUDE",
    "RESEARCH_LIMITATIONS",
    "band_area_mean_series",
    "band_definitions",
    "cell_area_mean_series",
    "compute_spatial_diagnostics",
    "equal_longitude_mean",
    "lagged_correlation",
    "latitude_band_weights",
    "latitude_cell_edges",
    "linear_regression",
    "pearson_summary",
    "seasonal_extremes",
    "zscore_series",
]
