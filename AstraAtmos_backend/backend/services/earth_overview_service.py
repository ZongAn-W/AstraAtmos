"""Numeric service behind the read-only 2D Earth overview APIs.

Reads the verified release snapshot from :class:`DatasetRegistry` and answers
three questions: one field, the coverage-wide mean, and one native point's
series. Daily responses retain their original contract; three-hour responses
use UTC timestamps and bounded reads with a separate display aggregation.

All values are original physical units taken straight from the NetCDF variable;
nothing here normalizes, interpolates, regrids or converts units.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
from typing import Any, Mapping, Optional

import numpy as np

from services.dataset_registry import DatasetRegistry
from services.earth_dataset_metadata import VerifiedEarthRelease
from services.earth_scales import color_range

VARIABLE_IDS = ("TO3", "U10M", "V10M", "T2M", "SWGDN")
WIND_VARIABLE_IDS = ("U10M", "V10M")

AGGREGATION = "cos_lat_sample_mean"
GLOBAL_AGGREGATION = "spherical_cell_area_mean"
SELECTION = "nearest_grid_point"

_ISO_DATE_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


class EarthOverviewError(ValueError):
    """An overview request error carrying a stable code and HTTP status."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def require_date_index(value: Any, dates: np.ndarray) -> int:
    """Map a strict ``YYYY-MM-DD`` string to its index in the daily timeline.

    Timestamps, Mars years, Ls values and integer sample indexes are rejected.
    A well formed date that this release simply does not contain - including a
    non-leap ``02-29`` - is reported as out of range and never clamped.
    """
    match = _ISO_DATE_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise EarthOverviewError("invalid_date", "Use YYYY-MM-DD")
    year, month, day = (int(part) for part in match.groups())
    if not 1 <= month <= 12:
        # Month 0 or 13 is not a date at all.
        raise EarthOverviewError("invalid_date", "Invalid calendar date")
    try:
        parsed = dt.date(year, month, day)
    except ValueError:
        # A real month with an impossible day, such as a non-leap ``02-29``:
        # well formed input for a day this release does not contain.
        raise EarthOverviewError(
            "date_out_of_range", "Date is outside the available range"
        ) from None
    target = np.datetime64(parsed, "D")
    index = int(np.searchsorted(dates, target))
    if index == len(dates) or dates[index] != target:
        # A real day outside the published period; never clamped.
        raise EarthOverviewError(
            "date_out_of_range", "Date is outside the available range"
        )
    return index


def nearest_grid_index(axis: np.ndarray, value: Any, bounds: Optional[tuple[float, float]] = None) -> int:
    """Return the nearest sample index, rejecting points outside coverage.

    Ties resolve to the smaller index on both ends of the stack. The point is
    never clamped to the edge and longitude is never wrapped.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EarthOverviewError(
            "point_outside_coverage", "Point is outside the data coverage"
        ) from exc
    lower, upper = bounds if bounds is not None else (float(axis[0]), float(axis[-1]))
    if not np.isfinite(numeric) or numeric < lower or numeric > upper:
        raise EarthOverviewError(
            "point_outside_coverage", "Point is outside the data coverage"
        )
    return int(np.argmin(np.abs(axis.astype(np.float64) - numeric)))


def regional_sample_mean(field: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Cosine-latitude weighted mean over the sampled grid points.

    This weights each sampled latitude row by ``cos(lat)``; it is not a
    conservative area integral and not a global total. The full coverage area is
    always used, so the result must be described as the coverage mean.
    """
    values = np.asarray(field)
    weights = np.cos(np.deg2rad(np.asarray(latitude, dtype="float64")))
    if values.ndim != 3:
        raise EarthOverviewError("invalid_dataset", "Field must be [time, lat, lon]")
    if weights.shape[0] != values.shape[1]:
        raise EarthOverviewError("invalid_dataset", "Latitude axis does not match the field")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise EarthOverviewError("invalid_dataset", "Field or latitude contains invalid values")
    if weights.sum() == 0:
        raise EarthOverviewError("invalid_dataset", "Latitude weights sum to zero")
    means = np.einsum(
        "thw,h->t", values.astype("float64"), weights
    ) / (values.shape[2] * weights.sum())
    return means


def regional_cell_area_mean(field: np.ndarray, latitude: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Area mean for equal longitude cells and explicit latitude cell bounds."""
    values = np.asarray(field, dtype="float64")
    if values.ndim != 3 or values.shape[1] != len(latitude):
        raise EarthOverviewError("invalid_dataset", "Field and latitude axis do not match")
    lower, upper = float(bounds[0]), float(bounds[1])
    edges = np.linspace(lower, upper, len(latitude) + 1)
    weights = np.diff(np.sin(np.deg2rad(edges)))
    if np.any(weights <= 0):
        raise EarthOverviewError("invalid_dataset", "Latitude cell bounds are invalid")
    return np.einsum("tij,i->t", values, weights) / (values.shape[2] * weights.sum())


def _coverage(metadata: Mapping[str, Any]) -> dict:
    grid = metadata["grid"]
    return {
        "latitude_range": [float(grid["latitude_range"][0]), float(grid["latitude_range"][1])],
        "longitude_range": [float(grid["longitude_range"][0]), float(grid["longitude_range"][1])],
        "wrap_longitude": bool(grid["wrap_longitude"]),
    }


class EarthOverviewService:
    """Read-only numeric layer for the Earth overview endpoints."""

    def __init__(self, registry: DatasetRegistry):
        self._registry = registry
        self._lock = threading.Lock()
        # Derived series/statistics keyed by (fingerprint, variable); cleared
        # whenever the verified release identity changes.
        self._series_cache: dict[tuple, dict] = {}
        self._cached_fingerprint: Optional[str] = None
        from services.earth_3hourly_overview import ThreehourOverview
        self._threehour = ThreehourOverview(self)

    # ── public API ─────────────────────────────────────────────────────
    def get_field(
        self, dataset_id: str, expected_fingerprint: str, date: Optional[str] = None,
        variable: str = "TO3", *, timestamp: Optional[str] = None, render_stride: int = 4,
    ) -> dict:
        release = self._release(dataset_id, expected_fingerprint)
        variable_id, units = self._require_variable(release, variable)
        if release.metadata["dataset_id"] == "earth_merra2_3hourly_v1":
            return self._threehour.field(release, variable_id, units, timestamp, render_stride)
        dates = release.dates
        index = require_date_index(date, dates)
        field = np.asarray(release.fields[variable_id][index], dtype="float32")
        if not np.isfinite(field).all():
            raise EarthOverviewError("invalid_dataset", "Field contains invalid values")

        color_range = self._color_range(release, variable_id)
        statistics = self._statistics(release, variable_id, index)
        return {
            **self._identity(release, variable_id, units),
            "date": str(dates[index]),
            "calendar": release.metadata["time"]["calendar"],
            "lat": [float(value) for value in release.latitude],
            "lon": [float(value) for value in release.longitude],
            "dimension_order": ["lat", "lon"],
            "field": [[float(value) for value in row] for row in field],
            "coverage": _coverage(release.metadata),
            "color_range": color_range,
            "statistics": statistics,
        }

    def get_regional_series(
        self,
        dataset_id: str,
        expected_fingerprint: str,
        variable: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict:
        release = self._release(dataset_id, expected_fingerprint)
        variable_id, units = self._require_variable(release, variable)
        first, last = self._date_range(release, start, end)
        if release.metadata["dataset_id"] == "earth_merra2_3hourly_v1":
            return self._threehour.regional(release, variable_id, units, first, last)
        series = self._regional_series(release, variable_id)
        return {
            **self._identity(release, variable_id, units),
            "start": str(release.dates[first]),
            "end": str(release.dates[last]),
            "dates": [str(value) for value in release.dates[first:last + 1]],
            "values": [float(value) for value in series[first:last + 1]],
            "aggregation": (GLOBAL_AGGREGATION if release.metadata["grid"].get("coverage") == "global" else AGGREGATION),
            "coverage": _coverage(release.metadata),
        }

    def get_point_series(
        self,
        dataset_id: str,
        expected_fingerprint: str,
        variable: str,
        lat: Any,
        lon: Any,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict:
        release = self._release(dataset_id, expected_fingerprint)
        variable_id, units = self._require_variable(release, variable)
        first, last = self._date_range(release, start, end)
        if release.metadata["dataset_id"] == "earth_merra2_3hourly_v1":
            return self._threehour.point(release, variable_id, units, lat, lon, first, last)
        grid = release.metadata["grid"]
        lat_index = nearest_grid_index(release.latitude, lat, tuple(grid["cell_bounds"]["latitude"]))
        lon_index = nearest_grid_index(release.longitude, lon, tuple(grid["cell_bounds"]["longitude"]))
        values = np.asarray(release.fields[variable_id], dtype="float32")[:, lat_index, lon_index]
        if not np.isfinite(values).all():
            raise EarthOverviewError("invalid_dataset", "Series contains invalid values")
        return {
            **self._identity(release, variable_id, units),
            "start": str(release.dates[first]),
            "end": str(release.dates[last]),
            "requested": {"lat": float(lat), "lon": float(lon)},
            "grid_point": {
                "lat": float(release.latitude[lat_index]),
                "lon": float(release.longitude[lon_index]),
                "lat_index": lat_index,
                "lon_index": lon_index,
            },
            "selection": SELECTION,
            "dates": [str(value) for value in release.dates[first:last + 1]],
            "values": [float(value) for value in values[first:last + 1]],
        }

    # ── internals ──────────────────────────────────────────────────────
    def _release(self, dataset_id: str, expected_fingerprint: str) -> VerifiedEarthRelease:
        release = self._registry.get_earth_overview_snapshot(dataset_id, expected_fingerprint)
        if release.metadata["dataset_id"] == "earth_merra2_3hourly_v1":
            from services.earth_3hourly_overview import _source_path
            _source_path(release)
        return release
    def _require_variable(self, release: VerifiedEarthRelease, variable: Any) -> tuple[str, str]:
        if not isinstance(variable, str) or variable not in VARIABLE_IDS:
            raise EarthOverviewError("unsupported_variable", "Unsupported variable")
        for declared in release.metadata["variables"]:
            if declared["id"] == variable:
                return variable, str(declared["units"])
        raise EarthOverviewError("invalid_dataset", "Variable is missing from the release")

    def _date_range(
        self, release: VerifiedEarthRelease, start: Optional[str], end: Optional[str]
    ) -> tuple[int, int]:
        dates = release.dates
        index_for = require_date_index
        if release.metadata["dataset_id"] == "earth_merra2_3hourly_v1":
            from services.earth_3hourly_overview import require_timestamp_index
            index_for = require_timestamp_index
        first = index_for(start, dates) if start is not None else 0
        last = index_for(end, dates) if end is not None else len(dates) - 1
        if first > last:
            raise EarthOverviewError("invalid_date_range", "start must not be after end")
        return first, last

    def _identity(self, release: VerifiedEarthRelease, variable: str, units: str) -> dict:
        metadata = release.metadata
        return {
            "dataset_id": metadata["dataset_id"],
            "dataset_version": metadata["dataset_version"],
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "planet": "earth",
            "variable": variable,
            "units": units,
        }

    def _series(self, release: VerifiedEarthRelease) -> dict:
        """Return per-variable derived arrays for the current release identity.

        The caller must already hold ``self._lock``.
        """
        fingerprint = release.metadata["dataset_fingerprint"]
        if self._cached_fingerprint != fingerprint:
            # A published change invalidates every derived series.
            self._series_cache = {}
            self._cached_fingerprint = fingerprint
        return self._series_cache

    def _regional_series(self, release: VerifiedEarthRelease, variable: str) -> np.ndarray:
        with self._lock:
            cache = self._series(release)
            key = ("regional", variable)
            cached = cache.get(key)
            if cached is None:
                if release.metadata["grid"].get("coverage") == "global":
                    cached = regional_cell_area_mean(
                        np.asarray(release.fields[variable], dtype="float32"), release.latitude,
                        tuple(release.metadata["grid"]["cell_bounds"]["latitude"]),
                    )
                else:
                    cached = regional_sample_mean(
                        np.asarray(release.fields[variable], dtype="float32"), release.latitude
                    )
                cache[key] = cached
            return cached

    def _statistics(
        self, release: VerifiedEarthRelease, variable: str, index: int
    ) -> dict:
        field = np.asarray(release.fields[variable][index], dtype="float32")
        means = self._regional_series(release, variable)
        return {
            "min": float(field.min()),
            "max": float(field.max()),
            "regional_mean": float(means[index]),
            "valid_count": int(field.size),
        }

    def _color_range(self, release: VerifiedEarthRelease, variable: str) -> dict:
        with self._lock:
            cache = self._series(release)
            key = ("color_range", variable)
            cached = cache.get(key)
            if cached is None:
                # Single shared rule; see services.earth_scales.color_range.
                cached = color_range(
                    np.asarray(release.fields[variable], dtype="float32"), variable
                )
                cache[key] = cached
            return dict(cached)


__all__ = [
    "AGGREGATION",
    "EarthOverviewError",
    "EarthOverviewService",
    "SELECTION",
    "VARIABLE_IDS",
    "WIND_VARIABLE_IDS",
    "nearest_grid_index",
    "regional_sample_mean",
    "regional_cell_area_mean",
    "require_date_index",
]
