"""Bounded, mask-aware reads for the three-hour UTC Earth overview.

Display maps aggregate native cells; point queries never use that display grid.
The publication's validated statistics define a fixed colour scale without
scanning the two-year volume on each request.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
from collections import OrderedDict

import numpy as np
import xarray as xr

from services.earth_dataset import _threehour_path
from services.earth_overview_service import EarthOverviewError, nearest_grid_index
from services.netcdf_read_lock import netcdf_read_lock

DATASET_ID = "earth_merra2_3hourly_v1"
_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|\+00:00)$")
TEMPORAL = {"frequency_hours": 3, "step_unit": "hour", "step": 3, "time_zone": "UTC"}


def iso_timestamp(value):
    return np.datetime_as_string(np.datetime64(value, "s"), unit="s") + "Z"


def require_timestamp_index(value, dates):
    if not isinstance(value, str) or not _UTC_PATTERN.fullmatch(value):
        raise EarthOverviewError("invalid_timestamp", "Use an ISO datetime in UTC with seconds and Z")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        target = np.datetime64(parsed.replace(tzinfo=None), "ns")
    except (ValueError, OverflowError):
        raise EarthOverviewError("invalid_timestamp", "Invalid UTC datetime") from None
    index = int(np.searchsorted(dates, target))
    if index == len(dates) or dates[index] != target:
        raise EarthOverviewError("timestamp_out_of_range", "Timestamp is not on the published three-hour axis")
    return index


def _source_path(release):
    try:
        return _threehour_path(release)
    except (ValueError, OSError) as exc:
        raise EarthOverviewError("dataset_version_changed", "The dataset publication changed; reload the catalog", 409) from exc


def _read_values(ds, release, variable, start, stop, lat_slice, lon_slice):
    if not 0 <= start < stop <= len(release.dates) or stop - start > 8:
        raise ValueError("Three-hour reads must select one to eight published frames")
    if (not np.array_equal(ds.time.isel(time=slice(start, stop)).values, release.dates[start:stop])
            or not np.array_equal(ds.lat.values, release.latitude)
            or not np.array_equal(ds.lon.values, release.longitude)):
        raise EarthOverviewError("dataset_version_changed", "Published coordinates changed", 409)
    selection = {"time": slice(start, stop), "lat": lat_slice, "lon": lon_slice}
    values = np.asarray(ds[variable].isel(**selection).values, dtype="float32")
    valid = np.asarray(ds[f"{variable}_valid_mask"].isel(**selection).values)
    if (values.shape != valid.shape or not np.isin(valid, [0, 1]).all()
            or not np.array_equal(valid.astype(bool), np.isfinite(values))
            or not np.isnan(values[valid == 0]).all()
            or np.any(np.abs(values[valid == 1]) >= 1e14)):
        raise EarthOverviewError("invalid_dataset", "Field values do not match the published missing-value mask", 503)
    return values


def read_threehour_values(release, variable, start, stop, *, lat_slice=slice(None), lon_slice=slice(None)):
    """Read at most eight native frames, preserving masked observations as NaN."""
    path = _source_path(release)
    try:
        with netcdf_read_lock(), xr.open_dataset(path, engine="netcdf4") as ds:
            result = _read_values(ds, release, variable, start, stop, lat_slice, lon_slice)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, EarthOverviewError):
            raise
        _source_path(release)
        raise EarthOverviewError("dataset_unavailable", "Cannot read the published Earth data", 503) from exc
    _source_path(release)
    return result


def iter_threehour_chunks(release, variable, start=0, stop=None):
    """Yield (first index, <=8 native frames), opening only the selected variable."""
    stop = len(release.dates) if stop is None else stop
    path = _source_path(release)
    ds = None
    try:
        with netcdf_read_lock():
            ds = xr.open_dataset(path, engine="netcdf4")
        for first in range(start, stop, 8):
            with netcdf_read_lock():
                values = _read_values(ds, release, variable, first, min(first + 8, stop), slice(None), slice(None))
            yield first, values
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, EarthOverviewError):
            raise
        _source_path(release)
        raise EarthOverviewError("dataset_unavailable", "Cannot read the published Earth data", 503) from exc
    finally:
        if ds is not None:
            with netcdf_read_lock():
                ds.close()
    _source_path(release)


def latitude_area_weights(latitude):
    edges = np.concatenate((np.asarray(latitude, dtype="float64") - .375, [float(latitude[-1]) + .375]))
    return np.diff(np.sin(np.deg2rad(edges)))


def threehour_area_mean(values, latitude):
    values = np.asarray(values, dtype="float64")
    valid = np.isfinite(values)
    weights = latitude_area_weights(latitude)[None, :, None]
    numerator = np.sum(np.where(valid, values, 0.) * weights, axis=(1, 2))
    denominator = np.sum(valid * weights, axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def render_block_mean(field, latitude, stride=4):
    """Spherical area mean over valid native cells of each display block."""
    if stride not in (4, 8):
        raise EarthOverviewError("invalid_render_resolution", "Use render_stride 4 or 8")
    height, width = field.shape
    shape = (height // stride, stride, width // stride, stride)
    values = np.asarray(field, dtype="float64").reshape(shape)
    valid = np.isfinite(values)
    weights = latitude_area_weights(latitude).reshape(height // stride, stride, 1, 1)
    numerator = np.sum(np.where(valid, values, 0.) * weights, axis=(1, 3))
    denominator = np.sum(valid * weights, axis=(1, 3))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def nullable_values(values):
    array = np.asarray(values)
    return np.where(np.isfinite(array), array, None).tolist()


class OverviewCache:
    """Byte-bounded LRU containing JSON bytes, isolated by complete identity."""

    def __init__(self, max_bytes=16 * 1024 * 1024, max_entries=128):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries, self._bytes, self._lock = OrderedDict(), 0, threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return json.loads(entry)

    def put(self, key, value):
        payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > self.max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            self._bytes -= len(previous) if previous else 0
            self._entries[key] = payload
            self._bytes += len(payload)
            while self._entries and (self._bytes > self.max_bytes or len(self._entries) > self.max_entries):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= len(evicted)


class ThreehourOverview:
    def __init__(self, service):
        self.service, self.cache = service, OverviewCache()

    def _key(self, release, variable, kind, *selection):
        metadata = release.metadata
        return ("earth", metadata["dataset_id"], metadata["dataset_version"],
                metadata["dataset_fingerprint"], variable, kind, *selection)

    def _store(self, release, key, payload):
        self.service._release(release.metadata["dataset_id"], release.metadata["dataset_fingerprint"])
        _source_path(release)
        self.cache.put(key, payload)
        return payload

    @staticmethod
    def _temporal(release, first, last):
        timestamps = [iso_timestamp(value) for value in release.dates[first:last + 1]]
        return {**TEMPORAL, "start": timestamps[0], "end": timestamps[-1], "dates": timestamps, "timestamps": timestamps}

    def field(self, release, variable, units, timestamp, stride):
        index = require_timestamp_index(timestamp, release.dates)
        if stride not in (4, 8):
            raise EarthOverviewError("invalid_render_resolution", "Use render_stride 4 or 8")
        timestamp = iso_timestamp(release.dates[index])
        key = self._key(release, variable, "field", timestamp, stride)
        _source_path(release)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        native = read_threehour_values(release, variable, index, index + 1)[0]
        rendered = render_block_mean(native, release.latitude, stride)
        finite = native[np.isfinite(native)]
        declared = release.metadata.get("manifest", {}).get("variables", {}).get(variable, {})
        low, high = declared.get("min"), declared.get("max")
        # The verified publication supplies extrema, including the all-missing
        # case. Fallback is only for trusted service fixtures with no manifest.
        low = float(low) if low is not None else (float(finite.min()) if finite.size else 0.)
        high = float(high) if high is not None else (float(finite.max()) if finite.size else 1.)
        centered = variable in ("U10M", "V10M")
        if centered:
            bound = max(abs(low), abs(high)); low, high = -bound, bound
        shape = [240 // stride, 480 // stride]
        payload = {
            **self.service._identity(release, variable, units), **TEMPORAL,
            "date": timestamp, "timestamp": timestamp,
            "calendar": release.metadata["time"].get("calendar", "proleptic_gregorian"),
            "lat": np.asarray(release.latitude).reshape(-1, stride).mean(axis=1).tolist(),
            "lon": np.asarray(release.longitude).reshape(-1, stride).mean(axis=1).tolist(),
            "dimension_order": ["lat", "lon"], "field": nullable_values(rendered),
            "coverage": {"latitude_range": [-90., 90.], "longitude_range": [-180., 180.], "wrap_longitude": True},
            "grid_shape": shape, "render_grid_shape": shape, "source_grid_shape": [240, 480],
            "render_method": f"spherical_cell_area_mean_{stride}x{stride}",
            "render": {"method": "spherical_cell_area_block_mean", "block_shape": [stride, stride],
                       "missing_policy": "valid_cell_area_mean", "native_resolution_for_points": True},
            "color_range": {"min": low, "max": high, "scope": "dataset", "centered_on_zero": centered},
            "statistics": {"min": float(finite.min()) if finite.size else None,
                           "max": float(finite.max()) if finite.size else None,
                           "regional_mean": nullable_values(threehour_area_mean(native[None], release.latitude))[0],
                           "valid_count": int(finite.size)},
        }
        return self._store(release, key, payload)

    def regional(self, release, variable, units, first, last):
        key = self._key(release, variable, "regional", first, last)
        _source_path(release)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        means = np.empty(last - first + 1, dtype="float64")
        for index, values in iter_threehour_chunks(release, variable, first, last + 1):
            means[index - first:index - first + len(values)] = threehour_area_mean(values, release.latitude)
        payload = {**self.service._identity(release, variable, units), **self._temporal(release, first, last),
                   "values": nullable_values(means), "aggregation": "spherical_cell_area_mean",
                   "missing_policy": "valid_cell_area_mean", "source_grid_shape": [240, 480],
                   "coverage": {"latitude_range": [-90., 90.], "longitude_range": [-180., 180.], "wrap_longitude": True}}
        return self._store(release, key, payload)

    def point(self, release, variable, units, lat, lon, first, last):
        row = nearest_grid_index(release.latitude, lat, (-90., 90.))
        column = nearest_grid_index(release.longitude, lon, (-180., 180.))
        key = self._key(release, variable, "point", first, last, float(lat), float(lon), row, column)
        _source_path(release)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        values = np.empty(last - first + 1, dtype="float32")
        for index in range(first, last + 1, 8):
            block = read_threehour_values(release, variable, index, min(index + 8, last + 1),
                                          lat_slice=row, lon_slice=column)
            values[index - first:index - first + len(block)] = block
        payload = {**self.service._identity(release, variable, units), **self._temporal(release, first, last),
                   "requested": {"lat": float(lat), "lon": float(lon)},
                   "grid_point": {"lat": float(release.latitude[row]), "lon": float(release.longitude[column]),
                                  "lat_index": row, "lon_index": column},
                   "selection": "nearest_grid_point", "values": nullable_values(values),
                   "source_grid_shape": [240, 480], "missing_policy": "preserve_null"}
        return self._store(release, key, payload)
