"""Shared Mars training and inference data preparation.

The two Mars dataset identities intentionally bind to different physical
sources.  This module is the single implementation of file discovery, MCD
field mapping, Ls handling, grid normalization and training-only scaling.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

from services.netcdf_read_lock import netcdf_read_lock
from services.mars_dataset_identity import (
    canonical_dataset_identity,
    directory_fingerprint,
    normalize_mars_dataset_identity,
    volume_dataset_identity,
)
from services.mars_checkpoint import (
    MCD_WINDOW_POLICY,
    MARS_AUXILIARY_CHANNEL_ORDER,
    MARS_LEGACY_SPLIT_POLICY,
    OPENMARS_WINDOW_POLICY,
    mars_input_channel_order,
    validate_mars_normalization,
)
from services.ozone_units import normalize_ozone_column_units
from services.training_split import (
    STRICT_TIMELINE_SPLIT_POLICY,
    TrainingSplitError,
    split_sample_ranges,
    strict_window_split_metadata,
)

CHANNEL_ORDER = list(MARS_AUXILIARY_CHANNEL_ORDER)
MCD_VARS_MAP = {
    "U": ("U_Wind", "u"),
    "V": ("V_Wind", "v"),
    "D": ("Dust_Optical_Depth", "dustq"),
    "S": ("Solar_Flux_DN", "fluxsurf_dn_sw"),
    "T": ("Temperature", "temp"),
}
# Raw MCD releases use several historical names for dust.  Resolution is
# case-insensitive, but the canonical names are tried first.
RAW_FIELD_CANDIDATES = {
    "U": ("U", "U_Wind", "u"),
    "V": ("V", "V_Wind", "v"),
    "D": ("Dust_Optical_Depth", "DUST", "DUSTQ", "dustq", "dust"),
    "S": ("FSDS", "Solar_Flux_DN", "fluxsurf_dn_sw", "SWGDN"),
    "T": ("T", "Temperature", "temp"),
}
RAW_MCD_TARGET_LAT = np.arange(87.5, -90.0, -5.0, dtype=np.float32)


class MarsDataError(ValueError):
    """Raised when the bound Mars data source cannot satisfy the contract."""


@dataclass(frozen=True)
class MarsYearSegment:
    """A slice of one source file; all index ranges are half-open."""

    mars_year: int
    start: int
    end: int
    source_file: Path
    file_start: int
    file_end: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mars_year": self.mars_year, "start": self.start, "end": self.end,
            "source_file": self.source_file.name,
            "file_start": self.file_start, "file_end": self.file_end,
        }


@dataclass(frozen=True)
class MarsYearBlock:
    """One independently windowed Mars-year data block."""

    mars_year: int
    start: int
    end: int
    source_files: tuple[str, ...]
    values: Any
    ls: np.ndarray | None
    segments: tuple[MarsYearSegment, ...] = ()


def resolve_forecast_window(
    ls_values: Any,
    ls_start: Any,
    window: int,
    horizon: int,
    candidate_starts: Any = None,
    candidate_mars_years: Any = None,
    candidate_blocks: Any = None,
) -> dict[str, Any]:
    """Resolve a complete Mars forecast window on the actual Ls timeline.

    ``ls_start`` identifies the first input time point. Candidate starts are
    restricted to windows with both a complete input and target segment, and
    periodic angular distance makes 0 and 360 degrees equivalent. Windows may
    cross a Mars-year boundary.
    """
    try:
        values = np.asarray(ls_values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MarsDataError("Mars Ls values must be a one-dimensional numeric array") from exc
    if values.size == 0:
        raise MarsDataError("Mars Ls timeline is empty")
    if not np.all(np.isfinite(values)):
        raise MarsDataError("Mars Ls timeline contains non-finite values")
    try:
        requested = float(ls_start)
    except (TypeError, ValueError) as exc:
        raise MarsDataError("ls_start must be a finite number") from exc
    if not np.isfinite(requested):
        raise MarsDataError("ls_start must be a finite number")
    try:
        window = int(window)
        horizon = int(horizon)
    except (TypeError, ValueError) as exc:
        raise MarsDataError("window and horizon must be positive integers") from exc
    if window <= 0 or horizon <= 0:
        raise MarsDataError("window and horizon must be positive integers")
    max_start = int(values.size) - window - horizon
    if max_start < 0:
        raise MarsDataError(
            f"Not enough Ls values for a complete forecast window: "
            f"time={values.size}, window={window}, horizon={horizon}"
        )

    requested %= 360.0
    if candidate_starts is None:
        starts = np.arange(max_start + 1, dtype=np.int64)
    else:
        starts = np.asarray(candidate_starts, dtype=np.int64).reshape(-1)
        if starts.size == 0:
            raise MarsDataError("Mars dataset has no complete windows")
        if np.any(starts < 0) or np.any(starts > max_start):
            raise MarsDataError("Mars window start indices contain an incomplete window")
    blocks = tuple(candidate_blocks or ())
    block_indices = None
    if blocks:
        block_starts = np.asarray([block.start for block in blocks], dtype=np.int64)
        block_ends = np.asarray([block.end for block in blocks], dtype=np.int64)
        block_indices = np.searchsorted(block_starts, starts, side="right") - 1
        if np.any(block_indices < 0):
            raise MarsDataError("Mars window start indices are outside the data blocks")
    candidates = values[starts]
    distances = np.abs(((candidates - requested + 180.0) % 360.0) - 180.0)
    # ``argmin`` is deterministic and returns the first index on a tie.
    candidate_index = int(np.argmin(distances))
    sample_index = int(starts[candidate_index])
    selected_year = None
    selected_block = None
    if candidate_mars_years is not None and len(np.asarray(candidate_mars_years).reshape(-1)):
        years = np.asarray(candidate_mars_years).reshape(-1)
        if len(years) != len(starts):
            raise MarsDataError("Mars-year metadata does not match window starts")
        selected_year = int(years[candidate_index])
        ordered_years = []
        for year in years.tolist():
            if year not in ordered_years:
                ordered_years.append(year)
        selected_block = int(ordered_years.index(selected_year))
    if block_indices is not None:
        selected_block = int(block_indices[candidate_index])
        block_years = np.asarray([blocks[index].mars_year for index in block_indices])
        if candidate_mars_years is not None and not np.array_equal(np.asarray(candidate_mars_years).reshape(-1), block_years):
            raise MarsDataError("Mars-year metadata does not match the actual data blocks")
        selected_year = int(block_years[candidate_index])
    return {
        "sample_index": sample_index,
        "block_index": selected_block,
        "mars_year": selected_year,
        "input_ls": values[sample_index : sample_index + window].astype(float).tolist(),
        "target_ls": values[
            sample_index + window : sample_index + window + horizon
        ].astype(float).tolist(),
    }


@dataclass(frozen=True)
class MarsDataArrays:
    ozone: np.ndarray
    ls: np.ndarray | None
    features: dict[str, np.ndarray]
    latitude: np.ndarray | None
    longitude: np.ndarray | None
    dataset_id: str
    source_type: str
    data_dir: Path
    manifest: tuple[dict[str, Any], ...]
    fingerprint: str
    blocks: tuple[MarsYearBlock, ...]
    data_directories: tuple[Path, ...] = ()


@dataclass(frozen=True)
class MarsScaledVolume:
    values: np.ndarray
    y_scaled: np.ndarray
    ls: np.ndarray | None
    y_mean: float
    y_std: float
    height: int
    width: int
    latitude: np.ndarray | None
    longitude: np.ndarray | None
    split_idx: int
    input_means: tuple[np.ndarray, ...]
    input_stds: tuple[np.ndarray, ...]
    dataset_id: str
    source_type: str
    data_dir: Path
    manifest: tuple[dict[str, Any], ...]
    fingerprint: str
    sample_starts: np.ndarray
    sample_mars_years: tuple[int, ...]
    blocks: tuple[MarsYearBlock, ...]
    train_sample_end: int
    data_directories: tuple[Path, ...] = ()
    split_ranges: dict[str, dict[str, Any]] | None = None
    split_window_starts: dict[str, tuple[int, ...]] | None = None
    split_policy: str = "legacy_compatibility"


def natural_sort_key(value: Any) -> list[Any]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", str(value))]


def _mars_years_from_name(path: Path) -> tuple[int, ...]:
    matches = re.findall(r"(?:^|[_-])MY(\d{1,3})(?=[_\.-]|$)", path.name, re.IGNORECASE)
    if not matches:
        raise MarsDataError(f"Unable to identify Mars year from data file: {path.name}")
    return tuple(int(year) for year in matches)


def _mars_year_from_name(path: Path) -> int:
    years = _mars_years_from_name(path)
    if len(set(years)) != 1:
        raise MarsDataError(f"Multiple Mars years require Ls-based segmentation: {path.name}")
    return years[0]


def _openmars_year_segments(path: Path, ls: np.ndarray, start: int) -> tuple[MarsYearSegment, ...]:
    years = _mars_years_from_name(path)
    if not np.all(np.isfinite(ls)) or np.any(ls < 0) or np.any(ls > 360):
        raise MarsDataError(f"OpenMARS Ls must be finite and within 0..360 in {path.name}")
    differences = np.diff(ls)
    if np.any((differences < 0) & (differences >= -180.0)):
        raise MarsDataError(f"OpenMARS Ls moves backwards without a year wrap in {path.name}")
    cuts = np.flatnonzero(differences < -180.0) + 1
    first_year = years[0]
    # Named endpoints corroborate the observed wraps, never determine their positions.
    if len(years) > 1 and (years[-1] != first_year + len(cuts) or any(a > b for a, b in zip(years, years[1:]))):
        raise MarsDataError(f"OpenMARS filename Mars years disagree with Ls year boundaries in {path.name}")
    edges = [0, *cuts.tolist(), len(ls)]
    return tuple(
        MarsYearSegment(first_year + index, start + left, start + right, path, left, right)
        for index, (left, right) in enumerate(zip(edges, edges[1:]))
    )


def _file_time_length(dataset: Any, dataset_name: str) -> int:
    variable_name = "O3COL" if dataset_name == "mcd_overview_raw" else "o3col"
    if variable_name not in dataset.variables:
        return 0
    shape = dataset.variables[variable_name].shape
    if not shape:
        return 0
    return int(shape[0] * shape[1]) if dataset_name == "mcd_overview_raw" and len(shape) >= 4 else int(shape[0])


def _discover_year_segments(directory: Path, dataset_name: str) -> tuple[MarsYearSegment, ...]:
    segments = []
    start = 0
    for path in sorted(directory.glob("*.nc"), key=natural_sort_key):
        with netcdf_read_lock(), netCDF4.Dataset(str(path)) as dataset:
            length = _file_time_length(dataset, dataset_name)
            if length <= 0:
                continue
            if dataset_name == "openmars":
                ls = _raw_ls(dataset, path, required=True)
                if len(ls) != length:
                    raise MarsDataError(f"OpenMARS Ls length does not match ozone time steps in {path.name}")
                segments.extend(_openmars_year_segments(path, ls, start))
            else:
                segments.append(MarsYearSegment(_mars_year_from_name(path), start, start + length, path, 0, length))
        start += length
    if not segments:
        raise MarsDataError(f"No Mars-year data files found in {directory}")
    return tuple(segments)


def _make_year_blocks(
    data: "MarsDataArrays",
    segments: tuple[MarsYearSegment, ...],
) -> tuple[MarsYearBlock, ...]:
    expected = sum(segment.end - segment.start for segment in segments)
    if expected != len(data.ozone):
        raise MarsDataError(
            "Mars-year file lengths do not match the aligned data timeline: "
            f"files={expected}, timeline={len(data.ozone)}"
        )
    blocks = []
    for segment in segments:
        start, end = segment.start, segment.end
        blocks.append(
            MarsYearBlock(
                mars_year=segment.mars_year,
                start=start,
                end=end,
                source_files=(segment.source_file.name,),
                values={
                    "ozone": data.ozone[start:end],
                    "features": {
                        key: value[start:end] for key, value in data.features.items()
                    },
                },
                ls=None if data.ls is None else np.asarray(data.ls[start:end]),
                segments=(segment,),
            )
        )
    return tuple(blocks)


def parse_channels(value: Any) -> list[str]:
    if value is None:
        raw = []
    elif isinstance(value, str):
        try:
            parsed = json.loads(value) if value.strip().startswith("[") else None
        except Exception:
            parsed = None
        raw = parsed if isinstance(parsed, list) else value.replace("+", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = []
    selected = {str(item).strip().upper() for item in raw if str(item).strip()}
    return [channel for channel in CHANNEL_ORDER if channel in selected]


def _clean(value: Any) -> np.ndarray:
    array = np.asanyarray(value)
    if np.ma.isMaskedArray(array) and np.any(np.ma.getmaskarray(array)):
        raise MarsDataError("Mars data contains masked or missing values; training and prediction require complete finite fields")
    try:
        result = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MarsDataError("Mars data contains non-numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise MarsDataError("Mars data contains NaN or Inf values; training and prediction require finite fields")
    return result


def _raw_ls(dataset: Any, path: Path, *, required: bool = False) -> np.ndarray | None:
    for name in ("LS", "Ls", "ls"):
        if name in dataset.variables:
            return _clean(dataset.variables[name][:]).reshape(-1)
    if required:
        raise MarsDataError(f"Missing Ls variable in {path}")
    return None


def _unwrap(values: Any) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32).reshape(-1)
    out = raw.copy()
    offset = 0.0
    for index in range(1, len(out)):
        if raw[index] < raw[index - 1] - 180.0:
            offset += 360.0
        out[index] += offset
    return out


def _expand_ls(dataset: Any, path: Path, variable_name: str) -> np.ndarray:
    values = _raw_ls(dataset, path, required=True)
    shape = dataset.variables[variable_name].shape
    if len(shape) < 4:
        if len(values) < int(shape[0]):
            raise MarsDataError(f"Not enough Ls values in {path}")
        return values[: int(shape[0])]
    solar_count, hour_count = int(shape[0]), int(shape[1])
    if len(values) == solar_count * hour_count:
        return values
    if len(values) < solar_count:
        raise MarsDataError(f"Not enough Ls values in {path}")
    expanded = np.zeros(solar_count * hour_count, dtype=np.float32)
    for index in range(solar_count):
        start = float(values[index])
        if index < solar_count - 1:
            end = float(values[index + 1])
            if end < start:
                end += 360.0
        else:
            step = float(values[1] - values[0]) if solar_count > 1 else 0.5
            if step <= 0:
                step += 360.0
            end = start + step
        expanded[index * hour_count:(index + 1) * hour_count] = np.linspace(start, end, hour_count, endpoint=False)
    return expanded % 360.0


def _merge_hours(value: Any) -> np.ndarray:
    array = _clean(value)
    if array.ndim == 4:
        return array.reshape(array.shape[0] * array.shape[1], array.shape[2], array.shape[3])
    if array.ndim == 3:
        return array
    raise MarsDataError(f"Expected MCD variable to be 3D or 4D, got shape {array.shape}")


def _coords(dataset: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    lat_name = "lat" if "lat" in dataset.variables else "latitude" if "latitude" in dataset.variables else None
    lon_name = "lon" if "lon" in dataset.variables else "longitude" if "longitude" in dataset.variables else None
    if not lat_name or not lon_name:
        return None, None
    def axis(name):
        return _clean(dataset.variables[name][:])
    return axis(lat_name), axis(lon_name)


def _fit_lat_grid(data: Any, lat_values: Any) -> np.ndarray:
    field = _clean(data)
    if field.ndim != 3:
        raise MarsDataError(f"Expected raw MCD field to be 3D after time expansion, got shape {field.shape}")
    lat = np.asarray(lat_values, dtype=np.float32).reshape(-1)
    if len(lat) != field.shape[1] or not np.all(np.isfinite(lat)):
        raise MarsDataError("Raw MCD latitude coordinates do not match the field or contain non-finite values")
    if field.shape[1] == len(RAW_MCD_TARGET_LAT) and np.allclose(lat, RAW_MCD_TARGET_LAT, atol=1e-3):
        result = field
    elif field.shape[1] == len(RAW_MCD_TARGET_LAT) and np.allclose(lat, RAW_MCD_TARGET_LAT[::-1], atol=1e-3):
        result = field[:, ::-1, :]
    elif field.shape[1] == len(RAW_MCD_TARGET_LAT) + 1 and lat.shape[0] == field.shape[1]:
        diffs = np.diff(lat)
        if not np.allclose(np.abs(diffs), 5.0, atol=1e-3):
            result = None
        elif lat[0] > lat[-1]:
            result = (field[:, :-1, :] + field[:, 1:, :]) * 0.5
        else:
            result = ((field[:, :-1, :] + field[:, 1:, :]) * 0.5)[:, ::-1, :]
    else:
        result = None
    if result is None:
        order = np.argsort(lat)
        sorted_lat = lat[order]
        flat = field[:, order, :].transpose(0, 2, 1).reshape(-1, field.shape[1])
        out = np.empty((flat.shape[0], len(RAW_MCD_TARGET_LAT)), dtype=np.float32)
        for index, row in enumerate(flat):
            out[index] = np.interp(RAW_MCD_TARGET_LAT, sorted_lat, row)
        result = out.reshape(field.shape[0], field.shape[2], len(RAW_MCD_TARGET_LAT)).transpose(0, 2, 1)
    return result[:, :, :72].astype(np.float32)


def _resolve_raw_variable(dataset: Any, channel: str, path: Path) -> str:
    by_lower = {str(name).lower(): str(name) for name in dataset.variables}
    for candidate in RAW_FIELD_CANDIDATES[channel]:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    if channel == "D":
        raise MarsDataError(
            f"Raw MCD file {path} has no Dust variable; cannot train with Dust channel"
        )
    raise MarsDataError(f"Raw MCD file {path} is missing variable for channel {channel}")


def resolve_data_directory(training_dataset: Any, *, openmars_dir: Any, mcd_dir: Any, raw_dir: Any) -> tuple[str, str, Path]:
    dataset = str(training_dataset or "openmars_mcd").strip().lower()
    if dataset == "mcd_overview":
        return dataset, "mcd_raw_3h", Path(raw_dir).expanduser()
    if dataset == "openmars_mcd":
        return dataset, "openmars_mcd", Path(mcd_dir).expanduser()
    raise MarsDataError(f"Unsupported Mars training dataset: {training_dataset}")


def _load_raw(directory: Path, selected: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    ozone_parts, ls_parts = [], []
    feature_parts = {MCD_VARS_MAP[ch][1]: [] for ch in selected}
    latitude = longitude = None
    files = sorted(directory.glob("*.nc"), key=natural_sort_key)
    for path in files:
        with netcdf_read_lock(), netCDF4.Dataset(str(path)) as ds:
            if "O3COL" not in ds.variables:
                continue
            lat_name = "lat" if "lat" in ds.variables else "latitude" if "latitude" in ds.variables else None
            if lat_name is None:
                raise MarsDataError(f"Raw MCD file {path} is missing lat")
            lat = _clean(ds.variables[lat_name][:])
            if lat.ndim != 1 or not lat.size or not np.all(np.isfinite(lat)):
                raise MarsDataError(f"Raw MCD latitude coordinates must be finite and one-dimensional in {path.name}")
            lon_name = "lon" if "lon" in ds.variables else "longitude" if "longitude" in ds.variables else None
            lon = None if lon_name is None else _clean(ds.variables[lon_name][:])
            if lon is not None:
                source_shape = ds.variables["O3COL"].shape[-2:]
                if len(lat) != source_shape[0] or lon.ndim != 1 or len(lon) != source_shape[1] or not np.all(np.isfinite(lon)):
                    raise MarsDataError(f"Raw MCD spatial coordinates do not match the field or contain non-finite values in {path.name}")
            if not ozone_parts:
                latitude = RAW_MCD_TARGET_LAT.copy() if lon is not None else None
                longitude = None if lon is None else lon[:72].copy()
            elif (longitude is None) != (lon is None) or (lon is not None and not np.array_equal(longitude, lon[:72])):
                raise MarsDataError(f"Spatial grid mismatch in {path}")
            ozone_parts.append(normalize_ozone_column_units(_fit_lat_grid(ds.variables["O3COL"][:], lat), getattr(ds.variables["O3COL"], "units", None), allow_mcd_legacy_heuristic=True))
            raw_ls = _raw_ls(ds, path)
            if raw_ls is not None:
                ls_parts.append(raw_ls)
            for channel in selected:
                name = _resolve_raw_variable(ds, channel, path)
                feature_parts[MCD_VARS_MAP[channel][1]].append(_fit_lat_grid(ds.variables[name][:], lat))
    if not ozone_parts:
        raise FileNotFoundError(f"No raw 3h MCD .nc files found in {directory}")
    ozone = _clean(np.concatenate(ozone_parts, axis=0))
    ls = np.concatenate(ls_parts, axis=0).astype(np.float32).reshape(-1) if ls_parts else None
    features = {key: _clean(np.concatenate(values, axis=0)) for key, values in feature_parts.items()}
    count = min(([len(ls)] if ls is not None else []) + [len(ozone)] + [len(value) for value in features.values()])
    if ls is not None and len(ls) != len(ozone):
        ls = None
        count = min([len(ozone)] + [len(value) for value in features.values()])
    return ozone[:count], None if ls is None else ls[:count], {key: value[:count] for key, value in features.items()}, latitude, longitude


def _load_overview(directory: Path, selected: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    ozone_parts, ls_parts = [], []
    feature_parts = {MCD_VARS_MAP[ch][1]: [] for ch in selected}
    latitude = longitude = None
    for path in sorted(directory.glob("*.nc"), key=natural_sort_key):
        with netcdf_read_lock(), netCDF4.Dataset(str(path)) as ds:
            if "o3col" not in ds.variables:
                continue
            ozone = _clean(ds.variables["o3col"][:])
            if ozone.ndim != 3:
                raise MarsDataError(f"Invalid MCD overview o3col shape in {path}: {ozone.shape}")
            file_latitude, file_longitude = _coords(ds)
            if not ozone_parts:
                latitude, longitude = file_latitude, file_longitude
            elif not np.array_equal(latitude, file_latitude) or not np.array_equal(longitude, file_longitude):
                raise MarsDataError(f"Spatial grid mismatch in {path}")
            ozone_parts.append(ozone)
            raw_ls = _raw_ls(ds, path)
            if raw_ls is not None:
                ls_parts.append(raw_ls)
            for channel in selected:
                variable = MCD_VARS_MAP[channel][0]
                if variable not in ds.variables:
                    raise MarsDataError(f"MCD overview file {path} is missing variable: {variable}")
                feature_parts[MCD_VARS_MAP[channel][1]].append(_clean(ds.variables[variable][:]))
    if not ozone_parts:
        raise FileNotFoundError(f"No MCD overview .nc files found in {directory}")
    ozone = _clean(np.concatenate(ozone_parts, axis=0))
    ls = np.concatenate(ls_parts, axis=0).astype(np.float32).reshape(-1) if ls_parts else None
    features = {key: _clean(np.concatenate(values, axis=0)) for key, values in feature_parts.items()}
    count = min(([len(ls)] if ls is not None else []) + [len(ozone)] + [len(value) for value in features.values()])
    if ls is not None and len(ls) != len(ozone):
        ls = None
        count = min([len(ozone)] + [len(value) for value in features.values()])
    return ozone[:count], None if ls is None else ls[:count], {key: value[:count] for key, value in features.items()}, latitude, longitude


def _load_openmars(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    ozone_parts, ls_parts = [], []
    latitude = longitude = None
    for path in sorted(directory.glob("*.nc"), key=natural_sort_key):
        with netcdf_read_lock(), netCDF4.Dataset(str(path)) as ds:
            if "o3col" not in ds.variables:
                continue
            ozone = _clean(ds.variables["o3col"][:])
            if ozone.ndim == 4:
                ozone = np.nanmean(ozone, axis=1)
            if ozone.ndim != 3:
                raise MarsDataError(f"Invalid OpenMars o3col shape in {path}: {ozone.shape}")
            file_latitude, file_longitude = _coords(ds)
            if not ozone_parts:
                latitude, longitude = file_latitude, file_longitude
            elif not np.array_equal(latitude, file_latitude) or not np.array_equal(longitude, file_longitude):
                raise MarsDataError(f"Spatial grid mismatch in {path}")
            ozone_parts.append(ozone)
            raw_ls = _raw_ls(ds, path, required=True)
            ls_parts.append(raw_ls)
    if not ozone_parts:
        raise FileNotFoundError(f"No OpenMars .nc files found in {directory}")
    ozone = _clean(np.concatenate(ozone_parts, axis=0))
    ls = np.concatenate(ls_parts, axis=0).astype(np.float32).reshape(-1)
    count = min(len(ozone), len(ls))
    return ozone[:count], ls[:count], latitude, longitude


def load_mars_arrays(*, training_dataset: Any, openmars_dir: Any, mcd_dir: Any, raw_dir: Any, selected_channels: Any) -> MarsDataArrays:
    dataset, source_type, source_dir = resolve_data_directory(training_dataset, openmars_dir=openmars_dir, mcd_dir=mcd_dir, raw_dir=raw_dir)
    identity = canonical_dataset_identity(training_dataset=dataset, openmars_dir=openmars_dir, mcd_dir=mcd_dir, raw_dir=raw_dir)
    selected = parse_channels(selected_channels)
    if dataset == "mcd_overview":
        has_raw = False
        for path in sorted(source_dir.glob("*.nc"), key=natural_sort_key):
            with netcdf_read_lock(), netCDF4.Dataset(str(path)) as ds:
                if "O3COL" in ds.variables:
                    has_raw = True
                    break
        if has_raw:
            ozone, ls, features, lat, lon = _load_raw(source_dir, selected)
            segments = _discover_year_segments(source_dir, "mcd_overview_raw")
        else:
            ozone, ls, features, lat, lon = _load_overview(source_dir, selected)
            segments = _discover_year_segments(source_dir, "mcd_overview")
    else:
        ozone, ls, lat, lon = _load_openmars(Path(openmars_dir).expanduser())
        segments = _discover_year_segments(Path(openmars_dir).expanduser(), "openmars")
        # OpenMARS + MCD interpolation is kept in this shared path so inference
        # cannot accidentally use the raw overview directory.
        mcd_ls_parts, mcd_parts = [], {MCD_VARS_MAP[ch][1]: [] for ch in selected}
        for path in sorted(source_dir.glob("*.nc"), key=natural_sort_key):
            with netcdf_read_lock(), netCDF4.Dataset(str(path)) as ds:
                first = MCD_VARS_MAP[selected[0]][0] if selected else None
                if first and first not in ds.variables:
                    continue
                if not selected:
                    continue
                missing = [MCD_VARS_MAP[ch][0] for ch in selected if MCD_VARS_MAP[ch][0] not in ds.variables]
                if missing:
                    raise MarsDataError(f"MCD file {path} is missing variables: {missing}")
                for ch in selected:
                    name, short = MCD_VARS_MAP[ch]
                    mcd_parts[short].append(_merge_hours(ds.variables[name][:]))
                mcd_ls_parts.append(_expand_ls(ds, path, first))
        if selected and not mcd_ls_parts:
            raise FileNotFoundError(f"No usable MCD files found in {source_dir}")
        if selected:
            mcd_ls = _unwrap(np.concatenate(mcd_ls_parts, axis=0))
            target_ls = _unwrap(ls)
            order = np.argsort(mcd_ls)
            mcd_ls = mcd_ls[order]
            features = {short: _clean(interp1d(mcd_ls, np.concatenate(parts, axis=0)[order], axis=0, bounds_error=False, fill_value="extrapolate")(target_ls)) for short, parts in mcd_parts.items()}
        else:
            features = {}
    if canonical_dataset_identity(training_dataset=dataset, openmars_dir=openmars_dir, mcd_dir=mcd_dir, raw_dir=raw_dir) != identity:
        raise MarsDataError("Mars data files changed while loading the training volume")
    data = MarsDataArrays(
        ozone=ozone,
        ls=ls,
        features=features,
        latitude=lat,
        longitude=lon,
        dataset_id=dataset,
        source_type=source_type,
        data_dir=source_dir.resolve(),
        manifest=tuple(identity["file_manifest"]),
        fingerprint=identity["dataset_fingerprint"],
        blocks=(),
        data_directories=tuple(Path(value) for value in identity["data_directories"]),
    )
    return replace(data, blocks=_make_year_blocks(data, segments))


def prepare_scaled_volume(*, training_dataset: Any, openmars_dir: Any, mcd_dir: Any, raw_dir: Any, selected_channels: Any, window: int, horizon: int, split_ratios: dict[str, Any], require_ls: bool = True, normalization: dict[str, Any] | None = None, split_policy: str | None = None) -> MarsScaledVolume:
    data = load_mars_arrays(training_dataset=training_dataset, openmars_dir=openmars_dir, mcd_dir=mcd_dir, raw_dir=raw_dir, selected_channels=selected_channels)
    if require_ls and data.ls is None:
        raise MarsDataError("Mars dataset does not contain Ls values")
    selected = parse_channels(selected_channels)
    input_channel_order = mars_input_channel_order(selected)
    arrays = [
        data.ozone if channel == "O3" else data.features[MCD_VARS_MAP[channel][1]]
        for channel in input_channel_order
    ]
    count = min(len(array) for array in arrays)
    height = min(array.shape[1] for array in arrays)
    width = min(array.shape[2] for array in arrays)
    raw = np.stack([_clean(array[:count, :height, :width]) for array in arrays], axis=-1)
    ls = None if data.ls is None else np.asarray(data.ls[:count], dtype=np.float32).reshape(-1)
    # Windows are formed on the complete chronological timeline. A window may
    # cross an MY boundary; the start index is still labelled with the MY that
    # contains that start for display and deterministic tie-breaking.
    legal_starts = list(range(0, count - int(window) - int(horizon) + 1))
    sample_years: list[int] = []
    prepared_blocks: list[MarsYearBlock] = []
    for block in data.blocks:
        block_end = min(int(block.end), count)
        prepared_blocks.append(
            replace(
                block,
                end=block_end,
                ls=None if ls is None else ls[int(block.start):block_end],
            )
        )
    for start in legal_starts:
        year = next(
            (int(block.mars_year) for block in data.blocks
             if int(block.start) <= start < int(block.end)),
            -1,
        )
        sample_years.append(year)
    sample_starts = np.asarray(legal_starts, dtype=np.int64)
    sample_count = int(sample_starts.size)
    if sample_count <= 0:
        raise MarsDataError(
            f"No complete windows: time={count}, window={window}, horizon={horizon}"
        )
    # Allocate the retained windows on the complete timeline. Boundary gaps
    # keep input and target indices disjoint across splits. Tiny historical
    # fixtures may not have enough points for the strict contract; those remain
    # on the explicit legacy path.
    try:
        if split_policy == MARS_LEGACY_SPLIT_POLICY:
            raise TrainingSplitError("legacy checkpoint split policy")
        strict = strict_window_split_metadata(
            count,
            int(window),
            int(horizon),
            split_ratios,
            blocks=data.blocks,
        )
        split_policy = STRICT_TIMELINE_SPLIT_POLICY
        split_window_starts = {
            name: tuple(
                start
                for left, right in strict[name]["window_ranges"]
                for start in range(int(left), int(right))
            )
            for name in ("train", "validation", "test")
        }
        split_ranges = {
            name: dict(strict[name])
            for name in ("train", "validation", "test")
        }
    except TrainingSplitError:
        # Preserve old tiny-data behavior for compatibility readers.  A real
        # task with enough data always takes the strict path above.
        try:
            legacy_ranges = split_sample_ranges(sample_count, split_ratios)
        except TrainingSplitError:
            # A few legacy readers use a three-sample fixture only to inspect
            # tensor shapes. Keep every sample addressable without claiming a
            # strict training split for that fixture.
            if sample_count < 3:
                raise MarsDataError("Mars dataset has too few windows for a training split")
            first = max(1, sample_count - 2)
            legacy_ranges = {
                "train": (0, first),
                "validation": (first, first + 1),
                "test": (first + 1, sample_count),
            }
        split_window_starts = {
            name: tuple(sample_starts[left:right].tolist())
            for name, (left, right) in legacy_ranges.items()
        }
        split_policy = "legacy_compatibility"
        split_ranges = {
            name: {
                "raw_start": int(sample_starts[left]) if right > left else None,
                "raw_end": int(sample_starts[right - 1] + window + horizon) if right > left else None,
                "window_count": int(right - left),
                "window_ranges": [[int(sample_starts[left]), int(sample_starts[right - 1] + 1)]] if right > left else [],
                "input_time_start": int(sample_starts[left]) if right > left else None,
                "input_time_end": int(sample_starts[right - 1] + window) if right > left else None,
                "target_time_start": int(sample_starts[left] + window) if right > left else None,
                "target_time_end": int(sample_starts[right - 1] + window + horizon) if right > left else None,
            }
            for name, (left, right) in legacy_ranges.items()
        }
    train_starts = np.asarray(split_window_starts["train"], dtype=np.int64)
    train_sample_end = int(len(train_starts))
    fit_input_indices = np.unique(
        np.concatenate([np.arange(start, start + int(window)) for start in train_starts])
    )
    fit_target_indices = np.unique(
        np.concatenate(
            [
                np.arange(start + int(window), start + int(window) + int(horizon))
                for start in train_starts
            ]
        )
    )
    split_idx = min(count, int(fit_input_indices[-1]) + 1)
    scaled = np.empty_like(raw, dtype=np.float32)
    means, stds = [], []
    use_saved = normalization is not None
    if use_saved:
        try:
            validate_mars_normalization(
                normalization, selected_channels=selected, grid_shape=(height, width)
            )
        except ValueError as exc:
            raise MarsDataError(str(exc)) from exc
    for idx, channel in enumerate(input_channel_order):
        if use_saved:
            saved_idx = normalization["input_channel_order"].index(channel)
            mean = np.asarray(normalization["input_mean"][saved_idx], dtype=np.float32)
            scale = np.asarray(normalization["input_scale"][saved_idx], dtype=np.float32)
        else:
            scaler = StandardScaler().fit(raw[fit_input_indices, ..., idx].reshape(len(fit_input_indices), -1))
            mean = scaler.mean_.reshape(height, width).astype(np.float32)
            scale = scaler.scale_.reshape(height, width).astype(np.float32)
        scaled[..., idx] = ((raw[..., idx] - mean) / (scale + 1e-6)).astype(np.float32)
        means.append(mean)
        stds.append(scale)
    target = raw[..., 0]
    y_mean = float((normalization or {}).get("target_mean", target[fit_target_indices].mean()))
    y_std = float((normalization or {}).get("target_scale", target[fit_target_indices].std()))
    return MarsScaledVolume(
        values=scaled,
        y_scaled=((target - y_mean) / (y_std + 1e-6)).astype(np.float32),
        ls=ls,
        y_mean=y_mean,
        y_std=y_std,
        height=height,
        width=width,
        latitude=data.latitude,
        longitude=data.longitude,
        split_idx=split_idx,
        input_means=tuple(means),
        input_stds=tuple(stds),
        dataset_id=data.dataset_id,
        source_type=data.source_type,
        data_dir=data.data_dir,
        manifest=data.manifest,
        fingerprint=data.fingerprint,
        sample_starts=sample_starts,
        sample_mars_years=tuple(sample_years),
        blocks=tuple(prepared_blocks),
        train_sample_end=train_sample_end,
        data_directories=data.data_directories,
        split_ranges=split_ranges,
        split_window_starts=split_window_starts,
        split_policy=split_policy,
    )


def identity_snapshot(*, training_dataset: Any, openmars_dir: Any, mcd_dir: Any, raw_dir: Any, selected_channels: Any, window: int, horizon: int, split_ratios: dict[str, Any], normalization: MarsScaledVolume | None = None) -> dict[str, Any]:
    dataset, source_type, source_dir = resolve_data_directory(training_dataset, openmars_dir=openmars_dir, mcd_dir=mcd_dir, raw_dir=raw_dir)
    identity = canonical_dataset_identity(training_dataset=dataset, openmars_dir=openmars_dir, mcd_dir=mcd_dir, raw_dir=raw_dir)
    if normalization is not None and volume_dataset_identity(normalization) != identity:
        raise MarsDataError("Mars data files changed since the training volume was loaded")
    snapshot = {
        **identity,
        "manifest": identity["file_manifest"],
        "selected_channels": parse_channels(selected_channels),
        "window": int(window),
        "horizon": int(horizon),
        "split_ratios": {key: float(value) for key, value in split_ratios.items()},
        "version_status": "verified",
        "window_policy": OPENMARS_WINDOW_POLICY if dataset == "openmars_mcd" else MCD_WINDOW_POLICY,
    }
    if normalization is not None:
        snapshot["normalization"] = {"input_means": [array.tolist() for array in normalization.input_means], "input_stds": [array.tolist() for array in normalization.input_stds], "y_mean": normalization.y_mean, "y_std": normalization.y_std, "split_idx": normalization.split_idx}
        snapshot["split_policy"] = getattr(normalization, "split_policy", "legacy_compatibility")
        snapshot["split_ranges"] = getattr(normalization, "split_ranges", None) or {}
        snapshot["split_window_counts"] = {
            name: len(starts)
            for name, starts in (getattr(normalization, "split_window_starts", None) or {}).items()
        }
        snapshot["year_blocks"] = [
            {
                "mars_year": int(block.mars_year),
                "start": int(block.start),
                "end": int(block.end),
                "source_files": list(block.source_files),
                "segments": [segment.to_metadata() for segment in block.segments],
                "window_count": int(np.sum((normalization.sample_starts >= block.start) & (normalization.sample_starts < block.end))),
            }
            for block in normalization.blocks
        ]
    return snapshot


def assert_identity_current(snapshot: dict[str, Any], *, openmars_dir: Any, mcd_dir: Any, raw_dir: Any) -> dict[str, Any]:
    from services.dataset_identity import DatasetRequestError
    try:
        recorded = normalize_mars_dataset_identity(snapshot)
        current = canonical_dataset_identity(training_dataset=recorded["dataset_id"], openmars_dir=openmars_dir, mcd_dir=mcd_dir, raw_dir=raw_dir)
    except (ValueError, TypeError, OSError) as exc:
        raise DatasetRequestError("dataset_version_changed", "The Mars training data identity is incomplete or unavailable; retrain the model", status_code=409) from exc
    if recorded != current:
        raise DatasetRequestError("dataset_version_changed", "The Mars training data source changed since this task was trained; retrain the model", status_code=409)
    return current
