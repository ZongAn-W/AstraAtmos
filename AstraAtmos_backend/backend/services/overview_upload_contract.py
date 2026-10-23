from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from config import N_LAT, N_LON
from services.ozone_units import normalize_ozone_column_units

DIRECT_TARGET_LAT = np.arange(87.5, -90.0, -5.0, dtype=np.float32)
DIRECT_TARGET_LON = np.arange(-180.0, 180.0, 5.0, dtype=np.float32)
DIRECT_SAMPLES_PER_SOL = 8

MCD_CORE_FIELDS = ["Temperature", "U_Wind", "V_Wind", "Solar_Flux_DN"]
MCD_OPTIONAL_FIELDS = ["Pressure", "Dust_Optical_Depth"]
MCD_REQUIRED_FIELDS = ["o3col", *MCD_CORE_FIELDS]

_LAT_ALIASES = ("lat", "latitude")
_LON_ALIASES = ("lon", "longitude")
_LS_ALIASES = ("Ls", "ls", "LS", "L_s", "solar_longitude")
_COUNT_ALIASES = ("count", "counts", "n_obs", "observation_count")

_RAW_MCD_FIELD_MAP = {
    "o3col": "O3COL",
    "Temperature": "T",
    "U_Wind": "U",
    "V_Wind": "V",
    "Solar_Flux_DN": "FSDS",
    "Pressure": "PS",
}

_READY_MCD_ALIASES = {
    "o3col": ("o3col", "O3COL"),
    "Temperature": ("Temperature",),
    "U_Wind": ("U_Wind",),
    "V_Wind": ("V_Wind",),
    "Solar_Flux_DN": ("Solar_Flux_DN",),
    "Pressure": ("Pressure",),
    "Dust_Optical_Depth": ("Dust_Optical_Depth",),
}


@dataclass
class OverviewUploadValidationResult:
    is_valid: bool = False
    data_type: Optional[str] = None
    mars_year: Optional[int] = None
    ls_start: Optional[float] = None
    ls_end: Optional[float] = None
    lat_points: int = 0
    lon_points: int = 0
    ls_points: int = 0
    variables: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


def _find_name(ds: xr.Dataset, aliases: tuple[str, ...]) -> str | None:
    names = list(ds.data_vars) + list(ds.coords)
    lowered = {name.lower(): name for name in names}
    for alias in aliases:
        if alias in names:
            return alias
        found = lowered.get(alias.lower())
        if found:
            return found
    return None


def _data_var(ds: xr.Dataset, aliases: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in ds.data_vars}
    for alias in aliases:
        if alias in ds.data_vars:
            return alias
        found = lowered.get(alias.lower())
        if found:
            return found
    return None


def _values_1d(ds: xr.Dataset, name: str) -> np.ndarray:
    values = np.asarray(ds[name].values, dtype=np.float32).reshape(-1)
    return values


def _daily_mean(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 4:
        arr = np.nanmean(arr, axis=1)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError(f"field must be 3D or 4D, got {arr.shape}")


def _trim_to_groups(values: np.ndarray, group_size: int = DIRECT_SAMPLES_PER_SOL) -> np.ndarray:
    arr = np.asarray(values)
    usable = (arr.shape[0] // group_size) * group_size
    if usable == 0:
        raise ValueError("not enough time samples to build overview groups")
    return arr[:usable]


def _mean_by_sample_group(values: np.ndarray, group_size: int = DIRECT_SAMPLES_PER_SOL) -> np.ndarray:
    arr = _trim_to_groups(np.asarray(values, dtype=np.float32), group_size)
    grouped = arr.reshape((arr.shape[0] // group_size, group_size, *arr.shape[1:]))
    return np.nanmean(grouped, axis=1).astype(np.float32)


def _circular_mean_degrees(values: np.ndarray, group_size: int = DIRECT_SAMPLES_PER_SOL) -> np.ndarray:
    arr = _trim_to_groups(np.asarray(values, dtype=np.float64), group_size)
    grouped = arr.reshape((arr.shape[0] // group_size, group_size))
    radians = np.deg2rad(grouped)
    mean_sin = np.nanmean(np.sin(radians), axis=1)
    mean_cos = np.nanmean(np.cos(radians), axis=1)
    return np.mod(np.rad2deg(np.arctan2(mean_sin, mean_cos)), 360.0).astype(np.float32)


def _regrid_latitude(data: np.ndarray, source_lat: np.ndarray, target_lat: np.ndarray = DIRECT_TARGET_LAT) -> np.ndarray:
    field = np.asarray(data, dtype=np.float64)
    source = np.asarray(source_lat, dtype=np.float64)
    target = np.asarray(target_lat, dtype=np.float64)
    if source.shape[0] == target.shape[0] and np.allclose(source, target, atol=1e-3):
        return field.astype(np.float32)

    order = np.argsort(source)
    source_sorted = source[order]
    field_sorted = field[:, order, :]
    regridded = np.empty((field.shape[0], target.shape[0], field.shape[2]), dtype=np.float32)
    for lon_idx in range(field.shape[2]):
        interpolator = interp1d(
            source_sorted,
            field_sorted[:, :, lon_idx],
            axis=1,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        regridded[:, :, lon_idx] = interpolator(target).astype(np.float32)
    return regridded


def _normalize_lon(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(lon, dtype=np.float32)
    if values.shape[0] != N_LON:
        return values, None
    normalized = ((values + 180.0) % 360.0) - 180.0
    order = np.argsort(normalized)
    sorted_lon = normalized[order]
    if np.allclose(sorted_lon, DIRECT_TARGET_LON, atol=1e-3):
        if np.array_equal(order, np.arange(values.shape[0])):
            return sorted_lon.astype(np.float32), None
        return sorted_lon.astype(np.float32), order
    return values.astype(np.float32), None


def _reorder_longitude(data: np.ndarray, order: np.ndarray | None) -> np.ndarray:
    if order is None:
        return data
    return np.asarray(data)[..., order]


def _overview_lat_order(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(lat, dtype=np.float32)
    if values.shape[0] != N_LAT:
        raise ValueError(f"Data Overview grid must have {N_LAT} latitude points, got {values.shape[0]}")
    if np.allclose(values, DIRECT_TARGET_LAT, atol=1e-3):
        return DIRECT_TARGET_LAT.astype(np.float32), None
    reverse_order = np.arange(values.shape[0] - 1, -1, -1)
    if np.allclose(values[reverse_order], DIRECT_TARGET_LAT, atol=1e-3):
        return DIRECT_TARGET_LAT.astype(np.float32), reverse_order
    raise ValueError("Data Overview grid coordinates must use 5-degree cell centers")


def _overview_lon_order(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(lon, dtype=np.float32)
    if values.shape[0] != N_LON:
        raise ValueError(f"Data Overview grid must have {N_LON} longitude points, got {values.shape[0]}")
    normalized, order = _normalize_lon(values)
    if not np.allclose(normalized, DIRECT_TARGET_LON, atol=1e-3):
        raise ValueError("Data Overview grid coordinates must use 5-degree cell centers")
    return DIRECT_TARGET_LON.astype(np.float32), order


def _normalize_overview_grid(lat: np.ndarray, lon: np.ndarray, fields: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    target_lat, lat_order = _overview_lat_order(lat)
    target_lon, lon_order = _overview_lon_order(lon)
    normalized_fields = {}
    for name, values in fields.items():
        arr = np.asarray(values)
        if arr.ndim != 3:
            raise ValueError(f"field {name} must be 3D, got {arr.shape}")
        if arr.shape[1:] != (len(lat), len(lon)):
            raise ValueError(f"field {name} grid {arr.shape[1:]} does not match lat/lon {(len(lat), len(lon))}")
        if lat_order is not None:
            arr = arr[:, lat_order, :]
        if lon_order is not None:
            arr = arr[:, :, lon_order]
        normalized_fields[name] = arr
    return target_lat, target_lon, normalized_fields


def _sort_time(data: dict) -> dict:
    ls = np.asarray(data["ls"], dtype=np.float32)
    order = np.argsort(ls)
    out = dict(data)
    out["ls"] = ls[order]
    for name, values in list(out.items()):
        if name == "ls":
            continue
        if name in {"data_type", "lat", "lon", "mars_year"}:
            continue
        arr = np.asarray(values)
        if arr.ndim >= 1 and arr.shape[0] == order.shape[0]:
            out[name] = arr[order]
    return out


def _finite_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.isfinite(arr).sum()) / max(int(arr.size), 1)


def _valid_ls(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr


def _extract_mars_year(ds: xr.Dataset, filename: str = "") -> Optional[int]:
    if "MY" in ds.data_vars:
        values = np.asarray(ds["MY"].values).reshape(-1)
        valid = values[np.isfinite(values.astype(float))]
        if valid.size:
            return int(valid[0])
    for attr in ("mars_year", "MY", "Mars_Year", "year"):
        if attr in ds.attrs:
            try:
                return int(ds.attrs[attr])
            except (TypeError, ValueError):
                pass
    if filename:
        match = re.search(r"[Mm][Yy]_?(\d{2,3})", Path(filename).name)
        if match:
            return int(match.group(1))
    return None


def _filename_data_type_hint(filename: str = "") -> str | None:
    stem = Path(filename or "").name.lower()
    if "nomad" in stem:
        return "nomad"
    if "openmars" in stem:
        return "openmars"
    if re.search(r"(^|[_\-.])mcd([_\-.]|$)", stem):
        return "mcd"
    return None


def _has_any(ds: xr.Dataset, aliases_by_field: dict[str, tuple[str, ...]]) -> bool:
    return any(_data_var(ds, aliases) is not None for aliases in aliases_by_field.values())


def is_raw_mcd_dataset(ds: xr.Dataset) -> bool:
    return all(raw_name in ds.data_vars for raw_name in ("O3COL", "T", "U", "V", "FSDS", "PS"))


def is_ready_mcd_dataset(ds: xr.Dataset) -> bool:
    return _has_any(ds, _READY_MCD_ALIASES) and any(
        _data_var(ds, aliases) is not None
        for field, aliases in _READY_MCD_ALIASES.items()
        if field != "o3col"
    )


def classify_overview_upload_dataset(ds: xr.Dataset, filename: str = "") -> str | None:
    filename_hint = _filename_data_type_hint(filename)
    if _data_var(ds, _COUNT_ALIASES) is not None and _data_var(ds, ("o3col", "O3COL")) is not None:
        return "nomad"
    if is_ready_mcd_dataset(ds):
        return "mcd"
    if _has_any(ds, {key: (value,) for key, value in _RAW_MCD_FIELD_MAP.items()}):
        if any(name in ds.data_vars for name in ("U", "V", "T", "FSDS", "PS")):
            return "mcd"
    if filename_hint in {"mcd", "nomad"}:
        return filename_hint
    if _data_var(ds, ("o3col", "O3COL")) is not None:
        return filename_hint or "openmars"
    if filename_hint == "openmars":
        return "openmars"
    return None


def _normalize_ozone_source(
    ds: xr.Dataset,
    data_type: str,
    filename: str = "",
    sort: bool = True,
    normalize_mcd_ozone: bool = False,
) -> dict:
    lat_name = _find_name(ds, _LAT_ALIASES)
    lon_name = _find_name(ds, _LON_ALIASES)
    ls_name = _find_name(ds, _LS_ALIASES)
    o3_name = _data_var(ds, ("o3col", "O3COL"))
    if not lat_name:
        raise ValueError("missing lat/latitude coordinate")
    if not lon_name:
        raise ValueError("missing lon/longitude coordinate")
    if not ls_name:
        raise ValueError("missing Ls/ls variable")
    if not o3_name:
        raise ValueError("missing o3col ozone field")

    lat = _values_1d(ds, lat_name)
    lon = _values_1d(ds, lon_name)
    ls = _values_1d(ds, ls_name)
    o3 = _daily_mean(ds[o3_name].values)
    if normalize_mcd_ozone:
        o3 = normalize_ozone_column_units(
            o3,
            ds[o3_name].attrs.get("units"),
            allow_mcd_legacy_heuristic=True,
        )
    if o3.shape[0] != ls.shape[0]:
        raise ValueError(f"o3col time dimension {o3.shape[0]} does not match Ls length {ls.shape[0]}")
    if o3.shape[1:] != (lat.shape[0], lon.shape[0]):
        raise ValueError(f"o3col grid {o3.shape[1:]} does not match lat/lon {(lat.shape[0], lon.shape[0])}")

    grid_fields: dict[str, np.ndarray] = {"o3col": o3.astype(np.float32)}
    if data_type == "nomad":
        count_name = _data_var(ds, _COUNT_ALIASES)
        if not count_name:
            raise ValueError("NOMAD upload missing count field")
        count = np.asarray(ds[count_name].values)
        if count.shape != o3.shape:
            raise ValueError(f"NOMAD count shape {count.shape} must match o3col shape {o3.shape}")
        grid_fields["count"] = count.astype(np.int32)
        uncertainty_name = _data_var(ds, ("uncertainty", "o3_abund_uncty"))
        if uncertainty_name:
            uncertainty = _daily_mean(ds[uncertainty_name].values)
            if uncertainty.shape == o3.shape:
                grid_fields["uncertainty"] = uncertainty.astype(np.float32)

    lat, lon, grid_fields = _normalize_overview_grid(lat, lon, grid_fields)
    out = {
        "data_type": data_type,
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "ls": ls.astype(np.float32),
        "mars_year": _extract_mars_year(ds, filename),
    }
    out.update(grid_fields)
    return _sort_time(out) if sort else out


def _normalize_ready_mcd(ds: xr.Dataset, filename: str = "") -> dict:
    out = _normalize_ozone_source(ds, "mcd", filename, sort=False, normalize_mcd_ozone=True)
    lat_name = _find_name(ds, _LAT_ALIASES)
    lon_name = _find_name(ds, _LON_ALIASES)
    source_lat = _values_1d(ds, lat_name) if lat_name else out["lat"]
    source_lon = _values_1d(ds, lon_name) if lon_name else out["lon"]
    for output_name in [*MCD_CORE_FIELDS, *MCD_OPTIONAL_FIELDS]:
        input_name = _data_var(ds, _READY_MCD_ALIASES[output_name])
        if input_name is None:
            if output_name in MCD_CORE_FIELDS:
                raise ValueError(f"MCD upload missing required field {output_name}")
            continue
        values = _daily_mean(ds[input_name].values)
        _, _, normalized = _normalize_overview_grid(source_lat, source_lon, {output_name: values})
        values = normalized[output_name]
        if values.shape != out["o3col"].shape:
            raise ValueError(f"MCD field {output_name} shape {values.shape} must match o3col shape {out['o3col'].shape}")
        out[output_name] = values.astype(np.float32)
    return _sort_time(out)


def _normalize_raw_mcd(ds: xr.Dataset, filename: str = "") -> dict:
    lat_name = _find_name(ds, _LAT_ALIASES)
    lon_name = _find_name(ds, _LON_ALIASES)
    ls_name = _find_name(ds, _LS_ALIASES)
    if not lat_name:
        raise ValueError("missing lat/latitude coordinate")
    if not lon_name:
        raise ValueError("missing lon/longitude coordinate")
    if not ls_name:
        raise ValueError("missing LS/Ls variable")
    missing = [out_name for out_name, raw_name in _RAW_MCD_FIELD_MAP.items() if raw_name not in ds.data_vars]
    missing_required = [name for name in missing if name in MCD_REQUIRED_FIELDS]
    if missing_required:
        raise ValueError(f"MCD upload missing required fields: {', '.join(missing_required)}")

    source_lat = _values_1d(ds, lat_name)
    source_lon = _values_1d(ds, lon_name)
    raw_ls = _values_1d(ds, ls_name)
    if source_lon.shape[0] != N_LON:
        raise ValueError(f"MCD upload lon grid must have {N_LON} points, got {source_lon.shape[0]}")

    data_vars = {}
    for output_name, raw_name in _RAW_MCD_FIELD_MAP.items():
        if raw_name not in ds.data_vars:
            continue
        daily = _mean_by_sample_group(ds[raw_name].values)
        if output_name == "o3col":
            daily = normalize_ozone_column_units(
                daily,
                ds[raw_name].attrs.get("units"),
                allow_mcd_legacy_heuristic=True,
            )
        data_vars[output_name] = _regrid_latitude(daily, source_lat)
    data_vars["Dust_Optical_Depth"] = np.full_like(data_vars["o3col"], np.nan, dtype=np.float32)

    ls = _circular_mean_degrees(raw_ls)
    lon, lon_order = _normalize_lon(source_lon)
    if not np.allclose(lon, DIRECT_TARGET_LON, atol=1e-3):
        raise ValueError("Data Overview grid coordinates must use 5-degree cell centers")
    out = {
        "data_type": "mcd",
        "lat": DIRECT_TARGET_LAT.astype(np.float32),
        "lon": lon.astype(np.float32),
        "ls": ls.astype(np.float32),
        "mars_year": _extract_mars_year(ds, filename),
    }
    out.update({name: _reorder_longitude(np.asarray(values, dtype=np.float32), lon_order) for name, values in data_vars.items()})
    return _sort_time(out)


def normalize_overview_upload_dataset(ds: xr.Dataset, filename: str = "") -> dict:
    data_type = classify_overview_upload_dataset(ds, filename)
    if data_type is None:
        raise ValueError("unsupported Data Overview upload type; expected MCD, OpenMARS, or gridded NOMAD")
    if data_type == "openmars":
        return _normalize_ozone_source(ds, "openmars", filename)
    if data_type == "nomad":
        return _normalize_ozone_source(ds, "nomad", filename)
    if data_type == "mcd":
        if all(raw_name in ds.data_vars for raw_name in ("O3COL", "T", "U", "V", "FSDS")):
            return _normalize_raw_mcd(ds, filename)
        return _normalize_ready_mcd(ds, filename)
    raise ValueError(f"unsupported Data Overview upload type: {data_type}")


def validate_overview_upload_dataset(
    ds: xr.Dataset,
    filename: str = "",
    allow_ready_mcd: bool = True,
) -> OverviewUploadValidationResult:
    result = OverviewUploadValidationResult(
        data_type=classify_overview_upload_dataset(ds, filename),
        variables=list(ds.data_vars),
    )
    if result.data_type is None:
        if _find_name(ds, _LAT_ALIASES) and _find_name(ds, _LON_ALIASES) and _find_name(ds, _LS_ALIASES) and "MY" in ds.data_vars:
            result.error = "OpenMARS upload missing o3col ozone field"
            return result
        result.error = "Unsupported dataset. Upload MCD, OpenMARS, or gridded NOMAD data for Data Overview."
        return result
    try:
        data = normalize_overview_upload_dataset(ds, filename)
    except ValueError as exc:
        result.error = str(exc)
        return result

    if result.data_type == "mcd" and not allow_ready_mcd and not is_raw_mcd_dataset(ds):
        result.error = (
            "MCD uploads must use the raw 3-hour MCD file format with O3COL, T, U, V, FSDS, and PS fields, "
            "for example MCD_MY33_global_3h_5deg_10m_ls_lst.nc"
        )
        return result

    result.data_type = data["data_type"]
    result.mars_year = data.get("mars_year")
    result.lat_points = int(np.asarray(data.get("lat", [])).shape[0])
    result.lon_points = int(np.asarray(data.get("lon", [])).shape[0])
    ls = _valid_ls(data.get("ls", []))
    if ls.size == 0:
        result.error = "Ls variable contains no valid values"
        return result
    if float(np.nanmin(ls)) < -0.5 or float(np.nanmax(ls)) > 360.5:
        result.error = f"Ls values out of range 0-360: [{float(np.nanmin(ls)):.1f}, {float(np.nanmax(ls)):.1f}]"
        return result
    if result.lat_points != N_LAT or result.lon_points != N_LON:
        result.error = f"Data Overview grid must be {N_LAT}x{N_LON}, got {result.lat_points}x{result.lon_points}"
        return result
    if _finite_ratio(data.get("o3col")) < 0.10:
        result.error = "o3col valid data ratio is too low"
        return result
    if result.data_type == "mcd":
        missing = [name for name in MCD_REQUIRED_FIELDS if name not in data]
        if missing:
            result.error = f"MCD upload missing required fields: {', '.join(missing)}"
            return result
    if result.data_type == "nomad":
        count = np.asarray(data.get("count"))
        if count.shape != np.asarray(data.get("o3col")).shape:
            result.error = "NOMAD count shape must match o3col"
            return result
        if not np.any(count > 0):
            result.error = "NOMAD count has no observed cells"
            return result

    result.ls_points = int(ls.shape[0])
    result.ls_start = float(round(float(np.nanmin(ls)), 2))
    result.ls_end = float(round(float(np.nanmax(ls)), 2))
    result.is_valid = True
    return result
