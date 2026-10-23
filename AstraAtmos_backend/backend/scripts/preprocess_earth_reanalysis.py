"""Build the immutable MERRA-2 Earth daily release from raw hourly files.

The processor deliberately reads one day and one variable at a time.  It
validates the source archive before writing any output, computes a complete
24-hour UTC mean, and then conservatively aggregates the native 0.5 x 0.625
degree grid into global 5 x 5 degree cells using spherical cell-overlap
weights.  Missing values, duplicate dates, inconsistent coordinates, and
incomplete hourly files are hard errors; no value is filled or silently
skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence
from functools import lru_cache
from scipy.sparse import csr_matrix

import netCDF4
import numpy as np

from services.earth_dataset import CHANNELS, SCHEMA, SPLITS, UNITS, split_days


RAW_VARIABLES = {
    "TO3": ("slv", "Dobsons", "Total column ozone"),
    "U10M": ("slv", "m s-1", "10 m eastward wind"),
    "V10M": ("slv", "m s-1", "10 m northward wind"),
    "T2M": ("slv", "K", "2 m air temperature"),
    "SWGDN": ("rad", "W m-2", "Surface incoming shortwave flux"),
}

_DATE_RE = re.compile(r"(?:^|[._-])(?P<date>20\d{6})(?:[._-]|$)")
_PRODUCT_RE = re.compile(r"(?:^|[._-])(?P<product>slv|rad)(?:[._-]|$)", re.I)
_FILL_LIMIT = 1e14


def _date_from_name(path: Path) -> date:
    match = _DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse YYYYMMDD from {path.name}")
    try:
        return datetime.strptime(match.group("date"), "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid date in {path.name}") from exc


def _product_from_name(path: Path) -> str:
    match = _PRODUCT_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot identify slv/rad product in {path.name}")
    return match.group("product").lower()


def discover_merra2_files(source_root: Path) -> dict[date, dict[str, Path]]:
    """Return a strictly paired ``{date: {slv, rad}}`` source index."""
    source_root = Path(source_root)
    found: dict[date, dict[str, Path]] = {}
    for product in ("slv", "rad"):
        directory = source_root / product
        if not directory.is_dir():
            raise ValueError(f"missing MERRA-2 source directory: {directory}")
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".nc", ".nc4", ".netcdf"}:
                continue
            parsed_product = _product_from_name(path)
            if parsed_product != product:
                raise ValueError(f"file {path.name} is in {product}/ but names {parsed_product}")
            day = _date_from_name(path)
            bucket = found.setdefault(day, {})
            if product in bucket:
                raise ValueError(f"duplicate {product} file for {day}: {bucket[product].name} and {path.name}")
            bucket[product] = path
    missing = [f"{day}: {sorted(set(('slv', 'rad')) - set(items))}" for day, items in found.items() if len(items) != 2]
    if missing:
        raise ValueError("missing paired daily files: " + "; ".join(missing[:8]))
    if not found:
        raise ValueError(f"missing paired daily files: no MERRA-2 .nc/.nc4 files found below {source_root}")
    return dict(sorted(found.items()))


def _centers_to_edges(values: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("coordinate centres must be finite and strictly increasing")
    edges = np.empty(len(values) + 1, dtype="float64")
    edges[1:-1] = (values[:-1] + values[1:]) / 2.0
    step = float(np.median(np.diff(values)))
    edges[0] = values[0] - step / 2.0 if lower is None else lower
    edges[-1] = values[-1] + step / 2.0 if upper is None else upper
    if np.any(np.diff(edges) <= 0):
        raise ValueError("coordinate bounds must be strictly increasing")
    return edges


def build_overlap_weights(
    source_centres: Sequence[float],
    target_centres: Sequence[float],
    *,
    source_bounds: tuple[float, float] | None = None,
    target_bounds: tuple[float, float] | None = None,
    spherical: bool = False,
) -> np.ndarray:
    """Build target-by-source normalized overlap weights.

    Latitude weights use ``sin(north)-sin(south)`` when ``spherical`` is true;
    longitude weights use angular overlap.  Bounds are clipped to the stated
    source/target domain, which is essential for the +/-90 degree cells and
    for the global +/-180 degree seam.
    """
    source = np.asarray(source_centres, dtype="float64")
    target = np.asarray(target_centres, dtype="float64")
    periodic = (not spherical and source_bounds is not None
                and np.isclose(source_bounds[1] - source_bounds[0], 360.0)
                and np.isclose(np.median(np.diff(source)) * len(source), 360.0))
    source_edges = _centers_to_edges(
        source,
        *(source_bounds if source_bounds is not None and not periodic else (None, None)),
    )
    target_edges = _centers_to_edges(
        target,
        *(target_bounds if target_bounds is not None else (None, None)),
    )
    overlap = np.zeros((len(target), len(source)), dtype="float64")
    for shift in (-360., 0., 360.) if periodic else (0.,):
        left = np.maximum(target_edges[:-1, None], source_edges[None, :-1] + shift)
        right = np.minimum(target_edges[1:, None], source_edges[None, 1:] + shift)
        if spherical:
            area = np.sin(np.deg2rad(right)) - np.sin(np.deg2rad(left))
        else:
            area = right - left
        overlap += np.where(right > left, area, 0.)
    totals = overlap.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("target grid contains a cell with no source overlap")
    expected = np.diff(np.sin(np.deg2rad(target_edges))) if spherical else np.diff(target_edges)
    if not np.allclose(totals, expected, rtol=1e-10, atol=1e-12):
        raise ValueError("target cells are not completely covered by source cells")
    return overlap / totals[:, None]


@lru_cache(maxsize=8)
def _cached_weights(source, target, source_bounds, target_bounds, spherical):
    return csr_matrix(build_overlap_weights(source, target, source_bounds=source_bounds,
                                           target_bounds=target_bounds, spherical=spherical))


def aggregate_to_grid(
    values: np.ndarray,
    source_lat: Sequence[float],
    source_lon: Sequence[float],
    target_lat: Sequence[float],
    target_lon: Sequence[float],
    *,
    source_lat_bounds: tuple[float, float] | None = None,
    source_lon_bounds: tuple[float, float] | None = None,
    target_lat_bounds: tuple[float, float] | None = None,
    target_lon_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Conservatively aggregate ``[..., source_lat, source_lon]`` to a grid."""
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[-2:] != (len(source_lat), len(source_lon)):
        raise ValueError("values must end with the source latitude/longitude dimensions")
    if not np.isfinite(array).all() or np.any(np.abs(array) >= _FILL_LIMIT):
        raise ValueError("source values contain missing or invalid numbers")
    lat_weights = _cached_weights(tuple(source_lat), tuple(target_lat), source_lat_bounds, target_lat_bounds, True)
    lon_weights = _cached_weights(tuple(source_lon), tuple(target_lon), source_lon_bounds, target_lon_bounds, False)
    flat = array.astype("float64").reshape(-1, len(source_lat), len(source_lon))
    result = np.stack([(lon_weights @ (lat_weights @ frame).T).T for frame in flat])
    return result.reshape(*array.shape[:-2], len(target_lat), len(target_lon))



def _target_grid() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.arange(-87.5, 90.0, 5.0, dtype="float32"),
        np.arange(-177.5, 180.0, 5.0, dtype="float32"),
    )


def _read_daily_mean(path: Path, variable: str, expected_unit: str, expected_date: date,
                     reference_lat: np.ndarray | None, reference_lon: np.ndarray | None, quality: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    with netCDF4.Dataset(path, "r") as source:
        if "time" not in source.dimensions or source.dimensions["time"].size != 24:
            raise ValueError(f"{path.name}/{variable}: expected exactly 24 hourly records")
        if variable not in source.variables:
            raise ValueError(f"{path.name} is missing variable {variable}")
        lat = np.asarray(source.variables["lat"][:], dtype="float64")
        lon = np.asarray(source.variables["lon"][:], dtype="float64")
        _centers_to_edges(lat)
        _centers_to_edges(lon)
        if reference_lat is not None and (lat.shape != reference_lat.shape or not np.array_equal(lat, reference_lat)):
            raise ValueError(f"{path.name}: latitude coordinates do not match the archive")
        if reference_lon is not None and (lon.shape != reference_lon.shape or not np.array_equal(lon, reference_lon)):
            raise ValueError(f"{path.name}: longitude coordinates do not match the archive")
        time_var = source.variables["time"]
        time_var.set_auto_mask(False)
        try:
            decoded = netCDF4.num2date(
                time_var[:], time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
        except Exception as exc:
            raise ValueError(f"{path.name}: cannot decode hourly time coordinate") from exc
        if len(decoded) != 24 or any(
            (item.date() != expected_date or item.hour != index or item.minute != 30 or item.second != 0)
            for index, item in enumerate(decoded)
        ):
            raise ValueError(f"{path.name}: time must contain 00:30..23:30 UTC for {expected_date}")
        variable_obj = source.variables[variable]
        if getattr(variable_obj, "units", None) != expected_unit:
            raise ValueError(f"{path.name}/{variable}: expected units {expected_unit!r}")
        if variable_obj.dimensions != ("time", "lat", "lon"):
            raise ValueError(f"{path.name}/{variable}: invalid dimension order")
        if float(getattr(variable_obj, "scale_factor", 1.0)) != 1.0 or float(getattr(variable_obj, "add_offset", 0.0)) != 0.0:
            raise ValueError(f"{path.name}/{variable}: packed variables are not supported")
        variable_obj.set_auto_maskandscale(False)
        raw = variable_obj[:]
        if np.ma.isMaskedArray(raw) and np.ma.getmaskarray(raw).any():
            raise ValueError(f"{path.name}/{variable}: missing or masked source values")
        values = np.asarray(raw, dtype="float64")
        if values.shape != (24, len(lat), len(lon)) or not np.isfinite(values).all() or np.any(np.abs(values) >= _FILL_LIMIT):
            raise ValueError(f"{path.name}/{variable}: missing or invalid source values")
        for attribute in ("_FillValue", "missing_value"):
            for missing in np.asarray(getattr(variable_obj, attribute, [])).ravel():
                if np.any(values == missing):
                    raise ValueError(f"{path.name}/{variable}: missing source values")
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f"source changed while reading: {path.name}")
        if quality is not None:
            quality.update(raw_min=float(values.min()), raw_max=float(values.max()),
                           values_checked=int(values.size), missing_count=0,
                           hourly_values_sha256=hashlib.sha256(np.asarray(raw).tobytes()).hexdigest())
        return values.mean(axis=0, dtype="float64"), lat, lon


def sha256_file(path: Path) -> str:
    """Hash a complete file in bounded memory and report source I/O failures clearly."""
    path = Path(path)
    try:
        before = path.stat()
        if not path.is_file():
            raise ValueError(f"source file is not a readable regular file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except FileNotFoundError as exc:
        raise ValueError(f"source file is missing while hashing: {path}") from exc
    except PermissionError as exc:
        raise ValueError(f"source file is not readable while hashing: {path}") from exc
    except OSError as exc:
        raise ValueError(f"source file could not be read while hashing: {path}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"source file changed while hashing: {path.name}")
    return digest.hexdigest()


def _source_inventory(index: Mapping[date, Mapping[str, Path]]) -> list[dict]:
    payload = []
    for day in sorted(index):
        pair = index[day]
        for product in ("slv", "rad"):
            path = pair[product]
            try:
                info = path.stat()
                digest = sha256_file(path)
                after = path.stat()
            except FileNotFoundError as exc:
                raise ValueError(f"source file is missing while building inventory: {path}") from exc
            except PermissionError as exc:
                raise ValueError(f"source file is not readable while building inventory: {path}") from exc
            except OSError as exc:
                raise ValueError(f"source file could not be inventoried: {path}: {exc}") from exc
            if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"source file changed while hashing: {path.name}")
            payload.append({"date": day.isoformat(), "product": product,
                            "name": path.name, "bytes": info.st_size,
                            "mtime_ns": info.st_mtime_ns, "sha256": digest})
    return payload


def _source_fingerprint(index: Mapping[date, Mapping[str, Path]]) -> str:
    return _fingerprint_inventory(_source_inventory(index))


def _fingerprint_inventory(inventory: list[dict]) -> str:
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _date_range(index: Mapping[date, Mapping[str, Path]], start_date: str | None, end_date: str | None) -> list[date]:
    dates = list(index)
    if start_date:
        start = date.fromisoformat(start_date)
        if start not in index: raise ValueError("requested start date is missing")
        dates = [item for item in dates if item >= start]
    if end_date:
        end = date.fromisoformat(end_date)
        if end not in index: raise ValueError("requested end date is missing")
        dates = [item for item in dates if item <= end]
    if len(dates) < 3:
        raise ValueError("selected MERRA-2 period must contain at least three daily dates")
    if any(dates[i + 1] - dates[i] != timedelta(days=1) for i in range(len(dates) - 1)):
        raise ValueError("selected MERRA-2 dates are not continuous")
    return dates


def preprocess_merra2(
    source_root: Path,
    output_dir: Path,
    *,
    target_lat: Sequence[float] | None = None,
    target_lon: Sequence[float] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    train_end: str = "2020-12-31",
    validation_end: str = "2021-06-30",
) -> Path:
    """Create ``earth_merra2_daily.nc`` and its manifest in a new directory."""
    output_dir = Path(output_dir)
    output = output_dir / "earth_merra2_daily.nc"
    manifest_path = output_dir / "manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Dataset already exists in {output_dir}; choose a new output directory")
    index = discover_merra2_files(Path(source_root))
    dates = _date_range(index, start_date, end_date)
    production_grid = target_lat is None and target_lon is None
    target_lat = np.asarray(_target_grid()[0] if target_lat is None else target_lat, dtype="float32")
    target_lon = np.asarray(_target_grid()[1] if target_lon is None else target_lon, dtype="float32")
    if len(target_lat) < 2 or len(target_lon) < 2:
        raise ValueError("target grid must contain at least two latitude and longitude cells")
    if np.any(np.diff(target_lat) <= 0) or np.any(np.diff(target_lon) <= 0):
        raise ValueError("target coordinates must be strictly increasing")
    target_lat_bounds = (float(target_lat[0] - (target_lat[1] - target_lat[0]) / 2),
                         float(target_lat[-1] + (target_lat[-1] - target_lat[-2]) / 2))
    target_lon_bounds = (float(target_lon[0] - (target_lon[1] - target_lon[0]) / 2),
                         float(target_lon[-1] + (target_lon[-1] - target_lon[-2]) / 2))
    if np.isclose(target_lat_bounds[0], -90.0) and np.isclose(target_lat_bounds[1], 90.0):
        target_lat_bounds = (-90.0, 90.0)
    if np.isclose(target_lon_bounds[0], -180.0) and np.isclose(target_lon_bounds[1], 180.0):
        target_lon_bounds = (-180.0, 180.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_lat = source_lon = None
    source_lat_bounds = source_lon_bounds = None
    selected_sources = {day: index[day] for day in dates}
    source_inventory = _source_inventory(selected_sources)
    source_fingerprint = _fingerprint_inventory(source_inventory)
    dates64 = np.asarray(dates, dtype="datetime64[D]")
    split = split_days(dates64, train_end, validation_end)
    fields = {name: np.empty((len(dates), len(target_lat), len(target_lon)), dtype="float32") for name in CHANNELS}
    quality_records = []
    for day_index, day in enumerate(dates):
        for variable in CHANNELS:
            product, unit, _ = RAW_VARIABLES[variable]
            quality = {"date": day.isoformat(), "variable": variable, "file": index[day][product].name}
            mean, source_lat, source_lon = _read_daily_mean(
                index[day][product], variable, unit, day, source_lat, source_lon, quality
            )
            if source_lat_bounds is None:
                source_lat_bounds = (-90.0, 90.0) if source_lat[0] == -90 and source_lat[-1] == 90 else None
                source_lon_bounds = (-180.0, 180.0) if np.isclose(np.diff(source_lon)[0] * len(source_lon), 360) else None
                if production_grid and (source_lat_bounds is None or source_lon_bounds is None):
                    raise ValueError("production release requires a complete global source grid")
            aggregated = aggregate_to_grid(
                mean, source_lat, source_lon, target_lat, target_lon,
                source_lat_bounds=source_lat_bounds,
                source_lon_bounds=source_lon_bounds,
                target_lat_bounds=target_lat_bounds,
                target_lon_bounds=target_lon_bounds,
            )
            fields[variable][day_index] = aggregated.astype("float32")
            if production_grid:
                source_edges = _centers_to_edges(source_lat, -90, 90)
                source_area = np.diff(np.sin(np.deg2rad(source_edges)))
                target_edges = _centers_to_edges(target_lat, -90, 90)
                target_area = np.diff(np.sin(np.deg2rad(target_edges)))
                source_mean = float(np.sum(mean * source_area[:, None]) / (2 * len(source_lon)))
                target_mean = float(np.sum(aggregated * target_area[:, None]) / (2 * len(target_lon)))
                stored_mean = float(np.sum(fields[variable][day_index] * target_area[:, None]) / (2 * len(target_lon)))
                if not np.isclose(source_mean, target_mean, rtol=1e-11, atol=1e-10):
                    raise ValueError(f"area conservation failed: {day}/{variable}")
                quality.update(source_area_mean=source_mean, target_area_mean=target_mean,
                               stored_area_mean=stored_mean, conservation_error=abs(source_mean-target_mean),
                               float32_max_error=float(np.max(abs(aggregated-fields[variable][day_index]))))
            quality_records.append(quality)
        if day_index % 10 == 0 or day_index == len(dates) - 1:
            print(f"processed {day.isoformat()} ({day_index + 1}/{len(dates)})", flush=True)

    train_mask = split == SPLITS["train"]
    train_stats = {
        name: {"mean": float(fields[name][train_mask].mean(dtype="float64")),
               "std": float(fields[name][train_mask].std(dtype="float64")),
               "scale_std": float(fields[name][train_mask].std(dtype="float64")) if fields[name][train_mask].std(dtype="float64") >= 1e-6 else 1.0}
        for name in CHANNELS
    }
    root_attrs = {
        "planet": "Earth", "schema": SCHEMA, "source": "NASA MERRA-2",
        "source_product": "MERRA-2 tavg1_2d_slv_Nx + tavg1_2d_rad_Nx",
        "source_sha256": source_fingerprint, "temporal_resolution": "1 day",
        "train_end": train_end, "validation_end": validation_end,
        "processing": "UTC daily mean followed by spherical cell-overlap area-weighted aggregation to global 5 degree cells",
        "spatial_method": "spherical_area_weighted_overlap",
        "temporal_method": "mean_of_complete_source_steps",
        "source_date_start": dates[0].isoformat(), "source_date_end": dates[-1].isoformat(),
    }
    import xarray as xr
    data_vars = {
        name: (("time", "lat", "lon"), fields[name],
               {"units": unit, "long_name": label, "cell_methods": "time: mean; area: mean"})
        for name, unit, label in ((name, RAW_VARIABLES[name][1] if name != "TO3" else "DU", RAW_VARIABLES[name][2]) for name in CHANNELS)
    }
    ds = xr.Dataset(
        data_vars,
        coords={"time": dates64.astype("datetime64[ns]"),
                "lat": ("lat", target_lat, {"units": "degrees_north", "bounds": "lat_bounds"}),
                "lon": ("lon", target_lon, {"units": "degrees_east", "bounds": "lon_bounds"})},
        attrs=root_attrs,
    )
    ds["lat_bounds"] = (("lat", "bounds"), np.column_stack((_centers_to_edges(target_lat)[:-1], _centers_to_edges(target_lat)[1:])))
    ds["lon_bounds"] = (("lon", "bounds"), np.column_stack((_centers_to_edges(target_lon)[:-1], _centers_to_edges(target_lon)[1:])))
    ds["split"] = ("time", split, {"flag_values": np.array([0, 1, 2], dtype="int8"), "flag_meanings": "train validation test"})
    output_dir.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 4, "shuffle": True,
                       "chunksizes": (1, len(target_lat), len(target_lon))} for name in CHANNELS}
    ds.to_netcdf(output, engine="netcdf4", encoding=encoding)
    ds.close()
    manifest = {
        "schema": SCHEMA, "planet": "Earth", "dataset_id": "earth_merra2_daily_v2", "dataset_version": "v2",
        "source": {"product": "MERRA-2", "root_label": "MERRA2/raw", "date_start": dates[0].isoformat(),
                    "date_end": dates[-1].isoformat(), "daily_file_count": len(dates),
                    "source_sha256": source_fingerprint},
        "source_sha256": source_fingerprint,
        "source_inventory_version": 2,
        "source_fingerprint_basis": "SHA256 of canonical sorted date/product/name/bytes/mtime_ns/full-file-sha256 inventory",
        "source_files": source_inventory,
        "data_file": output.name,
        "dimensions": {"time": len(dates), "lat": len(target_lat), "lon": len(target_lon), "bounds": 2},
        "time_start": dates[0].isoformat(), "time_end": dates[-1].isoformat(), "cadence": "daily mean",
        "latitude_range": [float(target_lat_bounds[0]), float(target_lat_bounds[1])],
        "longitude_range": [float(target_lon_bounds[0]), float(target_lon_bounds[1])],
        "cell_bounds": {"latitude": [float(target_lat_bounds[0]), float(target_lat_bounds[1])],
                        "longitude": [float(target_lon_bounds[0]), float(target_lon_bounds[1])]},
        "channel_order": list(CHANNELS), "variables": {}, "splits": {},
        "normalization": {"method": "per_channel_train_mean_std", "fit_split": "train", "reduction": "unweighted time/latitude/longitude population statistics (ddof=0)", "minimum_std": 1e-6, "train_stats": train_stats},
        "processing": {"temporal_method": "mean_of_complete_source_steps", "spatial_method": "spherical_area_weighted_overlap",
                        "target_grid": "global 5x5 degree cells", "source_grid": "MERRA-2 0.5x0.625 degree",
                        "longitude_seam": "periodic global cells at -180/180", "fill_values": "rejected", "arithmetic": "float64 UTC daily means and overlap aggregation; float32 storage",
                        "source_cells": "midpoint latitude bounds clipped at poles; periodic midpoint longitude cells",
                        "source_value_digests": "verification/preprocess_report.json contains SHA256 of each unscaled variable array in C order"},
        "limitations": ["Daily means do not retain hourly variation", "Physical values are stored in source units", "Training and prediction remain disabled"],
    }
    for name, unit in zip(CHANNELS, UNITS):
        manifest["variables"][name] = {"units": unit, "dtype": "float32", "min": float(fields[name].min()), "max": float(fields[name].max())}
    for name, code in SPLITS.items():
        selected = dates64[split == code]
        manifest["splits"][name] = {"start": str(selected[0]), "end": str(selected[-1]), "days": int(len(selected))}
    manifest["data_bytes"] = output.stat().st_size
    manifest["data_sha256"] = _sha256(output)
    final_source_inventory = _source_inventory(selected_sources)
    if [item["sha256"] for item in source_inventory] != [item["sha256"] for item in final_source_inventory]:
        raise ValueError("source archive changed during preprocessing")
    if source_inventory != final_source_inventory:
        raise ValueError("source archive metadata changed during preprocessing")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    verification_dir = output_dir / "verification"
    verification_dir.mkdir(exist_ok=True)
    (verification_dir / "preprocess_report.json").write_text(json.dumps({
        "source_sha256": source_fingerprint, "data_sha256": manifest["data_sha256"],
        "source_fingerprint_basis": manifest["source_fingerprint_basis"],
        "source_files": source_inventory,
        "days": len(dates), "grid": [len(target_lat), len(target_lon)], "train_stats": train_stats,
        "variables_checked": len(quality_records), "missing_values": 0,
        "source_values_checked": sum(item["values_checked"] for item in quality_records),
        "quality": quality_records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _sha256(path: Path) -> str:
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/earth/merra2_daily_v2"))
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-06-30")
    args = parser.parse_args()
    result = preprocess_merra2(args.source_root, args.output_dir, start_date=args.start_date,
                               end_date=args.end_date, train_end=args.train_end,
                               validation_end=args.validation_end)
    print(f"Created {result} ({result.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
