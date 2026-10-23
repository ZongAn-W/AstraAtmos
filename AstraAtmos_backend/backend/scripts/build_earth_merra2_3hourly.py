"""Stream raw hourly MERRA-2 slv/rad into an external three-hour Earth release.

This offline builder never changes the registered daily releases. Each group of
three hourly mean fields is averaged, then conservatively regridded by spherical
cell overlap. Missing support is propagated, never renormalized or filled with
zero. Existing output is verified and reused; incompatible or partial output is
retained and rejected. Use a fresh external directory after an interrupted build.

Dataset identity is bound to *content*, not to how the content was obtained.
Every published manifest declares its fingerprint rule: rule 1 is the legacy
whole-manifest digest kept for already registered publications, rule 2 hashes the
same manifest without the source_mtimes provenance block and with source_files
projected onto (date, product, path, bytes, sha256). File-system timestamps
(mtime_ns) and the redundant name suffix never move the identity, while a changed
path, size, digest or inventory membership always does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import netCDF4
import numpy as np
from scipy.sparse import csr_matrix

# Direct execution works from any working directory; no project config import or
# environment activation is required beyond the project's specified interpreter.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.dataset_identity import build_dataset_fingerprint
from services.netcdf_read_lock import netcdf_read_lock
from scripts.preprocess_earth_reanalysis import build_overlap_weights, sha256_file

DATASET_ID = "earth_merra2_3hourly_v1"
DATASET_VERSION = "v1"
SCHEMA = "astraatmos_earth_3hourly_v1"
DATA_FILE = "earth_merra2_3hourly.nc"
DATA_FILENAME = DATA_FILE
CHANNELS = ("TO3", "U10M", "V10M", "T2M", "SWGDN")
UNITS = ("DU", "m s-1", "m s-1", "K", "W m-2")
RAW_VARIABLES = {
    "TO3": ("slv", ("Dobsons", "DU"), "Total column ozone"),
    "U10M": ("slv", ("m s-1",), "10 m eastward wind"),
    "V10M": ("slv", ("m s-1",), "10 m northward wind"),
    "T2M": ("slv", ("K",), "2 m air temperature"),
    "SWGDN": ("rad", ("W m-2",), "Surface incoming shortwave flux"),
}
TIME_UNITS = "hours since 2020-01-01 00:00:00 UTC"
CALENDAR = "proleptic_gregorian"
FINGERPRINT_RULE_LEGACY = 1
FINGERPRINT_RULE_CONTENT = 2
FINGERPRINT_RULE = FINGERPRINT_RULE_CONTENT
SOURCE_INVENTORY_VERSION = 3
# Content identity of one source inventory entry. name is a redundant suffix of
# path and mtime_ns is file-system metadata, so neither may enter an identity
# digest; both stay in the manifest for human and change-detection use.
SOURCE_IDENTITY_FIELDS = ("date", "product", "path", "bytes", "sha256")
FINGERPRINT_BASIS = (
    "build_dataset_fingerprint(dataset_id, dataset_version, "
    "SHA256(canonical manifest excluding dataset_fingerprint, source_mtimes and "
    "fingerprint_rule, with source_files projected to date/product/path/bytes/sha256), "
    "data_sha256); canonical JSON: sort_keys=True, separators=(',', ':'), ensure_ascii=True"
)
SOURCE_FINGERPRINT_BASIS = (
    "SHA256(canonical sorted date/product/path/bytes/full-file-content-SHA256 identity "
    "inventory); name and mtime_ns are provenance only and never enter an identity digest"
)
_DATE_RE = re.compile(r"(?:^|[._-])(?P<date>20\d{6})(?:[._-]|$)")
_PRODUCT_RE = re.compile(r"(?:^|[._-])(?P<product>slv|rad)(?:[._-]|$)", re.I)
_FILL_LIMIT = 1e14


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def fingerprint_rule(manifest: Mapping) -> int:
    """Resolve the identity rule of a manifest; unknown values fail closed.

    A missing field means the legacy rule. Anything that is not exactly 1 or 2
    (0, negative, string, float, bool) is refused rather than silently digesting
    as the legacy rule, because a silent fallback would publish a package whose
    identity no consumer can reproduce.
    """
    value = manifest.get("fingerprint_rule", FINGERPRINT_RULE_LEGACY)
    if (isinstance(value, bool) or not isinstance(value, int)
            or value not in (FINGERPRINT_RULE_LEGACY, FINGERPRINT_RULE_CONTENT)):
        raise ValueError(f"unsupported fingerprint_rule: {value!r}")
    return value


def source_inventory_version(manifest: Mapping) -> int:
    """Resolve the source inventory basis; unknown values fail closed."""
    value = manifest.get("source_inventory_version", 2)
    if isinstance(value, bool) or not isinstance(value, int) or value not in (2, SOURCE_INVENTORY_VERSION):
        raise ValueError(f"unsupported source_inventory_version: {value!r}")
    return value


def source_identity_projection(files) -> list:
    """Project inventory entries onto the fields that define source content."""
    projection = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("source inventory entry must be a JSON object")
        missing = [field for field in SOURCE_IDENTITY_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"source inventory entry lacks {', '.join(missing)}")
        projection.append({field: entry[field] for field in SOURCE_IDENTITY_FIELDS})
    return projection


def source_inventory_sha256(files, version: int) -> str:
    """Digest an inventory with the basis its version declares.

    Version 3 hashes the content identity projection; earlier versions hash every
    recorded field, including name and mtime_ns.
    """
    if version not in (2, SOURCE_INVENTORY_VERSION):
        raise ValueError(f"unsupported source_inventory_version: {version!r}")
    payload = source_identity_projection(files) if version == SOURCE_INVENTORY_VERSION else files
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def manifest_content_sha256(manifest: dict) -> str:
    """Digest the manifest content that defines the dataset identity.

    Rule 1 keeps the historical behaviour byte for byte: every manifest field
    except the self-referential dataset_fingerprint. Rule 2 additionally excludes
    the source_mtimes provenance block and the rule marker itself, and projects
    source_files onto its identity fields, so file-system metadata cannot move a
    published identity.
    """
    rule = fingerprint_rule(manifest)
    if rule == FINGERPRINT_RULE_LEGACY:
        return hashlib.sha256(canonical_json({
            key: value for key, value in manifest.items() if key != "dataset_fingerprint"
        })).hexdigest()
    payload = {key: value for key, value in manifest.items()
               if key not in ("dataset_fingerprint", "source_mtimes", "fingerprint_rule")}
    files = payload.get("source_files")
    if not isinstance(files, list):
        raise ValueError("source_files must be a JSON array")
    payload["source_files"] = source_identity_projection(files)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def dataset_fingerprint(manifest: dict) -> str:
    return build_dataset_fingerprint(manifest["dataset_id"], manifest["dataset_version"],
                                     manifest_content_sha256(manifest), manifest["data_sha256"])


def target_grid() -> tuple[np.ndarray, np.ndarray]:
    return (np.arange(240, dtype="float64") * 0.75 - 89.625,
            np.arange(480, dtype="float64") * 0.75 - 179.625)


def _values_and_valid(values: np.ndarray, valid_mask: np.ndarray | None = None):
    array = np.asarray(np.ma.getdata(values), dtype="float64")
    valid = (~np.ma.getmaskarray(values) & np.isfinite(array)
             & (np.abs(array) < _FILL_LIMIT))
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != array.shape:
            raise ValueError("valid_mask must match values shape")
        valid &= mask
    return array, valid


def temporal_mean_three_hours(values: np.ndarray,
                              valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Mean consecutive groups of three on axis 0; any missing member -> NaN."""
    array, valid = _values_and_valid(values, valid_mask)
    if array.ndim < 1 or array.shape[0] == 0 or array.shape[0] % 3:
        raise ValueError("hourly time length must be a positive multiple of three")
    shape = (array.shape[0] // 3, 3, *array.shape[1:])
    complete = valid.reshape(shape).all(axis=1)
    # Invalid arithmetic cannot contaminate valid output; completeness determines
    # all emitted values. This is not a reduced-support or zero-filled average.
    with np.errstate(invalid="ignore", over="ignore"):
        means = array.reshape(shape).mean(axis=1, dtype="float64")
    return np.where(complete, means, np.nan)


def three_hour_mean(values: np.ndarray, valid_mask: np.ndarray | None = None):
    means = temporal_mean_three_hours(values, valid_mask)
    return means, np.isfinite(means)


def _global_coordinates(lat: Sequence[float], lon: Sequence[float]):
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    for name, axis in (("latitude", lat), ("longitude", lon)):
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
            raise ValueError(f"source {name} must be a finite 1-D coordinate")
        delta = np.diff(axis)
        if np.any(delta <= 0) or not np.allclose(delta, delta[0], atol=1e-9, rtol=0):
            raise ValueError(f"source {name} must be uniformly spaced and increasing")
    if lat[0] < -90 or lat[-1] > 90 or not np.isclose(lat[0], -90) or not np.isclose(lat[-1], 90):
        raise ValueError("source latitude must include both global poles")
    if lon[0] < -180 or lon[-1] >= 180 or not np.isclose(np.diff(lon)[0] * len(lon), 360):
        raise ValueError("source longitude must cover one periodic global domain in [-180,180)")
    return lat, lon


@lru_cache(maxsize=4)
def _regrid_weights(lat: tuple[float, ...], lon: tuple[float, ...]):
    target_lat, target_lon = target_grid()
    lat_dense = build_overlap_weights(lat, target_lat, source_bounds=(-90., 90.),
                                     target_bounds=(-90., 90.), spherical=True)
    lon_dense = build_overlap_weights(lon, target_lon, source_bounds=(-180., 180.),
                                     target_bounds=(-180., 180.), spherical=False)
    return (csr_matrix(lat_dense), csr_matrix(lon_dense),
            csr_matrix((lat_dense > 0).astype("float64")),
            csr_matrix((lon_dense > 0).astype("float64")))


def conservative_regrid(values: np.ndarray, source_lat: Sequence[float],
                        source_lon: Sequence[float],
                        valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Regrid [...,lat,lon]; any missing nonzero-overlap support -> target NaN.

    Latitude overlaps use sin(north)-sin(south), longitude overlaps are periodic
    angular widths. Midpoint native cells are clipped at the poles. The output
    uses the exact 240 x 480 global 0.75 degree cell centres.
    """
    lat, lon = _global_coordinates(source_lat, source_lon)
    array, valid = _values_and_valid(values, valid_mask)
    if array.ndim < 2 or array.shape[-2:] != (len(lat), len(lon)):
        raise ValueError("values must end with the source latitude/longitude shape")
    lat_weight, lon_weight, lat_support, lon_support = _regrid_weights(tuple(lat), tuple(lon))
    frames = array.reshape(-1, len(lat), len(lon))
    masks = valid.reshape(frames.shape)
    result = np.empty((len(frames), 240, 480), dtype="float64")
    for index, (frame, mask) in enumerate(zip(frames, masks)):
        # Zero is only an internal sparse multiply placeholder. Every target
        # touched by invalid support is subsequently marked missing, never zero.
        safe = np.where(mask, frame, 0.0)
        mapped = (lon_weight @ (lat_weight @ safe).T).T
        invalid_support = (lon_support @ (lat_support @ (~mask).astype("float64")).T).T
        result[index] = np.where(invalid_support == 0, mapped, np.nan)
    return result.reshape(*array.shape[:-2], 240, 480)


def regrid_three_hour_mean(values, valid_mask, source_lat, source_lon):
    output = conservative_regrid(temporal_mean_three_hours(values, valid_mask), source_lat, source_lon)
    return output, np.isfinite(output)


def _signature(path: Path) -> tuple[int, int]:
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise ValueError(f"source file is missing: {path}") from exc
    except PermissionError as exc:
        raise ValueError(f"source file is not readable: {path}") from exc
    except OSError as exc:
        raise ValueError(f"source file metadata could not be read: {path}: {exc}") from exc
    if not path.is_file():
        raise ValueError(f"source path is not a regular file: {path}")
    return info.st_size, info.st_mtime_ns


def _period(start_date: str, end_date: str, smoke_days: int | None):
    first, last = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if first < date(2020, 1, 1) or last > date(2021, 12, 31) or last < first:
        raise ValueError("requested period must be within 2020-01-01..2021-12-31")
    if smoke_days is not None:
        if not 7 <= smoke_days <= 30:
            raise ValueError("smoke_days must be between 7 and 30")
        smoke_last = first + timedelta(days=smoke_days - 1)
        if smoke_last > last:
            raise ValueError("requested end date does not contain smoke_days")
        last = smoke_last
    if (last - first).days + 1 < 7:
        raise ValueError("selected period must contain at least seven complete UTC days")
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    return days


def discover_source_files(source_root: Path, days: Sequence[date]):
    """Index only selected slv/rad daily products; chm is never inspected."""
    selected = set(days)
    found = {day: {} for day in days}
    for product in ("slv", "rad"):
        folder = source_root / product
        if not folder.is_dir():
            raise ValueError(f"missing source product directory: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".nc", ".nc4", ".netcdf"}:
                continue
            match = _DATE_RE.search(path.name)
            if match is None:
                raise ValueError(f"cannot parse source date: {path.name}")
            day = datetime.strptime(match.group("date"), "%Y%m%d").date()
            if day not in selected:
                continue
            named_product = _PRODUCT_RE.search(path.name)
            if named_product is None or named_product.group("product").lower() != product:
                raise ValueError(f"source product does not match its directory: {path.name}")
            if product in found[day]:
                raise ValueError(f"duplicate {product} source file for {day}")
            found[day][product] = path
    for day, pair in found.items():
        if set(pair) != {"slv", "rad"}:
            raise ValueError(f"missing slv/rad source pair for {day}")
    return found


def _read_hourly(path: Path, variable: str, day: date,
                 reference_lat: np.ndarray | None, reference_lon: np.ndarray | None):
    before = _signature(path)
    with netcdf_read_lock(), netCDF4.Dataset(path, "r") as source:
        if variable not in source.variables or "time" not in source.variables:
            raise ValueError(f"{path.name}: missing time or {variable}")
        if "lat" not in source.variables or "lon" not in source.variables:
            raise ValueError(f"{path.name}: missing global coordinates")
        lat, lon = _global_coordinates(source.variables["lat"][:], source.variables["lon"][:])
        if reference_lat is not None and (not np.array_equal(lat, reference_lat) or not np.array_equal(lon, reference_lon)):
            raise ValueError(f"{path.name}: source coordinates changed within archive")
        time_var = source.variables["time"]
        # MERRA-2 integer-minute coordinates carry a float valid_range near
        # +/-1e15 which netCDF4 cannot safely cast. Validate decoded coordinate
        # values explicitly rather than relying on that automatic mask.
        time_var.set_auto_mask(False)
        time_values = np.asarray(time_var[:], dtype="float64")
        if not np.isfinite(time_values).all() or np.any(np.abs(time_values) >= _FILL_LIMIT):
            raise ValueError(f"{path.name}: missing or invalid time coordinate")
        for attribute in ("_FillValue", "missing_value"):
            if any(np.any(time_values == missing)
                   for missing in np.asarray(getattr(time_var, attribute, [])).ravel()):
                raise ValueError(f"{path.name}: missing time coordinate")
        calendar = getattr(time_var, "calendar", "standard")
        if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
            raise ValueError(f"{path.name}: time requires a Gregorian calendar")
        decoded = netCDF4.num2date(time_values, time_var.units,
                                  calendar=calendar,
                                  only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        expected = [datetime.combine(day, datetime.min.time()) + timedelta(hours=i, minutes=30)
                    for i in range(24)]
        if time_var.dimensions != ("time",) or len(decoded) != 24 or any(
            item.replace(tzinfo=None) != wanted for item, wanted in zip(decoded, expected)
        ):
            raise ValueError(f"{path.name}: time must contain exactly 00:30..23:30 UTC for {day}")
        source_var = source.variables[variable]
        raw_unit = getattr(source_var, "units", None)
        if raw_unit not in RAW_VARIABLES[variable][1]:
            raise ValueError(f"{path.name}/{variable}: unexpected unit {raw_unit!r}")
        if source_var.dimensions != ("time", "lat", "lon"):
            raise ValueError(f"{path.name}/{variable}: expected dimension order time,lat,lon")
        source_var.set_auto_maskandscale(True)
        raw = source_var[:]
        values, valid = _values_and_valid(raw)
        if values.shape != (24, len(lat), len(lon)):
            raise ValueError(f"{path.name}/{variable}: wrong hourly shape")
        # netCDF4 masks declared _FillValue/missing_value before auto scaling.
        # The finite-value test also catches NaN/Inf and undeclared fill sentinels.
        if before != _signature(path):
            raise ValueError(f"source file changed while reading: {path.name}")
    return values, valid, lat, lon, raw_unit


def _source_inventory(index: dict, source_root: Path):
    inventory, signatures, digests = [], {}, {}
    for day in sorted(index):
        pair = index[day]
        for product in ("slv", "rad"):
            path = pair[product]
            signature = _signature(path)
            digest = sha256_file(path)
            if signature != _signature(path):
                raise ValueError(f"source file changed while hashing: {path.name}")
            signatures[path] = signature
            digests[(day.isoformat(), product)] = digest
            inventory.append({"date": day.isoformat(), "product": product,
                              "path": path.relative_to(source_root).as_posix(), "name": path.name,
                              "bytes": signature[0], "sha256": digest})
    return inventory, signatures, digests


def source_mtimes(signatures: Mapping[Path, tuple], source_root: Path) -> dict:
    """Provenance block: cheap change detection, never part of any identity digest."""
    return {path.relative_to(source_root).as_posix(): signature[1]
            for path, signature in signatures.items()}


def _assert_source_digests(index: dict, source_root: Path,
                           expected: dict[tuple[str, str], str]) -> None:
    _, _, current = _source_inventory(index, source_root)
    if current != expected:
        raise ValueError("source archive content changed during build")


def _safe_output_dir(output_dir: Path, source_root: Path) -> Path:
    output = Path(output_dir).resolve()
    if output == REPOSITORY_ROOT or REPOSITORY_ROOT in output.parents:
        raise ValueError("output directory must be outside the Git checkout")
    if output == source_root or source_root in output.parents:
        raise ValueError("output directory must not be inside the raw source archive")
    return output


def _split_code(day: date) -> int:
    return 0 if day <= date(2020, 12, 31) else (1 if day <= date(2021, 6, 30) else 2)


def _times(days: Sequence[date]):
    epoch = datetime(2020, 1, 1)
    first = datetime.combine(days[0], datetime.min.time())
    lower = (first - epoch).total_seconds() / 3600 + np.arange(len(days) * 8) * 3
    return lower + 1.5, np.column_stack((lower, lower + 3))


def _iso_time(hours: float) -> str:
    return (datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(hours=float(hours))).isoformat().replace("+00:00", "Z")


def _new_output(output: Path, days: Sequence[date], source_sha: str):
    """Create an exclusive NetCDF writer. Caller holds netcdf_read_lock."""
    dataset = netCDF4.Dataset(output, "w", format="NETCDF4", clobber=False)
    try:
        dataset.createDimension("time", len(days) * 8)
        dataset.createDimension("lat", 240)
        dataset.createDimension("lon", 480)
        dataset.createDimension("bounds", 2)
        dataset.setncatts({"Conventions": "CF-1.8", "planet": "Earth", "schema": SCHEMA,
                          "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
                          "source": "NASA MERRA-2 tavg1_2d_slv_Nx + tavg1_2d_rad_Nx",
                          "source_sha256": source_sha, "frequency_hours": 3,
                          "step_unit": "hour", "step": 3, "time_zone": "UTC",
                          "spatial_method": "spherical_area_weighted_overlap",
                          "temporal_method": "mean_of_three_complete_hourly_samples",
                          "missing_support": "strict propagation; no support renormalization",
                          "train_end": "2020-12-31", "validation_end": "2021-06-30"})
        times, time_bounds = _times(days)
        time_var = dataset.createVariable("time", "f8", ("time",), fill_value=False)
        time_var.setncatts({"units": TIME_UNITS, "calendar": CALENDAR, "standard_name": "time",
                           "axis": "T", "bounds": "time_bounds", "time_zone": "UTC"})
        time_var[:] = times
        bound_var = dataset.createVariable("time_bounds", "f8", ("time", "bounds"), fill_value=False)
        bound_var.setncatts({"units": TIME_UNITS, "calendar": CALENDAR})
        bound_var[:] = time_bounds
        lat, lon = target_grid()
        for name, centres, unit, standard in (("lat", lat, "degrees_north", "latitude"),
                                               ("lon", lon, "degrees_east", "longitude")):
            axis = dataset.createVariable(name, "f8", (name,), fill_value=False)
            axis.setncatts({"units": unit, "standard_name": standard,
                           "axis": "Y" if name == "lat" else "X", "bounds": f"{name}_bounds"})
            axis[:] = centres
            bounds = dataset.createVariable(f"{name}_bounds", "f8", (name, "bounds"), fill_value=False)
            bounds.units = unit
            bounds[:] = np.column_stack((centres - .375, centres + .375))
        split = dataset.createVariable("split", "i1", ("time",), fill_value=False)
        split.setncatts({"flag_values": np.array([0, 1, 2], dtype="int8"),
                        "flag_meanings": "train validation test"})
        split[:] = np.repeat([_split_code(day) for day in days], 8)
        for name, unit in zip(CHANNELS, UNITS):
            field = dataset.createVariable(name, "f4", ("time", "lat", "lon"),
                                           zlib=True, complevel=4, shuffle=True,
                                           chunksizes=(1, 240, 480), fill_value=np.float32(np.nan))
            field.setncatts({"units": unit, "long_name": RAW_VARIABLES[name][2],
                            "cell_methods": "time: mean area: mean",
                            "ancillary_variables": f"{name}_valid_mask"})
            mask = dataset.createVariable(f"{name}_valid_mask", "u1", ("time", "lat", "lon"),
                                          zlib=True, complevel=4, chunksizes=(1, 240, 480), fill_value=False)
            mask.setncatts({"long_name": f"Valid complete temporal and spatial support for {name}",
                           "flag_values": np.array([0, 1], dtype="uint8"),
                           "flag_meanings": "missing valid"})
        return dataset
    except Exception:
        dataset.close()
        raise


def verify_existing_release(output_dir: Path, *, days: Sequence[date] | None = None,
                            source_sha256: str | None = None) -> dict:
    """Verify hashes and basic storage contract before accepting a repeat build."""
    output_dir = Path(output_dir)
    path, manifest_path = output_dir / DATA_FILE, output_dir / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("existing output is partial; retained without overwrite; use a fresh directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("dataset_id") != DATASET_ID
            or manifest.get("dataset_version") != DATASET_VERSION or manifest.get("data_file") != DATA_FILE):
        raise ValueError("existing output identity does not match this builder")
    if path.stat().st_size != manifest.get("data_bytes") or sha256_file(path) != manifest.get("data_sha256"):
        raise ValueError("existing NetCDF SHA-256 or size mismatch")
    if manifest.get("source_inventory_version") != SOURCE_INVENTORY_VERSION:
        raise ValueError("existing output uses another source inventory version; use a fresh directory")
    if manifest.get("dataset_fingerprint") != dataset_fingerprint(manifest):
        raise ValueError("existing manifest fingerprint mismatch")
    if source_sha256 is not None and manifest.get("source_sha256") != source_sha256:
        raise ValueError("existing output source inventory does not match requested sources")
    if days is not None and (manifest.get("source", {}).get("date_start") != days[0].isoformat()
                            or manifest.get("source", {}).get("date_end") != days[-1].isoformat()):
        raise ValueError("existing output period does not match requested period")
    with netcdf_read_lock(), netCDF4.Dataset(path, "r") as stored:
        if getattr(stored, "dataset_id", None) != DATASET_ID or getattr(stored, "source_sha256", None) != manifest["source_sha256"]:
            raise ValueError("existing NetCDF identity does not match its manifest")
        lat, lon = target_grid()
        if not np.array_equal(stored.variables["lat"][:], lat) or not np.array_equal(stored.variables["lon"][:], lon):
            raise ValueError("existing NetCDF target grid mismatch")
        time = np.asarray(stored.variables["time"][:])
        if len(time) != manifest["dimensions"]["time"] or not np.all(np.diff(time) == 3):
            raise ValueError("existing NetCDF time axis has duplicates or gaps")
        if days is not None and not np.array_equal(time, _times(days)[0]):
            raise ValueError("existing NetCDF time axis differs from requested period")
        for name, unit in zip(CHANNELS, UNITS):
            field = stored.variables[name]
            mask = stored.variables[f"{name}_valid_mask"]
            if field.shape != (len(time), 240, 480) or mask.shape != field.shape or field.units != unit:
                raise ValueError(f"existing NetCDF variable contract mismatch: {name}")
    return manifest


def build_dataset(source_root: Path | str, output_dir: Path | str, *,
                  start_date: str = "2020-01-01", end_date: str = "2021-12-31",
                  smoke_days: int | None = None, verify_existing: bool = True) -> Path:
    """Build or verify/reuse matching external output; return the NetCDF path."""
    source_root = Path(source_root).resolve()
    output_dir = _safe_output_dir(Path(output_dir), source_root)
    days = _period(start_date, end_date, smoke_days)
    index = discover_source_files(source_root, days)
    inventory, signatures, source_digests = _source_inventory(index, source_root)
    source_sha = source_inventory_sha256(inventory, SOURCE_INVENTORY_VERSION)
    output, manifest_path = output_dir / DATA_FILE, output_dir / "manifest.json"
    if output.exists() or manifest_path.exists():
        if not verify_existing:
            raise ValueError("output already exists; refusing to overwrite")
        verify_existing_release(output_dir, days=days, source_sha256=source_sha)
        _assert_source_digests(index, source_root, source_digests)
        print(f"Verified existing release: {output}", flush=True)
        return output
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: {"source_values": 0, "source_missing": 0, "temporal_values": 0,
                     "temporal_missing": 0, "target_values": 0, "target_missing": 0,
                     "min": None, "max": None, "source_units": None,
                     "max_conservation_error": 0., "max_float32_error": 0.,
                     "complete_support_frames": 0} for name in CHANNELS}
    reference_lat = reference_lon = None
    # The shared reentrant lock covers every NetCDF C-library operation here,
    # including nested source readers while the streaming writer remains open.
    with netcdf_read_lock():
        stored = _new_output(output, days, source_sha)
        try:
            for day_index, day in enumerate(days):
                for name in CHANNELS:
                    path = index[day][RAW_VARIABLES[name][0]]
                    if _signature(path) != signatures[path]:
                        raise ValueError(f"source archive changed before reading: {path.name}")
                    hourly, valid, lat, lon, raw_unit = _read_hourly(path, name, day, reference_lat, reference_lon)
                    reference_lat, reference_lon = lat, lon
                    temporal = temporal_mean_three_hours(hourly, valid)
                    regridded = conservative_regrid(temporal, lat, lon)
                    emitted = regridded.astype("float32")
                    emitted_valid = np.isfinite(emitted)
                    if np.any(np.isfinite(regridded) & ~emitted_valid):
                        raise ValueError(f"{day}/{name}: float32 storage overflow")
                    selection = slice(day_index * 8, (day_index + 1) * 8)
                    stored.variables[name][selection] = emitted
                    stored.variables[f"{name}_valid_mask"][selection] = emitted_valid.astype("uint8")
                    stats = counts[name]
                    stats["source_values"] += int(hourly.size)
                    stats["source_missing"] += int((~valid).sum())
                    stats["temporal_values"] += int(temporal.size)
                    stats["temporal_missing"] += int((~np.isfinite(temporal)).sum())
                    stats["target_values"] += int(emitted.size)
                    stats["target_missing"] += int((~emitted_valid).sum())
                    stats["source_units"] = raw_unit
                    if emitted_valid.any():
                        minimum, maximum = float(emitted[emitted_valid].min()), float(emitted[emitted_valid].max())
                        stats["min"] = minimum if stats["min"] is None else min(stats["min"], minimum)
                        stats["max"] = maximum if stats["max"] is None else max(stats["max"], maximum)
                        stats["max_float32_error"] = max(stats["max_float32_error"], float(np.max(np.abs(
                            regridded[emitted_valid] - emitted[emitted_valid].astype("float64")))))
                    # Exact area-mean conservation is applicable only to complete
                    # support frames; partial support is reported, not imputed.
                    edges = np.concatenate(([-90.], (lat[:-1] + lat[1:]) / 2, [90.]))
                    source_area = np.diff(np.sin(np.deg2rad(edges)))
                    target_area = np.diff(np.sin(np.deg2rad(np.arange(241) * .75 - 90)))
                    for frame, mapped in zip(temporal, regridded):
                        if np.isfinite(frame).all():
                            native_mean = float(np.sum(frame.mean(axis=1) * source_area) / source_area.sum())
                            target_mean = float(np.sum(mapped.mean(axis=1) * target_area) / target_area.sum())
                            error = abs(native_mean - target_mean)
                            if error > max(1e-10, abs(native_mean) * 1e-10):
                                raise ValueError(f"{day}/{name}: area-mean conservation failed")
                            stats["max_conservation_error"] = max(stats["max_conservation_error"], error)
                            stats["complete_support_frames"] += 1
                stored.sync()
                print(f"Processed {day.isoformat()} ({day_index + 1}/{len(days)} UTC days)", flush=True)
            _assert_source_digests(index, source_root, source_digests)
        finally:
            stored.close()
    times, bounds = _times(days)
    variables = {}
    for name, unit in zip(CHANNELS, UNITS):
        stats = counts[name]
        variables[name] = {"units": unit, "dtype": "float32", "valid_mask": f"{name}_valid_mask",
                           **stats, "missing_count": stats["target_missing"],
                           "missing_rate": stats["target_missing"] / stats["target_values"],
                           "source_missing_rate": stats["source_missing"] / stats["source_values"]}
    splits = {}
    for label, code in (("train", 0), ("validation", 1), ("test", 2)):
        selected = [day for day in days if _split_code(day) == code]
        splits[label] = {"start": selected[0].isoformat() if selected else None,
                         "end": selected[-1].isoformat() if selected else None,
                         "days": len(selected), "steps": len(selected) * 8}
    manifest = {
        "schema": SCHEMA, "planet": "Earth", "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION, "data_file": DATA_FILE,
        "build_mode": "smoke" if smoke_days is not None else "full" if len(days) == 731 else "subset",
        "source": {"product": "MERRA-2 tavg1_2d_slv_Nx + tavg1_2d_rad_Nx",
                   "root_label": "MERRA2/raw", "date_start": days[0].isoformat(),
                   "date_end": days[-1].isoformat(), "daily_file_count": len(days),
                   "products": ["slv", "rad"], "excluded_products": ["chm"]},
        "source_files": inventory, "source_sha256": source_sha,
        "source_mtimes": source_mtimes(signatures, source_root),
        "source_inventory_version": SOURCE_INVENTORY_VERSION,
        "source_fingerprint_basis": SOURCE_FINGERPRINT_BASIS,
        "fingerprint_rule": FINGERPRINT_RULE,
        "dimensions": {"time": len(times), "lat": 240, "lon": 480, "bounds": 2},
        "time_start": _iso_time(times[0]), "time_end": _iso_time(times[-1]),
        "frequency_hours": 3, "step_unit": "hour", "step": 3, "time_zone": "UTC",
        "time_kind": "iso-datetime", "cadence": "3 hour mean", "grid_shape": [240, 480],
        "grid": {"latitude": {"start": -89.625, "end": 89.625, "step": .75, "count": 240},
                 "longitude": {"start": -179.625, "end": 179.625, "step": .75, "count": 480},
                 "latitude_bounds": [-90., 90.], "longitude_bounds": [-180., 180.]},
        "channel_order": list(CHANNELS), "variables": variables, "splits": splits,
        "training_profile": {"target": "TO3", "target_unit": "DU", "window": 56, "horizon": 24,
                             "step_unit": "hour", "step": 3, "grid_shape": [240, 480]},
        "time_rule": {"source": "24 hourly means centred 00:30..23:30 UTC per day",
                      "method": "arithmetic mean of three consecutive hourly samples",
                      "target_centres": "01:30,04:30,07:30,10:30,13:30,16:30,19:30,22:30 UTC",
                      "target_bounds": "[00:00,03:00),[03:00,06:00),...,[21:00,24:00) UTC",
                      "interval_start": _iso_time(bounds[0, 0]), "interval_end_exclusive": _iso_time(bounds[-1, 1])},
        "processing": {"temporal_method": "mean_of_three_complete_hourly_samples",
                       "spatial_method": "spherical_area_weighted_overlap",
                       "latitude_weights": "sin(north)-sin(south) overlap of midpoint source cells clipped at poles",
                       "longitude_weights": "angular overlap with periodic shifts -360,0,+360 degrees",
                       "source_grid_shape": [len(reference_lat), len(reference_lon)],
                       "source_latitude_step": float(reference_lat[1] - reference_lat[0]),
                       "source_longitude_step": float(reference_lon[1] - reference_lon[0]),
                       "target_grid": "global 0.75 x 0.75 degree cell means",
                       "missing_policy": "any missing hourly member invalidates its source three-hour cell; any missing nonzero-overlap source cell invalidates target; no renormalization or zero fill",
                       "arithmetic": "float64 temporal mean and conservative regrid; float32 storage",
                       "ozone_unit_rule": "source Dobsons or DU mapped to DU without numerical scaling"},
        "limitations": ["Offline data product only; no registry, training, prediction or overview integration is enabled by this builder",
                        "Conservative grid-cell remapping cannot recover finer-scale variability absent from the source"],
        "fingerprint_basis": FINGERPRINT_BASIS,
        "data_bytes": output.stat().st_size, "data_sha256": sha256_file(output),
    }
    manifest["dataset_fingerprint"] = dataset_fingerprint(manifest)
    _assert_source_digests(index, source_root, source_digests)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def default_output_dir() -> Path:
    configured = os.environ.get("ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR")
    return Path(configured) if configured else REPOSITORY_ROOT.parent / "data" / "earth" / "merra2_3hourly_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"E:\AAOzone\data\MERRA2\raw"))
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--smoke-days", type=int, choices=range(7, 31), metavar="7..30")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Explicitly select the default verify/reuse behavior; existing files are never overwritten")
    args = parser.parse_args()
    try:
        output = build_dataset(args.source_root, args.output_dir, start_date=args.start_date,
                               end_date=args.end_date, smoke_days=args.smoke_days)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Build refused/failed: {exc}\nExisting or partial outputs have been retained.\n")
    print(f"Release ready: {output} ({output.stat().st_size / 1024**2:.2f} MiB)", flush=True)


if __name__ == "__main__":
    main()
