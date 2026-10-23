"""Verify a three-hourly Earth release, optionally against independent raw stencils.

All fields are scanned in daily chunks. Raw comparisons integrate source cell
overlaps directly, without calling the builder's temporal/regridding functions.
No input file is modified. The optional JSON report must be a new file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_identity import build_dataset_fingerprint
from services.netcdf_read_lock import netcdf_read_lock

CHANNELS = ("TO3", "U10M", "V10M", "T2M", "SWGDN")
UNITS = dict(TO3="DU", U10M="m s-1", V10M="m s-1", T2M="K", SWGDN="W m-2")
DATASET_ID = "earth_merra2_3hourly_v1"
DATA_FILENAME = "earth_merra2_3hourly.nc"

# This verifier reimplements the identity rules on purpose. It must not import
# the builder's or the service's digest helpers, otherwise a shared bug would be
# verified by itself.
FINGERPRINT_RULE_LEGACY = 1
FINGERPRINT_RULE_CONTENT = 2
SOURCE_INVENTORY_FULL = 2
SOURCE_INVENTORY_IDENTITY = 3
SOURCE_IDENTITY_FIELDS = ("date", "product", "path", "bytes", "sha256")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fingerprint_rule(manifest) -> int:
    """Missing means the legacy rule; anything unsupported fails closed."""
    value = manifest.get("fingerprint_rule", FINGERPRINT_RULE_LEGACY)
    _require(not isinstance(value, bool) and isinstance(value, int)
             and value in (FINGERPRINT_RULE_LEGACY, FINGERPRINT_RULE_CONTENT),
             f"unsupported fingerprint_rule: {value!r}")
    return value


def _source_inventory_version(manifest) -> int:
    """Missing means the full-inventory basis; anything unsupported fails closed."""
    value = manifest.get("source_inventory_version", SOURCE_INVENTORY_FULL)
    _require(not isinstance(value, bool) and isinstance(value, int)
             and value in (SOURCE_INVENTORY_FULL, SOURCE_INVENTORY_IDENTITY),
             f"unsupported source_inventory_version: {value!r}")
    return value


def _source_identity(entry: dict) -> dict:
    _require(all(field in entry for field in SOURCE_IDENTITY_FIELDS),
             "source inventory entry lacks content identity fields")
    return {field: entry[field] for field in SOURCE_IDENTITY_FIELDS}


def _source_inventory_sha256(files, version: int) -> str:
    payload = ([_source_identity(entry) for entry in files]
               if version == SOURCE_INVENTORY_IDENTITY else files)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _manifest_content_sha256(manifest) -> str:
    """Rule 1 hashes the whole manifest, rule 2 only its content identity."""
    if _fingerprint_rule(manifest) == FINGERPRINT_RULE_LEGACY:
        content = {key: value for key, value in manifest.items()
                   if key != "dataset_fingerprint"}
    else:
        content = {key: value for key, value in manifest.items()
                   if key not in ("dataset_fingerprint", "source_mtimes", "fingerprint_rule")}
        _require(isinstance(content.get("source_files"), list),
                 "manifest requires a source_files array")
        content["source_files"] = [_source_identity(entry) for entry in content["source_files"]]
    return hashlib.sha256(_canonical(content)).hexdigest()


def _utc_labels(time_var) -> list[str]:
    time_var.set_auto_mask(False)
    calendar = getattr(time_var, "calendar", "standard")
    _require(calendar in ("standard", "gregorian", "proleptic_gregorian"),
             "time must use a Gregorian calendar")
    decoded = netCDF4.num2date(time_var[:], time_var.units, calendar=calendar,
                              only_use_cftime_datetimes=False,
                              only_use_python_datetimes=True)
    return [value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in decoded]


def _source_edges(values, latitude=False):
    values = np.asarray(values, dtype="float64")
    edges = np.r_[values[0] - (values[1] - values[0]) / 2,
                  (values[:-1] + values[1:]) / 2,
                  values[-1] + (values[-1] - values[-2]) / 2]
    if latitude:
        edges[0], edges[-1] = -90.0, 90.0
    return edges


def _independent_cell(raw, valid, lat_edges, lon_edges, south, north, west, east):
    """Integrate one target cell using explicit periodic source cell overlaps."""
    lat_left = np.maximum(lat_edges[:-1], south)
    lat_right = np.minimum(lat_edges[1:], north)
    lat_weights = np.where(lat_right > lat_left,
                           np.sin(np.deg2rad(lat_right)) - np.sin(np.deg2rad(lat_left)), 0)
    lon_weights = np.zeros(len(lon_edges) - 1)
    for shift in (-360.0, 0.0, 360.0):
        left = np.maximum(lon_edges[:-1] + shift, west)
        right = np.minimum(lon_edges[1:] + shift, east)
        lon_weights += np.maximum(0, right - left)
    rows, cols = np.flatnonzero(lat_weights > 0), np.flatnonzero(lon_weights > 0)
    _require(len(rows) > 0 and len(cols) > 0, "raw stencil has no coverage")
    patch = raw[:, rows][:, :, cols]
    patch_valid = valid[:, rows][:, :, cols]
    if not patch_valid.all():
        return float("nan")
    area = lat_weights[rows, None] * lon_weights[None, cols]
    expected_area = ((math.sin(math.radians(north)) - math.sin(math.radians(south)))
                     * (east - west))
    _require(np.isclose(area.sum(), expected_area, rtol=1e-10, atol=1e-12),
             "independent stencil does not cover the entire target cell")
    return float(np.sum(patch.mean(axis=0, dtype="float64") * area) / area.sum())


def _compare_raw(dataset, manifest, raw_root: Path) -> dict:
    files = manifest["source_files"]
    days = sorted({entry["date"] for entry in files})
    selected = sorted({days[0], days[len(days) // 2], days[-1]})
    by_key = {(entry["date"], entry["product"]): entry for entry in files}
    source_times = _utc_labels(dataset["time"])
    samples = 0
    masked = 0
    max_error = {name: 0.0 for name in CHANNELS}
    hashes_checked = 0
    targets = ((0, 0), (0, 479), (239, 0), (239, 479),
               (120, 240), (60, 120), (180, 360), (120, 0), (120, 479))
    for day in selected:
        day_index = days.index(day)
        for product, names in (("slv", CHANNELS[:4]), ("rad", ("SWGDN",))):
            entry = by_key[day, product]
            relative = Path(entry["path"])
            _require(not relative.is_absolute() and ".." not in relative.parts,
                     "invalid source relative path")
            path = raw_root / relative
            _require(path.is_file() and path.stat().st_size == entry["bytes"],
                     f"source file missing or changed: {relative}")
            _require(_sha256(path) == entry["sha256"], f"source SHA mismatch: {relative}")
            hashes_checked += 1
            with netcdf_read_lock(), netCDF4.Dataset(path, "r") as source:
                lat_edges = _source_edges(source["lat"][:], latitude=True)
                lon_edges = _source_edges(source["lon"][:])
                labels = _utc_labels(source["time"])
                expected = [f"{day}T{hour:02d}:30:00Z" for hour in range(24)]
                _require(labels == expected, "raw timestamps are not 00:30..23:30 UTC")
                for name in names:
                    variable = source[name]
                    # Both tools decode CF packing/missing/valid-range metadata;
                    # the averaging and direct overlap integration stay independent.
                    variable.set_auto_maskandscale(True)
                    accepted_units = ("Dobsons", "DU") if name == "TO3" else (UNITS[name],)
                    _require(getattr(variable, "units", "") in accepted_units,
                             f"raw unit mismatch: {name}")
                    for bin_index in (0, 3, 7):
                        hour = 3 * bin_index
                        decoded = variable[hour:hour + 3]
                        raw = np.asarray(np.ma.getdata(decoded), dtype="float64")
                        valid = (~np.ma.getmaskarray(decoded) & np.isfinite(raw)
                                 & (np.abs(raw) < 1e14))
                        time_index = day_index * 8 + bin_index
                        _require(source_times[time_index] == f"{day}T{hour + 1:02d}:30:00Z",
                                 "three-hour group is labelled with the wrong centre")
                        field = np.asarray(dataset[name][time_index], dtype="float64")
                        for row, col in targets:
                            south, north = dataset["lat_bounds"][row]
                            west, east = dataset["lon_bounds"][col]
                            expected_value = _independent_cell(
                                raw, valid, lat_edges, lon_edges, south, north, west, east)
                            actual = float(field[row, col])
                            if np.isnan(expected_value):
                                _require(np.isnan(actual), f"missing raw contributor was filled: {name}")
                                masked += 1
                            else:
                                _require(np.isfinite(actual) and
                                         np.isclose(actual, expected_value, rtol=2e-7, atol=2e-5),
                                         f"raw mean/regrid mismatch: {day}/{name}/{bin_index}/{row}/{col}")
                                max_error[name] = max(max_error[name], abs(actual - expected_value))
                            samples += 1
    return {"method": "independent_three_hour_mean_and_spherical_overlap_stencils",
            "dates": selected, "bins": [0, 3, 7], "cells_per_bin": len(targets),
            "source_sha256_files_checked": hashes_checked, "samples_checked": samples,
            "masked_samples": masked, "maximum_absolute_errors": max_error}


def verify_package(package_dir, *, raw_root=None) -> dict:
    root = Path(package_dir)
    manifest_path, data_path = root / "manifest.json", root / DATA_FILENAME
    _require(manifest_path.is_file() and data_path.is_file(), "package is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("dataset_id") == DATASET_ID and manifest.get("dataset_version") == "v1",
             "wrong dataset identity")
    _require(manifest.get("data_file") == DATA_FILENAME, "wrong data filename")
    data_sha = _sha256(data_path)
    _require(data_sha == manifest.get("data_sha256"), "NetCDF SHA-256 mismatch")
    _require(data_path.stat().st_size == manifest.get("data_bytes"), "NetCDF size mismatch")
    rule = _fingerprint_rule(manifest)
    inventory_version = _source_inventory_version(manifest)
    content_sha = _manifest_content_sha256(manifest)
    fingerprint = build_dataset_fingerprint(DATASET_ID, "v1", content_sha, data_sha)
    _require(fingerprint == manifest.get("dataset_fingerprint"), "dataset fingerprint mismatch")
    files = manifest.get("source_files", [])
    _require(bool(files) and _source_inventory_sha256(files, inventory_version) == manifest.get("source_sha256"),
             "source inventory digest mismatch")
    source_mtimes = manifest.get("source_mtimes")
    if source_mtimes is not None:
        _require(isinstance(source_mtimes, dict)
                 and set(source_mtimes) == {entry.get("path") for entry in files}
                 and all(type(value) is int and value >= 0 for value in source_mtimes.values()),
                 "source mtimes provenance block mismatch")
    days = sorted({entry["date"] for entry in files})
    _require(len(files) == len(days) * 2 and
             len({(entry["date"], entry["product"]) for entry in files}) == len(files) and
             all({entry["product"] for entry in files if entry["date"] == day} == {"slv", "rad"}
                 for day in days), "source inventory must contain exactly one SLV/RAD pair per day")
    start = date.fromisoformat(days[0])
    _require(days == [(start + timedelta(days=index)).isoformat() for index in range(len(days))],
             "source inventory has missing dates")
    statistics = {}
    with netcdf_read_lock(), netCDF4.Dataset(data_path, "r") as dataset:
        dataset.set_auto_mask(False)
        _require(getattr(dataset, "dataset_id", "") == DATASET_ID, "NetCDF identity mismatch")
        _require(getattr(dataset, "schema", "") == "astraatmos_earth_3hourly_v1"
                 and getattr(dataset, "dataset_version", "") == "v1"
                 and getattr(dataset, "source_sha256", "") == manifest["source_sha256"],
                 "NetCDF schema, version or provenance mismatch")
        shape = [len(dataset.dimensions[name]) for name in ("time", "lat", "lon")]
        _require(shape == [len(days) * 8, 240, 480], "wrong release dimensions")
        _require(manifest.get("dimensions") == dict(time=shape[0], lat=240, lon=480, bounds=2),
                 "manifest dimension mismatch")
        _require(manifest.get("channel_order") == list(CHANNELS)
                 and manifest.get("frequency_hours") == 3
                 and manifest.get("step") == 3 and manifest.get("step_unit") == "hour"
                 and manifest.get("time_zone") == "UTC" and manifest.get("grid_shape") == [240, 480],
                 "manifest time, grid or channel contract mismatch")
        labels = _utc_labels(dataset["time"])
        expected_labels = [f"{day}T{hour:02d}:30:00Z" for day in days for hour in range(1, 24, 3)]
        _require(labels == expected_labels, "time has duplicates, gaps, or incorrectly centred samples")
        _require(manifest.get("time_start") == labels[0] and manifest.get("time_end") == labels[-1],
                 "manifest time range mismatch")
        minutes = np.asarray([datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ") for value in labels],
                             dtype="datetime64[m]")
        _require(np.all(np.diff(minutes) == np.timedelta64(3, "h")), "time interval must be three hours")
        np.testing.assert_array_equal(dataset["lat"][:], -89.625 + np.arange(240) * 0.75)
        np.testing.assert_array_equal(dataset["lon"][:], -179.625 + np.arange(480) * 0.75)
        for axis, size, edge in (("lat", 240, -90.0), ("lon", 480, -180.0)):
            expected = np.column_stack((edge + np.arange(size) * .75,
                                        edge + (np.arange(size) + 1) * .75))
            np.testing.assert_array_equal(dataset[f"{axis}_bounds"][:], expected)
        bounds_var = dataset["time_bounds"]
        bounds_var.set_auto_mask(False)
        units = getattr(bounds_var, "units", dataset["time"].units)
        decoded_bounds = netCDF4.num2date(bounds_var[:], units,
                                         calendar=getattr(dataset["time"], "calendar", "standard"),
                                         only_use_cftime_datetimes=False,
                                         only_use_python_datetimes=True)
        for index, centre in enumerate(minutes):
            left, right = [np.datetime64(value, "m") for value in decoded_bounds[index]]
            _require(centre - left == np.timedelta64(90, "m") and
                     right - centre == np.timedelta64(90, "m"), "wrong three-hour time bounds")
        total = shape[0] * shape[1] * shape[2]
        for name in CHANNELS:
            variable, mask = dataset[name], dataset[f"{name}_valid_mask"]
            _require(variable.dimensions == ("time", "lat", "lon") and
                     variable.dtype == np.dtype("float32") and
                     getattr(variable, "units", "") == UNITS[name], f"wrong variable contract: {name}")
            _require(mask.dimensions == variable.dimensions, f"wrong mask dimensions: {name}")
            variable.set_auto_maskandscale(False)
            mask.set_auto_maskandscale(False)
            missing, minimum, maximum = 0, math.inf, -math.inf
            for begin in range(0, shape[0], 8):
                values = np.asarray(variable[begin:begin + 8])
                valid = np.asarray(mask[begin:begin + 8])
                _require(np.isin(valid, [0, 1]).all(), f"nonbinary validity mask: {name}")
                _require(np.array_equal(valid.astype(bool), np.isfinite(values)), f"mask/value mismatch: {name}")
                _require(np.isnan(values[valid == 0]).all(), f"missing values must stay NaN: {name}")
                good = values[valid == 1]
                _require((np.abs(good) < 1e14).all(), f"unmasked fill value: {name}")
                missing += int(np.count_nonzero(valid == 0))
                if len(good):
                    minimum, maximum = min(minimum, float(good.min())), max(maximum, float(good.max()))
            statistics[name] = {"units": UNITS[name], "missing_count": missing,
                                "missing_rate": missing / total,
                                "min": minimum if np.isfinite(minimum) else None,
                                "max": maximum if np.isfinite(maximum) else None}
            declared = manifest.get("variables", {}).get(name, {})
            _require(declared.get("units") == UNITS[name] and declared.get("dtype") == "float32"
                     and declared.get("valid_mask") == f"{name}_valid_mask"
                     and declared.get("target_missing") == missing and declared.get("target_values") == total
                     and declared.get("missing_rate") == statistics[name]["missing_rate"]
                     and declared.get("min") == statistics[name]["min"]
                     and declared.get("max") == statistics[name]["max"],
                     f"manifest variable statistics mismatch: {name}")
        raw_comparison = _compare_raw(dataset, manifest, Path(raw_root)) if raw_root else None
    return {"validation": "passed", "dataset_id": DATASET_ID, "shape": shape,
            "time_start": labels[0], "time_end": labels[-1], "frequency_hours": 3,
            "grid": {"latitude": [-89.625, 89.625], "longitude": [-179.625, 179.625],
                     "step_degrees": .75, "coverage": "global", "wrap_longitude": True},
            "variables": statistics, "data_sha256": data_sha,
            "manifest_file_sha256": _sha256(manifest_path),
            "manifest_content_sha256": content_sha, "dataset_fingerprint": fingerprint,
            "fingerprint_rule": rule, "source_inventory_version": inventory_version,
            "raw_comparison": raw_comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", type=Path, help="independently compare first/middle/last source days")
    parser.add_argument("--report", type=Path, help="write a new JSON report; existing files are refused")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("report already exists; choose a new path")
    report = verify_package(args.package_dir, raw_root=args.raw_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
