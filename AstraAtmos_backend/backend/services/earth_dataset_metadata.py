"""Verify a fixed Earth compact release and describe it, without training support.

The package directory comes from server configuration only. Client input never
selects a path, and the manifest may only reference the fixed data file name.

Verification order (each step fails with a stable reason code):

1. both files exist                     -> ``package_missing``
2. manifest raw-byte SHA-256            -> ``manifest_fingerprint_mismatch``
3. manifest is a JSON object            -> ``invalid_manifest``
4. manifest ``data_file`` fixed name    -> ``invalid_manifest``
5. NetCDF raw-byte SHA-256              -> ``data_fingerprint_mismatch``
6. manifest self reported hash/size     -> ``manifest_metadata_mismatch``
7. ``load_earth_dataset()`` data contract -> ``invalid_dataset``
8. manifest vs NetCDF metadata          -> ``manifest_metadata_mismatch``
9. file signatures changed mid check    -> ``package_changed_during_verification``
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np
import xarray as xr

from services.dataset_identity import build_dataset_fingerprint
from services.earth_dataset import (
    CHANNELS, SCHEMA, SPLITS, UNITS, load_earth_dataset,
    THREE_HOURLY_SCHEMA, THREE_HOURLY_DATASET_ID, THREE_HOURLY_5DEG_DATASET_ID,
    THREE_HOURLY_DATASET_IDS, THREE_HOURLY_DATASET_VERSION,
    THREE_HOURLY_TRAINING_PROFILE, THREE_HOURLY_5DEG_TRAINING_PROFILE,
    validate_earth_3hourly_dataset,
)
from services.netcdf_read_lock import netcdf_read_lock

logger = logging.getLogger("astraatmos.datasets.earth")

VERIFIER_VERSION = "earth_3hourly_verifier_v2"

MANIFEST_FILE_NAME = "manifest.json"
DATA_FILE_NAME = "earth_merra2_daily.nc"
THREE_HOURLY_DATA_FILE_NAME = "earth_merra2_3hourly.nc"

GREGORIAN_CALENDARS = frozenset({"standard", "gregorian", "proleptic_gregorian"})

VARIABLE_LABELS = {
    "TO3": "Total column ozone",
    "U10M": "10 m eastward wind",
    "V10M": "10 m northward wind",
    "T2M": "2 m air temperature",
    "SWGDN": "Surface incoming shortwave flux",
}
TARGET_CHANNEL = "TO3"

REASON_PACKAGE_MISSING = "package_missing"
REASON_PACKAGE_UNREADABLE = "package_unreadable"
REASON_MANIFEST_FINGERPRINT_MISMATCH = "manifest_fingerprint_mismatch"
REASON_DATA_FINGERPRINT_MISMATCH = "data_fingerprint_mismatch"
REASON_INVALID_MANIFEST = "invalid_manifest"
REASON_MANIFEST_METADATA_MISMATCH = "manifest_metadata_mismatch"
REASON_INVALID_DATASET = "invalid_dataset"
REASON_PACKAGE_CHANGED = "package_changed_during_verification"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# Three-hour identity rules. Rule 1 is the legacy whole-manifest digest that the
# already registered publication is frozen to; rule 2 binds identity to source
# content only. A manifest declares its rule with an optional integer field.
THREE_HOURLY_FINGERPRINT_RULE_LEGACY = 1
THREE_HOURLY_FINGERPRINT_RULE_CONTENT = 2
# Source inventory bases. Version 2 (and any manifest without the field) hashes
# every recorded field, including name and mtime_ns; version 3 hashes the content
# identity projection only.
THREE_HOURLY_SOURCE_INVENTORY_FULL = 2
THREE_HOURLY_SOURCE_INVENTORY_IDENTITY = 3
# Content identity of one inventory entry. name is a redundant suffix of path and
# mtime_ns is file-system metadata, so neither may enter an identity digest.
THREE_HOURLY_SOURCE_IDENTITY_FIELDS = ("date", "product", "path", "bytes", "sha256")


class EarthPackageError(RuntimeError):
    """A package verification failure carrying a stable public reason code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_signature(package_dir: Any, *, data_file_name: str = DATA_FILE_NAME) -> tuple:
    """Return a stable source identity for the package files.

    ``st_ctime`` is creation time on Windows and therefore cannot be used as a
    modification signal.  The shared helper obtains the NTFS ChangeTime and
    also includes the volume/file identity.  Keep the historical four-column
    tuple shape for existing callers and cache files, but use the fourth column
    for that identity token rather than Windows creation time.
    """
    from services.earth_preparation_cache import file_identity

    root = Path(package_dir).expanduser()
    parts = []
    for name in (MANIFEST_FILE_NAME, data_file_name):
        path = root / name
        try:
            identity = file_identity(path)
        except OSError:
            parts.append((str(path), None, None, None))
        else:
            token = f"{identity['device']}:{identity['inode']}:{identity['change_ns']}"
            parts.append((str(path.resolve()), identity["bytes"], identity["mtime_ns"], token))
    return tuple(parts)


def read_verified_manifest(
    package_dir: Any,
    expected_manifest_sha256: str,
    expected_data_sha256: str,
    *,
    data_file_name: str = DATA_FILE_NAME,
) -> tuple[dict, Path]:
    """Validate both artifact hashes and the manifest self description."""
    root = Path(package_dir).expanduser()
    manifest_path = root / MANIFEST_FILE_NAME
    data_path = root / data_file_name
    if not manifest_path.is_file() or not data_path.is_file():
        raise EarthPackageError(REASON_PACKAGE_MISSING, "package files are missing")

    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise EarthPackageError(
            REASON_MANIFEST_FINGERPRINT_MISMATCH, "manifest.json is not the released version"
        )
    try:
        manifest = json.loads(manifest_bytes)
    except ValueError as exc:
        raise EarthPackageError(REASON_INVALID_MANIFEST, f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise EarthPackageError(REASON_INVALID_MANIFEST, "manifest must be a JSON object")
    if manifest.get("data_file") != data_path.name:
        raise EarthPackageError(
            REASON_INVALID_MANIFEST, "manifest data_file must be the fixed package data file"
        )
    if file_sha256(data_path) != expected_data_sha256:
        raise EarthPackageError(
            REASON_DATA_FINGERPRINT_MISMATCH, "data file is not the released version"
        )
    if manifest.get("data_sha256") != expected_data_sha256:
        raise EarthPackageError(
            REASON_MANIFEST_METADATA_MISMATCH,
            "manifest data_sha256 does not match the released data file",
        )
    if manifest.get("data_bytes") != data_path.stat().st_size:
        raise EarthPackageError(
            REASON_MANIFEST_METADATA_MISMATCH,
            "manifest data_bytes does not match the released data file",
        )
    return manifest, data_path


def _assert_manifest_matches(ds, manifest: Mapping[str, Any]) -> None:
    """Compare manifest claims against the decoded NetCDF content item by item."""
    dates = ds.time.values.astype("datetime64[D]")
    lat = np.asarray(ds.lat.values, dtype="float64")
    lon = np.asarray(ds.lon.values, dtype="float64")

    checks = [
        manifest.get("schema") == SCHEMA,
        manifest.get("planet") == "Earth",
        manifest.get("dimensions") == {name: int(size) for name, size in ds.sizes.items()},
        manifest.get("time_start") == str(dates[0]),
        manifest.get("time_end") == str(dates[-1]),
        manifest.get("channel_order") == list(CHANNELS),
        manifest.get("latitude_range") == ([float(ds.lat_bounds.values[0, 0]), float(ds.lat_bounds.values[-1, 1])] if "lat_bounds" in ds else [float(lat.min()), float(lat.max())]),
        manifest.get("longitude_range") == ([float(ds.lon_bounds.values[0, 0]), float(ds.lon_bounds.values[-1, 1])] if "lon_bounds" in ds else [float(lon.min()), float(lon.max())]),
        manifest.get("source_sha256") == ds.attrs.get("source_sha256"),
        manifest.get("cadence") == "daily mean",
        ds.attrs.get("temporal_resolution") == "1 day",
    ]
    declared_variables = manifest.get("variables")
    if not isinstance(declared_variables, dict):
        declared_variables = {}
    splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    for name, unit in zip(CHANNELS, UNITS):
        variable = declared_variables.get(name)
        if not isinstance(variable, dict):
            variable = {}
        checks.extend([
            variable.get("units") == unit == ds[name].attrs.get("units"),
            variable.get("dtype") == "float32" == str(ds[name].dtype),
            variable.get("min") == float(ds[name].min()),
            variable.get("max") == float(ds[name].max()),
        ])
    for name, code in SPLITS.items():
        selected = dates[ds["split"].values == code]
        if len(selected) == 0:
            checks.append(False)
            continue
        checks.append(splits.get(name) == {
            "start": str(selected[0]),
            "end": str(selected[-1]),
            "days": int(len(selected)),
        })
    if not all(checks):
        raise EarthPackageError(
            REASON_MANIFEST_METADATA_MISMATCH, "manifest metadata does not match the data file"
        )

    if not all(np.isfinite(np.diff(axis)).all() for axis in (lat, lon)):
        raise EarthPackageError(REASON_INVALID_DATASET, "grid coordinates are not finite")
    if not (
        np.allclose(np.diff(lat), np.diff(lat)[0])
        and np.allclose(np.diff(lon), np.diff(lon)[0])
    ):
        raise EarthPackageError(REASON_INVALID_DATASET, "grid coordinates are not uniform")
    if "lat_bounds" in ds:
        expected_bounds = {
            "latitude": [float(ds.lat_bounds.values[0, 0]), float(ds.lat_bounds.values[-1, 1])],
            "longitude": [float(ds.lon_bounds.values[0, 0]), float(ds.lon_bounds.values[-1, 1])],
        }
        if manifest.get("cell_bounds") != expected_bounds:
            raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH, "cell bounds mismatch")
        normalization = manifest.get("normalization", {})
        stats = normalization.get("train_stats", {}) if isinstance(normalization, dict) else {}
        if not isinstance(normalization, dict) or normalization.get("fit_split") != "train":
            raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH, "normalization must use train dates")
        for name in CHANNELS:
            train_values = ds[name].values[ds['split'].values == SPLITS['train']]
            mean = float(train_values.mean(dtype='float64'))
            std = float(train_values.std(dtype='float64'))
            expected = {'mean': mean, 'std': std, 'scale_std': std if std >= 1e-6 else 1.0}
            declared = stats.get(name, {})
            if any(key not in declared or not np.isclose(declared[key], value, rtol=1e-12, atol=1e-12)
                   for key, value in expected.items()):
                raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH, "training normalization statistics mismatch")


def _decode_calendar(ds) -> str:
    encoding = getattr(getattr(ds, "time", None), "encoding", None) or {}
    calendar = encoding.get("calendar")
    if not isinstance(calendar, str) or calendar.strip().lower() not in GREGORIAN_CALENDARS:
        raise EarthPackageError(
            REASON_INVALID_DATASET,
            "time coordinate must use a Gregorian calendar",
        )
    return calendar.strip().lower()


def readonly_array(values, dtype=None) -> np.ndarray:
    """Copy ``values`` into a read-only NumPy array."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class VerifiedEarthRelease:
    """A verified snapshot of the fixed Earth release.

    Daily releases hold fields for the existing overview and model APIs. The
    three-hour snapshot holds coordinates and a server-only file path, with no
    field arrays; its large volume stays on disk. The server never returns this
    object over HTTP or includes the path in descriptor metadata.
    """

    metadata: dict
    signature: tuple
    dates: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    fields: Mapping[str, np.ndarray]
    data_path: Optional[Path] = None
    verification_cached: bool = False


def read_earth_metadata(
    package_dir: Any,
    *,
    expected_manifest_sha256: str,
    expected_data_sha256: str,
    dataset_id: Optional[str] = None,
    dataset_version: Optional[str] = None,
    data_file_name: str = DATA_FILE_NAME,
) -> dict:
    """Return verified Earth metadata, or raise :class:`EarthPackageError`."""
    return read_earth_release(
        package_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_data_sha256=expected_data_sha256,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        data_file_name=data_file_name,
    ).metadata


def read_earth_release(
    package_dir: Any,
    *,
    expected_manifest_sha256: str,
    expected_data_sha256: str,
    dataset_id: Optional[str] = None,
    dataset_version: Optional[str] = None,
    data_file_name: str = DATA_FILE_NAME,
) -> VerifiedEarthRelease:
    """Verify the release and return its metadata plus read-only data arrays.

    Uses the same pinned-hash, manifest semantics and before/after signature
    checks as the metadata-only path; both share this single verification body.
    """
    if not _SHA256_PATTERN.match(expected_manifest_sha256 or "") or not _SHA256_PATTERN.match(
        expected_data_sha256 or ""
    ):
        raise EarthPackageError(
            REASON_INVALID_MANIFEST, "expected release hashes are not valid sha256 digests"
        )

    signature_before = package_signature(package_dir, data_file_name=data_file_name)
    manifest, data_path = read_verified_manifest(
        package_dir,
        expected_manifest_sha256,
        expected_data_sha256,
        data_file_name=data_file_name,
    )
    try:
        ds = load_earth_dataset(data_path)
    except EarthPackageError:
        raise
    except Exception as exc:
        raise EarthPackageError(REASON_INVALID_DATASET, str(exc)) from exc

    with ds:
        try:
            calendar = _decode_calendar(ds)
            _assert_manifest_matches(ds, manifest)
            if dataset_id == 'earth_merra2_daily_v2' and (
                manifest.get('dataset_id') != dataset_id or manifest.get('dataset_version') != dataset_version
                or 'lat_bounds' not in ds or 'lon_bounds' not in ds
            ):
                raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH, 'v2 release identity or bounds mismatch')
            metadata = _extract_metadata(
                ds,
                manifest,
                calendar=calendar,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                manifest_sha256=expected_manifest_sha256,
                data_sha256=expected_data_sha256,
            )
            release = VerifiedEarthRelease(
                metadata=metadata,
                signature=signature_before,
                dates=readonly_array(ds.time.values, "datetime64[D]"),
                latitude=readonly_array(ds.lat.values),
                longitude=readonly_array(ds.lon.values),
                fields=MappingProxyType({
                    name: readonly_array(ds[name].values, "float32") for name in CHANNELS
                }),
            )
        except EarthPackageError:
            raise
        except Exception as exc:
            raise EarthPackageError(REASON_INVALID_DATASET, str(exc)) from exc

    if package_signature(package_dir, data_file_name=data_file_name) != signature_before:
        raise EarthPackageError(
            REASON_PACKAGE_CHANGED, "package files changed during verification"
        )
    return release


def _extract_metadata(
    ds,
    manifest: Mapping[str, Any],
    *,
    calendar: str,
    dataset_id: Optional[str],
    dataset_version: Optional[str],
    manifest_sha256: str,
    data_sha256: str,
) -> dict:
    dates = ds.time.values.astype("datetime64[D]")
    lat = np.asarray(ds.lat.values, dtype="float64")
    lon = np.asarray(ds.lon.values, dtype="float64")

    latitude_step = float(np.diff(lat)[0])
    longitude_step = float(np.diff(lon)[0])
    latitude_bounds = ([float(ds.lat_bounds.values[0, 0]), float(ds.lat_bounds.values[-1, 1])]
                       if 'lat_bounds' in ds else [float(lat.min()), float(lat.max())])
    longitude_bounds = ([float(ds.lon_bounds.values[0, 0]), float(ds.lon_bounds.values[-1, 1])]
                        if 'lon_bounds' in ds else [float(lon.min()), float(lon.max())])
    coverage = "global" if (
        'lat_bounds' in ds and 'lon_bounds' in ds
        and np.allclose(latitude_bounds, [-90., 90.], rtol=0, atol=1e-8)
        and np.allclose(longitude_bounds, [-180., 180.], rtol=0, atol=1e-8)
    ) else "regional"
    splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    limitations = manifest.get("limitations")
    if not isinstance(limitations, list):
        limitations = []

    return {
        "dataset_fingerprint": build_dataset_fingerprint(
            dataset_id, dataset_version, manifest_sha256, data_sha256
        ),
        "manifest_sha256": manifest_sha256,
        "data_sha256": data_sha256,
        # The verified manifest itself, so downstream services can report the
        # published provenance (source digest, processing description) without
        # re-reading the package. It carries no server path: the manifest may
        # only name the fixed data file, which was already checked above.
        "manifest": dict(manifest),
        "schema": SCHEMA,
        "time": {
            "kind": "date",
            "calendar": calendar,
            "start": str(dates[0]),
            "end": str(dates[-1]),
            "count": int(len(dates)),
            "step": int((dates[1] - dates[0]) / np.timedelta64(1, "D")),
            "step_unit": "day",
        },
        "grid": {
            "shape": [int(ds.sizes["lat"]), int(ds.sizes["lon"])],
            "dimension_order": ["lat", "lon"],
            "latitude_range": [float(latitude_bounds[0]), float(latitude_bounds[1])],
            "longitude_range": [float(longitude_bounds[0]), float(longitude_bounds[1])],
            "cell_bounds": {
                "latitude": [float(latitude_bounds[0]), float(latitude_bounds[1])],
                "longitude": [float(longitude_bounds[0]), float(longitude_bounds[1])],
            },
            "latitude_step": latitude_step,
            "longitude_step": longitude_step,
            "latitude_order": "ascending" if latitude_step > 0 else "descending",
            "longitude_order": "ascending" if longitude_step > 0 else "descending",
            "coverage": coverage,
            "wrap_longitude": bool(coverage == "global"),
            "latitude_values": [float(value) for value in lat],
            "longitude_values": [float(value) for value in lon],
        },
        "channel_order": list(CHANNELS),
        "variables": [
            {
                "id": name,
                "label": VARIABLE_LABELS[name],
                "units": unit,
                "role": "target_and_input" if name == TARGET_CHANNEL else "optional_input",
            }
            for name, unit in zip(CHANNELS, UNITS)
        ],
        "splits": {
            name: {
                "start": str(splits[name]["start"]),
                "end": str(splits[name]["end"]),
                "days": int(splits[name]["days"]),
            }
            for name in SPLITS
        },
        "limitations": [str(item) for item in limitations],
    }


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def threehour_fingerprint_rule(manifest: Mapping[str, Any]) -> int:
    """Resolve the manifest's identity rule; unknown values fail closed.

    A missing ``fingerprint_rule`` is the legacy rule, which is what the already
    registered publication relies on. Every other unsupported value (0, negative,
    string, float, bool) is refused instead of silently digesting as the legacy
    rule: a silent fallback would report a fingerprint no builder can reproduce.
    """
    value = manifest.get("fingerprint_rule", THREE_HOURLY_FINGERPRINT_RULE_LEGACY)
    if (isinstance(value, bool) or not isinstance(value, int)
            or value not in (THREE_HOURLY_FINGERPRINT_RULE_LEGACY,
                             THREE_HOURLY_FINGERPRINT_RULE_CONTENT)):
        raise ValueError(f"unsupported three-hour fingerprint_rule: {value!r}")
    return value


def threehour_source_inventory_version(manifest: Mapping[str, Any]) -> int:
    """Resolve the source inventory basis; unknown values fail closed."""
    value = manifest.get("source_inventory_version", THREE_HOURLY_SOURCE_INVENTORY_FULL)
    if (isinstance(value, bool) or not isinstance(value, int)
            or value not in (THREE_HOURLY_SOURCE_INVENTORY_FULL,
                             THREE_HOURLY_SOURCE_INVENTORY_IDENTITY)):
        raise ValueError(f"unsupported three-hour source_inventory_version: {value!r}")
    return value


def threehour_source_identity_projection(files) -> list:
    """Project source inventory entries onto their content identity fields."""
    projection = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("invalid three-hour source inventory entry")
        missing = [field for field in THREE_HOURLY_SOURCE_IDENTITY_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"three-hour source inventory entry lacks {', '.join(missing)}")
        projection.append({field: entry[field] for field in THREE_HOURLY_SOURCE_IDENTITY_FIELDS})
    return projection


def threehour_source_inventory_sha256(files, version: int) -> str:
    """Digest an inventory with the basis its declared version requires.

    Version 3 projects onto content identity, so a pure mtime drift cannot make a
    published release look damaged; earlier versions keep the full entry digest.
    """
    if version not in (THREE_HOURLY_SOURCE_INVENTORY_FULL,
                       THREE_HOURLY_SOURCE_INVENTORY_IDENTITY):
        raise ValueError(f"unsupported three-hour source_inventory_version: {version!r}")
    payload = (threehour_source_identity_projection(files)
               if version == THREE_HOURLY_SOURCE_INVENTORY_IDENTITY else files)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def threehour_manifest_content_sha256(manifest: Mapping[str, Any]) -> str:
    """Digest the manifest content that defines the three-hour dataset identity.

    Rule 1 keeps the historical basis byte for byte: every field except the
    self-referential ``dataset_fingerprint``. Rule 2 additionally excludes the
    ``source_mtimes`` provenance block and the rule marker itself, and projects
    ``source_files`` onto (date, product, path, bytes, sha256), so neither a
    timestamp nor the redundant ``name`` suffix can move a published identity.
    """
    rule = threehour_fingerprint_rule(manifest)
    if rule == THREE_HOURLY_FINGERPRINT_RULE_LEGACY:
        return hashlib.sha256(_canonical_json({
            key: value for key, value in manifest.items() if key != "dataset_fingerprint"
        })).hexdigest()
    payload = {key: value for key, value in manifest.items()
               if key not in ("dataset_fingerprint", "source_mtimes", "fingerprint_rule")}
    files = payload.get("source_files")
    if not isinstance(files, list):
        raise ValueError("three-hour manifest requires a source_files array")
    payload["source_files"] = threehour_source_identity_projection(files)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def threehour_release_identity(manifest: Mapping[str, Any], *,
                               verified_content_sha256: Optional[str] = None) -> dict:
    """Resolve the identity triple behind ``dataset_fingerprint``.

    ``verified_content_sha256`` is the frozen digest of an already verified
    publication (registry descriptor or shared verification proof). Supplying it
    short-circuits a legacy publication: the 1462 entry source inventory is never
    re-projected and never re-hashed, the persisted value is combined with the
    manifest's own id, version and data digest, and the result must reproduce the
    published ``dataset_fingerprint`` exactly. Without it the cold path falls back
    to the rule the manifest declares, which for every legacy publication is rule 1
    over the full manifest.
    """
    rule = threehour_fingerprint_rule(manifest)
    inventory_version = threehour_source_inventory_version(manifest)
    dataset_id = manifest["dataset_id"]
    dataset_version = manifest["dataset_version"]
    data_sha256 = manifest["data_sha256"]
    if verified_content_sha256 is None:
        content_sha256 = threehour_manifest_content_sha256(manifest)
    else:
        if (not isinstance(verified_content_sha256, str)
                or not _SHA256_PATTERN.fullmatch(verified_content_sha256)):
            raise ValueError("persisted manifest_content_sha256 is not a sha256 digest")
        content_sha256 = verified_content_sha256
    fingerprint = build_dataset_fingerprint(dataset_id, dataset_version,
                                            content_sha256, data_sha256)
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise EarthPackageError(REASON_MANIFEST_FINGERPRINT_MISMATCH,
                                "three-hour dataset fingerprint does not match its manifest")
    return {"dataset_id": dataset_id, "dataset_version": dataset_version,
            "manifest_content_sha256": content_sha256, "data_sha256": data_sha256,
            "dataset_fingerprint": fingerprint,
            "fingerprint_rule": rule, "source_inventory_version": inventory_version}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number: {value}")
    return result


def _assert_threehour_manifest_identity(manifest):
    if (not isinstance(manifest, dict) or manifest.get("schema") != THREE_HOURLY_SCHEMA
            or manifest.get("planet") != "Earth"
            or manifest.get("dataset_id") not in THREE_HOURLY_DATASET_IDS
            or manifest.get("dataset_version") != THREE_HOURLY_DATASET_VERSION
            or manifest.get("data_file") != THREE_HOURLY_DATA_FILE_NAME):
        raise EarthPackageError(REASON_INVALID_MANIFEST, "wrong three-hour release identity or filename")
    if (not _SHA256_PATTERN.fullmatch(str(manifest.get("data_sha256", "")))
            or not _SHA256_PATTERN.fullmatch(str(manifest.get("dataset_fingerprint", "")))
            or not _SHA256_PATTERN.fullmatch(str(manifest.get("source_sha256", "")))
            or type(manifest.get("data_bytes")) is not int or manifest["data_bytes"] < 1):
        raise EarthPackageError(REASON_INVALID_MANIFEST, "manifest must include valid release hashes and size")


def _iso_utc(value) -> str:
    return np.datetime_as_string(np.datetime64(value, "s"), unit="s") + "Z"


def _assert_threehour_manifest_matches(ds, manifest, statistics):
    """Verify all descriptor claims and the source inventory without requiring raw files."""
    times = ds.time.values
    days = times.astype("datetime64[D]")
    source = manifest.get("source")
    files = manifest.get("source_files")
    processing = manifest.get("processing")
    rule = manifest.get("time_rule")
    lowres = manifest.get("dataset_id") == THREE_HOURLY_5DEG_DATASET_ID
    grid_shape = (36, 72) if lowres else (240, 480)
    grid_step = 5.0 if lowres else .75
    expected_grid = {
        "latitude": {"start": -90 + grid_step / 2, "end": 90 - grid_step / 2, "step": grid_step, "count": grid_shape[0]},
        "longitude": {"start": -180 + grid_step / 2, "end": 180 - grid_step / 2, "step": grid_step, "count": grid_shape[1]},
        "latitude_bounds": [-90., 90.], "longitude_bounds": [-180., 180.],
    }
    checks = [
        manifest.get("dimensions") == {name: int(size) for name, size in ds.sizes.items()},
        manifest.get("time_start") == _iso_utc(times[0]),
        manifest.get("time_end") == _iso_utc(times[-1]),
        manifest.get("frequency_hours") == 3,
        manifest.get("step_unit") == "hour", manifest.get("step") == 3,
        manifest.get("time_zone") == "UTC", manifest.get("time_kind") == "iso-datetime",
        manifest.get("cadence") == "3 hour mean", manifest.get("grid_shape") == list(grid_shape),
        manifest.get("grid") == expected_grid,
        manifest.get("channel_order") == list(CHANNELS),
        manifest.get("training_profile") == (THREE_HOURLY_5DEG_TRAINING_PROFILE if lowres else THREE_HOURLY_TRAINING_PROFILE),
        manifest.get("source_sha256") == ds.attrs.get("source_sha256"),
        isinstance(source, dict), isinstance(files, list) and bool(files),
        isinstance(processing, dict), isinstance(rule, dict),
        manifest.get("build_mode") in ("smoke", "subset", "full"),
    ]
    if not all(checks):
        raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH, "three-hour manifest contract mismatch")
    expected_days = [str(value) for value in np.unique(days)]
    checks = [
        source.get("date_start") == expected_days[0], source.get("date_end") == expected_days[-1],
        source.get("daily_file_count") == len(expected_days), source.get("products") == ["slv", "rad"],
        source.get("excluded_products") == ["chm"],
        processing.get("temporal_method") == "mean_of_three_complete_hourly_samples",
        processing.get("spatial_method") == "spherical_area_weighted_overlap",
        rule.get("interval_start") == _iso_utc(ds.time_bounds.values[0, 0]),
        rule.get("interval_end_exclusive") == _iso_utc(ds.time_bounds.values[-1, 1]),
    ]
    if manifest["build_mode"] == "full":
        checks.append(expected_days[0] == "2020-01-01" and expected_days[-1] == "2021-12-31"
                      and len(times) == 5848)
    if manifest["build_mode"] == "smoke":
        checks.append(7 <= len(expected_days) <= 30)
    # The source digest basis is versioned and must stay symmetric with the
    # builder: version 3 (and therefore the projection branch) verifies by content
    # identity only, while a manifest without the field stays on the full entry
    # digest that the already registered publication was published with.
    inventory_version = threehour_source_inventory_version(manifest)
    projected = inventory_version == THREE_HOURLY_SOURCE_INVENTORY_IDENTITY
    keys = []
    for item in files:
        if not isinstance(item, dict):
            raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH, "invalid source inventory entry")
        product, day, relative = item.get("product"), item.get("date"), item.get("path")
        if not isinstance(relative, str):
            raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH, "invalid source inventory path")
        path = Path(relative)
        checks.extend([
            product in ("slv", "rad"), day in expected_days,
            not path.is_absolute() and ".." not in path.parts and ":" not in relative
            and "\\" not in relative and len(path.parts) == 2,
            path.parts[0] == product if path.parts else False,
            path.name == item.get("name"),
            type(item.get("bytes")) is int and item["bytes"] > 0,
            bool(_SHA256_PATTERN.fullmatch(str(item.get("sha256", "")))),
        ])
        if projected:
            # Version 3 keeps file-system metadata in the optional top-level
            # source_mtimes block; an entry that still carries one must be well
            # formed, but it never takes part in the identity or the source digest.
            checks.append("mtime_ns" not in item
                          or (type(item["mtime_ns"]) is int and item["mtime_ns"] >= 0))
        else:
            checks.append(type(item.get("mtime_ns")) is int and item["mtime_ns"] >= 0)
        keys.append((day, product))
    checks.extend([
        len(keys) == 2 * len(expected_days), len(set(keys)) == len(keys),
        set(keys) == {(day, product) for day in expected_days for product in ("slv", "rad")},
        threehour_source_inventory_sha256(files, inventory_version) == manifest["source_sha256"],
    ])
    source_mtimes = manifest.get("source_mtimes")
    if source_mtimes is not None:
        # Optional provenance block: never an identity input, but a recorded block
        # must stay internally consistent so change detection can trust it.
        checks.append(isinstance(source_mtimes, dict)
                      and set(source_mtimes) == {item.get("path") for item in files
                                                 if isinstance(item, dict)}
                      and all(type(value) is int and value >= 0
                              for value in source_mtimes.values()))
    variables = manifest.get("variables")
    if not isinstance(variables, dict) or set(variables) != set(CHANNELS):
        checks.append(False)
    else:
        for name in CHANNELS:
            entry = variables[name]
            checks.append(isinstance(entry, dict))
            if isinstance(entry, dict):
                checks.extend(entry.get(key) == value for key, value in statistics[name].items())
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        checks.append(False)
    else:
        for name, code in SPLITS.items():
            selected = days[ds.split.values == code]
            expected = {"start": str(selected[0]) if len(selected) else None,
                        "end": str(selected[-1]) if len(selected) else None,
                        "days": int(len(selected) // 8), "steps": int(len(selected))}
            checks.append(splits.get(name) == expected)
    if not all(checks):
        raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH,
                                "three-hour source, variable statistics or split metadata mismatch")


def _verification_proof_path(cache_dir: Any, package_dir: Any) -> Path:
    from services.earth_preparation_cache import digest_json
    root = Path(cache_dir).expanduser().resolve()
    package = str(Path(package_dir).expanduser().resolve())
    return root / "verification" / f"{digest_json({'package': package, 'file': THREE_HOURLY_DATA_FILE_NAME})}.json"


def _proof_matches_manifest(release: VerifiedEarthRelease, manifest: dict, raw: bytes) -> bool:
    """Check every cached descriptor field against the currently read manifest.

    Arrays have fixed scientific semantics and must also match the manifest;
    this does not trust a deserialized metadata dictionary merely because the
    top-level fingerprint and source signature happen to agree.
    """
    try:
        dates, lat, lon = release.dates, release.latitude, release.longitude
        if (np.isnat(dates).any() or len(dates) % 8
                or np.any(np.diff(dates) != np.timedelta64(3, "h"))
                or _iso_utc(dates[0]) != manifest["time_start"]
                or _iso_utc(dates[-1]) != manifest["time_end"]
                or dates[0] != dates[0].astype("datetime64[D]") + np.timedelta64(90, "m")
                or dates[-1] != dates[-1].astype("datetime64[D]") + np.timedelta64(1350, "m")
                or len(dates) != manifest["dimensions"]["time"]
                or not np.array_equal(lat, -90 + (np.arange(len(lat)) + .5) * (180 / len(lat)))
                or not np.array_equal(lon, -180 + (np.arange(len(lon)) + .5) * (360 / len(lon)))):
            return False
        metadata = release.metadata
        calendar = metadata["time"]["calendar"]
        if calendar not in GREGORIAN_CALENDARS:
            return False
        # Reuse the frozen content digest of the verified publication instead of
        # re-projecting the whole source inventory. The frozen value and the
        # manifest's own id, version and data digest must reproduce the published
        # fingerprint exactly; any disagreement is a proof miss, never a silent
        # fallback to the cached metadata.
        frozen_content_sha256 = metadata.get("manifest_content_sha256")
        if not isinstance(frozen_content_sha256, str):
            return False
        resolved = threehour_release_identity(
            manifest, verified_content_sha256=frozen_content_sha256)
        if resolved["dataset_fingerprint"] != metadata.get("dataset_fingerprint"):
            return False
        expected = _threehour_metadata(manifest, raw, calendar, dates, lat, lon,
                                       manifest["variables"])
        # Registration adds these stable ids to in-memory snapshots, so allow
        # them only when equal to their fixed values.
        candidate = dict(metadata)
        for name, value in (("dataset_id", manifest["dataset_id"]),
                            ("dataset_version", THREE_HOURLY_DATASET_VERSION)):
            if name in candidate and candidate.pop(name) != value:
                return False
        expected.pop("dataset_id", None)
        expected.pop("dataset_version", None)
        return candidate == expected
    except (EarthPackageError, KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def _threehour_metadata(manifest, raw, calendar, dates, lat, lon, statistics) -> dict:
    profile = THREE_HOURLY_5DEG_TRAINING_PROFILE if manifest.get("dataset_id") == THREE_HOURLY_5DEG_DATASET_ID else THREE_HOURLY_TRAINING_PROFILE
    step = 180.0 / profile["grid_shape"][0]
    return {
        "dataset_id": manifest["dataset_id"], "dataset_version": manifest["dataset_version"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_content_sha256": threehour_manifest_content_sha256(manifest),
        "data_sha256": manifest["data_sha256"],
        "manifest": manifest, "schema": THREE_HOURLY_SCHEMA,
        "frequency_hours": 3, "step_unit": "hour", "step": 3,
        "grid_shape": list(profile["grid_shape"]),
        "training_profile": dict(profile),
        "time": {"kind": "datetime", "calendar": calendar, "time_zone": "UTC",
                 "start": manifest["time_start"], "end": manifest["time_end"],
                 "count": len(dates), "step": 3, "step_unit": "hour",
                 "frequency_hours": 3, "label": "interval_center",
                 "interval_start": manifest["time_rule"]["interval_start"],
                 "interval_end_exclusive": manifest["time_rule"]["interval_end_exclusive"]},
        "grid": {"shape": list(profile["grid_shape"]), "dimension_order": ["lat", "lon"],
                 "latitude_range": [-90., 90.], "longitude_range": [-180., 180.],
                 "cell_bounds": {"latitude": [-90., 90.], "longitude": [-180., 180.]},
                 "latitude_step": step,
                 "longitude_step": step,
                 "latitude_order": "ascending", "longitude_order": "ascending",
                 "coverage": "global", "wrap_longitude": True,
                 "latitude_values": lat.tolist(), "longitude_values": lon.tolist()},
        "channel_order": list(CHANNELS),
        "variables": [{"id": name, "label": VARIABLE_LABELS[name], "units": unit,
                       "role": "target_and_input" if name == TARGET_CHANNEL else "optional_input",
                       "missing_rate": statistics[name]["missing_rate"],
                       "valid_mask": statistics[name]["valid_mask"]}
                      for name, unit in zip(CHANNELS, UNITS)],
        "splits": dict(manifest["splits"]),
        "limitations": [str(item) for item in manifest.get("limitations", [])],
    }


def _release_from_verification_proof(proof: Mapping[str, Any], *, package_dir: Path,
                                     signature: tuple) -> VerifiedEarthRelease | None:
    """Rehydrate the lightweight release after a completed full verification."""
    from services.earth_preparation_cache import canonical_json, normalize_signature
    try:
        if proof.get("schema") != VERIFIER_VERSION:
            return None
        if (proof.get("dataset_id") not in THREE_HOURLY_DATASET_IDS
                or proof.get("dataset_version") != THREE_HOURLY_DATASET_VERSION
                or type(proof.get("created_at")) not in (int, float)
                or not math.isfinite(proof["created_at"])):
            return None
        # The proof is written atomically, but a torn/partially edited JSON file
        # can still parse.  Verify an integrity digest over the complete payload
        # before using any metadata from it.
        proof_digest = proof.get("proof_sha256")
        if not isinstance(proof_digest, str) or not _SHA256_PATTERN.fullmatch(proof_digest):
            return None
        unsigned = {key: value for key, value in proof.items() if key != "proof_sha256"}
        if hashlib.sha256(canonical_json(unsigned)).hexdigest() != proof_digest:
            return None
        if proof.get("source_path") != str(package_dir.resolve()):
            return None
        if normalize_signature(proof.get("package_signature")) != signature:
            return None
        metadata = proof["metadata"]
        dates = np.asarray(proof["dates"], dtype="datetime64[ns]")
        latitude = readonly_array(proof["latitude"], "float64")
        longitude = readonly_array(proof["longitude"], "float64")
        if not isinstance(metadata, dict):
            return None
        expected_shape = (36, 72) if proof.get("dataset_id") == THREE_HOURLY_5DEG_DATASET_ID else (240, 480)
        if (not isinstance(metadata, dict) or dates.ndim != 1 or len(dates) < 8
                or latitude.shape != (expected_shape[0],) or longitude.shape != (expected_shape[1],)
                or metadata.get("dataset_id") not in (None, *THREE_HOURLY_DATASET_IDS)
                or metadata.get("schema") != THREE_HOURLY_SCHEMA
                or metadata.get("manifest_sha256") != proof.get("manifest_sha256")
                or metadata.get("manifest_content_sha256") != proof.get("manifest_content_sha256")
                or metadata.get("data_sha256") != proof.get("data_sha256")
                or metadata.get("dataset_fingerprint") != proof.get("dataset_fingerprint")
                or not _SHA256_PATTERN.fullmatch(str(proof.get("manifest_sha256", "")))
                or not _SHA256_PATTERN.fullmatch(str(proof.get("data_sha256", "")))
                or not _SHA256_PATTERN.fullmatch(str(proof.get("dataset_fingerprint", "")))):
            return None
        return VerifiedEarthRelease(
            metadata=metadata, signature=signature, dates=readonly_array(dates, "datetime64[ns]"),
            latitude=latitude, longitude=longitude, fields=MappingProxyType({}),
            data_path=package_dir / THREE_HOURLY_DATA_FILE_NAME, verification_cached=True,
        )
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        return None


def _write_verification_proof(cache_dir: Any, package_dir: Path, release: VerifiedEarthRelease) -> None:
    from services.earth_preparation_cache import atomic_write_json, canonical_json, signature_json
    payload = {
        "schema": VERIFIER_VERSION,
        "source_path": str(package_dir.resolve()),
        "package_signature": signature_json(release.signature),
        "dataset_id": release.metadata.get("dataset_id", THREE_HOURLY_DATASET_ID),
        "dataset_version": release.metadata.get("dataset_version", THREE_HOURLY_DATASET_VERSION),
        "manifest_sha256": release.metadata.get("manifest_sha256"),
        "manifest_content_sha256": release.metadata.get("manifest_content_sha256"),
        "data_sha256": release.metadata.get("data_sha256"),
        "dataset_fingerprint": release.metadata.get("dataset_fingerprint"),
        "metadata": release.metadata,
        "dates": [np.datetime_as_string(value, unit="ns") for value in release.dates],
        "latitude": release.latitude.tolist(), "longitude": release.longitude.tolist(),
        "created_at": time.time(),
    }
    payload["proof_sha256"] = hashlib.sha256(
        canonical_json(payload)
    ).hexdigest()
    atomic_write_json(_verification_proof_path(cache_dir, package_dir), payload)


def read_earth_3hourly_release(package_dir: Any, *, verification_cache_dir: Any = None,
                              force_full: bool = False, progress=None) -> VerifiedEarthRelease:
    """Verify the server-configured three-hour package and return a catalog snapshot.

    The builder's canonical manifest content hash defines identity. Its actual
    raw-byte hash is retained separately for diagnostics and cache signatures.
    Fields remain on disk: registering a complete two-year release never creates
    a full in-memory data snapshot or opens daily training/prediction paths.

    A shared proof is optional for direct callers.  ``force_full=True`` always
    repeats the complete raw-file SHA-256 and scientific data/mask scan, then
    atomically replaces the proof.  A hit rechecks the manifest and source file
    identities before and after reading the proof.
    """
    started = time.perf_counter()
    root = Path(package_dir).expanduser().resolve()
    _verification_log(progress, "data validation checking shared proof")
    if verification_cache_dir is None:
        release = _read_earth_3hourly_release(root, force_full=force_full, progress=progress)
    else:
        from services.earth_preparation_cache import preparation_lock
        proof_path = _verification_proof_path(verification_cache_dir, root)
        with preparation_lock(proof_path.with_suffix(".lock"), progress=progress):
            # Capture the source signature only after the builder lock: a
            # process waiting for another verifier can then reuse its proof.
            release = _read_earth_3hourly_release(
                root, verification_cache_dir=verification_cache_dir,
                force_full=force_full, progress=progress)
    _verification_log(progress, f"data validation {'hit' if release.verification_cached else 'completed full scan'} "
                      f"elapsed={time.perf_counter() - started:.3f}s")
    return release


def _verification_log(progress, message):
    message = f"Earth preparation: {message}"
    logger.info(message)
    if progress is not None:
        progress(message)


def _read_earth_3hourly_release(root: Path, *, verification_cache_dir=None,
                               force_full=False, progress=None) -> VerifiedEarthRelease:
    before = package_signature(root, data_file_name=THREE_HOURLY_DATA_FILE_NAME)
    manifest_path, data_path = root / MANIFEST_FILE_NAME, root / THREE_HOURLY_DATA_FILE_NAME
    try:
        if not manifest_path.is_file() or not data_path.is_file():
            raise EarthPackageError(REASON_PACKAGE_MISSING, "three-hour package files are missing")
        if any(any(value is None for value in part) for part in before):
            # A failed Windows ChangeTime query must not become a reusable
            # all-None signature.  Reliable identity is mandatory even when
            # the file bytes can otherwise be read.
            raise EarthPackageError(REASON_PACKAGE_UNREADABLE, "cannot obtain reliable source file identity")
    except OSError as exc:
        raise EarthPackageError(REASON_PACKAGE_UNREADABLE, str(exc)) from exc

    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_unique_json_object,
                              parse_constant=_invalid_json_constant,
                              parse_float=_finite_json_float)
    except OSError as exc:
        raise EarthPackageError(REASON_PACKAGE_UNREADABLE, str(exc)) from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EarthPackageError(REASON_INVALID_MANIFEST, str(exc)) from exc
    _assert_threehour_manifest_identity(manifest)
    # The cold path recomputes the rule the manifest declares. A legacy
    # publication therefore falls back to the rule 1 whole-manifest digest, which
    # is byte-for-byte the value frozen into its descriptors and proof.
    try:
        threehour_release_identity(manifest)
    except (ValueError, RecursionError) as exc:
        raise EarthPackageError(REASON_INVALID_MANIFEST, str(exc)) from exc
    if verification_cache_dir is not None and not force_full:
        from services.earth_preparation_cache import _read_json
        try:
            proof = _read_json(_verification_proof_path(verification_cache_dir, root))
        except RecursionError:
            proof = None
        if proof is not None:
            cached = _release_from_verification_proof(proof, package_dir=root, signature=before)
            if cached is not None and _proof_matches_manifest(cached, manifest, raw):
                if package_signature(root, data_file_name=THREE_HOURLY_DATA_FILE_NAME) != before:
                    raise EarthPackageError(REASON_PACKAGE_CHANGED, "package changed while reading its proof")
                return cached
    _verification_log(progress, "data validation proof miss; performing complete SHA-256 and scientific scan")
    try:
        if data_path.stat().st_size != manifest["data_bytes"]:
            raise EarthPackageError(REASON_MANIFEST_METADATA_MISMATCH, "NetCDF size mismatch")
        sha_started = time.perf_counter()
        if progress is None:
            actual_sha = file_sha256(data_path)
        else:
            from services.earth_preparation_cache import sha256_file
            actual_sha = sha256_file(data_path, progress=progress, stage="source SHA-256")
        _verification_log(progress, f"source SHA-256 elapsed={time.perf_counter() - sha_started:.3f}s")
        if actual_sha != manifest["data_sha256"]:
            raise EarthPackageError(REASON_DATA_FINGERPRINT_MISMATCH, "NetCDF SHA-256 mismatch")
        scan_started = time.perf_counter()
        _verification_log(progress, "scientific data/mask validation started")
        with netcdf_read_lock(), xr.open_dataset(data_path, engine="netcdf4", mask_and_scale=False) as ds:
            calendar = _decode_calendar(ds)
            # Preserve the no-callback call for direct callers; training also
            # gets per-variable chunk progress during the large data scan.
            statistics = (validate_earth_3hourly_dataset(ds) if progress is None
                          else validate_earth_3hourly_dataset(ds, progress=progress))
            _assert_threehour_manifest_matches(ds, manifest, statistics)
            lat, lon = readonly_array(ds.lat.values), readonly_array(ds.lon.values)
            metadata = _threehour_metadata(manifest, raw, calendar, ds.time.values, lat, lon, statistics)
            release = VerifiedEarthRelease(metadata=metadata, signature=before,
                                           dates=readonly_array(ds.time.values, "datetime64[ns]"),
                                           latitude=lat, longitude=lon, fields=MappingProxyType({}),
                                           data_path=data_path)
    except EarthPackageError:
        raise
    except OSError as exc:
        # netCDF errors use negative errno values (e.g. -51 for an unknown
        # format). They describe invalid content, while filesystem/permission
        # failures remain an unreadable package; NC_EPERM is -37.
        reason = (REASON_INVALID_DATASET
                  if isinstance(exc.errno, int) and exc.errno < 0 and exc.errno != -37
                  else REASON_PACKAGE_UNREADABLE)
        raise EarthPackageError(reason, str(exc)) from exc
    except Exception as exc:
        raise EarthPackageError(REASON_INVALID_DATASET, str(exc)) from exc
    _verification_log(progress, f"scientific data/mask validation elapsed={time.perf_counter() - scan_started:.3f}s")
    if package_signature(root, data_file_name=THREE_HOURLY_DATA_FILE_NAME) != before:
        raise EarthPackageError(REASON_PACKAGE_CHANGED, "package files changed during verification")
    if verification_cache_dir is not None:
        try:
            _write_verification_proof(verification_cache_dir, root, release)
        except (OSError, TypeError, ValueError):
            logger.warning("Could not publish Earth verification proof", exc_info=True)
    return release
