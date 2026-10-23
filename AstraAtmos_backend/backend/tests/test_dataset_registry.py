"""Earth release verification contract for the dataset registry."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_identity import build_dataset_fingerprint  # noqa: E402
from services.earth_dataset_metadata import (  # noqa: E402
    DATA_FILE_NAME,
    REASON_DATA_FINGERPRINT_MISMATCH,
    REASON_INVALID_DATASET,
    REASON_INVALID_MANIFEST,
    REASON_MANIFEST_FINGERPRINT_MISMATCH,
    REASON_MANIFEST_METADATA_MISMATCH,
    REASON_PACKAGE_MISSING,
    EarthPackageError,
    read_earth_metadata,
)

DATASET_ID = "earth_merra2_daily_v1"
DATASET_VERSION = "v1"


def read_release(release):
    return read_earth_metadata(
        release["earth_package_dir"],
        expected_manifest_sha256=release["expected_manifest_sha256"],
        expected_data_sha256=release["expected_data_sha256"],
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
    )


def reason_of(release, *, package_dir=None):
    with pytest.raises(EarthPackageError) as exc:
        read_earth_metadata(
            package_dir or release["earth_package_dir"],
            expected_manifest_sha256=release["expected_manifest_sha256"],
            expected_data_sha256=release["expected_data_sha256"],
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
        )
    return exc.value.reason


def copy_release(release, tmp_path):
    destination = tmp_path / "copied_release"
    destination.mkdir()
    for name in (DATA_FILE_NAME, "manifest.json"):
        shutil.copy2(Path(release["earth_package_dir"]) / name, destination / name)
    copied = dict(release)
    copied["earth_package_dir"] = destination
    return copied


def rebuild_manifest_with_trusted_hashes(package_dir, mutate):
    """Rewrite the manifest and point its self reported hashes at the real file."""
    package_dir = Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    data_path = package_dir / DATA_FILE_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["data_bytes"] = data_path.stat().st_size
    manifest["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "earth_package_dir": package_dir,
        "expected_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }


def rewrite_dataset(package_dir, mutate):
    """Rewrite the NetCDF after applying ``mutate(ds)``.

    ``mutate`` may change the dataset in place or return a new dataset.
    """
    package_dir = Path(package_dir)
    data_path = package_dir / DATA_FILE_NAME
    with xr.open_dataset(data_path, engine="netcdf4") as source:
        ds = source.load()
    updated = mutate(ds)
    if updated is not None:
        ds = updated
    temp_path = package_dir / "rewrite_tmp.nc"
    encoding = {
        name: {"dtype": "float32", "zlib": True, "complevel": 4, "shuffle": True}
        for name in ("TO3", "U10M", "V10M", "T2M", "SWGDN")
    }
    ds.to_netcdf(temp_path, engine="netcdf4", encoding=encoding)
    ds.close()
    temp_path.replace(data_path)
    return data_path


def test_release_metadata_reports_real_dates_units_and_coordinates(earth_release):
    result = read_release(earth_release)

    assert result["schema"] == "astraatmos_earth_daily_v1"
    assert result["manifest_sha256"] == earth_release["expected_manifest_sha256"]
    assert result["data_sha256"] == earth_release["expected_data_sha256"]
    assert result["dataset_fingerprint"] == build_dataset_fingerprint(
        DATASET_ID, DATASET_VERSION,
        earth_release["expected_manifest_sha256"],
        earth_release["expected_data_sha256"],
    )

    time = result["time"]
    assert time == {
        "kind": "date",
        "calendar": "proleptic_gregorian",
        "start": "2020-01-01",
        "end": "2021-12-31",
        "count": 731,
        "step": 1,
        "step_unit": "day",
    }

    grid = result["grid"]
    assert grid["shape"] == [31, 49]
    assert grid["dimension_order"] == ["lat", "lon"]
    assert grid["latitude_range"] == [-60.0, 60.0]
    assert grid["longitude_range"] == [-120.0, 120.0]
    assert grid["latitude_step"] == 4.0
    assert grid["longitude_step"] == 5.0
    assert grid["latitude_order"] == "ascending"
    assert grid["longitude_order"] == "ascending"
    assert grid["coverage"] == "regional"
    assert grid["wrap_longitude"] is False
    assert np.allclose(grid["latitude_values"], np.arange(-60, 61, 4))
    assert np.allclose(grid["longitude_values"], np.arange(-120, 121, 5))

    assert result["channel_order"] == ["TO3", "U10M", "V10M", "T2M", "SWGDN"]
    assert [variable["units"] for variable in result["variables"]] == [
        "DU", "m s-1", "m s-1", "K", "W m-2",
    ]
    assert result["variables"][0] == {
        "id": "TO3", "label": "Total column ozone", "units": "DU",
        "role": "target_and_input",
    }
    assert all(
        variable["role"] == "optional_input" for variable in result["variables"][1:]
    )
    assert result["splits"] == {
        "train": {"start": "2020-01-01", "end": "2020-12-31", "days": 366},
        "validation": {"start": "2021-01-01", "end": "2021-06-30", "days": 181},
        "test": {"start": "2021-07-01", "end": "2021-12-31", "days": 184},
    }
    assert result["limitations"]
    assert all(isinstance(item, str) for item in result["limitations"])


def test_metadata_is_json_serializable_without_numpy_scalars(earth_release):
    result = read_release(earth_release)
    json.dumps(result)

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert not isinstance(value, np.generic), value
            assert not isinstance(value, np.ndarray), value

    walk(result)


@pytest.mark.parametrize("missing", [DATA_FILE_NAME, "manifest.json"])
def test_missing_package_file_is_reported_not_raised_as_crash(earth_release, tmp_path, missing):
    copied = copy_release(earth_release, tmp_path)
    (Path(copied["earth_package_dir"]) / missing).unlink()

    assert reason_of(copied) == REASON_PACKAGE_MISSING


def test_missing_directory_is_reported(earth_release, tmp_path):
    assert reason_of(earth_release, package_dir=tmp_path / "absent") == REASON_PACKAGE_MISSING


def test_changed_data_file_is_not_accepted_as_the_released_version(earth_release, tmp_path):
    copied = copy_release(earth_release, tmp_path)
    data_path = Path(copied["earth_package_dir"]) / DATA_FILE_NAME
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 0xFF
    data_path.write_bytes(bytes(payload))

    assert reason_of(copied) == REASON_DATA_FINGERPRINT_MISMATCH


def test_changed_manifest_is_not_accepted_as_the_released_version(earth_release, tmp_path):
    copied = copy_release(earth_release, tmp_path)
    manifest_path = Path(copied["earth_package_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["limitations"] = ["tampered"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    assert reason_of(copied) == REASON_MANIFEST_FINGERPRINT_MISMATCH


def test_replacing_data_and_manifest_hashes_still_fails_the_fixed_release(earth_release, tmp_path):
    """R4: a self consistent replacement cannot impersonate the released v1."""
    copied = copy_release(earth_release, tmp_path)
    rewrite_dataset(copied["earth_package_dir"], lambda ds: ds["TO3"].__setitem__(
        (slice(None), slice(None), slice(None)), ds["TO3"].values + 1.0
    ))
    rebuilt = rebuild_manifest_with_trusted_hashes(copied["earth_package_dir"], lambda manifest: None)

    # The rebuilt package is internally consistent, so the pinned release hashes
    # of the original release remain the authority.
    assert rebuilt["expected_data_sha256"] != earth_release["expected_data_sha256"]
    with pytest.raises(EarthPackageError) as exc:
        read_earth_metadata(
            rebuilt["earth_package_dir"],
            expected_manifest_sha256=earth_release["expected_manifest_sha256"],
            expected_data_sha256=earth_release["expected_data_sha256"],
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
        )
    assert exc.value.reason == REASON_MANIFEST_FINGERPRINT_MISMATCH


def test_trusted_fingerprints_still_reject_wrong_self_reported_hashes(earth_release, tmp_path):
    """A trusted-hash fixture whose manifest under-reports its own data hash."""
    copied = copy_release(earth_release, tmp_path)
    package_dir = Path(copied["earth_package_dir"])
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_sha256"] = "0" * 64
    manifest["data_bytes"] = 123
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    trusted = {
        "earth_package_dir": package_dir,
        "expected_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_data_sha256": hashlib.sha256(
            (package_dir / DATA_FILE_NAME).read_bytes()
        ).hexdigest(),
    }
    assert reason_of(trusted) == REASON_MANIFEST_METADATA_MISMATCH


def test_trusted_fingerprints_still_reject_inconsistent_manifest_metadata(earth_release, tmp_path):
    """R5: semantic checks must not be skipped just because the hashes were trusted."""
    copied = copy_release(earth_release, tmp_path)
    trusted = rebuild_manifest_with_trusted_hashes(
        copied["earth_package_dir"],
        lambda manifest: manifest.update({
            "time_start": "2019-01-01",
            "channel_order": ["TO3"],
            "variables": {**manifest["variables"], "TO3": {
                **manifest["variables"]["TO3"], "units": "ppb", "min": -1.0, "max": 1.0,
            }},
            "splits": {**manifest["splits"], "train": {"start": "2019-01-01", "end": "2019-12-31", "days": 365}},
        }),
    )

    assert reason_of(trusted) == REASON_MANIFEST_METADATA_MISMATCH


def test_trusted_fingerprints_still_reject_wrong_manifest_planet_and_schema(earth_release, tmp_path):
    copied = copy_release(earth_release, tmp_path)
    trusted = rebuild_manifest_with_trusted_hashes(
        copied["earth_package_dir"],
        lambda manifest: manifest.update({"planet": "Mars", "schema": "astraatmos_mars_v1"}),
    )

    assert reason_of(trusted) == REASON_MANIFEST_METADATA_MISMATCH


def test_manifest_data_file_must_be_the_fixed_package_file(earth_release, tmp_path):
    copied = copy_release(earth_release, tmp_path)
    trusted = rebuild_manifest_with_trusted_hashes(
        copied["earth_package_dir"],
        lambda manifest: manifest.update({"data_file": "../outside.nc"}),
    )

    assert reason_of(trusted) == REASON_INVALID_MANIFEST


def test_manifest_that_is_not_a_json_object_is_rejected(earth_release, tmp_path):
    copied = copy_release(earth_release, tmp_path)
    manifest_path = Path(copied["earth_package_dir"]) / "manifest.json"
    manifest_path.write_text("[1, 2, 3]", encoding="utf-8")

    trusted = {
        "earth_package_dir": copied["earth_package_dir"],
        "expected_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_data_sha256": copied["expected_data_sha256"],
    }
    assert reason_of(trusted) == REASON_INVALID_MANIFEST


def set_time_calendar(package_dir, calendar):
    """Change only the calendar attribute of the time variable.

    Values and their numeric encoding stay untouched, so the manifest and the
    decoded content agree everywhere except the calendar itself.
    """
    import netCDF4

    data_path = Path(package_dir) / DATA_FILE_NAME
    with netCDF4.Dataset(data_path, "r+") as ds:
        original_name = "time_original"
        if "time" in ds.variables:
            ds.renameVariable("time", original_name)
        source = ds.variables[original_name]
        attributes = {name: source.getncattr(name) for name in source.ncattrs()}
        target = ds.createVariable("time", source.dtype, source.dimensions)
        target[:] = source[:]
        for name, value in attributes.items():
            target.setncattr(name, value)
        target.setncattr("calendar", calendar)
    return data_path


def test_unsupported_calendar_is_rejected_instead_of_treated_as_mars_time(earth_release, tmp_path):
    copied = copy_release(earth_release, tmp_path)
    set_time_calendar(copied["earth_package_dir"], "360_day")
    trusted = rebuild_manifest_with_trusted_hashes(copied["earth_package_dir"], lambda manifest: None)

    assert reason_of(trusted) == REASON_INVALID_DATASET


def test_nonuniform_grid_is_rejected(earth_release, tmp_path):
    copied = copy_release(earth_release, tmp_path)

    def corrupt_latitude(ds):
        # Keep the declared range so the nonuniformity check is what rejects it.
        values = np.arange(-60, 61, 4, dtype=np.float32)
        values[15] -= np.float32(0.5)
        ds = ds.assign_coords(lat=("lat", values))
        ds["lat"].attrs["units"] = "degrees_north"
        return ds

    rewrite_dataset(copied["earth_package_dir"], corrupt_latitude)
    trusted = rebuild_manifest_with_trusted_hashes(copied["earth_package_dir"], lambda manifest: None)

    assert reason_of(trusted) == REASON_INVALID_DATASET


def test_invalid_expected_digest_is_rejected_before_touching_files(earth_release):
    with pytest.raises(EarthPackageError) as exc:
        read_earth_metadata(
            earth_release["earth_package_dir"],
            expected_manifest_sha256="not-a-digest",
            expected_data_sha256=earth_release["expected_data_sha256"],
        )
    assert exc.value.reason == REASON_INVALID_MANIFEST
