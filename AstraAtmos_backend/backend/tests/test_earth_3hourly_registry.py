"""Registration and API contracts for the independent three-hour Earth release.

Only synthetic SLV/RAD sources are used. All publications live in pytest's
workspace temporary directory; these tests neither read nor alter deployed data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from routers.datasets import router as dataset_router
from routers.earth_overview import router as overview_router
from schemas.training import TrainingStartRequest
from scripts.build_earth_merra2_3hourly import DATASET_ID, DATA_FILE, build_dataset, dataset_fingerprint
from services.dataset_identity import DatasetRequestError, build_dataset_fingerprint, resolve_dataset_id
from services.dataset_registry import DatasetRegistry
from services.earth_overview_service import EarthOverviewService
from test_build_earth_merra2_3hourly import make_source_archive


CHANNELS = ["TO3", "U10M", "V10M", "T2M", "SWGDN"]
UNITS = ["DU", "m s-1", "m s-1", "K", "W m-2"]
NEW_CAPABILITIES = {
    "metadata": True, "training": True, "web_overview": True, "trained_prediction": True,
}
IDENTITY_OVERRIDE_FIELDS = [
    "dataset_version", "dataset_fingerprint", "dataset_snapshot", "fingerprint", "snapshot",
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture(scope="module")
def threehour_release(tmp_path_factory):
    root = tmp_path_factory.mktemp("threehour_catalog_source")
    raw = make_source_archive(root / "raw", bad="missing")
    return build_dataset(raw, root / "release", smoke_days=7).parent


def copy_release(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    for name in (DATA_FILE, "manifest.json"):
        shutil.copy2(source / name, destination / name)
    return destination


def manifest_of(package: Path) -> dict:
    return json.loads((package / "manifest.json").read_text(encoding="utf-8"))


def republish_manifest(package: Path, manifest: dict) -> None:
    """Recompute publication hashes so contract checks run after byte checks."""
    manifest["data_sha256"] = sha256(package / DATA_FILE)
    manifest["data_bytes"] = (package / DATA_FILE).stat().st_size
    manifest["dataset_fingerprint"] = dataset_fingerprint(manifest)
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def registry_for(package: Path, tmp_path: Path, **daily_release) -> DatasetRegistry:
    kwargs = {"earth_package_dir": tmp_path / "missing_daily", **daily_release}
    return DatasetRegistry(**kwargs, earth_3hourly_package_dir=package)


def client_for(registry: DatasetRegistry) -> TestClient:
    app = FastAPI()
    app.state.dataset_registry = registry
    app.state.earth_overview_service = EarthOverviewService(registry)
    app.include_router(dataset_router, prefix="/api")
    app.include_router(overview_router, prefix="/api")
    return TestClient(app)


def downgrade_to_legacy(package: Path) -> dict:
    """Rewrite a workspace copy as the pre-rule legacy publication (rule 1)."""
    manifest = manifest_of(package)
    mtimes = manifest.pop("source_mtimes", {})
    manifest.pop("fingerprint_rule", None)
    manifest["source_inventory_version"] = 2
    files = []
    for entry in manifest["source_files"]:
        entry = dict(entry)
        entry["mtime_ns"] = mtimes.get(entry["path"], 0)
        files.append(entry)
    manifest["source_files"] = files
    manifest["source_sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    manifest["source_fingerprint_basis"] = (
        "SHA256(canonical sorted date/product/path/name/bytes/mtime_ns/full-file-sha256 inventory)")
    # The NetCDF records the same provenance digest, so the copy must agree.
    with netCDF4.Dataset(package / DATA_FILE, "a") as data:
        data.source_sha256 = manifest["source_sha256"]
    republish_manifest(package, manifest)
    return manifest


def test_legacy_manifest_without_fingerprint_rule_still_registers(threehour_release, tmp_path):
    """A pre-rule publication keeps its own identity and stays loadable."""
    package = copy_release(threehour_release, tmp_path / "legacy_rule")
    manifest = downgrade_to_legacy(package)
    assert "fingerprint_rule" not in manifest and "source_mtimes" not in manifest
    assert manifest["source_inventory_version"] == 2
    expected_content = hashlib.sha256(json.dumps(
        {key: value for key, value in manifest.items() if key != "dataset_fingerprint"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    registry = registry_for(package, tmp_path)
    descriptor = registry.get_dataset(DATASET_ID)
    assert descriptor["availability"] == "available"
    assert descriptor["availability_reason"] is None
    assert descriptor["manifest_content_sha256"] == expected_content
    assert descriptor["dataset_fingerprint"] == manifest["dataset_fingerprint"]
    assert descriptor["dataset_fingerprint"] == build_dataset_fingerprint(
        DATASET_ID, "v1", descriptor["manifest_content_sha256"], descriptor["data_sha256"])


def test_builder_release_registers_as_exact_datetime_contract(threehour_release, tmp_path):
    registry = registry_for(threehour_release, tmp_path)
    with client_for(registry) as client:
        response = client.get(f"/api/datasets/{DATASET_ID}")
        assert response.status_code == 200
        descriptor = response.json()
        catalog = client.get("/api/datasets").json()["items"]

    assert descriptor["availability"] == "available"
    assert descriptor["availability_reason"] is None
    assert descriptor["schema"] == "astraatmos_earth_3hourly_v1"
    assert descriptor["dataset_version"] == "v1"
    assert descriptor["planet"] == "earth"
    assert descriptor["capabilities"] == NEW_CAPABILITIES
    assert [item["dataset_id"] for item in catalog] == [
        "openmars_mcd", "mcd_overview", DATASET_ID,
    ]
    assert catalog[-1] == descriptor
    assert descriptor["time"]["kind"] == "datetime"
    assert descriptor["time"]["time_zone"] == "UTC"
    assert descriptor["time"]["start"] == "2020-01-01T01:30:00Z"
    assert descriptor["time"]["end"] == "2020-01-07T22:30:00Z"
    assert descriptor["time"]["count"] == 56
    assert descriptor["time"]["interval_start"] == "2020-01-01T00:00:00Z"
    assert descriptor["time"]["interval_end_exclusive"] == "2020-01-08T00:00:00Z"
    for source in (descriptor, descriptor["time"], descriptor["training_profile"]):
        assert source["frequency_hours"] == 3
        assert source["step_unit"] == "hour"
        assert source["step"] == 3
    assert descriptor["grid_shape"] == descriptor["grid"]["shape"] == [240, 480]
    assert descriptor["grid"]["coverage"] == "global"
    assert descriptor["grid"]["wrap_longitude"] is True
    np.testing.assert_array_equal(descriptor["grid"]["latitude_values"], -89.625 + np.arange(240) * .75)
    np.testing.assert_array_equal(descriptor["grid"]["longitude_values"], -179.625 + np.arange(480) * .75)
    assert descriptor["channel_order"] == CHANNELS
    assert [entry["id"] for entry in descriptor["variables"]] == CHANNELS
    assert [entry["units"] for entry in descriptor["variables"]] == UNITS
    assert descriptor["variables"][0]["missing_rate"] > 0
    assert descriptor["variables"][0]["valid_mask"] == "TO3_valid_mask"
    profile = descriptor["training_profile"]
    assert (profile["window"], profile["horizon"], profile["target"], profile["target_unit"]) == (56, 24, "TO3", "DU")
    assert profile["grid_shape"] == [240, 480]
    assert descriptor["splits"]["train"]["steps"] == 56
    manifest = manifest_of(threehour_release)
    assert descriptor["dataset_fingerprint"] == manifest["dataset_fingerprint"]
    assert descriptor["data_sha256"] == sha256(threehour_release / DATA_FILE)
    assert descriptor["manifest_sha256"] == sha256(threehour_release / "manifest.json")
    assert descriptor["dataset_fingerprint"] == build_dataset_fingerprint(
        DATASET_ID, "v1", descriptor["manifest_content_sha256"], descriptor["data_sha256"],
    )


def test_new_identity_isolated_from_retired_daily_id(
    threehour_release, earth_global_release, tmp_path,
):
    registry = registry_for(threehour_release, tmp_path, **earth_global_release)
    descriptor = registry.get_dataset(DATASET_ID)
    assert descriptor["availability"] == "available"
    for read in (registry.get_dataset, registry.build_training_binding):
        with pytest.raises(DatasetRequestError) as error:
            read("earth_merra2_daily_v2")
        assert error.value.code == "dataset_retired"
    assert Path(earth_global_release["earth_package_dir"]).is_dir()
    assert descriptor["dataset_fingerprint"] != build_dataset_fingerprint(
        "earth_merra2_daily_v2", "v1", descriptor["manifest_content_sha256"], descriptor["data_sha256"],
    )


@pytest.mark.parametrize("missing", ["directory", "manifest", "netcdf"])
def test_missing_new_release_returns_stable_catalog_without_daily_fallback(
    threehour_release, earth_global_release, tmp_path, missing,
):
    package = tmp_path / "missing_new"
    if missing != "directory":
        package.mkdir()
        retained_name = DATA_FILE if missing == "manifest" else "manifest.json"
        shutil.copy2(threehour_release / retained_name, package / retained_name)
    registry = registry_for(package, tmp_path, **earth_global_release)
    with client_for(registry) as client:
        first = client.get(f"/api/datasets/{DATASET_ID}")
        second = client.get(f"/api/datasets/{DATASET_ID}")
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        catalog = client.get("/api/datasets")
    descriptor = first.json()
    assert descriptor["availability"] == "missing"
    assert descriptor["availability_reason"] == "package_missing"
    assert descriptor["dataset_fingerprint"] is None
    assert descriptor["time"]["kind"] == "datetime"
    assert descriptor["frequency_hours"] == 3
    assert descriptor["grid_shape"] == [240, 480]
    assert descriptor["capabilities"] == NEW_CAPABILITIES
    assert catalog.status_code == 200
    assert catalog.json()["items"][-1]["availability"] == "missing"
    assert len(catalog.json()["items"]) == 3


@pytest.mark.parametrize("bad,reason", [
    ("malformed_json", "invalid_manifest"), ("array", "invalid_manifest"),
    ("duplicate_json_key", "invalid_manifest"), ("non_finite_json", "invalid_manifest"),
    ("overflow_number", "invalid_manifest"), ("negative_overflow_number", "invalid_manifest"),
    ("deep_json", "invalid_manifest"),
    ("wrong_dataset_id", "invalid_manifest"), ("wrong_version", "invalid_manifest"),
    ("path_traversal", "invalid_manifest"), ("fingerprint", "manifest_fingerprint_mismatch"),
    ("netcdf_sha", "data_fingerprint_mismatch"), ("size", "manifest_metadata_mismatch"),
    ("frequency", "manifest_metadata_mismatch"), ("training_window", "manifest_metadata_mismatch"),
    ("source_inventory", "manifest_metadata_mismatch"), ("variable_statistics", "manifest_metadata_mismatch"),
])
def test_invalid_manifest_has_stable_reason_without_daily_fallback(
    threehour_release, earth_global_release, tmp_path, bad, reason,
):
    package = copy_release(threehour_release, tmp_path / "bad_manifest")
    manifest = manifest_of(package)
    raw = None
    if bad == "malformed_json":
        raw = "{unfinished"
    elif bad == "array":
        raw = "[]"
    elif bad == "duplicate_json_key":
        raw = '{"dataset_id":"first","dataset_id":"second"}'
    elif bad == "non_finite_json":
        raw = '{"missing_rate":NaN}'
    elif bad in ("overflow_number", "negative_overflow_number"):
        number = "1e999" if bad == "overflow_number" else "-1e999"
        raw = json.dumps(manifest)[:-1] + ', "extra": ' + number + '}'
    elif bad == "deep_json":
        raw = json.dumps(manifest)[:-1] + ', "extra": ' + '[' * 5000 + '0' + ']' * 5000 + '}'
    elif bad == "wrong_dataset_id":
        manifest["dataset_id"] = "earth_merra2_daily_v2"
    elif bad == "wrong_version":
        manifest["dataset_version"] = "v2"
    elif bad == "path_traversal":
        manifest["data_file"] = "../earth_merra2_3hourly.nc"
    elif bad == "fingerprint":
        manifest["dataset_fingerprint"] = "0" * 64
    elif bad == "netcdf_sha":
        manifest["data_sha256"] = "0" * 64
    elif bad == "size":
        manifest["data_bytes"] += 1
    elif bad == "frequency":
        manifest["frequency_hours"] = 1
    elif bad == "training_window":
        manifest["training_profile"]["window"] = 7
    elif bad == "source_inventory":
        manifest["source_files"][0]["path"] = "../private/raw.nc4"
    elif bad == "variable_statistics":
        manifest["variables"]["TO3"]["missing_rate"] = .5
    if raw is None:
        if bad != "fingerprint":
            manifest["dataset_fingerprint"] = dataset_fingerprint(manifest)
        raw = json.dumps(manifest)
    (package / "manifest.json").write_text(raw, encoding="utf-8")
    registry = registry_for(package, tmp_path, **earth_global_release)
    with client_for(registry) as client:
        descriptor = client.get(f"/api/datasets/{DATASET_ID}").json()
        repeated = client.get(f"/api/datasets/{DATASET_ID}").json()
        daily = client.get("/api/datasets/earth_merra2_daily_v2")
        catalog = client.get("/api/datasets")
    assert descriptor == repeated
    assert descriptor["availability"] == "invalid"
    assert descriptor["availability_reason"] == reason
    assert descriptor["dataset_fingerprint"] is None
    assert daily.status_code == 409
    assert daily.json()["detail"]["code"] == "dataset_retired"
    assert catalog.status_code == 200
    assert len(catalog.json()["items"]) == 3
    assert catalog.json()["items"][-1] == descriptor


@pytest.mark.parametrize("bad", [
    "duplicate_time", "missing_step", "hourly_time", "not_utc", "bad_time_bounds", "latitude",
    "longitude_seam", "latitude_bounds", "missing_variable", "to3_unit", "wind_unit", "mask", "nan_without_mask",
])
def test_rehashed_invalid_netcdf_still_fails_the_data_contract(threehour_release, tmp_path, bad):
    package = copy_release(threehour_release, tmp_path / "bad_data")
    manifest = manifest_of(package)
    with netCDF4.Dataset(package / DATA_FILE, "a") as data:
        data.set_auto_mask(False)
        if bad == "duplicate_time":
            data["time"][1] = data["time"][0]
        elif bad == "missing_step":
            data["time"][1:] = data["time"][1:] + 3
        elif bad == "hourly_time":
            data["time"][:] = np.arange(56) + 1.5
        elif bad == "not_utc":
            data["time"].time_zone = "Asia/Shanghai"
        elif bad == "bad_time_bounds":
            data["time_bounds"][0, 1] = 4
        elif bad == "latitude":
            data["lat"][10] = data["lat"][10] + .125
        elif bad == "longitude_seam":
            data["lon"][-1] = 180
        elif bad == "latitude_bounds":
            data["lat_bounds"][0, 0] = -89.9
        elif bad == "missing_variable":
            data.renameVariable("SWGDN", "unused_SWGDN")
        elif bad == "to3_unit":
            data["TO3"].units = "kg m-2"
        elif bad == "wind_unit":
            data["U10M"].units = "km h-1"
        elif bad == "mask":
            data["TO3_valid_mask"][0, 0, 100] = 0
        elif bad == "nan_without_mask":
            data["TO3"][0, 0, 100] = np.nan
    republish_manifest(package, manifest)
    registry = registry_for(package, tmp_path)
    with client_for(registry) as client:
        response = client.get(f"/api/datasets/{DATASET_ID}")
    assert response.status_code == 200
    assert response.json()["availability"] == "invalid"
    assert response.json()["availability_reason"] == "invalid_dataset"


def test_data_byte_corruption_invalidates_positive_cache(threehour_release, tmp_path):
    package = copy_release(threehour_release, tmp_path / "changed_bytes")
    registry = registry_for(package, tmp_path)
    original = registry.get_dataset(DATASET_ID)
    assert original["availability"] == "available"
    registry.get_dataset(DATASET_ID)
    assert registry.verification_count == 1
    data_path = package / DATA_FILE
    with data_path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 0xFF]))
    changed = registry.get_dataset(DATASET_ID)
    assert changed["availability"] == "invalid"
    assert changed["availability_reason"] == "data_fingerprint_mismatch"
    assert changed["dataset_fingerprint"] is None
    assert registry.verification_count == 2
    assert registry.get_dataset(DATASET_ID) == changed
    assert registry.verification_count == 2


def test_manifest_format_change_invalidates_cache_but_preserves_content_identity(threehour_release, tmp_path):
    package = copy_release(threehour_release, tmp_path / "reformatted")
    registry = registry_for(package, tmp_path)
    before = registry.get_dataset(DATASET_ID)
    manifest = manifest_of(package)
    (package / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    after = registry.get_dataset(DATASET_ID)
    assert after["availability"] == "available"
    assert before["dataset_fingerprint"] == after["dataset_fingerprint"]
    assert before["manifest_content_sha256"] == after["manifest_content_sha256"]
    assert before["manifest_sha256"] != after["manifest_sha256"]
    assert registry.verification_count == 2


@pytest.mark.parametrize("initial", ["missing", "invalid"])
def test_negative_cache_refreshes_when_release_files_are_published_or_repaired(threehour_release, tmp_path, initial):
    package = tmp_path / "repairable"
    if initial == "invalid":
        copy_release(threehour_release, package)
        (package / "manifest.json").write_text("{bad", encoding="utf-8")
    registry = registry_for(package, tmp_path)
    first = registry.get_dataset(DATASET_ID)
    assert first["availability"] == initial
    assert registry.get_dataset(DATASET_ID) == first
    assert registry.verification_count == 1
    if initial == "missing":
        copy_release(threehour_release, package)
    else:
        shutil.copy2(threehour_release / "manifest.json", package / "manifest.json")
    restored = registry.get_dataset(DATASET_ID)
    assert restored["availability"] == "available"
    assert restored["dataset_fingerprint"] == manifest_of(threehour_release)["dataset_fingerprint"]
    assert registry.verification_count == 2


def test_descriptor_does_not_leak_paths_and_cannot_mutate_cached_contract(threehour_release, tmp_path):
    registry = registry_for(threehour_release, tmp_path)
    with client_for(registry) as client:
        detail = client.get(f"/api/datasets/{DATASET_ID}").text
        catalog = client.get("/api/datasets").text
    for response in (detail, catalog):
        assert str(threehour_release) not in response
        assert "earth_3hourly_package_dir" not in response
        assert "source_files" not in response
        assert "source_sha256" not in response
    first = registry.get_dataset(DATASET_ID)
    first["training_profile"]["window"] = 7
    first["grid"]["shape"][0] = 36
    first["time"]["start"] = "2020-01-01"
    second = registry.get_dataset(DATASET_ID)
    assert second["training_profile"]["window"] == 56
    assert second["grid"]["shape"] == [240, 480]
    assert second["time"]["start"] == "2020-01-01T01:30:00Z"


def test_registered_new_id_binds_training_and_overview_rejects_date_only(threehour_release, tmp_path):
    assert resolve_dataset_id(DATASET_ID) == DATASET_ID
    assert resolve_dataset_id(None, {"training_dataset": DATASET_ID}) == DATASET_ID
    registry = registry_for(threehour_release, tmp_path)
    binding = registry.build_training_binding(DATASET_ID)
    assert binding["dataset_id"] == DATASET_ID
    assert binding["dataset_identity_status"] == "verified"
    snapshot = json.loads(binding["dataset_snapshot"])
    assert snapshot["time"]["kind"] == "datetime"
    assert snapshot["training_profile"]["window"] == 56
    assert snapshot["training_profile"]["horizon"] == 24
    with client_for(registry) as client:
        descriptor = client.get(f"/api/datasets/{DATASET_ID}").json()
        response = client.get(f"/api/datasets/{DATASET_ID}/overview/field", params={
            "date": "2020-01-01", "variable": "TO3", "expected_fingerprint": descriptor["dataset_fingerprint"],
        })
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_timestamp"


@pytest.mark.parametrize("field", IDENTITY_OVERRIDE_FIELDS)
def test_identity_override_in_hyperparameters_is_rejected_by_resolver(field):
    with pytest.raises(DatasetRequestError) as error:
        resolve_dataset_id(DATASET_ID, {field: "forged"})
    assert error.value.code == "client_identity_not_allowed"
    assert error.value.status_code == 400


@pytest.mark.parametrize("field", IDENTITY_OVERRIDE_FIELDS)
def test_training_request_schema_rejects_top_level_client_identity_overrides(field):
    request = {"model_script": "earth_daily.py", "model_name": "test", "dataset_id": DATASET_ID,
               "hyperparameters": {"window": 56, "horizon": 24}}
    request[field] = "forged"
    with pytest.raises(ValidationError, match="identity is generated by the server"):
        TrainingStartRequest.model_validate(request)


@pytest.mark.parametrize("field", IDENTITY_OVERRIDE_FIELDS)
def test_nested_identity_keeps_schema_acceptance_and_resolver_400_contract(field):
    request = TrainingStartRequest.model_validate({
        "model_script": "earth_daily.py", "model_name": "test", "dataset_id": DATASET_ID,
        "hyperparameters": {field: "forged"},
    })
    with pytest.raises(DatasetRequestError) as error:
        resolve_dataset_id(request.dataset_id, request.hyperparameters)
    assert error.value.code == "client_identity_not_allowed"
    assert error.value.status_code == 400
