"""Three-hour UTC overview against a genuinely verified 240 by 480 package."""

import copy
import json

import netCDF4
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.earth_overview import router
from services.dataset_identity import DatasetRequestError
from services.dataset_registry import DatasetRegistry
from services.earth_dataset_metadata import VerifiedEarthRelease
from services.earth_3hourly_overview import (
    OverviewCache, read_threehour_values, render_block_mean, threehour_area_mean,
)
from services.earth_overview_service import EarthOverviewError, EarthOverviewService
from test_earth_3hourly_registry import (
    DATASET_ID, DATA_FILE, copy_release, threehour_release,
)

PREFIX = f"/api/datasets/{DATASET_ID}/overview"


@pytest.fixture
def overview(threehour_release, tmp_path):
    registry = DatasetRegistry(tmp_path / "missing_daily", earth_3hourly_package_dir=threehour_release)
    release = registry.get_earth_snapshot(DATASET_ID)
    return EarthOverviewService(registry), registry, release


@pytest.fixture
def client(overview):
    service, registry, release = overview
    app = FastAPI()
    app.state.earth_overview_service = service
    app.state.dataset_registry = registry
    app.include_router(router, prefix="/api")
    with TestClient(app) as connection:
        yield connection, release.metadata["dataset_fingerprint"]


def arguments(release, **extra):
    return {"dataset_id": DATASET_ID, "expected_fingerprint": release.metadata["dataset_fingerprint"],
            "variable": "TO3", **extra}


def test_field_is_selected_timestamp_and_server_preview(overview):
    service, _, release = overview
    result = service.get_field(**arguments(release), timestamp="2020-01-01T04:30:00Z")
    assert result["timestamp"] == result["date"] == "2020-01-01T04:30:00Z"
    assert result["frequency_hours"] == result["step"] == 3
    assert result["step_unit"] == "hour" and result["time_zone"] == "UTC"
    assert result["source_grid_shape"] == [240, 480]
    assert result["grid_shape"] == result["render_grid_shape"] == [60, 120]
    assert np.asarray(result["field"]).shape == (60, 120)
    assert np.allclose(result["field"], 16.)
    assert result["render"]["method"] == "spherical_cell_area_block_mean"
    assert result["render_method"] == "spherical_cell_area_mean_4x4"
    assert result["statistics"]["valid_count"] == 115_200
    assert len(json.dumps(result).encode()) < 160_000
    assert result["color_range"]["min"] == 6.
    assert result["color_range"]["max"] == 676.
    later = service.get_field(**arguments(release), timestamp="2020-01-07T22:30:00Z")
    assert later["color_range"] == result["color_range"]


def test_area_render_is_weighted_not_stride_sampling():
    latitude = -89.625 + np.arange(240) * .75
    native = np.repeat(np.arange(240, dtype="float32")[:, None], 480, axis=1)
    rendered = render_block_mean(native, latitude)
    edges = np.deg2rad(np.arange(-90., -86.999, .75))
    weights = np.diff(np.sin(edges))
    expected = np.dot(np.arange(4), weights) / weights.sum()
    assert rendered[0, 0] == pytest.approx(expected)
    assert rendered[0, 0] != native[0, 0]
    assert rendered[0, 0] != native[:4, :4].mean()


def test_missing_values_become_null_not_zero(overview):
    service, _, release = overview
    result = service.get_field(**arguments(release), timestamp="2020-01-01T01:30:00Z")
    assert any(value is None for row in result["field"] for value in row)
    assert result["statistics"]["valid_count"] < 115_200
    assert all(value == pytest.approx(6.) for row in result["field"] for value in row if value is not None)
    point = service.get_point_series(**arguments(release), lat=.375, lon=-179.625,
                                     start="2020-01-01T01:30:00Z", end="2020-01-01T07:30:00Z")
    assert point["values"] == [None, 16., 26.]
    assert point["missing_policy"] == "preserve_null"


def test_point_series_uses_native_cell_and_threehour_axis(overview):
    service, _, release = overview
    result = service.get_point_series(**arguments(release), lat=.375, lon=.375)
    assert result["grid_point"] == {"lat": .375, "lon": .375, "lat_index": 120, "lon_index": 240}
    assert result["source_grid_shape"] == [240, 480]
    assert len(result["timestamps"]) == len(result["values"]) == 56
    assert result["timestamps"] == result["dates"]
    axis = np.asarray([value[:-1] for value in result["timestamps"]], dtype="datetime64[s]")
    assert np.all(np.diff(axis) == np.timedelta64(3, "h"))
    assert result["values"][:3] == [6., 16., 26.]


def test_regional_series_masked_area_means_and_window(overview):
    service, _, release = overview
    result = service.get_regional_series(**arguments(release), start="2020-01-01T01:30:00Z",
                                         end="2020-01-01T22:30:00Z")
    assert result["values"] == pytest.approx([6., 16., 26., 36., 46., 56., 66., 76.])
    assert result["aggregation"] == "spherical_cell_area_mean"
    assert result["missing_policy"] == "valid_cell_area_mean"
    assert len(result["timestamps"]) == 8


def test_all_missing_has_no_invented_mean():
    field = np.full((1, 240, 480), np.nan)
    latitude = -89.625 + np.arange(240) * .75
    assert np.isnan(threehour_area_mean(field, latitude)[0])
    assert np.isnan(render_block_mean(field[0], latitude)).all()


def test_streaming_reads_at_most_eight_frames(overview, monkeypatch):
    import services.earth_3hourly_overview as module
    service, _, release = overview
    original, sizes = module._read_values, []
    def bounded(ds, current, variable, start, stop, latitude, longitude):
        sizes.append(stop - start)
        return original(ds, current, variable, start, stop, latitude, longitude)
    monkeypatch.setattr(module, "_read_values", bounded)
    service.get_regional_series(**arguments(release))
    service.get_point_series(**arguments(release), lat=0., lon=0.)
    assert max(sizes) == 8
    assert sum(sizes) == 112
    assert release.fields == {}


@pytest.mark.parametrize("value,code", [
    ("2020-01-01", "invalid_timestamp"),
    ("2020-01-01T01:30:00", "invalid_timestamp"),
    ("2020-01-01T09:30:00+08:00", "invalid_timestamp"),
    ("2020-02-30T01:30:00Z", "invalid_timestamp"),
    ("2020-01-01T02:30:00Z", "timestamp_out_of_range"),
    ("2021-01-01T01:30:00Z", "timestamp_out_of_range"),
])
def test_timestamp_validation(overview, value, code):
    service, _, release = overview
    with pytest.raises(EarthOverviewError) as error:
        service.get_field(**arguments(release), timestamp=value)
    assert error.value.status_code == 422 and error.value.code == code


def test_cache_hits_and_variable_timestamp_resolution_isolation(overview, monkeypatch):
    import services.earth_3hourly_overview as module
    service, _, release = overview
    original, calls = module.read_threehour_values, []
    def counted(*args, **kwargs):
        calls.append(args[1:])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "read_threehour_values", counted)
    first = service.get_field(**arguments(release), timestamp="2020-01-01T04:30:00Z")
    first["field"][0][0] = -999.
    again = service.get_field(**arguments(release), timestamp="2020-01-01T04:30:00+00:00")
    assert again["field"][0][0] == 16.
    assert len(calls) == 1
    service.get_field(**arguments(release), timestamp="2020-01-01T07:30:00Z")
    service.get_field(**arguments(release, variable="T2M"), timestamp="2020-01-01T04:30:00Z")
    coarse = service.get_field(**arguments(release), timestamp="2020-01-01T04:30:00Z", render_stride=8)
    assert coarse["render_grid_shape"] == [30, 60]
    assert len(calls) == 4


def test_cache_identity_includes_planet_dataset_version_fingerprint(overview):
    service, _, release = overview
    key = service._threehour._key(release, "TO3", "field", "2020-01-01T04:30:00Z", 4)
    assert key[:4] == ("earth", DATASET_ID, "v1", release.metadata["dataset_fingerprint"])
    for name, value in (("dataset_id", "earth_merra2_daily_v2"), ("dataset_version", "v2"),
                        ("dataset_fingerprint", "a" * 64)):
        changed = copy.deepcopy(release.metadata)
        changed[name] = value
        alternate = VerifiedEarthRelease(changed, release.signature, release.dates, release.latitude,
                                         release.longitude, release.fields, release.data_path)
        assert service._threehour._key(alternate, "TO3", "field", "2020-01-01T04:30:00Z", 4) != key


def test_changed_fingerprint_rejected_before_cache(overview):
    service, _, release = overview
    service.get_field(**arguments(release), timestamp="2020-01-01T04:30:00Z")
    with pytest.raises(DatasetRequestError) as error:
        service.get_field(**arguments(release, expected_fingerprint="a" * 64), timestamp="2020-01-01T04:30:00Z")
    assert error.value.code == "dataset_version_changed"


def test_changed_package_never_returns_cached_field(threehour_release, tmp_path):
    package = copy_release(threehour_release, tmp_path / "changed_package")
    registry = DatasetRegistry(tmp_path / "missing", earth_3hourly_package_dir=package)
    release = registry.get_earth_snapshot(DATASET_ID)
    service = EarthOverviewService(registry)
    service.get_field(**arguments(release), timestamp="2020-01-01T04:30:00Z")
    with netCDF4.Dataset(package / DATA_FILE, "r+") as ds:
        ds["TO3"][1, 0, 0] = 999.
    with pytest.raises(DatasetRequestError) as error:
        service.get_field(**arguments(release), timestamp="2020-01-01T04:30:00Z")
    assert error.value.code == "dataset_unavailable"


def test_cache_is_bounded_and_copies_outputs():
    cache = OverviewCache(max_bytes=200, max_entries=2)
    cache.put("one", {"value": [1]})
    result = cache.get("one"); result["value"][0] = 7
    assert cache.get("one")["value"] == [1]
    cache.put("two", {"value": [2]}); cache.put("three", {"value": [3]})
    assert cache.get("one") is None
    cache.put("large", {"value": "x" * 300})
    assert cache.get("large") is None
    assert cache._bytes <= 200 and len(cache._entries) <= 2


def test_timestamp_http_field_response_stays_small(client):
    connection, fingerprint = client
    response = connection.get(PREFIX + "/field", params={"variable": "TO3", "expected_fingerprint": fingerprint,
                                                        "timestamp": "2020-01-01T04:30:00Z"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_grid_shape"] == [240, 480] and body["grid_shape"] == [60, 120]
    assert body["timestamp"] == "2020-01-01T04:30:00Z"
    assert len(response.content) < 160_000
    assert "frequency_hours" in body and body["time_zone"] == "UTC"
    assert body["render_method"] == "spherical_cell_area_mean_4x4"


def test_http_missing_values_roundtrip_as_null(client):
    connection, fingerprint = client
    response = connection.get(PREFIX + "/field", params={"variable": "TO3", "expected_fingerprint": fingerprint,
                                                        "timestamp": "2020-01-01T01:30:00Z"})
    assert response.status_code == 200
    assert any(value is None for row in response.json()["field"] for value in row)


@pytest.mark.parametrize("selection,code", [
    ({"date": "2020-01-01"}, "invalid_timestamp"),
    ({"timestamp": "2020-01-01T02:30:00Z"}, "timestamp_out_of_range"),
    ({"timestamp": "2020-01-01T04:30:00Z", "render_stride": 1}, "invalid_render_resolution"),
])
def test_http_strict_time_and_preview_resolution(client, selection, code):
    connection, fingerprint = client
    response = connection.get(PREFIX + "/field", params={"variable": "TO3", "expected_fingerprint": fingerprint, **selection})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == code


def test_http_point_and_regional_keep_threehour_timestamps(client):
    connection, fingerprint = client
    common = {"variable": "TO3", "expected_fingerprint": fingerprint,
              "start": "2020-01-01T01:30:00Z", "end": "2020-01-01T07:30:00Z"}
    point = connection.get(PREFIX + "/point-series", params={**common, "lat": .375, "lon": .375})
    regional = connection.get(PREFIX + "/regional-series", params=common)
    assert point.status_code == regional.status_code == 200
    assert point.json()["timestamps"] == regional.json()["timestamps"]
    assert len(point.json()["timestamps"]) == 3
    assert point.json()["grid_point"]["lat_index"] == 120


def test_daily_responses_keep_date_only_and_no_added_null_fields(earth_global_release):
    registry = DatasetRegistry(**earth_global_release)
    dataset_id = "earth_merra2_daily_v2"
    fingerprint = registry.get_dataset(dataset_id)["dataset_fingerprint"]
    app = FastAPI(); app.state.earth_overview_service = EarthOverviewService(registry)
    app.include_router(router, prefix="/api")
    with TestClient(app) as connection:
        response = connection.get(f"/api/datasets/{dataset_id}/overview/field", params={
            "date": "2020-01-01", "variable": "TO3", "expected_fingerprint": fingerprint})
    assert response.status_code == 200
    body = response.json()
    assert body["date"] == "2020-01-01" and len(body["field"]) == 36
    assert "timestamp" not in body and "render" not in body and "frequency_hours" not in body
