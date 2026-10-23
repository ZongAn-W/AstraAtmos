"""HTTP contract for the read-only 2D Earth overview endpoints."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routers.earth_overview import router  # noqa: E402
from services.dataset_registry import DatasetRegistry  # noqa: E402
from services.earth_dataset_metadata import DATA_FILE_NAME  # noqa: E402
from services.earth_overview_service import EarthOverviewService  # noqa: E402

DATASET_ID = "earth_merra2_daily_v1"
PREFIX = f"/api/datasets/{DATASET_ID}/overview"


@pytest.fixture
def earth_client(earth_spatial_release):
    registry = DatasetRegistry(**earth_spatial_release)
    fingerprint = registry.get_dataset(DATASET_ID)["dataset_fingerprint"]
    app = FastAPI()
    app.state.dataset_registry = registry
    app.state.earth_overview_service = EarthOverviewService(registry)
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        yield client, fingerprint


def params(fingerprint, **overrides):
    base = {"variable": "TO3", "expected_fingerprint": fingerprint}
    base.update(overrides)
    return base


# ── success contracts ──────────────────────────────────────────────────

def test_field_and_series_agree(earth_client):
    client, fingerprint = earth_client
    common = params(fingerprint)
    field = client.get(PREFIX + "/field", params={**common, "date": "2020-02-29"})
    point = client.get(PREFIX + "/point-series", params={**common, "lat": 0, "lon": 0})
    region = client.get(PREFIX + "/regional-series", params=common)
    assert field.status_code == point.status_code == region.status_code == 200
    f, p, r = field.json(), point.json(), region.json()
    index = p["dates"].index(f["date"])
    row, col = p["grid_point"]["lat_index"], p["grid_point"]["lon_index"]
    assert f["field"][row][col] == p["values"][index]
    assert f["statistics"]["regional_mean"] == pytest.approx(r["values"][index], rel=1e-12)
    assert f["units"] == p["units"] == r["units"] == "DU"


def test_field_response_shape_and_identity(earth_client):
    client, fingerprint = earth_client
    response = client.get(PREFIX + "/field", params=params(fingerprint, date="2020-01-01"))
    assert response.status_code == 200
    body = response.json()

    assert body["dataset_id"] == DATASET_ID
    assert body["dataset_version"] == "v1"
    assert body["dataset_fingerprint"] == fingerprint
    assert body["planet"] == "earth"
    assert body["variable"] == "TO3"
    assert body["units"] == "DU"
    assert body["date"] == "2020-01-01"
    assert body["calendar"] == "proleptic_gregorian"
    assert body["dimension_order"] == ["lat", "lon"]
    assert len(body["lat"]) == 31 and len(body["lon"]) == 49
    assert len(body["field"]) == 31 and all(len(row) == 49 for row in body["field"])
    assert body["lat"][0] == -60.0 and body["lat"][-1] == 60.0
    assert body["lon"][0] == -120.0 and body["lon"][-1] == 120.0
    assert body["field"][0][0] == pytest.approx(0.0)
    assert body["coverage"] == {
        "latitude_range": [-60.0, 60.0],
        "longitude_range": [-120.0, 120.0],
        "wrap_longitude": False,
    }
    assert body["color_range"]["scope"] == "dataset"
    assert body["statistics"]["valid_count"] == 1519


@pytest.mark.parametrize("variable,units", [
    ("TO3", "DU"), ("U10M", "m s-1"), ("V10M", "m s-1"), ("T2M", "K"), ("SWGDN", "W m-2"),
])
def test_every_variable_is_served_with_original_units(earth_client, variable, units):
    client, fingerprint = earth_client
    response = client.get(
        PREFIX + "/field", params=params(fingerprint, date="2020-06-15", variable=variable)
    )
    assert response.status_code == 200, response.text
    assert response.json()["units"] == units


def test_regional_series_defaults_to_the_full_period(earth_client):
    client, fingerprint = earth_client
    response = client.get(PREFIX + "/regional-series", params=params(fingerprint))
    assert response.status_code == 200
    body = response.json()
    assert body["aggregation"] == "cos_lat_sample_mean"
    assert body["start"] == "2020-01-01" and body["end"] == "2021-12-31"
    assert len(body["dates"]) == len(body["values"]) == 731
    assert body["coverage"]["longitude_range"] == [-120.0, 120.0]


def test_regional_series_accepts_a_window(earth_client):
    client, fingerprint = earth_client
    response = client.get(
        PREFIX + "/regional-series",
        params=params(fingerprint, start="2020-02-28", end="2020-03-01"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dates"] == ["2020-02-28", "2020-02-29", "2020-03-01"]
    assert body["start"] == "2020-02-28" and body["end"] == "2020-03-01"


def test_point_series_snaps_to_the_nearest_grid_point(earth_client):
    client, fingerprint = earth_client
    response = client.get(
        PREFIX + "/point-series", params=params(fingerprint, lat=1.7, lon=-2.2)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["selection"] == "nearest_grid_point"
    assert body["requested"] == {"lat": 1.7, "lon": -2.2}
    assert body["grid_point"]["lat"] == 0.0
    assert body["grid_point"]["lon"] == 0.0
    assert body["grid_point"]["lat_index"] == 15
    assert body["grid_point"]["lon_index"] == 24
    assert len(body["dates"]) == len(body["values"]) == 731


def test_responses_never_leak_server_paths(earth_client, earth_spatial_release):
    client, fingerprint = earth_client
    package_dir = str(Path(earth_spatial_release["earth_package_dir"]))
    for response in (
        client.get(PREFIX + "/field", params=params(fingerprint, date="2020-01-01")),
        client.get(PREFIX + "/regional-series", params=params(fingerprint)),
        client.get(PREFIX + "/point-series", params=params(fingerprint, lat=0, lon=0)),
        client.get("/api/datasets"),
    ):
        assert package_dir not in response.text
        assert ".nc" not in response.text


def test_openapi_documents_all_three_overview_routes(earth_spatial_release):
    registry = DatasetRegistry(**earth_spatial_release)
    app = FastAPI()
    app.state.dataset_registry = registry
    app.state.earth_overview_service = EarthOverviewService(registry)
    app.include_router(router, prefix="/api")

    schema = app.openapi()
    for suffix in ("/field", "/regional-series", "/point-series"):
        # The documented path keeps the dataset id as a template parameter.
        assert f"/api/datasets/{{dataset_id}}/overview{suffix}" in schema["paths"]
    assert "EarthFieldResponse" in schema["components"]["schemas"]
    assert "EarthRegionalSeriesResponse" in schema["components"]["schemas"]
    assert "EarthPointSeriesResponse" in schema["components"]["schemas"]


# ── error protocol ─────────────────────────────────────────────────────

def test_unknown_dataset_id_returns_404(earth_client):
    client, fingerprint = earth_client
    response = client.get(
        "/api/datasets/missing/overview/field",
        params={"variable": "TO3", "expected_fingerprint": fingerprint, "date": "2020-01-01"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "unknown_dataset"


@pytest.mark.parametrize("dataset_id", ["openmars_mcd", "mcd_overview"])
def test_mars_datasets_have_no_overview(earth_client, dataset_id):
    client, fingerprint = earth_client
    response = client.get(
        f"/api/datasets/{dataset_id}/overview/regional-series",
        params={"variable": "TO3", "expected_fingerprint": fingerprint},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_overview_not_supported"


def test_stale_fingerprint_returns_409_version_changed(earth_client):
    client, _fingerprint = earth_client
    response = client.get(
        PREFIX + "/field",
        params={"variable": "TO3", "expected_fingerprint": "0" * 64, "date": "2020-01-01"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_version_changed"


@pytest.mark.parametrize("fingerprint", ["abc", "A" * 64, "z" * 64, ""])
def test_malformed_fingerprint_is_a_validation_error(earth_client, fingerprint):
    client, _fingerprint = earth_client
    response = client.get(
        PREFIX + "/field",
        params={"variable": "TO3", "expected_fingerprint": fingerprint, "date": "2020-01-01"},
    )
    assert response.status_code == 422


def test_missing_package_returns_503_with_a_safe_reason(tmp_path):
    app = FastAPI()
    registry = DatasetRegistry(tmp_path / "absent")
    app.state.dataset_registry = registry
    app.state.earth_overview_service = EarthOverviewService(registry)
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        response = client.get(
            PREFIX + "/field",
            params={"variable": "TO3", "expected_fingerprint": "a" * 64, "date": "2020-01-01"},
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "dataset_unavailable"
    assert detail["availability_reason"] == "package_missing"
    assert str(tmp_path) not in response.text


def test_corrupted_package_returns_503_with_the_verification_reason(earth_spatial_release, tmp_path):
    destination = tmp_path / "broken"
    destination.mkdir()
    for name in (DATA_FILE_NAME, "manifest.json"):
        shutil.copy2(Path(earth_spatial_release["earth_package_dir"]) / name, destination / name)
    data_path = destination / DATA_FILE_NAME
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 0xFF
    data_path.write_bytes(bytes(payload))

    registry = DatasetRegistry(
        destination,
        expected_manifest_sha256=earth_spatial_release["expected_manifest_sha256"],
        expected_data_sha256=earth_spatial_release["expected_data_sha256"],
    )
    app = FastAPI()
    app.state.dataset_registry = registry
    app.state.earth_overview_service = EarthOverviewService(registry)
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        response = client.get(
            PREFIX + "/field",
            params={"variable": "TO3", "expected_fingerprint": "a" * 64, "date": "2020-01-01"},
        )
    assert response.status_code == 503
    assert response.json()["detail"]["availability_reason"] == "data_fingerprint_mismatch"


@pytest.mark.parametrize("date,code", [
    ("2019-12-31", "date_out_of_range"),
    ("2022-01-01", "date_out_of_range"),
    ("2021-02-29", "date_out_of_range"),
    ("2020-13-01", "invalid_date"),
    ("2020-1-1", "invalid_date"),
    ("not-a-date", "invalid_date"),
])
def test_bad_dates_return_422_with_stable_codes(earth_client, date, code):
    client, fingerprint = earth_client
    response = client.get(PREFIX + "/field", params=params(fingerprint, date=date))
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == code


def test_start_after_end_returns_422(earth_client):
    client, fingerprint = earth_client
    response = client.get(
        PREFIX + "/regional-series",
        params=params(fingerprint, start="2021-01-01", end="2020-01-01"),
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_date_range"


@pytest.mark.parametrize("variable", ["wind", "O3", "temperature", "to3", "U10"])
def test_unsupported_variable_returns_422(earth_client, variable):
    client, fingerprint = earth_client
    response = client.get(
        PREFIX + "/field", params=params(fingerprint, date="2020-01-01", variable=variable)
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_variable"


@pytest.mark.parametrize("lat,lon", [
    (60.5, 0), (-61, 0), (0, 121), (0, -121), (0, 240), (90, 0), (-90, 180),
])
def test_points_outside_coverage_return_422_without_wrapping(earth_client, lat, lon):
    client, fingerprint = earth_client
    response = client.get(
        PREFIX + "/point-series", params=params(fingerprint, lat=lat, lon=lon)
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "point_outside_coverage"


def test_missing_required_parameters_are_validation_errors(earth_client):
    client, fingerprint = earth_client
    assert client.get(PREFIX + "/field", params={"variable": "TO3"}).status_code == 422
    assert client.get(
        PREFIX + "/field", params={"expected_fingerprint": fingerprint, "date": "2020-01-01"}
    ).status_code == 422
    assert client.get(
        PREFIX + "/point-series", params=params(fingerprint, lat=0)
    ).status_code == 422


def test_error_detail_shape_is_consistent(earth_client):
    client, fingerprint = earth_client
    response = client.get(PREFIX + "/field", params=params(fingerprint, date="2019-12-31"))
    detail = response.json()["detail"]
    assert set(detail) == {"code", "message"}
    assert isinstance(detail["message"], str) and detail["message"]
    json.dumps(detail)


def test_overview_routes_do_not_require_authentication(earth_client):
    client, fingerprint = earth_client
    # Public read-only, matching the first-stage catalog permissions.
    assert client.get(
        PREFIX + "/field", params=params(fingerprint, date="2020-01-01")
    ).status_code == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
