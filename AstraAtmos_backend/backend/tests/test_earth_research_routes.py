"""HTTP contract for the Earth scientific analysis endpoints."""

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routers.earth_analysis import router  # noqa: E402
from services.dataset_registry import DatasetRegistry  # noqa: E402
from services.earth_research_service import EarthResearchService  # noqa: E402

DATASET_ID = "earth_merra2_daily_v2"
PREFIX = "/api/analysis/earth/overview"
FINGERPRINT = "0" * 64
CHANNELS = ("TO3", "U10M", "V10M", "T2M", "SWGDN")
UNITS = ("DU", "m s-1", "m s-1", "K", "W m-2")


def build_app(release, *, with_service=True) -> FastAPI:
    registry = DatasetRegistry(**release, earth_dataset_id=DATASET_ID)
    app = FastAPI()
    app.state.dataset_registry = registry
    if with_service:
        app.state.earth_research_service = EarthResearchService(registry)
    app.include_router(router, prefix="/api")
    return app


@pytest.fixture
def earth_research_client(earth_global_release):
    app = build_app(earth_global_release)
    fingerprint = app.state.dataset_registry.get_dataset(DATASET_ID)["dataset_fingerprint"]
    with TestClient(app) as client:
        yield client, fingerprint


def params(fingerprint, **overrides):
    base = {"dataset_id": DATASET_ID, "expected_fingerprint": fingerprint}
    base.update(overrides)
    return base


def insight_body(fingerprint=None, **overrides):
    body = {
        "planet": "earth",
        "dataset_id": DATASET_ID,
        "year": 2020,
        "variable": "TO3",
        "date": "2020-06-15",
        "scope": "global",
        "question": "What does this series show?",
        "locale": "en",
        "summary": {
            "units": "DU",
            "cards": [{"card": "seasonal", "values": {"peak_month": 4}, "notes": ["client note"]}],
        },
    }
    if fingerprint is not None:
        body["expected_fingerprint"] = fingerprint
    body.update(overrides)
    return body


# ── success contracts ──────────────────────────────────────────────────

def test_context_reports_global_v2_geometry(earth_research_client):
    client, fingerprint = earth_research_client
    response = client.get(PREFIX + "/context", params={"expected_fingerprint": fingerprint})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["planet"] == "earth"
    assert body["dataset_id"] == DATASET_ID
    assert body["dataset_version"] == "v2"
    assert body["dataset_fingerprint"] == fingerprint
    assert body["schema"] == "astraatmos_earth_daily_v1"
    assert body["source_meta"]["source"] == "NASA MERRA-2"
    assert body["source_meta"]["cadence"] == "daily mean"
    assert body["time"]["kind"] == "iso-date"
    assert body["time"]["step_unit"] == "day"
    assert body["time"]["years"] == [2020, 2021]

    geometry = body["geometry"]
    assert geometry["shape"] == [36, 72]
    assert geometry["dimension_order"] == ["lat", "lon"]
    assert geometry["lat_bounds"] == [-90.0, 90.0]
    assert geometry["lon_bounds"] == [-180.0, 180.0]
    assert geometry["wrap_longitude"] is True
    assert geometry["coverage"] == "global"
    assert len(geometry["lat_centers"]) == 36
    assert len(geometry["lon_centers"]) == 72
    assert geometry["cell_bounds"]["latitude"] == [-90.0, 90.0]
    assert geometry["cell_bounds"]["longitude"] == [-180.0, 180.0]

    assert [item["id"] for item in body["variables"]] == list(CHANNELS)
    assert [item["units"] for item in body["variables"]] == list(UNITS)
    for item in body["variables"]:
        assert item["color_range"]["scope"] == "dataset"
        assert item["color_range"]["centered_on_zero"] is (item["id"] in ("U10M", "V10M"))


def test_context_omits_or_defaults_the_fingerprint(earth_research_client):
    client, fingerprint = earth_research_client
    defaulted = client.get(PREFIX + "/context", params={"dataset_id": DATASET_ID})
    explicit = client.get(PREFIX + "/context", params={"expected_fingerprint": fingerprint})
    assert defaulted.status_code == explicit.status_code == 200
    assert defaulted.json() == explicit.json()

    no_arguments = client.get(PREFIX + "/context")
    assert no_arguments.status_code == 200
    assert no_arguments.json()["dataset_id"] == DATASET_ID


def test_context_reports_diurnal_as_unavailable_with_a_reason(earth_research_client):
    client, fingerprint = earth_research_client
    body = client.get(PREFIX + "/context", params=params(fingerprint)).json()
    assert body["capabilities"]["diurnal"] is False
    assert body["unavailable"] == {"diurnal": "daily_data_has_no_diurnal_samples"}
    assert body["capabilities"]["polar"] is True
    assert body["capabilities"]["researchSuite"] is True
    assert body["capabilities"]["spatialDiagnostics"] is True
    assert body["capabilities"]["aiInsight"] is True


def test_context_polar_scope_uses_real_sampled_latitudes(earth_research_client):
    client, fingerprint = earth_research_client
    body = client.get(PREFIX + "/context", params=params(fingerprint)).json()
    scope = body["polar_scope"]
    assert scope["min_abs_latitude"] == 60
    assert scope["sampling"] == "daily_mean"
    assert scope["note"] == "diurnal_variation_not_resolvable"
    assert [item["id"] for item in scope["bands"]] == ["south_polar", "north_polar"]
    for band in scope["bands"]:
        assert band["grid_point_count"] == 6
        assert len(band["latitude_values"]) == 6
        assert abs(band["min_latitude"]) >= 60 and abs(band["max_latitude"]) >= 60


@pytest.mark.parametrize("year,day_count", [(2020, 366), (2021, 365)])
def test_research_suite_returns_continuous_real_dates(earth_research_client, year, day_count):
    client, fingerprint = earth_research_client
    response = client.get(
        PREFIX + "/research-suite", params=params(fingerprint, year=year)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["planet"] == "earth"
    assert body["dataset_id"] == DATASET_ID
    assert body["dataset_fingerprint"] == fingerprint
    assert body["year"] == year
    assert body["day_count"] == day_count == len(body["dates"])
    assert body["start"] == f"{year}-01-01"
    assert body["end"] == f"{year}-12-31"
    if year == 2020:
        assert "2020-02-29" in body["dates"]

    for variable in CHANNELS:
        assert len(body["regional_series"][variable]) == day_count
        assert len(body["seasonal"][variable]["z"]) == 36
        for band_id in ("south_polar", "tropics", "north_polar"):
            assert len(body["band_series"][band_id][variable]) == day_count
    assert set(body["relationships"]) == {
        "global", "south_polar", "south_mid", "tropics", "north_mid", "north_polar",
    }
    global_scope = body["relationships"]["global"]["scope"]
    assert global_scope["kind"] == "global"
    assert global_scope["band_id"] is None
    assert global_scope["units_note"]
    assert body["relationships"]["tropics"]["scope"]["kind"] == "band"
    assert body["relationships"]["tropics"]["scope"]["band_id"] == "tropics"


def test_research_suite_json_size_is_measured(earth_research_client):
    """The 2020 suite payload size is part of the delivered contract."""
    client, fingerprint = earth_research_client
    response = client.get(PREFIX + "/research-suite", params=params(fingerprint, year=2020))
    assert response.status_code == 200
    # The transport encoding is what the browser receives. Measured here so the
    # payload size is a checked contract rather than a claim in a report.
    print(f"research-suite 2020 JSON bytes: {len(response.content)}")
    assert 100_000 < len(response.content) < 8_000_000
    assert len(json.dumps(response.json())) > len(response.content) // 2


@pytest.mark.parametrize("variable,units", list(zip(CHANNELS, UNITS)))
def test_spatial_diagnostics_serves_every_variable(earth_research_client, variable, units):
    client, fingerprint = earth_research_client
    response = client.get(
        PREFIX + "/spatial-diagnostics",
        params=params(fingerprint, year=2020, variable=variable),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["variable"] == variable
    assert body["units"] == units
    assert body["year"] == 2020
    assert body["reference"] == "annual_mean_minus_equal_longitude_mean"
    assert len(body["anomaly"]) == 36
    assert all(len(row) == 72 for row in body["anomaly"])
    assert body["color_range"]["centered_on_zero"] is True
    assert body["color_range"]["min"] == pytest.approx(-body["color_range"]["max"])
    for band in body["bands"]:
        assert band["grid_point_count"] == (12 if band["id"] == "tropics" else 6)
        assert band["rms"] is not None and band["rms"] >= 0.0


def test_spatial_diagnostics_anomaly_is_zonal_and_matches_an_independent_mean(
    earth_research_client, earth_global_release
):
    import numpy as np

    client, fingerprint = earth_research_client
    body = client.get(
        PREFIX + "/spatial-diagnostics",
        params=params(fingerprint, year=2020, variable="TO3"),
    ).json()
    anomaly = np.asarray(body["anomaly"], dtype="float64")
    # Every row deviates from its own equal-longitude mean.
    assert np.allclose(anomaly.sum(axis=1), 0.0, atol=1e-9)

    registry = DatasetRegistry(**earth_global_release, earth_dataset_id=DATASET_ID)
    release = registry.get_earth_overview_snapshot(DATASET_ID, fingerprint)
    field = np.asarray(release.fields["TO3"], dtype="float64")[:366]
    annual = field.mean(axis=0)
    expected = annual - annual.mean(axis=1, keepdims=True)
    assert np.allclose(anomaly, expected, rtol=1e-9, atol=1e-9)


def test_research_suite_regional_series_matches_an_independent_area_mean(
    earth_research_client, earth_global_release
):
    import numpy as np

    client, fingerprint = earth_research_client
    body = client.get(
        PREFIX + "/research-suite", params=params(fingerprint, year=2020)
    ).json()

    registry = DatasetRegistry(**earth_global_release, earth_dataset_id=DATASET_ID)
    release = registry.get_earth_overview_snapshot(DATASET_ID, fingerprint)
    latitude = np.asarray(release.latitude, dtype="float64")
    edges = np.linspace(-90.0, 90.0, latitude.size + 1)
    weights = np.diff(np.sin(np.deg2rad(edges)))
    # Independent recomputation, not the service helper.
    for variable in CHANNELS:
        field = np.asarray(release.fields[variable], dtype="float64")[:366]
        expected = np.einsum("tij,i->t", field, weights) / (field.shape[2] * weights.sum())
        assert np.allclose(
            body["regional_series"][variable], expected, rtol=1e-9, atol=1e-9
        )


def test_polar_dynamics_reports_six_plus_six_rows(earth_research_client):
    client, fingerprint = earth_research_client
    response = client.get(PREFIX + "/polar-dynamics", params=params(fingerprint, year=2020))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["planet"] == "earth"
    assert [item["id"] for item in body["bands"]] == ["south_polar", "north_polar"]
    assert [item["hemisphere"] for item in body["bands"]] == ["south", "north"]
    assert [item["grid_point_count"] for item in body["bands"]] == [6, 6]
    assert body["polar_scope"]["note"] == "diurnal_variation_not_resolvable"
    assert body["day_count"] == 366
    for band in body["bands"]:
        assert band["days"] == 366
        for variable, units in zip(CHANNELS, UNITS):
            metrics = band["variables"][variable]
            assert metrics["units"] == units
            assert len(metrics["series"]) == 366
            assert metrics["reason"] is None
    for variable, units in zip(CHANNELS, UNITS):
        contrast = body["hemisphere_contrast"][variable]
        assert set(contrast) == {"n", "r", "reason", "mean_difference"}
    assert body["hemisphere_contrast"]["TO3"]["n"] == 366


# ── insight ────────────────────────────────────────────────────────────

def test_insight_fallback_answers_without_a_copilot_service(earth_research_client):
    client, fingerprint = earth_research_client
    response = client.post(PREFIX + "/insight", json=insight_body(fingerprint))
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["planet"] == "earth"
    assert body["dataset_id"] == DATASET_ID
    assert body["dataset_version"] == "v2"
    assert body["dataset_fingerprint"] == fingerprint
    assert body["source"] == "NASA MERRA-2"
    assert body["year"] == 2020
    assert body["date_range"] == {"start": "2020-01-01", "end": "2020-12-31"}
    assert body["variable"] == "TO3"
    assert body["units"] == "DU"
    assert body["scope"] == "global"
    assert body["time_kind"] == "iso-date"
    assert body["calendar"] == "proleptic_gregorian"
    assert body["model"] == "builtin-digest"
    assert body["generated_at"]

    answer = body["answer"]
    assert "Earth" in answer
    assert "NASA MERRA-2" in answer
    assert "daily mean" in answer
    assert "2020-01-01" in answer and "2020-12-31" in answer
    assert "TO3" in answer and "DU" in answer
    for forbidden in ("Ls", "MY", "Mars", "mars", "dust"):
        assert forbidden not in answer, forbidden

    limitations = " ".join(body["limitations"]).lower()
    assert "utc daily means" in limitations
    assert "diurnal" in limitations
    assert "global 5 degree grid" in limitations
    assert "associations" in limitations and "causation" in limitations

    digest = body["digest"]
    assert digest["variable"] == "TO3"
    assert digest["units"] == "DU"
    assert digest["date_range"] == body["date_range"]
    cards = {card["card"]: card for card in digest["cards"]}
    assert "series" in cards
    assert cards["series"]["values"]["mean"] is not None
    # Client notes are merged as text only; client numbers never become data.
    assert any(card["notes"] == ["client note"] for card in digest["cards"])


def test_insight_accepts_a_band_scope_and_omitted_fingerprint(earth_research_client):
    client, _fingerprint = earth_research_client
    response = client.post(
        PREFIX + "/insight",
        json=insight_body(scope="north_polar", year=2021, variable="SWGDN", date="2021-06-15"),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scope"] == "north_polar"
    assert body["year"] == 2021
    assert body["variable"] == "SWGDN"
    assert body["units"] == "W m-2"
    assert body["digest"]["scope_label"]


def test_insight_ignores_a_copilot_service_without_an_api_key(earth_research_client, monkeypatch):
    """A configured-but-keyless Copilot must not be called; the fallback answers."""
    client, fingerprint = earth_research_client
    calls = []

    class KeylessCopilot:
        api_key = ""
        model = "some-model"

        async def chat(self, question, context=None, history=None):
            calls.append(question)
            return "should not be used"

    client.app.state.copilot_service = KeylessCopilot()
    response = client.post(PREFIX + "/insight", json=insight_body(fingerprint))
    assert response.status_code == 200
    assert calls == []
    assert response.json()["model"] == "builtin-digest"


def test_insight_body_carries_the_documented_fields(earth_research_client):
    client, fingerprint = earth_research_client
    body = client.post(PREFIX + "/insight", json=insight_body(fingerprint)).json()
    assert set(body) == {
        "planet", "dataset_id", "dataset_version", "dataset_fingerprint", "schema",
        "source", "year", "date_range", "variable", "units", "scope", "scope_label",
        "time_kind", "calendar", "model", "answer", "digest", "limitations", "generated_at",
    }


# ── error protocol ─────────────────────────────────────────────────────

def test_stale_fingerprint_returns_409_version_changed(earth_research_client):
    client, _fingerprint = earth_research_client
    for path, extra in (
        ("/context", {}),
        ("/research-suite", {"year": 2020}),
        ("/spatial-diagnostics", {"year": 2020, "variable": "TO3"}),
        ("/polar-dynamics", {"year": 2020}),
    ):
        response = client.get(
            PREFIX + path, params=params(FINGERPRINT, **extra)
        )
        assert response.status_code == 409, path
        assert response.json()["detail"]["code"] == "dataset_version_changed"
    response = client.post(PREFIX + "/insight", json=insight_body(FINGERPRINT))
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_version_changed"


@pytest.mark.parametrize("dataset_id", ["openmars_mcd", "mcd_overview"])
def test_mars_datasets_are_rejected_with_409(earth_research_client, dataset_id):
    client, fingerprint = earth_research_client
    for path, extra in (
        ("/context", {}),
        ("/research-suite", {"year": 2020}),
        ("/spatial-diagnostics", {"year": 2020, "variable": "TO3"}),
        ("/polar-dynamics", {"year": 2020}),
    ):
        response = client.get(
            PREFIX + path,
            params={"dataset_id": dataset_id, "expected_fingerprint": fingerprint, **extra},
        )
        assert response.status_code == 409, path
        assert response.json()["detail"]["code"] == "dataset_overview_not_supported"
    response = client.post(
        PREFIX + "/insight", json=insight_body(fingerprint, dataset_id=dataset_id)
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_overview_not_supported"


def test_unknown_dataset_returns_404(earth_research_client):
    client, fingerprint = earth_research_client
    response = client.get(
        PREFIX + "/context",
        params={"dataset_id": "missing", "expected_fingerprint": fingerprint},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "unknown_dataset"
    response = client.get(
        PREFIX + "/research-suite",
        params={"dataset_id": "missing", "expected_fingerprint": fingerprint, "year": 2020},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("fingerprint", ["abc", "A" * 64, "z" * 64, "", "0" * 63])
def test_malformed_fingerprint_is_a_validation_error(earth_research_client, fingerprint):
    client, _fingerprint = earth_research_client
    response = client.get(
        PREFIX + "/context", params={"dataset_id": DATASET_ID, "expected_fingerprint": fingerprint}
    )
    assert response.status_code == 422


def test_missing_required_query_parameters_are_validation_errors(earth_research_client):
    client, fingerprint = earth_research_client
    assert client.get(
        PREFIX + "/research-suite", params={"dataset_id": DATASET_ID}
    ).status_code == 422
    assert client.get(
        PREFIX + "/research-suite",
        params={"expected_fingerprint": fingerprint, "year": 2020},
    ).status_code == 422
    assert client.get(
        PREFIX + "/spatial-diagnostics",
        params={"dataset_id": DATASET_ID, "year": 2020},
    ).status_code == 422
    assert client.get(
        PREFIX + "/polar-dynamics", params={"dataset_id": DATASET_ID}
    ).status_code == 422
    assert client.get(PREFIX + "/research-suite", params={"dataset_id": DATASET_ID, "year": "x"}).status_code == 422


def test_missing_dataset_returns_503_with_a_safe_reason(tmp_path):
    app = FastAPI()
    registry = DatasetRegistry(tmp_path / "absent")
    app.state.dataset_registry = registry
    app.state.earth_research_service = EarthResearchService(registry)
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        response = client.get(
            PREFIX + "/context", params={"dataset_id": DATASET_ID}
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "dataset_unavailable"
    assert detail["availability_reason"] == "package_missing"
    assert str(tmp_path) not in response.text


def test_year_outside_the_release_returns_422(earth_research_client):
    client, fingerprint = earth_research_client
    for year in (2019, 2022):
        response = client.get(
            PREFIX + "/research-suite", params=params(fingerprint, year=year)
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "year_out_of_range"
    response = client.get(
        PREFIX + "/polar-dynamics", params=params(fingerprint, year=2019)
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "year_out_of_range"


@pytest.mark.parametrize("variable", ["O3", "to3", "wind", "temperature"])
def test_unsupported_variable_returns_422(earth_research_client, variable):
    client, fingerprint = earth_research_client
    response = client.get(
        PREFIX + "/spatial-diagnostics",
        params=params(fingerprint, year=2020, variable=variable),
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_variable"


def test_insight_rejects_a_mars_planet(earth_research_client):
    client, fingerprint = earth_research_client
    response = client.post(PREFIX + "/insight", json=insight_body(fingerprint, planet="mars"))
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "planet_not_supported"


def test_insight_rejects_a_non_earth_planet_other_than_mars(earth_research_client):
    client, fingerprint = earth_research_client
    for planet in ("venus", "EARTH", "earth "):
        response = client.post(PREFIX + "/insight", json=insight_body(fingerprint, planet=planet))
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "planet_not_supported"


@pytest.mark.parametrize("size", [4001, 20_000])
def test_insight_rejects_an_oversized_summary_list(earth_research_client, size):
    client, fingerprint = earth_research_client
    body = insight_body(fingerprint)
    body["summary"]["cards"][0]["values"]["blob"] = list(range(size))
    response = client.post(PREFIX + "/insight", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "insight_payload_too_large"


def test_insight_rejects_a_deeply_nested_summary(earth_research_client):
    client, fingerprint = earth_research_client
    body = insight_body(fingerprint)
    nested = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    body["summary"]["cards"][0]["values"]["deep"] = nested
    response = client.post(PREFIX + "/insight", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "insight_payload_too_large"


def test_insight_rejects_unknown_request_fields(earth_research_client):
    client, fingerprint = earth_research_client
    body = insight_body(fingerprint)
    body["field"] = [[1.0] * 72] * 36
    response = client.post(PREFIX + "/insight", json=body)
    assert response.status_code == 422

    body = insight_body(fingerprint)
    body["summary"]["cards"][0]["unexpected"] = 1
    assert client.post(PREFIX + "/insight", json=body).status_code == 422


def test_insight_rejects_a_bad_scope_and_variable(earth_research_client):
    client, fingerprint = earth_research_client
    response = client.post(
        PREFIX + "/insight", json=insight_body(fingerprint, scope="antarctica")
    )
    assert response.status_code == 422
    response = client.post(
        PREFIX + "/insight", json=insight_body(fingerprint, variable="O3")
    )
    assert response.status_code == 422
    response = client.post(
        PREFIX + "/insight", json=insight_body(fingerprint, expected_fingerprint="abc")
    )
    assert response.status_code == 422


def test_error_detail_shape_is_consistent(earth_research_client):
    client, fingerprint = earth_research_client
    response = client.get(
        PREFIX + "/research-suite", params=params(fingerprint, year=2019)
    )
    detail = response.json()["detail"]
    assert set(detail) == {"code", "message"}
    assert isinstance(detail["message"], str) and detail["message"]
    json.dumps(detail)


# ── hygiene ────────────────────────────────────────────────────────────

def test_responses_never_leak_paths_or_data_file_names(earth_research_client, earth_global_release):
    client, fingerprint = earth_research_client
    package_dir = str(Path(earth_global_release["earth_package_dir"]))
    responses = [
        client.get(PREFIX + "/context", params=params(fingerprint)),
        client.get(PREFIX + "/research-suite", params=params(fingerprint, year=2020)),
        client.get(PREFIX + "/spatial-diagnostics", params=params(fingerprint, year=2020, variable="TO3")),
        client.get(PREFIX + "/polar-dynamics", params=params(fingerprint, year=2020)),
        client.post(PREFIX + "/insight", json=insight_body(fingerprint)),
        client.get(PREFIX + "/research-suite", params=params(fingerprint, year=2019)),
        client.get(PREFIX + "/context", params={"dataset_id": "openmars_mcd"}),
    ]
    for response in responses:
        assert package_dir not in response.text
        assert ".nc" not in response.text
        assert "Traceback" not in response.text
        assert "manifest.json" not in response.text


def test_openapi_documents_all_five_routes(earth_global_release):
    app = build_app(earth_global_release)
    schema = app.openapi()
    documented = sorted(path for path in schema["paths"] if "analysis/earth/overview" in path)
    assert documented == [
        "/api/analysis/earth/overview/context",
        "/api/analysis/earth/overview/insight",
        "/api/analysis/earth/overview/polar-dynamics",
        "/api/analysis/earth/overview/research-suite",
        "/api/analysis/earth/overview/spatial-diagnostics",
    ]
    assert schema["paths"]["/api/analysis/earth/overview/insight"]["post"]
    assert schema["paths"]["/api/analysis/earth/overview/context"]["get"]
    for name in (
        "EarthContextResponse",
        "EarthResearchSuiteResponse",
        "EarthSpatialDiagnosticsResponse",
        "EarthPolarDynamicsResponse",
        "EarthInsightRequest",
        "EarthInsightResponse",
    ):
        assert name in schema["components"]["schemas"], name


def test_analysis_routes_do_not_require_authentication(earth_research_client):
    client, fingerprint = earth_research_client
    assert client.get(PREFIX + "/context", params=params(fingerprint)).status_code == 200
    assert client.post(PREFIX + "/insight", json=insight_body(fingerprint)).status_code == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
