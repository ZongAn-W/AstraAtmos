import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth.dependencies import get_optional_user  # noqa: E402
from config import OVERVIEW_MCD_VARIABLES  # noqa: E402
from routers.analysis import router  # noqa: E402
from services.analysis_service import AnalysisService  # noqa: E402
from services.user_overview_source_service import UserMcdOverviewDataView  # noqa: E402


class FakeOfficialOverview:
    def get_available_years(self):
        return [27]

    def get_ls_range(self, mars_year):
        return (0.0, 360.0)

    def get_ozone_capabilities(self):
        return {
            "openmars": True,
            "nomad": False,
            "diff_pairs": ["MCD-OpenMARS"],
            "coverage": {
                "openmars": {"34": [{"start": 0.0, "end": 90.0}]},
                "nomad": {},
            },
        }

    def get_ozone_overlay_payload(self, mars_year, ls):
        return {
            "mars_year": mars_year,
            "requested_ls": ls,
            "anchor_ls": ls,
            "mcd": {"source": "mcd", "points": [], "minVal": 0.0, "maxVal": 1.0, "ls": ls},
            "openmars": None,
            "nomad": None,
            "available_sources": ["mcd"],
            "diff_candidates": [],
            "validation": {"nomad": None},
            "capabilities": self.get_ozone_capabilities(),
        }


class FakeUserDataService:
    async def get_loaded_dataset(self, upload_id):
        lat = np.array([-2.5, 2.5], dtype=np.float32)
        lon = np.array([0.0, 5.0], dtype=np.float32)
        ls = np.array([10.0, 20.0], dtype=np.float32)
        field = np.ones((2, 2, 2), dtype=np.float32)
        if upload_id == 123:
            return {
                "data_type": "mcd",
                "lat": lat,
                "lon": lon,
                "ls": ls,
                "o3col": field,
                "Temperature": field * 180.0,
                "U_Wind": field,
                "V_Wind": field,
                "Solar_Flux_DN": field,
            }
        if upload_id == 456:
            return {"data_type": "openmars", "lat": lat, "lon": lon, "ls": ls, "o3col": field * 2.0}
        if upload_id == 789:
            return {
                "data_type": "nomad",
                "lat": lat,
                "lon": lon,
                "ls": ls,
                "o3col": field * 3.0,
                "count": np.ones((2, 2, 2), dtype=np.int32),
            }
        raise ValueError("missing upload")

    async def get_loaded_mcd_hourly_dataset(self, upload_id):
        lat = np.array([-45.0, 0.0, 45.0], dtype=np.float32)
        lon = np.array([0.0, 5.0], dtype=np.float32)
        ls = np.array([10.0, 20.0], dtype=np.float32)
        hourly = np.stack([
            np.full((8, 3, 2), 1.0, dtype=np.float32),
            np.full((8, 3, 2), 2.0, dtype=np.float32),
        ], axis=0)
        if upload_id == 123:
            return {
                "data_type": "mcd",
                "lat": lat,
                "lon": lon,
                "ls": ls,
                "O3COL_hourly": hourly,
            }
        raise ValueError("missing upload")


def build_client(monkeypatch):
    import routers.analysis as analysis_module

    async def fake_upload_record(upload_id, current_user, expected_types):
        data_type = {123: "mcd", 456: "openmars", 789: "nomad"}[upload_id]
        assert data_type in expected_types
        return SimpleNamespace(
            id=upload_id,
            user_id=7,
            data_type=data_type,
            mars_year=34,
            filename=f"{data_type}.nc",
            status="valid",
        )

    monkeypatch.setattr(analysis_module, "_get_accessible_upload_record", fake_upload_record)

    app = FastAPI()
    official = FakeOfficialOverview()
    app.state.mcd_overview_service = official
    official_view = UserMcdOverviewDataView(
        upload_id=0,
        mars_year=27,
        data={
            "lat": np.array([-2.5, 2.5], dtype=np.float32),
            "lon": np.array([0.0, 5.0], dtype=np.float32),
            "ls": np.array([10.0, 20.0], dtype=np.float32),
            "o3col": np.ones((2, 2, 2), dtype=np.float32),
            "Temperature": np.ones((2, 2, 2), dtype=np.float32),
            "U_Wind": np.ones((2, 2, 2), dtype=np.float32),
            "V_Wind": np.ones((2, 2, 2), dtype=np.float32),
            "Solar_Flux_DN": np.ones((2, 2, 2), dtype=np.float32),
        },
    )
    app.state.mcd_overview_analysis_service = AnalysisService(official_view, mcd_variables=OVERVIEW_MCD_VARIABLES)
    app.state.user_data_service = FakeUserDataService()
    app.dependency_overrides[get_optional_user] = lambda: SimpleNamespace(id=7, role="user")
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_overview_info_uses_uploaded_mcd(monkeypatch):
    client = build_client(monkeypatch)

    response = client.get("/api/explore/overview/info?mcd_upload_id=123")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available_years"] == [34]
    assert payload["timeline"] == {"min": 10.0, "max": 20.0, "step": 5.0}
    assert payload["source_meta"]["effective_source"] == "user_mcd"
    assert payload["source_meta"]["upload_id"] == 123


def test_overview_info_for_uploaded_mcd_keeps_ozone_coverage_for_timeline_matching(monkeypatch):
    client = build_client(monkeypatch)

    response = client.get("/api/explore/overview/info?mcd_upload_id=123")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ozone_capabilities"]["coverage"]["mcd"]["34"] == [{"start": 10.0, "end": 20.0}]
    assert payload["ozone_capabilities"]["coverage"]["openmars"]["34"] == [{"start": 0.0, "end": 90.0}]


def test_overview_globe_uses_uploaded_mcd(monkeypatch):
    client = build_client(monkeypatch)

    response = client.get("/api/explore/overview/globe?mcd_upload_id=123&ls=20&variable=Temperature")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mars_year"] == 34
    assert payload["variable"] == "Temperature"
    assert payload["source_meta"]["effective_source"] == "user_mcd"


def test_overview_ozone_sources_can_include_uploaded_openmars_and_nomad(monkeypatch):
    client = build_client(monkeypatch)

    response = client.get(
        "/api/explore/overview/ozone-sources?mcd_upload_id=123&openmars_upload_id=456&nomad_upload_id=789&ls=10"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["available_sources"] == ["mcd", "openmars", "nomad"]
    assert payload["mcd"]["source"] == "mcd"
    assert payload["openmars"]["source"] == "openmars"
    assert payload["nomad"]["source"] == "nomad"
    assert payload["validation"]["nomad"]["sample_count"] > 0


def test_overview_diurnal_uses_uploaded_mcd_hourly_ozone(monkeypatch):
    client = build_client(monkeypatch)

    response = client.get("/api/explore/overview/diurnal?mcd_upload_id=123&ls=20&lat_band=Equatorial%20(30S-30N)")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["source_key"] == "O3COL_hourly"
    assert payload["hours"] == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]


def test_legacy_personal_overview_source_is_rejected(monkeypatch):
    client = build_client(monkeypatch)

    response = client.get("/api/explore/overview/info?data_source=personal")

    assert response.status_code == 400
    assert "mcd_upload_id" in response.json()["detail"]
