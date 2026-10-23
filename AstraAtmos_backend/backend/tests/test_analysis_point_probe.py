import sys
from pathlib import Path

import numpy as np
import xarray as xr
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import OVERVIEW_MCD_VARIABLES  # noqa: E402
from routers.analysis import router  # noqa: E402
from services.analysis_service import AnalysisService  # noqa: E402
from services.mcd_overview_data_service import McdOverviewDataService  # noqa: E402


class FakeBaseDataService:
    def get_openmars_data(self, mars_year: int) -> dict:
        raise ValueError(f"MY{mars_year} OpenMARS missing")

    def get_mcd_data(self, mars_year: int) -> dict:
        raise ValueError(f"MY{mars_year} runtime MCD missing")


def write_probe_overview(path: Path):
    lat = np.array([2.5, -2.5], dtype=np.float32)
    lon = np.array([-180.0, -175.0], dtype=np.float32)
    ls = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    ozone = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[11.0, 21.0], [31.0, 41.0]],
            [[12.0, 22.0], [32.0, 42.0]],
        ],
        dtype=np.float32,
    )
    env = np.ones_like(ozone)
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), ls),
            "o3col": (("time", "lat", "lon"), ozone),
            "Pressure": (("time", "lat", "lon"), env),
            "Temperature": (("time", "lat", "lon"), env * 180.0),
            "U_Wind": (("time", "lat", "lon"), env),
            "V_Wind": (("time", "lat", "lon"), env),
            "Dust_Optical_Depth": (("time", "lat", "lon"), env),
            "Solar_Flux_DN": (("time", "lat", "lon"), env),
        },
        coords={"time": np.arange(3), "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    ds.close()


def build_client(overview_dir: Path) -> TestClient:
    app = FastAPI()
    overview_service = McdOverviewDataService(
        FakeBaseDataService(),
        overview_dir=overview_dir,
        nomad_dir=overview_dir / "missing_nomad",
    )
    app.state.mcd_overview_service = overview_service
    app.state.mcd_overview_analysis_service = AnalysisService(overview_service, mcd_variables=OVERVIEW_MCD_VARIABLES)
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_overview_point_probe_returns_nearest_point_current_value_and_annual_comparison(tmp_path):
    overview_dir = tmp_path / "mcd_overview"
    overview_dir.mkdir()
    write_probe_overview(overview_dir / "MCD_MY34_overview.nc")
    client = build_client(overview_dir)

    response = client.get("/api/explore/overview/point-probe?my=34&lat=2.2&lng=-176&ls=11&variable=o3col")

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested"] == {"lat": 2.2, "lng": -176.0, "ls": 11.0}
    assert payload["gridPoint"] == {"lat": 2.5, "lng": -175.0}
    assert payload["current"] == {"ls": 10.0, "value": 21.0}
    assert payload["series"]["point"] == [20.0, 21.0, 22.0]
    assert payload["series"]["globalMean"] == [25.0, 26.0, 27.0]
    assert payload["series"]["latitudeMean"] == [15.0, 16.0, 17.0]
    assert payload["comparison"]["pointMinusGlobal"] == -5.0
    assert payload["comparison"]["pointMinusLatitudeMean"] == 5.0
    assert payload["variable"] == "o3col"
