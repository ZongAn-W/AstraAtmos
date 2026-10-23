import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.analysis_service import AnalysisService  # noqa: E402
from services.user_data_service import UserDataService  # noqa: E402
from services.user_overview_source_service import (  # noqa: E402
    UserMcdOverviewDataView,
    build_uploaded_nomad_validation,
    build_uploaded_ozone_layer,
)


def _uploaded_mcd():
    lat = np.array([-2.5, 2.5], dtype=np.float32)
    lon = np.array([0.0, 5.0], dtype=np.float32)
    ls = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    ozone = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 1
    env = np.ones((3, 2, 2), dtype=np.float32)
    return {
        "data_type": "mcd",
        "lat": lat,
        "lon": lon,
        "ls": ls,
        "o3col": ozone,
        "Temperature": env * 180.0,
        "U_Wind": env * 5.0,
        "V_Wind": env * 2.0,
        "Solar_Flux_DN": env * 90.0,
    }


def test_uploaded_mcd_view_can_drive_analysis_service():
    view = UserMcdOverviewDataView(upload_id=12, mars_year=34, data=_uploaded_mcd())
    service = AnalysisService(view, mcd_variables=["Temperature", "U_Wind", "V_Wind", "Solar_Flux_DN"])

    globe = service.get_globe_data(34, 20.0, variable="Temperature")
    heatmap = service.get_seasonal_heatmap(34, variable="o3col")
    corr = service.get_correlation_matrix(34)

    assert globe["mars_year"] == 34
    assert globe["variable"] == "Temperature"
    assert globe["points"]
    assert heatmap["x"] == [10.0, 20.0, 30.0]
    assert corr["variable_names"] == ["o3col", "Temperature", "U_Wind", "V_Wind", "Solar_Flux_DN"]


def test_uploaded_mcd_view_rejects_wrong_year():
    view = UserMcdOverviewDataView(upload_id=12, mars_year=34, data=_uploaded_mcd())

    with pytest.raises(ValueError, match="MY34"):
        view.get_openmars_data(27)


def test_uploaded_ozone_layer_uses_nearest_ls():
    layer = build_uploaded_ozone_layer(_uploaded_mcd(), ls=19.0, source="openmars")

    assert layer["source"] == "openmars"
    assert layer["ls"] == 20.0
    assert layer["points"]
    assert layer["minVal"] <= layer["maxVal"]


def test_uploaded_nomad_validation_compares_shared_cells():
    mcd_layer = {
        "source": "mcd",
        "ls": 10.0,
        "points": [
            {"lat": -2.5, "lng": 0.0, "val": 10.0},
            {"lat": 2.5, "lng": 5.0, "val": 20.0},
        ],
    }
    nomad_layer = {
        "source": "nomad",
        "ls": 10.0,
        "points": [
            {"lat": -2.5, "lng": 0.0, "val": 8.0, "count": 3},
            {"lat": 2.5, "lng": 5.0, "val": 22.0, "count": 4},
        ],
    }

    validation = build_uploaded_nomad_validation(mcd_layer, nomad_layer)

    assert validation["sample_count"] == 2
    assert validation["bias"] == 0.0
    assert validation["mae"] == 2.0
    assert validation["points"][0]["count"] == 3


def test_user_data_service_sorts_nomad_count_with_ls(tmp_path):
    path = tmp_path / "nomad.nc"
    field = np.ones((2, 36, 72), dtype=np.float32)
    count = np.ones((2, 36, 72), dtype=np.int32)
    field[0] *= 30.0
    field[1] *= 10.0
    count[0] *= 3
    count[1] *= 1
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([30.0, 10.0], dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), field),
            "count": (("time", "lat", "lon"), count),
        },
        coords={
            "time": np.arange(2),
            "lat": np.linspace(87.5, -87.5, 36, dtype=np.float32),
            "lon": np.linspace(-180.0, 175.0, 72, dtype=np.float32),
        },
    )
    ds.to_netcdf(path)

    loaded = UserDataService()._load_nc_file(str(path))

    assert loaded["data_type"] == "nomad"
    assert loaded["ls"].tolist() == [10.0, 30.0]
    assert loaded["o3col"][:, 0, 0].tolist() == [10.0, 30.0]
    assert loaded["count"][:, 0, 0].tolist() == [1, 3]


def test_user_data_service_normalizes_reference_mcd_upload_for_overview(tmp_path):
    path = tmp_path / "MCD_MY34_global_3h_5deg_10m_ls_lst.nc"
    shape = (16, 37, 72)
    ds = xr.Dataset(
        data_vars={
            "LS": (("time",), np.linspace(10.0, 20.0, shape[0], dtype=np.float32)),
            "O3COL": (("time", "lat", "lon"), np.ones(shape, dtype=np.float32)),
            "T": (("time", "lat", "lon"), np.full(shape, 180.0, dtype=np.float32)),
            "U": (("time", "lat", "lon"), np.full(shape, 5.0, dtype=np.float32)),
            "V": (("time", "lat", "lon"), np.full(shape, 2.0, dtype=np.float32)),
            "FSDS": (("time", "lat", "lon"), np.full(shape, 90.0, dtype=np.float32)),
            "PS": (("time", "lat", "lon"), np.full(shape, 6.0, dtype=np.float32)),
        },
        coords={
            "time": np.arange(shape[0], dtype=np.int32),
            "lat": np.linspace(90.0, -90.0, shape[1], dtype=np.float32),
            "lon": np.linspace(-180.0, 175.0, shape[2], dtype=np.float32),
        },
        attrs={"mars_year": 34},
    )
    ds.to_netcdf(path)
    ds.close()

    loaded = UserDataService()._load_nc_file(str(path))

    assert loaded["data_type"] == "mcd"
    assert loaded["mars_year"] == 34
    assert loaded["o3col"].shape == (2, 36, 72)
    assert loaded["Temperature"].shape == loaded["o3col"].shape
    assert loaded["U_Wind"].shape == loaded["o3col"].shape
    assert loaded["V_Wind"].shape == loaded["o3col"].shape
    assert loaded["Solar_Flux_DN"].shape == loaded["o3col"].shape
    assert loaded["Pressure"].shape == loaded["o3col"].shape
