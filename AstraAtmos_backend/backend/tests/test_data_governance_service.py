import sys
from pathlib import Path

import numpy as np
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.models import UploadRecord  # noqa: E402
from services.data_governance_service import DataGovernanceService  # noqa: E402


def test_governance_scores_raw_mcd_upload_against_normalized_overview_contract(tmp_path):
    path = tmp_path / "MCD_MY34_global_3h_5deg_10m_ls_lst.nc"
    raw_shape = (16, 37, 72)
    ds = xr.Dataset(
        data_vars={
            "LS": (("time",), np.linspace(10.0, 20.0, raw_shape[0], dtype=np.float32)),
            "O3COL": (("time", "lat", "lon"), np.ones(raw_shape, dtype=np.float32)),
            "T": (("time", "lat", "lon"), np.full(raw_shape, 180.0, dtype=np.float32)),
            "U": (("time", "lat", "lon"), np.full(raw_shape, 5.0, dtype=np.float32)),
            "V": (("time", "lat", "lon"), np.full(raw_shape, 2.0, dtype=np.float32)),
            "FSDS": (("time", "lat", "lon"), np.full(raw_shape, 90.0, dtype=np.float32)),
        },
        coords={
            "time": np.arange(raw_shape[0], dtype=np.int32),
            "lat": np.linspace(90.0, -90.0, raw_shape[1], dtype=np.float32),
            "lon": np.linspace(-180.0, 175.0, raw_shape[2], dtype=np.float32),
        },
    )
    ds.to_netcdf(path)
    record = UploadRecord(
        id=7,
        user_id=1,
        filename=path.name,
        file_path=str(path),
        data_type="mcd",
        status="valid",
        mars_year=34,
    )
    service = DataGovernanceService()

    meta = service._get_dataset_meta(record, path)
    quality = service._get_quality_metrics(record, path, meta)

    assert meta["data_type"] == "mcd"
    assert meta["lat_points"] == 36
    assert meta["lon_points"] == 72
    assert meta["ls_points"] == 2
    assert {"o3col", "Temperature", "U_Wind", "V_Wind", "Solar_Flux_DN"}.issubset(meta["variables"])
    assert "O3COL" not in meta["variables"]
    assert quality["metrics"]["variable_completeness"] == 1.0
    assert quality["metrics"]["variables"]["missing"] == []
    assert quality["metrics"]["grid_compatibility"]["score"] == 100.0
    assert "Grid is incompatible with native 36x72 resolution" not in quality["issues"]
