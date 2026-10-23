import sys
from pathlib import Path

import numpy as np
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.upload_service import UploadService, ValidationResult  # noqa: E402
from config import MAX_UPLOAD_SIZE_MB, MCD_CACHE_DIR  # noqa: E402


def _raw_mcd_dataset():
    shape = (16, 37, 72)
    return xr.Dataset(
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
        attrs={"mars_year": 33},
    )


def _ready_mcd_dataset():
    shape = (2, 36, 72)
    return xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), np.ones(shape, dtype=np.float32)),
            "Temperature": (("time", "lat", "lon"), np.full(shape, 180.0, dtype=np.float32)),
            "U_Wind": (("time", "lat", "lon"), np.full(shape, 5.0, dtype=np.float32)),
            "V_Wind": (("time", "lat", "lon"), np.full(shape, 2.0, dtype=np.float32)),
            "Solar_Flux_DN": (("time", "lat", "lon"), np.full(shape, 90.0, dtype=np.float32)),
        },
        coords={
            "time": np.arange(shape[0], dtype=np.int32),
            "lat": np.linspace(87.5, -87.5, shape[1], dtype=np.float32),
            "lon": np.linspace(-180.0, 175.0, shape[2], dtype=np.float32),
        },
        attrs={"mars_year": 33},
    )


def test_upload_accepts_raw_mcd_reference_shape():
    result = UploadService()._validate_dataset(
        _raw_mcd_dataset(),
        ValidationResult(),
        "MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
    )

    assert result.is_valid is True
    assert result.data_type == "mcd"
    assert result.mars_year == 33
    assert result.lat_points == 36
    assert result.lon_points == 72


def test_upload_rejects_internal_mcd_overview_file():
    result = UploadService()._validate_dataset(
        _ready_mcd_dataset(),
        ValidationResult(),
        "MCD_MY33_overview.nc",
    )

    assert result.is_valid is False
    assert result.data_type == "mcd"
    assert "raw 3-hour MCD" in result.error


def test_upload_rejects_raw_mcd_without_pressure_needed_for_cache_build():
    ds = _raw_mcd_dataset().drop_vars("PS")

    result = UploadService()._validate_dataset(
        ds,
        ValidationResult(),
        "MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
    )

    assert result.is_valid is False
    assert result.data_type == "mcd"
    assert "PS" in result.error


def test_upload_size_limit_allows_reference_raw_mcd_files():
    assert MAX_UPLOAD_SIZE_MB >= 300
    assert MCD_CACHE_DIR.name == "mcd_cache"
