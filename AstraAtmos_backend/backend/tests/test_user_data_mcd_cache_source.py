import asyncio
import sys
from pathlib import Path

import numpy as np
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.user_data_service import UserDataService  # noqa: E402


def _write_ready_overview(path: Path):
    shape = (2, 36, 72)
    ds = xr.Dataset(
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
    ds.to_netcdf(path)
    ds.close()


def _write_raw_3h_mcd(path: Path):
    shape = (16, 2, 2)
    ds = xr.Dataset(
        data_vars={
            "O3COL": (("time", "lat", "lon"), np.arange(np.prod(shape), dtype=np.float32).reshape(shape)),
            "LS": (("time",), np.linspace(10.0, 20.0, shape[0], dtype=np.float32)),
        },
        coords={
            "time": np.arange(shape[0], dtype=np.int32),
            "lat": np.array([-45.0, 45.0], dtype=np.float32),
            "lon": np.array([0.0, 5.0], dtype=np.float32),
        },
        attrs={"mars_year": 33},
    )
    ds["O3COL"].attrs["units"] = "kg m-2"
    ds.to_netcdf(path)
    ds.close()


def test_user_data_service_uses_ready_mcd_overview_artifact(tmp_path):
    async def run():
        overview = tmp_path / "MCD_MY33_overview.nc"
        _write_ready_overview(overview)

        class FakeMcdCacheService:
            async def get_ready_artifact_path(self, upload_id, cache_type):
                assert upload_id == 42
                assert cache_type == "mcd_overview"
                return overview

        service = UserDataService(mcd_cache_service=FakeMcdCacheService())

        async def fake_record(upload_id):
            assert upload_id == 42
            return {
                "file_path": str(tmp_path / "raw.nc"),
                "data_type": "mcd",
            }

        service._get_record_file_info = fake_record
        data = await service._get_data(42)

        assert data["data_type"] == "mcd"
        assert data["o3col"].shape == (2, 36, 72)

    asyncio.run(run())


def test_user_data_service_keeps_approved_index_priority_for_mcd(tmp_path):
    async def run():
        approved = tmp_path / "approved.nc"
        _write_ready_overview(approved)

        class FakeMcdCacheService:
            async def get_ready_artifact_path(self, upload_id, cache_type):
                return None

        service = UserDataService(mcd_cache_service=FakeMcdCacheService())
        service._approved_index[42] = approved

        async def fake_record(upload_id):
            assert upload_id == 42
            return {
                "file_path": str(tmp_path / "raw.nc"),
                "data_type": "mcd",
            }

        service._get_record_file_info = fake_record
        data = await service._get_data(42)

        assert data["data_type"] == "mcd"
        assert data["o3col"].shape == (2, 36, 72)

    asyncio.run(run())


def test_user_data_service_loads_mcd_hourly_dataset_from_mcd_3h_artifact(tmp_path):
    async def run():
        raw_3h = tmp_path / "MCD_MY33_global_3h_5deg_10m_ls_lst.nc"
        _write_raw_3h_mcd(raw_3h)

        class FakeMcdCacheService:
            async def get_ready_artifact_path(self, upload_id, cache_type):
                assert upload_id == 42
                assert cache_type == "mcd_3h"
                return raw_3h

        service = UserDataService(mcd_cache_service=FakeMcdCacheService())

        async def fake_record(upload_id):
            assert upload_id == 42
            return {
                "file_path": str(tmp_path / "missing.nc"),
                "data_type": "mcd",
            }

        service._get_record_file_info = fake_record
        hourly = await service.get_loaded_mcd_hourly_dataset(42)

        assert hourly["data_type"] == "mcd"
        assert hourly["ls"].shape == (2,)
        assert hourly["O3COL_hourly"].shape == (2, 8, 2, 2)
        assert hourly["raw_3h_source_file"] == str(raw_3h)

    asyncio.run(run())
