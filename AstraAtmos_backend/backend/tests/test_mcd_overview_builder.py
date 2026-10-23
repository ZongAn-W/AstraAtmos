import sys
from pathlib import Path

import pytest
import xarray as xr
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.build_mcd_overview_dataset import (  # noqa: E402
    _default_reference_dir,
    build_overview_dataset,
    build_year,
    interpolate_reference_ozone,
)

KG_M2_PER_UM_ATM_O3 = 2.14e-6


def _workspace_reference_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "Data" / "MCD_Output_global_10m_ls_lst"


def test_build_overview_dataset_creates_backend_runtime_shape(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    base_mcd = repo_root / "AstraAtmos_backend" / "backend" / "data" / "mcd" / "MCD_MY27_Lat-90-90_real.nc"
    ozone_ref = _workspace_reference_dir() / "MCD_MY27_global_3h_5deg_10m_ls_lst.nc"
    output_path = tmp_path / "MCD_MY27_overview.nc"

    build_overview_dataset(base_mcd, ozone_ref, output_path)

    ds = xr.open_dataset(output_path)
    try:
        assert "o3col" in ds.data_vars
        assert "Dust_Optical_Depth" in ds.data_vars
        assert ds["o3col"].shape == ds["Temperature"].shape
        assert ds["o3col"].shape == ds["Dust_Optical_Depth"].shape
        assert ds["lat"].shape == (36,)
        assert ds["lon"].shape == (72,)
        assert ds["Ls"].shape == (669,)
        assert float(ds["lat"][0]) == pytest.approx(87.5, abs=1e-3)
        assert float(ds["lat"][-1]) == pytest.approx(-87.5, abs=1e-3)
        assert float(ds["o3col"].isnull().mean()) < 0.05
    finally:
        ds.close()


def test_interpolate_reference_ozone_preserves_samples_after_ls_wrap():
    reference_ls = np.array([359.5, 0.5, 1.5, 2.5], dtype=np.float32)
    reference_ozone = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(-1, 1, 1)
    target_ls = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    result = interpolate_reference_ozone(reference_ozone, reference_ls, target_ls)

    np.testing.assert_allclose(result[:, 0, 0], [2.0, 3.0, 4.0], rtol=1e-6)


def test_build_overview_from_reference_dataset_creates_direct_mcd_overview(tmp_path):
    reference_path = _workspace_reference_dir() / "MCD_MY34_global_3h_5deg_10m_ls_lst.nc"
    output_path = tmp_path / "MCD_MY34_overview.nc"

    from scripts.build_mcd_overview_dataset import build_overview_from_reference_dataset

    build_overview_from_reference_dataset(reference_path, output_path)

    ds = xr.open_dataset(output_path)
    try:
        assert ds["o3col"].shape[1:] == (36, 72)
        assert ds["Temperature"].shape == ds["o3col"].shape
        assert ds["U_Wind"].shape == ds["o3col"].shape
        assert ds["V_Wind"].shape == ds["o3col"].shape
        assert ds["Pressure"].shape == ds["o3col"].shape
        assert ds["Solar_Flux_DN"].shape == ds["o3col"].shape
        assert ds["Dust_Optical_Depth"].shape == ds["o3col"].shape
        assert ds["lat"].shape == (36,)
        assert ds["lon"].shape == (72,)
        assert float(ds["lat"][0]) == pytest.approx(87.5, abs=1e-3)
        assert float(ds["lat"][-1]) == pytest.approx(-87.5, abs=1e-3)
        assert float(ds["lon"][0]) == pytest.approx(-180.0, abs=1e-3)
        assert float(ds["lon"][-1]) == pytest.approx(175.0, abs=1e-3)
        assert float(ds["Ls"].min()) >= 0.0
        assert float(ds["Ls"].max()) < 360.0
        assert float(ds["o3col"].isnull().mean()) < 0.05
        assert ds.attrs["build_mode"] == "reference_direct"
    finally:
        ds.close()


def test_build_overview_from_reference_dataset_writes_o3col_in_um_atm(tmp_path):
    from scripts.build_mcd_overview_dataset import build_overview_from_reference_dataset

    reference_path = tmp_path / "MCD_MY34_global_3h_5deg_10m_ls_lst.nc"
    output_path = tmp_path / "MCD_MY34_overview.nc"
    shape = (8, 37, 72)
    ref = xr.Dataset(
        data_vars={
            "LS": (("time",), np.linspace(10.0, 12.0, shape[0], dtype=np.float32)),
            "O3COL": (("time", "lat", "lon"), np.full(shape, 2.0 * KG_M2_PER_UM_ATM_O3, dtype=np.float32)),
            "T": (("time", "lat", "lon"), np.full(shape, 180.0, dtype=np.float32)),
            "U": (("time", "lat", "lon"), np.full(shape, 5.0, dtype=np.float32)),
            "V": (("time", "lat", "lon"), np.full(shape, 2.0, dtype=np.float32)),
            "PS": (("time", "lat", "lon"), np.full(shape, 6.0, dtype=np.float32)),
            "FSDS": (("time", "lat", "lon"), np.full(shape, 90.0, dtype=np.float32)),
        },
        coords={
            "time": np.arange(shape[0], dtype=np.int32),
            "lat": np.linspace(90.0, -90.0, shape[1], dtype=np.float32),
            "lon": np.linspace(-180.0, 175.0, shape[2], dtype=np.float32),
        },
    )
    ref["O3COL"].attrs["units"] = "kg m-2"
    ref.to_netcdf(reference_path)
    ref.close()

    build_overview_from_reference_dataset(reference_path, output_path)

    ds = xr.open_dataset(output_path)
    try:
        assert float(ds["o3col"].mean()) == pytest.approx(2.0, rel=1e-5)
        assert ds["o3col"].attrs["units"] == "um-atm"
    finally:
        ds.close()


def test_default_reference_dir_prefers_workspace_data_directory():
    assert _default_reference_dir() == _workspace_reference_dir()


def test_build_year_auto_uses_downloaded_reference_dataset(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    base_mcd_dir = repo_root / "AstraAtmos_backend" / "backend" / "data" / "mcd"

    output_path = build_year(
        28,
        _workspace_reference_dir(),
        tmp_path,
        base_mcd_dir=base_mcd_dir,
        mode="auto",
    )

    ds = xr.open_dataset(output_path)
    try:
        assert ds.attrs["build_mode"] == "reference_direct"
        assert ds["Ls"].shape == (687,)
        assert float(ds["Temperature"].min()) > 100.0
    finally:
        ds.close()
