import importlib.util
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _load_demo3_module():
    script_path = BACKEND_DIR / "models" / "training_scripts" / "demo3.py"
    spec = importlib.util.spec_from_file_location("astraatmos_demo3_training_loader", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_overview_file(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 6)
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 3)

        ls = ds.createVariable("Ls", "f4", ("time",))
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        ls[:] = np.linspace(offset, offset + 5.0, 6, dtype=np.float32) % 360.0
        lat[:] = np.array([-45.0, 45.0], dtype=np.float32)
        lon[:] = np.array([0.0, 120.0, 240.0], dtype=np.float32)

        base = np.arange(36, dtype=np.float32).reshape(6, 2, 3) + offset
        for var_name, delta in {
            "o3col": 0.0,
            "U_Wind": 10.0,
            "V_Wind": 20.0,
            "Dust_Optical_Depth": 30.0,
            "Solar_Flux_DN": 40.0,
            "Temperature": 50.0,
        }.items():
            var = ds.createVariable(var_name, "f4", ("time", "lat", "lon"))
            var[:] = base + delta


def _write_raw_3h_mcd_file(path: Path, offset: float, *, include_dust: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 10)
        ds.createDimension("lat", 37)
        ds.createDimension("lon", 72)

        ls = ds.createVariable("LS", "f4", ("time",))
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        ls[:] = np.linspace(offset, offset + 1.0, 10, dtype=np.float32) % 360.0
        lat[:] = np.linspace(90.0, -90.0, 37, dtype=np.float32)
        lon[:] = np.linspace(-180.0, 175.0, 72, dtype=np.float32)

        base = np.arange(10 * 37 * 72, dtype=np.float32).reshape(10, 37, 72) + offset
        raw_variables = {
            "O3COL": 0.0,
            "U": 10.0,
            "T": 20.0,
            "V": 30.0,
            "FSDS": 40.0,
        }
        if include_dust:
            raw_variables["Dust_Optical_Depth"] = 50.0
        for var_name, delta in raw_variables.items():
            var = ds.createVariable(var_name, "f4", ("time", "lat", "lon"))
            var[:] = base + delta
        ds["O3COL"].units = "um-atm"


def test_official_training_loader_builds_tensors_from_mcd_overview(tmp_path):
    workspace_tmp = tmp_path
    overview_dir = workspace_tmp / "mcd_overview"
    first_file = overview_dir / "MCD_MY24_overview.nc"
    second_file = overview_dir / "MCD_MY25_overview.nc"
    _write_overview_file(first_file, 0.0)
    _write_overview_file(second_file, 10.0)

    demo3 = _load_demo3_module()
    x_torch, ls_torch, y_torch, height, width = demo3.prepare_training_tensors(
        openmars_dir=workspace_tmp / "openmars",
        mcd_dir=workspace_tmp / "MCD",
        mcd_overview_dir=overview_dir,
        selected_channels=["U", "T"],
        window=2,
        horizon=2,
        training_dataset="mcd_overview",
    )

    assert list(x_torch.shape) == [9, 2, 3, 2, 3]
    assert list(y_torch.shape) == [9, 2, 1, 2, 3]
    assert list(ls_torch.shape) == [9, 2]
    assert height == 2
    assert width == 3


def test_mars_year_blocks_allow_cross_year_windows(tmp_path):
    overview_dir = tmp_path / "mcd_overview"
    _write_overview_file(overview_dir / "MCD_MY27_overview.nc", 27.0)
    _write_overview_file(overview_dir / "MCD_MY28_overview.nc", 28.0)
    demo3 = _load_demo3_module()

    prepared = demo3._prepare_training_data(
        openmars_dir=tmp_path / "openmars",
        mcd_dir=tmp_path / "MCD",
        mcd_overview_dir=overview_dir,
        selected_channels=["U"],
        window=2,
        horizon=2,
        training_dataset="mcd_overview",
    )

    assert prepared.volume_metadata.sample_starts.tolist() == list(range(9))
    assert prepared.volume_metadata.sample_mars_years == (27, 27, 27, 27, 27, 27, 28, 28, 28)


def test_zero_validation_ratio_reaches_runner_split_without_reallocation(tmp_path):
    overview_dir = tmp_path / "mcd_overview"
    _write_overview_file(overview_dir / "MCD_MY27_overview.nc", 27.0)
    _write_overview_file(overview_dir / "MCD_MY28_overview.nc", 28.0)
    demo3 = _load_demo3_module()

    prepared = demo3._prepare_training_data(
        openmars_dir=tmp_path / "openmars",
        mcd_dir=tmp_path / "MCD",
        mcd_overview_dir=overview_dir,
        selected_channels=["U"],
        window=2,
        horizon=2,
        training_dataset="mcd_overview",
        train_ratio=0.8,
        validation_ratio=0.0,
        test_ratio=0.2,
    )
    train, validation, test = demo3._split_training_data(
        prepared,
        {"train_ratio": 0.8, "validation_ratio": 0.0, "test_ratio": 0.2},
    )
    assert len(train) == 5
    assert len(validation) == 0
    assert len(test) == 1


def test_mars_file_without_year_metadata_fails_explicitly(tmp_path):
    overview_dir = tmp_path / "mcd_overview"
    _write_overview_file(overview_dir / "MCD_without_year_overview.nc", 27.0)
    demo3 = _load_demo3_module()

    with pytest.raises(ValueError, match="identify Mars year"):
        demo3.prepare_training_tensors(
            openmars_dir=tmp_path / "openmars",
            mcd_dir=tmp_path / "MCD",
            mcd_overview_dir=overview_dir,
            selected_channels=["U"],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
        )


def test_official_training_loader_builds_tensors_from_raw_3h_mcd_dataset(tmp_path):
    workspace_tmp = tmp_path
    raw_dir = workspace_tmp / "MCD_Output_global_10m_ls_lst"
    first_file = raw_dir / "MCD_MY24_global_3h_5deg_10m_ls_lst.nc"
    second_file = raw_dir / "MCD_MY25_global_3h_5deg_10m_ls_lst.nc"
    _write_raw_3h_mcd_file(first_file, 0.0)
    _write_raw_3h_mcd_file(second_file, 2.0)

    demo3 = _load_demo3_module()
    x_torch, ls_torch, y_torch, height, width = demo3.prepare_training_tensors(
        openmars_dir=workspace_tmp / "openmars",
        mcd_dir=workspace_tmp / "MCD",
        mcd_overview_dir=raw_dir,
        selected_channels=["U", "D", "T"],
        window=2,
        horizon=2,
        training_dataset="mcd_overview",
    )

    assert list(x_torch.shape) == [17, 2, 4, 36, 72]
    assert list(y_torch.shape) == [17, 2, 1, 36, 72]
    assert list(ls_torch.shape) == [17, 2]
    assert height == 36
    assert width == 72


def test_raw_mcd_dust_channel_is_rejected_when_field_is_missing(tmp_path):
    raw_dir = tmp_path / "MCD_Output_global_10m_ls_lst"
    path = raw_dir / "MCD_MY24_global_3h_5deg_10m_ls_lst.nc"
    _write_raw_3h_mcd_file(path, 0.0, include_dust=False)
    demo3 = _load_demo3_module()

    with pytest.raises(ValueError, match="Dust"):
        demo3.prepare_training_tensors(
            openmars_dir=tmp_path / "openmars",
            mcd_dir=tmp_path / "MCD",
            mcd_overview_dir=raw_dir,
            selected_channels=["D"],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
        )
