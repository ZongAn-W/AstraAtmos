"""预测体积缓存契约：单次预测只物化所需滑窗，且数值与原路径完全一致。

覆盖：

- ``prepare_tensors(return_scaled_volume=True)`` 的产物切窗后与逐样本展开路径 bitwise 相同；
- 上传模型测试集分区（``_window_stack`` / ``_uploaded_task_test_windows``）与展开切片相同；
- ``InferenceService._prepare_uploaded_task_volume`` 第二次调用命中进程内缓存，
  文件清单变化（含大小/修改时间）后不会复用旧体积。
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest
import torch

from services.inference_service import InferenceService
from services.prediction_volume_cache import clear_scaled_volumes
from services.training_split import split_sample_ranges
from training_backbones import user_model_runner

BACKEND_DIR = Path(__file__).resolve().parents[1]

TIME_STEPS = 12
WINDOW = 2
HORIZON = 2
CHANNELS = ["T"]
CHANNEL_VARIABLES = {"Temperature": 100.0}


def _write_overview_file(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("time", TIME_STEPS)
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        dataset.createVariable("lat", "f4", ("lat",))[:] = np.array([-45.0, 45.0], dtype=np.float32)
        dataset.createVariable("lon", "f4", ("lon",))[:] = np.array([0.0, 120.0, 240.0], dtype=np.float32)
        dataset.createVariable("Ls", "f4", ("time",))[:] = np.linspace(
            offset, offset + 60.0, TIME_STEPS, dtype=np.float32
        ) % 360.0
        base = np.arange(TIME_STEPS * 2 * 3, dtype=np.float32).reshape(TIME_STEPS, 2, 3) + offset
        dataset.createVariable("o3col", "f4", ("time", "lat", "lon"))[:] = base
        for name, delta in CHANNEL_VARIABLES.items():
            dataset.createVariable(name, "f4", ("time", "lat", "lon"))[:] = base + delta


@pytest.fixture(autouse=True)
def _isolated_volume_cache():
    clear_scaled_volumes()
    yield
    clear_scaled_volumes()


@pytest.fixture
def overview_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "mcd_overview"
    _write_overview_file(directory / "MCD_MY24_overview.nc", 0.0)
    _write_overview_file(directory / "MCD_MY25_overview.nc", 12.0)
    return directory


def _prepare(overview_dir: Path, **kwargs):
    return user_model_runner.prepare_tensors(
        overview_dir,
        overview_dir,
        CHANNELS,
        WINDOW,
        HORIZON,
        training_dataset="mcd_overview",
        mcd_overview_dir=overview_dir,
        return_ls=True,
        require_coordinates=False,
        **kwargs,
    )


def test_scaled_volume_windows_match_expanded_samples(overview_dir: Path):
    x_torch, y_torch, ls_torch, y_mean, y_std, height, width = _prepare(overview_dir)
    volume = _prepare(overview_dir, return_scaled_volume=True)

    # mcd_overview 数据源会把目录内所有文件按时间拼接，并加载全部 MCD 通道。
    total_time = int(volume.values.shape[0])
    channels = int(volume.values.shape[-1])
    assert volume.values.shape == (total_time, height, width, channels)
    assert volume.y_scaled.shape == (total_time, height, width)
    assert channels > 0
    assert volume.split_idx > 0
    assert volume.y_mean == pytest.approx(y_mean, rel=1e-12)
    assert volume.y_std == pytest.approx(y_std, rel=1e-12)

    assert int(x_torch.shape[0]) == int(volume.sample_starts.size)

    for index, start in enumerate(volume.sample_starts.tolist()):
        x_window = torch.from_numpy(
            np.ascontiguousarray(volume.values[start:start + WINDOW])
        ).permute(0, 3, 1, 2).float()
        assert torch.equal(x_window, x_torch[index])

        y_window = torch.from_numpy(
            np.ascontiguousarray(volume.y_scaled[start + WINDOW:start + WINDOW + HORIZON])
        ).unsqueeze(1).float()
        assert torch.equal(y_window, y_torch[index])

        ls_window = torch.from_numpy(
            np.ascontiguousarray(np.asarray(volume.ls)[start:start + WINDOW])
        ).reshape(WINDOW, 1).float()
        assert torch.equal(ls_window, ls_torch[index].reshape(WINDOW, 1))


def test_test_partition_windows_match_expanded_slice(overview_dir: Path):
    legacy_ratios = {
        "train_ratio": 0.8,
        "validation_ratio": 0.0,
        "test_ratio": 0.2,
    }
    x_torch, y_torch, ls_torch, _y_mean, _y_std, _height, _width = _prepare(
        overview_dir, **legacy_ratios
    )
    volume = _prepare(overview_dir, return_scaled_volume=True, **legacy_ratios)

    service = InferenceService()
    x_test, y_test, ls_test = service._uploaded_task_test_windows(volume, WINDOW, HORIZON)

    split = split_sample_ranges(
        int(x_torch.shape[0]), legacy_ratios
    )["test"][0]
    assert torch.equal(x_test, x_torch[split:])
    assert torch.equal(y_test, y_torch[split:])
    assert ls_test is not None
    assert torch.equal(ls_test, ls_torch[split:])


def test_uploaded_task_volume_hits_cache_on_second_call(overview_dir: Path):
    service = InferenceService()
    hypers = {
        "selected_channels": CHANNELS,
        "training_dataset": "mcd_overview",
        "window": WINDOW,
        "horizon": HORIZON,
    }

    _channels, first = service._prepare_uploaded_task_volume(hypers, WINDOW, HORIZON)
    _channels, second = service._prepare_uploaded_task_volume(hypers, WINDOW, HORIZON)

    # 命中缓存时返回同一对象，说明没有重新读取与重新标准化。
    assert second is first


def test_volume_signature_separates_split_and_window_parameters(overview_dir: Path):
    common = {
        "openmars_dir": overview_dir,
        "mcd_dir": overview_dir,
        "selected_channels": CHANNELS,
        "training_dataset": "mcd_overview",
        "mcd_overview_dir": overview_dir,
        "cache_prefix": "uploaded",
    }
    from services.prediction_volume_cache import volume_signature

    first = volume_signature(
        **common,
        split_ratios={"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1},
        window=2,
        horizon=2,
    )
    different_split = volume_signature(
        **common,
        split_ratios={"train_ratio": 0.5, "validation_ratio": 0.1, "test_ratio": 0.4},
        window=2,
        horizon=2,
    )
    different_window = volume_signature(
        **common,
        split_ratios={"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1},
        window=3,
        horizon=2,
    )

    assert first != different_split
    assert first != different_window


def test_volume_cache_invalidates_when_files_change(overview_dir: Path, tmp_path: Path):
    from services.prediction_volume_cache import (
        get_scaled_volume,
        put_scaled_volume,
        volume_signature,
    )

    signature = volume_signature(
        openmars_dir=overview_dir,
        mcd_dir=overview_dir,
        selected_channels=CHANNELS,
        training_dataset="mcd_overview",
        mcd_overview_dir=overview_dir,
        cache_prefix="test",
    )

    volume = _prepare(overview_dir, return_scaled_volume=True)
    put_scaled_volume(signature, volume)
    assert get_scaled_volume(signature) is volume

    # 文件内容变化后签名必须变化，避免复用过期体积。
    target = overview_dir / "MCD_MY25_overview.nc"
    _write_overview_file(target, 24.0)

    changed = volume_signature(
        openmars_dir=overview_dir,
        mcd_dir=overview_dir,
        selected_channels=CHANNELS,
        training_dataset="mcd_overview",
        mcd_overview_dir=overview_dir,
        cache_prefix="test",
    )

    assert changed != signature
    assert get_scaled_volume(changed) is None


def test_volume_signature_tracks_separate_overview_files(overview_dir: Path, tmp_path: Path):
    from services.prediction_volume_cache import volume_signature

    openmars_dir = tmp_path / "openmars"
    mcd_dir = tmp_path / "mcd"
    common = {
        "openmars_dir": openmars_dir,
        "mcd_dir": mcd_dir,
        "selected_channels": CHANNELS,
        "training_dataset": "mcd_overview",
        "mcd_overview_dir": overview_dir,
        "cache_prefix": "uploaded",
    }
    first = volume_signature(**common)
    _write_overview_file(overview_dir / "MCD_MY25_overview.nc", 24.0)
    changed = volume_signature(**common)

    assert changed != first
