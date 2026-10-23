import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services import inference_service as inference_module  # noqa: E402
from services.inference_service import InferenceService  # noqa: E402
from services.prediction_volume_cache import ScaledVolume, clear_scaled_volumes  # noqa: E402
from training_backbones import user_model_runner  # noqa: E402
from training_backbones.uploaded_model_contract import (  # noqa: E402
    attach_uploaded_model_contract,
)


@pytest.fixture(autouse=True)
def _isolate_volume_cache():
    clear_scaled_volumes()
    yield
    clear_scaled_volumes()


def test_nearest_sequence_index_uses_complete_merged_ls_axis():
    # The prediction selector sees all server files on one timeline. A repeated
    # Ls value therefore resolves deterministically to the first nearest sample.
    ls_axis = torch.tensor([[5.0], [90.0], [90.0], [180.0]])
    assert InferenceService._nearest_sequence_index(ls_axis, 90.0) == 1
    assert InferenceService._nearest_sequence_index(ls_axis, 92.0) == 1
LS_MODEL_SPEC = {
    "name": "InferenceLsModel",
    "auxiliary_inputs": {
        "ls": {
            "required": True,
            "shape": ["batch", "window"],
            "dtype": "float32",
            "unit": "degree",
        }
    },
}

LS_TOPOGRAPHY_MODEL_SPEC = {
    "name": "InferenceLsTopographyModel",
    "auxiliary_inputs": {
        **LS_MODEL_SPEC["auxiliary_inputs"],
        "topography": {
            "required": True,
            "shape": ["batch", 1, "height", "width"],
            "dtype": "float32",
            "unit": "meter",
        },
    },
}


class RecordingLsModel(torch.nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.ls_calls = []

    def forward(self, x, ls):
        self.ls_calls.append(ls.detach().cpu().clone())
        return x[:, -1:, :1].repeat(1, self.horizon, 1, 1, 1)


class RecordingLsTopographyModel(RecordingLsModel):
    def __init__(self, horizon):
        super().__init__(horizon)
        self.topography_calls = []

    def forward(self, x, ls, topography):
        self.ls_calls.append(ls.detach().cpu().clone())
        self.topography_calls.append(topography.detach().cpu().clone())
        return x[:, -1:, :1].repeat(1, self.horizon, 1, 1, 1)


def _setup_uploaded_inference(monkeypatch, *, with_topography=False, sample_count=10):
    window = 4
    horizon = 2
    total_time = sample_count + window + horizon - 1
    # 预测路径现在接收连续体积并自行切窗；夹具从同一份连续数组派生
    # x / ls，使断言中的 ls[i:j] 切片语义与旧的逐样本展开完全一致。
    volume_values = np.arange(total_time * 4, dtype=np.float32).reshape(total_time, 2, 2, 1)
    # ls[t] = t，使 ls[i:j] 与逐样本展开的窗口一一对应（与原夹具取值一致）。
    volume_ls = np.arange(total_time, dtype=np.float32).reshape(total_time, 1)
    x = torch.from_numpy(
        np.stack([
            np.ascontiguousarray(volume_values[i:i + window]).transpose(0, 3, 1, 2)
            for i in range(sample_count)
        ])
    ).float()
    y = torch.zeros(sample_count, horizon, 1, 2, 2)
    ls = torch.from_numpy(
        np.stack([volume_ls[i:i + window, 0] for i in range(sample_count)])
    ).float()
    assert x.shape == (sample_count, window, 1, 2, 2)
    assert ls.shape == (sample_count, window)
    if with_topography:
        model = attach_uploaded_model_contract(
            RecordingLsTopographyModel(horizon),
            LS_TOPOGRAPHY_MODEL_SPEC,
        )
    else:
        model = attach_uploaded_model_contract(RecordingLsModel(horizon), LS_MODEL_SPEC)
    prepare_calls = []
    topography_calls = []
    latitude = torch.tensor([45.0, -45.0]).numpy()
    longitude = torch.tensor([-90.0, 90.0]).numpy()
    static_topography = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

    def fake_prepare_tensors(*args, **kwargs):
        assert kwargs.get("return_ls") is True
        assert kwargs.get("return_coordinates") is True
        assert kwargs.get("require_coordinates") is False
        assert kwargs.get("return_scaled_volume") is True
        prepare_calls.append((args, kwargs))
        return ScaledVolume(
            values=volume_values,
            # 目标通道为单通道，与生产实现一致（y_scaled 为 3D）。
            y_scaled=volume_values[..., 0],
            ls=volume_ls[:, 0].copy(),
            y_mean=0.0,
            y_std=1.0,
            height=2,
            width=2,
            latitude=latitude,
            longitude=longitude,
            split_idx=int(0.8 * sample_count),
        )

    def fake_prepare_topography(target_latitude, target_longitude, **kwargs):
        topography_calls.append((target_latitude, target_longitude))
        return static_topography

    monkeypatch.setattr(user_model_runner, "prepare_tensors", fake_prepare_tensors)
    monkeypatch.setattr(user_model_runner, "load_uploaded_model", lambda path, config: model)
    monkeypatch.setattr(
        inference_module,
        "prepare_topography_grid",
        fake_prepare_topography,
        raising=False,
    )

    service = InferenceService.__new__(InferenceService)
    service.device = torch.device("cpu")
    service.openmars_dir = Path("unused-openmars")
    service.mcd_dir = Path("unused-mcd")
    service._load_task_state_dict = lambda task: {}

    task = SimpleNamespace(metrics="{}")
    hypers = {
        "window": window,
        "horizon": horizon,
        "batch_size": 16,
        "selected_channels": [],
        "custom_model_params": {},
        "_uploaded_model_param_schema": {},
        "_uploaded_model_path": "unused.source",
    }
    return (
        service,
        task,
        hypers,
        model,
        ls,
        prepare_calls,
        static_topography,
        topography_calls,
    )


def test_uploaded_volume_preparation_passes_task_split_ratios(monkeypatch):
    service, _task, hypers, _model, _ls, prepare_calls, *_rest = _setup_uploaded_inference(monkeypatch)
    hypers.update({"train_ratio": 0.5, "validation_ratio": 0.1, "test_ratio": 0.4})

    service._prepare_uploaded_task_volume(hypers, hypers["window"], hypers["horizon"])

    kwargs = prepare_calls[-1][1]
    assert kwargs["train_ratio"] == 0.5
    assert kwargs["validation_ratio"] == 0.1
    assert kwargs["test_ratio"] == 0.4


def test_uploaded_volume_preparation_uses_legacy_split_for_old_tasks(monkeypatch):
    service, _task, hypers, _model, _ls, prepare_calls, *_rest = _setup_uploaded_inference(monkeypatch)

    service._prepare_uploaded_task_volume(hypers, hypers["window"], hypers["horizon"])

    kwargs = prepare_calls[-1][1]
    assert kwargs["train_ratio"] == 0.8
    assert kwargs["validation_ratio"] == 0.0
    assert kwargs["test_ratio"] == 0.2


def test_uploaded_volume_preparation_marks_invalid_split_metadata_as_legacy(monkeypatch):
    service, _task, hypers, _model, _ls, prepare_calls, *_rest = _setup_uploaded_inference(monkeypatch)
    hypers.update({"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": "invalid"})

    service._prepare_uploaded_task_volume(hypers, hypers["window"], hypers["horizon"])

    kwargs = prepare_calls[-1][1]
    assert kwargs["train_ratio"] == 0.8
    assert kwargs["validation_ratio"] == 0.0
    assert kwargs["test_ratio"] == 0.2


def test_uploaded_old_task_test_window_matches_legacy_split_boundary(monkeypatch):
    service, _task, hypers, _model, ls, *_rest = _setup_uploaded_inference(
        monkeypatch, sample_count=6
    )
    _channels, volume = service._prepare_uploaded_task_volume(
        hypers, hypers["window"], hypers["horizon"]
    )

    _x_test, _y_test, ls_test = service._uploaded_task_test_windows(
        volume, hypers["window"], hypers["horizon"]
    )

    assert torch.equal(ls_test, ls[5:9])


def test_uploaded_formal_prediction_receives_selected_history_ls(monkeypatch):
    service, task, hypers, model, ls, *_rest = _setup_uploaded_inference(monkeypatch)

    service._predict_uploaded_task_window(task, hypers, ls_start=0.0, horizon=2)

    assert torch.equal(model.ls_calls[-1], ls[0:1])


def test_uploaded_window_returns_actual_input_and_target_ls(monkeypatch):
    service, task, hypers, _model, _ls, *_rest = _setup_uploaded_inference(monkeypatch)

    result = service._predict_uploaded_task_window(
        task,
        hypers,
        ls_start=1.5,
        horizon=2,
    )

    assert result[5] == [1.0, 2.0, 3.0, 4.0]
    assert result[6] == [5.0, 6.0]


def test_uploaded_test_set_metrics_receives_matching_ls_batches(monkeypatch):
    service, task, hypers, model, ls, *_rest = _setup_uploaded_inference(monkeypatch)

    service._uploaded_task_test_set_arrays(task, hypers, horizon=2)

    assert torch.equal(model.ls_calls[-1], ls[8:10])


@pytest.mark.parametrize(
    ("ratios", "test_start"),
    [
        pytest.param({}, 40, id="legacy-80-20"),
        pytest.param(
            {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1},
            45,
            id="default-70-20-10",
        ),
        pytest.param(
            {"train_ratio": 0.5, "validation_ratio": 0.1, "test_ratio": 0.4},
            30,
            id="custom-50-10-40",
        ),
    ],
)
@pytest.mark.parametrize("horizon", [1, 2])
def test_uploaded_metrics_entry_uses_task_test_partition(monkeypatch, ratios, test_start, horizon):
    service, task, hypers, model, ls, *_rest = _setup_uploaded_inference(
        monkeypatch, sample_count=50
    )
    hypers.update(ratios)

    metrics = service._uploaded_task_test_set_metrics(task, hypers, horizon=horizon)

    assert torch.equal(torch.cat(model.ls_calls), ls[test_start:])
    assert len(metrics["per_step"]) == horizon
    # The model repeats the last input; each subsequent truth frame increases by 4.
    expected_errors = 4 * np.arange(1, horizon + 1)
    assert metrics["overall"]["rmse"] == pytest.approx(
        np.sqrt(np.mean(expected_errors ** 2)), rel=1e-5
    )
    assert metrics["overall"]["mae"] == pytest.approx(np.mean(expected_errors), rel=1e-5)
    expected_source = "task_metadata" if ratios else "legacy_compatibility"
    assert metrics["split_meta"]["source"] == expected_source
    assert metrics["split_meta"]["legacy_compatibility"] is (not bool(ratios))


def test_uploaded_permutation_importance_keeps_history_ls_fixed(monkeypatch):
    service, task, hypers, model, ls, *_rest = _setup_uploaded_inference(monkeypatch)

    service._uploaded_task_permutation_importance(
        task,
        hypers,
        selected_variables=["Ozone"],
        horizon=2,
    )

    assert len(model.ls_calls) >= 2
    assert all(torch.equal(call, ls[8:10]) for call in model.ls_calls)


def test_uploaded_test_results_receive_matching_ls(monkeypatch):
    service, task, hypers, model, ls, *_rest = _setup_uploaded_inference(monkeypatch)

    asyncio.run(service._get_uploaded_model_test_results(task, hypers))

    assert torch.equal(model.ls_calls[-1], ls[8:10])


def test_all_uploaded_inference_paths_reuse_persisted_training_dataset(monkeypatch):
    service, task, hypers, _model, _ls, prepare_calls, *_rest = _setup_uploaded_inference(monkeypatch)
    hypers["training_dataset"] = "mcd_overview"
    data_dirs = {"MCD_RAW_3H_DIR": "persisted-mcd-overview"}

    service._predict_uploaded_task_window(task, hypers, ls_start=0.0, horizon=2, data_dirs=data_dirs)
    service._uploaded_task_test_set_arrays(task, hypers, horizon=2, data_dirs=data_dirs)
    service._uploaded_task_permutation_importance(
        task,
        hypers,
        selected_variables=["Ozone"],
        horizon=2,
        data_dirs=data_dirs,
    )
    asyncio.run(service._get_uploaded_model_test_results(task, hypers, data_dirs=data_dirs))

    assert len(prepare_calls) == 4
    for _args, kwargs in prepare_calls:
        assert kwargs["training_dataset"] == "mcd_overview"
        assert kwargs["mcd_overview_dir"] == Path("persisted-mcd-overview")


def test_all_uploaded_inference_paths_receive_static_topography(monkeypatch):
    (
        service,
        task,
        hypers,
        model,
        ls,
        _prepare_calls,
        static_topography,
        topography_preparations,
    ) = _setup_uploaded_inference(monkeypatch, with_topography=True)

    service._predict_uploaded_task_window(task, hypers, ls_start=0.0, horizon=2)
    service._uploaded_task_test_set_arrays(task, hypers, horizon=2)
    service._uploaded_task_permutation_importance(
        task,
        hypers,
        selected_variables=["Ozone"],
        horizon=2,
    )
    asyncio.run(service._get_uploaded_model_test_results(task, hypers))

    assert len(topography_preparations) == 4
    assert len(model.topography_calls) >= 5
    assert all(
        call.shape[1:] == (1, 2, 2) for call in model.topography_calls
    )
    assert all(
        torch.equal(call[0], static_topography) for call in model.topography_calls
    )
    permutation_ls_calls = model.ls_calls[2:-1]
    permutation_topography_calls = model.topography_calls[2:-1]
    assert len(permutation_ls_calls) >= 2
    assert all(torch.equal(call, ls[8:10]) for call in permutation_ls_calls)
    assert all(
        torch.equal(call, permutation_topography_calls[0])
        for call in permutation_topography_calls
    )


def test_ls_only_inference_does_not_prepare_mola(monkeypatch):
    service, task, hypers, _model, _ls, _prepare, _static, topography_calls = (
        _setup_uploaded_inference(monkeypatch)
    )

    service._predict_uploaded_task_window(task, hypers, ls_start=0.0, horizon=2)

    assert topography_calls == []
