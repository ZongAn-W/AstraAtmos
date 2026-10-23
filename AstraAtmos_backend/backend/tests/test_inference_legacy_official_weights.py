import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def load_inference_module():
    return importlib.import_module("services.inference_service")


class UncachedAnalysis:
    async def get_or_compute(self, *, compute, **kwargs):
        return await compute()


def build_legacy_task(tmp_path: Path):
    model_path = tmp_path / "legacy-official.pth"
    from core.predict_model import PredRNNv2

    legacy_model = PredRNNv2(
        input_dim=2,
        hidden_dims=[1],
        height=8,
        width=8,
        horizon=2,
    )
    torch.save(legacy_model.state_dict(), model_path)

    hypers = {
        "window": 3,
        "horizon": 2,
        "stlstm_hidden_dims": [1],
        "selected_channels": ["U"],
        "model_architecture": "predrnnv2",
    }
    task = SimpleNamespace(
        id=7,
        output_model_path=str(model_path),
        hyperparameters=json.dumps(hypers),
        custom_model_name="legacy-official",
        model_source="official",
        model_script="demo3-U.py",
    )
    return task, hypers


def stub_prediction_data():
    torch.manual_seed(0)
    x_torch = torch.randn(6, 3, 2, 8, 8)
    y_torch = torch.randn(6, 2, 1, 8, 8)
    ls_torch = torch.tensor(
        [
            [10.0, 15.0, 20.0],
            [25.0, 30.0, 35.0],
            [40.0, 45.0, 50.0],
            [55.0, 60.0, 65.0],
            [70.0, 75.0, 80.0],
            [85.0, 90.0, 95.0],
        ],
        dtype=torch.float32,
    )
    return x_torch, y_torch, ls_torch, 5.0, 2.0


def test_inference_service_uses_central_mcd_directory(tmp_path, monkeypatch):
    inference_module = load_inference_module()
    expected = tmp_path / "mcd"
    monkeypatch.setattr(inference_module, "MCD_DIR", expected)

    service = inference_module.InferenceService(analysis_cache=UncachedAnalysis())

    assert service.mcd_dir == expected


def test_predict_task_supports_legacy_official_weights(tmp_path):
    inference_module = load_inference_module()
    task, hypers = build_legacy_task(tmp_path)
    service = inference_module.InferenceService(analysis_cache=UncachedAnalysis())

    async def fake_prepare_task_prediction_context(
        task_id,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ):
        assert task_id == 7
        return task, hypers, {}, None

    service._prepare_task_prediction_context = fake_prepare_task_prediction_context
    service._prepare_data = lambda used_mcd_vars, window, horizon, **kwargs: stub_prediction_data()
    # The legacy loader normally supplies these axes with its window metadata.
    service._last_prepared_window_metadata = {
        "latitude": torch.linspace(87.5, -87.5, 8).numpy(),
        "longitude": torch.linspace(-160.0, 120.0, 8).numpy(),
    }

    result = asyncio.run(
        service.predict_task(
            task_id=7,
            ls_start=90.0,
            horizon=2,
            current_user=SimpleNamespace(id=7, role="user"),
        )
    )

    assert result["horizon"] == 2
    assert result["model_info"]["training_model_source"] == "official"
    assert result["selected_variables"] == ["U_Wind"]
    assert len(result["prediction"]) == 2
    assert result["metrics"]["aggregation"]["overall"] == "mean_over_forecast_steps"
    assert result["metrics"]["split_meta"]["source"] == "legacy_compatibility"


def test_task_test_set_metrics_supports_legacy_official_weights(tmp_path):
    inference_module = load_inference_module()
    task, hypers = build_legacy_task(tmp_path)
    service = inference_module.InferenceService(analysis_cache=UncachedAnalysis())

    async def fake_prepare_task_prediction_context(
        task_id,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ):
        assert task_id == 7
        return task, hypers, {}, None

    service._prepare_task_prediction_context = fake_prepare_task_prediction_context
    service._prepare_data = lambda used_mcd_vars, window, horizon, **kwargs: stub_prediction_data()

    result = asyncio.run(
        service.task_test_set_metrics(
            task_id=7,
            ls_start=90.0,
            horizon=2,
            current_user=SimpleNamespace(id=7, role="user"),
        )
    )

    assert result["overall"]["step"] == 0
    assert len(result["per_step"]) == 2
    assert result["aggregation"]["overall"] == "pooled_test_set_pixels"
    assert result["split_meta"]["source"] == "legacy_compatibility"
