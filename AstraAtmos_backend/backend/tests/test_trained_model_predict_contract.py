import asyncio
import sys
import types
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routers import predict  # noqa: E402
from schemas.predict import PredictRequest  # noqa: E402


AUTHENTICATED_USER = types.SimpleNamespace(id=7, role="user")


class FakeTrainingInferenceService:
    def __init__(self):
        self.prediction_calls = []
        self.metric_calls = []
        self.test_set_metric_calls = []
        self.compare_metric_calls = []
        self.compare_error_calls = []
        self.compare_pfi_calls = []
        self.error_distribution_calls = []
        self.pfi_calls = []

    async def predict_task(self, **kwargs):
        self.prediction_calls.append(kwargs)
        return {
            "ground_truth": [_field(1.0)],
            "prediction": [_field(2.0)],
            "residual": [_field(1.0)],
            "selected_variables": ["Temperature"],
            "horizon": 1,
            "ls_values": [95.0],
            "model_info": {
                "model_source": "trained_task",
                "training_task_id": kwargs["task_id"],
            },
            "metrics": {
                "overall": {"step": 0, "rmse": 1.0, "mae": 1.0, "ssim": 0.0, "r2": 0.0},
                "per_step": [{"step": 1, "rmse": 1.0, "mae": 1.0, "ssim": 0.0, "r2": 0.0}],
            },
        }

    async def task_metrics(self, **kwargs):
        self.metric_calls.append(kwargs)
        return {
            "overall": {"step": 0, "rmse": 1.0, "mae": 1.0, "ssim": 0.0, "r2": 0.0},
            "per_step": [{"step": 1, "rmse": 1.0, "mae": 1.0, "ssim": 0.0, "r2": 0.0}],
        }

    async def task_test_set_metrics(self, **kwargs):
        self.test_set_metric_calls.append(kwargs)
        return {
            "overall": {"step": 0, "rmse": 3.47, "mae": 2.46, "ssim": 0.49, "r2": 0.64},
            "per_step": [{"step": 1, "rmse": 3.47, "mae": 2.46, "ssim": 0.49, "r2": 0.64}],
        }

    async def compare_task_test_set_metrics(self, **kwargs):
        self.compare_metric_calls.append(kwargs)
        return {
            "items": [
                {
                    "task_id": 12,
                    "model_name": "model-a",
                    "model_source": "official",
                    "architecture": "predrnnv2",
                    "selected_channels": ["U", "V", "D"],
                    "hyperparameters": {"window": 3, "horizon": 3, "batch_size": 16},
                    "metrics": {
                        "overall": {"step": 0, "rmse": 3.47, "mae": 2.46, "ssim": 0.49, "r2": 0.64},
                        "per_step": [{"step": 1, "rmse": 3.47, "mae": 2.46, "ssim": 0.49, "r2": 0.64}],
                    },
                },
                {
                    "task_id": 18,
                    "model_name": "model-b",
                    "model_source": "uploaded",
                    "architecture": "custom",
                    "selected_channels": ["T"],
                    "hyperparameters": {"window": 4, "horizon": 3, "batch_size": 8},
                    "metrics": {
                        "overall": {"step": 0, "rmse": 2.9, "mae": 2.1, "ssim": 0.52, "r2": 0.7},
                        "per_step": [{"step": 1, "rmse": 2.9, "mae": 2.1, "ssim": 0.52, "r2": 0.7}],
                    },
                },
            ]
        }

    async def compare_task_error_distributions(self, **kwargs):
        self.compare_error_calls.append(kwargs)
        return {"items": []}

    async def compare_task_permutation_importance(self, **kwargs):
        self.compare_pfi_calls.append(kwargs)
        return {"items": []}

    async def task_permutation_importance(self, **kwargs):
        self.pfi_calls.append(kwargs)
        return {
            "items": [
                {"name": "Ozone", "importance": 0.55},
                {"name": "Temperature", "importance": 0.25},
            ],
            "baseline_metric": "r2",
            "baseline_value": 0.5,
        }

    async def task_error_distribution(self, **kwargs):
        self.error_distribution_calls.append(kwargs)
        return {
            "scatter": {"trues": [1.0, 2.0], "preds": [1.5, 2.5], "density": [1.0, 1.0]},
            "hist_trues": {"bin_edges": [1.0, 2.0], "counts": [2]},
            "hist_preds": {"bin_edges": [1.5, 2.5], "counts": [2]},
            "hist_errors": {"bin_edges": [0.5, 0.6], "counts": [2]},
            "mae": 0.5,
            "rmse": 0.5,
        }


def _field(value):
    return {
        "points": [{"lat": 0.0, "lng": 0.0, "val": value}],
        "lat": [0.0],
        "lon": [0.0],
        "field": [[value]],
        "minVal": value,
        "maxVal": value,
    }


def _request(service=None):
    state = type("State", (), {})()
    if service is not None:
        state.training_inference_service = service
    app = type("App", (), {"state": state})()
    return type("Request", (), {"app": app})()


def test_predict_run_rejects_a_request_without_task_id():
    body = PredictRequest(selected_variables=["U_Wind"], horizon=3, ls_start=90)

    with pytest.raises(predict.HTTPException) as exc:
        asyncio.run(
            predict.run_prediction(
                _request(),
                body,
                current_user=None,
            )
        )

    assert exc.value.status_code == 400
    assert "training_task_id" in exc.value.detail


def test_predict_request_has_no_public_mars_year_field():
    body = PredictRequest.model_validate({"training_task_id": 42, "ls_start": 90, "mars_year": 27})
    assert "mars_year" not in body.model_fields
    assert "mars_year" not in body.model_dump()


def test_predict_run_uses_training_task_inference_when_task_id_is_present():
    service = FakeTrainingInferenceService()
    body = PredictRequest(training_task_id=42, selected_variables=["U_Wind"], horizon=3, ls_start=90)

    payload = asyncio.run(
        predict.run_prediction(
            _request(service), body, current_user=AUTHENTICATED_USER
        )
    )

    assert payload["model_info"]["model_source"] == "trained_task"
    assert payload["model_info"]["training_task_id"] == 42
    assert payload["metrics"]["overall"]["rmse"] == 1.0
    assert service.prediction_calls[0]["task_id"] == 42
    assert service.prediction_calls[0]["current_user"] is AUTHENTICATED_USER
    assert "mars_year" not in service.prediction_calls[0]
    assert "mars_year" not in payload["model_info"]
    assert "mars_year" not in payload["source_meta"]


async def test_predict_metrics_uses_training_task_test_set_metrics_when_task_id_is_present():
    service = FakeTrainingInferenceService()
    body = PredictRequest(training_task_id=42, selected_variables=["U_Wind"], horizon=3, ls_start=90)

    payload = await predict.get_eval_metrics(
        _request(service), body, current_user=AUTHENTICATED_USER
    )

    assert payload["overall"]["rmse"] == 3.47
    assert service.test_set_metric_calls[0]["task_id"] == 42
    assert service.metric_calls == []
    assert "mars_year" not in service.test_set_metric_calls[0]
    assert "mars_year" not in payload["source_meta"]


async def test_trained_model_request_fails_when_training_inference_service_is_missing():
    body = PredictRequest(training_task_id=42, ls_start=90)
    try:
        await predict.run_prediction(
            _request(), body, current_user=AUTHENTICATED_USER
        )
    except predict.HTTPException as exc:
        assert exc.status_code == 500
        assert "training inference service unavailable" in exc.detail
    else:
        raise AssertionError("Expected HTTPException for missing training inference service")


def test_trained_model_request_returns_model_file_detail_when_weight_disappears():
    class MissingWeightInferenceService:
        async def predict_task(self, **kwargs):
            raise ValueError("Model file not found")

    body = PredictRequest(training_task_id=42, ls_start=90)

    try:
        asyncio.run(
            predict.run_prediction(
                _request(MissingWeightInferenceService()),
                body,
                current_user=AUTHENTICATED_USER,
            )
        )
    except predict.HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "Model file not found"
    else:
        raise AssertionError("Expected HTTPException for missing model weight")


def test_predict_metrics_wraps_unexpected_training_errors():
    class FailingMetricsInferenceService:
        async def task_test_set_metrics(self, **kwargs):
            raise RuntimeError("NetCDF: Can't open HDF5 attribute")

    body = PredictRequest(
        training_task_id=42,
        selected_variables=["U_Wind"],
        horizon=1,
        ls_start=90,
    )

    try:
        asyncio.run(
            predict.get_eval_metrics(
                _request(FailingMetricsInferenceService()),
                body,
                current_user=AUTHENTICATED_USER,
            )
        )
    except predict.HTTPException as exc:
        assert exc.status_code == 500
        assert "NetCDF: Can't open HDF5 attribute" in exc.detail
    else:
        raise AssertionError("Expected HTTPException for metrics failure")


async def test_permutation_importance_uses_training_task_service_when_task_id_is_present():
    service = FakeTrainingInferenceService()

    payload = await predict.get_permutation_importance(
        _request(service),
        vars="Temperature",
        training_task_id=42,
        ls_start=90,
        horizon=3,
        current_user=AUTHENTICATED_USER,
    )

    assert payload["items"][0]["name"] == "Ozone"
    assert service.pfi_calls[0]["task_id"] == 42
    assert service.pfi_calls[0]["selected_variables"] == ["Temperature"]
    assert "mars_year" not in service.pfi_calls[0]


async def test_error_distribution_uses_training_task_test_set_when_task_id_is_present():
    service = FakeTrainingInferenceService()

    payload = await predict.get_error_distribution(
        _request(service),
        vars="Temperature",
        training_task_id=42,
        horizon=3,
        current_user=AUTHENTICATED_USER,
    )

    assert payload["mae"] == 0.5
    assert service.error_distribution_calls[0]["task_id"] == 42
    assert service.error_distribution_calls[0]["selected_variables"] == ["Temperature"]


async def test_training_model_compare_uses_batch_test_set_metrics_service():
    service = FakeTrainingInferenceService()

    payload = await predict.compare_training_models(
        _request(service),
        predict.TrainingModelCompareRequest(task_ids=[12, 18], horizon=3),
        current_user=AUTHENTICATED_USER,
    )

    assert payload["items"][0]["task_id"] == 12
    assert payload["items"][1]["metrics"]["overall"]["r2"] == 0.7
    assert service.compare_metric_calls[0]["task_ids"] == [12, 18]
    assert service.compare_metric_calls[0]["horizon"] == 3
    assert service.compare_metric_calls[0]["current_user"] is AUTHENTICATED_USER


async def test_trained_model_analysis_requires_authenticated_user_before_service_call():
    service = FakeTrainingInferenceService()
    body = PredictRequest(training_task_id=42, horizon=3, ls_start=90)
    compare_body = predict.TrainingModelCompareRequest(task_ids=[12, 18], horizon=3)
    requests = (
        lambda: predict.run_prediction(_request(service), body, current_user=None),
        lambda: predict.get_eval_metrics(_request(service), body, current_user=None),
        lambda: predict.get_error_distribution(
            _request(service), vars="Temperature", training_task_id=42, current_user=None
        ),
        lambda: predict.get_permutation_importance(
            _request(service), vars="Temperature", training_task_id=42, current_user=None
        ),
        lambda: predict.compare_training_models(
            _request(service), compare_body, current_user=None
        ),
        lambda: predict.compare_training_model_error_distributions(
            _request(service), compare_body, current_user=None
        ),
        lambda: predict.compare_training_model_pfi(
            _request(service), compare_body, current_user=None
        ),
    )

    for call in requests:
        try:
            await call()
        except predict.HTTPException as exc:
            assert exc.status_code == 401
        else:
            raise AssertionError("Expected anonymous trained-model analysis to return 401")

    assert service.prediction_calls == []
    assert service.test_set_metric_calls == []
    assert service.error_distribution_calls == []
    assert service.pfi_calls == []
    assert service.compare_metric_calls == []
    assert service.compare_error_calls == []
    assert service.compare_pfi_calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--asyncio-mode=auto"]))
