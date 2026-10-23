import asyncio
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from services import inference_service as inference_module
from services.inference_service import InferenceService


def _metric(value):
    return {
        "overall": {
            "step": 0,
            "rmse": value,
            "mae": value,
            "ssim": 0.5,
            "r2": 0.5,
        },
        "per_step": [],
    }


def test_metric_contract_replaces_stale_cached_aggregation():
    service = InferenceService()
    metrics = _metric(1.0)
    metrics["aggregation"] = {
        "overall": "mean_over_forecast_steps",
        "per_step": "per_forecast_step",
    }

    normalized = service._with_test_set_contract(metrics, {})

    assert normalized["aggregation"]["overall"] == "pooled_test_set_pixels"
    assert normalized["split_meta"]["source"] == "legacy_compatibility"


class RecordingCache:
    def __init__(self):
        self.calls = []

    async def get_or_compute(self, **kwargs):
        self.calls.append(
            {
                key: value
                for key, value in kwargs.items()
                if key != "compute"
            }
        )
        value = kwargs["compute"]()
        return await value if inspect.isawaitable(value) else value


def test_single_trained_analyses_use_effective_cache_parameters(monkeypatch):
    async def run():
        cache = RecordingCache()
        service = InferenceService(analysis_cache=cache)
        task = SimpleNamespace(
            id=12,
            user_id=7,
            model_source="official",
            custom_model_name="A",
        )
        hypers = {"horizon": 3, "selected_channels": ["U", "T"]}

        async def prepare(**kwargs):
            return task, hypers, {}, None

        def prediction(**kwargs):
            return {
                "ground_truth": [],
                "prediction": [],
                "residual": [],
                "selected_variables": ["U_Wind", "Temperature"],
                "horizon": kwargs["horizon"],
                "ls_values": [],
                "model_info": {"training_task_id": task.id},
                "metrics": _metric(1.0),
            }

        monkeypatch.setattr(
            service, "_prepare_task_prediction_context", prepare
        )
        monkeypatch.setattr(
            service, "_predict_task_with_context", prediction
        )
        monkeypatch.setattr(
            service,
            "_official_task_test_set_metrics",
            lambda *args, **kwargs: _metric(1.0),
        )
        monkeypatch.setattr(
            service,
            "_official_task_test_set_arrays",
            lambda *args, **kwargs: (
                np.zeros((1, 3, 2, 2)),
                np.ones((1, 3, 2, 2)),
                3,
            ),
        )
        monkeypatch.setattr(
            service,
            "_task_permutation_importance_with_context",
            lambda **kwargs: {
                "items": [],
                "baseline_metric": "r2",
                "baseline_value": 0.5,
            },
        )
        user = SimpleNamespace(id=7, role="user")
        prediction_result = await service.predict_task(
            12, 90.0, 3, current_user=user
        )
        test_metrics = await service.task_test_set_metrics(
            12, 90.0, 3, current_user=user
        )
        await service.task_error_distribution(
            12, ["Temperature"], 3, current_user=user
        )
        await service.task_permutation_importance(
            12,
            ["U_Wind", "Temperature"],
            90.0,
            3,
            current_user=user,
        )
        return cache.calls, prediction_result, test_metrics

    calls, prediction_result, test_metrics = asyncio.run(run())
    assert [call["analysis_type"] for call in calls] == [
        "prediction",
        "metrics",
        "error_distribution",
        "pfi",
    ]
    assert prediction_result["metrics"]["aggregation"]["overall"] == "mean_over_forecast_steps"
    assert test_metrics["aggregation"]["overall"] == "pooled_test_set_pixels"
    assert test_metrics["split_meta"]["source"] == "legacy_compatibility"


def test_comparisons_reuse_per_task_cached_primitives(monkeypatch):
    async def run():
        cache = RecordingCache()
        service = InferenceService(analysis_cache=cache)
        tasks = {
            task_id: SimpleNamespace(
                id=task_id,
                user_id=7,
                model_source="official",
                custom_model_name=f"Model {task_id}",
            )
            for task_id in (12, 18)
        }
        hypers = {"horizon": 3, "selected_channels": ["U"]}

        async def prepare(task_id, **kwargs):
            return tasks[task_id], hypers, {}, None

        monkeypatch.setattr(
            service, "_prepare_task_prediction_context", prepare
        )
        monkeypatch.setattr(
            service,
            "_official_task_test_set_metrics",
            lambda *args, **kwargs: _metric(1.0),
        )
        monkeypatch.setattr(
            service,
            "_official_task_test_set_arrays",
            lambda *args, **kwargs: (
                np.zeros((1, 3, 2, 2)),
                np.ones((1, 3, 2, 2)),
                3,
            ),
        )
        monkeypatch.setattr(
            service,
            "_task_permutation_importance_with_context",
            lambda **kwargs: {
                "items": [],
                "baseline_metric": "r2",
                "baseline_value": 0.5,
            },
        )
        monkeypatch.setattr(
            service,
            "_task_selected_channels",
            lambda task, task_hypers: ["U"],
        )
        monkeypatch.setattr(
            service,
            "_task_compare_metadata",
            lambda task, task_hypers: {
                "task_id": task.id,
                "model_name": task.custom_model_name,
            },
        )
        user = SimpleNamespace(id=7, role="user")
        metrics = await service.compare_task_test_set_metrics(
            [12, 18], 3, current_user=user
        )
        errors = await service.compare_task_error_distributions(
            [12, 18], 3, current_user=user
        )
        pfi = await service.compare_task_permutation_importance(
            [12, 18], 3, current_user=user
        )
        return cache.calls, metrics, errors, pfi

    calls, metrics, errors, pfi = asyncio.run(run())
    assert [call["analysis_type"] for call in calls] == [
        "metrics",
        "metrics",
        "error_distribution",
        "error_distribution",
        "pfi",
        "pfi",
    ]
    assert [item["task_id"] for item in metrics["items"]] == [12, 18]
    assert [item["task_id"] for item in errors["items"]] == [12, 18]
    assert [item["task_id"] for item in pfi["items"]] == [12, 18]


def test_unauthenticated_context_is_rejected_before_database_access(
    monkeypatch,
):
    def fail_sessionmaker():
        raise AssertionError("database access must follow authentication")

    monkeypatch.setattr(
        inference_module, "async_session_maker", fail_sessionmaker
    )
    service = InferenceService()

    with pytest.raises(PermissionError, match="Authentication is required"):
        asyncio.run(
            service._prepare_task_prediction_context(
                task_id=12, current_user=None
            )
        )
