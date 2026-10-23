"""Retired daily datasets cannot enter any application data workflow."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.dependencies import get_current_user
from config import resolve_default_earth_dataset_id
from routers.datasets import router as datasets_router
from routers.earth_overview import router as overview_router
from routers.earth_predict import router as predict_router
from services.dataset_identity import (
    DatasetRequestError, RETIRED_DATASET_IDS, is_earth_training_task,
    normalize_dataset_id, resolve_dataset_id,
)
from services.dataset_registry import DatasetRegistry
from services.earth_overview_service import EarthOverviewService
from services.earth_prediction_service import build_prediction_context, run_earth_prediction
from services.training_service import TrainingService


@pytest.fixture
def registry(tmp_path):
    return DatasetRegistry(earth_package_dir=tmp_path / "missing_daily")


def assert_retired(call):
    with pytest.raises(DatasetRequestError) as error:
        call()
    assert error.value.code == "dataset_retired"
    assert error.value.status_code == 409


def test_threehour_is_the_only_default_and_earth_catalog_entry(registry):
    assert resolve_default_earth_dataset_id(None) == "earth_merra2_3hourly_v1"
    assert resolve_default_earth_dataset_id(" EARTH_MERRA2_3HOURLY_V1 ") == "earth_merra2_3hourly_v1"
    for dataset_id in RETIRED_DATASET_IDS:
        with pytest.raises(ValueError):
            resolve_default_earth_dataset_id(dataset_id)
    items = registry.list_datasets()
    assert [item["dataset_id"] for item in items] == ["openmars_mcd", "mcd_overview", "earth_merra2_3hourly_v1"]
    assert items[-1]["availability"] == "missing"
    assert items[-1]["training_profile"]["window"] == 56


@pytest.mark.parametrize("dataset_id", RETIRED_DATASET_IDS)
def test_retired_id_is_still_recognized_for_historical_records(dataset_id):
    assert normalize_dataset_id(dataset_id) == dataset_id
    assert is_earth_training_task(SimpleNamespace(dataset_id=dataset_id))
    assert_retired(lambda: resolve_dataset_id(dataset_id, {}))
    assert_retired(lambda: resolve_dataset_id(None, {"training_dataset": dataset_id}))


@pytest.mark.parametrize("dataset_id", RETIRED_DATASET_IDS)
def test_retired_registry_rejects_before_opening_any_package(registry, dataset_id):
    assert_retired(lambda: registry.get_dataset(dataset_id))
    assert_retired(lambda: registry.build_training_binding(dataset_id))
    assert_retired(lambda: registry.get_earth_snapshot(dataset_id))
    assert_retired(lambda: registry.get_earth_overview_snapshot(dataset_id, "f" * 64))
    assert registry.verification_count == 0


@pytest.mark.parametrize("dataset_id", RETIRED_DATASET_IDS)
def test_training_rejects_before_registry_database_or_model_access(dataset_id, monkeypatch):
    service = TrainingService()

    def forbidden(*args, **kwargs):
        pytest.fail("Retired training reached data preparation")

    monkeypatch.setattr(service, "_default_dataset_registry", forbidden)
    assert_retired(lambda: asyncio.run(service.start_training(
        user_id=1, model_script="earth_daily.py", hyperparameters={},
        custom_model_name="retired", dataset_id=dataset_id,
    )))


@pytest.mark.parametrize("dataset_id", RETIRED_DATASET_IDS)
@pytest.mark.parametrize("legacy", [False, True])
def test_prediction_rejects_even_without_a_checkpoint_file(registry, dataset_id, legacy):
    task = SimpleNamespace(dataset_id=None if legacy else dataset_id, status="completed",
                           hyperparameters=json.dumps({"training_dataset": dataset_id}),
                           output_model_path="does-not-exist.pth")
    assert_retired(lambda: build_prediction_context(task, registry))
    assert_retired(lambda: run_earth_prediction(task, "2021-07-08", registry))


@pytest.mark.parametrize("dataset_id", RETIRED_DATASET_IDS)
@pytest.mark.parametrize("legacy", [False, True])
def test_http_catalog_overview_and_prediction_report_retirement(registry, dataset_id, legacy):
    task = SimpleNamespace(id=1, user_id=1, dataset_id=None if legacy else dataset_id, status="completed",
                           hyperparameters=json.dumps({"training_dataset": dataset_id}),
                           output_model_path="does-not-exist.pth")

    async def get_task(task_id):
        return task

    app = FastAPI()
    app.state.dataset_registry = registry
    app.state.earth_overview_service = EarthOverviewService(registry)
    app.state.training_service = SimpleNamespace(get_task=get_task)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, role="user")
    for router in (datasets_router, overview_router, predict_router):
        app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        catalog = client.get("/api/datasets").json()
        assert catalog["default_earth_dataset_id"] == "earth_merra2_3hourly_v1"
        responses = [
            client.get(f"/api/datasets/{dataset_id}"),
            client.get(f"/api/datasets/{dataset_id}/overview/field", params={
                "expected_fingerprint": "f" * 64, "date": "2021-07-08", "variable": "TO3",
            }),
            client.get("/api/earth/predict/context", params={"training_task_id": 1}),
            client.post("/api/earth/predict/run", json={"training_task_id": 1, "forecast_origin": "2021-07-08"}),
        ]
    for response in responses:
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "dataset_retired"


@pytest.mark.parametrize("dataset_id", RETIRED_DATASET_IDS)
@pytest.mark.parametrize("legacy", [False, True])
def test_queued_daily_task_cannot_launch_a_subprocess(tmp_path, monkeypatch, dataset_id, legacy):
    import services.training_service as module

    task = SimpleNamespace(dataset_id=None if legacy else dataset_id, status="queued",
                           hyperparameters=json.dumps({"training_dataset": dataset_id}))

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, *_):
            return task

        async def commit(self):
            pytest.fail("Retired task was marked running")

    def forbidden(*_, **__):
        pytest.fail("Retired task launched a subprocess")

    monkeypatch.setattr(module, "async_session_maker", Session)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    log_file = tmp_path / "retired.log"
    assert_retired(lambda: asyncio.run(TrainingService()._run_training_subprocess(
        1, "earth_daily.py", {}, log_file, tmp_path / "retired.pth",
    )))
    assert task.status == "queued"
    assert not log_file.exists()
