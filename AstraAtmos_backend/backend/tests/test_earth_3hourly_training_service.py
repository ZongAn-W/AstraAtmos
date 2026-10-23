"""Three-hour tasks are queued with the server-selected contract and identity."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from services.dataset_identity import DatasetRequestError, EARTH_DATASET_3HOURLY_ID
from services.dataset_registry import DatasetRegistry


def binding():
    snapshot = {
        "dataset_id": EARTH_DATASET_3HOURLY_ID, "planet": "earth",
        "time": {"kind": "datetime", "step": 3, "step_unit": "hour", "time_zone": "UTC", "start": "2020-01-01T01:30:00Z",
                 "end": "2021-12-31T22:30:00Z", "count": 5848},
        "grid": {"shape": [240, 480]},
    }
    return {
        "dataset_id": EARTH_DATASET_3HOURLY_ID, "dataset_version": "v1",
        "dataset_fingerprint": "a" * 64, "dataset_identity_status": "verified",
        "dataset_snapshot": json.dumps(snapshot),
    }


class ServerRegistry:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.lookups = []

    def build_training_binding(self, dataset_id):
        assert dataset_id == EARTH_DATASET_3HOURLY_ID
        return binding()

    def get_earth_snapshot(self, dataset_id, expected_fingerprint=None):
        assert dataset_id == EARTH_DATASET_3HOURLY_ID
        assert expected_fingerprint == "a" * 64
        self.lookups.append((dataset_id, expected_fingerprint))
        return SimpleNamespace(data_path=self.data_path, dates=np.datetime64("2020-01-01T01:30", "ns") + np.arange(5848) * np.timedelta64(3, "h"))


@pytest.fixture
def service_environment(tmp_path, monkeypatch):
    import services.training_service as module
    from database.models import Base
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'threehour-tasks.db'}")

    async def prepare():
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(prepare())
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(module, "async_session_maker", sessions)
    logs, results = tmp_path / "logs", tmp_path / "results"
    logs.mkdir()
    results.mkdir()
    monkeypatch.setattr(module, "LOGS_DIR", logs)
    monkeypatch.setattr(module, "OUTPUT_MODELS_DIR", results)
    yield module, ServerRegistry(tmp_path / "server-package" / "earth_merra2_3hourly.nc")
    asyncio.run(engine.dispose())


def test_start_queues_56_to_24_with_only_the_server_data_path(service_environment):
    module, registry = service_environment
    service = module.TrainingService()
    service._scheduler_started = True

    async def scenario():
        return await service.start_training(
            user_id=1, custom_model_name="3-hour official", model_script="client_script.py",
            hyperparameters={"epochs": 1, "batch_size": 1, "selected_channels": ["SWGDN", "U10M"]},
            dataset_id=EARTH_DATASET_3HOURLY_ID, dataset_registry=registry,
        )

    task = asyncio.run(scenario())
    assert task.status == "queued"
    assert task.model_script == "earth_daily.py"
    assert task.dataset_id == EARTH_DATASET_3HOURLY_ID
    assert task.dataset_fingerprint == "a" * 64
    assert json.loads(task.dataset_snapshot)["time"]["kind"] == "datetime"
    hypers = json.loads(task.hyperparameters)
    assert (hypers["window"], hypers["horizon"]) == (56, 24)
    assert hypers["selected_channels"] == ["U10M", "SWGDN"]
    assert hypers["training_dataset"] == EARTH_DATASET_3HOURLY_ID
    assert hypers["_earth_metrics_schema"] == "earth_training_metrics_3hourly_v3"
    assert hypers["_earth_metrics_aggregation"] == "forecast_origin_lead_cell_area_weighted"
    assert "data_path" not in hypers
    plan = asyncio.run(service._prepare_training_execution(task, dataset_registry=registry))
    spec = plan["earth_training_spec"]
    assert spec["data_path"] == str(registry.data_path)
    assert spec["training_profile"]["frequency_hours"] == 3
    assert spec["training_profile"]["grid_shape"] == [240, 480]
    assert spec["dataset_binding"]["dataset_snapshot"] == task.dataset_snapshot
    assert spec["metrics_contract"] == {
        "schema": "earth_training_metrics_3hourly_v3",
        "aggregation": "forecast_origin_lead_cell_area_weighted",
    }
    assert registry.lookups == [(EARTH_DATASET_3HOURLY_ID, "a" * 64)] * 2


@pytest.mark.parametrize("source,upload_id,hypers", [
    ("official", None, {"model_architecture": "uploaded"}),
    ("official", None, {"model_source": "uploaded"}),
    ("unknown", None, {}),
])
def test_unsupported_models_stop_before_package_database_or_user_code(service_environment, monkeypatch, source, upload_id, hypers):
    module, _ = service_environment

    class ForbiddenRegistry:
        def build_training_binding(self, *_):
            pytest.fail("unsupported requests must not read a package")

    def forbidden_session():
        pytest.fail("unsupported requests must not open a database session")

    service = module.TrainingService()
    with monkeypatch.context() as admission_patch:
        admission_patch.setattr(module, "async_session_maker", forbidden_session)
        with pytest.raises(DatasetRequestError) as error:
            asyncio.run(service.start_training(
                user_id=1, custom_model_name="rejected", model_script="demo3.py",
                hyperparameters=hypers, model_source=source, uploaded_model_id=upload_id,
                dataset_id=EARTH_DATASET_3HOURLY_ID, dataset_registry=ForbiddenRegistry(),
            ))
    assert (error.value.status_code, error.value.code) == (409, "dataset_training_configuration_not_supported")
    assert asyncio.run(service.get_all_tasks()) == []


def test_missing_new_package_does_not_open_database_or_queue(service_environment, tmp_path, monkeypatch):
    module, _ = service_environment

    def forbidden_session():
        pytest.fail("missing package requests must not open a database session")

    registry = DatasetRegistry(tmp_path / "daily", earth_3hourly_package_dir=tmp_path / "new")
    service = module.TrainingService()
    with monkeypatch.context() as admission_patch:
        admission_patch.setattr(module, "async_session_maker", forbidden_session)
        with pytest.raises(DatasetRequestError) as error:
            asyncio.run(service.start_training(
                user_id=1, custom_model_name="missing", model_script="demo3.py", hyperparameters={},
                dataset_id=EARTH_DATASET_3HOURLY_ID, dataset_registry=registry,
            ))
    assert (error.value.status_code, error.value.code) == (503, "dataset_unavailable")
    assert error.value.availability_reason == "package_missing"
    assert asyncio.run(service.get_all_tasks()) == []


def test_default_registry_uses_the_independent_threehour_environment_path(tmp_path, monkeypatch):
    import services.training_service as module

    monkeypatch.setattr(module, "EARTH_MERRA2_DIR", tmp_path / "daily-v2")
    monkeypatch.setattr(module, "EARTH_MERRA2_V1_DIR", tmp_path / "daily-v1")
    monkeypatch.setattr(module, "EARTH_MERRA2_3HOURLY_DIR", tmp_path / "threehour")
    registry = module.TrainingService._default_dataset_registry()
    assert registry._earth_specs[EARTH_DATASET_3HOURLY_ID]["path"] == tmp_path / "threehour"
    assert registry._earth_specs["earth_merra2_daily_v2"]["path"] == tmp_path / "daily-v2"
    assert registry._earth_specs["earth_merra2_daily_v1"]["path"] == tmp_path / "daily-v1"


def test_restarted_queued_task_retains_its_frozen_identity_and_windows(tmp_path, monkeypatch):
    import services.training_service as module

    registry = ServerRegistry(tmp_path / "verified.nc")
    service = module.TrainingService()
    monkeypatch.setattr(service, "_default_dataset_registry", lambda: registry)
    task = SimpleNamespace(
        id=19, **binding(), hyperparameters=json.dumps({
            "training_dataset": EARTH_DATASET_3HOURLY_ID, "window": 56, "horizon": 24,
            "selected_channels": [], "model_source": "official", "epochs": 1, "_data_source": "default",
        }),
    )
    spec = service._restore_3hourly_training_spec(task)
    assert spec["task_id"] == 19
    assert spec["dataset_binding"] == binding()
    assert (spec["hyperparameters"]["window"], spec["hyperparameters"]["horizon"]) == (56, 24)
    assert spec["data_path"] == str(registry.data_path)
    assert "_data_source" not in spec["hyperparameters"]


@pytest.mark.parametrize("field", ["dataset_snapshot", "dataset_fingerprint", "dataset_version", "dataset_identity_status"])
def test_restarted_task_refuses_missing_identity_before_loading_data(tmp_path, monkeypatch, field):
    import services.training_service as module

    service = module.TrainingService()
    row = binding()
    row[field] = None
    task = SimpleNamespace(id=3, **row, hyperparameters=json.dumps({"training_dataset": EARTH_DATASET_3HOURLY_ID}))

    def forbidden_registry():
        pytest.fail("unverified queued tasks must not load a dataset")

    monkeypatch.setattr(service, "_default_dataset_registry", forbidden_registry)
    with pytest.raises(DatasetRequestError) as error:
        service._restore_3hourly_training_spec(task)
    assert (error.value.status_code, error.value.code) == (409, "dataset_identity_not_verified")


def test_restarted_task_rejects_a_changed_package(tmp_path, monkeypatch):
    import services.training_service as module

    class ChangedRegistry:
        def get_earth_snapshot(self, dataset_id, expected_fingerprint):
            assert expected_fingerprint == "a" * 64
            raise DatasetRequestError("dataset_version_changed", "changed", status_code=409)

    service = module.TrainingService()
    monkeypatch.setattr(service, "_default_dataset_registry", ChangedRegistry)
    task = SimpleNamespace(id=7, **binding(), hyperparameters=json.dumps({"training_dataset": EARTH_DATASET_3HOURLY_ID}))
    with pytest.raises(DatasetRequestError) as error:
        service._restore_3hourly_training_spec(task)
    assert error.value.code == "dataset_version_changed"
