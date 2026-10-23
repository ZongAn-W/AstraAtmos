"""HTTP contract for the Earth training and prediction routes.

These drive the real FastAPI routers with a temporary SQLite database so the
public status codes and error shapes are verified, not just the service layer.
"""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.dataset_registry import DatasetRegistry


@pytest.fixture
def earth_app(earth_global_release, tmp_path, monkeypatch):
    """A real router stack over a temporary database and Earth package."""
    import routers.datasets as datasets_router
    import routers.earth_predict as earth_predict_router
    import routers.predict as predict_router
    import routers.training as training_router
    import services.inference_service as inference_service
    import services.training_service as training_service
    from auth.dependencies import get_current_user, get_optional_user
    from database.models import Base, User
    from services.earth_prediction_service import build_prediction_context, run_earth_prediction
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'routes.db'}")

    async def prepare():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(prepare())
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(training_service, "async_session_maker", sessions)
    monkeypatch.setattr(inference_service, "async_session_maker", sessions)

    registry = DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **earth_global_release)

    user = User(id=1, email="earth@example.com", username="earth", password_hash="x", role="user")
    other = User(id=2, email="other@example.com", username="other", password_hash="x", role="user")

    async def seed():
        async with sessions() as session:
            session.add_all([user, other])
            await session.commit()

    asyncio.run(seed())

    app = FastAPI()
    app.include_router(training_router.router, prefix="/api")
    app.include_router(predict_router.router, prefix="/api")
    app.include_router(earth_predict_router.router, prefix="/api")
    app.include_router(datasets_router.router, prefix="/api")
    app.state.dataset_registry = registry
    app.state.training_service = training_service.TrainingService()
    # The Mars prediction routes read this service from app.state.
    app.state.training_inference_service = inference_service.InferenceService()

    current = {"user": user}
    app.dependency_overrides[get_current_user] = lambda: current["user"]
    # The Mars prediction routes authenticate optionally; override both so the
    # scene rejection is what gets exercised, not the auth guard.
    app.dependency_overrides[get_optional_user] = lambda: current["user"]
    return {
        "app": app,
        "client": TestClient(app),
        "registry": registry,
        "sessions": sessions,
        "current": current,
        "user": user,
        "other": other,
        "tmp": tmp_path,
    }


def _seed_earth_task(env, *, status="completed", dataset_id="earth_merra2_daily_v2", user_id=1, artifact=None):
    """Insert a task row that the routes can read; returns its id."""
    import database.models as models

    registry = env["registry"]
    binding = registry.build_training_binding("earth_merra2_daily_v2")

    async def insert():
        async with env["sessions"]() as session:
            row = models.ModelTrainingTask(
                user_id=user_id,
                model_script="earth_daily.py" if dataset_id.startswith("earth") else "demo3.py",
                model_source="official",
                hyperparameters=json.dumps({
                    "training_dataset": dataset_id,
                    "selected_channels": ["U10M"],
                    "window": 7,
                    "horizon": 3,
                }),
                custom_model_name="Earth Route Task",
                status=status,
                output_model_path=str(artifact) if artifact else None,
                dataset_id=dataset_id,
                dataset_version=binding["dataset_version"] if dataset_id.startswith("earth") else None,
                dataset_fingerprint=binding["dataset_fingerprint"] if dataset_id.startswith("earth") else None,
                dataset_identity_status="verified" if dataset_id.startswith("earth") else None,
                dataset_snapshot=binding["dataset_snapshot"] if dataset_id.startswith("earth") else None,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.id

    return asyncio.run(insert())


def _real_artifact(env, task_id):
    from services.earth_training_artifact import (
        build_checkpoint_payload,
        build_metrics_block,
        normalization_from_release,
        save_earth_artifact_atomic,
    )
    from training_backbones.earth_daily_model import create_earth_forecaster

    registry = env["registry"]
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    order = ["TO3", "U10M"]
    payload = build_checkpoint_payload(
        model=create_earth_forecaster(order),
        input_channel_order=order,
        linear_hidden_layers=2,
        dataset_binding=binding,
        normalization=normalization_from_release(
            registry.get_earth_snapshot("earth_merra2_daily_v2"), order
        ),
        run={
            "task_id": task_id, "seed": 1, "optimizer": "Adam",
            "loss": "normalized_mse_grid_uniform", "best_epoch": 1,
            "epochs_completed": 1, "run_complete": True, "device": "cpu",
        },
        metrics=build_metrics_block(
            validation={"overall": {"rmse": 1.0, "mae": 0.5}, "by_lead": [
                {"lead_day": i, "rmse": 1.0, "mae": 0.5} for i in (1, 2, 3)]},
            test={"overall": {"rmse": 4.0, "mae": 2.0}, "by_lead": [
                {"lead_day": i, "rmse": 4.0, "mae": 2.0} for i in (1, 2, 3)]},
            validation_window_count=172,
            test_window_count=175,
        ),
        split_window_counts={"train": 357, "validation": 172, "test": 175},
        task_id=task_id,
    )
    target = env["tmp"] / f"task_{task_id}_Earth_Route_Task.pth"
    save_earth_artifact_atomic(payload, target)
    return target


# ── prediction routes ──────────────────────────────────────────────────────

def test_context_route_returns_origins_grid_and_units(earth_app):
    artifact = None
    task_id = _seed_earth_task(earth_app, artifact=earth_app["tmp"] / "later.pth")
    artifact = _real_artifact(earth_app, task_id)
    _update_artifact_path(earth_app, task_id, artifact)

    response = earth_app["client"].get(f"/api/earth/predict/context?training_task_id={task_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["planet"] == "earth"
    assert body["dataset_id"] == "earth_merra2_daily_v2"
    assert body["target_unit"] == "DU"
    assert (body["window"], body["horizon"]) == (7, 3)
    assert body["grid"]["shape"] == [36, 72]
    assert len(body["grid"]["latitude"]) == 36
    assert body["origins"]["start"] == "2020-01-07"
    assert body["origins"]["end"] == "2021-12-28"
    assert body["origins"]["count"] == 722
    assert body["metrics"]["splits"]["test"]["window_count"] == 175


def test_run_route_returns_three_du_days_and_metrics(earth_app):
    task_id = _seed_earth_task(earth_app)
    artifact = _real_artifact(earth_app, task_id)
    _update_artifact_path(earth_app, task_id, artifact)

    response = earth_app["client"].post(
        "/api/earth/predict/run",
        json={"training_task_id": task_id, "forecast_origin": "2021-07-08"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["forecast_origin"] == "2021-07-08"
    assert body["origin_split"] == "test"
    assert body["target_dates"] == ["2021-07-09", "2021-07-10", "2021-07-11"]
    assert body["target_unit"] == "DU"
    for kind in ("prediction", "reference", "residual"):
        assert len(body[kind]) == 3
        assert len(body[kind][0]["field"]) == 36
        assert len(body[kind][0]["field"][0]) == 72
    assert body["metrics"]["unit"] == "DU"
    assert body["metrics"]["reference_available"] is True


def test_out_of_range_origin_returns_422_with_a_stable_code(earth_app):
    task_id = _seed_earth_task(earth_app)
    artifact = _real_artifact(earth_app, task_id)
    _update_artifact_path(earth_app, task_id, artifact)

    response = earth_app["client"].post(
        "/api/earth/predict/run",
        json={"training_task_id": task_id, "forecast_origin": "2021-12-31"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "earth_prediction_origin_out_of_range"


def test_daily_task_still_rejects_a_datetime_origin(earth_app):
    task_id = _seed_earth_task(earth_app)
    artifact = _real_artifact(earth_app, task_id)
    _update_artifact_path(earth_app, task_id, artifact)
    response = earth_app["client"].post(
        "/api/earth/predict/run",
        json={"training_task_id": task_id, "forecast_origin": "2021-07-08T01:30:00Z"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_earth_prediction_origin"


def test_changed_fingerprint_returns_409(earth_app):
    task_id = _seed_earth_task(earth_app)
    artifact = _real_artifact(earth_app, task_id)
    # Keep the artifact bound to the real release but point the task at a stale one.
    _update_artifact_path(earth_app, task_id, artifact, fingerprint="0" * 64)

    response = earth_app["client"].post(
        "/api/earth/predict/run",
        json={"training_task_id": task_id, "forecast_origin": "2021-07-08"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_version_changed"


def test_incomplete_task_returns_409(earth_app):
    task_id = _seed_earth_task(earth_app, status="running")
    response = earth_app["client"].get(f"/api/earth/predict/context?training_task_id={task_id}")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "earth_prediction_task_not_completed"


def test_another_users_task_returns_403(earth_app):
    task_id = _seed_earth_task(earth_app, user_id=2)
    response = earth_app["client"].get(f"/api/earth/predict/context?training_task_id={task_id}")
    assert response.status_code == 403


def test_unknown_task_returns_404(earth_app):
    response = earth_app["client"].get("/api/earth/predict/context?training_task_id=987654")
    assert response.status_code == 404


def test_mars_task_on_the_earth_route_returns_409(earth_app):
    task_id = _seed_earth_task(earth_app, dataset_id="openmars_mcd")
    response = earth_app["client"].get(f"/api/earth/predict/context?training_task_id={task_id}")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_prediction_not_supported"


def test_damaged_artifact_returns_409(earth_app):
    task_id = _seed_earth_task(earth_app)
    broken = earth_app["tmp"] / "broken.pth"
    broken.write_bytes(b"definitely not a checkpoint")
    _update_artifact_path(earth_app, task_id, broken)
    response = earth_app["client"].post(
        "/api/earth/predict/run",
        json={"training_task_id": task_id, "forecast_origin": "2021-07-08"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_earth_training_artifact"


def test_run_route_requires_a_well_formed_body(earth_app):
    response = earth_app["client"].post("/api/earth/predict/run", json={})
    assert response.status_code == 422


# ── Mars routes keep rejecting Earth ───────────────────────────────────────

def test_mars_predict_route_rejects_an_earth_task_with_409(earth_app):
    task_id = _seed_earth_task(earth_app)
    artifact = _real_artifact(earth_app, task_id)
    _update_artifact_path(earth_app, task_id, artifact)
    response = earth_app["client"].post(
        "/api/predict/run",
        json={"training_task_id": task_id, "horizon": 3, "ls_start": 90},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "dataset_prediction_not_supported"


def test_mars_metrics_route_rejects_an_earth_task_with_409(earth_app):
    task_id = _seed_earth_task(earth_app)
    artifact = _real_artifact(earth_app, task_id)
    _update_artifact_path(earth_app, task_id, artifact)
    response = earth_app["client"].post(
        "/api/predict/metrics",
        json={"training_task_id": task_id, "horizon": 3, "ls_start": 90},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_prediction_not_supported"


def test_training_action_test_rejects_an_earth_task_before_data_prep(earth_app):
    task_id = _seed_earth_task(earth_app)
    artifact = _real_artifact(earth_app, task_id)
    _update_artifact_path(earth_app, task_id, artifact)
    response = earth_app["client"].post(f"/api/training/tasks/{task_id}/action?action=test")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "dataset_prediction_not_supported"


def test_earth_task_response_exposes_scene_flags(earth_app):
    task_id = _seed_earth_task(earth_app)
    response = earth_app["client"].get("/api/training/tasks")
    assert response.status_code == 200
    task = next(item for item in response.json() if item["id"] == task_id)
    assert task["is_earth_task"] is True
    # The Mars prediction paths do not accept this task.
    assert task["trained_prediction_supported"] is False
    assert task["dataset_id"] == "earth_merra2_daily_v2"
    assert task["dataset_identity_status"] == "verified"


def test_mars_task_response_keeps_the_old_flags(earth_app):
    task_id = _seed_earth_task(earth_app, dataset_id="openmars_mcd")
    response = earth_app["client"].get("/api/training/tasks")
    task = next(item for item in response.json() if item["id"] == task_id)
    assert task["is_earth_task"] is False
    assert task["trained_prediction_supported"] is True


# ── training start route ───────────────────────────────────────────────────

def test_start_route_rejects_unsupported_earth_configuration(earth_app):
    response = earth_app["client"].post(
        "/api/training/start",
        json={
            "model_script": "demo3.py",
            "model_name": "Route Earth Reject",
            "model_source": "official",
            "dataset_id": "earth_merra2_daily_v2",
            "hyperparameters": {
                "training_dataset": "earth_merra2_daily_v2",
                "model_architecture": "simvp",
                "selected_channels": ["U10M"],
            },
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_training_configuration_not_supported"


def test_start_route_rejects_client_supplied_identity(earth_app):
    response = earth_app["client"].post(
        "/api/training/start",
        json={
            "model_script": "demo3.py",
            "model_name": "Route Earth Identity",
            "dataset_id": "earth_merra2_daily_v2",
            "dataset_fingerprint": "x" * 64,
            "hyperparameters": {"training_dataset": "earth_merra2_daily_v2"},
        },
    )
    assert response.status_code == 422


def test_start_route_rejects_unknown_dataset(earth_app):
    response = earth_app["client"].post(
        "/api/training/start",
        json={
            "model_script": "demo3.py",
            "model_name": "Route Unknown",
            "dataset_id": "not_a_dataset",
            "hyperparameters": {},
        },
    )
    # The id parser rejects an unregistered id as a client error before any task
    # row, upload load or subprocess is reached.
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unknown_dataset"


def test_dataset_catalog_publishes_the_earth_training_profile(earth_app):
    response = earth_app["client"].get("/api/datasets/earth_merra2_daily_v2")
    assert response.status_code == 200
    body = response.json()
    assert body["capabilities"]["training"] is True
    assert body["capabilities"]["trained_prediction"] is True
    profile = body["training_profile"]
    assert profile["profile_id"] == "earth_daily_dlinear_v1"
    assert profile["model_architectures"] == ["dlinear", "uploaded"]
    assert profile["model_sources"] == ["official", "uploaded"]
    assert (profile["window"], profile["horizon"]) == (7, 3)
    assert profile["target_unit"] == "DU"
    assert profile["optional_channels"] == ["U10M", "V10M", "T2M", "SWGDN"]
    assert body["availability"] == "available"


def test_mars_descriptor_has_no_training_profile(earth_app):
    response = earth_app["client"].get("/api/datasets/openmars_mcd")
    assert response.status_code == 200
    assert response.json()["training_profile"] is None


def _update_artifact_path(env, task_id, artifact, *, fingerprint=None):
    import database.models as models

    async def update():
        async with env["sessions"]() as session:
            row = await session.get(models.ModelTrainingTask, task_id)
            row.output_model_path = str(artifact)
            if fingerprint is not None:
                row.dataset_fingerprint = fingerprint
            await session.commit()

    asyncio.run(update())
