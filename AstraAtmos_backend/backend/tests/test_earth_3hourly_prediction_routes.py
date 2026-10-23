"""Three-hour Earth HTTP errors, datetime DTOs, ownership and Mars isolation."""

import asyncio
import copy
import json
from datetime import datetime, timezone

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_earth_3hourly_prediction_service import (
    DATASET_ID, ORIGIN, threehour_prediction_bundle, synthetic_training_release,
)


@pytest.fixture
def threehour_app(threehour_prediction_bundle, tmp_path, monkeypatch):
    import routers.earth_predict as earth_router
    import routers.predict as mars_router
    import routers.training as training_router
    import services.earth_prediction_service as earth_service
    import services.inference_service as inference_service
    import services.training_service as training_service
    import services.training_tag_service as tag_service
    from auth.dependencies import get_current_user, get_optional_user
    from database.models import Base, User
    from services.earth_prediction_cache import EarthPredictionCache
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'threehour_routes.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def prepare():
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add_all([
                User(id=1, email="earth@example.com", username="earth", password_hash="x", role="user"),
                User(id=2, email="other@example.com", username="other", password_hash="x", role="user"),
            ])
            await session.commit()

    asyncio.run(prepare())
    monkeypatch.setattr(training_service, "async_session_maker", sessions)
    monkeypatch.setattr(inference_service, "async_session_maker", sessions)
    monkeypatch.setattr(tag_service, "async_session_maker", sessions)
    monkeypatch.setattr(earth_service, "EARTH_PREDICTION_CACHE", EarthPredictionCache())
    app = FastAPI()
    app.include_router(earth_router.router, prefix="/api")
    app.include_router(mars_router.router, prefix="/api")
    app.include_router(training_router.router, prefix="/api")
    app.state.dataset_registry = threehour_prediction_bundle.registry
    app.state.training_service = training_service.TrainingService()
    app.state.training_inference_service = inference_service.InferenceService()
    current = {"user": User(id=1, email="earth@example.com", username="earth", password_hash="x", role="user")}
    app.dependency_overrides[get_current_user] = lambda: current["user"]
    app.dependency_overrides[get_optional_user] = lambda: current["user"]
    with TestClient(app) as client:
        yield dict(client=client, app=app, sessions=sessions, bundle=threehour_prediction_bundle,
                   current=current, tmp=tmp_path)
    asyncio.run(engine.dispose())


def seed_task(env, *, status="completed", user_id=1, dataset_id=DATASET_ID, fingerprint=None, legacy_metrics=False):
    from database.models import ModelTrainingTask
    binding = env["bundle"].registry.build_training_binding(DATASET_ID)
    if fingerprint:
        binding["dataset_fingerprint"] = fingerprint
    binding["dataset_id"] = dataset_id

    async def insert():
        async with env["sessions"]() as session:
            row = ModelTrainingTask(
                user_id=user_id, model_script="earth_daily.py", model_source="official",
                custom_model_name="Threehour backtest", status=status,
                queued_at=datetime.now(timezone.utc),
                hyperparameters=json.dumps({"training_dataset": dataset_id, "window": 56,
                                            "horizon": 24, "selected_channels": ["U10M"]}), **binding,
            )
            session.add(row)
            await session.flush()
            payload = copy.deepcopy(env["bundle"].payload)
            if legacy_metrics:
                from test_earth_metrics_v2 import legacy_payload
                payload = legacy_payload(payload)
            payload["run"]["task_id"] = row.id
            path = env["tmp"] / f"threehour_task_{row.id}.pth"
            torch.save(payload, path)
            row.output_model_path = str(path)
            await session.commit()
            return row.id, path

    return asyncio.run(insert())


def test_mixed_v1_v2_comparison_preserves_missing_metrics_without_inference(threehour_app, monkeypatch):
    import services.earth_prediction_service as service
    ids = [seed_task(threehour_app, legacy_metrics=True)[0], seed_task(threehour_app)[0]]
    monkeypatch.setattr(service, "earth_forward_for_model", lambda *a, **k: pytest.fail("Comparison reran inference"))
    response = threehour_app["client"].post("/api/earth/predict/training-models/compare", json={"task_ids": ids})
    assert response.status_code == 200, response.text
    old, new = response.json()["items"]
    assert old["metrics"]["overall"] == {"rmse": 1.0, "mae": 1.0}
    assert "r2" not in old["metrics"]["per_step"][0]
    assert "metric_policy" not in old["metrics"]
    assert new["metrics"]["schema"] == "earth_training_metrics_3hourly_v2"
    assert new["metrics"]["overall"]["r2"] == 0


def test_http_context_preserves_utc_datetimes_and_publishes_complete_origins(threehour_app):
    task_id, _ = seed_task(threehour_app)
    response = threehour_app["client"].get("/api/earth/predict/context", params={"training_task_id": task_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["planet"] == "earth" and body["dataset_id"] == DATASET_ID
    assert body["frequency_hours"] == body["step"] == 3
    assert body["step_unit"] == "hour" and body["time_zone"] == "UTC"
    assert body["timestamp_rule"] == "interval_center"
    assert (body["window"], body["horizon"]) == (56, 24)
    assert body["target"] == "TO3" and body["target_unit"] == "DU"
    assert body["grid"]["shape"] == [240, 480]
    assert body["origins"]["count"] == 161
    assert body["origins"]["timestamps"][0] == "2020-01-07T22:30:00Z"
    assert body["origins"]["timestamps"][-1] == "2020-01-27T22:30:00Z"
    assert len(body["metrics"]["splits"]["test"]["by_lead"]) == 24


def test_earth_comparison_reads_verified_test_metrics_and_exports_without_inference(threehour_app, monkeypatch):
    from routers import research_export
    import services.earth_prediction_service as service
    from services.research_export_sources import EXPORT_SOURCES
    from services.research_figure import prepare_figure
    from schemas.research_export import ResearchExportRequest

    ids = [seed_task(threehour_app)[0] for _ in range(2)]
    monkeypatch.setattr(service, "earth_forward_for_model", lambda *a, **k: pytest.fail("Comparison reran inference"))
    response = threehour_app["client"].post("/api/earth/predict/training-models/compare", json={"task_ids": ids})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metric_source"] == "verified_checkpoint_test_metrics"
    assert [item["task_id"] for item in body["items"]] == ids
    refs = []
    for item in body["items"]:
        assert item["metrics"]["overall"]["rmse"] == item["metrics"]["overall"]["mae"] == 1.0
        assert set(item["metrics"]["overall"]) == {"mse", "rmse", "mae", "r2", "mape", "smape"}
        assert item["metrics"]["metric_units"]["mse"] == "DU^2"
        assert len(item["metrics"]["per_step"]) == 24
        assert item["metrics"]["split_meta"]["window_count"] == 1
        refs.append(item["metrics"]["export_ref"])
    request = ResearchExportRequest(kind="step_curves", metric="rmse", sources=refs,
                                    options={"format": "svg", "language": "en", "unit": "um-atm"})
    figure = prepare_figure([EXPORT_SOURCES.get(ref, 1) for ref in refs], request)
    assert figure["unit"] == "DU"
    assert len(figure["curves"]) == 2
    assert figure["curves"][0]["y"].tolist() == [1.0] * 24
    r2_figure = prepare_figure([EXPORT_SOURCES.get(ref, 1) for ref in refs],
                              request.model_copy(update={"metric": "r2"}))
    assert r2_figure["value_unit"] == "dimensionless"
    assert r2_figure["curves"][0]["y"].tolist() == [0.0] * 24
    threehour_app["app"].include_router(research_export.router, prefix="/api")
    exported = threehour_app["client"].post("/api/predict/research-export/download", json=request.model_dump())
    assert exported.status_code == 200, exported.text
    # A reference cannot be relabelled as another task's metrics.
    refs[0]["task_id"] = ids[1]
    assert threehour_app["client"].post("/api/predict/research-export/download", json={**request.model_dump(), "sources": refs}).status_code == 409


def test_earth_comparison_enforces_access_identity_and_selection(threehour_app):
    first, _ = seed_task(threehour_app)
    private, _ = seed_task(threehour_app, user_id=2)
    changed, _ = seed_task(threehour_app, fingerprint="c" * 64)
    client = threehour_app["client"]
    endpoint = "/api/earth/predict/training-models/compare"
    assert client.post(endpoint, json={"task_ids": [first]}).status_code == 422
    assert client.post(endpoint, json={"task_ids": [first, first]}).status_code == 422
    assert client.post(endpoint, json={"task_ids": [first, True]}).status_code == 422
    assert client.post(endpoint, json={"task_ids": [first, private]}).status_code == 403
    assert client.post(endpoint, json={"task_ids": [first, changed]}).status_code == 409
    mars, _ = seed_task(threehour_app, dataset_id="openmars_mcd")
    assert client.post(endpoint, json={"task_ids": [first, mars]}).status_code == 409


def test_earth_comparison_rejects_different_test_windows(threehour_app, monkeypatch):
    import services.earth_prediction_service as service
    first, _ = seed_task(threehour_app)
    second, _ = seed_task(threehour_app)
    original = service._load_checkpoint

    def altered_test_count(task):
        checkpoint = original(task)
        if task.id == second:
            checkpoint.metrics["splits"]["test"]["window_count"] += 1
        return checkpoint

    monkeypatch.setattr(service, "_load_checkpoint", altered_test_count)
    response = threehour_app["client"].post("/api/earth/predict/training-models/compare", json={"task_ids": [first, second]})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "earth_comparison_incompatible"


def test_http_run_returns_24_full_fields_and_hour_metrics(threehour_app, monkeypatch):
    # Only this route test serializes all global fields; other tests exercise the
    # public errors/context without retaining multiple large JSON responses.
    task_id, _ = seed_task(threehour_app)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        response = threehour_app["client"].post("/api/earth/predict/run", json={
            "training_task_id": task_id, "forecast_origin": ORIGIN,
        })
    finally:
        torch.set_num_threads(previous)
    assert response.status_code == 200, response.text[:1000]
    body = response.json()
    assert body["forecast_origin"] == ORIGIN and body["origin_split"] == "test"
    assert body["dataset_fingerprint"] == threehour_app["bundle"].task.dataset_fingerprint
    assert body["target_unit"] == "DU" and body["grid"]["shape"] == [240, 480]
    assert len(body["input_timestamps"]) == 56
    assert len(body["target_timestamps"]) == 24
    assert body["target_timestamps"][0] == "2020-01-21T04:30:00Z"
    assert body["target_timestamps"][-1] == "2020-01-24T01:30:00Z"
    assert len(body["cache_key"]) == 64
    for kind in ("prediction", "reference", "residual"):
        assert len(body[kind]) == 24
        assert all(len(entry["field"]) == 240 and len(entry["field"][0]) == 480 for entry in body[kind])
    assert [row["lead_step"] for row in body["metrics"]["by_lead"]] == list(range(1, 25))
    assert [row["lead_hours"] for row in body["metrics"]["by_lead"]] == list(range(3, 73, 3))
    assert [row["horizon_hours"] for row in body["metrics"]["by_horizon"]] == [24, 48, 72]
    assert all("lead_day" not in row for row in body["metrics"]["by_lead"])
    keys = {"mse", "rmse", "mae", "r2", "mape", "smape"}
    assert set(body["metrics"]["overall"]) == keys
    for row in [*body["metrics"]["by_lead"], *body["metrics"]["by_horizon"]]:
        assert keys <= row.keys()
    assert body["metrics"]["metric_units"]["mse"] == "DU^2"
    assert body["metrics"]["schema"] == "earth_training_metrics_3hourly_v2"
    # Export the already returned synthetic result; prohibit additional inference.
    from routers import research_export, earth_predict
    from io import BytesIO
    from zipfile import ZipFile
    from scipy.io import loadmat
    import numpy as np
    threehour_app["app"].include_router(research_export.router, prefix="/api")
    monkeypatch.setattr(earth_predict, "run_earth_prediction", lambda *args: pytest.fail("Export reran inference"))
    payload = {"sources": [body["export_ref"]], "kind": "triptych", "step": 23, "bundle": True,
               "options": {"format": "pdf", "language": "en", "unit": "um-atm", "height_mm": 60}}
    exported = threehour_app["client"].post("/api/predict/research-export/download", json=payload)
    assert exported.status_code == 200, exported.text[:500] if exported.status_code != 200 else ""
    with ZipFile(BytesIO(exported.content)) as archive:
        mat = loadmat(BytesIO(archive.read("figure.mat")), simplify_cells=True)["figure_data"]
        assert mat["unit"] == "DU"  # Earth must never use Mars conversions.
        np.testing.assert_array_equal(mat["reference"], body["reference"][23]["field"])
        np.testing.assert_array_equal(mat["latitude"], body["grid"]["latitude"])
        np.testing.assert_allclose(mat["residual"], mat["prediction"] - mat["reference"], atol=1e-5)
        assert mat["target_time"] == body["target_timestamps"][23]
    payload["sources"][0]["origin"] = "2020-01-22T01:30:00Z"
    assert threehour_app["client"].post("/api/predict/research-export/download", json=payload).status_code == 409
    payload["sources"][0]["origin"] = body["forecast_origin"]
    threehour_app["current"]["user"].id = 2
    assert threehour_app["client"].post("/api/predict/research-export/download", json=payload).status_code == 403
    threehour_app["current"]["user"].id = 1
    from services.dataset_identity import DatasetRequestError
    import services.earth_prediction_service as earth_service
    def changed_release(*args):
        raise DatasetRequestError("dataset_version_changed", "Synthetic release changed", status_code=409)
    monkeypatch.setattr(earth_service, "build_prediction_context", changed_release)
    assert threehour_app["client"].post("/api/predict/research-export/download", json=payload).status_code == 409


@pytest.mark.parametrize("origin,code", [
    ("2020-01-21", "invalid_earth_prediction_origin"),
    ("2020-01-21T01:30:00", "invalid_earth_prediction_origin"),
    ("2020-01-21T02:30:00Z", "earth_prediction_origin_out_of_range"),
    ("2020-01-07T19:30:00Z", "earth_prediction_origin_out_of_range"),
    ("2020-01-28T01:30:00Z", "earth_prediction_origin_out_of_range"),
])
def test_http_origin_errors_have_stable_detail_codes(threehour_app, origin, code):
    task_id, _ = seed_task(threehour_app)
    response = threehour_app["client"].post("/api/earth/predict/run", json={
        "training_task_id": task_id, "forecast_origin": origin,
    })
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == code


@pytest.mark.parametrize("endpoint", ["context", "run"])
def test_http_dataset_change_is_409_for_context_and_run(threehour_app, endpoint):
    task_id, _ = seed_task(threehour_app, fingerprint="0" * 64)
    if endpoint == "context":
        response = threehour_app["client"].get("/api/earth/predict/context", params={"training_task_id": task_id})
    else:
        response = threehour_app["client"].post("/api/earth/predict/run", json={"training_task_id": task_id, "forecast_origin": ORIGIN})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_version_changed"


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_http_unfinished_task_is_stably_rejected(threehour_app, status):
    task_id, _ = seed_task(threehour_app, status=status)
    response = threehour_app["client"].get("/api/earth/predict/context", params={"training_task_id": task_id})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "earth_prediction_task_not_completed"


def test_http_invalid_checkpoint_is_stably_rejected(threehour_app):
    task_id, path = seed_task(threehour_app)
    path.write_bytes(b"invalid serialized artifact")
    response = threehour_app["client"].post("/api/earth/predict/run", json={"training_task_id": task_id, "forecast_origin": ORIGIN})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_earth_training_artifact"
    assert "traceback" not in response.text.lower()
    assert str(path) not in response.text


def test_http_missing_checkpoint_is_stably_rejected_without_path_details(threehour_app):
    task_id, path = seed_task(threehour_app)
    path.unlink()
    response = threehour_app["client"].post(
        "/api/earth/predict/run", json={"training_task_id": task_id, "forecast_origin": ORIGIN}
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_earth_training_artifact"
    assert "artifact" in detail["message"].lower()
    assert str(path) not in response.text
    assert "traceback" not in response.text.lower()


def test_http_unreadable_checkpoint_is_stably_rejected_without_internal_details(threehour_app, monkeypatch):
    task_id, path = seed_task(threehour_app)
    import services.earth_prediction_service as service

    def unreadable(*args, **kwargs):
        raise PermissionError(f"cannot read {path}")

    monkeypatch.setattr(service, "load_earth_training_artifact", unreadable)
    response = threehour_app["client"].post(
        "/api/earth/predict/run", json={"training_task_id": task_id, "forecast_origin": ORIGIN}
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_earth_training_artifact"
    assert str(path) not in response.text
    assert "traceback" not in response.text.lower()


def test_http_invalid_checkpoint_run_record_is_rejected_before_response_validation(threehour_app):
    task_id, path = seed_task(threehour_app)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["run"]["best_epoch"] = [1]
    torch.save(payload, path)
    response = threehour_app["client"].get("/api/earth/predict/context", params={"training_task_id": task_id})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_earth_training_artifact"


@pytest.mark.parametrize("endpoint", ["context", "run"])
def test_http_ownership_applies_to_both_endpoints(threehour_app, endpoint):
    task_id, _ = seed_task(threehour_app, user_id=2)
    if endpoint == "context":
        response = threehour_app["client"].get("/api/earth/predict/context", params={"training_task_id": task_id})
    else:
        response = threehour_app["client"].post("/api/earth/predict/run", json={"training_task_id": task_id, "forecast_origin": ORIGIN})
    assert response.status_code == 403


def test_http_unknown_task_returns_404_and_mars_task_is_409(threehour_app):
    response = threehour_app["client"].get("/api/earth/predict/context", params={"training_task_id": 987654})
    assert response.status_code == 404
    task_id, _ = seed_task(threehour_app, dataset_id="openmars_mcd")
    response = threehour_app["client"].get("/api/earth/predict/context", params={"training_task_id": task_id})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dataset_prediction_not_supported"


@pytest.mark.parametrize("endpoint", ["run", "metrics", "pfi", "compare", "compare_pfi"])
def test_threehour_tasks_are_409_on_all_mars_prediction_paths(threehour_app, endpoint):
    task_id, _ = seed_task(threehour_app)
    client = threehour_app["client"]
    if endpoint in ("run", "metrics"):
        response = client.post(f"/api/predict/{endpoint}", json={"training_task_id": task_id, "horizon": 24, "ls_start": 90})
    elif endpoint == "pfi":
        response = client.get("/api/predict/permutation-importance", params={"training_task_id": task_id, "horizon": 24})
    elif endpoint in ("compare", "compare_pfi"):
        other_id, _ = seed_task(threehour_app)
        path = "compare" if endpoint == "compare" else "compare-pfi"
        response = client.post(f"/api/predict/training-models/{path}", json={"task_ids": [task_id, other_id], "horizon": 24})
    else:
        response = client.post(f"/api/training/tasks/{task_id}/action", params={"action": "test"})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "dataset_prediction_not_supported"


def test_earth_source_task_cannot_enter_mars_transfer_start_route(threehour_app, monkeypatch):
    task_id, _ = seed_task(threehour_app)
    registry = threehour_app["app"].state.dataset_registry
    original_binding = registry.build_training_binding

    def binding(dataset_id):
        if dataset_id == "openmars_mcd":
            return {"dataset_id": dataset_id, "dataset_version": "v1", "dataset_fingerprint": "a" * 64,
                    "dataset_identity_status": "verified", "dataset_snapshot": "{}"}
        return original_binding(dataset_id)

    monkeypatch.setattr(registry, "build_training_binding", binding)
    response = threehour_app["client"].post("/api/training/start", json={
        "model_script": "demo3.py", "model_name": "Mars rejected transfer", "dataset_id": "openmars_mcd",
        "hyperparameters": {"transfer_learning": True, "transfer_source_type": "task", "transfer_source_task_id": task_id},
    })
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "dataset_transfer_not_supported"
