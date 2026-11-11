"""The task-list API must preserve variable identity for old and new tasks."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.dependencies import get_current_user


def make_task(task_id, *, model_script="demo3.py", dataset_id="mcd_overview", hypers):
    return SimpleNamespace(
        id=task_id,
        user_id=1,
        model_script=model_script,
        model_source="official",
        uploaded_model_id=None,
        uploaded_model_version=None,
        status="completed",
        start_time=datetime.now(timezone.utc),
        queued_at=None,
        queue_position=None,
        end_time=None,
        hyperparameters=json.dumps(hypers),
        log_file_path=None,
        output_model_path=None,
        custom_model_name=f"task-{task_id}",
        metrics=None,
        progress=100,
        current_epoch=1,
        total_epochs=1,
        current_loss=None,
        eta=None,
        loss_history=None,
        dataset_id=dataset_id,
        dataset_version="v1",
        dataset_fingerprint="a" * 64,
        dataset_identity_status="verified",
        dataset_snapshot=None,
    )


@pytest.mark.parametrize(("task", "expected_target", "expected_inputs", "expected_unit"), [
    (make_task(1, hypers={"target_variable": "T", "selected_channels": '["U", "V"]'}),
     "T", ["O3", "U", "V"], "K"),
    (make_task(2, model_script="demo3-DT.py", hypers={}),
     "O3", ["O3", "D", "T"], "um-atm"),
    (make_task(6, model_script="demo3-DT.py", hypers={"selected_channels": ["legacy-unknown"]}),
     "O3", ["O3", "D", "T"], "um-atm"),
    (make_task(3, hypers={"target_variable": "T", "input_variables": '["U", "T", "O3"]'}),
     "T", ["O3", "U", "T"], "K"),
    (make_task(4, model_script="earth_daily.py", dataset_id="earth_merra2_3hourly_v1",
               hypers={"training_dataset": "earth_merra2_3hourly_v1",
                       "selected_channels": '["T2M", "U10M"]'}),
     "TO3", ["TO3", "U10M", "T2M"], "DU"),
    (make_task(5, model_script="earth_daily.py", dataset_id="earth_merra2_3hourly_v1",
               hypers={"training_dataset": "earth_merra2_3hourly_v1", "target_variable": "T2M",
                       "selected_channels": '["U10M"]'}),
     "T2M", ["TO3", "U10M"], "K"),
])
def test_task_list_http_serializes_variable_identity(monkeypatch, task, expected_target, expected_inputs, expected_unit):
    import routers.training as training_router

    async def get_all_tasks():
        return [task]

    async def get_task_tags(task_ids, user_id):
        return {task_id: [] for task_id in task_ids}

    monkeypatch.setattr(training_router.training_service, "get_all_tasks", get_all_tasks)
    monkeypatch.setattr(training_router.training_tag_service, "get_task_tags", get_task_tags)
    app = FastAPI()
    app.include_router(training_router.router, prefix="/api")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, role="user")

    with TestClient(app) as client:
        response = client.get("/api/training/tasks")

    assert response.status_code == 200, response.text
    [body] = response.json()
    assert body["target_variable"] == expected_target
    assert body["input_variables"] == expected_inputs
    assert body["target_unit"] == expected_unit
