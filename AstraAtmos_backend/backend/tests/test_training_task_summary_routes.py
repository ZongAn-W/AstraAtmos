"""Task summaries keep selection metadata and permissions without frozen arrays."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.dependencies import get_current_user


def task_record(task_id, user_id, weight_path):
    return SimpleNamespace(
        id=task_id, user_id=user_id, model_script="demo3.py", model_source="official",
        uploaded_model_id=None, uploaded_model_version=None, status="completed",
        start_time=datetime(2026, 10, 1, tzinfo=timezone.utc), queued_at=None,
        queue_position=None, end_time=None,
        hyperparameters=json.dumps({"target_variable": "T", "input_variables": ["U", "T"]}),
        log_file_path=None, output_model_path=str(weight_path), custom_model_name=f"task-{task_id}",
        metrics='{"test":{"rmse":1.5}}', progress=100, current_epoch=1, total_epochs=1,
        current_loss=0.1, eta=None, loss_history='{"train":[0.1]}',
        dataset_id="mcd_overview", dataset_version="v1", dataset_fingerprint="a" * 64,
        dataset_identity_status="verified",
        dataset_snapshot=json.dumps({"normalization": {"input_means": [0.123456] * 20000}}),
    )


@pytest.fixture
def task_list_client(monkeypatch, tmp_path):
    import routers.training as training_router

    weight = tmp_path / "weights.pth"
    weight.write_bytes(b"non-empty weight artifact")
    tasks = [task_record(1, 1, weight), task_record(2, 2, weight)]
    calls = {"queries": 0, "tag_requests": []}
    user = SimpleNamespace(id=1, role="user")

    async def get_all_tasks():
        calls["queries"] += 1
        return tasks

    async def get_task_tags(task_ids, user_id):
        calls["tag_requests"].append((task_ids, user_id))
        return {task_id: [{"id": user_id, "name": f"private-tag-{user_id}"}] for task_id in task_ids}

    monkeypatch.setattr(training_router.training_service, "get_all_tasks", get_all_tasks)
    monkeypatch.setattr(training_router.training_tag_service, "get_task_tags", get_task_tags)
    app = FastAPI()
    app.include_router(training_router.router, prefix="/api")
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        yield client, tasks, calls, user, app


def test_summary_preserves_selection_identity_and_omits_snapshot(task_list_client):
    client, _, calls, _, _ = task_list_client
    full = client.get("/api/training/tasks")
    summary = client.get("/api/training/tasks?summary=true")
    assert full.status_code == summary.status_code == 200
    [expected] = full.json()
    expected["dataset_snapshot"] = None
    assert summary.json() == [expected]
    assert expected["model_available"] is True
    assert expected["target_variable"] == "T"
    assert expected["input_variables"] == ["U", "T"]
    assert expected["target_unit"] == "K"
    assert expected["dataset_fingerprint"] == "a" * 64
    assert len(summary.content) < len(full.content) / 20
    assert calls["tag_requests"] == [([1], 1), ([1], 1)]


def test_summary_skips_the_snapshot_before_schema_validation(task_list_client):
    client, tasks, _, _, _ = task_list_client

    class UnreadableSnapshot(SimpleNamespace):
        @property
        def dataset_snapshot(self):
            raise AssertionError("summary must not read or decode the snapshot")

    values = vars(tasks[0]).copy()
    values.pop("dataset_snapshot")
    tasks[0] = UnreadableSnapshot(**values)
    response = client.get("/api/training/tasks?summary=true")
    assert response.status_code == 200, response.text
    assert response.json()[0]["dataset_snapshot"] is None


def test_false_summary_retains_the_default_full_contract(task_list_client):
    client, _, _, _, _ = task_list_client
    default = client.get("/api/training/tasks")
    explicit_full = client.get("/api/training/tasks?summary=false")
    assert default.json() == explicit_full.json()
    assert len(default.json()[0]["dataset_snapshot"]["normalization"]["input_means"]) == 20000


def test_admin_summaries_keep_visible_tasks_and_account_private_tags(task_list_client):
    client, _, calls, user, _ = task_list_client
    user.role = "admin"
    response = client.get("/api/training/tasks?summary=true")
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [1, 2]
    assert all(item["tags"] == [{"id": 1, "name": "private-tag-1"}] for item in response.json())
    assert calls["tag_requests"] == [([1, 2], 1)]


def test_summary_still_requires_authentication(task_list_client):
    client, _, calls, _, app = task_list_client
    app.dependency_overrides.clear()
    response = client.get("/api/training/tasks?summary=true")
    assert response.status_code == 401
    assert calls["queries"] == 0


def test_invalid_summary_query_is_rejected(task_list_client):
    client, _, calls, _, _ = task_list_client
    response = client.get("/api/training/tasks?summary=maybe")
    assert response.status_code == 422
    assert calls["queries"] == 0
