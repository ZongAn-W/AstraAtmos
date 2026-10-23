"""Earth task-test dispatch, bounds, ownership and shared inference coordination."""
import asyncio
import threading

import pytest
import torch

from test_earth_3hourly_prediction_routes import threehour_app, seed_task
from test_earth_3hourly_prediction_service import threehour_prediction_bundle, synthetic_training_release


def test_http_diagnostics_runs_real_earth_checkpoint_and_shares_cached_action(threehour_app, monkeypatch):
    import services.earth_diagnostics as diagnostics
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diagnostics, "EARTH_DIAGNOSTICS_CACHE", diagnostics.EarthDiagnosticsCache())
    task_id, _ = seed_task(threehour_app)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        response = threehour_app["client"].post("/api/earth/predict/diagnostics",
            json={"training_task_id": task_id, "sample_windows": 1, "scatter_points": 16,
                  "histogram_bins": 8, "pfi_repeats": 1, "include_pfi": False})
    finally:
        torch.set_num_threads(previous)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["planet"] == "earth" and result["target_unit"] == "DU"
    assert result["scope"]["split"] == "test"
    assert result["scope"]["window_count"] == 1
    assert result["scope"]["valid_points"] == 24 * 240 * 480
    assert sum(result["histogram"]["counts"]) == result["scope"]["valid_points"]
    assert len(result["scatter"]["reference"]) == 16
    assert result["full_test_metrics"]["metrics"]["overall"]["rmse"] == 1.


def test_context_dispatch_and_cache_are_shared_with_task_action(threehour_app, monkeypatch):
    import services.earth_diagnostics as diagnostics
    import routers.training as training

    task_id, _ = seed_task(threehour_app)
    client = threehour_app["client"]
    context = client.get("/api/earth/predict/diagnostics/context", params={"training_task_id": task_id})
    assert context.status_code == 200, context.text
    assert context.json()["available"] and context.json()["target_unit"] == "DU"
    calls = []

    def compute(task, registry, **parameters):
        calls.append(parameters)
        assert registry is threehour_app["bundle"].registry
        return {"planet": "earth", "task_id": task.id, "status": "completed", "target_unit": "DU"}

    async def forbidden(*args, **kwargs):
        pytest.fail("Earth action=test entered Mars inference")

    monkeypatch.setattr(diagnostics, "compute_earth_diagnostics", compute)
    monkeypatch.setattr(training.inference_service, "get_test_results", forbidden)
    monkeypatch.setattr(training.training_service, "prepare_task_inference_data_env", forbidden)
    action = client.post(f"/api/training/tasks/{task_id}/action", params={"action": "test"})
    direct = client.post("/api/earth/predict/diagnostics", json={"training_task_id": task_id})
    assert action.status_code == direct.status_code == 200
    assert action.json()["data"] == direct.json()
    assert calls[0] == calls[1]  # Same defaults => same verified core cache identity.


@pytest.mark.parametrize("parameters", [{"sample_windows": 9}, {"sample_windows": 0},
    {"scatter_points": 20001}, {"histogram_bins": 101}, {"pfi_repeats": 6},
    {"seed": -1}, {"sample_windows": True}, {"dataset_fingerprint": "fake"}])
def test_diagnostics_parameter_bounds(threehour_app, parameters):
    task_id, _ = seed_task(threehour_app)
    response = threehour_app["client"].post("/api/earth/predict/diagnostics",
        json={"training_task_id": task_id, **parameters})
    assert response.status_code == 422


def test_permission_retirement_and_invalid_checkpoint_precede_compute(threehour_app, monkeypatch):
    import services.earth_diagnostics as diagnostics
    task_id, _ = seed_task(threehour_app, user_id=2)
    retired_id, _ = seed_task(threehour_app, dataset_id="earth_merra2_daily_v2")
    damaged_id, path = seed_task(threehour_app)
    path.write_bytes(b"invalid checkpoint")
    client = threehour_app["client"]
    monkeypatch.setattr(diagnostics, "compute_earth_diagnostics", lambda *a, **k: pytest.fail("Unauthorized compute"))
    assert client.post("/api/earth/predict/diagnostics", json={"training_task_id": task_id}).status_code == 403
    response = client.post("/api/earth/predict/diagnostics", json={"training_task_id": retired_id})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "dataset_retired"
    response = client.get("/api/earth/predict/diagnostics/context", params={"training_task_id": damaged_id})
    assert response.status_code == 409
    assert response.json()["detail"]["message"]


def test_mars_action_test_keeps_its_existing_environment(threehour_app, monkeypatch):
    import routers.training as training
    task_id, _ = seed_task(threehour_app, dataset_id="openmars_mcd")
    calls = []

    async def prepare(task, **kwargs):
        calls.append("prepare")
        return {"ASTRAATMOS_OPENMARS_DIR": "verified"}, None

    async def compute(task_id, data_dirs=None):
        calls.append(data_dirs)
        return {"y_true": [1], "y_pred": [2], "metrics": {"rmse": 1}}

    monkeypatch.setattr(training.training_service, "prepare_task_inference_data_env", prepare)
    monkeypatch.setattr(training.inference_service, "get_test_results", compute)
    response = threehour_app["client"].post(f"/api/training/tasks/{task_id}/action", params={"action": "test"})
    assert response.status_code == 200 and response.json()["data"]["y_pred"] == [2]
    assert calls == ["prepare", {"ASTRAATMOS_OPENMARS_DIR": "verified"}]


def test_compute_slot_serializes_different_requests_after_cancellation():
    from services.inference_compute import run_inference_compute
    entered, release, second = threading.Event(), threading.Event(), threading.Event()

    def slow():
        entered.set()
        assert release.wait(5)

    async def verify():
        first = asyncio.create_task(run_inference_compute(lambda: asyncio.to_thread(slow)))
        assert await asyncio.to_thread(entered.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        other = asyncio.create_task(run_inference_compute(lambda: asyncio.to_thread(second.set)))
        await asyncio.sleep(.1)
        assert not second.is_set()
        release.set()
        await asyncio.wait_for(other, 5)
        assert second.is_set()

    asyncio.run(verify())
