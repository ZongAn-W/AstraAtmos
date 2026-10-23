import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from schemas.training import TrainingTaskResponse  # noqa: E402
from services import inference_service as inference_module  # noqa: E402
from services import training_service as training_module  # noqa: E402


ARTIFACT_ERROR = (
    "Training process exited successfully but no valid model weight file was produced"
)


class _EmptyStdout:
    def __init__(self, lines=None):
        self.lines = iter(lines or [])

    def readline(self):
        return next(self.lines, "")

    def close(self):
        return None


class _SuccessfulProcess:
    pid = 4321

    def __init__(self, lines=None):
        self.stdout = _EmptyStdout(lines)

    def wait(self):
        return 0


class _FailedProcess:
    pid = 9876

    def __init__(self, lines):
        self.stdout = _EmptyStdout(lines)

    def wait(self):
        return 1


class _TaskSession:
    def __init__(self, task, commit_side_effect=None):
        self.task = task
        self.commit_side_effect = commit_side_effect

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, _model, task_id):
        assert task_id == self.task.id
        return self.task

    async def commit(self):
        if self.commit_side_effect:
            raise self.commit_side_effect
        return None


class _BroadcastRecorder:
    def __init__(self, delay=0):
        self.messages = []
        self.delay = delay

    async def broadcast_to_task(self, task_id, message):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.messages.append((task_id, message))


def _run_successful_training(
    monkeypatch,
    tmp_path,
    artifact_bytes,
    *,
    output_is_directory=False,
    simulate_delayed_progress=False,
    hyperparameters=None,
    dataset_id=None,
):
    output_path = tmp_path / "weights.pth"
    if output_is_directory:
        output_path.mkdir()
    elif artifact_bytes is not None:
        output_path.write_bytes(artifact_bytes)

    log_file = tmp_path / "training.log"
    task = SimpleNamespace(
        id=17,
        status="pending",
        total_epochs=0,
        pid=None,
        loss_history=None,
        progress=100.0,
        end_time=None,
        metrics=None,
        hyperparameters=json.dumps(hyperparameters or {"epochs": 2}),
        dataset_id=dataset_id,
        dataset_fingerprint=None,
        dataset_identity_status="unversioned",
        dataset_snapshot=None,
    )
    broadcaster = _BroadcastRecorder(delay=0.1 if simulate_delayed_progress else 0)
    process_lines = ["final epoch\n"] if simulate_delayed_progress else None

    monkeypatch.setattr(training_module, "async_session_maker", lambda: _TaskSession(task))
    monkeypatch.setattr(
        training_module.subprocess,
        "Popen",
        lambda *args, **kwargs: _SuccessfulProcess(process_lines),
    )

    service = training_module.TrainingService()
    if simulate_delayed_progress:
        monkeypatch.setattr(
            service,
            "_parse_progress_from_log",
            lambda *args, **kwargs: {"progress": 100.0},
        )

        async def delayed_progress_update(_task_id, progress_data, _ws_manager):
            await asyncio.sleep(0.05)
            task.progress = progress_data["progress"]

        monkeypatch.setattr(service, "_update_task_progress", delayed_progress_update)

    from services import ws_manager

    monkeypatch.setattr(ws_manager, "manager", broadcaster)

    asyncio.run(
        service._run_training_subprocess(
            task_id=task.id,
            script_name="demo3.py",
            hyperparameters=hyperparameters or {"epochs": 2},
            log_file=log_file,
            output_path=output_path,
        )
    )
    return task, log_file, broadcaster


def test_successful_process_with_valid_weight_completes_task(monkeypatch, tmp_path):
    task, _log_file, broadcaster = _run_successful_training(
        monkeypatch, tmp_path, b"non-empty model weights"
    )

    assert task.status == "completed"
    assert task.progress == 100.0
    assert json.loads(task.metrics) == {"note": "completed"}
    assert broadcaster.messages[-1][1]["status"] == "completed"


def test_successful_process_without_weight_fails_task(monkeypatch, tmp_path):
    task, log_file, broadcaster = _run_successful_training(monkeypatch, tmp_path, None)

    assert task.status == "failed"
    assert task.progress < 100.0
    assert ARTIFACT_ERROR in json.loads(task.metrics)["error"]
    assert ARTIFACT_ERROR in log_file.read_text(encoding="utf-8")
    assert broadcaster.messages[-1][1]["status"] == "failed"


@pytest.mark.parametrize("fault", [None, "input_channel_order", "input_scale"])
def test_mars_checkpoint_normalization_controls_task_completion(monkeypatch, tmp_path, fault):
    import io
    import numpy as np
    import torch
    from services.mars_checkpoint import build_mars_checkpoint

    from services.mars_dataset_identity import canonical_dataset_identity
    identity = canonical_dataset_identity(training_dataset="openmars_mcd", openmars_dir=tmp_path / "openmars", mcd_dir=tmp_path)

    payload = build_mars_checkpoint(
        model_state_dict={"bias": torch.zeros(1)},
        volume=SimpleNamespace(
            input_means=(np.zeros((2, 3)),), input_stds=(np.ones((2, 3)),),
            y_mean=5.0, y_std=2.0, height=2, width=3,
            dataset_id="openmars_mcd", source_type="openmars_mcd",
            data_dir=tmp_path, data_directories=tuple(identity["data_directories"]),
            manifest=tuple(identity["file_manifest"]), fingerprint=identity["dataset_fingerprint"],
        ),
        selected_channels=[], window=3, horizon=3,
        split_ratios={"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}, seed=11,
    )
    if fault is not None:
        payload["normalization"][fault] = []
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    task, log_file, broadcaster = _run_successful_training(monkeypatch, tmp_path, buffer.getvalue())
    if fault is None:
        assert task.status == "completed"
        assert task.progress == 100.0
    else:
        assert task.status == "failed"
        assert "Mars checkpoint" in json.loads(task.metrics)["error"]
        assert "Mars checkpoint" in log_file.read_text(encoding="utf-8")
    assert broadcaster.messages[-1][1]["status"] == task.status


@pytest.mark.parametrize("dataset_id", ["mcd_overview", "openmars_mcd"])
@pytest.mark.parametrize("fault", [None, "data_directories", "file_manifest", "file_fingerprint", "schema"])
def test_mars_artifact_rebuilds_a_verified_compatible_task_snapshot(monkeypatch, tmp_path, dataset_id, fault):
    import io
    import numpy as np
    import torch
    from services.mars_checkpoint import build_mars_checkpoint
    from services.mars_data_service import identity_snapshot, assert_identity_current
    from services.mars_dataset_identity import canonical_dataset_identity

    directories = {name: tmp_path / name for name in ("raw", "openmars", "mcd")}
    for directory in directories.values():
        directory.mkdir()
    kwargs = dict(training_dataset=dataset_id, openmars_dir=directories["openmars"],
                  mcd_dir=directories["mcd"], raw_dir=directories["raw"])
    identity = canonical_dataset_identity(**kwargs)
    volume = SimpleNamespace(
        input_means=(np.zeros((2, 3)),), input_stds=(np.ones((2, 3)),),
        y_mean=5.0, y_std=2.0, height=2, width=3, split_idx=9,
        dataset_id=dataset_id, source_type=identity["source_type"], data_dir=Path(identity["data_dir"]),
        data_directories=tuple(identity["data_directories"]), manifest=tuple(identity["file_manifest"]),
        fingerprint=identity["dataset_fingerprint"], blocks=(), sample_starts=np.array([], dtype=np.int64),
    )
    ratios = {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}
    payload = build_mars_checkpoint(model_state_dict={"bias": torch.zeros(1)}, volume=volume,
                                    selected_channels=[], window=3, horizon=3, split_ratios=ratios, seed=11)
    if fault == "schema":
        payload.pop("schema")
    elif fault is not None:
        payload["data_binding"].pop(fault)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    task, log_file, _ = _run_successful_training(
        monkeypatch, tmp_path, buffer.getvalue(), dataset_id=dataset_id,
        hyperparameters={"training_dataset": dataset_id, "window": 3, "horizon": 3, "selected_channels": []},
    )
    if fault is None:
        assert task.status == "completed" and task.dataset_identity_status == "verified"
        snapshot = json.loads(task.dataset_snapshot)
        expected = identity_snapshot(**kwargs, selected_channels=[], window=3, horizon=3, split_ratios=ratios, normalization=volume)
        assert snapshot == expected
        current = assert_identity_current(snapshot, openmars_dir=directories["openmars"], mcd_dir=directories["mcd"], raw_dir=directories["raw"])
        assert task.dataset_fingerprint == current["dataset_fingerprint"]
    else:
        assert task.status == "failed" and task.dataset_identity_status != "verified"
        assert "Mars checkpoint" in log_file.read_text(encoding="utf-8")
        assert task.dataset_snapshot is None


def test_successful_process_with_empty_weight_fails_task(monkeypatch, tmp_path):
    task, log_file, _broadcaster = _run_successful_training(monkeypatch, tmp_path, b"")

    assert task.status == "failed"
    assert task.progress < 100.0
    assert ARTIFACT_ERROR in json.loads(task.metrics)["error"]
    assert ARTIFACT_ERROR in log_file.read_text(encoding="utf-8")


def test_successful_process_with_directory_output_fails_task(monkeypatch, tmp_path):
    task, _log_file, _broadcaster = _run_successful_training(
        monkeypatch,
        tmp_path,
        None,
        output_is_directory=True,
    )

    assert task.status == "failed"
    assert task.progress < 100.0
    assert ARTIFACT_ERROR in json.loads(task.metrics)["error"]


def test_invalid_artifact_final_state_wins_over_queued_progress_update(monkeypatch, tmp_path):
    task, _log_file, _broadcaster = _run_successful_training(
        monkeypatch,
        tmp_path,
        None,
        simulate_delayed_progress=True,
    )

    assert task.status == "failed"
    assert task.progress < 100.0


def _run_failed_training(
    monkeypatch,
    tmp_path,
    log_lines,
    notification_side_effect=None,
    notification_commit_side_effect=None,
    return_session_calls=False,
):
    output_path = tmp_path / "failed.pth"
    log_file = tmp_path / "failed.log"
    task = SimpleNamespace(
        id=23,
        user_id=5,
        custom_model_name="OOM model",
        status="pending",
        total_epochs=0,
        pid=None,
        loss_history=None,
        progress=0.0,
        end_time=None,
        metrics=None,
    )
    broadcaster = _BroadcastRecorder()
    session_calls = []

    def session_factory():
        session_calls.append(len(session_calls) + 1)
        commit_side_effect = (
            notification_commit_side_effect if len(session_calls) > 4 else None
        )
        return _TaskSession(task, commit_side_effect=commit_side_effect)

    monkeypatch.setattr(training_module, "async_session_maker", session_factory)
    monkeypatch.setattr(
        training_module.subprocess,
        "Popen",
        lambda *args, **kwargs: _FailedProcess(log_lines),
    )
    notification_writer = AsyncMock(
        side_effect=notification_side_effect,
        return_value=True,
    )
    monkeypatch.setattr(
        training_module,
        "ensure_cuda_oom_notification",
        notification_writer,
    )
    from services import ws_manager

    monkeypatch.setattr(ws_manager, "manager", broadcaster)

    asyncio.run(
        training_module.TrainingService()._run_training_subprocess(
            task_id=task.id,
            script_name="demo3.py",
            hyperparameters={"epochs": 2},
            log_file=log_file,
            output_path=output_path,
        )
    )
    result = (task, broadcaster, notification_writer)
    if return_session_calls:
        return (*result, session_calls)
    return result


def test_cuda_oom_failure_records_metrics_and_notification(monkeypatch, tmp_path):
    task, broadcaster, notification_writer = _run_failed_training(
        monkeypatch,
        tmp_path,
        ["torch.OutOfMemoryError: CUDA out of memory\n"],
    )

    assert task.status == "failed"
    assert json.loads(task.metrics)["error_code"] == "cuda_out_of_memory"
    notification_writer.assert_awaited_once()
    assert broadcaster.messages[-1][1]["status"] == "failed"


def test_ordinary_subprocess_failure_does_not_create_oom_notification(monkeypatch, tmp_path):
    task, _broadcaster, notification_writer = _run_failed_training(
        monkeypatch,
        tmp_path,
        ["ValueError: invalid training data\n"],
    )

    assert task.status == "failed"
    notification_writer.assert_not_awaited()


def test_notification_write_failure_does_not_block_terminal_task_state(monkeypatch, tmp_path):
    task, broadcaster, notification_writer = _run_failed_training(
        monkeypatch,
        tmp_path,
        ["CUDA out of memory\n"],
        notification_side_effect=RuntimeError("notification database unavailable"),
    )

    assert task.status == "failed"
    notification_writer.assert_awaited_once()
    assert broadcaster.messages[-1][1]["status"] == "failed"


def test_notification_commit_failure_does_not_block_terminal_task_state(monkeypatch, tmp_path):
    task, broadcaster, notification_writer, session_calls = _run_failed_training(
        monkeypatch,
        tmp_path,
        ["CUDA out of memory\n"],
        notification_commit_side_effect=RuntimeError("notification commit failed"),
        return_session_calls=True,
    )

    assert task.status == "failed"
    notification_writer.assert_awaited_once()
    assert len(session_calls) == 5
    assert broadcaster.messages[-1][1]["status"] == "failed"


def _task_response_payload(output_model_path):
    return {
        "id": 29,
        "model_script": "demo3.py",
        "status": "completed",
        "start_time": datetime.now(timezone.utc),
        "end_time": datetime.now(timezone.utc),
        "hyperparameters": "{}",
        "log_file_path": None,
        "output_model_path": str(output_model_path),
        "custom_model_name": "Historical broken task",
        "metrics": None,
        "progress": 100.0,
        "current_epoch": 1,
        "total_epochs": 1,
        "current_loss": None,
        "eta": None,
        "loss_history": None,
    }


def test_historical_completed_task_without_weight_is_unavailable(tmp_path):
    response = TrainingTaskResponse.model_validate(
        _task_response_payload(tmp_path / "missing.pth")
    )

    assert response.model_available is False
    assert response.model_dump()["model_available"] is False


def test_failed_task_with_leftover_weight_is_unavailable(tmp_path):
    weight_path = tmp_path / "leftover.pth"
    weight_path.write_bytes(b"stale weights")
    payload = _task_response_payload(weight_path)
    payload["status"] = "failed"

    response = TrainingTaskResponse.model_validate(payload)

    assert response.model_available is False


def test_inference_rechecks_weight_after_task_list_load(monkeypatch, tmp_path):
    missing_path = tmp_path / "moved-after-list-load.pth"
    task = SimpleNamespace(
        id=31,
        user_id=7,
        status="completed",
        output_model_path=str(missing_path),
        hyperparameters="{}",
    )

    monkeypatch.setattr(
        inference_module,
        "async_session_maker",
        lambda: _TaskSession(task),
    )
    service = inference_module.InferenceService()

    with pytest.raises(ValueError, match="Model file not found"):
        asyncio.run(
            service.predict_task(
                task_id=task.id,
                ls_start=90.0,
                current_user=SimpleNamespace(id=7, role="user"),
            )
        )


def test_inference_reports_clear_error_when_weight_disappears_during_load(monkeypatch, tmp_path):
    model_path = tmp_path / "removed-during-load.pth"
    model_path.write_bytes(b"weights")
    task = SimpleNamespace(output_model_path=str(model_path))
    service = inference_module.InferenceService()

    def remove_then_load(*args, **kwargs):
        model_path.unlink()
        raise FileNotFoundError(str(model_path))

    monkeypatch.setattr(inference_module.torch, "load", remove_then_load)

    with pytest.raises(ValueError, match="Model file not found"):
        service._load_task_state_dict(task)


def test_parse_progress_accepts_annotated_validation_loss_and_scientific_notation():
    service = training_module.TrainingService()
    history = {"train": [], "val": []}
    lines = [
        "Epoch 1/2 Loss=1.2e-1 (area-weighted normalized MSE) Val Loss=6.75e-2",
        "Epoch 2/2 Loss=8.0e-2 (normalized MSE) Val Loss=5.0e-2",
    ]
    results = [service._parse_progress_from_log(line, 2, 0.0, history) for line in lines]
    assert [item["val_loss"] for item in results] == [0.0675, 0.05]
    assert history == {"train": [0.12, 0.08], "val": [0.0675, 0.05]}


def test_parse_progress_replays_missing_validation_value_without_losing_history():
    service = training_module.TrainingService()
    history = {"train": [0.12], "val": [None]}
    result = service._parse_progress_from_log(
        "Epoch 1/2 Loss=0.12 (area-weighted normalized MSE) Val Loss=0.067518",
        2, 0.0, history,
    )
    assert result["loss_history"] == {"train": [0.12], "val": [0.067518]}
