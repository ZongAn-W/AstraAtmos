"""Earth training through the shared task service and the HTTP routes.

The service test drives the real subprocess wrapper with a stub process so the
completion gate is exercised end to end: a task only becomes ``completed`` after
the parent strictly reloads the published checkpoint.
"""

import asyncio
import json
from pathlib import Path

import pytest
import torch

from services.dataset_registry import DatasetRegistry
from services.earth_training_artifact import (
    build_checkpoint_payload,
    build_metrics_block,
    normalization_from_release,
    save_earth_artifact_atomic,
)
from training_backbones.earth_daily_model import create_earth_forecaster


def _split_metrics(rmse=3.0, mae=1.5):
    return {
        "overall": {"rmse": rmse, "mae": mae},
        "by_lead": [
            {"lead_day": 1, "rmse": rmse, "mae": mae},
            {"lead_day": 2, "rmse": rmse, "mae": mae},
            {"lead_day": 3, "rmse": rmse, "mae": mae},
        ],
    }


def _write_artifact(registry, output_path, *, task_id, binding=None, corrupt=None):
    binding = binding or registry.build_training_binding("earth_merra2_daily_v2")
    order = ["TO3", "U10M"]
    normalization = normalization_from_release(
        registry.get_earth_snapshot("earth_merra2_daily_v2"), order
    )
    payload = build_checkpoint_payload(
        model=create_earth_forecaster(order),
        input_channel_order=order,
        linear_hidden_layers=2,
        dataset_binding=binding,
        normalization=normalization,
        run={
            "task_id": task_id,
            "seed": 11,
            "optimizer": "Adam",
            "loss": "normalized_mse_grid_uniform",
            "best_epoch": 1,
            "epochs_completed": 1,
            "run_complete": True,
            "device": "cpu",
        },
        metrics=build_metrics_block(
            validation=_split_metrics(2.0, 1.0),
            test=_split_metrics(4.0, 2.0),
            validation_window_count=172,
            test_window_count=175,
        ),
        split_window_counts={"train": 357, "validation": 172, "test": 175},
        task_id=task_id,
    )
    if corrupt is not None:
        corrupt(payload)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if corrupt is None:
        save_earth_artifact_atomic(payload, target)
    else:
        torch.save(payload, target)
    return target


class StubProcess:
    """A subprocess stand-in that exits with a fixed code and writes an artifact."""

    def __init__(self, on_start, returncode=0):
        self.pid = 4242
        self.returncode = returncode
        self._on_start = on_start

    def start(self):
        self._on_start()

    def readline(self):
        return ""

    def close(self):
        """The service closes stdout after reading it."""

    def wait(self):
        return self.returncode

    @property
    def stdout(self):
        return self


@pytest.fixture
def service_env(earth_global_release, tmp_path, monkeypatch):
    """Point the service at a temporary database, log dir and results dir.

    The real user database must never be touched by these tests, so both the
    training service and the inference service are rebound to a fresh SQLite file.
    """
    import services.inference_service as inference_service
    import services.training_service as training_service
    from database.models import Base
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'earth-service.db'}")

    async def prepare():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(prepare())
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(training_service, "async_session_maker", sessions)
    monkeypatch.setattr(inference_service, "async_session_maker", sessions)

    logs_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(training_service, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(training_service, "OUTPUT_MODELS_DIR", results_dir)
    registry = DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **earth_global_release)
    return {"registry": registry, "logs": logs_dir, "results": results_dir, "tmp": tmp_path}


def _run_service_case(service_env, monkeypatch, *, writer, returncode=0, name="Earth DLinear"):
    """Create an Earth task and drive the real subprocess wrapper to completion."""
    import services.training_service as training_service

    service = training_service.TrainingService()
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env", {})
        # The Earth runner receives only --output_path; the spec rides in the env.
        captured["output_path"] = args[-1]
        try:
            writer(Path(captured["output_path"]))
        except Exception as exc:  # pragma: no cover - surfaces test bugs loudly
            raise AssertionError(f"the stub artifact writer failed: {exc!r}") from exc
        return StubProcess(lambda: None, returncode=returncode)

    monkeypatch.setattr(training_service.subprocess, "Popen", fake_popen)

    async def scenario():
        task = await service.start_training(
            user_id=1,
            model_script="demo3.py",
            hyperparameters={
                "training_dataset": "earth_merra2_daily_v2",
                "selected_channels": ["U10M"],
                "epochs": 1,
                "batch_size": 8,
            },
            custom_model_name=name,
            data_source="default",
            model_source="official",
            dataset_id="earth_merra2_daily_v2",
            dataset_registry=service_env["registry"],
        )
        # Let the fire-and-forget subprocess coroutine run to completion.
        for _ in range(60):
            await asyncio.sleep(0.05)
            refreshed = await service.get_task(task.id)
            if refreshed.status in ("completed", "failed"):
                return refreshed
        return await service.get_task(task.id)

    return asyncio.run(scenario()), captured


def test_earth_task_completes_only_after_the_checkpoint_is_verified(service_env, monkeypatch):
    def writer(output_path):
        # The task id is discoverable from the output file name; use a matching id.
        task_id = int(output_path.stem.split("_")[1])
        _write_artifact(service_env["registry"], output_path, task_id=task_id)

    task, captured = _run_service_case(service_env, monkeypatch, writer=writer)
    assert task.status == "completed"
    assert task.dataset_id == "earth_merra2_daily_v2"
    assert task.dataset_version == "v2"
    assert task.dataset_identity_status == "verified"
    assert task.dataset_fingerprint
    assert json.loads(task.dataset_snapshot)["planet"] == "earth"
    # Metrics come from the verified checkpoint, not from log scraping.
    metrics = json.loads(task.metrics)
    assert metrics["unit"] == "DU"
    assert metrics["splits"]["test"]["window_count"] == 175
    assert metrics["splits"]["test"]["overall"]["rmse"] == pytest.approx(4.0)
    # The Earth contract travels through the environment, never the CLI.
    assert captured["env"]["ASTRAATMOS_EARTH_TRAINING_SPEC"]
    assert captured["args"][-2] == "--output_path"
    assert "--training_dataset" not in captured["args"]
    assert "--selected_channels" not in captured["args"]
    spec = json.loads(captured["env"]["ASTRAATMOS_EARTH_TRAINING_SPEC"])
    assert spec["dataset_binding"]["dataset_fingerprint"] == task.dataset_fingerprint
    # The task remembers its strict parameters for later prediction.
    hypers = json.loads(task.hyperparameters)
    assert hypers["window"] == 7 and hypers["horizon"] == 3
    assert hypers["selected_channels"] == ["U10M"]
    assert hypers["training_dataset"] == "earth_merra2_daily_v2"


def test_exit_zero_without_a_file_fails_the_task(service_env, monkeypatch):
    task, _ = _run_service_case(service_env, monkeypatch, writer=lambda path: None, name="Earth NoFile")
    assert task.status == "failed"
    assert "no valid model weight file" in json.loads(task.metrics)["error"].lower()


def test_exit_zero_with_garbage_fails_the_task(service_env, monkeypatch):
    def writer(output_path):
        Path(output_path).write_bytes(b"not a checkpoint")

    task, _ = _run_service_case(service_env, monkeypatch, writer=writer, name="Earth Garbage")
    assert task.status == "failed"
    metrics = json.loads(task.metrics)
    assert metrics["error_code"] == "invalid_earth_training_artifact"


def test_artifact_bound_to_another_task_fails_the_task(service_env, monkeypatch):
    def writer(output_path):
        # A valid container, but for a different task id.
        _write_artifact(service_env["registry"], output_path, task_id=999999)

    task, _ = _run_service_case(service_env, monkeypatch, writer=writer, name="Earth WrongTask")
    assert task.status == "failed"
    assert json.loads(task.metrics)["error_code"] == "invalid_earth_training_artifact"


def test_artifact_with_a_tampered_identity_fails_the_task(service_env, monkeypatch):
    def corrupt(payload):
        payload["dataset_binding"]["dataset_fingerprint"] = "0" * 64

    def writer(output_path):
        task_id = int(Path(output_path).stem.split("_")[1])
        _write_artifact(service_env["registry"], output_path, task_id=task_id, corrupt=corrupt)

    task, _ = _run_service_case(service_env, monkeypatch, writer=writer, name="Earth Tampered")
    assert task.status == "failed"
    assert json.loads(task.metrics)["error_code"] == "invalid_earth_training_artifact"


def test_non_zero_exit_never_publishes_a_completed_task(service_env, monkeypatch):
    def writer(output_path):
        task_id = int(Path(output_path).stem.split("_")[1])
        _write_artifact(service_env["registry"], output_path, task_id=task_id)

    task, _ = _run_service_case(
        service_env, monkeypatch, writer=writer, returncode=1, name="Earth Crash"
    )
    assert task.status == "failed"


def test_a_user_stop_is_not_overwritten_by_a_late_exit_zero(service_env, monkeypatch):
    """The terminal stop state must survive the completion callback."""
    import services.training_service as training_service

    assert training_service._task_was_stopped(json.dumps({"note": "Stopped by user"})) is True
    assert training_service._task_was_stopped(json.dumps({"error": "x"})) is False
    assert training_service._task_was_stopped(None) is False
    assert training_service._task_was_stopped("{not json") is False


def test_earth_requests_are_rejected_before_creating_a_task(service_env, monkeypatch):
    """Unsupported configurations must not write a row, load a model or spawn a process."""
    import services.training_service as training_service
    from services.dataset_identity import DatasetRequestError

    service = training_service.TrainingService()
    spawned = {"count": 0}
    monkeypatch.setattr(
        training_service.subprocess,
        "Popen",
        lambda *args, **kwargs: spawned.__setitem__("count", spawned["count"] + 1),
    )

    cases = [
        # `uploaded` is a supported Earth source now, so this case fails on the
        # missing upload service rather than on capability. What matters is that it
        # still fails before any row, model load or subprocess.
        ({"model_source": "uploaded", "uploaded_model_id": "m1"}, {}, None),
        ({"model_source": "official"}, {"model_architecture": "simvp"}, "dataset_training_configuration_not_supported"),
        ({"model_source": "official"}, {"use_sphere": True}, "dataset_training_configuration_not_supported"),
        ({"model_source": "official"}, {"transfer_learning": True}, "dataset_training_configuration_not_supported"),
        ({"model_source": "official"}, {"window": 3}, "invalid_earth_training_parameters"),
        ({"model_source": "official"}, {"selected_channels": ["NOPE"]}, "invalid_earth_training_parameters"),
        ({"model_source": "official"}, {"epochs": 0}, "invalid_earth_training_parameters"),
        ({"model_source": "official"}, {"dataset_fingerprint": "x"}, "client_identity_not_allowed"),
    ]
    for options, extra, expected_code in cases:
        hyperparameters = {
            "training_dataset": "earth_merra2_daily_v2",
            "selected_channels": ["U10M"],
            **extra,
        }
        with pytest.raises((DatasetRequestError, ValueError)) as error:
            asyncio.run(
                service.start_training(
                    user_id=1,
                    model_script="demo3.py",
                    hyperparameters=hyperparameters,
                    custom_model_name=f"Rejected {expected_code} {len(extra)}",
                    model_source=options["model_source"],
                    uploaded_model_id=options.get("uploaded_model_id"),
                    dataset_id="earth_merra2_daily_v2",
                    dataset_registry=service_env["registry"],
                )
            )
        if expected_code is None:
            assert "user_model_service" in str(error.value)
        else:
            assert error.value.code == expected_code
    assert spawned["count"] == 0


def test_dataset_id_conflict_is_still_refused(service_env, monkeypatch):
    import services.training_service as training_service
    from services.dataset_identity import DatasetRequestError

    service = training_service.TrainingService()
    with pytest.raises(DatasetRequestError) as error:
        asyncio.run(
            service.start_training(
                user_id=1,
                model_script="demo3.py",
                hyperparameters={"training_dataset": "openmars_mcd"},
                custom_model_name="Conflict",
                dataset_id="earth_merra2_daily_v2",
                dataset_registry=service_env["registry"],
            )
        )
    assert error.value.code == "dataset_id_conflict"


def test_mars_training_still_rejects_earth_through_the_legacy_runner_contract():
    """The Mars runner keeps its own strict check, so Earth cannot leak into it."""
    from services.dataset_identity import DatasetRequestError, require_training_dataset

    for dataset_id in ("earth_merra2_daily_v1", "earth_merra2_daily_v2"):
        with pytest.raises(DatasetRequestError) as error:
            require_training_dataset(dataset_id)
        assert error.value.code == "dataset_training_not_supported"
        assert error.value.status_code == 409
    assert require_training_dataset("openmars_mcd") == "openmars_mcd"
