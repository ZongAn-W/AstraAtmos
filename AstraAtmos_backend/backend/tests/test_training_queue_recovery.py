"""Exercise persisted Mars transfer tasks through the real queue lifecycle.

Only the subprocess boundary is replaced: requests, ORM rows, ownership checks,
queue ordering, and execution preparation all use the production services.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.engine import Base
from database.models import ModelTrainingTask, TrainingWeightFile, User, UserModelPackage
from services import training_service as training_module
from services import training_weight_service as weight_module
from services import user_model_service as user_model_module
from services.dataset_registry import DatasetRegistry
from services.training_channels import normalize_training_hyperparameters
from services.training_weight_service import TrainingWeightService
from services.user_model_service import UserModelService


class RecordingTrainingService(training_module.TrainingService):
    """Keep a first run open while requests join the production FIFO queue."""

    def __init__(self, harness, *, block_first=False):
        super().__init__()
        self.harness = harness
        self.calls = []
        self.block_first = block_first
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def _run_training_subprocess(self, task_id, **spec):
        self.calls.append({"task_id": task_id, **spec})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            async with self.harness.sessions() as session:
                running = (await session.execute(
                    select(ModelTrainingTask).where(ModelTrainingTask.status == "running")
                )).scalars().all()
                assert [task.id for task in running] == [task_id]
            if self.block_first and len(self.calls) == 1:
                self.first_started.set()
                await self.release_first.wait()
            async with self.harness.sessions() as session:
                task = await session.get(ModelTrainingTask, task_id)
                task.status = "completed"
                task.end_time = datetime.now(timezone.utc)
                await session.commit()
        finally:
            self.active -= 1


class QueueHarness:
    def __init__(self, tmp_path, sessions):
        self.root = tmp_path
        self.sessions = sessions
        self.registry = DatasetRegistry(tmp_path / "no-earth-release")
        self.weights = TrainingWeightService(tmp_path / "uploaded-weights", sessions)
        self.models = UserModelService(tmp_path / "uploaded-models", sessions)
        self.services = []
        self.next_name = 0

    def service(self, *, block_first=False):
        service = RecordingTrainingService(self, block_first=block_first)
        self.services.append(service)
        return service

    async def enqueue(self, service, hyperparameters=None, *, model_source="official", is_admin=False):
        self.next_name += 1
        return await service.start_training(
            user_id=1,
            model_script="demo3.py",
            hyperparameters=hyperparameters or {},
            custom_model_name=f"queue-regression-{self.next_name}",
            model_source=model_source,
            uploaded_model_id="uploaded-model" if model_source == "uploaded" else None,
            user_model_service=self.models,
            training_weight_service=self.weights,
            dataset_registry=self.registry,
            is_admin=is_admin,
        )

    async def block_queue(self):
        service = self.service(block_first=True)
        await service.start()
        blocker = await self.enqueue(service)
        await asyncio.wait_for(service.first_started.wait(), timeout=5)
        return service, blocker

    async def task(self, task_id):
        async with self.sessions() as session:
            return await session.get(ModelTrainingTask, task_id)

    async def wait_terminal(self, task_id):
        async def poll():
            while True:
                task = await self.task(task_id)
                if task.status in {"completed", "failed", "cancelled"}:
                    return task
                await asyncio.sleep(0.01)
        return await asyncio.wait_for(poll(), timeout=5)

    async def seed_source(self, model_source, source_type, *, owner=1):
        weight_path = self.root / f"{model_source}-{source_type}-source.pth"
        torch.save({"weight": torch.ones(1)}, weight_path)
        hypers = normalize_training_hyperparameters({"model_source": model_source})
        if model_source == "uploaded":
            hypers.update({"_uploaded_model_id": "uploaded-model", "_uploaded_model_version": 1})
        async with self.sessions() as session:
            if source_type == "task":
                source = ModelTrainingTask(
                    user_id=owner, model_script="demo3.py", model_source=model_source,
                    uploaded_model_id="uploaded-model" if model_source == "uploaded" else None,
                    uploaded_model_version=1 if model_source == "uploaded" else None,
                    hyperparameters=json.dumps(hypers), status="completed",
                    output_model_path=str(weight_path), dataset_id="openmars_mcd",
                )
                session.add(source)
                await session.flush()
                target = {"transfer_source_type": "task", "transfer_source_task_id": source.id}
                source_id = source.id
            else:
                source = TrainingWeightFile(
                    id="uploaded-weight", user_id=owner, original_filename=weight_path.name,
                    storage_path=str(weight_path), content_hash=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                    status="ready", file_size=weight_path.stat().st_size,
                )
                session.add(source)
                source_id = source.id
                target = {"transfer_source_type": "upload", "transfer_weight_id": source.id}
            await session.commit()
        target["transfer_learning"] = True
        return target, source_id, weight_path


def run_queue_case(tmp_path, monkeypatch, scenario):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'queue.db'}")
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        harness = QueueHarness(tmp_path, sessions)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)
        monkeypatch.setattr(weight_module, "TRAINING_WEIGHTS_DIR", tmp_path / "uploaded-weights")
        monkeypatch.setattr(user_model_module, "USER_MODELS_DIR", tmp_path / "uploaded-models")
        monkeypatch.setattr(training_module.TrainingService, "_default_dataset_registry", staticmethod(lambda: harness.registry))
        scripts, logs, results = (tmp_path / name for name in ("scripts", "logs", "results"))
        for directory in (scripts, logs, results):
            directory.mkdir()
        (scripts / "demo3.py").write_text("# Subprocess is recorded, never executed.\n", encoding="utf-8")
        monkeypatch.setattr(training_module, "MODELS_DIR", scripts)
        monkeypatch.setattr(training_module, "LOGS_DIR", logs)
        monkeypatch.setattr(training_module, "OUTPUT_MODELS_DIR", results)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            async with sessions() as session:
                session.add_all([
                    User(id=1, email="owner@example.test", username="owner", password_hash="unused"),
                    User(id=2, email="other@example.test", username="other", password_hash="unused"),
                ])
                model_path = tmp_path / "uploaded-model.source"
                source_text = "# The runner is stubbed; no user model is executed.\n"
                model_path.write_text(source_text, encoding="utf-8")
                session.add(UserModelPackage(
                    id="uploaded-model", user_id=1, display_name="queue model", version=1,
                    original_filename="queue_model.py", storage_path=str(model_path),
                    content_hash=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    param_schema="{}", validation_status="valid", validation_report='{"mars":{"compatible":true}}',
                ))
                await session.commit()
            await scenario(harness)
        finally:
            for service in harness.services:
                await service.stop()
            await engine.dispose()
    asyncio.run(run())


@pytest.mark.parametrize("model_source", ["official", "uploaded"])
@pytest.mark.parametrize("source_type", ["task", "upload"])
@pytest.mark.parametrize("restart", [False, True], ids=["live-queue", "recovered-queue"])
def test_transfer_execution_restores_weight_path_from_persisted_task(tmp_path, monkeypatch, model_source, source_type, restart):
    async def scenario(harness):
        hypers, _source_id, weight_path = await harness.seed_source(model_source, source_type)
        original, _blocker = await harness.block_queue()
        target = await harness.enqueue(original, hypers, model_source=model_source)
        assert (await harness.task(target.id)).status == "queued"
        persisted_hypers = json.loads((await harness.task(target.id)).hyperparameters)
        # The durable request retains the source ID; the resolved host path must
        # be derived again when the scheduler finally executes the task.
        assert "ASTRAATMOS_TRANSFER_WEIGHT_PATH" not in persisted_hypers
        if restart:
            await original.stop()
            service = harness.service()
            await service.start()
        else:
            service = original
            original.release_first.set()
        assert (await harness.wait_terminal(target.id)).status == "completed"
        calls = [call for call in service.calls if call["task_id"] == target.id]
        assert len(calls) == 1
        assert calls[0]["env_overrides"] == {"ASTRAATMOS_TRANSFER_WEIGHT_PATH": str(weight_path)}
        assert calls[0]["hyperparameters"] == persisted_hypers
        assert calls[0]["script_name"] == ("__user_model_runner__" if model_source == "uploaded" else "demo3.py")
        assert calls[0]["earth_training_spec"] is None
        assert service.max_active == 1
    run_queue_case(tmp_path, monkeypatch, scenario)


FAILURE_CASES = [
    ("task", "missing-file"),
    ("task", "missing-record"),
    ("task", "empty-artifact"),
    ("task", "not-completed"),
    ("task", "foreign-owner"),
    ("task", "incompatible-model"),
    ("upload", "missing-file"),
    ("upload", "deleted-record"),
    ("upload", "invalid-status"),
    ("upload", "foreign-owner"),
]


@pytest.mark.parametrize("source_type,failure", FAILURE_CASES)
@pytest.mark.parametrize("restart", [False, True], ids=["live-queue", "recovered-queue"])
def test_transfer_failure_is_terminal_before_subprocess(tmp_path, monkeypatch, source_type, failure, restart):
    async def scenario(harness):
        hypers, source_id, weight_path = await harness.seed_source("official", source_type)
        original, _blocker = await harness.block_queue()
        target = await harness.enqueue(original, hypers)
        async with harness.sessions() as session:
            source = await session.get(ModelTrainingTask if source_type == "task" else TrainingWeightFile, source_id)
            if failure == "missing-file":
                weight_path.unlink()
            elif failure == "missing-record":
                await session.delete(source)
            elif failure == "empty-artifact":
                weight_path.write_bytes(b"")
            elif failure == "not-completed":
                source.status = "failed"
            elif failure == "foreign-owner":
                source.user_id = 2
            elif failure == "incompatible-model":
                incompatible = json.loads(source.hyperparameters)
                incompatible["window"] += 1
                source.hyperparameters = json.dumps(incompatible)
            elif failure == "deleted-record":
                source.deleted_at = datetime.now(timezone.utc)
            elif failure == "invalid-status":
                source.status = "invalid"
            await session.commit()
        if restart:
            await original.stop()
            executing = harness.service()
            await executing.start()
        else:
            executing = original
            original.release_first.set()
        failed = await harness.wait_terminal(target.id)
        assert failed.status == "failed"
        assert failed.end_time is not None
        assert failed.pid is None
        metrics = json.loads(failed.metrics)
        assert metrics["error_code"] == "training_execution_preparation_failed"
        assert metrics["error"]
        assert not Path(failed.log_file_path).exists()
        assert not [call for call in executing.calls if call["task_id"] == target.id]
    run_queue_case(tmp_path, monkeypatch, scenario)


@pytest.mark.parametrize("revoke_admin", [False, True], ids=["admin-retained", "admin-revoked"])
def test_recovered_foreign_transfer_checks_current_admin_role(tmp_path, monkeypatch, revoke_admin):
    async def scenario(harness):
        async with harness.sessions() as session:
            user = await session.get(User, 1)
            user.role = "admin"
            await session.commit()
        hypers, _source_id, weight_path = await harness.seed_source("official", "task", owner=2)
        original, _blocker = await harness.block_queue()
        target = await harness.enqueue(original, hypers, is_admin=True)
        await original.stop()
        if revoke_admin:
            async with harness.sessions() as session:
                user = await session.get(User, 1)
                user.role = "user"
                await session.commit()
        recovered = harness.service()
        await recovered.start()
        result = await harness.wait_terminal(target.id)
        if revoke_admin:
            assert result.status == "failed"
            assert json.loads(result.metrics)["error_code"] == "training_execution_preparation_failed"
            assert not recovered.calls
        else:
            assert result.status == "completed"
            assert recovered.calls[0]["env_overrides"] == {"ASTRAATMOS_TRANSFER_WEIGHT_PATH": str(weight_path)}
    run_queue_case(tmp_path, monkeypatch, scenario)


def test_recovered_fifo_continues_after_preparation_failure_with_one_active_task(tmp_path, monkeypatch):
    async def scenario(harness):
        bad_hypers, _source_id, bad_path = await harness.seed_source("official", "task")
        original, _blocker = await harness.block_queue()
        bad = await harness.enqueue(original, bad_hypers)
        first = await harness.enqueue(original)
        second = await harness.enqueue(original)
        assert [(await harness.task(task.id)).queue_position for task in (bad, first, second)] == [1, 2, 3]
        bad_path.unlink()
        await original.stop()
        recovered = harness.service(block_first=True)
        await recovered.start()
        await asyncio.wait_for(recovered.first_started.wait(), timeout=5)
        assert (await harness.wait_terminal(bad.id)).status == "failed"
        assert json.loads((await harness.task(bad.id)).metrics)["error_code"] == "training_execution_preparation_failed"
        assert (await harness.task(first.id)).status == "running"
        assert (await harness.task(second.id)).status == "queued"
        assert (await harness.task(second.id)).queue_position == 1
        recovered.release_first.set()
        assert (await harness.wait_terminal(second.id)).status == "completed"
        assert [call["task_id"] for call in recovered.calls] == [first.id, second.id]
        assert recovered.max_active == 1
    run_queue_case(tmp_path, monkeypatch, scenario)


def test_cancelling_queued_task_prevents_launch_and_reindexes_fifo(tmp_path, monkeypatch):
    async def scenario(harness):
        service, blocker = await harness.block_queue()
        cancelled = await harness.enqueue(service)
        next_task = await harness.enqueue(service)
        assert not await service.cancel_training(blocker.id)
        assert await service.cancel_training(cancelled.id)
        assert not await service.cancel_training(cancelled.id)
        assert (await harness.task(cancelled.id)).status == "cancelled"
        assert (await harness.task(next_task.id)).queue_position == 1
        service.release_first.set()
        assert (await harness.wait_terminal(next_task.id)).status == "completed"
        assert [call["task_id"] for call in service.calls] == [blocker.id, next_task.id]
    run_queue_case(tmp_path, monkeypatch, scenario)


@pytest.mark.parametrize("mutation", ["deleted", "invalid", "version", "digest", "missing-source"])
def test_recovered_uploaded_model_keeps_frozen_source_identity(tmp_path, monkeypatch, mutation):
    async def scenario(harness):
        original, _blocker = await harness.block_queue()
        target = await harness.enqueue(original, model_source="uploaded")
        async with harness.sessions() as session:
            package = await session.get(UserModelPackage, "uploaded-model")
            if mutation == "deleted":
                package.deleted_at = datetime.now(timezone.utc)
            elif mutation == "invalid":
                package.validation_status = "invalid"
            elif mutation == "version":
                package.version += 1
            elif mutation == "digest":
                source = Path(package.storage_path)
                source.write_text("# Changed since admission\n", encoding="utf-8")
                # Even an updated package digest cannot replace the task's hash.
                package.content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            else:
                Path(package.storage_path).unlink()
            await session.commit()
        await original.stop()
        recovered = harness.service()
        await recovered.start()
        failed = await harness.wait_terminal(target.id)
        assert failed.status == "failed"
        assert json.loads(failed.metrics)["error_code"] == "training_execution_preparation_failed"
        assert not recovered.calls
    run_queue_case(tmp_path, monkeypatch, scenario)
