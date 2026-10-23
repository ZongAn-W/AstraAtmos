import asyncio
import ast
import sys
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.engine import Base  # noqa: E402
from database.models import ModelTrainingTask  # noqa: E402
from services import training_service as training_module  # noqa: E402


async def _create_session_maker(tmp_path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rename.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _task(name: str, *, status: str = "completed", user_id: int = 7):
    return ModelTrainingTask(
        user_id=user_id,
        model_script="demo3.py",
        model_source="official",
        hyperparameters="{}",
        custom_model_name=name,
        status=status,
        output_model_path=f"D:/models/{name}.pth",
    )


def test_rename_completed_model_trims_name_and_preserves_artifact_path(tmp_path, monkeypatch):
    async def run():
        engine, sessions = await _create_session_maker(tmp_path)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)

        try:
            async with sessions() as session:
                task = _task("Before")
                session.add(task)
                await session.commit()
                task_id = task.id
                original_path = task.output_model_path

            renamed = await training_module.TrainingService().rename_model(task_id, "  After  ")

            assert renamed.custom_model_name == "After"
            assert renamed.output_model_path == original_path

            async with sessions() as session:
                stored = await session.get(ModelTrainingTask, task_id)
                assert stored.custom_model_name == "After"
                assert stored.output_model_path == original_path
        finally:
            await engine.dispose()

    asyncio.run(run())


def test_rename_model_rejects_duplicates_and_unknown_status(tmp_path, monkeypatch):
    async def run():
        engine, sessions = await _create_session_maker(tmp_path)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)

        try:
            async with sessions() as session:
                existing = _task("Existing")
                completed = _task("Target")
                unknown = _task("Unknown", status="archived")
                session.add_all([existing, completed, unknown])
                await session.commit()

            service = training_module.TrainingService()
            with pytest.raises(ValueError, match="已被使用"):
                await service.rename_model(completed.id, " Existing ")
            with pytest.raises(ValueError, match="不允许"):
                await service.rename_model(unknown.id, "New name")
        finally:
            await engine.dispose()


@pytest.mark.parametrize("status", ["queued", "pending", "running", "completed", "failed", "cancelled"])
def test_rename_model_all_user_visible_statuses(tmp_path, monkeypatch, status):
    async def run():
        status_dir = tmp_path / status
        status_dir.mkdir()
        engine, sessions = await _create_session_maker(status_dir)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)
        try:
            async with sessions() as session:
                task = _task(f"Before {status}", status=status)
                session.add(task)
                await session.commit()
                task_id = task.id
            renamed = await training_module.TrainingService().rename_model(task_id, f"After {status}")
            assert renamed.custom_model_name == f"After {status}"
        finally:
            await engine.dispose()

    asyncio.run(run())


def test_rename_route_checks_task_access_before_renaming():
    route_path = BACKEND_DIR / "routers" / "training.py"
    module = ast.parse(route_path.read_text(encoding="utf-8"))
    route = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "rename_training_model"
    )

    access_check = route.body[0]
    assert isinstance(access_check, ast.Expr)
    assert isinstance(access_check.value, ast.Await)
    assert access_check.value.value.func.id == "_get_task_with_access_check"
