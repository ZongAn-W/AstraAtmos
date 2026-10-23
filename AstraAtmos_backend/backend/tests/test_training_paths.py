import asyncio
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.engine import Base
from database.init_db import migrate_training_task_output_paths
from database.models import ModelTrainingTask
from services.training_paths import build_task_output_path, sanitize_model_name


def test_sanitize_model_name_keeps_ascii_and_removes_unicode():
    assert sanitize_model_name("MST-BEST") == "MST-BEST"
    assert sanitize_model_name("自定义测试") == "model"
    assert sanitize_model_name("模型 21") == "21"


def test_build_task_output_path_is_ascii(tmp_path: Path):
    output = build_task_output_path(3, "自定义测试1", results_dir=tmp_path)

    assert output == tmp_path / "task_3_1.pth"
    output.relative_to(tmp_path).as_posix().encode("ascii")


def test_sanitize_model_name_bounds_long_components():
    value = "a" * 400

    sanitized = sanitize_model_name(value)

    assert len(sanitized) <= 120
    assert sanitized == sanitize_model_name(value)
    assert sanitized != sanitize_model_name("a" * 399 + "b")


def test_training_task_path_migration_moves_file_and_is_idempotent(tmp_path: Path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
            connect_args={"check_same_thread": False},
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        legacy_results_dir = "\u8bad\u7ec3\u7ed3\u679c"
        old_dir = tmp_path / "models" / legacy_results_dir
        old_dir.mkdir(parents=True)
        legacy_filename = "task_3_\u81ea\u5b9a\u4e49\u6d4b\u8bd51.pth"
        old_path = old_dir / legacy_filename
        old_path.write_bytes(b"weights")
        results_dir = tmp_path / "models" / "training_results"

        async with sessionmaker() as session:
            session.add(
                ModelTrainingTask(
                    id=3,
                    model_script="demo3.py",
                    hyperparameters="{}",
                    custom_model_name="自定义测试1",
                    output_model_path=str(old_path),
                )
            )
            await session.commit()

        assert await migrate_training_task_output_paths(sessionmaker, results_dir) == 1

        migrated_path = results_dir / "task_3_1.pth"
        assert migrated_path.read_bytes() == b"weights"
        assert not old_path.exists()
        async with sessionmaker() as session:
            task = await session.get(ModelTrainingTask, 3)
            assert task.output_model_path == str(migrated_path.resolve())

        assert await migrate_training_task_output_paths(sessionmaker, results_dir) == 0
        assert migrated_path.read_bytes() == b"weights"
        await engine.dispose()

    asyncio.run(run())


def test_training_task_path_migration_preflights_all_conflicts(tmp_path: Path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'conflict.db'}",
            connect_args={"check_same_thread": False},
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        old_dir = tmp_path / "legacy"
        old_dir.mkdir()
        first_source = old_dir / "first.pth"
        second_source = old_dir / "second.pth"
        first_source.write_bytes(b"first")
        second_source.write_bytes(b"second")
        results_dir = tmp_path / "training_results"
        results_dir.mkdir()
        second_destination = build_task_output_path(2, "second", results_dir)
        second_destination.write_bytes(b"unrelated")

        async with sessionmaker() as session:
            session.add_all([
                ModelTrainingTask(
                    id=1,
                    model_script="demo3.py",
                    hyperparameters="{}",
                    custom_model_name="first",
                    output_model_path=str(first_source),
                ),
                ModelTrainingTask(
                    id=2,
                    model_script="demo3.py",
                    hyperparameters="{}",
                    custom_model_name="second",
                    output_model_path=str(second_source),
                ),
            ])
            await session.commit()

        with pytest.raises(FileExistsError, match="migration conflict"):
            await migrate_training_task_output_paths(sessionmaker, results_dir)

        assert first_source.read_bytes() == b"first"
        assert not build_task_output_path(1, "first", results_dir).exists()
        async with sessionmaker() as session:
            first_task = await session.get(ModelTrainingTask, 1)
            assert first_task.output_model_path == str(first_source)
        await engine.dispose()

    asyncio.run(run())


def test_training_task_path_migration_rejects_existing_destination_for_missing_source(tmp_path: Path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'missing.db'}",
            connect_args={"check_same_thread": False},
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        missing_source = tmp_path / "missing" / "model.pth"
        results_dir = tmp_path / "training_results"
        results_dir.mkdir()
        destination = build_task_output_path(3, "model", results_dir)
        destination.write_bytes(b"unrelated")
        async with sessionmaker() as session:
            session.add(
                ModelTrainingTask(
                    id=3,
                    model_script="demo3.py",
                    hyperparameters="{}",
                    custom_model_name="model",
                    output_model_path=str(missing_source),
                )
            )
            await session.commit()

        with pytest.raises(FileExistsError, match="migration conflict"):
            await migrate_training_task_output_paths(sessionmaker, results_dir)

        async with sessionmaker() as session:
            task = await session.get(ModelTrainingTask, 3)
            assert task.output_model_path == str(missing_source)
        assert destination.read_bytes() == b"unrelated"
        await engine.dispose()

    asyncio.run(run())


def test_training_task_path_migration_restores_files_when_commit_fails(tmp_path: Path):
    class FailingCommitSession(AsyncSession):
        async def commit(self):
            raise RuntimeError("commit failed")

    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'rollback.db'}",
            connect_args={"check_same_thread": False},
        )
        setup_sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        failing_sessionmaker = async_sessionmaker(
            engine,
            class_=FailingCommitSession,
            expire_on_commit=False,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        source = tmp_path / "legacy" / "model.pth"
        source.parent.mkdir()
        source.write_bytes(b"weights")
        results_dir = tmp_path / "training_results"
        async with setup_sessionmaker() as session:
            session.add(
                ModelTrainingTask(
                    id=4,
                    model_script="demo3.py",
                    hyperparameters="{}",
                    custom_model_name="rollback",
                    output_model_path=str(source),
                )
            )
            await session.commit()

        with pytest.raises(RuntimeError, match="commit failed"):
            await migrate_training_task_output_paths(failing_sessionmaker, results_dir)

        destination = build_task_output_path(4, "rollback", results_dir)
        assert source.read_bytes() == b"weights"
        assert not destination.exists()
        async with setup_sessionmaker() as session:
            task = await session.get(ModelTrainingTask, 4)
            assert task.output_model_path == str(source)
        await engine.dispose()

    asyncio.run(run())


def test_training_task_path_migration_restores_files_when_database_rollback_fails(
    tmp_path: Path,
):
    class FailingCommitAndRollbackSession(AsyncSession):
        async def commit(self):
            raise RuntimeError("commit failed")

        async def rollback(self):
            raise RuntimeError("rollback failed")

    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'rollback_failure.db'}",
            connect_args={"check_same_thread": False},
        )
        setup_sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        failing_sessionmaker = async_sessionmaker(
            engine,
            class_=FailingCommitAndRollbackSession,
            expire_on_commit=False,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        source = tmp_path / "legacy" / "model.pth"
        source.parent.mkdir()
        source.write_bytes(b"weights")
        results_dir = tmp_path / "training_results"
        async with setup_sessionmaker() as session:
            session.add(
                ModelTrainingTask(
                    id=5,
                    model_script="demo3.py",
                    hyperparameters="{}",
                    custom_model_name="rollback-failure",
                    output_model_path=str(source),
                )
            )
            await session.commit()

        with pytest.raises(RuntimeError, match="rollback failed"):
            await migrate_training_task_output_paths(failing_sessionmaker, results_dir)

        destination = build_task_output_path(5, "rollback-failure", results_dir)
        assert source.read_bytes() == b"weights"
        assert not destination.exists()
        async with setup_sessionmaker() as session:
            task = await session.get(ModelTrainingTask, 5)
            assert task.output_model_path == str(source)
        await engine.dispose()

    asyncio.run(run())
