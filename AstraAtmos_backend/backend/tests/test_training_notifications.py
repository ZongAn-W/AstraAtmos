import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.engine import Base
from database.models import ModelTrainingTask, Notification, User
from services.training_notifications import ensure_cuda_oom_notification


def test_oom_notification_is_owner_scoped_and_idempotent(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'notifications.db'}")
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with sessionmaker() as session:
            user = User(
                email="owner@example.com",
                username="Owner",
                password_hash="test-hash",
            )
            session.add(user)
            await session.flush()
            task = ModelTrainingTask(
                user_id=user.id,
                model_script="demo3.py",
                hyperparameters="{}",
                custom_model_name="PredRNN OOM",
            )
            session.add(task)
            await session.flush()

            assert await ensure_cuda_oom_notification(session, task) is True
            assert await ensure_cuda_oom_notification(session, task) is False
            await session.commit()

            rows = (
                await session.execute(
                    select(Notification).where(
                        Notification.related_training_task_id == task.id
                    )
                )
            ).scalars().all()

        await engine.dispose()
        return user.id, task.id, rows

    user_id, task_id, rows = asyncio.run(run())
    assert len(rows) == 1
    assert rows[0].user_id == user_id
    assert rows[0].type == "training_oom"
    assert rows[0].related_training_task_id == task_id
    assert rows[0].related_upload_id is None
    assert "PredRNN OOM" in rows[0].content


def test_ownerless_task_does_not_create_notification(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ownerless.db'}")
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with sessionmaker() as session:
            task = ModelTrainingTask(model_script="demo3.py", hyperparameters="{}")
            session.add(task)
            await session.flush()
            created = await ensure_cuda_oom_notification(session, task)
        await engine.dispose()
        return created

    assert asyncio.run(run()) is False
