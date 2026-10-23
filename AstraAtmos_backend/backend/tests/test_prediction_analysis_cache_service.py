import asyncio
import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.engine import Base
from database.models import ModelTrainingTask, PredictionAnalysisCache, User
from services.prediction_analysis_cache import PredictionAnalysisCacheService


def _metric(value):
    return {
        "overall": {
            "step": 0,
            "rmse": value,
            "mae": value,
            "ssim": 0.5,
            "r2": 0.5,
        },
        "per_step": [],
    }


async def _database(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'cache.db'}"
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    openmars = tmp_path / "openmars"
    mcd = tmp_path / "mcd"
    openmars.mkdir(exist_ok=True)
    mcd.mkdir(exist_ok=True)
    (openmars / "MY27.nc").write_bytes(b"openmars")
    (mcd / "MY27.nc").write_bytes(b"mcd")
    data_dirs = {
        "ASTRAATMOS_OPENMARS_DIR": str(openmars),
        "ASTRAATMOS_MCD_DIR": str(mcd),
    }

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with sessions() as session:
        user_a = User(
            email="a@example.com", username="A", password_hash="x"
        )
        user_b = User(
            email="b@example.com", username="B", password_hash="x"
        )
        session.add_all([user_a, user_b])
        await session.flush()
        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"weights")
        task = ModelTrainingTask(
            user_id=user_a.id,
            model_script="demo3.py",
            hyperparameters=json.dumps({"horizon": 3}),
            status="completed",
            output_model_path=str(model_path),
        )
        session.add(task)
        await session.commit()

    return engine, sessions, user_a.id, user_b.id, task, data_dirs


def test_ready_result_persists_across_service_instances(tmp_path):
    async def run():
        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            return _metric(1.25)

        kwargs = {
            "user_id": user_a,
            "task": task,
            "analysis_type": "metrics",
            "request_params": {"horizon": 3},
            "data_dirs": data_dirs,
            "compute": compute,
        }
        first = await PredictionAnalysisCacheService(
            sessions
        ).get_or_compute(**kwargs)
        second = await PredictionAnalysisCacheService(
            sessions
        ).get_or_compute(**kwargs)
        await engine.dispose()
        return calls, first, second

    calls, first, second = asyncio.run(run())
    assert calls == 1
    assert first == second == _metric(1.25)


def test_cache_rows_are_isolated_by_requesting_account(tmp_path):
    async def run():
        engine, sessions, user_a, user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            return _metric(float(calls))

        service = PredictionAnalysisCacheService(sessions)
        common = {
            "task": task,
            "analysis_type": "metrics",
            "request_params": {"horizon": 3},
            "data_dirs": data_dirs,
            "compute": compute,
        }
        await service.get_or_compute(user_id=user_a, **common)
        await service.get_or_compute(user_id=user_b, **common)
        async with sessions() as session:
            rows = (
                await session.execute(select(PredictionAnalysisCache))
            ).scalars().all()
        await engine.dispose()
        return calls, {(row.user_id, row.status) for row in rows}

    calls, rows = asyncio.run(run())
    assert calls == 2
    assert len(rows) == 2
    assert all(status == "ready" for _user_id, status in rows)


def test_concurrent_callers_share_one_computation(tmp_path):
    async def run():
        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return _metric(2.0)

        kwargs = {
            "user_id": user_a,
            "task": task,
            "analysis_type": "metrics",
            "request_params": {"horizon": 3},
            "data_dirs": data_dirs,
            "compute": compute,
        }
        first = asyncio.create_task(
            PredictionAnalysisCacheService(sessions).get_or_compute(**kwargs)
        )
        await started.wait()
        second = asyncio.create_task(
            PredictionAnalysisCacheService(sessions).get_or_compute(**kwargs)
        )
        await asyncio.sleep(0.1)
        assert calls == 1
        gate.set()
        results = await asyncio.gather(first, second)
        await engine.dispose()
        return calls, results

    calls, results = asyncio.run(run())
    assert calls == 1
    assert results == [_metric(2.0), _metric(2.0)]


def test_changed_artifact_replaces_ready_payload(tmp_path):
    async def run():
        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            return _metric(float(calls))

        service = PredictionAnalysisCacheService(sessions)
        kwargs = {
            "user_id": user_a,
            "task": task,
            "analysis_type": "metrics",
            "request_params": {"horizon": 3},
            "data_dirs": data_dirs,
            "compute": compute,
        }
        first = await service.get_or_compute(**kwargs)
        task.hyperparameters = json.dumps({"horizon": 4})
        second = await service.get_or_compute(**kwargs)
        async with sessions() as session:
            count = len(
                (
                    await session.execute(select(PredictionAnalysisCache))
                ).scalars().all()
            )
        await engine.dispose()
        return calls, first, second, count

    calls, first, second, count = asyncio.run(run())
    assert calls == 2
    assert first != second
    assert count == 1


def test_corrupt_payload_is_recomputed(tmp_path):
    async def run():
        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            return _metric(float(calls))

        service = PredictionAnalysisCacheService(sessions)
        kwargs = {
            "user_id": user_a,
            "task": task,
            "analysis_type": "metrics",
            "request_params": {"horizon": 3},
            "data_dirs": data_dirs,
            "compute": compute,
        }
        await service.get_or_compute(**kwargs)
        async with sessions() as session:
            row = (
                await session.execute(select(PredictionAnalysisCache))
            ).scalar_one()
            row.payload = b"broken"
            await session.commit()
        result = await service.get_or_compute(**kwargs)
        await engine.dispose()
        return calls, result

    calls, result = asyncio.run(run())
    assert calls == 2
    assert result == _metric(2.0)


def test_expired_lease_is_reclaimed(tmp_path):
    async def run():
        from datetime import datetime, timedelta, timezone

        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        service = PredictionAnalysisCacheService(sessions)
        await service.get_or_compute(
            user_id=user_a,
            task=task,
            analysis_type="metrics",
            request_params={"horizon": 3},
            data_dirs=data_dirs,
            compute=lambda: _metric(1.0),
        )
        async with sessions() as session:
            row = (
                await session.execute(select(PredictionAnalysisCache))
            ).scalar_one()
            row.status = "computing"
            row.payload = None
            row.lease_token = "expired-token"
            row.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                seconds=1
            )
            await session.commit()
        result = await service.get_or_compute(
            user_id=user_a,
            task=task,
            analysis_type="metrics",
            request_params={"horizon": 3},
            data_dirs=data_dirs,
            compute=lambda: _metric(3.0),
        )
        async with sessions() as session:
            row = (
                await session.execute(select(PredictionAnalysisCache))
            ).scalar_one()
            final = (row.status, row.lease_token, row.payload is not None)
        await engine.dispose()
        return result, final

    result, final = asyncio.run(run())
    assert result == _metric(3.0)
    assert final == ("ready", None, True)


def test_abandoned_active_lease_is_reclaimed_after_wait_budget(tmp_path):
    async def run():
        from datetime import datetime, timedelta, timezone

        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        service = PredictionAnalysisCacheService(
            sessions,
            lease_seconds=60.0,
            poll_seconds=0.01,
            wait_timeout_seconds=0.03,
        )
        await service.get_or_compute(
            user_id=user_a,
            task=task,
            analysis_type="metrics",
            request_params={"horizon": 3},
            data_dirs=data_dirs,
            compute=lambda: _metric(1.0),
        )
        async with sessions() as session:
            row = (
                await session.execute(select(PredictionAnalysisCache))
            ).scalar_one()
            row.status = "computing"
            row.payload = None
            row.lease_token = "abandoned-token"
            row.lease_expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=5
            )
            await session.commit()

        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            return _metric(5.0)

        result = await asyncio.wait_for(
            service.get_or_compute(
                user_id=user_a,
                task=task,
                analysis_type="metrics",
                request_params={"horizon": 3},
                data_dirs=data_dirs,
                compute=compute,
            ),
            timeout=0.5,
        )
        async with sessions() as session:
            row = (
                await session.execute(select(PredictionAnalysisCache))
            ).scalar_one()
            final = (row.status, row.lease_token, row.payload is not None)
        await engine.dispose()
        return calls, result, final

    calls, result, final = asyncio.run(run())
    assert calls == 1
    assert result == _metric(5.0)
    assert final == ("ready", None, True)


def test_compute_failure_is_not_cached(tmp_path):
    async def run():
        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        service = PredictionAnalysisCacheService(sessions)
        attempts = 0

        async def compute():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("scientific failure")
            return _metric(3.0)

        kwargs = {
            "user_id": user_a,
            "task": task,
            "analysis_type": "metrics",
            "request_params": {"horizon": 3},
            "data_dirs": data_dirs,
            "compute": compute,
        }
        with pytest.raises(RuntimeError, match="scientific failure"):
            await service.get_or_compute(**kwargs)
        result = await service.get_or_compute(**kwargs)
        async with sessions() as session:
            row = (
                await session.execute(select(PredictionAnalysisCache))
            ).scalar_one()
            status = row.status
        await engine.dispose()
        return attempts, result, status

    attempts, result, status = asyncio.run(run())
    assert attempts == 2
    assert result == _metric(3.0)
    assert status == "ready"


def test_publish_failure_returns_fresh_result(tmp_path, monkeypatch):
    async def run():
        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        service = PredictionAnalysisCacheService(sessions)

        async def fail_publish(*args, **kwargs):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr(service, "_publish", fail_publish)
        result = await service.get_or_compute(
            user_id=user_a,
            task=task,
            analysis_type="metrics",
            request_params={"horizon": 3},
            data_dirs=data_dirs,
            compute=lambda: _metric(4.0),
        )
        await engine.dispose()
        return result

    assert asyncio.run(run()) == _metric(4.0)


def test_deleting_training_task_removes_analysis_cache_rows(
    tmp_path, monkeypatch
):
    async def run():
        engine, sessions, user_a, _user_b, task, data_dirs = (
            await _database(tmp_path)
        )
        await PredictionAnalysisCacheService(sessions).get_or_compute(
            user_id=user_a,
            task=task,
            analysis_type="metrics",
            request_params={"horizon": 3},
            data_dirs=data_dirs,
            compute=lambda: _metric(1.0),
        )
        async with sessions() as session:
            stored_task = await session.get(ModelTrainingTask, task.id)
            stored_task.log_file_path = None
            stored_task.output_model_path = None
            await session.commit()

        from services import training_service

        monkeypatch.setattr(
            training_service, "async_session_maker", sessions
        )
        deleted = await training_service.TrainingService().delete_task(task.id)

        async with sessions() as session:
            remaining_task = await session.get(ModelTrainingTask, task.id)
            remaining_cache_rows = (
                await session.execute(select(PredictionAnalysisCache))
            ).scalars().all()
        await engine.dispose()
        return deleted, remaining_task, remaining_cache_rows

    deleted, remaining_task, remaining_cache_rows = asyncio.run(run())
    assert deleted is True
    assert remaining_task is None
    assert remaining_cache_rows == []
