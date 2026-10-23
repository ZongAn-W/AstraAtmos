import asyncio
import json
import inspect
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.engine import Base
from database import models


def test_metadata_adds_private_tag_tables_to_existing_database(tmp_path):
    async def run():
        assert "training_model_tags" in Base.metadata.tables
        assert "training_task_tags" in Base.metadata.tables
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
        try:
            async with engine.begin() as conn:
                legacy_tables = [table for name, table in Base.metadata.tables.items()
                                 if name not in {"training_model_tags", "training_task_tags"}]
                await conn.run_sync(lambda sync: Base.metadata.create_all(sync, tables=legacy_tables))
                await conn.execute(models.ModelTrainingTask.__table__.insert().values(id=81, model_script="demo3.py", hyperparameters="{}", status="completed"))
                await conn.run_sync(Base.metadata.create_all)
                assert (await conn.execute(text("SELECT id FROM model_training_tasks"))).scalar_one() == 81
                assert (await conn.execute(text("SELECT count(*) FROM training_task_tags"))).scalar_one() == 0
                assert (await conn.execute(text("SELECT count(*) FROM training_model_tags"))).scalar_one() == 0
        finally:
            await engine.dispose()
    asyncio.run(run())


async def _setup(tmp_path, monkeypatch, *, foreign_keys=True):
    assert (BACKEND_DIR / "services" / "training_tag_service.py").is_file(), "Private tag service is not implemented"
    from services import training_tag_service as tag_module
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tags.db'}")
    if foreign_keys:
        @event.listens_for(engine.sync_engine, "connect")
        def enable_foreign_keys(dbapi_connection, _):
            dbapi_connection.execute("PRAGMA foreign_keys=ON")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(tag_module, "async_session_maker", sessions)
    async with sessions() as session:
        session.add_all([models.User(id=user_id, email=f"u{user_id}@example.com", username=f"u{user_id}", password_hash="test", role="admin" if user_id == 3 else "user") for user_id in (1, 2, 3)])
        await session.flush()
        session.add_all([models.ModelTrainingTask(id=task_id, user_id=user_id, model_script="demo3.py", model_source="official", hyperparameters='{"horizon": 3}', custom_model_name=f"Model {task_id}", status="completed", output_model_path=str(tmp_path / f"{task_id}.pth")) for task_id, user_id in ((11, 1), (12, 1), (21, 2))])
        await session.commit()
    return engine, sessions, tag_module.TrainingTagService()


def test_tag_crud_normalizes_names_and_enforces_private_uniqueness(tmp_path, monkeypatch):
    async def run():
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        try:
            tag = await service.create_tag(1, "  Straße 火星  ")
            assert (tag.name, tag.name_key) == ("Straße 火星", "strasse 火星")
            with pytest.raises(ValueError, match="already exists"):
                await service.create_tag(1, "STRASSE 火星")
            other = await service.create_tag(2, "STRASSE 火星")
            assert [item.id for item in await service.list_tags(1)] == [tag.id]
            assert [item.id for item in await service.list_tags(3)] == []
            second = await service.create_tag(1, "Second")
            with pytest.raises(ValueError, match="already exists"):
                await service.rename_tag(second.id, 1, "strasse 火星")
            for user_id in (2, 3):
                with pytest.raises(FileNotFoundError):
                    await service.rename_tag(tag.id, user_id, "stolen")
                with pytest.raises(FileNotFoundError):
                    await service.delete_tag(tag.id, user_id)
            renamed = await service.rename_tag(tag.id, 1, "  新名称  ")
            assert renamed.name == "新名称"
            await service.delete_tag(tag.id, 1)
            assert [item.id for item in await service.list_tags(1)] == [second.id]
            assert [item.id for item in await service.list_tags(2)] == [other.id]
            async with sessions() as session:
                assert await session.get(models.ModelTrainingTask, 11) is not None
        finally:
            await engine.dispose()
    asyncio.run(run())


@pytest.mark.parametrize("name", ["", " \t\n", "火" * 65])
def test_tag_name_validation_rejects_empty_and_overlong(tmp_path, monkeypatch, name):
    async def run():
        engine, _, service = await _setup(tmp_path, monkeypatch)
        try:
            with pytest.raises(ValueError):
                await service.create_tag(1, name)
            valid = await service.create_tag(1, "火" * 64)
            with pytest.raises(ValueError):
                await service.rename_tag(valid.id, 1, name)
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_admin_overlay_replace_and_delete_preserve_other_users_tags_and_files(tmp_path, monkeypatch):
    async def run():
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        try:
            owner_tag = await service.create_tag(1, "Owner")
            admin_tag = await service.create_tag(3, "Admin")
            await service.replace_task_tags(11, [owner_tag.id], 1)
            await service.replace_task_tags(11, [admin_tag.id], 3, is_admin=True)
            assert await service.get_task_tags([11], 1) == {11: [{"id": owner_tag.id, "name": "Owner"}]}
            assert await service.get_task_tags([11], 3) == {11: [{"id": admin_tag.id, "name": "Admin"}]}
            with pytest.raises(PermissionError):
                await service.replace_task_tags(11, [], 2)
            with pytest.raises(FileNotFoundError):
                await service.replace_task_tags(11, [owner_tag.id], 3, is_admin=True)
            await service.replace_task_tags(11, [], 3, is_admin=True)
            assert await service.get_task_tags([11], 3) == {11: []}
            assert len((await service.get_task_tags([11], 1))[11]) == 1
            weight = tmp_path / "11.pth"
            weight.write_bytes(b"model")
            await service.delete_tag(owner_tag.id, 1)
            assert weight.read_bytes() == b"model"
            async with sessions() as session:
                assert (await session.execute(select(models.TrainingTaskTag))).scalars().all() == []
                assert await session.get(models.ModelTrainingTask, 11) is not None
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_bulk_assignment_is_atomic_and_idempotent(tmp_path, monkeypatch):
    async def run():
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        try:
            tag = await service.create_tag(1, "Batch")
            foreign = await service.create_tag(2, "Private")
            for task_ids, tag_ids, error in [([11, 21], [tag.id], PermissionError), ([11, 999], [tag.id], FileNotFoundError), ([11, 12], [tag.id, foreign.id], FileNotFoundError)]:
                with pytest.raises(error):
                    await service.update_task_tags(task_ids, tag_ids, "add", 1)
                assert await service.get_task_tags([11, 12], 1) == {11: [], 12: []}
            for _ in range(2):
                tasks = await service.update_task_tags([11, 12, 11], [tag.id, tag.id], "add", 1)
                assert [task.id for task in tasks] == [11, 12]
            async with sessions() as session:
                assert len((await session.execute(select(models.TrainingTaskTag))).scalars().all()) == 2
            with pytest.raises(PermissionError):
                await service.update_task_tags([11, 21], [tag.id], "remove", 1)
            assert len((await service.get_task_tags([11], 1))[11]) == 1
            for _ in range(2):
                await service.update_task_tags([11, 12], [tag.id], "remove", 1)
            assert await service.get_task_tags([11, 12], 1) == {11: [], 12: []}
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_http_tag_contract_validation_auth_and_private_task_responses(tmp_path, monkeypatch):
    async def run():
        import httpx
        from fastapi import FastAPI
        from routers import training as router_module
        from services import training_service as training_module
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)
        app = FastAPI()
        app.include_router(router_module.router, prefix="/api")
        user = SimpleNamespace(id=1, role="user")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                assert (await client.get("/api/training/tags")).status_code == 401
                app.dependency_overrides[router_module.get_current_user] = lambda: user
                for name in ("", "  ", "火" * 65):
                    assert (await client.post("/api/training/tags", json={"name": name})).status_code == 422
                created = await client.post("/api/training/tags", json={"name": "  火星  "})
                assert created.status_code == 200
                tag = created.json()
                assert tag == {"id": tag["id"], "name": "火星"}
                assert (await client.post("/api/training/tags", json={"name": "火星"})).status_code == 409
                assert (await client.get("/api/training/tags")).json() == [tag]
                for ids in ([True], [-1], [1.1], ["1"]):
                    assert (await client.put("/api/training/tasks/11/tags", json={"tag_ids": ids})).status_code == 422
                attached = await client.put("/api/training/tasks/11/tags", json={"tag_ids": [tag["id"]]})
                assert attached.status_code == 200
                assert attached.json()["tags"] == [tag]
                assert attached.json()["model_available"] is False
                listed = (await client.get("/api/training/tasks")).json()
                assert {task["id"] for task in listed} == {11, 12}
                assert next(task for task in listed if task["id"] == 11)["tags"] == [tag]
                renamed = await client.patch("/api/training/tasks/11/name", json={"model_name": "Renamed model"})
                assert renamed.json()["tags"] == [tag]
                assert (await client.put("/api/training/tasks/21/tags", json={"tag_ids": [tag["id"]]})).status_code == 403
                assert (await client.put("/api/training/tasks/11/tags", json={"tag_ids": [999]})).status_code == 404
                assert (await client.patch("/api/training/task-tags", json={"task_ids": [11], "tag_ids": [], "operation": "replace"})).status_code == 422
                bulk = await client.patch("/api/training/task-tags", json={"task_ids": [11, 12], "tag_ids": [tag["id"]], "operation": "add"})
                assert bulk.status_code == 200
                assert [item["id"] for item in bulk.json()] == [11, 12]
                assert all(item["tags"] == [tag] for item in bulk.json())
                changed = (await client.patch(f"/api/training/tags/{tag['id']}", json={"name": "New tag"})).json()
                assert changed == {"id": tag["id"], "name": "New tag"}
                user.id, user.role = 3, "admin"
                assert (await client.get("/api/training/tags")).json() == []
                assert all(item["tags"] == [] for item in (await client.get("/api/training/tasks")).json())
                assert (await client.delete(f"/api/training/tags/{tag['id']}")).status_code == 404
                admin_tag = (await client.post("/api/training/tags", json={"name": "Admin"})).json()
                bulk = await client.patch("/api/training/task-tags", json={"task_ids": [11, 21], "tag_ids": [admin_tag["id"]], "operation": "add"})
                assert bulk.status_code == 200
                assert [item["tags"] for item in bulk.json()] == [[admin_tag], [admin_tag]]
                user.id, user.role = 1, "user"
                assert next(item for item in (await client.get("/api/training/tasks")).json() if item["id"] == 11)["tags"] == [changed]
                assert (await client.delete(f"/api/training/tags/{tag['id']}")).json() == {"status": "success"}
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_start_commits_tags_and_paths_before_scheduling_and_defaults_to_untagged(tmp_path, monkeypatch):
    async def run():
        from schemas.training import TrainingStartRequest
        from services import training_service as training_module
        assert "tag_ids" in TrainingStartRequest.model_fields
        assert "tag_ids" in inspect.signature(training_module.TrainingService.start_training).parameters
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)
        observed = []
        def schedule(coroutine):
            coroutine.close()
            with sqlite3.connect(tmp_path / "tags.db") as db:
                observed.append((db.execute("SELECT log_file_path, output_model_path, hyperparameters FROM model_training_tasks WHERE custom_model_name LIKE 'Started%' ORDER BY id DESC LIMIT 1").fetchone(), db.execute("SELECT task_id, tag_id FROM training_task_tags").fetchall()))
        monkeypatch.setattr(training_module, "asyncio", SimpleNamespace(create_task=schedule))
        try:
            tag = await service.create_tag(1, "Start")
            start_service = training_module.TrainingService()
            task = await start_service.start_training(1, "demo3.py", {"tags": ["accidental nested tag"], "tag_ids": [tag.id]}, custom_model_name="Started tagged", tag_ids=[tag.id, tag.id])
            assert observed[0][0][:2] == (task.log_file_path, task.output_model_path)
            assert observed[0][1] == [(task.id, tag.id)]
            assert "tag_ids" not in json.loads(task.hyperparameters)
            assert "tags" not in json.loads(task.hyperparameters)
            default_request = TrainingStartRequest(model_script="demo3.py", model_name="Default")
            assert default_request.tag_ids == []
            untagged = await start_service.start_training(1, "demo3.py", {}, custom_model_name="Started default")
            assert await service.get_task_tags([untagged.id], 1) == {untagged.id: []}
            assert untagged.hyperparameters == task.hyperparameters
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_start_rejects_foreign_or_unknown_tags_without_task_or_subprocess(tmp_path, monkeypatch):
    async def run():
        from services import training_service as training_module
        assert "tag_ids" in inspect.signature(training_module.TrainingService.start_training).parameters
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)
        def unexpected_schedule(coroutine):
            coroutine.close()
            pytest.fail("Invalid tag must not launch a subprocess")
        monkeypatch.setattr(training_module, "asyncio", SimpleNamespace(create_task=unexpected_schedule))
        try:
            foreign = await service.create_tag(2, "Foreign")
            for tag_ids in ([foreign.id], [999]):
                with pytest.raises(FileNotFoundError):
                    await training_module.TrainingService().start_training(1, "demo3.py", {}, custom_model_name="Invalid tags", tag_ids=tag_ids, is_admin=True)
            async with sessions() as session:
                assert len((await session.execute(select(models.ModelTrainingTask))).scalars().all()) == 3
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_start_commit_failure_rolls_back_task_paths_and_tags(tmp_path, monkeypatch):
    async def run():
        from services import training_service as training_module
        assert "tag_ids" in inspect.signature(training_module.TrainingService.start_training).parameters
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        class FailedCommitSession(AsyncSession):
            async def commit(self):
                raise RuntimeError("commit failed")
        failed_sessions = async_sessionmaker(engine, class_=FailedCommitSession, expire_on_commit=False)
        monkeypatch.setattr(training_module, "async_session_maker", failed_sessions)
        def unexpected_schedule(coroutine):
            coroutine.close()
            pytest.fail("Uncommitted task must not launch a subprocess")
        monkeypatch.setattr(training_module, "asyncio", SimpleNamespace(create_task=unexpected_schedule))
        try:
            tag = await service.create_tag(1, "Start")
            with pytest.raises(RuntimeError, match="commit failed"):
                await training_module.TrainingService().start_training(1, "demo3.py", {}, custom_model_name="Failed commit", tag_ids=[tag.id])
            async with sessions() as session:
                assert len((await session.execute(select(models.ModelTrainingTask))).scalars().all()) == 3
                assert (await session.execute(select(models.TrainingTaskTag))).scalars().all() == []
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_task_deletion_removes_links_from_all_owners_without_deleting_tags(tmp_path, monkeypatch):
    async def run():
        from services import training_service as training_module
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)
        try:
            owner = await service.create_tag(1, "Owner")
            admin = await service.create_tag(3, "Admin")
            await service.replace_task_tags(11, [owner.id], 1)
            await service.replace_task_tags(11, [admin.id], 3, is_admin=True)
            # Production SQLite also needs explicit cleanup when FK enforcement is off.
            async with engine.connect() as conn:
                await conn.execute(text("PRAGMA foreign_keys=OFF"))
            assert await training_module.TrainingService().delete_task(11)
            async with sessions() as session:
                assert await session.get(models.ModelTrainingTask, 11) is None
                assert (await session.execute(select(models.TrainingTaskTag))).scalars().all() == []
                assert len((await session.execute(select(models.TrainingModelTag))).scalars().all()) == 2
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_tag_changes_do_not_change_prediction_fingerprint(tmp_path, monkeypatch):
    async def run():
        from services.prediction_analysis_cache import build_artifact_fingerprint
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        try:
            async with sessions() as session:
                task = await session.get(models.ModelTrainingTask, 11)
            dirs = {"ASTRAATMOS_OPENMARS_DIR": str(tmp_path), "ASTRAATMOS_MCD_DIR": str(tmp_path)}
            Path(task.output_model_path).write_bytes(b"model")
            before = build_artifact_fingerprint(task, dirs)
            tag = await service.create_tag(1, "Test")
            await service.replace_task_tags(11, [tag.id], 1)
            await service.rename_tag(tag.id, 1, "Renamed")
            async with sessions() as session:
                task_after = await session.get(models.ModelTrainingTask, 11)
            assert task_after.hyperparameters == task.hyperparameters
            assert task_after.output_model_path == task.output_model_path
            assert build_artifact_fingerprint(task_after, dirs) == before
            await service.delete_tag(tag.id, 1)
            async with sessions() as session:
                task_after_delete = await session.get(models.ModelTrainingTask, 11)
            assert build_artifact_fingerprint(task_after_delete, dirs) == before
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_bulk_and_hydration_chunk_large_id_lists_without_n_plus_one(tmp_path, monkeypatch):
    async def run():
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        try:
            task_ids = list(range(100, 1101))
            async with sessions() as session:
                session.add_all([models.ModelTrainingTask(id=task_id, user_id=1, model_script="demo3.py", hyperparameters="{}") for task_id in task_ids])
                session.add_all([models.TrainingModelTag(id=tag_id, user_id=1, name=f"Tag {tag_id}", name_key=f"tag {tag_id}") for tag_id in range(100, 603)])
                await session.commit()
            selects = []
            @event.listens_for(engine.sync_engine, "before_cursor_execute")
            def bounded_parameters(conn, cursor, statement, parameters, context, executemany):
                if not executemany:
                    assert len(parameters) <= 999
                if statement.startswith("SELECT"):
                    selects.append(statement)
            await service.update_task_tags(task_ids, [100], "add", 1)
            selects.clear()
            mapped = await service.get_task_tags(task_ids, 1)
            assert all(value == [{"id": 100, "name": "Tag 100"}] for value in mapped.values())
            assert len(selects) <= 3
            await service.replace_task_tags(11, list(range(100, 603)), 1)
            assert len((await service.get_task_tags([11], 1))[11]) == 503
            await service.update_task_tags(task_ids + [11], list(range(100, 603)), "remove", 1)
            assert all(value == [] for value in (await service.get_task_tags(task_ids + [11], 1)).values())
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_bulk_database_failure_rolls_back_preceding_chunks(tmp_path, monkeypatch):
    async def run():
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        try:
            async with sessions() as session:
                session.add_all([models.TrainingModelTag(id=tag_id, user_id=1, name=f"Tag {tag_id}", name_key=f"tag {tag_id}") for tag_id in range(100, 501)])
                await session.commit()
            inserts = []
            @event.listens_for(engine.sync_engine, "before_cursor_execute")
            def fail_second_insert(conn, cursor, statement, parameters, context, executemany):
                if statement.startswith("INSERT INTO training_task_tags"):
                    inserts.append(statement)
                    if len(inserts) == 2:
                        raise RuntimeError("Simulated write failure")
            with pytest.raises(RuntimeError, match="Simulated write failure"):
                await service.update_task_tags([11], list(range(100, 501)), "add", 1)
            assert await service.get_task_tags([11], 1) == {11: []}
        finally:
            await engine.dispose()
    asyncio.run(run())


def test_http_start_returns_private_tags_and_invalid_tag_never_schedules(tmp_path, monkeypatch):
    async def run():
        import httpx
        from fastapi import FastAPI
        from routers import training as router_module
        from services import training_service as training_module
        engine, sessions, service = await _setup(tmp_path, monkeypatch)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)
        scheduled = []
        def schedule(coroutine):
            coroutine.close()
            scheduled.append(True)
        monkeypatch.setattr(training_module, "asyncio", SimpleNamespace(create_task=schedule))
        app = FastAPI()
        app.include_router(router_module.router, prefix="/api")
        app.dependency_overrides[router_module.get_current_user] = lambda: SimpleNamespace(id=1, role="user")
        try:
            own = await service.create_tag(1, "Start")
            foreign = await service.create_tag(2, "Hidden")
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                payload = {"model_script": "demo3.py", "model_name": "HTTP start", "tag_ids": [foreign.id]}
                assert (await client.post("/api/training/start", json=payload)).status_code == 404
                assert scheduled == []
                payload["tag_ids"] = [own.id]
                response = await client.post("/api/training/start", json=payload)
                assert response.status_code == 200
                assert response.json()["tags"] == [{"id": own.id, "name": "Start"}]
                assert len(scheduled) == 1
                payload = {"model_script": "demo3.py", "model_name": "HTTP default"}
                response = await client.post("/api/training/start", json=payload)
                assert response.status_code == 200
                assert response.json()["tags"] == []
        finally:
            await engine.dispose()
    asyncio.run(run())


@pytest.mark.parametrize("operation", ["add", "remove", "replace", "start", "rename", "delete"])
def test_tag_writes_lock_ownership_until_commit_with_production_sqlite(tmp_path, monkeypatch, operation):
    """A concurrent deletion must not recycle a validated ID for another owner."""
    async def run():
        from services import training_service as training_module
        from services import training_tag_service as tag_module
        engine, sessions, service = await _setup(tmp_path, monkeypatch, foreign_keys=False)
        monkeypatch.setattr(training_module, "async_session_maker", sessions)
        paused, resume = asyncio.Event(), asyncio.Event()
        pending = None
        try:
            tag = await service.create_tag(1, "Original")
            if operation == "remove":
                await service.replace_task_tags(11, [tag.id], 1)
            original_validate = tag_module.validate_owned_tags
            original_owned = service._owned_tag
            validations = 0

            async def validate_then_pause(session, user_id, tag_ids):
                nonlocal validations
                result = await original_validate(session, user_id, tag_ids)
                validations += 1
                # Start has a preflight read before its final creation transaction.
                if validations == (2 if operation == "start" else 1):
                    paused.set()
                    await resume.wait()
                return result

            async def owned_then_pause(session, tag_id, user_id):
                result = await original_owned(session, tag_id, user_id)
                paused.set()
                await resume.wait()
                return result

            monkeypatch.setattr(tag_module, "validate_owned_tags", validate_then_pause)
            monkeypatch.setattr(training_module, "validate_owned_tags", validate_then_pause)
            monkeypatch.setattr(service, "_owned_tag", owned_then_pause)
            monkeypatch.setattr(training_module, "asyncio", SimpleNamespace(create_task=lambda coroutine: coroutine.close()))
            if operation in ("add", "remove"):
                work = service.update_task_tags([11], [tag.id], operation, 1)
            elif operation == "replace":
                work = service.replace_task_tags(11, [tag.id], 1)
            elif operation == "start":
                work = training_module.TrainingService().start_training(1, "demo3.py", {}, custom_model_name="Concurrent start", tag_ids=[tag.id])
            elif operation == "rename":
                work = service.rename_tag(tag.id, 1, "Renamed")
            else:
                work = service.delete_tag(tag.id, 1)
            pending = asyncio.create_task(work)
            await asyncio.wait_for(paused.wait(), timeout=5)
            race_blocked = False
            with sqlite3.connect(tmp_path / "tags.db", timeout=0) as other:
                assert other.execute("PRAGMA foreign_keys").fetchone()[0] == 0
                try:
                    other.execute("DELETE FROM training_task_tags WHERE tag_id = ?", (tag.id,))
                    other.execute("DELETE FROM training_model_tags WHERE id = ?", (tag.id,))
                    other.execute("INSERT INTO training_model_tags (user_id, name, name_key, created_at) VALUES (2, 'Private replacement', 'private replacement', CURRENT_TIMESTAMP)")
                    other.commit()
                except sqlite3.OperationalError as exc:
                    assert "locked" in str(exc)
                    race_blocked = True
                    other.rollback()
            resume.set()
            await asyncio.wait_for(pending, timeout=5)
            assert race_blocked, "Ownership validation allowed a concurrent delete and cross-account ID reuse"
            assert await service.list_tags(2) == []
        finally:
            resume.set()
            if pending is not None and not pending.done():
                await pending
            await engine.dispose()
    asyncio.run(run())


@pytest.mark.parametrize("operation", ["create", "rename"])
def test_tag_write_response_cannot_read_another_owners_reused_id(tmp_path, monkeypatch, operation):
    async def run():
        engine, _, service = await _setup(tmp_path, monkeypatch, foreign_keys=False)
        try:
            original_tag = await service.create_tag(1, "Original") if operation == "rename" else None
            commit = service._commit_name

            async def commit_then_reuse_id(session):
                await commit(session)
                with sqlite3.connect(tmp_path / "tags.db") as other:
                    other.execute("DELETE FROM training_model_tags WHERE user_id = 1")
                    other.execute("INSERT INTO training_model_tags (user_id, name, name_key, created_at) VALUES (2, 'Private secret', 'private secret', CURRENT_TIMESTAMP)")
                    other.commit()

            monkeypatch.setattr(service, "_commit_name", commit_then_reuse_id)
            result = await service.create_tag(1, "Requested") if operation == "create" else await service.rename_tag(original_tag.id, 1, "Requested")
            assert (result.user_id, result.name) == (1, "Requested")
            assert [tag.name for tag in await service.list_tags(2)] == ["Private secret"]
        finally:
            await engine.dispose()
    asyncio.run(run())
