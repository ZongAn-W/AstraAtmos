"""Exercise model management through HTTP against a temporary database."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI

from test_user_model_service import RecordingValidator, _create_user, _make_sessionmaker
from auth.dependencies import get_current_user
from database.models import ModelTrainingTask, UserModelPackage
from routers.user_models import router
from services.user_model_service import UserModelService


SOURCE = "# 原始模型源码\r\nMODEL_SPEC = {'name': 'StoredTiny'}\r\n".encode("utf-8")


@asynccontextmanager
async def management_client(tmp_path, authenticated=True):
    engine, sessions = await _make_sessionmaker(tmp_path / "models.db")
    try:
        owner = await _create_user(sessions, "owner@example.com")
        other = await _create_user(sessions, "other@example.com")
        service = UserModelService(tmp_path / "uploads", sessions, RecordingValidator())
        package = await service.create_from_source(owner.id, "自定义模型.py", SOURCE)
        identity = {"user": owner}
        app = FastAPI()
        app.state.user_model_service = service
        app.include_router(router, prefix="/api")
        if authenticated:
            app.dependency_overrides[get_current_user] = lambda: identity["user"]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, service, package, identity, other
    finally:
        await engine.dispose()


def test_owner_can_rename_and_download_original_source_without_changing_task_snapshot(tmp_path):
    async def scenario():
        async with management_client(tmp_path) as (client, service, package, _, _other):
            snapshot = '{"uploaded_model_name":"StoredTiny","uploaded_model_version":1}'
            async with service.sessionmaker() as session:
                task = ModelTrainingTask(
                    user_id=package.user_id, model_script="user_model_runner.py",
                    hyperparameters=snapshot,
                    uploaded_model_id=package.id,
                )
                session.add(task)
                await session.commit()
                task_id = task.id
            response = await client.patch(
                f"/api/user-models/{package.id}", json={"display_name": "  新模型名称  "}
            )
            assert response.status_code == 200
            renamed = response.json()
            assert renamed["display_name"] == "新模型名称"
            for field in ("id", "version", "original_filename", "content_hash", "validation_status"):
                assert renamed[field] == getattr(package, field)
            assert (await client.get("/api/user-models")).json()["items"][0]["display_name"] == "新模型名称"
            revalidated = await client.post(f"/api/user-models/{package.id}/validate")
            assert revalidated.json()["display_name"] == "新模型名称"
            downloaded = await client.get(f"/api/user-models/{package.id}/download")
            assert downloaded.status_code == 200
            assert downloaded.content == SOURCE
            assert quote("自定义模型.py") in downloaded.headers["content-disposition"]
            assert downloaded.headers["cache-control"] == "private, no-store"
            assert Path(package.storage_path).read_bytes() == SOURCE
            async with service.sessionmaker() as session:
                assert (await session.get(ModelTrainingTask, task_id)).hyperparameters == snapshot
    asyncio.run(scenario())


@pytest.mark.parametrize("body", [
    {}, {"display_name": ""}, {"display_name": "   "},
    {"display_name": "x" * 121}, {"display_name": 42}, {"display_name": None},
])
def test_invalid_name_is_rejected_without_writing(tmp_path, body):
    async def scenario():
        async with management_client(tmp_path) as (client, service, package, _, _other):
            response = await client.patch(f"/api/user-models/{package.id}", json=body)
            assert response.status_code in (400, 422)
            stored = await service.get_package_for_user(package.id, package.user_id)
            assert stored.display_name == package.display_name
            assert stored.updated_at == package.updated_at
    asyncio.run(scenario())


def test_other_users_missing_and_deleted_models_are_rejected(tmp_path):
    async def scenario():
        async with management_client(tmp_path) as (client, service, package, identity, other):
            owner = identity["user"]
            identity["user"] = other
            assert (await client.get(f"/api/user-models/{package.id}/download")).status_code == 403
            assert (await client.patch(f"/api/user-models/{package.id}", json={"display_name": "stolen"})).status_code == 403
            identity["user"] = owner
            for model_id in ("missing", package.id):
                if model_id == package.id:
                    await service.soft_delete_package(package.id, owner.id)
                assert (await client.get(f"/api/user-models/{model_id}/download")).status_code == 404
                assert (await client.patch(f"/api/user-models/{model_id}", json={"display_name": "new"})).status_code == 404
    asyncio.run(scenario())


def test_both_management_actions_require_login(tmp_path):
    async def scenario():
        async with management_client(tmp_path, authenticated=False) as (client, _, package, _identity, _other):
            assert (await client.get(f"/api/user-models/{package.id}/download")).status_code == 401
            assert (await client.patch(f"/api/user-models/{package.id}", json={"display_name": "new"})).status_code == 401
    asyncio.run(scenario())


def test_download_rejects_missing_and_outside_storage_files(tmp_path):
    async def scenario():
        async with management_client(tmp_path) as (client, service, package, _, _other):
            outside = tmp_path / "private.py"
            outside.write_bytes(b"private file")
            for path in (service.storage_root / "missing.source", outside):
                async with service.sessionmaker() as session:
                    stored = await session.get(UserModelPackage, package.id)
                    stored.storage_path = str(path)
                    await session.commit()
                response = await client.get(f"/api/user-models/{package.id}/download")
                assert response.status_code == 404
                assert b"private file" not in response.content
    asyncio.run(scenario())
