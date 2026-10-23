import asyncio
import hashlib
import json
import sys
import tempfile
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.engine import Base  # noqa: E402
from database.models import User  # noqa: E402
from config import MAX_USER_MODEL_SIZE_KB  # noqa: E402
from routers.user_models import _serialize_package  # noqa: E402
from services.user_model_service import UserModelService  # noqa: E402
from services.user_model_validator import VALIDATION_TIMEOUT_CODE  # noqa: E402


VALID_MODEL = b"""
import torch
from torch import nn

MODEL_SPEC = {
    "name": "StoredTiny",
    "description": "Tiny stored model for service tests.",
    "parameters": {
        "hidden_dim": {
            "type": "int",
            "default": 4,
            "min": 1,
            "max": 16,
        },
    },
}


class StoredTiny(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        last = x[:, -1, :1]
        return last.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)


def build_model(config):
    return StoredTiny(config["horizon"])
"""


class RecordingValidator:
    def __init__(self):
        self.paths = []

    def validate_file(self, path):
        self.paths.append(Path(path))
        return SimpleNamespace(
            ok=True,
            errors=[],
            warnings=[],
            display_name="StoredTiny",
            description="recorded validation",
            param_schema={},
            output_shape=[2, 3, 1, 8, 16],
            report_dict=lambda: {
                "ok": True,
                "errors": [],
                "warnings": [],
                "output_shape": [2, 3, 1, 8, 16],
            },
        )


async def _make_sessionmaker(db_path: Path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_user(sessionmaker, email: str) -> User:
    async with sessionmaker() as session:
        user = User(
            email=email,
            username=email.split("@", 1)[0],
            password_hash="test-hash",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def test_service_creates_lists_revalidates_and_soft_deletes_model():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        db_path = root / "test.db"
        upload_root = root / "uploads"
        engine, sessionmaker = await _make_sessionmaker(db_path)

        try:
            user = await _create_user(sessionmaker, "owner@example.com")
            service = UserModelService(
                storage_root=upload_root,
                sessionmaker=sessionmaker,
            )

            package = await service.create_from_source(
                user.id,
                "stored tiny.py",
                VALID_MODEL,
            )

            assert package.validation_status == "valid"
            assert Path(package.storage_path).exists()
            assert package.version == 1
            assert package.display_name == "StoredTiny"

            packages = await service.list_user_packages(user.id)
            assert [item.id for item in packages] == [package.id]

            revalidated = await service.revalidate_package(package.id, user.id)
            assert revalidated.validation_status == "valid"

            await service.soft_delete_package(package.id, user.id)
            assert await service.list_user_packages(user.id) == []
            assert Path(package.storage_path).exists()
        finally:
            await engine.dispose()


@pytest.mark.parametrize("operation", ["upload", "revalidate"])
async def test_slow_validation_keeps_other_requests_responsive(tmp_path, operation):
    engine, sessionmaker = await _make_sessionmaker(tmp_path / "test.db")
    entered, release = threading.Event(), threading.Event()
    worker_threads = []
    validator = RecordingValidator()
    service = UserModelService(
        storage_root=tmp_path / "uploads", sessionmaker=sessionmaker, validator=validator,
    )
    pending = None
    try:
        user = await _create_user(sessionmaker, "responsive@example.com")
        package = await service.create_from_source(user.id, "model.py", VALID_MODEL)
        original = validator.validate_file
        def blocked(path):
            worker_threads.append(threading.get_ident())
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Validation blocked the event loop")
            return original(path)
        validator.validate_file = blocked
        coroutine = (service.create_from_source(user.id, "other.py", VALID_MODEL)
                     if operation == "upload" else service.revalidate_package(package.id, user.id))
        pending = asyncio.create_task(coroutine)
        assert await asyncio.to_thread(entered.wait, 2)
        packages = await asyncio.wait_for(service.list_user_packages(user.id), timeout=2)
        assert [item.id for item in packages] == [package.id]
        assert not pending.done()
        assert len(worker_threads) == 1
        assert worker_threads[0] != threading.get_ident()
        release.set()
        assert (await pending).validation_status == "valid"
    finally:
        release.set()
        if pending is not None and not pending.done():
            await pending
        await engine.dispose()


async def test_service_validates_temporary_file_before_permanent_storage_write():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        db_path = root / "test.db"
        upload_root = root / "uploads"
        engine, sessionmaker = await _make_sessionmaker(db_path)
        validator = RecordingValidator()

        try:
            user = await _create_user(sessionmaker, "owner@example.com")
            service = UserModelService(
                storage_root=upload_root,
                sessionmaker=sessionmaker,
                validator=validator,
            )

            package = await service.create_from_source(
                user.id,
                "stored_tiny.py",
                VALID_MODEL,
            )

            assert len(validator.paths) == 1
            assert upload_root not in validator.paths[0].parents
            assert Path(package.storage_path).exists()
            assert Path(package.storage_path).suffix != ".py"
        finally:
            await engine.dispose()


async def test_service_rejects_non_owner_access():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        db_path = root / "test.db"
        upload_root = root / "uploads"
        engine, sessionmaker = await _make_sessionmaker(db_path)

        try:
            owner = await _create_user(sessionmaker, "owner@example.com")
            other = await _create_user(sessionmaker, "other@example.com")
            service = UserModelService(
                storage_root=upload_root,
                sessionmaker=sessionmaker,
            )
            package = await service.create_from_source(
                owner.id,
                "owner_model.py",
                VALID_MODEL,
            )

            try:
                await service.get_package_for_user(package.id, other.id)
            except PermissionError:
                pass
            else:
                raise AssertionError("non-owner access should raise PermissionError")
        finally:
            await engine.dispose()


async def test_service_rejects_non_py_original_filename_before_writing():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        db_path = root / "test.db"
        upload_root = root / "uploads"
        engine, sessionmaker = await _make_sessionmaker(db_path)

        try:
            user = await _create_user(sessionmaker, "owner@example.com")
            service = UserModelService(
                storage_root=upload_root,
                sessionmaker=sessionmaker,
            )

            try:
                await service.create_from_source(user.id, "model.txt", VALID_MODEL)
            except ValueError:
                pass
            else:
                raise AssertionError("non-.py filenames should raise ValueError")

            assert await service.list_user_packages(user.id) == []
            assert not any(upload_root.rglob("*"))
        finally:
            await engine.dispose()


async def test_service_rejects_oversized_source_before_creating_package():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        db_path = root / "test.db"
        upload_root = root / "uploads"
        engine, sessionmaker = await _make_sessionmaker(db_path)

        try:
            user = await _create_user(sessionmaker, "owner@example.com")
            service = UserModelService(
                storage_root=upload_root,
                sessionmaker=sessionmaker,
            )
            oversized_source = b"x" * (MAX_USER_MODEL_SIZE_KB * 1024 + 1)

            try:
                await service.create_from_source(user.id, "too_large.py", oversized_source)
            except ValueError:
                pass
            else:
                raise AssertionError("oversized sources should raise ValueError")

            assert await service.list_user_packages(user.id) == []
            assert not any(upload_root.rglob("*"))
        finally:
            await engine.dispose()


def test_serialize_package_normalizes_malformed_json_shapes():
    now = datetime.now(timezone.utc)
    package = SimpleNamespace(
        id="package-id",
        user_id=1,
        display_name="BrokenJsonModel",
        version=1,
        original_filename="broken.py",
        content_hash="a" * 64,
        param_schema="[]",
        description=None,
        validation_status="invalid",
        validation_report='{"errors":"bad"}',
        created_at=now,
        updated_at=now,
    )

    response = _serialize_package(package)

    assert response.param_schema == {}
    assert response.validation_report.ok is False
    assert response.validation_report.errors == []
    assert response.validation_report.warnings == []
    assert response.validation_report.output_shape is None


@pytest.mark.parametrize("code", [VALIDATION_TIMEOUT_CODE, None, 120, [], {}, False])
def test_serialize_package_preserves_only_string_validation_codes(code):
    package = SimpleNamespace(
        id="package-id", user_id=1, display_name="TimedOutModel", version=1,
        original_filename="model.py", content_hash="a" * 64, param_schema="{}",
        description=None, validation_status="invalid", created_at=None, updated_at=None,
        validation_report=json.dumps({"ok": False, "errors": ["budget exceeded"], "code": code}),
    )
    response = _serialize_package(package)
    assert response.validation_report.ok is False
    assert response.validation_report.errors == ["budget exceeded"]
    assert response.validation_report.code == (code if isinstance(code, str) else None)


def _cached_compatibility_package(tmp_path, report):
    source = b"verified uploaded source"
    path = tmp_path / "uploaded.source"
    path.write_bytes(source)
    return SimpleNamespace(
        id="timed-out-package", user_id=1, display_name="TimedOutModel", version=1,
        storage_path=str(path), content_hash=hashlib.sha256(source).hexdigest(),
        validation_status="invalid", validation_report=json.dumps(report),
    )


@pytest.mark.parametrize("dataset_id", ["earth_merra2", "earth_merra2_3hourly_v1"])
@pytest.mark.parametrize("report,status,code", [
    ({"ok": False, "code": VALIDATION_TIMEOUT_CODE, "errors": ["budget exceeded"]},
     "unknown", VALIDATION_TIMEOUT_CODE),
    ({"ok": False, "errors": ["User model validation timed out after 30.0 seconds"]},
     "unknown", VALIDATION_TIMEOUT_CODE),
    ({"ok": False, "errors": ["model forward timeout is unsupported"]},
     "unavailable", "uploaded_model_contract_invalid"),
])
def test_cached_timeout_is_unknown_and_real_validation_failure_unavailable(tmp_path, dataset_id, report, status, code):
    package = _cached_compatibility_package(tmp_path, report)
    service = UserModelService(storage_root=tmp_path / "storage")
    async def get_package(*args):
        return package
    service.get_package_for_user = get_package
    verdict = asyncio.run(service.get_earth_compatibility(package.id, 1, dataset_id=dataset_id))
    assert verdict["compatible"] is False
    assert verdict["status"] == status
    assert verdict["code"] == code
    assert verdict["reasons"] == report["errors"]


@pytest.mark.parametrize("source_state,expected", [
    ("missing", "uploaded_model_missing"), ("tampered", "uploaded_model_tampered"),
])
def test_cached_source_integrity_errors_take_priority_over_timeout(tmp_path, source_state, expected):
    package = _cached_compatibility_package(tmp_path, {"ok": False, "code": VALIDATION_TIMEOUT_CODE,
        "errors": ["User model validation timed out after 30.0 seconds"]})
    if source_state == "missing":
        package.storage_path = str(tmp_path / "never-created.source")
    else:
        Path(package.storage_path).write_bytes(b"changed uploaded source")
    service = UserModelService(storage_root=tmp_path / "storage")
    async def get_package(*args):
        return package
    service.get_package_for_user = get_package
    verdict = asyncio.run(service.get_earth_compatibility(package.id, 1, dataset_id="earth_merra2_3hourly_v1"))
    assert verdict["compatible"] is False and verdict["status"] == "unavailable"
    assert verdict["code"] == expected


async def _run_tests():
    await test_service_creates_lists_revalidates_and_soft_deletes_model()
    await test_service_validates_temporary_file_before_permanent_storage_write()
    await test_service_rejects_non_owner_access()
    await test_service_rejects_non_py_original_filename_before_writing()
    await test_service_rejects_oversized_source_before_creating_package()
    test_serialize_package_normalizes_malformed_json_shapes()


if __name__ == "__main__":
    asyncio.run(_run_tests())
    print("user model service tests passed")
