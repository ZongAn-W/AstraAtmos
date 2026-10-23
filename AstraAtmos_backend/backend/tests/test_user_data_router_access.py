import asyncio
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.models import UploadRecord, User  # noqa: E402
from routers import user_data as user_data_router  # noqa: E402


class FakeSession:
    def __init__(self, record):
        self.record = record

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, _model, _upload_id):
        return self.record


def _user(user_id=7, role="user"):
    return User(id=user_id, email=f"user{user_id}@example.com", username=f"user{user_id}", password_hash="x", role=role)


def _record(status="valid", user_id=7):
    return UploadRecord(
        id=42,
        user_id=user_id,
        filename="upload.nc",
        file_path="upload.nc",
        data_type="mcd",
        status=status,
    )


def _patch_record(monkeypatch, record):
    monkeypatch.setattr(user_data_router, "async_session_maker", lambda: FakeSession(record))


def test_user_data_visualization_access_rejects_invalid_owner_upload(monkeypatch):
    _patch_record(monkeypatch, _record(status="invalid", user_id=7))

    async def run():
        with pytest.raises(Exception) as exc:
            await user_data_router._check_access(42, _user(7), require_active=True)
        assert getattr(exc.value, "status_code", None) == 403

    asyncio.run(run())


def test_user_data_summary_access_still_allows_invalid_owner_upload(monkeypatch):
    record = _record(status="invalid", user_id=7)
    _patch_record(monkeypatch, record)

    async def run():
        assert await user_data_router._check_access(42, _user(7), require_active=False) is record

    asyncio.run(run())
