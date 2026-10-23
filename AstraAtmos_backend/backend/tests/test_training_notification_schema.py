import asyncio
from types import SimpleNamespace

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from database.init_db import _patch_notification_table_columns
from routers.notification import serialize_notification


def test_legacy_notification_table_gets_training_task_link(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE notifications ("
                    "id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
                    "type VARCHAR(30) NOT NULL, title VARCHAR(200) NOT NULL, "
                    "content TEXT, is_read BOOLEAN NOT NULL, "
                    "related_upload_id INTEGER, created_at DATETIME NOT NULL)"
                )
            )
            await _patch_notification_table_columns(conn)
            columns = await conn.execute(text("PRAGMA table_info(notifications)"))
            names = {row[1] for row in columns.fetchall()}
        await engine.dispose()
        return names

    assert "related_training_task_id" in asyncio.run(run())


def test_notification_serializer_exposes_training_task_link():
    payload = serialize_notification(
        SimpleNamespace(
            id=7,
            type="training_oom",
            title="训练失败：GPU 显存不足",
            content="训练任务 #19 因 GPU 显存不足而失败。",
            is_read=False,
            related_upload_id=None,
            related_training_task_id=19,
            created_at=SimpleNamespace(isoformat=lambda: "2026-08-14T10:00:00"),
        )
    )

    assert payload["related_training_task_id"] == 19
    assert payload["related_upload_id"] is None
