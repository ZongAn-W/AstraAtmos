from __future__ import annotations

from typing import Any

from sqlalchemy import select


TRAINING_OOM_NOTIFICATION_TYPE = "training_oom"
TRAINING_OOM_NOTIFICATION_TITLE = "训练失败：GPU 显存不足"


def _notification_content(task: Any) -> str:
    name = str(task.custom_model_name or "").strip()
    name_label = f"（{name}）" if name else ""
    return (
        f"训练任务 #{task.id}{name_label} 因 GPU 显存不足而失败。"
        "请等待其他训练任务完成，或减小批大小后重试。"
    )


async def ensure_cuda_oom_notification(session, task: Any) -> bool:
    if task.user_id is None:
        return False

    from database.models import Notification

    existing = await session.execute(
        select(Notification.id).where(
            Notification.user_id == task.user_id,
            Notification.type == TRAINING_OOM_NOTIFICATION_TYPE,
            Notification.related_training_task_id == task.id,
        )
    )
    if existing.scalar_one_or_none() is not None:
        return False

    session.add(
        Notification(
            user_id=task.user_id,
            type=TRAINING_OOM_NOTIFICATION_TYPE,
            title=TRAINING_OOM_NOTIFICATION_TITLE,
            content=_notification_content(task),
            related_training_task_id=task.id,
        )
    )
    return True
