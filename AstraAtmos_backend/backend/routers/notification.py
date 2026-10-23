"""
通知路由
--------
前缀 /notification，挂载到 /api/notification/*

路由：
  GET   /list              获取当前用户通知列表
  GET   /unread-count      获取未读通知数量
  POST  /mark-read/{id}   标记单条通知为已读
  POST  /mark-all-read    标记全部通知为已读
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, update

from auth.dependencies import get_current_user
from database.engine import async_session_maker
from database.models import Notification, User

router = APIRouter(prefix="/notification", tags=["通知"])


def serialize_notification(notification: Notification) -> dict:
    return {
        "id": notification.id,
        "type": notification.type,
        "title": notification.title,
        "content": notification.content,
        "is_read": notification.is_read,
        "related_upload_id": notification.related_upload_id,
        "related_training_task_id": notification.related_training_task_id,
        "created_at": notification.created_at.isoformat(),
    }


# ─── GET /list ────────────────────────────────────────────────────────────────

@router.get("/list")
async def get_notifications(
    current_user: User = Depends(get_current_user),
):
    """获取当前用户的通知列表（最新 50 条）。"""
    async with async_session_maker() as db:
        stmt = (
            select(Notification)
            .where(Notification.user_id == current_user.id)
            .order_by(Notification.created_at.desc())
            .limit(50)
        )
        rows = (await db.execute(stmt)).scalars().all()

    return [serialize_notification(notification) for notification in rows]


# ─── GET /unread-count ────────────────────────────────────────────────────────

@router.get("/unread-count")
async def get_unread_count(
    current_user: User = Depends(get_current_user),
):
    """获取当前用户未读通知数量。"""
    async with async_session_maker() as db:
        stmt = select(func.count()).where(
            Notification.user_id == current_user.id,
            Notification.is_read == False,  # noqa: E712
        )
        count = (await db.execute(stmt)).scalar_one()

    return {"count": count}


# ─── POST /mark-read/{id} ─────────────────────────────────────────────────────

@router.post("/mark-read/{notification_id}")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
):
    """标记指定通知为已读。"""
    async with async_session_maker() as db:
        notif = await db.get(Notification, notification_id)
        if notif and notif.user_id == current_user.id:
            notif.is_read = True
            await db.commit()

    return {"message": "已标记为已读"}


# ─── POST /mark-all-read ──────────────────────────────────────────────────────

@router.post("/mark-all-read")
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
):
    """标记当前用户所有通知为已读。"""
    async with async_session_maker() as db:
        stmt = (
            update(Notification)
            .where(
                Notification.user_id == current_user.id,
                Notification.is_read == False,  # noqa: E712
            )
            .values(is_read=True)
        )
        await db.execute(stmt)
        await db.commit()

    return {"message": "全部已读"}
