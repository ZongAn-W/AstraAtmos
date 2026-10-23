"""
用户反馈路由
前缀：/feedback（挂载到 /api/feedback/*）
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select

from auth.dependencies import get_optional_user, require_admin
from config import DATA_DIR
from database.engine import async_session_maker
from database.models import Feedback, User

logger = logging.getLogger("astraatmos.feedback")

router = APIRouter(prefix="/feedback", tags=["用户反馈"])

FEEDBACK_UPLOADS_DIR = DATA_DIR / "feedback_uploads"
ALLOWED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_SCREENSHOT_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_CONTENT_LEN = 2000


# ─── POST /submit ─────────────────────────────────────────────────────────────

@router.post("/submit")
async def submit_feedback(
    type: str = Form(...),
    content: str = Form(...),
    contact_email: Optional[str] = Form(None),
    screenshot: Optional[UploadFile] = File(None),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """提交反馈（无需登录）。支持可选截图上传。"""
    if type not in ("bug", "suggestion", "other"):
        raise HTTPException(status_code=400, detail="type 必须是 bug、suggestion 或 other")
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="反馈内容不能为空")
    if len(content) > MAX_CONTENT_LEN:
        raise HTTPException(status_code=400, detail=f"反馈内容不能超过 {MAX_CONTENT_LEN} 字")

    # 保存截图
    screenshot_path = None
    if screenshot and screenshot.filename:
        ext = Path(screenshot.filename).suffix.lower()
        if ext not in ALLOWED_IMG_EXTS:
            raise HTTPException(status_code=400, detail="截图格式不支持，请上传 PNG/JPG/GIF/WebP")
        file_bytes = await screenshot.read()
        if len(file_bytes) > MAX_SCREENSHOT_BYTES:
            raise HTTPException(status_code=400, detail="截图不能超过 5MB")
        FEEDBACK_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = FEEDBACK_UPLOADS_DIR / filename
        save_path.write_bytes(file_bytes)
        screenshot_path = str(save_path)

    async with async_session_maker() as db:
        feedback = Feedback(
            user_id=current_user.id if current_user else None,
            type=type,
            content=content.strip(),
            contact_email=contact_email.strip() if contact_email else None,
            screenshot_path=screenshot_path,
        )
        db.add(feedback)
        await db.commit()
        await db.refresh(feedback)

    logger.info(
        "收到反馈 #%d: type=%s, user=%s",
        feedback.id, type,
        current_user.id if current_user else "anonymous",
    )
    return {"message": "反馈已提交，感谢您的意见！", "id": feedback.id}


# ─── GET /list ────────────────────────────────────────────────────────────────

@router.get("/list")
async def get_feedback_list(
    current_user: User = Depends(require_admin),
    status: Optional[str] = None,
):
    """获取所有反馈列表（仅管理员）。可选按状态过滤。"""
    async with async_session_maker() as db:
        stmt = select(Feedback).order_by(Feedback.created_at.desc())
        if status:
            stmt = stmt.where(Feedback.status == status)
        rows = (await db.execute(stmt)).scalars().all()

        result = []
        for f in rows:
            user_info = None
            if f.user_id:
                user = await db.get(User, f.user_id)
                if user:
                    user_info = {"username": user.username, "email": user.email}

            result.append({
                "id": f.id,
                "type": f.type,
                "content": f.content,
                "contact_email": f.contact_email,
                "has_screenshot": f.screenshot_path is not None,
                "status": f.status,
                "admin_note": f.admin_note,
                "user": user_info,
                "created_at": f.created_at.isoformat(),
                "resolved_at": f.resolved_at.isoformat() if f.resolved_at else None,
            })

    return result


# ─── GET /count ───────────────────────────────────────────────────────────────

@router.get("/count")
async def get_feedback_count(
    current_user: User = Depends(require_admin),
):
    """获取待处理反馈数量（仅管理员）。"""
    async with async_session_maker() as db:
        stmt = (
            select(func.count())
            .select_from(Feedback)
            .where(Feedback.status == "pending")
        )
        count = (await db.execute(stmt)).scalar() or 0
    return {"pending_count": count}


# ─── POST /{feedback_id}/resolve ──────────────────────────────────────────────

@router.post("/{feedback_id}/resolve")
async def resolve_feedback(
    feedback_id: int,
    current_user: User = Depends(require_admin),
):
    """标记反馈为已处理（仅管理员）。"""
    async with async_session_maker() as db:
        feedback = await db.get(Feedback, feedback_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="反馈不存在")
        feedback.status = "resolved"
        feedback.resolved_at = datetime.now(timezone.utc)
        await db.commit()
    return {"message": "已标记为已处理"}
