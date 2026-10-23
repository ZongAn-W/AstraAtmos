"""
文件上传路由
-----------
前缀 /upload，挂载到 /api/upload/*

路由：
  POST   /nc                      上传并校验 .nc 文件
  GET    /my-uploads              查询当前用户上传记录
  DELETE /{upload_id}             删除上传记录和文件
  POST   /{upload_id}/contribute  贡献文件给网站（送审）
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from auth.dependencies import get_current_user, require_admin
from config import ALLOWED_NC_EXTENSIONS, APPROVED_DIR, PENDING_REVIEW_DIR
from database.engine import async_session_maker
from database.models import DatasetLineageEvent, Notification, UploadRecord, User

logger = logging.getLogger("astraatmos.upload_router")

router = APIRouter(prefix="/upload", tags=["文件上传"])


def _svc(request: Request):
    return request.app.state.upload_service


def _enqueue_personal_cache_rebuild(request: Request | None, user_id: int | None) -> None:
    """Deprecated: raw user uploads no longer build fused personal data caches."""
    return


def _empty_cache_status() -> dict:
    return {"jobs": [], "artifacts": []}


def _success_payload(record: UploadRecord, result, cache_status: dict | None = None) -> dict:
    status = "cache_building" if record.data_type == "mcd" else "valid"
    return {
        "upload_id": record.id,
        "status": status,
        "data_type": result.data_type,
        "mars_year": result.mars_year,
        "ls_range": [result.ls_start, result.ls_end],
        "grid_size": [result.lat_points, result.lon_points],
        "variables": result.variables,
        "warnings": result.warnings,
        "cache_status": cache_status or _empty_cache_status(),
        "message": "文件已保存，缓存正在后台生成" if status == "cache_building" else "文件校验通过，数据已保存",
    }


def _upload_list_item(r: UploadRecord, cache_status: dict | None = None) -> dict:
    return {
        "id": r.id,
        "filename": r.filename,
        "data_type": r.data_type,
        "mars_year": r.mars_year,
        "ls_start": r.ls_start,
        "ls_end": r.ls_end,
        "status": r.status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "file_size": r.file_size,
        "is_public": r.is_public,
        "validation_message": r.validation_message,
        "cache_status": cache_status or _empty_cache_status(),
    }


async def _add_lineage_event(
    db,
    upload_id: int,
    event_type: str,
    actor: Optional[User] = None,
    detail: Optional[str] = None,
) -> None:
    db.add(
        DatasetLineageEvent(
            upload_id=upload_id,
            event_type=event_type,
            event_detail=detail,
            actor_user_id=actor.id if actor else None,
            actor_role=actor.role if actor else None,
        )
    )


# ─── POST /nc ────────────────────────────────────────────────────────────────

@router.post("/nc")
async def upload_nc_file(
    request: Request,
    file: UploadFile = File(...),
    mars_year: Optional[int] = Form(None),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    """上传并校验 .nc 数据文件（需要认证）。"""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_NC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件格式 '{suffix}'，"
                f"请上传 {' / '.join(ALLOWED_NC_EXTENSIONS)} 格式的文件"
            ),
        )

    cache_status = _empty_cache_status()
    async with async_session_maker() as db:
        try:
            record, result = await _svc(request).process_upload(
                file=file,
                user_id=current_user.id,
                db=db,
                mars_year=mars_year,
                description=description,
            )
            event_type = "validated" if result.is_valid else "validation_failed"
            detail = (
                f"data_type={result.data_type or 'unknown'}; "
                f"mars_year={result.mars_year}; "
                f"ls=({result.ls_start},{result.ls_end}); "
                f"warnings={len(result.warnings)}"
            )
            if not result.is_valid and result.error:
                detail = f"{detail}; error={result.error}"
            await _add_lineage_event(
                db=db,
                upload_id=record.id,
                event_type=event_type,
                actor=current_user,
                detail=detail,
            )
            await db.commit()
            if result.is_valid:
                try:
                    await request.app.state.data_governance_service.prime_quality_snapshot(record.id)
                except Exception as exc:
                    logger.warning("precompute governance quality snapshot failed: %s", exc)
                if result.data_type == "mcd":
                    await request.app.state.mcd_cache_service.enqueue_upload(record.id)
                    record.status = "cache_building"
                    cache_status = await request.app.state.mcd_cache_service.get_cache_status(record.id)
                    enqueue_cache_job = getattr(request.app.state, "enqueue_mcd_cache_job", None)
                    if enqueue_cache_job:
                        enqueue_cache_job()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"文件存储失败: {exc}")

    if result.is_valid:
        _enqueue_personal_cache_rebuild(request, current_user.id)
        return _success_payload(record, result, cache_status)
    return {
        "upload_id": record.id,
        "status": "invalid",
        "error": result.error,
        "message": "文件校验失败",
    }


# ─── GET /my-uploads ─────────────────────────────────────────────────────────

@router.get("/my-uploads")
async def get_my_uploads(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """获取当前用户的所有上传记录（需要认证）。"""
    async with async_session_maker() as db:
        stmt = (
            select(UploadRecord)
            .where(UploadRecord.user_id == current_user.id)
            .order_by(UploadRecord.created_at.desc())
        )
        rows = (await db.execute(stmt)).scalars().all()

    result = []
    for r in rows:
        cache_status = _empty_cache_status()
        if r.data_type == "mcd":
            cache_status = await request.app.state.mcd_cache_service.get_cache_status(r.id)
        result.append(_upload_list_item(r, cache_status))
    return result


@router.get("/{upload_id}/cache-status")
async def get_upload_cache_status(
    request: Request,
    upload_id: int,
    current_user: User = Depends(get_current_user),
):
    async with async_session_maker() as db:
        record = await db.get(UploadRecord, upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="上传记录不存在")
    if record.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问此记录")
    if record.data_type != "mcd":
        return {"upload_id": upload_id, **_empty_cache_status()}
    return await request.app.state.mcd_cache_service.get_cache_status(upload_id)


# ─── DELETE /{upload_id} ─────────────────────────────────────────────────────

@router.delete("/{upload_id}")
async def delete_upload(
    request: Request,
    upload_id: int,
    current_user: User = Depends(get_current_user),
):
    """删除指定上传记录及其文件（仅限本人或管理员）。"""
    target_user_id: int | None = None
    should_delete_mcd_cache = False
    async with async_session_maker() as db:
        record = await db.get(UploadRecord, upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="上传记录不存在")
        if record.user_id != current_user.id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="无权删除此记录")
        target_user_id = record.user_id
        should_delete_mcd_cache = record.data_type == "mcd"

        # 删除文件目录（original.nc 所在的 upload_id 目录）
        file_dir = Path(record.file_path).parent
        if file_dir.exists():
            try:
                shutil.rmtree(file_dir)
            except OSError as exc:
                logger.warning("删除上传目录失败: %s", exc)
        if should_delete_mcd_cache:
            try:
                await request.app.state.mcd_cache_service.delete_upload_cache(record.id, session=db)
            except Exception as exc:
                logger.warning("删除 MCD 缓存失败: %s", exc)

        await db.delete(record)
        await db.commit()

    _enqueue_personal_cache_rebuild(request, target_user_id)
    return {"message": "上传记录已删除"}


# ─── GET /pending-reviews ─────────────────────────────────────────────────────

@router.get("/pending-reviews")
async def get_pending_reviews(
    current_user: User = Depends(require_admin),
):
    """获取所有待审核上传记录（仅管理员）。"""
    async with async_session_maker() as db:
        stmt = (
            select(UploadRecord)
            .options(selectinload(UploadRecord.uploader))
            .where(UploadRecord.status == "pending_review")
            .order_by(UploadRecord.created_at.asc())
        )
        rows = (await db.execute(stmt)).scalars().all()

        # 必须在 session 内构建结果，否则 session 关闭后访问 lazy 关系会报错
        result = [
            {
                "id": r.id,
                "filename": r.filename,
                "data_type": r.data_type,
                "mars_year": r.mars_year,
                "ls_start": r.ls_start,
                "ls_end": r.ls_end,
                "file_size": r.file_size,
                "created_at": r.created_at.isoformat(),
                "uploader_username": r.uploader.username if r.uploader else None,
                "uploader_email": r.uploader.email if r.uploader else None,
                "validation_message": r.validation_message,
                "description": r.description,
            }
            for r in rows
        ]

    return result


# ─── POST /{upload_id}/review ─────────────────────────────────────────────────

class ReviewBody(BaseModel):
    action: str  # "approve" | "reject"
    reason: Optional[str] = ""


@router.post("/{upload_id}/review")
async def review_upload(
    upload_id: int,
    body: ReviewBody,
    request: Request,
    current_user: User = Depends(require_admin),
):
    """审核上传记录：通过或拒绝（仅管理员）。"""
    if body.action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action 必须是 approve 或 reject")

    target_user_id: int | None = None
    final_status = ""
    async with async_session_maker() as db:
        record = await db.get(UploadRecord, upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="上传记录不存在")
        if record.status != "pending_review":
            raise HTTPException(
                status_code=400,
                detail=f"该记录当前状态为 '{record.status}'，不可审核",
            )

        now = datetime.now(timezone.utc)
        record.reviewed_at = now
        record.reviewed_by = current_user.id
        target_user_id = record.user_id

        if body.action == "approve":
            src = PENDING_REVIEW_DIR / str(record.id) / "original.nc"
            dest_dir = APPROVED_DIR / str(record.id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            if src.exists():
                try:
                    shutil.move(str(src), str(dest_dir / "original.nc"))
                except OSError as exc:
                    logger.warning("审核通过时移动文件失败: %s", exc)
            record.status = "approved"
            record.validation_message = "审核通过"
            notif = Notification(
                user_id=record.user_id,
                type="approved",
                title="数据贡献审核通过",
                content=f"您贡献的数据文件《{record.filename}》已通过审核，已添加到网站数据库。",
                related_upload_id=record.id,
            )
        else:
            record.status = "rejected"
            record.validation_message = body.reason or "审核未通过"
            notif = Notification(
                user_id=record.user_id,
                type="rejected",
                title="数据贡献审核未通过",
                content=f"您贡献的数据文件《{record.filename}》未通过审核。"
                + (f"原因：{body.reason}" if body.reason else ""),
                related_upload_id=record.id,
            )

        lineage_event_type = "approved" if body.action == "approve" else "rejected"
        lineage_detail = (
            "approved by admin review"
            if body.action == "approve"
            else (body.reason or "rejected by admin review")
        )

        db.add(notif)
        await _add_lineage_event(
            db=db,
            upload_id=record.id,
            event_type=lineage_event_type,
            actor=current_user,
            detail=lineage_detail,
        )
        if body.action == "approve":
            approved_path = APPROVED_DIR / str(record.id) / "original.nc"
            fallback_path = Path(record.file_path) if record.file_path else None
            if approved_path.exists():
                effective_path = str(approved_path)
                effective_status = "active"
            elif fallback_path and fallback_path.exists():
                effective_path = str(fallback_path)
                effective_status = "active_fallback_user_uploads"
            else:
                effective_path = ""
                effective_status = "approved_but_missing"
            await _add_lineage_event(
                db=db,
                upload_id=record.id,
                event_type="activated",
                actor=current_user,
                detail=f"effective_status={effective_status}; effective_path={effective_path or 'None'}",
            )
        await db.commit()
        final_status = record.status

    _enqueue_personal_cache_rebuild(request, target_user_id)

    # 审核通过后触发用户数据服务热更新索引
    if body.action == "approve":
        try:
            request.app.state.user_data_service.reload_approved()
        except Exception as exc:
            logger.warning("刷新 approved 索引失败: %s", exc)

    return {"message": "审核完成", "status": final_status}


# ─── POST /{upload_id}/contribute ────────────────────────────────────────────

class ContributeBody(BaseModel):
    description: Optional[str] = None


@router.post("/{upload_id}/contribute")
async def contribute_upload(
    request: Request,
    upload_id: int,
    body: ContributeBody = ContributeBody(),
    current_user: User = Depends(get_current_user),
):
    """将校验通过的文件贡献给网站（标记为待审核）。"""
    async with async_session_maker() as db:
        record = await db.get(UploadRecord, upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="上传记录不存在")
        if record.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权操作此记录")
        if record.status != "valid":
            raise HTTPException(
                status_code=400,
                detail="只有校验通过（valid）的文件才能贡献给网站",
            )
        if record.data_type == "mcd":
            cache_status = await request.app.state.mcd_cache_service.get_cache_status(record.id)
            ready_types = {
                item.get("cache_type")
                for item in cache_status.get("artifacts", [])
                if item.get("status") == "ready"
            }
            if not {"mcd_overview", "mcd_3h"}.issubset(ready_types):
                raise HTTPException(status_code=400, detail="MCD 缓存尚未生成完成，暂不能贡献")

        # 拷贝到 pending_review/{record_id}/original.nc
        dest_dir = PENDING_REVIEW_DIR / str(record.id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(record.file_path, dest_dir / "original.nc")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"文件拷贝失败: {exc}")

        record.is_public = True
        record.status = "pending_review"
        if body.description:
            record.description = body.description
        await _add_lineage_event(
            db=db,
            upload_id=record.id,
            event_type="submitted_for_review",
            actor=current_user,
            detail=body.description or "submitted to admin review",
        )
        await db.commit()

    _enqueue_personal_cache_rebuild(request, current_user.id)
    return {"message": "感谢贡献！文件已提交审核", "status": "pending_review"}


# ─── GET /approved-datasets ───────────────────────────────────────────────────

@router.get("/approved-datasets")
async def get_approved_datasets(
    current_user: User = Depends(require_admin),
):
    """获取所有已通过审核的上传记录（仅管理员）。"""
    async with async_session_maker() as db:
        stmt = (
            select(UploadRecord)
            .options(selectinload(UploadRecord.uploader))
            .where(UploadRecord.status == "approved")
            .order_by(UploadRecord.reviewed_at.desc())
        )
        rows = (await db.execute(stmt)).scalars().all()

        result = [
            {
                "id": r.id,
                "filename": r.filename,
                "data_type": r.data_type,
                "mars_year": r.mars_year,
                "ls_start": r.ls_start,
                "ls_end": r.ls_end,
                "file_size": r.file_size,
                "created_at": r.created_at.isoformat(),
                "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                "uploader_username": r.uploader.username if r.uploader else None,
                "uploader_email": r.uploader.email if r.uploader else None,
                "description": r.description,
            }
            for r in rows
        ]

    return result


# ─── POST /{upload_id}/revoke ─────────────────────────────────────────────────

@router.post("/{upload_id}/revoke")
async def revoke_upload(
    request: Request,
    upload_id: int,
    current_user: User = Depends(require_admin),
):
    """撤销已通过的数据集（仅管理员），并通知上传者。"""
    target_user_id: int | None = None
    final_status = ""
    async with async_session_maker() as db:
        record = await db.get(UploadRecord, upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="上传记录不存在")
        if record.status != "approved":
            raise HTTPException(
                status_code=400,
                detail=f"该记录当前状态为 '{record.status}'，无法撤销",
            )

        target_user_id = record.user_id

        # 删除 approved 目录中的文件
        approved_file = APPROVED_DIR / str(record.id) / "original.nc"
        if approved_file.exists():
            try:
                shutil.rmtree(APPROVED_DIR / str(record.id))
            except OSError as exc:
                logger.warning("撤销时删除文件失败: %s", exc)

        record.status = "rejected"
        record.validation_message = "管理员已撤销该数据集"

        notif = Notification(
            user_id=record.user_id,
            type="revoked",
            title="数据集已被撤销",
            content=f"您贡献的数据文件《{record.filename}》已被管理员从数据库中撤销。",
            related_upload_id=record.id,
        )
        db.add(notif)
        await _add_lineage_event(
            db=db,
            upload_id=record.id,
            event_type="revoked",
            actor=current_user,
            detail="revoked by admin",
        )
        await db.commit()
        final_status = record.status

    _enqueue_personal_cache_rebuild(request, target_user_id)
    return {"message": "已撤销", "status": final_status}
