"""
用户数据分析路由
前缀：/user-data（挂载到 /api/user-data/*）

提供与 /api/explore/* 相同格式的响应，但数据源是用户上传的文件。
权限规则：上传者本人或管理员可访问所有状态；其他登录用户只能访问 approved 数据集。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select

from auth.dependencies import get_current_user, require_admin
from database.engine import async_session_maker
from database.models import UploadRecord, User

logger = logging.getLogger("astraatmos.user_data_router")

router = APIRouter(prefix="/user-data", tags=["用户数据分析"])


def _svc(request: Request):
    return request.app.state.user_data_service


ACTIVE_VISUALIZATION_STATUSES = {"valid", "pending_review", "approved"}


async def _check_access(upload_id: int, current_user: User, require_active: bool = False) -> UploadRecord:
    """
    检查当前用户是否有权访问该数据集。
    规则：上传者本人 / 管理员 → 所有状态可访问；其他用户 → 仅 approved。
    返回 UploadRecord，或抛出 HTTPException。
    """
    async with async_session_maker() as db:
        record = await db.get(UploadRecord, upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="数据集不存在")
    has_direct_access = record.user_id == current_user.id or current_user.role == "admin"
    if has_direct_access or record.status == "approved":
        if require_active and record.status not in ACTIVE_VISUALIZATION_STATUSES:
            raise HTTPException(status_code=403, detail="Dataset status is not available for visualization")
        return record
    raise HTTPException(status_code=403, detail="无权访问此数据集")


# ─── GET /summary/{upload_id} ─────────────────────────────────────────────────

@router.get("/summary/{upload_id}")
async def get_dataset_summary(
    request: Request,
    upload_id: int,
    current_user: User = Depends(get_current_user),
):
    """获取用户数据集的基本信息摘要"""
    await _check_access(upload_id, current_user)
    try:
        return await _svc(request).get_dataset_summary(upload_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── GET /globe/{upload_id} ───────────────────────────────────────────────────

@router.get("/globe/{upload_id}")
async def get_user_globe_data(
    request: Request,
    upload_id: int,
    ls: float = Query(10.0, ge=0, le=360),
    current_user: User = Depends(get_current_user),
):
    """获取用户数据集的 3D 点云"""
    await _check_access(upload_id, current_user, require_active=True)
    try:
        return await _svc(request).get_globe_data(upload_id, ls)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── GET /seasonal-heatmap/{upload_id} ───────────────────────────────────────

@router.get("/seasonal-heatmap/{upload_id}")
async def get_user_heatmap(
    request: Request,
    upload_id: int,
    variable: str = Query("o3col"),
    current_user: User = Depends(get_current_user),
):
    """获取用户数据集的 Ls-纬度热力图"""
    await _check_access(upload_id, current_user, require_active=True)
    try:
        return await _svc(request).get_seasonal_heatmap(upload_id, variable)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── GET /seasonal-bands/{upload_id} ─────────────────────────────────────────

@router.get("/seasonal-bands/{upload_id}")
async def get_user_bands(
    request: Request,
    upload_id: int,
    current_user: User = Depends(get_current_user),
):
    """获取用户数据集的纬度带折线图"""
    await _check_access(upload_id, current_user, require_active=True)
    try:
        return await _svc(request).get_seasonal_bands(upload_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── POST /reload-approved ────────────────────────────────────────────────────

@router.post("/reload-approved")
async def reload_approved_data(
    request: Request,
    current_user: User = Depends(require_admin),
):
    """
    热更新：重新扫描 approved 目录（仅管理员）。
    审核通过新数据后调用此接口，无需重启服务。
    """
    svc = _svc(request)
    svc.reload_approved()
    return {
        "message": "approved 数据索引已刷新",
        "count": len(svc.get_approved_list()),
    }
