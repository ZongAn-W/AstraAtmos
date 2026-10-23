"""
Data governance routes.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from auth.dependencies import get_current_user
from database.models import User

router = APIRouter(prefix="/upload/governance", tags=["数据治理"])


def _svc(request: Request):
    return request.app.state.data_governance_service


@router.get("/overview")
async def get_governance_overview(
    request: Request,
    scope: str = Query("mine", description="mine | all"),
    current_user: User = Depends(get_current_user),
):
    """
    Governance overview.
    - scope=mine: only current user's datasets
    - scope=all: all datasets (admin only)
    """
    try:
        return await _svc(request).get_overview(scope=scope, current_user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.get("/quality/{upload_id}")
async def get_governance_quality(
    request: Request,
    upload_id: int,
    current_user: User = Depends(get_current_user),
):
    """
    Single-dataset quality score.
    Admin can access all datasets; non-admin can only access own datasets.
    """
    try:
        return await _svc(request).get_quality(upload_id=upload_id, current_user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.get("/lineage/{upload_id}")
async def get_governance_lineage(
    request: Request,
    upload_id: int,
    current_user: User = Depends(get_current_user),
):
    """
    Single-dataset lineage information.
    Admin can access all datasets; non-admin can only access own datasets.
    """
    try:
        return await _svc(request).get_lineage(upload_id=upload_id, current_user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
