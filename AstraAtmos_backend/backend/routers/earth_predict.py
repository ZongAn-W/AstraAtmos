"""Earth historical prediction endpoints.

These are the only prediction endpoints that accept an Earth task. The Mars
endpoints under ``/predict`` keep rejecting Earth tasks, and nothing here falls
back to Mars data preparation, Mars channels, MY/Ls or the Mars prediction cache.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from auth.dependencies import get_current_user
from database.models import User
from schemas.earth_predict import (
    EarthPredictContextResponse,
    EarthPredictRequest,
    EarthPredictResponse,
    EarthCompareRequest,
)
from schemas.earth_diagnostics import EarthDiagnosticsRequest
from services.inference_compute import run_inference_compute
from services.dataset_identity import (
    DatasetRequestError, require_active_dataset, training_task_dataset_id,
)
from services.earth_dataset_metadata import EarthPackageError
from services.earth_prediction_service import (
    build_prediction_context,
    run_earth_prediction,
    compare_earth_test_metrics,
    _load_checkpoint,
    _require_completed_earth_task,
)
from services.research_export_sources import ArtifactGuardError, task_guard, register_earth

logger = logging.getLogger("astraatmos.earth.predict")

router = APIRouter(prefix="/earth/predict", tags=["Earth Prediction"])


def _error_response(exc: Exception) -> HTTPException:
    if isinstance(exc, DatasetRequestError):
        detail = {"code": exc.code, "message": str(exc)}
        if exc.availability_reason:
            detail["availability_reason"] = exc.availability_reason
        return HTTPException(status_code=exc.status_code, detail=detail)
    if isinstance(exc, EarthPackageError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "dataset_unavailable",
                "message": "The registered Earth dataset is not available",
                "availability_reason": exc.reason,
            },
        )
    if isinstance(exc, ArtifactGuardError):
        return HTTPException(
            status_code=409,
            detail={
                "code": "invalid_earth_training_artifact",
                "message": "The Earth model artifact is unavailable",
            },
        )
    raise exc


async def _load_task(request: Request, task_id: int, current_user: User):
    """Resolve the task and enforce the existing ownership rules."""
    from services.training_service import TrainingService

    service = getattr(request.app.state, "training_service", None) or TrainingService()
    task = await service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id is not None and task.user_id != current_user.id:
        role = (getattr(current_user, "role", "") or "").lower()
        if role != "admin":
            raise HTTPException(status_code=403, detail="No permission to access this task")
    return task


def _registry(request: Request):
    registry = getattr(request.app.state, "dataset_registry", None)
    if registry is None:
        raise HTTPException(status_code=500, detail="dataset registry unavailable")
    return registry


@router.get("/diagnostics/context")
async def get_earth_diagnostics_context(
    request: Request, training_task_id: int,
    current_user: User = Depends(get_current_user),
):
    task = await _load_task(request, training_task_id, current_user)
    try:
        context = await asyncio.to_thread(build_prediction_context, task, _registry(request))
        test_range = (context.metrics.get("split_ranges") or {}).get("test")
        return {"available": True, "planet": "earth", "task_id": task.id,
                "target_unit": "DU", "window": context.window, "horizon": context.horizon,
                "test_range": test_range, "input_channel_order": context.input_channel_order,
                "defaults": EarthDiagnosticsRequest(training_task_id=task.id).model_dump(exclude={"training_task_id"})}
    except (DatasetRequestError, EarthPackageError) as exc:
        raise _error_response(exc) from exc


async def diagnostics_for_task(request: Request, task, payload: EarthDiagnosticsRequest, current_user: User):
    """Shared by the Earth page and the legacy task action=test entry."""
    from services.earth_diagnostics import compute_earth_diagnostics
    try:
        require_active_dataset(training_task_dataset_id(task))
        from services.earth_prediction_service import _require_completed_earth_task
        _require_completed_earth_task(task)
        guard = task_guard(task)
        parameters = payload.model_dump(exclude={"training_task_id"})
        async def compute():
            current_task = await _load_task(request, task.id, current_user)
            if guard != task_guard(current_task):
                raise DatasetRequestError("invalid_earth_training_artifact", "Task changed while diagnostics were queued", status_code=409)
            return await asyncio.to_thread(compute_earth_diagnostics, current_task, _registry(request), **parameters)

        result = await run_inference_compute(compute)
        current_task = await _load_task(request, task.id, current_user)
        if guard != task_guard(current_task):
            raise DatasetRequestError("invalid_earth_training_artifact", "Model changed during diagnostics", status_code=409)
        return result
    except (DatasetRequestError, EarthPackageError) as exc:
        raise _error_response(exc) from exc
    except OSError as exc:
        raise HTTPException(409, detail={"code": "invalid_earth_training_artifact", "message": "Model artifact is unavailable"}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Earth diagnostics failed: task_id=%s", task.id)
        raise HTTPException(500, detail={"code": "earth_diagnostics_failed", "message": f"Earth diagnostics failed: {exc}"}) from exc


@router.post("/diagnostics")
async def run_earth_diagnostics(
    request: Request, payload: EarthDiagnosticsRequest,
    current_user: User = Depends(get_current_user),
):
    task = await _load_task(request, payload.training_task_id, current_user)
    return await diagnostics_for_task(request, task, payload, current_user)


@router.get("/context", response_model=EarthPredictContextResponse)
async def get_earth_predict_context(
    request: Request,
    training_task_id: int,
    current_user: User = Depends(get_current_user),
):
    """Return the selectable historical forecast origins and the trained grid."""
    task = await _load_task(request, training_task_id, current_user)
    try:
        context = await asyncio.to_thread(
            build_prediction_context, task, _registry(request)
        )
    except (DatasetRequestError, EarthPackageError) as exc:
        raise _error_response(exc) from exc
    except Exception as exc:  # pragma: no cover - surfaced for troubleshooting
        logger.exception("Earth prediction context failed: task_id=%s", training_task_id)
        raise HTTPException(status_code=500, detail=f"Earth prediction context failed: {exc}") from exc
    return {
        "planet": "earth",
        "task_id": context.task_id,
        "dataset_id": context.dataset_id,
        "dataset_version": context.dataset_version,
        "dataset_fingerprint": context.dataset_fingerprint,
        "target": context.target,
        "target_unit": context.target_unit,
        "window": context.window,
        "horizon": context.horizon,
        "input_channel_order": context.input_channel_order,
        "input_units": context.input_units,
        "grid": {
            "shape": context.grid_shape,
            "latitude": context.latitude,
            "longitude": context.longitude,
        },
        "origins": context.origins,
        "training_split_end": context.training_split_end,
        "metrics": context.metrics,
        "run": context.run,
        "model": context.model,
        "warnings": context.warnings,
        **getattr(context, "temporal", {}),
    }


@router.post("/training-models/compare")
async def compare_earth_models(request: Request, payload: EarthCompareRequest,
                               current_user: User = Depends(get_current_user)):
    if any(task_id <= 0 for task_id in payload.task_ids) or len(set(payload.task_ids)) != len(payload.task_ids):
        raise HTTPException(422, detail={"code": "invalid_earth_comparison", "message": "Select distinct positive task IDs"})
    tasks = [await _load_task(request, task_id, current_user) for task_id in payload.task_ids]
    try:
        from services.research_export_sources import register_earth_metrics
        for task in tasks:
            require_active_dataset(training_task_dataset_id(task))
        guards = [task_guard(task) if task.output_model_path else None for task in tasks]
        result = await asyncio.to_thread(compare_earth_test_metrics, tasks, _registry(request))
        for task, guard, item in zip(tasks, guards, result["items"]):
            if task_guard(task) != guard:
                raise DatasetRequestError("invalid_earth_training_artifact", "Model changed during comparison", status_code=409)
            item["metrics"] = register_earth_metrics(item, task, current_user.id)
        return result
    except (DatasetRequestError, EarthPackageError) as exc:
        raise _error_response(exc) from exc
    except ArtifactGuardError as exc:
        raise HTTPException(409, detail={"code": "invalid_earth_training_artifact", "message": "Model artifact is unavailable"}) from exc


@router.post("/run", response_model=EarthPredictResponse)
async def run_earth_predict(
    request: Request,
    payload: EarthPredictRequest,
    current_user: User = Depends(get_current_user),
):
    """Forecast three-hour leads after a historical UTC origin."""
    task = await _load_task(request, payload.training_task_id, current_user)
    registry = _registry(request)
    try:
        require_active_dataset(training_task_dataset_id(task))
        _require_completed_earth_task(task)
        # Validate the artifact before task_guard() so missing, unreadable, or
        # malformed checkpoints never fall through to the generic 500 branch.
        _load_checkpoint(task)
        export_guard = task_guard(task)
        result = await run_inference_compute(lambda: asyncio.to_thread(
            run_earth_prediction, task, payload.forecast_origin, registry
        ))
        current_task = await _load_task(request, payload.training_task_id, current_user)
        _require_completed_earth_task(current_task)
        if task_guard(current_task) != export_guard:
            raise DatasetRequestError(
                "invalid_earth_training_artifact",
                "The model artifact changed during prediction",
                status_code=409,
            )
        result = await asyncio.to_thread(register_earth, result, current_task, current_user.id)
    except (DatasetRequestError, EarthPackageError) as exc:
        raise _error_response(exc) from exc
    except ArtifactGuardError as exc:
        raise _error_response(exc) from exc
    except Exception as exc:  # pragma: no cover - surfaced for troubleshooting
        logger.exception(
            "Earth prediction failed: task_id=%s origin=%s",
            payload.training_task_id,
            payload.forecast_origin,
        )
        raise HTTPException(status_code=500, detail=f"Earth prediction failed: {exc}") from exc
    return result
