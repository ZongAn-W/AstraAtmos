"""
棰勬祴鍒嗘瀽椤?鈥?API 璺敱
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Query, Body

from auth.dependencies import get_optional_user
from database.models import User

from schemas.predict import (
    PredictRequest, PredictResponse,
    EvalMetricsResponse, DiurnalResponse,
    TrainingModelCompareRequest, TrainingModelCompareResponse,
    ErrorDistributionResponse, PermutationImportanceResponse,
)
from config import DEFAULT_MARS_YEAR
from services.analysis_service import AnalysisService
from services.dataset_identity import DatasetRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["棰勬祴鍒嗘瀽"])


def _get_analysis_service(request: Request):
    return request.app.state.analysis_service


def _get_training_inference_service(request: Request):
    service = getattr(request.app.state, "training_inference_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="training inference service unavailable")
    return service


def _require_trained_model_user(current_user: User | None) -> User:
    if current_user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication is required for trained-model analysis",
        )
    return current_user


def _require_training_task_id(training_task_id: int | None) -> int:
    """预测分析只支持已训练模型；默认 PredRNNv2 预测链路已下线。"""
    if not training_task_id:
        raise HTTPException(
            status_code=400,
            detail="training_task_id is required: the default PredRNNv2 prediction chain is no longer available",
        )
    return training_task_id


def _normalize_predict_source(data_source: str | None) -> str:
    requested = (data_source or "default").strip().lower()
    if requested == "personal":
        raise HTTPException(
            status_code=400,
            detail="Personal raw uploads are only available in Data Overview; prediction uses server-managed datasets.",
        )
    if requested not in ("default",):
        raise HTTPException(status_code=400, detail="data_source must be 'default'")
    return "default"


async def _resolve_diurnal_context(
    request: Request,
    my: int,
    data_source: str,
    current_user: User | None,
) -> tuple[AnalysisService, dict, int]:
    _normalize_predict_source(data_source)
    return (
        _get_analysis_service(request),
        {
            "requested_source": "default",
            "effective_source": "default",
            "fallback": False,
            "message": None,
            "mars_year": my,
        },
        my,
    )


# 鈹€鈹€鈹€ 鏍稿績棰勬祴鎺ュ彛 鈹€鈹€鈹€

@router.post("/run", response_model=PredictResponse)
async def run_prediction(
    request: Request,
    body: PredictRequest = Body(...),
    current_user: User | None = Depends(get_optional_user),
):
    """
    执行预测分析。
    只支持已训练模型：请求必须携带 training_task_id（默认 PredRNNv2 预测链路已下线）。
    返回真值场、预测场与差值场。
    """
    try:
        training_task_id = _require_training_task_id(body.training_task_id)
        current_user = _require_trained_model_user(current_user)
        service = _get_training_inference_service(request)
        result = await service.predict_task(
            task_id=training_task_id,
            ls_start=body.ls_start,
            horizon=body.horizon,
            current_user=current_user,
            data_service=getattr(request.app.state, "data_service", None),
            personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
        )
        source_meta = dict(result.get("source_meta") or {
            "requested_source": "training_task",
            "effective_source": "training_task",
            "fallback": False,
            "message": None,
        })
        source_meta.pop("mars_year", None)
        model_info = dict(result.get("model_info") or {})
        model_info.pop("requested_mars_year", None)
        model_info.pop("mars_year", None)
        return {
            "export_ref": result.get("export_ref"),
            "ground_truth": result["ground_truth"],
            "prediction": result["prediction"],
            "residual": result["residual"],
            "selected_variables": result["selected_variables"],
            "horizon": result["horizon"],
            "input_ls_values": result.get("input_ls_values", []),
            "ls_values": result["ls_values"],
            "model_info": model_info,
            "metrics": result.get("metrics"),
            "source_meta": source_meta,
        }
    except DatasetRequestError as e:
        detail = {"code": e.code, "message": str(e)}
        if e.availability_reason:
            detail["availability_reason"] = e.availability_reason
        raise HTTPException(status_code=e.status_code, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("trained-model prediction failed")
        raise HTTPException(status_code=500, detail=f"预测错误: {e}")


# 鈹€鈹€鈹€ 璇勪及鎸囨爣 鈹€鈹€鈹€

@router.post("/metrics", response_model=EvalMetricsResponse)
async def get_eval_metrics(
    request: Request,
    body: PredictRequest = Body(...),
    current_user: User | None = Depends(get_optional_user),
):
    """获取预测评估指标（RMSE, MAE, SSIM, R²）；只支持已训练模型。"""
    try:
        training_task_id = _require_training_task_id(body.training_task_id)
        current_user = _require_trained_model_user(current_user)
        service = _get_training_inference_service(request)
        metrics = await service.task_test_set_metrics(
            task_id=training_task_id,
            ls_start=body.ls_start,
            horizon=body.horizon,
            current_user=current_user,
            data_service=getattr(request.app.state, "data_service", None),
            personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
        )
        metrics = dict(metrics)
        metrics["source_meta"] = dict(metrics.get("source_meta") or {
            "requested_source": "training_task",
            "effective_source": "training_task",
            "fallback": False,
            "message": None,
        })
        metrics["source_meta"].pop("mars_year", None)
        return metrics
    except DatasetRequestError as e:
        detail = {"code": e.code, "message": str(e)}
        if e.availability_reason:
            detail["availability_reason"] = e.availability_reason
        raise HTTPException(status_code=e.status_code, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("prediction metrics calculation failed")
        raise HTTPException(
            status_code=500,
            detail=f"prediction metrics error: {e}",
        )


@router.post("/training-models/compare", response_model=TrainingModelCompareResponse)
async def compare_training_models(
    request: Request,
    body: TrainingModelCompareRequest = Body(...),
    current_user: User | None = Depends(get_optional_user),
):
    """Compare completed training models using full test-set metrics."""
    try:
        current_user = _require_trained_model_user(current_user)
        service = _get_training_inference_service(request)
        return await service.compare_task_test_set_metrics(
            task_ids=body.task_ids,
            horizon=body.horizon,
            current_user=current_user,
            data_service=getattr(request.app.state, "data_service", None),
            personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
        )
    except DatasetRequestError as e:
        detail = {"code": e.code, "message": str(e)}
        if e.availability_reason:
            detail["availability_reason"] = e.availability_reason
        raise HTTPException(status_code=e.status_code, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise


@router.post("/training-models/compare-error-distribution")
async def compare_training_model_error_distributions(
    request: Request,
    body: TrainingModelCompareRequest = Body(...),
    current_user: User | None = Depends(get_optional_user),
):
    """Compare completed training models using full test-set error distributions."""
    try:
        current_user = _require_trained_model_user(current_user)
        service = _get_training_inference_service(request)
        return await service.compare_task_error_distributions(
            task_ids=body.task_ids,
            horizon=body.horizon,
            current_user=current_user,
            data_service=getattr(request.app.state, "data_service", None),
            personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
        )
    except DatasetRequestError as e:
        detail = {"code": e.code, "message": str(e)}
        if e.availability_reason:
            detail["availability_reason"] = e.availability_reason
        raise HTTPException(status_code=e.status_code, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise


@router.post("/training-models/compare-pfi")
async def compare_training_model_pfi(
    request: Request,
    body: TrainingModelCompareRequest = Body(...),
    current_user: User | None = Depends(get_optional_user),
):
    """Compare completed training models using full test-set permutation importance."""
    try:
        current_user = _require_trained_model_user(current_user)
        service = _get_training_inference_service(request)
        return await service.compare_task_permutation_importance(
            task_ids=body.task_ids,
            horizon=body.horizon,
            current_user=current_user,
            data_service=getattr(request.app.state, "data_service", None),
            personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
        )
    except DatasetRequestError as e:
        detail = {"code": e.code, "message": str(e)}
        if e.availability_reason:
            detail["availability_reason"] = e.availability_reason
        raise HTTPException(status_code=e.status_code, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise


# 鈹€鈹€鈹€ 鏄煎鍙樺寲 鈹€鈹€鈹€

@router.get("/diurnal", response_model=DiurnalResponse)
async def get_diurnal_data(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR),
    ls: float = Query(90.0, ge=0, le=360),
    lat_band: str = Query("Equatorial (30S-30N)", description="纬度带名称"),
    data_source: str = Query("default", description="default"),
    current_user: User | None = Depends(get_optional_user),
):
    """获取指定纬度带的臭氧昼夜变化曲线"""
    try:
        vs, source_meta, resolved_year = await _resolve_diurnal_context(
            request, my, data_source, current_user
        )
        result = vs.get_diurnal_data(resolved_year, ls, lat_band)
        result["source_meta"] = source_meta
        return result
    except DatasetRequestError as e:
        detail = {"code": e.code, "message": str(e)}
        if e.availability_reason:
            detail["availability_reason"] = e.availability_reason
        raise HTTPException(status_code=e.status_code, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/error-distribution", response_model=ErrorDistributionResponse)
async def get_error_distribution(
    request: Request,
    vars: str = Query("Temperature,Dust_Optical_Depth,Solar_Flux_DN,U_Wind,V_Wind"),
    training_task_id: int | None = Query(None, ge=1),
    horizon: int = Query(3, ge=1, le=30),
    current_user: User | None = Depends(get_optional_user),
):
    """鑾峰彇鏁翠釜娴嬭瘯闆嗕笂鐨勮宸垎甯冦€佹牳瀵嗗害鏁ｇ偣鍙婃煴鐘跺浘鏁版嵁"""
    try:
        selected_variables = [v.strip() for v in vars.split(",") if v.strip()]
        training_task_id = _require_training_task_id(training_task_id)
        current_user = _require_trained_model_user(current_user)
        service = _get_training_inference_service(request)
        return await service.task_error_distribution(
            task_id=training_task_id,
            selected_variables=selected_variables,
            horizon=horizon,
            current_user=current_user,
            data_service=getattr(request.app.state, "data_service", None),
            personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
        )
    except DatasetRequestError as e:
        detail = {"code": e.code, "message": str(e)}
        if e.availability_reason:
            detail["availability_reason"] = e.availability_reason
        raise HTTPException(status_code=e.status_code, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("prediction error-distribution calculation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/permutation-importance", response_model=PermutationImportanceResponse)
async def get_permutation_importance(
    request: Request,
    vars: str = Query("Temperature,Dust_Optical_Depth,Solar_Flux_DN,U_Wind,V_Wind"),
    training_task_id: int | None = Query(None, ge=1),
    ls_start: float = Query(90.0, ge=0, le=360),
    horizon: int = Query(3, ge=1, le=30),
    current_user: User | None = Depends(get_optional_user),
):
    """鑾峰彇鎺掑垪鐗瑰緛閲嶈鎬?(Permutation Feature Importance) 鍒嗚В缁撴灉"""
    try:
        selected_variables = [v.strip() for v in vars.split(",") if v.strip()]
        training_task_id = _require_training_task_id(training_task_id)
        current_user = _require_trained_model_user(current_user)
        service = _get_training_inference_service(request)
        return await service.task_permutation_importance(
            task_id=training_task_id,
            selected_variables=selected_variables,
            ls_start=ls_start,
            horizon=horizon,
            current_user=current_user,
            data_service=getattr(request.app.state, "data_service", None),
            personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
        )
    except DatasetRequestError as e:
        detail = {"code": e.code, "message": str(e)}
        if e.availability_reason:
            detail["availability_reason"] = e.availability_reason
        raise HTTPException(status_code=e.status_code, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("permutation importance calculation failed")
        raise HTTPException(status_code=500, detail=str(e))


