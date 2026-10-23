from pathlib import Path
import json
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect

from auth.dependencies import get_current_user
from database.models import ModelTrainingTask, User
from schemas.training import (
    LogResponse,
    RenameTrainingModelRequest,
    ReplaceTrainingTaskTagsRequest,
    TrainingStartRequest,
    TrainingTagNameRequest,
    TrainingTagResponse,
    TrainingTaskResponse,
    UpdateTrainingTaskTagsRequest,
    TrainingWeightFileListResponse,
    TrainingWeightFileResponse,
)
from services.dataset_identity import DatasetRequestError, is_earth_training_task
from services.inference_service import InferenceService, require_mars_prediction_task
from services.training_service import TrainingService
from services.training_weight_service import TrainingWeightService
from services.training_tag_service import TagNameConflict, TrainingTagService

router = APIRouter(tags=["Training"])

training_service = TrainingService()
inference_service = InferenceService()
training_weight_service = TrainingWeightService()
training_tag_service = TrainingTagService()


async def _serialize_tasks(tasks, current_user: User) -> list[TrainingTaskResponse]:
    task_tags = await training_tag_service.get_task_tags([task.id for task in tasks], current_user.id)
    return [TrainingTaskResponse.model_validate(task).model_copy(update={
        "tags": [TrainingTagResponse(**tag) for tag in task_tags[task.id]],
    }) for task in tasks]


def _tag_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        status = 404
    elif isinstance(exc, PermissionError):
        status = 403
    elif isinstance(exc, TagNameConflict):
        status = 409
    else:
        status = 400
    return HTTPException(status_code=status, detail=str(exc))


@router.get("/training/tags", response_model=list[TrainingTagResponse])
async def list_training_tags(current_user: User = Depends(get_current_user)):
    return await training_tag_service.list_tags(current_user.id)


@router.post("/training/tags", response_model=TrainingTagResponse)
async def create_training_tag(req: TrainingTagNameRequest, current_user: User = Depends(get_current_user)):
    try:
        return await training_tag_service.create_tag(current_user.id, req.name)
    except ValueError as exc:
        raise _tag_error(exc) from exc


@router.patch("/training/tags/{tag_id}", response_model=TrainingTagResponse)
async def rename_training_tag(tag_id: int, req: TrainingTagNameRequest, current_user: User = Depends(get_current_user)):
    try:
        return await training_tag_service.rename_tag(tag_id, current_user.id, req.name)
    except (FileNotFoundError, ValueError) as exc:
        raise _tag_error(exc) from exc


@router.delete("/training/tags/{tag_id}")
async def delete_training_tag(tag_id: int, current_user: User = Depends(get_current_user)):
    try:
        await training_tag_service.delete_tag(tag_id, current_user.id)
    except FileNotFoundError as exc:
        raise _tag_error(exc) from exc
    return {"status": "success"}


@router.put("/training/tasks/{task_id}/tags", response_model=TrainingTaskResponse)
async def replace_training_task_tags(
    task_id: int, req: ReplaceTrainingTaskTagsRequest, current_user: User = Depends(get_current_user),
):
    try:
        task = await training_tag_service.replace_task_tags(
            task_id, req.tag_ids, current_user.id, is_admin=_is_admin(current_user),
        )
        return (await _serialize_tasks([task], current_user))[0]
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        raise _tag_error(exc) from exc


@router.patch("/training/task-tags", response_model=list[TrainingTaskResponse])
async def update_training_task_tags(
    req: UpdateTrainingTaskTagsRequest, current_user: User = Depends(get_current_user),
):
    try:
        tasks = await training_tag_service.update_task_tags(
            req.task_ids, req.tag_ids, req.operation, current_user.id, is_admin=_is_admin(current_user),
        )
        return await _serialize_tasks(tasks, current_user)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        raise _tag_error(exc) from exc


def _is_admin(user: User) -> bool:
    return (getattr(user, "role", "") or "").lower() == "admin"


def _service_weight(request: Request) -> TrainingWeightService:
    return getattr(request.app.state, "training_weight_service", None) or training_weight_service


def _parse_report(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _serialize_weight(record) -> TrainingWeightFileResponse:
    return TrainingWeightFileResponse(
        id=record.id,
        user_id=record.user_id,
        original_filename=record.original_filename,
        content_hash=record.content_hash,
        file_size=record.file_size,
        status=record.status,
        validation_report=_parse_report(record.validation_report),
        created_at=record.created_at,
    )


async def _get_task_with_access_check(task_id: int, current_user: User) -> ModelTrainingTask:
    task = await training_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not _is_admin(current_user) and task.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="No permission to access this task")

    return task


@router.websocket("/ws/training/{task_id}")
async def training_ws(websocket: WebSocket, task_id: str):
    from services.ws_manager import manager as ws_manager

    await ws_manager.connect(websocket, task_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, task_id)
    except Exception:
        ws_manager.disconnect(websocket, task_id)


@router.get("/training/scripts", response_model=List[str])
async def get_scripts():
    return training_service.get_available_scripts()


@router.post("/training/weights", response_model=TrainingWeightFileResponse)
async def upload_training_weight(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    try:
        source = await file.read()
        record = await _service_weight(request).create_from_upload(
            user_id=current_user.id,
            original_filename=file.filename or "",
            content=source,
        )
        return _serialize_weight(record)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/training/weights", response_model=TrainingWeightFileListResponse)
async def list_training_weights(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    records = await _service_weight(request).list_user_weights(current_user.id)
    return TrainingWeightFileListResponse(items=[_serialize_weight(record) for record in records])


@router.delete("/training/weights/{weight_id}")
async def delete_training_weight(
    weight_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    try:
        await _service_weight(request).soft_delete_weight(weight_id, current_user.id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"status": "success", "message": "Training weight deleted"}


@router.post("/training/start", response_model=TrainingTaskResponse)
async def start_training(
    req: TrainingStartRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    try:
        task = await training_service.start_training(
            user_id=current_user.id,
            model_script=req.model_script,
            hyperparameters=req.hyperparameters,
            custom_model_name=req.model_name,
            data_source=req.data_source,
            data_service=getattr(request.app.state, "data_service", None),
            personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
            model_source=req.model_source,
            uploaded_model_id=req.uploaded_model_id,
            user_model_service=getattr(request.app.state, "user_model_service", None),
            training_weight_service=_service_weight(request),
            is_admin=_is_admin(current_user),
            tag_ids=req.tag_ids,
            dataset_id=req.dataset_id,
            dataset_registry=getattr(request.app.state, "dataset_registry", None),
        )
        return (await _serialize_tasks([task], current_user))[0]
    except DatasetRequestError as e:
        raise HTTPException(status_code=e.status_code, detail={
            "code": e.code, "message": str(e),
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/tasks", response_model=List[TrainingTaskResponse])
async def get_tasks(current_user: User = Depends(get_current_user)):
    tasks = await training_service.get_all_tasks()
    if not _is_admin(current_user):
        tasks = [task for task in tasks if task.user_id == current_user.id]
    return await _serialize_tasks(tasks, current_user)


@router.patch("/training/tasks/{task_id}/name", response_model=TrainingTaskResponse)
async def rename_training_model(
    task_id: int,
    req: RenameTrainingModelRequest,
    current_user: User = Depends(get_current_user),
):
    await _get_task_with_access_check(task_id, current_user)
    try:
        task = await training_service.rename_model(task_id, req.model_name)
        return (await _serialize_tasks([task], current_user))[0]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/training/tasks/{task_id}/logs", response_model=LogResponse)
async def get_task_logs(task_id: int, current_user: User = Depends(get_current_user)):
    task = await _get_task_with_access_check(task_id, current_user)

    if not task.log_file_path or not Path(task.log_file_path).exists():
        try:
            metrics = json.loads(task.metrics or "{}")
        except (TypeError, ValueError):
            metrics = {}
        error = metrics.get("error") if isinstance(metrics, dict) else None
        if error:
            return LogResponse(lines=[f"[Training failed before logs started] {error}"])
        return LogResponse(lines=["[No logs available yet]"])

    try:
        with open(task.log_file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if not lines:
            return LogResponse(lines=["[Waiting for log output...]"])
        return LogResponse(lines=lines[-500:])
    except Exception as e:
        return LogResponse(lines=[f"Error reading logs: {e}"])


@router.post("/training/tasks/{task_id}/stop")
async def stop_task(task_id: int, current_user: User = Depends(get_current_user)):
    await _get_task_with_access_check(task_id, current_user)

    success = await training_service.stop_training(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot stop task (not found or not running)")
    return {"message": "Task stopped", "status": "success"}


@router.post("/training/tasks/{task_id}/cancel")
async def cancel_task(task_id: int, current_user: User = Depends(get_current_user)):
    await _get_task_with_access_check(task_id, current_user)
    success = await training_service.cancel_training(task_id)
    if not success:
        raise HTTPException(status_code=409, detail="Cannot cancel task (only queued tasks can be cancelled)")
    return {"message": "Task cancelled", "status": "success"}


@router.delete("/training/tasks/{task_id}")
async def delete_task(task_id: int, current_user: User = Depends(get_current_user)):
    await _get_task_with_access_check(task_id, current_user)

    success = await training_service.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete task (not found)")
    return {"message": "Task deleted", "status": "success"}


@router.post("/training/tasks/{task_id}/action")
async def perform_task_action(
    task_id: int,
    action: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    task = await _get_task_with_access_check(task_id, current_user)

    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Cannot perform action on incomplete task")

    if action == "test":
        if is_earth_training_task(task):
            from routers.earth_predict import diagnostics_for_task
            from schemas.earth_diagnostics import EarthDiagnosticsRequest
            results = await diagnostics_for_task(request, task, EarthDiagnosticsRequest(training_task_id=task_id), current_user)
            return {"status": "success", "data": results}
        try:
            require_mars_prediction_task(task)
        except DatasetRequestError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        temp_data_root = None
        try:
            data_dirs, temp_data_root = await training_service.prepare_task_inference_data_env(
                task,
                data_service=getattr(request.app.state, "data_service", None),
                personal_source_service=getattr(request.app.state, "personal_data_source_service", None),
            )
            from services.inference_compute import run_inference_compute
            results = await run_inference_compute(lambda: inference_service.get_test_results(task_id, data_dirs=data_dirs))
            return {"status": "success", "data": results}
        except DatasetRequestError:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
        finally:
            training_service.cleanup_temp_data_root(temp_data_root)

    return {"message": f"Action '{action}' executed for task {task_id}", "status": "success"}
