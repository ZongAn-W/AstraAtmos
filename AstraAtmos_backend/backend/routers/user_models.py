import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from auth.dependencies import get_current_user
from database.models import User, UserModelPackage
from schemas.user_models import UserModelListResponse, UserModelPackageResponse, UserModelRenameRequest
from services.user_model_service import UserModelService

router = APIRouter(prefix="/user-models", tags=["User Models"])
REPO_ROOT = Path(__file__).resolve().parents[3]
UPLOADED_MODEL_DOWNLOAD_ASSETS = {
    "earth-3hourly-5deg-template": {
        "path": REPO_ROOT / "docs" / "earth-3hourly-5deg-model-template.py",
        "download_name": "astraatmos_earth_3hourly_5deg_model_v1.py",
        "media_type": "text/x-python; charset=utf-8",
    },
    "earth-3hourly-5deg-guide": {
        "path": REPO_ROOT / "docs" / "earth-merra2-5deg.md",
        "download_name": "astraatmos_earth_3hourly_5deg_model_v1.md",
        "media_type": "text/markdown; charset=utf-8",
    },
    "guide": {
        "path": REPO_ROOT / "docs" / "uploaded-model-training.md",
        "download_name": "astraatmos_uploaded_model_guide.md",
        "media_type": "text/markdown; charset=utf-8",
    },
    "template": {
        "path": REPO_ROOT / "docs" / "uploaded-model-template.py",
        "download_name": "astraatmos_uploaded_model_template.py",
        "media_type": "text/x-python; charset=utf-8",
    },
    "earth-3hourly-template": {
        "path": REPO_ROOT / "docs" / "earth-3hourly-uploaded-model-template.py",
        "download_name": "astraatmos_earth_3hourly_model_v1.py",
        "media_type": "text/x-python; charset=utf-8",
    },
    "earth-3hourly-shared-gate-template": {
        "path": REPO_ROOT / "docs" / "earth-3hourly-four-positions-shared-no-after-decoder-model.py",
        "download_name": "astraatmos_earth_3hourly_shared_gate_model_v1.py",
        "media_type": "text/x-python; charset=utf-8",
    },
    "earth-3hourly-guide": {
        "path": REPO_ROOT / "docs" / "earth-3hourly-uploaded-model.md",
        "download_name": "astraatmos_earth_3hourly_model_v1.md",
        "media_type": "text/markdown; charset=utf-8",
    },
    "earth-3hourly-fullgrid-template": {
        "path": REPO_ROOT / "docs" / "earth-3hourly-fullgrid-model-template.py",
        "download_name": "astraatmos_earth_3hourly_fullgrid_model_v2.py",
        "media_type": "text/x-python; charset=utf-8",
    },
    "earth-3hourly-fullgrid-guide": {
        "path": REPO_ROOT / "docs" / "earth-3hourly-fullgrid-model.md",
        "download_name": "astraatmos_earth_3hourly_fullgrid_model_v2.md",
        "media_type": "text/markdown; charset=utf-8",
    },
}


def get_uploaded_model_download_assets() -> dict[str, dict[str, Any]]:
    return UPLOADED_MODEL_DOWNLOAD_ASSETS


def get_uploaded_model_download_asset(kind: str) -> dict[str, Any]:
    asset = UPLOADED_MODEL_DOWNLOAD_ASSETS.get((kind or "").strip().lower())
    if not asset or not asset["path"].is_file():
        raise HTTPException(status_code=404, detail="Uploaded model download asset not found")
    return asset


def _service(request: Request) -> UserModelService:
    return getattr(request.app.state, "user_model_service", None) or UserModelService()


def _parse_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _normalize_param_schema(value: str | None) -> dict[str, Any]:
    parsed = _parse_json(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _normalize_output_shape(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return value


def _normalize_validation_report(value: str | None) -> dict[str, Any]:
    parsed = _parse_json(value, {})
    if not isinstance(parsed, dict):
        parsed = {}

    report: dict[str, Any] = {
        "ok": parsed.get("ok") if isinstance(parsed.get("ok"), bool) else False,
        "errors": _normalize_string_list(parsed.get("errors")),
        "warnings": _normalize_string_list(parsed.get("warnings")),
        "output_shape": _normalize_output_shape(parsed.get("output_shape")),
    }
    if isinstance(parsed.get("code"), str):
        report["code"] = parsed["code"]
    # The Earth capability result is part of the upload contract now, so it must
    # survive serialization; the training page reads it before offering the model
    # for an Earth experiment. It is only published when the model opted in.
    datasets = parsed.get("datasets")
    if isinstance(datasets, dict) and datasets:
        report["datasets"] = datasets
    earth = parsed.get("earth")
    if isinstance(earth, dict):
        report["earth"] = {
            "compatible": earth.get("compatible") if isinstance(earth.get("compatible"), bool) else False,
            "errors": _normalize_string_list(earth.get("errors")),
            "output_shape": _normalize_output_shape(earth.get("output_shape")),
        }
    earth_datasets = parsed.get("earth_datasets")
    if isinstance(earth_datasets, dict):
        report["earth_datasets"] = {
            str(dataset_id): {
                "dataset_id": str(dataset_id),
                "status": value.get("status", "unknown") if isinstance(value, dict) else "unknown",
                "code": value.get("code") if isinstance(value, dict) else None,
                "contract_schema": value.get("contract_schema") if isinstance(value, dict) else None,
                "compatible": bool(value.get("compatible")) if isinstance(value, dict) else False,
                "errors": _normalize_string_list(value.get("errors")) if isinstance(value, dict) else [],
                "output_shape": _normalize_output_shape(value.get("output_shape")) if isinstance(value, dict) else None,
            }
            for dataset_id, value in earth_datasets.items()
        }
    if isinstance(parsed.get("mars"), dict):
        report["mars"] = {"compatible": parsed["mars"].get("compatible") is True}
    return report


def _serialize_package(package: UserModelPackage) -> UserModelPackageResponse:
    return UserModelPackageResponse(
        id=package.id,
        user_id=package.user_id,
        display_name=package.display_name,
        version=package.version,
        original_filename=package.original_filename,
        content_hash=package.content_hash,
        param_schema=_normalize_param_schema(package.param_schema),
        description=package.description,
        validation_status=package.validation_status,
        validation_report=_normalize_validation_report(package.validation_report),
        created_at=package.created_at.isoformat() if package.created_at else None,
        updated_at=package.updated_at.isoformat() if package.updated_at else None,
    )


@router.post("", response_model=UserModelPackageResponse)
async def upload_user_model(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".py":
        raise HTTPException(status_code=400, detail="Uploaded model must be a .py file")

    source = await file.read()
    try:
        package = await _service(request).create_from_source(
            user_id=current_user.id,
            original_filename=filename,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_package(package)


@router.get("", response_model=UserModelListResponse)
async def list_user_models(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    packages = await _service(request).list_user_packages(current_user.id)
    return UserModelListResponse(
        items=[_serialize_package(package) for package in packages]
    )


@router.get("/downloads/{kind}")
async def download_uploaded_model_asset(kind: str):
    asset = get_uploaded_model_download_asset(kind)
    return FileResponse(
        path=asset["path"],
        filename=asset["download_name"],
        media_type=asset["media_type"],
    )


@router.get("/{model_id}", response_model=UserModelPackageResponse)
async def get_user_model(
    model_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    try:
        package = await _service(request).get_package_for_user(model_id, current_user.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return _serialize_package(package)


@router.patch("/{model_id}", response_model=UserModelPackageResponse)
async def rename_user_model(
    model_id: str,
    payload: UserModelRenameRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    try:
        package = await _service(request).rename_package(model_id, current_user.id, payload.display_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_package(package)


@router.get("/{model_id}/download")
async def download_user_model(
    model_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    try:
        source_path, filename = await _service(request).get_source_for_download(model_id, current_user.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return FileResponse(
        path=source_path,
        filename=filename,
        media_type="application/octet-stream",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/{model_id}/earth-compatibility")
async def get_user_model_earth_compatibility(
    model_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    dataset_id: str = "earth_merra2",
):
    """Whether this uploaded model may be trained on Earth MERRA-2 data.

    A model validated for Mars is not automatically usable on Earth, so the
    training page asks this before offering the model for an Earth experiment.
    """
    try:
        return await _service(request).get_earth_compatibility(
            model_id, current_user.id, dataset_id=dataset_id.strip().lower()
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "unknown_dataset_id", "message": str(exc)})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/{model_id}/validate", response_model=UserModelPackageResponse)
async def revalidate_user_model(
    model_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    try:
        package = await _service(request).revalidate_package(model_id, current_user.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return _serialize_package(package)


@router.delete("/{model_id}")
async def delete_user_model(
    model_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    try:
        await _service(request).soft_delete_package(model_id, current_user.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return {"status": "success", "message": "Uploaded model deleted"}
