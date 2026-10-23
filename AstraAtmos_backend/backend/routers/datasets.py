"""Read-only server dataset catalog endpoints."""

from fastapi import APIRouter, HTTPException, Request

from config import DEFAULT_EARTH_DATASET_ID
from schemas.datasets import DatasetDescriptor, DatasetListResponse
from services.dataset_identity import DatasetRequestError

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=DatasetListResponse, response_model_exclude_unset=True)
def list_datasets(request: Request):
    return {
        "items": request.app.state.dataset_registry.list_datasets(),
        "default_earth_dataset_id": getattr(
            request.app.state, "default_earth_dataset_id", DEFAULT_EARTH_DATASET_ID
        ),
    }


@router.get("/{dataset_id}", response_model=DatasetDescriptor, response_model_exclude_unset=True)
def get_dataset(dataset_id: str, request: Request):
    try:
        return request.app.state.dataset_registry.get_dataset(dataset_id)
    except DatasetRequestError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
