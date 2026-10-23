"""Read-only 2D Earth overview endpoints.

Handlers are synchronous so FastAPI runs the NumPy work in its thread pool
instead of blocking the event loop.
"""

from fastapi import APIRouter, HTTPException, Query, Request

from schemas.earth_overview import (
    EarthFieldResponse,
    EarthPointSeriesResponse,
    EarthRegionalSeriesResponse,
)
from services.dataset_identity import DatasetRequestError
from services.earth_dataset_metadata import EarthPackageError
from services.earth_overview_service import EarthOverviewError

router = APIRouter(prefix="/datasets/{dataset_id}/overview", tags=["Earth Overview"])

FINGERPRINT_QUERY = Query(..., pattern=r"^[0-9a-f]{64}$")


def _error_response(exc: Exception) -> HTTPException:
    """Map a service/registry failure to the documented public error shape."""
    if isinstance(exc, DatasetRequestError):
        detail = {"code": exc.code, "message": str(exc)}
        if exc.availability_reason:
            detail["availability_reason"] = exc.availability_reason
        return HTTPException(status_code=exc.status_code, detail=detail)
    if isinstance(exc, EarthOverviewError):
        return HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        )
    if isinstance(exc, EarthPackageError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "dataset_unavailable",
                "message": "The registered Earth dataset is not available",
                "availability_reason": exc.reason,
            },
        )
    raise exc


@router.get("/field", response_model=EarthFieldResponse)
def get_field(
    request: Request,
    dataset_id: str,
    date: str | None = Query(default=None),
    timestamp: str | None = Query(default=None),
    render_stride: int = Query(default=4),
    variable: str = Query(...),
    expected_fingerprint: str = FINGERPRINT_QUERY,
):
    try:
        return request.app.state.earth_overview_service.get_field(
            dataset_id=dataset_id,
            expected_fingerprint=expected_fingerprint,
            date=date,
            timestamp=timestamp,
            render_stride=render_stride,
            variable=variable,
        )
    except (DatasetRequestError, EarthOverviewError, EarthPackageError) as exc:
        raise _error_response(exc) from exc


@router.get("/regional-series", response_model=EarthRegionalSeriesResponse)
def get_regional_series(
    request: Request,
    dataset_id: str,
    variable: str = Query(...),
    expected_fingerprint: str = FINGERPRINT_QUERY,
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
):
    try:
        return request.app.state.earth_overview_service.get_regional_series(
            dataset_id=dataset_id,
            expected_fingerprint=expected_fingerprint,
            variable=variable,
            start=start,
            end=end,
        )
    except (DatasetRequestError, EarthOverviewError, EarthPackageError) as exc:
        raise _error_response(exc) from exc


@router.get("/point-series", response_model=EarthPointSeriesResponse)
def get_point_series(
    request: Request,
    dataset_id: str,
    variable: str = Query(...),
    expected_fingerprint: str = FINGERPRINT_QUERY,
    lat: float = Query(...),
    lon: float = Query(...),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
):
    try:
        return request.app.state.earth_overview_service.get_point_series(
            dataset_id=dataset_id,
            expected_fingerprint=expected_fingerprint,
            variable=variable,
            lat=lat,
            lon=lon,
            start=start,
            end=end,
        )
    except (DatasetRequestError, EarthOverviewError, EarthPackageError) as exc:
        raise _error_response(exc) from exc
