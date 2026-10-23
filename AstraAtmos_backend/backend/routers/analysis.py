"""
Data overview / exploration routes.
"""

import logging
import asyncio

from cachetools import LRUCache
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from auth.dependencies import get_optional_user
from config import DEFAULT_MARS_YEAR, MCD_VARIABLES, OVERVIEW_MCD_VARIABLES
from database.engine import async_session_maker
from database.models import UploadRecord, User
from schemas.explore import (
    CorrelationResponse,
    GlobeDataResponse,
    HeatmapResponse,
    OverviewInfoResponse,
    OverviewOzoneSourcesResponse,
    PointProbeResponse,
    SeasonalBandsResponse,
)
from services.analysis_service import AnalysisService
from services.personal_data_source_service import SingleYearDataView
from services.user_overview_source_service import (
    UserMcdOverviewDataView,
    build_uploaded_nomad_validation,
    build_uploaded_ozone_layer,
)

router = APIRouter(prefix="/explore", tags=["数据探索"])
logger = logging.getLogger(__name__)

_ALLOWED_SOURCES = ("default", "personal")


def _get_personal_analysis_service_cache(request: Request) -> LRUCache:
    cache = getattr(request.app.state, "personal_analysis_service_cache", None)
    if cache is None:
        cache = LRUCache(maxsize=16)
        request.app.state.personal_analysis_service_cache = cache
    return cache


def _personal_cache_key(current_user: User | None, resolution) -> tuple:
    return (
        int(current_user.id if current_user else 0),
        int(resolution.mars_year),
        str(resolution.effective_source),
        str(getattr(resolution, "signature_hash", "") or ""),
    )


def _with_source_meta(payload: dict, source_meta: dict) -> dict:
    out = dict(payload)
    out["source_meta"] = source_meta
    return out


_RAW_UPLOAD_ACTIVE_STATUSES = {"valid", "pending_review", "approved"}


def _reject_legacy_personal(data_source: str) -> None:
    requested = (data_source or "default").strip().lower()
    if requested == "personal":
        raise HTTPException(
            status_code=400,
            detail="data_source=personal has been retired; use mcd_upload_id for Data Overview uploaded MCD data",
        )
    if requested not in ("default", "official"):
        raise HTTPException(status_code=400, detail="data_source must be 'default'")


async def _get_accessible_upload_record(
    upload_id: int,
    current_user: User | None,
    expected_types: set[str],
) -> UploadRecord:
    if current_user is None:
        raise HTTPException(status_code=401, detail="sign in before using uploaded datasets")
    async with async_session_maker() as db:
        record = await db.get(UploadRecord, upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail="uploaded dataset not found")
    if record.user_id != current_user.id and current_user.role != "admin" and record.status != "approved":
        raise HTTPException(status_code=403, detail="no permission to access uploaded dataset")
    if record.status not in _RAW_UPLOAD_ACTIVE_STATUSES:
        raise HTTPException(status_code=400, detail=f"uploaded dataset status '{record.status}' is not usable")
    data_type = str(record.data_type or "").lower()
    if data_type not in expected_types:
        allowed = ", ".join(sorted(expected_types))
        raise HTTPException(status_code=400, detail=f"uploaded dataset must be one of: {allowed}")
    return record


def _uploaded_source_meta(record: UploadRecord, effective_source: str) -> dict:
    return {
        "requested_source": "user_upload",
        "effective_source": effective_source,
        "fallback": False,
        "message": None,
        "mars_year": int(record.mars_year or DEFAULT_MARS_YEAR),
        "upload_id": int(record.id),
        "upload_filename": record.filename,
        "data_type": record.data_type,
    }


def _normalize_source(data_source: str) -> str:
    s = (data_source or "default").strip().lower()
    if s not in _ALLOWED_SOURCES:
        raise HTTPException(status_code=400, detail="data_source must be 'default' or 'personal'")
    return s


def _validate_overview_variable(variable: str, *, include_ozone: bool = False) -> str:
    allowed = (["o3col"] if include_ozone else []) + OVERVIEW_MCD_VARIABLES
    if variable not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"variable must be one of: {', '.join(allowed)}",
        )
    return variable


async def _resolve_analysis_context(
    request: Request,
    my: int,
    data_source: str,
    current_user: User | None,
) -> tuple[AnalysisService, dict, int]:
    requested = _normalize_source(data_source)

    if requested == "default":
        return (
            request.app.state.analysis_service,
            {
                "requested_source": "default",
                "effective_source": "default",
                "fallback": False,
                "message": None,
                "mars_year": my,
            },
            my,
        )

    resolver = request.app.state.personal_data_source_service
    resolution = await resolver.resolve_for_year("personal", my, current_user.id if current_user else None)

    if resolution.effective_source == "default":
        service = request.app.state.analysis_service
    else:
        cache = _get_personal_analysis_service_cache(request)
        key = _personal_cache_key(current_user, resolution)
        cached_service = cache.get(key)
        if cached_service is not None:
            logger.info(
                "personal analysis service cache hit (uid=%s, MY%s, mode=%s)",
                current_user.id if current_user else "anon",
                resolution.mars_year,
                resolution.effective_source,
            )
            service = cached_service
        else:
            data_view = SingleYearDataView(
                mars_year=resolution.mars_year,
                openmars_data=resolution.openmars_data,
                aligned_mcd_data=resolution.aligned_mcd_data,
                mcd_raw_data=resolution.mcd_raw_data,
            )
            service = AnalysisService(data_view)
            cache[key] = service
            logger.info(
                "personal analysis service cache miss -> create (uid=%s, MY%s, mode=%s)",
                current_user.id if current_user else "anon",
                resolution.mars_year,
                resolution.effective_source,
            )

    return service, resolution.source_meta(), resolution.mars_year


def _overview_source_meta(data_source: str, mars_year: int) -> dict:
    _reject_legacy_personal(data_source)
    requested = (data_source or "default").strip().lower()
    return {
        "requested_source": "official" if requested == "official" else "default",
        "effective_source": "default",
        "fallback": False,
        "message": None,
        "mars_year": mars_year,
    }


async def _resolve_overview_context(
    request: Request,
    my: int,
    data_source: str,
    current_user: User | None,
    mcd_upload_id: int | None = None,
    include_mcd_hourly: bool = False,
) -> tuple[AnalysisService, dict, int]:
    _reject_legacy_personal(data_source)
    if not mcd_upload_id:
        return request.app.state.mcd_overview_analysis_service, _overview_source_meta(data_source, my), my

    record = await _get_accessible_upload_record(mcd_upload_id, current_user, {"mcd"})
    data = await request.app.state.user_data_service.get_loaded_dataset(record.id)
    hourly_data = None
    if include_mcd_hourly:
        hourly_data = await request.app.state.user_data_service.get_loaded_mcd_hourly_dataset(record.id)
    resolved_year = int(record.mars_year or DEFAULT_MARS_YEAR)
    view = UserMcdOverviewDataView(upload_id=record.id, mars_year=resolved_year, data=data, hourly_data=hourly_data)
    service = AnalysisService(view, mcd_variables=OVERVIEW_MCD_VARIABLES)
    return service, _uploaded_source_meta(record, "user_mcd"), resolved_year


@router.get("/overview/info", response_model=OverviewInfoResponse)
async def get_overview_info(
    request: Request,
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        _reject_legacy_personal(data_source)
        if mcd_upload_id:
            record = await _get_accessible_upload_record(mcd_upload_id, current_user, {"mcd"})
            data = await request.app.state.user_data_service.get_loaded_dataset(record.id)
            resolved_year = int(record.mars_year or DEFAULT_MARS_YEAR)
            view = UserMcdOverviewDataView(upload_id=record.id, mars_year=resolved_year, data=data)
            ls_min, ls_max = view.get_ls_range(resolved_year)
            official_capabilities = dict(request.app.state.mcd_overview_service.get_ozone_capabilities())
            coverage = dict(official_capabilities.get("coverage") or {})
            coverage["mcd"] = {
                str(resolved_year): [{"start": float(ls_min), "end": float(ls_max)}]
            }
            official_capabilities["coverage"] = coverage
            return {
                "available_years": [resolved_year],
                "timeline": {
                    "min": float(ls_min),
                    "max": float(ls_max),
                    "step": 5.0,
                },
                "ozone_capabilities": official_capabilities,
                "source_meta": _uploaded_source_meta(record, "user_mcd"),
            }

        overview_service = request.app.state.mcd_overview_service
        years = overview_service.get_available_years()
        primary_year = DEFAULT_MARS_YEAR if DEFAULT_MARS_YEAR in years else years[0]
        ls_min, ls_max = overview_service.get_ls_range(primary_year)
        return {
            "available_years": years,
            "timeline": {
                "min": float(ls_min),
                "max": float(ls_max),
                "step": 5.0,
            },
            "ozone_capabilities": overview_service.get_ozone_capabilities(),
            "source_meta": _overview_source_meta(data_source, primary_year),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取 MCD 总览信息失败: {exc}")


@router.get("/overview/globe", response_model=GlobeDataResponse)
async def get_overview_globe_data(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    ls: float = Query(10.0, ge=0, le=360, description="太阳黄经 Ls"),
    variable: str = Query("o3col", description="显示变量", enum=["o3col"] + OVERVIEW_MCD_VARIABLES),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        variable = _validate_overview_variable(variable, include_ozone=True)
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_globe_data, resolved_year, ls, variable=variable)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MCD 总览球体数据处理错误: {exc}")

@router.get("/overview/point-probe", response_model=PointProbeResponse)
async def get_overview_point_probe(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="Mars year"),
    lat: float = Query(..., ge=-90, le=90, description="Clicked latitude"),
    lng: float = Query(..., ge=-180, le=180, description="Clicked longitude"),
    ls: float = Query(0.0, ge=0, le=360, description="Solar longitude Ls"),
    variable: str = Query("o3col", description="Display variable", enum=["o3col"] + OVERVIEW_MCD_VARIABLES),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        variable = _validate_overview_variable(variable, include_ozone=True)
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_point_probe, resolved_year, lat, lng, ls, variable=variable)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MCD point probe processing error: {exc}")


@router.get("/overview/seasonal-heatmap", response_model=HeatmapResponse)
async def get_overview_seasonal_heatmap(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_seasonal_heatmap, resolved_year, variable="o3col")
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/overview/env-heatmap", response_model=HeatmapResponse)
async def get_overview_env_variable_heatmap(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    variable: str = Query(..., description="变量名", enum=OVERVIEW_MCD_VARIABLES),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        variable = _validate_overview_variable(variable)
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_env_variable_heatmap, resolved_year, variable)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/overview/correlation", response_model=CorrelationResponse)
async def get_overview_correlation_matrix(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_correlation_matrix, resolved_year)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/overview/diurnal")
async def get_overview_diurnal(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    ls: float = Query(90.0, description="太阳黄经 Ls"),
    lat_band: str = Query("Equatorial (30S-30N)", description="纬度带名称"),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id, include_mcd_hourly=True
        )
        result = service.get_diurnal_data(resolved_year, ls, lat_band)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/overview/coupling")
async def get_overview_coupling(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    var1: str = Query("o3col", description="变量1"),
    var2: str = Query("Temperature", description="变量2", enum=OVERVIEW_MCD_VARIABLES),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        var2 = _validate_overview_variable(var2)
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_coupling_data, resolved_year, var1, var2)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/overview/zonal-anomaly")
async def get_overview_zonal_anomaly(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    variable: str = Query("o3col", description="变量名", enum=["o3col"] + OVERVIEW_MCD_VARIABLES),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        variable = _validate_overview_variable(variable, include_ozone=True)
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_zonal_anomalies, resolved_year, variable)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/overview/solar-photochemical")
async def get_overview_solar_photochemical(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    lat_band: str = Query("Equatorial (30S-30N)", description="纬度带名称"),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_solar_photochemical, resolved_year, lat_band)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/overview/polar-dynamics")
async def get_overview_polar_dynamics(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_polar_dynamics, resolved_year)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/overview/research-suite")
async def get_overview_research_suite(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_research_suite, resolved_year)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/overview/phase-space")
async def get_overview_phase_space(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    driver: str = Query("Temperature", description="驱动变量", enum=OVERVIEW_MCD_VARIABLES),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        driver = _validate_overview_variable(driver)
        service, source_meta, resolved_year = await _resolve_overview_context(
            request, my, data_source, current_user, mcd_upload_id
        )
        result = await asyncio.to_thread(service.get_phase_space, resolved_year, driver)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/overview/ozone-sources", response_model=OverviewOzoneSourcesResponse)
async def get_overview_ozone_sources(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    ls: float = Query(10.0, ge=0, le=360, description="太阳黄经 Ls"),
    data_source: str = Query("default", description="default"),
    mcd_upload_id: int | None = Query(None, ge=1),
    openmars_upload_id: int | None = Query(None, ge=1),
    nomad_upload_id: int | None = Query(None, ge=1),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        _reject_legacy_personal(data_source)
        overview_service = request.app.state.mcd_overview_service
        payload = overview_service.get_ozone_overlay_payload(my, ls)

        if mcd_upload_id:
            mcd_record = await _get_accessible_upload_record(mcd_upload_id, current_user, {"mcd"})
            mcd_data = await request.app.state.user_data_service.get_loaded_dataset(mcd_record.id)
            mcd_layer = build_uploaded_ozone_layer(mcd_data, ls, "mcd")
            payload["mcd"] = mcd_layer
            payload["anchor_ls"] = mcd_layer["ls"]
            payload["mars_year"] = int(mcd_record.mars_year or my)
        else:
            mcd_layer = payload["mcd"]

        if openmars_upload_id:
            openmars_record = await _get_accessible_upload_record(openmars_upload_id, current_user, {"openmars"})
            openmars_data = await request.app.state.user_data_service.get_loaded_dataset(openmars_record.id)
            payload["openmars"] = build_uploaded_ozone_layer(openmars_data, payload["anchor_ls"], "openmars")

        if nomad_upload_id:
            nomad_record = await _get_accessible_upload_record(nomad_upload_id, current_user, {"nomad"})
            nomad_data = await request.app.state.user_data_service.get_loaded_dataset(nomad_record.id)
            payload["nomad"] = build_uploaded_ozone_layer(nomad_data, payload["anchor_ls"], "nomad")

        payload["available_sources"] = [
            source for source in ("mcd", "openmars", "nomad") if payload.get(source) is not None
        ]
        payload["diff_candidates"] = []
        if payload.get("openmars") is not None:
            payload["diff_candidates"].append("MCD-OpenMARS")
        if payload.get("nomad") is not None:
            payload["diff_candidates"].append("MCD-NOMAD")
        payload["validation"] = {
            "nomad": build_uploaded_nomad_validation(mcd_layer, payload.get("nomad"))
        }
        payload["capabilities"] = {
            "openmars": payload.get("openmars") is not None,
            "nomad": payload.get("nomad") is not None,
            "diff_pairs": payload["diff_candidates"],
            "coverage": payload.get("capabilities", {}).get("coverage", {}),
        }
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MCD 总览臭氧多源数据处理错误: {exc}")


@router.get("/globe", response_model=GlobeDataResponse)
async def get_globe_data(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    ls: float = Query(10.0, ge=0, le=360, description="太阳黄经 Ls"),
    variable: str = Query("o3col", description="显示变量", enum=["o3col"] + MCD_VARIABLES),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_globe_data, resolved_year, ls, variable=variable)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"数据处理错误: {exc}")


@router.get("/seasonal-heatmap", response_model=HeatmapResponse)
async def get_seasonal_heatmap(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_seasonal_heatmap, resolved_year, variable="o3col")
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/seasonal-bands", response_model=SeasonalBandsResponse)
async def get_seasonal_bands(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_seasonal_bands, resolved_year)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/env-heatmap", response_model=HeatmapResponse)
async def get_env_variable_heatmap(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    variable: str = Query(..., description="变量名", enum=MCD_VARIABLES),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_env_variable_heatmap, resolved_year, variable)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/correlation", response_model=CorrelationResponse)
async def get_correlation_matrix(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_correlation_matrix, resolved_year)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/info")
async def get_data_info(
    request: Request,
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        resolver = request.app.state.personal_data_source_service
        requested = _normalize_source(data_source)
        return await resolver.get_data_info(requested, current_user.id if current_user else None)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取数据源信息失败: {exc}")


@router.get("/personal-build-status")
async def get_personal_build_status():
    raise HTTPException(
        status_code=410,
        detail=(
            "personal-build-status has been retired; user uploads now feed Data Overview "
            "raw-source visualization, while training and prediction use server-managed datasets"
        ),
    )


@router.get("/coupling")
async def get_coupling(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    var1: str = Query("o3col", description="变量1"),
    var2: str = Query("Dust_Optical_Depth", description="变量2"),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_coupling_data, resolved_year, var1, var2)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/zonal-anomaly")
async def get_zonal_anomaly(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    variable: str = Query("o3col", description="变量名"),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_zonal_anomalies, resolved_year, variable)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/solar-photochemical")
async def get_solar_photochemical(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    lat_band: str = Query("Equatorial (30S-30N)", description="纬度带名称"),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_solar_photochemical, resolved_year, lat_band)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/polar-dynamics")
async def get_polar_dynamics(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_polar_dynamics, resolved_year)
        return _with_source_meta(result, source_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/research-suite")
async def get_research_suite(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_research_suite, resolved_year)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/phase-space")
async def get_phase_space(
    request: Request,
    my: int = Query(DEFAULT_MARS_YEAR, description="火星年"),
    driver: str = Query("Dust_Optical_Depth", description="驱动变量", enum=MCD_VARIABLES),
    data_source: str = Query("default", description="default | personal"),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        service, source_meta, resolved_year = await _resolve_analysis_context(
            request, my, data_source, current_user
        )
        result = await asyncio.to_thread(service.get_phase_space, resolved_year, driver)
        return _with_source_meta(result, source_meta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
