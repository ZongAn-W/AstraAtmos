"""Authenticated scientific export, using results only (never inference)."""
import asyncio
import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from auth.dependencies import get_current_user
from database.models import User
from schemas.research_export import ResearchExportRequest
from services.dataset_identity import DatasetRequestError
from services.research_export_sources import (
    EXPORT_SOURCES, ArtifactGuardError, ExportUnavailable, task_guard,
)
from services.research_figure import prepare_figure, render_figure, build_bundle, resolve_fonts

router = APIRouter(prefix="/predict/research-export", tags=["Scientific export"])
logger = logging.getLogger("astraatmos.research_export")
EXPORT_SLOTS = threading.BoundedSemaphore(2)


async def validate_sources(request, payload, user):
    sources = [EXPORT_SOURCES.get(ref.model_dump(), user.id) for ref in payload.sources]
    for source in sources:
        if source.ref["planet"] == "mars":
            from routers.predict import _get_training_inference_service
            from services.prediction_analysis_cache import build_artifact_fingerprint
            service = _get_training_inference_service(request)
            task, _, dirs, temp = await service._prepare_task_prediction_context(
                task_id=source.ref["task_id"], current_user=user,
                data_service=getattr(request.app.state, "data_service", None),
                personal_source_service=getattr(request.app.state, "personal_data_source_service", None))
            try:
                guard = build_artifact_fingerprint(task, dirs)
            finally:
                service._cleanup_temp_data_root(temp)
        else:
            from routers.earth_predict import _load_task, _registry
            from services.earth_prediction_service import build_prediction_context
            task = await _load_task(request, source.ref["task_id"], user)
            context = await asyncio.to_thread(build_prediction_context, task, _registry(request))
            if context.dataset_fingerprint != source.metadata["dataset_fingerprint"]:
                raise ExportUnavailable("Dataset identity changed. Refresh the result.")
            guard = task_guard(task)
        if guard != source.guard:
            raise ExportUnavailable("Model or dataset changed after this result. Refresh the result.")
    return sources


@router.get("/capabilities")
async def capabilities(current_user: User = Depends(get_current_user)):
    return {"fonts": {name: resolve_fonts(name) for name in ("auto", "sans", "serif")},
            "chinese_available": len(resolve_fonts()) > 1, "ttl_seconds": EXPORT_SOURCES.ttl,
            "max_grid_cells": 1_000_000, "max_scatter_points": 250_000}


async def _export(request, payload, user, preview=False):
    if not EXPORT_SLOTS.acquire(blocking=False):
        raise HTTPException(429, detail="Export server busy; retry shortly")
    owns_slot = True
    try:
        sources = await validate_sources(request, payload, user)
        def draw():
            data = prepare_figure(sources, payload)
            content = render_figure(data, preview=preview)
            if payload.bundle and not preview:
                content = build_bundle(data, content)
            return content
        render_task = asyncio.create_task(asyncio.to_thread(draw))
        try:
            content = await asyncio.shield(render_task)
        except asyncio.CancelledError:
            # A disconnected preview cannot cancel a native render thread.
            # Retain its slot until it finishes, preventing unbounded work.
            def finished(task):
                if not task.cancelled():
                    task.exception()
                EXPORT_SLOTS.release()
            render_task.add_done_callback(finished)
            owns_slot = False
            raise
        # Recheck expiry, ownership and file identities before publishing bytes.
        await validate_sources(request, payload, user)
        fmt = "png" if preview else "zip" if payload.bundle else payload.options.format
        media = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf", "zip": "application/zip"}[fmt]
        headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        if not preview:
            headers["Content-Disposition"] = f'attachment; filename="astraatmos-{payload.kind}.{fmt}"'
        return Response(content, media_type=media, headers=headers)
    except ArtifactGuardError as exc:
        raise HTTPException(
            409,
            detail={"code": "invalid_earth_training_artifact", "message": "The Earth model artifact is unavailable"},
        ) from exc
    except PermissionError as exc:
        raise HTTPException(403, detail=str(exc)) from exc
    except (ExportUnavailable, FileNotFoundError) as exc:
        raise HTTPException(409, detail=str(exc) if isinstance(exc, ExportUnavailable) else "Model/result no longer available") from exc
    except DatasetRequestError as exc:
        raise HTTPException(exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scientific export failed")
        raise HTTPException(500, detail="Scientific export failed; check the server log") from exc
    finally:
        if owns_slot:
            EXPORT_SLOTS.release()


@router.post("/preview")
async def preview(request: Request, payload: ResearchExportRequest, current_user: User = Depends(get_current_user)):
    return await _export(request, payload, current_user, preview=True)


@router.post("/download")
async def download(request: Request, payload: ResearchExportRequest, current_user: User = Depends(get_current_user)):
    return await _export(request, payload, current_user)
