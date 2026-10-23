"""Earth scientific analysis endpoints (context, suite, diagnostics, insight).

Handlers are synchronous ``def`` functions so FastAPI runs the NumPy work on its
thread pool instead of blocking the event loop. Every error is mapped through
:func:`_error_response`, which never echoes a server path or a traceback.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from config import DEFAULT_EARTH_DATASET_ID

from schemas.earth_research import (
    EarthContextResponse,
    EarthInsightRequest,
    EarthInsightResponse,
    EarthPolarDynamicsResponse,
    EarthResearchSuiteResponse,
    EarthSpatialDiagnosticsResponse,
)
from services.dataset_identity import DatasetRequestError
from services.earth_dataset_metadata import EarthPackageError
from services.earth_overview_service import EarthOverviewError
from services.earth_research_service import CADENCE, PLANET, SOURCE_LABEL

logger = logging.getLogger("astraatmos.datasets.earth")

router = APIRouter(prefix="/analysis/earth/overview", tags=["Earth Analysis"])

DEFAULT_DATASET_ID = DEFAULT_EARTH_DATASET_ID
FINGERPRINT_PATTERN = r"^[0-9a-f]{64}$"
MAX_DATASET_ID_LENGTH = 128

# The insight endpoint speaks about Earth only.
SUPPORTED_PLANET = "earth"

# Depth below the top level request object at which the summary stops being a
# digest. A raw [36][72] grid is depth 4 and is therefore still allowed through
# the depth gate and caught by the element-count gate below, which is the
# documented contract: digests are small, fields are not.
MAX_SUMMARY_DEPTH = 6
MAX_SUMMARY_ELEMENTS = 4000

# A stuck upstream must not hold the request open indefinitely.
COPILOT_TIMEOUT_SECONDS = 20.0


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
        # ``exc.reason`` is a stable code; ``exc.detail`` may quote a path and
        # is deliberately not returned.
        return HTTPException(
            status_code=503,
            detail={
                "code": "dataset_unavailable",
                "message": "The registered Earth dataset is not available",
                "availability_reason": exc.reason,
            },
        )
    raise exc


def _summary_limits(node: Any, depth: int = 0) -> tuple[int, int]:
    """Return the maximum depth and the largest list length inside ``node``."""
    deepest = depth
    widest = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                widest = max(widest, len(key))
            child_depth, child_width = _summary_limits(value, depth + 1)
            deepest = max(deepest, child_depth)
            widest = max(widest, child_width)
    elif isinstance(node, (list, tuple)):
        widest = max(widest, len(node))
        for item in node:
            child_depth, child_width = _summary_limits(item, depth + 1)
            deepest = max(deepest, child_depth)
            widest = max(widest, child_width)
    return deepest, widest


def _guard_summary(body: EarthInsightRequest) -> None:
    """Reject anything that is not a small statistics digest.

    Documented rule: the AI endpoint accepts *statistics*, never a raw field.
    A list longer than :data:`MAX_SUMMARY_ELEMENTS` or a nesting depth beyond
    :data:`MAX_SUMMARY_DEPTH` is refused with ``insight_payload_too_large``; the
    server recomputes every number from the verified release anyway.
    """
    if body.summary is None:
        return
    payload = body.summary.model_dump()
    deepest, widest = _summary_limits(payload)
    if widest > MAX_SUMMARY_ELEMENTS or deepest > MAX_SUMMARY_DEPTH:
        raise EarthOverviewError(
            "insight_payload_too_large",
            "The insight summary must be a small statistics digest, not a field",
            422,
        )


def _limitations(digest: dict) -> list[str]:
    scope = digest.get("scope_label") or digest.get("scope") or "global"
    if digest.get("dataset_id") == "earth_merra2_3hourly_v1":
        return [
            "Annual analysis uses UTC daily means aggregated from eight three-hour means; within-day variation is excluded",
            f"Scope is {scope} on the native global 0.75 degree grid",
            "Reported correlations are associations, not evidence of causation",
        ]
    return [
        f"Values are UTC daily means from {SOURCE_LABEL}; the package carries no diurnal cycle",
        f"Scope is {scope} on the published global 5 degree grid",
        "Reported correlations are associations, not evidence of causation",
    ]


def _fallback_answer(digest: dict) -> str:
    """Deterministic answer built from the digest, with no external call."""
    variable = digest["variable"]
    units = digest["units"]
    start = digest["date_range"]["start"]
    end = digest["date_range"]["end"]
    scope_label = digest.get("scope_label") or digest.get("scope") or "global"
    series = next(
        (card["values"] for card in digest["cards"] if card["card"] == "series"), {}
    )
    mean = series.get("mean")
    low = series.get("min_value")
    high = series.get("max_value")
    low_date = series.get("min_date")
    high_date = series.get("max_date")
    parts = [
        f"Earth: {SOURCE_LABEL} daily mean {variable} in {units}, "
        f"{start} to {end} ({digest['day_count']} days), scope {scope_label}."
    ]
    if mean is not None:
        parts.append(f"Annual mean {mean:.3f} {units}.")
    if low is not None and high is not None:
        parts.append(
            f"Range {low:.3f} {units} on {low_date} to {high:.3f} {units} on {high_date}."
        )
    correlations = [
        card["values"] for card in digest["cards"] if card["card"] == "correlation"
    ]
    strongest = None
    for values in correlations:
        if values.get("r") is not None and (
            strongest is None or abs(values["r"]) > abs(strongest["r"])
        ):
            strongest = values
    if strongest is not None:
        parts.append(
            f"Strongest same-day association within {scope_label}: r="
            f"{strongest['r']:.3f} against {strongest['against']} (n={strongest['n']})."
        )
    parts.append(
        "These are UTC daily means with no diurnal cycle, and the correlations are "
        "associations rather than causation."
    )
    return " ".join(parts)


def _call_copilot(request: Request, question: str, context: dict) -> Optional[str]:
    """Ask the shared Copilot service, or return ``None`` to use the fallback.

    The handler runs in FastAPI's thread pool, so the coroutine is scheduled onto
    the loop that is serving this request, which ``main`` publishes as
    ``app.state.event_loop``. A missing loop, a missing service, a missing API
    key and any transport error all degrade to the deterministic digest answer so
    the endpoint always answers.
    """
    copilot = getattr(request.app.state, "copilot_service", None)
    if copilot is None:
        return None
    if not getattr(copilot, "api_key", None):
        return None
    loop = getattr(request.app.state, "event_loop", None)
    if loop is None or loop.is_closed() or not loop.is_running():
        return None
    try:
        future = asyncio.run_coroutine_threadsafe(
            copilot.chat(question, context), loop
        )
        answer = future.result(timeout=COPILOT_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("Earth insight AI call failed; using the builtin digest answer")
        return None
    text = str(answer).strip() if answer else ""
    return text or None


@router.get("/context", response_model=EarthContextResponse)
def get_context(
    request: Request,
    dataset_id: str = Query(default=DEFAULT_DATASET_ID, max_length=MAX_DATASET_ID_LENGTH),
    expected_fingerprint: Optional[str] = Query(default=None, pattern=FINGERPRINT_PATTERN),
):
    try:
        return request.app.state.earth_research_service.get_context(
            dataset_id=dataset_id,
            expected_fingerprint=expected_fingerprint,
        )
    except (DatasetRequestError, EarthOverviewError, EarthPackageError) as exc:
        raise _error_response(exc) from exc


@router.get("/research-suite", response_model=EarthResearchSuiteResponse)
def get_research_suite(
    request: Request,
    dataset_id: str = Query(..., max_length=MAX_DATASET_ID_LENGTH),
    year: int = Query(...),
    expected_fingerprint: Optional[str] = Query(default=None, pattern=FINGERPRINT_PATTERN),
):
    try:
        return request.app.state.earth_research_service.get_research_suite(
            dataset_id=dataset_id,
            year=year,
            expected_fingerprint=expected_fingerprint,
        )
    except (DatasetRequestError, EarthOverviewError, EarthPackageError) as exc:
        raise _error_response(exc) from exc


@router.get("/spatial-diagnostics", response_model=EarthSpatialDiagnosticsResponse)
def get_spatial_diagnostics(
    request: Request,
    dataset_id: str = Query(..., max_length=MAX_DATASET_ID_LENGTH),
    year: int = Query(...),
    variable: str = Query(..., max_length=16),
    expected_fingerprint: Optional[str] = Query(default=None, pattern=FINGERPRINT_PATTERN),
):
    try:
        return request.app.state.earth_research_service.get_spatial_diagnostics(
            dataset_id=dataset_id,
            year=year,
            variable=variable,
            expected_fingerprint=expected_fingerprint,
        )
    except (DatasetRequestError, EarthOverviewError, EarthPackageError) as exc:
        raise _error_response(exc) from exc


@router.get("/polar-dynamics", response_model=EarthPolarDynamicsResponse)
def get_polar_dynamics(
    request: Request,
    dataset_id: str = Query(..., max_length=MAX_DATASET_ID_LENGTH),
    year: int = Query(...),
    expected_fingerprint: Optional[str] = Query(default=None, pattern=FINGERPRINT_PATTERN),
):
    try:
        return request.app.state.earth_research_service.get_polar_dynamics(
            dataset_id=dataset_id,
            year=year,
            expected_fingerprint=expected_fingerprint,
        )
    except (DatasetRequestError, EarthOverviewError, EarthPackageError) as exc:
        raise _error_response(exc) from exc


@router.post("/insight", response_model=EarthInsightResponse)
def post_insight(request: Request, body: EarthInsightRequest):
    """Answer one question about an Earth aggregate series.

    The body carries *statistics only*. Every reported number is recomputed from
    the verified release by ``EarthResearchService.build_insight_summary``; the
    client ``summary`` cannot inject a value, and is merged as supplementary text
    notes. A raw 36x72 field is rejected with ``insight_payload_too_large``.
    """
    try:
        if body.planet != SUPPORTED_PLANET:
            # This endpoint must never describe Mars.
            raise EarthOverviewError(
                "planet_not_supported", "Only the Earth planet is supported", 422
            )
        _guard_summary(body)
        service = request.app.state.earth_research_service
        digest = service.build_insight_summary(
            dataset_id=body.dataset_id,
            payload=body.model_dump(),
            expected_fingerprint=body.expected_fingerprint,
        )
    except (DatasetRequestError, EarthOverviewError, EarthPackageError) as exc:
        raise _error_response(exc) from exc

    limitations = _limitations(digest)
    question = (body.question or "").strip() or (
        f"Describe the {digest['variable']} daily series for {digest['scope_label']} "
        f"in {digest['year']}."
    )
    context = {
        "planet": PLANET,
        "source": SOURCE_LABEL,
        "cadence": digest.get("cadence", CADENCE),
        "variable": digest["variable"],
        "units": digest["units"],
        "year": digest["year"],
        "scope": digest["scope"],
        "scope_label": digest["scope_label"],
        "date_range": digest["date_range"],
        "digest": digest,
        "limitations": limitations,
    }
    answer = _call_copilot(request, question, context)
    model = getattr(
        getattr(request.app.state, "copilot_service", None), "model", None
    )
    if answer is None:
        answer = _fallback_answer(digest)
        model = "builtin-digest"
    elif not model:
        model = "copilot"

    return {
        "planet": PLANET,
        "dataset_id": digest["dataset_id"],
        "dataset_version": digest["dataset_version"],
        "dataset_fingerprint": digest["dataset_fingerprint"],
        "source": SOURCE_LABEL,
        "year": digest["year"],
        "date_range": digest["date_range"],
        "variable": digest["variable"],
        "units": digest["units"],
        "scope": digest["scope"],
        "scope_label": digest["scope_label"],
        "time_kind": "iso-date",
        "calendar": digest["calendar"],
        "model": model,
        "answer": answer,
        "digest": digest,
        "limitations": limitations,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
