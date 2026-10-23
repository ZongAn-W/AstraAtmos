from __future__ import annotations

import asyncio
import gzip
import hashlib
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import MCD_DIR, MCD_RAW_3H_DIR, OPENMARS_DIR
from database.engine import async_session_maker
from database.models import PredictionAnalysisCache
from services.mars_checkpoint import MCD_WINDOW_POLICY, OPENMARS_WINDOW_POLICY
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError


CACHE_SCHEMA_VERSION = 1
MARS_OUTPUT_COORDINATE_POLICY = "loader_coordinates_v1"
SUPPORTED_ANALYSIS_TYPES = {
    "prediction",
    "metrics",
    "error_distribution",
    "pfi",
}
logger = logging.getLogger("astraatmos.prediction_analysis_cache")


class CachePayloadError(ValueError):
    pass


@dataclass(frozen=True)
class CacheClaim:
    action: str
    row_id: int | None = None
    token: str | None = None
    payload: dict[str, Any] | None = None


def _lease_is_active(value, now):
    if value is None:
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value > now


async def _resolve_compute(compute):
    from services.inference_compute import run_inference_compute
    return await run_inference_compute(compute)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _variables(values: list[str] | None) -> list[str]:
    return sorted(
        {str(value).strip() for value in values or [] if str(value).strip()}
    )


def normalize_request(
    analysis_type: str, params: dict[str, Any]
) -> dict[str, Any]:
    if analysis_type not in SUPPORTED_ANALYSIS_TYPES:
        raise ValueError(
            f"Unsupported prediction analysis type: {analysis_type}"
        )

    horizon = int(params["horizon"])
    if analysis_type == "prediction":
        ls_start = float(params["ls_start"]) % 360.0
        if ls_start == 0.0:
            ls_start = 0.0
        return {
            "horizon": horizon,
            "ls_start": ls_start,
        }
    if analysis_type in {"metrics", "error_distribution"}:
        return {"horizon": horizon}
    return {
        "horizon": horizon,
        "selected_variables": _variables(params.get("selected_variables")),
    }


def build_request_hash(
    analysis_type: str, params: dict[str, Any]
) -> str:
    normalized = normalize_request(analysis_type, params)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _dataset_manifest(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*.nc"), key=lambda item: item.name.lower())
    return [_file_signature(path) for path in paths]


def build_artifact_fingerprint(
    task,
    data_dirs=None,
    *,
    schema_version=CACHE_SCHEMA_VERSION,
) -> str:
    data_dirs = data_dirs or {}
    model_path = Path(task.output_model_path).expanduser().resolve()
    openmars_dir = Path(
        data_dirs.get("ASTRAATMOS_OPENMARS_DIR") or OPENMARS_DIR
    ).expanduser().resolve()
    mcd_dir = Path(
        data_dirs.get("ASTRAATMOS_MCD_DIR") or MCD_DIR
    ).expanduser().resolve()
    raw_dir = Path(
        data_dirs.get("MCD_RAW_3H_DIR")
        or data_dirs.get("ASTRAATMOS_MCD_RAW_3H_DIR")
        or MCD_RAW_3H_DIR
    ).expanduser().resolve()

    try:
        hyperparameters = json.loads(task.hyperparameters or "{}")
    except (TypeError, json.JSONDecodeError):
        hyperparameters = {}

    identity = {
        "schema_version": int(schema_version),
        "task_id": int(task.id),
        "model_source": str(
            getattr(task, "model_source", "official") or "official"
        ),
        "uploaded_model_id": getattr(task, "uploaded_model_id", None),
        "uploaded_model_version": getattr(
            task, "uploaded_model_version", None
        ),
        "hyperparameters": hyperparameters,
        "dataset_id": getattr(task, "dataset_id", None) or hyperparameters.get("training_dataset"),
        "dataset_snapshot": getattr(task, "dataset_snapshot", None),
        "model": _file_signature(model_path),
        "openmars": _dataset_manifest(openmars_dir),
        "mcd": _dataset_manifest(mcd_dir),
        "mcd_raw_3h": _dataset_manifest(raw_dir),
    }
    dataset_id = identity["dataset_id"] or "openmars_mcd"
    if dataset_id in ("openmars_mcd", "mcd_overview"):
        identity["window_policy"] = OPENMARS_WINDOW_POLICY if dataset_id == "openmars_mcd" else MCD_WINDOW_POLICY
    if (identity["dataset_id"] or "openmars_mcd") in ("openmars_mcd", "mcd_overview"):
        identity["output_coordinate_policy"] = MARS_OUTPUT_COORDINATE_POLICY
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _validate_result(
    analysis_type: str, result: Any
) -> dict[str, Any]:
    required = {
        "prediction": {
            "ground_truth",
            "prediction",
            "residual",
            "horizon",
            "model_info",
        },
        "metrics": {"overall", "per_step"},
        "error_distribution": {
            "scatter",
            "hist_trues",
            "hist_preds",
            "hist_errors",
            "mae",
            "rmse",
        },
        "pfi": {"items", "baseline_metric", "baseline_value"},
    }[analysis_type]
    if not isinstance(result, dict) or not required.issubset(result):
        raise CachePayloadError(
            f"Invalid {analysis_type} cache payload"
        )
    return result


def encode_payload(
    analysis_type: str, result: dict[str, Any]
) -> bytes:
    envelope = {
        "analysis_type": analysis_type,
        "payload": _validate_result(analysis_type, result),
        "schema_version": CACHE_SCHEMA_VERSION,
    }
    return gzip.compress(_canonical_json(envelope), mtime=0)


def decode_payload(
    analysis_type: str, payload: bytes
) -> dict[str, Any]:
    try:
        envelope = json.loads(gzip.decompress(payload))
    except Exception as exc:
        raise CachePayloadError(
            "Invalid compressed cache payload"
        ) from exc

    if envelope.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CachePayloadError("Cache schema version mismatch")
    if envelope.get("analysis_type") != analysis_type:
        raise CachePayloadError("Cache analysis type mismatch")
    return _validate_result(analysis_type, envelope.get("payload"))


class PredictionAnalysisCacheService:
    def __init__(
        self,
        sessionmaker=async_session_maker,
        *,
        lease_seconds=1800.0,
        poll_seconds=0.05,
        wait_timeout_seconds=10.0,
    ):
        self.sessionmaker = sessionmaker
        self.lease_seconds = float(lease_seconds)
        self.poll_seconds = float(poll_seconds)
        self.wait_timeout_seconds = (
            None
            if wait_timeout_seconds is None
            else float(wait_timeout_seconds)
        )

    async def get_or_compute(
        self,
        *,
        user_id: int,
        task,
        analysis_type: str,
        request_params: dict[str, Any],
        data_dirs: dict[str, str] | None,
        compute,
    ) -> dict[str, Any]:
        if not user_id:
            raise PermissionError(
                "Authentication is required for trained-model analysis"
            )

        scope = {
            "user_id": int(user_id),
            "training_task_id": int(task.id),
            "analysis_type": analysis_type,
            "request_hash": build_request_hash(
                analysis_type, request_params
            ),
        }
        fingerprint = build_artifact_fingerprint(task, data_dirs)

        try:
            wait_started_at = None
            waiting_for = None
            reclaim = None
            while True:
                claim = await self._lookup_or_claim(
                    scope, fingerprint, reclaim=reclaim
                )
                reclaim = None
                if claim.action == "hit":
                    return claim.payload
                if claim.action == "retry":
                    wait_started_at = None
                    waiting_for = None
                    await asyncio.sleep(self.poll_seconds)
                    continue
                if claim.action == "wait":
                    now = time.monotonic()
                    current_wait = (claim.row_id, claim.token)
                    if waiting_for != current_wait:
                        waiting_for = current_wait
                        wait_started_at = now

                    elapsed = (
                        now - wait_started_at
                        if wait_started_at is not None
                        else 0.0
                    )
                    if (
                        self.wait_timeout_seconds is not None
                        and elapsed >= self.wait_timeout_seconds
                    ):
                        logger.warning(
                            "prediction analysis cache wait timed out; "
                            "reclaiming row=%s type=%s hash=%s",
                            claim.row_id,
                            scope["analysis_type"],
                            scope["request_hash"][:12],
                        )
                        reclaim = {
                            "row_id": claim.row_id,
                            "token": claim.token,
                        }
                        wait_started_at = None
                        waiting_for = None
                        continue

                    sleep_for = self.poll_seconds
                    if self.wait_timeout_seconds is not None:
                        remaining = max(
                            0.0, self.wait_timeout_seconds - elapsed
                        )
                        sleep_for = min(sleep_for, remaining)
                    await asyncio.sleep(sleep_for)
                    continue
                break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "prediction analysis cache lookup failed: %s",
                scope["request_hash"][:12],
            )
            return await _resolve_compute(compute)

        try:
            result = await _resolve_compute(compute)
        except asyncio.CancelledError:
            await self._mark_failed(
                claim.row_id, claim.token, "cancelled"
            )
            raise
        except Exception as exc:
            try:
                await self._mark_failed(
                    claim.row_id, claim.token, str(exc)[:500]
                )
            except Exception:
                logger.exception(
                    "prediction analysis cache failure state could not be stored"
                )
            raise

        try:
            await self._publish(
                claim.row_id,
                claim.token,
                encode_payload(analysis_type, result),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "prediction analysis cache publish failed: %s",
                scope["request_hash"][:12],
            )
        return result

    async def _lookup_or_claim(
        self, scope, fingerprint, *, reclaim=None
    ) -> CacheClaim:
        now = datetime.now(timezone.utc)
        async with self.sessionmaker() as session:
            row = (
                await session.execute(
                    select(PredictionAnalysisCache).where(
                        PredictionAnalysisCache.user_id
                        == scope["user_id"],
                        PredictionAnalysisCache.training_task_id
                        == scope["training_task_id"],
                        PredictionAnalysisCache.analysis_type
                        == scope["analysis_type"],
                        PredictionAnalysisCache.request_hash
                        == scope["request_hash"],
                    )
                )
            ).scalar_one_or_none()

            if row is None:
                token = str(uuid.uuid4())
                row = PredictionAnalysisCache(
                    **scope,
                    artifact_fingerprint=fingerprint,
                    status="computing",
                    lease_token=token,
                    lease_expires_at=now
                    + timedelta(seconds=self.lease_seconds),
                    updated_at=now,
                )
                session.add(row)
                try:
                    await session.commit()
                    await session.refresh(row)
                except IntegrityError:
                    await session.rollback()
                    return CacheClaim("retry")
                return CacheClaim(
                    "compute", row_id=row.id, token=token
                )

            if (
                row.status == "ready"
                and row.artifact_fingerprint == fingerprint
                and row.payload
            ):
                try:
                    return CacheClaim(
                        "hit",
                        row_id=row.id,
                        payload=decode_payload(
                            scope["analysis_type"], row.payload
                        ),
                    )
                except CachePayloadError:
                    pass

            if (
                row.status == "computing"
                and row.artifact_fingerprint == fingerprint
                and _lease_is_active(row.lease_expires_at, now)
            ):
                should_reclaim = (
                    reclaim is not None
                    and reclaim.get("row_id") == row.id
                    and reclaim.get("token") == row.lease_token
                )
                if not should_reclaim:
                    return CacheClaim(
                        "wait", row_id=row.id, token=row.lease_token
                    )

            token = str(uuid.uuid4())
            conditions = [PredictionAnalysisCache.id == row.id]
            if row.status == "computing":
                conditions.append(
                    PredictionAnalysisCache.lease_token
                    == row.lease_token
                )
            else:
                conditions.extend(
                    [
                        PredictionAnalysisCache.status == row.status,
                        PredictionAnalysisCache.artifact_fingerprint
                        == row.artifact_fingerprint,
                    ]
                )
            result = await session.execute(
                update(PredictionAnalysisCache)
                .where(*conditions)
                .values(
                    artifact_fingerprint=fingerprint,
                    status="computing",
                    payload=None,
                    lease_token=token,
                    lease_expires_at=now
                    + timedelta(seconds=self.lease_seconds),
                    last_error=None,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                await session.rollback()
                return CacheClaim("retry")
            await session.commit()
            return CacheClaim(
                "compute", row_id=row.id, token=token
            )

    async def _publish(self, row_id, token, payload):
        async with self.sessionmaker() as session:
            result = await session.execute(
                update(PredictionAnalysisCache)
                .where(
                    PredictionAnalysisCache.id == row_id,
                    PredictionAnalysisCache.lease_token == token,
                    PredictionAnalysisCache.status == "computing",
                )
                .values(
                    status="ready",
                    payload=payload,
                    lease_token=None,
                    lease_expires_at=None,
                    last_error=None,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            if result.rowcount != 1:
                await session.rollback()
                raise RuntimeError(
                    "Prediction analysis cache lease was lost"
                )
            await session.commit()

    async def _mark_failed(self, row_id, token, error):
        async with self.sessionmaker() as session:
            await session.execute(
                update(PredictionAnalysisCache)
                .where(
                    PredictionAnalysisCache.id == row_id,
                    PredictionAnalysisCache.status == "computing",
                    PredictionAnalysisCache.lease_token == token,
                )
                .values(
                    status="failed",
                    payload=None,
                    lease_token=None,
                    lease_expires_at=None,
                    last_error=str(error)[:500],
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
