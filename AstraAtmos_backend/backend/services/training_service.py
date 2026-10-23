from __future__ import annotations

import os
import sys
import json
import hashlib
import asyncio
import logging
import traceback
import psutil
import subprocess
import re
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import time
import numpy as np
import netCDF4 as nc
from sqlalchemy import delete, select, update

from config import (
    EARTH_MERRA2_DIR,
    EARTH_MERRA2_V1_DIR,
    EARTH_MERRA2_3HOURLY_DIR,
    EARTH_MERRA2_3HOURLY_5DEG_DIR,
    EARTH_TRAINING_CACHE_DIR,
    MCD_VARIABLES,
    TRAINING_RESULTS_DIR,
    TRAINING_SCRIPTS_DIR,
    USER_UPLOADS_DIR,
)
from database.engine import async_session_maker
from database.models import ModelTrainingTask, PredictionAnalysisCache, TrainingTaskTag, User
from services.data_service import DataService
from services.dataset_identity import (
    DatasetRequestError,
    EARTH_THREE_HOURLY_DATASET_IDS,
    is_earth_training_task,
    resolve_dataset_id,
    require_training_dataset,
    require_active_dataset,
    training_task_dataset_id,
)
from services.dataset_registry import DatasetRegistry
from services.earth_training_artifact import (
    EarthArtifactError,
    load_earth_training_artifact,
    verify_earth_model_reload_isolated,
)
from services.earth_training_contract import (
    EARTH_3HOURLY_METRICS_SCHEMA_V2,
    EARTH_3HOURLY_METRICS_SCHEMA_V3,
    EARTH_METRICS_SCHEMA_KEY,
    EARTH_METRICS_AGGREGATION_KEY,
    EARTH_TRAINING_SCRIPT,
    build_earth_training_spec,
    canonical_channel_order,
    earth_3hourly_metrics_contract,
    earth_task_metrics_contract,
    earth_spec_metrics_contract,
    is_earth_dataset_id,
    require_earth_training_configuration,
)
from services.uploaded_model_source import (
    UploadedModelSourceError,
    build_reference,
    resolve_source_text_strict,
)
from services.user_model_validator import UserModelValidator
from training_backbones.uploaded_model_earth_gate import (
    EarthCompatibility,
    evaluate_package_earth_compatibility,
)
from services.personal_data_source_service import PersonalDataSourceService
from services.model_artifacts import is_valid_model_weight_file
from services.mars_checkpoint import validate_mars_checkpoint, mars_checkpoint_identity_snapshot
from services.training_failures import CUDA_OOM_ERROR_CODE, classify_training_log
from services.training_channels import (
    ARCHITECTURE_PARAM_KEYS,
    UNIFIED_TRAINING_SCRIPT,
    build_hyperparameter_args,
    normalize_training_hyperparameters,
)
from services.training_notifications import ensure_cuda_oom_notification
from services.training_paths import build_task_output_path
from services.training_tag_service import add_task_tag_links, begin_tag_write, validate_owned_tags
import config

logger = logging.getLogger("astraatmos.training")

INVALID_MODEL_ARTIFACT_ERROR = (
    "Training process exited successfully but no valid model weight file was produced"
)

MODELS_DIR = TRAINING_SCRIPTS_DIR
LOGS_DIR = Path(__file__).parent.parent / "logs" / "training"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MODELS_DIR = TRAINING_RESULTS_DIR
OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_training_data_source(data_source: str | None) -> str:
    source = (data_source or "default").strip().lower()
    if source == "personal":
        raise ValueError(
            "Personal raw uploads are only available in Data Overview; training uses server-managed datasets."
        )
    if source not in ("default",):
        raise ValueError("training data_source must be 'default'")
    return source


def _task_was_stopped(raw_metrics: Any) -> bool:
    """Return True when the stored metrics record a user stop.

    Stopping a task writes ``{"note": "Stopped by user"}``. The completion
    callback must not overwrite that terminal state with ``completed`` when the
    killed process happens to exit with code 0.
    """
    if not raw_metrics:
        return False
    try:
        parsed = json.loads(raw_metrics) if isinstance(raw_metrics, str) else raw_metrics
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("note") == "Stopped by user"


class TrainingService:
    def __init__(self, earth_model_validator: Any | None = None):
        # Retained for legacy daily compatibility; three-hour tasks reuse upload proofs.
        self._earth_model_validator = earth_model_validator
        self._scheduler_started = False
        self._scheduler_task: asyncio.Task | None = None
        self._queue_event: asyncio.Event | None = None

    async def start(self) -> None:
        if self._scheduler_started:
            return
        self._scheduler_started = True
        self._queue_event = asyncio.Event()
        async with async_session_maker() as session:
            result = await session.execute(
                select(ModelTrainingTask).where(ModelTrainingTask.status == "running")
            )
            recovery_time = datetime.now(timezone.utc)
            for task in result.scalars().all():
                log_path = Path(task.log_file_path or "")
                output_path = Path(task.output_model_path or "")
                saved_marker = False
                if log_path.is_file() and output_path.is_file() and is_valid_model_weight_file(output_path):
                    try:
                        saved_marker = "Model saved:" in log_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        saved_marker = False

                task.end_time = recovery_time
                task.pid = None
                earth_artifact = None
                if saved_marker and is_earth_training_task(task):
                    try:
                        require_active_dataset(training_task_dataset_id(task))
                        earth_artifact = await asyncio.to_thread(
                            load_earth_training_artifact, output_path,
                            {key: getattr(task, key, None) for key in (
                                "dataset_id", "dataset_version", "dataset_fingerprint",
                                "dataset_identity_status", "dataset_snapshot")},
                            json.loads(task.hyperparameters or "{}"), task.id,
                        )
                        await asyncio.to_thread(verify_earth_model_reload_isolated, output_path)
                    except Exception as exc:
                        task.status = "failed"
                        task.metrics = json.dumps({"error_code": "invalid_earth_training_artifact", "error": str(exc)})
                        continue
                if saved_marker:
                    task.status = "completed"
                    task.progress = 100.0
                    parsed_metrics = earth_artifact.metrics if earth_artifact is not None else self._extract_metrics_from_log(log_path)
                    task.metrics = json.dumps(
                        parsed_metrics or {"note": "completed after server restart"}
                    )
                else:
                    task.status = "failed"
                    task.metrics = json.dumps(
                        {"note": "Training interrupted by server restart"}
                    )
            await session.commit()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._wake_scheduler()

    async def stop(self) -> None:
        self._scheduler_started = False
        task = self._scheduler_task
        self._scheduler_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _wake_scheduler(self) -> None:
        if self._queue_event is not None:
            self._queue_event.set()

    async def _recalculate_queue_positions(self, session) -> None:
        result = await session.execute(
            select(ModelTrainingTask)
            .where(ModelTrainingTask.status == "queued")
            .order_by(ModelTrainingTask.queued_at.asc(), ModelTrainingTask.id.asc())
        )
        for position, task in enumerate(result.scalars().all(), start=1):
            task.queue_position = position

    async def _scheduler_loop(self) -> None:
        while self._scheduler_started:
            task_id = None
            async with async_session_maker() as session:
                result = await session.execute(
                    select(ModelTrainingTask)
                    .where(ModelTrainingTask.status == "queued")
                    .order_by(ModelTrainingTask.queued_at.asc(), ModelTrainingTask.id.asc())
                    .limit(1)
                )
                task = result.scalars().first()
                if task is not None:
                    claim = await session.execute(
                        update(ModelTrainingTask)
                        .where(ModelTrainingTask.id == task.id, ModelTrainingTask.status == "queued")
                        .values(status="running", start_time=datetime.now(timezone.utc), queue_position=None)
                    )
                    if claim.rowcount:
                        task_id = task.id
                    await session.commit()
                    await self._recalculate_queue_positions(session)
                    await session.commit()
            if task_id is None:
                self._queue_event.clear()
                try:
                    await asyncio.wait_for(self._queue_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._execute_training_task(task_id)

    async def _execute_training_task(
        self, task_id: int, *, dataset_registry: DatasetRegistry | None = None,
        training_weight_service: Any | None = None, user_model_service: Any | None = None,
    ) -> None:
        """Launch fresh and recovered tasks from the same persisted facts."""
        try:
            async with async_session_maker() as session:
                task = await session.get(ModelTrainingTask, task_id)
                if task is None or task.status != "running":
                    return
            spec = await self._prepare_training_execution(
                task, dataset_registry=dataset_registry,
                training_weight_service=training_weight_service, user_model_service=user_model_service,
            )
            await self._run_training_subprocess(task_id, **spec)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Training task %s could not be prepared for execution", task_id)
            async with async_session_maker() as session:
                row = await session.get(ModelTrainingTask, task_id)
                if row is not None and row.status == "running":
                    row.status = "failed"
                    row.end_time = datetime.now(timezone.utc)
                    row.metrics = json.dumps({
                        "error_code": getattr(exc, "code", "training_execution_preparation_failed"),
                        "error": str(exc),
                    })
                    await session.commit()

    async def _prepare_training_execution(
        self,
        task: Any,
        *,
        dataset_registry: DatasetRegistry | None = None,
        training_weight_service: Any | None = None,
        user_model_service: Any | None = None,
        is_admin: bool | None = None,
    ) -> dict[str, Any]:
        """Own runner selection, dependency checks and restart-safe launch details.

        This also validates admission after the task id has been allocated. No
        launch details are cached in memory: actual execution rechecks the saved
        task and its dependencies, including current transfer-source permission.
        """
        try:
            hyperparameters = json.loads(task.hyperparameters)
        except (TypeError, ValueError) as exc:
            raise ValueError("Training task hyperparameters are damaged") from exc
        if not isinstance(hyperparameters, dict):
            raise ValueError("Training task hyperparameters must be an object")
        dataset_id = resolve_dataset_id(getattr(task, "dataset_id", None), hyperparameters)
        require_active_dataset(dataset_id)
        _normalize_training_data_source(hyperparameters.get("_data_source"))
        earth = is_earth_dataset_id(dataset_id)
        model_source = getattr(task, "model_source", None) or hyperparameters.get("model_source", "official")
        if model_source not in ("official", "uploaded"):
            raise ValueError("Training task model source is invalid")
        if hyperparameters.get("model_source", model_source) != model_source:
            raise ValueError("Training task model source does not match its saved parameters")
        expected_script = EARTH_TRAINING_SCRIPT if earth else (
            "__user_model_runner__" if model_source == "uploaded" else UNIFIED_TRAINING_SCRIPT
        )
        if task.model_script != expected_script:
            raise ValueError("Training task runner does not match its dataset and model source")
        runner_path = (
            Path(__file__).parent.parent / "training_backbones" / "user_model_runner.py"
            if expected_script == "__user_model_runner__" else MODELS_DIR.joinpath(expected_script)
        )
        if not runner_path.exists():
            raise FileNotFoundError(f"Training runner {expected_script} is missing")
        for name in ("log_file_path", "output_model_path"):
            if not isinstance(getattr(task, name, None), str) or not getattr(task, name).strip():
                raise ValueError(f"Training task {name} is missing")

        earth_spec = None
        env_overrides = {}
        if earth:
            earth_spec = await asyncio.to_thread(
                self._restore_3hourly_training_spec, task, dataset_registry
            )
        else:
            require_training_dataset(dataset_id)
            if model_source == "uploaded":
                if user_model_service is None:
                    from services.user_model_service import UserModelService
                    user_model_service = UserModelService(sessionmaker=async_session_maker)
                package = await user_model_service.get_package_for_user(
                    task.uploaded_model_id, task.user_id
                )
                if (package.validation_status != "valid"
                        or package.version != task.uploaded_model_version
                        or package.id != hyperparameters.get("_uploaded_model_id")
                        or package.version != hyperparameters.get("_uploaded_model_version")):
                    raise ValueError("Training task uploaded model no longer matches its frozen reference")
                source_path = Path(package.storage_path)
                if source_path != Path(hyperparameters.get("_uploaded_model_path") or ""):
                    raise ValueError("Training task uploaded model path does not match its frozen reference")
                source_hash = await asyncio.to_thread(
                    lambda: hashlib.sha256(source_path.read_bytes()).hexdigest()
                )
                expected_hash = hyperparameters.get("_uploaded_model_content_hash", package.content_hash)
                if source_hash != package.content_hash or source_hash != expected_hash:
                    raise ValueError("Training task uploaded model source has changed")
            if (hyperparameters.get("transfer_learning")
                    and hyperparameters.get("transfer_source_type") == "upload"
                    and training_weight_service is None):
                from services.training_weight_service import TrainingWeightService
                training_weight_service = TrainingWeightService(sessionmaker=async_session_maker)
            env_overrides = await self._resolve_transfer_source(
                user_id=task.user_id, hyperparameters=hyperparameters,
                training_weight_service=training_weight_service, is_admin=is_admin,
            )
        return {
            "script_name": expected_script,
            "hyperparameters": hyperparameters,
            "log_file": Path(task.log_file_path),
            "output_path": Path(task.output_model_path),
            "env_overrides": env_overrides,
            "temp_data_root": None,
            "earth_training_spec": earth_spec,
        }

    async def cancel_training(self, task_id: int) -> bool:
        async with async_session_maker() as session:
            task = await session.get(ModelTrainingTask, task_id)
            if task is None or task.status != "queued":
                return False
            cancelled = await session.execute(
                update(ModelTrainingTask)
                .where(ModelTrainingTask.id == task_id, ModelTrainingTask.status == "queued")
                .values(status="cancelled", end_time=datetime.now(timezone.utc),
                        queue_position=None, metrics=json.dumps({"note": "Cancelled while queued"}))
            )
            if not cancelled.rowcount:
                return False
            await self._recalculate_queue_positions(session)
            await session.commit()
        self._wake_scheduler()
        return True

    def _earth_validator(self) -> Any:
        if self._earth_model_validator is None:
            from training_backbones.uploaded_model_earth_gate import build_validator

            self._earth_model_validator = build_validator()
        return self._earth_model_validator

    def get_available_scripts(self) -> list[str]:
        script_path = MODELS_DIR / UNIFIED_TRAINING_SCRIPT
        if not script_path.exists():
            return []
        return [UNIFIED_TRAINING_SCRIPT]

    async def start_training(
        self,
        user_id: int | None,
        model_script: str,
        hyperparameters: dict,
        custom_model_name: str | None = None,
        data_source: str = "default",
        data_service: DataService | None = None,
        personal_source_service: PersonalDataSourceService | None = None,
        model_source: str = "official",
        uploaded_model_id: str | None = None,
        user_model_service: Any | None = None,
        training_weight_service: Any | None = None,
        is_admin: bool = False,
        tag_ids: list[int] | None = None,
        dataset_id: str | None = None,
        dataset_registry: DatasetRegistry | None = None,
    ) -> ModelTrainingTask:
        requested_model_source = model_source
        model_source = (model_source or "official").strip().lower()
        if model_source not in ("official", "uploaded"):
            model_source = "official"

        if not custom_model_name or not custom_model_name.strip():
            raise ValueError("模型命名不能为空")

        # Dataset identity is resolved and authorized before any database write,
        # uploaded package load or subprocess scheduling happens.
        resolved_dataset_id = resolve_dataset_id(dataset_id, hyperparameters or {})
        earth_training = is_earth_dataset_id(resolved_dataset_id)
        earth_source = (
            requested_model_source if resolved_dataset_id in EARTH_THREE_HOURLY_DATASET_IDS
            else model_source if earth_training else "official"
        )
        earth_reference: dict | None = None
        earth_model_warnings: list[str] = []
        if not earth_training:
            # The Mars runners keep rejecting Earth identities; Earth has its own
            # preparation and training path below.
            require_training_dataset(resolved_dataset_id)
        registry = dataset_registry or self._default_dataset_registry()
        if earth_training and resolved_dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            # Reject unsupported Earth configurations (SPHERE, transfer learning,
            # wrong architecture, missing/extra uploaded id) and validate the strict
            # parameter contract before the task row exists.
            require_earth_training_configuration(
                model_source=earth_source,
                uploaded_model_id=uploaded_model_id,
                hyperparameters=hyperparameters or {},
                dataset_id=resolved_dataset_id,
            )
        dataset_binding = registry.build_training_binding(resolved_dataset_id)
        if earth_training and resolved_dataset_id not in EARTH_THREE_HOURLY_DATASET_IDS:
            # Preserve the daily package-before-configuration error precedence.
            require_earth_training_configuration(
                model_source=earth_source, uploaded_model_id=uploaded_model_id,
                hyperparameters=hyperparameters or {}, dataset_id=resolved_dataset_id,
            )

        earth_task_split = None
        if resolved_dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            from services.earth_task_split import build_earth_task_split, timeline_from_snapshot
            from services.training_split import TrainingSplitError
            normalized = require_earth_training_configuration(
                model_source=earth_source, uploaded_model_id=uploaded_model_id,
                hyperparameters=hyperparameters or {}, dataset_id=resolved_dataset_id,
            )
            try:
                earth_task_split = build_earth_task_split(
                    timeline_from_snapshot(dataset_binding["dataset_snapshot"]),
                    normalized["window"], normalized["horizon"], normalized,
                )
            except TrainingSplitError as exc:
                raise DatasetRequestError("invalid_earth_training_parameters", str(exc), status_code=422) from exc

        source = _normalize_training_data_source(data_source)

        async with async_session_maker() as session:
            await validate_owned_tags(session, user_id, tag_ids or [])
            existing = await session.execute(
                select(ModelTrainingTask).where(ModelTrainingTask.custom_model_name == custom_model_name.strip())
            )
            if existing.scalars().first():
                raise ValueError(f"模型名称 '{custom_model_name}' 已被使用，请换一个名称")

        if earth_training:
            # Earth runs its own server-side script; the client never picks it.
            model_script, raw_hypers = EARTH_TRAINING_SCRIPT, dict(hyperparameters or {})
            if earth_source == "uploaded":
                # Resolve the uploaded model now and keep the pinned reference, so a
                # later upload by the same user cannot change this task's code.
                earth_reference, earth_model_warnings = await self._resolve_earth_uploaded_model(
                    user_id=user_id,
                    uploaded_model_id=uploaded_model_id,
                    user_model_service=user_model_service,
                    custom_model_params=(hyperparameters or {}).get("custom_model_params"),
                    dataset_id=resolved_dataset_id,
                    input_channel_order=canonical_channel_order((hyperparameters or {}).get("selected_channels")),
                    window=(hyperparameters or {}).get('window', 56),
                    horizon=(hyperparameters or {}).get('horizon', 24),
                    batch_size=normalized["batch_size"] if resolved_dataset_id in EARTH_THREE_HOURLY_DATASET_IDS else 8,
                )
        elif model_source == "uploaded":
            model_script, raw_hypers = await self._resolve_uploaded_training_entrypoint(
                user_id=user_id,
                uploaded_model_id=uploaded_model_id,
                hyperparameters=hyperparameters,
                user_model_service=user_model_service,
            )
        else:
            model_script, raw_hypers = self._resolve_training_entrypoint(
                user_id=user_id,
                model_source=model_source,
                uploaded_model_id=uploaded_model_id,
                hyperparameters=hyperparameters,
                user_model_service=user_model_service,
            )

        if model_script == "__user_model_runner__":
            runner_path = Path(__file__).parent.parent / "training_backbones" / "user_model_runner.py"
            if not runner_path.exists():
                raise FileNotFoundError(f"Script {runner_path.name} not found in {runner_path.parent}")
        elif not MODELS_DIR.joinpath(model_script).exists():
            raise FileNotFoundError(f"Script {model_script} not found in {MODELS_DIR}")

        preserved_keys = ["model_source"]
        if model_source == "uploaded":
            preserved_keys.extend([
                "_uploaded_model_id",
                "_uploaded_model_version",
                "_uploaded_model_content_hash",
                "_uploaded_model_name",
                "_uploaded_model_filename",
                "_uploaded_model_path",
                "_uploaded_model_param_schema",
                "custom_model_params",
            ])
        preserved_hypers = {key: raw_hypers[key] for key in preserved_keys if key in raw_hypers}
        # The resolved id is the authority; the legacy key is kept in sync so the
        # training CLI and older readers see the same dataset.
        raw_hypers = {**raw_hypers, "training_dataset": resolved_dataset_id}
        if earth_training:
            # Earth never runs through the tolerant Mars normalizer: a silently
            # clamped window or a channel filtered down to an empty list would
            # train a different model than the user asked for.
            payload_hypers = require_earth_training_configuration(
                model_source=model_source,
                uploaded_model_id=uploaded_model_id,
                hyperparameters=raw_hypers,
                dataset_id=resolved_dataset_id,
            )
            payload_hypers["training_dataset"] = resolved_dataset_id
        else:
            payload_hypers = normalize_training_hyperparameters(raw_hypers)
        payload_hypers.update(preserved_hypers)
        # Labels are organization metadata, never model/cache identity or runner arguments.
        payload_hypers.pop("tag_ids", None)
        payload_hypers.pop("tags", None)
        payload_hypers["_data_source"] = source
        if resolved_dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            metrics_contract = earth_3hourly_metrics_contract(EARTH_3HOURLY_METRICS_SCHEMA_V3)
            payload_hypers[EARTH_METRICS_SCHEMA_KEY] = metrics_contract["schema"]
            payload_hypers[EARTH_METRICS_AGGREGATION_KEY] = metrics_contract["aggregation"]
        if earth_task_split is not None:
            from services.earth_task_split import EARTH_TASK_SPLIT_POLICY, TASK_SPLIT_POLICY_KEY
            payload_hypers["_earth_task_split"] = earth_task_split
            payload_hypers[TASK_SPLIT_POLICY_KEY] = EARTH_TASK_SPLIT_POLICY
        if earth_training and earth_reference is not None:
            # Record the pinned uploaded model identity on the task so history,
            # copy-config and prediction can all name the exact trained model. The
            # values come from the server-resolved package, never from the request.
            payload_hypers["_uploaded_model_id"] = earth_reference.package_id
            payload_hypers["_uploaded_model_version"] = earth_reference.version
            payload_hypers["_uploaded_model_content_hash"] = earth_reference.content_hash
            payload_hypers["_uploaded_model_name"] = earth_reference.display_name
            payload_hypers["_uploaded_model_filename"] = Path(earth_reference.source_path).name
            payload_hypers["_uploaded_model_param_schema"] = dict(earth_reference.param_schema)
            payload_hypers["custom_model_params"] = dict(earth_reference.custom_model_params)
            if resolved_dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
                payload_hypers["_earth_uploaded_reference"] = earth_reference.checkpoint_reference()
                from training_backbones.earth_3hourly_uploaded_contract import contract_schema_for_reference
                payload_hypers["_earth_uploaded_contract_schema"] = contract_schema_for_reference(payload_hypers["_earth_uploaded_reference"])
                payload_hypers["_earth_uploaded_reference"]["contract_schema"] = payload_hypers["_earth_uploaded_contract_schema"]
                from services.earth_derived_inputs import freeze_topography_for_reference
                try:
                    terrain = freeze_topography_for_reference(
                        payload_hypers["_earth_uploaded_reference"],
                        registry.get_earth_snapshot(resolved_dataset_id,
                            expected_fingerprint=dataset_binding["dataset_fingerprint"]),
                    )
                except (ValueError, OSError, RuntimeError) as exc:
                    raise DatasetRequestError("earth_topography_invalid", str(exc), status_code=409) from exc
                if terrain is not None:
                    payload_hypers["_earth_topography"] = terrain
        if not earth_training:
            await self._resolve_transfer_source(
                user_id=user_id,
                is_admin=is_admin,
                hyperparameters=payload_hypers,
                training_weight_service=training_weight_service,
            )

        async with async_session_maker() as session:
            # Recheck in the same transaction that writes the task and its associations.
            await begin_tag_write(session)
            tags = await validate_owned_tags(session, user_id, tag_ids or [])
            task = ModelTrainingTask(
                user_id=user_id,
                model_script=model_script,
                model_source=model_source,
                uploaded_model_id=payload_hypers.get("_uploaded_model_id"),
                uploaded_model_version=payload_hypers.get("_uploaded_model_version"),
                hyperparameters=json.dumps(payload_hypers),
                custom_model_name=custom_model_name,
                status="queued",
                queued_at=datetime.now(timezone.utc),
                **dataset_binding,
            )
            session.add(task)
            await session.flush()

            task_id = task.id
            log_file = LOGS_DIR / f"task_{task_id}.log"

            output_path = build_task_output_path(
                task_id,
                custom_model_name,
                results_dir=OUTPUT_MODELS_DIR,
            )

            task.log_file_path = str(log_file)
            task.output_model_path = str(output_path)
            await add_task_tag_links(session, [task_id], [tag.id for tag in tags])

            # Validate through the same preparation used at actual execution.
            # Only task facts are committed; no environment or plan survives
            # solely in this process's memory.
            await self._prepare_training_execution(
                task, dataset_registry=registry,
                training_weight_service=training_weight_service,
                user_model_service=user_model_service, is_admin=is_admin,
            )
            if earth_reference is not None:
                logger.info(
                    "Earth task %s pinned uploaded model %s v%s (%s…)",
                    task_id, earth_reference.package_id,
                    earth_reference.version, earth_reference.content_hash[:12],
                )

            await session.commit()
            logger.info(
                "Training task queued: id=%s script=%s source=%s effective=%s",
                task_id,
                model_script,
                source,
                payload_hypers.get("_effective_data_source", source),
            )

            if getattr(self, "_scheduler_started", False):
                await self._recalculate_queue_positions(session)
                await session.commit()
                self._wake_scheduler()
            else:
                task.status = "running"
                await session.commit()
                asyncio.create_task(self._execute_training_task(
                    task_id, dataset_registry=registry,
                    training_weight_service=training_weight_service,
                    user_model_service=user_model_service,
                ))

            return task

    @staticmethod
    def _default_dataset_registry() -> DatasetRegistry:
        return DatasetRegistry(
            EARTH_MERRA2_DIR, earth_dataset_id="earth_merra2_daily_v2",
            legacy_earth_package_dir=EARTH_MERRA2_V1_DIR,
            earth_3hourly_package_dir=EARTH_MERRA2_3HOURLY_DIR,
            earth_3hourly_5deg_package_dir=EARTH_MERRA2_3HOURLY_5DEG_DIR,
            verification_cache_dir=EARTH_TRAINING_CACHE_DIR,
        )

    def _restore_3hourly_training_spec(
        self, task: Any, dataset_registry: DatasetRegistry | None = None,
    ) -> dict:
        """Build the Earth runner spec from frozen task facts at every launch."""
        raw_hypers = json.loads(task.hyperparameters or "{}")
        from services.earth_task_split import required_task_split
        from services.training_split import TrainingSplitError
        try:
            task_split = required_task_split(raw_hypers)
        except TrainingSplitError as exc:
            raise DatasetRequestError("invalid_earth_training_parameters", str(exc), status_code=409) from exc
        public_hypers = {key: value for key, value in raw_hypers.items() if not key.startswith("_")}
        normalized = require_earth_training_configuration(
            model_source=raw_hypers.get("model_source", "official"),
            uploaded_model_id=raw_hypers.get("_uploaded_model_id"),
            hyperparameters=public_hypers, dataset_id=task.dataset_id,
        )
        metrics_contract = earth_task_metrics_contract(raw_hypers, executable=True)
        if metrics_contract is None:
            # Truly old unmarked queue rows predate area weighting. Keep their
            # existing statistical meaning rather than upgrading while launching.
            metrics_contract = earth_3hourly_metrics_contract(EARTH_3HOURLY_METRICS_SCHEMA_V2, executable=True)
        binding = {
            key: getattr(task, key, None) for key in (
                "dataset_id", "dataset_version", "dataset_fingerprint",
                "dataset_identity_status", "dataset_snapshot",
            )
        }
        if (binding["dataset_identity_status"] != "verified"
                or not binding["dataset_version"] or not binding["dataset_fingerprint"]
                or not binding["dataset_snapshot"]):
            raise DatasetRequestError(
                "dataset_identity_not_verified",
                "A queued three-hourly task must retain its verified dataset identity",
                status_code=409,
            )
        release = (dataset_registry or self._default_dataset_registry()).get_earth_snapshot(
            binding["dataset_id"], expected_fingerprint=binding["dataset_fingerprint"],
        )
        reference = raw_hypers.get("_earth_uploaded_reference")
        if normalized.get("model_source") == "uploaded":
            if (not isinstance(reference, dict)
                    or reference.get("package_id") != task.uploaded_model_id
                    or reference.get("version") != task.uploaded_model_version
                    or reference.get("content_hash") != raw_hypers.get("_uploaded_model_content_hash")):
                raise DatasetRequestError("uploaded_model_reference_missing", "Queued task has no matching frozen model reference", status_code=409)
            from training_backbones.earth_3hourly_uploaded_contract import contract_schema_for_reference
            try:
                schema = contract_schema_for_reference(reference)
            except (ValueError, TypeError) as exc:
                raise DatasetRequestError('uploaded_model_contract_invalid', str(exc), status_code=409) from exc
            if raw_hypers.get('_earth_uploaded_contract_schema', schema) != schema:
                raise DatasetRequestError('uploaded_model_contract_invalid', 'Frozen spatial contract disagrees with the model source', status_code=409)
            from services.earth_derived_inputs import topography_for_reference
            try:
                topography_for_reference(reference, release, raw_hypers.get("_earth_topography"))
            except (ValueError, OSError, RuntimeError) as exc:
                raise DatasetRequestError("earth_topography_invalid", str(exc), status_code=409) from exc
        if task_split is not None:
            from services.earth_task_split import validate_earth_task_split
            try:
                validate_earth_task_split(task_split, release.dates,
                                          normalized["window"], normalized["horizon"], normalized)
            except TrainingSplitError as exc:
                raise DatasetRequestError("invalid_earth_training_parameters", str(exc), status_code=409) from exc
        spec = build_earth_training_spec(
            task_split=task_split,
            task_id=task.id, dataset_binding=binding, hyperparameters=normalized,
            uploaded_model=raw_hypers.get("_earth_uploaded_reference"),
            metrics_contract=metrics_contract,
            topography=raw_hypers.get("_earth_topography"),
        )
        spec["data_path"] = str(Path(release.data_path).resolve())
        return spec

    def _resolve_training_entrypoint(
        self,
        user_id: int | None,
        model_source: str,
        uploaded_model_id: str | None,
        hyperparameters: dict | None,
        user_model_service: Any | None,
    ) -> tuple[str, dict]:
        uploaded_only_keys = {
            "_uploaded_model_id",
            "_uploaded_model_version",
            "_uploaded_model_content_hash",
            "_uploaded_model_name",
            "_uploaded_model_filename",
            "_uploaded_model_path",
            "_uploaded_model_param_schema",
            "custom_model_params",
        }
        payload = {
            key: value
            for key, value in (hyperparameters or {}).items()
            if key not in uploaded_only_keys
        }
        payload["model_source"] = "official"
        return UNIFIED_TRAINING_SCRIPT, payload

    async def _resolve_earth_uploaded_model(
        self,
        *,
        user_id: int | None,
        uploaded_model_id: Any,
        user_model_service: Any | None,
        custom_model_params: Any,
        dataset_id: str = "earth_merra2_daily_v2",
        input_channel_order: list[str] | None = None,
        window: int = 56,
        horizon: int = 24,
        batch_size: int = 8,
    ) -> tuple[dict, list[str]]:
        """Pin one uploaded model for an Earth run.

        Returns ``(reference, warnings)`` where the reference carries the package
        identity, the verified parameter schema, the validated custom parameters and
        the hash-verified source text. Everything is captured now, so the task keeps
        training exactly this code even if the user uploads a newer version later.
        """
        if user_id is None:
            raise ValueError("user_id is required for uploaded model training")
        if not uploaded_model_id:
            raise DatasetRequestError(
                "invalid_earth_training_parameters",
                "uploaded_model_id is required when model_source is 'uploaded'",
                status_code=422,
            )
        if user_model_service is None:
            raise ValueError("user_model_service is required for uploaded model training")

        try:
            package = await user_model_service.get_package_for_user(uploaded_model_id, user_id)
        except (FileNotFoundError, PermissionError):
            raise
        if package is None:
            raise DatasetRequestError(
                "uploaded_model_not_found",
                "The uploaded model could not be found for this account",
                status_code=404,
            )
        if getattr(package, "validation_status", None) != "valid":
            raise DatasetRequestError(
                "uploaded_model_invalid",
                "The uploaded model must be valid before it can be trained on Earth data",
                status_code=422,
            )

        # Earth compatibility is its own question: a model validated for Mars is not
        # thereby usable on the Earth feed.
        try:
            param_schema = json.loads(getattr(package, "param_schema", None) or "{}")
        except Exception:
            param_schema = {}
        if not isinstance(param_schema, dict):
            param_schema = {}
        resolved_params, param_errors = UserModelValidator.normalize_custom_params(
            param_schema, custom_model_params
        )
        if param_errors:
            raise DatasetRequestError(
                "invalid_earth_training_parameters",
                param_errors[0],
                status_code=422,
            )

        if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            from services.user_model_service import UserModelService
            from training_backbones.earth_3hourly_uploaded_contract import contract_profile

            stored = await asyncio.to_thread(
                UserModelService.earth_compatibility_for_package, package, dataset_id=dataset_id,
            )
            verdict = EarthCompatibility(
                compatible=stored['compatible'], reasons=stored['reasons'],
                warnings=stored['warnings'], datasets=stored['datasets'],
                output_shape=stored['output_shape'], code=stored['code'],
                dataset_id=dataset_id, status=stored['status'],
            )
            if (verdict.code == 'uploaded_model_compatibility_unknown'
                    and dataset_id not in verdict.datasets):
                verdict.code = 'uploaded_model_not_earth_3hourly_compatible'
            if verdict.compatible:
                maximum = contract_profile(stored['contract_schema'])['max_batch_size']
                if not 1 <= batch_size <= maximum:
                    raise DatasetRequestError('invalid_earth_training_parameters',
                        f"Earth training requires batch_size between 1 and {maximum}", status_code=422)
        else:
            verdict = await asyncio.to_thread(
                evaluate_package_earth_compatibility, package, validator=self._earth_validator(),
                dataset_id=dataset_id,
            )
        if not verdict.compatible:
            from services.user_model_validator import VALIDATION_TIMEOUT_CODE
            prefix = ("Earth upload validation did not finish: "
                      if verdict.code == VALIDATION_TIMEOUT_CODE else
                      "The uploaded model is not compatible with this Earth dataset: ")
            raise DatasetRequestError(
                verdict.code or "uploaded_model_not_earth_compatible",
                prefix + "; ".join(verdict.reasons or ["unsupported model"]), status_code=422,
            )

        contract_schema = None
        if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            from training_backbones.earth_3hourly_uploaded_contract import contract_profile
            contract_schema = (verdict.datasets.get(dataset_id) or {}).get('schema')
            try:
                contract_profile(contract_schema)
            except (ValueError, TypeError) as exc:
                raise DatasetRequestError('uploaded_model_contract_invalid',
                    'The Earth upload report contains no supported spatial contract', status_code=422) from exc

        reference = build_reference(
            package=package,
            param_schema=param_schema,
            custom_model_params=resolved_params,
            embed_source=True,
            contract_schema=contract_schema,
        )
        # The server must be able to prove the code it pins is the code it validated.
        resolve_source_text_strict(
            {
                "source_path": reference.source_path,
                "content_hash": reference.content_hash,
                "source_text": reference.source_text,
            }
        )
        if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            from services.earth_derived_inputs import declaration_for_reference
            try:
                declaration = declaration_for_reference(reference.checkpoint_reference())
                if "topography" in declaration:
                    from services.earth_topography import load_earth_topography
                    report = json.loads(getattr(package, "validation_report", None) or "{}")
                    proof = (report.get("earth_datasets") or {}).get(dataset_id) or {}
                    asset = (load_earth_topography(dataset_id=dataset_id)
                             if dataset_id == "earth_merra2_3hourly_5deg_v1" else load_earth_topography())
                    if (proof.get("derived_inputs") != declaration
                            or proof.get("topography_identity") != asset.identity):
                        raise ValueError("Terrain validation evidence differs from the current asset; revalidate this uploaded model")
            except (ValueError, OSError, RuntimeError) as exc:
                raise DatasetRequestError("earth_topography_invalid", str(exc), status_code=409) from exc
        warnings = list(verdict.warnings)
        if verdict.output_shape:
            warnings.append(
                "Earth upload validation output shape: " + "x".join(str(v) for v in verdict.output_shape)
            )
        return reference, warnings

    async def _resolve_uploaded_training_entrypoint(
        self,
        user_id: int | None,
        uploaded_model_id: str | None,
        hyperparameters: dict | None,
        user_model_service: Any | None,
    ) -> tuple[str, dict]:
        if user_id is None:
            raise ValueError("user_id is required for uploaded model training")
        if not uploaded_model_id:
            raise ValueError("uploaded_model_id is required for uploaded model training")
        if user_model_service is None:
            raise ValueError("user_model_service is required for uploaded model training")

        package = await user_model_service.get_package_for_user(uploaded_model_id, user_id)
        report = json.loads(getattr(package, "validation_report", None) or "{}")
        if isinstance(report, dict) and (report.get("mars") or {}).get("compatible") is False:
            raise ValueError("The uploaded model is not compatible with Mars data")
        if getattr(package, "validation_status", None) != "valid":
            raise ValueError("Uploaded model package must be valid before training")

        try:
            param_schema = json.loads(getattr(package, "param_schema", None) or "{}")
        except Exception:
            param_schema = {}
        if not isinstance(param_schema, dict):
            param_schema = {}

        payload = dict(hyperparameters or {})
        payload["model_source"] = "uploaded"
        payload["_uploaded_model_id"] = getattr(package, "id", uploaded_model_id)
        payload["_uploaded_model_version"] = getattr(package, "version", None)
        payload["_uploaded_model_content_hash"] = getattr(package, "content_hash", None)
        payload["_uploaded_model_name"] = getattr(package, "display_name", None)
        payload["_uploaded_model_filename"] = getattr(package, "original_filename", None)
        payload["_uploaded_model_path"] = getattr(package, "storage_path", None)
        payload["_uploaded_model_param_schema"] = param_schema
        payload.setdefault("custom_model_params", {})
        return "__user_model_runner__", payload

    async def _resolve_transfer_source(
        self,
        user_id: int | None,
        hyperparameters: dict,
        training_weight_service: Any | None,
        is_admin: bool | None = False,
    ) -> dict[str, str]:
        if not hyperparameters.get("transfer_learning"):
            return {}

        source_type = str(hyperparameters.get("transfer_source_type") or "task").strip().lower()
        if source_type == "upload":
            if training_weight_service is None:
                raise ValueError("training_weight_service is required for uploaded transfer weights")
            weight_id = str(hyperparameters.get("transfer_weight_id") or "").strip()
            if not weight_id:
                raise ValueError("transfer_weight_id is required for uploaded transfer weights")
            if user_id is None:
                raise ValueError("user_id is required for uploaded transfer weights")
            record = await training_weight_service.get_weight_for_user(weight_id, user_id)
            if getattr(record, "status", None) != "ready":
                raise ValueError("Uploaded transfer weight must be ready before training")
            weight_path = Path(getattr(record, "storage_path", "") or "")
            if not is_valid_model_weight_file(weight_path):
                raise FileNotFoundError("Uploaded transfer weight file is missing")
            return {"ASTRAATMOS_TRANSFER_WEIGHT_PATH": str(weight_path)}

        source_task_id = int(hyperparameters.get("transfer_source_task_id") or 0)
        if source_task_id <= 0:
            raise ValueError("transfer_source_task_id is required for task transfer learning")

        async with async_session_maker() as session:
            source_task = await session.get(ModelTrainingTask, source_task_id)
            if is_admin is None:
                owner = await session.get(User, user_id) if user_id is not None else None
                is_admin = bool(owner is not None and owner.is_active and owner.role == "admin")

        if source_task is None:
            raise ValueError("Transfer source task not found")
        if not is_admin and getattr(source_task, "user_id", None) not in (None, user_id):
            raise PermissionError("No permission to access this transfer source task")
        if getattr(source_task, "status", None) != "completed":
            raise ValueError("Transfer source task must be completed")

        # An Earth (MERRA-2) checkpoint is a different container and a different
        # planet: it can never be a Mars transfer source, and its tensors must not
        # be extracted and passed off as a Mars state dict.
        if is_earth_training_task(source_task):
            raise DatasetRequestError(
                "dataset_transfer_not_supported",
                "Earth tasks cannot be used as a transfer learning source for Mars models",
                status_code=409,
            )

        weight_path = Path(getattr(source_task, "output_model_path", "") or "")
        if not is_valid_model_weight_file(weight_path):
            raise FileNotFoundError("Transfer source task weight file is missing")

        source_hypers = self._parse_task_hyperparameters(source_task)
        self._validate_transfer_task_compatibility(source_task, source_hypers, hyperparameters)
        return {"ASTRAATMOS_TRANSFER_WEIGHT_PATH": str(weight_path)}

    @staticmethod
    def _parse_task_hyperparameters(task: Any) -> dict:
        try:
            parsed = json.loads(getattr(task, "hyperparameters", "") or "{}")
        except Exception:
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    def _validate_transfer_task_compatibility(
        self,
        source_task: Any,
        source_hypers: dict,
        target_hypers: dict,
    ) -> None:
        source_model = str(getattr(source_task, "model_source", None) or source_hypers.get("model_source") or "official")
        target_model = str(target_hypers.get("model_source") or "official")
        if source_model != target_model:
            raise ValueError("Transfer source task model source does not match current training")

        keys = [
            "selected_channels",
            "window",
            "horizon",
            "use_sphere",
        ]
        if target_model == "official":
            keys.append("model_architecture")
            keys.extend(key for key in target_hypers if key in ARCHITECTURE_PARAM_KEYS)
        else:
            if getattr(source_task, "uploaded_model_id", None) != target_hypers.get("_uploaded_model_id"):
                raise ValueError("Transfer source uploaded model does not match current uploaded model")
            if getattr(source_task, "uploaded_model_version", None) != target_hypers.get("_uploaded_model_version"):
                raise ValueError("Transfer source uploaded model version does not match current uploaded model")

        for key in keys:
            if source_hypers.get(key) != target_hypers.get(key):
                raise ValueError(f"Transfer source task configuration does not match: {key}")

    async def _prepare_personal_training_env(
        self,
        user_id: int | None,
        task_id: int,
        data_service: DataService | None,
        personal_source_service: PersonalDataSourceService | None,
    ) -> tuple[dict[str, str], Path | None, str, str | None]:
        if user_id is None:
            return {}, None, "default", "personal source requested without user id; fallback to default"
        if data_service is None or personal_source_service is None:
            return {}, None, "default", "personal source resolver unavailable; fallback to default"

        temp_root = Path(tempfile.mkdtemp(prefix=f"astraatmos_train_{task_id}_"))
        openmars_dir = temp_root / "openmars"
        mcd_dir = temp_root / "MCD"
        openmars_dir.mkdir(parents=True, exist_ok=True)
        mcd_dir.mkdir(parents=True, exist_ok=True)

        has_personal = False
        try:
            years = data_service.get_available_years() or [27, 28]
            for year in years:
                resolution = await personal_source_service.resolve_for_year("personal", year, user_id)
                if resolution.effective_source != "default":
                    has_personal = True

                my = int(resolution.mars_year)
                openmars_path = openmars_dir / f"openmars_my{my}_ls_personal.nc"
                self._write_openmars_nc(openmars_path, resolution.openmars_data)

                mcd_src = resolution.mcd_raw_data or data_service.get_mcd_data(my)
                mcd_path = mcd_dir / f"MCD_MY{my}_Lat-90-90_real.nc"
                self._write_mcd_nc(mcd_path, mcd_src, resolution.openmars_data)

            effective_source = "personal" if has_personal else "default"
            note = None if has_personal else "personal datasets unavailable; fallback to default training data"
            env = {
                "ASTRAATMOS_OPENMARS_DIR": str(openmars_dir),
                "ASTRAATMOS_MCD_DIR": str(mcd_dir),
            }
            return env, temp_root, effective_source, note
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise

    async def prepare_task_inference_data_env(
        self,
        task: ModelTrainingTask,
        data_service: DataService | None,
        personal_source_service: PersonalDataSourceService | None,
    ) -> tuple[dict[str, str], Path | None]:
        return {}, None

    def cleanup_temp_data_root(self, temp_data_root: Path | None) -> None:
        if temp_data_root is not None:
            shutil.rmtree(temp_data_root, ignore_errors=True)

    def _write_openmars_nc(self, file_path: Path, openmars_data: dict[str, Any]) -> None:
        lat = np.asarray(openmars_data.get("lat"), dtype=np.float32).reshape(-1)
        lon = np.asarray(openmars_data.get("lon"), dtype=np.float32).reshape(-1)
        ls = np.asarray(openmars_data.get("ls"), dtype=np.float32).reshape(-1)
        o3 = np.asarray(openmars_data.get("o3col"), dtype=np.float32)

        if o3.ndim == 4:
            o3 = np.nanmean(o3, axis=1)
        if o3.ndim != 3:
            raise ValueError(f"Invalid openmars o3col shape: {o3.shape}")

        n_time = min(len(ls), o3.shape[0])
        if n_time <= 0:
            raise ValueError("Empty openmars timeline")

        ls = ls[:n_time]
        o3 = o3[:n_time, : len(lat), : len(lon)]

        sort_idx = np.argsort(ls)
        ls = ls[sort_idx]
        o3 = o3[sort_idx]

        with nc.Dataset(str(file_path), "w", format="NETCDF4") as ds:
            ds.createDimension("time", n_time)
            ds.createDimension("lat", len(lat))
            ds.createDimension("lon", len(lon))

            v_ls = ds.createVariable("Ls", "f4", ("time",))
            v_lat = ds.createVariable("lat", "f4", ("lat",))
            v_lon = ds.createVariable("lon", "f4", ("lon",))
            v_o3 = ds.createVariable("o3col", "f4", ("time", "lat", "lon"), zlib=True)

            v_ls[:] = ls
            v_lat[:] = lat
            v_lon[:] = lon
            v_o3[:] = o3

    def _write_mcd_nc(self, file_path: Path, mcd_data: dict[str, Any], openmars_data: dict[str, Any]) -> None:
        lat_raw = mcd_data.get("lat")
        lon_raw = mcd_data.get("lon")
        ls_raw = mcd_data.get("ls")

        lat = np.asarray(lat_raw if lat_raw is not None else openmars_data.get("lat"), dtype=np.float32).reshape(-1)
        lon = np.asarray(lon_raw if lon_raw is not None else openmars_data.get("lon"), dtype=np.float32).reshape(-1)
        ls = np.asarray(ls_raw if ls_raw is not None else openmars_data.get("ls"), dtype=np.float32).reshape(-1)

        if ls.size == 0:
            raise ValueError("Empty MCD ls timeline")

        hourly_vars: dict[str, np.ndarray] = {}
        max_hour = 1
        min_time = int(ls.shape[0])

        for var in MCD_VARIABLES:
            arr = None
            hourly_key = f"{var}_hourly"
            if hourly_key in mcd_data and mcd_data[hourly_key] is not None:
                arr = np.asarray(mcd_data[hourly_key], dtype=np.float32)
            elif var in mcd_data and mcd_data[var] is not None:
                arr = np.asarray(mcd_data[var], dtype=np.float32)

            if arr is None:
                raise ValueError(f"MCD variable missing: {var}")

            if arr.ndim == 3:
                arr = arr[:, None, :, :]
            elif arr.ndim != 4:
                raise ValueError(f"Invalid MCD shape for {var}: {arr.shape}")

            hourly_vars[var] = arr
            max_hour = max(max_hour, int(arr.shape[1]))
            min_time = min(min_time, int(arr.shape[0]))

        lat_size = min(int(len(lat)), *[int(v.shape[2]) for v in hourly_vars.values()])
        lon_size = min(int(len(lon)), *[int(v.shape[3]) for v in hourly_vars.values()])
        if min_time <= 0 or lat_size <= 0 or lon_size <= 0:
            raise ValueError("Invalid MCD dimensions after alignment")

        ls = ls[:min_time]
        lat = lat[:lat_size]
        lon = lon[:lon_size]

        normalized: dict[str, np.ndarray] = {}
        for var, arr in hourly_vars.items():
            arr = arr[:min_time, :, :lat_size, :lon_size]
            h = int(arr.shape[1])
            if h < max_hour:
                repeat_factor = int(np.ceil(max_hour / h))
                arr = np.repeat(arr, repeat_factor, axis=1)[:, :max_hour, :, :]
            elif h > max_hour:
                arr = arr[:, :max_hour, :, :]
            normalized[var] = arr

        with nc.Dataset(str(file_path), "w", format="NETCDF4") as ds:
            ds.createDimension("sol", min_time)
            ds.createDimension("hour", max_hour)
            ds.createDimension("lat", lat_size)
            ds.createDimension("lon", lon_size)

            v_ls = ds.createVariable("Ls", "f4", ("sol",))
            v_lat = ds.createVariable("lat", "f4", ("lat",))
            v_lon = ds.createVariable("lon", "f4", ("lon",))
            v_ls[:] = ls
            v_lat[:] = lat
            v_lon[:] = lon

            for var in MCD_VARIABLES:
                v = ds.createVariable(var, "f4", ("sol", "hour", "lat", "lon"), zlib=True)
                v[:] = normalized[var]

    async def _run_training_subprocess(
        self,
        task_id: int,
        script_name: str,
        hyperparameters: dict,
        log_file: Path,
        output_path: Path,
        env_overrides: dict[str, str] | None = None,
        temp_data_root: Path | None = None,
        earth_training_spec: dict | None = None,
    ):
        total_epochs = hyperparameters.get("epochs", 1)
        async with async_session_maker() as session:
            task = await session.get(ModelTrainingTask, task_id)
            if task:
                require_active_dataset(training_task_dataset_id(task))
                task.status = "running"
                task.total_epochs = total_epochs
                await session.commit()

        if script_name == "__user_model_runner__":
            script_path = Path(__file__).parent.parent / "training_backbones" / "user_model_runner.py"
        else:
            script_path = MODELS_DIR / script_name
        python_exe = getattr(config, "TRAINING_PYTHON_PATH", sys.executable)

        args = [python_exe, str(script_path)]
        if earth_training_spec is None:
            args.extend(build_hyperparameter_args(hyperparameters))
            if script_name == "__user_model_runner__":
                args.extend([
                    "--uploaded_model_path",
                    str(hyperparameters["_uploaded_model_path"]),
                    "--uploaded_model_param_schema",
                    json.dumps(hyperparameters.get("_uploaded_model_param_schema") or {}),
                ])
        # The Earth runner receives its identity and parameters through the
        # environment only: never as CLI arguments that would leak a snapshot or
        # let a flag silently change the dataset.
        args.extend(["--output_path", str(output_path)])

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"--- 训练任务 {task_id} 已启动 ---\n")
            f.flush()

        try:
            process_env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            }
            if env_overrides:
                process_env.update(env_overrides)
            if earth_training_spec is not None:
                process_env["ASTRAATMOS_EARTH_TRAINING_SPEC"] = json.dumps(earth_training_spec)

            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(MODELS_DIR),
                env=process_env,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            logger.info("Training subprocess started: task_id=%s pid=%s script=%s", task_id, process.pid, script_name)

            async with async_session_maker() as session:
                task = await session.get(ModelTrainingTask, task_id)
                if task:
                    task.pid = process.pid
                    await session.commit()

            start_time_ts = time.time()
            from services.ws_manager import manager as ws_manager
            loop = asyncio.get_event_loop()

            loss_history_buf = {"train": [], "val": []}
            pending_progress_updates = []
            async with async_session_maker() as session:
                task = await session.get(ModelTrainingTask, task_id)
                if task and task.loss_history:
                    try:
                        loss_history_buf = json.loads(task.loss_history)
                    except Exception:
                        pass

            def read_and_parse_thread():
                with open(log_file, "a", encoding="utf-8") as f:
                    for line in iter(process.stdout.readline, ""):
                        f.write(line)
                        f.flush()

                        progress_data = self._parse_progress_from_log(line, total_epochs, start_time_ts, loss_history_buf)
                        if progress_data:
                            pending_progress_updates.append(
                                asyncio.run_coroutine_threadsafe(
                                    self._update_task_progress(task_id, progress_data, ws_manager),
                                    loop,
                                )
                            )
                    process.stdout.close()

            await asyncio.to_thread(read_and_parse_thread)
            if pending_progress_updates:
                await asyncio.gather(
                    *(asyncio.wrap_future(future) for future in pending_progress_updates),
                    return_exceptions=True,
                )
            returncode = await asyncio.to_thread(process.wait)
            failure_code = (
                classify_training_log(log_file) if returncode != 0 else None
            )
            model_artifact_valid = returncode == 0 and is_valid_model_weight_file(output_path)
            # Earth runs add a real contract check on top of "the file exists and
            # is not empty": the checkpoint must reload strictly, match the task
            # binding and carry the task's own metrics.
            earth_artifact: dict | None = None
            earth_artifact_error: Exception | None = None
            mars_snapshot: dict | None = None
            if earth_training_spec is not None and model_artifact_valid:
                try:
                    expected_hypers = dict(earth_training_spec["hyperparameters"])
                    if earth_training_spec["dataset_binding"]["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS:
                        from services.earth_task_split import (
                            TASK_SPLIT_KEY, TASK_SPLIT_POLICY_KEY, required_task_split,
                        )
                        frozen_hypers = hyperparameters
                        async with async_session_maker() as session:
                            frozen_task = await session.get(ModelTrainingTask, task_id)
                            if frozen_task is not None:
                                frozen_hypers = json.loads(frozen_task.hyperparameters or "{}")
                        task_split = required_task_split(frozen_hypers)
                        if task_split != earth_training_spec.get("task_split"):
                            raise EarthArtifactError("Training spec partitions do not match the frozen task")
                        if frozen_hypers.get("_earth_topography") != earth_training_spec.get("topography"):
                            raise EarthArtifactError("Training spec terrain does not match the frozen task")
                        expected_hypers.update({
                            key: frozen_hypers[key]
                            for key in (TASK_SPLIT_KEY, TASK_SPLIT_POLICY_KEY, EARTH_METRICS_SCHEMA_KEY,
                                        EARTH_METRICS_AGGREGATION_KEY, "_earth_topography")
                            if key in frozen_hypers
                        })
                        execution_contract = earth_spec_metrics_contract(earth_training_spec)
                        frozen_contract = earth_task_metrics_contract(frozen_hypers, executable=True)
                        if frozen_contract is None:
                            frozen_contract = earth_3hourly_metrics_contract(EARTH_3HOURLY_METRICS_SCHEMA_V2, executable=True)
                        if execution_contract != frozen_contract:
                            raise EarthArtifactError("Training spec metrics do not match the frozen task contract")
                    if "task_split" in earth_training_spec:
                        expected_hypers["_earth_task_split"] = earth_training_spec["task_split"]
                    uploaded = earth_training_spec.get("uploaded_model")
                    if uploaded:
                        expected_hypers.update({
                            "_uploaded_model_id": uploaded["package_id"],
                            "_uploaded_model_version": uploaded["version"],
                            "_uploaded_model_content_hash": uploaded["content_hash"],
                            "custom_model_params": uploaded["custom_model_params"],
                        })
                    earth_artifact = await asyncio.to_thread(
                        load_earth_training_artifact,
                        output_path,
                        earth_training_spec["dataset_binding"],
                        expected_hypers,
                        earth_training_spec["task_id"],
                    )
                    if earth_training_spec["dataset_binding"]["dataset_id"] in EARTH_THREE_HOURLY_DATASET_IDS:
                        if any(earth_artifact.metrics.get(key) != value for key, value in execution_contract.items()):
                            raise EarthArtifactError("Earth checkpoint metrics do not match the frozen execution contract")
                        await asyncio.to_thread(verify_earth_model_reload_isolated, output_path)
                except Exception as exc:  # noqa: BLE001 - reported as a task failure
                    earth_artifact_error = exc
                    model_artifact_valid = False
            elif model_artifact_valid and not earth_training_spec:
                try:
                    import torch
                    try:
                        checkpoint = await asyncio.to_thread(
                            torch.load, output_path, map_location="cpu", weights_only=True
                        )
                    except Exception:
                        # Pre-versioned Mars artifacts remain readable as legacy
                        # weights. New runner-produced envelopes are validated
                        # strictly below; arbitrary legacy bytes keep the older
                        # file-existence contract for compatibility.
                        checkpoint = None
                    task_hypers = self._parse_task_hyperparameters(task)
                    if isinstance(checkpoint, dict) and ("schema" in checkpoint or isinstance(checkpoint.get("model_state_dict"), dict) or "data_binding" in checkpoint):
                        validate_mars_checkpoint(
                            checkpoint,
                            selected_channels=list(task_hypers.get("selected_channels") or []),
                            window=int(task_hypers.get("window", 3)),
                            horizon=int(task_hypers.get("horizon", 3)),
                        )
                        mars_snapshot = mars_checkpoint_identity_snapshot(checkpoint)
                        expected_dataset = getattr(task, "dataset_id", None) or task_hypers.get("training_dataset")
                        if expected_dataset and mars_snapshot["dataset_id"] != expected_dataset:
                            raise ValueError("Mars checkpoint data identity does not match the training task")
                except Exception as exc:  # noqa: BLE001 - artifact contract failure
                    earth_artifact_error = exc
                    model_artifact_valid = False
            status = "completed" if model_artifact_valid else "failed"

            if returncode == 0 and not model_artifact_valid:
                detail = (
                    str(earth_artifact_error)
                    if earth_artifact_error is not None
                    else INVALID_MODEL_ARTIFACT_ERROR
                )
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[System error]: {detail}\n")
                logger.error(
                    "Training artifact rejected: task_id=%s output_path=%s detail=%s",
                    task_id,
                    output_path,
                    detail,
                )

            async with async_session_maker() as session:
                task = await session.get(ModelTrainingTask, task_id)
                if task:
                    # A user stop is terminal: a late exit-0 completion callback
                    # must never turn a stopped run into a completed one.
                    stopped = _task_was_stopped(task.metrics)
                    if stopped:
                        status = "failed"
                    task.status = status
                    task.end_time = datetime.now(timezone.utc)
                    task.progress = 100.0 if status == "completed" else task.progress
                    if stopped:
                        task.metrics = json.dumps({"note": "Stopped by user"})
                    elif status == "completed":
                        if earth_artifact is not None:
                            # The published DU metrics come from the checkpoint the
                            # parent just verified, never from log scraping.
                            task.metrics = json.dumps(earth_artifact.metrics)
                        else:
                            # New Mars runners publish a checkpoint envelope
                            # containing the exact source manifest and
                            # normalization contract. Persist the same snapshot
                            # on the task so prediction does not trust a new
                            # request or a mutable environment.
                            if mars_snapshot is not None:
                                task.dataset_snapshot = json.dumps(mars_snapshot, sort_keys=True)
                                task.dataset_fingerprint = mars_snapshot["dataset_fingerprint"]
                                task.dataset_identity_status = "verified"
                            else:
                                task.dataset_identity_status = "legacy"
                            parsed_metrics = self._extract_metrics_from_log(log_file)
                            task.metrics = json.dumps(parsed_metrics) if parsed_metrics else json.dumps({"note": "completed"})
                    elif returncode == 0:
                        task.progress = min(float(task.progress or 0.0), 99.0)
                        if earth_artifact_error is not None:
                            task.metrics = json.dumps({
                                "error_code": getattr(
                                    earth_artifact_error,
                                    "code",
                                    "invalid_earth_training_artifact",
                                ),
                                "error": str(earth_artifact_error),
                            })
                        else:
                            task.metrics = json.dumps({"error": INVALID_MODEL_ARTIFACT_ERROR})
                    elif failure_code == CUDA_OOM_ERROR_CODE:
                        task.metrics = json.dumps(
                            {
                                "error_code": CUDA_OOM_ERROR_CODE,
                                "error": "GPU memory is exhausted",
                            }
                        )
                    await session.commit()

            if task and failure_code == CUDA_OOM_ERROR_CODE:
                try:
                    async with async_session_maker() as notification_session:
                        notification_task = await notification_session.get(
                            ModelTrainingTask,
                            task_id,
                        )
                        if notification_task and await ensure_cuda_oom_notification(
                            notification_session,
                            notification_task,
                        ):
                            await notification_session.commit()
                except Exception:
                    logger.exception(
                        "Could not create CUDA OOM notification: task_id=%s",
                        task_id,
                    )

            if task:
                await ws_manager.broadcast_to_task(str(task_id), {
                    "type": "status_update",
                    "task_id": task_id,
                    "status": status,
                })
                logger.info("Training task finished: id=%s status=%s", task_id, status)

        except Exception:
            error_msg = traceback.format_exc()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\\n[系统错误]:\\n{error_msg}")

            logger.error(f"Task {task_id} failed with error: {error_msg}")

            async with async_session_maker() as session:
                task = await session.get(ModelTrainingTask, task_id)
                if task:
                    task.status = "failed"
                    task.end_time = datetime.now(timezone.utc)
                    await session.commit()
        finally:
            if temp_data_root is not None:
                shutil.rmtree(temp_data_root, ignore_errors=True)

    async def _update_task_progress(self, task_id: int, progress_data: dict, ws_manager):
        async with async_session_maker() as session:
            update_values = {
                "progress": progress_data["progress"],
                "current_epoch": progress_data["current_epoch"],
                "current_loss": progress_data["current_loss"],
                "eta": progress_data["eta"],
            }
            if "loss_history" in progress_data:
                update_values["loss_history"] = json.dumps(progress_data["loss_history"])

            await session.execute(
                update(ModelTrainingTask)
                .where(ModelTrainingTask.id == task_id)
                .values(**update_values)
            )
            await session.commit()

        await ws_manager.broadcast_to_task(str(task_id), {
            "type": "training_update",
            "task_id": task_id,
            "data": progress_data,
        })

    async def get_task(self, task_id: int) -> ModelTrainingTask:
        async with async_session_maker() as session:
            task = await session.get(ModelTrainingTask, task_id)
            return task

    async def get_all_tasks(self) -> list[ModelTrainingTask]:
        async with async_session_maker() as session:
            result = await session.execute(select(ModelTrainingTask).order_by(ModelTrainingTask.id.desc()))
            return result.scalars().all()

    async def rename_model(self, task_id: int, model_name: str) -> ModelTrainingTask:
        normalized_name = (model_name or "").strip()
        if not normalized_name:
            raise ValueError("模型名称不能为空")
        if len(normalized_name) > 255:
            raise ValueError("模型名称不能超过 255 个字符")

        async with async_session_maker() as session:
            task = await session.get(ModelTrainingTask, task_id)
            if not task:
                raise FileNotFoundError("Task not found")
            # The display name belongs to the experiment record, so it is safe
            # to edit it while queued/running as well as after completion.
            # Keep the status check deliberately open for historical and
            # cancelled records; permission is enforced by the router.
            allowed_statuses = {"queued", "pending", "running", "completed", "failed", "cancelled"}
            if str(task.status or "").lower() not in allowed_statuses:
                raise ValueError("当前状态不允许重命名")

            existing = await session.execute(
                select(ModelTrainingTask).where(
                    ModelTrainingTask.custom_model_name == normalized_name,
                    ModelTrainingTask.id != task_id,
                )
            )
            if existing.scalars().first():
                raise ValueError(f"模型名称 '{normalized_name}' 已被使用，请换一个名称")

            task.custom_model_name = normalized_name
            await session.commit()
            await session.refresh(task)
            return task

    async def stop_training(self, task_id: int):
        async with async_session_maker() as session:
            task = await session.get(ModelTrainingTask, task_id)
            if not task or task.status != "running" or not task.pid:
                return False

            try:
                parent = psutil.Process(task.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()

                _, alive = psutil.wait_procs([parent] + parent.children(), timeout=3)
                for p in alive:
                    p.kill()

                task.status = "failed"
                task.end_time = datetime.now(timezone.utc)
                task.metrics = json.dumps({"note": "Stopped by user"})
                await session.commit()
                return True
            except psutil.NoSuchProcess:
                task.status = "failed"
                await session.commit()
                return True
            except Exception as e:
                logger.error(f"Error stopping task {task_id}: {e}")
                return False

    async def delete_task(self, task_id: int):
        async with async_session_maker() as session:
            task = await session.get(ModelTrainingTask, task_id)
            if not task:
                return False

            queued = task.status == "queued"

            if task.status == "running":
                await self.stop_training(task_id)
                await session.refresh(task)

            if task.log_file_path and os.path.exists(task.log_file_path):
                try:
                    os.remove(task.log_file_path)
                except Exception:
                    pass

            if task.output_model_path and os.path.exists(task.output_model_path):
                try:
                    os.remove(task.output_model_path)
                except Exception:
                    pass

            await session.execute(
                delete(PredictionAnalysisCache).where(
                    PredictionAnalysisCache.training_task_id == task.id
                )
            )
            await session.execute(
                delete(TrainingTaskTag).where(TrainingTaskTag.task_id == task.id)
            )
            await session.delete(task)
            await session.commit()
            if queued:
                async with async_session_maker() as position_session:
                    await self._recalculate_queue_positions(position_session)
                    await position_session.commit()
                self._wake_scheduler()
            return True

    def _extract_metrics_from_log(self, log_file: Path) -> dict | None:
        if not log_file.exists():
            return None
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            metrics = {}
            patterns = {
                "mse": r"MSE:\s*([\d\.]+)",
                "rmse": r"RMSE:\s*([\d\.]+)",
                "r2": r"R-Squared:\s*([\d\.\-]+)",
                "mape": r"MAPE:\s*([\d\.]+)%",
                "smape": r"SMAPE:\s*([\d\.]+)%",
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metrics[key] = float(match.group(1))

            return metrics if metrics else None
        except Exception as e:
            logger.error(f"Error extracting metrics from log: {e}")
            return None

    async def test_model(self, task_id: int):
        return {"id": task_id, "status": "testing"}

    def _parse_progress_from_log(self, line: str, total_epochs: int, start_time: float, history: dict = None) -> dict | None:
        number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        batch_pattern = rf"Epoch\s+(\d+)/(\d+)\s+Batch\s+(\d+)/(\d+)\s+Loss=({number_pattern})"
        batch_match = re.search(batch_pattern, line)
        if batch_match:
            current_ep = int(batch_match.group(1))
            total_ep_from_log = int(batch_match.group(2))
            batch_idx = int(batch_match.group(3))
            batch_total = max(1, int(batch_match.group(4)))
            loss = float(batch_match.group(5))

            effective_total_epochs = max(total_epochs, total_ep_from_log)
            completed_units = max(0.0, (current_ep - 1) + (batch_idx / batch_total))
            progress = min(99.9, (completed_units / max(1, effective_total_epochs)) * 100.0)

            elapsed = time.time() - start_time
            if completed_units > 0:
                total_est = elapsed / completed_units * effective_total_epochs
                remaining = max(0, total_est - elapsed)
                m, s = divmod(int(remaining), 60)
                h, m = divmod(m, 60)
                eta = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            else:
                eta = "--:--"

            return {
                "progress": round(progress, 2),
                "current_epoch": current_ep,
                "total_epochs": effective_total_epochs,
                "current_loss": round(loss, 4),
                "eta": eta,
            }

        # Keep the metric label between train and validation loss optional: the
        # Earth runner annotates it as ``(area-weighted normalized MSE)``.  The
        # old expression stopped at ``Loss=...`` and therefore silently stored
        # every Earth validation value as ``null``.
        epoch_pattern = (
            rf"Epoch\s+(\d+)/(\d+)\s+Loss=({number_pattern})"
            rf"(?:\s+\([^\r\n]*?\))?(?:\s+Val Loss=({number_pattern}))?"
        )
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            current_ep = int(epoch_match.group(1))
            total_ep_from_log = int(epoch_match.group(2))
            loss = float(epoch_match.group(3))
            val_loss = float(epoch_match.group(4)) if epoch_match.group(4) else None
            effective_total_epochs = max(total_epochs, total_ep_from_log)
            progress = (current_ep / max(1, effective_total_epochs)) * 100

            if history is not None:
                # Progress can be replayed after a restart.  Fill a missing
                # validation value for an already recorded epoch instead of
                # treating the existing ``null`` as final.
                while len(history["train"]) < current_ep:
                    history["train"].append(None)
                while len(history["val"]) < current_ep:
                    history["val"].append(None)
                history["train"][current_ep - 1] = loss
                if val_loss is not None:
                    history["val"][current_ep - 1] = val_loss

            elapsed = time.time() - start_time
            if current_ep > 0:
                total_est = elapsed / current_ep * effective_total_epochs
                remaining = max(0, total_est - elapsed)
                m, s = divmod(int(remaining), 60)
                h, m = divmod(m, 60)
                eta = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            else:
                eta = "--:--"

            return {
                "progress": round(progress, 2),
                "current_epoch": current_ep,
                "total_epochs": effective_total_epochs,
                "current_loss": round(loss, 4),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "eta": eta,
                "loss_history": history,
            }
        return None
