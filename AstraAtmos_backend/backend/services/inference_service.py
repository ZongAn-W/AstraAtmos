from __future__ import annotations

import asyncio
import hashlib
import json
import torch
import numpy as np
import glob
import re
import netCDF4 as nc
from pathlib import Path

from config import MCD_DIR, MCD_RAW_3H_DIR, MOLA_TOPOGRAPHY_PATH, OPENMARS_DIR
from database.models import ModelTrainingTask
from database.engine import async_session_maker
from core.metrics import compute_error_distribution, compute_metrics, compute_test_set_metrics
from core.predict_model import PredRNNv2 as LegacyPredRNNv2
from services.training_channels import (
    extract_architecture_params,
    get_channels_from_hyperparameters,
    get_task_channel_suffix,
)
from services.prediction_horizon import validate_prediction_horizon
from services.netcdf_read_lock import netcdf_read_lock
from services.model_artifacts import is_valid_model_weight_file
from services.dataset_identity import DatasetRequestError, is_earth_training_task
from services.mars_checkpoint import (
    MARS_LEGACY_SPLIT_POLICY,
    MarsCheckpointIdentityError,
    checkpoint_normalization,
    checkpoint_state_dict,
    mars_input_channel_order,
    mars_checkpoint_identity_snapshot,
    validate_mars_checkpoint,
)
from services.mars_data_service import (
    assert_identity_current,
    prepare_scaled_volume,
    parse_channels as parse_mars_channels,
    resolve_forecast_window,
)
from services.mars_dataset_identity import has_mars_file_identity
from services.prediction_analysis_cache import PredictionAnalysisCacheService
from services.training_split import (
    LEGACY_SPLIT_RATIOS,
    normalize_split_ratios,
    split_sample_ranges,
    split_time_boundary,
)
from training_backbones.model_zoo import (
    build_forecaster,
    normalize_model_architecture,
    normalize_use_sphere,
)
from training_backbones.mola_topography import prepare_topography_grid
from training_backbones.uploaded_model_contract import (
    expand_topography_batch,
    run_uploaded_model,
    uploaded_model_requires_topography,
)


def require_mars_prediction_task(task) -> None:
    """Reject an Earth task on every Mars prediction path.

    Earth is predicted by ``services.earth_prediction_service`` from its own
    verified release and checkpoint. Letting an Earth task continue here would
    either read OpenMARS/MCD (Mars) fields for an ozone field measured in DU, or
    silently reuse the Mars prediction cache, so the rejection is explicit and
    carries a stable code instead of degrading into a 400/500.
    """
    if is_earth_training_task(task):
        raise DatasetRequestError(
            "dataset_prediction_not_supported",
            "Earth tasks must be predicted through the Earth historical prediction endpoint",
            status_code=409,
        )


class InferenceService:
    def __init__(self, analysis_cache=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = Path(__file__).parent.parent
        self.openmars_dir = self.base_dir / "data" / "openmars"
        self.mcd_dir = Path(MCD_DIR)
        self.analysis_cache = analysis_cache or PredictionAnalysisCacheService()

    @staticmethod
    def _task_split_metadata(hypers: dict) -> dict:
        """Resolve persisted split metadata without applying today's defaults to old tasks."""
        configured = {
            key: hypers[key]
            for key in ("train_ratio", "validation_ratio", "test_ratio")
            if key in hypers
        }
        if len(configured) == 3:
            try:
                ratios = normalize_split_ratios(configured)
            except ValueError:
                # Corrupt or non-numeric metadata is equivalent to missing
                # metadata for inference; keep the historical boundary explicit.
                pass
            else:
                return {
                    "ratios": ratios,
                    "source": "task_metadata",
                    "legacy_compatibility": False,
                }
        return {
            "ratios": dict(LEGACY_SPLIT_RATIOS),
            "source": "legacy_compatibility",
            "legacy_compatibility": True,
        }

    @staticmethod
    def _task_split_ratios_for_loading(hypers: dict) -> dict:
        """Return validated task ratios while preserving the loader's old default."""
        configured = {
            key: hypers[key]
            for key in ("train_ratio", "validation_ratio", "test_ratio")
            if key in hypers
        }
        if not configured:
            return {}
        return InferenceService._task_split_metadata(hypers)["ratios"]

    @staticmethod
    def _task_test_split_start(sample_count: int, split_meta: dict) -> int:
        """Return the test boundary under task metadata or legacy 80/20 ratios."""
        count = int(sample_count)
        if count <= 1:
            return 0
        try:
            return split_sample_ranges(count, split_meta["ratios"])["test"][0]
        except ValueError:
            return min(max(1, int(count * split_meta["ratios"]["train_ratio"])), count - 1)

    @classmethod
    def _with_test_set_contract(
        cls,
        metrics: dict,
        hypers: dict,
        *,
        overall_aggregation: str = "pooled_test_set_pixels",
    ) -> dict:
        result = dict(metrics)
        split_meta = cls._task_split_metadata(hypers)
        result["split_meta"] = split_meta
        result["aggregation"] = {
            "overall": overall_aggregation,
            "per_step": (
                "per_forecast_step"
                if overall_aggregation == "mean_over_forecast_steps"
                else "pooled_test_set_pixels_per_forecast_step"
            ),
        }
        return result

    def _load_task_state_dict(self, task):
        payload = self._load_task_checkpoint(task)
        state = checkpoint_state_dict(payload)
        if state is not None:
            return state
        # Legacy Mars checkpoints were bare state_dict files.
        return payload

    def _load_task_checkpoint(self, task):
        model_path = getattr(task, "output_model_path", None)
        if not is_valid_model_weight_file(model_path):
            raise ValueError("Model file not found")
        try:
            return torch.load(model_path, map_location=self.device, weights_only=True)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
            raise ValueError("Model file not found") from exc

    @staticmethod
    def _checkpoint_normalization(task) -> dict | None:
        try:
            payload = torch.load(getattr(task, "output_model_path", ""), map_location="cpu", weights_only=True)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schema"):
            validate_mars_checkpoint(payload)
        return checkpoint_normalization(payload)

    @staticmethod
    def _checkpoint_split_policy(task) -> str:
        try:
            payload = torch.load(getattr(task, "output_model_path", ""), map_location="cpu", weights_only=True)
        except Exception:
            return MARS_LEGACY_SPLIT_POLICY
        contract = payload.get("training_contract") if isinstance(payload, dict) else None
        return str(contract.get("split_policy") or MARS_LEGACY_SPLIT_POLICY) if isinstance(contract, dict) else MARS_LEGACY_SPLIT_POLICY

    @staticmethod
    def _mars_normalization_cache_key(normalization, selected_channels) -> str:
        if normalization is None:
            return "legacy_refit"
        if not isinstance(normalization, dict) or normalization.get("input_channel_order") != mars_input_channel_order(selected_channels):
            raise ValueError("Mars checkpoint input_channel_order does not match the model inputs")
        # Different trained tasks can share files/channels but have different
        # input and target statistics. Their scaled volumes must stay separate.
        encoded = json.dumps(normalization, sort_keys=True, allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def predict_task(
        self,
        task_id: int,
        ls_start: float,
        horizon: int = 3,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ) -> dict:
        task, hypers, data_dirs, temp_data_root = await self._prepare_task_prediction_context(
            task_id=task_id,
            current_user=current_user,
            data_service=data_service,
            personal_source_service=personal_source_service,
        )
        try:
            validate_prediction_horizon(hypers, horizon)
            return await self._cached_prediction(
                task=task,
                hypers=hypers,
                data_dirs=data_dirs,
                user_id=current_user.id,
                ls_start=ls_start,
                horizon=horizon,
            )
        finally:
            self._cleanup_temp_data_root(temp_data_root)

    async def task_metrics(
        self,
        task_id: int,
        ls_start: float,
        horizon: int = 3,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ) -> dict:
        result = await self.predict_task(
            task_id=task_id,
            ls_start=ls_start,
            horizon=horizon,
            current_user=current_user,
            data_service=data_service,
            personal_source_service=personal_source_service,
        )
        return result["metrics"]

    async def task_test_set_metrics(
        self,
        task_id: int,
        ls_start: float,
        horizon: int = 3,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ) -> dict:
        task, hypers, data_dirs, temp_data_root = await self._prepare_task_prediction_context(
            task_id=task_id,
            current_user=current_user,
            data_service=data_service,
            personal_source_service=personal_source_service,
        )
        try:
            validate_prediction_horizon(hypers, horizon)
            return await self._cached_test_set_metrics(
                task,
                hypers,
                data_dirs,
                current_user.id,
                horizon,
            )
        finally:
            self._cleanup_temp_data_root(temp_data_root)

    async def compare_task_test_set_metrics(
        self,
        task_ids: list[int],
        horizon: int = 3,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ) -> dict:
        unique_task_ids = self._normalize_compare_task_ids(task_ids)
        items = []
        for task_id in unique_task_ids:
            task, hypers, data_dirs, temp_data_root = await self._prepare_task_prediction_context(
                task_id=task_id,
                current_user=current_user,
                data_service=data_service,
                personal_source_service=personal_source_service,
            )
            try:
                validate_prediction_horizon(hypers, horizon)
                metrics = await self._cached_test_set_metrics(
                    task,
                    hypers,
                    data_dirs,
                    current_user.id,
                    horizon,
                )
                items.append({
                    **self._task_compare_metadata(task, hypers),
                    "metrics": metrics,
                })
            finally:
                self._cleanup_temp_data_root(temp_data_root)
        return {"items": items}

    async def task_error_distribution(
        self,
        task_id: int,
        selected_variables: list[str] | None = None,
        horizon: int = 3,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ) -> dict:
        task, hypers, data_dirs, temp_data_root = await self._prepare_task_prediction_context(
            task_id=task_id,
            current_user=current_user,
            data_service=data_service,
            personal_source_service=personal_source_service,
        )
        try:
            validate_prediction_horizon(hypers, horizon)
            return await self._cached_error_distribution(
                task,
                hypers,
                data_dirs,
                current_user.id,
                horizon,
            )
        finally:
            self._cleanup_temp_data_root(temp_data_root)

    async def compare_task_error_distributions(
        self,
        task_ids: list[int],
        horizon: int = 3,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ) -> dict:
        unique_task_ids = self._normalize_compare_task_ids(task_ids)
        items = []
        for task_id in unique_task_ids:
            task, hypers, data_dirs, temp_data_root = await self._prepare_task_prediction_context(
                task_id=task_id,
                current_user=current_user,
                data_service=data_service,
                personal_source_service=personal_source_service,
            )
            try:
                validate_prediction_horizon(hypers, horizon)
                distribution = await self._cached_error_distribution(
                    task,
                    hypers,
                    data_dirs,
                    current_user.id,
                    horizon,
                )
                items.append({
                    "task_id": int(task.id),
                    "model_name": self._task_model_name(task),
                    "distribution": distribution,
                })
            finally:
                self._cleanup_temp_data_root(temp_data_root)
        return {"items": items}

    async def task_permutation_importance(
        self,
        task_id: int,
        selected_variables: list[str],
        ls_start: float,
        horizon: int = 3,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ) -> dict:
        task, hypers, data_dirs, temp_data_root = await self._prepare_task_prediction_context(
            task_id=task_id,
            current_user=current_user,
            data_service=data_service,
            personal_source_service=personal_source_service,
        )
        try:
            validate_prediction_horizon(hypers, horizon)
            return await self._cached_permutation_importance(
                task,
                hypers,
                data_dirs,
                current_user.id,
                horizon,
                selected_variables,
            )
        finally:
            self._cleanup_temp_data_root(temp_data_root)

    async def compare_task_permutation_importance(
        self,
        task_ids: list[int],
        horizon: int = 3,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ) -> dict:
        unique_task_ids = self._normalize_compare_task_ids(task_ids)
        items = []
        for task_id in unique_task_ids:
            task, hypers, data_dirs, temp_data_root = await self._prepare_task_prediction_context(
                task_id=task_id,
                current_user=current_user,
                data_service=data_service,
                personal_source_service=personal_source_service,
            )
            try:
                validate_prediction_horizon(hypers, horizon)
                selected_variables = self._channels_to_variable_names(self._task_selected_channels(task, hypers))
                pfi = await self._cached_permutation_importance(
                    task,
                    hypers,
                    data_dirs,
                    current_user.id,
                    horizon,
                    selected_variables,
                )
                items.append({
                    "task_id": int(task.id),
                    "model_name": self._task_model_name(task),
                    **pfi,
                })
            finally:
                self._cleanup_temp_data_root(temp_data_root)
        return {"items": items}

    async def _prepare_task_prediction_context(
        self,
        task_id: int,
        current_user=None,
        data_service=None,
        personal_source_service=None,
    ):
        if getattr(current_user, "id", None) is None:
            raise PermissionError(
                "Authentication is required for trained-model analysis"
            )

        async with async_session_maker() as session:
            task = await session.get(ModelTrainingTask, task_id)
            if not task:
                raise ValueError("Training task not found")

            role = str(getattr(current_user, "role", "") or "").lower()
            user_id = getattr(current_user, "id", None)
            if role != "admin" and task.user_id is not None and task.user_id != user_id:
                raise PermissionError("No permission to access this training task")

            if task.status != "completed":
                raise ValueError("Training task is not completed")
            if not is_valid_model_weight_file(task.output_model_path):
                raise ValueError("Model file not found")

            # Earth tasks are predicted through the Earth historical endpoint.
            # This rejection happens after the access check but before any Mars
            # data preparation or cache access, so an Earth task can never reach
            # the OpenMARS/MCD loaders or the Mars prediction cache.
            require_mars_prediction_task(task)

            try:
                hypers = json.loads(task.hyperparameters or "{}")
            except Exception:
                hypers = {}

            # The task identity column wins over any legacy hyperparameter. This
            # prevents a caller from selecting a different source at prediction
            # time; the legacy marker is retained only when no snapshot exists.
            effective_dataset = str(getattr(task, "dataset_id", None) or hypers.get("training_dataset") or "openmars_mcd").strip().lower()
            hypers["training_dataset"] = effective_dataset
            checkpoint = self._load_task_checkpoint(task)
            if isinstance(checkpoint, dict) and checkpoint.get("schema"):
                try:
                    validate_mars_checkpoint(
                        checkpoint,
                        selected_channels=list(hypers.get("selected_channels") or []),
                        window=int(hypers.get("window", 3)),
                        horizon=int(hypers.get("horizon", 3)),
                    )
                except MarsCheckpointIdentityError as exc:
                    raise DatasetRequestError("dataset_version_changed", "The Mars checkpoint data identity is incomplete or inconsistent; retrain the model", status_code=409) from exc

            data_dirs, temp_data_root = await self._prepare_task_data_env(
                task=task,
                hypers=hypers,
                data_service=data_service,
                personal_source_service=personal_source_service,
                checkpoint=checkpoint,
            )
            return task, hypers, data_dirs, temp_data_root

    async def _cached_prediction(
        self,
        task,
        hypers,
        data_dirs,
        user_id,
        ls_start,
        horizon,
    ):
        from services.research_export_sources import mars_result_guard
        export_guard = mars_result_guard(task, data_dirs)
        async def compute():
            return await asyncio.to_thread(self._predict_task_with_context,
                task=task,
                hypers=hypers,
                ls_start=ls_start,
                horizon=horizon,
                data_dirs=data_dirs,
            )

        result = await self.analysis_cache.get_or_compute(
            user_id=user_id,
            task=task,
            analysis_type="prediction",
            request_params={
                "ls_start": ls_start,
                "horizon": horizon,
            },
            data_dirs=data_dirs,
            compute=compute,
        )
        if isinstance(result, dict) and isinstance(result.get("metrics"), dict):
            result = dict(result)
            result["metrics"] = self._with_test_set_contract(
                result["metrics"],
                hypers,
                overall_aggregation="mean_over_forecast_steps",
            )
        from services.research_export_sources import register_mars
        return register_mars(result, task=task, hypers=hypers, data_dirs=data_dirs,
                             user_id=user_id, analysis="prediction", horizon=horizon,
                             origin=str(float(ls_start)), variables=result.get("selected_variables", []), expected_guard=export_guard)

    async def _cached_test_set_metrics(
        self, task, hypers, data_dirs, user_id, horizon
    ):
        from services.research_export_sources import mars_result_guard
        export_guard = mars_result_guard(task, data_dirs)
        async def compute():
            if getattr(task, "model_source", "official") == "uploaded":
                return await asyncio.to_thread(self._uploaded_task_test_set_metrics,
                    task,
                    hypers,
                    horizon,
                    data_dirs,
                )
            return await asyncio.to_thread(self._official_task_test_set_metrics,
                task,
                hypers,
                horizon,
                data_dirs,
            )

        result = await self.analysis_cache.get_or_compute(
            user_id=user_id,
            task=task,
            analysis_type="metrics",
            request_params={"horizon": horizon},
            data_dirs=data_dirs,
            compute=compute,
        )
        from services.research_export_sources import register_mars
        return register_mars(self._with_test_set_contract(result, hypers), task=task, hypers=hypers,
                             data_dirs=data_dirs, user_id=user_id, analysis="metrics", horizon=horizon, expected_guard=export_guard)

    async def _cached_error_distribution(
        self, task, hypers, data_dirs, user_id, horizon
    ):
        async def compute():
            if getattr(task, "model_source", "official") == "uploaded":
                truth_raw, pred_raw, actual_horizon = await asyncio.to_thread(self._uploaded_task_test_set_arrays,
                    task,
                    hypers,
                    horizon,
                    data_dirs,
                )
            else:
                truth_raw, pred_raw, actual_horizon = await asyncio.to_thread(self._official_task_test_set_arrays,
                    task,
                    hypers,
                    horizon,
                    data_dirs,
                )
            return await asyncio.to_thread(
                compute_error_distribution,
                truth_raw[:, :actual_horizon],
                pred_raw[:, :actual_horizon],
            )

        return await self.analysis_cache.get_or_compute(
            user_id=user_id,
            task=task,
            analysis_type="error_distribution",
            request_params={"horizon": horizon},
            data_dirs=data_dirs,
            compute=compute,
        )

    async def _cached_permutation_importance(
        self,
        task,
        hypers,
        data_dirs,
        user_id,
        horizon,
        selected_variables,
    ):
        from services.research_export_sources import mars_result_guard
        export_guard = mars_result_guard(task, data_dirs)
        async def compute():
            return await asyncio.to_thread(self._task_permutation_importance_with_context,
                task=task,
                hypers=hypers,
                selected_variables=selected_variables,
                horizon=horizon,
                data_dirs=data_dirs,
            )

        result = await self.analysis_cache.get_or_compute(
            user_id=user_id,
            task=task,
            analysis_type="pfi",
            request_params={
                "horizon": horizon,
                "selected_variables": selected_variables,
            },
            data_dirs=data_dirs,
            compute=compute,
        )
        from services.research_export_sources import register_mars
        return register_mars(result, task=task, hypers=hypers, data_dirs=data_dirs, user_id=user_id,
                             analysis="pfi", horizon=horizon, variables=sorted(set(selected_variables)), expected_guard=export_guard)

    async def _prepare_task_data_env(
        self,
        task,
        hypers: dict,
        data_service=None,
        personal_source_service=None,
        checkpoint=None,
    ):
        dataset_id = str(getattr(task, "dataset_id", None) or hypers.get("training_dataset") or "openmars_mcd").strip().lower()
        if dataset_id not in ("mcd_overview", "openmars_mcd"):
            raise DatasetRequestError("dataset_prediction_not_supported", "This Mars prediction path only supports registered Mars datasets", status_code=409)
        if checkpoint is None and getattr(task, "output_model_path", None):
            checkpoint = self._load_task_checkpoint(task)
        snapshots = []
        raw_snapshot = getattr(task, "dataset_snapshot", None)
        try:
            parsed = json.loads(raw_snapshot) if isinstance(raw_snapshot, str) else raw_snapshot
        except (TypeError, ValueError):
            parsed = None
        if has_mars_file_identity(parsed):
            snapshots.append(parsed)
        if isinstance(checkpoint, dict):
            if "schema" in checkpoint or isinstance(checkpoint.get("model_state_dict"), dict) or "data_binding" in checkpoint:
                try:
                    snapshots.append(mars_checkpoint_identity_snapshot(checkpoint))
                except (ValueError, TypeError) as exc:
                    raise DatasetRequestError("dataset_version_changed", "The Mars checkpoint lacks a complete data identity; retrain the model", status_code=409) from exc
            elif has_mars_file_identity(checkpoint.get("dataset_identity")):
                snapshots.append(checkpoint["dataset_identity"])
        if snapshots:
            for snapshot in snapshots:
                if snapshot.get("dataset_id") != dataset_id:
                    raise DatasetRequestError("dataset_version_changed", "The Mars data identity does not match the training task", status_code=409)
                task_fingerprint = getattr(task, "dataset_fingerprint", None)
                if task_fingerprint and task_fingerprint != (snapshot.get("dataset_fingerprint") or snapshot.get("file_fingerprint")):
                    raise DatasetRequestError("dataset_version_changed", "The Mars task fingerprint disagrees with its saved data identity", status_code=409)
                assert_identity_current(
                    snapshot,
                    openmars_dir=self.openmars_dir,
                    mcd_dir=self.mcd_dir,
                    raw_dir=MCD_RAW_3H_DIR,
                )
            hypers["_dataset_identity_status"] = "verified"
        elif getattr(task, "dataset_fingerprint", None) or getattr(task, "dataset_identity_status", None) == "verified":
            raise DatasetRequestError("dataset_version_changed", "The verified Mars task is missing its data identity; retrain the model", status_code=409)
        else:
            hypers["_dataset_identity_status"] = "legacy"
        if dataset_id == "mcd_overview":
            return {"MCD_RAW_3H_DIR": str(MCD_RAW_3H_DIR)}, None
        return {
            "ASTRAATMOS_OPENMARS_DIR": str(self.openmars_dir),
            "ASTRAATMOS_MCD_DIR": str(self.mcd_dir),
        }, None

    @staticmethod
    def _cleanup_temp_data_root(temp_data_root: Path | None) -> None:
        if temp_data_root is None:
            return
        from services.training_service import TrainingService

        TrainingService().cleanup_temp_data_root(temp_data_root)

    @staticmethod
    def _normalize_compare_task_ids(task_ids: list[int]) -> list[int]:
        unique = []
        seen = set()
        for raw_id in task_ids or []:
            try:
                task_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if task_id <= 0 or task_id in seen:
                continue
            seen.add(task_id)
            unique.append(task_id)
        if len(unique) < 2:
            raise ValueError("At least two training tasks are required for comparison")
        return unique

    @staticmethod
    def _task_model_name(task) -> str:
        return getattr(task, "custom_model_name", None) or f"Task #{getattr(task, 'id', '')}"

    def _task_selected_channels(self, task, hypers: dict) -> list[str]:
        channels = get_channels_from_hyperparameters(hypers)
        if channels:
            return channels
        return [channel for channel in get_task_channel_suffix(task) if channel]

    def _task_compare_metadata(self, task, hypers: dict) -> dict:
        model_source = str(getattr(task, "model_source", None) or hypers.get("model_source") or "official").lower()
        selected_channels = self._task_selected_channels(task, hypers)
        architecture = "uploaded" if model_source == "uploaded" else normalize_model_architecture(
            hypers.get("model_architecture", "predrnnv2")
        )
        data_source = str(hypers.get("_effective_data_source") or hypers.get("_data_source") or "default").lower()
        safe_keys = (
            "window",
            "horizon",
            "epochs",
            "batch_size",
            "learning_rate",
            "early_stopping_patience",
            "seed",
            "use_sphere",
            "model_architecture",
        )
        safe_hypers = {key: hypers[key] for key in safe_keys if key in hypers}
        safe_hypers.update({
            "model_source": model_source,
            "selected_channels": selected_channels,
            "_data_source": data_source,
        })
        return {
            "task_id": int(task.id),
            "model_name": self._task_model_name(task),
            "model_source": model_source,
            "architecture": architecture,
            "selected_channels": selected_channels,
            "hyperparameters": safe_hypers,
        }

    def _predict_task_with_context(
        self,
        task,
        hypers: dict,
        ls_start: float,
        horizon: int,
        data_dirs: dict[str, str] | None = None,
    ) -> dict:
        checkpoint_payload = self._load_task_checkpoint(task)
        legacy_normalization = not bool(checkpoint_normalization(checkpoint_payload))
        if getattr(task, "model_source", "official") == "uploaded":
            pred_scaled, truth_scaled, y_mean, y_std, selected_channels, input_ls, target_ls, window_meta, latitude, longitude = self._predict_uploaded_task_window(
                task=task,
                hypers=hypers,
                ls_start=ls_start,
                horizon=horizon,
                data_dirs=data_dirs,
            )
            model_info = {
                "model_source": "trained_task",
                "training_task_id": task.id,
                "training_model_source": "uploaded",
                "model_name": task.custom_model_name,
                "selected_channels": selected_channels,
                "weight_file": Path(task.output_model_path).name,
                "normalization_source": "legacy_refit" if legacy_normalization else "checkpoint",
                "legacy_compatibility": legacy_normalization,
            }
        else:
            pred_scaled, truth_scaled, y_mean, y_std, selected_channels, input_ls, target_ls, window_meta, latitude, longitude = self._predict_official_task_window(
                task=task,
                hypers=hypers,
                ls_start=ls_start,
                horizon=horizon,
                data_dirs=data_dirs,
            )
            model_info = {
                "model_source": "trained_task",
                "training_task_id": task.id,
                "training_model_source": "official",
                "model_name": task.custom_model_name,
                "architecture": normalize_model_architecture(hypers.get("model_architecture", "predrnnv2")),
                "selected_channels": selected_channels,
                "weight_file": Path(task.output_model_path).name,
                "normalization_source": "legacy_refit" if legacy_normalization else "checkpoint",
                "legacy_compatibility": legacy_normalization,
            }

        actual_horizon = min(
            int(horizon),
            int(pred_scaled.shape[0]),
            int(truth_scaled.shape[0]),
            len(target_ls),
        )
        if actual_horizon <= 0:
            raise ValueError("Mars prediction returned no complete target Ls values")
        pred_raw = pred_scaled[:actual_horizon] * (y_std + 1e-6) + y_mean
        truth_raw = truth_scaled[:actual_horizon] * (y_std + 1e-6) + y_mean
        if pred_raw.shape != truth_raw.shape or pred_raw.ndim != 3:
            raise ValueError("Mars prediction and ground truth spatial shapes must match")
        lat_arr, lon_arr = self._prediction_coordinates(latitude, longitude, pred_raw.shape[-2:])
        residual_raw = pred_raw - truth_raw
        metrics = self._with_test_set_contract(
            compute_metrics(truth_raw, pred_raw),
            hypers,
            overall_aggregation="mean_over_forecast_steps",
        )

        input_ls_values = [float(value) for value in input_ls]
        ls_values = [float(value) for value in target_ls[:actual_horizon]]
        selected_variables = self._channels_to_variable_names(selected_channels)
        model_info["requested_ls_start"] = float(ls_start)
        model_info.update({key: value for key, value in window_meta.items() if value is not None})

        return {
            "ground_truth": self._fields_to_dicts(truth_raw, lat_arr, lon_arr, include_points=False),
            "prediction": self._fields_to_dicts(pred_raw, lat_arr, lon_arr, include_points=False),
            "residual": self._fields_to_dicts(residual_raw, lat_arr, lon_arr, include_points=False),
            "selected_variables": selected_variables,
            "horizon": actual_horizon,
            "input_ls_values": input_ls_values,
            "ls_values": ls_values,
            "model_info": model_info,
            "metrics": metrics,
            "source_meta": {
                "requested_source": "training_task",
                "effective_source": "training_task",
                "fallback": False,
                "dataset_identity_status": hypers.get("_dataset_identity_status", "legacy"),
                "message": (
                    "legacy dataset identity; source snapshot was unavailable"
                    if hypers.get("_dataset_identity_status") == "legacy"
                    else None
                ),
            },
        }

    def _predict_official_task_window(self, task, hypers: dict, ls_start: float, horizon: int, data_dirs=None):
        window = int(hypers.get("window", 3))
        task_horizon = int(hypers.get("horizon", horizon or 3))
        hidden_dims = hypers.get("stlstm_hidden_dims", [64, 64, 64])
        model_architecture = normalize_model_architecture(hypers.get("model_architecture", "predrnnv2"))
        use_sphere = normalize_use_sphere(hypers)
        architecture_params = extract_architecture_params(hypers)
        active_vars = get_task_channel_suffix(task)
        mcd_vars_map = {
            'U': ('U_Wind', 'u'),
            'V': ('V_Wind', 'v'),
            'D': ('Dust_Optical_Depth', 'dustq'),
            'S': ('Solar_Flux_DN', 'fluxsurf_dn_sw'),
            'T': ('Temperature', 'temp')
        }
        used_mcd_vars = [mcd_vars_map[channel] for channel in active_vars if channel in mcd_vars_map]
        checkpoint_state = self._load_task_state_dict(task)
        legacy_checkpoint = self._is_legacy_official_state_dict(checkpoint_state)
        if legacy_checkpoint:
            x_torch, y_torch, ls_torch, y_mean, y_std = self._prepare_data(
                used_mcd_vars, window, task_horizon, data_dirs=data_dirs,
                split_policy=MARS_LEGACY_SPLIT_POLICY,
            )
            if ls_torch is None:
                raise ValueError("Mars prediction requires an Ls timeline")
            metadata = getattr(self, "_last_prepared_window_metadata", None)
            latitude = None if metadata is None else metadata.get("latitude")
            longitude = None if metadata is None else metadata.get("longitude")
            if metadata and metadata.get("ls_values") is not None:
                window_info = resolve_forecast_window(
                    metadata["ls_values"], ls_start, window, task_horizon,
                    candidate_starts=metadata["sample_starts"],
                    candidate_mars_years=metadata["sample_mars_years"],
                    candidate_blocks=metadata["blocks"],
                )
                sample_idx = int(np.flatnonzero(metadata["sample_starts"] == window_info["sample_index"])[0])
                target_ls = window_info["target_ls"]
            elif metadata and metadata.get("input_ls") is not None:
                sequence_starts = np.asarray(metadata["input_ls"], dtype=np.float32)[:, 0]
                window_info = resolve_forecast_window(
                    sequence_starts,
                    ls_start,
                    1,
                    1,
                    candidate_starts=np.arange(len(sequence_starts), dtype=np.int64),
                    candidate_mars_years=metadata.get("sample_mars_years"),
                )
                sample_idx = window_info["sample_index"]
                target_ls = np.asarray(metadata["target_ls"])[sample_idx].astype(float).tolist()
                window_info["input_ls"] = np.asarray(metadata["input_ls"])[sample_idx].astype(float).tolist()
                window_info["target_ls"] = target_ls
            else:
                # A legacy test double may not expose the shared metadata.
                # Preserve its historical nearest-start behavior explicitly.
                ls_rows = ls_torch.detach().cpu().numpy()
                timeline = np.concatenate([ls_rows[0], ls_rows[1:, -1]])
                if task_horizon > 1:
                    timeline = np.concatenate(
                        [timeline, np.repeat(timeline[-1:], task_horizon - 1)]
                    )
                window_info = resolve_forecast_window(
                    timeline,
                    ls_start,
                    window,
                    task_horizon,
                )
                sample_idx = window_info["sample_index"]
                window_info = {
                    "input_ls": ls_torch[sample_idx].detach().cpu().numpy().astype(float).tolist(),
                    "target_ls": timeline[
                        sample_idx + window : sample_idx + window + task_horizon
                    ].astype(float).tolist(),
                }
                target_ls = window_info["target_ls"]
            x_sample = x_torch[sample_idx:sample_idx + 1].to(self.device)
            ls_sample = ls_torch[sample_idx:sample_idx + 1].to(self.device)
            truth_sample = y_torch[sample_idx, :, 0]
            data_height, data_width = int(x_torch.shape[-2]), int(x_torch.shape[-1])
        else:
            # New checkpoints use the shared continuous-volume preparation.
            volume = self._load_official_task_volume(
                used_mcd_vars, window, task_horizon, data_dirs=data_dirs,
                split_ratios=self._task_split_ratios_for_loading(hypers),
                normalization=self._checkpoint_normalization(task),
                split_policy=self._checkpoint_split_policy(task),
            )
            window_info = resolve_forecast_window(
                volume.ls,
                ls_start,
                window,
                task_horizon,
                candidate_starts=volume.sample_starts,
                candidate_mars_years=volume.sample_mars_years,
                candidate_blocks=volume.blocks,
            )
            sample_idx = window_info["sample_index"]
            x_sample = self._window_slice(volume.values, sample_idx, window).unsqueeze(0).to(self.device)
            ls_sample = self._window_slice(
                np.asarray(volume.ls, dtype=np.float32).reshape(-1, 1), sample_idx, window
            ).unsqueeze(0).transpose(1, 2).to(self.device)
            truth_sample = self._window_slice(volume.y_scaled, sample_idx + window, task_horizon)
            y_mean, y_std = volume.y_mean, volume.y_std
            data_height, data_width = int(volume.values.shape[1]), int(volume.values.shape[2])
            latitude, longitude = volume.latitude, volume.longitude

        latitude, longitude = self._prediction_coordinates(latitude, longitude, (data_height, data_width))

        model, uses_legacy_loader = self._load_official_task_model(
            task=task,
            model_architecture=model_architecture,
            input_channels=1 + len(used_mcd_vars),
            selected_channels=list(active_vars),
            hidden_dims=hidden_dims,
            height=data_height,
            width=data_width,
            window=window,
            horizon=task_horizon,
            use_sphere=use_sphere,
            architecture_params=architecture_params,
        )

        with torch.no_grad():
            pred = self._run_official_task_model(
                model,
                x_sample,
                ls_sample,
                uses_legacy_loader,
            )[0, :, 0].cpu().numpy()
        # 官方模型输出为 (horizon, H, W)（单通道不再保留通道维），
        # 与 `y_torch[idx, :, 0]` 的原始语义一致。
        truth = truth_sample.squeeze(1).numpy() if truth_sample.ndim == 4 else truth_sample.numpy()
        return (
            pred,
            truth,
            y_mean,
            y_std,
            list(active_vars),
            window_info["input_ls"],
            window_info["target_ls"],
            {
                "sample_index": int(window_info.get("sample_index", sample_idx)),
                "mars_year": window_info.get("mars_year"),
                "block_index": window_info.get("block_index"),
            },
            latitude,
            longitude,
        )

    @staticmethod
    def _window_slice(volume: np.ndarray, start: int, length: int) -> torch.Tensor:
        """从连续体积切出 ``length`` 步窗口，并转换为训练时使用的张量布局。

        支持与 :meth:`_window_stack` 相同的三种输入布局：

        - ``[time, height, width]`` / ``[time, height, width, channels]``
          -> ``[length, channels, height, width]``（单通道目标为 ``[length, 1, h, w]``）
        - ``[time, features]``（如 Ls）-> ``[length, features]``

        与逐样本滑窗展开后的切片逐元素一致。
        """
        end = min(int(start) + int(length), int(volume.shape[0]))
        window = np.ascontiguousarray(volume[int(start):end])
        tensor = torch.from_numpy(window).float()
        if window.ndim == 2:
            return tensor
        if window.ndim == 3:
            window = window[..., None]
        return torch.from_numpy(np.ascontiguousarray(window)).permute(0, 3, 1, 2).float()

    @staticmethod
    def _window_stack(volume: np.ndarray, start: int, count: int, length: int) -> torch.Tensor:
        """把 ``count`` 个连续滑窗一次性转成张量。

        支持两种输入布局，输出与 ``prepare_tensors`` 逐样本展开后的切片堆叠一致：

        - ``[time, height, width]`` / ``[time, height, width, channels]``
          -> ``[count, length, channels, height, width]``
        - ``[time, features]``（如 Ls）-> ``[count, length, features]``；
          单列特征降为 ``[count, length]``，与 ``ls_seq`` 的展开结果一致。
        """
        if count <= 0:
            raise ValueError("window count must be positive")
        starts = np.arange(int(start), int(start) + int(count), dtype=np.int64)
        return InferenceService._window_stack_at_starts(volume, starts, length)
    @staticmethod
    def _window_stack_at_starts(volume: np.ndarray, starts: np.ndarray, length: int) -> torch.Tensor:
        """Build windows at explicit starts on the merged chronological timeline."""
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        if starts.size == 0:
            return torch.empty(0)
        windows = [InferenceService._window_slice(volume, int(start), int(length)) for start in starts]
        result = torch.stack(windows, dim=0)
        if np.asarray(volume).ndim == 2 and result.shape[-1] == 1:
            return result[..., 0]
        return result

    def _uploaded_task_test_windows(self, volume, window: int, horizon: int, split_ratios=None):
        """按任务保存的时间顺序划分取测试分区滑窗。

        样本数口径与 ``prepare_tensors`` 的逐样本展开相同：
        ``sample_count = total_time - window - horizon + 1``，
        使用任务保存的完整三项比例；没有完整比例元数据的旧任务继续使用 80/0/20。
        """
        sample_starts = getattr(volume, "sample_starts", None)
        if sample_starts is None:
            sample_starts = np.arange(
                int(volume.values.shape[0]) - int(window) - int(horizon) + 1,
                dtype=np.int64,
            )
        sample_starts = np.asarray(sample_starts, dtype=np.int64)
        sample_count = int(sample_starts.size)
        if sample_count <= 0:
            raise ValueError("Not enough time steps to build the requested windows")
        persisted_starts = getattr(volume, "split_window_starts", None)
        if persisted_starts and persisted_starts.get("test"):
            selected_starts = np.asarray(persisted_starts["test"], dtype=np.int64)
        else:
            split_meta = self._task_split_metadata(split_ratios or {})
            test_start = self._task_test_split_start(sample_count, split_meta)
            selected_starts = sample_starts[test_start:]
        if selected_starts.size <= 0:
            return torch.empty(0), torch.empty(0), None
        x_test = self._window_stack_at_starts(volume.values, selected_starts, window)
        y_test = self._window_stack_at_starts(volume.y_scaled, selected_starts + window, horizon)
        ls_test = (
            None
            if volume.ls is None
            else self._window_stack_at_starts(
                np.asarray(volume.ls, dtype=np.float32).reshape(-1, 1),
                selected_starts,
                window,
            )
        )
        return x_test, y_test, ls_test

    @staticmethod
    def _uploaded_task_split_ratios(hypers: dict) -> dict[str, float]:
        """Return the ratios used to fit an uploaded task's standardization."""
        return InferenceService._task_split_metadata(hypers)["ratios"]

    def _prepare_uploaded_task_volume(
        self,
        hypers: dict,
        window: int,
        horizon: int,
        data_dirs=None,
        normalization=None,
    ):
        """取上传模型所需的标准化体积：优先命中进程内缓存。"""
        from services.prediction_volume_cache import (
            ScaledVolume,
            get_scaled_volume,
            put_scaled_volume,
            volume_signature,
        )
        from training_backbones.user_model_runner import (
            parse_selected_channels,
            prepare_tensors,
        )

        directories = data_dirs or {}
        selected_channels = parse_selected_channels(hypers.get("selected_channels", []))
        training_dataset = hypers.get("training_dataset", "openmars_mcd")
        openmars_dir = Path(directories.get("ASTRAATMOS_OPENMARS_DIR") or self.openmars_dir)
        mcd_dir = Path(directories.get("ASTRAATMOS_MCD_DIR") or self.mcd_dir)
        mcd_overview_dir = Path(
            directories.get("MCD_RAW_3H_DIR")
            or directories.get("ASTRAATMOS_MCD_RAW_3H_DIR")
            or MCD_RAW_3H_DIR
        )
        split_ratios = self._uploaded_task_split_ratios(hypers)

        def load():
            return prepare_tensors(
                openmars_dir,
                mcd_dir,
                selected_channels,
                window,
                horizon,
                training_dataset=training_dataset,
                mcd_overview_dir=mcd_overview_dir,
                return_ls=True,
                return_coordinates=True,
                require_coordinates=False,
                return_scaled_volume=True,
                normalization=normalization,
                train_ratio=split_ratios["train_ratio"],
                validation_ratio=split_ratios["validation_ratio"],
                test_ratio=split_ratios["test_ratio"],
            )

        signature = volume_signature(
            openmars_dir=openmars_dir,
            mcd_dir=mcd_dir,
            selected_channels=selected_channels,
            training_dataset=str(training_dataset),
            mcd_overview_dir=mcd_overview_dir,
            cache_prefix="uploaded:" + self._mars_normalization_cache_key(normalization, selected_channels),
            split_ratios=split_ratios,
            window=window,
            horizon=horizon,
        )
        cached = get_scaled_volume(signature) if not data_dirs else None
        if not isinstance(cached, ScaledVolume):
            cached = load()
            if not data_dirs:
                put_scaled_volume(signature, cached)
        # ``prepare_tensors`` applies checkpoint statistics while constructing
        # the volume; never refit or transform it a second time here.
        return selected_channels, cached

    def _prepare_uploaded_topography(self, model, latitude, longitude):
        if not uploaded_model_requires_topography(model):
            return None
        return prepare_topography_grid(
            latitude,
            longitude,
            asset_path=MOLA_TOPOGRAPHY_PATH,
        ).to(self.device)

    def _run_uploaded_task_model(
        self,
        model,
        x_batch,
        ls_batch,
        topography_grid,
        context,
    ):
        x_device = x_batch.to(self.device)
        topography_batch = None
        if uploaded_model_requires_topography(model):
            topography_batch = expand_topography_batch(
                x_device,
                topography_grid,
                context,
            )
        return run_uploaded_model(
            model,
            x_device,
            ls=ls_batch.to(self.device) if ls_batch is not None else None,
            topography=topography_batch,
            context=context,
        )

    def _predict_uploaded_task_window(self, task, hypers: dict, ls_start: float, horizon: int, data_dirs=None):
        from training_backbones.user_model_runner import (
            assert_prediction_shape,
            build_uploaded_model_config,
            load_uploaded_model,
        )

        window = int(hypers.get("window", 3))
        task_horizon = int(hypers.get("horizon", horizon or 3))
        selected_channels, volume = self._prepare_uploaded_task_volume(
            hypers,
            window,
            task_horizon,
            data_dirs,
            normalization=self._checkpoint_normalization(task),
        )
        window_info = resolve_forecast_window(
            volume.ls,
            ls_start,
            window,
            task_horizon,
            candidate_starts=volume.sample_starts,
            candidate_mars_years=volume.sample_mars_years,
            candidate_blocks=volume.blocks,
        )
        sample_idx = window_info["sample_index"]
        latitude, longitude = self._prediction_coordinates(volume.latitude, volume.longitude, (volume.height, volume.width))
        channel_count = int(volume.values.shape[-1])
        config = build_uploaded_model_config(
            in_channels=channel_count,
            window=window,
            horizon=task_horizon,
            height=volume.height,
            width=volume.width,
            selected_channels=selected_channels,
            custom_model_params=hypers.get("custom_model_params", {}),
            param_schema=hypers.get("_uploaded_model_param_schema", {}),
        )
        model_path = hypers.get("_uploaded_model_path")
        if not model_path:
            raise ValueError("Uploaded model path is missing from task hyperparameters")
        model = load_uploaded_model(Path(model_path), config).to(self.device)
        state_dict = self._load_task_state_dict(task)
        model.load_state_dict(state_dict)
        model.eval()
        topography_grid = self._prepare_uploaded_topography(
            model,
            volume.latitude,
            volume.longitude,
        )

        x_window = self._window_slice(volume.values, sample_idx, window)
        x_sample = x_window.unsqueeze(0).to(self.device)
        if volume.ls is not None:
            ls_window = self._window_slice(
                np.asarray(volume.ls, dtype=np.float32).reshape(-1, 1), sample_idx, window
            ).unsqueeze(0).transpose(1, 2)
            ls_sample = ls_window.squeeze(1).to(self.device)
        else:
            ls_sample = None
        truth_tensor = self._window_slice(
            volume.y_scaled, sample_idx + window, task_horizon
        ).unsqueeze(0)
        with torch.no_grad():
            pred_tensor = self._run_uploaded_task_model(
                model,
                x_sample,
                ls_sample,
                topography_grid,
                "uploaded prediction",
            )
            assert_prediction_shape(pred_tensor, truth_tensor.to(self.device), "uploaded prediction")
        pred = pred_tensor[0, :, 0].cpu().numpy()
        truth = truth_tensor[0, :, 0].cpu().numpy()
        return (
            pred,
            truth,
            volume.y_mean,
            volume.y_std,
            selected_channels,
            window_info["input_ls"],
            window_info["target_ls"],
            {
                "sample_index": int(sample_idx),
                "mars_year": window_info.get("mars_year"),
                "block_index": window_info.get("block_index"),
            },
            latitude,
            longitude,
        )

    def _official_task_test_set_metrics(self, task, hypers: dict, horizon: int, data_dirs=None):
        truth_raw, pred_raw, actual_horizon = self._official_task_test_set_arrays(
            task,
            hypers,
            horizon,
            data_dirs=data_dirs,
        )
        return self._with_test_set_contract(
            compute_test_set_metrics(truth_raw, pred_raw, horizon=actual_horizon),
            hypers,
        )

    def _official_task_test_set_arrays(self, task, hypers: dict, horizon: int, data_dirs=None):
        window = int(hypers.get("window", 3))
        task_horizon = int(hypers.get("horizon", horizon or 3))
        hidden_dims = hypers.get("stlstm_hidden_dims", [64, 64, 64])
        model_architecture = normalize_model_architecture(hypers.get("model_architecture", "predrnnv2"))
        use_sphere = normalize_use_sphere(hypers)
        architecture_params = extract_architecture_params(hypers)
        active_vars = get_task_channel_suffix(task)
        mcd_vars_map = {
            'U': ('U_Wind', 'u'),
            'V': ('V_Wind', 'v'),
            'D': ('Dust_Optical_Depth', 'dustq'),
            'S': ('Solar_Flux_DN', 'fluxsurf_dn_sw'),
            'T': ('Temperature', 'temp')
        }
        used_mcd_vars = [mcd_vars_map[channel] for channel in active_vars if channel in mcd_vars_map]
        split_ratios = self._task_split_ratios_for_loading(hypers)
        prepare_kwargs = {"data_dirs": data_dirs}
        if split_ratios:
            prepare_kwargs["split_ratios"] = split_ratios
        checkpoint_norm = self._checkpoint_normalization(task)
        if checkpoint_norm:
            prepare_kwargs["normalization"] = checkpoint_norm
        x_torch, y_torch, ls_torch, y_mean, y_std = self._prepare_data(
            used_mcd_vars,
            window,
            task_horizon,
            **prepare_kwargs,
        )

        model, uses_legacy_loader = self._load_official_task_model(
            task=task,
            model_architecture=model_architecture,
            input_channels=1 + len(used_mcd_vars),
            selected_channels=list(active_vars),
            hidden_dims=hidden_dims,
            height=int(x_torch.shape[-2]),
            width=int(x_torch.shape[-1]),
            window=window,
            horizon=task_horizon,
            use_sphere=use_sphere,
            architecture_params=architecture_params,
        )

        split_meta = self._task_split_metadata(hypers)
        split_window_starts = getattr(self, "_last_prepared_window_metadata", {}).get("split_window_starts")
        if split_window_starts and split_window_starts.get("test"):
            positions = {
                int(start): index
                for index, start in enumerate(np.asarray(getattr(self, "_last_prepared_window_metadata", {}).get("sample_starts", ()), dtype=np.int64).tolist())
            }
            test_indices = [positions[int(start)] for start in split_window_starts["test"] if int(start) in positions]
            x_test = x_torch[test_indices]
            y_test = y_torch[test_indices]
            ls_test = ls_torch[test_indices]
        else:
            split = self._task_test_split_start(len(x_torch), split_meta)
            x_test = x_torch[split:]
            y_test = y_torch[split:]
            ls_test = ls_torch[split:]
        if len(x_test) == 0:
            raise ValueError("No trained model test samples are available")

        pred_batches = []
        truth_batches = []
        batch_size = max(1, min(16, int(hypers.get("batch_size", 16) or 16)))
        actual_horizon = min(int(horizon), int(task_horizon))
        with torch.no_grad():
            for start in range(0, len(x_test), batch_size):
                end = min(start + batch_size, len(x_test))
                pred = self._run_official_task_model(
                    model,
                    x_test[start:end].to(self.device),
                    ls_test[start:end].to(self.device),
                    uses_legacy_loader,
                ).cpu().numpy()
                pred_batches.append(pred[:, :actual_horizon, 0] * (y_std + 1e-6) + y_mean)
                truth_batches.append(
                    y_test[start:end, :actual_horizon, 0].cpu().numpy() * (y_std + 1e-6) + y_mean
                )

        return (
            np.concatenate(truth_batches, axis=0),
            np.concatenate(pred_batches, axis=0),
            actual_horizon,
        )

    def _uploaded_task_test_set_metrics(self, task, hypers: dict, horizon: int, data_dirs=None):
        truth_raw, pred_raw, actual_horizon = self._uploaded_task_test_set_arrays(
            task,
            hypers,
            horizon,
            data_dirs=data_dirs,
        )
        return self._with_test_set_contract(
            compute_test_set_metrics(truth_raw, pred_raw, horizon=actual_horizon),
            hypers,
        )

    def _uploaded_task_test_set_arrays(self, task, hypers: dict, horizon: int, data_dirs=None):
        from training_backbones.user_model_runner import (
            assert_prediction_shape,
            build_uploaded_model_config,
            load_uploaded_model,
        )

        window = int(hypers.get("window", 3))
        task_horizon = int(hypers.get("horizon", horizon or 3))
        selected_channels, volume = self._prepare_uploaded_task_volume(
            hypers,
            window,
            task_horizon,
            data_dirs,
            normalization=self._checkpoint_normalization(task),
        )
        config = build_uploaded_model_config(
            in_channels=int(volume.values.shape[-1]),
            window=window,
            horizon=task_horizon,
            height=volume.height,
            width=volume.width,
            selected_channels=selected_channels,
            custom_model_params=hypers.get("custom_model_params", {}),
            param_schema=hypers.get("_uploaded_model_param_schema", {}),
        )
        model_path = hypers.get("_uploaded_model_path")
        if not model_path:
            raise ValueError("Uploaded model path is missing from task hyperparameters")
        model = load_uploaded_model(Path(model_path), config).to(self.device)
        state_dict = self._load_task_state_dict(task)
        model.load_state_dict(state_dict)
        model.eval()
        topography_grid = self._prepare_uploaded_topography(
            model,
            volume.latitude,
            volume.longitude,
        )

        x_test, y_test, ls_test = self._uploaded_task_test_windows(
            volume, window, task_horizon,
            {key: hypers[key] for key in ("train_ratio", "validation_ratio", "test_ratio") if key in hypers},
        )
        if len(x_test) == 0:
            raise ValueError("No uploaded model test samples are available")

        pred_batches = []
        truth_batches = []
        batch_size = max(1, min(16, int(hypers.get("batch_size", 16) or 16)))
        actual_horizon = min(int(horizon), int(task_horizon))
        with torch.no_grad():
            for start in range(0, len(x_test), batch_size):
                end = min(start + batch_size, len(x_test))
                truth_tensor = y_test[start:end].to(self.device)
                ls_batch = ls_test[start:end].to(self.device) if ls_test is not None else None
                pred = self._run_uploaded_task_model(
                    model,
                    x_test[start:end],
                    ls_batch,
                    topography_grid,
                    "uploaded test-set metrics",
                )
                assert_prediction_shape(pred, truth_tensor, "uploaded test-set metrics")
                pred_np = pred.cpu().numpy()
                pred_batches.append(pred_np[:, :actual_horizon, 0] * (volume.y_std + 1e-6) + volume.y_mean)
                truth_batches.append(
                    truth_tensor[:, :actual_horizon, 0].cpu().numpy() * (volume.y_std + 1e-6) + volume.y_mean
                )

        return (
            np.concatenate(truth_batches, axis=0),
            np.concatenate(pred_batches, axis=0),
            actual_horizon,
        )

    def _task_permutation_importance_with_context(
        self,
        task,
        hypers: dict,
        selected_variables: list[str],
        horizon: int,
        data_dirs=None,
    ) -> dict:
        if getattr(task, "model_source", "official") == "uploaded":
            return self._uploaded_task_permutation_importance(
                task=task,
                hypers=hypers,
                selected_variables=selected_variables,
                horizon=horizon,
                data_dirs=data_dirs,
            )
        return self._official_task_permutation_importance(
            task=task,
            hypers=hypers,
            selected_variables=selected_variables,
            horizon=horizon,
            data_dirs=data_dirs,
        )

    def _official_task_permutation_importance(self, task, hypers: dict, selected_variables: list[str], horizon: int, data_dirs=None):
        from sklearn.metrics import r2_score

        window = int(hypers.get("window", 3))
        task_horizon = int(hypers.get("horizon", horizon or 3))
        hidden_dims = hypers.get("stlstm_hidden_dims", [64, 64, 64])
        model_architecture = normalize_model_architecture(hypers.get("model_architecture", "predrnnv2"))
        use_sphere = normalize_use_sphere(hypers)
        architecture_params = extract_architecture_params(hypers)
        active_channels = list(get_task_channel_suffix(task))
        mcd_vars_map = {
            'U': ('U_Wind', 'u'),
            'V': ('V_Wind', 'v'),
            'D': ('Dust_Optical_Depth', 'dustq'),
            'S': ('Solar_Flux_DN', 'fluxsurf_dn_sw'),
            'T': ('Temperature', 'temp')
        }
        used_mcd_vars = [mcd_vars_map[channel] for channel in active_channels if channel in mcd_vars_map]
        split_ratios = self._task_split_ratios_for_loading(hypers)
        prepare_kwargs = {"data_dirs": data_dirs}
        if split_ratios:
            prepare_kwargs["split_ratios"] = split_ratios
        checkpoint_norm = self._checkpoint_normalization(task)
        if checkpoint_norm:
            prepare_kwargs["normalization"] = checkpoint_norm
        x_torch, y_torch, ls_torch, y_mean, y_std = self._prepare_data(
            used_mcd_vars,
            window,
            task_horizon,
            **prepare_kwargs,
        )
        model, uses_legacy_loader = self._load_official_task_model(
            task=task,
            model_architecture=model_architecture,
            input_channels=1 + len(used_mcd_vars),
            selected_channels=active_channels,
            hidden_dims=hidden_dims,
            height=int(x_torch.shape[-2]),
            width=int(x_torch.shape[-1]),
            window=window,
            horizon=task_horizon,
            use_sphere=use_sphere,
            architecture_params=architecture_params,
        )

        split_meta = self._task_split_metadata(hypers)
        split_window_starts = getattr(self, "_last_prepared_window_metadata", {}).get("split_window_starts")
        if split_window_starts and split_window_starts.get("test"):
            positions = {
                int(start): index
                for index, start in enumerate(np.asarray(getattr(self, "_last_prepared_window_metadata", {}).get("sample_starts", ()), dtype=np.int64).tolist())
            }
            test_indices = [positions[int(start)] for start in split_window_starts["test"] if int(start) in positions]
            x_test = x_torch[test_indices].clone()
            y_test = y_torch[test_indices]
            ls_test = ls_torch[test_indices]
        else:
            split = self._task_test_split_start(len(x_torch), split_meta)
            x_test = x_torch[split:].clone()
            y_test = y_torch[split:]
            ls_test = ls_torch[split:]
        if len(x_test) == 0:
            return {"items": [], "baseline_metric": "r2", "baseline_value": 0.0}

        sample_size = min(40, len(x_test))
        sample_indices = np.linspace(0, len(x_test) - 1, sample_size, dtype=int)
        x_sample = x_test[sample_indices]
        y_sample = y_test[sample_indices]
        ls_sample = ls_test[sample_indices]

        def score(batch):
            with torch.no_grad():
                pred = self._run_official_task_model(
                    model,
                    batch.to(self.device),
                    ls_sample.to(self.device),
                    uses_legacy_loader,
                ).cpu().numpy()
            truth = y_sample.numpy()
            pred_raw = pred[:, :horizon, 0] * (y_std + 1e-6) + y_mean
            truth_raw = truth[:, :horizon, 0] * (y_std + 1e-6) + y_mean
            valid = np.isfinite(truth_raw) & np.isfinite(pred_raw)
            if np.count_nonzero(valid) < 10:
                return 0.0
            return float(r2_score(truth_raw[valid], pred_raw[valid]))

        baseline = score(x_sample)
        name_by_channel = {
            "U": "U_Wind",
            "V": "V_Wind",
            "D": "Dust_Optical_Depth",
            "S": "Solar_Flux_DN",
            "T": "Temperature",
        }
        selected_set = set(selected_variables or self._channels_to_variable_names(active_channels))
        selected_set.add("Ozone")
        items = []
        feature_pairs = [(0, "Ozone")] + [
            (index, name_by_channel[channel])
            for index, channel in enumerate(active_channels, start=1)
            if channel in name_by_channel
        ]
        for channel_index, name in feature_pairs:
            if name not in selected_set:
                continue
            shuffled = x_sample.clone()
            order = torch.randperm(shuffled.shape[0])
            shuffled[:, :, channel_index] = shuffled[order, :, channel_index]
            items.append({
                "name": name,
                "importance": round(float(baseline - score(shuffled)), 6),
            })

        items.sort(key=lambda item: item["importance"], reverse=True)
        return {"items": items, "baseline_metric": "r2", "baseline_value": round(float(baseline), 4),
                "sampling": {"method": "evenly_spaced_test_windows", "sample_size": sample_size,
                             "test_windows": len(x_test), "test_window_indices": sample_indices.tolist(),
                             "permutation": "one_random_permutation_per_feature", "seed": None}}

    def _uploaded_task_permutation_importance(self, task, hypers: dict, selected_variables: list[str], horizon: int, data_dirs=None):
        from sklearn.metrics import r2_score
        from training_backbones.user_model_runner import (
            assert_prediction_shape,
            build_uploaded_model_config,
            load_uploaded_model,
        )

        window = int(hypers.get("window", 3))
        task_horizon = int(hypers.get("horizon", horizon or 3))
        selected_channels, volume = self._prepare_uploaded_task_volume(
            hypers,
            window,
            task_horizon,
            data_dirs,
            normalization=self._checkpoint_normalization(task),
        )
        config = build_uploaded_model_config(
            in_channels=int(volume.values.shape[-1]),
            window=window,
            horizon=task_horizon,
            height=volume.height,
            width=volume.width,
            selected_channels=selected_channels,
            custom_model_params=hypers.get("custom_model_params", {}),
            param_schema=hypers.get("_uploaded_model_param_schema", {}),
        )
        model_path = hypers.get("_uploaded_model_path")
        if not model_path:
            raise ValueError("Uploaded model path is missing from task hyperparameters")
        model = load_uploaded_model(Path(model_path), config).to(self.device)
        state_dict = self._load_task_state_dict(task)
        model.load_state_dict(state_dict)
        model.eval()
        topography_grid = self._prepare_uploaded_topography(
            model,
            volume.latitude,
            volume.longitude,
        )

        x_test, y_test, ls_test = self._uploaded_task_test_windows(
            volume, window, task_horizon,
            {key: hypers[key] for key in ("train_ratio", "validation_ratio", "test_ratio") if key in hypers},
        )
        x_test = x_test.clone()
        if len(x_test) == 0:
            return {"items": [], "baseline_metric": "r2", "baseline_value": 0.0}
        sample_size = min(40, len(x_test))
        sample_indices = np.linspace(0, len(x_test) - 1, sample_size, dtype=int)
        x_sample = x_test[sample_indices]
        y_sample = y_test[sample_indices]
        ls_sample = ls_test[sample_indices] if ls_test is not None else None

        def score(batch):
            with torch.no_grad():
                pred = self._run_uploaded_task_model(
                    model,
                    batch,
                    ls_sample,
                    topography_grid,
                    "uploaded permutation importance",
                )
                assert_prediction_shape(pred, y_sample.to(self.device), "uploaded permutation importance")
                pred_np = pred.cpu().numpy()
            truth = y_sample.numpy()
            pred_raw = pred_np[:, :horizon, 0] * (volume.y_std + 1e-6) + volume.y_mean
            truth_raw = truth[:, :horizon, 0] * (volume.y_std + 1e-6) + volume.y_mean
            valid = np.isfinite(truth_raw) & np.isfinite(pred_raw)
            if np.count_nonzero(valid) < 10:
                return 0.0
            return float(r2_score(truth_raw[valid], pred_raw[valid]))

        baseline = score(x_sample)
        name_by_channel = {
            "U": "U_Wind",
            "V": "V_Wind",
            "D": "Dust_Optical_Depth",
            "S": "Solar_Flux_DN",
            "T": "Temperature",
        }
        selected_set = set(selected_variables or self._channels_to_variable_names(selected_channels))
        selected_set.add("Ozone")
        items = []
        feature_pairs = [(0, "Ozone")] + [
            (index, name_by_channel[channel])
            for index, channel in enumerate(selected_channels, start=1)
            if channel in name_by_channel
        ]
        for channel_index, name in feature_pairs:
            if name not in selected_set:
                continue
            shuffled = x_sample.clone()
            order = torch.randperm(shuffled.shape[0])
            shuffled[:, :, channel_index] = shuffled[order, :, channel_index]
            items.append({
                "name": name,
                "importance": round(float(baseline - score(shuffled)), 6),
            })
        items.sort(key=lambda item: item["importance"], reverse=True)
        return {"items": items, "baseline_metric": "r2", "baseline_value": round(float(baseline), 4),
                "sampling": {"method": "evenly_spaced_test_windows", "sample_size": sample_size,
                             "test_windows": len(x_test), "test_window_indices": sample_indices.tolist(),
                             "permutation": "one_random_permutation_per_feature", "seed": None}}

    @staticmethod
    def _nearest_sequence_index(ls_torch: torch.Tensor, ls_start: float) -> int:
        values = ls_torch[:, 0].detach().cpu().numpy().reshape(-1)
        if values.size == 0:
            raise ValueError("Mars Ls timeline is empty")
        padded = np.concatenate([values, values[-1:]])
        return int(resolve_forecast_window(padded, ls_start, 1, 1)["sample_index"])

    @staticmethod
    def _is_legacy_official_state_dict(state_dict) -> bool:
        keys = list(getattr(state_dict, "keys", lambda: [])())
        has_legacy_layers = any(key.startswith("layers.") for key in keys)
        has_legacy_head = any(key.startswith("conv_last.") for key in keys)
        has_adapter_keys = any(
            key.startswith("projector.") or key.startswith("backbone.")
            for key in keys
        )
        return has_legacy_layers and has_legacy_head and not has_adapter_keys

    def _load_official_task_model(
        self,
        task,
        model_architecture: str,
        input_channels: int,
        selected_channels: list[str],
        hidden_dims: list[int],
        height: int,
        width: int,
        window: int,
        horizon: int,
        use_sphere: bool,
        architecture_params: dict,
    ):
        state_dict = self._load_task_state_dict(task)
        uses_legacy_loader = self._is_legacy_official_state_dict(state_dict)

        if uses_legacy_loader:
            model = LegacyPredRNNv2(
                input_dim=int(input_channels),
                hidden_dims=list(hidden_dims or [64, 64, 64]),
                height=int(height),
                width=int(width),
                horizon=int(horizon),
            ).to(self.device)
        else:
            model = build_forecaster(
                architecture=model_architecture,
                input_channels=int(input_channels),
                selected_channels=list(selected_channels),
                hidden_dims=list(hidden_dims or [64, 64, 64]),
                height=int(height),
                width=int(width),
                window=int(window),
                horizon=int(horizon),
                use_sphere=use_sphere,
                architecture_params=architecture_params,
            ).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()
        return model, uses_legacy_loader

    @staticmethod
    def _run_official_task_model(model, x_batch: torch.Tensor, ls_batch: torch.Tensor, uses_legacy_loader: bool):
        if uses_legacy_loader:
            return model(x_batch)
        return model(x_batch, ls_batch)

    @staticmethod
    def _channels_to_variable_names(channels: list[str] | str) -> list[str]:
        channel_map = {
            "U": "U_Wind",
            "V": "V_Wind",
            "D": "Dust_Optical_Depth",
            "S": "Solar_Flux_DN",
            "T": "Temperature",
        }
        return [channel_map[channel] for channel in list(channels or "") if channel in channel_map]

    @staticmethod
    def _prediction_coordinates(latitude, longitude, spatial_shape):
        """Preserve loader order and reject guessed or inconsistent output axes."""
        axes = []
        for name, values, size in zip(("latitude", "longitude"), (latitude, longitude), spatial_shape):
            if values is None:
                raise ValueError(f"Mars prediction {name} coordinates are missing from the data source")
            try:
                axis = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Mars prediction {name} coordinates must be numeric") from exc
            if axis.ndim != 1 or axis.size == 0:
                raise ValueError(f"Mars prediction {name} coordinates must be a non-empty one-dimensional array")
            if axis.size != size:
                raise ValueError(f"Mars prediction {name} coordinate length {axis.size} does not match field size {size}")
            if not np.all(np.isfinite(axis)):
                raise ValueError(f"Mars prediction {name} coordinates must be finite")
            axes.append(axis)
        return tuple(axes)

    @staticmethod
    def _fields_to_dicts(
        fields: np.ndarray,
        lat_arr: np.ndarray,
        lon_arr: np.ndarray,
        include_points: bool = True,
    ) -> list[dict]:
        lat_arr, lon_arr = InferenceService._prediction_coordinates(lat_arr, lon_arr, fields.shape[-2:])
        result = []
        lat_list = [float(v) for v in lat_arr]
        lon_list = [float(v) for v in lon_arr]
        for step in range(fields.shape[0]):
            field = fields[step]
            points = []
            if include_points:
                for i, lat in enumerate(lat_arr):
                    for j, lon in enumerate(lon_arr):
                        val = float(field[i, j])
                        if not np.isnan(val):
                            points.append({
                                "lat": float(lat),
                                "lng": float(lon) if lon <= 180 else float(lon - 360),
                                "val": val,
                            })

            valid = field[~np.isnan(field)]
            result.append({
                "points": points,
                "lat": lat_list,
                "lon": lon_list,
                "field": np.nan_to_num(field).tolist(),
                "minVal": float(np.nanmin(valid)) if len(valid) > 0 else 0.0,
                "maxVal": float(np.nanmax(valid)) if len(valid) > 0 else 1.0,
            })
        return result

    async def get_test_results(self, task_id: int, data_dirs: dict[str, str] | None = None):
        """获取训练任务的测试结果（散点图数据）"""
        async with async_session_maker() as session:
            task = await session.get(ModelTrainingTask, task_id)
            if not task or not is_valid_model_weight_file(task.output_model_path):
                raise ValueError("Model file not found")

            # Same isolation as the cached prediction path: this direct loader
            # must not read Mars channels for an Earth task.
            require_mars_prediction_task(task)

            # 1. 解析任务参数
            hypers = json.loads(task.hyperparameters)
            hypers["training_dataset"] = str(getattr(task, "dataset_id", None) or hypers.get("training_dataset") or "openmars_mcd").strip().lower()
            verified_dirs, _ = await self._prepare_task_data_env(task, hypers)
            data_dirs = {**(data_dirs or {}), **verified_dirs}
            if getattr(task, "model_source", "official") == "uploaded":
                return await self._get_uploaded_model_test_results(task, hypers, data_dirs=data_dirs)

            window = hypers.get("window", 3)
            horizon = hypers.get("horizon", 3)
            hidden_dims = hypers.get("stlstm_hidden_dims", [64, 64, 64])
            model_architecture = normalize_model_architecture(hypers.get("model_architecture", "predrnnv2"))
            use_sphere = normalize_use_sphere(hypers)
            architecture_params = extract_architecture_params(hypers)
            
            # 2. 识别使用的变量
            active_vars = get_task_channel_suffix(task)
            # UDST -> ['U_Wind', 'Dust_Optical_Depth', 'Solar_Flux_DN', 'Temperature']
            mcd_vars_map = {
                'U': ('U_Wind', 'u'), 
                'V': ('V_Wind', 'v'), 
                'D': ('Dust_Optical_Depth', 'dustq'), 
                'S': ('Solar_Flux_DN', 'fluxsurf_dn_sw'), 
                'T': ('Temperature', 'temp')
            }
            used_mcd_vars = [mcd_vars_map[c] for c in active_vars if c in mcd_vars_map]
            base_input_dim = 1 + len(used_mcd_vars)

            # 3. 加载并预处理数据 (简化版，复用脚本逻辑)
            split_ratios = self._task_split_ratios_for_loading(hypers)
            prepare_kwargs = {"data_dirs": data_dirs}
            if split_ratios:
                prepare_kwargs["split_ratios"] = split_ratios
            checkpoint_norm = self._checkpoint_normalization(task)
            if checkpoint_norm:
                prepare_kwargs["normalization"] = checkpoint_norm
            X_torch, y_torch, ls_torch, y_mean, y_std = self._prepare_data(
                used_mcd_vars,
                window,
                horizon,
                **prepare_kwargs,
            )
            
            # 4. 加载模型
            model, uses_legacy_loader = self._load_official_task_model(
                task=task,
                model_architecture=model_architecture,
                input_channels=base_input_dim,
                selected_channels=list(active_vars),
                hidden_dims=hidden_dims,
                height=int(X_torch.shape[-2]),
                width=int(X_torch.shape[-1]),
                window=int(window),
                horizon=int(horizon),
                use_sphere=use_sphere,
                architecture_params=architecture_params,
            )

            # 5. 执行推理 (仅针对测试集)
            split_meta = self._task_split_metadata(hypers)
            split = self._task_test_split_start(len(X_torch), split_meta)
            X_test = X_torch[split:]
            y_test_true = y_torch[split:]
            
            with torch.no_grad():
                # 为了性能，限制测试点数
                sample_size = min(len(X_test), 50) 
                indices = np.linspace(0, len(X_test)-1, sample_size, dtype=int)
                
                test_xb = X_test[indices].to(self.device)
                test_yb = y_test_true[indices]
                test_lsb = ls_torch[split:][indices].to(self.device)
                
                preds = self._run_official_task_model(
                    model,
                    test_xb,
                    test_lsb,
                    uses_legacy_loader,
                ).cpu().numpy()
                trues = test_yb.numpy()

            # 6. 反标准化还原物理值
            y_pred_raw = preds * (y_std + 1e-6) + y_mean
            y_true_raw = trues * (y_std + 1e-6) + y_mean
            
            # 展平数据并进行子采样，防止前端渲染海量点位时卡顿（建议上限 50k 点）
            y_true_flat = y_true_raw.flatten()
            y_pred_flat = y_pred_raw.flatten()
            
            if len(y_true_flat) > 50000:
                step = len(y_true_flat) // 50000
                y_true_flat = y_true_flat[::step]
                y_pred_flat = y_pred_flat[::step]
            
            metric_meta = self._with_test_set_contract({}, hypers)
            return {
                "y_true": y_true_flat.tolist(),
                "y_pred": y_pred_flat.tolist(),
                "metrics": json.loads(task.metrics) if task.metrics else {},
                "metric_meta": {
                    "aggregation": metric_meta["aggregation"],
                    "split_meta": metric_meta["split_meta"],
                },
            }

    async def _get_uploaded_model_test_results(self, task, hypers, data_dirs=None):
        from training_backbones.user_model_runner import (
            assert_prediction_shape,
            build_uploaded_model_config,
            load_uploaded_model,
        )

        window = int(hypers.get("window", 3))
        horizon = int(hypers.get("horizon", 3))
        task_horizon = horizon

        selected_channels, volume = self._prepare_uploaded_task_volume(
            hypers,
            window,
            task_horizon,
            data_dirs,
            normalization=self._checkpoint_normalization(task),
        )
        config = build_uploaded_model_config(
            in_channels=int(volume.values.shape[-1]),
            window=window,
            horizon=task_horizon,
            height=volume.height,
            width=volume.width,
            selected_channels=selected_channels,
            custom_model_params=hypers.get("custom_model_params", {}),
            param_schema=hypers.get("_uploaded_model_param_schema", {}),
        )

        model_path = hypers.get("_uploaded_model_path")
        if not model_path:
            raise ValueError("Uploaded model path is missing from task hyperparameters")
        model = load_uploaded_model(Path(model_path), config).to(self.device)
        state_dict = self._load_task_state_dict(task)
        model.load_state_dict(state_dict)
        model.eval()
        topography_grid = self._prepare_uploaded_topography(
            model,
            volume.latitude,
            volume.longitude,
        )

        x_test, y_test, ls_test = self._uploaded_task_test_windows(
            volume, window, task_horizon,
            {key: hypers[key] for key in ("train_ratio", "validation_ratio", "test_ratio") if key in hypers},
        )
        y_test_true = y_test
        if len(x_test) == 0:
            raise ValueError("No uploaded model test samples are available")

        with torch.no_grad():
            sample_size = min(len(x_test), 50)
            indices = np.linspace(0, len(x_test) - 1, sample_size, dtype=int)
            test_xb = x_test[indices].to(self.device)
            test_yb = y_test_true[indices]
            test_lsb = ls_test[indices].to(self.device) if ls_test is not None else None
            pred_tensor = self._run_uploaded_task_model(
                model,
                test_xb,
                test_lsb,
                topography_grid,
                "uploaded inference test",
            )
            assert_prediction_shape(
                pred_tensor,
                test_yb.to(self.device),
                "uploaded inference test",
            )
            preds = pred_tensor.cpu().numpy()
            trues = test_yb.numpy()

        y_pred_raw = preds * (volume.y_std + 1e-6) + volume.y_mean
        y_true_raw = trues * (volume.y_std + 1e-6) + volume.y_mean
        y_true_flat = y_true_raw.flatten()
        y_pred_flat = y_pred_raw.flatten()

        if len(y_true_flat) > 50000:
            step = int(np.ceil(len(y_true_flat) / 50000))
            y_true_flat = y_true_flat[::step]
            y_pred_flat = y_pred_flat[::step]

        metric_meta = self._with_test_set_contract({}, hypers)
        return {
            "y_true": y_true_flat.tolist(),
            "y_pred": y_pred_flat.tolist(),
            "metrics": json.loads(task.metrics) if task.metrics else {},
            "metric_meta": {
                "aggregation": metric_meta["aggregation"],
                "split_meta": metric_meta["split_meta"],
            },
        }

    def _prepare_data(self, used_mcd_vars, window, horizon, data_dirs: dict[str, str] | None = None, split_ratios=None, normalization=None, split_policy=None):
        """复用训练脚本中的数据加载逻辑"""
        volume = self._load_official_task_volume(used_mcd_vars, window, horizon, data_dirs, split_ratios, normalization, split_policy)
        X_scaled = volume.values
        sample_starts = getattr(volume, "sample_starts", None)
        if sample_starts is None:
            sample_starts = np.arange(
                int(X_scaled.shape[0]) - int(window) - int(horizon) + 1,
                dtype=np.int64,
            )

        X_seq, y_seq, ls_seq, target_ls_seq = [], [], [], []
        for i in np.asarray(sample_starts, dtype=np.int64).tolist():
            X_seq.append(X_scaled[i: i + window])
            y_seq.append(volume.y_scaled[i + window: i + window + horizon])
            if volume.ls is not None:
                ls_seq.append(volume.ls[i: i + window])
                target_ls_seq.append(volume.ls[i + window: i + window + horizon])

        X_torch = torch.tensor(np.array(X_seq)).permute(0, 1, 4, 2, 3).float()
        y_torch = torch.tensor(np.array(y_seq)).unsqueeze(2).float()
        ls_torch = torch.tensor(np.array(ls_seq)).float() if ls_seq else None
        self._last_prepared_window_metadata = {
            "sample_starts": np.asarray(sample_starts, dtype=np.int64),
            "input_ls": None if volume.ls is None else np.asarray(ls_seq),
            "target_ls": None if volume.ls is None else np.asarray(target_ls_seq),
            "sample_mars_years": tuple(volume.sample_mars_years),
            "ls_values": volume.ls,
            "blocks": tuple(volume.blocks),
            "latitude": volume.latitude,
            "longitude": volume.longitude,
            "split_ranges": volume.split_ranges,
            "split_window_starts": volume.split_window_starts,
            "split_policy": volume.split_policy,
        }

        return X_torch, y_torch, ls_torch, volume.y_mean, volume.y_std

    def _load_official_task_volume(self, used_mcd_vars, window, horizon, data_dirs=None, split_ratios=None, normalization=None, split_policy=None):
        from services.prediction_volume_cache import ScaledVolume, get_scaled_volume, put_scaled_volume, volume_signature
        directories = data_dirs or {}
        dataset_id = "mcd_overview" if directories.get("MCD_RAW_3H_DIR") or directories.get("ASTRAATMOS_MCD_RAW_3H_DIR") else "openmars_mcd"
        raw_dir = Path(directories.get("MCD_RAW_3H_DIR") or directories.get("ASTRAATMOS_MCD_RAW_3H_DIR") or MCD_RAW_3H_DIR)
        channel_map = {"U_Wind": "U", "V_Wind": "V", "Dust_Optical_Depth": "D", "Solar_Flux_DN": "S", "Temperature": "T"}
        selected_channels = [channel_map[name] for name, _ in used_mcd_vars if name in channel_map]
        signature = volume_signature(
            openmars_dir=Path(directories.get("ASTRAATMOS_OPENMARS_DIR") or self.openmars_dir),
            mcd_dir=Path(directories.get("ASTRAATMOS_MCD_DIR") or self.mcd_dir),
            selected_channels=selected_channels,
            training_dataset=dataset_id,
            mcd_overview_dir=raw_dir,
            cache_prefix=(
                "official:" + repr(tuple(sorted((split_ratios or {}).items())))
                + ":" + self._mars_normalization_cache_key(normalization, selected_channels)
            ),
            split_ratios=split_ratios,
            window=window,
            horizon=horizon,
        )
        if not data_dirs:
            cached = get_scaled_volume(signature)
            if isinstance(cached, ScaledVolume):
                return cached
        volume = self._shared_volume_to_cache(
            prepare_scaled_volume(
                training_dataset=dataset_id,
                openmars_dir=Path(directories.get("ASTRAATMOS_OPENMARS_DIR") or self.openmars_dir),
                mcd_dir=Path(directories.get("ASTRAATMOS_MCD_DIR") or self.mcd_dir),
                raw_dir=raw_dir,
                selected_channels=selected_channels,
                window=window,
                horizon=horizon,
                split_ratios=split_ratios or dict(LEGACY_SPLIT_RATIOS),
                normalization=normalization,
                split_policy=split_policy,
            )
        )
        if not data_dirs:
            put_scaled_volume(signature, volume)
        return volume

    @staticmethod
    def _shared_volume_to_cache(volume):
        from services.prediction_volume_cache import ScaledVolume
        return ScaledVolume(
            values=volume.values, y_scaled=volume.y_scaled, ls=volume.ls,
            y_mean=volume.y_mean, y_std=volume.y_std, height=volume.height,
            width=volume.width, latitude=volume.latitude, longitude=volume.longitude,
            split_idx=volume.split_idx, input_means=volume.input_means,
            input_stds=volume.input_stds, dataset_id=volume.dataset_id,
            source_type=volume.source_type, data_dir=volume.data_dir,
            manifest=volume.manifest, dataset_fingerprint=volume.fingerprint,
            sample_starts=volume.sample_starts,
            sample_mars_years=volume.sample_mars_years,
            blocks=volume.blocks,
            train_sample_end=volume.train_sample_end,
            data_directories=volume.data_directories,
            split_ranges=volume.split_ranges,
            split_window_starts=volume.split_window_starts,
            split_policy=volume.split_policy,
        )
