"""Bounded, account-scoped snapshots of already computed results.

No model execution or cache-miss computation is allowed in the export path.
Private guards stay in memory; only whitelisted metadata enters output files.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import numpy as np


class ExportUnavailable(ValueError):
    pass


class ArtifactGuardError(OSError):
    """Raised when a task's model artifact cannot be fingerprinted safely.

    The underlying filesystem exception is intentionally kept out of the public
    response: it may contain an absolute path or implementation details.
    """

    code = "invalid_earth_training_artifact"
    status_code = 409

    def __init__(self, message="The Earth model artifact is unavailable"):
        super().__init__(message)


class ExportSnapshotUnavailable(ExportUnavailable):
    code = "earth_export_snapshot_unavailable"


@dataclass(frozen=True)
class DiskSnapshot:
    path: str
    size: int
    key: str

    def load_step(self, step: int) -> dict:
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                return {
                    "reference": np.asarray(archive["reference"][step], dtype=np.float64),
                    "prediction": np.asarray(archive["prediction"][step], dtype=np.float64),
                    "residual": np.asarray(archive["residual"][step], dtype=np.float64),
                    "latitude": np.asarray(archive["latitude"], dtype=np.float64),
                    "longitude": np.asarray(archive["longitude"], dtype=np.float64),
                }
        except (OSError, ValueError, IndexError, KeyError) as exc:
            raise ExportUnavailable("Result snapshot is missing or unreadable. Refresh the prediction.") from exc


@dataclass(frozen=True)
class FigureSource:
    ref: dict
    user_id: int
    guard: str
    data: object
    metadata: dict
    expires: float
    size: int


def task_guard(task):
    try:
        raw_path = getattr(task, "output_model_path", None)
        if not raw_path:
            raise OSError("missing model artifact")
        path = Path(raw_path)
        stat = path.stat()
        if not path.is_file():
            raise OSError("model artifact is not a file")
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactGuardError() from exc
    try:
        resolved_path = str(path.resolve())
        value = [resolved_path, stat.st_size, stat.st_mtime_ns,
                 task.hyperparameters, getattr(task, "dataset_snapshot", None),
                 getattr(task, "dataset_fingerprint", None), task.status]
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactGuardError() from exc
    return hashlib.sha256(json.dumps(value, default=str, sort_keys=True).encode()).hexdigest()


def _size(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_size(v) for v in value.values())
    if isinstance(value, list):
        return sum(_size(v) for v in value)
    return len(str(value).encode()) + 32


class ExportSourceStore:
    def __init__(self, max_bytes=128 * 1024 ** 2, max_entries=32, ttl=7200,
                 disk_max_bytes=1024 * 1024 ** 3):
        self.max_bytes, self.max_entries, self.ttl = max_bytes, max_entries, ttl
        self.disk_max_bytes = disk_max_bytes
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.bytes = 0
        self.disk_bytes = 0
        self._disk_refs = {}

    def _release(self, source):
        if isinstance(source.data, DiskSnapshot):
            snapshot = source.data
            entry = self._disk_refs.get(snapshot.key)
            if entry is None:
                return
            refs, size, path = entry
            if refs > 1:
                self._disk_refs[snapshot.key] = (refs - 1, size, path)
                return
            self._disk_refs.pop(snapshot.key, None)
            self.disk_bytes -= size
            try:
                os.unlink(path)
            except OSError:
                pass
            return
        self.bytes -= source.size

    def remove(self, ref):
        with self.lock:
            source = self.entries.pop(ref.get("id"))
            self._release(source)

    def invalidate_task_exports(self, *, task_id, user_id, planet="earth"):
        """Drop prior prediction snapshots before publishing a refreshed source."""
        with self.lock:
            for key, source in list(self.entries.items()):
                if (source.user_id == int(user_id)
                        and source.ref.get("task_id") == int(task_id)
                        and source.ref.get("planet") == planet):
                    self.entries.pop(key, None)
                    self._release(source)

    def register(self, *, user_id, task_id, planet, analysis, horizon, origin,
                 variables, guard, data, metadata, step=None):
        disk = data if isinstance(data, DiskSnapshot) else None
        size = (disk.size if disk and disk.key not in self._disk_refs else 0) if disk else _size(data) + _size(metadata)
        if disk:
            if disk.key not in self._disk_refs and self.disk_bytes + disk.size > self.disk_max_bytes:
                return None
        elif size > self.max_bytes:
            return None
        data = data if disk else copy.deepcopy(data)
        ref = dict(id=uuid.uuid4().hex, task_id=int(task_id), planet=planet,
                   analysis=analysis, horizon=int(horizon), origin=str(origin), variables=list(variables))
        if step is not None:
            ref["step"] = int(step)
        source = FigureSource(ref, int(user_id), guard, data, copy.deepcopy(metadata), time.monotonic() + self.ttl, size)
        with self.lock:
            for key, old in list(self.entries.items()):
                if old.expires <= time.monotonic():
                    self._release(self.entries.pop(key))
            self.entries[ref["id"]] = source
            if disk:
                if disk.key in self._disk_refs:
                    refs, disk_size, path = self._disk_refs[disk.key]
                    self._disk_refs[disk.key] = (refs + 1, disk_size, path)
                else:
                    self._disk_refs[disk.key] = (1, disk.size, disk.path)
                    self.disk_bytes += disk.size
            else:
                self.bytes += size
            while self.bytes > self.max_bytes or self.disk_bytes > self.disk_max_bytes or len(self.entries) > self.max_entries:
                _, old = self.entries.popitem(last=False)
                self._release(old)
        return ref

    def get(self, ref, user_id):
        with self.lock:
            source = self.entries.get(ref["id"])
            if source is None or source.expires <= time.monotonic():
                raise ExportUnavailable("Result expired or unavailable. Refresh the prediction/analysis result.")
            if source.user_id != user_id:
                raise PermissionError("No permission to export this result")
            if source.ref != ref:
                raise ExportUnavailable("Result conditions do not match the selected model, origin, horizon or variables.")
            self.entries.move_to_end(ref["id"])
            if isinstance(source.data, DiskSnapshot):
                step = int(source.ref.get("step", 0))
                return replace(source, data=source.data.load_step(step))
            return source


EXPORT_SOURCES = ExportSourceStore(max_entries=512)


def mars_result_guard(task, data_dirs):
    from services.prediction_analysis_cache import build_artifact_fingerprint
    try:
        return build_artifact_fingerprint(task, data_dirs)
    except (AttributeError, FileNotFoundError):
        return None


def register_mars(result, *, task, hypers, data_dirs, user_id, analysis, horizon, origin="", variables=(), expected_guard=None):
    guard = mars_result_guard(task, data_dirs)
    if guard is None or (expected_guard is not None and guard != expected_guard):
        return {**result, "export_ref": None}

    metadata = dict(model=str(task.custom_model_name or f"Task {task.id}"), task_id=int(task.id),
                    dataset_id=hypers.get("training_dataset", "openmars_mcd"), planet="mars",
                    source_unit="um-atm", window=int(hypers.get("window", 3)), horizon=int(horizon),
                    selected_variables=list(variables),
                    dataset_identity_status=hypers.get("_dataset_identity_status", "legacy"))
    metadata["result_artifact_fingerprint"] = guard
    if getattr(task, "dataset_fingerprint", None):
        metadata["dataset_fingerprint"] = task.dataset_fingerprint
    if analysis == "prediction":
        data = {key: [np.asarray(row["field"], dtype=np.float64) for row in result[key]]
                for key in ("ground_truth", "prediction", "residual")}
        data["latitude"] = [np.asarray(row.get("lat", [])) for row in result["ground_truth"]]
        data["longitude"] = [np.asarray(row.get("lon", [])) for row in result["ground_truth"]]
        # Preserve and validate the axes of all three fields, including old caches.
        for key in ("prediction", "residual"):
            data[f"{key}_latitude"] = [np.asarray(row.get("lat", [])) for row in result[key]]
            data[f"{key}_longitude"] = [np.asarray(row.get("lon", [])) for row in result[key]]
        metadata.update(scope="current_prediction_window", requested_ls_start=float(origin),
                        input_ls_values=result.get("input_ls_values", []), ls_values=result.get("ls_values", []))
        info = result.get("model_info", {})
        for key in ("sample_index", "mars_year", "target_mars_year", "input_mars_year", "input_mars_years", "target_mars_years", "normalization_source", "legacy_compatibility"):
            if key in info:
                metadata[key] = info[key]
    elif analysis == "metrics":
        data = {key: result.get(key) for key in ("overall", "per_step", "aggregation", "split_meta")}
        metadata.update(scope="full_test_set", split_meta=result.get("split_meta", {}), aggregation=result.get("aggregation", {}))
    else:
        data = {key: result.get(key) for key in ("items", "baseline_metric", "baseline_value", "sampling")}
        metadata.update(scope="sampled_test_set", sampling=result.get("sampling") or {"method": "legacy_unknown", "maximum_windows": 40})
    ref = EXPORT_SOURCES.register(user_id=user_id, task_id=task.id, planet="mars", analysis=analysis,
                                  horizon=horizon, origin=origin, variables=variables,
                                  guard=guard, data=data, metadata=metadata)
    return {**result, "export_ref": ref}


def register_earth(result, task, user_id):
    data = {key: [np.asarray(row["field"], dtype=np.float64) for row in result[key]]
            for key in ("reference", "prediction", "residual")}
    data["latitude"] = np.asarray(result["grid"]["latitude"])
    data["longitude"] = np.asarray(result["grid"]["longitude"])
    keys = ("task_id", "dataset_id", "dataset_version", "dataset_fingerprint", "forecast_origin",
            "origin_split", "input_dates", "target_dates", "input_timestamps", "target_timestamps",
            "frequency_hours", "time_zone", "window", "horizon", "model_architecture", "model_source")
    metadata = {key: result[key] for key in keys if key in result}
    metadata.update(planet="earth", scope="current_prediction_window", source_unit="DU",
                    model=str(task.custom_model_name or f"Task {task.id}"),
                    source_warnings=list(result.get("warnings") or []),
                    model_source_status=result.get("model_source_status", "official"))
    # Keep the prediction snapshot self-describing.  New checkpoints expose
    # cell-area weighting while historical checkpoints retain their original
    # grid-uniform policy; exports must not infer a policy from the task age.
    metrics = result.get("metrics") or {}
    for key in ("schema", "aggregation", "metric_units", "metric_policy"):
        if key in metrics:
            metadata[key if key != "schema" else "metrics_schema"] = metrics[key]
    EXPORT_SOURCES.invalidate_task_exports(task_id=task.id, user_id=user_id, planet="earth")
    ref = EXPORT_SOURCES.register(user_id=user_id, task_id=task.id, planet="earth", analysis="prediction",
                                  horizon=result["horizon"], origin=result["forecast_origin"], variables=[],
                                  guard=task_guard(task), data=data, metadata=metadata)
    return {**result, "export_ref": ref}


def register_earth_metrics(item, task, user_id):
    result = item["metrics"]
    data = {key: result[key] for key in ("overall", "per_step", "aggregation", "split_meta")}
    metadata = {key: item[key] for key in ("task_id", "dataset_id", "dataset_version", "dataset_fingerprint")}
    metadata.update(planet="earth", scope="full_test_set", source_unit="DU", window=item['window'], horizon=item['horizon'],
                    model=item["model_name"], split_meta=result["split_meta"], aggregation=result["aggregation"],
                    metric_source="verified_checkpoint_test_metrics", frequency_hours=3)
    for key in ("schema", "metric_units", "metric_policy"):
        if key in result:
            metadata[key] = result[key]
    ref = EXPORT_SOURCES.register(user_id=user_id, task_id=task.id, planet="earth", analysis="metrics",
                                  horizon=item['horizon'], origin="", variables=[], guard=task_guard(task), data=data, metadata=metadata)
    return {**result, "export_ref": ref}
