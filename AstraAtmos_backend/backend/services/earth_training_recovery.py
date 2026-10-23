"""Explicit audited recovery for metrics-contract failures or reload timeouts."""

from __future__ import annotations

import copy
from contextlib import contextmanager, closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid

from services.dataset_identity import EARTH_DATASET_3HOURLY_ID, EARTH_THREE_HOURLY_DATASET_IDS
from services.earth_dataset import (
    _fit_threehour_normalization_uncached, threehour_global_cell_area_weights,
)
from services.earth_dataset_metadata import (
    THREE_HOURLY_DATA_FILE_NAME, file_sha256, package_signature,
    read_earth_3hourly_release,
)
from services.earth_training_artifact import (
    EarthArtifactError, assert_release_matches_binding, load_earth_training_artifact,
    verify_earth_model_reload_isolated, MODEL_RELOAD_TIMEOUT_MESSAGE,
)
from services.earth_training_contract import (
    CELL_AREA_WEIGHTED_AGGREGATION,
    EARTH_3HOURLY_METRICS_SCHEMA_V2, EARTH_3HOURLY_METRICS_SCHEMA_V3,
    normalize_earth_training_hyperparameters,
)

FAILURE_MESSAGE = "Checkpoint metrics do not match the task's required metrics contract"
AUDIT_SCHEMA = "earth_metrics_contract_recovery_v1"
RELOAD_AUDIT_SCHEMA = "earth_reload_timeout_recovery_v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


@contextmanager
def _readonly(database):
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _task_row(connection, task_id):
    row = connection.execute("SELECT * FROM model_training_tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise EarthArtifactError("Recovery task does not exist")
    return dict(row)


def _binding(row):
    result = {key: row[key] for key in (
        "dataset_id", "dataset_version", "dataset_fingerprint", "dataset_identity_status", "dataset_snapshot",
    )}
    result["dataset_snapshot"] = json.loads(result["dataset_snapshot"])
    return result


def _source_evidence(hypers, checkpoint, row):
    if row["model_source"] != checkpoint.model_source:
        raise EarthArtifactError("Recovery model source differs from task")
    if checkpoint.model_source != "uploaded":
        return {"model_source": checkpoint.model_source}
    actual = checkpoint.uploaded_model
    frozen = hypers.get("_earth_uploaded_reference")
    if not isinstance(frozen, dict):
        raise EarthArtifactError("Recovery requires the task's frozen uploaded source")
    for key in ("package_id", "version", "content_hash", "source_text", "param_schema", "custom_model_params", "contract_schema"):
        if frozen.get(key) != actual.get(key):
            raise EarthArtifactError(f"Recovery uploaded source {key} differs from task")
    if (row["uploaded_model_id"] != actual["package_id"]
            or row["uploaded_model_version"] != actual["version"]):
        raise EarthArtifactError("Recovery uploaded package identity differs from task columns")
    source_hash = hashlib.sha256(actual["source_text"].encode("utf-8")).hexdigest()
    source_path = Path(actual["source_path"])
    if source_hash != actual["content_hash"] or not source_path.is_file() or file_sha256(source_path) != source_hash:
        raise EarthArtifactError("Recovery source file or embedded source SHA-256 is invalid")
    return {"model_source": "uploaded", "package_id": actual["package_id"], "version": actual["version"],
            "source_path": str(source_path), "source_sha256": source_hash}


def verify_recovery(database, task_id, package_dir, *, expected_checkpoint_sha256,
                    timeout_seconds=300.0, progress=None, failure_kind="metrics-contract"):
    """Read and verify all evidence without changing DB, weights, data or logs."""
    def report(message):
        if progress is not None:
            progress(message)

    with _readonly(database) as connection:
        row = _task_row(connection, task_id)
    hypers = json.loads(row["hyperparameters"])
    failure = json.loads(row["metrics"] or "{}")
    if failure_kind not in {"metrics-contract", "reload-timeout"}:
        raise EarthArtifactError("Unsupported recovery failure kind")
    reload_timeout = failure_kind == "reload-timeout"
    failure_message = MODEL_RELOAD_TIMEOUT_MESSAGE if reload_timeout else FAILURE_MESSAGE
    if (row["status"] != "failed" or row["model_script"] != "earth_daily.py"
            or row["dataset_id"] not in EARTH_THREE_HOURLY_DATASET_IDS
            or row["dataset_version"] != "v1" or row["dataset_identity_status"] != "verified"
            or row["end_time"] is None
            or failure.get("error_code") != "invalid_earth_training_artifact"
            or failure.get("error") != failure_message
            or (not reload_timeout and (
                row["dataset_id"] != EARTH_DATASET_3HOURLY_ID
                or hypers.get("_earth_metrics_schema") != EARTH_3HOURLY_METRICS_SCHEMA_V2))
            or (reload_timeout and (
                hypers.get("_earth_metrics_schema") != EARTH_3HOURLY_METRICS_SCHEMA_V3
                or hypers.get("_earth_metrics_aggregation") != CELL_AREA_WEIGHTED_AGGREGATION))):
        expected = "a failed v3 Earth task with the exact reload-timeout failure" if reload_timeout else (
            "a failed v2 Earth task with the exact metrics-contract failure")
        raise EarthArtifactError("Recovery requires " + expected)
    # Validate the original frozen declaration before proposing the explicit migration.
    from services.earth_task_split import required_task_split
    from services.earth_training_contract import earth_task_metrics_contract
    earth_task_metrics_contract(hypers, executable=True)
    required_task_split(hypers)
    path = Path(row["output_model_path"])
    weight_sha = file_sha256(path)
    if weight_sha != expected_checkpoint_sha256:
        raise EarthArtifactError("Recovery checkpoint SHA-256 differs from the explicitly selected artifact")
    migrated = copy.deepcopy(hypers)
    if not reload_timeout:
        migrated["_earth_metrics_schema"] = EARTH_3HOURLY_METRICS_SCHEMA_V3
        migrated["_earth_metrics_aggregation"] = CELL_AREA_WEIGHTED_AGGREGATION
    checkpoint = load_earth_training_artifact(
        path, expected_binding=_binding(row), expected_hyperparameters=migrated, expected_task_id=task_id,
    )
    if (checkpoint.metrics["schema"] != EARTH_3HOURLY_METRICS_SCHEMA_V3
            or checkpoint.metrics["aggregation"] != CELL_AREA_WEIGHTED_AGGREGATION
            or checkpoint.run["loss"] != "normalized_mse_cell_area_weighted"):
        raise EarthArtifactError("Recovery requires a genuine v3 area-weighted completed run")
    if _binding(row)["dataset_snapshot"] != checkpoint.dataset_binding["dataset_snapshot"]:
        raise EarthArtifactError("Recovery task and checkpoint dataset snapshots differ")
    task_params = normalize_earth_training_hyperparameters(
        {key: value for key, value in hypers.items() if not key.startswith("_")},
        dataset_id=row["dataset_id"],
    )
    if checkpoint.run.get("hyperparameters") != task_params:
        raise EarthArtifactError("Recovery checkpoint training parameters differ from the frozen task")
    if (checkpoint.run.get("seed") != task_params["seed"]
            or checkpoint.run["epochs_completed"] != row["current_epoch"]
            or not 1 <= checkpoint.run["epochs_completed"] <= task_params["epochs"]
            or row["total_epochs"] != task_params["epochs"]):
        raise EarthArtifactError("Recovery checkpoint epoch/seed evidence differs from task")
    if checkpoint.run.get("optimizer") != "Adam":
        raise EarthArtifactError("Recovery checkpoint optimizer differs from the Earth runner")
    for name in ("validation", "test"):
        loss = checkpoint.run.get("final_" + name + "_loss")
        expected_loss = checkpoint.metrics["splits"][name]["overall"]["mse"] / checkpoint.normalization["scale"][0] ** 2
        if (type(loss) not in (int, float) or not math.isfinite(loss)
                or not math.isclose(loss, expected_loss, rel_tol=1e-5, abs_tol=1e-7)):
            raise EarthArtifactError(f"Recovery {name} loss disagrees with physical area-weighted MSE")
    source = _source_evidence(hypers, checkpoint, row)
    log_path = Path(row["log_file_path"])
    log_sha = file_sha256(log_path)
    log = log_path.read_text(encoding="utf-8")
    if (failure_message not in log or "Earth training completed" not in log
            or "Model saved: " + str(path) not in log):
        raise EarthArtifactError("Recovery requires the preserved save/completion/failure log")
    report("Checkpoint, frozen parameters, source and complete metrics verified")
    # No shared-proof shortcut: recompute release hash, scientific contract and train statistics.
    release = read_earth_3hourly_release(package_dir, force_full=True, progress=progress)
    # The low-level verifier returns scientific metadata; registry identity is
    # added only after verifying the active three-hour publication.
    release.metadata.setdefault("dataset_id", row["dataset_id"])
    release.metadata.setdefault("dataset_version", "v1")
    assert_release_matches_binding(release, checkpoint.dataset_binding)
    expected_snapshot = checkpoint.dataset_binding["dataset_snapshot"]
    for key in ("schema", "data_sha256", "grid", "time", "splits", "variables", "channel_order"):
        if release.metadata.get(key) != expected_snapshot.get(key):
            raise EarthArtifactError(f"Recovery release {key} differs from frozen dataset snapshot")
    weights = threehour_global_cell_area_weights(release)
    normalization = _fit_threehour_normalization_uncached(
        release, checkpoint.model_config["input_channel_order"],
        task_split=migrated["_earth_task_split"], progress=progress,
    )
    if normalization != checkpoint.normalization:
        raise EarthArtifactError("Recovery train-only normalization differs from recomputed source statistics")
    report("Release identity, task partitions, latitude bounds and train-only normalization verified")
    verify_earth_model_reload_isolated(path, timeout_seconds=timeout_seconds)
    report("Isolated strict model reload, eval independence and output round trip verified")
    plan = {"schema": RELOAD_AUDIT_SCHEMA if reload_timeout else AUDIT_SCHEMA,
            "failure_kind": failure_kind, "original_task": row, "migrated_hyperparameters": migrated,
            "checkpoint_path": str(path), "checkpoint_sha256": weight_sha,
            "log_sha256": log_sha, "package_dir": str(Path(package_dir).resolve()),
            "package_signature": release.signature, "metrics": checkpoint.metrics,
            "verification": {"dataset_fingerprint": release.metadata["dataset_fingerprint"],
                             "data_sha256": release.metadata["data_sha256"],
                             "normalization": normalization, "source": source,
                             "cell_weight_sum": float(weights.sum()), "isolated_reload": "passed",
                             "run": checkpoint.run}}
    _assert_evidence_unchanged(plan)
    return plan


def _assert_evidence_unchanged(plan):
    if (file_sha256(Path(plan["checkpoint_path"])) != plan["checkpoint_sha256"]
            or file_sha256(Path(plan["original_task"]["log_file_path"])) != plan["log_sha256"]
            or package_signature(plan["package_dir"], data_file_name=THREE_HOURLY_DATA_FILE_NAME)
            != plan["package_signature"]):
        raise EarthArtifactError("Recovery evidence changed after validation")
    source = plan["verification"]["source"]
    if source["model_source"] == "uploaded" and file_sha256(Path(source["source_path"])) != source["source_sha256"]:
        raise EarthArtifactError("Recovery source changed after validation")


def recover_task(database, task_id, package_dir, *, expected_checkpoint_sha256,
                 audit_root, timeout_seconds=300.0, progress=None, failure_kind="metrics-contract"):
    """Verify first, then backup with SQLite API and update exactly one unchanged row."""
    plan = verify_recovery(database, task_id, package_dir,
                           expected_checkpoint_sha256=expected_checkpoint_sha256,
                           timeout_seconds=timeout_seconds, progress=progress, failure_kind=failure_kind)
    operation_id = str(uuid.uuid4())
    directory = Path(audit_root).resolve() / f"task_{task_id}_{operation_id}"
    directory.mkdir(parents=True, exist_ok=False)
    backup_path = directory / "database.sqlite3"
    audit_path = directory / "recovery.json"
    log_backup = directory / "original-training.log"
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN IMMEDIATE")
        if _task_row(connection, task_id) != plan["original_task"]:
            raise EarthArtifactError("Recovery task changed while validation was running")
        _assert_evidence_unchanged(plan)
        # A separate read-only connection can backup while the reserved write lock
        # prevents concurrent task writes. The writer has not changed any pages yet.
        with _readonly(database) as source, closing(sqlite3.connect(backup_path)) as backup:
            source.backup(backup)
            backup.row_factory = sqlite3.Row
            if (backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                    or _task_row(backup, task_id) != plan["original_task"]):
                raise EarthArtifactError("Recovery database backup did not preserve the original task")
        log_backup.write_bytes(Path(plan["original_task"]["log_file_path"]).read_bytes())
        if file_sha256(log_backup) != plan["log_sha256"]:
            raise EarthArtifactError("Recovery failure log backup differs from the original")
        migrated = copy.deepcopy(plan["migrated_hyperparameters"])
        reload_timeout = plan.get("failure_kind") == "reload-timeout"
        recovery_key = "_earth_reload_recovery" if reload_timeout else "_earth_metrics_recovery"
        migrated[recovery_key] = {
            "schema": plan["schema"], "operation_id": operation_id,
            "recovered_utc": datetime.now(timezone.utc).isoformat(),
            "from_schema": EARTH_3HOURLY_METRICS_SCHEMA_V3 if reload_timeout else EARTH_3HOURLY_METRICS_SCHEMA_V2,
            "to_schema": EARTH_3HOURLY_METRICS_SCHEMA_V3,
            "reason": ("Completed checkpoint passed isolated reload after the original timeout" if reload_timeout else
                       "Task marker incorrectly froze v2; verified saved run uses area weighting"),
            "original_failure": json.loads(plan["original_task"]["metrics"]),
            "checkpoint_sha256": plan["checkpoint_sha256"],
            "database_backup": str(backup_path), "audit_path": str(audit_path),
        }
        plan.update(operation_id=operation_id, database_backup=str(backup_path),
                    database_backup_sha256=file_sha256(backup_path), log_backup=str(log_backup),
                    migrated_hyperparameters=migrated, intended_status="completed", intended_progress=100.0)
        # Write the complete original row and immutable evidence before committing.
        audit_path.write_text(_json(plan), encoding="utf-8")
        _assert_evidence_unchanged(plan)
        cursor = connection.execute(
            "UPDATE model_training_tasks SET hyperparameters=?, metrics=?, status='completed', progress=100.0, pid=NULL WHERE id=? AND status='failed'",
            (_json(migrated), _json(plan["metrics"]), task_id),
        )
        if cursor.rowcount != 1:
            raise EarthArtifactError("Recovery did not update exactly one failed task")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {"task_id": task_id, "status": "completed", "progress": 100.0,
            "checkpoint_sha256": plan["checkpoint_sha256"], "audit_path": str(audit_path),
            "database_backup": str(backup_path), "test_metrics": plan["metrics"]["splits"]["test"]["overall"]}
