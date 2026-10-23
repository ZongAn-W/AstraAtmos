"""Audited task recovery on disposable SQLite databases and synthetic checkpoints.

The global NetCDF fixture is synthetic. It trains no model, and only the release
publication reader is replaced for the full verification test: strict checkpoint
loading, latitude weights, uncached train statistics and isolated model reload
remain real. Transaction tests reuse that known evidence to test backup and races.
"""

import copy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services import earth_training_recovery as recovery
from services.earth_dataset import threehour_global_cell_area_weights
from services.earth_dataset_metadata import file_sha256
from services.earth_training_artifact import (
    EarthArtifactError, ErrorAccumulator, build_metrics_block,
    load_earth_training_artifact, save_earth_artifact_atomic,
)
from services.earth_training_contract import (
    CELL_AREA_WEIGHTED_AGGREGATION, EARTH_ALLOWED_PARAMETER_KEYS,
    EARTH_3HOURLY_METRICS_SCHEMA_V2, EARTH_3HOURLY_METRICS_SCHEMA_V3,
    normalize_earth_training_hyperparameters,
)
from test_earth_3hourly_training_runner import synthetic_training_release
from test_earth_diagnostics_integration import _create_bundle

TASK_ID = 97


@pytest.fixture(scope="module")
def recovery_template(synthetic_training_release, tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        bundle = _create_bundle(synthetic_training_release,
                                tmp_path_factory.mktemp("recovery-template"),
                                "uploaded", task_id=TASK_ID)
        payload = copy.deepcopy(bundle.payload)
        params = normalize_earth_training_hyperparameters(
            {key: value for key, value in json.loads(bundle.task.hyperparameters).items()
             if key in EARTH_ALLOWED_PARAMETER_KEYS}, dataset_id=bundle.task.dataset_id)
        params["epochs"] = 3
        payload["run"].update(hyperparameters=params, seed=params["seed"],
                              epochs_completed=3, best_epoch=2,
                              loss="normalized_mse_cell_area_weighted")
        weights = threehour_global_cell_area_weights(bundle.release)
        # Unequal errors by latitude prove the v3 block is actually weighted.
        target = np.full((1, 1, 1, 240, 480), 10., dtype="float64")
        prediction = target + np.linspace(1., 4., 240)[None, None, None, :, None]
        values = {}
        counts = payload["training_contract"]["split_window_counts"]
        for name in ("validation", "test"):
            accumulator = ErrorAccumulator(dataset_id=bundle.task.dataset_id, horizon=1)
            for _ in range(counts[name]):
                accumulator.update(prediction, target, spatial_weights=weights)
            values[name] = accumulator.result()
        payload["metrics"] = build_metrics_block(
            validation=values["validation"], test=values["test"],
            validation_window_count=counts["validation"],
            test_window_count=counts["test"], dataset_id=bundle.task.dataset_id,
            horizon=1, aggregation=CELL_AREA_WEIGHTED_AGGREGATION)
        for name in ("validation", "test"):
            payload["run"]["final_" + name + "_loss"] = (
                payload["metrics"]["splits"][name]["overall"]["mse"] /
                payload["normalization"]["scale"][0] ** 2)
        hypers = copy.deepcopy(params)
        hypers.update(_earth_task_split=bundle.split,
                      _earth_metrics_schema=EARTH_3HOURLY_METRICS_SCHEMA_V2,
                      _earth_uploaded_reference=copy.deepcopy(payload["model_ref"]["uploaded_model"]))
        reference = payload["model_ref"]["uploaded_model"]
        hypers.update(_uploaded_model_id=reference["package_id"],
                      _uploaded_model_version=reference["version"],
                      _uploaded_model_content_hash=reference["content_hash"],
                      _earth_uploaded_contract_schema=reference["contract_schema"])
        return SimpleNamespace(bundle=bundle, payload=payload, hypers=hypers)
    finally:
        torch.set_num_threads(previous)


def _rows(database):
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM model_training_tasks ORDER BY id")]


def _update_task(case, **changes):
    with sqlite3.connect(case.database) as connection:
        assignments = ", ".join(key + "=?" for key in changes)
        connection.execute("UPDATE model_training_tasks SET " + assignments + " WHERE id=?",
                           [*changes.values(), TASK_ID])


@pytest.fixture
def recovery_case(recovery_template, tmp_path, monkeypatch):
    template = recovery_template
    payload = copy.deepcopy(template.payload)
    hypers = copy.deepcopy(template.hypers)
    source = tmp_path / "uploaded_model.py"
    source.write_bytes(payload["model_ref"]["uploaded_model"]["source_text"].encode("utf-8"))
    payload["model_ref"]["uploaded_model"]["source_path"] = str(source)
    hypers["_earth_uploaded_reference"]["source_path"] = str(source)
    path = tmp_path / "task_97_synthetic.pth"
    save_earth_artifact_atomic(payload, path, strict_reload=False)
    log = tmp_path / "task_97.log"
    log.write_text("Earth training completed\nModel saved: " + str(path) + "\n" +
                   recovery.FAILURE_MESSAGE + "\n", encoding="utf-8")
    binding = copy.deepcopy(payload["dataset_binding"])
    binding["dataset_snapshot"] = json.dumps(binding["dataset_snapshot"])
    row = dict(id=TASK_ID, **binding, model_script="earth_daily.py", model_source="uploaded",
               uploaded_model_id=payload["model_ref"]["uploaded_model"]["package_id"],
               uploaded_model_version=1, hyperparameters=json.dumps(hypers), status="failed",
               output_model_path=str(path), end_time="2026-10-11 10:03:00",
               start_time="2026-10-11 10:00:00", created_at="2026-10-11 09:59:00",
               current_epoch=3, total_epochs=3, progress=100., pid=12345, user_id=1,
               log_file_path=str(log), metrics=json.dumps({"error_code": "invalid_earth_training_artifact",
                                                        "error": recovery.FAILURE_MESSAGE}))
    database = tmp_path / "tasks.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE model_training_tasks (" + ", ".join(
            key + (" INTEGER" if type(value) is int else " REAL" if type(value) is float else " TEXT")
            for key, value in row.items()) + ")")
        sql = "INSERT INTO model_training_tasks VALUES (" + ",".join("?" for _ in row) + ")"
        connection.execute(sql, list(row.values()))
        other = copy.deepcopy(row)
        other.update(id=98, status="queued", metrics='{"unrelated": true}', progress=0.)
        connection.execute(sql, list(other.values()))
    case = SimpleNamespace(database=database, path=path, source=source, log=log,
                           payload=payload, hypers=hypers, release=template.bundle.release,
                           package_dir=template.bundle.release.data_path.parent,
                           audit_root=tmp_path / "audits")
    # Synthetic dates cannot satisfy the production publication's full-year rule.
    # Every downstream identity/snapshot/normalization/checkpoint gate is retained.
    monkeypatch.setattr(recovery, "read_earth_3hourly_release", lambda *a, **k: case.release)
    return case


def _verify(case):
    return recovery.verify_recovery(case.database, TASK_ID, case.package_dir,
                                    expected_checkpoint_sha256=file_sha256(case.path))


def _recover(case, **overrides):
    arguments = dict(expected_checkpoint_sha256=file_sha256(case.path), audit_root=case.audit_root)
    arguments.update(overrides)
    return recovery.recover_task(case.database, TASK_ID, case.package_dir, **arguments)


@pytest.fixture
def reload_timeout_case(recovery_case):
    case = recovery_case
    hypers = copy.deepcopy(case.hypers)
    hypers.update(_earth_metrics_schema=EARTH_3HOURLY_METRICS_SCHEMA_V3,
                  _earth_metrics_aggregation=CELL_AREA_WEIGHTED_AGGREGATION)
    _update_task(case, hyperparameters=json.dumps(hypers), progress=99.,
                 metrics=json.dumps({"error_code": "invalid_earth_training_artifact",
                                     "error": recovery.MODEL_RELOAD_TIMEOUT_MESSAGE}))
    case.log.write_text("Earth training completed\nModel saved: " + str(case.path) + "\n" +
                        recovery.MODEL_RELOAD_TIMEOUT_MESSAGE + "\n", encoding="utf-8")
    case.hypers = hypers
    return case


def _fast_gates(case, monkeypatch):
    """Retain the real strict checkpoint loader; bound repetition of proven scans."""
    monkeypatch.setattr(recovery, "_fit_threehour_normalization_uncached",
                        lambda *a, **k: copy.deepcopy(case.payload["normalization"]))
    monkeypatch.setattr(recovery, "verify_earth_model_reload_isolated", lambda *a, **k: None)


def _write_tampered_checkpoint(case, mutate):
    payload = copy.deepcopy(case.payload)
    mutate(payload)
    # Deliberately bypass the writer to model a malformed artifact on disk.
    torch.save(payload, case.path)


def test_full_synthetic_recovery_verification_is_readonly_and_reload_is_isolated(recovery_case):
    case = recovery_case
    before = _rows(case.database)
    digests = [file_sha256(path) for path in (case.database, case.path, case.log, case.source)]
    plan = _verify(case)
    assert plan["metrics"]["schema"] == EARTH_3HOURLY_METRICS_SCHEMA_V3
    assert plan["verification"]["isolated_reload"] == "passed"
    assert plan["verification"]["normalization"] == case.payload["normalization"]
    assert plan["verification"]["cell_weight_sum"] == pytest.approx(1.)
    assert plan["metrics"] == load_earth_training_artifact(case.path).metrics
    assert _rows(case.database) == before
    assert [file_sha256(path) for path in (case.database, case.path, case.log, case.source)] == digests
    assert not case.audit_root.exists()


def test_reload_timeout_verification_requires_explicit_kind_and_keeps_contract(reload_timeout_case):
    case = reload_timeout_case
    before = _rows(case.database)
    with pytest.raises(EarthArtifactError, match="exact metrics-contract failure"):
        _verify(case)
    plan = recovery.verify_recovery(case.database, TASK_ID, case.package_dir,
        expected_checkpoint_sha256=file_sha256(case.path), failure_kind="reload-timeout")
    assert plan['schema'] == recovery.RELOAD_AUDIT_SCHEMA
    assert plan['migrated_hyperparameters'] == case.hypers
    assert plan['verification']['isolated_reload'] == 'passed'
    assert _rows(case.database) == before
    assert not case.audit_root.exists()


def test_reload_timeout_recovery_backs_up_and_preserves_original_contract(reload_timeout_case, monkeypatch):
    case = reload_timeout_case
    _fast_gates(case, monkeypatch)
    before = _rows(case.database)
    digests = {path: file_sha256(path) for path in (case.path, case.source, case.log)}
    result = _recover(case, failure_kind='reload-timeout')
    after = _rows(case.database)
    assert after[1:] == before[1:]
    assert after[0]['status'] == 'completed' and after[0]['progress'] == 100.
    assert after[0]['pid'] is None
    changed = {key for key in before[0] if before[0][key] != after[0][key]}
    assert changed == {'hyperparameters', 'metrics', 'status', 'progress', 'pid'}
    hypers = json.loads(after[0]['hyperparameters'])
    audit = hypers.pop('_earth_reload_recovery')
    assert hypers == case.hypers
    assert audit['original_failure'] == json.loads(before[0]['metrics'])
    assert audit['schema'] == recovery.RELOAD_AUDIT_SCHEMA
    assert audit['from_schema'] == audit['to_schema'] == EARTH_3HOURLY_METRICS_SCHEMA_V3
    assert _rows(result['database_backup']) == before
    assert all(file_sha256(path) == digest for path, digest in digests.items())
    assert load_earth_training_artifact(case.path, expected_hyperparameters=hypers,
        expected_task_id=TASK_ID).metrics == json.loads(after[0]['metrics'])
    with pytest.raises(EarthArtifactError, match='exact reload-timeout failure'):
        _recover(case, failure_kind='reload-timeout')


@pytest.mark.parametrize('damage', ['failure', 'schema', 'metric', 'source', 'reload'])
def test_reload_timeout_recovery_rejects_unverified_evidence(reload_timeout_case, monkeypatch, damage):
    case = reload_timeout_case
    _fast_gates(case, monkeypatch)
    if damage == 'failure':
        _update_task(case, metrics=json.dumps({'error_code':'invalid_earth_training_artifact',
                                              'error':'Stopped by user'}))
    elif damage == 'schema':
        hypers = copy.deepcopy(case.hypers)
        hypers['_earth_metrics_schema'] = EARTH_3HOURLY_METRICS_SCHEMA_V2
        _update_task(case, hyperparameters=json.dumps(hypers))
    elif damage == 'metric':
        _write_tampered_checkpoint(case, lambda p: p['metrics']['splits']['test']['overall'].update(rmse=99.))
    elif damage == 'source':
        case.source.write_text('# changed source\n', encoding='utf-8')
    else:
        def fail(*args, **kwargs):
            raise EarthArtifactError(recovery.MODEL_RELOAD_TIMEOUT_MESSAGE)
        monkeypatch.setattr(recovery, 'verify_earth_model_reload_isolated', fail)
    before = _rows(case.database)
    with pytest.raises(EarthArtifactError):
        _recover(case, failure_kind='reload-timeout')
    assert _rows(case.database) == before
    assert not case.audit_root.exists()


@pytest.mark.parametrize("damage, match", [
    ("sha", "SHA-256"),
    ("task_id", "different task"),
    ("dataset", "different dataset"),
    ("partition", "partition|split|range"),
    ("metric", "rmse|RMSE|consisten"),
    ("aggregation", "aggregation"),
    ("loss", "loss"),
    ("parameters", "training parameters"),
    ("epoch", "epoch/seed"),
    ("normalization", "normalization"),
    ("embedded_source", "source|digest"),
    ("frozen_source", "source content_hash"),
    ("source_file", "source file"),
    ("failure", "exact metrics-contract failure"),
    ("unverified_identity", "exact metrics-contract failure"),
])
def test_invalid_evidence_keeps_task_failed_without_any_database_mutation(
        recovery_case, monkeypatch, damage, match):
    case = recovery_case
    _fast_gates(case, monkeypatch)
    expected = None
    if damage == "sha":
        expected = "0" * 64
    elif damage == "task_id":
        _write_tampered_checkpoint(case, lambda p: p["run"].update(task_id=96))
    elif damage == "dataset":
        _update_task(case, dataset_fingerprint="f" * 64)
    elif damage == "partition":
        hypers = copy.deepcopy(case.hypers)
        hypers["_earth_task_split"]["ranges"]["train"]["raw_end"] -= 1
        _update_task(case, hyperparameters=json.dumps(hypers))
    elif damage == "metric":
        _write_tampered_checkpoint(case, lambda p: p["metrics"]["splits"]["test"]["overall"].update(rmse=99.))
    elif damage == "aggregation":
        _write_tampered_checkpoint(case, lambda p: p["metrics"].update(aggregation="forecast_origin_lead_grid_uniform"))
    elif damage == "loss":
        _write_tampered_checkpoint(case, lambda p: p["run"].update(loss="normalized_mse_grid_uniform"))
    elif damage == "parameters":
        hypers = copy.deepcopy(case.hypers)
        hypers["learning_rate"] = .002
        _update_task(case, hyperparameters=json.dumps(hypers))
    elif damage == "epoch":
        _update_task(case, current_epoch=2)
    elif damage == "normalization":
        _write_tampered_checkpoint(case, lambda p: p["normalization"]["mean"].__setitem__(0, 1.))
    elif damage == "embedded_source":
        _write_tampered_checkpoint(case, lambda p: p["model_ref"]["uploaded_model"].update(source_text="# invalid source\n"))
    elif damage == "frozen_source":
        hypers = copy.deepcopy(case.hypers)
        hypers["_earth_uploaded_reference"]["content_hash"] = "0" * 64
        _update_task(case, hyperparameters=json.dumps(hypers))
    elif damage == "source_file":
        case.source.write_text("# replaced model source\n", encoding="utf-8")
    elif damage == "failure":
        _update_task(case, metrics='{"error": "a different failure"}')
    elif damage == "unverified_identity":
        _update_task(case, dataset_identity_status="legacy_inferred")
    before = _rows(case.database)
    db_digest = file_sha256(case.database)
    with pytest.raises(EarthArtifactError, match=match):
        _recover(case, **({"expected_checkpoint_sha256": expected} if expected else {}))
    assert _rows(case.database) == before
    assert _rows(case.database)[0]["status"] == "failed"
    assert file_sha256(case.database) == db_digest
    assert not case.audit_root.exists()


def test_isolated_reload_failure_does_not_create_backup_or_restore_task(recovery_case, monkeypatch):
    case = recovery_case
    _fast_gates(case, monkeypatch)
    calls = []

    def fail_reload(path, **kwargs):
        calls.append(path)
        raise EarthArtifactError("Synthetic isolated model reload failed")

    monkeypatch.setattr(recovery, "verify_earth_model_reload_isolated", fail_reload)
    before = _rows(case.database)
    db_digest = file_sha256(case.database)
    with pytest.raises(EarthArtifactError, match="isolated model reload failed"):
        _recover(case)
    assert calls == [case.path]
    assert _rows(case.database) == before
    assert file_sha256(case.database) == db_digest
    assert not case.audit_root.exists()


def test_unknown_public_training_parameter_cannot_be_silently_dropped(recovery_case, monkeypatch):
    case = recovery_case
    _fast_gates(case, monkeypatch)
    hypers = copy.deepcopy(case.hypers)
    hypers["unexpected_training_option"] = True
    _update_task(case, hyperparameters=json.dumps(hypers))
    before = _rows(case.database)
    with pytest.raises(ValueError, match="Unsupported Earth training parameter"):
        _recover(case)
    assert _rows(case.database) == before
    assert not case.audit_root.exists()


def test_genuine_equal_weighted_v2_checkpoint_cannot_be_migrated_as_v3(
        recovery_case, recovery_template, monkeypatch):
    case = recovery_case
    _fast_gates(case, monkeypatch)
    v2 = copy.deepcopy(recovery_template.bundle.payload)
    v2["run"].update(hyperparameters=copy.deepcopy(case.payload["run"]["hyperparameters"]),
                     seed=case.payload["run"]["seed"], epochs_completed=3, best_epoch=2)
    v2["model_ref"]["uploaded_model"]["source_path"] = str(case.source)
    save_earth_artifact_atomic(v2, case.path, strict_reload=False)
    assert load_earth_training_artifact(case.path).metrics["schema"] == EARTH_3HOURLY_METRICS_SCHEMA_V2
    before = _rows(case.database)
    with pytest.raises(EarthArtifactError, match="required metrics contract"):
        _recover(case)
    assert _rows(case.database) == before
    assert not case.audit_root.exists()


def test_failed_sqlite_backup_rolls_back_before_status_mutation(recovery_case, monkeypatch):
    case = recovery_case
    _fast_gates(case, monkeypatch)
    before = _rows(case.database)
    digest = file_sha256(case.database)
    original_connect = sqlite3.connect
    backup_calls = []

    class FailingBackup(sqlite3.Connection):
        def backup(self, target, *args, **kwargs):
            backup_calls.append(True)
            raise sqlite3.OperationalError("Synthetic backup failure")

    def connect(*args, **kwargs):
        kwargs.setdefault("factory", FailingBackup)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(recovery.sqlite3, "connect", connect)
    with pytest.raises(sqlite3.OperationalError, match="backup failure"):
        _recover(case)
    assert backup_calls == [True]
    assert _rows(case.database) == before
    assert file_sha256(case.database) == digest


def test_success_uses_sqlite_backup_preserves_history_and_only_changes_target(recovery_case, monkeypatch):
    case = recovery_case
    _fast_gates(case, monkeypatch)
    before = _rows(case.database)
    hashes = {path: file_sha256(path) for path in (case.path, case.log, case.source)}
    checkpoints = load_earth_training_artifact(case.path).metrics
    events = []
    original_connect = sqlite3.connect

    class BackupSpy(sqlite3.Connection):
        def backup(self, target, *args, **kwargs):
            assert _rows(case.database) == before
            events.append("backup")
            return super().backup(target, *args, **kwargs)

        def execute(self, sql, *args, **kwargs):
            if sql.startswith("UPDATE model_training_tasks"):
                assert events == ["backup"]
                events.append("update")
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        kwargs.setdefault("factory", BackupSpy)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(recovery.sqlite3, "connect", connect)
    result = _recover(case)
    after = _rows(case.database)
    assert events == ["backup", "update"]
    assert after[1] == before[1]
    assert after[0]["status"] == "completed" and after[0]["progress"] == 100.
    assert after[0]["pid"] is None
    for key in ("created_at", "start_time", "end_time", "current_epoch", "total_epochs"):
        assert after[0][key] == before[0][key]
    assert json.loads(after[0]["metrics"]) == checkpoints
    hypers = json.loads(after[0]["hyperparameters"])
    assert hypers["_earth_metrics_schema"] == EARTH_3HOURLY_METRICS_SCHEMA_V3
    assert hypers["_earth_metrics_aggregation"] == CELL_AREA_WEIGHTED_AGGREGATION
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["original_task"] == before[0]
    assert hypers["_earth_metrics_recovery"]["original_failure"] == json.loads(before[0]["metrics"])
    assert _rows(result["database_backup"]) == before
    assert Path(audit["log_backup"]).read_bytes() == case.log.read_bytes()
    assert all(file_sha256(path) == digest for path, digest in hashes.items())
    # The migrated task remains strictly readable through the ordinary loader.
    assert load_earth_training_artifact(case.path, expected_hyperparameters=hypers,
                                        expected_task_id=TASK_ID).metrics == checkpoints
    with pytest.raises(EarthArtifactError, match="exact metrics-contract failure"):
        _recover(case)
    assert _rows(case.database) == after


@pytest.mark.parametrize("changed", ["task", "checkpoint", "log", "source", "package"])
def test_changed_evidence_after_verification_is_rejected(recovery_case, monkeypatch, changed):
    case = recovery_case
    _fast_gates(case, monkeypatch)
    plan = _verify(case)
    if changed == "task":
        _update_task(case, progress=91.)
    elif changed == "checkpoint":
        with case.path.open("ab") as output:
            output.write(b"changed after verification")
    elif changed == "log":
        case.log.write_text("changed after verification", encoding="utf-8")
    elif changed == "source":
        case.source.write_text("changed after verification", encoding="utf-8")
    else:
        monkeypatch.setattr(recovery, "package_signature", lambda *a, **k: ("changed",))
    monkeypatch.setattr(recovery, "verify_recovery", lambda *a, **k: plan)
    before = _rows(case.database)
    digest = file_sha256(case.database)
    with pytest.raises(EarthArtifactError, match="changed"):
        _recover(case)
    assert _rows(case.database) == before
    assert _rows(case.database)[0]["status"] == "failed"
    assert file_sha256(case.database) == digest
