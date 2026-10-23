"""Mars training/checkpoint/task/prediction identities must form one contract."""

import asyncio
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import netCDF4
import numpy as np
import pytest
import torch

from services import inference_service as inference_module
from services.dataset_identity import DatasetRequestError
from services.mars_checkpoint import build_mars_checkpoint, mars_checkpoint_identity_snapshot
from services.mars_data_service import assert_identity_current, identity_snapshot, load_mars_arrays, prepare_scaled_volume
from services.mars_dataset_identity import canonical_dataset_identity


RATIOS = {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}
DATASETS = ["mcd_overview", "openmars_mcd"]


@pytest.fixture
def mars_sources(tmp_path):
    directories = {name: tmp_path / name for name in ("raw", "openmars", "mcd")}
    for directory in directories.values():
        directory.mkdir()
    paths = {name: directory / f"{name}_MY27.nc" for name, directory in directories.items()}
    for name, path in paths.items():
        height, width = (37, 72) if name == "raw" else (2, 3)
        with netCDF4.Dataset(str(path), "w") as dataset:
            for dimension, size in (("time", 12), ("hour", 1), ("lat", height), ("lon", width)):
                dataset.createDimension(dimension, size)
            dataset.createVariable("lat", "f4", ("lat",))[:] = np.linspace(90, -90, height)
            dataset.createVariable("lon", "f4", ("lon",))[:] = np.linspace(-180, 175, width)
            dataset.createVariable("Ls", "f4", ("time",))[:] = np.arange(12) * 5
            field = np.arange(12 * height * width, dtype=np.float32).reshape(12, height, width) + 10
            if name == "mcd":
                dataset.createVariable("U_Wind", "f4", ("time", "hour", "lat", "lon"))[:] = field[:, None]
            else:
                target = dataset.createVariable("O3COL" if name == "raw" else "o3col", "f4", ("time", "lat", "lon"))
                target[:] = field
                target.units = "um-atm"
                if name == "raw":
                    dataset.createVariable("U", "f4", ("time", "lat", "lon"))[:] = field + 100
    return SimpleNamespace(directories=directories, paths=paths, tmp_path=tmp_path)


def _kwargs(sources, dataset_id):
    return dict(training_dataset=dataset_id, openmars_dir=sources.directories["openmars"],
                mcd_dir=sources.directories["mcd"], raw_dir=sources.directories["raw"], selected_channels=["U"])


def _artifact(sources, dataset_id):
    kwargs = _kwargs(sources, dataset_id)
    volume = prepare_scaled_volume(**kwargs, window=2, horizon=2, split_ratios=RATIOS)
    payload = build_mars_checkpoint(model_state_dict={"bias": torch.zeros(1)}, volume=volume,
                                    selected_channels=["U"], window=2, horizon=2, split_ratios=RATIOS, seed=11)
    return volume, payload


def _task_and_service(sources, dataset_id, monkeypatch, *, snapshot=True):
    _, payload = _artifact(sources, dataset_id)
    path = sources.tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    hypers = dict(training_dataset=dataset_id, window=2, horizon=2, selected_channels=["U"], **RATIOS)
    task = SimpleNamespace(id=42, user_id=7, status="completed", dataset_id=dataset_id,
                           model_source="official", model_script="demo3.py", custom_model_name="probe",
                           output_model_path=str(path), hyperparameters=json.dumps(hypers),
                           dataset_snapshot=json.dumps(mars_checkpoint_identity_snapshot(payload)) if snapshot else None,
                           dataset_fingerprint=payload["data_binding"]["file_fingerprint"], dataset_identity_status="verified")
    service = inference_module.InferenceService()
    service.device = torch.device("cpu")
    service.openmars_dir = sources.directories["openmars"]
    service.mcd_dir = sources.directories["mcd"]
    monkeypatch.setattr(inference_module, "MCD_RAW_3H_DIR", sources.directories["raw"])
    return task, hypers, service, payload


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_four_stages_use_exactly_the_same_fingerprint(mars_sources, dataset_id):
    kwargs = _kwargs(mars_sources, dataset_id)
    arrays = load_mars_arrays(**kwargs)
    volume, payload = _artifact(mars_sources, dataset_id)
    snapshot = identity_snapshot(**kwargs, window=2, horizon=2, split_ratios=RATIOS, normalization=volume)
    current = assert_identity_current(snapshot, openmars_dir=kwargs["openmars_dir"],
                                      mcd_dir=kwargs["mcd_dir"], raw_dir=kwargs["raw_dir"])
    assert arrays.fingerprint == snapshot["dataset_fingerprint"] == payload["data_binding"]["file_fingerprint"] == current["dataset_fingerprint"]
    assert volume.fingerprint == current["dataset_fingerprint"]
    assert list(arrays.manifest) == snapshot["file_manifest"] == payload["data_binding"]["file_manifest"] == current["file_manifest"]
    assert list(map(str, arrays.data_directories)) == current["data_directories"]
    assert mars_checkpoint_identity_snapshot(payload) == snapshot


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_existing_file_identity_serialization_is_preserved(mars_sources, dataset_id):
    kwargs = _kwargs(mars_sources, dataset_id)
    identity = canonical_dataset_identity(**{key: value for key, value in kwargs.items() if key != "selected_channels"})
    if dataset_id == "mcd_overview":
        previous_payload = {"path": str(kwargs["raw_dir"].resolve()), "files": identity["file_manifest"]}
        assert identity["data_directories"] == [str(kwargs["raw_dir"].resolve())]
    else:
        previous_payload = {"openmars_dir": str(kwargs["openmars_dir"].resolve()),
                            "mcd_dir": str(kwargs["mcd_dir"].resolve()), "files": identity["file_manifest"]}
        assert identity["data_directories"] == [str(kwargs["openmars_dir"].resolve()), str(kwargs["mcd_dir"].resolve())]
        assert {entry["root"] for entry in identity["file_manifest"]} == {"openmars", "mcd"}
    previous_fingerprint = hashlib.sha256(json.dumps(previous_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert identity["dataset_fingerprint"] == previous_fingerprint


def test_same_filename_in_two_roots_keeps_both_manifest_entries(mars_sources):
    for root in ("openmars", "mcd"):
        shutil.copy2(mars_sources.paths[root], mars_sources.directories[root] / "shared.nc")
    identity = canonical_dataset_identity(training_dataset="openmars_mcd", openmars_dir=mars_sources.directories["openmars"], mcd_dir=mars_sources.directories["mcd"])
    assert [(entry["root"], entry["name"]) for entry in identity["file_manifest"] if entry["name"] == "shared.nc"] == [("openmars", "shared.nc"), ("mcd", "shared.nc")]


@pytest.mark.parametrize("dataset_id", DATASETS)
@pytest.mark.parametrize("snapshot_format", ["canonical", "binding_aliases", "checkpoint_only", "no_primary_dir"])
def test_prepare_task_data_env_verifies_both_datasets_without_legacy_fallback(mars_sources, dataset_id, snapshot_format, monkeypatch):
    task, hypers, service, payload = _task_and_service(mars_sources, dataset_id, monkeypatch)
    if snapshot_format == "binding_aliases":
        binding = payload["data_binding"]
        task.dataset_snapshot = json.dumps({key: binding[key] for key in (
            "dataset_id", "planet", "data_source_type", "data_directories", "file_manifest", "file_fingerprint",
        )})
    elif snapshot_format == "checkpoint_only":
        task.dataset_snapshot = None
    elif snapshot_format == "no_primary_dir":
        snapshot = json.loads(task.dataset_snapshot)
        snapshot.pop("data_dir")
        task.dataset_snapshot = json.dumps(snapshot)
    spy = Mock(wraps=inference_module.assert_identity_current)
    monkeypatch.setattr(inference_module, "assert_identity_current", spy)
    directories, temporary = asyncio.run(service._prepare_task_data_env(task, hypers))
    assert spy.call_count >= 1
    assert hypers["_dataset_identity_status"] == "verified" and temporary is None
    if dataset_id == "mcd_overview":
        assert directories == {"MCD_RAW_3H_DIR": str(mars_sources.directories["raw"])}
    else:
        assert directories == {"ASTRAATMOS_OPENMARS_DIR": str(mars_sources.directories["openmars"]), "ASTRAATMOS_MCD_DIR": str(mars_sources.directories["mcd"])}


def _change_file(sources, root, action):
    path = sources.paths[root]
    if action == "add":
        shutil.copy2(path, sources.directories[root] / f"added_{root}_MY28.nc")
    elif action == "delete":
        path.unlink()  # One explicitly selected fixture file.
    else:
        original = path.stat()
        with netCDF4.Dataset(str(path), "a") as dataset:
            name = "O3COL" if root == "raw" else "o3col" if root == "openmars" else "U_Wind"
            dataset.variables[name][0] = dataset.variables[name][0] + 1
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))


@pytest.mark.parametrize("dataset_id,root", [("mcd_overview", "raw"), ("openmars_mcd", "openmars"), ("openmars_mcd", "mcd")])
@pytest.mark.parametrize("action", ["modify", "add", "delete"])
@pytest.mark.parametrize("model_source", ["official", "uploaded"])
@pytest.mark.parametrize("entry", ["predict", "test"])
def test_source_changes_block_prediction_before_cache_windows_and_inference(mars_sources, dataset_id, root, action, model_source, entry, monkeypatch):
    task, _, service, _ = _task_and_service(mars_sources, dataset_id, monkeypatch)
    task.model_source = model_source
    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def get(self, model, task_id):
            return task
    monkeypatch.setattr(inference_module, "async_session_maker", lambda: Session())
    cache = AsyncMock()
    monkeypatch.setattr(service, "_cached_prediction", cache)
    guards = []
    for name in ("_load_official_task_volume", "_prepare_uploaded_task_volume", "_run_official_task_model", "_run_uploaded_task_model"):
        guard = Mock(side_effect=AssertionError("Data identity must be checked first"))
        guards.append(guard)
        monkeypatch.setattr(service, name, guard)
    _change_file(mars_sources, root, action)
    with pytest.raises(DatasetRequestError) as raised:
        if entry == "predict":
            asyncio.run(service.predict_task(task.id, 0, 2, current_user=SimpleNamespace(id=7, role="user")))
        else:
            asyncio.run(service.get_test_results(task.id))
    assert raised.value.code == "dataset_version_changed" and raised.value.status_code == 409
    cache.assert_not_called()
    for guard in guards:
        guard.assert_not_called()


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_prediction_route_returns_dataset_version_changed(mars_sources, dataset_id, monkeypatch):
    from fastapi import HTTPException
    from routers import predict
    from schemas.predict import PredictRequest
    task, _, service, _ = _task_and_service(mars_sources, dataset_id, monkeypatch)
    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def get(self, model, task_id):
            return task
    monkeypatch.setattr(inference_module, "async_session_maker", lambda: Session())
    monkeypatch.setattr(predict, "_get_training_inference_service", lambda request: service)
    _change_file(mars_sources, "raw" if dataset_id == "mcd_overview" else "mcd", "modify")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(predict.run_prediction(request, PredictRequest(training_task_id=42, ls_start=0, horizon=2),
                                           current_user=SimpleNamespace(id=7, role="user")))
    assert raised.value.status_code == 409
    assert raised.value.detail["code"] == "dataset_version_changed"


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_bare_state_dict_with_only_registry_metadata_stays_legacy(mars_sources, dataset_id, monkeypatch):
    task, hypers, service, _ = _task_and_service(mars_sources, dataset_id, monkeypatch)
    state = {"weight": torch.ones(1)}
    torch.save(state, task.output_model_path)
    task.dataset_fingerprint = None
    task.dataset_identity_status = "legacy_inferred"
    task.dataset_snapshot = json.dumps({"planet": "mars", "dataset_id": dataset_id, "binding_basis": "historical_default"})
    asyncio.run(service._prepare_task_data_env(task, hypers))
    assert hypers["_dataset_identity_status"] == "legacy"
    assert service._checkpoint_normalization(task) is None
    assert torch.equal(service._load_task_state_dict(task)["weight"], state["weight"])


@pytest.mark.parametrize("dataset_id", DATASETS)
@pytest.mark.parametrize("missing", ["data_directories", "file_manifest", "file_fingerprint"])
def test_new_schema_with_incomplete_binding_is_rejected(mars_sources, dataset_id, missing, monkeypatch):
    task, hypers, service, payload = _task_and_service(mars_sources, dataset_id, monkeypatch, snapshot=False)
    payload["data_binding"].pop(missing)
    torch.save(payload, task.output_model_path)
    with pytest.raises(DatasetRequestError, match="complete data identity"):
        asyncio.run(service._prepare_task_data_env(task, hypers))
    assert hypers.get("_dataset_identity_status") != "legacy"


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_partial_task_metadata_cannot_silently_fall_back_to_legacy(mars_sources, dataset_id, monkeypatch):
    task, hypers, service, _ = _task_and_service(mars_sources, dataset_id, monkeypatch)
    torch.save({"weight": torch.ones(1)}, task.output_model_path)
    task.dataset_fingerprint = None
    task.dataset_identity_status = "legacy_inferred"
    task.dataset_snapshot = json.dumps({"dataset_id": dataset_id, "data_directories": ["incomplete"]})
    with pytest.raises(DatasetRequestError):
        asyncio.run(service._prepare_task_data_env(task, hypers))
    assert hypers.get("_dataset_identity_status") != "legacy"


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_task_fingerprint_conflict_is_rejected(mars_sources, dataset_id, monkeypatch):
    task, hypers, service, _ = _task_and_service(mars_sources, dataset_id, monkeypatch)
    task.dataset_fingerprint = "0" * 64
    with pytest.raises(DatasetRequestError, match="disagrees"):
        asyncio.run(service._prepare_task_data_env(task, hypers))


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_checkpoints_retain_captured_identity_after_source_changes(mars_sources, dataset_id):
    volume, payload = _artifact(mars_sources, dataset_id)
    original = copy.deepcopy(payload["data_binding"])
    _change_file(mars_sources, "raw" if dataset_id == "mcd_overview" else "openmars", "modify")
    rebuilt = build_mars_checkpoint(model_state_dict={}, volume=volume, selected_channels=["U"],
                                    window=2, horizon=2, split_ratios=RATIOS, seed=11)
    assert rebuilt["data_binding"] == original
    kwargs = _kwargs(mars_sources, dataset_id)
    with pytest.raises(DatasetRequestError):
        assert_identity_current(mars_checkpoint_identity_snapshot(rebuilt), openmars_dir=kwargs["openmars_dir"],
                                 mcd_dir=kwargs["mcd_dir"], raw_dir=kwargs["raw_dir"])


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_invalid_checkpoint_fingerprint_returns_version_changed_at_public_entry(mars_sources, dataset_id, monkeypatch):
    task, _, service, payload = _task_and_service(mars_sources, dataset_id, monkeypatch, snapshot=False)
    payload["data_binding"].update(dataset_fingerprint="0" * 64, file_fingerprint="0" * 64)
    torch.save(payload, task.output_model_path)
    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def get(self, model, task_id):
            return task
    monkeypatch.setattr(inference_module, "async_session_maker", lambda: Session())
    cache = AsyncMock()
    monkeypatch.setattr(service, "_cached_prediction", cache)
    with pytest.raises(DatasetRequestError) as raised:
        asyncio.run(service.predict_task(42, 0, 2, current_user=SimpleNamespace(id=7, role="user")))
    assert raised.value.code == "dataset_version_changed" and raised.value.status_code == 409
    cache.assert_not_called()
