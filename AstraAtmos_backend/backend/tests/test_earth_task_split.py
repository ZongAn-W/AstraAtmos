"""Task partitions: raw timeline, isolation, persisted contracts and backtesting."""

import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import torch

from services.dataset_identity import DatasetRequestError
from services.earth_dataset import (EarthThreeHourlyWindows, fit_threehour_normalization,
                                    build_threehour_training_cache)
from services.earth_dataset_metadata import package_signature
from services.earth_task_split import (EARTH_TASK_SPLIT_POLICY, build_earth_task_split,
                                      validate_earth_task_split, timeline_from_snapshot,
                                      TASK_SPLIT_POLICY_KEY, required_task_split)
from services.earth_training_contract import normalize_earth_training_hyperparameters, build_earth_training_spec
from services.earth_training_artifact import (EarthArtifactError, load_earth_training_artifact,
                                             validate_checkpoint_payload, threehour_origin_split,
                                             threehour_origin_index)
from services.earth_prediction_service import build_prediction_context, compare_earth_test_metrics
from services.training_split import TrainingSplitError
from models.training_scripts.earth_daily import run_training
from test_earth_3hourly_training_runner import synthetic_training_release, registry_for
from test_earth_3hourly_training_service import service_environment
from test_earth_3hourly_uploaded import package_at, source_with_spec, spec as upload_spec
from test_earth_3hourly_artifact import _payload

DATASET_ID = "earth_merra2_3hourly_v1"
RATIOS = {"train_ratio": .34, "validation_ratio": .33, "test_ratio": .33}


def dates(count=5848):
    return np.datetime64("2020-01-01T01:30", "ns") + np.arange(count) * np.timedelta64(3, "h")


@pytest.mark.parametrize("ratios,sizes", [({}, [4094, 1169, 585]),
    ({"train_ratio": .6, "validation_ratio": .25, "test_ratio": .15}, [3509, 1462, 877])])
def test_raw_allocation_covers_the_release_and_keeps_all_windows_inside(ratios, sizes):
    split = build_earth_task_split(dates(), 56, 24, ratios)
    assert split == build_earth_task_split(dates(), 56, 24, ratios)
    assert [item["step_count"] for item in split["ranges"].values()] == sizes
    cursor = 0
    occupied = set()
    for item in split["ranges"].values():
        assert item["raw_start"] == cursor
        steps = set(range(item["raw_start"], item["raw_end"]))
        assert not occupied.intersection(steps)
        occupied.update(steps)
        starts = range(item["raw_start"], item["raw_end"] - 79)
        assert len(starts) == item["window_count"]
        for start in starts:
            assert start >= item["raw_start"] and start + 80 <= item["raw_end"]
        cursor = item["raw_end"]
    assert occupied == set(range(5848))


@pytest.mark.parametrize("key", ["train_ratio", "validation_ratio", "test_ratio"])
@pytest.mark.parametrize("value", [None, True, "0.7", [], {}, float("nan"), float("inf"), -1, 0])
def test_invalid_ratios_are_parameter_errors(key, value):
    with pytest.raises(DatasetRequestError) as error:
        normalize_earth_training_hyperparameters({key: value}, dataset_id=DATASET_ID)
    assert error.value.status_code == 422


def test_sum_and_short_partition_do_not_silently_change_ratios():
    with pytest.raises(TrainingSplitError, match="sum to 1"):
        build_earth_task_split(dates(), 56, 24, {"train_ratio": .6})
    with pytest.raises(TrainingSplitError, match="validation partition has 48 time steps"):
        build_earth_task_split(dates(240), 56, 24, {})
    with pytest.raises(TrainingSplitError, match="continuous"):
        build_earth_task_split(np.delete(dates(), 10), 56, 24, {})


@pytest.mark.parametrize("value", [[], False, "", 0])
def test_corrupt_empty_task_parameters_are_not_legacy(value):
    with pytest.raises(TrainingSplitError, match="hyperparameters must be an object"):
        required_task_split(value)


@pytest.fixture(scope="module", params=["official", "uploaded"])
def custom_smoke(request, synthetic_training_release, tmp_path_factory):
    release = synthetic_training_release
    registry = registry_for(release)
    binding = registry.build_training_binding(DATASET_ID)
    root = tmp_path_factory.mktemp("task-split-" + request.param)
    split = build_earth_task_split(release.dates, 55, 24, RATIOS)
    reference = None
    if request.param == "uploaded":
        declaration = upload_spec()
        declaration["datasets"][DATASET_ID]["window"] = [55, 56]
        source = source_with_spec(declaration)
        package, validation = package_at(root, source)
        assert validation.ok
        reference = {"package_id": package.id, "display_name": package.display_name, "version": 1,
                     "content_hash": package.content_hash, "source_path": package.storage_path,
                     "source_text": source, "param_schema": validation.param_schema,
                     "custom_model_params": {"bias": True}}
    spec = build_earth_training_spec(task_id=1201, dataset_binding=binding, task_split=split,
        hyperparameters={**RATIOS, "window": 55, "horizon": 24, "epochs": 1, "batch_size": 8,
                         "selected_channels": ["U10M"]}, uploaded_model=reference,
        metrics_contract={"schema": "earth_training_metrics_3hourly_v2",
                          "aggregation": "forecast_origin_lead_grid_uniform"})
    output = root / "custom.pth"
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        result = run_training(spec, output, registry, device="cpu", cache_root=root / "cache")
    finally:
        torch.set_num_threads(previous)
    hypers = {**spec["hyperparameters"], "_earth_task_split": split}
    task = SimpleNamespace(id=1201, **binding, status="completed", output_model_path=str(output),
                           hyperparameters=json.dumps(hypers), custom_model_name=request.param)
    return SimpleNamespace(task=task, output=output, split=split, registry=registry,
                           result=result, release=release, hypers=hypers)


def test_both_model_sources_train_custom_windows_and_strictly_reload(custom_smoke):
    smoke = custom_smoke
    checkpoint = load_earth_training_artifact(smoke.output, expected_binding=smoke.registry.build_training_binding(DATASET_ID),
                                             expected_hyperparameters=smoke.hypers, expected_task_id=1201)
    assert checkpoint.training_contract["split_policy"] == EARTH_TASK_SPLIT_POLICY
    assert checkpoint.payload["run"]["task_split_policy"] == EARTH_TASK_SPLIT_POLICY
    assert smoke.result["split_window_counts"] == {"train": 4, "validation": 1, "test": 1}
    assert checkpoint.normalization["fit_step_count"] == 82
    assert checkpoint.normalization["fit_time_end"] == smoke.split["ranges"]["train"]["date_end"]
    assert checkpoint.dataset_binding["dataset_snapshot"]["splits"] == smoke.release.metadata["splits"]


@pytest.mark.parametrize("mutation", ["missing", "unknown_policy", "boundary", "bool_index", "windows", "normalization", "metrics", "ratios", "missing_ratios", "normalization_strategy", "range_type", "run_ratios", "run_window_type", "aggregation"])
def test_new_checkpoint_metadata_is_never_treated_as_legacy(custom_smoke, mutation):
    payload = copy.deepcopy(load_earth_training_artifact(custom_smoke.output).payload)
    contract = payload["training_contract"]
    if mutation == "missing":
        contract.pop("task_split")
    elif mutation == "unknown_policy":
        contract["split_policy"] = "unknown"
    elif mutation == "boundary":
        contract["task_split"]["ranges"]["train"]["raw_end"] -= 1
    elif mutation == "bool_index":
        contract["task_split"]["ranges"]["train"]["raw_start"] = False
    elif mutation == "windows":
        contract["split_window_counts"]["test"] += 1
    elif mutation == "normalization":
        payload["normalization"]["fit_time_end"] = contract["split_ranges"]["validation"]["date_end"]
    elif mutation == "metrics":
        payload["metrics"]["splits"]["test"]["window_count"] += 1
    elif mutation == "missing_ratios":
        contract.pop("split_ratios")
    elif mutation == "normalization_strategy":
        payload["normalization"]["task_split"]["ranges"]["train"]["raw_start"] = False
    elif mutation == "range_type":
        contract["split_ranges"]["train"]["raw_start"] = False
    elif mutation == "run_ratios":
        payload["run"]["hyperparameters"]["train_ratio"] = .5
    elif mutation == "run_window_type":
        payload["run"]["hyperparameters"]["window"] = 55.0
    elif mutation == "aggregation":
        payload["metrics"]["aggregation"] = "sample_subset"
    else:
        contract["split_ratios"] = {"train_ratio": .4, "validation_ratio": .3, "test_ratio": .3}
    with pytest.raises((EarthArtifactError, ValueError)):
        validate_checkpoint_payload(payload)


def test_task_checkpoint_ratio_or_fixed_split_mismatch_is_rejected(custom_smoke):
    for hypers in ({**custom_smoke.hypers, "train_ratio": .5},
                   {key: value for key, value in custom_smoke.hypers.items() if key != "_earth_task_split"}):
        with pytest.raises(EarthArtifactError):
            load_earth_training_artifact(custom_smoke.output, expected_hyperparameters=hypers)


def test_existing_custom_artifact_without_run_marker_keeps_its_task_contract(custom_smoke, tmp_path):
    payload = copy.deepcopy(load_earth_training_artifact(custom_smoke.output).payload)
    payload["run"].pop("task_split_policy")
    path = tmp_path / "existing-custom.pth"
    torch.save(payload, path)
    hypers = {**custom_smoke.hypers, "_earth_metrics_schema": "earth_training_metrics_3hourly_v2"}
    checkpoint = load_earth_training_artifact(path, expected_hyperparameters=hypers)
    assert checkpoint.training_contract["task_split"] == custom_smoke.split


def test_origin_labels_and_uploaded_cross_partition_rule_use_task_boundaries(custom_smoke):
    smoke = custom_smoke
    # The last origin must leave the complete target window inside train;
    # the next origin crosses into validation and is rejected for both sources.
    train_end = smoke.split["ranges"]["train"]["raw_end"]
    origin = np.datetime_as_string(smoke.release.dates[train_end - 24 - 1], unit="s") + "Z"
    crossing = np.datetime_as_string(smoke.release.dates[train_end - 24], unit="s") + "Z"
    assert threehour_origin_split(smoke.release, origin, window=55, horizon=24, task_split=smoke.split) == "train"
    with pytest.raises(EarthArtifactError) as error:
        threehour_origin_split(smoke.release, crossing, window=55, horizon=24, task_split=smoke.split)
    assert error.value.code == "earth_prediction_origin_out_of_range"
    context = build_prediction_context(smoke.task, smoke.registry)
    assert context.metrics["task_split"] == smoke.split
    assert context.training_split_end == smoke.split["ranges"]["train"]["date_end"]
    assert context.origins["count"] == 6
    assert threehour_origin_index(smoke.release, origin, window=55, horizon=24,
                                  strict_split=True, task_split=smoke.split) == train_end - 24 - 1


def test_legacy_threehour_checkpoint_remains_a_manifest_contract():
    payload, _, _ = _payload()
    assert validate_checkpoint_payload(payload)["training_contract"]["split_policy"] == "published_manifest_splits"
    assert "task_split" not in payload["training_contract"]


def test_changed_validation_and_test_data_cannot_affect_task_normalization(synthetic_training_release, tmp_path):
    import shutil
    release = synthetic_training_release
    root = tmp_path / "copy"
    root.mkdir()
    path = root / release.data_path.name
    shutil.copyfile(release.data_path, path)
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    modified = replace(release, data_path=path, signature=package_signature(root, data_file_name=path.name))
    split = build_earth_task_split(release.dates, 55, 24, RATIOS)
    expected = fit_threehour_normalization(release, ["TO3"], task_split=split)
    with netCDF4.Dataset(path, "r+") as ds:
        ds["TO3"][82:] = np.float32(500000)
    modified = replace(modified, signature=package_signature(root, data_file_name=path.name))
    assert fit_threehour_normalization(modified, ["TO3"], task_split=split) == expected


def test_cache_reuses_same_train_statistics_with_different_validation_test_boundaries(synthetic_training_release, tmp_path):
    release = synthetic_training_release
    first = build_earth_task_split(release.dates, 2, 1, {"train_ratio": .5, "validation_ratio": .25, "test_ratio": .25})
    second = build_earth_task_split(release.dates, 2, 1, {"train_ratio": .5, "validation_ratio": .3, "test_ratio": .2})
    one = fit_threehour_normalization(release, ["TO3"], task_split=first)
    two = fit_threehour_normalization(release, ["TO3"], task_split=second)
    assert one["mean"] == two["mean"] and one["scale"] == two["scale"]
    cache = build_threehour_training_cache(release, ["TO3"], one, tmp_path)
    dataset = EarthThreeHourlyWindows.from_release(release, window=2, horizon=1,
        selected_channels=[], normalization=two, task_split=second)
    dataset.use_training_cache(cache)
    assert dataset.normalization['task_split'] == second
    assert build_threehour_training_cache(release, ['TO3'], two, tmp_path) == cache


def test_dataset_rejects_a_fixed_partition_for_different_window_settings(synthetic_training_release):
    release = synthetic_training_release
    split = build_earth_task_split(release.dates, 2, 1, RATIOS)
    with pytest.raises(TrainingSplitError, match="window counts are inconsistent"):
        EarthThreeHourlyWindows.from_release(release, window=3, horizon=1, task_split=split)


def test_queue_restart_keeps_fixed_split_and_rejects_corruption(service_environment, monkeypatch):
    module, registry = service_environment
    service = module.TrainingService()
    service._scheduler_started = True
    task = asyncio.run(service.start_training(user_id=1, custom_model_name="custom queue", model_script="ignored.py",
        hyperparameters={"train_ratio": .6, "validation_ratio": .25, "test_ratio": .15},
        dataset_id=DATASET_ID, dataset_registry=registry))
    plan = asyncio.run(service._prepare_training_execution(task, dataset_registry=registry))
    original = plan["earth_training_spec"]
    monkeypatch.setattr(service, "_default_dataset_registry", lambda: registry)
    restored = service._restore_3hourly_training_spec(task)
    assert restored["task_split"] == original["task_split"]
    raw = json.loads(task.hyperparameters)
    assert raw[TASK_SPLIT_POLICY_KEY] == EARTH_TASK_SPLIT_POLICY
    raw["_earth_task_split"]["ranges"]["test"].pop("window_count")
    task.hyperparameters = json.dumps(raw)
    with pytest.raises(DatasetRequestError):
        service._restore_3hourly_training_spec(task)
    raw.pop("_earth_task_split")
    task.hyperparameters = json.dumps(raw)
    with pytest.raises(DatasetRequestError, match="required frozen partition"):
        service._restore_3hourly_training_spec(task)
    # A real legacy queue entry has neither new-task version marker.
    raw.pop(TASK_SPLIT_POLICY_KEY)
    raw.pop("_earth_metrics_schema")
    raw.pop("_earth_metrics_aggregation")
    task.hyperparameters = json.dumps(raw)
    assert "task_split" not in service._restore_3hourly_training_spec(task)


@pytest.mark.parametrize("mutation", ["missing_split", "null_split", "unknown_policy", "null_policy", "unknown_metrics", "old_v2_missing_split"])
def test_new_queue_metadata_cannot_downgrade_to_manifest(service_environment, monkeypatch, mutation):
    module, registry = service_environment
    service = module.TrainingService()
    service._scheduler_started = True
    task = asyncio.run(service.start_training(user_id=1, custom_model_name="strict queue " + mutation,
        model_script="ignored.py", hyperparameters={}, dataset_id=DATASET_ID, dataset_registry=registry))
    raw = json.loads(task.hyperparameters)
    if mutation in ("missing_split", "old_v2_missing_split"):
        raw.pop("_earth_task_split")
        if mutation == "old_v2_missing_split":
            raw.pop(TASK_SPLIT_POLICY_KEY)
    elif mutation == "null_split":
        raw["_earth_task_split"] = None
    elif mutation in ("unknown_policy", "null_policy"):
        raw[TASK_SPLIT_POLICY_KEY] = "future_policy" if mutation == "unknown_policy" else None
    else:
        raw["_earth_metrics_schema"] = "future_metrics"
    task.hyperparameters = json.dumps(raw)
    monkeypatch.setattr(service, "_default_dataset_registry", lambda: pytest.fail("reject before reading the release"))
    with pytest.raises(DatasetRequestError) as error:
        service._restore_3hourly_training_spec(task)
    assert error.value.status_code == 409


def test_existing_custom_queue_without_policy_marker_keeps_frozen_split(service_environment, monkeypatch):
    module, registry = service_environment
    service = module.TrainingService()
    service._scheduler_started = True
    task = asyncio.run(service.start_training(user_id=1, custom_model_name="existing custom queue",
        model_script="ignored.py", hyperparameters={}, dataset_id=DATASET_ID, dataset_registry=registry))
    raw = json.loads(task.hyperparameters)
    raw.pop(TASK_SPLIT_POLICY_KEY)
    task.hyperparameters = json.dumps(raw)
    monkeypatch.setattr(service, "_default_dataset_registry", lambda: registry)
    assert service._restore_3hourly_training_spec(task)["task_split"] == raw["_earth_task_split"]


@pytest.mark.parametrize("marker", ["policy", "v2", "unknown_policy", "unknown_metrics"])
def test_new_task_cannot_read_a_manifest_checkpoint(tmp_path, marker):
    payload, _, _ = _payload()
    path = tmp_path / "manifest.pth"
    torch.save(payload, path)
    hypers = {"training_dataset": DATASET_ID, "window": 56, "horizon": 24}
    if marker in ("policy", "unknown_policy"):
        hypers[TASK_SPLIT_POLICY_KEY] = EARTH_TASK_SPLIT_POLICY if marker == "policy" else "future_policy"
    else:
        hypers["_earth_metrics_schema"] = "earth_training_metrics_3hourly_v2" if marker == "v2" else "future_metrics"
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(path, expected_hyperparameters=hypers)


@pytest.mark.parametrize("policy", ["future_policy", "published_manifest_splits", None])
def test_checkpoint_run_marker_cannot_disagree_with_task_policy(custom_smoke, policy):
    payload = copy.deepcopy(load_earth_training_artifact(custom_smoke.output).payload)
    payload["run"]["task_split_policy"] = policy
    with pytest.raises(EarthArtifactError, match="run split policy"):
        validate_checkpoint_payload(payload)


def test_checkpoint_run_marker_prevents_manifest_downgrade():
    payload, _, _ = _payload()
    payload["run"]["task_split_policy"] = EARTH_TASK_SPLIT_POLICY
    with pytest.raises(EarthArtifactError, match="run split policy"):
        validate_checkpoint_payload(payload)


@pytest.mark.parametrize("marker,expected", [("policy", "failed"), ("v2", "failed"), (None, "completed")])
def test_restart_completion_rejects_new_tasks_without_frozen_split(service_environment, tmp_path, marker, expected):
    from database.models import ModelTrainingTask
    module, _ = service_environment
    payload, _, _ = _payload()
    path = tmp_path / "restart.pth"
    torch.save(payload, path)
    log = tmp_path / "restart.log"
    log.write_text(f"Model saved: {path}\n", encoding="utf-8")
    binding = copy.deepcopy(payload["dataset_binding"])
    binding["dataset_snapshot"] = json.dumps(binding["dataset_snapshot"])
    hypers = {"training_dataset": DATASET_ID, "window": 56, "horizon": 24,
              "selected_channels": ["U10M"], "model_source": "official"}
    if marker == "policy":
        hypers[TASK_SPLIT_POLICY_KEY] = EARTH_TASK_SPLIT_POLICY
    elif marker == "v2":
        hypers["_earth_metrics_schema"] = "earth_training_metrics_3hourly_v2"

    async def scenario():
        async with module.async_session_maker() as session:
            session.add(ModelTrainingTask(id=22, user_id=1, model_script="earth_daily.py", model_source="official",
                status="running", output_model_path=str(path), log_file_path=str(log),
                hyperparameters=json.dumps(hypers), **binding))
            await session.commit()
        service = module.TrainingService()
        await service.start()
        await service.stop()
        return await service.get_task(22)

    task = asyncio.run(scenario())
    assert task.status == expected
    if expected == "completed":
        assert json.loads(task.metrics) == payload["metrics"]
    else:
        assert json.loads(task.metrics)["error_code"] == "invalid_earth_training_artifact"


@pytest.mark.parametrize("mutation", ["task_missing_split", "task_and_spec_missing_split", "all_task_markers_missing", "unknown_policy"])
def test_exit_zero_cannot_complete_after_frozen_task_metadata_is_lost(service_environment, monkeypatch, mutation):
    module, registry = service_environment
    service = module.TrainingService()
    service._scheduler_started = True
    payload, _, _ = _payload()
    artifact_reads = []

    def unexpected_artifact_read(*args, **kwargs):
        artifact_reads.append(args)
        raise EarthArtifactError("Corrupt task must be rejected before artifact loading")

    monkeypatch.setattr(module, "load_earth_training_artifact", unexpected_artifact_read)

    class Process:
        pid = 921
        returncode = 0
        def __init__(self):
            self.stdout = self
        def readline(self):
            return ""
        def close(self):
            pass
        def wait(self):
            return 0

    def popen(args, **kwargs):
        spec = json.loads(kwargs["env"]["ASTRAATMOS_EARTH_TRAINING_SPEC"])
        output = copy.deepcopy(payload)
        output["run"]["task_id"] = spec["task_id"]
        torch.save(output, args[-1])
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    async def scenario():
        task = await service.start_training(user_id=1, custom_model_name="split completion " + mutation,
            model_script="ignored.py", hyperparameters={}, dataset_id=DATASET_ID, dataset_registry=registry)
        plan = await service._prepare_training_execution(task, dataset_registry=registry)
        async with module.async_session_maker() as session:
            row = await session.get(module.ModelTrainingTask, task.id)
            raw = json.loads(row.hyperparameters)
            if mutation == "unknown_policy":
                raw[TASK_SPLIT_POLICY_KEY] = "future_policy"
            else:
                raw.pop("_earth_task_split")
                if mutation == "all_task_markers_missing":
                    raw.pop(TASK_SPLIT_POLICY_KEY)
                    raw.pop("_earth_metrics_schema")
            row.hyperparameters = json.dumps(raw)
            await session.commit()
        if mutation == "task_and_spec_missing_split":
            plan["earth_training_spec"].pop("task_split")
        await service._run_training_subprocess(task.id, **plan)
        return await service.get_task(task.id)

    task = asyncio.run(scenario())
    assert task.status == "failed"
    assert json.loads(task.metrics)["error_code"] == "invalid_earth_training_artifact"
    assert artifact_reads == []


def test_short_partition_is_rejected_before_a_database_session(service_environment, monkeypatch):
    module, registry = service_environment
    original = registry.build_training_binding
    def small_binding(dataset_id):
        binding = original(dataset_id)
        snapshot = json.loads(binding["dataset_snapshot"])
        snapshot["time"].update(count=240, end="2020-01-30T22:30:00Z")
        binding["dataset_snapshot"] = json.dumps(snapshot)
        return binding
    registry.build_training_binding = small_binding
    monkeypatch.setattr(module, "async_session_maker", lambda: pytest.fail("must reject before DB"))
    with pytest.raises(DatasetRequestError, match="validation partition") as error:
        asyncio.run(module.TrainingService().start_training(user_id=1, custom_model_name="short",
            model_script="ignored.py", hyperparameters={}, dataset_id=DATASET_ID, dataset_registry=registry))
    assert error.value.status_code == 422


def test_comparison_rejects_different_test_ranges_before_ranking(custom_smoke, monkeypatch):
    import services.earth_prediction_service as module
    checkpoint = load_earth_training_artifact(custom_smoke.output)
    another = copy.deepcopy(checkpoint)
    another.training_contract["split_ranges"]["test"]["date_start"] = "2020-01-22T01:30:00Z"
    tasks = [custom_smoke.task, copy.copy(custom_smoke.task)]
    tasks[1].id = 1202
    monkeypatch.setattr(module, "_load_checkpoint", lambda task: checkpoint if task.id == 1201 else another)
    monkeypatch.setattr(module, "_validate_checkpoint_split_ranges", lambda *_: None)
    with pytest.raises(DatasetRequestError, match="different test time ranges") as error:
        compare_earth_test_metrics(tasks, custom_smoke.registry)
    assert error.value.code == "earth_comparison_incompatible"
