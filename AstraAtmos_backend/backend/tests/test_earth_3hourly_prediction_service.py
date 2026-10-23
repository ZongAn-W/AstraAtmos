"""Real global-grid 56-to-24 inference, identity gates and Earth cache isolation.

The fixture uses three contiguous ten-day partitions, as the runner smoke does.
It bypasses the separately tested production manifest dates, while inference
reads real NetCDF fields and restores a real strictly validated checkpoint.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from services.dataset_identity import DatasetRequestError
from services.earth_dataset import fit_threehour_normalization
from services.earth_model_source import ModelSourcePlan, build_earth_model_for_plan, earth_forward_for_model
from services.earth_prediction_cache import EarthPredictionCache, build_earth_prediction_cache_key
from services.earth_prediction_service import build_prediction_context, run_earth_prediction, _model_source_state
from services.earth_training_artifact import (
    build_checkpoint_payload, build_metrics_block, compute_earth_metrics,
    save_earth_artifact_atomic,
)
from services.netcdf_read_lock import netcdf_read_lock
from test_earth_3hourly_training_runner import registry_for, synthetic_training_release

DATASET_ID = "earth_merra2_3hourly_v1"
ORIGIN = "2020-01-21T01:30:00Z"


def test_release_identity_gate_does_not_require_default_window(synthetic_training_release):
    """A short verified release can predict a model trained with a short window."""
    from dataclasses import replace
    from services.earth_prediction_service import _sync_release
    from services.earth_training_artifact import EarthArtifactError, threehour_available_origin_range

    parent = synthetic_training_release
    metadata = copy.deepcopy(parent.metadata)
    dates = parent.dates[:24]
    metadata["time"].update(end="2020-01-03T22:30:00Z", count=24)
    metadata["splits"] = {
        "train": {"start": "2020-01-01", "end": "2020-01-03", "days": 3, "steps": 24},
        "validation": {"start": None, "end": None, "days": 0, "steps": 0},
        "test": {"start": None, "end": None, "days": 0, "steps": 0},
    }
    release = replace(parent, metadata=metadata, dates=dates)
    registry = registry_for(release)
    binding = registry.build_training_binding(DATASET_ID)
    binding["dataset_snapshot"] = json.loads(binding["dataset_snapshot"])
    assert _sync_release(registry, binding) is release
    assert threehour_available_origin_range(release, window=7, horizon=1)["count"] == 17
    with pytest.raises(EarthArtifactError, match="56-step"):
        threehour_available_origin_range(release)


@pytest.fixture(scope="module")
def threehour_prediction_bundle(synthetic_training_release, tmp_path_factory):
    release = synthetic_training_release
    registry = registry_for(release)
    binding = registry.build_training_binding(DATASET_ID)
    order = ["TO3", "U10M"]
    normalization = fit_threehour_normalization(release, order)
    model, _, _ = build_earth_model_for_plan(ModelSourcePlan("official", order, 2, dataset_id=DATASET_ID))
    metrics = compute_earth_metrics(np.ones((1, 24, 1, 2, 3)), np.zeros((1, 24, 1, 2, 3)), dataset_id=DATASET_ID)
    # The synthetic constant prediction/truth extends uniformly over the grid.
    for row in metrics["by_lead"]:
        row["target_statistics"]["count"] = 240 * 480
    ranges = {
        name: {"date_start": entry["start"] + "T01:30:00Z", "date_end": entry["end"] + "T22:30:00Z", "window_count": 1}
        for name, entry in release.metadata["splits"].items()
    }
    payload = build_checkpoint_payload(
        model=model, input_channel_order=order, linear_hidden_layers=2,
        dataset_binding=binding, normalization=normalization,
        run={"task_id": 701, "seed": 5, "optimizer": "Adam", "loss": "normalized_mse_grid_uniform",
             "best_epoch": 1, "epochs_completed": 1, "run_complete": True, "device": "cpu"},
        metrics=build_metrics_block(validation=metrics, test=metrics, validation_window_count=1,
                                    test_window_count=1, dataset_id=DATASET_ID),
        split_window_counts={"train": 1, "validation": 1, "test": 1}, split_ranges=ranges, task_id=701,
    )
    path = tmp_path_factory.mktemp("threehour_prediction_weights") / "task_701.pth"
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        save_earth_artifact_atomic(payload, path)
    finally:
        torch.set_num_threads(previous)
    task = SimpleNamespace(id=701, **binding, user_id=1, status="completed", output_model_path=str(path),
                           hyperparameters=json.dumps({"training_dataset": DATASET_ID, "window": 56,
                                                       "horizon": 24, "selected_channels": ["U10M"]}))
    return SimpleNamespace(registry=registry, release=release, task=task, model=model,
                           normalization=normalization, payload=payload)


@pytest.fixture
def threehour_prediction_task(threehour_prediction_bundle):
    bundle = threehour_prediction_bundle
    return bundle.registry, copy.deepcopy(bundle.task)


def test_context_uses_datetime_origins_and_actual_global_grid(threehour_prediction_task):
    registry, task = threehour_prediction_task
    context = build_prediction_context(task, registry)
    assert context.dataset_id == DATASET_ID and context.dataset_version == "v1"
    assert context.target == "TO3" and context.target_unit == "DU"
    assert (context.window, context.horizon) == (56, 24)
    assert context.grid_shape == [240, 480]
    assert len(context.latitude) == 240 and len(context.longitude) == 480
    assert context.latitude[0] == -89.625 and context.longitude[-1] == 179.625
    assert context.input_channel_order == ["TO3", "U10M"]
    assert context.input_units == ["DU", "m s-1"]
    assert context.temporal == {"frequency_hours": 3, "step_unit": "hour", "step": 3,
                               "time_zone": "UTC", "timestamp_rule": "interval_center"}
    assert context.origins["kind"] == "datetime"
    assert context.origins["start"] == "2020-01-07T22:30:00Z"
    assert context.origins["end"] == "2020-01-27T22:30:00Z"
    assert context.origins["count"] == 161
    assert len(context.origins["timestamps"]) == 161
    assert context.origins["timestamps"][0] == context.origins["start"]
    assert context.origins["timestamps"][-1] == context.origins["end"]
    assert context.training_split_end == "2020-01-10T22:30:00Z"


def test_real_prediction_reads_bounded_windows_returns_du_fields_and_uses_saved_statistics(
    threehour_prediction_bundle, monkeypatch,
):
    import services.earth_prediction_service as service
    bundle = threehour_prediction_bundle
    cache = EarthPredictionCache()
    monkeypatch.setattr(service, "EARTH_PREDICTION_CACHE", cache)
    calls = []
    reads = []
    original_forward = service.earth_forward_for_model
    original_read = service._threehour_values

    def record_forward(model, inputs, **kwargs):
        calls.append(tuple(inputs.shape))
        return original_forward(model, inputs, **kwargs)

    monkeypatch.setattr(service, "earth_forward_for_model", record_forward)

    def record_read(source, channel, start, stop, **kwargs):
        reads.append((channel, start, stop))
        return original_read(source, channel, start, stop, **kwargs)

    monkeypatch.setattr(service, "_threehour_values", record_read)

    def forbidden_refit(*args, **kwargs):
        pytest.fail("prediction must use the checkpoint normalization without fitting data")

    monkeypatch.setattr("services.earth_dataset.fit_threehour_normalization", forbidden_refit)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        result = run_earth_prediction(bundle.task, ORIGIN, bundle.registry, device="cpu")
    finally:
        torch.set_num_threads(previous)
    assert result["planet"] == "earth" and result["dataset_id"] == DATASET_ID
    assert result["dataset_fingerprint"] == bundle.task.dataset_fingerprint
    assert result["forecast_origin"] == ORIGIN and result["origin_split"] == "test"
    assert result["target_unit"] == "DU" and result["target"] == "TO3"
    assert result["frequency_hours"] == 3 and result["timestamp_rule"] == "interval_center"
    assert result["grid"]["shape"] == [240, 480]
    assert len(result["input_timestamps"]) == 56
    assert result["input_timestamps"][0] == "2020-01-14T04:30:00Z"
    assert result["input_timestamps"][-1] == ORIGIN
    assert len(result["target_timestamps"]) == 24
    assert result["target_timestamps"][0] == "2020-01-21T04:30:00Z"
    assert result["target_timestamps"][-1] == "2020-01-24T01:30:00Z"
    assert calls and all(shape[1:3] == (56, 2) for shape in calls)
    assert max(shape[-2] for shape in calls) <= 24
    assert max(shape[-1] for shape in calls) <= 48
    assert reads and max(stop - start for _, start, stop in reads) <= 8
    assert min(start for _, start, _ in reads) == 105
    assert max(stop for _, _, stop in reads) == 185
    arrays = {}
    for kind in ("prediction", "reference", "residual"):
        arrays[kind] = np.asarray([entry["field"] for entry in result[kind]], dtype="float32")
        assert arrays[kind].shape == (24, 240, 480)
        assert np.isfinite(arrays[kind]).all()
        assert all(entry["valid_cells"] == 240 * 480 for entry in result[kind])
    np.testing.assert_allclose(arrays["residual"], arrays["prediction"] - arrays["reference"], rtol=1e-5, atol=1e-5)
    with netcdf_read_lock(), xr.open_dataset(bundle.release.data_path, engine="netcdf4", mask_and_scale=False) as source:
        expected_reference = np.asarray(source["TO3"].isel(time=slice(161, 185)).values, dtype="float32")
        np.testing.assert_array_equal(arrays["reference"], expected_reference)
        inputs = np.stack([source[name].isel(time=slice(105, 161), lat=slice(0, 2), lon=slice(0, 3)).values
                           for name in ("TO3", "U10M")], axis=1)
    for index in range(2):
        inputs[:, index] = (inputs[:, index] - bundle.normalization["mean"][index]) / bundle.normalization["scale"][index]
    bundle.model.eval()
    with torch.no_grad():
        # Corner equality is a property of the grid-point model, not of the
        # spatial contract: the shared contract fixes every call to 24x48, so a
        # 2x3 patch is only meaningful when the model is called directly.
        expected_prediction = bundle.model(
            torch.from_numpy(inputs[None].astype("float32"))
        )[0, :, 0].numpy()
    expected_prediction = expected_prediction * bundle.normalization["scale"][0] + bundle.normalization["mean"][0]
    np.testing.assert_allclose(arrays["prediction"][:, :2, :3], expected_prediction, rtol=1e-5, atol=1e-4)
    metrics = result["metrics"]
    assert metrics["unit"] == "DU" and metrics["reference_available"] is True
    assert [row["lead_step"] for row in metrics["by_lead"]] == list(range(1, 25))
    assert [row["lead_hours"] for row in metrics["by_lead"]] == list(range(3, 73, 3))
    assert [row["horizon_hours"] for row in metrics["by_horizon"]] == [24, 48, 72]
    direct = compute_earth_metrics(arrays["prediction"][None, :, None], arrays["reference"][None, :, None], dataset_id=DATASET_ID)
    for group in ("by_lead", "by_horizon"):
        for actual, expected in zip(metrics[group], direct[group]):
            assert actual["rmse"] == pytest.approx(expected["rmse"], rel=1e-10)
            assert actual["mae"] == pytest.approx(expected["mae"], rel=1e-10)
    assert metrics["overall"] == pytest.approx(direct["overall"], rel=1e-10)
    key = result["cache_key"]
    assert len(key) == 64
    del result, arrays
    # A second request must reuse the arrays, while still rechecking the identity.
    before = len(calls)
    monkeypatch.setattr(service, "_field_list", lambda values: [{"shape": list(values.shape)}])
    hit = run_earth_prediction(bundle.task, ORIGIN, bundle.registry, device="cpu")
    assert hit["cache_key"] == key and len(calls) == before
    changed = copy.deepcopy(bundle.task)
    changed.dataset_fingerprint = "f" * 64
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(changed, ORIGIN, bundle.registry, device="cpu")
    assert (error.value.status_code, error.value.code) == (409, "dataset_version_changed")
    assert len(calls) == before


def test_cached_threehour_prediction_rechecks_checkpoint_after_cache_read(threehour_prediction_bundle, monkeypatch):
    import services.earth_prediction_service as service

    bundle = threehour_prediction_bundle
    cache = EarthPredictionCache()
    monkeypatch.setattr(service, "EARTH_PREDICTION_CACHE", cache)
    run_earth_prediction(bundle.task, ORIGIN, bundle.registry, device="cpu", cache=cache)
    monkeypatch.setattr(service, "earth_forward_for_model",
                        lambda *args, **kwargs: pytest.fail("the cached result should not rerun inference"))
    original_sha = service._threehour_checkpoint_sha(bundle.task)
    calls = {"count": 0}
    def change_after_cache_key(task):
        calls["count"] += 1
        return original_sha if calls["count"] == 1 else "changed"
    monkeypatch.setattr(service, "_threehour_checkpoint_sha", change_after_cache_key)

    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(bundle.task, ORIGIN, bundle.registry, device="cpu", cache=cache)
    assert (error.value.status_code, error.value.code) == (409, "invalid_earth_training_artifact")


def test_uploaded_source_warning_state_refreshes_without_rebuilding(tmp_path):
    from services.uploaded_model_source import hash_source_bytes

    source = tmp_path / "uploaded_model.py"
    content = b"MODEL_SPEC = {}\n"
    source.write_bytes(content)
    reference = {"source_path": str(source), "content_hash": hash_source_bytes(content), "source_text": "MODEL_SPEC = {}\n"}
    checkpoint = SimpleNamespace(model_source="uploaded", uploaded_model=reference)

    assert _model_source_state(checkpoint) == ("available", [])
    source.unlink()
    status, warnings = _model_source_state(checkpoint)
    assert status == "missing" and warnings
    source.write_bytes(b"tampered")
    status, warnings = _model_source_state(checkpoint)
    assert status == "tampered" and warnings
    source.write_bytes(content)
    assert _model_source_state(checkpoint) == ("available", [])
    assert _model_source_state(SimpleNamespace(model_source="official", uploaded_model=None)) == ("official", [])


@pytest.mark.parametrize("origin,code", [
    ("2020-01-21", "invalid_earth_prediction_origin"),
    ("2020-01-21T01:30:00", "invalid_earth_prediction_origin"),
    ("2020-01-21T01:30:00+08:00", "invalid_earth_prediction_origin"),
    ("2020-01-21T02:30:00Z", "earth_prediction_origin_out_of_range"),
    ("2020-01-01T01:30:00Z", "earth_prediction_origin_out_of_range"),
    ("2020-01-07T19:30:00Z", "earth_prediction_origin_out_of_range"),
    ("2020-01-28T01:30:00Z", "earth_prediction_origin_out_of_range"),
    ("2021-07-08T01:30:00Z", "earth_prediction_origin_out_of_range"),
])
def test_incomplete_or_non_utc_origins_are_stably_rejected(threehour_prediction_task, origin, code):
    registry, task = threehour_prediction_task
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(task, origin, registry, device="cpu")
    assert (error.value.status_code, error.value.code) == (422, code)


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled"])
def test_unfinished_tasks_rejected_before_reading_checkpoint(threehour_prediction_task, status):
    registry, task = threehour_prediction_task
    task.status = status
    task.output_model_path = "must-not-be-opened.pth"
    with pytest.raises(DatasetRequestError) as error:
        build_prediction_context(task, registry)
    assert (error.value.status_code, error.value.code) == (409, "earth_prediction_task_not_completed")


@pytest.mark.parametrize("kind", ["missing", "damaged", "wrong_window", "wrong_grid", "wrong_time_rule", "wrong_task", "wrong_state",
                                  "architecture_structure", "time_structure", "splits_structure", "timestamp_offset",
                                  "run_epoch", "run_seed", "run_device", "run_created"])
def test_invalid_checkpoint_stably_rejected(threehour_prediction_bundle, tmp_path, kind):
    bundle = threehour_prediction_bundle
    task = copy.deepcopy(bundle.task)
    path = tmp_path / "bad_checkpoint.pth"
    if kind == "damaged":
        path.write_bytes(b"not a checkpoint")
    elif kind != "missing":
        payload = copy.deepcopy(bundle.payload)
        if kind == "wrong_window":
            payload["training_contract"]["window"] = 7
        elif kind == "wrong_grid":
            payload["model_config"]["height"] = 36
        elif kind == "wrong_time_rule":
            payload["training_contract"]["timestamp_rule"] = "date_only"
        elif kind == "wrong_task":
            payload["run"]["task_id"] = 999
        elif kind == "wrong_state":
            name = next(iter(payload["model_state_dict"]))
            payload["model_state_dict"][name] = torch.zeros(1)
        elif kind == "architecture_structure":
            payload["model_config"]["architecture_params"] = [1]
        elif kind == "time_structure":
            payload["dataset_binding"]["dataset_snapshot"]["time"] = [1]
        elif kind == "splits_structure":
            payload["dataset_binding"]["dataset_snapshot"]["splits"] = [1]
        elif kind == "timestamp_offset":
            payload["training_contract"]["split_ranges"]["train"]["date_start"] = "2020-01-01T04:30:00+03:00"
        elif kind == "run_epoch":
            payload["run"]["best_epoch"] = [1]
        elif kind == "run_seed":
            payload["run"]["seed"] = [5]
        elif kind == "run_device":
            payload["run"]["device"] = ["cpu"]
        elif kind == "run_created":
            payload["run"]["created_utc"] = ["2026-10-07T00:00:00Z"]
        torch.save(payload, path)
    task.output_model_path = str(path)
    with pytest.raises(DatasetRequestError) as error:
        build_prediction_context(task, bundle.registry)
    assert (error.value.status_code, error.value.code) == (409, "invalid_earth_training_artifact")


@pytest.mark.parametrize("change", ["fingerprint", "version", "time", "time_axis", "grid", "longitude", "splits"])
def test_release_changes_are_version_errors(threehour_prediction_bundle, change):
    bundle = threehour_prediction_bundle
    altered = copy.copy(bundle.release)
    metadata = copy.deepcopy(bundle.release.metadata)
    if change == "fingerprint":
        metadata["dataset_fingerprint"] = "f" * 64
    elif change == "version":
        metadata["dataset_version"] = "v2"
    elif change == "time":
        metadata["frequency_hours"] = 24
    elif change == "time_axis":
        dates = bundle.release.dates.copy()
        dates[100] += np.timedelta64(1, "h")
        object.__setattr__(altered, "dates", dates)
    elif change == "grid":
        object.__setattr__(altered, "latitude", np.arange(36))
    elif change == "longitude":
        longitude = bundle.release.longitude.copy()
        longitude[0] += .125
        object.__setattr__(altered, "longitude", longitude)
    elif change == "splits":
        metadata["splits"]["test"]["steps"] = 88
    object.__setattr__(altered, "metadata", metadata)
    registry = SimpleNamespace(get_earth_snapshot=lambda *args, **kwargs: altered)
    with pytest.raises(DatasetRequestError) as error:
        build_prediction_context(bundle.task, registry)
    assert (error.value.status_code, error.value.code) == (409, "dataset_version_changed")


@pytest.mark.parametrize("change", ["window", "horizon", "channels", "invalid_json", "snapshot_time", "snapshot_grid"])
def test_task_contract_or_snapshot_cannot_override_the_checkpoint(threehour_prediction_task, change):
    registry, task = threehour_prediction_task
    hypers = json.loads(task.hyperparameters)
    if change == "window":
        hypers["window"] = 7
    elif change == "horizon":
        hypers["horizon"] = 3
    elif change == "channels":
        hypers["selected_channels"] = []
    elif change.startswith("snapshot"):
        snapshot = json.loads(task.dataset_snapshot)
        if change == "snapshot_time":
            snapshot["time"]["label"] = "date_only"
        else:
            snapshot["grid"]["shape"] = [36, 72]
        task.dataset_snapshot = json.dumps(snapshot)
    task.hyperparameters = "{unfinished" if change == "invalid_json" else json.dumps(hypers)
    with pytest.raises(DatasetRequestError) as error:
        build_prediction_context(task, registry)
    assert (error.value.status_code, error.value.code) == (409, "invalid_earth_training_artifact")


def test_missing_observations_return_a_stable_error_without_zero_fill(threehour_prediction_task, monkeypatch):
    import services.earth_prediction_service as service
    registry, task = threehour_prediction_task
    monkeypatch.setattr(service, "EARTH_PREDICTION_CACHE", EarthPredictionCache())

    def unavailable(*args, **kwargs):
        raise ValueError("Earth three-hour data contains missing values in TO3")

    monkeypatch.setattr(service, "_threehour_values", unavailable)
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(task, ORIGIN, registry, device="cpu")
    assert (error.value.status_code, error.value.code) == (503, "earth_prediction_data_unavailable")


@pytest.mark.parametrize("output_value", [float("nan"), np.finfo("float32").max])
def test_nonfinite_du_predictions_are_rejected_without_caching(
    threehour_prediction_bundle, tmp_path, monkeypatch, output_value,
):
    import services.earth_prediction_service as service
    bundle = threehour_prediction_bundle
    payload = copy.deepcopy(bundle.payload)
    # This is a valid positive saved scale; a finite model output may still
    # overflow when converted back to float32 DU.
    payload["normalization"]["scale"][0] = 10.0
    path = tmp_path / "nonfinite_prediction.pth"
    save_earth_artifact_atomic(payload, path)
    task = copy.deepcopy(bundle.task)
    task.output_model_path = str(path)
    cache = EarthPredictionCache()
    monkeypatch.setattr(cache, "put", lambda *args: pytest.fail("Invalid DU predictions must never be cached"))
    monkeypatch.setattr(service, "earth_forward_for_model", lambda model, inputs, **kw:
                        torch.full((1, 24, 1, 24, 48), float(output_value)))
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(task, ORIGIN, bundle.registry, device="cpu", cache=cache)
    assert (error.value.status_code, error.value.code) == (409, "invalid_earth_training_artifact")


def test_task_snapshot_accepts_a_server_mapping(threehour_prediction_task):
    registry, task = threehour_prediction_task
    task.dataset_snapshot = json.loads(task.dataset_snapshot)
    assert build_prediction_context(task, registry).origins["count"] == 161


def cache_identity():
    return dict(planet="earth", dataset_id=DATASET_ID, dataset_version="v1", dataset_fingerprint="a" * 64,
                task_id=701, forecast_origin=ORIGIN, target_timestamps=["2020-01-21T04:30:00Z"],
                checkpoint_sha256="b" * 64)


@pytest.mark.parametrize("field,value", [
    ("planet", "mars"), ("dataset_id", "earth_merra2_daily_v2"), ("dataset_version", "v2"),
    ("dataset_fingerprint", "f" * 64), ("task_id", 702), ("forecast_origin", "2020-01-21T04:30:00Z"),
    ("target_timestamps", ["2020-01-21T07:30:00Z"]), ("checkpoint_sha256", "c" * 64),
])
def test_cache_identity_separates_every_required_axis(field, value):
    baseline = cache_identity()
    altered = {**baseline, field: value}
    assert build_earth_prediction_cache_key(**baseline) != build_earth_prediction_cache_key(**altered)
    assert build_earth_prediction_cache_key(**baseline) == build_earth_prediction_cache_key(**dict(reversed(list(baseline.items()))))


def test_array_cache_is_immutable_bounded_and_does_not_store_serialized_fields():
    cache = EarthPredictionCache(max_bytes=500, max_entries=1)
    def item(value):
        return {"task_id": value, "nested": {"tag": "original"},
                **{name: np.full((2, 3, 4), value, dtype="float32")
                   for name in ("_prediction_du", "_reference_du", "_residual_du")}}
    first = item(1)
    cache.put("first", first)
    first["_prediction_du"][0, 0, 0] = 99
    found = cache.get("first")
    assert found["_prediction_du"][0, 0, 0] == 1
    assert found["_prediction_du"].flags.writeable is False
    found["nested"]["tag"] = "changed"
    assert cache.get("first")["nested"]["tag"] == "original"
    cache.put("second", item(2))
    assert cache.get("first") is None and cache.get("second")["task_id"] == 2
    cache.put("too-large", {**item(3), "_prediction_du": np.ones((1000,), dtype="float32")})
    assert cache.get("too-large") is None
