"""Global NetCDF and verified checkpoint diagnostics, without real-data training.

The shared synthetic release has real 240x480 fields and masks. Tiny temporal
windows bound inference while keeping task/checkpoint/source/data identity
checks, Earth normalization, physical units and spatial tile reads intact.
"""

import copy
from datetime import date, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services.dataset_identity import DatasetRequestError
from services.earth_dataset import fit_threehour_normalization
from services.earth_dataset_metadata import VerifiedEarthRelease, package_signature
from services.earth_diagnostics import EarthDiagnosticsCache, compute_earth_diagnostics
from services.earth_model_source import ModelSourcePlan, build_earth_model_for_plan
from services.earth_task_split import build_earth_task_split
from services.earth_training_artifact import (
    FULL_METRIC_KEYS, build_checkpoint_payload, build_metrics_block,
    compute_earth_metrics, save_earth_artifact_atomic,
)
from services.earth_training_contract import EARTH_3HOURLY_METRICS_SCHEMA_V2
from services.netcdf_read_lock import netcdf_read_lock
from scripts.build_earth_merra2_3hourly import _new_output
from training_backbones.earth_3hourly_uploaded_contract import build_config
from test_earth_3hourly_training_runner import synthetic_training_release, registry_for
from test_earth_3hourly_uploaded import package_at, source_with_spec, spec as uploaded_spec
from test_earth_metrics_v2 import legacy_payload

DATASET_ID = "earth_merra2_3hourly_v1"
RATIOS = {"train_ratio": .5, "validation_ratio": .25, "test_ratio": .25}


@pytest.fixture(scope="module", autouse=True)
def integration_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _saved_metrics(horizon, count):
    # Deliberately distinct saved metrics make accidental replacement by a
    # two-window diagnostic evaluation observable. This is a synthetic fixture,
    # never a claimed accuracy measurement of the real production package.
    result = compute_earth_metrics(np.full((1, horizon, 1, 1, 1), 12.),
                                   np.full((1, horizon, 1, 1, 1), 10.), DATASET_ID)
    for row in result["by_lead"]:
        row["target_statistics"]["count"] = count * 240 * 480
    return result


def _create_bundle(release, root, source, *, window=2, horizon=1, custom_split=True,
                   task_id=2401, polarity=1):
    root.mkdir(exist_ok=True)
    registry = registry_for(release)
    binding = registry.build_training_binding(DATASET_ID)
    split = build_earth_task_split(release.dates, window, horizon, RATIOS) if custom_split else None
    normalization = fit_threehour_normalization(release, ["TO3", "U10M"], task_split=split)
    reference = None
    if source == "uploaded":
        declaration = uploaded_spec()
        declaration["datasets"][DATASET_ID]["window"] = [window]
        declaration["datasets"][DATASET_ID]["horizon"] = [horizon]
        text = source_with_spec(declaration)
        package, validation = package_at(root, text)
        assert validation.ok, validation.errors
        reference = {"package_id": package.id, "display_name": package.display_name, "version": 1,
                     "content_hash": package.content_hash, "source_path": package.storage_path,
                     "source_text": text, "param_schema": validation.param_schema,
                     "custom_model_params": {"bias": True}, "input_channel_order": ["TO3", "U10M"],
                     "build_config": build_config(["TO3", "U10M"], {"bias": True}, window=window, horizon=horizon)}
    plan = ModelSourcePlan(source, ["TO3", "U10M"], 2, dataset_id=DATASET_ID,
                           window=window, horizon=horizon, uploaded_model=reference)
    model, _, _ = build_earth_model_for_plan(plan)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        if source == "uploaded":
            # A verified trainable uploaded model uses historical ozone only.
            # The actual auxiliary input remains present and varies by window.
            model.mix.weight[0, 0, 0, 0] = 1.
            model.temporal.weight[:, -1] = polarity
    if split:
        ranges = copy.deepcopy(split["ranges"])
    else:
        ranges = {name: {"date_start": entry["start"] + "T01:30:00Z",
                         "date_end": entry["end"] + "T22:30:00Z",
                         "window_count": entry["steps"] - window - horizon + 1}
                  for name, entry in release.metadata["splits"].items()}
    counts = {name: entry["window_count"] for name, entry in ranges.items()}
    metrics = build_metrics_block(
        validation=_saved_metrics(horizon, counts["validation"]),
        test=_saved_metrics(horizon, counts["test"]),
        validation_window_count=counts["validation"], test_window_count=counts["test"],
        dataset_id=DATASET_ID, horizon=horizon)
    hypers = {"training_dataset": DATASET_ID, "model_source": source,
              "selected_channels": ["U10M"], "window": window, "horizon": horizon, **RATIOS}
    if split:
        hypers["_earth_task_split"] = split
    if reference:
        hypers.update(_uploaded_model_id=reference["package_id"], _uploaded_model_version=1,
                      _uploaded_model_content_hash=reference["content_hash"], custom_model_params={"bias": True})
    payload = build_checkpoint_payload(
        model=model, input_channel_order=["TO3", "U10M"], linear_hidden_layers=2,
        dataset_binding=binding, normalization=normalization,
        run={"task_id": task_id, "optimizer": "Adam", "loss": "normalized_mse_grid_uniform",
             "best_epoch": 1, "epochs_completed": 1, "run_complete": True, "hyperparameters": hypers},
        metrics=metrics, split_window_counts=counts, split_ranges=ranges, task_split=split,
        split_ratios=RATIOS, task_id=task_id, model_source=source, uploaded_model=reference)
    path = root / "checkpoint.pth"
    save_earth_artifact_atomic(payload, path)
    task = SimpleNamespace(id=task_id, **binding, user_id=1, status="completed", output_model_path=str(path),
                           hyperparameters=json.dumps(hypers), custom_model_name=source)
    return SimpleNamespace(task=task, registry=registry, release=release, payload=payload,
                           source=source, split=split, path=path, root=root)


@pytest.fixture(scope="module", params=["official", "uploaded"])
def diagnostic_bundle(request, synthetic_training_release, tmp_path_factory):
    source = request.param
    # Both nondefault and unequal windows are accepted from frozen task metadata.
    return _create_bundle(synthetic_training_release,
                          tmp_path_factory.mktemp("diagnostics-" + source), source,
                          window=3 if source == "official" else 2,
                          horizon=2 if source == "official" else 1)


def _compute(bundle, **overrides):
    parameters = dict(sample_windows=2, scatter_points=7, histogram_bins=8, seed=42,
                      pfi_repeats=1, include_pfi=True, device="cpu", cache=EarthDiagnosticsCache())
    parameters.update(overrides)
    return compute_earth_diagnostics(bundle.task, bundle.registry, **parameters)


def test_global_verified_diagnostics_use_fixed_task_partition_and_bounded_reads(diagnostic_bundle, monkeypatch):
    import services.earth_diagnostics as service
    bundle = diagnostic_bundle
    reads, calls = [], []
    original_read, original_forward = service._threehour_values, service.earth_forward_for_model

    def read(ds, channel, start, stop, **kwargs):
        reads.append((start, stop, kwargs["lat_slice"], kwargs["lon_slice"]))
        return original_read(ds, channel, start, stop, **kwargs)

    def forward(model, inputs, **kwargs):
        calls.append(tuple(inputs.shape))
        return original_forward(model, inputs, **kwargs)

    monkeypatch.setattr(service, "_threehour_values", read)
    monkeypatch.setattr(service, "earth_forward_for_model", forward)
    monkeypatch.setattr("services.earth_dataset.fit_threehour_normalization",
                        lambda *a, **k: pytest.fail("Diagnostics refitted task normalization"))
    result = _compute(bundle)
    assert result["scope"]["grid_shape"] == [240, 480]
    assert result["scope"]["window_count"] == 2
    assert result["scope"]["available_window_count"] == bundle.split["ranges"]["test"]["window_count"]
    assert result["scope"]["valid_points"] == 2 * result["horizon"] * 240 * 480
    assert sum(result["histogram"]["counts"]) == result["scope"]["valid_points"]
    assert len(result["scatter"]["prediction"]) == 7
    assert result["full_test_metrics"]["metrics"]["overall"]["rmse"] == 2.
    assert result["full_test_metrics"]["schema"] == EARTH_3HOURLY_METRICS_SCHEMA_V2
    assert set(result["sample_metrics"]["overall"]) == set(FULL_METRIC_KEYS)
    assert result["unit"] == result["pfi"]["unit"] == "DU"
    assert reads and all(stop - start <= 8 for start, stop, _, _ in reads)
    test = bundle.split["ranges"]["test"]
    assert min(start for start, _, _, _ in reads) >= test["raw_start"]
    assert max(stop for _, stop, _, _ in reads) <= test["raw_end"]
    assert calls and all(shape[0] <= 2 and shape[-2:] == (24, 48) for shape in calls)
    assert all(lat.stop - lat.start <= 24 and lon.stop - lon.start <= 48 for _, _, lat, lon in reads)
    if bundle.source == "uploaded":
        assert result["sample_metrics"]["overall"]["rmse"] == pytest.approx(.01, abs=1e-5)
        assert result["histogram"]["summary"]["mean"] == pytest.approx(-.01, abs=1e-5)
        assert result["pfi"]["items"][0]["importance_mean"] > 0
        assert result["pfi"]["items"][1]["importance_mean"] == pytest.approx(0, abs=1e-8)
        np.testing.assert_allclose(np.array(result["scatter"]["prediction"]) - result["scatter"]["reference"],
                                   -.01, atol=1e-5)


def test_legacy_v1_metric_gaps_remain_missing_and_checkpoint_is_not_rewritten(diagnostic_bundle, tmp_path):
    bundle = diagnostic_bundle
    payload = legacy_payload(bundle.payload)
    path = tmp_path / "legacy-v1.pth"
    save_earth_artifact_atomic(payload, path)
    task = copy.copy(bundle.task)
    task.output_model_path = str(path)
    before = path.read_bytes()
    result = compute_earth_diagnostics(task, bundle.registry, sample_windows=2, scatter_points=4,
                                       histogram_bins=5, include_pfi=False, device="cpu", cache=EarthDiagnosticsCache())
    assert set(result["full_test_metrics"]["metrics"]["overall"]) == {"rmse", "mae"}
    assert set(result["sample_metrics"]["overall"]) == set(FULL_METRIC_KEYS)
    assert path.read_bytes() == before


def test_cache_rechecks_model_and_task_data_identity(diagnostic_bundle, monkeypatch):
    import services.earth_diagnostics as service
    bundle = diagnostic_bundle
    cache = EarthDiagnosticsCache()
    first = _compute(bundle, include_pfi=False, cache=cache)
    monkeypatch.setattr(service, "_read_tile", lambda *a, **k: pytest.fail("Cache hit reread fields"))
    second = _compute(bundle, include_pfi=False, cache=cache)
    assert second["cache"]["hit"] and first["cache"]["key"] == second["cache"]["key"]
    changed = copy.copy(bundle.task)
    changed.dataset_fingerprint = "f" * 64
    with pytest.raises(DatasetRequestError) as error:
        compute_earth_diagnostics(changed, bundle.registry, cache=cache)
    assert error.value.code == "dataset_version_changed"
    changed = copy.copy(bundle.task)
    hypers = json.loads(changed.hyperparameters)
    hypers["window"] += 1
    changed.hyperparameters = json.dumps(hypers)
    with pytest.raises(DatasetRequestError, match="window"):
        compute_earth_diagnostics(changed, bundle.registry, cache=cache)


def test_valid_checkpoint_replacement_invalidates_cached_snapshot(diagnostic_bundle, tmp_path):
    bundle = diagnostic_bundle
    path = tmp_path / "replaceable.pth"
    save_earth_artifact_atomic(bundle.payload, path)
    task = copy.copy(bundle.task)
    task.output_model_path = str(path)
    cache = EarthDiagnosticsCache()
    arguments = dict(sample_windows=2, scatter_points=4, histogram_bins=5, include_pfi=False,
                     device="cpu", cache=cache)
    first = compute_earth_diagnostics(task, bundle.registry, **arguments)
    replacement = copy.deepcopy(bundle.payload)
    # A valid different run snapshot must be a new identity even with equal weights.
    replacement["run"]["seed"] = 989
    save_earth_artifact_atomic(replacement, path)
    second = compute_earth_diagnostics(task, bundle.registry, **arguments)
    assert not second["cache"]["hit"] and first["cache"]["key"] != second["cache"]["key"]


def test_published_manifest_compatibility_never_crosses_into_validation(synthetic_training_release, tmp_path):
    bundle = _create_bundle(synthetic_training_release, tmp_path, "official", window=2, horizon=1,
                            custom_split=False, task_id=2402)
    result = _compute(bundle, include_pfi=False)
    assert result["full_test_metrics"]["split_policy"] == "published_manifest_splits"
    assert result["scope"]["available_window_count"] == 78
    assert all(row["raw_start"] >= 160 for row in result["scope"]["windows"])
    assert result["scope"]["time_range"]["input_start"] >= "2020-01-21T01:30:00Z"


def test_pfi_retains_negative_du_importance(synthetic_training_release, tmp_path):
    bundle = _create_bundle(synthetic_training_release, tmp_path, "uploaded", polarity=-1, task_id=2403)
    result = _compute(bundle)
    ozone, unused = result["pfi"]["items"]
    assert ozone["importance_mean"] < 0
    assert ozone["repeats"][0]["delta_rmse"] < 0
    assert unused["importance_mean"] == pytest.approx(0, abs=1e-10)


def test_uploaded_source_state_is_rechecked_even_on_cached_result(diagnostic_bundle, monkeypatch):
    if diagnostic_bundle.source != "uploaded":
        pytest.skip("Uploaded source identity is not present on official checkpoints")
    bundle = diagnostic_bundle
    path = Path(bundle.payload["model_ref"]["uploaded_model"]["source_path"])
    before = path.read_bytes()
    cache = EarthDiagnosticsCache()
    first = _compute(bundle, include_pfi=False, cache=cache)
    import services.earth_diagnostics as service
    monkeypatch.setattr(service, "_read_tile", lambda *a, **k: pytest.fail("Cached snapshot reran inference"))
    try:
        path.write_bytes(before + b"\n# source changed after training\n")
        hit = _compute(bundle, include_pfi=False, cache=cache)
        assert hit["cache"]["hit"] and hit["cache"]["key"] == first["cache"]["key"]
        assert any("no longer matches" in message for message in hit["warnings"])
        # The pinned embedded source is verified and remains authoritative; it
        # never executes the changed on-disk content.
        path.unlink()
        missing = _compute(bundle, include_pfi=False, cache=cache)
        assert missing["cache"]["hit"]
        assert any("unavailable" in message for message in missing["warnings"])
    finally:
        path.write_bytes(before)


def test_release_file_mutation_is_rejected_before_cached_diagnostics(diagnostic_bundle, monkeypatch):
    bundle = diagnostic_bundle
    cache = EarthDiagnosticsCache()
    _compute(bundle, include_pfi=False, cache=cache)
    import services.earth_dataset_metadata as metadata
    monkeypatch.setattr(metadata, "package_signature", lambda *a, **k: ("changed",))
    with pytest.raises(DatasetRequestError) as error:
        _compute(bundle, include_pfi=False, cache=cache)
    assert error.value.code == "dataset_version_changed"


def test_identical_real_global_windows_report_meaningless_pfi(synthetic_training_release, tmp_path):
    path = tmp_path / "earth_merra2_3hourly.nc"
    times = np.datetime64("2020-01-01T01:30", "ns") + np.arange(96) * np.timedelta64(3, "h")
    with netcdf_read_lock():
        ds = _new_output(path, [date(2020, 1, 1) + timedelta(days=index) for index in range(12)], "a" * 64)
        try:
            for channel in ("TO3", "U10M", "V10M", "T2M", "SWGDN"):
                for step in range(96):
                    ds[channel][step] = np.float32(100)
                    ds[channel + "_valid_mask"][step] = np.uint8(1)
        finally:
            ds.close()
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    metadata = copy.deepcopy(synthetic_training_release.metadata)
    metadata["time"].update(start="2020-01-01T01:30:00Z", end="2020-01-12T22:30:00Z", count=96)
    metadata["splits"] = {name: {"start": f"2020-01-{index * 4 + 1:02d}",
                                  "end": f"2020-01-{index * 4 + 4:02d}", "steps": 32, "days": 4}
                          for index, name in enumerate(("train", "validation", "test"))}
    release = VerifiedEarthRelease(
        metadata=metadata, signature=package_signature(tmp_path, data_file_name=path.name),
        dates=times, latitude=synthetic_training_release.latitude,
        longitude=synthetic_training_release.longitude, fields=synthetic_training_release.fields, data_path=path)
    bundle = _create_bundle(release, tmp_path / "model", "uploaded", task_id=2404)
    result = _compute(bundle)
    assert result["pfi"]["status"] == "unavailable"
    assert result["pfi"]["reason_code"] == "earth_pfi_identical_windows"
    assert all(row["status"] == "unavailable" and row["importance_mean"] is None
               and row["reason_code"] == "earth_pfi_identical_channel_windows" for row in result["pfi"]["items"])


def test_embedded_upload_source_digest_is_validated_before_inference(diagnostic_bundle, tmp_path, monkeypatch):
    if diagnostic_bundle.source != "uploaded":
        pytest.skip("Embedded source identity only belongs to uploaded checkpoints")
    payload = copy.deepcopy(diagnostic_bundle.payload)
    payload["model_ref"]["uploaded_model"]["source_text"] += "\n# invalid embedded copy\n"
    path = tmp_path / "tampered.pth"
    torch.save(payload, path)  # Deliberately bypass writer validation to model a damaged file.
    task = copy.copy(diagnostic_bundle.task)
    task.output_model_path = str(path)
    monkeypatch.setattr("services.earth_diagnostics._read_tile",
                        lambda *a, **k: pytest.fail("Invalid source reached inference"))
    with pytest.raises(DatasetRequestError, match="source identity"):
        compute_earth_diagnostics(task, diagnostic_bundle.registry, device="cpu", cache=EarthDiagnosticsCache())
