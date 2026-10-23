"""56-to-24 checkpoint isolation, bounded strict reload and physical lead metrics."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services.dataset_identity import EARTH_DATASET_3HOURLY_ID
from services.earth_model_source import (
    EarthModelBuildError, ModelSourcePlan, build_earth_model_for_plan,
    earth_forward_for_model,
)
from services.earth_training_artifact import (
    EarthArtifactError, ErrorAccumulator, available_origin_range,
    build_checkpoint_payload, build_earth_model_from_checkpoint,
    build_metrics_block, compute_earth_metrics, forecast_dates,
    load_earth_training_artifact, origin_split, save_earth_artifact_atomic,
    validate_checkpoint_payload,
)
from services.earth_training_contract import (
    EARTH_3HOURLY_ARTIFACT_SCHEMA, EARTH_3HOURLY_IMPLEMENTATION_ID,
    EARTH_3HOURLY_METRICS_SCHEMA_V2,
)

DATASET_ID = EARTH_DATASET_3HOURLY_ID
SPLITS = {
    "train": {"start": "2020-01-01", "end": "2020-12-31", "steps": 2928},
    "validation": {"start": "2021-01-01", "end": "2021-06-30", "steps": 1448},
    "test": {"start": "2021-07-01", "end": "2021-12-31", "steps": 1472},
}
COUNTS = {name: entry["steps"] - 79 for name, entry in SPLITS.items()}


def _binding():
    snapshot = {
        "dataset_id": DATASET_ID, "dataset_version": "v1", "planet": "earth",
        "dataset_fingerprint": "a" * 64, "data_sha256": "b" * 64,
        "grid": {"shape": [240, 480]},
        "frequency_hours": 3, "step_unit": "hour", "step": 3,
        "time": {"kind": "datetime", "time_zone": "UTC", "label": "interval_center"},
        "splits": copy.deepcopy(SPLITS),
    }
    return {
        "dataset_id": DATASET_ID, "dataset_version": "v1",
        "dataset_fingerprint": "a" * 64, "dataset_identity_status": "verified",
        "dataset_snapshot": snapshot,
    }


def _metrics(window_count=None):
    target = np.full((2, 24, 1, 2, 3), 280.0)
    errors = np.arange(1, 25, dtype=float)[None, :, None, None, None]
    metrics = compute_earth_metrics(target + errors, target, dataset_id=DATASET_ID)
    if window_count is not None:
        # Repeating the constant synthetic grid preserves every metric/moment.
        for row in metrics["by_lead"]:
            row["target_statistics"]["count"] = window_count * 240 * 480
    return metrics


def _payload(order=("TO3", "U10M")):
    plan = ModelSourcePlan("official", list(order), 2, dataset_id=DATASET_ID)
    model, config, warnings = build_earth_model_for_plan(plan)
    assert not warnings
    normalization = {
        "method": "per_channel_standard", "fit_split": "train",
        "fit_date_start": "2020-01-01T01:30:00Z", "fit_date_end": "2020-12-31T22:30:00Z",
        "fit_time_start": "2020-01-01T01:30:00Z", "fit_time_end": "2020-12-31T22:30:00Z",
        "fit_step_count": 2928, "channel_order": list(order),
        "mean": [280.0] * len(order), "scale": [10.0] * len(order),
        "frequency_hours": 3, "step_unit": "hour", "step": 3,
        "time_zone": "UTC", "timestamp_rule": "interval_center",
        "ddof": 0, "epsilon": 1e-6, "target_channel_index": 0,
        "constant_channel_mask": [False] * len(order),
    }
    ranges = {
        name: {
            "date_start": entry["start"] + "T01:30:00Z",
            "date_end": entry["end"] + "T22:30:00Z",
            "window_count": COUNTS[name],
        }
        for name, entry in SPLITS.items()
    }
    metrics = build_metrics_block(
        validation=_metrics(COUNTS["validation"]), test=_metrics(COUNTS["test"]),
        validation_window_count=COUNTS["validation"], test_window_count=COUNTS["test"],
        dataset_id=DATASET_ID,
    )
    payload = build_checkpoint_payload(
        model=model, input_channel_order=order, linear_hidden_layers=2,
        dataset_binding=_binding(), normalization=normalization,
        run={"task_id": 22, "optimizer": "Adam", "loss": "normalized_mse_grid_uniform",
             "best_epoch": 1, "epochs_completed": 1, "run_complete": True},
        metrics=metrics, split_window_counts=COUNTS, split_ranges=ranges, task_id=22,
    )
    return payload, model, config


def test_metrics_include_all_leads_and_cumulative_24_48_72_hours():
    metrics = _metrics()
    assert [row["lead_step"] for row in metrics["by_lead"]] == list(range(1, 25))
    assert [row["lead_hours"] for row in metrics["by_lead"]] == list(range(3, 73, 3))
    assert metrics["overall"]["mae"] == pytest.approx(12.5)
    for row, steps in zip(metrics["by_horizon"], (8, 16, 24)):
        assert row["horizon_hours"] == steps * 3
        assert row["lead_steps"] == steps
        assert row["mae"] == pytest.approx((steps + 1) / 2)
        assert row["rmse"] == pytest.approx(np.sqrt(np.mean(np.arange(1, steps + 1) ** 2)))


def test_chunked_spatial_and_batch_metrics_match_full_array():
    rng = np.random.default_rng(33)
    truth = rng.normal(280, 8, (4, 24, 1, 6, 8))
    prediction = truth + rng.normal(0, 2, truth.shape)
    expected = compute_earth_metrics(prediction, truth, dataset_id=DATASET_ID)
    accumulator = ErrorAccumulator(dataset_id=DATASET_ID)
    for sample in range(4):
        for latitude in range(0, 6, 2):
            accumulator.update(prediction[sample:sample + 1, :, :, latitude:latitude + 2],
                               truth[sample:sample + 1, :, :, latitude:latitude + 2])
    actual = accumulator.result()
    for key in ("rmse", "mae"):
        assert actual["overall"][key] == pytest.approx(expected["overall"][key], rel=1e-12)
        for group in ("by_lead", "by_horizon"):
            assert [row[key] for row in actual[group]] == pytest.approx([row[key] for row in expected[group]], rel=1e-12)


@pytest.mark.parametrize("shape", [(1, 0, 1, 2, 2), (1, 24, 2, 2, 2), (1, 241, 1, 2, 2)])
def test_threehour_metrics_reject_wrong_horizon_or_target_channels(shape):
    with pytest.raises(EarthArtifactError):
        compute_earth_metrics(np.zeros(shape), np.zeros(shape), dataset_id=DATASET_ID)


def test_daily_metrics_keep_their_existing_day_labels():
    metrics = compute_earth_metrics(np.ones((1, 3, 1, 2, 2)), np.zeros((1, 3, 1, 2, 2)))
    assert [row["lead_day"] for row in metrics["by_lead"]] == [1, 2, 3]
    assert "by_horizon" not in metrics


@pytest.mark.parametrize("order", [("TO3",), ("TO3", "V10M", "SWGDN"), ("TO3", "U10M", "V10M", "T2M", "SWGDN")])
def test_threehour_checkpoint_roundtrip_preserves_contract_and_tile_output(tmp_path, order):
    payload, model, config = _payload(order)
    target = tmp_path / "earth_3hourly_22.pth"
    save_earth_artifact_atomic(payload, target)
    checkpoint = load_earth_training_artifact(
        target, expected_binding=_binding(), expected_task_id=22,
        expected_hyperparameters={"training_dataset": DATASET_ID, "window": 56, "horizon": 24,
                                  "selected_channels": list(order[1:])},
    )
    rebuilt, warnings = build_earth_model_from_checkpoint(checkpoint)
    assert not warnings
    assert checkpoint.payload["artifact_schema"] == EARTH_3HOURLY_ARTIFACT_SCHEMA
    assert checkpoint.model_config["implementation_id"] == EARTH_3HOURLY_IMPLEMENTATION_ID
    assert checkpoint.metrics["schema"] == EARTH_3HOURLY_METRICS_SCHEMA_V2
    contract = checkpoint.training_contract
    assert (contract["window"], contract["horizon"], contract["frequency_hours"]) == (56, 24, 3)
    assert contract["step_unit"] == "hour" and contract["step"] == 3
    assert contract["time_zone"] == "UTC" and contract["timestamp_rule"] == "interval_center"
    assert contract["time_aggregation"] == "three_hour_mean" and contract["timestamp_offset_minutes"] == 90
    assert contract["grid_shape"] == [240, 480]
    assert checkpoint.normalization == payload["normalization"]
    assert config["window"] == 56 and config["horizon"] == 24
    assert [config["height"], config["width"]] == [240, 480]
    model.eval()
    probe = torch.randn(1, 56, len(order), 24, 48)
    with torch.no_grad():
        first = earth_forward_for_model(model, probe, model_source="official", horizon=24, height=24, width=48)
        second = earth_forward_for_model(rebuilt, probe, model_source="official", horizon=24, height=24, width=48)
    assert first.shape == (1, 24, 1, 24, 48)
    assert torch.equal(first, second)
    raw = torch.load(target, map_location="cpu", weights_only=True)
    json.dumps({key: value for key, value in raw.items() if key != "model_state_dict"})


def test_strict_threehour_reload_probe_has_bounded_spatial_shape(tmp_path, monkeypatch):
    import services.earth_training_artifact as artifact
    payload, _, _ = _payload()
    seen = []
    original = artifact.earth_forward_for_model
    def checked(model, inputs, **kwargs):
        seen.append(tuple(inputs.shape))
        return original(model, inputs, **kwargs)
    monkeypatch.setattr(artifact, "earth_forward_for_model", checked)
    save_earth_artifact_atomic(payload, tmp_path / "bounded.pth")
    assert seen == [(2, 56, 2, 24, 48)] * 2


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(artifact_schema="astraatmos_earth_forecast_checkpoint_v1"),
    lambda p: p["model_config"].update(window=7),
    lambda p: p["model_config"].update(horizon=3),
    lambda p: p["model_config"].update(height=36),
    lambda p: p["model_config"].update(implementation_id="astraatmos_gridpoint_dlinear_v1"),
    lambda p: p["training_contract"].update(step_unit="day"),
    lambda p: p["training_contract"].update(step=1),
    lambda p: p["training_contract"].update(timestamp_rule="date_only"),
    lambda p: p["training_contract"].update(time_zone="local"),
    lambda p: p["training_contract"].update(grid_shape=[36, 72]),
    lambda p: p["training_contract"].update(split_policy="legacy_compatibility"),
    lambda p: p["training_contract"]["split_ranges"]["validation"].update(date_start="2020-12-31T01:30:00Z"),
    lambda p: p["normalization"].update(fit_time_end="2021-01-01T01:30:00Z"),
    lambda p: p["normalization"].update(fit_step_count=5848),
    lambda p: p["normalization"].update(target_channel_index=1),
    lambda p: p["normalization"].update(ddof=1),
    lambda p: p["normalization"].update(epsilon=1),
    lambda p: p["normalization"].update(constant_channel_mask=[1, 1]),
    lambda p: p["dataset_binding"].update(dataset_id="earth_merra2_daily_v2"),
    lambda p: p["metrics"].update(schema="earth_training_metrics_v1"),
    lambda p: p["metrics"]["splits"]["test"]["by_lead"][0].update(lead_hours=24),
    lambda p: p["metrics"]["splits"]["test"]["by_horizon"][0].update(rmse=999),
])
def test_daily_contract_fields_or_tampered_threehour_fields_are_rejected(mutate):
    payload, _, _ = _payload()
    mutate(payload)
    with pytest.raises((EarthArtifactError, ValueError)):
        validate_checkpoint_payload(payload)


def test_threehour_checkpoint_cannot_load_as_daily_task(tmp_path):
    payload, _, _ = _payload()
    target = tmp_path / "3hourly.pth"
    save_earth_artifact_atomic(payload, target)
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(target, expected_hyperparameters={"training_dataset": "earth_merra2_daily_v2", "window": 7})


def test_old_daily_checkpoint_stays_loadable_but_cannot_load_as_threehour_task(earth_global_release, tmp_path, monkeypatch):
    from services.dataset_registry import DatasetRegistry
    from services.earth_training_artifact import normalization_from_release
    # Archived artifacts can still be read; the public daily runtime remains retired.
    # Build this historical fixture through a test-only registry admission override.
    import services.dataset_registry as registry_module
    monkeypatch.setattr(registry_module, "require_active_dataset", lambda *_: None)
    registry = DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **earth_global_release)
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    model, _, _ = build_earth_model_for_plan(ModelSourcePlan("official", ["TO3"], 2))
    metrics = compute_earth_metrics(np.ones((1, 3, 1, 2, 2)), np.zeros((1, 3, 1, 2, 2)))
    payload = build_checkpoint_payload(
        model=model, input_channel_order=["TO3"], linear_hidden_layers=2,
        dataset_binding=binding, normalization=normalization_from_release(release, ["TO3"]),
        run={"task_id": 11, "optimizer": "Adam", "loss": "mse", "best_epoch": 1,
             "epochs_completed": 1, "run_complete": True},
        metrics=build_metrics_block(validation=metrics, test=metrics,
                                    validation_window_count=172, test_window_count=175),
        split_window_counts={"train": 357, "validation": 172, "test": 175},
    )
    target = tmp_path / "daily.pth"
    save_earth_artifact_atomic(payload, target)
    checkpoint = load_earth_training_artifact(target, expected_binding=binding)
    assert checkpoint.training_contract["window"] == 7 and checkpoint.training_contract["horizon"] == 3
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(target, expected_binding=_binding())
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(target, expected_hyperparameters={"training_dataset": DATASET_ID, "window": 56, "horizon": 24})


def test_uploaded_threehour_plan_requires_a_verified_source():
    plan = ModelSourcePlan("uploaded", ["TO3"], 2, uploaded_model={}, dataset_id=DATASET_ID)
    with pytest.raises(EarthModelBuildError) as error:
        build_earth_model_for_plan(plan)
    assert error.value.code == "uploaded_model_missing"


def test_threehour_checkpoint_is_not_accepted_as_mars_transfer_weights(tmp_path):
    from services.training_weight_service import TrainingWeightService
    payload, _, _ = _payload()
    target = tmp_path / "threehour-transfer.pth"
    torch.save(payload, target)
    report = TrainingWeightService._validate_weight_file(target)
    assert report["ok"] is False
    assert report["artifact_kind"] == "earth_forecast_checkpoint"


@pytest.mark.parametrize("function", [available_origin_range, lambda r: forecast_dates(r, "2020-01-07"),
                                      lambda r: origin_split(r, "2020-01-07")])
def test_daily_prediction_date_helpers_reject_threehour_release(function):
    release = SimpleNamespace(metadata={"dataset_id": DATASET_ID})
    with pytest.raises(EarthArtifactError) as error:
        function(release)
    assert error.value.code == "dataset_prediction_not_supported"
