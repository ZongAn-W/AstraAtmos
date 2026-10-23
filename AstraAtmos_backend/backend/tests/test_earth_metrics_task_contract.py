"""Frozen v2/v3 task contracts stay strict across artifact reads and reloads."""

import copy

import numpy as np
import pytest
import torch

from services.dataset_identity import EARTH_DATASET_3HOURLY_ID as DATASET_ID
from services.earth_model_source import earth_forward_for_model
from services.earth_task_split import (
    EARTH_TASK_SPLIT_POLICY, build_earth_task_split, required_task_split, utc,
)
from services.earth_training_artifact import (
    EarthArtifactError, build_earth_model_from_checkpoint, build_metrics_block,
    compute_earth_metrics, load_earth_training_artifact, save_earth_artifact_atomic,
    validate_checkpoint_payload,
)
from services.earth_training_contract import (
    EARTH_3HOURLY_METRICS_SCHEMA_V2 as V2,
    EARTH_3HOURLY_METRICS_SCHEMA_V3 as V3,
    CELL_AREA_WEIGHTED_AGGREGATION as AREA,
    GRID_UNIFORM_AGGREGATION as UNIFORM,
)
from services.training_split import TrainingSplitError
from test_earth_3hourly_artifact import _payload


def _split_metrics(window_count, aggregation):
    truth = np.full((2, 24, 1, 2, 3), 280.0)
    errors = np.arange(1, 25, dtype=float)[None, :, None, None, None]
    errors = errors + np.array([0.0, 10.0])[None, None, None, :, None]
    weights = np.array([[0.05] * 3, [0.85 / 3] * 3]) if aggregation == AREA else None
    metrics = compute_earth_metrics(truth + errors, truth, DATASET_ID, spatial_weights=weights)
    # This synthetic pattern is repeated over each whole global grid/origin.
    # Constant targets preserve the moments exactly under this repetition.
    for row in metrics["by_lead"]:
        row["target_statistics"]["count"] = window_count * 240 * 480
        if aggregation == AREA:
            row["target_statistics"]["weight_sum"] = float(window_count)
    return metrics


def _task_payload(schema):
    payload, model, _ = _payload()
    aggregation = AREA if schema == V3 else UNIFORM
    dates = np.datetime64("2020-01-01T01:30", "ns") + np.arange(5848) * np.timedelta64(3, "h")
    ratios = {"train_ratio": .7, "validation_ratio": .2, "test_ratio": .1}
    split = build_earth_task_split(dates, 56, 24, ratios)
    counts = {name: row["window_count"] for name, row in split["ranges"].items()}
    payload["dataset_binding"]["dataset_snapshot"]["time"].update(
        start=utc(dates[0]), end=utc(dates[-1]), count=len(dates),
    )
    payload["training_contract"].update(
        split_policy=EARTH_TASK_SPLIT_POLICY, task_split=copy.deepcopy(split),
        split_ratios=dict(ratios), split_ranges=copy.deepcopy(split["ranges"]),
        split_window_counts=counts,
    )
    train = split["ranges"]["train"]
    payload["normalization"].update(
        task_split=copy.deepcopy(split), fit_step_count=train["step_count"],
        fit_date_start=train["date_start"], fit_date_end=train["date_end"],
        fit_time_start=train["date_start"], fit_time_end=train["date_end"],
    )
    hypers = {
        **ratios, "training_dataset": DATASET_ID, "window": 56, "horizon": 24,
        "selected_channels": ["U10M"], "model_source": "official",
        "_earth_metrics_schema": schema, "_earth_metrics_aggregation": aggregation,
        "_earth_task_split": copy.deepcopy(split),
        "_earth_task_split_policy": EARTH_TASK_SPLIT_POLICY,
    }
    payload["run"].update(
        task_split_policy=EARTH_TASK_SPLIT_POLICY,
        hyperparameters=copy.deepcopy(hypers),
        loss="normalized_mse_cell_area_weighted" if schema == V3 else "normalized_mse_grid_uniform",
    )
    payload["metrics"] = build_metrics_block(
        validation=_split_metrics(counts["validation"], aggregation),
        test=_split_metrics(counts["test"], aggregation),
        validation_window_count=counts["validation"], test_window_count=counts["test"],
        dataset_id=DATASET_ID, horizon=24, aggregation=aggregation,
    )
    return payload, model, hypers


@pytest.mark.parametrize("schema,aggregation", [(V2, UNIFORM), (V3, AREA)])
def test_frozen_task_roundtrip_preserves_statistics_weights_and_model(tmp_path, schema, aggregation):
    payload, source_model, hypers = _task_payload(schema)
    path = tmp_path / "task.pth"
    save_earth_artifact_atomic(payload, path)
    before = path.read_bytes()
    checkpoint = load_earth_training_artifact(
        path, expected_hyperparameters=hypers, expected_task_id=22,
        expected_binding=payload["dataset_binding"],
    )
    rebuilt, warnings = build_earth_model_from_checkpoint(checkpoint)
    assert not warnings
    assert checkpoint.metrics["schema"] == schema
    assert checkpoint.metrics["aggregation"] == aggregation
    assert checkpoint.metrics == payload["metrics"]
    probe = torch.randn(1, 56, 2, 2, 3)
    source_model.eval()
    with torch.no_grad():
        kwargs = {"model_source": "official", "horizon": 24, "height": 2, "width": 3}
        assert torch.equal(earth_forward_for_model(source_model, probe, **kwargs),
                           earth_forward_for_model(rebuilt, probe, **kwargs))
    assert path.read_bytes() == before
    # Pre-aggregation-marker v2 tasks derive the same frozen equal-weight policy.
    if schema == V2:
        hypers.pop("_earth_metrics_aggregation")
        assert load_earth_training_artifact(path, expected_hyperparameters=hypers).metrics == payload["metrics"]


def test_v2_and_v3_have_distinct_synthetic_statistical_results():
    v2, _, _ = _task_payload(V2)
    v3, _, _ = _task_payload(V3)
    assert v2["metrics"]["splits"]["test"]["overall"]["mae"] == pytest.approx(17.5)
    assert v3["metrics"]["splits"]["test"]["overall"]["mae"] == pytest.approx(21.0)


@pytest.mark.parametrize("schema,other,aggregation", [(V2, V3, AREA), (V3, V2, UNIFORM)])
def test_artifact_cannot_be_read_under_other_frozen_task_version(tmp_path, schema, other, aggregation):
    payload, _, hypers = _task_payload(schema)
    path = tmp_path / "version-mismatch.pth"
    torch.save(payload, path)
    hypers.update(_earth_metrics_schema=other, _earth_metrics_aggregation=aggregation)
    with pytest.raises(EarthArtifactError, match="required metrics contract"):
        load_earth_training_artifact(path, expected_hyperparameters=hypers)


@pytest.mark.parametrize("schema", [V2, V3])
@pytest.mark.parametrize("mutation", ["aggregation", "policy", "loss", "run_version", "unknown_schema"])
def test_checkpoint_statistical_contract_mutations_are_rejected(schema, mutation):
    payload, _, _ = _task_payload(schema)
    other_schema, other_aggregation = (V2, UNIFORM) if schema == V3 else (V3, AREA)
    if mutation == "aggregation":
        payload["metrics"]["aggregation"] = other_aggregation
    elif mutation == "policy":
        payload["metrics"]["metric_policy"]["spatial_weighting"] = "unverified_weights"
    elif mutation == "loss":
        payload["run"]["loss"] = "normalized_mse_grid_uniform" if schema == V3 else "normalized_mse_cell_area_weighted"
    elif mutation == "run_version":
        payload["run"]["hyperparameters"].update(
            _earth_metrics_schema=other_schema, _earth_metrics_aggregation=other_aggregation,
        )
    else:
        payload["metrics"]["schema"] = "earth_training_metrics_3hourly_v999"
    with pytest.raises(EarthArtifactError):
        validate_checkpoint_payload(payload)


@pytest.mark.parametrize("schema,aggregation", [(V2, UNIFORM), (V3, AREA)])
def test_known_task_markers_require_and_preserve_frozen_split(schema, aggregation):
    _, _, hypers = _task_payload(schema)
    assert required_task_split(hypers) == hypers["_earth_task_split"]
    hypers.pop("_earth_task_split")
    with pytest.raises(TrainingSplitError, match="required frozen partition"):
        required_task_split(hypers)


@pytest.mark.parametrize("markers", [
    {"_earth_metrics_schema": None},
    {"_earth_metrics_schema": "future_metrics"},
    {"_earth_metrics_schema": "earth_training_metrics_3hourly_v1"},
    {"_earth_metrics_schema": V3, "_earth_metrics_aggregation": UNIFORM},
    {"_earth_metrics_schema": V2, "_earth_metrics_aggregation": AREA},
    {"_earth_metrics_schema": V3, "_earth_metrics_aggregation": None},
    {"_earth_metrics_aggregation": AREA},
])
def test_task_split_rejects_unknown_or_inconsistent_metric_markers(markers):
    _, _, hypers = _task_payload(V3)
    hypers.pop("_earth_metrics_schema")
    hypers.pop("_earth_metrics_aggregation")
    hypers.update(markers)
    with pytest.raises(TrainingSplitError):
        required_task_split(hypers)


def test_unknown_task_schema_still_rejects_a_valid_artifact(tmp_path):
    payload, _, hypers = _task_payload(V3)
    path = tmp_path / "unknown-task-contract.pth"
    torch.save(payload, path)
    hypers["_earth_metrics_schema"] = "future_metrics"
    with pytest.raises(EarthArtifactError, match="unsupported"):
        load_earth_training_artifact(path, expected_hyperparameters=hypers)
