"""Physical TO3 formulas, streaming aggregation and versioned artifact validation."""
import copy
import asyncio
import json

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from services.dataset_identity import EARTH_DATASET_3HOURLY_ID as DATASET_ID
from services.earth_training_artifact import (
    ErrorAccumulator, EarthArtifactError, FULL_METRIC_KEYS, METRIC_UNITS,
    compute_earth_metrics, validate_checkpoint_payload, load_earth_training_artifact,
    save_earth_artifact_atomic, build_earth_model_from_checkpoint,
)
from services.earth_dataset import cell_area_weights_from_lat_bounds
from services.earth_training_contract import EARTH_3HOURLY_METRICS_SCHEMA
from test_earth_3hourly_artifact import _payload
from test_earth_3hourly_training_service import service_environment


def direct(prediction, truth):
    prediction, truth = prediction.ravel(), truth.ravel()
    error = prediction - truth
    mse = np.mean(error ** 2)
    m2 = np.sum((truth - truth.mean()) ** 2)
    return dict(mse=mse, rmse=np.sqrt(mse), mae=np.mean(np.abs(error)),
                r2=1 - np.sum(error ** 2) / m2 if m2 > 0 else float(np.sum(error ** 2) == 0),
                mape=100 * np.mean(np.abs(error) / (np.abs(truth) + 1e-8)),
                smape=100 * np.mean(2 * np.abs(error) / (np.abs(truth) + np.abs(prediction) + 1e-8)))


def shaped(values):
    return np.asarray(values, dtype=np.float64).reshape(1, 1, 1, 1, -1)


def test_area_weighted_tile_loss_keeps_global_tile_share_in_denominator():
    from models.training_scripts.earth_daily import _weighted_mse

    # These two synthetic tiles represent a high-area and low-area tile from a
    # 100-tile global partition. Local normalization would make both tiles
    # contribute equally; the fixed 1/100 denominator preserves global area.
    output = torch.tensor([[[[[1.0]]]], [[[[3.0]]]]])
    target = torch.zeros_like(output)
    weights = torch.tensor([[[0.90]], [[0.01]]])
    loss, denominator = _weighted_mse(
        output, target, weights, sample_weight_denominator=0.01,
    )
    assert denominator.item() == pytest.approx(0.02)
    assert loss.item() == pytest.approx((0.90 + 9 * 0.01) / 0.02)


def test_cell_area_weights_use_lat_bounds_and_sum_to_one():
    bounds = np.array([[-90.0, 0.0], [0.0, 90.0]])
    weights = cell_area_weights_from_lat_bounds(bounds, 2)
    expected_row = np.array([0.25, 0.25])
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(expected_row)
    assert weights[1] == pytest.approx(expected_row)


def test_hand_calculated_negative_r2():
    metrics = compute_earth_metrics(shaped([2, 2, 1]), shaped([1, 2, 3]), DATASET_ID)["overall"]
    assert metrics == pytest.approx(dict(mse=5/3, rmse=np.sqrt(5/3), mae=1, r2=-1.5,
        mape=100/3 * (1/(1 + 1e-8) + 2/(3 + 1e-8)),
        smape=100/3 * (2/(3 + 1e-8) + 4/(4 + 1e-8))), rel=1e-14)


@pytest.mark.parametrize("truth,prediction", [([1, 2, 3], [1, 2, 3]), ([7, 7, 7], [7, 7, 7]),
    ([7, 7, 7], [8, 8, 8]), ([0, 0], [0, 0]), ([0, 1e-12], [1, 1])])
def test_perfect_constant_zero_and_near_zero(truth, prediction):
    result = compute_earth_metrics(shaped(prediction), shaped(truth), DATASET_ID)
    assert result["overall"] == pytest.approx(direct(shaped(prediction), shaped(truth)))
    assert all(np.isfinite(value) for value in result["overall"].values())


@pytest.mark.parametrize("horizon", [1, 9, 25, 240])
def test_streaming_batches_spatial_blocks_and_tail_match_one_shot(horizon):
    rng = np.random.default_rng(82)
    truth = rng.normal(280, 5, (5, horizon, 1, 3, 7))
    truth[:, :, :, 0, 0] = 0
    prediction = truth + rng.normal(0, 2, truth.shape)
    expected = compute_earth_metrics(prediction, truth, DATASET_ID)
    for batch_size, height, width in [(1, 1, 2), (2, 2, 3), (4, 3, 7)]:
        accumulator = ErrorAccumulator(DATASET_ID, horizon)
        for batch in range(0, 5, batch_size):
            for lat in range(0, 3, height):
                for lon in range(0, 7, width):
                    sl = (slice(batch, batch + batch_size), slice(None), slice(None), slice(lat, lat + height), slice(lon, lon + width))
                    accumulator.update(prediction[sl], truth[sl])
        actual = accumulator.result()
        assert actual["overall"] == pytest.approx(direct(prediction, truth), rel=1e-12)
        for group in ("by_lead", "by_horizon"):
            for a, b in zip(actual[group], expected[group]):
                assert {key: a[key] for key in FULL_METRIC_KEYS} == pytest.approx({key: b[key] for key in FULL_METRIC_KEYS}, rel=1e-12)
        for row in actual["by_horizon"]:
            assert {key: row[key] for key in FULL_METRIC_KEYS} == pytest.approx(direct(prediction[:, :row["lead_steps"]], truth[:, :row["lead_steps"]]), rel=1e-12)
        assert actual["by_horizon"][-1]["horizon_hours"] == horizon * 3


def test_stable_target_variance_and_cumulative_r2_not_mean_of_lead_r2():
    truth = (1e8 + np.arange(27, dtype=float)).reshape(3, 3, 1, 1, 3)
    prediction = truth + np.arange(3)[None, :, None, None, None] * 3
    accumulator = ErrorAccumulator(DATASET_ID, 3)
    for i in range(3):
        accumulator.update(prediction[i:i+1], truth[i:i+1])
    result = accumulator.result()
    assert result["overall"] == pytest.approx(direct(prediction, truth), rel=1e-12)
    assert result["overall"]["r2"] != pytest.approx(np.mean([row["r2"] for row in result["by_lead"]]))


@pytest.mark.parametrize("model_source", ["official", "uploaded"])
def test_evaluation_uses_denormalized_physical_values_and_keeps_loss_separate(model_source):
    from models.training_scripts.earth_daily import _evaluate_split

    class Echo(torch.nn.Module):
        _astraatmos_earth_3hourly_contract = model_source == "uploaded"

        def forward(self, inputs, time=None):
            return inputs

    dataset = SimpleNamespace(horizon=9, _release=SimpleNamespace(metadata={"dataset_id": DATASET_ID}),
                              denormalize_ozone=lambda value: value * 10 + 100)
    truth = (torch.arange(5 * 9 * 6, dtype=torch.float32).reshape(5, 9, 1, 2, 3) / 100).repeat(1, 1, 1, 12, 16)
    prediction = truth + .2
    loader = [(prediction[i:i+2], truth[i:i+2]) for i in range(0, 5, 2)]
    result, loss = _evaluate_split(Echo(), loader, dataset, torch.device("cpu"),
                                   loss_function=torch.nn.MSELoss(), model_source=model_source)
    expected = direct((prediction * 10 + 100).numpy(), (truth * 10 + 100).numpy())
    assert result["overall"] == pytest.approx(expected, rel=2e-6)
    assert result["overall"]["mse"] == pytest.approx(4, rel=2e-6)
    assert loss == pytest.approx(.04, rel=2e-6)
    assert [row["horizon_hours"] for row in result["by_horizon"]] == [24, 27]


@pytest.mark.parametrize("group", ["overall", "by_lead", "by_horizon"])
@pytest.mark.parametrize("key", FULL_METRIC_KEYS)
@pytest.mark.parametrize("bad", [None, True, "1.2", float("nan"), float("inf")])
def test_v2_rejects_missing_illegal_and_nonfinite_metrics(group, key, bad):
    payload, _, _ = _payload()
    row = payload["metrics"]["splits"]["test"][group]
    row = row if group == "overall" else row[0]
    if bad is None:
        row.pop(key)
    else:
        row[key] = bad
    with pytest.raises(EarthArtifactError):
        validate_checkpoint_payload(payload)


def legacy_payload(payload):
    payload = copy.deepcopy(payload)
    metrics = payload["metrics"]
    metrics["schema"] = EARTH_3HOURLY_METRICS_SCHEMA
    metrics.pop("metric_units", None)
    metrics.pop("metric_policy", None)
    for split in metrics["splits"].values():
        for row in [split["overall"], *split["by_lead"], *split["by_horizon"]]:
            for key in (*FULL_METRIC_KEYS, "target_statistics"):
                if key not in ("rmse", "mae"):
                    row.pop(key, None)
    return payload


def test_legacy_checkpoint_stays_readable_predictable_and_unmodified(tmp_path):
    payload, _, _ = _payload()
    legacy = legacy_payload(payload)
    path = tmp_path / "legacy.pth"
    save_earth_artifact_atomic(legacy, path)
    before = path.read_bytes()
    checkpoint = load_earth_training_artifact(path)
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    assert model is not None
    assert checkpoint.metrics == legacy["metrics"]
    assert set(checkpoint.metrics["splits"]["test"]["overall"]) == {"rmse", "mae"}
    assert path.read_bytes() == before
    with pytest.raises(EarthArtifactError, match="required metrics contract"):
        load_earth_training_artifact(path, expected_hyperparameters={"_earth_metrics_schema": "earth_training_metrics_3hourly_v2"})


def test_v2_preserves_units_and_rejects_tampered_policy_or_statistics():
    payload, _, _ = _payload()
    assert validate_checkpoint_payload(payload)["metrics"]["metric_units"] == METRIC_UNITS
    for mutation in [lambda p: p["metrics"]["metric_units"].update(mse="DU"),
                     lambda p: p["metrics"]["metric_policy"].update(denominator_epsilon=1e-6),
                     lambda p: p["metrics"]["splits"]["test"]["by_lead"][0]["target_statistics"].update(m2=12),
                     lambda p: p["metrics"]["splits"]["test"]["by_horizon"][0].update(r2=-2)]:
        changed = copy.deepcopy(payload)
        mutation(changed)
        with pytest.raises(EarthArtifactError):
            validate_checkpoint_payload(changed)


def test_checkpoint_rejects_partial_grid_metrics_and_numeric_overflow():
    payload, _, _ = _payload()
    for split in payload["metrics"]["splits"].values():
        for row in split["by_lead"]:
            row["target_statistics"]["count"] //= 2
    with pytest.raises(EarthArtifactError, match="every forecast-origin/grid"):
        validate_checkpoint_payload(payload)
    payload, _, _ = _payload()
    payload["metrics"]["splits"]["test"]["overall"]["mse"] = 10 ** 500
    with pytest.raises(EarthArtifactError, match="finite"):
        validate_checkpoint_payload(payload)


@pytest.mark.parametrize("case,expected", [("v2_missing_task_split", "failed"), ("legacy", "completed"),
    ("new_task_legacy_artifact", "failed"), ("missing_r2", "failed")])
def test_restart_recovers_checkpoint_metrics_and_enforces_new_task_version(service_environment, tmp_path, case, expected):
    from database.models import ModelTrainingTask
    module, _ = service_environment
    payload, _, _ = _payload()
    if case in ("legacy", "new_task_legacy_artifact"):
        payload = legacy_payload(payload)
    elif case == "missing_r2":
        payload["metrics"]["splits"]["test"]["overall"].pop("r2")
    path = tmp_path / "restart.pth"
    torch.save(payload, path)
    log = tmp_path / "restart.log"
    log.write_text(f"Model saved: {path}\n", encoding="utf-8")
    binding = copy.deepcopy(payload["dataset_binding"])
    binding["dataset_snapshot"] = json.dumps(binding["dataset_snapshot"])
    hypers = {"training_dataset": DATASET_ID, "window": 56, "horizon": 24, "selected_channels": ["U10M"], "model_source": "official"}
    if case != "legacy":
        hypers["_earth_metrics_schema"] = "earth_training_metrics_3hourly_v2"

    async def scenario():
        async with module.async_session_maker() as session:
            session.add(ModelTrainingTask(id=22, user_id=1, model_script="earth_daily.py", model_source="official",
                status="running", output_model_path=str(path), log_file_path=str(log), hyperparameters=json.dumps(hypers), **binding))
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
