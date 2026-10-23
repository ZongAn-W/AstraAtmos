"""Earth checkpoint: DU metrics, strict validation, atomic save and CPU reload."""

import json

import numpy as np
import pytest
import torch

from services.dataset_registry import DatasetRegistry
from services.earth_dataset import CHANNELS
from services.earth_training_artifact import (
    EarthArtifactError,
    ErrorAccumulator,
    assert_release_matches_binding,
    available_origin_range,
    build_checkpoint_payload,
    build_earth_model_from_checkpoint,
    build_metrics_block,
    compute_earth_metrics,
    forecast_dates,
    load_earth_training_artifact,
    normalization_from_release,
    save_earth_artifact_atomic,
)
from training_backbones.earth_daily_model import (
    create_earth_forecaster,
    earth_forward,
    state_dict_to_cpu,
)


def _registry(fixture):
    return DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **fixture)


def _split_metrics(rmse=1.0, mae=0.5):
    return {
        "overall": {"rmse": rmse, "mae": mae},
        "by_lead": [
            {"lead_day": 1, "rmse": rmse, "mae": mae},
            {"lead_day": 2, "rmse": rmse * 2, "mae": mae * 2},
            {"lead_day": 3, "rmse": rmse * 3, "mae": mae * 3},
        ],
    }


def _payload(earth_global_release, *, order=("TO3", "U10M"), task_id=11, model=None):
    registry = _registry(earth_global_release)
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    normalization = normalization_from_release(
        registry.get_earth_snapshot("earth_merra2_daily_v2"), list(order)
    )
    forecaster = model or create_earth_forecaster(list(order))
    return {
        "payload": build_checkpoint_payload(
            model=forecaster,
            input_channel_order=list(order),
            linear_hidden_layers=2,
            dataset_binding=binding,
            normalization=normalization,
            run={
                "task_id": task_id,
                "seed": 11,
                "optimizer": "Adam",
                "loss": "normalized_mse_grid_uniform",
                "best_epoch": 1,
                "epochs_completed": 1,
                "best_validation_loss": 0.5,
                "stopped_early": False,
                "run_complete": True,
                "device": "cpu",
                "torch_version": torch.__version__,
            },
            metrics=build_metrics_block(
                validation=_split_metrics(2.0, 1.0),
                test=_split_metrics(3.0, 1.5),
                validation_window_count=172,
                test_window_count=175,
            ),
            split_window_counts={"train": 357, "validation": 172, "test": 175},
            task_id=task_id,
        ),
        "binding": binding,
        "normalization": normalization,
        "model": forecaster,
    }


# ── metrics ────────────────────────────────────────────────────────────────

def test_metrics_are_physical_and_split_by_lead():
    reference = np.full((2, 3, 1, 36, 72), 250.0)
    prediction = reference + np.array([1.0, 2.0, 3.0])[None, :, None, None, None]
    result = compute_earth_metrics(prediction, reference)
    assert result["overall"]["mae"] == pytest.approx(2.0)
    assert result["overall"]["rmse"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert [row["lead_day"] for row in result["by_lead"]] == [1, 2, 3]
    assert [row["rmse"] for row in result["by_lead"]] == pytest.approx([1, 2, 3])
    assert [row["mae"] for row in result["by_lead"]] == pytest.approx([1, 2, 3])


def test_metrics_reject_shape_mismatch_and_non_finite_values():
    reference = np.zeros((1, 3, 1, 36, 72))
    with pytest.raises(EarthArtifactError):
        compute_earth_metrics(np.zeros((1, 3, 1, 36, 71)), reference)
    with pytest.raises(EarthArtifactError):
        compute_earth_metrics(np.zeros((1, 3, 1, 36)), reference)
    broken = reference.copy()
    broken[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(EarthArtifactError):
        compute_earth_metrics(broken, reference)


def test_accumulator_matches_the_array_version_exactly():
    rng = np.random.default_rng(7)
    prediction = rng.normal(280.0, 20.0, size=(5, 3, 1, 36, 72))
    reference = rng.normal(280.0, 20.0, size=(5, 3, 1, 36, 72))
    expected = compute_earth_metrics(prediction, reference)
    accumulator = ErrorAccumulator()
    for start in range(0, len(prediction), 2):
        accumulator.update(prediction[start:start + 2], reference[start:start + 2])
    actual = accumulator.result()
    # Batch-wise accumulation matches the whole-array result up to float64
    # summation order, so the tolerance is far below any physical significance.
    for key in ("overall",):
        assert actual[key]["rmse"] == pytest.approx(expected[key]["rmse"], rel=1e-12)
        assert actual[key]["mae"] == pytest.approx(expected[key]["mae"], rel=1e-12)
    for actual_row, expected_row in zip(actual["by_lead"], expected["by_lead"]):
        assert actual_row["rmse"] == pytest.approx(expected_row["rmse"], rel=1e-12)
        assert actual_row["mae"] == pytest.approx(expected_row["mae"], rel=1e-12)


def test_accumulator_requires_at_least_one_sample():
    with pytest.raises(EarthArtifactError):
        ErrorAccumulator().result()


def test_metrics_block_carries_the_window_count_inside_each_split():
    block = build_metrics_block(
        validation=_split_metrics(), test=_split_metrics(),
        validation_window_count=172, test_window_count=175,
    )
    assert block["unit"] == "DU"
    assert block["target"] == "TO3"
    assert block["aggregation"] == "forecast_origin_lead_grid_uniform"
    assert block["splits"]["test"]["window_count"] == 175
    assert block["splits"]["validation"]["window_count"] == 172
    with pytest.raises(EarthArtifactError):
        build_metrics_block(
            validation=_split_metrics(), test={"overall": {"rmse": float("inf")}},
            validation_window_count=1, test_window_count=1,
        )


# ── save / load round trip ─────────────────────────────────────────────────

@pytest.mark.parametrize("order", [("TO3",), ("TO3", "U10M", "SWGDN"), tuple(CHANNELS)])
def test_cpu_round_trip_reproduces_the_standardized_output(earth_global_release, tmp_path, order):
    prepared = _payload(earth_global_release, order=order)
    target = tmp_path / "task_11_earth.pth"
    from services.earth_training_artifact import save_earth_artifact_atomic

    saved = save_earth_artifact_atomic(prepared["payload"], target, strict_reload=True)
    assert saved == str(target)
    assert target.is_file()
    assert not (tmp_path / "task_11_earth.pth.partial").exists()

    checkpoint = load_earth_training_artifact(
        saved,
        expected_binding=prepared["binding"],
        expected_task_id=11,
        expected_hyperparameters={"selected_channels": list(order[1:])},
    )
    model, _warnings = build_earth_model_from_checkpoint(checkpoint)
    inputs = torch.randn(2, 7, len(order), 36, 72)
    original = prepared["model"]
    original.eval()
    with torch.no_grad():
        assert torch.equal(earth_forward(original, inputs), earth_forward(model, inputs))

    # The container only holds plain values and CPU tensors.
    raw = torch.load(target, map_location="cpu", weights_only=True)
    assert raw["artifact_schema"] == "astraatmos_earth_forecast_checkpoint_v1"
    for key, value in raw["model_state_dict"].items():
        assert isinstance(value, torch.Tensor)
        assert value.device.type == "cpu"
        assert not value.requires_grad
    assert isinstance(raw["run"]["torch_version"], str)
    assert type(raw["run"]["task_id"]) is int
    json.dumps(raw["metrics"])


def test_du_output_is_identical_after_reload(earth_global_release, tmp_path):
    from services.earth_training_artifact import save_earth_artifact_atomic

    prepared = _payload(earth_global_release)
    target = tmp_path / "task_11_earth.pth"
    save_earth_artifact_atomic(prepared["payload"], target)
    checkpoint = load_earth_training_artifact(target, expected_binding=prepared["binding"], expected_task_id=11)
    model, _warnings = build_earth_model_from_checkpoint(checkpoint)
    inputs = torch.randn(3, 7, 2, 36, 72)
    mean = checkpoint.normalization["mean"][0]
    scale = checkpoint.normalization["scale"][0]
    original = prepared["model"]
    original.eval()
    model.eval()
    with torch.no_grad():
        before = earth_forward(original, inputs)[:, :, 0].numpy() * scale + mean
        after = earth_forward(model, inputs)[:, :, 0].numpy() * scale + mean
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-4)
    assert np.isfinite(after).all()


def test_reload_uses_the_saved_normalization_not_current_defaults(earth_global_release, tmp_path):
    from services.earth_training_artifact import save_earth_artifact_atomic

    prepared = _payload(earth_global_release, order=("TO3", "SWGDN"))
    target = tmp_path / "task_11_earth.pth"
    save_earth_artifact_atomic(prepared["payload"], target)

    # A different run over the same release with a different selection produces a
    # different block; the loaded checkpoint must keep its own.
    other = _payload(earth_global_release, order=("TO3", "U10M"), task_id=12)
    assert other["normalization"]["channel_order"] == ["TO3", "U10M"]
    checkpoint = load_earth_training_artifact(target, expected_task_id=11)
    assert checkpoint.normalization["channel_order"] == ["TO3", "SWGDN"]
    assert checkpoint.normalization["mean"] == prepared["normalization"]["mean"]


def test_reference_output_is_reproducible_from_the_checkpoint_alone(earth_global_release, tmp_path):
    from services.earth_training_artifact import save_earth_artifact_atomic

    prepared = _payload(earth_global_release)
    target = tmp_path / "task_11_earth.pth"
    save_earth_artifact_atomic(prepared["payload"], target)
    checkpoint = load_earth_training_artifact(target)
    rebuilt, _warnings = build_earth_model_from_checkpoint(checkpoint)
    rebuilt.eval()
    rebuilt_again, _warnings = build_earth_model_from_checkpoint(checkpoint)
    rebuilt_again.eval()
    inputs = torch.randn(1, 7, 2, 36, 72)
    with torch.no_grad():
        assert torch.equal(earth_forward(rebuilt, inputs), earth_forward(rebuilt_again, inputs))


# ── failure modes ──────────────────────────────────────────────────────────

def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(EarthArtifactError) as error:
        load_earth_training_artifact(tmp_path / "nope.pth")
    assert "missing" in str(error.value).lower()


def test_non_empty_garbage_is_not_a_valid_artifact(tmp_path):
    target = tmp_path / "garbage.pth"
    torch.save({"model_state_dict": {"a": torch.zeros(2)}}, target)
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(target)


def test_bare_mars_state_dict_is_rejected(earth_global_release, tmp_path):
    target = tmp_path / "mars.pth"
    torch.save(state_dict_to_cpu(create_earth_forecaster(["TO3"])), target)
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(target)


def test_wrong_task_identity_is_rejected(earth_global_release, tmp_path):
    from services.earth_training_artifact import save_earth_artifact_atomic

    prepared = _payload(earth_global_release, task_id=11)
    target = tmp_path / "task_11.pth"
    save_earth_artifact_atomic(prepared["payload"], target)
    with pytest.raises(EarthArtifactError) as error:
        load_earth_training_artifact(target, expected_task_id=99)
    assert "different task" in str(error.value)


def test_changed_dataset_fingerprint_reports_the_version_error(earth_global_release, tmp_path):
    from services.earth_training_artifact import save_earth_artifact_atomic

    prepared = _payload(earth_global_release)
    target = tmp_path / "task_11.pth"
    save_earth_artifact_atomic(prepared["payload"], target)
    stale = dict(prepared["binding"])
    stale["dataset_fingerprint"] = "0" * 64
    with pytest.raises(EarthArtifactError) as error:
        load_earth_training_artifact(target, expected_binding=stale)
    assert error.value.code == "dataset_version_changed"


def test_wrong_channel_order_is_rejected(earth_global_release, tmp_path):
    from services.earth_training_artifact import save_earth_artifact_atomic

    prepared = _payload(earth_global_release)
    target = tmp_path / "task_11.pth"
    save_earth_artifact_atomic(prepared["payload"], target)
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(
            target, expected_hyperparameters={"selected_channels": ["SWGDN"]}
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"artifact_schema": "something_else"}),
        lambda payload: payload["model_config"].update({"architecture": "simvp"}),
        lambda payload: payload["model_config"].update({"window": 3}),
        lambda payload: payload["model_config"].update({"height": 31}),
        lambda payload: payload["model_config"].update({"use_sphere": True}),
        lambda payload: payload["model_config"].update({"input_channel_order": ["TO3", "T2M"]}),
        lambda payload: payload["training_contract"].update({"target": "T2M"}),
        lambda payload: payload["training_contract"].update({"target_unit": "um-atm"}),
        lambda payload: payload["training_contract"].update({"strict_split_windows": False}),
        lambda payload: payload["training_contract"].update({"split_window_counts": {}}),
        lambda payload: payload["normalization"].update({"fit_split": "validation"}),
        lambda payload: payload["normalization"].update({"scale": [0.0, 1.0]}),
        lambda payload: payload["dataset_binding"].update({"dataset_id": "openmars_mcd"}),
        lambda payload: payload["dataset_binding"]["dataset_snapshot"].update({"planet": "mars"}),
        lambda payload: payload["run"].update({"run_complete": False}),
        lambda payload: payload.pop("run"),
        lambda payload: payload["metrics"].update({"unit": "ppm"}),
        lambda payload: payload["metrics"]["splits"].pop("test"),
        lambda payload: payload["metrics"]["splits"]["test"].update({"window_count": 0}),
        lambda payload: payload["metrics"]["splits"]["test"]["overall"].update({"rmse": float("nan")}),
    ],
)
def test_corrupted_artifacts_are_rejected(earth_global_release, tmp_path, mutate):
    import copy as copy_module

    from services.earth_training_artifact import save_earth_artifact_atomic

    prepared = _payload(earth_global_release)
    target = tmp_path / "task_11.pth"
    save_earth_artifact_atomic(prepared["payload"], target)

    raw = torch.load(target, map_location="cpu", weights_only=True)
    mutated = copy_module.deepcopy(raw)
    mutate(mutated)
    broken = tmp_path / "broken.pth"
    torch.save(mutated, broken)
    with pytest.raises(Exception):
        load_earth_training_artifact(broken)


def test_validation_happens_before_the_final_path_is_published(earth_global_release, tmp_path):
    from services.earth_training_artifact import save_earth_artifact_atomic

    prepared = _payload(earth_global_release)
    broken = dict(prepared["payload"])
    broken["training_contract"] = dict(prepared["payload"]["training_contract"])
    broken["training_contract"]["target_unit"] = "ppm"
    target = tmp_path / "task_11.pth"
    with pytest.raises(EarthArtifactError):
        save_earth_artifact_atomic(broken, target)
    assert not target.exists()
    assert not (tmp_path / "task_11.pth.partial").exists()


def test_atomic_save_replaces_an_existing_file(earth_global_release, tmp_path):
    from services.earth_training_artifact import save_earth_artifact_atomic

    target = tmp_path / "task_11.pth"
    first = _payload(earth_global_release, order=("TO3",))
    save_earth_artifact_atomic(first["payload"], target)
    second = _payload(earth_global_release, order=("TO3", "T2M"))
    save_earth_artifact_atomic(second["payload"], target)
    checkpoint = load_earth_training_artifact(target)
    assert checkpoint.training_contract["input_channel_order"] == ["TO3", "T2M"]
    assert not (tmp_path / "task_11.pth.partial").exists()


# ── origin range and dates ─────────────────────────────────────────────────

def test_available_origin_range_needs_input_and_reference_days(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    origins = available_origin_range(release)
    # 731 days, origin included in the seven-day input and 3 days out: the
    # first legal origin is day 7 and the last is day 728.
    assert origins["start"] == "2020-01-07"
    assert origins["end"] == "2021-12-28"
    assert origins["count"] == 722
    assert origins["dates"][0] == "2020-01-07"
    assert origins["dates"][-1] == "2021-12-28"
    assert origins["window"] == 7 and origins["horizon"] == 3
    assert origins["input_offset_days"] == -6
    assert origins["target_offset_days"] == 1


def test_forecast_dates_are_the_three_following_days(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    assert forecast_dates(release, "2021-07-08") == ["2021-07-09", "2021-07-10", "2021-07-11"]
    assert forecast_dates(release, "2020-01-07") == ["2020-01-08", "2020-01-09", "2020-01-10"]


@pytest.mark.parametrize("origin", ["2020-01-01", "2020-01-06", "2021-12-29", "2021-12-31", "2019-05-05"])
def test_origins_without_input_or_reference_days_are_rejected(earth_global_release, origin):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    with pytest.raises(EarthArtifactError) as error:
        forecast_dates(release, origin)
    assert error.value.code in {
        "earth_prediction_origin_out_of_range",
        "invalid_earth_prediction_origin",
    }


def test_the_first_and_last_legal_origins_are_accepted(earth_global_release):
    """The boundary itself is legal: exactly 6 input days before and 3 reference days after."""
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    origins = available_origin_range(release)
    assert forecast_dates(release, origins["start"]) == ["2020-01-08", "2020-01-09", "2020-01-10"]
    assert forecast_dates(release, origins["end"]) == ["2021-12-29", "2021-12-30", "2021-12-31"]
    # One day earlier there are only 5 input days before the origin, so the very
    # first selectable date is exactly the published start of the range.
    assert origins["dates"][0] == "2020-01-07"
    with pytest.raises(EarthArtifactError):
        forecast_dates(release, "2020-01-05")


def test_release_binding_mismatch_is_reported(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    assert_release_matches_binding(release, registry.build_training_binding("earth_merra2_daily_v2"))
    with pytest.raises(EarthArtifactError) as error:
        assert_release_matches_binding(release, {"dataset_fingerprint": "0" * 64})
    assert error.value.code == "dataset_version_changed"
    with pytest.raises(EarthArtifactError):
        assert_release_matches_binding(release, {"dataset_id": "earth_merra2_daily_v1"})
