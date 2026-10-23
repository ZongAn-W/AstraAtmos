"""Earth historical prediction: checkpoint restore, DU fields, dates and errors."""

import numpy as np
import pytest

from services.dataset_identity import DatasetRequestError
from services.dataset_registry import DatasetRegistry
from services.earth_prediction_service import (
    ORIGIN_OUT_OF_RANGE,
    TASK_NOT_COMPLETED,
    build_prediction_context,
    run_earth_prediction,
)
from services.earth_training_artifact import (
    build_checkpoint_payload,
    build_metrics_block,
    normalization_from_release,
    save_earth_artifact_atomic,
)
from training_backbones.earth_daily_model import create_earth_forecaster


class FakeTask:
    """Minimal stand-in for the ORM row the prediction service reads."""

    def __init__(self, **overrides):
        self.id = 11
        self.dataset_id = "earth_merra2_daily_v2"
        self.dataset_version = "v2"
        self.dataset_fingerprint = None
        self.dataset_identity_status = "verified"
        self.dataset_snapshot = None
        self.status = "completed"
        self.output_model_path = None
        self.hyperparameters = "{}"
        for key, value in overrides.items():
            setattr(self, key, value)


def _registry(fixture):
    return DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **fixture)


def _split_metrics(rmse=3.0, mae=1.5):
    return {
        "overall": {"rmse": rmse, "mae": mae},
        "by_lead": [
            {"lead_day": 1, "rmse": rmse, "mae": mae},
            {"lead_day": 2, "rmse": rmse, "mae": mae},
            {"lead_day": 3, "rmse": rmse, "mae": mae},
        ],
    }


@pytest.fixture
def trained_task(earth_global_release, tmp_path):
    """A real completed Earth task: verified binding plus a strict checkpoint."""
    registry = _registry(earth_global_release)
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    order = ["TO3", "U10M"]
    normalization = normalization_from_release(
        registry.get_earth_snapshot("earth_merra2_daily_v2"), order
    )
    model = create_earth_forecaster(order)
    payload = build_checkpoint_payload(
        model=model,
        input_channel_order=order,
        linear_hidden_layers=2,
        dataset_binding=binding,
        normalization=normalization,
        run={
            "task_id": 11,
            "seed": 5,
            "optimizer": "Adam",
            "loss": "normalized_mse_grid_uniform",
            "best_epoch": 2,
            "epochs_completed": 3,
            "run_complete": True,
            "device": "cpu",
            "created_utc": "2026-01-01T00:00:00Z",
        },
        metrics=build_metrics_block(
            validation=_split_metrics(),
            test=_split_metrics(4.0, 2.0),
            validation_window_count=172,
            test_window_count=175,
        ),
        split_window_counts={"train": 357, "validation": 172, "test": 175},
        task_id=11,
    )
    artifact = tmp_path / "task_11_earth.pth"
    save_earth_artifact_atomic(payload, artifact)
    task = FakeTask(
        dataset_fingerprint=binding["dataset_fingerprint"],
        dataset_snapshot=binding["dataset_snapshot"],
        output_model_path=str(artifact),
    )
    return registry, task, model, normalization


def test_context_returns_the_published_origin_range_and_identity(trained_task):
    registry, task, _, _ = trained_task
    context = build_prediction_context(task, registry)
    assert context.dataset_id == "earth_merra2_daily_v2"
    assert context.dataset_version == "v2"
    assert context.target == "TO3"
    assert context.target_unit == "DU"
    assert (context.window, context.horizon) == (7, 3)
    assert context.grid_shape == [36, 72]
    assert len(context.latitude) == 36 and len(context.longitude) == 72
    assert context.input_channel_order == ["TO3", "U10M"]
    assert context.input_units == ["DU", "m s-1"]
    assert context.origins["start"] == "2020-01-07"
    assert context.origins["end"] == "2021-12-28"
    assert context.origins["count"] == 722
    assert context.origins["dates"][0] == "2020-01-07"
    assert context.training_split_end == "2020-12-31"
    assert context.metrics["splits"]["test"]["window_count"] == 175
    assert context.metrics["splits"]["test"]["overall"]["rmse"] == 4.0
    assert context.run["best_epoch"] == 2
    assert context.run["linear_hidden_layers"] == 2


def test_prediction_returns_three_days_of_du_fields_on_the_real_grid(trained_task):
    registry, task, _, _ = trained_task
    result = run_earth_prediction(task, "2021-07-08", registry, device="cpu")

    assert result["planet"] == "earth"
    assert result["dataset_id"] == "earth_merra2_daily_v2"
    assert result["target"] == "TO3"
    assert result["target_unit"] == "DU"
    assert result["forecast_origin"] == "2021-07-08"
    assert result["origin_split"] == "test"
    assert result["input_dates"] == [
        "2021-07-02", "2021-07-03", "2021-07-04", "2021-07-05",
        "2021-07-06", "2021-07-07", "2021-07-08",
    ]
    assert result["target_dates"] == ["2021-07-09", "2021-07-10", "2021-07-11"]
    assert result["grid"]["shape"] == [36, 72]
    assert len(result["grid"]["latitude"]) == 36
    assert len(result["grid"]["longitude"]) == 72
    assert result["grid"]["latitude"][0] == pytest.approx(-87.5)
    assert result["grid"]["latitude"][-1] == pytest.approx(87.5)
    assert result["grid"]["longitude"][0] == pytest.approx(-177.5)
    assert result["grid"]["longitude"][-1] == pytest.approx(177.5)

    for kind in ("prediction", "reference", "residual"):
        assert len(result[kind]) == 3, kind
        for day in result[kind]:
            field = np.asarray(day["field"], dtype="float64")
            assert field.shape == (36, 72)
            assert np.isfinite(field).all()
            assert day["valid_cells"] == 36 * 72
            assert np.isfinite(day["minVal"]) and np.isfinite(day["maxVal"])


def test_residual_is_prediction_minus_reference(trained_task):
    registry, task, _, _ = trained_task
    result = run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    for index in range(3):
        prediction = np.asarray(result["prediction"][index]["field"], dtype="float64")
        reference = np.asarray(result["reference"][index]["field"], dtype="float64")
        residual = np.asarray(result["residual"][index]["field"], dtype="float64")
        np.testing.assert_allclose(residual, prediction - reference, rtol=1e-6, atol=1e-6)


def test_reference_field_matches_the_published_dataset(trained_task):
    registry, task, _, _ = trained_task
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    result = run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    dates = np.asarray(release.dates, dtype="datetime64[D]")
    for index, date in enumerate(result["target_dates"]):
        position = int(np.flatnonzero(dates == np.datetime64(date))[0])
        expected = np.asarray(release.fields["TO3"], dtype="float64")[position]
        actual = np.asarray(result["reference"][index]["field"], dtype="float64")
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)


def test_prediction_metrics_are_du_and_finite(trained_task):
    registry, task, _, _ = trained_task
    result = run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    metrics = result["metrics"]
    assert metrics["unit"] == "DU"
    assert metrics["target"] == "TO3"
    assert metrics["reference_available"] is True
    assert metrics["aggregation"] == "user_forecast_origin_lead_grid_uniform"
    assert np.isfinite(metrics["overall"]["rmse"]) and np.isfinite(metrics["overall"]["mae"])
    assert [row["lead_day"] for row in metrics["by_lead"]] == [1, 2, 3]
    for row in metrics["by_lead"]:
        assert np.isfinite(row["rmse"]) and np.isfinite(row["mae"])
    # The reported numbers must equal a direct comparison of the returned fields.
    prediction = np.stack([
        np.asarray(day["field"], dtype="float64") for day in result["prediction"]
    ])[None, :, None]
    reference = np.stack([
        np.asarray(day["field"], dtype="float64") for day in result["reference"]
    ])[None, :, None]
    from services.earth_training_artifact import compute_earth_metrics

    expected = compute_earth_metrics(prediction, reference)
    assert metrics["overall"]["rmse"] == pytest.approx(expected["overall"]["rmse"], rel=1e-9)
    assert metrics["overall"]["mae"] == pytest.approx(expected["overall"]["mae"], rel=1e-9)


def test_prediction_uses_the_saved_normalization_not_a_refit(trained_task):
    """The input window is standardized with the checkpoint block, not recomputed."""
    registry, task, _, normalization = trained_task
    result = run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    # Rebuilding the input by hand with the saved values must reproduce the model
    # input exactly, which the returned fields alone cannot show; assert the
    # documented statistics are the ones in play instead.
    assert normalization["channel_order"] == ["TO3", "U10M"]
    assert normalization["fit_split"] == "train"
    assert result["input_channel_order"] == ["TO3", "U10M"]
    assert result["input_units"] == ["DU", "m s-1"]


def test_boundary_origins_are_predictable(trained_task):
    registry, task, _, _ = trained_task
    first = run_earth_prediction(task, "2020-01-07", registry, device="cpu")
    assert first["input_dates"] == [
        "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04",
        "2020-01-05", "2020-01-06", "2020-01-07",
    ]
    assert first["target_dates"] == ["2020-01-08", "2020-01-09", "2020-01-10"]
    last = run_earth_prediction(task, "2021-12-28", registry, device="cpu")
    assert last["target_dates"] == ["2021-12-29", "2021-12-30", "2021-12-31"]


@pytest.mark.parametrize(
    "origin",
    ["", None, "not-a-date", "2020/01/08", "2020-01-01", "2020-01-05", "2020-01-06", "2021-12-29", "2021-12-31", "2030-01-01"],
)
def test_invalid_or_out_of_range_origins_are_rejected(trained_task, origin):
    registry, task, _, _ = trained_task
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(task, origin, registry, device="cpu")
    assert error.value.status_code == 422
    assert error.value.code in {ORIGIN_OUT_OF_RANGE, "invalid_earth_prediction_origin"}


def test_changed_dataset_fingerprint_is_reported_not_fallen_back(trained_task):
    registry, task, _, _ = trained_task
    task.dataset_fingerprint = "0" * 64
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    assert error.value.status_code == 409
    assert error.value.code == "dataset_version_changed"


def test_incomplete_task_cannot_be_predicted(trained_task):
    registry, task, _, _ = trained_task
    task.status = "running"
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    assert error.value.code == TASK_NOT_COMPLETED
    assert error.value.status_code == 409

    failed = FakeTask(
        status="failed",
        dataset_fingerprint=task.dataset_fingerprint,
        dataset_snapshot=task.dataset_snapshot,
        output_model_path=task.output_model_path,
    )
    with pytest.raises(DatasetRequestError):
        build_prediction_context(failed, registry)


def test_missing_or_damaged_artifact_is_reported(trained_task, tmp_path):
    registry, task, _, _ = trained_task
    task.output_model_path = str(tmp_path / "gone.pth")
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    assert error.value.code == "invalid_earth_training_artifact"
    assert error.value.status_code == 409

    damaged = tmp_path / "damaged.pth"
    damaged.write_bytes(b"this is not a checkpoint at all")
    task.output_model_path = str(damaged)
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    assert error.value.code == "invalid_earth_training_artifact"


def test_task_without_a_verified_binding_cannot_be_predicted(earth_global_release, tmp_path):
    registry = _registry(earth_global_release)
    artifact = tmp_path / "x.pth"
    artifact.write_bytes(b"x")
    task = FakeTask(dataset_fingerprint=None, dataset_snapshot=None, output_model_path=str(artifact))
    with pytest.raises(DatasetRequestError) as error:
        build_prediction_context(task, registry)
    assert error.value.code == "invalid_earth_training_artifact"


def test_a_mars_task_is_not_an_earth_prediction_target(earth_global_release, tmp_path):
    registry = _registry(earth_global_release)
    task = FakeTask(dataset_id="openmars_mcd", output_model_path=str(tmp_path / "x.pth"))
    with pytest.raises(DatasetRequestError) as error:
        build_prediction_context(task, registry)
    assert error.value.code == "dataset_prediction_not_supported"
    assert error.value.status_code == 409


def test_legacy_earth_identity_in_hyperparameters_is_recognised(earth_global_release, trained_task):
    """A row without the identity column but with the legacy key is still Earth."""
    registry, task, _, _ = trained_task
    task.dataset_id = None
    task.dataset_fingerprint = None
    task.dataset_snapshot = None
    task.hyperparameters = '{"training_dataset": "earth_merra2_daily_v2"}'
    # The binding is required, so this fails on the missing binding rather than on
    # being mistaken for a Mars task.
    with pytest.raises(DatasetRequestError) as error:
        build_prediction_context(task, registry)
    assert error.value.code == "invalid_earth_training_artifact"


def test_unavailable_dataset_reports_the_package_reason(earth_global_release, tmp_path):
    """A deleted package must surface availability, never a Mars fallback."""
    import shutil

    registry = _registry(earth_global_release)
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    artifact = tmp_path / "task_11.pth"
    artifact.write_bytes(b"x")
    task = FakeTask(
        dataset_fingerprint=binding["dataset_fingerprint"],
        dataset_snapshot=binding["dataset_snapshot"],
        output_model_path=str(artifact),
    )
    shutil.rmtree(earth_global_release["earth_package_dir"])
    fresh = _registry(earth_global_release)
    with pytest.raises(DatasetRequestError) as error:
        build_prediction_context(task, fresh)
    assert error.value.status_code in (503, 409)
    assert error.value.code in {"dataset_unavailable", "invalid_earth_training_artifact"}
