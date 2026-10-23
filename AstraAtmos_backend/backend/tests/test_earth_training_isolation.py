"""Earth tasks must never reach the Mars prediction, compare, PFI or transfer paths."""

import asyncio
import json

import pytest
import torch

from services.dataset_identity import (
    DatasetRequestError,
    is_earth_training_task,
)
from services.inference_service import InferenceService, require_mars_prediction_task
from services.training_weight_service import TrainingWeightService


class FakeTask:
    def __init__(self, **overrides):
        self.id = 5
        self.user_id = 1
        self.dataset_id = "earth_merra2_daily_v2"
        self.dataset_version = "v2"
        self.dataset_fingerprint = "a" * 64
        self.dataset_identity_status = "verified"
        self.dataset_snapshot = "{}"
        self.status = "completed"
        self.output_model_path = "unused.pth"
        self.hyperparameters = json.dumps({
            "training_dataset": "earth_merra2_daily_v2",
            "window": 7,
            "horizon": 3,
        })
        for key, value in overrides.items():
            setattr(self, key, value)


class FakeUser:
    id = 1
    role = "user"


# ── pure identification ────────────────────────────────────────────────────

def test_earth_task_identification_prefers_the_identity_column():
    assert is_earth_training_task(FakeTask()) is True
    assert is_earth_training_task(FakeTask(dataset_id="earth_merra2_daily_v1")) is True
    assert is_earth_training_task(FakeTask(dataset_id="openmars_mcd")) is False


def test_legacy_rows_fall_back_to_the_hyperparameters_key():
    assert is_earth_training_task(
        FakeTask(dataset_id=None, hyperparameters='{"training_dataset": "earth_merra2_daily_v2"}')
    ) is True
    assert is_earth_training_task(
        FakeTask(dataset_id=None, hyperparameters='{"training_dataset": "mcd_overview"}')
    ) is False
    # A damaged legacy row keeps the historical Mars behaviour instead of being
    # misrouted into the Earth endpoint.
    assert is_earth_training_task(FakeTask(dataset_id=None, hyperparameters="{not json")) is False
    assert is_earth_training_task(FakeTask(dataset_id=None, hyperparameters="")) is False
    assert is_earth_training_task(FakeTask(dataset_id=None, hyperparameters="[]")) is False


def test_require_mars_prediction_task_rejects_earth_with_a_stable_code():
    with pytest.raises(DatasetRequestError) as error:
        require_mars_prediction_task(FakeTask())
    assert error.value.status_code == 409
    assert error.value.code == "dataset_prediction_not_supported"
    require_mars_prediction_task(FakeTask(dataset_id="openmars_mcd"))


# ── the guard runs before any Mars work ────────────────────────────────────

def test_context_preparation_rejects_earth_before_touching_mars_data(monkeypatch):
    """No Mars loader or cache may run for an Earth task."""
    service = InferenceService()
    calls = {"prepare": 0}

    async def fake_data_env(**kwargs):
        calls["prepare"] += 1
        raise AssertionError("Mars data preparation must not run for an Earth task")

    monkeypatch.setattr(service, "_prepare_task_data_env", fake_data_env)
    monkeypatch.setattr(
        "services.inference_service.is_valid_model_weight_file", lambda path: True
    )

    class FakeSession:
        async def get(self, model, task_id):
            return FakeTask()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "services.inference_service.async_session_maker", lambda: FakeSession()
    )

    with pytest.raises(DatasetRequestError) as error:
        asyncio.run(
            service._prepare_task_prediction_context(task_id=5, current_user=FakeUser())
        )
    assert error.value.status_code == 409
    assert error.value.code == "dataset_prediction_not_supported"
    assert calls["prepare"] == 0


def test_access_control_still_wins_over_the_scene_guard(monkeypatch):
    """Another user's Earth task must fail on permission, not disclose the scene."""
    service = InferenceService()

    class OtherUser:
        id = 99
        role = "user"

    class FakeSession:
        async def get(self, model, task_id):
            return FakeTask(user_id=1)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "services.inference_service.async_session_maker", lambda: FakeSession()
    )
    with pytest.raises(PermissionError):
        asyncio.run(
            service._prepare_task_prediction_context(task_id=5, current_user=OtherUser())
        )


@pytest.mark.parametrize(
    "method_name",
    [
        "predict_task",
        "compare_task_test_set_metrics",
        "compare_task_error_distributions",
        "compare_task_permutation_importance",
    ],
)
def test_every_mars_prediction_entry_point_propagates_the_scene_rejection(
    monkeypatch, method_name
):
    """The rejection must reach the caller and must not be swallowed into a 500."""
    service = InferenceService()
    calls = {"cache": 0}

    async def reject(task_id, *args, **kwargs):
        raise DatasetRequestError(
            "dataset_prediction_not_supported",
            "Earth tasks must be predicted through the Earth historical prediction endpoint",
            status_code=409,
        )

    async def fake_cache(*args, **kwargs):
        calls["cache"] += 1
        raise AssertionError("the Mars prediction cache must not be consulted")

    monkeypatch.setattr(service, "_prepare_task_prediction_context", reject)
    monkeypatch.setattr(service.analysis_cache, "get_or_compute", fake_cache)

    method = getattr(service, method_name)
    with pytest.raises(DatasetRequestError) as error:
        if method_name == "predict_task":
            asyncio.run(method(task_id=5, ls_start=90, current_user=FakeUser()))
        else:
            # Comparison entry points require at least two distinct tasks; the
            # rejection must still happen before any of them is computed.
            asyncio.run(method([5, 6], current_user=FakeUser()))
    assert error.value.code == "dataset_prediction_not_supported"
    assert error.value.status_code == 409
    assert calls["cache"] == 0


def test_test_set_loader_rejects_earth_before_loading_mars_channels(monkeypatch):
    service = InferenceService()
    monkeypatch.setattr(
        "services.inference_service.is_valid_model_weight_file", lambda path: True
    )
    monkeypatch.setattr(
        service, "_prepare_data", lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Mars channel preparation must not run for an Earth task")
        )
    )

    class FakeSession:
        async def get(self, model, task_id):
            return FakeTask()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "services.inference_service.async_session_maker", lambda: FakeSession()
    )
    with pytest.raises(DatasetRequestError) as error:
        asyncio.run(service.get_test_results(5))
    assert error.value.code == "dataset_prediction_not_supported"


def test_compare_rejects_a_mixed_earth_and_mars_selection(monkeypatch):
    """A mixed comparison refuses the whole request instead of dropping Earth."""
    service = InferenceService()
    seen = []

    class EarthTask(FakeTask):
        pass

    class MarsTask(FakeTask):
        def __init__(self):
            super().__init__(dataset_id="openmars_mcd", id=6)

    async def resolve(task_id, *args, **kwargs):
        seen.append(task_id)
        row = EarthTask() if task_id == 5 else MarsTask()
        require_mars_prediction_task(row)
        return row, {}, {}, None

    monkeypatch.setattr(service, "_prepare_task_prediction_context", resolve)
    with pytest.raises(DatasetRequestError) as error:
        asyncio.run(service.compare_task_test_set_metrics([5, 6], current_user=FakeUser()))
    assert error.value.code == "dataset_prediction_not_supported"
    # The Earth task is examined first and the Mars task is never evaluated, so no
    # partial comparison can be produced.
    assert seen == [5]


# ── transfer learning ──────────────────────────────────────────────────────

def test_earth_task_cannot_be_a_mars_transfer_source(monkeypatch):
    from services.training_service import TrainingService

    service = TrainingService()

    class FakeSession:
        async def get(self, model, task_id):
            return FakeTask()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "services.training_service.async_session_maker", lambda: FakeSession()
    )
    hyperparameters = {
        "transfer_learning": True,
        "transfer_source_type": "task",
        "transfer_source_task_id": 5,
    }
    with pytest.raises(DatasetRequestError) as error:
        asyncio.run(
            service._resolve_transfer_source(
                user_id=1, hyperparameters=hyperparameters, training_weight_service=None
            )
        )
    assert error.value.code == "dataset_transfer_not_supported"
    assert error.value.status_code == 409


def test_earth_checkpoint_upload_is_reported_as_incompatible(tmp_path):
    artifact = tmp_path / "earth.pth"
    torch.save(
        {
            "artifact_schema": "astraatmos_earth_forecast_checkpoint_v1",
            "model_state_dict": {"a": torch.zeros(2)},
        },
        artifact,
    )
    report = TrainingWeightService._validate_weight_file(artifact)
    assert report["ok"] is False
    assert report["artifact_kind"] == "earth_forecast_checkpoint"
    assert "Earth" in report["errors"][0]

    # A plain Mars state dict keeps working.
    mars = tmp_path / "mars.pth"
    torch.save({"a": torch.zeros(2)}, mars)
    assert TrainingWeightService._validate_weight_file(mars)["ok"] is True


def test_earth_checkpoint_is_not_extracted_into_a_mars_state_dict(tmp_path):
    """Even with a plausible state dict inside, the container is refused."""
    artifact = tmp_path / "earth.pth"
    torch.save(
        {
            "artifact_schema": "astraatmos_earth_forecast_checkpoint_v1",
            "model_state_dict": {"backbone.linear.weight": torch.zeros(4, 4)},
        },
        artifact,
    )
    report = TrainingWeightService._validate_weight_file(artifact)
    assert report["ok"] is False
    assert "tensor_count" not in report
