"""Uploaded factories keep the validated float32 contract throughout a task."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.training_scripts.earth_daily import run_training
from services.earth_model_source import build_uploaded_earth_model
from services.earth_prediction_cache import EarthPredictionCache
from services.earth_prediction_service import run_earth_prediction
from services.earth_training_artifact import (
    build_earth_model_from_checkpoint,
    load_earth_training_artifact,
)
from services.earth_training_contract import build_earth_training_spec
from services.earth_task_split import build_earth_task_split
from services.user_model_validator import UserModelValidator
from training_backbones.earth_3hourly_uploaded_contract import DATASET_ID, forward
from test_earth_3hourly_training_runner import synthetic_training_release, registry_for


TEMPLATE = Path(__file__).resolve().parents[3] / "docs" / "earth-3hourly-uploaded-model-template.py"


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def double_factory_reference(tmp_path_factory):
    source = TEMPLATE.read_text(encoding="utf-8").replace(
        "return EarthThreeHourLinear(config)",
        "return EarthThreeHourLinear(config).double()",
    )
    path = tmp_path_factory.mktemp("double_factory") / "model.py"
    path.write_text(source, encoding="utf-8")
    validation = UserModelValidator(timeout_seconds=None).validate_file(path)
    assert validation.ok, validation.errors
    assert validation.earth_compatibilities[DATASET_ID]["compatible"]
    return {
        "package_id": "double-factory-model",
        "display_name": "Double factory",
        "version": 1,
        "content_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_path": str(path),
        "source_text": source,
        "param_schema": validation.param_schema,
        "custom_model_params": {"bias": True},
    }


def test_validated_double_factory_runs_first_float32_batch(double_factory_reference):
    model, _, _ = build_uploaded_earth_model(
        reference=double_factory_reference,
        input_channel_order=["TO3", "U10M"],
        window=56,
        horizon=24,
        height=240,
        width=480,
        dataset_id=DATASET_ID,
    )
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    inputs = torch.randn(2, 56, 2, 24, 48)
    output = forward(model, inputs)
    assert output.shape == (2, 24, 1, 24, 48)
    assert output.dtype == torch.float32
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_double_factory_trains_reloads_and_predicts(
    double_factory_reference, synthetic_training_release, tmp_path, monkeypatch,
):
    import services.earth_prediction_service as prediction_service

    registry = registry_for(synthetic_training_release)
    binding = registry.build_training_binding(DATASET_ID)
    ratios = {name + "_ratio": 1 / 3 for name in ("train", "validation", "test")}
    spec = build_earth_training_spec(
        task_id=916,
        dataset_binding=binding,
        uploaded_model=double_factory_reference,
        task_split=build_earth_task_split(synthetic_training_release.dates, 56, 24, ratios),
        hyperparameters={**ratios, "epochs": 1, "batch_size": 8, "selected_channels": ["U10M"]},
    )
    output_path = tmp_path / "double_factory.pth"
    result = run_training(spec, output_path, registry, device="cpu", cache_root=tmp_path / "cache")
    task_hypers = {
        **spec["hyperparameters"],
        "_earth_task_split": spec["task_split"],
        "_uploaded_model_id": double_factory_reference["package_id"],
        "_uploaded_model_version": double_factory_reference["version"],
        "_uploaded_model_content_hash": double_factory_reference["content_hash"],
    }
    checkpoint = load_earth_training_artifact(
        output_path, expected_binding=binding, expected_hyperparameters=task_hypers, expected_task_id=916,
    )
    assert result["epochs_completed"] == 1
    assert all(value.dtype == torch.float32 for value in checkpoint.payload["model_state_dict"].values())
    reloaded, _ = build_earth_model_from_checkpoint(checkpoint)
    assert all(parameter.dtype == torch.float32 for parameter in reloaded.parameters())
    assert forward(reloaded, torch.randn(1, 56, 2, 24, 48)).dtype == torch.float32

    task = SimpleNamespace(
        id=916, **binding, user_id=1, status="completed", output_model_path=str(output_path),
        hyperparameters=json.dumps(task_hypers),
    )
    monkeypatch.setattr(prediction_service, "EARTH_PREDICTION_CACHE", EarthPredictionCache())
    prediction = run_earth_prediction(task, "2020-01-27T22:30:00Z", registry, device="cpu")
    assert prediction["model_source"] == "uploaded"
    assert prediction["origin_split"] == "test"
    fields = np.asarray([row["field"] for row in prediction["prediction"]], dtype="float32")
    assert fields.shape == (24, 240, 480)
    assert np.isfinite(fields).all()
