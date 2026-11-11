"""Environment task identity survives API serialization and transfer selection."""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from schemas.training import TrainingTaskResponse
from services.training_service import TrainingService


def response(hypers, dataset_id="mcd_overview"):
    return TrainingTaskResponse(
        id=1, model_script="demo3.py", status="queued", start_time=datetime.now(), end_time=None,
        hyperparameters=json.dumps(hypers), log_file_path=None, output_model_path=None,
        custom_model_name="Environment task", metrics=None, progress=0, current_epoch=0,
        total_epochs=1, current_loss=None, eta=None, loss_history=None, dataset_id=dataset_id,
    ).model_dump()


def test_api_serializes_explicit_temperature_target_and_independent_inputs():
    result = response({"target_variable": "T", "input_variables": ["O3", "U"]})
    assert result["target_variable"] == "T"
    assert result["input_variables"] == ["O3", "U"]
    assert result["target_unit"] == "K"


def test_active_catalog_exposes_target_choices_even_when_data_is_missing(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routers.datasets import router
    from services.dataset_registry import DatasetRegistry
    app = FastAPI()
    app.state.dataset_registry = DatasetRegistry(tmp_path / "missing")
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        response = client.get("/api/datasets")
        assert response.status_code == 200
        catalog = {item["dataset_id"]: item for item in response.json()["items"]}
        assert catalog["mcd_overview"]["supported_targets"] == ["O3", "U", "V", "D", "S", "T"]
        earth = catalog["earth_merra2_3hourly_v1"]
        assert earth["supported_targets"] == ["TO3", "U10M", "V10M", "T2M", "SWGDN"]
        assert earth["training_profile"]["supported_targets"] == earth["supported_targets"]
        assert earth["availability"] == "missing"
        assert client.get("/api/datasets/earth_merra2_3hourly_v1").json() == earth


@pytest.mark.parametrize("dataset_id,ozone,auxiliary", [
    ("mcd_overview", "O3", "T"), ("earth_merra2_3hourly_v1", "TO3", "T2M"),
])
def test_legacy_tasks_keep_implicit_ozone_input(dataset_id, ozone, auxiliary):
    result = response({"selected_channels": [auxiliary]}, dataset_id)
    assert result["target_variable"] == ozone
    assert result["input_variables"] == [ozone, auxiliary]
    assert result["target_unit"] == ("DU" if dataset_id.startswith("earth_") else "um-atm")


def test_transfer_accepts_equivalent_old_and_explicit_ozone_configuration():
    service = TrainingService.__new__(TrainingService)
    task = SimpleNamespace(model_source="official")
    shared = dict(window=3, horizon=3, use_sphere=False, model_architecture="dlinear")
    service._validate_transfer_task_compatibility(
        task, {**shared, "selected_channels": ["U"]},
        {**shared, "target_variable": "O3", "input_variables": ["O3", "U"]},
    )


@pytest.mark.parametrize("replacement,reason", [
    ({"target_variable": "T"}, "target_variable"),
    ({"input_variables": ["O3", "T"]}, "input_variables"),
])
def test_transfer_rejects_a_different_target_or_input_identity(replacement, reason):
    service = TrainingService.__new__(TrainingService)
    task = SimpleNamespace(model_source="official")
    hypers = dict(target_variable="T", input_variables=["O3", "U"], window=3,
                  horizon=3, use_sphere=False, model_architecture="dlinear")
    # Source ozone must never silently become a temperature-trained model.
    source = {**hypers, "target_variable": "O3"}
    current = {**source, **replacement}
    with pytest.raises(ValueError, match=reason):
        service._validate_transfer_task_compatibility(task, source, current)


@pytest.mark.parametrize("name", ["target_variable", "input_variables", "out_channels"])
def test_uploaded_parameters_cannot_replace_environment_task_identity(name):
    from training_backbones.user_model_runner import build_uploaded_model_config
    with pytest.raises(ValueError, match="cannot override"):
        build_uploaded_model_config(
            in_channels=2, window=3, horizon=2, height=8, width=16,
            selected_channels=["U"], custom_model_params={},
            param_schema={name: {"type": "integer", "default": 2}},
            target_variable="T", input_variables=["O3", "U"],
        )
