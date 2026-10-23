import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from schemas.training import TrainingStartRequest  # noqa: E402
from schemas.user_models import (  # noqa: E402
    UserModelPackageResponse,
    UserModelValidationReport,
)


def _dump_model(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_training_start_request_defaults_to_official_model_source():
    request = TrainingStartRequest(
        model_script="demo3.py",
        model_name="baseline",
        hyperparameters={"epochs": 1},
    )

    assert request.model_source == "official"
    assert request.uploaded_model_id is None


def test_training_start_request_accepts_uploaded_model_source_and_id():
    request = TrainingStartRequest(
        model_source="uploaded",
        uploaded_model_id="4d24f680-5029-47d9-9890-a56a6247b20e",
        model_script="demo3.py",
        model_name="custom",
        hyperparameters={"custom_model_params": {"hidden_dim": 64}},
    )

    assert request.model_source == "uploaded"
    assert request.uploaded_model_id == "4d24f680-5029-47d9-9890-a56a6247b20e"


def test_user_model_package_response_serializes_validation_and_param_schema():
    validation_report = UserModelValidationReport(
        ok=True,
        errors=[],
        warnings=["hidden_dim defaults to 64"],
        output_shape=[2, 3, 1, 8, 16],
    )
    response = UserModelPackageResponse(
        id="4d24f680-5029-47d9-9890-a56a6247b20e",
        user_id=7,
        display_name="custom",
        version=1,
        original_filename="custom_model.py",
        content_hash="a" * 64,
        param_schema={"hidden_dim": {"type": "integer", "default": 64}},
        description=None,
        validation_status="valid",
        validation_report=validation_report,
        created_at="2026-06-27T00:00:00Z",
        updated_at="2026-06-27T00:00:00Z",
    )

    serialized = _dump_model(response)
    assert serialized["validation_report"]["ok"] is True
    assert serialized["param_schema"]["hidden_dim"]["default"] == 64


if __name__ == "__main__":
    test_training_start_request_defaults_to_official_model_source()
    test_training_start_request_accepts_uploaded_model_source_and_id()
    test_user_model_package_response_serializes_validation_and_param_schema()
    print("user model schema tests passed")
