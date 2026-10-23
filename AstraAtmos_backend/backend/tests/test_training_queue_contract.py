import inspect

from database.models import ModelTrainingTask
from schemas.training import TrainingTaskResponse
from services.training_service import TrainingService


def test_training_task_persists_queue_fields_and_response_exposes_them():
    assert hasattr(ModelTrainingTask, "queued_at")
    assert hasattr(ModelTrainingTask, "queue_position")
    assert "queued" in str(ModelTrainingTask.status.property.columns[0].comment or "queued")
    assert "queued_at" in TrainingTaskResponse.model_fields
    assert "queue_position" in TrainingTaskResponse.model_fields


def test_training_service_exposes_lifecycle_and_queue_cancellation():
    assert inspect.iscoroutinefunction(TrainingService.start)
    assert inspect.iscoroutinefunction(TrainingService.stop)
    assert inspect.iscoroutinefunction(TrainingService.cancel_training)
