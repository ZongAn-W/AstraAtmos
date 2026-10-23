import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routers import predict as predict_router  # noqa: E402
from services.training_service import TrainingService  # noqa: E402


def test_training_rejects_personal_source_before_database_work():
    service = TrainingService()

    async def run():
        with pytest.raises(ValueError, match="Data Overview"):
            await service.start_training(
                user_id=7,
                model_script="demo3.py",
                hyperparameters={},
                custom_model_name="personal-blocked",
                data_source="personal",
            )

    asyncio.run(run())


def test_training_task_personal_inference_env_is_noop():
    service = TrainingService()
    task = SimpleNamespace(id=42, user_id=7, hyperparameters='{"_data_source":"personal"}')

    env, temp_root = asyncio.run(
        service.prepare_task_inference_data_env(
            task,
            data_service=object(),
            personal_source_service=object(),
        )
    )

    assert env == {}
    assert temp_root is None


def test_predict_context_rejects_personal_source():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    async def run():
        with pytest.raises(Exception) as exc:
            await predict_router._resolve_diurnal_context(request, 27, "personal", None)
        assert getattr(exc.value, "status_code", None) == 400
        assert "Data Overview" in getattr(exc.value, "detail", "")

    asyncio.run(run())
