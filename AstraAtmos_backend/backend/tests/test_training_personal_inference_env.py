import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def test_training_service_ignores_historical_personal_inference_data_env():
    from services.training_service import TrainingService

    class RecordingTrainingService(TrainingService):
        def __init__(self):
            self.prepare_calls = []

        async def _prepare_personal_training_env(
            self,
            user_id,
            task_id,
            data_service,
            personal_source_service,
        ):
            self.prepare_calls.append((user_id, task_id, data_service, personal_source_service))
            raise AssertionError("historical personal tasks should not rebuild personal data")

    data_service = object()
    personal_source_service = object()
    task = SimpleNamespace(
        id=42,
        user_id=7,
        hyperparameters=json.dumps({"_data_source": "personal"}),
    )
    service = RecordingTrainingService()

    env, temp_root = asyncio.run(
        service.prepare_task_inference_data_env(
            task,
            data_service=data_service,
            personal_source_service=personal_source_service,
        )
    )

    assert env == {}
    assert temp_root is None
    assert service.prepare_calls == []


def test_training_service_uses_default_inference_env_for_default_tasks():
    from services.training_service import TrainingService

    class RecordingTrainingService(TrainingService):
        def __init__(self):
            self.prepare_calls = []

        async def _prepare_personal_training_env(self, **kwargs):
            self.prepare_calls.append(kwargs)
            raise AssertionError("default tasks should not rebuild personal data")

    task = SimpleNamespace(
        id=42,
        user_id=7,
        hyperparameters=json.dumps({"_data_source": "default"}),
    )
    service = RecordingTrainingService()

    env, temp_root = asyncio.run(
        service.prepare_task_inference_data_env(
            task,
            data_service=object(),
            personal_source_service=object(),
        )
    )

    assert env == {}
    assert temp_root is None
    assert service.prepare_calls == []


def test_test_action_ignores_historical_personal_data_env():
    from routers import training as training_router

    class FakeTrainingService:
        def __init__(self):
            self.task = SimpleNamespace(
                id=42,
                user_id=7,
                status="completed",
                hyperparameters=json.dumps({"_data_source": "personal"}),
            )
            self.prepare_calls = []
            self.cleanup_calls = []

        async def get_task(self, task_id):
            assert task_id == 42
            return self.task

        async def prepare_task_inference_data_env(
            self,
            task,
            data_service,
            personal_source_service,
        ):
            self.prepare_calls.append((task, data_service, personal_source_service))
            return {}, None

        def cleanup_temp_data_root(self, temp_root):
            self.cleanup_calls.append(temp_root)

    class FakeInferenceService:
        def __init__(self):
            self.calls = []

        async def get_test_results(self, task_id, data_dirs=None):
            self.calls.append((task_id, data_dirs))
            return {"ok": True}

    original_training_service = training_router.training_service
    original_inference_service = training_router.inference_service
    fake_training_service = FakeTrainingService()
    fake_inference_service = FakeInferenceService()
    training_router.training_service = fake_training_service
    training_router.inference_service = fake_inference_service

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        data_service="data-service",
        personal_data_source_service="personal-service",
    )))
    current_user = SimpleNamespace(id=7, role="user")

    try:
        result = asyncio.run(
            training_router.perform_task_action(
                task_id=42,
                action="test",
                request=request,
                current_user=current_user,
            )
        )
    finally:
        training_router.training_service = original_training_service
        training_router.inference_service = original_inference_service

    assert result == {"status": "success", "data": {"ok": True}}
    assert fake_training_service.prepare_calls == [
        (fake_training_service.task, "data-service", "personal-service")
    ]
    assert fake_inference_service.calls == [(42, {})]
    assert fake_training_service.cleanup_calls == [None]


def test_inference_rebuild_passes_saved_architecture_params(monkeypatch, tmp_path):
    import torch
    from services import inference_service as inference_module

    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"fake")
    task = SimpleNamespace(
        id=42,
        output_model_path=str(model_path),
        hyperparameters=json.dumps({
            "window": 4,
            "horizon": 2,
            "selected_channels": ["D"],
            "model_architecture": "patchtst",
            "stlstm_hidden_dims": [4, 4, 4],
            "patch_len": 3,
            "stride": 2,
            "d_model": 96,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 192,
            "dropout": 0.2,
        }),
        metrics="{}",
        model_script="demo3.py",
    )

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, model, task_id):
            assert task_id == 42
            return task

    monkeypatch.setattr(inference_module, "async_session_maker", lambda: FakeSession())
    monkeypatch.setattr(inference_module.torch, "load", lambda *args, **kwargs: {})

    captured = {}

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            pass

        def eval(self):
            pass

        def __call__(self, xb, ls=None):
            return torch.zeros((xb.shape[0], 2, 1, xb.shape[-2], xb.shape[-1]), device=xb.device)

    def fake_build_forecaster(**kwargs):
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(inference_module, "build_forecaster", fake_build_forecaster)

    service = inference_module.InferenceService()

    def fake_prepare_data(used_mcd_vars, window, horizon, data_dirs=None):
        assert window == 4
        assert horizon == 2
        x_torch = torch.zeros((5, 4, 2, 3, 3))
        y_torch = torch.zeros((5, 2, 1, 3, 3))
        ls_torch = torch.zeros((5, 4))
        return x_torch, y_torch, ls_torch, 0.0, 1.0

    monkeypatch.setattr(service, "_prepare_data", fake_prepare_data)

    result = asyncio.run(service.get_test_results(42))

    assert result["metrics"] == {}
    assert captured["architecture"] == "patchtst"
    assert captured["architecture_params"] == {
        "patch_len": 3,
        "stride": 2,
        "d_model": 96,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 192,
        "dropout": 0.2,
    }


def test_inference_test_action_passes_ls_even_without_sphere(monkeypatch, tmp_path):
    import torch
    from services import inference_service as inference_module

    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"fake")
    task = SimpleNamespace(
        id=42,
        output_model_path=str(model_path),
        hyperparameters=json.dumps({
            "window": 3,
            "horizon": 2,
            "selected_channels": ["D"],
            "model_architecture": "convlstm_phase_gated_mst",
            "hidden_dim": 4,
            "use_sphere": False,
        }),
        metrics="{}",
        model_script="demo3.py",
    )

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, model, task_id):
            assert task_id == 42
            return task

    monkeypatch.setattr(inference_module, "async_session_maker", lambda: FakeSession())
    monkeypatch.setattr(inference_module.torch, "load", lambda *args, **kwargs: {})

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            pass

        def eval(self):
            pass

        def __call__(self, xb, ls):
            assert ls is not None
            assert ls.shape[:2] == xb.shape[:2]
            return torch.zeros((xb.shape[0], 2, 1, xb.shape[-2], xb.shape[-1]), device=xb.device)

    monkeypatch.setattr(inference_module, "build_forecaster", lambda **kwargs: FakeModel())

    service = inference_module.InferenceService()

    def fake_prepare_data(used_mcd_vars, window, horizon, data_dirs=None):
        x_torch = torch.zeros((5, 3, 2, 3, 3))
        y_torch = torch.zeros((5, 2, 1, 3, 3))
        ls_torch = torch.zeros((5, 3))
        return x_torch, y_torch, ls_torch, 0.0, 1.0

    monkeypatch.setattr(service, "_prepare_data", fake_prepare_data)

    result = asyncio.run(service.get_test_results(42))

    assert result["metrics"] == {}


if __name__ == "__main__":
    test_training_service_ignores_historical_personal_inference_data_env()
    test_training_service_uses_default_inference_env_for_default_tasks()
    test_test_action_ignores_historical_personal_data_env()
    test_inference_test_action_passes_ls_even_without_sphere()
    print("training personal inference environment tests passed")
