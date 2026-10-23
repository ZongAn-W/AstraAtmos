import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def import_training_service():
    from services.training_service import TrainingService
    return TrainingService


def import_training_router():
    from routers import training
    return training


class FakePackage:
    id = "4d24f680-5029-47d9-9890-a56a6247b20e"
    user_id = 3
    display_name = "MarsCustomModel"
    version = 2
    original_filename = "mars_custom_model.py"
    validation_status = "valid"
    storage_path = "D:/tmp/model.py"
    param_schema = '{"hidden_dim": {"type": "integer", "default": 32}}'


class FakeUserModelService:
    async def get_package_for_user(self, package_id, user_id):
        assert package_id == FakePackage.id
        assert user_id == FakePackage.user_id
        return FakePackage


class FakeTransferTask:
    def __init__(self, status="completed", output_model_path="D:/tmp/source.pth", user_id=3):
        self.id = 44
        self.user_id = user_id
        self.status = status
        self.output_model_path = output_model_path
        self.model_source = "official"
        self.uploaded_model_id = None
        self.uploaded_model_version = None
        self.hyperparameters = json.dumps({
            "model_source": "official",
            "model_architecture": "predrnnv2",
            "selected_channels": ["U", "D"],
            "window": 3,
            "horizon": 3,
            "use_sphere": False,
        })


class FakeTransferSession:
    def __init__(self, task):
        self.task = task

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, model, task_id):
        return self.task if task_id == 44 else None


class FakeTransferSessionMaker:
    def __init__(self, task):
        self.task = task

    def __call__(self):
        return FakeTransferSession(self.task)


class FakeScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return list(self._items)


class FakeExecuteResult:
    def __init__(self, scalar=None, items=None):
        self._scalar = scalar
        self._items = items or []

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return FakeScalarResult(self._items)


class FakeSession:
    def __init__(self):
        self.records = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def add(self, record):
        self.records[record.id] = record

    async def commit(self):
        return None

    async def refresh(self, record):
        return None

    async def execute(self, query):
        return FakeExecuteResult(items=list(self.records.values()))


class FakeSessionMaker:
    def __init__(self):
        self.session = FakeSession()

    def __call__(self):
        return self.session


class FakeTrainingWeightFile:
    user_id = object()
    deleted_at = object()
    created_at = object()
    id = object()

    def __init__(
        self,
        id,
        user_id,
        original_filename,
        storage_path,
        content_hash,
        file_size,
        status,
        validation_report,
    ):
        self.id = id
        self.user_id = user_id
        self.original_filename = original_filename
        self.storage_path = storage_path
        self.content_hash = content_hash
        self.file_size = file_size
        self.status = status
        self.validation_report = validation_report
        self.deleted_at = None


async def test_training_weight_service_stores_valid_weight_and_rejects_bad_extension(tmp_path):
    from services import training_weight_service

    class FakeTorch:
        @staticmethod
        def load(path, map_location=None, weights_only=True):
            return {"layer.weight": object()}

    fake_sessionmaker = FakeSessionMaker()
    with patch.object(training_weight_service, "torch", FakeTorch):
        service = training_weight_service.TrainingWeightService(
            storage_root=tmp_path,
            sessionmaker=fake_sessionmaker,
        )

        record = await service.create_from_upload(
            user_id=3,
            original_filename="source.pth",
            content=b"pretend-pytorch-state",
        )

        assert record.user_id == 3
        assert record.original_filename == "source.pth"
        assert Path(record.storage_path).is_file()
        assert record.file_size == len(b"pretend-pytorch-state")
        assert record.status == "ready"
        assert json.loads(record.validation_report)["ok"] is True

        try:
            await service.create_from_upload(
                user_id=3,
                original_filename="notes.txt",
                content=b"bad",
            )
        except ValueError as exc:
            assert ".pth" in str(exc)
        else:
            raise AssertionError("Expected invalid extension to be rejected")


def _transfer_request_hypers():
    return {
        "model_source": "official",
        "model_architecture": "predrnnv2",
        "selected_channels": ["U", "D"],
        "window": 3,
        "horizon": 3,
        "use_sphere": False,
        "transfer_learning": True,
        "transfer_source_type": "task",
        "transfer_source_task_id": 44,
        "transfer_load_mode": "strict",
        "freeze_mode": "none",
        "finetune_learning_rate": 0.0001,
    }


async def test_transfer_source_rejects_incomplete_source_task(monkeypatch):
    TrainingService = import_training_service()
    from services import training_service as training_module
    monkeypatch.setattr(
        training_module, "async_session_maker",
        FakeTransferSessionMaker(FakeTransferTask(status="running")),
    )
    service = TrainingService()

    try:
        await service._resolve_transfer_source(
            user_id=3,
            hyperparameters=_transfer_request_hypers(),
            training_weight_service=None,
        )
    except ValueError as exc:
        assert "completed" in str(exc)
    else:
        raise AssertionError("Expected incomplete source task to be rejected")


async def test_transfer_source_task_injects_weight_path_for_compatible_completed_task(monkeypatch, tmp_path):
    TrainingService = import_training_service()
    from services import training_service as training_module
    source_path = tmp_path / "source.pth"
    source_path.write_bytes(b"weights")
    monkeypatch.setattr(
        training_module, "async_session_maker",
        FakeTransferSessionMaker(FakeTransferTask(output_model_path=str(source_path))),
    )
    service = TrainingService()

    env = await service._resolve_transfer_source(
        user_id=3,
        hyperparameters=_transfer_request_hypers(),
        training_weight_service=None,
    )

    assert env == {"ASTRAATMOS_TRANSFER_WEIGHT_PATH": str(source_path)}


async def test_uploaded_training_contract():
    TrainingService = import_training_service()
    service = TrainingService()

    official_script, official_hypers = service._resolve_training_entrypoint(
        user_id=3,
        model_source="official",
        uploaded_model_id=None,
        hyperparameters={"epochs": 1},
        user_model_service=None,
    )

    assert official_script == "demo3.py"
    assert official_hypers["epochs"] == 1
    assert official_hypers["model_source"] == "official"

    uploaded_script, uploaded_hypers = await service._resolve_uploaded_training_entrypoint(
        user_id=3,
        uploaded_model_id=FakePackage.id,
        hyperparameters={"custom_model_params": {"hidden_dim": 16}},
        user_model_service=FakeUserModelService(),
    )

    assert uploaded_script == "__user_model_runner__"
    assert uploaded_hypers["model_source"] == "uploaded"
    assert uploaded_hypers["_uploaded_model_id"] == FakePackage.id
    assert uploaded_hypers["_uploaded_model_version"] == FakePackage.version
    assert uploaded_hypers["_uploaded_model_name"] == FakePackage.display_name
    assert uploaded_hypers["_uploaded_model_filename"] == FakePackage.original_filename
    assert uploaded_hypers["_uploaded_model_path"] == FakePackage.storage_path
    assert uploaded_hypers["_uploaded_model_param_schema"]["hidden_dim"]["default"] == 32
    assert uploaded_hypers["custom_model_params"]["hidden_dim"] == 16


def test_official_entrypoint_strips_uploaded_private_fields():
    TrainingService = import_training_service()
    service = TrainingService()

    _script, hypers = service._resolve_training_entrypoint(
        user_id=3,
        model_source="official",
        uploaded_model_id=None,
        hyperparameters={
            "epochs": 1,
            "_uploaded_model_id": FakePackage.id,
            "_uploaded_model_version": FakePackage.version,
            "_uploaded_model_path": FakePackage.storage_path,
            "_uploaded_model_param_schema": {"hidden_dim": {"default": 32}},
            "custom_model_params": {"hidden_dim": 16},
        },
        user_model_service=None,
    )

    assert hypers["model_source"] == "official"
    assert "_uploaded_model_id" not in hypers
    assert "_uploaded_model_version" not in hypers
    assert "_uploaded_model_path" not in hypers
    assert "_uploaded_model_param_schema" not in hypers
    assert "custom_model_params" not in hypers


async def test_training_route_maps_permission_error_to_403(monkeypatch):
    training = import_training_router()

    class PermissionDeniedTrainingService:
        async def start_training(self, **kwargs):
            raise PermissionError("No permission to access this uploaded model")

    request = type("Request", (), {
        "app": type("App", (), {"state": type("State", (), {})()})()
    })()
    req = type("Req", (), {
        "model_script": "demo3.py",
        "hyperparameters": {},
        "model_name": "permission-case",
        "data_source": "default",
        "model_source": "uploaded",
        "uploaded_model_id": FakePackage.id,
        "tag_ids": [],
        "dataset_id": None,
    })()
    current_user = type("User", (), {"id": 3})()
    monkeypatch.setattr(training, "training_service", PermissionDeniedTrainingService())

    with pytest.raises(training.HTTPException) as error:
        await training.start_training(req, request, current_user)
    assert error.value.status_code == 403
    assert error.value.detail == "No permission to access this uploaded model"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--asyncio-mode=auto"]))
