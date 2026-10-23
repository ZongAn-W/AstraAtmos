"""Upload plumbing for Earth-capable models.

Three defects found during HTTP acceptance are pinned here so they cannot come back:

1. the upload response schema silently dropped the ``earth``/``datasets`` blocks
   (Pydantic models ignore extra keys), so the training page could not see the
   Earth verdict on the upload it had just made;
2. revalidation validated the stored file under its ``.source`` name, so every
   re-check reported "must use .py suffix" and marked a good model invalid;
3. the Earth compatibility gate hit the same ``.source`` problem when its dry-run
   ran against the stored path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from routers.user_models import _normalize_validation_report
from schemas.user_models import UserModelPackageResponse
from services.user_model_service import UserModelService
from services.user_model_validator import UserModelValidator
from training_backbones.uploaded_model_earth_gate import (
    evaluate_package_earth_compatibility,
)

EARTH_MODEL_SOURCE = '''import torch
from torch import nn

MODEL_SPEC = {
    "name": "EarthUploadSmoke",
    "parameters": {"hidden_dim": {"type": "int", "default": 8, "min": 2, "max": 32}},
    "datasets": {
        "earth_merra2": {
            "grid": [[36, 72]],
            "window": [7],
            "horizon": [3],
            "auxiliary_inputs": [],
        }
    },
}

class EarthUploadSmoke(nn.Module):
    def __init__(self, in_channels, horizon, hidden_dim):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.encoder(x[:, -1]).unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)

def build_model(config):
    return EarthUploadSmoke(
        in_channels=config["in_channels"],
        horizon=config["horizon"],
        hidden_dim=config["hidden_dim"],
    )
'''


class StoredPackage:
    """A package row as the service stores it: ``.source`` suffix, uuid prefix."""

    def __init__(self, storage_path: Path, original_filename: str, source: bytes):
        self.id = "pkg-1"
        self.display_name = "EarthUploadSmoke"
        self.version = 1
        self.storage_path = str(storage_path)
        self.original_filename = original_filename
        self.content_hash = hashlib.sha256(source).hexdigest()
        self.validation_status = "valid"
        self.param_schema = "{}"
        self.description = None
        self.validation_report = "{}"


def write_stored_package(tmp_path, source=EARTH_MODEL_SOURCE) -> StoredPackage:
    """Write source the way the service does: a ``.source`` file behind a uuid."""
    stored = tmp_path / "0f8a1b2c_EarthUploadSmoke.py.source"
    stored.write_bytes(source.encode("utf-8"))
    return StoredPackage(stored, "EarthUploadSmoke.py", source.encode("utf-8"))


# ── 1. the upload response must carry the Earth verdict ────────────────────

def test_response_schema_keeps_the_earth_block():
    payload = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "output_shape": [2, 3, 1, 8, 16],
        "datasets": {"earth_merra2": {"grid": [[36, 72]]}},
        "earth": {"compatible": True, "errors": [], "output_shape": [1, 3, 1, 36, 72]},
    }
    normalized = _normalize_validation_report(json.dumps(payload))
    assert normalized["earth"]["compatible"] is True
    assert normalized["datasets"]["earth_merra2"]["grid"] == [[36, 72]]

    # The response model is what the HTTP layer serializes through, so it must be
    # able to represent the block rather than silently dropping it.
    response = UserModelPackageResponse(
        id="pkg-1", user_id=1, display_name="M", version=1,
        original_filename="M.py", content_hash="a" * 64, param_schema={},
        description=None, validation_status="valid",
        validation_report=normalized, created_at=None, updated_at=None,
    )
    serialized = json.loads(response.model_dump_json())
    assert serialized["validation_report"]["earth"]["compatible"] is True
    assert serialized["validation_report"]["output_shape"] == [2, 3, 1, 8, 16]
    assert serialized["validation_report"]["datasets"]["earth_merra2"]["grid"] == [[36, 72]]


def test_response_schema_keeps_the_historical_shape_for_mars_only_models():
    normalized = _normalize_validation_report(json.dumps({
        "ok": True, "errors": [], "warnings": [], "output_shape": [2, 3, 1, 8, 16],
    }))
    assert "earth" not in normalized
    assert "datasets" not in normalized
    response = UserModelPackageResponse(
        id="pkg-1", user_id=1, display_name="M", version=1, original_filename="M.py",
        content_hash="a" * 64, param_schema={}, description=None,
        validation_status="valid", validation_report=normalized,
        created_at=None, updated_at=None,
    )
    serialized = json.loads(response.model_dump_json())
    assert serialized["validation_report"]["earth"] is None
    assert serialized["validation_report"]["ok"] is True


def test_report_normalizer_tolerates_junk():
    assert _normalize_validation_report(None)["ok"] is False
    assert _normalize_validation_report("not json")["errors"] == []
    assert _normalize_validation_report(json.dumps({"earth": "nope"})).get("earth") is None
    assert _normalize_validation_report(json.dumps({"datasets": {}})).get("datasets") is None


# ── 2 & 3. the stored ``.source`` name must not break validation ───────────

def test_upload_filename_uses_the_original_py_name(tmp_path):
    package = write_stored_package(tmp_path)
    assert UserModelService._upload_filename(package) == "EarthUploadSmoke.py"


def test_upload_filename_falls_back_to_a_cleaned_storage_name(tmp_path):
    package = write_stored_package(tmp_path)
    package.original_filename = ""
    assert UserModelService._upload_filename(package) == "0f8a1b2c_EarthUploadSmoke.py"


def test_validating_the_stored_path_directly_would_fail(tmp_path):
    """Documents the defect: the raw stored path is not a valid model filename."""
    package = write_stored_package(tmp_path)
    result = UserModelValidator().validate_file(Path(package.storage_path))
    assert result.ok is False
    assert any(".py suffix" in error for error in result.errors)


def test_earth_gate_dry_runs_the_stored_source_under_a_py_name(tmp_path):
    package = write_stored_package(tmp_path)
    verdict = evaluate_package_earth_compatibility(package, validator=UserModelValidator())
    assert verdict.compatible is True, verdict.reasons
    assert verdict.output_shape == [1, 3, 1, 36, 72]
    assert verdict.declares_earth is True


def test_earth_gate_still_reports_a_missing_file(tmp_path):
    package = write_stored_package(tmp_path)
    Path(package.storage_path).unlink()
    verdict = evaluate_package_earth_compatibility(package, validator=UserModelValidator())
    assert verdict.compatible is False
    assert any("missing" in reason for reason in verdict.reasons)


def test_safe_filename_keeps_py_and_sanitizes(tmp_path):
    assert UserModelService._safe_filename("my model.py") == "my_model.py"
    assert UserModelService._safe_filename("model") == "model.py"
    assert UserModelService._safe_filename("../../evil.py") == "evil.py"


@pytest.mark.parametrize(
    "original,expected",
    [
        ("EarthUploadSmoke.py", "EarthUploadSmoke.py"),
        ("nested/dir/EarthUploadSmoke.py", "EarthUploadSmoke.py"),
        ("weird name!.py", "weird_name_.py"),
    ],
)
def test_upload_filename_shapes(tmp_path, original, expected):
    package = write_stored_package(tmp_path)
    package.original_filename = original
    assert UserModelService._upload_filename(package) == expected
