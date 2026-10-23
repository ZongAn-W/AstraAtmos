"""Uploaded user models on the Earth feed: declaration, gate, checkpoint, predict.

Covers the contract that makes "works on Mars" insufficient for Earth:
the optional ``MODEL_SPEC.datasets`` declaration, the Earth dry-run, the pinned
uploaded-model reference inside the checkpoint, and rebuild after the original file
disappears or changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from services.dataset_registry import DatasetRegistry
from services.earth_training_artifact import (
    EarthArtifactError,
    build_checkpoint_payload,
    build_earth_model_from_checkpoint,
    build_metrics_block,
    load_earth_training_artifact,
    normalization_from_release,
    save_earth_artifact_atomic,
)
from services.earth_training_contract import (
    EARTH_UPLOADED_ARCHITECTURE,
    build_earth_training_spec,
    require_earth_training_configuration,
)
from services.dataset_identity import DatasetRequestError
from services.earth_model_source import (
    MODEL_SOURCE_OFFICIAL,
    MODEL_SOURCE_UPLOADED,
    EarthModelBuildError,
    ModelSourcePlan,
    build_earth_model_for_plan,
    earth_forward_for_model,
    uploaded_model_config,
)
from services.uploaded_model_source import (
    UploadedModelSourceError,
    build_reference,
    resolve_source_text,
    resolve_source_text_strict,
    verify_reference_source,
)
from training_backbones.uploaded_model_dataset_spec import (
    DatasetCapabilityError,
    declares_earth_feed,
    earth_incompatibility_reasons,
    normalize_dataset_declarations,
)
from training_backbones.uploaded_model_earth_gate import (
    evaluate_earth_compatibility,
    evaluate_package_earth_compatibility,
)

EARTH_MODEL_SOURCE = '''
import torch
from torch import nn

MODEL_SPEC = {
    "name": "EarthSmokeModel",
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

class EarthSmokeModel(nn.Module):
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
    return EarthSmokeModel(
        in_channels=config["in_channels"],
        horizon=config["horizon"],
        hidden_dim=config["hidden_dim"],
    )
'''

MARS_ONLY_MODEL_SOURCE = '''
import torch
from torch import nn

MODEL_SPEC = {"name": "MarsOnly", "parameters": {}}

class MarsOnly(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, -1:].repeat(1, self.horizon, 1, 1, 1)

def build_model(config):
    return MarsOnly(horizon=config["horizon"])
'''

WRONG_GRID_MODEL_SOURCE = '''
import torch
from torch import nn

MODEL_SPEC = {
    "name": "WrongGrid",
    "parameters": {},
    "datasets": {"earth_merra2": {"grid": [[31, 49]], "window": [7], "horizon": [3]}},
}

class WrongGrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x[:, -1]).unsqueeze(1).repeat(1, 3, 1, 1, 1)

def build_model(config):
    return WrongGrid()
'''

NEEDS_LS_MODEL_SOURCE = '''
import torch
from torch import nn

MODEL_SPEC = {
    "name": "NeedsLs",
    "parameters": {},
    "datasets": {
        "earth_merra2": {
            "grid": [[36, 72]],
            "window": [7],
            "horizon": [3],
            "auxiliary_inputs": ["ls"],
        }
    },
    "auxiliary_inputs": {
        "ls": {"required": True, "shape": ["batch", "window"], "dtype": "float32", "unit": "degree"}
    },
}

class NeedsLs(nn.Module):
    def forward(self, x, ls):
        return x[:, -1:]

def build_model(config):
    return NeedsLs()
'''


class FakePackage:
    """Minimal stand-in for a UserModelPackage row.

    The content hash is taken from the bytes actually written to disk: on Windows
    ``write_text`` translates newlines, so hashing the in-memory literal instead
    would make every package look tampered.
    """

    def __init__(self, path, source_bytes, *, model_id="pkg-1", name="EarthSmokeModel", version=2,
                 validation_status="valid"):
        on_disk = path.read_bytes()
        if source_bytes is not None and source_bytes != on_disk:
            raise AssertionError("fake package source_bytes must match the file on disk")
        self.id = model_id
        self.display_name = name
        self.version = version
        self.storage_path = str(path)
        self.original_filename = path.name
        self.content_hash = hashlib.sha256(on_disk).hexdigest()
        self.validation_status = validation_status

    @property
    def source_text(self) -> str:
        return Path(self.storage_path).read_text(encoding="utf-8")


def write_model(tmp_path, name, source):
    """Write a model file with exact bytes.

    ``write_text`` translates newlines on Windows, which would make the in-memory
    literal and the file on disk hash differently; writing bytes keeps
    ``sha256(source.encode("utf-8"))`` equal to the file's digest.
    """
    path = tmp_path / f"{name}.py"
    path.write_bytes(source.encode("utf-8"))
    return path


def model_digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(fixture):
    return DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **fixture)


# ── MODEL_SPEC.datasets declaration ────────────────────────────────────────

def test_missing_datasets_block_means_mars_only():
    spec = {"name": "Legacy", "parameters": {}}
    assert normalize_dataset_declarations(spec) == {}
    assert declares_earth_feed(spec) is False
    reasons = earth_incompatibility_reasons(
        spec, {}, height=36, width=72, channel_count=5, window=7, horizon=3
    )
    assert reasons and "does not declare" in reasons[0]


def test_earth_declaration_is_normalized_and_accepted():
    spec = {
        "datasets": {
            "earth_merra2": {
                "grid": [[36, 72]],
                "window": [7],
                "horizon": [3],
                "auxiliary_inputs": [],
                "target_leading_channels": 1,
            }
        }
    }
    declarations = normalize_dataset_declarations(spec)
    assert declarations["earth_merra2"]["grid"] == [[36, 72]]
    assert declares_earth_feed(spec) is True
    assert earth_incompatibility_reasons(
        spec, {}, height=36, width=72, channel_count=5, window=7, horizon=3
    ) == []


@pytest.mark.parametrize(
    "datasets",
    [
        "not-a-dict",
        {"unknown_feed": {}},
        {"earth_merra2": "not-a-dict"},
        {"earth_merra2": {"grid": [[36, 72]], "unexpected": 1}},
        {"earth_merra2": {"grid": [[36]]}},
        {"earth_merra2": {"grid": [[0, 72]]}},
        {"earth_merra2": {"window": [0]}},
        {"earth_merra2": {"auxiliary_inputs": "ls"}},
        {"earth_merra2": {"target_leading_channels": -1}},
    ],
)
def test_malformed_declarations_are_rejected(datasets):
    with pytest.raises(DatasetCapabilityError):
        normalize_dataset_declarations({"datasets": datasets})


def test_mars_only_auxiliary_inputs_disqualify_the_earth_feed():
    spec = {
        "datasets": {"earth_merra2": {"grid": [[36, 72]], "auxiliary_inputs": ["ls"]}},
        "auxiliary_inputs": {
            "ls": {"required": True, "shape": ["batch", "window"], "dtype": "float32", "unit": "degree"}
        },
    }
    reasons = earth_incompatibility_reasons(
        spec, {"ls": {}}, height=36, width=72, channel_count=5, window=7, horizon=3
    )
    assert any("no auxiliary inputs" in reason for reason in reasons)


def test_grid_window_horizon_and_channel_mismatches_are_explained():
    spec = {"datasets": {"earth_merra2": {"grid": [[31, 49]], "window": [3], "horizon": [1],
                                           "target_leading_channels": 9}}}
    reasons = earth_incompatibility_reasons(
        spec, {}, height=36, width=72, channel_count=5, window=7, horizon=3
    )
    joined = " ".join(reasons)
    assert "grid" in joined and "window" in joined and "horizon" in joined
    assert "target_leading_channels" in joined


# ── the Earth compatibility gate ───────────────────────────────────────────

def test_earth_ready_model_is_reported_compatible(tmp_path):
    package = FakePackage(write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE),
                          EARTH_MODEL_SOURCE.encode("utf-8"))
    verdict = evaluate_package_earth_compatibility(package)
    assert verdict.compatible, verdict.reasons
    assert verdict.declares_earth
    assert verdict.output_shape == [1, 3, 1, 36, 72]
    assert verdict.display_name == "EarthSmokeModel"
    assert verdict.version == 2


def test_mars_only_model_is_not_assumed_earth_compatible(tmp_path):
    """The whole point: a perfectly valid Mars model is refused for Earth."""
    from services.user_model_validator import UserModelValidator

    path = write_model(tmp_path, "MarsOnly", MARS_ONLY_MODEL_SOURCE)
    validator = UserModelValidator()
    result = validator.validate_file(path)
    assert result.ok is True, result.errors
    assert result.earth_ok is False

    package = FakePackage(path, MARS_ONLY_MODEL_SOURCE.encode("utf-8"), name="MarsOnly")
    verdict = evaluate_package_earth_compatibility(package, validator=validator)
    assert verdict.compatible is False
    assert any("does not declare" in reason for reason in verdict.reasons)


def test_declared_but_wrong_grid_is_refused(tmp_path):
    package = FakePackage(write_model(tmp_path, "WrongGrid", WRONG_GRID_MODEL_SOURCE),
                          WRONG_GRID_MODEL_SOURCE.encode("utf-8"), name="WrongGrid")
    verdict = evaluate_package_earth_compatibility(package)
    assert verdict.compatible is False
    assert any("grid" in reason for reason in verdict.reasons)


def test_declared_but_needs_ls_is_refused(tmp_path):
    package = FakePackage(write_model(tmp_path, "NeedsLs", NEEDS_LS_MODEL_SOURCE),
                          NEEDS_LS_MODEL_SOURCE.encode("utf-8"), name="NeedsLs")
    verdict = evaluate_package_earth_compatibility(package)
    assert verdict.compatible is False
    assert any("auxiliary" in reason for reason in verdict.reasons)


def test_tampered_file_is_refused_before_any_build(tmp_path):
    path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    package = FakePackage(path, EARTH_MODEL_SOURCE.encode("utf-8"))
    path.write_text(EARTH_MODEL_SOURCE + "\n# tampered\n", encoding="utf-8")
    verdict = evaluate_package_earth_compatibility(package)
    assert verdict.compatible is False
    assert any("no longer matches" in reason for reason in verdict.reasons)


def test_missing_file_is_refused(tmp_path):
    path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    package = FakePackage(path, EARTH_MODEL_SOURCE.encode("utf-8"))
    path.unlink()
    verdict = evaluate_package_earth_compatibility(package)
    assert verdict.compatible is False
    assert any("missing" in reason for reason in verdict.reasons)


def test_static_declaration_check_never_claims_compatibility_alone():
    """Without executing the model the verdict stays False with an explanation."""
    verdict = evaluate_earth_compatibility(source_text=EARTH_MODEL_SOURCE)
    assert verdict.compatible is False
    assert verdict.declares_earth is True
    assert any("tensor contract" in reason for reason in verdict.reasons)


def test_relative_import_model_is_refused(tmp_path):
    source = "from . import helper\nMODEL_SPEC = {}\n"
    verdict = evaluate_earth_compatibility(source_text=source)
    assert verdict.compatible is False
    assert any("relative import" in reason for reason in verdict.reasons)


# ── source references and hashing ──────────────────────────────────────────

def test_reference_pins_identity_and_source(tmp_path):
    path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    package = FakePackage(path, None)
    reference = build_reference(
        package=package,
        param_schema={"hidden_dim": {"type": "int", "default": 8}},
        custom_model_params={"hidden_dim": 16},
        embed_source=True,
    )
    assert reference.package_id == "pkg-1"
    assert reference.version == 2
    assert reference.content_hash == model_digest(path)
    assert reference.custom_model_params == {"hidden_dim": 16}
    stored = reference.checkpoint_reference()
    assert stored["source_text"] == package.source_text
    assert stored["source_available"] is True
    assert verify_reference_source(stored)["status"] == "available"


def test_reference_build_detects_a_drifted_file(tmp_path):
    path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    package = FakePackage(path, None)
    path.write_text(EARTH_MODEL_SOURCE + "# drift\n", encoding="utf-8")
    with pytest.raises(UploadedModelSourceError) as error:
        build_reference(package=package, param_schema={}, custom_model_params={}, embed_source=True)
    assert error.value.code == "uploaded_model_tampered"


def test_strict_resolution_refuses_missing_and_changed_files(tmp_path):
    path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    text = path.read_text(encoding="utf-8")
    reference = {"source_path": str(path), "content_hash": model_digest(path), "source_text": text}
    assert resolve_source_text_strict(reference)[0] == text

    path.unlink()
    with pytest.raises(UploadedModelSourceError) as missing:
        resolve_source_text_strict(reference)
    assert missing.value.code == "uploaded_model_missing"

    path.write_text("MODEL_SPEC = {}\n", encoding="utf-8")
    with pytest.raises(UploadedModelSourceError) as tampered:
        resolve_source_text_strict(reference)
    assert tampered.value.code == "uploaded_model_tampered"


def test_loose_resolution_falls_back_to_the_embedded_copy(tmp_path):
    path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    text = path.read_text(encoding="utf-8")
    reference = {"source_path": str(path), "content_hash": model_digest(path), "source_text": text}

    path.unlink()
    resolved, report = resolve_source_text(reference)
    assert resolved == text
    assert report["used_embedded_source"] is True

    path.write_text("MODEL_SPEC = {}\n", encoding="utf-8")
    resolved, report = resolve_source_text(reference)
    assert resolved == text
    assert report["status"] == "tampered"


def test_loose_resolution_fails_without_any_usable_source(tmp_path):
    path = tmp_path / "gone.py"
    with pytest.raises(UploadedModelSourceError) as error:
        resolve_source_text({"source_path": str(path), "content_hash": "a" * 64})
    assert error.value.code == "uploaded_model_missing"


# ── building the uploaded Earth model ──────────────────────────────────────

def test_uploaded_model_builds_on_the_earth_contract(tmp_path):
    model_path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    reference = {
        "package_id": "pkg-1",
        "display_name": "EarthSmokeModel",
        "version": 1,
        "content_hash": hashlib.sha256(EARTH_MODEL_SOURCE.encode("utf-8")).hexdigest(),
        "source_path": str(model_path),
        "source_text": EARTH_MODEL_SOURCE,
        "param_schema": {"hidden_dim": {"type": "int", "default": 8}},
        "custom_model_params": {"hidden_dim": 4},
    }
    model, config, warnings = build_earth_model_boost(reference, ["TO3", "U10M"])
    assert warnings == ()
    assert config["in_channels"] == 2
    assert config["window"] == 7 and config["horizon"] == 3
    assert config["height"] == 36 and config["width"] == 72
    assert config["selected_channels"] == ["TO3", "U10M"]
    assert config["target_channel"] == "TO3"
    assert config["hidden_dim"] == 4  # the validated custom value, not the default

    probe = torch.randn(2, 7, 2, 36, 72)
    output = earth_forward_for_model(model, probe, model_source=MODEL_SOURCE_UPLOADED)
    assert tuple(output.shape) == (2, 3, 1, 36, 72)
    assert torch.isfinite(output).all()


def build_earth_model_boost(reference, order):
    """Small helper so the build call in tests stays readable."""
    from services.earth_model_source import build_uploaded_earth_model

    return build_uploaded_earth_model(
        reference=reference,
        input_channel_order=order,
        window=7,
        horizon=3,
        height=36,
        width=72,
    )


def test_uploaded_build_uses_declared_defaults_when_schema_is_empty(tmp_path):
    """A model reading config["hidden_dim"] must not KeyError on a bare schema."""
    path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    reference = {
        "package_id": "pkg-1",
        "display_name": "EarthSmokeModel",
        "version": 1,
        "content_hash": hashlib.sha256(EARTH_MODEL_SOURCE.encode("utf-8")).hexdigest(),
        "source_path": str(path),
        "source_text": EARTH_MODEL_SOURCE,
        "param_schema": {},
        "custom_model_params": {},
    }
    model, config, _ = build_earth_model_boost(reference, ["TO3"])
    assert config["hidden_dim"] == 8  # MODEL_SPEC.parameters default
    assert isinstance(model, torch.nn.Module)


def test_uploaded_build_refuses_mars_only_auxiliary_models(tmp_path):
    ls_path = write_model(tmp_path, "NeedsLs", NEEDS_LS_MODEL_SOURCE)
    reference = {
        "package_id": "pkg-2",
        "display_name": "NeedsLs",
        "version": 1,
        "content_hash": hashlib.sha256(NEEDS_LS_MODEL_SOURCE.encode("utf-8")).hexdigest(),
        "source_path": str(ls_path),
        "source_text": NEEDS_LS_MODEL_SOURCE,
        "param_schema": {},
        "custom_model_params": {},
    }
    with pytest.raises(EarthModelBuildError) as error:
        build_earth_model_boost(reference, ["TO3"])
    assert "Ls" in str(error.value)


def test_uploaded_build_refuses_relative_imports_in_a_checkpoint_copy():
    reference = {
        "package_id": "pkg-3",
        "display_name": "Bad",
        "version": 1,
        "content_hash": "a" * 64,
        "source_text": "from . import x\nMODEL_SPEC = {'datasets': {'earth_merra2': {}}}\n",
        "param_schema": {},
        "custom_model_params": {},
    }
    with pytest.raises(EarthModelBuildError) as error:
        build_earth_model_boost(reference, ["TO3"])
    assert error.value.code == "uploaded_model_source_invalid"


def test_official_plan_still_builds_the_dlinear():
    plan = ModelSourcePlan(
        model_source=MODEL_SOURCE_OFFICIAL,
        input_channel_order=["TO3", "T2M"],
        linear_hidden_layers=2,
    )
    model, config, warnings = build_earth_model_for_plan(plan)
    assert warnings == ()
    assert config["architecture"] == "dlinear"
    output = earth_forward_for_model(model, torch.randn(2, 7, 2, 36, 72),
                                     model_source=MODEL_SOURCE_OFFICIAL)
    assert tuple(output.shape) == (2, 3, 1, 36, 72)


def test_uploaded_config_keeps_unknown_custom_values():
    config = uploaded_model_config(
        input_channel_order=["TO3"],
        window=7,
        horizon=3,
        height=36,
        width=72,
        param_schema={},
        custom_model_params={"custom_flag": "on"},
    )
    assert config["custom_flag"] == "on"
    assert config["in_channels"] == 1


def test_uploaded_forward_checks_grid_and_shape(tmp_path):
    grid_path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    reference = {
        "package_id": "pkg-1",
        "display_name": "EarthSmokeModel",
        "version": 1,
        "content_hash": hashlib.sha256(EARTH_MODEL_SOURCE.encode("utf-8")).hexdigest(),
        "source_path": str(grid_path),
        "source_text": EARTH_MODEL_SOURCE,
        "param_schema": {},
        "custom_model_params": {},
    }
    model, _, _ = build_earth_model_boost(reference, ["TO3"])
    with pytest.raises(EarthModelBuildError):
        earth_forward_for_model(model, torch.randn(2, 7, 1, 31, 49), model_source=MODEL_SOURCE_UPLOADED)
    with pytest.raises(EarthModelBuildError):
        earth_forward_for_model(model, torch.randn(2, 7, 1, 36, 72), model_source="nonsense")


# ── checkpoint reference and reload ────────────────────────────────────────

def _uploaded_payload(earth_global_release, tmp_path, *, task_id=77, order=("TO3", "U10M")):
    registry = _registry(earth_global_release)
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    model_path = write_model(tmp_path, "EarthSmokeModel", EARTH_MODEL_SOURCE)
    reference = {
        "package_id": "pkg-uploaded",
        "display_name": "EarthSmokeModel",
        "version": 3,
        "content_hash": hashlib.sha256(EARTH_MODEL_SOURCE.encode("utf-8")).hexdigest(),
        "source_path": str(model_path),
        "source_text": EARTH_MODEL_SOURCE,
        "param_schema": {"hidden_dim": {"type": "int", "default": 8}},
        "custom_model_params": {"hidden_dim": 6},
    }
    model, build_config, _ = build_earth_model_boost(reference, list(order))
    reference["build_config"] = build_config
    reference["input_channel_order"] = list(order)
    payload = build_checkpoint_payload(
        model=model,
        input_channel_order=list(order),
        linear_hidden_layers=2,
        dataset_binding=binding,
        normalization=normalization_from_release(
            registry.get_earth_snapshot("earth_merra2_daily_v2"), list(order)
        ),
        run={
            "task_id": task_id, "seed": 1, "optimizer": "Adam",
            "loss": "normalized_mse_grid_uniform", "best_epoch": 1,
            "epochs_completed": 1, "run_complete": True, "device": "cpu",
        },
        metrics=build_metrics_block(
            validation={"overall": {"rmse": 1.0, "mae": 0.5}, "by_lead": [
                {"lead_day": i, "rmse": 1.0, "mae": 0.5} for i in (1, 2, 3)]},
            test={"overall": {"rmse": 2.0, "mae": 1.0}, "by_lead": [
                {"lead_day": i, "rmse": 2.0, "mae": 1.0} for i in (1, 2, 3)]},
            validation_window_count=172, test_window_count=175,
        ),
        split_window_counts={"train": 357, "validation": 172, "test": 175},
        task_id=task_id,
        model_source=MODEL_SOURCE_UPLOADED,
        uploaded_model=reference,
    )
    return registry, binding, payload, model_path


def test_uploaded_checkpoint_round_trips_and_keeps_identity(earth_global_release, tmp_path):
    _registry_obj, binding, payload, model_path = _uploaded_payload(earth_global_release, tmp_path)
    target = tmp_path / "task_77_uploaded.pth"
    save_earth_artifact_atomic(payload, target, strict_reload=True)
    assert target.is_file()

    checkpoint = load_earth_training_artifact(target, expected_binding=binding, expected_task_id=77)
    assert checkpoint.model_source == MODEL_SOURCE_UPLOADED
    assert checkpoint.model_ref["architecture"] == EARTH_UPLOADED_ARCHITECTURE
    identity = checkpoint.model_identity()
    assert identity["uploaded_model_id"] == "pkg-uploaded"
    assert identity["uploaded_model_version"] == 3
    assert identity["uploaded_model_name"] == "EarthSmokeModel"
    assert identity["uploaded_model_source_embedded"] is True
    assert identity["uploaded_model_content_hash"] == payload["model_ref"]["uploaded_model"]["content_hash"]

    model, warnings = build_earth_model_from_checkpoint(checkpoint)
    assert warnings == ()
    probe = torch.randn(2, 7, 2, 36, 72)
    with torch.no_grad():
        first = earth_forward_for_model(model, probe, model_source=MODEL_SOURCE_UPLOADED)
        second = earth_forward_for_model(model, probe, model_source=MODEL_SOURCE_UPLOADED)
    assert torch.equal(first, second)


def test_uploaded_checkpoint_rebuilds_after_the_file_is_deleted(earth_global_release, tmp_path):
    _registry_obj, binding, payload, model_path = _uploaded_payload(earth_global_release, tmp_path, task_id=78)
    target = tmp_path / "task_78_uploaded.pth"
    save_earth_artifact_atomic(payload, target)
    model_path.unlink()

    checkpoint = load_earth_training_artifact(target, expected_binding=binding, expected_task_id=78)
    model, warnings = build_earth_model_from_checkpoint(checkpoint)
    assert isinstance(model, torch.nn.Module)
    assert warnings and "unavailable" in warnings[0]


def test_uploaded_checkpoint_rebuilds_with_a_warning_after_tampering(earth_global_release, tmp_path):
    _registry_obj, binding, payload, model_path = _uploaded_payload(earth_global_release, tmp_path, task_id=79)
    target = tmp_path / "task_79_uploaded.pth"
    save_earth_artifact_atomic(payload, target)
    model_path.write_text(EARTH_MODEL_SOURCE + "\n# tampered\n", encoding="utf-8")

    checkpoint = load_earth_training_artifact(target, expected_binding=binding, expected_task_id=79)
    model, warnings = build_earth_model_from_checkpoint(checkpoint)
    assert isinstance(model, torch.nn.Module)
    assert warnings and "no longer matches" in warnings[0]


def test_uploaded_checkpoint_without_embedded_source_is_rejected(earth_global_release, tmp_path):
    _registry_obj, binding, payload, _model_path = _uploaded_payload(earth_global_release, tmp_path, task_id=80)
    payload = dict(payload)
    payload["model_state_dict"] = dict(payload["model_state_dict"])
    payload["model_ref"] = json.loads(json.dumps(payload["model_ref"]))
    del payload["model_ref"]["uploaded_model"]["source_text"]
    with pytest.raises(EarthArtifactError) as error:
        save_earth_artifact_atomic(payload, tmp_path / "never.pth")
    assert "embed the verified model source" in str(error.value)


def test_uploaded_checkpoint_requires_package_and_hash(earth_global_release, tmp_path):
    _registry_obj, _binding, payload, _model_path = _uploaded_payload(earth_global_release, tmp_path, task_id=81)
    for field in ("package_id", "content_hash"):
        broken = dict(payload)
        broken["model_state_dict"] = dict(payload["model_state_dict"])
        broken["model_ref"] = json.loads(json.dumps(payload["model_ref"]))
        broken["model_ref"]["uploaded_model"][field] = ""
        with pytest.raises(EarthArtifactError):
            save_earth_artifact_atomic(broken, tmp_path / f"missing_{field}.pth")


def test_uploaded_checkpoint_rejects_a_channel_order_mismatch(earth_global_release, tmp_path):
    _registry_obj, _binding, payload, _model_path = _uploaded_payload(earth_global_release, tmp_path, task_id=82)
    payload["model_ref"]["uploaded_model"]["input_channel_order"] = ["TO3", "SWGDN"]
    with pytest.raises(EarthArtifactError):
        save_earth_artifact_atomic(payload, tmp_path / "mismatch.pth")


def test_official_checkpoint_without_model_ref_stays_readable(earth_global_release, tmp_path):
    """Backward compatibility: artifacts written before model_ref existed."""
    from training_backbones.earth_daily_model import create_earth_forecaster

    registry = _registry(earth_global_release)
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    order = ["TO3", "T2M"]
    payload = build_checkpoint_payload(
        model=create_earth_forecaster(order),
        input_channel_order=order,
        linear_hidden_layers=2,
        dataset_binding=binding,
        normalization=normalization_from_release(
            registry.get_earth_snapshot("earth_merra2_daily_v2"), order
        ),
        run={
            "task_id": 83, "seed": 1, "optimizer": "Adam", "loss": "normalized_mse_grid_uniform",
            "best_epoch": 1, "epochs_completed": 1, "run_complete": True, "device": "cpu",
        },
        metrics=build_metrics_block(
            validation={"overall": {"rmse": 1.0, "mae": 0.5}, "by_lead": [
                {"lead_day": i, "rmse": 1.0, "mae": 0.5} for i in (1, 2, 3)]},
            test={"overall": {"rmse": 2.0, "mae": 1.0}, "by_lead": [
                {"lead_day": i, "rmse": 2.0, "mae": 1.0} for i in (1, 2, 3)]},
            validation_window_count=172, test_window_count=175,
        ),
        split_window_counts={"train": 357, "validation": 172, "test": 175},
        task_id=83,
    )
    # Simulate an artifact from before the model_ref field existed.
    legacy = json.loads(json.dumps({k: v for k, v in payload.items() if k != "model_state_dict"}))
    legacy["model_state_dict"] = payload["model_state_dict"]
    legacy.pop("model_ref")
    target = tmp_path / "task_83_legacy.pth"
    torch.save(legacy, target)

    checkpoint = load_earth_training_artifact(target, expected_task_id=83)
    assert checkpoint.model_source == MODEL_SOURCE_OFFICIAL
    assert checkpoint.model_identity().get("uploaded_model_id") is None
    model, warnings = build_earth_model_from_checkpoint(checkpoint)
    assert warnings == ()
    output = earth_forward_for_model(model, torch.randn(1, 7, 2, 36, 72),
                                     model_source=MODEL_SOURCE_OFFICIAL)
    assert tuple(output.shape) == (1, 3, 1, 36, 72)


# ── spec and configuration contract ────────────────────────────────────────

def test_spec_marks_uploaded_source_from_the_pinned_reference():
    binding = {
        "dataset_id": "earth_merra2_daily_v2",
        "dataset_version": "v2",
        "dataset_fingerprint": "a" * 64,
        "dataset_identity_status": "verified",
        "dataset_snapshot": {"planet": "earth"},
    }
    spec = build_earth_training_spec(
        task_id=5,
        dataset_binding=binding,
        # Deliberately omits model_source: the pinned reference is authoritative.
        hyperparameters={"selected_channels": ["U10M"], "epochs": 1},
        uploaded_model={
            "package_id": "pkg-1", "display_name": "M", "version": 1,
            "content_hash": "b" * 64, "source_text": "MODEL_SPEC = {}\n",
        },
    )
    assert spec["hyperparameters"]["model_source"] == "uploaded"
    assert spec["hyperparameters"]["model_architecture"] == EARTH_UPLOADED_ARCHITECTURE
    assert spec["uploaded_model"]["package_id"] == "pkg-1"


def test_spec_drops_unknown_uploaded_reference_fields():
    binding = {
        "dataset_id": "earth_merra2_daily_v2",
        "dataset_version": "v2",
        "dataset_fingerprint": "a" * 64,
        "dataset_identity_status": "verified",
        "dataset_snapshot": {"planet": "earth"},
    }
    spec = build_earth_training_spec(
        task_id=5,
        dataset_binding=binding,
        hyperparameters={"selected_channels": []},
        uploaded_model={"package_id": "p", "content_hash": "c" * 64, "evil": "x"},
    )
    assert "evil" not in spec["uploaded_model"]


def test_configuration_requires_an_id_exactly_for_the_uploaded_source():
    ok = require_earth_training_configuration(
        model_source="uploaded", uploaded_model_id="pkg-1", hyperparameters={}
    )
    assert ok["model_source"] == "uploaded"
    assert ok["model_architecture"] == EARTH_UPLOADED_ARCHITECTURE

    with pytest.raises(DatasetRequestError) as missing:
        require_earth_training_configuration(
            model_source="uploaded", uploaded_model_id=None, hyperparameters={}
        )
    assert missing.value.status_code == 422

    with pytest.raises(DatasetRequestError) as extra:
        require_earth_training_configuration(
            model_source="official", uploaded_model_id="pkg-1", hyperparameters={}
        )
    assert extra.value.status_code == 422
