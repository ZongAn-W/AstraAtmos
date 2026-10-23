"""Earth three-hour seasonal coordinate: declaration, dispatch, freeze and reload.

The coordinate is *derived* from the UTC timestamp rather than read from the
package, so these tests cover the four things a real deployment depends on: the
declaration is exact, a declaring model is never called without the tensor, the
producer identity is frozen next to the numbers, and a model that declares
nothing keeps being called exactly as before.
"""

import ast
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.training_scripts.earth_daily import run_training
from services.earth_model_source import EarthModelBuildError, build_uploaded_earth_model
from services.earth_prediction_service import run_earth_prediction
from services.earth_season import SEASON_ALGORITHM
from services.earth_task_split import build_earth_task_split
from services.earth_training_artifact import (
    EarthArtifactError,
    build_earth_model_from_checkpoint,
    load_earth_training_artifact,
    validate_checkpoint_payload,
)
from services.earth_training_contract import build_earth_training_spec
from test_earth_3hourly_training_runner import registry_for, synthetic_training_release
from test_earth_3hourly_uploaded import TEMPLATE, package_at, spec as upload_spec
from training_backbones.earth_3hourly_uploaded_contract import (
    CONTRACT_SCHEMA,
    DATASET_ID,
    dry_run,
    forward,
    normalize_derived_inputs,
    prepare_model,
    probe_longitude,
    requires_longitude,
)
from training_backbones.uploaded_model_contract import LS_AUXILIARY_INPUT

LS_DECLARATION = {"ls": dict(LS_AUXILIARY_INPUT)}
TOPOGRAPHY_DECLARATION = {
    "topography": {"required": True, "shape": ["batch", 1, "height", "width"],
                   "dtype": "float32", "unit": "meter"}
}
ORIGIN = "2020-01-27T22:30:00Z"


# ── the declaration is exact ────────────────────────────────────────────────

def test_absent_declaration_means_no_derived_input():
    assert normalize_derived_inputs({}) == {}
    assert normalize_derived_inputs({"name": "model"}) == {}


def test_exact_seasonal_declaration_is_accepted():
    declared = {"derived_inputs": copy.deepcopy(LS_DECLARATION)}
    assert normalize_derived_inputs(declared) == LS_DECLARATION


@pytest.mark.parametrize("declaration", [TOPOGRAPHY_DECLARATION,
    {**LS_DECLARATION, **TOPOGRAPHY_DECLARATION}])
def test_exact_earth_topography_declaration_is_accepted(declaration):
    assert normalize_derived_inputs({"derived_inputs": declaration}) == declaration


@pytest.mark.parametrize("declared", [
    {"topography": {**TOPOGRAPHY_DECLARATION["topography"], "unit": "kilometer"}},
    {**LS_DECLARATION, "topography": {**TOPOGRAPHY_DECLARATION["topography"], "required": False}},
    {"ls": {**LS_AUXILIARY_INPUT, "unit": "radian"}},
    {"ls": {**LS_AUXILIARY_INPUT, "shape": ["batch", "window", 1]}},
    {"ls": {**LS_AUXILIARY_INPUT, "dtype": "float64"}},
    {"ls": {"required": True}},
    {"ls": LS_AUXILIARY_INPUT, "extra": LS_AUXILIARY_INPUT},
    "ls",
])
def test_any_other_declaration_is_refused(declared):
    with pytest.raises(ValueError):
        normalize_derived_inputs({"derived_inputs": declared})


# ── a declaring model is always handed the tensor ───────────────────────────

class _SeasonalModel(nn.Module):
    """Adds the coordinate before mixing, so the same inputs differ without it."""

    def __init__(self, window, channels, height, width, horizon):
        super().__init__()
        self.mix = nn.Conv2d(channels, 1, 1)
        self.temporal = nn.Linear(window, horizon)

    def forward(self, x, ls):
        x = x + ls[:, :, None, None, None]
        batch, time, channels, height, width = x.shape
        mixed = self.mix(x.reshape(batch * time, channels, height, width))
        mixed = mixed.reshape(batch, time, height, width).permute(0, 2, 3, 1)
        return self.temporal(mixed).permute(0, 3, 1, 2).unsqueeze(2)


def build_seasonal_model(config):
    return _SeasonalModel(config["window"], config["in_channels"],
                          config["height"], config["width"], config["horizon"])


def _prepared(derived_inputs):
    # The v1 contract fixes the spatial tile at 24x48, so the probe uses it.
    return prepare_model(build_seasonal_model(
        {"window": 8, "in_channels": 2, "height": 24, "width": 48, "horizon": 3}),
        window=8, contract_schema=CONTRACT_SCHEMA, derived_inputs=derived_inputs)


def test_declared_model_consumes_the_coordinate_and_refuses_to_run_without_it():
    model = _prepared(copy.deepcopy(LS_DECLARATION))
    assert requires_longitude(model)
    inputs = torch.zeros(2, 8, 2, 24, 48)
    first = forward(model, inputs, horizon=3, ls=probe_longitude(2, 8))
    shifted = forward(model, inputs, horizon=3, ls=probe_longitude(2, 8) + 1.0)
    assert not torch.allclose(first, shifted), "the model never received the coordinate"
    with pytest.raises(ValueError):
        forward(model, inputs, horizon=3)


class _ChannelsOnlyModel(nn.Module):
    """The same network without the coordinate, i.e. what every older model is."""

    def __init__(self, window, channels, height, width, horizon):
        super().__init__()
        self.mix = nn.Conv2d(channels, 1, 1)
        self.temporal = nn.Linear(window, horizon)

    def forward(self, x):
        batch, time, channels, height, width = x.shape
        mixed = self.mix(x.reshape(batch * time, channels, height, width))
        mixed = mixed.reshape(batch, time, height, width).permute(0, 2, 3, 1)
        return self.temporal(mixed).permute(0, 3, 1, 2).unsqueeze(2)


def test_undeclared_model_is_called_with_inputs_only():
    model = prepare_model(
        _ChannelsOnlyModel(8, 2, 24, 48, 3), window=8, contract_schema=CONTRACT_SCHEMA)
    assert not requires_longitude(model)
    inputs = torch.zeros(2, 8, 2, 24, 48)
    assert torch.equal(forward(model, inputs, horizon=3, ls=probe_longitude(2, 8)),
                       forward(model, inputs, horizon=3))


def test_admission_dry_run_supplies_the_coordinate_for_a_declaring_model():
    shape = dry_run(build_seasonal_model, ["TO3"], {}, window=8, horizon=3,
                    contract_schema=CONTRACT_SCHEMA, derived_inputs=copy.deepcopy(LS_DECLARATION))
    assert shape == [2, 3, 1, 24, 48]


def test_admission_dry_run_rejects_a_model_that_ignores_its_declaration():
    def build_channels_only(config):
        return nn.Conv2d(config["in_channels"], 1, 1)

    with pytest.raises((ValueError, TypeError)):
        dry_run(build_channels_only, ["TO3"], {}, window=8, horizon=3,
                contract_schema=CONTRACT_SCHEMA, derived_inputs=copy.deepcopy(LS_DECLARATION))


# ── end to end: upload, train, freeze, reload, predict ──────────────────────

def _with_spec(source, declaration):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "MODEL_SPEC" for target in node.targets):
            node.value = ast.parse(repr(declaration), mode="eval").body
    return ast.unparse(tree)


def declaring_source():
    """The shipped template with a two-argument forward and a matching declaration."""
    source = TEMPLATE.read_text(encoding="utf-8")
    # Anchor on the indented definition: the template's own comment also mentions
    # the single-argument form.
    definition = "\n    def forward(self, x):"
    assert source.count(definition) == 1
    assert source.count("batch, time, channels, height, width = x.shape") == 1
    source = source.replace(definition, "\n    def forward(self, x, ls):")
    source = source.replace(
        "batch, time, channels, height, width = x.shape",
        "x = x + ls[:, :, None, None, None] * 0.01\n        batch, time, channels, height, width = x.shape",
    )
    declaration = upload_spec()
    declaration["derived_inputs"] = copy.deepcopy(LS_DECLARATION)
    return _with_spec(source, declaration)


def _train(tmp_path, release, source, task_id):
    package, validation = package_at(tmp_path, source)
    assert validation.ok, validation.errors
    registry = registry_for(release)
    binding = registry.build_training_binding(DATASET_ID)
    ratios = {name + "_ratio": 1 / 3 for name in ("train", "validation", "test")}
    split = build_earth_task_split(release.dates, 56, 24, ratios)
    reference = {
        "package_id": package.id, "display_name": package.display_name, "version": 1,
        "content_hash": package.content_hash, "source_path": package.storage_path,
        "source_text": source, "param_schema": json.loads(package.param_schema or "{}"),
        "custom_model_params": {"bias": True},
    }
    spec = build_earth_training_spec(
        task_id=task_id, dataset_binding=binding, uploaded_model=reference, task_split=split,
        hyperparameters={**ratios, "epochs": 1, "batch_size": 8, "selected_channels": ["U10M"]},
    )
    path = tmp_path / "seasonal.pth"
    run_training(spec, path, registry, device="cpu", cache_root=tmp_path / "cache")
    return path, binding, split, registry, spec


def _task(task_id, binding, split, spec, path):
    return SimpleNamespace(
        id=task_id, **binding, user_id=1, status="completed", output_model_path=str(path),
        hyperparameters=json.dumps({**spec["hyperparameters"], "_earth_task_split": split}),
    )


def test_declaring_model_trains_freezes_the_algorithm_and_predicts(
    synthetic_training_release, tmp_path, monkeypatch,
):
    path, binding, split, registry, spec = _train(
        tmp_path, synthetic_training_release, declaring_source(), task_id=4801)
    checkpoint = load_earth_training_artifact(path)
    assert checkpoint.model_config["derived_inputs"] == ["ls"]
    assert checkpoint.model_config["season_algorithm"] == SEASON_ALGORITHM
    assert checkpoint.training_contract["derived_inputs"] == ["ls"]
    assert checkpoint.training_contract["season_algorithm"] == SEASON_ALGORITHM

    model, _ = build_earth_model_from_checkpoint(checkpoint)
    assert requires_longitude(model), "the rebuilt model lost its declaration"

    import services.earth_prediction_service as prediction
    monkeypatch.setattr(prediction, "_field_list", lambda values: [{"shape": list(values.shape)}])
    result = run_earth_prediction(_task(4801, binding, split, spec, path), ORIGIN, registry, device="cpu")
    assert result["horizon"] == 24 and result["origin_split"] == "test"


def test_frozen_declaration_is_validated(synthetic_training_release, tmp_path):
    path, _binding, _split, _registry, _spec = _train(
        tmp_path, synthetic_training_release, declaring_source(), task_id=4802)
    payload = copy.deepcopy(load_earth_training_artifact(path).payload)
    # The untampered payload must pass, so each mutation below is a real rejection.
    validate_checkpoint_payload(copy.deepcopy(payload))

    without_algorithm = copy.deepcopy(payload)
    without_algorithm["model_config"].pop("season_algorithm")
    unknown_input = copy.deepcopy(payload)
    unknown_input["model_config"]["derived_inputs"] = ["topography"]
    mirror_mismatch = copy.deepcopy(payload)
    mirror_mismatch["training_contract"]["derived_inputs"] = []
    stale_algorithm = copy.deepcopy(payload)
    stale_algorithm["model_config"]["season_algorithm"] = "solar_longitude_invented_v9"
    for mutated in (without_algorithm, unknown_input, mirror_mismatch, stale_algorithm):
        with pytest.raises(EarthArtifactError):
            validate_checkpoint_payload(mutated)


def test_model_without_a_declaration_keeps_its_previous_contract(
    synthetic_training_release, tmp_path, monkeypatch,
):
    source = TEMPLATE.read_text(encoding="utf-8")
    path, binding, split, registry, spec = _train(
        tmp_path, synthetic_training_release, source, task_id=4803)
    checkpoint = load_earth_training_artifact(path)
    assert "derived_inputs" not in checkpoint.model_config
    assert "season_algorithm" not in checkpoint.model_config
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    assert not requires_longitude(model)

    import services.earth_prediction_service as prediction
    monkeypatch.setattr(prediction, "_field_list", lambda values: [{"shape": list(values.shape)}])
    result = run_earth_prediction(_task(4803, binding, split, spec, path), ORIGIN, registry, device="cpu")
    assert result["horizon"] == 24


def test_mars_style_auxiliary_inputs_stay_refused_for_earth(tmp_path):
    declaration = upload_spec()
    declaration["auxiliary_inputs"] = copy.deepcopy(LS_DECLARATION)
    source = _with_spec(TEMPLATE.read_text(encoding="utf-8"), declaration)
    _package, validation = package_at(tmp_path, source)
    assert not validation.ok
    assert "derived_inputs" in " ".join(validation.errors)

    with pytest.raises(EarthModelBuildError):
        build_uploaded_earth_model(
            reference={"package_id": "p", "display_name": "d", "version": 1,
                       "content_hash": hashlib.sha256(source.encode()).hexdigest(),
                       "source_text": source, "param_schema": {}, "custom_model_params": {}},
            input_channel_order=["TO3"], window=56, horizon=24, height=240, width=480,
            dataset_id=DATASET_ID, contract_schema=CONTRACT_SCHEMA,
        )
