"""Earth terrain declarations, real per-location probes and protected dispatch."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from training_backbones.earth_3hourly_uploaded_contract import (
    CONTRACT_SCHEMA, DATASET_ID, DERIVED_INPUT_SPECS, FEED, FULL_GRID_CONTRACT_SCHEMA,
    derived_input_names, dry_run, forward, normalize_derived_inputs, prepare_model,
    probe_longitude, validate_eval_batch_independence,
)


LS = copy.deepcopy(DERIVED_INPUT_SPECS["ls"])
TERRAIN = copy.deepcopy(DERIVED_INPUT_SPECS["topography"])


class ChannelsOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.horizon = config["horizon"]

    def forward(self, x):
        return (self.weight * x[:, -1:, :1]).repeat(1, self.horizon, 1, 1, 1)


class Season(ChannelsOnly):
    def forward(self, x, ls):
        return super().forward(x) + self.weight * ls[:, -1:, None, None, None] / 360.0


class Terrain(ChannelsOnly):
    def forward(self, x, topography):
        return super().forward(x) + self.weight * topography[:, None] / 1000.0


class Both(ChannelsOnly):
    def forward(self, x, ls, topography):
        return (super().forward(x) + self.weight * topography[:, None] / 1000.0
                + self.weight * ls[:, -1:, None, None, None] / 360.0)


@pytest.fixture
def terrain_asset():
    rows, columns = np.indices((240, 480), dtype=np.float32)
    return SimpleNamespace(elevation=(rows * 11 + columns * 3 + rows * columns / 17).astype("float32"),
                           identity={"test": "nonconstant-asymmetric-terrain"})


@pytest.mark.parametrize("schema", [CONTRACT_SCHEMA, FULL_GRID_CONTRACT_SCHEMA])
@pytest.mark.parametrize("model_class,declaration", [
    (ChannelsOnly, {}), (Season, {"ls": LS}), (Terrain, {"topography": TERRAIN}),
    (Both, {"topography": TERRAIN, "ls": LS}),
])
def test_four_calls_keep_channel_count_and_propagate_terrain_gradients(schema, model_class, declaration):
    height, width = (24, 48) if schema == CONTRACT_SCHEMA else (240, 480)
    model = prepare_model(model_class({"horizon": 2}), window=3, contract_schema=schema,
                          derived_inputs=copy.deepcopy(declaration))
    assert derived_input_names(model) == [name for name in ("ls", "topography") if name in declaration]
    x = torch.ones(2, 3, 2, height, width)
    terrain = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width).repeat(2, 1, 1, 1)
    terrain.requires_grad_()
    output = forward(model, x, horizon=2, ls=probe_longitude(2, 3), topography=terrain)
    assert output.shape == (2, 2, 1, height, width) and x.shape[2] == 2
    output.mean().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad)
    if "topography" in declaration:
        assert terrain.grad is not None and torch.count_nonzero(terrain.grad) == terrain.numel()
        shifted = forward(model, x, horizon=2, ls=probe_longitude(2, 3), topography=terrain + 100)
        assert not torch.equal(output, shifted)


@pytest.mark.parametrize("field,value", [
    ("required", False), ("required", 1), ("shape", ["batch", True, "height", "width"]),
    ("shape", ["batch", "window", 1, "height", "width"]),
    ("dtype", "float64"), ("unit", "kilometer"), ("path", "arbitrary.nc"),
])
def test_exact_declaration_rejects_invalid_fields(field, value):
    declared = copy.deepcopy(TERRAIN)
    declared[field] = value
    with pytest.raises(ValueError):
        normalize_derived_inputs({"derived_inputs": {"topography": declared}})


@pytest.mark.parametrize("block", [None, "topography", {"mola": TERRAIN}, {"topography": {"required": True}}])
def test_invalid_or_unknown_declarations_are_refused(block):
    with pytest.raises(ValueError):
        normalize_derived_inputs({"derived_inputs": block})


@pytest.mark.parametrize("model_class,declaration", [
    (ChannelsOnly, {"topography": TERRAIN}), (Terrain, {"ls": LS}),
    (Both, {"topography": TERRAIN}), (Season, {"ls": LS, "topography": TERRAIN}),
])
def test_declared_arguments_must_match_explicit_forward(model_class, declaration):
    with pytest.raises(ValueError, match="forward"):
        prepare_model(model_class({"horizon": 1}), window=1, derived_inputs=declaration)


def test_missing_dtype_shape_device_and_nonfinite_terrain_fail():
    model = prepare_model(Terrain({"horizon": 1}), window=1, derived_inputs={"topography": TERRAIN})
    x = torch.zeros(2, 1, 1, 24, 48)
    invalid = [None, torch.zeros(2, 1, 24, 48, dtype=torch.float64), torch.zeros(1, 1, 24, 48),
               torch.full((2, 1, 24, 48), float("nan")), torch.empty(2, 1, 24, 48, device="meta")]
    for terrain in invalid:
        with pytest.raises(ValueError):
            forward(model, x, horizon=1, topography=terrain)


@pytest.mark.parametrize("schema,spatial", [(CONTRACT_SCHEMA, (24, 48)),
    (FULL_GRID_CONTRACT_SCHEMA, (240, 480))])
def test_model_mutation_cannot_contaminate_caller_or_next_call(schema, spatial):
    class Mutator(Terrain):
        def forward(self, x, topography):
            topography.add_(300)
            return super().forward(x, topography)

    model = prepare_model(Mutator({"horizon": 1}), window=1, contract_schema=schema,
                          derived_inputs={"topography": TERRAIN})
    x = torch.zeros(2, 1, 1, *spatial)
    base = torch.arange(spatial[0] * spatial[1], dtype=torch.float32).reshape(1, 1, *spatial)
    terrain = base.expand(2, -1, -1, -1)
    saved = base.clone()
    first = forward(model, x, horizon=1, topography=terrain)
    second = forward(model, x, horizon=1, topography=terrain)
    assert torch.equal(base, saved) and torch.equal(first, second)


@pytest.mark.parametrize("schema", [CONTRACT_SCHEMA, FULL_GRID_CONTRACT_SCHEMA])
def test_dry_run_uses_asset_in_all_eval_and_backward_variants(schema, terrain_asset):
    shapes = []
    class Recorder(Both):
        def forward(self, x, ls, topography):
            shapes.append((x.shape[0], tuple(topography.shape), bool(self.training)))
            return super().forward(x, ls, topography)

    result = dry_run(Recorder, ["TO3", "U10M"], {}, window=2, horizon=1, batch_size=2,
                     contract_schema=schema, derived_inputs={"ls": LS, "topography": TERRAIN},
                     topography=terrain_asset)
    spatial = (24, 48) if schema == CONTRACT_SCHEMA else (240, 480)
    assert result == [2, 1, 1, *spatial]
    assert any(training for _, _, training in shapes)
    assert all(shape == (batch, 1, *spatial) for batch, shape, _ in shapes)


def test_eval_neighbour_checks_preserve_recipient_terrain(terrain_asset):
    class Neighbours(Terrain):
        def forward(self, x, topography):
            return super().forward(x, topography) + topography.mean() / 1000.0

    model = prepare_model(Neighbours({"horizon": 1}), window=1, derived_inputs={"topography": TERRAIN})
    with pytest.raises(ValueError, match="sample-independent"):
        validate_eval_batch_independence(model, window=1, channels=1, horizon=1,
                                         batch_size=2, topography=terrain_asset)


def test_only_declaring_models_load_runtime_asset(monkeypatch):
    import services.earth_topography as terrain
    def missing(*args, **kwargs):
        raise ValueError("required terrain resource missing")
    monkeypatch.setattr(terrain, "load_earth_topography", missing)
    assert dry_run(ChannelsOnly, ["TO3"], {}, window=1, horizon=1) == [2, 1, 1, 24, 48]
    with pytest.raises(ValueError, match="resource missing"):
        dry_run(Terrain, ["TO3"], {}, window=1, horizon=1, derived_inputs={"topography": TERRAIN})


def test_upload_validator_uses_terrain_for_every_channel_order_and_freezes_proof(terrain_asset, monkeypatch):
    from services.user_model_validator import UserModelValidator
    import services.earth_topography as terrain
    seen = []
    def build(config):
        seen.append(config["selected_channels"])
        return Both(config)
    monkeypatch.setattr(terrain, "load_earth_topography", lambda: terrain_asset)
    feed = {**copy.deepcopy(FEED), "window": [2], "horizon": [1]}
    module = SimpleNamespace(MODEL_SPEC={"name": "Both", "parameters": {},
        "datasets": {DATASET_ID: feed}, "derived_inputs": {"topography": TERRAIN, "ls": LS}},
        build_model=build)
    result = UserModelValidator._validate_module(module, [])
    assert result.ok, result.errors
    verdict = result.earth_compatibilities[DATASET_ID]
    assert verdict["derived_inputs"] == {"ls": LS, "topography": TERRAIN}
    assert verdict["topography_identity"] == terrain_asset.identity
    assert len(seen) == 16 and len({tuple(order) for order in seen}) == 16


def test_upload_without_terrain_does_not_access_asset(monkeypatch):
    from services.user_model_validator import UserModelValidator
    import services.earth_topography as terrain
    def missing():
        raise FileNotFoundError("configured ETOPO missing")
    monkeypatch.setattr(terrain, "load_earth_topography", missing)
    feed = {**copy.deepcopy(FEED), "window": [2], "horizon": [1]}
    base = {"name": "ChannelsOnly", "parameters": {}, "datasets": {DATASET_ID: feed}}
    result = UserModelValidator._validate_module(SimpleNamespace(MODEL_SPEC=base, build_model=ChannelsOnly), [])
    assert result.ok and "derived_inputs" not in result.earth_compatibilities[DATASET_ID]
    topographic = {**base, "derived_inputs": {"topography": TERRAIN}}
    result = UserModelValidator._validate_module(SimpleNamespace(MODEL_SPEC=topographic, build_model=Terrain), [])
    assert not result.ok and "configured ETOPO missing" in " ".join(result.errors)


def test_rebuilt_source_must_keep_frozen_declaration(tmp_path):
    from services.earth_model_source import EarthModelBuildError, build_uploaded_earth_model
    from services.uploaded_model_source import hash_source_bytes
    path = Path(__file__).resolve().parents[3] / "docs" / "earth-3hourly-topography-model-template.py"
    source = path.read_text(encoding="utf-8")
    reference = {"source_text": source, "source_path": str(tmp_path / "missing.py"),
                 "content_hash": hash_source_bytes(source.encode()), "param_schema": {},
                 "custom_model_params": {}, "contract_schema": CONTRACT_SCHEMA}
    keywords = dict(reference=reference, input_channel_order=["TO3"], window=2, horizon=1,
                    height=240, width=480, dataset_id=DATASET_ID, contract_schema=CONTRACT_SCHEMA)
    rebuilt, _, _ = build_uploaded_earth_model(**keywords, derived_inputs=["topography"])
    assert derived_input_names(rebuilt) == ["topography"]
    with pytest.raises(EarthModelBuildError, match="derived inputs"):
        build_uploaded_earth_model(**keywords, derived_inputs=["ls"])


@pytest.mark.parametrize("template,schema,names", [
    ("earth-3hourly-topography-model-template.py", CONTRACT_SCHEMA, ["topography"]),
    ("earth-3hourly-fullgrid-topography-model-template.py", FULL_GRID_CONTRACT_SCHEMA, ["ls", "topography"]),
])
def test_shipped_templates_consume_terrain(template, schema, names):
    from services.earth_model_source import _load_uploaded_module
    from training_backbones.earth_3hourly_uploaded_contract import build_config
    path = Path(__file__).resolve().parents[3] / "docs" / template
    module = _load_uploaded_module(path.read_text(encoding="utf-8"), template)
    model = prepare_model(module.build_model(build_config(["TO3"], window=2, horizon=1,
                          contract_schema=schema)), window=2, contract_schema=schema,
                          derived_inputs=normalize_derived_inputs(module.MODEL_SPEC))
    assert derived_input_names(model) == names
    height, width = model._astraatmos_earth_spatial_shape
    x = torch.ones(1, 2, 1, height, width)
    terrain = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
    first = forward(model, x, horizon=1, ls=probe_longitude(1, 2), topography=terrain)
    second = forward(model, x, horizon=1, ls=probe_longitude(1, 2), topography=terrain + 700)
    assert not torch.allclose(first, second)
    first.mean().backward()
    assert model.terrain_scale.grad is not None and model.terrain_scale.grad.abs() > 0
