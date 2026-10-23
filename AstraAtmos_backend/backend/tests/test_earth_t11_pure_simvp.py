"""T11 ablation retains its SimVP layers and removes the recurrent/terrain paths."""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch

from models.training_scripts import earth_daily as runner
from training_backbones.earth_3hourly_uploaded_contract import (
    LOWRES_CONTRACT_SCHEMA, build_config, forward, prepare_model,
)

DOCS = Path(__file__).resolve().parents[3] / 'docs'
T11 = DOCS / 'earth-3hourly-5deg-shared-topography-model-template.py'
PURE = DOCS / 'earth-3hourly-5deg-t11-pure-simvp-model.py'
CHANNELS = ['TO3', 'U10M', 'V10M', 'T2M', 'SWGDN']
BACKBONE_CLASSES = ('SpatialEncoder', 'TemporalInceptionBlock', 'TemporalTranslator', 'SpatialDecoder')


def module_from(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_backbone_definitions_defaults_and_state_shapes_match_t11():
    original, pure = module_from(T11), module_from(PURE)
    for name in BACKBONE_CLASSES:
        classes = []
        for path in (T11, PURE):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            classes.append(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name))
        assert ast.dump(classes[0], include_attributes=False) == ast.dump(classes[1], include_attributes=False)
    assert pure.MODEL_SPEC['datasets'] == original.MODEL_SPEC['datasets']
    assert 'derived_inputs' not in pure.MODEL_SPEC
    assert set(pure.MODEL_SPEC['parameters']) == {
        'spatial_hidden_dim', 'temporal_hidden_dim', 'num_temporal_blocks', 'dropout'}
    for key, schema in pure.MODEL_SPEC['parameters'].items():
        assert schema == original.MODEL_SPEC['parameters'][key]
    config = build_config(CHANNELS, window=20, horizon=20, contract_schema=LOWRES_CONTRACT_SCHEMA)
    full, control = original.build_model(config), pure.build_model(config)
    assert set(dict(control.named_children())) == {'spatial_encoder', 'temporal_translator', 'spatial_decoder'}
    assert not any('convlstm' in name or 'gate' in name or 'encoder.layers.0.weight' == name
                   for name, _ in control.named_parameters())
    backbone_state = {key: value for key, value in full.state_dict().items()
                      if key.startswith(('spatial_encoder.', 'temporal_translator.', 'spatial_decoder.'))}
    assert control.state_dict().keys() == backbone_state.keys()
    for key, value in control.state_dict().items():
        if key == 'spatial_encoder.layers.0.weight':
            assert tuple(value.shape) == (64, 5, 3, 3)
            assert tuple(backbone_state[key].shape) == (64, 16, 3, 3)
        else:
            assert value.shape == backbone_state[key].shape


def test_pure_forward_and_gradients_equal_original_simvp_components():
    original, pure = module_from(T11), module_from(PURE)
    config = build_config(CHANNELS, window=20, horizon=20, contract_schema=LOWRES_CONTRACT_SCHEMA)
    model = pure.build_model(config).train()
    # Use T11's actual component implementations as an independent reference.
    components = torch.nn.ModuleDict({
        'spatial_encoder': original.SpatialEncoder(5, 64),
        'temporal_translator': original.TemporalTranslator(20, 20, 64, 128, 2, .1),
        'spatial_decoder': original.SpatialDecoder(64),
    }).train()
    components.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(2, 20, 5, 36, 72)
    torch.manual_seed(11)
    encoded = components['spatial_encoder'](x.reshape(40, 5, 36, 72))
    translated = components['temporal_translator'](encoded.reshape(2, 20 * 64, 18, 36))
    expected = components['spatial_decoder'](translated.reshape(40, 64, 18, 36), (36, 72))
    expected = expected.reshape(2, 20, 1, 36, 72)
    torch.manual_seed(11)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for name, value in model.named_parameters():
        other = dict(components.named_parameters())[name]
        assert value.grad is not None and torch.isfinite(value.grad).all()
        torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=0)


def test_control_needs_no_derived_inputs_and_rejects_wrong_contract():
    pure = module_from(PURE)
    config = build_config(CHANNELS, window=7, horizon=3, contract_schema=LOWRES_CONTRACT_SCHEMA)
    model = prepare_model(pure.build_model(config), window=7, contract_schema=LOWRES_CONTRACT_SCHEMA)
    with torch.no_grad():
        result = forward(model, torch.randn(1, 7, 5, 36, 72), horizon=3)
    assert result.shape == (1, 3, 1, 36, 72) and result.dtype == torch.float32
    with pytest.raises(ValueError, match='36x72 Earth contract'):
        pure.build_model({**config, 'height': 240})
    with pytest.raises(ValueError, match='spatial shape'):
        model(torch.randn(1, 7, 5, 24, 48))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_t11_batch32_20_to_20_cuda_autocast_checkpoint_and_update(monkeypatch):
    pure = module_from(PURE)
    config = build_config(CHANNELS, window=20, horizon=20, contract_schema=LOWRES_CONTRACT_SCHEMA)
    model = prepare_model(pure.build_model(config), window=20, contract_schema=LOWRES_CONTRACT_SCHEMA).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    batches = []
    real_checkpoint = pure.checkpoint
    def record(function, inputs, **kwargs):
        batches.append(len(inputs))
        return real_checkpoint(function, inputs, **kwargs)
    monkeypatch.setattr(pure, 'checkpoint', record)
    state = runner._cuda_math_state()
    try:
        acceleration = runner.configure_cuda_acceleration(torch.device('cuda'))
        scaler = runner.make_grad_scaler(acceleration)
        before = model.spatial_encoder.layers[0].weight.detach().clone()
        with runner.autocast_context(acceleration):
            prediction = forward(model, torch.randn(32, 20, 5, 36, 72, device='cuda'), horizon=20)
            loss, _ = runner._weighted_mse(prediction, torch.zeros_like(prediction))
        assert prediction.shape == (32, 20, 1, 36, 72)
        assert prediction.dtype == loss.dtype == torch.float32
        assert batches == [6, 6, 6, 6, 6, 2]
        assert runner._optimizer_step(model, optimizer, loss, scaler)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert not torch.equal(before, model.spatial_encoder.layers[0].weight)
    finally:
        runner._restore_cuda_math(state)
