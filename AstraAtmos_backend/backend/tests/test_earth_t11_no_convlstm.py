"""Check the T11 recurrence ablation against its original gated execution."""
import ast
import hashlib
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
CONTROL = DOCS / 'earth-3hourly-5deg-t11-no-convlstm-model.py'
CHANNELS = ['TO3', 'U10M', 'V10M', 'T2M', 'SWGDN']
RETAINED_CLASSES = (
    'LsHarmonicEncoder', 'TopographyEncoder', 'SharedTerrainAffineGate',
    'SpatialEncoder', 'TemporalInceptionBlock', 'TemporalTranslator',
    'SpatialDecoder', 'SharedGateTranslator',
)


def module_from(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_config(order=CHANNELS, window=3, horizon=2):
    return build_config(order, window=window, horizon=horizon,
                        contract_schema=LOWRES_CONTRACT_SCHEMA)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_retained_layers_and_contract_match_frozen_t11():
    assert hashlib.sha256(T11.read_bytes()).hexdigest() == (
        '38b4e6dd45318d3d364562e2e844f6ece4e9dfe1d2776c8598e1a89d2d019c4e')
    definitions = []
    for path in (T11, CONTROL):
        definitions.append({node.name: ast.dump(node, include_attributes=False)
                            for node in ast.parse(path.read_text(encoding='utf-8')).body
                            if isinstance(node, ast.ClassDef)})
    for name in RETAINED_CLASSES:
        assert definitions[0][name] == definitions[1][name]
    assert not any('ConvLSTM' in name for name in definitions[1]
                   if name != 'EarthThreeHour5DegreeGatedSimVPNoConvLSTMT11Control')
    original, control = module_from(T11), module_from(CONTROL)
    assert control.MODEL_SPEC['datasets'] == original.MODEL_SPEC['datasets']
    assert control.MODEL_SPEC['derived_inputs'] == original.MODEL_SPEC['derived_inputs']
    assert control.MODEL_SPEC['parameters'] == {
        key: value for key, value in original.MODEL_SPEC['parameters'].items()
        if key != 'convlstm_hidden_dim'}
    config = make_config(window=20, horizon=20)
    full, model = original.build_model(config), control.build_model(config)
    assert not hasattr(model, 'convlstm')
    assert sum(p.numel() for p in model.parameters()) == 580258
    assert sum(p.numel() for p in full.parameters()) == 598754
    expected = {key: value for key, value in full.state_dict().items()
                if not key.startswith('convlstm.')}
    assert model.state_dict().keys() == expected.keys()
    for key, value in model.state_dict().items():
        if key == 'spatial_encoder.layers.0.weight':
            assert value.shape == (64, 5, 3, 3)
            assert expected[key].shape == (64, 16, 3, 3)
        else:
            assert value.shape == expected[key].shape


def test_forward_and_gradients_match_t11_with_recurrence_bypassed():
    original, control = module_from(T11), module_from(CONTROL)
    model = control.build_model(make_config()).train()
    reference = original.build_model(make_config()).train()
    # Keep T11's forward path with recurrence bypassed and observations entering the encoder.
    reference.convlstm = torch.nn.Identity()
    reference.spatial_encoder = original.SpatialEncoder(5, 64)
    reference.load_state_dict(model.state_dict(), strict=True)
    shapes = []
    hook = model.shared_gate.register_forward_pre_hook(
        lambda _, args: shapes.append(tuple(args[0].shape)))
    inputs = torch.randn(2, 3, 5, 36, 72)
    ls = torch.tensor([[20., 20.1, 20.2], [170., 170.1, 170.2]])
    terrain = torch.randn(2, 1, 36, 72) * 2000
    torch.manual_seed(11)
    expected = reference(inputs, ls, terrain)
    torch.manual_seed(11)
    actual = model(inputs, ls, terrain)
    hook.remove()
    assert shapes == [(2, 3, 5, 36, 72), (2, 3, 5, 36, 72),
                      (2, 3, 64, 18, 36), (2, 1, 128, 18, 36),
                      (2, 2, 64, 18, 36)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad,
                                   dict(reference.named_parameters())[name].grad,
                                   rtol=0, atol=0)


@pytest.mark.parametrize('order', [CHANNELS[:1], CHANNELS])
def test_eval_round_trip_and_derived_inputs_are_sample_independent(order):
    control = module_from(CONTROL)
    config = make_config(order)
    model = prepare_model(control.build_model(config), window=3,
                          contract_schema=LOWRES_CONTRACT_SCHEMA,
                          derived_inputs=control.MODEL_SPEC['derived_inputs']).eval()
    inputs = torch.randn(2, 3, len(order), 36, 72)
    ls = torch.tensor([[20., 20.1, 20.2], [170., 170.1, 170.2]])
    terrain = torch.randn(2, 1, 36, 72) * 2000
    clone = control.build_model(config).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        actual = forward(model, inputs, horizon=2, ls=ls, topography=terrain)
        singleton = model(inputs[:1], ls[:1], terrain[:1])
        torch.testing.assert_close(actual, clone(inputs, ls, terrain), rtol=0, atol=0)
        torch.testing.assert_close(actual[:1], singleton, rtol=1e-5, atol=1e-6)
        assert not torch.equal(actual, model(inputs, ls + 90, terrain))
        assert not torch.equal(actual, model(inputs, ls, terrain + 1000))
    assert actual.shape == (2, 2, 1, 36, 72) and actual.dtype == torch.float32
    with pytest.raises(ValueError, match='36x72 Earth contract'):
        control.build_model({**config, 'height': 240})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_batch32_20_to_20_cuda_autocast_checkpoint_and_update(monkeypatch):
    control = module_from(CONTROL)
    config = make_config(window=20, horizon=20)
    model = prepare_model(control.build_model(config), window=20,
                          contract_schema=LOWRES_CONTRACT_SCHEMA,
                          derived_inputs=control.MODEL_SPEC['derived_inputs']).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    batches = []
    real_checkpoint = control.checkpoint

    def record(function, *inputs, **kwargs):
        batches.append(len(inputs[0]))
        return real_checkpoint(function, *inputs, **kwargs)

    monkeypatch.setattr(control, 'checkpoint', record)
    state = runner._cuda_math_state()
    try:
        acceleration = runner.configure_cuda_acceleration(torch.device('cuda'))
        scaler = runner.make_grad_scaler(acceleration)
        before = model.spatial_encoder.layers[0].weight.detach().clone()
        with runner.autocast_context(acceleration):
            prediction = forward(model, torch.randn(32, 20, 5, 36, 72, device='cuda'),
                                 horizon=20, ls=torch.rand(32, 20, device='cuda') * 360,
                                 topography=torch.randn(32, 1, 36, 72, device='cuda') * 2000)
            loss, _ = runner._weighted_mse(prediction, torch.zeros_like(prediction))
        assert prediction.shape == (32, 20, 1, 36, 72)
        assert prediction.dtype == loss.dtype == torch.float32
        assert batches == [6, 6, 6, 6, 6, 2]
        assert runner._optimizer_step(model, optimizer, loss, scaler)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert not torch.equal(before, model.spatial_encoder.layers[0].weight)
    finally:
        runner._restore_cuda_math(state)
