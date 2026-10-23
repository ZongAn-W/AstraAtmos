"""Execution optimizations preserve spatial identity, dtype checks and model weights."""
import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.training_scripts import earth_daily as runner
from services.earth_dataset import EarthThreeHourlyWindows
from services.earth_season import solar_longitude
from training_backbones.earth_3hourly_uploaded_contract import (
    LOWRES_CONTRACT_SCHEMA, build_config, forward, prepare_model,
)
from test_earth_5deg_dataset import lowres_release


def module_from(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precomputed_seasons_are_identical_and_isolated(lowres_release, monkeypatch):
    release, _ = lowres_release
    dataset = EarthThreeHourlyWindows.from_release(release, window=2, horizon=1)
    expected = solar_longitude(dataset.dates[3:5])
    monkeypatch.setattr('services.earth_season.solar_longitude', lambda _: pytest.fail('unexpected recomputation'))
    np.testing.assert_array_equal(dataset.window_longitude(3), expected)
    np.testing.assert_array_equal(dataset.window_longitudes([3, 3])[0], expected)
    values = dataset.window_longitude(3); values[:] = -1
    np.testing.assert_array_equal(dataset.window_longitude(3), expected)
    with pytest.raises(TypeError):
        dataset.window_longitudes([1.5])


def test_loss_and_denominator_accumulate_in_fp32():
    output = torch.full((17, 17, 1, 2, 2), .5, dtype=torch.bfloat16)
    loss, denominator = runner._weighted_mse(output, torch.zeros_like(output),
                                            torch.full((2, 2), .25), 1.0)
    assert loss.dtype == denominator.dtype == torch.float32
    assert denominator.item() == 289
    assert loss.item() == .25


def test_cuda_math_state_is_restored_on_failure(monkeypatch):
    initial = runner._cuda_math_state()
    try:
        torch.set_float32_matmul_precision('medium')
        state = runner._cuda_math_state()
        def fail(*args, **kwargs):
            runner.configure_cuda_acceleration(torch.device('cuda'))
            torch.backends.cudnn.deterministic = not state[-1]
            raise ValueError('training failed')
        monkeypatch.setattr(runner, '_run_training', fail)
        with pytest.raises(ValueError, match='training failed'):
            runner.run_training({}, None, None)
        assert runner._cuda_math_state() == state
    finally:
        runner._restore_cuda_math(initial)


def test_batched_terrain_matches_single_samples_without_extra_history_copy(monkeypatch):
    terrain = SimpleNamespace(elevation=np.arange(36*72, dtype='float32').reshape(36, 72))

    class Windows:
        def __len__(self):
            return 3
        def __getitem__(self, index):
            return np.full((2, 1, 36, 72), index, dtype='float32'), np.zeros((1, 1, 36, 72), dtype='float32')
        def read_windows(self, requests):
            return [self[index] for index, _, _ in requests]

    dataset = runner._ArrayDataset(Windows(), topography_asset=terrain)
    single = dataset[1]
    from services.earth_derived_inputs import topography_kwargs
    calls = []
    def record(*args, **kwargs):
        calls.append(args[1].shape[0])
        return topography_kwargs(*args, **kwargs)
    monkeypatch.setattr('services.earth_derived_inputs.topography_kwargs', record)
    samples = dataset.__getitems__([1, 2, 1])
    assert calls == [3]
    torch.testing.assert_close(samples[0]['topography'], single['topography'], rtol=0, atol=0)
    samples[0]['topography'].zero_()
    assert samples[2]['topography'].sum() > 0 and terrain.elevation.sum() > 0
    assert int(samples[1]['inputs'][0, 0, 0, 0]) == 2


def test_prefetch_preserves_order_propagates_errors_and_bounds_memory(monkeypatch):
    dataset = torch.utils.data.TensorDataset(torch.arange(7))
    loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    prefetched = runner._PrefetchLoader(loader)
    assert len(prefetched) == len(loader)
    assert torch.equal(torch.cat([batch[0] for batch in prefetched]), torch.arange(7))
    assert torch.equal(torch.cat([batch[0] for batch in prefetched]), torch.arange(7))
    class BrokenLoader:
        dataset = None
        def __iter__(self):
            yield 1
            raise ValueError('changed source')
    iterator = iter(runner._PrefetchLoader(BrokenLoader()))
    assert next(iterator) == 1
    with pytest.raises(ValueError, match='changed source'):
        next(iterator)
    source = SimpleNamespace(lat=np.zeros(240), lon=np.zeros(480), window=56, horizon=24,
                             input_channels=list(range(5)), _normalized_map=True)
    assert runner._prefetch_loader(loader, source, torch.device('cuda')) is loader
    source.lat, source.lon, source.window, source.horizon = np.zeros(36), np.zeros(72), 20, 20
    assert isinstance(runner._prefetch_loader(loader, source, torch.device('cuda')), runner._PrefetchLoader)
    monkeypatch.setenv('ASTRAATMOS_EARTH_PREFETCH', '0')
    assert runner._prefetch_loader(loader, source, torch.device('cuda')) is loader


@pytest.mark.parametrize('dtype', [torch.float16, torch.float64, torch.int32])
def test_strict_contract_still_rejects_non_fp32_output(dtype):
    class WrongDtype(torch.nn.Module):
        def forward(self, x):
            return x[:, :1, :1].to(dtype)
    model = prepare_model(WrongDtype(), window=2, contract_schema=LOWRES_CONTRACT_SCHEMA)
    with pytest.raises(ValueError, match='output'):
        forward(model, torch.zeros(1, 2, 1, 36, 72), horizon=1)


def test_cached_model_preserves_parameters_outputs_and_gradients():
    root = Path(__file__).resolve().parents[3] / 'docs'
    original = module_from(root / 'earth-3hourly-four-positions-shared-no-after-decoder-model.py')
    optimized = module_from(root / 'earth-3hourly-5deg-shared-topography-model-template.py')
    config = build_config(['TO3'], {'convlstm_hidden_dim': 4, 'spatial_hidden_dim': 4,
        'temporal_hidden_dim': 8, 'gate_hidden_dim': 4}, window=2, horizon=1, contract_schema=LOWRES_CONTRACT_SCHEMA)
    a, b = original.build_model(config), optimized.build_model(config)
    assert a.state_dict().keys() == b.state_dict().keys()
    b.load_state_dict(a.state_dict(), strict=True)
    x, ls, terrain = torch.randn(2, 2, 1, 36, 72), torch.tensor([[10., 11.], [20., 21.]]), torch.randn(2, 1, 36, 72)
    calls = []
    handle = b.shared_gate.topography_encoder.layers.register_forward_hook(lambda *args: calls.append(1))
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        for _ in range(2):
            a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
            torch.manual_seed(11); expected = a(x, ls, terrain)
            torch.manual_seed(11); actual = b(x, ls, terrain)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward(); actual.square().mean().backward()
            for left, right in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
        assert calls == [1, 1]
        assert not any('context' in key or 'cache' in key for key in b.__dict__)
    finally:
        handle.remove(); torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_autocast_returns_fp32_and_updates_weights(monkeypatch):
    state = runner._cuda_math_state()
    class Forecast(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layer = torch.nn.Conv2d(1, 1, 1)
        def forward(self, x):
            return self.layer(x[:, -1]).unsqueeze(1)
    model = prepare_model(Forecast(), window=2, contract_schema=LOWRES_CONTRACT_SCHEMA).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    try:
        monkeypatch.setenv('ASTRAATMOS_EARTH_AMP', '1')
        acceleration = runner.configure_cuda_acceleration(torch.device('cuda'))
        scaler = runner.make_grad_scaler(acceleration)
        before = model.layer.weight.detach().clone()
        with runner.autocast_context(acceleration):
            output = forward(model, torch.randn(2, 2, 1, 36, 72, device='cuda'), horizon=1)
            loss, _ = runner._weighted_mse(output, torch.zeros_like(output))
        assert output.dtype == loss.dtype == torch.float32
        assert runner._optimizer_step(model, optimizer, loss, scaler)
        assert not torch.equal(before, model.layer.weight)
    finally:
        runner._restore_cuda_math(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_checkpointed_batch_preserves_outputs_gradients_and_dropout_rng(monkeypatch):
    module = module_from(Path(__file__).resolve().parents[3] / 'docs' /
                         'earth-3hourly-5deg-shared-topography-model-template.py')
    config = build_config(['TO3'], {'convlstm_hidden_dim': 4, 'spatial_hidden_dim': 4,
        'temporal_hidden_dim': 8, 'gate_hidden_dim': 4}, window=20, horizon=20,
        contract_schema=LOWRES_CONTRACT_SCHEMA)
    expected_model, actual_model = module.build_model(config).cuda(), module.build_model(config).cuda()
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    x = torch.randn(16, 20, 1, 36, 72, device='cuda')
    ls, terrain = torch.full((16, 20), 30., device='cuda'), torch.randn(16, 1, 36, 72, device='cuda')
    calls = []
    real_checkpoint = module.checkpoint
    def record(function, *args, **kwargs):
        calls.append(len(args[0]))
        return real_checkpoint(function, *args, **kwargs)
    monkeypatch.setattr(module, 'checkpoint', record)
    torch.manual_seed(11)
    expected = torch.cat([expected_model._forward_impl(x[i:i+6], ls[i:i+6], terrain[i:i+6])
                          for i in range(0, 16, 6)])
    torch.manual_seed(11)
    actual = actual_model(x, ls, terrain)
    assert calls == [6, 6, 4]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward(); actual.square().mean().backward()
    for left, right in zip(expected_model.parameters(), actual_model.parameters()):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
    expected_model.eval(); actual_model.eval(); calls.clear()
    with torch.no_grad():
        torch.testing.assert_close(actual_model(x, ls, terrain), expected_model._forward_impl(x, ls, terrain),
                                   rtol=1e-4, atol=1e-6)
    assert not calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_fp16_scaler_skips_overflow_and_recovers():
    model = torch.nn.Linear(1, 1, bias=False).cuda()
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = runner.make_grad_scaler({'amp': True, 'amp_dtype': 'float16'})
    before, scale = model.weight.detach().clone(), scaler.get_scale()
    loss = model.weight.square().mean() * 1e35
    assert torch.isfinite(loss)
    assert not runner._optimizer_step(model, optimizer, loss, scaler)
    assert scaler.get_scale() < scale
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=True)
    assert runner._optimizer_step(model, optimizer, model.weight.square().mean(), scaler)
    assert not torch.equal(model.weight, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_training_saves_fp32_metrics_and_strictly_reloads(lowres_release, tmp_path, monkeypatch):
    from services.dataset_registry import DatasetRegistry
    from services.earth_derived_inputs import freeze_topography_for_reference
    from services.earth_training_artifact import load_earth_training_artifact
    from services.earth_training_contract import build_earth_training_spec
    from services.earth_task_split import build_earth_task_split
    from test_earth_topography import asset
    release, root = lowres_release
    terrain_root = tmp_path / 'terrain'; terrain_root.mkdir()
    asset(terrain_root, rows=36, columns=72)
    monkeypatch.setattr('services.earth_topography.EARTH_TOPOGRAPHY_5DEG_PATH', terrain_root / 'terrain.nc')
    template = Path(__file__).resolve().parents[3] / 'docs' / 'earth-3hourly-5deg-shared-topography-model-template.py'
    module = module_from(template)
    code = template.read_text(encoding='utf-8')
    reference = {'package_id': 'accelerated-template', 'version': 1,
        'content_hash': hashlib.sha256(code.encode()).hexdigest(), 'source_text': code,
        'source_path': str(template), 'display_name': module.MODEL_SPEC['name'],
        'param_schema': module.MODEL_SPEC['parameters'],
        'custom_model_params': {**{key: value['default'] for key, value in module.MODEL_SPEC['parameters'].items()},
                                'convlstm_hidden_dim': 4, 'spatial_hidden_dim': 4,
                                'temporal_hidden_dim': 8, 'gate_hidden_dim': 4},
        'contract_schema': LOWRES_CONTRACT_SCHEMA}
    registry = DatasetRegistry(root / 'missing_daily', earth_3hourly_5deg_package_dir=root / 'lowres')
    binding = registry.build_training_binding('earth_merra2_3hourly_5deg_v1')
    ratios = {'train_ratio': .5, 'validation_ratio': .25, 'test_ratio': .25}
    split = build_earth_task_split(release.dates, 2, 1, ratios)
    spec = build_earth_training_spec(task_id=1901, dataset_binding=binding, task_split=split,
        uploaded_model=reference, topography=freeze_topography_for_reference(reference, release),
        hyperparameters={**ratios, 'model_source': 'uploaded', 'window': 2, 'horizon': 1,
            'epochs': 1, 'batch_size': 4, 'selected_channels': ['U10M', 'T2M']})
    evaluate = runner._evaluate_split
    evaluation_modes = []
    def record(*args, **kwargs):
        evaluation_modes.append(bool((kwargs.get('acceleration') or {}).get('amp')))
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(runner, '_evaluate_split', record)
    monkeypatch.setenv('ASTRAATMOS_EARTH_AMP', '1')
    state = runner._cuda_math_state()
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        output = tmp_path / 'accelerated.pth'
        runner.run_training(spec, output, registry, device='cuda', cache_root=tmp_path / 'cache')
        saved = load_earth_training_artifact(output, expected_binding=binding, expected_task_id=1901)
        assert saved.run['acceleration']['amp'] is True
        assert saved.run['saved_metrics_precision'] == 'float32'
        assert evaluation_modes == [True, False, False]
        assert runner._cuda_math_state() == state
        assert saved.run['run_complete'] is True
    finally:
        torch.set_num_threads(previous)
