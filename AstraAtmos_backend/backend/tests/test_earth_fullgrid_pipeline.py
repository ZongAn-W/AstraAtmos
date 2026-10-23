"""Full-grid windows preserve source values and survive actual training/reload."""

import hashlib
import json

import numpy as np
import pytest
import torch

from models.training_scripts import earth_daily as runner
from services.earth_dataset import (
    EarthThreeHourlyWindows, THREE_HOURLY_FULL_GRID_CACHE_LAYOUT,
    build_threehour_training_cache,
)
from services.earth_task_split import build_earth_task_split
from services.earth_training_artifact import load_earth_training_artifact
from services.earth_training_contract import build_earth_training_spec
from training_backbones.earth_3hourly_uploaded_contract import FULL_GRID_FEED
from test_earth_3hourly_training_data import disk_release, _normalization
from test_earth_3hourly_training_runner import synthetic_training_release, registry_for


@pytest.fixture(scope='module')
def full_grid_windows(disk_release, tmp_path_factory):
    normalization = _normalization(disk_release, ['U10M'])
    windows = EarthThreeHourlyWindows.from_release(
        disk_release, selected_channels=['U10M'], normalization=normalization,
    )
    path = build_threehour_training_cache(
        disk_release, ['TO3', 'U10M'], normalization,
        tmp_path_factory.mktemp('full_grid_cache'), full_grid=True,
    )
    windows.use_training_cache(path)
    return windows, path


def test_full_grid_cache_equals_source_without_reconstructing_tiles(full_grid_windows, monkeypatch):
    windows, path = full_grid_windows
    uncached = EarthThreeHourlyWindows.from_release(
        windows._release, selected_channels=['U10M'], normalization=windows.normalization,
    )
    expected = uncached[1]
    def no_tiles(*args, **kwargs):
        pytest.fail('A full-grid window must not reconstruct spatial tiles')
    monkeypatch.setattr(windows, '_read_tile_cache', no_tiles)
    actual = windows[1]
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)
    assert actual[0].shape == (56, 2, 240, 480)
    assert actual[1].shape == (24, 1, 240, 480)
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    assert metadata['layout'] == THREE_HOURLY_FULL_GRID_CACHE_LAYOUT
    assert metadata['cache_version'] == 1
    assert tuple(metadata['storage_shape']) == windows._normalized_map.shape == (88, 2, 240, 480)
    assert 'spatial_tile_shape' not in metadata
    assert not windows._normalized_map.flags.writeable
    actual[0][:] = 0
    np.testing.assert_array_equal(windows[1][0], expected[0])


def test_full_grid_batch_preserves_window_order_and_source_identity(full_grid_windows, monkeypatch):
    windows, _ = full_grid_windows
    wrapper = runner._ArrayDataset(windows)
    assert len(wrapper) == len(windows)
    expected = [wrapper[index] for index in [3, 1, 3]]
    from services import earth_dataset_metadata
    checks = []
    signature = earth_dataset_metadata.package_signature
    def counted(*args, **kwargs):
        checks.append(args)
        return signature(*args, **kwargs)
    monkeypatch.setattr(earth_dataset_metadata, 'package_signature', counted)
    actual = wrapper.__getitems__([3, 1, 3])
    assert len(checks) == 2
    for before, after in zip(expected, actual):
        assert before.keys() == after.keys()
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


@pytest.mark.parametrize('field,value', [('cache_version', 2), ('storage_shape', [88, 2, 24, 48])])
def test_full_grid_cache_checks_its_version_and_storage_shape(full_grid_windows, tmp_path, field, value):
    windows, path = full_grid_windows
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    metadata[field] = value
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='version|layout'):
        windows.use_training_cache(tmp_path / 'normalized.npy')


def full_grid_reference():
    model_spec = {
        'name': 'FullGridTest', 'parameters': {},
        'datasets': {'earth_merra2_3hourly_v1': FULL_GRID_FEED},
    }
    source = 'import torch\nfrom torch import nn\nMODEL_SPEC = ' + repr(model_spec) + '''
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Conv2d(config['in_channels'], config['horizon'], 1)
    def forward(self, inputs):
        return self.projection(inputs[:, -1]).unsqueeze(2)
def build_model(config):
    return Model(config)
'''
    return {
        'package_id': 'full-grid-pipeline-test', 'display_name': 'FullGridTest', 'version': 1,
        'content_hash': hashlib.sha256(source.encode('utf-8')).hexdigest(),
        'source_text': source, 'param_schema': {}, 'custom_model_params': {},
        'contract_schema': FULL_GRID_FEED['schema'],
    }


def full_grid_spec(release):
    registry = registry_for(release)
    ratios = {name + '_ratio': 1 / 3 for name in ('train', 'validation', 'test')}
    return registry, build_earth_training_spec(
        task_id=702, dataset_binding=registry.build_training_binding('earth_merra2_3hourly_v1'),
        task_split=build_earth_task_split(release.dates, 56, 24, ratios),
        hyperparameters={**ratios, 'model_source': 'uploaded', 'epochs': 1,
                         'batch_size': 1, 'seed': 5, 'selected_channels': []},
        uploaded_model=full_grid_reference(),
    )


def test_full_grid_training_uses_one_global_sample_and_publishes_reloadable_weights(
    synthetic_training_release, tmp_path, monkeypatch, capsys,
):
    registry, spec = full_grid_spec(synthetic_training_release)
    def no_tiles(*args, **kwargs):
        pytest.fail('Full-grid training/evaluation must not create a tile dataset')
    monkeypatch.setattr(runner, '_SpatialTileDataset', no_tiles)
    output = tmp_path / 'full-grid.pth'
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        result = runner.run_training(spec, output, registry, device='cpu', cache_root=tmp_path / 'cache')
    finally:
        torch.set_num_threads(previous_threads)
    checkpoint = load_earth_training_artifact(output, expected_task_id=702)
    assert result['split_window_counts'] == {'train': 1, 'validation': 1, 'test': 1}
    assert checkpoint.run['batch_unit'] == 'global_time_window'
    assert checkpoint.run['memory_strategy'] == 'normalized_memmap_full_grid'
    assert checkpoint.run['spatial_grid_shape'] == [240, 480]
    assert 'spatial_tile_shape' not in checkpoint.run
    assert checkpoint.training_contract['uploaded_model_schema'] == FULL_GRID_FEED['schema']
    logs = capsys.readouterr().out
    assert 'normalization TO3' in logs
    assert 'spatial mode=full_grid, train samples=1, batches=1' in logs
    assert 'Epoch 1/1 Batch 1/1' in logs
    assert 'Earth training completed' in logs


@pytest.mark.parametrize('batch_size', [0, 65, True, 1.5])
def test_full_grid_worker_rejects_excessive_batch_before_data_preparation(synthetic_training_release, tmp_path, batch_size):
    registry, spec = full_grid_spec(synthetic_training_release)
    with pytest.raises(runner.EarthTrainingError, match='integer between 1 and 64'):
        runner._build_loaders(spec, registry, batch_size, 5, cache_root=tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


def test_full_grid_worker_accepts_larger_batches(synthetic_training_release, tmp_path):
    registry, spec = full_grid_spec(synthetic_training_release)
    spec['hyperparameters']['batch_size'] = 4
    prepared = runner._build_loaders(spec, registry, 4, 5, cache_root=tmp_path / 'cache')
    assert prepared['full_grid'] is True
    assert prepared['train_loader'].batch_size == 4
    assert type(prepared['train_loader'].dataset) is runner._ArrayDataset
