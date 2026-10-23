import json

import numpy as np
import pytest
import torch

from models.training_scripts.earth_daily import (
    EarthTrainingError, _SpatialTileDataset, _require_finite_gradients,
)
from services.earth_dataset import (
    EarthThreeHourlyWindows, THREE_HOURLY_TILE_CACHE_LAYOUT, build_threehour_training_cache,
)
from test_earth_3hourly_training_data import disk_release, _normalization


@pytest.fixture(scope='module')
def cached_windows(disk_release, tmp_path_factory):
    normalization = _normalization(disk_release, ['U10M'])
    windows = EarthThreeHourlyWindows.from_release(
        disk_release, selected_channels=['U10M'], normalization=normalization,
    )
    path = build_threehour_training_cache(
        disk_release, ['TO3', 'U10M'], normalization,
        tmp_path_factory.mktemp('tile_cache'),
    )
    windows.use_training_cache(path)
    return windows, path


@pytest.mark.parametrize('latitude,longitude', [
    (slice(0, 24), slice(0, 48)),
    (slice(216, 240), slice(432, 480)),
    (slice(20, 28), slice(45, 55)),
    (slice(5, 35, 2), slice(25, 75, 3)),
    (slice(40, 15, -2), slice(70, 35, -3)),
    (slice(None), slice(None)),
])
def test_tile_cache_preserves_global_tile_and_arbitrary_slices(cached_windows, latitude, longitude):
    windows, path = cached_windows
    uncached = EarthThreeHourlyWindows.from_release(
        windows._release, selected_channels=['U10M'], normalization=windows.normalization,
    )
    expected = uncached.read_window(1, lat_slice=latitude, lon_slice=longitude)
    actual = windows.read_window(1, lat_slice=latitude, lon_slice=longitude)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
        assert right.dtype == np.float32
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    assert metadata['layout'] == THREE_HOURLY_TILE_CACHE_LAYOUT
    assert tuple(metadata['storage_shape']) == windows._normalized_map.shape
    assert windows._normalized_map.flags.writeable is False


def test_existing_global_cache_remains_readable(cached_windows, tmp_path):
    windows, path = cached_windows
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    legacy_path = tmp_path / 'normalized.npy'
    legacy = np.lib.format.open_memmap(legacy_path, mode='w+', dtype='float32', shape=tuple(metadata['shape']))
    for tile in range(100):
        row, column = divmod(tile, 10)
        legacy[:, :, row * 24:(row + 1) * 24, column * 48:(column + 1) * 48] = windows._normalized_map[tile]
    legacy.flush()
    legacy._mmap.close()
    for key in ('layout', 'storage_shape', 'spatial_tile_shape', 'cache_identity_version',
                'cache_status', 'cache_key', 'identity', 'array_identity', 'array_sha256',
                'package_signature', 'proof_sha256'):
        metadata.pop(key, None)
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    old = EarthThreeHourlyWindows.from_release(
        windows._release, selected_channels=['U10M'], normalization=windows.normalization,
    )
    old.use_training_cache(legacy_path)
    for latitude, longitude in ((slice(20, 28), slice(45, 55)), (slice(216, 240), slice(432, 480))):
        for before, after in zip(old.read_window(1, lat_slice=latitude, lon_slice=longitude),
                                 windows.read_window(1, lat_slice=latitude, lon_slice=longitude)):
            np.testing.assert_array_equal(before, after)


def test_cached_batch_checks_identity_at_both_boundaries(cached_windows, monkeypatch):
    windows, _ = cached_windows
    requests = [(0, slice(0, 24), slice(0, 48)), (1, slice(24, 48), slice(48, 96))]
    expected = [windows.read_window(index, lat_slice=latitude, lon_slice=longitude)
                for index, latitude, longitude in requests]
    checks = []
    from services import earth_dataset_metadata
    signature = earth_dataset_metadata.package_signature
    def counted(*args, **kwargs):
        checks.append(args)
        return signature(*args, **kwargs)
    monkeypatch.setattr(earth_dataset_metadata, 'package_signature', counted)
    result = windows.read_windows(requests)
    assert len(checks) == 2
    for before, after in zip(expected, result):
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)


def test_cached_batch_rejects_source_change_during_reads(cached_windows, monkeypatch):
    windows, _ = cached_windows
    from services import earth_dataset_metadata
    calls = []
    def changed(*args, **kwargs):
        calls.append(args)
        return windows._release.signature if len(calls) == 1 else ('changed',)
    monkeypatch.setattr(earth_dataset_metadata, 'package_signature', changed)
    with pytest.raises(ValueError, match='changed since verification'):
        windows.read_windows([(0, slice(0, 24), slice(0, 48))])
    assert len(calls) == 2


@pytest.mark.parametrize('field,value', [
    ('layout', 'unsupported'), ('storage_shape', [100, 1, 2, 24, 48]), ('spatial_tile_shape', [48, 24]),
])
def test_cache_layout_metadata_is_verified(cached_windows, tmp_path, field, value):
    windows, path = cached_windows
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    metadata[field] = value
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='layout'):
        windows.use_training_cache(tmp_path / 'normalized.npy')


def test_batched_tile_fetch_preserves_sampler_order_and_duplicates(cached_windows):
    windows, _ = cached_windows
    wrapper = _SpatialTileDataset(windows)
    indices = [101, 0, 899, 101]
    expected = [wrapper[index] for index in indices]
    actual = wrapper.__getitems__(indices)
    for before, after in zip(expected, actual):
        assert before.keys() == after.keys()
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_finite_gradient_checks_use_one_boolean_without_modifying_gradients(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model(torch.ones(2, 2)).sum().backward()
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    booleans = []
    original = torch.Tensor.__bool__
    def counted(tensor):
        booleans.append(tensor)
        return original(tensor)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, '__bool__', counted)
        _require_finite_gradients(model)
    assert len(booleans) == 1
    for gradient, parameter in zip(gradients, model.parameters()):
        torch.testing.assert_close(gradient, parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_gradient_identifies_first_invalid_parameter(invalid):
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.zeros_like(model.weight)
    model.bias.grad = torch.full_like(model.bias, invalid)
    with pytest.raises(EarthTrainingError, match='Non-finite gradient for bias'):
        _require_finite_gradients(model)


def test_missing_gradients_keep_existing_behavior():
    _require_finite_gradients(torch.nn.Linear(2, 1))
