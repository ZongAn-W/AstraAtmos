"""Three-hour UTC windows stay on disk and fit training-only normalization."""

from __future__ import annotations

import copy
import shutil
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType

import netCDF4
import numpy as np
import pytest
import xarray as xr

from scripts.build_earth_merra2_3hourly import DATA_FILE, _new_output
from services.earth_dataset import (
    CHANNELS,
    EarthThreeHourlyWindows,
    THREE_HOURLY_SCHEMA,
    canonical_input_channels,
    fit_threehour_normalization,
    build_threehour_training_cache,
    threehour_release_split_codes,
)
from services.earth_dataset_metadata import VerifiedEarthRelease, package_signature
from services.netcdf_read_lock import netcdf_read_lock


def _metadata(times, *, boundaries=None):
    days = times.astype('datetime64[D]')
    if boundaries is None:
        boundaries = {
            'train': ('2020-01-01', '2020-12-31'),
            'validation': ('2021-01-01', '2021-06-30'),
            'test': ('2021-07-01', '2021-12-31'),
        }
    splits = {}
    for name, (start, end) in boundaries.items():
        selected = days[(days >= np.datetime64(start)) & (days <= np.datetime64(end))]
        splits[name] = {
            'start': str(selected[0]) if len(selected) else None,
            'end': str(selected[-1]) if len(selected) else None,
            'steps': len(selected), 'days': len(selected) // 8,
        }
    return {
        'dataset_id': 'earth_merra2_3hourly_v1', 'schema': THREE_HOURLY_SCHEMA,
        'frequency_hours': 3, 'step_unit': 'hour', 'step': 3,
        'time': {'kind': 'datetime', 'time_zone': 'UTC', 'label': 'interval_center'},
        'grid_shape': [240, 480], 'splits': splits,
    }


def _normalization(release, channels=None):
    channels = canonical_input_channels(channels)
    dates = release.dates[threehour_release_split_codes(release.dates, release.metadata) == 0]
    first = np.datetime_as_string(dates[0], unit='s') + 'Z'
    last = np.datetime_as_string(dates[-1], unit='s') + 'Z'
    return {
        'method': 'per_channel_standard', 'fit_split': 'train',
        'fit_date_start': first, 'fit_date_end': last,
        'fit_time_start': first, 'fit_time_end': last, 'fit_step_count': len(dates),
        'time_zone': 'UTC', 'frequency_hours': 3, 'step_unit': 'hour', 'step': 3,
        'timestamp_rule': 'interval_center', 'channel_order': channels,
        'mean': [0.] * len(channels), 'scale': [1.] * len(channels),
        'constant_channel_mask': [False] * len(channels), 'target_channel_index': 0,
        'ddof': 0, 'epsilon': 1e-6,
    }


@pytest.fixture(scope='module')
def disk_release(tmp_path_factory):
    root = tmp_path_factory.mktemp('threehour_training')
    days = [date(2020, 1, 1) + timedelta(days=index) for index in range(11)]
    path = root / DATA_FILE
    spatial = (np.arange(240, dtype='float32')[:, None] * .005
               + np.arange(480, dtype='float32')[None, :] * .001)
    with netcdf_read_lock():
        stored = _new_output(path, days, 'a' * 64)
        try:
            for index, channel in enumerate(CHANNELS):
                for step in range(88):
                    stored[channel][step] = np.float32(100. + index * 5 + step * .25) + spatial
                    stored[f'{channel}_valid_mask'][step] = np.uint8(1)
        finally:
            stored.close()
    (root / 'manifest.json').write_text('{}', encoding='utf-8')
    times = np.datetime64('2020-01-01T01:30', 'ns') + np.arange(88) * np.timedelta64(3, 'h')
    return VerifiedEarthRelease(
        metadata=_metadata(times), signature=package_signature(root, data_file_name=DATA_FILE),
        dates=times, latitude=-89.625 + np.arange(240) * .75,
        longitude=-179.625 + np.arange(480) * .75,
        fields=MappingProxyType({}), data_path=path,
    )


@pytest.fixture
def mutable_release(disk_release, tmp_path):
    root = tmp_path / 'release'
    shutil.copytree(disk_release.data_path.parent, root)
    return replace(disk_release, data_path=root / DATA_FILE,
                   signature=package_signature(root, data_file_name=DATA_FILE))


def _refresh_signature(release):
    return replace(release, signature=package_signature(release.data_path.parent, data_file_name=DATA_FILE))


def test_full_published_counts_and_boundaries(disk_release):
    times = np.arange(np.datetime64('2020-01-01T01:30', 'ns'),
                      np.datetime64('2022-01-01T00:00', 'ns'), np.timedelta64(3, 'h'))
    release = replace(disk_release, dates=times, metadata=_metadata(times), signature=())
    normalization = _normalization(release, [])
    datasets = {
        name: EarthThreeHourlyWindows.from_release(
            release, split=name, selected_channels=[], normalization=normalization)
        for name in ('train', 'validation', 'test')
    }
    assert [len(datasets[name]) for name in datasets] == [2849, 1369, 1393]
    for name, dataset in datasets.items():
        entry = release.metadata['splits'][name]
        assert dataset.dates.dtype == np.dtype('datetime64[ns]')
        assert str(dataset.dates[0].astype('datetime64[D]')) == entry['start']
        assert str(dataset.dates[-1].astype('datetime64[D]')) == entry['end']
        assert dataset.dates[len(dataset) - 1 + 56 + 24 - 1] == dataset.dates[-1]
        assert dataset.dates[56] - dataset.dates[55] == np.timedelta64(3, 'h')
        assert not hasattr(dataset, 'data')
        assert dict(dataset._release.fields) == {}


def test_held_out_split_cannot_fit_its_own_normalization(disk_release):
    times = np.arange(np.datetime64('2020-01-01T01:30', 'ns'),
                      np.datetime64('2022-01-01T00:00', 'ns'), np.timedelta64(3, 'h'))
    release = replace(disk_release, dates=times, metadata=_metadata(times), signature=())
    for split in ('validation', 'test'):
        with pytest.raises(ValueError, match='Only the training split'):
            EarthThreeHourlyWindows.from_release(release, split=split, normalization=None)


def test_training_normalization_matches_train_population_in_bounded_chunks(disk_release, monkeypatch):
    def reject_load(*args, **kwargs):
        raise AssertionError('Loading the entire three-hour volume is forbidden')
    monkeypatch.setattr(xr.Dataset, 'load', reject_load)
    original = xr.DataArray.isel
    selections = []
    def capture(self, *args, **kwargs):
        selection = kwargs.get('time')
        if self.name in CHANNELS and isinstance(selection, slice):
            selections.append(selection.stop - selection.start)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(xr.DataArray, 'isel', capture)
    normalization = fit_threehour_normalization(disk_release, ['U10M'])
    assert selections and max(selections) <= 8
    assert normalization['channel_order'] == ['TO3', 'U10M']
    assert normalization['fit_time_start'] == '2020-01-01T01:30:00Z'
    assert normalization['fit_time_end'] == '2020-01-11T22:30:00Z'
    assert normalization['fit_step_count'] == 88
    assert normalization['frequency_hours'] == 3
    with netcdf_read_lock(), netCDF4.Dataset(disk_release.data_path) as stored:
        for index, channel in enumerate(normalization['channel_order']):
            expected = np.asarray(stored[channel][:], dtype='float64')
            assert normalization['mean'][index] == pytest.approx(float(expected.mean()), rel=1e-6)
            assert normalization['scale'][index] == pytest.approx(float(expected.std()), rel=1e-6)


def test_held_out_values_cannot_change_training_fit(disk_release, mutable_release):
    metadata = _metadata(disk_release.dates, boundaries={
        'train': ('2020-01-01', '2020-01-05'),
        'validation': ('2020-01-06', '2020-01-08'),
        'test': ('2020-01-09', '2020-01-11'),
    })
    baseline = fit_threehour_normalization(replace(disk_release, metadata=metadata), [])
    with netcdf_read_lock(), netCDF4.Dataset(mutable_release.data_path, 'r+') as stored:
        stored['TO3'][40:] += np.float32(10000)
    changed = _refresh_signature(replace(mutable_release, metadata=metadata))
    refit = fit_threehour_normalization(changed, [])
    assert refit == baseline
    assert refit['fit_time_end'] == '2020-01-05T22:30:00Z'


def test_lazy_shapes_order_and_target_round_trip(disk_release):
    normalization = fit_threehour_normalization(disk_release, ['SWGDN', 'U10M'])
    windows = EarthThreeHourlyWindows.from_release(
        disk_release, selected_channels=['SWGDN', 'U10M'], normalization=normalization)
    assert len(windows) == 9
    assert windows.input_channels == ['TO3', 'U10M', 'SWGDN']
    assert windows.input_units == ['DU', 'm s-1', 'W m-2']
    assert not hasattr(windows, 'data')
    inputs, targets = windows[0]
    assert inputs.shape == (56, 3, 240, 480)
    assert targets.shape == (24, 1, 240, 480)
    assert inputs.dtype == targets.dtype == np.dtype('float32')
    with netcdf_read_lock(), netCDF4.Dataset(disk_release.data_path) as stored:
        expected = np.asarray(stored['TO3'][56:80])
    np.testing.assert_allclose(windows.denormalize_ozone(targets[:, 0]), expected, rtol=1e-6, atol=1e-4)
    inputs[:] = 0
    assert np.any(windows.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))[0] != 0)


def test_spatial_tiles_equal_full_window_and_bound_reads(disk_release, monkeypatch):
    windows = EarthThreeHourlyWindows.from_release(disk_release, selected_channels=[])
    full_inputs, full_targets = windows[8]
    original = xr.DataArray.isel
    spatial_selections = []
    def capture(self, *args, **kwargs):
        if self.name in CHANNELS:
            spatial_selections.append((kwargs.get('lat'), kwargs.get('lon')))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(xr.DataArray, 'isel', capture)
    inputs, targets = windows.read_window(8, lat_slice=slice(7, 9), lon_slice=slice(12, 15))
    assert inputs.shape == (56, 1, 2, 3)
    assert targets.shape == (24, 1, 2, 3)
    np.testing.assert_array_equal(inputs, full_inputs[:, :, 7:9, 12:15])
    np.testing.assert_array_equal(targets, full_targets[:, :, 7:9, 12:15])
    assert all(lat == slice(7, 9) and lon == slice(12, 15) for lat, lon in spatial_selections)


@pytest.mark.parametrize('index', [-1, 9, 100])
def test_invalid_window_index_is_rejected(disk_release, index):
    windows = EarthThreeHourlyWindows.from_release(disk_release, selected_channels=[],
                                                 normalization=_normalization(disk_release, []))
    with pytest.raises(IndexError):
        windows[index]


@pytest.mark.parametrize('window,horizon', [(7, 3), (56, 3), (7, 24), (55, 24)])
def test_custom_windows_use_threehour_steps_and_exact_target_slice(disk_release, window, horizon):
    dataset = EarthThreeHourlyWindows.from_release(disk_release, window=window, horizon=horizon, selected_channels=[])
    inputs, targets = dataset.read_window(0, lat_slice=slice(0, 2), lon_slice=slice(0, 3))
    assert inputs.shape == (window, 1, 2, 3)
    assert targets.shape == (horizon, 1, 2, 3)
    assert len(dataset) == len(dataset.dates) - window - horizon + 1
    assert dataset.dates[window] - dataset.dates[window - 1] == np.timedelta64(3, 'h')


@pytest.mark.parametrize('mutate', [
    lambda block: block.update(frequency_hours=24),
    lambda block: block.update(step_unit='day'),
    lambda block: block.update(fit_time_end='2021-01-01T01:30:00Z'),
    lambda block: block.update(timestamp_rule='date'),
    lambda block: block.update(channel_order=['TO3', 'U10M']),
    lambda block: block.update(scale=[0.]),
])
def test_daily_or_tampered_normalization_is_rejected(disk_release, mutate):
    normalization = _normalization(disk_release, [])
    mutate(normalization)
    with pytest.raises(ValueError):
        EarthThreeHourlyWindows.from_release(disk_release, selected_channels=[], normalization=normalization)


@pytest.mark.parametrize('channel', ['TO3', 'U10M'])
def test_selected_channel_missing_values_fail_training_fit(mutable_release, channel):
    with netcdf_read_lock(), netCDF4.Dataset(mutable_release.data_path, 'r+') as stored:
        stored[channel][0, 0, 0] = np.nan
        stored[f'{channel}_valid_mask'][0, 0, 0] = 0
    release = _refresh_signature(mutable_release)
    with pytest.raises(ValueError, match=f'missing or non-finite values in {channel}'):
        fit_threehour_normalization(release, ['U10M'])
    if channel == 'U10M':
        assert fit_threehour_normalization(release, [])['channel_order'] == ['TO3']


def test_held_out_missing_values_fail_window_read(mutable_release):
    normalization = fit_threehour_normalization(mutable_release, [])
    with netcdf_read_lock(), netCDF4.Dataset(mutable_release.data_path, 'r+') as stored:
        stored['TO3'][60, 0, 0] = np.nan
        stored['TO3_valid_mask'][60, 0, 0] = 0
    release = _refresh_signature(mutable_release)
    windows = EarthThreeHourlyWindows.from_release(release, selected_channels=[], normalization=normalization)
    with pytest.raises(ValueError, match='missing or non-finite values in TO3'):
        windows[0]


def test_changed_package_is_rejected_instead_of_training_new_bytes(mutable_release):
    windows = EarthThreeHourlyWindows.from_release(mutable_release, selected_channels=[],
                                                 normalization=_normalization(mutable_release, []))
    with netcdf_read_lock(), netCDF4.Dataset(mutable_release.data_path, 'r+') as stored:
        stored['TO3'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='changed since verification'):
        windows.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))


def test_constant_channel_and_valid_zero_keep_finite_scale(mutable_release):
    with netcdf_read_lock(), netCDF4.Dataset(mutable_release.data_path, 'r+') as stored:
        stored['TO3'][:] = np.float32(0)
    release = _refresh_signature(mutable_release)
    normalization = fit_threehour_normalization(release, [])
    assert normalization['mean'] == [0.]
    assert normalization['scale'] == [1.]
    assert normalization['constant_channel_mask'] == [True]
    windows = EarthThreeHourlyWindows.from_release(release, selected_channels=[], normalization=normalization)
    inputs, targets = windows.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))
    assert np.isfinite(inputs).all() and np.isfinite(targets).all()
    assert not np.any(inputs) and not np.any(targets)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'nan', 'off_center', 'overlap', 'uncovered', 'count'])
def test_invalid_axis_or_splits_are_rejected(disk_release, mutation):
    dates = disk_release.dates.copy()
    metadata = copy.deepcopy(disk_release.metadata)
    if mutation == 'duplicate':
        dates[10] = dates[9]
    elif mutation == 'missing':
        dates[10:] += np.timedelta64(3, 'h')
    elif mutation == 'nan':
        dates[10] = np.datetime64('NaT')
    elif mutation == 'off_center':
        dates += np.timedelta64(1, 'm')
    elif mutation == 'overlap':
        metadata['splits']['validation'] = {'start': '2020-01-01', 'end': '2020-01-11', 'steps': 88}
    elif mutation == 'uncovered':
        metadata['splits']['train'] = {'start': '2020-01-01', 'end': '2020-01-10', 'steps': 80}
    elif mutation == 'count':
        metadata['splits']['train']['steps'] -= 1
    with pytest.raises(ValueError):
        threehour_release_split_codes(dates, metadata)


def test_insufficient_smoke_steps_and_daily_release_are_rejected(disk_release):
    short = disk_release.dates[:56]
    with pytest.raises(ValueError, match='Not enough three-hour steps'):
        EarthThreeHourlyWindows.from_release(replace(disk_release, dates=short, metadata=_metadata(short)))
    daily = copy.deepcopy(disk_release.metadata)
    daily['schema'] = 'astraatmos_earth_daily_v1'
    with pytest.raises(ValueError, match='three-hour release'):
        EarthThreeHourlyWindows.from_release(replace(disk_release, metadata=daily))


def test_no_server_path_or_invalid_spatial_tile_is_rejected(disk_release):
    normalization = _normalization(disk_release, [])
    with pytest.raises(ValueError, match='no server data path'):
        EarthThreeHourlyWindows.from_release(replace(disk_release, data_path=None), selected_channels=[],
                                             normalization=normalization)
    windows = EarthThreeHourlyWindows.from_release(disk_release, selected_channels=[], normalization=normalization)
    for tile in ([0, 1], slice(0, 0)):
        with pytest.raises(ValueError):
            windows.read_window(0, lat_slice=tile)


def test_tensor_denormalization_preserves_type(disk_release):
    torch = pytest.importorskip('torch')
    windows = EarthThreeHourlyWindows.from_release(disk_release, selected_channels=[],
                                                 normalization=_normalization(disk_release, []))
    values = torch.zeros(2, 24)
    result = windows.denormalize_ozone(values)
    assert isinstance(result, torch.Tensor)
    assert result.device == values.device


def test_normalized_disk_cache_reads_tiles_without_reopening_netcdf(disk_release, tmp_path, monkeypatch):
    normalization = fit_threehour_normalization(disk_release, ['U10M'])
    windows = EarthThreeHourlyWindows.from_release(
        disk_release, selected_channels=['U10M'], normalization=normalization,
    )
    expected = windows.read_window(0, lat_slice=slice(2, 4), lon_slice=slice(6, 9))
    path = build_threehour_training_cache(
        disk_release, ['TO3', 'U10M'], normalization, tmp_path / 'cache',
    )
    windows.use_training_cache(path)
    assert isinstance(windows._normalized_map, np.memmap)
    assert windows._normalized_map.flags.writeable is False

    def no_reopen(*args, **kwargs):
        pytest.fail('Training tiles must use the disk map after preparation')
    monkeypatch.setattr(xr, 'open_dataset', no_reopen)
    actual = windows.read_window(0, lat_slice=slice(2, 4), lon_slice=slice(6, 9))
    for first, second in zip(actual, expected):
        np.testing.assert_allclose(first, second, atol=2e-6, rtol=2e-6)
    metadata_path = path.parent / 'metadata.json'
    metadata = __import__('json').loads(metadata_path.read_text())
    metadata['normalization']['mean'][0] += 1
    metadata_path.write_text(__import__('json').dumps(metadata))
    with pytest.raises(ValueError, match='normalization mismatch'):
        windows.use_training_cache(path)
