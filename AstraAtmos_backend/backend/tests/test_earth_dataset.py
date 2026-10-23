import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_earth_ozone_dataset import build_dataset
from services.earth_dataset import CHANNELS, EarthOzoneWindows, load_earth_dataset


def make_source(tmp_path, *, gap=False, nonfinite=False, wrong_variable=False):
    dates = np.arange('2020-12-20', '2021-01-19', dtype='datetime64[D]')
    if gap:
        dates[10:] += np.timedelta64(1, 'D')
    values = np.arange(30 * 2 * 3, dtype=np.float32).reshape(30, 2, 3) + 1
    # A future distribution shift must not affect training normalization.
    values[12:] += 10000
    if nonfinite:
        values[0, 0, 0] = np.nan
    source = tmp_path / 'source.npz'
    np.savez_compressed(
        source, **{name: values + i for i, name in enumerate(('O3', 'U', 'V', 'T', 'S'))},
        lat=np.array([-4, 0], dtype=np.float32),
        lon=np.array([-5, 0, 5], dtype=np.float32), time_iso=dates.astype(str),
        metadata=json.dumps({'source': 'MERRA2', 'temporal': 'daily',
                             'used_variables': {'O3': ['TO3'], 'U': ['U10M'],
                                                'V': ['V10M'], 'T': ['T2M'],
                                                'S': ['SWTDN' if wrong_variable else 'SWGDN']}}),
    )
    return source


def export(tmp_path, **kwargs):
    return build_dataset(make_source(tmp_path, **kwargs), tmp_path / 'earth',
                         train_end='2020-12-31', validation_end='2021-01-09')


def test_compact_roundtrip_preserves_data_dates_units_and_compression(tmp_path):
    path = export(tmp_path)
    with load_earth_dataset(path) as ds, np.load(tmp_path / 'source.npz') as src:
        assert ds.attrs['planet'] == 'Earth'
        assert ds.sizes == {'time': 30, 'lat': 2, 'lon': 3}
        assert ds.TO3.attrs['units'] == 'DU'
        assert ds.SWGDN.attrs['units'] == 'W m-2'
        np.testing.assert_array_equal(ds.TO3.values, src['O3'])
        np.testing.assert_array_equal(ds.time.values.astype('datetime64[D]'),
                                      src['time_iso'].astype('datetime64[D]'))
        assert 'Ls' not in ds and 'mars_year' not in ds
    with netCDF4.Dataset(path) as ds:
        assert ds['TO3'].filters()['zlib']
    manifest = json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
    assert manifest['splits']['train']['days'] == 12
    assert manifest['splits']['validation']['days'] == 9
    assert manifest['splits']['test']['days'] == 9
    assert len(manifest['source_sha256']) == 64


@pytest.mark.parametrize('invalid', ['gap', 'nonfinite', 'wrong_variable'])
def test_rejects_invalid_source_without_writing_dataset(tmp_path, invalid):
    with pytest.raises(ValueError):
        export(tmp_path, **{invalid: True})
    assert not (tmp_path / 'earth' / 'earth_merra2_daily.nc').exists()


def test_rejects_duplicate_grid_coordinates(tmp_path):
    path = export(tmp_path)
    with xr.open_dataset(path) as ds:
        bad = ds.load().assign_coords(lon=[0, 0, 5])
    bad_path = tmp_path / 'bad.nc'
    bad.to_netcdf(bad_path)
    with pytest.raises(ValueError, match='lon'):
        load_earth_dataset(bad_path)


def test_export_never_overwrites_existing_package(tmp_path):
    path = export(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        build_dataset(tmp_path / 'source.npz', path.parent,
                      train_end='2020-12-31', validation_end='2021-01-09')
    assert path.read_bytes() == original


def test_windows_do_not_cross_splits_and_stats_use_training_only(tmp_path):
    path = export(tmp_path)
    train = EarthOzoneWindows(path, split='train', window=3, horizon=2)
    val = EarthOzoneWindows(path, split='validation', window=3, horizon=2)
    test = EarthOzoneWindows(path, split='test', window=3, horizon=2)
    assert len(train) == 8 and len(val) == len(test) == 5
    assert train.dates[-1] < val.dates[0] < val.dates[-1] < test.dates[0]
    x, y = train[0]
    assert x.shape == (3, 5, 2, 3) and y.shape == (2, 1, 2, 3)
    assert x.dtype == y.dtype == np.float32
    with load_earth_dataset(path) as ds:
        np.testing.assert_allclose(train.mean[0], ds.TO3.values[:12].mean(), rtol=1e-6)
        np.testing.assert_allclose(train.denormalize_ozone(y[:, 0]),
                                   ds.TO3.values[3:5], rtol=1e-6)
    np.testing.assert_array_equal(train.mean, val.mean)
    np.testing.assert_array_equal(train.std, test.std)
    assert val[0][0].mean() > 100
    with pytest.raises(IndexError):
        train[len(train)]
    with pytest.raises(ValueError):
        EarthOzoneWindows(path, split='test', window=10, horizon=2)


def test_invalid_split_and_channel_contract_fail_clearly(tmp_path):
    path = export(tmp_path)
    with pytest.raises(ValueError):
        EarthOzoneWindows(path, split='all', window=3, horizon=2)
    assert CHANNELS[0] == 'TO3'
