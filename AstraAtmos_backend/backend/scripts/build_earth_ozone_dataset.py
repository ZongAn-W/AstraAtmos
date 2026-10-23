"""Package an existing MERRA-2 daily O3/U/V/T/S NPZ derivative as compact NetCDF.

This preserves the derivative's float32 values; it does not rescan the raw archive.
Run from the backend directory with `python -m scripts.build_earth_ozone_dataset`.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from services.earth_dataset import CHANNELS, SCHEMA, SPLITS, UNITS, split_days, validate_dataset

SOURCE_KEYS = ('O3', 'U', 'V', 'T', 'S')
LONG_NAMES = ('Total column ozone', '10 m eastward wind', '10 m northward wind',
              '2 m air temperature', 'Surface incoming shortwave flux')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_dataset(source_path, output_dir, *, train_end='2020-12-31', validation_end='2021-06-30'):
    source_path, output_dir = Path(source_path), Path(output_dir)
    output = output_dir / 'earth_merra2_daily.nc'
    manifest_path = output_dir / 'manifest.json'
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f'Dataset already exists in {output_dir}; choose a new output directory')
    with np.load(source_path, allow_pickle=False) as src:
        metadata = json.loads(str(src['metadata'].item()))
        if metadata.get('temporal') != 'daily' or metadata.get('source') != 'MERRA2':
            raise ValueError('Expected the MERRA2 daily derivative, not hourly or another source')
        for key, name in zip(SOURCE_KEYS, CHANNELS):
            if metadata.get('used_variables', {}).get(key) != [name]:
                raise ValueError(f'{key} must originate from {name}')
        # Daily labels must not be silently truncated from hourly timestamps.
        labels = src['time_iso'].astype(str)
        dates = labels.astype('datetime64[D]')
        if not np.array_equal(labels, dates.astype(str)):
            raise ValueError('time_iso must contain YYYY-MM-DD dates')
        ds = xr.Dataset(
            {name: (('time', 'lat', 'lon'), src[key].astype('float32'),
                    {'units': unit, 'long_name': label, 'cell_methods': 'time: mean'})
             for key, name, unit, label in zip(SOURCE_KEYS, CHANNELS, UNITS, LONG_NAMES)},
            coords={'time': dates.astype('datetime64[ns]'),
                    'lat': ('lat', src['lat'].astype('float32'), {'units': 'degrees_north'}),
                    'lon': ('lon', src['lon'].astype('float32'), {'units': 'degrees_east'})},
            attrs={'planet': 'Earth', 'schema': SCHEMA, 'source': 'NASA MERRA-2',
                   'temporal_resolution': '1 day', 'train_end': train_end,
                   'validation_end': validation_end,
                   'processing': 'Existing daily spatially subsampled derivative; no further resampling',
                   'source_file': source_path.name, 'source_sha256': sha256(source_path)},
        )
    ds['split'] = ('time', split_days(dates, train_end, validation_end),
                   {'flag_values': np.array([0, 1, 2], dtype='int8'),
                    'flag_meanings': 'train validation test'})
    validate_dataset(ds)
    manifest = {
        'schema': SCHEMA, 'planet': 'Earth', 'source_file': source_path.name,
        'source_bytes': source_path.stat().st_size, 'source_sha256': ds.attrs['source_sha256'],
        'data_file': output.name, 'dimensions': dict(ds.sizes),
        'time_start': str(dates[0]), 'time_end': str(dates[-1]), 'cadence': 'daily mean',
        'latitude_range': [float(ds.lat.min()), float(ds.lat.max())],
        'longitude_range': [float(ds.lon.min()), float(ds.lon.max())],
        'source_spatial_stride': {axis: metadata.get(f'{axis}_stride') for axis in ('lat', 'lon')},
        'channel_order': list(CHANNELS), 'variables': {}, 'splits': {},
        'normalization': 'Per-channel mean/std fitted on train days only by EarthOzoneWindows',
        'limitations': ['Regional grid, not a complete global map',
                        'Spatial point subsampling, not area averaging',
                        'Daily means do not retain hourly variation',
                        'No additional loss of float32 precision during packaging',
                        'Earth data reader is separate from the Mars web/API dataset registry'],
    }
    for name, unit in zip(CHANNELS, UNITS):
        manifest['variables'][name] = {'units': unit, 'dtype': 'float32',
                                       'min': float(ds[name].min()), 'max': float(ds[name].max())}
    for name, code in SPLITS.items():
        selected = dates[ds['split'].values == code]
        manifest['splits'][name] = {'start': str(selected[0]), 'end': str(selected[-1]),
                                   'days': len(selected)}
    output_dir.mkdir(parents=True, exist_ok=True)
    encoding = {name: {'dtype': 'float32', 'zlib': True, 'complevel': 4, 'shuffle': True,
                       'chunksizes': (1, ds.sizes['lat'], ds.sizes['lon'])} for name in CHANNELS}
    ds.to_netcdf(output, engine='netcdf4', encoding=encoding)
    ds.close()
    manifest['data_bytes'] = output.stat().st_size
    manifest['data_sha256'] = sha256(output)
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Existing MERRA2 daily prepared NPZ')
    parser.add_argument('--output-dir', type=Path, default=Path('data/earth/merra2_daily_v1'))
    parser.add_argument('--train-end', default='2020-12-31')
    parser.add_argument('--validation-end', default='2021-06-30')
    args = parser.parse_args()
    path = build_dataset(args.input, args.output_dir, train_end=args.train_end,
                         validation_end=args.validation_end)
    print(f'Created {path} ({path.stat().st_size / 1024**2:.2f} MiB)')


if __name__ == '__main__':
    main()
