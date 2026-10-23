"""Conservatively aggregate a verified three-hour 0.75 degree release to 5 degrees.

Time coordinates, bounds, variables, units and published splits are preserved.
Existing output is verified rather than overwritten; interrupted output is kept.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from services.earth_dataset import (CHANNELS, THREE_HOURLY_5DEG_DATASET_ID,
    THREE_HOURLY_5DEG_TRAINING_PROFILE, validate_earth_3hourly_dataset)
from services.earth_dataset_metadata import (file_sha256, package_signature,
    read_earth_3hourly_release, threehour_manifest_content_sha256)
from services.dataset_identity import build_dataset_fingerprint
from services.netcdf_read_lock import netcdf_read_lock


def overlap_weights(source_bounds, target_bounds, *, spherical=False):
    """Rows are target cells, columns source cells; each row sums to one."""
    source, target = np.asarray(source_bounds, dtype='float64'), np.asarray(target_bounds, dtype='float64')
    south = np.maximum(target[:, None, 0], source[None, :, 0])
    north = np.minimum(target[:, None, 1], source[None, :, 1])
    if spherical:
        overlap = np.maximum(0., np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south)))
        width = np.diff(np.sin(np.deg2rad(target)), axis=1)[:, 0]
    else:
        overlap = np.maximum(0., north - south)
        width = np.diff(target, axis=1)[:, 0]
    weights = overlap / width[:, None]
    if not np.allclose(weights.sum(axis=1), 1., rtol=0, atol=1e-12):
        raise ValueError('Source cells do not completely cover the target grid')
    return csr_matrix(weights)


def regrid(values, valid, lat_weights, lon_weights):
    """A missing cell with positive overlap invalidates the whole target cell."""
    lat_support, lon_support = lat_weights.copy(), lon_weights.copy()
    lat_support.data[:] = 1
    lon_support.data[:] = 1
    out = np.empty((len(values), lat_weights.shape[0], lon_weights.shape[0]), dtype='float32')
    for index, (field, mask) in enumerate(zip(values, valid)):
        good = mask.astype(bool) & np.isfinite(field)
        lat_mapped = lat_weights @ np.where(good, field, 0.).astype('float64')
        mapped = (lon_weights @ lat_mapped.T).T
        lat_missing = lat_support @ (~good).astype('float64')
        missing = (lon_support @ lat_missing.T).T > 0
        mapped[missing] = np.nan
        out[index] = mapped
    return out


def build_dataset(source_dir, output_dir, *, verification_cache=None, progress=print):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    checkout = BACKEND.parents[1]
    if output_dir == checkout or checkout in output_dir.parents or output_dir == source_dir:
        raise ValueError('Use a separate output directory outside the Git checkout')
    release = read_earth_3hourly_release(source_dir, verification_cache_dir=verification_cache)
    if release.metadata.get('dataset_id', 'earth_merra2_3hourly_v1') != 'earth_merra2_3hourly_v1' or len(release.latitude) != 240:
        raise ValueError('Source must be the native verified 0.75 degree release')
    if output_dir.exists():
        existing = read_earth_3hourly_release(output_dir)
        if (existing.metadata['manifest'].get('parent_release', {}).get('dataset_fingerprint') != release.metadata['dataset_fingerprint']
                or len(existing.latitude) != 36):
            raise ValueError('Existing output belongs to a different source or grid')
        return existing
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / 'earth_merra2_3hourly.nc'
    lat_bounds = np.column_stack((np.arange(36) * 5. - 90, (np.arange(36) + 1) * 5. - 90))
    lon_bounds = np.column_stack((np.arange(72) * 5. - 180, (np.arange(72) + 1) * 5. - 180))
    manifest = copy.deepcopy(release.metadata['manifest'])
    started, last = time.monotonic(), 0.
    with netcdf_read_lock(), netCDF4.Dataset(str(release.data_path)) as source, netCDF4.Dataset(str(output), 'w', clobber=False) as target:
        source.set_auto_mask(False)
        for name, dimension in source.dimensions.items():
            target.createDimension(name, 36 if name == 'lat' else 72 if name == 'lon' else len(dimension))
        target.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
        target.dataset_id = THREE_HOURLY_5DEG_DATASET_ID
        target.parent_dataset_fingerprint = release.metadata['dataset_fingerprint']
        target.regrid_algorithm = 'native_0p75_to_5deg_spherical_cell_overlap_v1'
        lat_weight = overlap_weights(source['lat_bounds'][:], lat_bounds, spherical=True)
        lon_weight = overlap_weights(source['lon_bounds'][:], lon_bounds)
        for name, variable in source.variables.items():
            attributes = {key: variable.getncattr(key) for key in variable.ncattrs() if key != '_FillValue'}
            options = {}
            if variable.dimensions == ('time', 'lat', 'lon'):
                options.update(zlib=True, complevel=4, chunksizes=(8, 36, 72))
            if '_FillValue' in variable.ncattrs():
                options['fill_value'] = variable.getncattr('_FillValue')
            created = target.createVariable(name, variable.dtype, variable.dimensions, **options)
            created.setncatts(attributes)
            if name == 'lat':
                created[:] = lat_bounds.mean(axis=1)
            elif name == 'lon':
                created[:] = lon_bounds.mean(axis=1)
            elif name == 'lat_bounds':
                created[:] = lat_bounds
            elif name == 'lon_bounds':
                created[:] = lon_bounds
            elif variable.dimensions != ('time', 'lat', 'lon'):
                created[:] = variable[:]
        for start in range(0, len(release.dates), 8):
            stop = min(start + 8, len(release.dates))
            for channel in CHANNELS:
                values = regrid(source[channel][start:stop], source[channel + '_valid_mask'][start:stop], lat_weight, lon_weight)
                target[channel][start:stop] = values
                target[channel + '_valid_mask'][start:stop] = np.isfinite(values).astype('uint8')
            if progress and (stop == len(release.dates) or time.monotonic() - last >= 5):
                progress(f'Earth 5 degree aggregation: {stop}/{len(release.dates)} frames ({time.monotonic() - started:.1f}s)')
                last = time.monotonic()
    if package_signature(source_dir, data_file_name='earth_merra2_3hourly.nc') != release.signature:
        raise ValueError('Source changed while building; partial output retained')
    with netcdf_read_lock(), xr.open_dataset(output, engine='netcdf4', mask_and_scale=False) as dataset:
        statistics = validate_earth_3hourly_dataset(dataset)
    manifest.update(dataset_id=THREE_HOURLY_5DEG_DATASET_ID,
        dimensions={'time': len(release.dates), 'lat': 36, 'lon': 72, 'bounds': 2},
        grid_shape=[36, 72],
        grid={'latitude': {'start': -87.5, 'end': 87.5, 'step': 5., 'count': 36},
              'longitude': {'start': -177.5, 'end': 177.5, 'step': 5., 'count': 72},
              'latitude_bounds': [-90., 90.], 'longitude_bounds': [-180., 180.]},
        training_profile=dict(THREE_HOURLY_5DEG_TRAINING_PROFILE), variables=statistics,
        parent_release={key: release.metadata[key] for key in ('dataset_fingerprint', 'manifest_sha256', 'data_sha256')},
        data_bytes=output.stat().st_size, data_sha256=file_sha256(output))
    manifest['processing'].update(target_grid='global 5 x 5 degree cell means',
        regrid_algorithm='native_0p75_to_5deg_spherical_cell_overlap_v1', source_grid_shape=[240, 480])
    # Preserve the parent's inventory/digest basis; a separate parent-release
    # identity records that the input here is a published three-hour package.
    manifest['dataset_fingerprint'] = build_dataset_fingerprint(THREE_HOURLY_5DEG_DATASET_ID, 'v1',
        threehour_manifest_content_sha256(manifest), manifest['data_sha256'])
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    verified = read_earth_3hourly_release(output_dir, verification_cache_dir=verification_cache)
    (output_dir / 'verification.json').write_text(json.dumps({
        'validation': 'passed', 'dataset_id': THREE_HOURLY_5DEG_DATASET_ID,
        'shape': [len(release.dates), 36, 72], 'time': verified.metadata['time'],
        'variables': statistics, 'dataset_fingerprint': verified.metadata['dataset_fingerprint'],
        'parent_release': manifest['parent_release']}, indent=2) + '\n', encoding='utf-8')
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verification-cache', type=Path)
    args = parser.parse_args()
    result = build_dataset(args.source_dir, args.output_dir, verification_cache=args.verification_cache)
    print(f"Release ready: {result.data_path}; fingerprint={result.metadata['dataset_fingerprint']}")
