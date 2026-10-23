"""Earth daily and three-hour data contracts, independent of Mars MY/Ls services.

Three-hour validation preserves UTC datetimes and explicit missing-value masks.
Daily windows retain their existing in-memory contract. Three-hour windows keep
the data on disk and read only one window or spatial tile at a time.

Two entry points share one implementation:

* :meth:`EarthOzoneWindows.__init__` keeps the original standalone behaviour of
  reading a package path directly, which the dataset verification and smoke
  scripts use.
* :meth:`EarthOzoneWindows.from_release` builds windows from an already verified
  :class:`VerifiedEarthRelease`, selects channels and reuses normalization
  statistics instead of fitting them again.

Normalization is always fitted on the **training dates only** and then reused by
validation, test and prediction. Nothing here mutates the release arrays: they
are read-only and shared with the overview API.
"""

from pathlib import Path
import operator
import json
import time
import os
import copy

import numpy as np
import xarray as xr

CHANNELS = ('TO3', 'U10M', 'V10M', 'T2M', 'SWGDN')
UNITS = ('DU', 'm s-1', 'm s-1', 'K', 'W m-2')
SPLITS = {'train': 0, 'validation': 1, 'test': 2}
SCHEMA = 'astraatmos_earth_daily_v1'
THREE_HOURLY_SCHEMA = 'astraatmos_earth_3hourly_v1'
THREE_HOURLY_DATASET_ID = 'earth_merra2_3hourly_v1'
THREE_HOURLY_DATASET_VERSION = 'v1'
THREE_HOURLY_5DEG_DATASET_ID = 'earth_merra2_3hourly_5deg_v1'
THREE_HOURLY_DATASET_IDS = (THREE_HOURLY_DATASET_ID, THREE_HOURLY_5DEG_DATASET_ID)
THREE_HOURLY_GRID_SHAPE = (240, 480)
THREE_HOURLY_5DEG_GRID_SHAPE = (36, 72)
THREE_HOURLY_TILE_SHAPE = (24, 48)
THREE_HOURLY_TILE_CACHE_LAYOUT = 'earth_spatial_tiles_v1'
THREE_HOURLY_FULL_GRID_CACHE_LAYOUT = 'earth_full_grid_v1'
THREE_HOURLY_CACHE_SCHEMA = 'astraatmos_earth_training_cache_v1'
THREE_HOURLY_TRAINING_PROFILE = {
    'target': 'TO3', 'target_unit': 'DU', 'window': 56, 'horizon': 24,
    'step_unit': 'hour', 'step': 3, 'grid_shape': [240, 480],
}
THREE_HOURLY_5DEG_TRAINING_PROFILE = {
    'target': 'TO3', 'target_unit': 'DU', 'window': 56, 'horizon': 24,
    'step_unit': 'hour', 'step': 3, 'grid_shape': [36, 72],
}


def threehourly_grid_shape(dataset_id):
    if dataset_id == THREE_HOURLY_5DEG_DATASET_ID:
        return THREE_HOURLY_5DEG_GRID_SHAPE
    if dataset_id == THREE_HOURLY_DATASET_ID:
        return THREE_HOURLY_GRID_SHAPE
    raise ValueError(f'Unsupported three-hour Earth dataset: {dataset_id}')

TARGET_CHANNEL = 'TO3'
TARGET_CHANNEL_INDEX = 0
OPTIONAL_CHANNELS = ('U10M', 'V10M', 'T2M', 'SWGDN')

# Published normalization contract. The stored values are valid float32 numbers
# computed with population statistics over the training dates.
NORMALIZATION_METHOD = 'per_channel_standard'
NORMALIZATION_CACHE_VERSION = 'earth_3hourly_normalization_v2'
NORMALIZATION_DDOF = 0
NORMALIZATION_EPSILON = 1e-6
MINIMUM_SCALE = 1e-6


def split_days(dates, train_end, validation_end):
    train_end = np.datetime64(train_end, 'D')
    validation_end = np.datetime64(validation_end, 'D')
    if np.isnat(train_end) or np.isnat(validation_end) or train_end >= validation_end:
        raise ValueError('train_end must be earlier than validation_end')
    splits = np.where(dates <= train_end, 0, np.where(dates <= validation_end, 1, 2)).astype('int8')
    if set(splits.tolist()) != {0, 1, 2}:
        raise ValueError('Each chronological split must contain at least one day')
    return splits


def validate_dataset(ds):
    if ds.attrs.get('schema') == THREE_HOURLY_SCHEMA:
        return validate_earth_3hourly_dataset(ds)
    if ds.attrs.get('planet') != 'Earth' or ds.attrs.get('schema') != SCHEMA:
        raise ValueError('Expected an AstraAtmos Earth daily dataset')
    for name, limit in (('lat', 90), ('lon', 180)):
        if name not in ds.coords or ds[name].dims != (name,):
            raise ValueError(f'{name} must be a one-dimensional coordinate')
        values = ds[name].values
        if (len(values) < 2 or not np.isfinite(values).all()
                or np.any(np.abs(values) > limit) or np.any(np.diff(values) <= 0)):
            raise ValueError(f'{name} must be finite, increasing and within +/-{limit}')
    has_bounds = ('lat_bounds' in ds, 'lon_bounds' in ds)
    if any(has_bounds) and not all(has_bounds):
        raise ValueError('Both latitude and longitude cell bounds are required')
    for axis, limit in (('lat', 90), ('lon', 180)):
        name = f'{axis}_bounds'
        if name not in ds:
            continue  # v1 point-sampled compatibility package
        bounds = ds[name].values
        if (ds[name].dims != (axis, 'bounds') or bounds.shape != (ds.sizes[axis], 2)
                or not np.isfinite(bounds).all() or np.any(np.abs(bounds) > limit)
                or np.any(bounds[:, 1] <= bounds[:, 0])
                or not np.allclose(bounds[:-1, 1], bounds[1:, 0], rtol=0, atol=1e-8)
                or not np.allclose(bounds.mean(axis=1), ds[axis].values, rtol=0, atol=1e-8)
                or not np.allclose(bounds[:, 1] - bounds[:, 0], np.diff(ds[axis])[0], rtol=0, atol=1e-8)):
            raise ValueError(f'{name} must describe contiguous uniform cells centred on {axis}')
    if 'time' not in ds.coords or ds.time.dims != ('time',):
        raise ValueError('time must be a one-dimensional coordinate')
    times = ds.time.values
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError('time must contain decoded Earth dates')
    dates = times.astype('datetime64[D]')
    if (len(dates) < 3 or np.isnat(dates).any() or np.any(times != dates)
            or np.any(np.diff(dates) != np.timedelta64(1, 'D'))):
        raise ValueError('time must contain continuous, unique daily dates at midnight')
    for name, unit in zip(CHANNELS, UNITS):
        if name not in ds or ds[name].dims != ('time', 'lat', 'lon'):
            raise ValueError(f'{name} must have dimensions (time, lat, lon)')
        if ds[name].attrs.get('units') != unit:
            raise ValueError(f'{name} must use units {unit}')
        values = ds[name].values
        if not np.isfinite(values).all() or np.any(np.abs(values) >= 1e14):
            raise ValueError(f'{name} contains missing or invalid values')
    if 'train_end' not in ds.attrs or 'validation_end' not in ds.attrs:
        raise ValueError('Missing chronological split boundaries')
    expected = split_days(dates, ds.attrs['train_end'], ds.attrs['validation_end'])
    if 'split' not in ds or ds['split'].dims != ('time',) or not np.array_equal(ds['split'], expected):
        raise ValueError('split does not match chronological boundaries')


def validate_earth_3hourly_dataset(ds, *, chunk_steps=8, progress=None):
    """Verify the three-hour data and masks under the shared NetCDF lock."""
    from services.netcdf_read_lock import netcdf_read_lock
    with netcdf_read_lock():
        return _validate_earth_3hourly_dataset(ds, chunk_steps=chunk_steps, progress=progress)


def _validate_earth_3hourly_dataset(ds, *, chunk_steps=8, progress=None):
    """Validate the three-hour product in bounded chunks, returning field statistics.

    Only coordinates and at most eight spatial frames per variable are read at
    once. Missing observations must be NaN and agree with their explicit binary
    validity masks; they are retained rather than filled or rejected outright.
    """
    dataset_id = ds.attrs.get('dataset_id')
    if (ds.attrs.get('planet') != 'Earth' or ds.attrs.get('schema') != THREE_HOURLY_SCHEMA
            or dataset_id not in THREE_HOURLY_DATASET_IDS
            or ds.attrs.get('dataset_version') != THREE_HOURLY_DATASET_VERSION):
        raise ValueError('Expected the Earth MERRA-2 three-hour v1 dataset')
    if (ds.attrs.get('frequency_hours') != 3 or ds.attrs.get('step_unit') != 'hour'
            or ds.attrs.get('step') != 3 or ds.attrs.get('time_zone') != 'UTC'):
        raise ValueError('Three-hour data must declare a three-hour UTC cadence')
    if (ds.attrs.get('temporal_method') != 'mean_of_three_complete_hourly_samples'
            or ds.attrs.get('spatial_method') != 'spherical_area_weighted_overlap'):
        raise ValueError('Unexpected temporal averaging or conservative regrid method')
    if set(ds.sizes) != {'time', 'lat', 'lon', 'bounds'} or ds.sizes.get('bounds') != 2:
        raise ValueError('Three-hour dimensions must be time, lat, lon and two interval bounds')
    height, width = threehourly_grid_shape(dataset_id)
    step = 180.0 / height
    lon_step = 360.0 / width
    for axis, size, centre, edge, axis_step in (('lat', height, -90 + step / 2, -90., step),
                                                ('lon', width, -180 + lon_step / 2, -180., lon_step)):
        expected = centre + np.arange(size) * axis_step
        if (axis not in ds.coords or ds[axis].dims != (axis,)
                or not np.array_equal(ds[axis].values, expected)):
            raise ValueError(f'{axis} must contain the exact global {axis_step} degree cell centres')
        unit = 'degrees_north' if axis == 'lat' else 'degrees_east'
        if ds[axis].attrs.get('units') != unit:
            raise ValueError(f'{axis} must use units {unit}')
        bounds_name = f'{axis}_bounds'
        bounds = np.column_stack((edge + np.arange(size) * axis_step,
                                  edge + (np.arange(size) + 1) * axis_step))
        if (bounds_name not in ds or ds[bounds_name].dims != (axis, 'bounds')
                or not np.array_equal(ds[bounds_name].values, bounds)
                or ds[axis].attrs.get('bounds') != bounds_name
                or ds[bounds_name].attrs.get('units') != unit):
            raise ValueError(f'{bounds_name} must cover contiguous global cells')
    if 'time' not in ds.coords or ds.time.dims != ('time',):
        raise ValueError('time must be a one-dimensional UTC coordinate')
    times = ds.time.values
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError('time must contain decoded Gregorian UTC datetimes')
    if (len(times) < 8 or len(times) % 8 or np.isnat(times).any()
            or np.any(np.diff(times) != np.timedelta64(3, 'h'))):
        raise ValueError('time must be continuous, unique and complete with a fixed three-hour interval')
    first_day = times[0].astype('datetime64[D]')
    last_day = times[-1].astype('datetime64[D]')
    if (times[0] != first_day + np.timedelta64(90, 'm')
            or times[-1] != last_day + np.timedelta64(1350, 'm')
            or first_day < np.datetime64('2020-01-01') or last_day > np.datetime64('2021-12-31')):
        raise ValueError('time must use 01:30 through 22:30 UTC centres within 2020-2021')
    if ds.time.attrs.get('time_zone') != 'UTC' or ds.time.attrs.get('bounds') != 'time_bounds':
        raise ValueError('time must declare UTC and its interval bounds')
    calendar = ds.time.encoding.get('calendar', ds.time.attrs.get('calendar'))
    if calendar not in ('standard', 'gregorian', 'proleptic_gregorian'):
        raise ValueError('time must use a Gregorian calendar')
    if 'time_bounds' not in ds or ds.time_bounds.dims != ('time', 'bounds'):
        raise ValueError('time_bounds must have dimensions (time, bounds)')
    expected_bounds = np.column_stack((times - np.timedelta64(90, 'm'), times + np.timedelta64(90, 'm')))
    if not np.array_equal(ds.time_bounds.values, expected_bounds):
        raise ValueError('time_bounds must contain the complete three-hour interval about each centre')
    if ds.attrs.get('train_end') != '2020-12-31' or ds.attrs.get('validation_end') != '2021-06-30':
        raise ValueError('Unexpected published chronological split boundaries')
    days = times.astype('datetime64[D]')
    expected_split = np.where(days <= np.datetime64('2020-12-31'), 0,
                              np.where(days <= np.datetime64('2021-06-30'), 1, 2)).astype('int8')
    if ('split' not in ds or ds.split.dims != ('time',)
            or not np.array_equal(ds.split.values, expected_split)):
        raise ValueError('split does not match the published chronological boundaries')
    chunk_steps = min(operator.index(chunk_steps), 8)
    if chunk_steps < 1:
        raise ValueError('chunk_steps must be positive')
    statistics = {}
    from services.netcdf_read_lock import netcdf_read_lock
    total = len(times) * height * width
    for name, unit in zip(CHANNELS, UNITS):
        if (name not in ds or ds[name].dims != ('time', 'lat', 'lon')
                or ds[name].shape != (len(times), height, width)
                or str(ds[name].dtype) != 'float32' or ds[name].attrs.get('units') != unit):
            raise ValueError(f'{name} must be float32 (time, lat, lon) in {unit}')
        for packing in ('scale_factor', 'add_offset'):
            if packing in ds[name].attrs or packing in ds[name].encoding:
                raise ValueError(f'{name} must contain unpacked float32 values')
        fill = ds[name].encoding.get('_FillValue', ds[name].attrs.get('_FillValue'))
        if fill is not None and (not np.isscalar(fill) or not np.isnan(fill)):
            raise ValueError(f'{name} must preserve missing values as NaN')
        mask_name = f'{name}_valid_mask'
        if (mask_name not in ds or ds[mask_name].dims != ds[name].dims
                or ds[mask_name].shape != ds[name].shape or str(ds[mask_name].dtype) != 'uint8'
                or ds[name].attrs.get('ancillary_variables') != mask_name):
            raise ValueError(f'{name} must include an explicit uint8 validity mask')
        missing, minimum, maximum = 0, None, None
        last_progress = time.monotonic()
        if progress is not None:
            progress(f'Earth preparation: scientific validation {name} 0/{len(times)} frames')
        for start in range(0, len(times), chunk_steps):
            with netcdf_read_lock():
                values = np.asarray(ds[name].isel(time=slice(start, start + chunk_steps)).values)
                valid = np.asarray(ds[mask_name].isel(time=slice(start, start + chunk_steps)).values)
            stop = min(start + chunk_steps, len(times))
            if progress is not None and (stop == len(times) or time.monotonic() - last_progress >= 5):
                progress(f'Earth preparation: scientific validation {name} {stop}/{len(times)} frames')
                last_progress = time.monotonic()
            if (not np.isin(valid, [0, 1]).all()
                    or not np.array_equal(valid.astype(bool), np.isfinite(values))
                    or not np.isnan(values[valid == 0]).all()):
                raise ValueError(f'{name} NaN values and binary validity mask disagree')
            good = values[valid == 1]
            if np.any(np.abs(good) >= 1e14):
                raise ValueError(f'{name} contains an unmasked invalid fill value')
            missing += int(np.count_nonzero(valid == 0))
            if good.size:
                low, high = float(good.min()), float(good.max())
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
        statistics[name] = {'units': unit, 'dtype': 'float32', 'valid_mask': mask_name,
                            'target_values': total, 'target_missing': missing,
                            'missing_count': missing, 'missing_rate': missing / total,
                            'min': minimum, 'max': maximum}
    return statistics


def load_earth_dataset(path):
    """Keep daily eager loading; validate three-hour data without loading its volume.

    A three-hour dataset retains its lazy file handle until the caller closes it
    (or uses it as a context manager). Its training and overview paths are not
    enabled by this reader.
    """
    from services.netcdf_read_lock import netcdf_read_lock
    with netcdf_read_lock():
        source = xr.open_dataset(Path(path), engine='netcdf4')
    if source.attrs.get('schema') == THREE_HOURLY_SCHEMA:
        try:
            validate_earth_3hourly_dataset(source)
        except Exception:
            source.close()
            raise
        return source
    with source:
        ds = source.load()
    validate_dataset(ds)
    return ds


def _require_daily_metadata(metadata):
    if (metadata.get('schema') == THREE_HOURLY_SCHEMA
            or (metadata.get('time') or {}).get('kind') == 'datetime'):
        raise ValueError('Three-hour data is catalog-only; daily windows and date indexing are unavailable')


def canonical_input_channels(selected_channels=None):
    """Return the canonical model input order for a channel selection.

    ``TO3`` is always the first input and the only target. Auxiliary variables
    follow ``U10M, V10M, T2M, SWGDN`` regardless of request order.
    """
    if selected_channels is None:
        return [TARGET_CHANNEL, *OPTIONAL_CHANNELS]
    requested = set()
    target_seen = False
    for item in selected_channels:
        name = str(item).strip().upper()
        if name == TARGET_CHANNEL:
            # TO3 is mandatory and already first; naming it is accepted, naming
            # it twice is still a duplicate.
            if target_seen:
                raise ValueError(f'Duplicate Earth input channel: {item}')
            target_seen = True
            continue
        if name not in OPTIONAL_CHANNELS:
            raise ValueError(f'Unsupported Earth input channel: {item}')
        if name in requested:
            raise ValueError(f'Duplicate Earth input channel: {item}')
        requested.add(name)
    return [TARGET_CHANNEL] + [name for name in OPTIONAL_CHANNELS if name in requested]


def input_units(channel_order):
    """Return the physical unit of each channel in model input order."""
    return [UNITS[CHANNELS.index(name)] for name in channel_order]


def stack_release_cube(release):
    """Stack the five published fields into ``[time, channel, lat, lon]`` float32."""
    _require_daily_metadata(release.metadata)
    return np.stack(
        [np.asarray(release.fields[name], dtype='float32') for name in CHANNELS], axis=1
    )


def fit_normalization(cube, input_channels, *, fit_dates=None):
    """Fit per-channel training statistics and return the serialisable contract.

    ``cube`` is ``[time, channel, lat, lon]`` for the **training dates only**, so
    validation and test values can never influence the statistics. Accumulation
    is float64 and the stored values are float32, matching the published manifest
    statistics. A near-constant channel keeps a scale of ``1.0`` and is flagged
    instead of being divided into NaN.
    """
    channels = list(input_channels)
    indices = [CHANNELS.index(name) for name in channels]
    array = np.asarray(cube, dtype='float64')
    if array.shape[1] != len(CHANNELS):
        raise ValueError('Normalization must be fitted on all five published channels')
    selected = array[:, indices]
    if not np.isfinite(selected).all():
        raise ValueError('Cannot fit normalization on non-finite values')
    mean = selected.mean(axis=(0, 2, 3))
    std = selected.std(axis=(0, 2, 3))
    constant = std < MINIMUM_SCALE
    scale = np.where(constant, 1.0, std)
    dates = [str(value) for value in (fit_dates if fit_dates is not None else [])]
    return {
        'method': NORMALIZATION_METHOD,
        'fit_split': 'train',
        'fit_date_start': dates[0] if dates else None,
        'fit_date_end': dates[-1] if dates else None,
        'ddof': NORMALIZATION_DDOF,
        'epsilon': NORMALIZATION_EPSILON,
        'channel_order': channels,
        'mean': [float(value) for value in mean.astype('float32')],
        'scale': [float(value) for value in scale.astype('float32')],
        'constant_channel_mask': [bool(value) for value in constant],
        'target_channel_index': TARGET_CHANNEL_INDEX,
    }


def validate_normalization(normalization, input_channels):
    """Return ``(mean, scale)`` float64 arrays for a saved normalization block.

    The saved channel order must match the model input order exactly: reusing a
    statistic fitted for different channels would silently rescale the inputs.
    """
    if not isinstance(normalization, dict):
        raise ValueError('Normalization must be a mapping')
    if normalization.get('method') != NORMALIZATION_METHOD:
        raise ValueError('Unsupported normalization method')
    if normalization.get('fit_split') != 'train':
        raise ValueError('Normalization must have been fitted on the training split')
    if list(normalization.get('channel_order') or []) != list(input_channels):
        raise ValueError('Normalization channel order does not match the model inputs')
    mean = np.asarray(normalization.get('mean'), dtype='float64')
    scale = np.asarray(normalization.get('scale'), dtype='float64')
    if mean.shape != (len(input_channels),) or scale.shape != (len(input_channels),):
        raise ValueError('Normalization vectors must have one value per input channel')
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Normalization contains non-finite values')
    if np.any(scale <= 0):
        raise ValueError('Normalization scale must be positive')
    return mean, scale


def normalize_cube(cube, input_channels, mean, scale):
    """Standardise a ``[time, channel, lat, lon]`` cube with saved statistics."""
    indices = [CHANNELS.index(name) for name in input_channels]
    selected = np.asarray(cube, dtype='float32')[:, indices, :, :]
    scaled = (selected.astype('float64') - mean[None, :, None, None]) / scale[None, :, None, None]
    return scaled.astype('float32')


class EarthOzoneWindows:
    """NumPy map-style dataset usable directly with torch.utils.data.DataLoader.

    Inputs: ``[window, C, lat, lon]``; targets: ``[horizon, 1, lat, lon]``.
    Both use per-channel normalization fitted exclusively on training dates.
    Windows are formed within each split, never across split boundaries.
    """

    def __init__(self, path, *, split='train', window=7, horizon=3):
        if split not in SPLITS:
            raise ValueError(f'Unknown split: {split}')
        self.window, self.horizon = operator.index(window), operator.index(horizon)
        if self.window < 1 or self.horizon < 1:
            raise ValueError('window and horizon must be positive integers')
        with load_earth_dataset(path) as ds:
            if ds.attrs.get('schema') != SCHEMA:
                raise ValueError('EarthOzoneWindows requires a daily release')
            cube = np.stack([ds[name].values for name in CHANNELS], axis=1).astype('float32')
            split_ids = ds['split'].values
            all_dates = ds.time.values.astype('datetime64[D]')
            training = split_ids == SPLITS['train']
            self.normalization = fit_normalization(
                cube[training], list(CHANNELS), fit_dates=all_dates[training]
            )
            mean, scale = validate_normalization(self.normalization, list(CHANNELS))
            self.input_channels = list(CHANNELS)
            self.input_units = input_units(self.input_channels)
            self.dates = all_dates[split_ids == SPLITS[split]]
            self.lat, self.lon = ds.lat.values.copy(), ds.lon.values.copy()
            self.data = normalize_cube(cube[split_ids == SPLITS[split]], self.input_channels, mean, scale)
        self.target_channel_index = TARGET_CHANNEL_INDEX
        self._set_compat_statistics()
        self._set_length()

    @classmethod
    def from_release(
        cls,
        release,
        *,
        split='train',
        window=7,
        horizon=3,
        selected_channels=None,
        normalization=None,
        require_split_coverage=True,
    ):
        """Build windows from a verified release with optional channel selection.

        Only the training split may fit statistics (``normalization=None``); the
        validation, test and prediction paths must reuse the saved block.
        ``require_split_coverage=False`` selects the whole release instead of one
        published split, which the prediction path uses for its input window.
        """
        if split not in SPLITS:
            raise ValueError(f'Unknown split: {split}')
        _require_daily_metadata(release.metadata)
        window = operator.index(window)
        horizon = operator.index(horizon)
        if window < 1 or horizon < 1:
            raise ValueError('window and horizon must be positive integers')
        channels = canonical_input_channels(selected_channels)
        if normalization is None and split != 'train':
            raise ValueError('Only the training split may fit normalization statistics')

        dataset = cls.__new__(cls)
        dataset.window, dataset.horizon = window, horizon
        dates = np.asarray(release.dates, dtype='datetime64[D]')
        dataset.lat = np.array(release.latitude, dtype='float64', copy=True)
        dataset.lon = np.array(release.longitude, dtype='float64', copy=True)
        cube = stack_release_cube(release)
        split_ids = release_split_codes(dates, release.metadata)

        if normalization is None:
            training = split_ids == SPLITS['train']
            dataset.normalization = fit_normalization(
                cube[training], channels, fit_dates=dates[training]
            )
        else:
            dataset.normalization = normalization
        mean, scale = validate_normalization(dataset.normalization, channels)

        selected = (split_ids == SPLITS[split]) if require_split_coverage else np.ones(len(dates), dtype=bool)
        dataset.dates = dates[selected]
        dataset.data = normalize_cube(cube[selected], channels, mean, scale)
        dataset.input_channels = channels
        dataset.input_units = input_units(channels)
        dataset.target_channel_index = TARGET_CHANNEL_INDEX
        dataset._set_compat_statistics()
        dataset._set_length()
        return dataset

    def _set_compat_statistics(self):
        """Expose the historical ``mean``/``std`` attributes for existing callers."""
        self.mean = np.asarray(self.normalization['mean'], dtype='float32')
        self.std = np.asarray(self.normalization['scale'], dtype='float32')

    def _set_length(self):
        self.length = len(self.dates) - self.window - self.horizon + 1
        if self.length < 1:
            raise ValueError('Not enough days in the selected split for window + horizon')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0 or index >= self.length:
            raise IndexError(index)
        forecast = index + self.window
        return (self.data[index:forecast].copy(),
                self.data[forecast:forecast + self.horizon, :1].copy())

    def denormalize_ozone(self, values):
        """Convert normalized target values back to Dobson units.

        Accepts either a NumPy array or a Tensor and returns the same kind; a
        tensor is never moved between devices implicitly.
        """
        mean = float(self.normalization['mean'][self.target_channel_index])
        scale = float(self.normalization['scale'][self.target_channel_index])
        if hasattr(values, 'dim') and hasattr(values, 'to'):
            return values * scale + mean
        return np.asarray(values) * scale + mean


def threehour_release_split_codes(dates, metadata):
    """Map full UTC timestamps to the release's contiguous published date blocks.

    Date-only split bounds describe complete UTC days. No timestamp is rounded
    for indexing, and an uncovered, overlapping or interleaved block is rejected.
    Empty published splits are supported for inspection of a smoke package.
    """
    if metadata.get('schema') != THREE_HOURLY_SCHEMA:
        raise ValueError('Earth three-hour windows require a three-hour release')
    times = np.asarray(dates, dtype='datetime64[ns]')
    if (times.ndim != 1 or not len(times) or np.isnat(times).any()
            or np.any(np.diff(times) != np.timedelta64(3, 'h'))
            or np.any((times - times.astype('datetime64[D]')) % np.timedelta64(3, 'h')
                      != np.timedelta64(90, 'm'))):
        raise ValueError('Three-hour timestamps must be continuous UTC interval centres')
    if (metadata.get('frequency_hours') != 3 or metadata.get('step_unit') != 'hour'
            or metadata.get('step') != 3 or (metadata.get('time') or {}).get('time_zone') != 'UTC'):
        raise ValueError('Three-hour release must use the fixed three-hour UTC cadence')
    codes = np.full(len(times), -1, dtype='int8')
    splits = metadata.get('splits') or {}
    for name, code in SPLITS.items():
        entry = splits.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f'Verified release has no published {name} split')
        if entry.get('start') is None and entry.get('end') is None:
            if entry.get('steps') != 0:
                raise ValueError(f'Empty {name} split has nonzero step count')
            continue
        try:
            start = np.datetime64(entry['start'], 'D')
            stop = np.datetime64(entry['end'], 'D') + np.timedelta64(1, 'D')
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f'Invalid published {name} split bounds') from exc
        if np.isnat(start) or np.isnat(stop) or start >= stop:
            raise ValueError(f'Invalid published {name} split bounds')
        selected = (times >= start) & (times < stop)
        if np.any(codes[selected] >= 0):
            raise ValueError('Published three-hour splits overlap')
        if type(entry.get('steps')) is not int or int(selected.sum()) != entry['steps']:
            raise ValueError(f'Published {name} split step count disagrees with the time axis')
        codes[selected] = code
    if np.any(codes < 0) or np.any(np.diff(codes.astype('int16')) < 0):
        raise ValueError('Published three-hour splits must cover one chronological time axis')
    return codes


def _threehour_path(release):
    """Check the server snapshot before each bounded read, without recomputing SHA."""
    from services.earth_dataset_metadata import package_signature
    path = getattr(release, 'data_path', None)
    if path is None:
        raise ValueError('Verified three-hour release has no server data path')
    path = Path(path)
    if release.signature and package_signature(path.parent, data_file_name=path.name) != release.signature:
        raise ValueError('Earth three-hour release changed since verification')
    return path


def cell_area_weights_from_lat_bounds(lat_bounds, longitude_count: int) -> np.ndarray:
    """Return globally normalized spherical cell weights from latitude bounds."""
    bounds = np.asarray(lat_bounds, dtype='float64')
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0])
            or np.any(bounds < -90.) or np.any(bounds > 90.)
            or type(longitude_count) is not int or longitude_count < 1):
        raise ValueError('Latitude bounds and longitude count are invalid for cell-area weights')
    row_area = (np.sin(np.deg2rad(bounds[:, 1]))
                - np.sin(np.deg2rad(bounds[:, 0])))
    if not np.isfinite(row_area).all() or np.any(row_area <= 0):
        raise ValueError('Latitude bounds produce invalid spherical cell areas')
    weights = np.broadcast_to(
        (row_area / (row_area.sum() * longitude_count))[:, None],
        (len(row_area), longitude_count),
    ).copy()
    weights /= weights.sum(dtype='float64')
    return weights


def threehour_global_cell_area_weights(release) -> np.ndarray:
    """Read exact cell bounds from a verified three-hour global release."""
    metadata = getattr(release, 'metadata', {}) or {}
    dataset_id = metadata.get('dataset_id') or THREE_HOURLY_DATASET_ID
    height, width = threehourly_grid_shape(dataset_id)
    step = 180.0 / height
    if (metadata.get('schema') != THREE_HOURLY_SCHEMA
            or metadata.get('dataset_version') not in (None, THREE_HOURLY_DATASET_VERSION)
            or metadata.get('grid_shape') != [height, width]
            or not np.array_equal(np.asarray(release.latitude), -90 + (np.arange(height) + .5) * step)
            or not np.array_equal(np.asarray(release.longitude), -180 + (np.arange(width) + .5) * step)):
        raise ValueError('Area weights require a verified three-hour global grid')
    path = _threehour_path(release)
    from services.netcdf_read_lock import netcdf_read_lock
    with netcdf_read_lock(), xr.open_dataset(path, engine='netcdf4', mask_and_scale=False) as ds:
        bounds = np.asarray(ds['lat_bounds'].values, dtype='float64')
        expected = np.column_stack((-90. + np.arange(height) * step,
                                    -90. + (np.arange(height) + 1) * step))
        if (ds['lat_bounds'].dims != ('lat', 'bounds')
                or bounds.shape != expected.shape or not np.array_equal(bounds, expected)):
            raise ValueError('Verified release latitude cell bounds changed')
    _threehour_path(release)
    return cell_area_weights_from_lat_bounds(bounds, width)


def _threehour_values(ds, channel, start, stop, *, lat_slice=slice(None), lon_slice=slice(None)):
    """Read one selected field and mask; never silently fill missing observations."""
    selection = {'time': slice(start, stop), 'lat': lat_slice, 'lon': lon_slice}
    values = np.asarray(ds[channel].isel(**selection).values, dtype='float32')
    valid = np.asarray(ds[f'{channel}_valid_mask'].isel(**selection).values)
    if (values.shape != valid.shape or not np.all(valid == 1)
            or not np.isfinite(values).all() or np.any(np.abs(values) >= 1e14)):
        raise ValueError(f'Earth three-hour training data contains missing or non-finite values in {channel}')
    return values


def _utc_timestamp(value):
    return np.datetime_as_string(np.datetime64(value, 's'), unit='s') + 'Z'


def _normalization_identity(release, channels, *, dates, start, stop):
    from services.earth_preparation_cache import signature_json
    return {
        'schema': NORMALIZATION_CACHE_VERSION,
        'dataset_id': release.metadata.get('dataset_id', THREE_HOURLY_DATASET_ID),
        'dataset_version': release.metadata.get('dataset_version', THREE_HOURLY_DATASET_VERSION),
        'dataset_fingerprint': release.metadata.get('dataset_fingerprint'),
        'package_signature': signature_json(release.signature),
        'channel_order': list(channels), 'method': NORMALIZATION_METHOD,
        'ddof': NORMALIZATION_DDOF, 'epsilon': NORMALIZATION_EPSILON,
        'fit_split': 'train', 'fit_time_start': _utc_timestamp(dates[start]),
        'fit_time_end': _utc_timestamp(dates[stop - 1]), 'fit_step_count': int(stop - start),
    }


def _normalization_cache_paths(cache_root, identity):
    from services.earth_preparation_cache import digest_json
    root = Path(cache_root).expanduser().resolve() / 'normalization'
    key = digest_json(identity)
    return root / f'{key}.json', root / f'{key}.lock'


def _load_normalization_cache(path, identity):
    from services.earth_preparation_cache import _read_json
    payload = _read_json(path)
    if payload is None or payload.get('schema') != NORMALIZATION_CACHE_VERSION:
        return None
    if payload.get('identity') != identity or not isinstance(payload.get('normalization'), dict):
        return None
    normalization = payload['normalization']
    try:
        validate_normalization(normalization, identity['channel_order'])
        from services.earth_preparation_cache import digest_json
        if payload.get('normalization_sha256') != digest_json(normalization):
            return None
        if any(normalization.get(key) != identity[key] for key in (
                'fit_split', 'fit_time_start', 'fit_time_end', 'fit_step_count',
                'method', 'ddof', 'epsilon')):
            return None
    except (TypeError, ValueError):
        return None
    return normalization


def _normalization_for_task(normalization, task_split):
    result = copy.deepcopy(normalization)
    result.pop('task_split', None)
    if task_split is not None:
        result['task_split'] = copy.deepcopy(task_split)
    return result


def _fit_threehour_normalization_uncached(release, channels=None, *, task_split=None, progress=None):
    """Fit stable population statistics on training frames in chunks of at most 8.

    This scans one channel at a time and merges float64 chunk moments. It never
    materialises the five-channel volume or any sliding-window expansion.
    Selected-channel missing observations are rejected; valid numeric zeros are
    retained. Validation/test frames cannot affect the fit.
    """
    from services.netcdf_read_lock import netcdf_read_lock
    selected_channels = canonical_input_channels(channels)
    dates = np.asarray(release.dates, dtype='datetime64[ns]')
    from services.earth_task_split import task_split_codes
    codes = task_split_codes(dates, release.metadata, task_split)
    indices = np.flatnonzero(codes == SPLITS['train'])
    if not len(indices):
        raise ValueError('No training timestamps are available to fit normalization')
    start, stop = int(indices[0]), int(indices[-1]) + 1
    path = _threehour_path(release)
    means, scales, constants = [], [], []
    with netcdf_read_lock(), xr.open_dataset(path, engine='netcdf4', mask_and_scale=False) as ds:
        if not np.array_equal(ds.time.values, dates):
            raise ValueError('Earth three-hour timestamps changed since verification')
        for channel in selected_channels:
            last_progress = time.monotonic()
            if progress is not None:
                progress(f'Earth preparation: normalization {channel} 0/{stop - start} train frames')
            count, mean, moment = 0, 0.0, 0.0
            for offset in range(start, stop, 8):
                chunk_stop = min(offset + 8, stop)
                values = _threehour_values(ds, channel, offset, chunk_stop).astype('float64')
                chunk_count = values.size
                chunk_mean = float(values.mean())
                chunk_moment = float(values.var(ddof=0)) * chunk_count
                total_count = count + chunk_count
                delta = chunk_mean - mean
                moment += chunk_moment + delta * delta * count * chunk_count / total_count
                mean += delta * chunk_count / total_count
                count = total_count
                if progress is not None and (chunk_stop == stop or time.monotonic() - last_progress >= 5):
                    progress(f'Earth preparation: normalization {channel} {chunk_stop - start}/{stop - start} train frames')
                    last_progress = time.monotonic()
            std = float(np.sqrt(max(moment / count, 0.0)))
            constant = std < MINIMUM_SCALE
            means.append(float(np.float32(mean)))
            scales.append(float(np.float32(1.0 if constant else std)))
            constants.append(bool(constant))
    _threehour_path(release)
    return {
        **({'task_split': task_split} if task_split is not None else {}),
        'method': NORMALIZATION_METHOD, 'fit_split': 'train',
        'fit_date_start': _utc_timestamp(dates[start]), 'fit_date_end': _utc_timestamp(dates[stop - 1]),
        'fit_time_start': _utc_timestamp(dates[start]), 'fit_time_end': _utc_timestamp(dates[stop - 1]),
        'fit_step_count': stop - start,
        'time_zone': 'UTC', 'frequency_hours': 3, 'step_unit': 'hour', 'step': 3,
        'timestamp_rule': 'interval_center',
        'ddof': NORMALIZATION_DDOF, 'epsilon': NORMALIZATION_EPSILON,
        'channel_order': selected_channels, 'mean': means, 'scale': scales,
        'constant_channel_mask': constants, 'target_channel_index': TARGET_CHANNEL_INDEX,
    }


def fit_threehour_normalization(release, channels=None, *, task_split=None, progress=None,
                                cache_root=None):
    """Return train-only statistics, reusing a verified content-addressed result."""
    started = time.perf_counter()
    _threehour_path(release)
    selected_channels = canonical_input_channels(channels)
    dates = np.asarray(release.dates, dtype='datetime64[ns]')
    from services.earth_task_split import task_split_codes
    codes = task_split_codes(dates, release.metadata, task_split)
    indices = np.flatnonzero(codes == SPLITS['train'])
    if not len(indices):
        raise ValueError('No training timestamps are available to fit normalization')
    start, stop = int(indices[0]), int(indices[-1]) + 1
    identity = _normalization_identity(
        release, selected_channels, dates=dates, start=start, stop=stop,
    )
    if cache_root is None:
        return _fit_threehour_normalization_uncached(
            release, selected_channels, task_split=task_split, progress=progress,
        )
    cache_path, lock_path = _normalization_cache_paths(cache_root, identity)
    cached = _load_normalization_cache(cache_path, identity)
    if cached is not None:
        _threehour_path(release)
        if progress is not None:
            progress(f'Earth preparation: normalization cache hit in {time.perf_counter() - started:.3f}s')
        return _normalization_for_task(cached, task_split)
    from services.earth_preparation_cache import atomic_write_json, preparation_lock
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with preparation_lock(lock_path, progress=progress):
        _threehour_path(release)
        cached = _load_normalization_cache(cache_path, identity)
        if cached is not None:
            _threehour_path(release)
            if progress is not None:
                progress('Earth preparation: normalization cache hit (waited for builder)')
            return _normalization_for_task(cached, task_split)
        if progress is not None:
            progress('Earth preparation: normalization cache miss; fitting train-only statistics')
        result = _fit_threehour_normalization_uncached(
            release, selected_channels, task_split=task_split, progress=progress,
        )
        stored = _normalization_for_task(result, None)
        from services.earth_preparation_cache import digest_json
        atomic_write_json(cache_path, {'schema': NORMALIZATION_CACHE_VERSION,
                                       'identity': identity, 'normalization': stored,
                                       'normalization_sha256': digest_json(stored)})
        _threehour_path(release)
        if progress is not None:
            progress(f'Earth preparation: normalization generated in {time.perf_counter() - started:.3f}s')
        return result


def _build_threehour_training_cache_uncached(release, channels, normalization, cache_root, *, full_grid=False,
                                             progress=None, folder_override, cache_identity):
    """Stream normalized fields to a unique disk map, without expanding windows.

    Compressed global chunks are decoded once here. Full-grid models use time-major
    fields; tile models retain spatially contiguous tile storage. Only <=8 frames
    of one channel are read.
    Interrupted caches are retained; source packages are always read-only.
    """
    from services.netcdf_read_lock import netcdf_read_lock
    order = canonical_input_channels(channels)
    mean, scale = validate_normalization(normalization, order)
    source_path = _threehour_path(release)
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(folder_override)
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'normalized.npy'
    shape = (len(release.dates), len(order), len(release.latitude), len(release.longitude))
    layout = THREE_HOURLY_FULL_GRID_CACHE_LAYOUT if full_grid else THREE_HOURLY_TILE_CACHE_LAYOUT
    storage_shape = shape if full_grid else (100, shape[0], shape[1], *THREE_HOURLY_TILE_SHAPE)
    mapped = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=storage_shape)
    last_progress = time.monotonic()
    if progress is not None:
        progress(f'Earth preparation: {layout} cache 0/{shape[0]} frames')
    try:
        with netcdf_read_lock(), xr.open_dataset(source_path, engine='netcdf4', mask_and_scale=False) as ds:
            if not np.array_equal(ds.time.values, release.dates):
                raise ValueError('Earth three-hour timestamps changed since verification')
            for start in range(0, shape[0], 8):
                stop = min(start + 8, shape[0])
                for index, channel in enumerate(order):
                    values = _threehour_values(ds, channel, start, stop)
                    values = ((values.astype('float64') - mean[index]) / scale[index]).astype('float32')
                    if full_grid:
                        mapped[start:stop, index] = values
                    else:
                        if tuple(values.shape[-2:]) != THREE_HOURLY_GRID_SHAPE:
                            raise ValueError('Spatial tile cache is only available for the native 240x480 release')
                        tiles = values.reshape(stop - start, 10, 24, 10, 48)
                        tiles = tiles.transpose(1, 3, 0, 2, 4).reshape(100, stop - start, 24, 48)
                        mapped[:, start:stop, index] = tiles
                if progress is not None and (stop == shape[0] or time.monotonic() - last_progress >= 5):
                    progress(f'Earth preparation: {layout} cache {stop}/{shape[0]} frames')
                    last_progress = time.monotonic()
        mapped.flush()
    finally:
        mapped._mmap.close()
    _threehour_path(release)
    metadata = {
        'schema': THREE_HOURLY_CACHE_SCHEMA, 'cache_version': 1,
        'cache_identity_version': 2,
        'cache_status': 'ready',
        'dataset_id': release.metadata.get('dataset_id', THREE_HOURLY_DATASET_ID),
        'dataset_version': release.metadata.get('dataset_version', THREE_HOURLY_DATASET_VERSION),
        'dataset_fingerprint': release.metadata.get('dataset_fingerprint'),
        'package_signature': cache_identity['package_signature'],
        'shape': list(shape), 'input_channel_order': order,
        'layout': layout, 'storage_shape': list(storage_shape),
        **({'spatial_tile_shape': list(THREE_HOURLY_TILE_SHAPE)} if not full_grid else {}),
        'normalization': normalization,
    }
    from services.earth_preparation_cache import (atomic_write_json, cache_file_identity,
        sha256_file, seal_payload, digest_json)
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    metadata['array_sha256'] = sha256_file(path, progress=progress)
    metadata['array_identity'] = cache_file_identity(path)
    metadata['identity'] = cache_identity
    metadata['cache_key'] = digest_json(cache_identity)
    atomic_write_json(folder / 'metadata.json', seal_payload(metadata))
    return path


def _training_cache_identity(release, order, normalization, *, full_grid):
    from services.earth_preparation_cache import signature_json
    shape = (len(release.dates), len(order), len(release.latitude), len(release.longitude))
    layout = THREE_HOURLY_FULL_GRID_CACHE_LAYOUT if full_grid else THREE_HOURLY_TILE_CACHE_LAYOUT
    storage_shape = shape if full_grid else (100, shape[0], shape[1], *THREE_HOURLY_TILE_SHAPE)
    return {
        'schema': THREE_HOURLY_CACHE_SCHEMA, 'cache_version': 1,
        'cache_identity_version': 2, 'dataset_id': release.metadata.get('dataset_id', THREE_HOURLY_DATASET_ID),
        'dataset_version': release.metadata.get('dataset_version', THREE_HOURLY_DATASET_VERSION),
        'dataset_fingerprint': release.metadata.get('dataset_fingerprint'),
        'package_signature': signature_json(release.signature),
        'input_channel_order': list(order), 'layout': layout,
        'shape': list(shape), 'storage_shape': list(storage_shape),
        'normalization_method_version': NORMALIZATION_CACHE_VERSION,
        'normalization': _normalization_content(normalization),
    }


def _normalization_content(normalization):
    """Only fields which define normalized bytes; partition windows are separate."""
    return {key: copy.deepcopy(value) for key, value in normalization.items()
            if key not in {'task_split', 'normalization_cache_version', 'dataset_id',
                           'dataset_version', 'dataset_fingerprint', 'package_signature'}}


def _training_cache_candidate(path, identity, *, release=None, progress=None, verify_payload=True):
    from services.earth_preparation_cache import (_read_json, verify_array_file, valid_seal)
    path = Path(path)
    if path.parent.name.startswith('.'):
        return None  # An unpublished/interrupted staging directory.
    metadata = _read_json(path.parent / 'metadata.json')
    if metadata is None:
        return None
    for key in ('schema', 'cache_version', 'dataset_id', 'dataset_version', 'dataset_fingerprint',
                'input_channel_order', 'layout', 'shape', 'storage_shape'):
        if metadata.get(key) != identity.get(key):
            if key == 'dataset_version' and metadata.get('dataset_version') is None:
                continue
            return None
    if (not isinstance(metadata.get('normalization'), dict)
            or _normalization_content(metadata['normalization']) != identity['normalization']):
        return None
    if metadata.get('cache_identity_version') not in (None, 2):
        return None
    if metadata.get('cache_identity_version') == 2:
        from services.earth_preparation_cache import digest_json
        if (metadata.get('cache_status') != 'ready' or metadata.get('identity') != identity
                or metadata.get('cache_key') != digest_json(identity) or not valid_seal(metadata)):
            return None
    try:
        mapped = np.load(path, mmap_mode='r', allow_pickle=False)
        if (mapped.shape != tuple(identity['storage_shape']) or mapped.dtype != np.dtype('float32')
                or path.stat().st_size != mapped.offset + mapped.nbytes):
            mapped._mmap.close()
            return None
        mapped._mmap.close()
        if metadata.get('cache_identity_version') is None:
            if release is None:
                return None
            return _certify_existing_training_cache(path, metadata, identity, release, progress=progress)
        if verify_payload and not verify_array_file(path, metadata.get('array_identity'), metadata.get('array_sha256'), progress=progress):
            return None
        return path
    except (OSError, ValueError, TypeError):
        return None


def _certify_existing_training_cache(path, metadata, identity, release, *, progress):
    """Certify an existing current-layout v1 volume against source bytes once.

    Original caches have no payload digest or completion identity. They cannot
    be trusted by shape alone, so compare every normalized float32 chunk before
    adding an atomic integrity certificate; the large array stays in place.
    """
    from services.earth_preparation_cache import (preparation_lock, _read_json,
        cache_file_identity, sha256_file, atomic_write_json, seal_payload, digest_json)
    with preparation_lock(path.parent / 'certification.lock', progress=progress):
        current = _read_json(path.parent / 'metadata.json')
        if current is not None and current.get('cache_identity_version') == 2:
            return _training_cache_candidate(path, identity, progress=progress)
        before = cache_file_identity(path)
        _threehour_path(release)
        started = time.perf_counter()
        if progress is not None:
            progress('Earth preparation: existing cache needs full source certification')
        from services.netcdf_read_lock import netcdf_read_lock
        mean, scale = validate_normalization(metadata['normalization'], identity['input_channel_order'])
        mapped = np.load(path, mmap_mode='r', allow_pickle=False)
        last = time.monotonic()
        try:
            with netcdf_read_lock(), xr.open_dataset(_threehour_path(release), engine='netcdf4', mask_and_scale=False) as ds:
                if not np.array_equal(ds.time.values, release.dates):
                    raise ValueError('Earth three-hour timestamps changed since verification')
                for start in range(0, len(release.dates), 8):
                    stop = min(start + 8, len(release.dates))
                    for index, channel in enumerate(identity['input_channel_order']):
                        values = _threehour_values(ds, channel, start, stop)
                        values = ((values.astype('float64') - mean[index]) / scale[index]).astype('float32')
                        if identity['layout'] == THREE_HOURLY_TILE_CACHE_LAYOUT:
                            values = values.reshape(stop - start, 10, 24, 10, 48).transpose(1, 3, 0, 2, 4).reshape(100, stop - start, 24, 48)
                            actual = mapped[:, start:stop, index]
                        else:
                            actual = mapped[start:stop, index]
                        if not np.array_equal(actual, values):
                            return None
                    if progress is not None and (stop == len(release.dates) or time.monotonic() - last >= 5):
                        progress(f'Earth preparation: existing cache certification {stop}/{len(release.dates)} frames')
                        last = time.monotonic()
        finally:
            mapped._mmap.close()
        _threehour_path(release)
        if cache_file_identity(path) != before:
            return None
        checksum = sha256_file(path, progress=progress)
        if cache_file_identity(path) != before:
            return None
        updated = dict(metadata, cache_identity_version=2, cache_status='ready',
                       dataset_version=THREE_HOURLY_DATASET_VERSION,
                       package_signature=identity['package_signature'], identity=identity,
                       cache_key=digest_json(identity), array_identity=before, array_sha256=checksum)
        atomic_write_json(path.parent / 'metadata.json', seal_payload(updated))
        if progress is not None:
            progress(f'Earth preparation: existing cache certified in {time.perf_counter() - started:.3f}s')
        return path


def build_threehour_training_cache(release, channels, normalization, cache_root, *, full_grid=False, progress=None):
    """Find or atomically publish the normalized volume for this data identity."""
    started = time.perf_counter()
    _threehour_path(release)
    order = canonical_input_channels(channels)
    validate_normalization(normalization, order)
    identity = _training_cache_identity(release, order, normalization, full_grid=full_grid)
    from services.earth_preparation_cache import digest_json, preparation_lock
    root = Path(cache_root).expanduser().resolve()
    final = root / 'earth_cache_v2' / digest_json(identity)
    path = final / 'normalized.npy'
    found = _training_cache_candidate(path, identity, release=release, progress=progress)
    if found is None and root.exists():
        for metadata_path in root.glob('**/metadata.json'):
            candidate = _training_cache_candidate(metadata_path.parent / 'normalized.npy', identity, release=release, progress=progress)
            if candidate is not None:
                found = candidate
                break
    if found is not None:
        _threehour_path(release)
        if progress is not None:
            progress(f"Earth preparation: {identity['layout']} cache hit in {time.perf_counter() - started:.3f}s")
        return found
    final.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root / 'locks' / f"{digest_json(identity)}.lock"
    with preparation_lock(lock_path, progress=progress):
        _threehour_path(release)
        found = _training_cache_candidate(path, identity, release=release, progress=progress)
        if found is None:
            for metadata_path in root.glob('**/metadata.json'):
                candidate = _training_cache_candidate(metadata_path.parent / 'normalized.npy', identity,
                                                      release=release, progress=progress)
                if candidate is not None:
                    found = candidate
                    break
        if found is not None:
            _threehour_path(release)
            if progress is not None:
                progress(f"Earth preparation: {identity['layout']} cache hit (waited for builder)")
            return found
        temporary = final.parent / f'.{final.name}.building-{os.getpid()}-{time.time_ns()}'
        if progress is not None:
            progress(f"Earth preparation: {identity['layout']} cache miss; generating normalized volume")
        built = _build_threehour_training_cache_uncached(
            release, order, normalization, root, full_grid=full_grid, progress=progress,
            folder_override=temporary, cache_identity=identity,
        )
        if final.exists():
            # Retain rejected published files. Never overwrite a volume which
            # another process could still have mapped.
            final = final.with_name(f'{final.name}-generation-{os.getpid()}-{time.time_ns()}')
        os.replace(temporary, final)
        _threehour_path(release)
        if progress is not None:
            progress(f"Earth preparation: {identity['layout']} cache generated in {time.perf_counter() - started:.3f}s")
        return final / built.name


class EarthThreeHourlyWindows:
    """Disk-backed map-style 56→24 dataset with optional spatial tile reads.

    ``__getitem__`` returns float32 ``[56,C,240,480]`` inputs and
    ``[24,1,240,480]`` targets. ``read_window`` accepts latitude/longitude slices
    to bound spatial memory further. Only lightweight coordinates/indices and
    normalization are retained between reads; there is no ``data`` cube and no
    full-window expansion. Optional normalized disk maps serve tile reads
    without repeated decompression; source reads use the shared NetCDF lock.
    """

    @classmethod
    def from_release(cls, release, *, split='train', window=56, horizon=24,
                     selected_channels=None, normalization=None, task_split=None):
        if split not in SPLITS:
            raise ValueError(f'Unknown split: {split}')
        from services.earth_training_contract import earth_training_profile
        dataset_id = str((release.metadata or {}).get('dataset_id') or THREE_HOURLY_DATASET_ID)
        profile = earth_training_profile(dataset_id, {'window': window, 'horizon': horizon})
        window, horizon = profile['window'], profile['horizon']
        channels = canonical_input_channels(selected_channels)
        dates = np.asarray(release.dates, dtype='datetime64[ns]')
        from services.earth_task_split import task_split_codes, validate_earth_task_split
        if task_split is not None:
            validate_earth_task_split(task_split, dates, window, horizon,
                                      task_split.get('ratios') if isinstance(task_split, dict) else None)
        codes = task_split_codes(dates, release.metadata, task_split)
        lat = np.asarray(release.latitude, dtype='float64')
        lon = np.asarray(release.longitude, dtype='float64')
        expected_shape = tuple(profile['grid_shape'])
        if (lat.shape != (expected_shape[0],) or lon.shape != (expected_shape[1],)):
            raise ValueError(f'Earth three-hour training requires the global {expected_shape[0]}x{expected_shape[1]} grid')
        indices = np.flatnonzero(codes == SPLITS[split])
        length = len(indices) - window - horizon + 1
        if length < 1:
            raise ValueError('Not enough three-hour steps in the selected split for window + horizon')
        if normalization is None:
            if split != 'train':
                raise ValueError('Only the training split may fit normalization statistics')
            normalization = fit_threehour_normalization(release, channels, task_split=task_split)
        mean, scale = validate_normalization(normalization, channels)
        train_dates = dates[codes == SPLITS['train']]
        if (normalization.get('task_split') != task_split
                or not len(train_dates) or normalization.get('frequency_hours') != 3
                or normalization.get('step_unit') != 'hour' or normalization.get('step') != 3
                or normalization.get('time_zone') != 'UTC'
                or normalization.get('timestamp_rule') != 'interval_center'
                or normalization.get('fit_step_count') != len(train_dates)
                or normalization.get('fit_time_start') != _utc_timestamp(train_dates[0])
                or normalization.get('fit_time_end') != _utc_timestamp(train_dates[-1])):
            raise ValueError('Normalization must belong to the three-hour UTC training interval')
        dataset = cls.__new__(cls)
        dataset._release = release
        dataset.data_path = _threehour_path(release)
        dataset.window, dataset.horizon = window, horizon
        dataset.length, dataset.split = length, split
        dataset._offset = int(indices[0])
        dataset.dates = dates[indices]
        dataset.lat, dataset.lon = lat.copy(), lon.copy()
        dataset.input_channels, dataset.input_units = channels, input_units(channels)
        dataset.target_channel_index = TARGET_CHANNEL_INDEX
        dataset.normalization = normalization
        # Precompute one seasonal coordinate per published UTC timestamp, then
        # slice windows without rerunning the solar-longitude formula.
        from services.earth_season import solar_longitude
        seasonal = np.asarray(solar_longitude(dataset.dates), dtype="float32")
        seasonal.setflags(write=False)
        dataset._seasonal_longitude = seasonal
        dataset._normalized_map = None
        dataset._normalized_cache_layout = None
        dataset.mean = mean.astype('float32')
        dataset.std = scale.astype('float32')
        return dataset

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.read_window(index)

    def use_training_cache(self, path, *, progress=None):
        """Attach a server-created map after checking identity and statistics."""
        _threehour_path(self._release)
        path = Path(path)
        metadata = json.loads((path.parent / 'metadata.json').read_text(encoding='utf-8'))
        shape = (len(self._release.dates), len(self.input_channels), len(self._release.latitude), len(self._release.longitude))
        from services.earth_preparation_cache import signature_json
        normalization_match = metadata.get('normalization') == self.normalization
        if metadata.get('cache_identity_version') == 2 and isinstance(metadata.get('normalization'), dict):
            normalization_match = _normalization_content(metadata['normalization']) == _normalization_content(self.normalization)
        if (metadata.get('dataset_id') != self._release.metadata.get('dataset_id', THREE_HOURLY_DATASET_ID)
                or metadata.get('dataset_fingerprint') != self._release.metadata.get('dataset_fingerprint')
                or metadata.get('input_channel_order') != self.input_channels
                or not normalization_match
                or metadata.get('shape') != list(shape)):
            raise ValueError('Three-hour training cache identity or normalization mismatch')
        if metadata.get('cache_identity_version') not in (None, 2):
            raise ValueError('Unsupported three-hour training cache identity version')
        if metadata.get('cache_identity_version') == 2:
            if metadata.get('cache_status') != 'ready' or metadata.get('package_signature') != signature_json(self._release.signature):
                raise ValueError('Three-hour training cache is incomplete or source identity changed')
        layout = metadata.get('layout')
        storage_shape = shape
        if metadata.get('schema') is not None and (metadata.get('schema') != THREE_HOURLY_CACHE_SCHEMA
                                                   or metadata.get('cache_version') != 1):
            raise ValueError('Unsupported three-hour training cache version')
        if layout == THREE_HOURLY_TILE_CACHE_LAYOUT:
            storage_shape = (100, shape[0], shape[1], *THREE_HOURLY_TILE_SHAPE)
            if (metadata.get('storage_shape') != list(storage_shape)
                    or metadata.get('spatial_tile_shape') != list(THREE_HOURLY_TILE_SHAPE)):
                raise ValueError('Three-hour training cache storage layout mismatch')
        elif layout == THREE_HOURLY_FULL_GRID_CACHE_LAYOUT:
            if (metadata.get('schema') != THREE_HOURLY_CACHE_SCHEMA
                    or metadata.get('cache_version') != 1
                    or metadata.get('storage_shape') != list(storage_shape)):
                raise ValueError('Three-hour training cache storage layout mismatch')
        elif layout is not None:
            raise ValueError('Unsupported three-hour training cache layout')
        identity = _training_cache_identity(self._release, self.input_channels, self.normalization,
                                            full_grid=layout != THREE_HOURLY_TILE_CACHE_LAYOUT)
        if metadata.get('cache_identity_version') is None:
            legacy = dict(metadata, schema=THREE_HOURLY_CACHE_SCHEMA, cache_version=1,
                          dataset_version=THREE_HOURLY_DATASET_VERSION,
                          layout=identity['layout'], storage_shape=list(storage_shape))
            if _certify_existing_training_cache(path, legacy, identity, self._release,
                                                progress=progress) is None:
                raise ValueError('Three-hour training cache integrity check failed')
            return self.use_training_cache(path, progress=progress)
        from services.earth_preparation_cache import (ReadOnlyFileGuard,
                                                      file_identity_matches, verify_array_file)
        guard = ReadOnlyFileGuard(path)
        mapped = None
        try:
            if _training_cache_candidate(path, identity, verify_payload=False) is None:
                raise ValueError('Three-hour training cache integrity check failed')
            # Keep the certified identity; never adopt a newer stat baseline.
            certified_identity = metadata.get('array_identity')
            if not file_identity_matches(path, certified_identity):
                raise ValueError('Three-hour training cache file changed after publication')
            mapped = np.load(path, mmap_mode='r', allow_pickle=False)
            if (mapped.shape != storage_shape or mapped.dtype != np.dtype('float32')
                    or path.stat().st_size != mapped.offset + mapped.nbytes):
                raise ValueError('Three-hour training cache shape or dtype mismatch')
            if not verify_array_file(path, certified_identity, metadata.get('array_sha256'),
                                     progress=progress):
                raise ValueError('Three-hour training cache changed after publication')
            _threehour_path(self._release)
            if not file_identity_matches(path, certified_identity):
                raise ValueError('Three-hour training cache file changed after publication')
        except BaseException:
            if mapped is not None:
                mapped._mmap.close()
            guard.close()
            raise
        self.close_training_cache()
        self._normalized_map = mapped
        self._normalized_cache_layout = layout
        self._normalized_cache_path = path
        self._normalized_cache_file_identity = certified_identity
        self._normalized_cache_guard = guard

    def close_training_cache(self):
        mapped = getattr(self, '_normalized_map', None)
        if mapped is not None:
            mapped._mmap.close()
            self._normalized_map = None
        guard = getattr(self, '_normalized_cache_guard', None)
        if guard is not None:
            guard.close()
            self._normalized_cache_guard = None

    def __del__(self):
        self.close_training_cache()

    def _check_cache_file(self):
        from services.earth_preparation_cache import file_identity_matches
        if not file_identity_matches(self._normalized_cache_path, self._normalized_cache_file_identity):
            raise ValueError('Three-hour training cache file changed after publication')

    def read_windows(self, requests):
        """Verify the source before and after a batch of read-only cache slices."""
        if self._normalized_map is None:
            return [self.read_window(index, lat_slice=lat, lon_slice=lon)
                    for index, lat, lon in requests]
        _threehour_path(self._release)
        self._check_cache_file()
        windows = [self._read_window(index, lat_slice=lat, lon_slice=lon, verify_source=False)
                   for index, lat, lon in requests]
        _threehour_path(self._release)
        self._check_cache_file()
        return windows

    def read_window(self, index, *, lat_slice=slice(None), lon_slice=slice(None)):
        """Read exactly one temporal window, optionally limited to a spatial tile."""
        return self._read_window(index, lat_slice=lat_slice, lon_slice=lon_slice, verify_source=True)

    def window_longitude(self, index):
        """Return one input window's seasonal coordinate as float32 ``[window]``.

        Derived from the partition's own timestamps, so it costs no extra read and
        cannot disagree with the data the same window returns. Models that do not
        declare the derived input never receive it.
        """
        index = operator.index(index)
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return self._seasonal_longitude[index:index + self.window].copy()

    def window_longitudes(self, indices):
        """Stack several windows' seasonal coordinates as float32 ``[N, window]``."""
        positions = np.asarray([operator.index(index) for index in indices], dtype="int64")
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= self.length):
            raise IndexError("Earth seasonal-coordinate window index is out of range")
        windows = np.lib.stride_tricks.sliding_window_view(self._seasonal_longitude, self.window)
        return np.array(windows[positions], dtype="float32", copy=True)

    def _read_window(self, index, *, lat_slice, lon_slice, verify_source):
        from services.netcdf_read_lock import netcdf_read_lock
        index = operator.index(index)
        if index < 0 or index >= self.length:
            raise IndexError(index)
        if not isinstance(lat_slice, slice) or not isinstance(lon_slice, slice):
            raise ValueError('Spatial tiles must use latitude and longitude slices')
        latitude = np.arange(len(self.lat))[lat_slice]
        longitude = np.arange(len(self.lon))[lon_slice]
        if not len(latitude) or not len(longitude):
            raise ValueError('Spatial tiles must contain at least one grid cell')
        start = self._offset + index
        forecast, stop = start + self.window, start + self.window + self.horizon
        if self._normalized_map is not None:
            if verify_source:
                _threehour_path(self._release)
                self._check_cache_file()
            if self._normalized_cache_layout == THREE_HOURLY_TILE_CACHE_LAYOUT:
                inputs, targets = self._read_tile_cache(start, forecast, stop, latitude, longitude)
            else:
                inputs = np.array(self._normalized_map[start:forecast, :, lat_slice, lon_slice], copy=True)
                targets = np.array(self._normalized_map[forecast:stop, :1, lat_slice, lon_slice], copy=True)
            if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
                raise ValueError('Three-hour training cache contains non-finite values')
            if verify_source:
                _threehour_path(self._release)
                self._check_cache_file()
            return inputs, targets
        inputs = np.empty((self.window, len(self.input_channels), len(latitude), len(longitude)), dtype='float32')
        targets = np.empty((self.horizon, 1, len(latitude), len(longitude)), dtype='float32')
        path = _threehour_path(self._release)
        with netcdf_read_lock(), xr.open_dataset(path, engine='netcdf4', mask_and_scale=False) as ds:
            if not np.array_equal(ds.time.isel(time=slice(start, stop)).values,
                                  self.dates[index:index + self.window + self.horizon]):
                raise ValueError('Earth three-hour timestamps changed since verification')
            for channel_index, channel in enumerate(self.input_channels):
                values = _threehour_values(ds, channel, start, forecast,
                                          lat_slice=lat_slice, lon_slice=lon_slice)
                np.subtract(values, self.mean[channel_index], out=inputs[:, channel_index])
                np.divide(inputs[:, channel_index], self.std[channel_index], out=inputs[:, channel_index])
                if channel == TARGET_CHANNEL:
                    values = _threehour_values(ds, channel, forecast, stop,
                                              lat_slice=lat_slice, lon_slice=lon_slice)
                    np.subtract(values, self.mean[channel_index], out=targets[:, 0])
                    np.divide(targets[:, 0], self.std[channel_index], out=targets[:, 0])
        _threehour_path(self._release)
        return inputs, targets

    def _read_tile_cache(self, start, forecast, stop, latitude, longitude):
        tile_height, tile_width = THREE_HOURLY_TILE_SHAPE
        if (len(latitude) == tile_height and len(longitude) == tile_width
                and latitude[0] % tile_height == 0 and longitude[0] % tile_width == 0
                and np.array_equal(latitude, latitude[0] + np.arange(tile_height))
                and np.array_equal(longitude, longitude[0] + np.arange(tile_width))):
            tile = int(latitude[0] // tile_height * 10 + longitude[0] // tile_width)
            return (
                np.array(self._normalized_map[tile, start:forecast], copy=True),
                np.array(self._normalized_map[tile, forecast:stop, :1], copy=True),
            )
        inputs = np.empty((forecast - start, len(self.input_channels), len(latitude), len(longitude)), dtype='float32')
        targets = np.empty((stop - forecast, 1, len(latitude), len(longitude)), dtype='float32')
        # Arbitrary slices can cross tiles or reverse/skip coordinates.
        for tile_row in np.unique(latitude // tile_height):
            output_rows = np.flatnonzero(latitude // tile_height == tile_row)
            local_rows = latitude[output_rows] % tile_height
            for tile_column in np.unique(longitude // tile_width):
                output_columns = np.flatnonzero(longitude // tile_width == tile_column)
                local_columns = longitude[output_columns] % tile_width
                tile = int(tile_row * 10 + tile_column)
                history = self._normalized_map[tile, start:forecast]
                future = self._normalized_map[tile, forecast:stop, :1]
                inputs[:, :, output_rows[:, None], output_columns] = history[:, :, local_rows[:, None], local_columns]
                targets[:, :, output_rows[:, None], output_columns] = future[:, :, local_rows[:, None], local_columns]
        return inputs, targets

    def denormalize_ozone(self, values):
        mean = float(self.normalization['mean'][self.target_channel_index])
        scale = float(self.normalization['scale'][self.target_channel_index])
        if hasattr(values, 'dim') and hasattr(values, 'to'):
            return values * scale + mean
        return np.asarray(values) * scale + mean


def release_split_codes(dates, metadata):
    """Derive the per-day split code from the verified manifest split ranges.

    The published manifest is the authority, not a recomputed boundary: a
    package whose split flags disagree with its manifest fails verification
    before reaching this function.
    """
    _require_daily_metadata(metadata)
    splits = metadata.get('splits') or {}
    codes = np.full(len(dates), -1, dtype='int8')
    for name, code in SPLITS.items():
        entry = splits.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f'Verified release has no published {name} split')
        start = np.datetime64(str(entry['start']), 'D')
        end = np.datetime64(str(entry['end']), 'D')
        codes[(dates >= start) & (dates <= end)] = code
    if np.any(codes < 0):
        raise ValueError('Published splits do not cover every release date')
    return codes


def release_date_index(release, date):
    """Return the position of ``date`` in the release, or raise ``KeyError``."""
    _require_daily_metadata(release.metadata)
    dates = np.asarray(release.dates, dtype='datetime64[D]')
    target = np.datetime64(date, 'D')
    matches = np.flatnonzero(dates == target)
    if matches.size == 0:
        raise KeyError(str(target))
    return int(matches[0])


def reference_ozone(release, start_date, horizon):
    """Return ``[horizon, lat, lon]`` reference TO3 in DU for a date range."""
    begin = release_date_index(release, start_date)
    stop = begin + int(horizon) - 1
    dates = np.asarray(release.dates, dtype='datetime64[D]')
    if stop >= len(dates):
        raise KeyError(str(np.datetime64(start_date, 'D') + np.timedelta64(horizon - 1, 'D')))
    span = dates[begin:stop + 1]
    if len(span) != int(horizon) or np.any(np.diff(span) != np.timedelta64(1, 'D')):
        raise ValueError('Reference dates are not contiguous in the release')
    field = np.asarray(release.fields[TARGET_CHANNEL], dtype='float32')
    return np.array(field[begin:stop + 1], dtype='float32', copy=True)


__all__ = [
    'CHANNELS',
    'EarthOzoneWindows',
    'EarthThreeHourlyWindows',
    'MINIMUM_SCALE',
    'NORMALIZATION_METHOD',
    'OPTIONAL_CHANNELS',
    'SCHEMA',
    'THREE_HOURLY_SCHEMA',
    'THREE_HOURLY_DATASET_ID',
    'THREE_HOURLY_DATASET_VERSION',
    'THREE_HOURLY_GRID_SHAPE',
    'THREE_HOURLY_TRAINING_PROFILE',
    'SPLITS',
    'TARGET_CHANNEL',
    'TARGET_CHANNEL_INDEX',
    'UNITS',
    'build_threehour_training_cache',
    'cell_area_weights_from_lat_bounds',
    'canonical_input_channels',
    'fit_normalization',
    'fit_threehour_normalization',
    'input_units',
    'load_earth_dataset',
    'normalize_cube',
    'reference_ozone',
    'release_date_index',
    'release_split_codes',
    'threehour_release_split_codes',
    'threehour_global_cell_area_weights',
    'split_days',
    'stack_release_cube',
    'validate_dataset',
    'validate_earth_3hourly_dataset',
    'validate_normalization',
]
