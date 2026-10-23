"""Render an Earth data preview and optionally exercise the existing model factory."""

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np

from scripts.build_earth_ozone_dataset import SOURCE_KEYS, sha256
from services.earth_dataset import CHANNELS, EarthOzoneWindows, load_earth_dataset


def compare_derivative(ds, source_path):
    with np.load(source_path, allow_pickle=False) as src:
        for key, name in zip(SOURCE_KEYS, CHANNELS):
            np.testing.assert_array_equal(ds[name].values, src[key])
        for coordinate in ('lat', 'lon'):
            np.testing.assert_array_equal(ds[coordinate].values, src[coordinate])
        np.testing.assert_array_equal(ds.time.values.astype('datetime64[D]'),
                                      src['time_iso'].astype('datetime64[D]'))
    return {'exact_match': True, 'values_compared': sum(ds[name].size for name in CHANNELS),
            'source_sha256': sha256(source_path)}


def compare_raw_cells(ds, raw_dir):
    """Independent scalar cell integrals; no preprocessing weight helpers."""
    dates = ds.time.values.astype('datetime64[D]')
    indices = sorted({0, min(59, len(dates)-1), len(dates)//2, 3*len(dates)//4, len(dates)-1})
    results = []
    for index in indices:
        day = str(dates[index])
        for group, names in (('slv', CHANNELS[:4]), ('rad', CHANNELS[4:])):
            paths = list((Path(raw_dir)/group).glob(f'*.{day.replace("-", "")}.nc4'))
            if len(paths) != 1:
                raise ValueError(f'Expected one {group} source for {day}')
            with netCDF4.Dataset(paths[0]) as source:
                lat = np.asarray(source['lat'][:], dtype='float64')
                lon = np.asarray(source['lon'][:], dtype='float64')
                lat_edges = np.r_[-90., (lat[1:]+lat[:-1])/2, 90.]
                step = lon[1]-lon[0]
                source_area = np.diff(np.sin(np.deg2rad(lat_edges)))
                target_area = np.sin(np.deg2rad(ds.lat_bounds.values[:, 1]))-np.sin(np.deg2rad(ds.lat_bounds.values[:, 0]))
                for name in names:
                    variable = source[name]
                    variable.set_auto_maskandscale(False)
                    hourly = np.asarray(variable[:], dtype='float64')
                    if not np.isfinite(hourly).all() or np.any(abs(hourly) >= 1e14):
                        raise ValueError(f'Invalid raw values: {day}/{name}')
                    daily = hourly.sum(axis=0) / 24.
                    errors = []
                    for i in (0, len(ds.lat)//2, len(ds.lat)-1):
                        south, north = ds.lat_bounds.values[i]
                        rows = []
                        for r in range(len(lat)):
                            lo, hi = max(south, lat_edges[r]), min(north, lat_edges[r+1])
                            if hi > lo:
                                rows.append((r, np.sin(np.deg2rad(hi))-np.sin(np.deg2rad(lo))))
                        for j in (0, len(ds.lon)//2, len(ds.lon)-1):
                            west, east = ds.lon_bounds.values[j]
                            columns = []
                            for c, centre in enumerate(lon):
                                width = sum(max(0., min(east, centre+step/2+shift)-max(west, centre-step/2+shift))
                                            for shift in (-360., 0., 360.))
                                if width > 0: columns.append((c, width))
                            weighted = sum(daily[r,c]*area*width for r,area in rows for c,width in columns)
                            denominator = sum(area for _,area in rows)*sum(width for _,width in columns)
                            expected = weighted / denominator
                            actual = float(ds[name].values[index,i,j])
                            np.testing.assert_allclose(actual, np.float32(expected), rtol=1e-7, atol=1e-6)
                            errors.append(abs(actual-expected))
                    raw_mean = float(np.sum(daily*source_area[:,None]) / (2*len(lon)))
                    stored_mean = float(np.sum(ds[name].values[index]*target_area[:,None]) / (2*len(ds.lon)))
                    np.testing.assert_allclose(stored_mean, raw_mean, rtol=1e-7, atol=1e-6)
                    results.append({'date': day, 'variable': name, 'cells': 9,
                                    'max_absolute_error': max(errors),
                                    'raw_area_mean': raw_mean, 'stored_area_mean': stored_mean})
    return results


def compare_raw_samples(ds, raw_dir):
    if 'lat_bounds' in ds and 'lon_bounds' in ds:
        return compare_raw_cells(ds, raw_dir)
    results = []
    dates = ds.time.values.astype('datetime64[D]')
    day_indices = sorted({0, min(59, len(dates) - 1), len(dates) // 2,
                          3 * len(dates) // 4, len(dates) - 1})
    row = [0, len(ds.lat) // 2, len(ds.lat) - 1]
    col = [0, len(ds.lon) // 2, len(ds.lon) - 1]
    for index in day_indices:
        day = str(dates[index])
        for group, names in (('slv', CHANNELS[:4]), ('rad', CHANNELS[4:])):
            paths = list((Path(raw_dir) / group).glob(f'*.{day.replace("-", "")}.nc4'))
            if len(paths) != 1:
                raise ValueError(f'Expected one {group} source for {day}, got {len(paths)}')
            with netCDF4.Dataset(paths[0]) as source:
                source_lat, source_lon = source['lat'][:], source['lon'][:]
                lat_indices = [int(np.argmin(abs(source_lat - ds.lat.values[i]))) for i in row]
                lon_indices = [int(np.argmin(abs(source_lon - ds.lon.values[j]))) for j in col]
                np.testing.assert_allclose(source_lat[lat_indices], ds.lat.values[row], atol=1e-5)
                np.testing.assert_allclose(source_lon[lon_indices], ds.lon.values[col], atol=1e-5)
                time_var = source['time']
                # MERRA-2's integer time variable may carry an out-of-range
                # floating valid_range attribute. Decode the actual 24 values.
                time_var.set_auto_mask(False)
                hours = netCDF4.num2date(time_var[:], time_var.units,
                                         calendar=getattr(time_var, 'calendar', 'standard'))
                if (len(hours) != 24 or any(t.strftime('%Y-%m-%d') != day for t in hours)
                        or [t.hour for t in hours] != list(range(24))):
                    raise ValueError(f'Incomplete hourly source for {day}')
                for name in names:
                    raw = source[name][:, lat_indices, lon_indices]
                    if np.ma.getmaskarray(raw).any():
                        raise ValueError(f'Missing raw sample for {day}/{name}')
                    expected = np.asarray(raw.mean(axis=0), dtype='float32')
                    actual = ds[name].values[index][np.ix_(row, col)]
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)
                    results.append({'date': day, 'variable': name, 'points': 9,
                                    'max_absolute_error': float(np.max(abs(actual - expected)))})
    return results


def render_preview(ds, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig = plt.figure(figsize=(12, 7.6), constrained_layout=True, facecolor='#f5f7fa')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.85])
    ozone = ds.TO3.values
    cells = 'lat_bounds' in ds and 'lon_bounds' in ds
    lo, hi = float(ozone.min()), float(ozone.max())
    for column, index in enumerate((0, min(182, len(ds.time) - 1))):
        ax = fig.add_subplot(grid[0, column])
        mesh = ax.pcolormesh(ds.lon.values, ds.lat.values, ozone[index], cmap='viridis',
                             vmin=lo, vmax=hi, shading='nearest')
        ax.set(title=f'Daily ozone | {str(ds.time.values[index])[:10]}',
               xlabel='Longitude (degrees east)', ylabel='Latitude (degrees north)')
        fig.colorbar(mesh, ax=ax, label='Total column ozone (DU)', shrink=0.8)
    ax = fig.add_subplot(grid[1, :])
    weights = (np.sin(np.deg2rad(ds.lat_bounds.values[:, 1])) - np.sin(np.deg2rad(ds.lat_bounds.values[:, 0])))[:, None] if cells else np.cos(np.deg2rad(ds.lat.values))[:, None]
    mean = (ozone * weights).sum(axis=(1, 2)) / (weights.sum() * len(ds.lon))
    dates = ds.time.values
    ax.plot(dates, mean, color='#263d70', linewidth=1.2)
    for code, label, color in ((0, 'Training', '#d5e9fc'), (1, 'Validation', '#fff1be'),
                                (2, 'Test', '#e8ddf5')):
        selected = dates[ds['split'].values == code]
        ax.axvspan(selected[0], selected[-1], color=color, alpha=0.65, label=label)
    ax.set(title='Global spherical cell-area mean' if cells else 'Latitude-weighted sample mean within the covered region', ylabel='Ozone (DU)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(loc='upper right', ncol=3, frameon=False)
    ax.grid(alpha=0.2)
    fig.suptitle('MERRA-2 Earth ozone | compact daily dataset\n'
                 f'{len(ds.time)} days | {len(ds.lat)} x {len(ds.lon)} grid | '
                 + ('global coverage, spherical overlap aggregation' if cells else 'regional coverage, spatial point sampling'), fontsize=14)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def training_smoke(path):
    import torch
    from torch.utils.data import DataLoader
    from training_backbones.model_zoo import build_forecaster

    torch.manual_seed(7)
    torch.set_num_threads(2)
    dataset = EarthOzoneWindows(path, window=7, horizon=3)
    x, y = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    model = build_forecaster('dlinear', input_channels=5, selected_channels=['U', 'V', 'T', 'S'],
                             hidden_dims=[8], height=len(dataset.lat), width=len(dataset.lon),
                             window=7, horizon=3, use_sphere=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    # This legacy backbone requires a second argument even in raw mode.
    # With use_sphere=False / use_phase_warp=False it is not used as a feature.
    prediction = model(x, torch.zeros(x.shape[:2], dtype=x.dtype))
    if prediction.shape != y.shape:
        raise ValueError(f'Model output {prediction.shape} != target {y.shape}')
    loss = torch.nn.functional.mse_loss(prediction, y)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    optimizer.zero_grad()
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('Nonfinite gradients')
    optimizer.step()
    if not any(not torch.equal(a, p) for a, p in zip(before, model.parameters())):
        raise ValueError('Optimizer did not update any weights')
    return {'model': 'dlinear', 'device': 'cpu', 'optimizer_steps': 1,
            'input_shape': list(x.shape), 'target_shape': list(y.shape),
            'finite_loss': True, 'weights_updated': True,
            'purpose': 'Data/model compatibility only; not a predictive skill evaluation',
            'channel_mean': dataset.mean.tolist(), 'channel_std': dataset.std.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-npz', type=Path)
    parser.add_argument('--raw-dir', type=Path)
    parser.add_argument('--training-smoke', action='store_true')
    args = parser.parse_args()
    report = {'data_file': args.dataset.name, 'data_sha256': sha256(args.dataset)}
    with load_earth_dataset(args.dataset) as ds:
        if args.source_npz:
            report['derivative_comparison'] = compare_derivative(ds, args.source_npz)
        if args.raw_dir:
            report['raw_sample_comparison'] = compare_raw_samples(ds, args.raw_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        render_preview(ds, args.output_dir / 'preview.png')
    if args.training_smoke:
        report['training_smoke'] = training_smoke(args.dataset)
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
