import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.preprocess_earth_reanalysis import (
    _fingerprint_inventory,
    _source_fingerprint,
    _source_inventory,
    aggregate_to_grid,
    build_overlap_weights,
    discover_merra2_files,
    preprocess_merra2,
)


def test_overlap_weights_are_normalized_and_respect_spherical_area():
    source_lat = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    target_lat = np.array([-45.0, 0.0, 45.0])
    weights = build_overlap_weights(
        source_lat, target_lat, source_bounds=(-75.0, 75.0), spherical=True
    )

    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    # The middle cell is symmetric; its southern and northern source rows have
    # equal area contributions, while the equator row has the largest weight.
    assert weights[1, 2] > weights[1, 1]
    assert weights[1, 1] == pytest.approx(weights[1, 3])


def test_aggregate_to_grid_preserves_a_constant_and_averages_a_linear_field():
    source_lat = np.array([-1.0, 0.0, 1.0])
    source_lon = np.array([-2.0, 0.0, 2.0])
    target_lat = np.array([-0.5, 0.5])
    target_lon = np.array([-1.0, 1.0])
    values = np.empty((2, 3, 3), dtype=np.float32)
    values[0] = 7.0
    values[1] = source_lat[:, None] + 3.0 * source_lon[None, :]

    result = aggregate_to_grid(
        values,
        source_lat,
        source_lon,
        target_lat,
        target_lon,
        source_lat_bounds=(-1.5, 1.5),
        source_lon_bounds=(-2.5, 2.5),
    )

    np.testing.assert_allclose(result[0], 7.0)
    # A linear field is reproduced at the target cell centres by symmetric
    # overlap averaging.
    expected = target_lat[:, None] + 3.0 * target_lon[None, :]
    np.testing.assert_allclose(result[1], expected, rtol=0, atol=3e-5)


def test_preprocess_merra2_builds_daily_area_weighted_package(tmp_path):
    source_root = tmp_path / "raw"
    (source_root / "slv").mkdir(parents=True)
    (source_root / "rad").mkdir(parents=True)
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    lat = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    lon = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    from netCDF4 import Dataset

    for day_index, date in enumerate(dates):
        for product, variables in (
            ("slv", ("TO3", "U10M", "V10M", "T2M")),
            ("rad", ("SWGDN",)),
        ):
            path = source_root / product / f"{product}.{date.replace('-', '')}.nc4"
            with Dataset(path, "w") as ds:
                ds.createDimension("time", 24)
                ds.createDimension("lat", len(lat))
                ds.createDimension("lon", len(lon))
                time = ds.createVariable("time", "f8", ("time",))
                time.units = f"hours since {date} 00:00:00"
                time[:] = np.arange(24) + 0.5
                ds.createVariable("lat", "f4", ("lat",))[:] = lat
                ds.createVariable("lon", "f4", ("lon",))[:] = lon
                for variable in variables:
                    value = day_index * 10 + (0 if variable == "TO3" else 1)
                    v = ds.createVariable(variable, "f4", ("time", "lat", "lon"))
                    v.units = {"TO3": "Dobsons", "U10M": "m s-1", "V10M": "m s-1", "T2M": "K", "SWGDN": "W m-2"}[variable]
                    v[:] = value

    output_dir = tmp_path / "release"
    output = preprocess_merra2(
        source_root,
        output_dir,
        target_lat=np.array([-0.5, 0.5], dtype=np.float32),
        target_lon=np.array([-1.0, 1.0], dtype=np.float32),
        train_end="2020-01-01",
        validation_end="2020-01-02",
    )

    assert output.name == "earth_merra2_daily.nc"
    with Dataset(output) as ds:
        assert ds.dimensions["time"].size == 3
        assert ds.dimensions["lat"].size == 2
        assert ds.dimensions["lon"].size == 2
        np.testing.assert_allclose(ds["TO3"][0], 0.0)
        np.testing.assert_allclose(ds["TO3"][2], 20.0)
        np.testing.assert_array_equal(ds["split"][:], [0, 1, 2])

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["processing"]["spatial_method"] == "spherical_area_weighted_overlap"
    assert manifest["processing"]["temporal_method"] == "mean_of_complete_source_steps"
    assert manifest["source"]["product"] == "MERRA-2"
    inventory = manifest["source_files"]
    assert len(inventory) == 6
    assert all({"date", "product", "name", "bytes", "mtime_ns", "sha256"} <= item.keys()
               for item in inventory)
    assert manifest["source_sha256"] == _fingerprint_inventory(inventory)
    assert "full-file-sha256" in manifest["source_fingerprint_basis"]
    report = json.loads((output_dir / "verification" / "preprocess_report.json").read_text(encoding="utf-8"))
    assert report["source_files"] == inventory
    assert report["source_sha256"] == manifest["source_sha256"]


def test_daily_source_inventory_hash_and_canonical_fingerprint_are_reproducible(tmp_path):
    source = make_raw(tmp_path / "raw")
    index = discover_merra2_files(source)

    first = _source_inventory(index)
    second = _source_inventory(index)

    assert first == second
    assert _source_fingerprint(index) == _source_fingerprint(index)
    assert _fingerprint_inventory(first) == _fingerprint_inventory(second)
    assert all(len(item["sha256"]) == 64 for item in first)
    canonical = json.dumps(first, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert _fingerprint_inventory(first) == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("change_size", [False, True], ids=["same-size-restored-mtime", "size-changed"])
def test_daily_preprocess_rejects_source_content_change_before_manifest(tmp_path, monkeypatch, change_size):
    import scripts.preprocess_earth_reanalysis as processor

    source_root = make_raw(tmp_path / "raw")
    target = source_root / "rad" / "MERRA2_400.tavg1_2d_rad_Nx.20210102.nc4"
    original_read = processor._read_daily_mean
    changed = False

    def read_then_change(path, *args, **kwargs):
        nonlocal changed
        result = original_read(path, *args, **kwargs)
        if path == target and not changed:
            before = target.stat()
            contents = bytearray(target.read_bytes())
            if change_size:
                contents.extend(b"x")
            else:
                contents[-1] ^= 1
            target.write_bytes(contents)
            if not change_size:
                os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
                after = target.stat()
                assert after.st_size == before.st_size
                assert after.st_mtime_ns == before.st_mtime_ns
            changed = True
        return result

    monkeypatch.setattr(processor, "_read_daily_mean", read_then_change)
    output_dir = tmp_path / "release"
    with pytest.raises(ValueError, match="source archive changed during preprocessing"):
        preprocess_merra2(source_root, output_dir, train_end="2020-12-31", validation_end="2021-01-01")

    assert changed
    assert not (output_dir / "manifest.json").exists()


def test_daily_preprocess_distinguishes_mtime_only_change_from_content_change(tmp_path, monkeypatch):
    import scripts.preprocess_earth_reanalysis as processor

    source_root = make_raw(tmp_path / "raw")
    target = source_root / "rad" / "MERRA2_400.tavg1_2d_rad_Nx.20210102.nc4"
    original_read = processor._read_daily_mean
    touched = False

    def read_then_touch(path, *args, **kwargs):
        nonlocal touched
        result = original_read(path, *args, **kwargs)
        if path == target and not touched:
            before = target.stat()
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
            touched = True
        return result

    monkeypatch.setattr(processor, "_read_daily_mean", read_then_touch)
    output_dir = tmp_path / "release"
    with pytest.raises(ValueError, match="source archive metadata changed during preprocessing"):
        preprocess_merra2(source_root, output_dir, train_end="2020-12-31", validation_end="2021-01-01")

    assert touched
    assert not (output_dir / "manifest.json").exists()


def test_preprocess_merra2_rejects_an_incomplete_day(tmp_path):
    source_root = tmp_path / "raw"
    (source_root / "slv").mkdir(parents=True)
    (source_root / "rad").mkdir(parents=True)
    (source_root / "slv" / "slv.20200101.nc4").touch()
    # A missing radiation file must never silently produce an aligned sample.
    with pytest.raises(ValueError, match="missing paired daily files"):
        preprocess_merra2(
            source_root,
            tmp_path / "release",
            target_lat=np.array([0.0], dtype=np.float32),
            target_lon=np.array([0.0], dtype=np.float32),
        )


def test_periodic_seam_and_spherical_conservation():
    # Native longitude centres include -180, so its cell contributes to both
    # ends of the target domain, never stretched into an oversized last cell.
    lat = np.arange(-90., 91., 30.)
    lon = np.arange(-180., 180., 30.)
    target_lat = np.arange(-75., 90., 30.)
    target_lon = np.arange(-165., 180., 30.)
    values = np.arange(len(lat) * len(lon)).reshape(len(lat), len(lon)).astype(float)
    result = aggregate_to_grid(values, lat, lon, target_lat, target_lon,
                               source_lat_bounds=(-90., 90.), source_lon_bounds=(-180., 180.))
    seam = np.zeros_like(values)
    seam[:, 0] = 1.
    edge_result = aggregate_to_grid(seam, lat, lon, target_lat, target_lon,
                                    source_lat_bounds=(-90., 90.), source_lon_bounds=(-180., 180.))
    np.testing.assert_allclose(edge_result[:, [0, -1]], 0.5)
    source_edges = np.r_[-90, (lat[:-1] + lat[1:]) / 2, 90]
    source_area = np.diff(np.sin(np.deg2rad(source_edges)))
    target_area = np.diff(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
    np.testing.assert_allclose(np.sum(values * source_area[:, None]) / (2 * len(lon)),
                               np.sum(result * target_area[:, None]) / (2 * len(target_lon)), atol=1e-12)


def make_raw(root, *, bad=None):
    from netCDF4 import Dataset
    for day_index, day in enumerate(('20201231', '20210101', '20210102')):
        for product, names in (('slv', ('TO3', 'U10M', 'V10M', 'T2M')), ('rad', ('SWGDN',))):
            directory = root / product
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'MERRA2_{401 if day_index == 0 else 400}.tavg1_2d_{product}_Nx.{day}.nc4'
            with Dataset(path, 'w') as ds:
                ds.createDimension('time', 24)
                ds.createDimension('lat', 7)
                ds.createDimension('lon', 12)
                t = ds.createVariable('time', 'f8', ('time',))
                iso = f'{day[:4]}-{day[4:6]}-{day[6:]}'
                t.units = f'hours since {iso} 00:00:00'
                t[:] = np.arange(24) + (0 if bad == 'hours' else .5)
                ds.createVariable('lat', 'f4', ('lat',))[:] = np.arange(-90, 91, 30)
                ds.createVariable('lon', 'f4', ('lon',))[:] = np.arange(-180, 180, 30)
                for name in names:
                    v = ds.createVariable(name, 'f4', ('time', 'lat', 'lon'), fill_value=1e15)
                    v.units = dict(TO3='Dobsons', U10M='m s-1', V10M='m s-1', T2M='K', SWGDN='W m-2')[name]
                    v[:] = day_index * 1000 + np.arange(24)[:, None, None]
                    if name == 'TO3':
                        if bad == 'fill': v[0, 0, 0] = 1e15
                        if bad == 'missing':
                            v.missing_value = np.float32(-9999)
                            v[0, 0, 0] = -9999
                        if bad == 'unit': v.units = 'kg m-2'
                        if bad == 'nan': v[0, 0, 0] = np.nan
    return root


@pytest.mark.parametrize('bad,match', [('hours', 'time'), ('fill', 'invalid'), ('missing', 'missing'), ('unit', 'units'), ('nan', 'invalid')])
def test_rejects_bad_raw_values_or_metadata(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        preprocess_merra2(make_raw(tmp_path / 'raw', bad=bad), tmp_path / 'out',
                          train_end='2020-12-31', validation_end='2021-01-01')
    assert not (tmp_path / 'out' / 'manifest.json').exists()


def test_normalization_fits_train_only_and_bounds_match_release(tmp_path):
    from services.earth_dataset import load_earth_dataset, EarthOzoneWindows
    from services.earth_dataset_metadata import read_earth_metadata, file_sha256
    output = preprocess_merra2(make_raw(tmp_path / 'raw'), tmp_path / 'out',
                              train_end='2020-12-31', validation_end='2021-01-01')
    m = json.loads(output.with_name('manifest.json').read_text())
    assert m['normalization']['train_stats']['TO3']['mean'] == pytest.approx(11.5)
    assert m['normalization']['train_stats']['TO3']['std'] == 0
    with load_earth_dataset(output) as ds:
        np.testing.assert_array_equal(ds.lat_bounds.values[[0, -1]], [[-90, -85], [85, 90]])
        np.testing.assert_allclose(ds.TO3.values[:, 0, 0], [11.5, 1011.5, 2011.5])
    metadata = read_earth_metadata(output.parent, expected_manifest_sha256=file_sha256(output.with_name('manifest.json')),
                                   expected_data_sha256=file_sha256(output), dataset_id='earth_merra2_daily_v2', dataset_version='v2')
    assert metadata['grid']['coverage'] == 'global'
    assert metadata['grid']['latitude_range'] == [-90, 90]


def test_date_discovery_rejects_duplicates_gaps_and_missing_endpoints(tmp_path):
    from scripts.preprocess_earth_reanalysis import discover_merra2_files
    root = make_raw(tmp_path / 'raw')
    first = next((root / 'slv').glob('*20201231*'))
    duplicate = first.with_name(first.name.replace('401', '400'))
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match='duplicate'):
        discover_merra2_files(root)
    root2 = make_raw(tmp_path / 'raw2')
    with pytest.raises(ValueError, match='start'):
        preprocess_merra2(root2, tmp_path / 'out', start_date='2020-12-30',
                          train_end='2020-12-31', validation_end='2021-01-01')


def test_preprocess_keeps_daily_package_compatible_while_registry_retires_it(tmp_path):
    from services.dataset_registry import DatasetRegistry
    from services.dataset_identity import DatasetRequestError
    output = preprocess_merra2(make_raw(tmp_path / 'raw'), tmp_path / 'merra2_daily_v2',
                              train_end='2020-12-31', validation_end='2021-01-01')
    manifest = json.loads(output.with_name('manifest.json').read_text(encoding='utf-8'))
    assert output.is_file()
    assert manifest['source_files']
    assert all('sha256' in item for item in manifest['source_files'])
    registry = DatasetRegistry(output.parent, earth_dataset_id='earth_merra2_daily_v2')
    with pytest.raises(DatasetRequestError) as error:
        registry.get_dataset('earth_merra2_daily_v2')
    assert error.value.code == 'dataset_retired'
    assert error.value.status_code == 409


@pytest.mark.parametrize('mutation', ['manifest_bounds', 'cell_gap', 'normalization'])
def test_release_rejects_unverified_bounds_and_train_statistics(tmp_path, mutation):
    from netCDF4 import Dataset
    from services.earth_dataset_metadata import file_sha256, read_earth_metadata, EarthPackageError
    output = preprocess_merra2(make_raw(tmp_path / 'raw'), tmp_path / 'merra2_daily_v2',
                              train_end='2020-12-31', validation_end='2021-01-01')
    path = output.with_name('manifest.json')
    manifest = json.loads(path.read_text())
    if mutation == 'manifest_bounds':
        manifest['cell_bounds']['latitude'] = [-80, 80]
    elif mutation == 'normalization':
        manifest['normalization']['train_stats']['TO3']['mean'] = 999
    else:
        with Dataset(output, 'r+') as ds:
            ds['lat_bounds'][10, 1] -= 1
        manifest['data_sha256'] = file_sha256(output)
        manifest['data_bytes'] = output.stat().st_size
    path.write_text(json.dumps(manifest))
    with pytest.raises(EarthPackageError):
        read_earth_metadata(output.parent, expected_manifest_sha256=file_sha256(path),
                            expected_data_sha256=file_sha256(output), dataset_id='earth_merra2_daily_v2', dataset_version='v2')
