"""Scientific alignment/semantics and immutable per-sample terrain tests."""
from dataclasses import replace
import os
from pathlib import Path
import struct

import netCDF4
import numpy as np
import pytest
import torch

from services.earth_topography import (
    ASSET_SCHEMA, OCEAN_POLICY, ICE_POLICY, REGRID_ALGORITHM,
    load_earth_topography, prepare_topography_batch, grid_identity, validate_identity,
)
from services.netcdf_read_lock import netcdf_read_lock
from scripts.build_earth_topography_asset import aggregate_windows, publish_asset, GSHHGLandMask


def grid(rows=4, columns=8, *, descending=False, longitude_zero=False):
    lat_edges = np.linspace(-90., 90., rows + 1)
    lon_edges = np.linspace(-180., 180., columns + 1)
    lat = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon = (lon_edges[:-1] + lon_edges[1:]) / 2
    lb = np.column_stack((lat_edges[:-1], lat_edges[1:]))
    ob = np.column_stack((lon_edges[:-1], lon_edges[1:]))
    if descending:
        lat, lb = lat[::-1], lb[::-1]
    if longitude_zero:
        lon, ob = np.mod(lon, 360), np.mod(ob, 360)
        order = np.argsort(lon)
        lon, ob = lon[order], ob[order]
    return lat, lon, lb, ob


def metadata():
    return {
        "schema": ASSET_SCHEMA, "source": "NOAA NCEI ETOPO 2022", "product": "ETOPO_2022_v1",
        "version": "2022_v1", "variant": "ice_surface", "source_resolution_arc_seconds": 60,
        "source_files": [{"name": "source.nc", "bytes": 100, "sha256": "a" * 64,
                          "url": "https://www.ngdc.noaa.gov/source.nc"}],
        "land_mask": {"source": "independent coastline", "version": "2.3.7", "sha256": "b" * 64,
                      "url": "https://www.ncei.noaa.gov/coast.b",
                      "method": "independent_binary_land_at_source_pixel_centres_v1"},
        "horizontal_datum": "WGS84 EPSG:4326", "vertical_datum": "EGM2008 geoid EPSG:3855",
        "ocean_policy": OCEAN_POLICY, "ice_policy": ICE_POLICY, "regrid_algorithm": REGRID_ALGORITHM,
    }


def asset(tmp_path, *, rows=4, columns=8):
    g = grid(rows, columns)
    values = (np.arange(rows)[:, None] * 100 + np.arange(columns)[None, :] * 7 - 20).astype("float32")
    return publish_asset(tmp_path / "terrain.nc", values, g, metadata()), g


def test_non_symmetric_descending_latitude_and_zero_longitude_align():
    source = grid(4, 8, descending=True, longitude_zero=True)
    values = (source[0][:, None] * 3 + np.cos(np.deg2rad(source[1]))[None, :] * 47
              + np.sin(np.deg2rad(source[1]))[None, :] * 13).astype("float32")
    target = grid(4, 8)
    result = aggregate_windows(lambda s, e: values[s:e], lambda lat, lon: np.ones((len(lat), len(lon)), bool), source, target, chunk_rows=1)
    expected = (target[0][:, None] * 3 + np.cos(np.deg2rad(target[1]))[None, :] * 47
                + np.sin(np.deg2rad(target[1]))[None, :] * 13).astype("float32")
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-5)


def test_dateline_wrap_polar_cover_and_spherical_area():
    source, target = grid(4, 8), grid(2, 4)
    # Rotate target longitude centres so first cell crosses the date line.
    target = (target[0], np.array([-180., -90., 0., 90.]), target[2],
              np.array([[-225., -135.], [-135., -45.], [-45., 45.], [45., 135.]]))
    values = np.arange(32, dtype="float32").reshape(4, 8)
    result = aggregate_windows(lambda s, e: values[s:e], lambda lat, lon: np.ones((len(lat), len(lon)), bool), source, target, chunk_rows=3)
    row_weights = np.diff(np.sin(np.deg2rad(np.array([-90., -45., 0.]))))
    expected_first = ((values[0, [7, 0]].mean() * row_weights[0]
                       + values[1, [7, 0]].mean() * row_weights[1]) / row_weights.sum())
    assert result[0, 0] == pytest.approx(expected_first)
    assert not np.isclose(result[0, 0], values[:2][:, [7, 0]].mean())
    assert np.isfinite(result).all()


def test_ocean_zero_negative_land_and_coastal_mixture():
    source, target = grid(4, 8), grid(2, 4)
    values = np.full((4, 8), -1000., dtype="float32")
    values[:, 0], values[:, 1] = -80., 400.
    def mask(lat, lon):
        result = np.zeros((len(lat), len(lon)), bool)
        result[:, 0] = True
        return result
    result = aggregate_windows(lambda s, e: values[s:e], mask, source, target, chunk_rows=2)
    np.testing.assert_allclose(result[:, 0], -40.)
    np.testing.assert_array_equal(result[:, 1:], 0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -99999.])
def test_nodata_fails_even_at_masked_ocean(invalid):
    g = grid()
    values = np.zeros((4, 8), dtype="float32"); values[1, 3] = invalid
    with pytest.raises(ValueError, match="NoData"):
        aggregate_windows(lambda s, e: values[s:e], lambda lat, lon: np.zeros((len(lat), len(lon)), bool), g, g)


def test_half_cell_shift_and_missing_polar_coverage_fail():
    g = grid()
    shifted = (g[0], g[1] + 1., g[2], g[3])
    with pytest.raises(ValueError, match="half-cell"):
        grid_identity(*shifted)
    missing = (g[0][1:], g[1], g[2][1:], g[3])
    with pytest.raises(ValueError, match="pole-to-pole"):
        grid_identity(*missing)


def test_runtime_asset_exact_grid_frozen_identity_and_no_mtime_proof(tmp_path):
    original, g = asset(tmp_path)
    path = tmp_path / "terrain.nc"
    assert load_earth_topography(*g, path=path, expected_identity=original.identity).identity == original.identity
    before = path.stat()
    # Same layout/size and restored mtime does not reuse a cached proof.
    with netcdf_read_lock(), netCDF4.Dataset(str(path), "r+") as dataset:
        dataset.variables["elevation"][0, 0] += 1
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="identity is corrupted"):
        load_earth_topography(*g, path=path, expected_identity=original.identity)


@pytest.mark.parametrize("field,value", [("units", "kilometer"), ("units", "m")])
def test_runtime_wrong_units_fail(tmp_path, field, value):
    _, g = asset(tmp_path)
    with netcdf_read_lock(), netCDF4.Dataset(str(tmp_path / "terrain.nc"), "r+") as dataset:
        dataset.variables["elevation"].setncattr(field, value)
    with pytest.raises(ValueError, match="units"):
        load_earth_topography(*g, path=tmp_path / "terrain.nc")


def test_runtime_wrong_dtype_fails(tmp_path):
    _, g = asset(tmp_path)
    path = tmp_path / "wrong.nc"
    with netcdf_read_lock(), netCDF4.Dataset(str(path), "w") as dataset:
        dataset.schema = ASSET_SCHEMA
        dataset.createDimension("latitude", 4); dataset.createDimension("longitude", 8); dataset.createDimension("bounds", 2)
        for name, dimensions in (("elevation", ("latitude", "longitude")),
                ("latitude", ("latitude",)), ("longitude", ("longitude",)),
                ("lat_bounds", ("latitude", "bounds")), ("lon_bounds", ("longitude", "bounds"))):
            dataset.createVariable(name, "f8", dimensions)
    with pytest.raises(ValueError, match="float32"):
        load_earth_topography(*g, path=path)


def test_runtime_missing_asset_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ASTRAATMOS_EARTH_TOPOGRAPHY_PATH"):
        load_earth_topography(*grid(), path=tmp_path / "missing.nc")


def test_v1_actual_different_sample_positions_shuffle_and_final_batch(tmp_path):
    terrain, _ = asset(tmp_path)
    x = torch.zeros(3, 2, 1, 2, 2, dtype=torch.float32)
    slices = [(slice(2, 4), slice(4, 6)), (slice(0, 2), slice(0, 2)), (slice(0, 2), slice(6, 8))]
    prepared = prepare_topography_batch(terrain, x, slices)
    for index, pair in enumerate(slices):
        np.testing.assert_array_equal(prepared[index, 0], terrain.elevation[pair])
    order = [2, 0, 1]
    torch.testing.assert_close(prepare_topography_batch(terrain, x[order], [slices[i] for i in order]), prepared[order])
    torch.testing.assert_close(prepare_topography_batch(terrain, x[:1], slices[:1]), prepared[:1])
    with pytest.raises(ValueError, match="actual per-sample"):
        prepare_topography_batch(terrain, x)


def test_v2_full_grid_storage_independence_and_gradient(tmp_path):
    terrain, _ = asset(tmp_path)
    x = torch.zeros(2, 2, 1, 4, 8, dtype=torch.float32)
    prepared = prepare_topography_batch(terrain, x)
    original = prepared[1].clone()
    prepared[0].add_(1000)
    torch.testing.assert_close(prepared[1], original)
    torch.testing.assert_close(prepare_topography_batch(terrain, x)[0], original)
    assert prepared.dtype == torch.float32 and prepared.device == x.device
    scale = torch.nn.Parameter(torch.tensor(.01))
    loss = (x[:, -1] + scale * prepare_topography_batch(terrain, x)).square().mean()
    loss.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad) and scale.grad != 0


def test_identity_missing_unknown_algorithm_and_grid_replacement(tmp_path):
    terrain, g = asset(tmp_path)
    bad = dict(terrain.identity); bad.pop("land_mask")
    with pytest.raises(ValueError, match="missing"):
        validate_identity(bad)
    bad = dict(terrain.identity); bad["regrid_algorithm"] = "unknown"
    with pytest.raises(ValueError, match="regrid_algorithm"):
        validate_identity(bad)
    bad = dict(terrain.identity); bad["asset_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frozen identity"):
        load_earth_topography(*g, path=tmp_path / "terrain.nc", expected_identity=bad)
    reordered = (g[0][::-1], g[1], g[2][::-1], g[3])
    with pytest.raises(ValueError, match="exactly match"):
        load_earth_topography(*reordered, path=tmp_path / "terrain.nc")


def _polygon(stream, identifier, level, points, *, polar=False):
    points = np.asarray(points, dtype=np.float64)
    west, east = (-180000000, 180000000) if polar else (int(points[:, 0].min() * 1e6), int(points[:, 0].max() * 1e6))
    south = -90000000 if polar else int(points[:, 1].min() * 1e6)
    header = [identifier, len(points), level | (15 << 8), west, east, south,
              int(points[:, 1].max() * 1e6), 1, 1, -1, identifier]
    stream.write(struct.pack(">11i", *header)); stream.write((points * 1e6).astype(">i4").tobytes())


def test_independent_coastline_negative_land_dateline_and_antarctic_pole(tmp_path):
    path = tmp_path / "coast.b"
    with path.open("wb") as stream:
        _polygon(stream, 0, 1, [[170, -10], [190, -10], [190, 10], [170, 10], [170, -10]])
        _polygon(stream, 1, 5, [[180, -75], [0, -70], [-180, -75]], polar=True)
        _polygon(stream, 2, 6, [[180, -76], [0, -71], [-180, -76]], polar=True)
    mask = GSHHGLandMask(path)
    values = mask(np.array([-89., 0., 40.]), np.array([-179., 0., 179.]))
    np.testing.assert_array_equal(values, [[True, True, True], [True, False, True], [False, False, False]])


def test_failed_publication_keeps_valid_previous_asset(tmp_path):
    terrain, g = asset(tmp_path)
    original = (tmp_path / "terrain.nc").read_bytes()
    bad = terrain.elevation.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        publish_asset(tmp_path / "terrain.nc", bad, g, metadata())
    assert (tmp_path / "terrain.nc").read_bytes() == original


def test_new_earth_netcdf_read_points_share_process_lock():
    from scripts import build_earth_topography_asset as builder
    from services import earth_topography as runtime
    assert builder.netcdf_read_lock is runtime.netcdf_read_lock is netcdf_read_lock
    for module in (builder, runtime):
        source = Path(module.__file__).read_text(encoding="utf-8")
        lines = source.splitlines()
        constructors = [line for line in lines if "netCDF4.Dataset(" in line]
        assert constructors and all("with netcdf_read_lock()," in line for line in constructors)


def test_builder_unknown_bytes_cannot_be_proven_by_name_or_supplied_hash(tmp_path):
    from scripts import build_earth_topography_asset as builder
    # Surface and bed use the same structural metadata. Claiming the official
    # surface name and its own SHA does not authenticate an arbitrary file.
    source = tmp_path / "ETOPO_2022_v1_60s_N90W180_surface.nc"
    source.write_bytes(b"unknown bedrock product with surface filename")
    with pytest.raises(ValueError, match="version/variant is unverified"):
        builder.verify_etopo_source(source, expected_sha256=builder.sha256_file(source))


@pytest.mark.parametrize("field,value", [("variant", "bedrock"), ("version", "2022_v2"),
                                         ("product", "ETOPO_2024_v1")])
def test_builder_rejects_wrong_product_variant_or_version(tmp_path, monkeypatch, field, value):
    from scripts import build_earth_topography_asset as builder
    source = tmp_path / "source.nc"
    source.write_bytes(b"synthetic provenance test only")
    digest = builder.sha256_file(source)
    record = dict(next(iter(builder.VERIFIED_ETOPO_SOURCES.values())))
    record.update(bytes=source.stat().st_size)
    record[field] = value
    monkeypatch.setattr(builder, "VERIFIED_ETOPO_SOURCES", {digest: record})
    with pytest.raises(ValueError, match="version/variant must be"):
        builder.verify_etopo_source(source)


def test_builder_reviewed_content_accepts_renamed_file_without_trusting_basename(tmp_path, monkeypatch):
    from scripts import build_earth_topography_asset as builder
    source = tmp_path / "renamed-local-copy.nc"
    source.write_bytes(b"synthetic reviewed bytes; not an official source")
    digest = builder.sha256_file(source)
    record = dict(next(iter(builder.VERIFIED_ETOPO_SOURCES.values())))
    record["bytes"] = source.stat().st_size
    monkeypatch.setattr(builder, "VERIFIED_ETOPO_SOURCES", {digest: record})
    actual, identity = builder.verify_etopo_source(source, expected_sha256=digest)
    assert actual == digest and identity == record
    # Returned metadata cannot alter the registry used by subsequent checks.
    identity["variant"] = "bedrock"
    assert builder.verify_etopo_source(source)[1]["variant"] == "ice_surface"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        builder.verify_etopo_source(source, expected_sha256="f" * 64)
    with pytest.raises(ValueError, match="resolution differs"):
        builder.verify_etopo_source(source, resolution=30)
    record["bytes"] += 1
    with pytest.raises(ValueError, match="byte length differs"):
        builder.verify_etopo_source(source)
    record["bytes"] -= 1
    source.write_bytes(b"modified" + source.read_bytes()[8:])
    with pytest.raises(ValueError, match="version/variant is unverified"):
        builder.verify_etopo_source(source)


def test_builder_unverified_source_keeps_existing_valid_asset(tmp_path, monkeypatch):
    from scripts import build_earth_topography_asset as builder
    existing, g = asset(tmp_path)
    output = tmp_path / "terrain.nc"
    original = output.read_bytes()
    source = tmp_path / "ETOPO_2022_v1_60s_N90W180_surface.nc"
    source.write_bytes(b"unverified source")
    def unexpected_mask(*args):
        pytest.fail("an unverified source must fail before reading the mask")
    monkeypatch.setattr(builder, "GSHHGLandMask", unexpected_mask)
    with pytest.raises(ValueError, match="version/variant is unverified"):
        builder.build(source, tmp_path / "missing-mask.b", output, g)
    assert output.read_bytes() == original
    assert not Path(str(output) + ".build.lock").exists()
    assert load_earth_topography(*g, path=output).identity == existing.identity


def test_builder_real_reviewed_60_second_surface_source():
    from scripts import build_earth_topography_asset as builder
    # Optional local evidence; the normal suite remains offline and portable.
    # This does not download, mutate or process the high-resolution source.
    source = Path(os.environ.get("ASTRAATMOS_ETOPO_TEST_SOURCE",
        str(Path(__file__).resolve().parents[4] / "earth-terrain-cache" /
            "ETOPO_2022_v1_60s_N90W180_surface.nc")))
    if not source.is_file():
        pytest.skip("reviewed official ETOPO source is not available locally")
    digest, identity = builder.verify_etopo_source(source)
    assert digest == "74bbc5c23e85188d93f0beba8116babf69e1a8f160e8dc9f053bf8df01b197b6"
    assert identity["version"] == "2022_v1" and identity["variant"] == "ice_surface"
    assert identity["source_resolution_arc_seconds"] == 60
    with netcdf_read_lock(), netCDF4.Dataset(str(source)) as dataset:
        assert dataset.variables["z"].shape == (10800, 21600)
        assert dataset.variables["z"].vert_crs_epsg == "EPSG:3855"
