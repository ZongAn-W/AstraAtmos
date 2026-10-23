"""Build a small Earth atmospheric lower-boundary asset from local NOAA ETOPO.

No dependency installs or application-time downloads. Source cells are masked
using independent GSHHG coastline polygons, then conservatively aggregated by
spherical overlap area. Every NetCDF read shares the service process lock.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import struct
import sys
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netCDF4
import numpy as np
from scipy.sparse import csr_matrix

from config import EARTH_MERRA2_3HOURLY_DIR, EARTH_TOPOGRAPHY_PATH
from services.netcdf_read_lock import netcdf_read_lock
from services.earth_topography import (
    ASSET_SCHEMA, REGRID_ALGORITHM, OCEAN_POLICY, ICE_POLICY, canonical_json,
    sha256_file, validate_grid, grid_identity, asset_content_identity,
    load_earth_topography, release_topography_grid,
)

ETOPO_URL = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/{resolution}s/{resolution}s_surface_elev_netcdf/ETOPO_2022_v1_{resolution}s_N90W180_surface.nc"
GSHHG_URL = "https://www.ncei.noaa.gov/data/oceans/archive/arc0234/0304143/1.1/data/0-data/gshhs_i.b"

# The official NetCDF carries CRS/registration metadata, but no version or
# ice-surface/bedrock identifier. Those products share the same grid and datum;
# a basename or a caller-supplied digest cannot prove which one was supplied.
# Pin the complete bytes of the Ice Surface file obtained from NOAA's HTTPS
# endpoint and checked against its THREDDS catalogue on 2026-10-13. This is a
# locally verified content pin, NOT a checksum published by NOAA. Reviewing a
# new official product requires adding its independently obtained bytes here.
VERIFIED_ETOPO_SOURCES = {
    "74bbc5c23e85188d93f0beba8116babf69e1a8f160e8dc9f053bf8df01b197b6": {
        "bytes": 478290125,
        "product": "ETOPO_2022_v1",
        "version": "2022_v1",
        "variant": "ice_surface",
        "source_resolution_arc_seconds": 60,
        "url": ETOPO_URL.format(resolution=60),
    },
}


def verify_etopo_source(source_path, *, resolution=60, expected_sha256=None):
    """Bind actual file contents to a reviewed official version and variant.

    This verification is offline. Unknown bytes fail closed even when the file
    is renamed to the official basename or the caller supplies its own hash.
    Only the reviewed 60-second release is currently accepted; 15/30-second or
    repacked NetCDF products need a separately reviewed content pin first.
    """
    source_path = Path(source_path)
    digest = sha256_file(source_path)
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("ETOPO source expected/official SHA-256 mismatch")
    identity = VERIFIED_ETOPO_SOURCES.get(digest)
    if identity is None:
        raise ValueError("ETOPO source version/variant is unverified: actual SHA-256 is not a reviewed NOAA Ice Surface product; filenames and supplied checksums are not provenance")
    if (identity["product"] != "ETOPO_2022_v1" or identity["version"] != "2022_v1"
            or identity["variant"] != "ice_surface"):
        raise ValueError("ETOPO source version/variant must be reviewed ETOPO 2022 v1 Ice Surface")
    if source_path.stat().st_size != identity["bytes"]:
        raise ValueError("ETOPO source byte length differs from reviewed official product")
    if type(resolution) is not int or resolution != identity["source_resolution_arc_seconds"]:
        raise ValueError("ETOPO source resolution differs from reviewed official product")
    return digest, dict(identity)


@contextmanager
def exclusive_build_lock(output):
    lock = Path(str(output) + ".build.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"Earth topography build lock already exists: {lock}; inspect its process before removing this one file") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()  # One exact lock file, never a recursive/batch deletion.


def download_atomic(url, destination, *, expected_sha256=None):
    """HTTPS download to one temporary path; preserve valid existing files."""
    destination = Path(destination)
    if destination.is_file():
        digest = sha256_file(destination)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Existing source SHA-256 differs: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.part")
    # NOAA serves byte ranges; bounded requests tolerate interrupted transfers
    # while every byte is still covered by the final SHA-256 and length check.
    probe = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(probe, timeout=30) as response:
        range_header = response.headers.get("Content-Range", "")
        total = int(range_header.rsplit("/", 1)[1]) if response.status == 206 and "/" in range_header else None
    with temporary.open("wb") as stream:
        if total is None:
            with urllib.request.urlopen(url, timeout=90) as response:
                expected_length = response.headers.get("Content-Length")
                for block in iter(lambda: response.read(64 * 1024), b""):
                    stream.write(block)
        else:
            expected_length = total
            stalled = 0
            while stream.tell() < total:
                offset = stream.tell()
                last = min(total - 1, offset + 2 * 1024 * 1024 - 1)
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{last}"})
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        actual_range = response.headers.get("Content-Range", "")
                        if response.status != 206 or actual_range != f"bytes {offset}-{last}/{total}":
                            raise ValueError("Earth terrain source server returned unexpected byte range")
                        for block in iter(lambda: response.read(64 * 1024), b""):
                            stream.write(block)
                except (OSError, http.client.IncompleteRead) as exc:
                    if isinstance(exc, http.client.IncompleteRead):
                        stream.write(exc.partial)
                    stalled = stalled + 1 if stream.tell() == offset else 0
                    if stalled >= 5:
                        raise ValueError("Earth terrain HTTPS transfer stalled; partial download preserved") from exc
                if stream.tell() > last + 1:
                    raise ValueError("Earth terrain server sent more bytes than requested")
        stream.flush()
        os.fsync(stream.fileno())
    if expected_length is not None and temporary.stat().st_size != int(expected_length):
        raise ValueError("Earth topography download is truncated")
    if expected_sha256 is not None and sha256_file(temporary) != expected_sha256:
        raise ValueError("Earth topography official/expected source checksum differs")
    os.replace(temporary, destination)
    return destination


def _cell_bounds_from_registered_source(coordinates, *, latitude=False):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 1 or len(coordinates) < 2 or not np.isfinite(coordinates).all():
        raise ValueError("ETOPO source coordinates must be finite one-dimensional arrays")
    steps = np.diff(coordinates)
    if not np.allclose(steps, steps[0], rtol=0, atol=1e-9) or steps[0] == 0:
        raise ValueError("ETOPO source coordinates must be uniformly monotonic")
    half = abs(float(steps[0])) / 2
    result = np.column_stack((coordinates - half, coordinates + half))
    if latitude and (not np.isclose(result.min(), -90, rtol=0, atol=1e-8)
                     or not np.isclose(result.max(), 90, rtol=0, atol=1e-8)):
        raise ValueError("ETOPO source pixel-centre latitude coverage is incomplete")
    if latitude:
        result = np.clip(result, -90, 90)
    return result


def latitude_overlap(target_bounds, source_bounds):
    target = np.sort(np.asarray(target_bounds, dtype=np.float64), axis=1)
    source = np.sort(np.asarray(source_bounds, dtype=np.float64), axis=1)
    rows, cols, values = [], [], []
    for row, (lo, hi) in enumerate(target):
        chosen = np.flatnonzero((source[:, 1] > lo + 1e-10) & (source[:, 0] < hi - 1e-10))
        lower, upper = np.maximum(lo, source[chosen, 0]), np.minimum(hi, source[chosen, 1])
        area = np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower))
        rows.extend([row] * len(chosen)); cols.extend(chosen.tolist()); values.extend(area.tolist())
    result = csr_matrix((values, (rows, cols)), shape=(len(target), len(source)))
    expected = np.sin(np.deg2rad(target[:, 1])) - np.sin(np.deg2rad(target[:, 0]))
    if not np.allclose(np.asarray(result.sum(axis=1)).ravel(), expected, rtol=1e-9, atol=1e-13):
        raise ValueError("ETOPO latitude overlap has incomplete coverage")
    return result


def longitude_overlap(target_bounds, source_bounds):
    target, source = np.asarray(target_bounds), np.asarray(source_bounds)
    source_start = np.mod(source[:, 0], 360.)
    source_width = np.mod(source[:, 1] - source[:, 0], 360.)
    rows, cols, values = [], [], []
    for row, (left, right) in enumerate(target):
        left = float(left) % 360; width = float((right - target[row, 0]) % 360)
        accumulated = np.zeros(len(source), dtype=np.float64)
        for offset in (-360., 0., 360.):
            lower = np.maximum(left, source_start + offset)
            upper = np.minimum(left + width, source_start + source_width + offset)
            accumulated += np.maximum(upper - lower, 0)
        chosen = np.flatnonzero(accumulated > 1e-10)
        rows.extend([row] * len(chosen)); cols.extend(chosen.tolist()); values.extend(np.deg2rad(accumulated[chosen]).tolist())
    result = csr_matrix((values, (rows, cols)), shape=(len(target), len(source)))
    expected = np.deg2rad(np.mod(target[:, 1] - target[:, 0], 360.))
    if not np.allclose(np.asarray(result.sum(axis=1)).ravel(), expected, rtol=1e-9, atol=1e-12):
        raise ValueError("ETOPO longitude overlap has incomplete coverage")
    return result


class GSHHGLandMask:
    """Independent GSHHG 2.3.7 intermediate-resolution coastline rasterizer.

    Level 1 + level 5 ice-front are land; level 2/4 inland water is zero, while
    level 3 lake islands are land again. Zero inland water is an explicit
    approximation, avoiding lake bathymetry being used as atmospheric surface.
    Polygon crossings use actual vertices rather than relief sign.
    """
    def __init__(self, path):
        self.polygons = []
        levels = {}
        with Path(path).open("rb") as stream:
            while header := stream.read(44):
                if len(header) != 44:
                    raise ValueError("GSHHG binary file has truncated header")
                values = struct.unpack(">11i", header)
                _, count, flag, west, east, south, north, *_ = values
                level, version = flag & 255, (flag >> 8) & 255
                if version != 15 or level not in (1, 2, 3, 4, 5, 6) or count < 3 or count > 10000000:
                    raise ValueError("GSHHG binary header/version/level is invalid")
                raw = stream.read(count * 8)
                if len(raw) != count * 8:
                    raise ValueError("GSHHG binary file has truncated polygon")
                levels[level] = levels.get(level, 0) + 1
                if level == 6:
                    continue
                points = np.frombuffer(raw, dtype=">i4").reshape(count, 2).astype(np.float64) / 1e6
                # Antarctica's open coast runs +180 to -180; its region is
                # closed through the south pole, not a chord at coastal lat.
                # The binary header explicitly marks south=-90 and 360 width.
                polar = south == -90000000 and east - west == 360000000
                if polar:
                    points = np.concatenate((points,
                        [[points[-1, 0], -90.], [points[0, 0], -90.]], points[:1]), axis=0)
                else:
                    # Unwrap using contiguous real vertices. Greenwich and
                    # date-line polygons cannot be shifted independently at
                    # 180 degrees: doing so introduces a false long chord.
                    points[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(points[:, 0])))
                if not np.array_equal(points[0], points[-1]):
                    points = np.concatenate((points, points[:1]), axis=0)
                if not np.isfinite(points).all() or np.any(np.abs(points[:, 1]) > 90):
                    raise ValueError("GSHHG polygon coordinates are invalid")
                hierarchy = 1 if level == 5 else level
                self.polygons.append((hierarchy, float(points[:, 1].min()), float(points[:, 1].max()), points))
        if not self.polygons or 1 not in levels or 5 not in levels or 6 not in levels:
            raise ValueError("GSHHG mask does not contain complete land and Antarctic coastlines")
        self.polygons.sort(key=lambda polygon: polygon[0])

    def __call__(self, latitude, longitude):
        latitude, longitude = np.asarray(latitude), np.asarray(longitude)
        mask = np.zeros((len(latitude), len(longitude)), dtype=bool)
        # Loop bounded source row windows; crossings scale with polygon edges,
        # not with 233 million points times coastline vertex count.
        for level, lower, upper, points in self.polygons:
            rows = np.flatnonzero((latitude >= lower) & (latitude <= upper))
            if not len(rows):
                continue
            x0, y0, x1, y1 = points[:-1, 0], points[:-1, 1], points[1:, 0], points[1:, 1]
            min_lon = points[:, 0].min()
            candidate_lon = np.mod(longitude - min_lon, 360.) + min_lon
            lon_order = np.argsort(candidate_lon)
            sorted_lon = candidate_lon[lon_order]
            for row in rows:
                y = latitude[row]
                edges = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
                intersections = np.sort(x0[edges] + (y - y0[edges]) * (x1[edges] - x0[edges]) / (y1[edges] - y0[edges]))
                if len(intersections) % 2:
                    raise ValueError("GSHHG coastline scanline has unmatched crossings")
                for a, b in intersections.reshape(-1, 2):
                    first, last = np.searchsorted(sorted_lon, (a, b), side="left")
                    mask[row, lon_order[first:last]] = bool(level % 2)
        return mask


def aggregate_windows(read_window, land_mask, source_grid, target_grid, *, chunk_rows=90, progress=None):
    """Conservative spherical means; all source cells must be valid, even sea."""
    source, target = validate_grid(*source_grid), validate_grid(*target_grid)
    lat_weights, lon_weights = latitude_overlap(target[2], source[2]), longitude_overlap(target[3], source[3])
    expected_area = np.asarray(lat_weights.sum(axis=1)) * np.asarray(lon_weights.sum(axis=1)).T
    sums = np.zeros(expected_area.shape, dtype=np.float64)
    coverage = np.zeros(expected_area.shape, dtype=np.float64)
    for start in range(0, len(source[0]), chunk_rows):
        stop = min(start + chunk_rows, len(source[0]))
        raw = np.asanyarray(read_window(start, stop))
        if np.ma.isMaskedArray(raw):
            raw = raw.filled(np.nan)
        raw = np.asarray(raw)
        if raw.shape != (stop - start, len(source[1])) or not np.isfinite(raw).all() or np.any(np.abs(raw) >= 99999):
            raise ValueError(f"ETOPO source NoData, incomplete window or invalid relief at rows {start}:{stop}")
        mask = np.asarray(land_mask(source[0][start:stop], source[1]))
        if mask.dtype != bool or mask.shape != raw.shape:
            raise ValueError("Independent land mask must provide a complete boolean source-pixel window")
        lower_boundary = np.where(mask, raw, np.float32(0))
        # Sparse overlaps avoid a dense 480 x 21,600 float64 weight copy.
        longitude_means = lon_weights.dot(lower_boundary.T).T
        sums += lat_weights[:, start:stop].dot(longitude_means)
        coverage += lat_weights[:, start:stop].dot(np.ones((stop - start, len(source[1])), dtype=np.float32) @ lon_weights.T)
        if progress:
            progress(f"aggregated source rows {stop}/{len(source[0])}")
    if not np.allclose(coverage, expected_area, rtol=1e-8, atol=1e-13) or np.any(coverage <= 0):
        raise ValueError("ETOPO aggregation coverage is incomplete; no fill or fallback is permitted")
    output = (sums / expected_area).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("ETOPO aggregation produced non-finite elevation")
    return output


def publish_asset(output, elevation, target_grid, metadata):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid = validate_grid(*target_grid)
    temporary = output.with_name(output.name + f".{uuid.uuid4().hex}.partial")
    with netcdf_read_lock(), netCDF4.Dataset(str(temporary), "w", format="NETCDF4") as dataset:
        dataset.createDimension("latitude", len(grid[0])); dataset.createDimension("longitude", len(grid[1])); dataset.createDimension("bounds", 2)
        for name, values, dimensions, units in (
                ("latitude", grid[0], ("latitude",), "degrees_north"),
                ("longitude", grid[1], ("longitude",), "degrees_east"),
                ("lat_bounds", grid[2], ("latitude", "bounds"), "degrees_north"),
                ("lon_bounds", grid[3], ("longitude", "bounds"), "degrees_east")):
            variable = dataset.createVariable(name, "f8", dimensions)
            variable.units = units; variable[:] = values
        variable = dataset.createVariable("elevation", "f4", ("latitude", "longitude"), zlib=True, complevel=6)
        variable.units = "meter"; variable.long_name = "atmospheric lower-boundary orography; source oceans zero"
        variable[:] = elevation
        dataset.schema = ASSET_SCHEMA
        dataset.provenance_json = canonical_json(metadata)
        dataset.target_grid_sha256 = grid_identity(*grid)
        dataset.asset_content_sha256 = asset_content_identity(elevation, grid, metadata)
    validated = load_earth_topography(*grid, path=temporary)
    os.replace(temporary, output)
    return load_earth_topography(*grid, path=output)


def build(source_path, mask_path, output, target_grid, *, resolution=60, source_sha256=None,
          mask_sha256=None, chunk_rows=90, progress=print):
    source_path, mask_path = Path(source_path), Path(mask_path)
    with exclusive_build_lock(output):
        before_source, source_identity = verify_etopo_source(
            source_path, resolution=resolution, expected_sha256=source_sha256)
        before_mask = sha256_file(mask_path)
        if mask_sha256 is not None and mask_sha256 != before_mask:
            raise ValueError("GSHHG expected/official SHA-256 mismatch")
        mask = GSHHGLandMask(mask_path)
        with netcdf_read_lock(), netCDF4.Dataset(str(source_path)) as dataset:
            if not {"lat", "lon", "z", "crs"}.issubset(dataset.variables):
                raise ValueError("Expected the official ETOPO Ice Surface NetCDF variables")
            variable = dataset.variables["z"]
            if (np.dtype(variable.dtype) != np.dtype("float32") or variable.dimensions != ("lat", "lon")
                    or getattr(variable, "units", "") != "meters"
                    or getattr(variable, "vert_crs_name", "") != "EGM2008"
                    or getattr(variable, "vert_crs_epsg", "") != "EPSG:3855"
                    or getattr(dataset, "GDAL_AREA_OR_POINT", "") != "Area" or getattr(dataset, "node_offset", None) != 1
                    or getattr(dataset.variables["lat"], "units", "") != "degrees_north"
                    or getattr(dataset.variables["lon"], "units", "") != "degrees_east"
                    or "WGS 84" not in getattr(dataset.variables["crs"], "spatial_ref", "")):
                raise ValueError("ETOPO source dtype/units/cell registration/horizontal or vertical datum is invalid")
            lat, lon = np.asarray(dataset.variables["lat"][:]), np.asarray(dataset.variables["lon"][:])
            expected_shape = (180 * 3600 // resolution, 360 * 3600 // resolution)
            if variable.shape != expected_shape or not np.isclose(abs(np.diff(lat)[0]) * 3600, resolution, rtol=0, atol=1e-6) or not np.isclose(abs(np.diff(lon)[0]) * 3600, resolution, rtol=0, atol=1e-6):
                raise ValueError("ETOPO source dimensions/resolution do not match the declared official product")
            source_grid = (lat, lon, _cell_bounds_from_registered_source(lat, latitude=True), _cell_bounds_from_registered_source(lon))
            variable.set_var_chunk_cache(size=128 * 1024 * 1024, nelems=1009, preemption=.75)
            if progress:
                progress(f"ETOPO {expected_shape} float32={int(np.prod(expected_shape))*4:,} uncompressed bytes; window {chunk_rows} rows={chunk_rows*len(lon)*4:,} bytes plus mask/overlaps; chunk cache=128 MiB")
            elevation = aggregate_windows(lambda start, stop: variable[start:stop, :], mask,
                                          source_grid, target_grid, chunk_rows=chunk_rows, progress=progress)
        if sha256_file(source_path) != before_source or sha256_file(mask_path) != before_mask:
            raise ValueError("Earth terrain source/mask changed during aggregation")
        metadata = {
            "schema": ASSET_SCHEMA, "source": "NOAA NCEI ETOPO 2022", "product": source_identity["product"],
            "version": source_identity["version"], "variant": source_identity["variant"],
            "source_resolution_arc_seconds": source_identity["source_resolution_arc_seconds"],
            "source_files": [{"name": source_path.name, "bytes": source_path.stat().st_size,
                              "sha256": before_source, "url": source_identity["url"]}],
            "land_mask": {"source": "GSHHG intermediate coastlines; land hierarchy 1/3 + Antarctic ice-front 5; ocean and inland water 2/4 zero",
                          "version": "2.3.7", "sha256": before_mask, "url": GSHHG_URL,
                          "method": "independent_binary_land_at_source_pixel_centres_v1"},
            "horizontal_datum": "WGS84 EPSG:4326", "vertical_datum": "EGM2008 geoid EPSG:3855",
            "ocean_policy": OCEAN_POLICY, "ice_policy": ICE_POLICY, "regrid_algorithm": REGRID_ALGORITHM,
        }
        # Rehash immediately before publication, even when size/mtime match.
        if sha256_file(source_path) != before_source or sha256_file(mask_path) != before_mask:
            raise ValueError("Earth terrain source/mask changed before publication")
        return publish_asset(output, elevation, target_grid, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local official ETOPO surface .nc")
    parser.add_argument("--land-mask", type=Path, help="local GSHHG 2.3.7 gshhs_i.b")
    parser.add_argument("--target-release", type=Path, default=EARTH_MERRA2_3HOURLY_DIR / "earth_merra2_3hourly.nc")
    parser.add_argument("--output", type=Path, default=EARTH_TOPOGRAPHY_PATH)
    parser.add_argument("--resolution", type=int, choices=(60,), default=60,
                        help="reviewed official source resolution in arc-seconds; new products require a reviewed content pin")
    parser.add_argument("--source-sha256"); parser.add_argument("--land-mask-sha256")
    parser.add_argument("--chunk-rows", type=int, default=90)
    parser.add_argument("--validate", "--validate-only", action="store_true", dest="validate")
    parser.add_argument("--download", action="store_true", help="explicit HTTPS source download; never used by runtime")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[4] / "earth-terrain-cache")
    args = parser.parse_args()
    grid = release_topography_grid(path=args.target_release)
    if args.validate:
        asset = load_earth_topography(*grid, path=args.output)
    else:
        if args.chunk_rows < 1 or args.chunk_rows > 1350:
            parser.error("--chunk-rows must be between 1 and 1350")
        source, mask = args.source, args.land_mask
        if args.download:
            source = source or args.cache_dir / f"ETOPO_2022_v1_{args.resolution}s_N90W180_surface.nc"
            mask = mask or args.cache_dir / "gshhs_i_2.3.7.b"
            source = download_atomic(ETOPO_URL.format(resolution=args.resolution), source, expected_sha256=args.source_sha256)
            mask = download_atomic(GSHHG_URL, mask, expected_sha256=args.land_mask_sha256)
        if source is None or mask is None:
            parser.error("provide --source and --land-mask, or explicitly request --download")
        asset = build(source, mask, args.output, grid, resolution=args.resolution,
                      source_sha256=args.source_sha256, mask_sha256=args.land_mask_sha256,
                      chunk_rows=args.chunk_rows)
    print(canonical_json({"output": str(args.output), "shape": list(asset.elevation.shape),
                          "minimum_meter": float(asset.elevation.min()), "maximum_meter": float(asset.elevation.max()),
                          "identity": asset.identity}))


if __name__ == "__main__":
    main()
