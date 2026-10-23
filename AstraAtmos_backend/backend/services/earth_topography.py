"""Earth ETOPO assets, content identities and shared per-sample terrain slicing.

No source download, interpolation, meteorological normalization or Mars code is
used here. The offline builder must already have aligned the asset to the exact
release coordinates and cell bounds. Every load hashes actual file bytes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import netCDF4
import numpy as np
import torch

from config import (EARTH_TOPOGRAPHY_PATH, EARTH_TOPOGRAPHY_5DEG_PATH,
                    EARTH_MERRA2_3HOURLY_DIR, EARTH_MERRA2_3HOURLY_5DEG_DIR)
from services.netcdf_read_lock import netcdf_read_lock

ASSET_SCHEMA = "astraatmos_earth_topography_v1"
REGRID_ALGORITHM = "spherical_cell_overlap_source_land_mask_v1"
OCEAN_POLICY = "independent_land_mask_source_centres_ocean_and_inland_water_zero_before_aggregation"
ICE_POLICY = "etopo_ice_surface_including_antarctic_ice_shelves"
IDENTITY_KEYS = frozenset({
    "schema", "asset_content_sha256", "asset_file_sha256", "target_grid_sha256",
    "source", "product", "version", "variant", "source_resolution_arc_seconds",
    "source_files", "land_mask", "horizontal_datum", "vertical_datum",
    "ocean_policy", "ice_policy", "regrid_algorithm",
})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> np.ndarray:
    value = np.asanyarray(value)
    if np.ma.isMaskedArray(value):
        value = value.filled(np.nan)
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"Earth topography {name} must contain only finite values")
    return array


def validate_grid(latitude: Any, longitude: Any, lat_bounds: Any, lon_bounds: Any):
    """Validate a global, cell-registered rectilinear grid without guessing it."""
    lat, lon = _finite(latitude, "latitude"), _finite(longitude, "longitude")
    lb, ob = _finite(lat_bounds, "lat_bounds"), _finite(lon_bounds, "lon_bounds")
    if (lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2
            or lb.shape != (len(lat), 2) or ob.shape != (len(lon), 2)):
        raise ValueError("Earth topography coordinates and bounds have invalid dimensions")
    if not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)):
        raise ValueError("Earth topography latitude must be strictly monotonic")
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lb) > 90):
        raise ValueError("Earth topography latitude exceeds the poles")
    # Preserve original row direction, allowing bounds endpoint direction too.
    low, high = np.min(lb, axis=1), np.max(lb, axis=1)
    order = np.argsort(low)
    if (np.any(high <= low) or not np.allclose(lat, (low + high) / 2, rtol=0, atol=1e-9)
            or not np.allclose(high[order][:-1], low[order][1:], rtol=0, atol=1e-9)
            or not np.isclose(low.min(), -90, rtol=0, atol=1e-9)
            or not np.isclose(high.max(), 90, rtol=0, atol=1e-9)):
        raise ValueError("Earth topography latitude bounds must cover pole-to-pole contiguous cells centred on latitude")
    widths = np.mod(ob[:, 1] - ob[:, 0], 360.)
    if np.any(widths <= 0) or np.any(widths >= 360):
        raise ValueError("Earth topography longitude bounds must have positive widths below 360 degrees")
    centres = np.mod(ob[:, 0] + widths / 2, 360.)
    if not np.allclose(np.mod(lon - centres + 180, 360) - 180, 0, rtol=0, atol=1e-9):
        raise ValueError("Earth topography longitude centres disagree with cell bounds (possible half-cell shift)")
    starts = np.mod(ob[:, 0], 360.)
    order = np.argsort(starts)
    cyclic_gaps = np.mod(np.roll(starts[order], -1) - starts[order], 360.)
    if (not np.isclose(widths.sum(), 360, rtol=0, atol=1e-8)
            or not np.allclose(widths[order], cyclic_gaps, rtol=0, atol=1e-9)):
        raise ValueError("Earth topography longitude bounds must cover one complete period without gaps or duplicates")
    return lat, lon, lb, ob


def grid_identity(latitude: Any, longitude: Any, lat_bounds: Any, lon_bounds: Any) -> str:
    grid = validate_grid(latitude, longitude, lat_bounds, lon_bounds)
    digest = hashlib.sha256(b"earth_topography_target_grid_v1\0")
    for name, array in zip(("latitude", "longitude", "lat_bounds", "lon_bounds"), grid):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(canonical_json(list(array.shape)).encode("ascii") + b"\0")
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def asset_content_identity(elevation: Any, grid: tuple, metadata: Mapping[str, Any]) -> str:
    """Semantic digest covers arrays + provenance, excludes any digest field.

    The raw NetCDF file hash is computed after publication, outside this digest;
    neither digest is embedded in its own input.
    """
    digest = hashlib.sha256(b"earth_topography_asset_content_v1\0")
    digest.update(canonical_json(dict(metadata)).encode("ascii") + b"\0")
    digest.update(grid_identity(*grid).encode("ascii") + b"\0")
    digest.update(np.asarray(elevation, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def validate_identity(identity: Any) -> dict:
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        raise ValueError("Earth topography identity has missing or unknown fields")
    for key in ("asset_content_sha256", "asset_file_sha256", "target_grid_sha256"):
        if not isinstance(identity[key], str) or not re.fullmatch(r"[a-f0-9]{64}", identity[key]):
            raise ValueError(f"Earth topography identity {key} must be a SHA-256 digest")
    expected = {
        "schema": ASSET_SCHEMA, "source": "NOAA NCEI ETOPO 2022", "product": "ETOPO_2022_v1",
        "version": "2022_v1", "variant": "ice_surface", "horizontal_datum": "WGS84 EPSG:4326",
        "vertical_datum": "EGM2008 geoid EPSG:3855", "ocean_policy": OCEAN_POLICY,
        "ice_policy": ICE_POLICY, "regrid_algorithm": REGRID_ALGORITHM,
    }
    for key, value in expected.items():
        if identity[key] != value:
            raise ValueError(f"Earth topography identity {key} expected {value!r}")
    if type(identity["source_resolution_arc_seconds"]) is not int or identity["source_resolution_arc_seconds"] not in (15, 30, 60):
        raise ValueError("Earth topography source resolution must be 15, 30 or 60 arc-seconds")
    files = identity["source_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Earth topography identity needs source_files")
    for entry in files:
        if (not isinstance(entry, dict) or set(entry) != {"name", "bytes", "sha256", "url"}
                or not isinstance(entry["name"], str) or not entry["name"]
                or type(entry["bytes"]) is not int or entry["bytes"] <= 0
                or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                or not isinstance(entry["url"], str) or not entry["url"].startswith("https://")):
            raise ValueError("Earth topography source file identity is invalid")
    mask = identity["land_mask"]
    if (not isinstance(mask, dict) or set(mask) != {"source", "version", "sha256", "url", "method"}
            or any(not isinstance(mask.get(key), str) or not mask[key] for key in mask)
            or not re.fullmatch(r"[a-f0-9]{64}", mask["sha256"])
            or not mask["url"].startswith("https://")):
        raise ValueError("Earth topography independent land-mask identity is invalid")
    if mask["method"] != "independent_binary_land_at_source_pixel_centres_v1":
        raise ValueError("Earth topography land-mask algorithm is unknown")
    return copy.deepcopy(identity)


@dataclass(frozen=True)
class EarthTopographyAsset:
    elevation: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    lat_bounds: np.ndarray
    lon_bounds: np.ndarray
    identity: dict


def release_topography_grid(release: Any = None, *, path: Any = None) -> tuple:
    """Read only actual release grid variables under the process-wide lock."""
    release_path = path if path is not None else getattr(release, "data_path", None)
    if release_path is None:
        release_path = EARTH_MERRA2_3HOURLY_DIR / "earth_merra2_3hourly.nc"
    with netcdf_read_lock(), netCDF4.Dataset(str(release_path)) as dataset:
        values = []
        for name, unit in (("lat", "degrees_north"), ("lon", "degrees_east"),
                           ("lat_bounds", None), ("lon_bounds", None)):
            if name not in dataset.variables:
                raise ValueError(f"Earth release missing actual topography target grid field {name}")
            variable = dataset.variables[name]
            if unit is not None and getattr(variable, "units", "") != unit:
                raise ValueError(f"Earth release {name} has invalid coordinate units")
            values.append(np.array(variable[:], dtype=np.float64, copy=True))
    grid = validate_grid(*values)
    if release is not None:
        for actual, expected in zip(grid[:2], (release.latitude, release.longitude)):
            if not np.array_equal(actual, np.asarray(expected)):
                raise ValueError("Earth release actual topography grid differs from verified release")
    return grid


def load_earth_topography(latitude: Any = None, longitude: Any = None,
                          lat_bounds: Any = None, lon_bounds: Any = None, *,
                          expected_identity: Any = None, path: Any = None,
                          release: Any = None, dataset_id=None) -> EarthTopographyAsset:
    lowres = (dataset_id == 'earth_merra2_3hourly_5deg_v1'
              or (release is not None and np.asarray(release.latitude).shape == (36,))
              or (latitude is not None and np.asarray(latitude).shape == (36,)))
    asset_path = Path((EARTH_TOPOGRAPHY_5DEG_PATH if lowres else EARTH_TOPOGRAPHY_PATH) if path is None else path)
    if not asset_path.is_file():
        raise FileNotFoundError(f"Earth ETOPO topography asset not found: {asset_path}; build offline and set ASTRAATMOS_EARTH_TOPOGRAPHY_PATH")
    supplied = (latitude, longitude, lat_bounds, lon_bounds)
    if all(value is None for value in supplied):
        target = (release_topography_grid(path=EARTH_MERRA2_3HOURLY_5DEG_DIR / 'earth_merra2_3hourly.nc')
                  if lowres and release is None else release_topography_grid(release))
    elif any(value is None for value in supplied):
        raise ValueError("Earth topography requires all actual target coordinates and cell bounds")
    else:
        target = validate_grid(*supplied)
    before = sha256_file(asset_path)
    try:
        with netcdf_read_lock(), netCDF4.Dataset(str(asset_path)) as dataset:
            if getattr(dataset, "schema", "") != ASSET_SCHEMA:
                raise ValueError("Earth topography asset schema is unknown")
            required = {"elevation", "latitude", "longitude", "lat_bounds", "lon_bounds"}
            if not required.issubset(dataset.variables):
                raise ValueError("Earth topography asset is missing elevation, coordinates or bounds")
            variable = dataset.variables["elevation"]
            if variable.dimensions != ("latitude", "longitude") or np.dtype(variable.dtype) != np.dtype("float32"):
                raise ValueError("Earth topography elevation must be float32(latitude, longitude)")
            if getattr(variable, "units", "") != "meter":
                raise ValueError("Earth topography elevation units must be meter")
            for name, dimensions, units in (("latitude", ("latitude",), "degrees_north"),
                    ("longitude", ("longitude",), "degrees_east"),
                    ("lat_bounds", ("latitude", "bounds"), "degrees_north"),
                    ("lon_bounds", ("longitude", "bounds"), "degrees_east")):
                item = dataset.variables[name]
                if item.dimensions != dimensions or getattr(item, "units", "") != units:
                    raise ValueError(f"Earth topography {name} dimensions/units are invalid")
            values = np.asanyarray(variable[:])
            if np.ma.isMaskedArray(values):
                values = values.filled(np.nan)
            elevation = np.array(values, dtype=np.float32, copy=True)
            grid = validate_grid(*(dataset.variables[name][:] for name in
                                    ("latitude", "longitude", "lat_bounds", "lon_bounds")))
            metadata = json.loads(dataset.provenance_json)
            embedded_content = str(dataset.asset_content_sha256)
            embedded_grid = str(dataset.target_grid_sha256)
    except (OSError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read Earth topography asset: {exc}") from exc
    if elevation.shape != (len(grid[0]), len(grid[1])) or not np.isfinite(elevation).all():
        raise ValueError("Earth topography elevation has invalid shape or non-finite/NoData values")
    if any(not np.array_equal(actual, expected) for actual, expected in zip(grid, target)):
        raise ValueError("Earth topography coordinates/cell bounds do not exactly match actual Earth release grid")
    grid_sha = grid_identity(*grid)
    content_sha = asset_content_identity(elevation, grid, metadata)
    if embedded_grid != grid_sha or embedded_content != content_sha:
        raise ValueError("Earth topography asset content/grid identity is corrupted")
    after = sha256_file(asset_path)
    if before != after:
        raise ValueError("Earth topography asset changed during loading")
    identity = validate_identity({**metadata, "asset_content_sha256": content_sha,
                                  "asset_file_sha256": after, "target_grid_sha256": grid_sha})
    if expected_identity is not None and identity != validate_identity(expected_identity):
        raise ValueError("Earth topography frozen identity differs from current asset")
    for array in (elevation, *grid):
        array.setflags(write=False)
    return EarthTopographyAsset(elevation, *grid, identity)


def prepare_topography_batch(asset: EarthTopographyAsset, inputs: torch.Tensor,
                              spatial_slices: Any = None) -> torch.Tensor:
    """Copy source terrain per actual sample; safe against model in-place writes.

    ``spatial_slices`` is a sequence of (latitude slice, longitude slice) pairs,
    one per batch member. Omit it only for whole-grid model inputs.
    """
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 5 or inputs.dtype != torch.float32:
        raise ValueError("Earth topography inputs must be float32 [B,T,C,H,W]")
    batch, height, width = inputs.shape[0], inputs.shape[-2], inputs.shape[-1]
    if spatial_slices is None:
        if asset.elevation.shape != (height, width):
            raise ValueError("Earth topography tile inputs require actual per-sample spatial slices")
        # np.repeat creates independent storage across all batch members.
        array = np.repeat(asset.elevation[None, None, :, :], batch, axis=0)
    else:
        if len(spatial_slices) != batch:
            raise ValueError("Earth topography needs exactly one spatial slice pair per sample")
        chunks = []
        for pair in spatial_slices:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not all(isinstance(item, slice) for item in pair)):
                raise ValueError("Earth topography spatial positions must be latitude/longitude slices")
            for item, size in zip(pair, asset.elevation.shape):
                if (item.step not in (None, 1) or item.start is None or item.stop is None
                        or not isinstance(item.start, (int, np.integer))
                        or not isinstance(item.stop, (int, np.integer))
                        or item.start < 0 or item.stop > size or item.stop <= item.start):
                    raise ValueError("Earth topography spatial slice is out of bounds or non-contiguous")
            chunk = asset.elevation[pair[0], pair[1]]
            if chunk.shape != (height, width):
                raise ValueError("Earth topography per-sample slice shape differs from meteorological inputs")
            chunks.append(chunk)
        array = np.stack(chunks, axis=0)[:, None].copy()
    return torch.from_numpy(array).to(device=inputs.device, dtype=torch.float32)
