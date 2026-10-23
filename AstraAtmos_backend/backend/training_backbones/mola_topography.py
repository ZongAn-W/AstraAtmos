from __future__ import annotations

from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import torch

from config import MOLA_TOPOGRAPHY_PATH
from services.netcdf_read_lock import netcdf_read_lock


def _as_finite_1d(values: Any, name: str, context: str) -> np.ndarray:
    array = np.asanyarray(values)
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(
            f"{context} {name} must be one-dimensional, got shape {array.shape}"
        )
    if array.size < 2:
        raise ValueError(f"{context} {name} must contain at least two values")
    if not np.isfinite(array).all():
        raise ValueError(f"{context} {name} must contain only finite values")
    return array


def validate_rectilinear_grid(
    latitude: Any,
    longitude: Any,
    *,
    context: str,
    require_global_longitude: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    lat = _as_finite_1d(latitude, "latitude", context)
    lon = _as_finite_1d(longitude, "longitude", context)
    if np.any(lat < -90.0) or np.any(lat > 90.0):
        raise ValueError(f"{context} latitude must stay within [-90, 90]")
    lat_steps = np.diff(lat)
    if not (np.all(lat_steps > 0.0) or np.all(lat_steps < 0.0)):
        raise ValueError(f"{context} latitude must be strictly monotonic")

    normalized_lon = np.mod(lon, 360.0)
    sorted_lon = np.sort(normalized_lon)
    if np.any(np.diff(sorted_lon) <= 1e-8):
        raise ValueError(
            f"{context} longitude must contain unique positions modulo 360"
        )
    if require_global_longitude:
        cyclic_gaps = np.diff(np.concatenate((sorted_lon, [sorted_lon[0] + 360.0])))
        spacing = float(np.median(cyclic_gaps))
        if not np.allclose(cyclic_gaps, spacing, rtol=0.0, atol=1e-4):
            raise ValueError(
                f"{context} longitude must cover a uniform complete global period"
            )
        if not np.isclose(spacing * len(lon), 360.0, rtol=0.0, atol=1e-4):
            raise ValueError(
                f"{context} longitude must cover a complete global period"
            )
    return lat, lon


def load_mola_asset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    asset_path = Path(path)
    if not asset_path.is_file():
        raise FileNotFoundError(f"MOLA topography asset not found: {asset_path}")

    try:
        with netcdf_read_lock(), netCDF4.Dataset(str(asset_path)) as dataset:
            missing = [
                name
                for name in ("elevation", "latitude", "longitude")
                if name not in dataset.variables
            ]
            if missing:
                raise ValueError(
                    f"MOLA asset {asset_path} is missing variables: {missing}"
                )
            elevation_variable = dataset.variables["elevation"]
            elevation_dtype = np.dtype(elevation_variable.dtype)
            expected_dtype = np.dtype(np.float32)
            if elevation_dtype != expected_dtype:
                raise ValueError(
                    f"MOLA asset {asset_path} elevation.dtype expected "
                    f"{expected_dtype.name!r}, got {elevation_dtype.name!r}"
                )
            unit = str(getattr(elevation_variable, "units", ""))
            if unit != "meter":
                raise ValueError(
                    f"MOLA asset {asset_path} elevation.units expected "
                    f"'meter', got {unit!r}"
                )
            elevation_value = np.asanyarray(elevation_variable[:])
            if np.ma.isMaskedArray(elevation_value):
                elevation_value = elevation_value.filled(np.nan)
            elevation = np.asarray(elevation_value)
            latitude, longitude = validate_rectilinear_grid(
                dataset.variables["latitude"][:],
                dataset.variables["longitude"][:],
                context=f"MOLA asset {asset_path}",
                require_global_longitude=True,
            )
    except OSError as exc:
        raise ValueError(f"Unable to read MOLA asset {asset_path}: {exc}") from exc

    expected_shape = (len(latitude), len(longitude))
    if elevation.shape != expected_shape:
        raise ValueError(
            f"MOLA asset {asset_path} elevation shape mismatch: "
            f"expected {expected_shape}, got {elevation.shape}"
        )
    if not np.isfinite(elevation).all():
        non_finite_count = int(np.count_nonzero(~np.isfinite(elevation)))
        raise ValueError(
            f"MOLA asset {asset_path} elevation.values expected all finite, "
            f"got {non_finite_count} non-finite value(s)"
        )
    return elevation, latitude, longitude


def _validate_target_coordinates(
    latitude: Any,
    longitude: Any,
    source_latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_lat = np.asanyarray(latitude)
    target_lon = np.asanyarray(longitude)
    if np.ma.isMaskedArray(target_lat):
        target_lat = target_lat.filled(np.nan)
    if np.ma.isMaskedArray(target_lon):
        target_lon = target_lon.filled(np.nan)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    if target_lat.ndim != 1 or target_lon.ndim != 1:
        raise ValueError(
            "MOLA target latitude and longitude must be one-dimensional"
        )
    if target_lat.size == 0 or target_lon.size == 0:
        raise ValueError("MOLA target latitude and longitude must not be empty")
    if not np.isfinite(target_lat).all() or not np.isfinite(target_lon).all():
        raise ValueError(
            "MOLA target latitude and longitude must contain only finite values"
        )
    source_min = float(np.min(source_latitude))
    source_max = float(np.max(source_latitude))
    if np.any(target_lat < source_min) or np.any(target_lat > source_max):
        raise ValueError(
            "MOLA target latitude is outside the source-center range "
            f"[{source_min}, {source_max}]"
        )
    return target_lat, target_lon


def resample_mola(
    source_elevation: Any,
    source_latitude: Any,
    source_longitude: Any,
    target_latitude: Any,
    target_longitude: Any,
) -> np.ndarray:
    source_lat, source_lon = validate_rectilinear_grid(
        source_latitude,
        source_longitude,
        context="MOLA source grid",
        require_global_longitude=True,
    )
    elevation = np.asarray(source_elevation, dtype=np.float32)
    expected_source_shape = (len(source_lat), len(source_lon))
    if elevation.shape != expected_source_shape:
        raise ValueError(
            "MOLA source elevation shape mismatch: "
            f"expected {expected_source_shape}, got {elevation.shape}"
        )
    if not np.isfinite(elevation).all():
        raise ValueError("MOLA source elevation must contain only finite values")
    target_lat, target_lon = _validate_target_coordinates(
        target_latitude,
        target_longitude,
        source_lat,
    )

    lat_order = np.argsort(source_lat)
    normalized_lon = np.mod(source_lon, 360.0)
    lon_order = np.argsort(normalized_lon)
    lat_sorted = source_lat[lat_order]
    lon_sorted = normalized_lon[lon_order]
    elevation_sorted = elevation[np.ix_(lat_order, lon_order)]

    lon_extended = np.concatenate(
        ([lon_sorted[-1] - 360.0], lon_sorted, [lon_sorted[0] + 360.0])
    )
    elevation_extended = np.concatenate(
        (elevation_sorted[:, -1:], elevation_sorted, elevation_sorted[:, :1]),
        axis=1,
    )
    normalized_target_lon = np.mod(target_lon, 360.0)
    along_longitude = np.stack(
        [
            np.interp(normalized_target_lon, lon_extended, row)
            for row in elevation_extended
        ],
        axis=0,
    )
    output = np.stack(
        [
            np.interp(target_lat, lat_sorted, along_longitude[:, index])
            for index in range(len(normalized_target_lon))
        ],
        axis=1,
    ).astype(np.float32)
    expected_output_shape = (len(target_lat), len(target_lon))
    if output.shape != expected_output_shape or not np.isfinite(output).all():
        raise ValueError(
            "MOLA resampling produced invalid output: "
            f"expected shape {expected_output_shape}, got {output.shape}"
        )
    return output


def prepare_topography_grid(
    target_latitude: Any,
    target_longitude: Any,
    *,
    asset_path: Path = MOLA_TOPOGRAPHY_PATH,
) -> torch.Tensor:
    target_shape = (
        int(np.size(target_latitude)),
        int(np.size(target_longitude)),
    )
    try:
        elevation, source_latitude, source_longitude = load_mola_asset(asset_path)
        output = resample_mola(
            elevation,
            source_latitude,
            source_longitude,
            target_latitude,
            target_longitude,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}; target spatial shape {target_shape}"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"MOLA asset {Path(asset_path)} cannot align source to "
            f"target spatial shape {target_shape}: {exc}"
        ) from exc
    return torch.from_numpy(output).unsqueeze(0)


def global_cell_center_coordinates(
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    height = int(height)
    width = int(width)
    if height < 2 or width < 2:
        raise ValueError("Global grid height and width must both be at least 2")
    latitude = 90.0 - 90.0 / height - np.arange(height) * (180.0 / height)
    longitude = -180.0 + np.arange(width) * (360.0 / width)
    return latitude.astype(np.float32), longitude.astype(np.float32)
