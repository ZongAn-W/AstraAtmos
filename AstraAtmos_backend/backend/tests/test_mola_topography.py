from pathlib import Path
import sys
import uuid

import netCDF4
import numpy as np
import pytest
import torch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training_backbones.mola_topography import (  # noqa: E402
    load_mola_asset,
    prepare_topography_grid,
    resample_mola,
    validate_rectilinear_grid,
)
from config import MOLA_TOPOGRAPHY_PATH  # noqa: E402
from scripts.build_mola_topography_asset import validate_asset  # noqa: E402


EXPECTED_PROVENANCE = {
    "source_archive": "MGS-M-MOLA-5-MEGDR-L3-V1",
    "source_product_id": "MEGT90N000CB.IMG",
    "source_product_resolution": "4 pixels per degree (0.25 degree)",
    "source_image_url": (
        "https://pds-geosciences.wustl.edu/mgs/"
        "mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004/megt90n000cb.img"
    ),
    "source_label_url": (
        "https://pds-geosciences.wustl.edu/mgs/"
        "mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004/megt90n000cb.lbl"
    ),
    "source_sha256": (
        "25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c"
    ),
    "preprocessing": (
        "Periodic 5-degree cell aggregation; arithmetic mean of exactly "
        "400 source 0.25-degree MOLA median-topography cells per output cell"
    ),
}


@pytest.fixture
def mola_tmp_path():
    directory = BACKEND_DIR / ".test_tmp" / f"mola_{uuid.uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    finally:
        for name in (
            "broken.nc",
            "non-finite.nc",
            "incomplete.nc",
            "mola.nc",
            "wrong-dtype.nc",
            "platform.nc",
        ):
            path = directory / name
            if path.exists():
                path.unlink()
        directory.rmdir()


def _periodic_field(latitude, longitude):
    return (
        np.asarray(latitude, dtype=np.float64)[:, None]
        + 10.0
        * np.cos(np.deg2rad(np.asarray(longitude, dtype=np.float64)))[None, :]
    ).astype(np.float32)


def _write_mola_fixture(
    path,
    latitude,
    longitude,
    elevation=None,
    *,
    missing_variable=None,
    elevation_dtype="f4",
    provenance=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    latitude = np.asarray(latitude, dtype=np.float32)
    longitude = np.asarray(longitude, dtype=np.float32)
    values = (
        _periodic_field(latitude, longitude)
        if elevation is None
        else np.asarray(elevation, dtype=np.float32)
    )
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("latitude", len(latitude))
        dataset.createDimension("longitude", len(longitude))
        if missing_variable != "latitude":
            dataset.createVariable("latitude", "f4", ("latitude",))[:] = latitude
        if missing_variable != "longitude":
            dataset.createVariable("longitude", "f4", ("longitude",))[:] = longitude
        if missing_variable != "elevation":
            variable = dataset.createVariable(
                "elevation", elevation_dtype, ("latitude", "longitude")
            )
            variable.units = "meter"
            variable[:] = values
        for name, value in (provenance or {}).items():
            dataset.setncattr(name, value)


def _write_platform_asset(path, *, elevation_dtype="f4"):
    latitude = np.arange(87.5, -90.0, -5.0, dtype=np.float32)
    longitude = np.arange(-180.0, 180.0, 5.0, dtype=np.float32)
    _write_mola_fixture(
        path,
        latitude,
        longitude,
        elevation_dtype=elevation_dtype,
        provenance=EXPECTED_PROVENANCE,
    )


def test_resample_preserves_descending_target_latitude():
    source_lat = np.array([-60.0, -20.0, 20.0, 60.0])
    source_lon = np.arange(-180.0, 180.0, 45.0)
    target_lat = np.array([40.0, 0.0, -40.0])
    target_lon = np.array([-135.0, -45.0, 45.0, 135.0])

    result = resample_mola(
        _periodic_field(source_lat, source_lon),
        source_lat,
        source_lon,
        target_lat,
        target_lon,
    )

    assert result.shape == (3, 4)
    assert np.all(result[0] > result[1])
    assert np.all(result[1] > result[2])


def test_resample_restores_ascending_target_latitude():
    source_lat = np.array([60.0, 20.0, -20.0, -60.0])
    source_lon = np.arange(0.0, 360.0, 45.0)
    target_lat = np.array([-40.0, 0.0, 40.0])

    result = resample_mola(
        _periodic_field(source_lat, source_lon),
        source_lat,
        source_lon,
        target_lat,
        np.array([0.0, 90.0, 180.0, 270.0]),
    )

    assert np.all(result[0] < result[1])
    assert np.all(result[1] < result[2])


def test_resample_wraps_longitude_boundary_without_a_seam():
    source_lat = np.array([-45.0, 45.0])
    source_lon = np.arange(-180.0, 180.0, 30.0)
    target_lon = np.array([-180.1, 179.9, -180.0, 180.0, -179.9, 180.1])

    result = resample_mola(
        _periodic_field(source_lat, source_lon),
        source_lat,
        source_lon,
        np.array([0.0]),
        target_lon,
    )[0]

    assert result[0] == pytest.approx(result[1], abs=1e-5)
    assert result[2] == pytest.approx(result[3], abs=1e-5)
    assert result[4] == pytest.approx(result[5], abs=1e-5)


def test_same_shape_different_coordinates_are_interpolated_not_copied():
    source_lat = np.array([-45.0, 45.0])
    source_lon = np.array([-180.0, -90.0, 0.0, 90.0])
    source = _periodic_field(source_lat, source_lon)

    result = resample_mola(
        source,
        source_lat,
        source_lon,
        np.array([-30.0, 30.0]),
        np.array([-135.0, -45.0, 45.0, 135.0]),
    )

    assert result.shape == source.shape
    assert not np.array_equal(result, source)


@pytest.mark.parametrize(("height", "width"), [(2, 4), (4, 8), (6, 12)])
def test_resample_supports_multiple_target_grid_sizes(height, width):
    source_lat = np.linspace(-75.0, 75.0, 6)
    source_lon = np.arange(-180.0, 180.0, 30.0)
    target_lat = np.linspace(-60.0, 60.0, height)
    target_lon = np.arange(width) * (360.0 / width) - 180.0

    result = resample_mola(
        _periodic_field(source_lat, source_lon),
        source_lat,
        source_lon,
        target_lat,
        target_lon,
    )

    assert result.shape == (height, width)
    assert np.isfinite(result).all()


def test_prepare_rejects_missing_asset_with_path_and_target_shape(mola_tmp_path):
    missing = mola_tmp_path / "missing.nc"

    with pytest.raises(FileNotFoundError) as exc_info:
        prepare_topography_grid(
            np.array([-45.0, 45.0]),
            np.array([-180.0, -90.0, 0.0, 90.0]),
            asset_path=missing,
        )

    message = str(exc_info.value)
    assert str(missing) in message
    assert "(2, 4)" in message


@pytest.mark.parametrize("missing_name", ["elevation", "latitude", "longitude"])
def test_load_rejects_missing_required_variable(mola_tmp_path, missing_name):
    path = mola_tmp_path / "broken.nc"
    _write_mola_fixture(
        path,
        [-45.0, 45.0],
        [-180.0, -90.0, 0.0, 90.0],
        missing_variable=missing_name,
    )

    with pytest.raises(ValueError, match=missing_name):
        load_mola_asset(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_load_rejects_non_finite_elevation(mola_tmp_path, bad):
    path = mola_tmp_path / "non-finite.nc"
    elevation = _periodic_field(
        [-45.0, 45.0], [-180.0, -90.0, 0.0, 90.0]
    )
    elevation[0, 0] = bad
    _write_mola_fixture(
        path,
        [-45.0, 45.0],
        [-180.0, -90.0, 0.0, 90.0],
        elevation,
    )

    with pytest.raises(ValueError) as exc_info:
        load_mola_asset(path)

    message = str(exc_info.value)
    assert "elevation.values" in message
    assert "all finite" in message
    assert "1 non-finite" in message


@pytest.mark.parametrize(
    ("raw_dtype", "actual_dtype"),
    [("i2", "int16"), ("f8", "float64")],
)
def test_load_rejects_non_float32_elevation_dtype(
    mola_tmp_path,
    raw_dtype,
    actual_dtype,
):
    path = mola_tmp_path / "wrong-dtype.nc"
    _write_mola_fixture(
        path,
        [-45.0, 45.0],
        [-180.0, -90.0, 0.0, 90.0],
        elevation_dtype=raw_dtype,
    )

    with pytest.raises(ValueError) as exc_info:
        load_mola_asset(path)

    message = str(exc_info.value)
    assert "elevation.dtype" in message
    assert "float32" in message
    assert actual_dtype in message


@pytest.mark.parametrize("field", EXPECTED_PROVENANCE)
def test_validate_asset_rejects_missing_required_provenance(mola_tmp_path, field):
    path = mola_tmp_path / "platform.nc"
    _write_platform_asset(path)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.delncattr(field)

    with pytest.raises(ValueError) as exc_info:
        validate_asset(path)

    message = str(exc_info.value)
    assert field in message
    assert repr(EXPECTED_PROVENANCE[field]) in message
    assert "<missing>" in message


@pytest.mark.parametrize("field", EXPECTED_PROVENANCE)
def test_validate_asset_rejects_wrong_required_provenance(mola_tmp_path, field):
    path = mola_tmp_path / "platform.nc"
    _write_platform_asset(path)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.setncattr(field, "unexpected")

    with pytest.raises(ValueError) as exc_info:
        validate_asset(path)

    message = str(exc_info.value)
    assert field in message
    assert repr(EXPECTED_PROVENANCE[field]) in message
    assert repr("unexpected") in message


def test_load_rejects_incomplete_periodic_longitude_grid(mola_tmp_path):
    path = mola_tmp_path / "incomplete.nc"
    _write_mola_fixture(path, [-45.0, 45.0], [-180.0, -90.0, 0.0])

    with pytest.raises(ValueError, match="global period"):
        load_mola_asset(path)


def test_validate_grid_accepts_float32_longitudes_converted_from_radians():
    latitude = np.arange(87.5, -90.0, -5.0, dtype=np.float32)
    longitude = np.rad2deg(
        np.linspace(-np.pi, np.pi, 72, endpoint=False, dtype=np.float32)
    ).astype(np.float32)

    actual_latitude, actual_longitude = validate_rectilinear_grid(
        latitude,
        longitude,
        context="OpenMars float32 grid",
        require_global_longitude=True,
    )

    assert np.array_equal(actual_latitude, latitude.astype(np.float64))
    assert np.array_equal(actual_longitude, longitude.astype(np.float64))


def test_prepare_static_topography_returns_float32_1hw_tensor(mola_tmp_path):
    path = mola_tmp_path / "mola.nc"
    _write_mola_fixture(path, [-45.0, 45.0], [-180.0, -90.0, 0.0, 90.0])

    result = prepare_topography_grid(
        np.array([-30.0, 30.0]),
        np.array([-135.0, -45.0, 45.0, 135.0]),
        asset_path=path,
    )

    assert result.shape == (1, 2, 4)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_committed_mola_asset_has_expected_grid_and_provenance():
    with netCDF4.Dataset(str(MOLA_TOPOGRAPHY_PATH)) as dataset:
        assert {"elevation", "latitude", "longitude"} <= set(dataset.variables)
        elevation = dataset.variables["elevation"]
        assert elevation.dtype == np.dtype("float32")
        assert elevation.shape == (36, 72)
        assert elevation.units == "meter"
        assert np.array_equal(
            dataset.variables["latitude"][:],
            np.arange(87.5, -90.0, -5.0, dtype=np.float32),
        )
        assert np.array_equal(
            dataset.variables["longitude"][:],
            np.arange(-180.0, 180.0, 5.0, dtype=np.float32),
        )
        assert dataset.source_product_id == "MEGT90N000CB.IMG"
        assert dataset.source_sha256 == (
            "25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c"
        )
        assert np.isfinite(elevation[:]).all()
