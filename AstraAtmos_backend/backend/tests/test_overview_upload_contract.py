import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.overview_upload_contract import (  # noqa: E402
    classify_overview_upload_dataset,
    normalize_overview_upload_dataset,
    validate_overview_upload_dataset,
)

KG_M2_PER_UM_ATM_O3 = 2.14e-6


def _grid_coords(lat_points=36, lon_points=72):
    return {
        "time": np.arange(2, dtype=np.int32),
        "lat": np.linspace(87.5, -87.5, lat_points, dtype=np.float32),
        "lon": np.linspace(-180.0, 175.0, lon_points, dtype=np.float32),
    }


def _field(value=1.0, lat_points=36, lon_points=72):
    return np.full((2, lat_points, lon_points), value, dtype=np.float32)


def test_overview_mcd_requires_ozone_and_core_environment_fields():
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "Temperature": (("time", "lat", "lon"), _field(180.0)),
            "U_Wind": (("time", "lat", "lon"), _field(5.0)),
            "V_Wind": (("time", "lat", "lon"), _field(2.0)),
            "Solar_Flux_DN": (("time", "lat", "lon"), _field(90.0)),
        },
        coords=_grid_coords(),
    )

    result = validate_overview_upload_dataset(ds)

    assert result.is_valid is False
    assert result.data_type == "mcd"
    assert "o3col" in result.error


def test_reference_mcd_upload_normalizes_raw_download_fields_for_data_overview():
    coords = _grid_coords(lat_points=37, lon_points=72)
    raw_shape = (16, 37, 72)
    ds = xr.Dataset(
        data_vars={
            "LS": (("time",), np.linspace(10.0, 20.0, raw_shape[0], dtype=np.float32)),
            "O3COL": (("time", "lat", "lon"), np.ones(raw_shape, dtype=np.float32)),
            "T": (("time", "lat", "lon"), np.full(raw_shape, 180.0, dtype=np.float32)),
            "U": (("time", "lat", "lon"), np.full(raw_shape, 5.0, dtype=np.float32)),
            "V": (("time", "lat", "lon"), np.full(raw_shape, 2.0, dtype=np.float32)),
            "FSDS": (("time", "lat", "lon"), np.full(raw_shape, 90.0, dtype=np.float32)),
            "PS": (("time", "lat", "lon"), np.full(raw_shape, 6.0, dtype=np.float32)),
        },
        coords={**coords, "time": np.arange(raw_shape[0], dtype=np.int32)},
        attrs={"mars_year": 34},
    )

    result = validate_overview_upload_dataset(ds)
    normalized = normalize_overview_upload_dataset(ds)

    assert result.is_valid is True
    assert result.data_type == "mcd"
    assert result.mars_year == 34
    assert normalized["data_type"] == "mcd"
    assert normalized["o3col"].shape == (2, 36, 72)
    assert normalized["Temperature"].shape == normalized["o3col"].shape
    assert normalized["U_Wind"].shape == normalized["o3col"].shape
    assert normalized["V_Wind"].shape == normalized["o3col"].shape
    assert normalized["Solar_Flux_DN"].shape == normalized["o3col"].shape
    assert normalized["Pressure"].shape == normalized["o3col"].shape
    assert normalized["lat"].shape == (36,)
    assert normalized["lon"].shape == (72,)
    assert normalized["mars_year"] == 34


def test_reference_mcd_upload_converts_raw_o3col_kg_m2_to_um_atm():
    coords = _grid_coords(lat_points=37, lon_points=72)
    raw_shape = (16, 37, 72)
    ds = xr.Dataset(
        data_vars={
            "LS": (("time",), np.linspace(10.0, 20.0, raw_shape[0], dtype=np.float32)),
            "O3COL": (("time", "lat", "lon"), np.full(raw_shape, 5.0 * KG_M2_PER_UM_ATM_O3, dtype=np.float32)),
            "T": (("time", "lat", "lon"), np.full(raw_shape, 180.0, dtype=np.float32)),
            "U": (("time", "lat", "lon"), np.full(raw_shape, 5.0, dtype=np.float32)),
            "V": (("time", "lat", "lon"), np.full(raw_shape, 2.0, dtype=np.float32)),
            "FSDS": (("time", "lat", "lon"), np.full(raw_shape, 90.0, dtype=np.float32)),
            "PS": (("time", "lat", "lon"), np.full(raw_shape, 6.0, dtype=np.float32)),
        },
        coords={**coords, "time": np.arange(raw_shape[0], dtype=np.int32)},
        attrs={"mars_year": 34},
    )
    ds["O3COL"].attrs["units"] = "kg m-2"

    normalized = normalize_overview_upload_dataset(ds)

    assert float(normalized["o3col"].mean()) == pytest.approx(5.0, rel=1e-5)


def test_ready_mcd_upload_converts_tiny_legacy_o3col_to_um_atm():
    coords = _grid_coords()
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), _field(6.0 * KG_M2_PER_UM_ATM_O3)),
            "Temperature": (("time", "lat", "lon"), _field(180.0)),
            "U_Wind": (("time", "lat", "lon"), _field(5.0)),
            "V_Wind": (("time", "lat", "lon"), _field(2.0)),
            "Solar_Flux_DN": (("time", "lat", "lon"), _field(90.0)),
        },
        coords=coords,
    )

    normalized = normalize_overview_upload_dataset(ds, filename="MCD_MY34_overview.nc")

    assert float(normalized["o3col"].mean()) == pytest.approx(6.0, rel=1e-5)


def test_ready_mcd_upload_sorts_ozone_and_environment_fields_together():
    coords = _grid_coords()
    first_time = np.full((36, 72), 30.0, dtype=np.float32)
    second_time = np.full((36, 72), 10.0, dtype=np.float32)
    values = np.stack([first_time, second_time], axis=0)
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([30.0, 10.0], dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), values),
            "Temperature": (("time", "lat", "lon"), values + 100.0),
            "U_Wind": (("time", "lat", "lon"), values + 200.0),
            "V_Wind": (("time", "lat", "lon"), values + 300.0),
            "Solar_Flux_DN": (("time", "lat", "lon"), values + 400.0),
        },
        coords=coords,
    )

    normalized = normalize_overview_upload_dataset(ds)

    assert normalized["ls"].tolist() == [10.0, 30.0]
    assert float(normalized["o3col"][0, 0, 0]) == 10.0
    assert float(normalized["Temperature"][0, 0, 0]) == 110.0
    assert float(normalized["U_Wind"][0, 0, 0]) == 210.0
    assert float(normalized["V_Wind"][0, 0, 0]) == 310.0
    assert float(normalized["Solar_Flux_DN"][0, 0, 0]) == 410.0


def test_reference_mcd_upload_reorders_zero_to_360_longitude_values():
    raw_shape = (8, 37, 72)
    source_lon = np.arange(0.0, 360.0, 5.0, dtype=np.float32)
    lon_field = np.broadcast_to(source_lon.reshape(1, 1, 72), raw_shape).astype(np.float32)
    ds = xr.Dataset(
        data_vars={
            "LS": (("time",), np.linspace(10.0, 12.0, raw_shape[0], dtype=np.float32)),
            "O3COL": (("time", "lat", "lon"), lon_field),
            "T": (("time", "lat", "lon"), lon_field + 100.0),
            "U": (("time", "lat", "lon"), lon_field + 200.0),
            "V": (("time", "lat", "lon"), lon_field + 300.0),
            "FSDS": (("time", "lat", "lon"), lon_field + 400.0),
        },
        coords={
            "time": np.arange(raw_shape[0], dtype=np.int32),
            "lat": np.linspace(90.0, -90.0, raw_shape[1], dtype=np.float32),
            "lon": source_lon,
        },
    )

    normalized = normalize_overview_upload_dataset(ds)

    assert normalized["lon"][:4].tolist() == [-180.0, -175.0, -170.0, -165.0]
    assert float(normalized["o3col"][0, 0, 0]) == 180.0
    assert float(normalized["o3col"][0, 0, -1]) == 175.0


def test_openmars_upload_requires_ozone_layer_contract():
    valid = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "MY": (("time",), np.array([34, 34], dtype=np.int32)),
            "o3col": (("time", "lat", "lon"), _field(2.0)),
        },
        coords=_grid_coords(),
    )
    invalid = valid.drop_vars("o3col")

    assert classify_overview_upload_dataset(valid) == "openmars"
    assert validate_overview_upload_dataset(valid).is_valid is True

    invalid_result = validate_overview_upload_dataset(invalid)
    assert invalid_result.is_valid is False
    assert "o3col" in invalid_result.error


def test_openmars_upload_normalizes_overview_grid_order():
    lat = np.linspace(-87.5, 87.5, 36, dtype=np.float32)
    lon = np.arange(0.0, 360.0, 5.0, dtype=np.float32)
    base = (
        np.arange(36, dtype=np.float32).reshape(1, 36, 1) * 1000.0
        + np.arange(72, dtype=np.float32).reshape(1, 1, 72)
    )
    o3 = np.repeat(base, repeats=2, axis=0)
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), o3),
        },
        coords={"time": np.arange(2), "lat": lat, "lon": lon},
    )

    normalized = normalize_overview_upload_dataset(ds, filename="openmars_MY34.nc")

    assert normalized["lat"][:4].tolist() == [87.5, 82.5, 77.5, 72.5]
    assert normalized["lon"][:4].tolist() == [-180.0, -175.0, -170.0, -165.0]
    assert float(normalized["o3col"][0, 0, 0]) == 35036.0
    assert float(normalized["o3col"][0, -1, -1]) == 35.0


def test_openmars_upload_rejects_irregular_overview_grid_coordinates():
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), _field(2.0)),
        },
        coords={
            "time": np.arange(2),
            "lat": np.linspace(-90.0, 90.0, 36, dtype=np.float32),
            "lon": np.linspace(-180.0, 175.0, 72, dtype=np.float32),
        },
    )

    result = validate_overview_upload_dataset(ds, filename="openmars_MY34.nc")

    assert result.is_valid is False
    assert "grid coordinates" in result.error


def test_nomad_upload_requires_gridded_count_matching_ozone_shape():
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), _field(3.0)),
            "count": (("time", "lat", "lon"), np.ones((2, 36, 72), dtype=np.int32)),
        },
        coords=_grid_coords(),
        attrs={"mars_year": 35},
    )
    bad_count = ds.assign({"count": (("time",), np.array([1, 2], dtype=np.int32))})

    result = validate_overview_upload_dataset(ds)
    normalized = normalize_overview_upload_dataset(ds)

    assert result.is_valid is True
    assert result.data_type == "nomad"
    assert normalized["data_type"] == "nomad"
    assert normalized["count"].shape == normalized["o3col"].shape
    assert normalized["mars_year"] == 35

    invalid_result = validate_overview_upload_dataset(bad_count)
    assert invalid_result.is_valid is False
    assert "count" in invalid_result.error


def test_nomad_filename_with_ozone_but_no_count_is_rejected_as_nomad():
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), _field(3.0)),
        },
        coords=_grid_coords(),
    )

    result = validate_overview_upload_dataset(ds, filename="NOMAD_ozone_MY34_gridded.nc")

    assert result.is_valid is False
    assert result.data_type == "nomad"
    assert "count" in result.error


def test_mcd_filename_with_ozone_but_no_environment_fields_is_rejected_as_mcd():
    ds = xr.Dataset(
        data_vars={
            "LS": (("time",), np.array([10.0, 20.0], dtype=np.float32)),
            "O3COL": (("time", "lat", "lon"), _field(3.0)),
        },
        coords=_grid_coords(),
    )

    result = validate_overview_upload_dataset(ds, filename="MCD_MY34_global.nc")

    assert result.is_valid is False
    assert result.data_type == "mcd"
    assert "Temperature" in result.error or "fields" in result.error


def test_unknown_dataset_is_rejected_with_supported_type_message():
    ds = xr.Dataset(
        data_vars={"Ls": (("time",), np.array([10.0, 20.0], dtype=np.float32))},
        coords=_grid_coords(),
    )

    result = validate_overview_upload_dataset(ds)

    assert result.is_valid is False
    assert result.data_type is None
    assert "MCD" in result.error
    assert "OpenMARS" in result.error
    assert "NOMAD" in result.error
