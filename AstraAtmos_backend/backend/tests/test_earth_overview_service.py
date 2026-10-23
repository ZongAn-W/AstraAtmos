"""Numeric contract for the 2D Earth overview service."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_identity import DatasetRequestError  # noqa: E402
from services.dataset_registry import DatasetRegistry  # noqa: E402
from services.earth_dataset import CHANNELS, UNITS  # noqa: E402
from services.earth_dataset_metadata import DATA_FILE_NAME  # noqa: E402
from services.earth_overview_service import (  # noqa: E402
    EarthOverviewError,
    EarthOverviewService,
    nearest_grid_index,
    regional_sample_mean,
    require_date_index,
)

DATASET_ID = "earth_merra2_daily_v1"


@pytest.fixture
def earth_overview(earth_spatial_release):
    registry = DatasetRegistry(**earth_spatial_release)
    descriptor = registry.get_dataset(DATASET_ID)
    assert descriptor["availability"] == "available", descriptor["availability_reason"]
    service = EarthOverviewService(registry)
    return service, descriptor, registry


def common(descriptor):
    return {
        "dataset_id": descriptor["dataset_id"],
        "expected_fingerprint": descriptor["dataset_fingerprint"],
    }


# ── spatial fixture sanity ─────────────────────────────────────────────

def test_spatial_fixture_is_not_spatially_constant(earth_overview):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")
    matrix = np.asarray(field["field"], dtype="float32")
    assert matrix.shape == (31, 49)
    # Row and column both matter, so flipping shows up.
    assert matrix[0, 0] != matrix[0, 1]
    assert matrix[0, 0] != matrix[1, 0]
    assert matrix[0, 0] == pytest.approx(0.0)
    # Day 0 contributes nothing, so the largest day-0 sample is row 30, column 48.
    assert matrix[30, 48] == pytest.approx(300.0 + 48.0)


# ── field ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("date,expected", [
    ("2020-01-01", 0.0),
    ("2020-02-29", 1000.0 * 59),
    ("2021-12-31", 1000.0 * 730),
])
def test_field_reports_real_dates_and_original_values(earth_overview, date, expected):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date=date, variable="TO3")
    assert field["date"] == date
    assert field["calendar"] == "proleptic_gregorian"
    # Original DU values, never normalized to a 0..1 range.
    assert field["field"][0][0] == pytest.approx(expected)
    assert field["statistics"]["min"] == pytest.approx(expected)
    assert field["statistics"]["max"] == pytest.approx(expected + 300.0 + 48.0)
    assert field["statistics"]["valid_count"] == 31 * 49


def test_field_coordinates_are_ascending_and_keep_dimension_order(earth_overview):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")
    assert field["dimension_order"] == ["lat", "lon"]
    assert field["lat"] == [float(value) for value in np.arange(-60, 61, 4)]
    assert field["lon"] == [float(value) for value in np.arange(-120, 121, 5)]
    assert len(field["field"]) == len(field["lat"]) == 31
    assert all(len(row) == len(field["lon"]) == 49 for row in field["field"])
    assert field["coverage"] == {
        "latitude_range": [-60.0, 60.0],
        "longitude_range": [-120.0, 120.0],
        "wrap_longitude": False,
    }


@pytest.mark.parametrize("variable,units", list(zip(CHANNELS, UNITS)))
def test_every_variable_returns_original_units(earth_overview, variable, units):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-03-01", variable=variable)
    assert field["variable"] == variable
    assert field["units"] == units


def test_field_row_zero_is_south_and_column_zero_is_west(earth_overview):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")
    matrix = np.asarray(field["field"], dtype="float32")
    # The synthetic value grows with latitude row and longitude column; if the
    # array were transposed or flipped these comparisons would fail.
    assert matrix[0, 0] < matrix[0, 10] < matrix[0, 48]
    assert matrix[0, 0] < matrix[10, 0] < matrix[30, 0]
    assert matrix[-1, -1] == matrix.max()


# ── colour range ───────────────────────────────────────────────────────

def test_color_range_is_fixed_across_dates_for_a_variable(earth_overview):
    service, descriptor, _registry = earth_overview
    first = service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")
    last = service.get_field(**common(descriptor), date="2021-12-31", variable="TO3")

    assert first["color_range"] == last["color_range"]
    assert first["color_range"]["scope"] == "dataset"
    assert first["color_range"]["centered_on_zero"] is False
    # Fixed over the whole data period, not recomputed per frame.
    assert first["color_range"]["min"] == pytest.approx(0.0)
    assert first["color_range"]["max"] == pytest.approx(730000.0 + 348.0)
    # The per-date statistics still move while the colour range does not.
    assert first["statistics"] != last["statistics"]


@pytest.mark.parametrize("variable", ["U10M", "V10M"])
def test_wind_color_range_is_symmetric_around_zero(earth_overview, variable):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-01-01", variable=variable)
    color_range = field["color_range"]
    assert color_range["centered_on_zero"] is True
    assert color_range["min"] == pytest.approx(-color_range["max"])


@pytest.mark.parametrize("variable", ["TO3", "T2M", "SWGDN"])
def test_scalar_variables_are_not_centered_on_zero(earth_overview, variable):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-01-01", variable=variable)
    assert field["color_range"]["centered_on_zero"] is False


# ── regional mean ──────────────────────────────────────────────────────

def test_cosine_weighting_is_not_an_unweighted_global_average():
    field = np.array([[[20, 20], [0, 0], [20, 20]]], dtype=np.float32)
    actual = regional_sample_mean(field, np.array([-60, 0, 60]))
    assert actual[0] == pytest.approx(10.0)
    assert actual[0] != pytest.approx(float(field.mean()))


def test_cosine_weighting_matches_independent_hand_computation(earth_overview):
    service, descriptor, _registry = earth_overview
    result = service.get_regional_series(**common(descriptor), variable="TO3")
    index = result["dates"].index("2020-02-29")

    # Independent expectation: build the field from the documented formula and
    # average with an explicit Python loop, not with the production function.
    rows = 31
    cols = 49
    day = 59
    expected_rows = [
        sum(1000.0 * day + 10.0 * row + col for col in range(cols)) / cols
        for row in range(rows)
    ]
    weights = [float(np.cos(np.deg2rad(-60 + 4 * row))) for row in range(rows)]
    expected = sum(value * weight for value, weight in zip(expected_rows, weights)) / sum(weights)

    assert result["values"][index] == pytest.approx(expected, rel=1e-12)
    assert result["aggregation"] == "cos_lat_sample_mean"


def test_field_statistics_regional_mean_equals_regional_series_value(earth_overview):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-02-29", variable="TO3")
    series = service.get_regional_series(**common(descriptor), variable="TO3")
    index = series["dates"].index(field["date"])
    assert field["statistics"]["regional_mean"] == pytest.approx(series["values"][index], rel=1e-12)


def test_regional_series_covers_every_day(earth_overview):
    service, descriptor, _registry = earth_overview
    series = service.get_regional_series(**common(descriptor), variable="TO3")
    assert series["start"] == "2020-01-01"
    assert series["end"] == "2021-12-31"
    assert len(series["dates"]) == len(series["values"]) == 731
    assert series["dates"][59] == "2020-02-29"
    assert all(np.isfinite(series["values"]))
    assert series["coverage"]["latitude_range"] == [-60.0, 60.0]


def test_regional_series_date_window_is_inclusive(earth_overview):
    service, descriptor, _registry = earth_overview
    windowed = service.get_regional_series(
        **common(descriptor), variable="TO3", start="2020-02-28", end="2020-03-01"
    )
    assert windowed["dates"] == ["2020-02-28", "2020-02-29", "2020-03-01"]
    assert len(windowed["values"]) == 3
    assert windowed["start"] == "2020-02-28"
    assert windowed["end"] == "2020-03-01"


# ── point series ───────────────────────────────────────────────────────

def test_point_series_returns_original_grid_values(earth_overview):
    service, descriptor, _registry = earth_overview
    point = service.get_point_series(**common(descriptor), variable="TO3", lat=0, lon=0)
    assert point["grid_point"] == {"lat": 0.0, "lon": 0.0, "lat_index": 15, "lon_index": 24}
    assert point["requested"] == {"lat": 0.0, "lon": 0.0}
    assert point["selection"] == "nearest_grid_point"
    assert point["units"] == "DU"
    assert len(point["dates"]) == len(point["values"]) == 731
    assert point["values"][0] == pytest.approx(10.0 * 15 + 24.0)


def test_point_series_values_match_the_field_on_the_same_date(earth_overview):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-02-29", variable="TO3")
    point = service.get_point_series(
        **common(descriptor), variable="TO3", lat=-4, lon=5,
    )
    index = point["dates"].index(field["date"])
    row = point["grid_point"]["lat_index"]
    col = point["grid_point"]["lon_index"]
    assert field["field"][row][col] == pytest.approx(point["values"][index])
    assert (row, col) == (14, 25)


@pytest.mark.parametrize("lat,lon,expected_lat,expected_lon", [
    (-60, -120, -60.0, -120.0),
    (60, 120, 60.0, 120.0),
    (0, 0, 0.0, 0.0),
])
def test_coverage_edges_are_selectable(earth_overview, lat, lon, expected_lat, expected_lon):
    service, descriptor, _registry = earth_overview
    point = service.get_point_series(**common(descriptor), variable="TO3", lat=lat, lon=lon)
    assert point["grid_point"]["lat"] == expected_lat
    assert point["grid_point"]["lon"] == expected_lon


def test_point_series_window_is_inclusive(earth_overview):
    service, descriptor, _registry = earth_overview
    point = service.get_point_series(
        **common(descriptor), variable="TO3", lat=0, lon=0,
        start="2020-02-28", end="2020-03-01",
    )
    assert point["dates"] == ["2020-02-28", "2020-02-29", "2020-03-01"]
    assert len(point["values"]) == 3


# ── date and point validation ──────────────────────────────────────────

@pytest.mark.parametrize("value,index", [
    ("2020-02-28", 58),
    ("2020-02-29", 59),
    ("2020-03-01", 60),
    ("2020-01-01", 0),
    ("2021-12-31", 730),
])
def test_accepted_dates_keep_consecutive_indexes(value, index):
    dates = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")
    assert require_date_index(value, dates) == index


@pytest.mark.parametrize("value,code", [
    ("2021-02-29", "date_out_of_range"),
    ("2019-12-31", "date_out_of_range"),
    ("2022-01-01", "date_out_of_range"),
    ("2020-13-01", "invalid_date"),
    ("2020-1-1", "invalid_date"),
    ("2020-01-01T00:00:00", "invalid_date"),
    ("27", "invalid_date"),
    ("", "invalid_date"),
    (27, "invalid_date"),
    ("2020/01/01", "invalid_date"),
])
def test_bad_dates_are_rejected_without_clamping(value, code):
    dates = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")
    with pytest.raises(EarthOverviewError) as exc:
        require_date_index(value, dates)
    assert exc.value.code == code


def test_date_out_of_range_is_not_clamped_to_edge(earth_overview):
    service, descriptor, _registry = earth_overview
    with pytest.raises(EarthOverviewError) as exc:
        service.get_field(**common(descriptor), date="2019-12-31", variable="TO3")
    assert exc.value.code == "date_out_of_range"


@pytest.mark.parametrize("lat,lon,code", [
    (60.001, 0, "point_outside_coverage"),
    (-60.001, 0, "point_outside_coverage"),
    (0, 120.001, "point_outside_coverage"),
    (0, -120.001, "point_outside_coverage"),
    (0, 240, "point_outside_coverage"),
    (0, -240, "point_outside_coverage"),
    (90, 0, "point_outside_coverage"),
    (float("nan"), 0, "point_outside_coverage"),
    (0, float("inf"), "point_outside_coverage"),
])
def test_points_outside_coverage_are_rejected_without_wrapping(earth_overview, lat, lon, code):
    service, descriptor, _registry = earth_overview
    with pytest.raises(EarthOverviewError) as exc:
        service.get_point_series(**common(descriptor), variable="TO3", lat=lat, lon=lon)
    assert exc.value.code == code


def test_nearest_grid_index_breaks_ties_towards_the_smaller_index():
    axis = np.array([-60.0, -56.0, -52.0, -48.0])
    # -58 is exactly between -60 and -56: the smaller index wins on both ends.
    assert nearest_grid_index(axis, -58.0) == 0
    assert nearest_grid_index(axis, -54.0) == 1
    assert nearest_grid_index(axis, -50.0) == 2
    assert nearest_grid_index(axis, -60.0) == 0
    assert nearest_grid_index(axis, -48.0) == 3
    assert nearest_grid_index(axis, -57.9) == 1


def test_point_series_never_wraps_longitude_across_the_antimeridian(earth_overview):
    service, descriptor, _registry = earth_overview
    # 240 degrees east is outside the regional coverage; wrapping it to -120
    # would silently answer for a different location.
    with pytest.raises(EarthOverviewError) as exc:
        service.get_point_series(**common(descriptor), variable="TO3", lat=0, lon=240)
    assert exc.value.code == "point_outside_coverage"


# ── variable / range validation ────────────────────────────────────────

@pytest.mark.parametrize("variable", ["wind", "O3", "temperature", "to3", "", "U10", "SWGDN "])
def test_unsupported_variables_are_rejected(earth_overview, variable):
    service, descriptor, _registry = earth_overview
    with pytest.raises(EarthOverviewError) as exc:
        service.get_field(**common(descriptor), date="2020-01-01", variable=variable)
    assert exc.value.code == "unsupported_variable"


def test_start_after_end_is_rejected(earth_overview):
    service, descriptor, _registry = earth_overview
    with pytest.raises(EarthOverviewError) as exc:
        service.get_regional_series(
            **common(descriptor), variable="TO3", start="2020-06-01", end="2020-01-01"
        )
    assert exc.value.code == "invalid_date_range"


def test_range_endpoints_outside_the_release_are_rejected(earth_overview):
    service, descriptor, _registry = earth_overview
    with pytest.raises(EarthOverviewError) as exc:
        service.get_regional_series(
            **common(descriptor), variable="TO3", start="2020-01-01", end="2023-01-01"
        )
    assert exc.value.code == "date_out_of_range"


# ── identity and availability ──────────────────────────────────────────

def test_unknown_and_mars_identifiers_fail_clearly(earth_overview):
    service, descriptor, _registry = earth_overview
    with pytest.raises(DatasetRequestError) as exc:
        service.get_field(
            dataset_id="missing", expected_fingerprint=descriptor["dataset_fingerprint"],
            date="2020-01-01", variable="TO3",
        )
    assert exc.value.code == "unknown_dataset" and exc.value.status_code == 404

    with pytest.raises(DatasetRequestError) as exc:
        service.get_field(
            dataset_id="openmars_mcd", expected_fingerprint=descriptor["dataset_fingerprint"],
            date="2020-01-01", variable="TO3",
        )
    assert exc.value.code == "dataset_overview_not_supported"
    assert exc.value.status_code == 409


def test_stale_fingerprint_is_rejected_rather_than_mixing_versions(earth_overview):
    service, descriptor, _registry = earth_overview
    with pytest.raises(DatasetRequestError) as exc:
        service.get_field(
            dataset_id=DATASET_ID, expected_fingerprint="0" * 64,
            date="2020-01-01", variable="TO3",
        )
    assert exc.value.code == "dataset_version_changed"
    assert exc.value.status_code == 409


def test_response_identity_matches_the_registry_descriptor(earth_overview):
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")
    assert field["dataset_id"] == descriptor["dataset_id"] == DATASET_ID
    assert field["dataset_version"] == descriptor["dataset_version"] == "v1"
    assert field["dataset_fingerprint"] == descriptor["dataset_fingerprint"]
    assert field["planet"] == "earth"


def test_missing_package_reports_unavailable_with_a_safe_reason(tmp_path):
    registry = DatasetRegistry(tmp_path / "absent")
    service = EarthOverviewService(registry)
    with pytest.raises(DatasetRequestError) as exc:
        service.get_field(
            dataset_id=DATASET_ID, expected_fingerprint="a" * 64,
            date="2020-01-01", variable="TO3",
        )
    assert exc.value.code == "dataset_unavailable"
    assert exc.value.status_code == 503
    assert exc.value.availability_reason == "package_missing"


def test_replaced_package_invalidates_the_cached_snapshot(earth_overview):
    service, descriptor, _registry = earth_overview
    assert service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")

    package_dir = Path(earth_overview[2]._earth_package_dir)
    data_path = package_dir / DATA_FILE_NAME
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 0xFF
    data_path.write_bytes(bytes(payload))

    with pytest.raises(DatasetRequestError) as exc:
        service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")
    assert exc.value.code == "dataset_unavailable"
    assert exc.value.availability_reason == "data_fingerprint_mismatch"

    # The catalog still answers, now reporting the failure.
    refreshed = earth_overview[2].get_dataset(DATASET_ID)
    assert refreshed["availability"] == "invalid"
    assert refreshed["availability_reason"] == "data_fingerprint_mismatch"


def test_disappearing_package_invalidates_the_cached_snapshot(earth_overview):
    service, descriptor, registry = earth_overview
    assert service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")

    package_dir = Path(registry._earth_package_dir)
    (package_dir / DATA_FILE_NAME).unlink()

    with pytest.raises(DatasetRequestError) as exc:
        service.get_regional_series(**common(descriptor), variable="TO3")
    assert exc.value.code == "dataset_unavailable"
    assert exc.value.availability_reason == "package_missing"


def test_overview_capability_is_declared_for_registered_earth(earth_overview):
    """Capability describes the wired entry, independent of file presence."""
    _service, descriptor, _registry = earth_overview
    assert descriptor["capabilities"]["web_overview"] is True
    assert descriptor["capabilities"]["metadata"] is True
    # Earth training and historical prediction entries are wired as well; this
    # test only pins the overview capability plus that the three are separate.
    assert descriptor["capabilities"]["training"] is True
    assert descriptor["capabilities"]["trained_prediction"] is True


def test_snapshot_arrays_are_read_only_and_survive_the_closed_file(earth_overview):
    _service, descriptor, registry = earth_overview
    release = registry.get_earth_overview_snapshot(
        DATASET_ID, descriptor["dataset_fingerprint"]
    )
    assert not release.dates.flags.writeable
    assert not release.latitude.flags.writeable
    assert not release.longitude.flags.writeable
    for name in CHANNELS:
        assert not release.fields[name].flags.writeable
        assert release.fields[name].shape == (731, 31, 49)
        assert release.fields[name].dtype == np.float32
    with pytest.raises((ValueError, TypeError)):
        release.latitude[0] = 999.0


def test_repeated_queries_verify_the_package_only_once(earth_overview):
    service, descriptor, registry = earth_overview
    for _ in range(3):
        service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")
        service.get_regional_series(**common(descriptor), variable="T2M")
        service.get_point_series(**common(descriptor), variable="SWGDN", lat=0, lon=0)
        registry.get_dataset(DATASET_ID)

    assert registry.verification_count == 1


def test_concurrent_reads_share_a_single_snapshot(earth_overview):
    import concurrent.futures

    service, descriptor, _registry = earth_overview

    def read(index):
        date = f"2020-{(index % 12) + 1:02d}-01"
        result = service.get_field(**common(descriptor), date=date, variable="TO3")
        return result["field"][0][0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(read, range(24)))

    assert len(values) == 24
    assert all(np.isfinite(value) for value in values)
    assert earth_overview[2].verification_count == 1


def test_constant_field_does_not_produce_nan_color_range(tmp_path, earth_release):
    """A release whose field is spatially constant must not divide by zero."""
    registry = DatasetRegistry(**earth_release)
    descriptor = registry.get_dataset(DATASET_ID)
    service = EarthOverviewService(registry)
    field = service.get_field(**common(descriptor), date="2020-01-01", variable="TO3")

    assert np.isfinite(field["color_range"]["min"])
    assert np.isfinite(field["color_range"]["max"])
    # The fixture is constant in space but not in time, so one date is a single
    # value; the frontend renders the ramp midpoint instead of dividing by zero.
    assert field["statistics"]["min"] == field["statistics"]["max"]
    assert np.isfinite(field["statistics"]["regional_mean"])


def test_field_values_are_not_normalized(earth_overview):
    """Original physical magnitudes must survive; TO3 is DU, not 0..1."""
    service, descriptor, _registry = earth_overview
    field = service.get_field(**common(descriptor), date="2021-12-31", variable="TO3")
    assert field["statistics"]["max"] > 100.0
    assert field["statistics"]["min"] >= 0.0
    assert field["field"][0][0] == pytest.approx(730000.0)


def test_service_does_not_reuse_the_training_normalizer(earth_overview):
    """The overview must not import the normalized-window training dataset."""
    source = (
        BACKEND_DIR / "services" / "earth_overview_service.py"
    ).read_text(encoding="utf-8")
    assert "EarthOzoneWindows" not in source
    assert "normalize_ozone" not in source
    assert "ozone_units" not in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
