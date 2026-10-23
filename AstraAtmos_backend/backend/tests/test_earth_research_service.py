"""Numeric contract for the Earth research service and its pure helpers."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_identity import DatasetRequestError  # noqa: E402
from services.dataset_registry import DatasetRegistry  # noqa: E402
from services.earth_dataset import CHANNELS, UNITS  # noqa: E402
from services.earth_overview_service import EarthOverviewError  # noqa: E402
from services.earth_research_service import (  # noqa: E402
    BAND_IDS,
    EarthResearchService,
    band_area_mean_series,
    band_definitions,
    cell_area_mean_series,
    compute_spatial_diagnostics,
    equal_longitude_mean,
    lagged_correlation,
    latitude_cell_edges,
    linear_regression,
    pearson_summary,
    seasonal_extremes,
    zscore_series,
)

DATASET_ID = "earth_merra2_daily_v2"

# Hand computable field: F[t, i, j] = 1000 * t + 10 * i + j.
LATITUDE = np.arange(-87.5, 90.0, 5.0)
LONGITUDE = np.arange(-177.5, 180.0, 5.0)
LAT_BOUNDS = (-90.0, 90.0)


def hand_field(shape=(4, 36, 72)) -> np.ndarray:
    time = np.arange(shape[0], dtype="float64")[:, None, None]
    row = np.arange(shape[1], dtype="float64")[None, :, None]
    column = np.arange(shape[2], dtype="float64")[None, None, :]
    return 1000.0 * time + 10.0 * row + column


@pytest.fixture
def earth_research(earth_global_release):
    registry = DatasetRegistry(**earth_global_release, earth_dataset_id=DATASET_ID)
    descriptor = registry.get_dataset(DATASET_ID)
    assert descriptor["availability"] == "available", descriptor["availability_reason"]
    service = EarthResearchService(registry)
    return service, descriptor, registry


def common(descriptor, **overrides):
    payload = {
        "dataset_id": descriptor["dataset_id"],
        "expected_fingerprint": descriptor["dataset_fingerprint"],
    }
    payload.update(overrides)
    return payload


# ── band partitions ────────────────────────────────────────────────────

def test_band_definitions_partition_every_row_exactly_once():
    definitions = band_definitions(LATITUDE)
    assert [item["id"] for item in definitions] == list(BAND_IDS)
    assigned = []
    for item in definitions:
        assigned.extend(item["latitude_values"])
    assert len(assigned) == LATITUDE.size == 36
    assert sorted(assigned) == sorted(float(value) for value in LATITUDE)


def test_band_definitions_use_real_cell_bounds_not_the_thresholds():
    definitions = {item["id"]: item for item in band_definitions(LATITUDE)}
    assert definitions["tropics"]["grid_point_count"] == 12
    assert definitions["tropics"]["min_latitude"] == -27.5
    assert definitions["tropics"]["max_latitude"] == 27.5
    assert definitions["south_polar"]["grid_point_count"] == 6
    assert definitions["south_polar"]["min_latitude"] == -87.5
    assert definitions["south_polar"]["max_latitude"] == -62.5
    assert definitions["north_polar"]["min_latitude"] == 62.5
    assert definitions["north_polar"]["max_latitude"] == 87.5
    assert [item["grid_point_count"] for item in definitions.values()] == [6, 6, 12, 6, 6]


def test_empty_band_reports_none_instead_of_fabricated_zeros():
    definitions = {item["id"]: item for item in band_definitions(np.array([-10.0, 0.0, 10.0]))}
    assert definitions["south_polar"]["grid_point_count"] == 0
    assert definitions["south_polar"]["min_latitude"] is None
    assert definitions["south_polar"]["max_latitude"] is None
    assert definitions["south_polar"]["latitude_values"] == []


# ── aggregation helpers ────────────────────────────────────────────────

def test_equal_longitude_mean_is_the_column_mean():
    field = hand_field()
    zonal = equal_longitude_mean(field)
    assert zonal.shape == (4, 36)
    # Column 0..71 mean of 1000t + 10i + j is 1000t + 10i + 35.5.
    assert zonal[0, 0] == pytest.approx(35.5)
    assert zonal[3, 35] == pytest.approx(3000.0 + 350.0 + 35.5)
    assert np.allclose(zonal, field.mean(axis=2))


def test_cell_area_mean_matches_an_independent_einsum():
    field = hand_field()
    edges = latitude_cell_edges(LATITUDE, LAT_BOUNDS)
    weights = np.diff(np.sin(np.deg2rad(edges)))
    expected = np.einsum("tij,i->t", field, weights) / (field.shape[2] * weights.sum())
    assert np.allclose(cell_area_mean_series(field, LATITUDE, LAT_BOUNDS), expected)


def test_band_area_mean_only_uses_the_band_rows():
    field = hand_field()
    definitions = {item["id"]: item for item in band_definitions(LATITUDE)}
    tropics_rows = [
        int(np.argmin(np.abs(LATITUDE - value)))
        for value in definitions["tropics"]["latitude_values"]
    ]
    series = band_area_mean_series(field, LATITUDE, tropics_rows, LAT_BOUNDS)
    edges = latitude_cell_edges(LATITUDE, LAT_BOUNDS)
    weights = np.diff(np.sin(np.deg2rad(edges)))[tropics_rows]
    expected = np.einsum("tij,i->t", field[:, tropics_rows, :], weights) / (
        field.shape[2] * weights.sum()
    )
    assert np.allclose(series, expected)
    # Band rows 12..23 average to 1000t + 10*17.5 + 35.5.
    assert series[0] == pytest.approx(210.5)


def test_band_area_mean_rejects_an_empty_band():
    with pytest.raises(EarthOverviewError) as excinfo:
        band_area_mean_series(hand_field(), LATITUDE, [])
    assert excinfo.value.code == "invalid_dataset"


def test_non_finite_samples_are_refused_not_patched():
    field = hand_field()
    field[1, 2, 3] = np.nan
    with pytest.raises(EarthOverviewError) as excinfo:
        cell_area_mean_series(field, LATITUDE, LAT_BOUNDS)
    assert excinfo.value.code == "invalid_dataset"
    with pytest.raises(EarthOverviewError):
        equal_longitude_mean(field)
    with pytest.raises(EarthOverviewError):
        compute_spatial_diagnostics(field, LATITUDE, LAT_BOUNDS)


# ── extremes ───────────────────────────────────────────────────────────

def test_seasonal_extremes_report_dates_and_peak_to_peak():
    values = np.array([3.0, 9.0, 1.0, 5.0])
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    result = seasonal_extremes(values, dates)
    assert result == {
        "max_value": 9.0, "max_date": "2020-01-02",
        "min_value": 1.0, "min_date": "2020-01-03",
        "peak_to_peak": 8.0,
    }


def test_seasonal_extremes_ties_resolve_to_the_earliest_date():
    values = np.array([5.0, 1.0, 5.0, 1.0, 5.0])
    dates = [f"2020-01-0{index + 1}" for index in range(5)]
    result = seasonal_extremes(values, dates)
    assert result["max_date"] == "2020-01-01"
    assert result["min_date"] == "2020-01-02"
    assert result["peak_to_peak"] == 4.0


def test_seasonal_extremes_require_matching_lengths():
    with pytest.raises(EarthOverviewError):
        seasonal_extremes(np.array([1.0, 2.0]), ["2020-01-01"])


# ── correlation ────────────────────────────────────────────────────────

def test_pearson_summary_perfect_positive_correlation():
    result = pearson_summary(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0]))
    assert result["r"] == pytest.approx(1.0)
    assert result["n"] == 4
    assert result["reason"] is None


def test_pearson_summary_perfect_negative_correlation():
    result = pearson_summary(np.array([1.0, 2.0, 3.0]), np.array([-1.0, -2.0, -3.0]))
    assert result["r"] == pytest.approx(-1.0)
    assert result["reason"] is None


def test_pearson_summary_constant_series_is_none_never_zero():
    constant = np.array([2.0, 2.0, 2.0, 2.0])
    varying = np.array([1.0, 2.0, 3.0, 4.0])
    for left, right in ((constant, varying), (varying, constant), (constant, constant)):
        result = pearson_summary(left, right)
        assert result["r"] is None
        assert result["reason"] == "constant_series"
        assert result["n"] == 4


def test_pearson_summary_needs_three_pairs():
    result = pearson_summary(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert result == {"r": None, "n": 2, "reason": "insufficient_samples"}


def test_lagged_correlation_peaks_at_the_documented_positive_lag():
    """A positive lag means the driver leads the target by that many days."""
    base = np.sin(np.arange(120, dtype="float64") * 0.15)
    driver = base
    target = np.zeros_like(base)
    lead = 7
    target[lead:] = base[:-lead]

    series = lagged_correlation(driver, target, max_lag=30)
    assert [item["lag_days"] for item in series] == list(range(-30, 31))
    best = max(
        (item for item in series if item["r"] is not None),
        key=lambda item: item["r"],
    )
    assert best["lag_days"] == lead
    assert best["r"] == pytest.approx(1.0)
    assert best["n"] == 120 - lead

    # No wrapping or padding: the overlap shrinks exactly with |k|.
    for item in series:
        assert item["n"] == 120 - abs(item["lag_days"])
    assert series[0]["lag_days"] == -30 and series[-1]["lag_days"] == 30
    assert series[0]["n"] == 90 and series[-1]["n"] == 90


def test_lagged_correlation_reports_insufficient_overlap_instead_of_padding():
    series = lagged_correlation(np.arange(4.0), np.arange(4.0), max_lag=3)
    edge = {item["lag_days"]: item for item in series}
    assert edge[0]["n"] == 4 and edge[0]["reason"] is None
    assert edge[3]["n"] == 1 and edge[3]["reason"] == "insufficient_samples"
    assert edge[-3]["n"] == 1 and edge[-3]["r"] is None


def test_linear_regression_recovers_the_exact_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 3.0 * x - 1.0
    result = linear_regression(x, y)
    assert result["slope"] == pytest.approx(3.0)
    assert result["intercept"] == pytest.approx(-1.0)


def test_linear_regression_is_none_when_undefined():
    assert linear_regression(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None
    assert linear_regression(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) is None


def test_zscore_series_uses_population_std_and_flags_constants():
    result = zscore_series(np.array([1.0, 2.0, 3.0]))
    assert result["reason"] is None
    assert result["values"] == pytest.approx([-1.224744871391589, 0.0, 1.224744871391589])

    constant = zscore_series(np.array([7.0, 7.0, 7.0]))
    assert constant["reason"] == "constant_series"
    assert constant["values"] == [None, None, None]
    assert all(not isinstance(value, float) for value in constant["values"])


# ── spatial diagnostics ────────────────────────────────────────────────

def test_compute_spatial_diagnostics_anomaly_rows_sum_to_zero():
    result = compute_spatial_diagnostics(hand_field(), LATITUDE, LAT_BOUNDS)
    anomaly = result["anomaly"]
    assert anomaly.shape == (36, 72)
    assert np.allclose(anomaly.sum(axis=1), 0.0, atol=1e-9)
    # Column j deviates from the row mean (j - 35.5) independently of time.
    assert anomaly[0, 0] == pytest.approx(-35.5)
    assert anomaly[35, 71] == pytest.approx(35.5)


def test_compute_spatial_diagnostics_band_rms_matches_hand_computation():
    result = compute_spatial_diagnostics(hand_field(), LATITUDE, LAT_BOUNDS)
    bands = {item["id"]: item for item in result["bands"]}
    anomaly = result["anomaly"]
    deviations = np.arange(72.0) - 35.5
    edges = latitude_cell_edges(LATITUDE, LAT_BOUNDS)
    weights = np.diff(np.sin(np.deg2rad(edges)))
    definitions = {item["id"]: item for item in band_definitions(LATITUDE)}
    for band_id, item in bands.items():
        rows = [
            int(np.argmin(np.abs(LATITUDE - value)))
            for value in definitions[band_id]["latitude_values"]
        ]
        band_weights = weights[rows][:, None]
        expected = math.sqrt(
            float(np.sum(band_weights * deviations ** 2) / np.sum(band_weights))
        )
        assert item["rms"] == pytest.approx(expected)
        assert item["peak_to_peak"] == pytest.approx(71.0)
        assert item["grid_point_count"] == len(rows)
    assert bands["tropics"]["grid_point_count"] == 12
    assert np.allclose(anomaly.sum(axis=1), 0.0, atol=1e-9)


def test_compute_spatial_diagnostics_empty_band_is_none():
    result = compute_spatial_diagnostics(
        np.ones((3, 3, 4)), np.array([-10.0, 0.0, 10.0]), (-15.0, 15.0)
    )
    bands = {item["id"]: item for item in result["bands"]}
    assert bands["south_polar"] == {
        "id": "south_polar", "rms": None, "peak_to_peak": None, "grid_point_count": 0,
    }
    assert bands["tropics"]["rms"] == pytest.approx(0.0)


# ── service level behaviour ────────────────────────────────────────────

def test_context_reports_global_geometry_and_no_filesystem_paths(earth_research, earth_global_release):
    service, descriptor, _registry = earth_research
    context = service.get_context(**common(descriptor))

    assert context["planet"] == "earth"
    assert context["dataset_id"] == DATASET_ID
    assert context["dataset_version"] == "v2"
    assert context["geometry"]["shape"] == [36, 72]
    assert context["geometry"]["lat_bounds"] == [-90.0, 90.0]
    assert context["geometry"]["lon_bounds"] == [-180.0, 180.0]
    assert context["geometry"]["wrap_longitude"] is True
    assert context["geometry"]["coverage"] == "global"
    assert context["geometry"]["dimension_order"] == ["lat", "lon"]
    assert len(context["geometry"]["lat_centers"]) == 36
    assert len(context["geometry"]["lon_centers"]) == 72
    assert context["time"]["years"] == [2020, 2021]
    assert context["time"]["count"] == 731
    assert context["capabilities"]["diurnal"] is False
    assert context["unavailable"] == {"diurnal": "daily_data_has_no_diurnal_samples"}
    assert [item["id"] for item in context["variables"]] == list(CHANNELS)
    assert context["source_meta"]["source"] == "NASA MERRA-2"
    assert context["source_meta"]["cadence"] == "daily mean"

    package_dir = str(Path(earth_global_release["earth_package_dir"]))
    serialized = repr(context)
    assert package_dir not in serialized
    assert ".nc" not in serialized


def test_context_color_range_matches_the_overview_field_endpoint(earth_research, earth_global_release):
    from services.earth_overview_service import EarthOverviewService

    service, descriptor, registry = earth_research
    overview = EarthOverviewService(registry)
    context = service.get_context(**common(descriptor))
    for variable, units in zip(CHANNELS, UNITS):
        field = overview.get_field(
            **common(descriptor), date="2020-01-01", variable=variable
        )
        declared = next(item for item in context["variables"] if item["id"] == variable)
        assert declared["units"] == units
        assert declared["color_range"] == field["color_range"], variable


def test_year_is_required_to_be_an_integer(earth_research):
    service, descriptor, _registry = earth_research
    for bad in ("2020", 2020.5, None, True):
        with pytest.raises(EarthOverviewError) as excinfo:
            service.get_research_suite(**common(descriptor), year=bad)
        assert excinfo.value.code == "year_out_of_range"
        assert excinfo.value.status_code == 422


def test_year_outside_the_release_is_rejected(earth_research):
    service, descriptor, _registry = earth_research
    for year in (2019, 2022, 1900):
        with pytest.raises(EarthOverviewError) as excinfo:
            service.get_research_suite(**common(descriptor), year=year)
        assert excinfo.value.code == "year_out_of_range"
        assert excinfo.value.status_code == 422


@pytest.mark.parametrize("year,expected", [(2020, 366), (2021, 365)])
def test_research_suite_dates_are_continuous_and_keep_the_leap_day(
    earth_research, year, expected
):
    service, descriptor, _registry = earth_research
    suite = service.get_research_suite(**common(descriptor), year=year)
    dates = suite["dates"]
    assert suite["day_count"] == expected == len(dates)
    assert dates[0] == f"{year}-01-01"
    assert dates[-1] == f"{year}-12-31"
    parsed = np.array(dates, dtype="datetime64[D]")
    assert np.all(np.diff(parsed) == np.timedelta64(1, "D"))
    assert suite["start"] == dates[0] and suite["end"] == dates[-1]
    if year == 2020:
        assert "2020-02-29" in dates
    else:
        assert "2021-02-29" not in dates


def test_research_suite_series_lengths_and_shapes(earth_research):
    service, descriptor, _registry = earth_research
    suite = service.get_research_suite(**common(descriptor), year=2020)
    day_count = suite["day_count"]

    for variable in CHANNELS:
        assert len(suite["regional_series"][variable]) == day_count
        assert len(suite["zscore"][variable]["values"]) == day_count
        seasonal = suite["seasonal"][variable]
        assert seasonal["aggregation"] == "equal_longitude_mean"
        assert len(seasonal["z"]) == 36
        assert all(len(row) == day_count for row in seasonal["z"])
        for band_id in BAND_IDS:
            assert len(suite["band_series"][band_id][variable]) == day_count
            assert suite["extremes"][band_id][variable]["peak_to_peak"] >= 0.0

    for scope_id in ("global",) + BAND_IDS:
        relationship = suite["relationships"][scope_id]
        correlation = relationship["correlation"]
        assert correlation["variable_order"] == list(CHANNELS)
        assert len(correlation["r"]) == len(correlation["n"]) == len(correlation["reason"]) == 5
        assert all(len(row) == 5 for row in correlation["r"])
        for index in range(5):
            assert correlation["r"][index][index] == pytest.approx(1.0)
            assert correlation["n"][index][index] == day_count
        assert len(relationship["solar_ozone"]["lag"]) == 61
        assert len(relationship["temperature_ozone"]["lag"]) == 61
        assert relationship["solar_ozone"]["slope_units"] == "DU/(W m-2)"
        assert relationship["temperature_ozone"]["slope_units"] == "DU/K"


def test_research_suite_aggregation_and_geometry(earth_research):
    service, descriptor, _registry = earth_research
    suite = service.get_research_suite(**common(descriptor), year=2020)
    assert suite["aggregation"] == {
        "regional": "spherical_cell_area_mean",
        "seasonal": "equal_longitude_mean",
        "band": "cell_area_weighted_band_mean",
        "scope": "global",
    }
    assert len(suite["latitude"]) == 36
    assert suite["latitude"] == [float(value) for value in LATITUDE]
    assert [item["id"] for item in suite["bands"]] == list(BAND_IDS)
    assert [item["grid_point_count"] for item in suite["bands"]] == [6, 6, 12, 6, 6]
    assert suite["variables"] == [
        {"id": name, "units": unit} for name, unit in zip(CHANNELS, UNITS)
    ]


def test_research_suite_regional_series_equals_an_independent_area_mean(earth_research):
    service, descriptor, registry = earth_research
    suite = service.get_research_suite(**common(descriptor), year=2020)
    release = registry.get_earth_overview_snapshot(
        DATASET_ID, descriptor["dataset_fingerprint"]
    )
    start = int(np.searchsorted(np.asarray(release.dates), np.datetime64("2020-01-01")))
    end = int(np.searchsorted(np.asarray(release.dates), np.datetime64("2021-01-01")))

    edges = np.linspace(-90.0, 90.0, len(release.latitude) + 1)
    weights = np.diff(np.sin(np.deg2rad(edges)))
    for variable in CHANNELS:
        field = np.asarray(release.fields[variable], dtype="float64")[start:end]
        expected = np.einsum("tij,i->t", field, weights) / (field.shape[2] * weights.sum())
        assert np.allclose(suite["regional_series"][variable], expected, rtol=1e-12)


def test_spatial_diagnostics_rows_sum_to_zero_and_reference_is_documented(earth_research):
    service, descriptor, _registry = earth_research
    body = service.get_spatial_diagnostics(**common(descriptor), year=2020, variable="TO3")
    anomaly = np.asarray(body["anomaly"], dtype="float64")
    assert anomaly.shape == (36, 72)
    assert np.allclose(anomaly.sum(axis=1), 0.0, atol=1e-9)
    assert body["reference"] == "annual_mean_minus_equal_longitude_mean"
    assert body["color_range"]["centered_on_zero"] is True
    assert body["color_range"]["min"] == pytest.approx(-body["color_range"]["max"])
    assert body["units"] == "DU"
    assert body["lat"] == [float(value) for value in LATITUDE]
    assert len(body["lon"]) == 72


def test_spatial_diagnostics_zero_anomaly_keeps_a_zero_range(tmp_path):
    """A perfectly zonal release yields an all-zero anomaly, never NaN."""
    import hashlib
    import json

    import xarray as xr

    from services.earth_dataset import SCHEMA, SPLITS, split_days

    dates = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")
    split = split_days(dates, "2020-01-05", "2020-01-07")
    lat = np.arange(-87.5, 90.0, 5.0)
    lon = np.arange(-177.5, 180.0, 5.0)
    lat_edges = np.column_stack((np.arange(-90.0, 90.0, 5.0), np.arange(-85.0, 95.0, 5.0)))
    lon_edges = np.column_stack((np.arange(-180.0, 180.0, 5.0), np.arange(-175.0, 185.0, 5.0)))
    # Constant along longitude, so every zonal anomaly is exactly zero.
    base = np.broadcast_to(
        (300.0 + np.arange(len(dates), dtype="float64"))[:, None, None],
        (len(dates), len(lat), len(lon)),
    )
    fields = {name: base + offset for offset, name in enumerate(CHANNELS)}
    package_dir = tmp_path / "merra2_daily_v2"
    package_dir.mkdir(parents=True, exist_ok=True)
    data_path = package_dir / "earth_merra2_daily.nc"
    stored = {name: np.asarray(values, dtype="float32") for name, values in fields.items()}
    training = {name: np.asarray(values, dtype="float64")[split == SPLITS["train"]] for name, values in stored.items()}
    train_stats = {
        name: {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "scale_std": float(values.std()) if float(values.std()) >= 1e-6 else 1.0,
        }
        for name, values in training.items()
    }
    dataset = xr.Dataset(
        {
            name: (("time", "lat", "lon"), stored[name], {"units": unit})
            for name, unit in zip(CHANNELS, UNITS)
        },
        coords={
            "time": ("time", dates.astype("datetime64[ns]")),
            "lat": ("lat", lat.astype("float32"), {"units": "degrees_north", "bounds": "lat_bounds"}),
            "lon": ("lon", lon.astype("float32"), {"units": "degrees_east", "bounds": "lon_bounds"}),
        },
        attrs={
            "planet": "Earth", "schema": SCHEMA, "source": "NASA MERRA-2",
            "source_sha256": "a" * 64, "temporal_resolution": "1 day",
            "train_end": "2020-01-05", "validation_end": "2020-01-07",
        },
    )
    dataset["lat_bounds"] = (("lat", "bounds"), lat_edges)
    dataset["lon_bounds"] = (("lon", "bounds"), lon_edges)
    dataset["split"] = ("time", split)
    dataset.to_netcdf(data_path, engine="netcdf4")
    dataset.close()
    manifest = {
        "schema": SCHEMA, "planet": "Earth", "dataset_id": DATASET_ID,
        "dataset_version": "v2", "source_sha256": "a" * 64,
        "data_file": data_path.name,
        "dimensions": {"time": len(dates), "lat": len(lat), "lon": len(lon), "bounds": 2},
        "time_start": str(dates[0]), "time_end": str(dates[-1]), "cadence": "daily mean",
        "latitude_range": [-90.0, 90.0], "longitude_range": [-180.0, 180.0],
        "cell_bounds": {"latitude": [-90.0, 90.0], "longitude": [-180.0, 180.0]},
        "channel_order": list(CHANNELS),
        "variables": {
            name: {
                "units": unit, "dtype": "float32",
                "min": float(stored[name].min()), "max": float(stored[name].max()),
            }
            for name, unit in zip(CHANNELS, UNITS)
        },
        "splits": {
            name: {
                "start": str(dates[split == code][0]),
                "end": str(dates[split == code][-1]),
                "days": int((split == code).sum()),
            }
            for name, code in SPLITS.items()
        },
        "normalization": {"fit_split": "train", "train_stats": train_stats},
    }
    manifest["data_bytes"] = data_path.stat().st_size
    manifest["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    registry = DatasetRegistry(
        package_dir,
        expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        expected_data_sha256=manifest["data_sha256"],
        earth_dataset_id=DATASET_ID,
    )
    descriptor = registry.get_dataset(DATASET_ID)
    assert descriptor["availability"] == "available", descriptor["availability_reason"]
    service = EarthResearchService(registry)
    body = service.get_spatial_diagnostics(
        **common(descriptor), year=2020, variable="TO3"
    )
    assert body["color_range"] == {"min": 0.0, "max": 0.0, "centered_on_zero": True}
    assert np.allclose(np.asarray(body["anomaly"], dtype="float64"), 0.0)


def test_polar_dynamics_bands_and_hemisphere_contrast(earth_research):
    service, descriptor, _registry = earth_research
    body = service.get_polar_dynamics(**common(descriptor), year=2020)
    assert [item["id"] for item in body["bands"]] == ["south_polar", "north_polar"]
    assert [item["hemisphere"] for item in body["bands"]] == ["south", "north"]
    assert [item["grid_point_count"] for item in body["bands"]] == [6, 6]
    assert body["polar_scope"]["min_abs_latitude"] == 60
    assert body["polar_scope"]["sampling"] == "daily_mean"
    assert body["polar_scope"]["note"] == "diurnal_variation_not_resolvable"
    for band in body["bands"]:
        assert band["days"] == body["day_count"] == 366
        for variable, units in zip(CHANNELS, UNITS):
            metrics = band["variables"][variable]
            assert metrics["units"] == units
            assert metrics["reason"] is None
            assert len(metrics["series"]) == 366
            assert metrics["days"] == 366
            assert metrics["min_value"] <= metrics["mean"] <= metrics["max_value"]
            assert metrics["peak_to_peak"] == pytest.approx(
                metrics["max_value"] - metrics["min_value"]
            )
    contrast = body["hemisphere_contrast"]
    for variable in CHANNELS:
        assert contrast[variable]["n"] == 366
        assert contrast[variable]["reason"] is None
        assert contrast[variable]["r"] is not None


def test_insight_summary_is_recomputed_from_the_release(earth_research):
    service, descriptor, _registry = earth_research
    digest = service.build_insight_summary(
        dataset_id=descriptor["dataset_id"],
        expected_fingerprint=descriptor["dataset_fingerprint"],
        payload={
            "year": 2020, "variable": "TO3", "scope": "tropics", "date": "2020-06-15",
            "summary": {"units": "DU", "cards": [
                {"card": "seasonal", "values": {"mean": 99999.0}, "notes": ["client note"]},
            ]},
        },
    )
    assert digest["variable"] == "TO3"
    assert digest["units"] == "DU"
    assert digest["scope"] == "tropics"
    assert digest["date_range"] == {"start": "2020-01-01", "end": "2020-12-31"}
    assert digest["date"] == "2020-06-15"
    cards = {card["card"]: card for card in digest["cards"]}
    assert "series" in cards
    # Client numbers are never echoed back as data; only notes survive.
    assert cards["series"]["values"]["mean"] != 99999.0
    assert any(card["notes"] == ["client note"] for card in digest["cards"])
    assert cards["series"]["values"]["scope"] == "tropics"


def test_insight_summary_rejects_an_unknown_scope(earth_research):
    service, descriptor, _registry = earth_research
    with pytest.raises(EarthOverviewError) as excinfo:
        service.build_insight_summary(
            dataset_id=descriptor["dataset_id"],
            expected_fingerprint=descriptor["dataset_fingerprint"],
            payload={"year": 2020, "variable": "TO3", "scope": "antarctica"},
        )
    assert excinfo.value.code == "unsupported_scope"


def test_insight_summary_rejects_a_date_outside_the_year(earth_research):
    service, descriptor, _registry = earth_research
    with pytest.raises(EarthOverviewError) as excinfo:
        service.build_insight_summary(
            dataset_id=descriptor["dataset_id"],
            expected_fingerprint=descriptor["dataset_fingerprint"],
            payload={"year": 2020, "variable": "TO3", "date": "2021-06-15"},
        )
    assert excinfo.value.code == "date_out_of_range"


def test_unsupported_variable_is_rejected_everywhere(earth_research):
    service, descriptor, _registry = earth_research
    for variable in ("O3", "to3", "wind", ""):
        with pytest.raises(EarthOverviewError) as excinfo:
            service.get_spatial_diagnostics(**common(descriptor), year=2020, variable=variable)
        assert excinfo.value.code == "unsupported_variable"
        assert excinfo.value.status_code == 422


def test_stale_fingerprint_is_a_version_conflict(earth_research):
    service, descriptor, _registry = earth_research
    with pytest.raises(DatasetRequestError) as excinfo:
        service.get_context(dataset_id=descriptor["dataset_id"], expected_fingerprint="0" * 64)
    assert excinfo.value.code == "dataset_version_changed"
    assert excinfo.value.status_code == 409


@pytest.mark.parametrize("dataset_id", ["openmars_mcd", "mcd_overview"])
def test_mars_datasets_have_no_earth_analysis(earth_research, dataset_id):
    service, descriptor, _registry = earth_research
    with pytest.raises(DatasetRequestError) as excinfo:
        service.get_context(dataset_id=dataset_id)
    assert excinfo.value.code == "dataset_overview_not_supported"
    assert excinfo.value.status_code == 409


def test_unknown_dataset_is_not_found(earth_research):
    service, _descriptor, _registry = earth_research
    with pytest.raises(DatasetRequestError) as excinfo:
        service.get_context(dataset_id="missing")
    assert excinfo.value.code == "unknown_dataset"
    assert excinfo.value.status_code == 404


def test_release_arrays_are_never_modified_and_stay_read_only(earth_research):
    service, descriptor, registry = earth_research
    release = registry.get_earth_overview_snapshot(
        DATASET_ID, descriptor["dataset_fingerprint"]
    )
    before = {name: np.asarray(release.fields[name]).copy() for name in CHANNELS}
    dates_before = np.asarray(release.dates).copy()

    for _ in range(2):
        service.get_context(**common(descriptor))
        service.get_research_suite(**common(descriptor), year=2020)
        service.get_spatial_diagnostics(**common(descriptor), year=2020, variable="TO3")
        service.get_polar_dynamics(**common(descriptor), year=2020)
        service.build_insight_summary(
            dataset_id=descriptor["dataset_id"],
            expected_fingerprint=descriptor["dataset_fingerprint"],
            payload={"year": 2020, "variable": "U10M"},
        )

    for name in CHANNELS:
        field = release.fields[name]
        assert not field.flags.writeable
        assert np.array_equal(np.asarray(field), before[name])
    assert not release.latitude.flags.writeable
    assert not release.longitude.flags.writeable
    assert not release.dates.flags.writeable
    assert np.array_equal(np.asarray(release.dates), dates_before)


def test_repeated_calls_return_identical_payloads(earth_research):
    service, descriptor, _registry = earth_research
    first = service.get_research_suite(**common(descriptor), year=2021)
    second = service.get_research_suite(**common(descriptor), year=2021)
    assert first == second
    assert service.get_context(**common(descriptor)) == service.get_context(**common(descriptor))


def test_omitting_the_fingerprint_uses_the_registry_verified_release(earth_research):
    service, descriptor, _registry = earth_research
    default = service.get_context(dataset_id=descriptor["dataset_id"])
    explicit = service.get_context(**common(descriptor))
    assert default["dataset_fingerprint"] == descriptor["dataset_fingerprint"]
    assert default == explicit


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
