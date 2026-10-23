from __future__ import annotations

import numpy as np
import pytest

from services.mars_data_service import MarsDataError, resolve_forecast_window


def test_uniform_axis_uses_nearest_legal_window_and_real_labels():
    result = resolve_forecast_window(np.arange(12, dtype=np.float32) * 10.0, 21.0, 3, 2)
    assert result["sample_index"] == 2
    assert result["input_ls"] == [20.0, 30.0, 40.0]
    assert result["target_ls"] == [50.0, 60.0]


def test_non_uniform_axis_does_not_use_proportional_index_mapping():
    values = np.array([0.0, 3.0, 4.0, 40.0, 120.0, 180.0, 270.0], dtype=np.float32)
    result = resolve_forecast_window(values, 5.0, 2, 2)
    assert result["sample_index"] == 2
    assert result["input_ls"] == [4.0, 40.0]
    assert result["target_ls"] == [120.0, 180.0]


def test_zero_and_360_are_the_same_periodic_position():
    values = np.array([358.0, 1.0, 4.0, 20.0, 40.0], dtype=np.float32)
    assert resolve_forecast_window(values, 0.0, 1, 2)["sample_index"] == 1
    assert resolve_forecast_window(values, 360.0, 1, 2)["sample_index"] == 1


def test_wraparound_distance_selects_nearest_window_across_a_year_boundary():
    values = np.array([358.0, 1.0, 4.0, 20.0, 40.0], dtype=np.float32)
    result = resolve_forecast_window(values, 359.0, 1, 2)
    assert result["sample_index"] == 0
    assert result["input_ls"] == [358.0]
    assert result["target_ls"] == [1.0, 4.0]


def test_start_is_never_after_last_complete_window():
    values = np.arange(8, dtype=np.float32)
    result = resolve_forecast_window(values, 999.0, 3, 2)
    assert result["sample_index"] <= len(values) - 3 - 2


@pytest.mark.parametrize(
    "values, ls_start, window, horizon",
    [
        ([], 0.0, 1, 1),
        ([0.0, 1.0], 0.0, 2, 1),
        ([0.0, float("nan"), 2.0], 0.0, 1, 1),
        ([0.0, 1.0, 2.0], float("nan"), 1, 1),
    ],
)
def test_invalid_timeline_or_window_returns_clear_error(values, ls_start, window, horizon):
    with pytest.raises(MarsDataError):
        resolve_forecast_window(values, ls_start, window, horizon)


def test_shared_resolver_gives_same_index_for_official_and_uploaded_paths():
    values = np.array([355.0, 2.0, 11.0, 45.0, 90.0, 180.0], dtype=np.float32)
    official = resolve_forecast_window(values, 360.0, 2, 2)
    uploaded = resolve_forecast_window(values, 0.0, 2, 2)
    assert official["sample_index"] == uploaded["sample_index"]
    assert official["input_ls"] == uploaded["input_ls"]
    assert official["target_ls"] == uploaded["target_ls"]


def test_repeated_ls_uses_first_year_ordered_legal_window():
    values = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    result = resolve_forecast_window(
        values,
        10.0,
        2,
        1,
        candidate_starts=np.array([0, 3], dtype=np.int64),
        candidate_mars_years=(27, 28),
    )

    assert result["sample_index"] == 0
    assert result["mars_year"] == 27
