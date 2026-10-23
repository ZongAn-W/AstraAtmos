"""Exact UTC backtest centres and additive prediction response compatibility."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from schemas.earth_predict import (
    EarthPredictContextResponse,
    EarthPredictMetrics,
    EarthPredictOrigins,
    EarthPredictRequest,
    EarthPredictResponse,
)
from services.dataset_identity import EARTH_DATASET_3HOURLY_ID
from services.earth_training_artifact import (
    EarthArtifactError,
    threehour_available_origin_range,
    threehour_forecast_timestamps,
    threehour_origin_index,
    threehour_origin_split,
)
from services.earth_task_split import build_earth_task_split


@pytest.fixture
def release():
    times = np.arange(np.datetime64("2020-01-01T01:30", "ns"),
                      np.datetime64("2022-01-01T01:30", "ns"), np.timedelta64(3, "h"))
    return SimpleNamespace(dates=times, metadata={
        "dataset_id": EARTH_DATASET_3HOURLY_ID,
        "schema": "astraatmos_earth_3hourly_v1",
        "frequency_hours": 3, "step_unit": "hour", "step": 3,
        "time": {"kind": "datetime", "time_zone": "UTC", "label": "interval_center"},
        "splits": {
            "train": {"start": "2020-01-01", "end": "2020-12-31", "steps": 2928},
            "validation": {"start": "2021-01-01", "end": "2021-06-30", "steps": 1448},
            "test": {"start": "2021-07-01", "end": "2021-12-31", "steps": 1472},
        },
    })


def test_context_has_all_5769_complete_origins_without_truncating_time(release):
    origins = threehour_available_origin_range(release)
    assert origins["start"] == "2020-01-07T22:30:00Z"
    assert origins["end"] == "2021-12-28T22:30:00Z"
    assert origins["count"] == 5769
    assert origins["timestamps"] == origins["dates"]
    assert origins["input_offset_hours"] == -165
    assert origins["target_offset_hours"] == 3
    assert (origins["window"], origins["horizon"]) == (56, 24)
    assert (origins["kind"], origins["time_zone"], origins["frequency_hours"]) == ("datetime", "UTC", 3)
    assert "input_offset_days" not in origins
    delta = np.diff(np.asarray([value[:-1] for value in origins["timestamps"]], dtype="datetime64[ns]"))
    assert np.all(delta == np.timedelta64(3, "h"))


def test_targets_are_24_exact_centres_at_three_to_72_hours(release):
    origin = "2021-07-08T01:30:00Z"
    targets = threehour_forecast_timestamps(release, origin)
    assert len(targets) == 24
    assert targets[0] == "2021-07-08T04:30:00Z"
    assert targets[-1] == "2021-07-11T01:30:00Z"
    assert threehour_forecast_timestamps(release, origin.replace("Z", "+00:00")) == targets
    assert threehour_forecast_timestamps(release, "2021-07-08T01:30:00.000000000Z") == targets


@pytest.mark.parametrize("origin,split", [
    ("2020-07-08T01:30:00Z", "train"),
    ("2021-01-01T01:30:00Z", "validation"),
    ("2021-07-01T01:30:00Z", "test"),
])
def test_legacy_split_is_the_published_origin_label(release, origin, split):
    assert threehour_origin_split(release, origin) == split


def test_task_split_range_accepts_only_complete_single_partition_windows(release):
    task_split = build_earth_task_split(
        release.dates, 56, 24,
        {"train_ratio": 0.5, "validation_ratio": 0.25, "test_ratio": 0.25},
    )
    origins = threehour_available_origin_range(
        release, strict_split=True, task_split=task_split,
    )
    assert origins["count"] == sum(
        entry["window_count"] for entry in task_split["ranges"].values()
    )
    assert origins["count"] < threehour_available_origin_range(release)["count"]

    train_end = task_split["ranges"]["train"]["raw_end"]
    last_train_index = train_end - task_split["horizon"] - 1
    first_crossing_index = train_end - task_split["horizon"]
    last_train_origin = _origin_for_index(release, last_train_index)
    first_crossing_origin = _origin_for_index(release, first_crossing_index)

    assert threehour_origin_index(
        release, last_train_origin, strict_split=True, task_split=task_split,
    ) == last_train_index
    assert threehour_origin_split(
        release, last_train_origin, task_split=task_split,
    ) == "train"
    with pytest.raises(EarthArtifactError) as error:
        threehour_origin_index(
            release, first_crossing_origin, strict_split=True, task_split=task_split,
        )
    assert error.value.code == "earth_prediction_origin_out_of_range"
    with pytest.raises(EarthArtifactError) as error:
        threehour_origin_split(
            release, first_crossing_origin, task_split=task_split,
        )
    assert error.value.code == "earth_prediction_origin_out_of_range"
    with pytest.raises(EarthArtifactError) as error:
        threehour_forecast_timestamps(
            release, first_crossing_origin, strict_split=True, task_split=task_split,
        )
    assert error.value.code == "earth_prediction_origin_out_of_range"


def _origin_for_index(release, index):
    return np.datetime_as_string(release.dates[index], unit="s") + "Z"


@pytest.mark.parametrize("origin", [
    "2021-07-08", "2021-07-08T01:30:00", "2021-07-08 01:30:00Z",
    "2021-07-08T01:30:00+08:00", "2021-07-08T01:30:00-00:00",
    "2021-07-08T01:30:00z", "2021-07-08T01:30Z", "2021-02-30T01:30:00Z",
    "2021-07-08T01:30:00.000000001Z", "2021-07-08T01:30:01Z",
    "2021-07-08T02:30:00Z", "2022-01-01T01:30:00Z", None,
])
def test_invalid_origins_are_rejected_without_rounding(release, origin):
    with pytest.raises(EarthArtifactError) as error:
        threehour_origin_index(release, origin)
    expected_code = ("earth_prediction_origin_out_of_range" if origin in (
        "2021-07-08T02:30:00Z", "2022-01-01T01:30:00Z") else "invalid_earth_prediction_origin")
    assert error.value.code == expected_code


@pytest.mark.parametrize("index", [0, 54, -24, -1])
def test_published_but_incomplete_window_origins_have_stable_range_code(release, index):
    origin = np.datetime_as_string(release.dates[index], unit="s") + "Z"
    with pytest.raises(EarthArtifactError) as error:
        threehour_origin_index(release, origin)
    assert error.value.code == "earth_prediction_origin_out_of_range"


def test_first_last_and_minimal_80_frame_packages_are_usable(release):
    origins = threehour_available_origin_range(release)
    assert threehour_origin_index(release, origins["start"]) == 55
    assert threehour_origin_index(release, origins["end"]) == 5823
    release.dates = release.dates[:80]
    minimal = threehour_available_origin_range(release)
    assert minimal["count"] == 1
    assert minimal["start"] == minimal["end"]
    release.dates = release.dates[:79]
    with pytest.raises(EarthArtifactError) as error:
        threehour_available_origin_range(release)
    assert error.value.code == "earth_prediction_origin_out_of_range"


@pytest.mark.parametrize("case", ["duplicate", "gap", "descending", "nat", "off_centre"])
def test_damaged_axis_never_offers_origins(release, case):
    release.dates = release.dates.copy()
    if case == "duplicate":
        release.dates[60] = release.dates[59]
    elif case == "gap":
        release.dates = np.delete(release.dates, 60)
    elif case == "descending":
        release.dates = release.dates[::-1]
    elif case == "nat":
        release.dates[60] = np.datetime64("NaT")
    else:
        release.dates += np.timedelta64(30, "m")
    with pytest.raises(EarthArtifactError) as error:
        threehour_available_origin_range(release)
    assert error.value.code == "invalid_earth_training_artifact"


def test_split_corruption_is_a_stable_artifact_error(release):
    release.metadata["splits"]["train"]["steps"] -= 1
    with pytest.raises(EarthArtifactError) as error:
        threehour_origin_split(release, "2021-07-08T01:30:00Z")
    assert error.value.code == "invalid_earth_training_artifact"


def test_threehour_helpers_do_not_accept_daily_identity(release):
    release.metadata["dataset_id"] = "earth_merra2_daily_v2"
    with pytest.raises(EarthArtifactError):
        threehour_available_origin_range(release)


def _daily_origins():
    return {"start": "2020-01-07", "end": "2021-12-28", "count": 722,
            "dates": ["2020-01-07", "2021-12-28"], "window": 7, "horizon": 3,
            "input_offset_days": -6, "target_offset_days": 1}


def _daily_metrics():
    return {"unit": "DU", "target": "TO3", "aggregation": "user_forecast_origin_lead_grid_uniform",
            "reference_available": True, "overall": {"rmse": 1.0, "mae": 0.5},
            "by_lead": [{"lead_day": day, "rmse": 1.0, "mae": 0.5} for day in (1, 2, 3)]}


def _daily_context():
    return {"task_id": 42, "dataset_id": "earth_merra2_daily_v2", "dataset_version": "v2",
            "dataset_fingerprint": "a" * 64, "target": "TO3", "target_unit": "DU",
            "window": 7, "horizon": 3, "grid": {"shape": [36, 72]}, "origins": _daily_origins()}


def test_daily_schema_retains_existing_defaults_and_has_no_new_null_fields():
    assert EarthPredictOrigins(**_daily_origins()).model_dump() == _daily_origins()
    assert EarthPredictMetrics(**_daily_metrics()).model_dump() == _daily_metrics()
    context = EarthPredictContextResponse(**_daily_context()).model_dump()
    assert "frequency_hours" not in context
    assert "timestamp_rule" not in context
    assert context["grid"]["latitude_range"] is None
    assert context["model"]["uploaded_model_id"] is None
    assert context["warnings"] == []
    run_payload = _daily_context()
    run_payload.pop("origins")
    run_payload.update(forecast_origin="2021-07-08", origin_split="test", metrics=_daily_metrics())
    result = EarthPredictResponse(**run_payload).model_dump()
    assert "target_timestamps" not in result
    assert "input_timestamps" not in result
    assert "frequency_hours" not in result
    assert result["model_source"] is None
    assert result["input_dates"] == []


def test_threehour_response_temporal_fields_and_metrics_survive_serialization(release):
    origins = threehour_available_origin_range(release)
    parsed_origins = EarthPredictOrigins(**origins).model_dump()
    assert parsed_origins == origins
    assert "input_offset_days" not in parsed_origins
    metrics = _daily_metrics()
    metrics["by_lead"] = [{"lead_step": i, "lead_hours": i * 3, "rmse": 1.0, "mae": 0.5}
                           for i in range(1, 25)]
    metrics["by_horizon"] = [{"horizon_hours": hours, "lead_steps": hours // 3,
                              "rmse": 1.0, "mae": 0.5} for hours in (24, 48, 72)]
    temporal = {"frequency_hours": 3, "step_unit": "hour", "step": 3,
                "time_zone": "UTC", "timestamp_rule": "interval_center"}
    context_payload = _daily_context()
    context_payload.update(dataset_id=EARTH_DATASET_3HOURLY_ID, dataset_version="v1", window=56,
                           horizon=24, origins=origins, grid={"shape": [240, 480]}, **temporal)
    context = EarthPredictContextResponse(**context_payload).model_dump()
    assert all(context[name] == value for name, value in temporal.items())
    run_payload = deepcopy(context_payload)
    run_payload.pop("origins")
    targets = threehour_forecast_timestamps(release, "2021-07-08T01:30:00Z")
    run_payload.update(forecast_origin="2021-07-08T01:30:00Z", origin_split="test", metrics=metrics,
                       input_timestamps=[origins["start"]], target_timestamps=targets, target_dates=targets)
    result = EarthPredictResponse(**run_payload).model_dump(mode="json")
    assert len(result["target_timestamps"]) == 24
    assert result["target_timestamps"] == result["target_dates"]
    assert "lead_day" not in result["metrics"]["by_lead"][0]
    assert result["metrics"]["by_lead"][-1]["lead_hours"] == 72
    assert result["metrics"]["by_horizon"][-1]["lead_steps"] == 24


def test_request_schema_accepts_both_daily_and_utc_datetime_lengths():
    for origin in ("2021-07-08", "2021-07-08T01:30:00Z", "2021-07-08T01:30:00.000000000+00:00"):
        assert EarthPredictRequest(training_task_id=42, forecast_origin=origin).forecast_origin == origin
    with pytest.raises(ValidationError):
        EarthPredictRequest(training_task_id=42, forecast_origin="a" * 41)


def test_nonfinite_response_metrics_remain_rejected():
    metrics = _daily_metrics()
    metrics["by_lead"][0]["rmse"] = float("nan")
    with pytest.raises(ValidationError):
        EarthPredictMetrics(**metrics)
