"""Regression coverage for complete-window Earth historical predictions.

These tests build verified checkpoints from the shared synthetic release.  No
training is performed; the checkpoint builder is the same one used by the
diagnostics integration tests.
"""

import numpy as np
import pytest
import torch

from services.dataset_identity import DatasetRequestError
from services.earth_dataset import EarthThreeHourlyWindows
from services.earth_prediction_cache import (
    TASK_SPLIT_BOUNDARY_VERSION,
    EarthPredictionCache,
    build_earth_prediction_cache_key,
)
from services.earth_prediction_service import (
    build_prediction_context,
    run_earth_prediction,
)
from services.earth_diagnostics import EarthDiagnosticsCache, compute_earth_diagnostics

from test_earth_3hourly_training_runner import synthetic_training_release
from test_earth_diagnostics_integration import _create_bundle

WINDOW, HORIZON = 3, 2


@pytest.fixture(scope="module", autouse=True)
def prediction_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _origin_for_index(release, index):
    return np.datetime_as_string(release.dates[index], unit="s") + "Z"


def _expected_indices(bundle, split_name=None):
    names = [split_name] if split_name else ["train", "validation", "test"]
    return [index for name in names
            for index in range(bundle.split["ranges"][name]["raw_start"] + WINDOW - 1,
                               bundle.split["ranges"][name]["raw_end"] - HORIZON)]


@pytest.fixture(scope="module")
def task_bundles(synthetic_training_release, tmp_path_factory):
    return {source: _create_bundle(
        synthetic_training_release,
        tmp_path_factory.mktemp("task-split-prediction-" + source),
        source, window=WINDOW, horizon=HORIZON, custom_split=True,
        task_id=2810 if source == "official" else 2811,
    ) for source in ("official", "uploaded")}


@pytest.fixture(params=["official", "uploaded"])
def task_bundle(request, task_bundles):
    return task_bundles[request.param]


def test_context_uses_identical_exact_complete_window_origins_for_both_sources(task_bundles):
    origins = []
    for bundle in task_bundles.values():
        context = build_prediction_context(bundle.task, bundle.registry)
        expected = [_origin_for_index(bundle.release, index) for index in _expected_indices(bundle)]
        assert context.origins["dates"] == context.origins["timestamps"] == expected
        assert context.origins["count"] == 228 == sum(
            entry["window_count"] for entry in bundle.split["ranges"].values()
        )
        assert context.metrics["split_policy"] == "earth_raw_utc_timeline_v1"
        origins.append(context.origins)
    assert origins[0] == origins[1]


def test_actual_run_acceptance_matches_context_and_never_reads_data_on_cache_hits(
    task_bundle, monkeypatch
):
    """Every context origin reaches the run validator; no extra origin is accepted."""
    import services.earth_prediction_service as service

    bundle = task_bundle
    context = build_prediction_context(bundle.task, bundle.registry)
    calls = []
    sentinel = object()

    class SentinelCache:
        def get(self, key):
            calls.append(key)
            return sentinel

        def put(self, key, result):
            pytest.fail("a cache hit must not write a new prediction")

    monkeypatch.setattr(service, "_serialize_threehour_result", lambda cached: cached)
    monkeypatch.setattr(
        service,
        "_read_threehour_forecast_block",
        lambda *args, **kwargs: pytest.fail("accepted cache hits must not read source data"),
    )
    cache = SentinelCache()
    accepted = []
    for index in range(len(bundle.release.dates)):
        origin = _origin_for_index(bundle.release, index)
        previous_calls = len(calls)
        try:
            result = run_earth_prediction(bundle.task, origin, bundle.registry, device="cpu", cache=cache)
        except DatasetRequestError as error:
            assert (error.status_code, error.code) == (422, "earth_prediction_origin_out_of_range")
            assert len(calls) == previous_calls
        else:
            accepted.append(origin)
            assert result is sentinel
    assert accepted == context.origins["dates"]
    assert len(calls) == context.origins["count"]


def test_legal_window_runs_real_inference_and_crossing_origin_is_rejected_before_cache(
    task_bundle, monkeypatch
):
    import services.earth_prediction_service as service

    bundle = task_bundle
    context = build_prediction_context(bundle.task, bundle.registry)
    test_range = bundle.split["ranges"]["test"]
    legal = _origin_for_index(bundle.release, test_range["raw_start"] + WINDOW - 1)

    # This is a real model/data run, with a private cache so it cannot reuse an
    # artifact produced by another test.
    result = run_earth_prediction(
        bundle.task,
        legal,
        bundle.registry,
        device="cpu",
        cache=EarthPredictionCache(max_bytes=32 * 1024 * 1024),
    )
    assert result["forecast_origin"] == legal
    assert result["origin_split"] == "test"
    assert legal in context.origins["dates"]
    assert result["input_timestamps"] == [
        _origin_for_index(bundle.release, index)
        for index in range(test_range["raw_start"], test_range["raw_start"] + WINDOW)
    ]
    assert len(result["target_timestamps"]) == HORIZON
    assert np.asarray([entry["field"] for entry in result["prediction"]]).shape == (HORIZON, 240, 480)
    assert np.isfinite(np.asarray([entry["field"] for entry in result["prediction"]])).all()

    crossing = _origin_for_index(bundle.release, bundle.split["ranges"]["train"]["raw_end"] - 1)
    class ForbiddenCache:
        def get(self, key):
            pytest.fail("cross-boundary origins must be rejected before cache lookup")

        def put(self, key, result):
            pytest.fail("cross-boundary origins must not populate the cache")

    monkeypatch.setattr(
        service,
        "_read_threehour_forecast_block",
        lambda *args, **kwargs: pytest.fail("cross-boundary origins must be rejected before data reads"),
    )
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(bundle.task, crossing, bundle.registry, device="cpu", cache=ForbiddenCache())
    assert (error.value.status_code, error.value.code) == (
        422,
        "earth_prediction_origin_out_of_range",
    )


def test_diagnostics_test_dataset_and_scope_use_the_same_task_split(task_bundle, monkeypatch):
    import services.earth_diagnostics as diagnostics

    bundle = task_bundle
    checkpoint = bundle.payload["training_contract"]
    dataset = EarthThreeHourlyWindows.from_release(
        bundle.release,
        split="test",
        window=checkpoint["window"],
        horizon=checkpoint["horizon"],
        selected_channels=checkpoint["input_channel_order"],
        normalization=bundle.payload["normalization"],
        task_split=checkpoint["task_split"],
    )
    context = build_prediction_context(bundle.task, bundle.registry)
    expected_test = [_origin_for_index(bundle.release, index)
                     for index in _expected_indices(bundle, "test")]
    dataset_origins = [_origin_for_index(bundle.release, dataset._offset + index + WINDOW - 1)
                       for index in range(len(dataset))]
    assert dataset_origins == expected_test
    assert [origin for origin in context.origins["dates"] if origin in expected_test] == expected_test
    reads = []
    original_read = diagnostics._threehour_values

    def record_read(source, channel, start, stop, **kwargs):
        reads.append((start, stop))
        return original_read(source, channel, start, stop, **kwargs)

    monkeypatch.setattr(diagnostics, "_threehour_values", record_read)
    result = compute_earth_diagnostics(
        bundle.task,
        bundle.registry,
        sample_windows=2,
        scatter_points=1,
        histogram_bins=5,
        include_pfi=True,
        pfi_windows=1,
        pfi_repeats=1,
        device="cpu",
        cache=EarthDiagnosticsCache(),
    )
    assert result["scope"]["available_window_count"] == len(dataset)
    assert result["scope"]["split"] == "test"
    assert all(row["forecast_origin"] in expected_test for row in result["scope"]["windows"])
    assert all(row["forecast_origin"] in expected_test for row in result["pfi"]["scope"]["windows"])
    assert result["sample_metrics"]["window_count"] == result["scope"]["window_count"] == 2
    assert sum(result["histogram"]["counts"]) == 2 * HORIZON * 240 * 480
    test_range = bundle.split["ranges"]["test"]
    assert reads and all(test_range["raw_start"] <= start < stop <= test_range["raw_end"]
                         for start, stop in reads)


def test_prediction_cache_identity_includes_split_boundary_policy():
    baseline = dict(
        planet="earth",
        dataset_id="earth_merra2_3hourly_v1",
        dataset_version="v1",
        dataset_fingerprint="a" * 64,
        task_id=2812,
        forecast_origin="2020-01-01T04:30:00Z",
        target_timestamps=["2020-01-01T07:30:00Z"],
        checkpoint_sha256="b" * 64,
    )
    legacy = build_earth_prediction_cache_key(**baseline)
    strict = build_earth_prediction_cache_key(
        **baseline, split_boundary_policy=TASK_SPLIT_BOUNDARY_VERSION
    )
    assert legacy != strict

