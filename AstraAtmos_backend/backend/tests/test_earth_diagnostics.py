"""Deterministic small-grid checks for Earth diagnostic statistics and PFI."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services.dataset_identity import DatasetRequestError
from services.earth_diagnostics import (
    EarthDiagnosticsCache,
    compute_earth_diagnostics,
    permutation_mappings,
    select_window_indices,
)


class LastHistoricalOzone(torch.nn.Module):
    def forward(self, values):
        return values[:, -1:, 0:1]


def install_synthetic_diagnostics(monkeypatch, *, available=4, task_split=None):
    import services.earth_diagnostics as service

    class FakeDataset(SimpleNamespace):
        def __len__(self):
            return self.length

    dates = np.datetime64("2020-01-01T00", "h") + np.arange(20).astype("timedelta64[h]")
    release = SimpleNamespace(dates=dates, latitude=np.array([-1., 1.]), longitude=np.arange(4, dtype=float))
    dataset = FakeDataset(
        _release=release, _offset=0, window=2, horizon=1, input_channels=["TO3", "U10M"],
        normalization={"mean": [0., 0.], "scale": [1., 1.]}, lat=release.latitude,
        lon=release.longitude, length=available,
    )
    contract = {
        "window": 2, "horizon": 1, "input_channel_order": ["TO3", "U10M"],
        "input_units": ["DU", "m s-1"], "split_policy": "test_policy",
        "task_split": task_split, "split_ranges": {"test": {"date_start": "2020-01-01", "date_end": "2020-01-02"}},
    }
    checkpoint = SimpleNamespace(
        training_contract=contract,
        dataset_binding={"dataset_id": "earth_merra2_3hourly_v1", "dataset_version": "v1",
                         "dataset_fingerprint": "f" * 64, "dataset_snapshot": {"grid": {}}},
        normalization=dataset.normalization, model_source="official",
        model_config={}, metrics={"schema": "synthetic_v2", "unit": "DU", "aggregation": "grid_uniform",
                                  "splits": {"test": {"window_count": available,
                                                       "overall": {"rmse": 9.}}}},
        model_identity=lambda: {"model_source": "official", "model_architecture": "synthetic"},
    )
    task = SimpleNamespace(id=81, dataset_id="earth_merra2_3hourly_v1", status="completed", user_id=1,
                           output_model_path="synthetic.pth", hyperparameters="{}")
    calls = {"task_split": None, "tile": None, "model_inputs": []}

    def from_release(cls, release_arg, **kwargs):
        calls["task_split"] = kwargs.get("task_split")
        return dataset

    monkeypatch.setattr(service.EarthThreeHourlyWindows, "from_release", classmethod(from_release))
    monkeypatch.setattr(service, "_require_completed_earth_task", lambda _: None)
    monkeypatch.setattr(service, "_threehour_checkpoint_sha", lambda _: "a" * 64)
    monkeypatch.setattr(service, "_load_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(service, "_sync_release", lambda *_: release)
    monkeypatch.setattr(service, "_validate_checkpoint_split_ranges", lambda *_: None)
    monkeypatch.setattr(service, "_model_source_warnings", lambda _: [])
    monkeypatch.setattr(service, "build_earth_model_from_checkpoint", lambda _: (LastHistoricalOzone(), []))
    monkeypatch.setattr(service, "earth_forward_for_model", lambda model, inputs, **_: model(inputs))
    monkeypatch.setattr(service, "TILE_SHAPE", (1, 2))
    monkeypatch.setattr(service, "BATCH_SIZE", 2)

    values = np.array([1., 5., 9., 13.])

    def read_tile(_ds, _dataset, selected, lat, lon):
        calls["tile"] = (lat, lon)
        width = min(2, len(dataset.lon) - lon)
        marker = lat * 20. + lon * 2.
        refs = np.stack([np.full((1, 1, 1, width), values[index] + marker) for index in selected])
        inputs = np.zeros((len(selected), 2, 2, 1, width), dtype="float32")
        for row, index in enumerate(selected):
            inputs[row, :, 0] = refs[row, 0] + 1.  # Prediction is reference + 1 DU.
            inputs[row, :, 1] = 100. + index * 100. + marker
        return inputs, refs.astype("float32")

    monkeypatch.setattr(service, "_read_tile", read_tile)

    @contextmanager
    def open_dataset(_):
        yield object()

    monkeypatch.setattr(service, "_open_verified_dataset", open_dataset)
    original_predict = service._predict_tile

    def record_predict(model, inputs, cp, device):
        calls["model_inputs"].append((calls["tile"], np.array(inputs[:, -1, :, 0, 0], copy=True)))
        return original_predict(model, inputs, cp, device)

    monkeypatch.setattr(service, "_predict_tile", record_predict)
    return service, task, SimpleNamespace(registry=object(), release=release), dataset, checkpoint, calls


def run_synthetic(monkeypatch, *, seed=42, task_split=None, cache=None):
    service, task, bundle, dataset, checkpoint, calls = install_synthetic_diagnostics(
        monkeypatch, task_split=task_split)
    result = service.compute_earth_diagnostics(
        task, bundle.registry, sample_windows=2, scatter_points=7, histogram_bins=8,
        seed=seed, pfi_repeats=2, include_pfi=True, device="cpu", cache=cache or EarthDiagnosticsCache(),
    )
    return result, calls


def test_physical_residual_scatter_pairing_histogram_and_checkpoint_scope(monkeypatch):
    result, _ = run_synthetic(monkeypatch)
    assert result["unit"] == result["target_unit"] == "DU"
    assert result["full_test_metrics"]["scope"] == "full_test"
    assert result["full_test_metrics"]["source"] == "verified_checkpoint_test_metrics"
    assert result["full_test_metrics"]["metrics"]["overall"]["rmse"] == 9.
    assert result["scope"]["split"] == "test" and result["scope"]["window_count"] == 2
    assert result["scope"]["available_window_count"] == 4
    assert result["scope"]["grid_shape"] == [2, 4]
    assert result["scope"]["spatial_coverage"]["all_grid_cells"] is True
    assert result["scope"]["valid_points"] == 16
    assert result["sample_metrics"]["overall"]["mae"] == pytest.approx(1.)
    assert result["histogram"]["residual_definition"] == "prediction-reference"
    assert result["histogram"]["summary"] == pytest.approx(
        {"mean": 1., "std": 0., "min": 1., "max": 1., "count": 16})
    assert sum(result["histogram"]["counts"]) == 16
    assert len(result["scatter"]["reference"]) == len(result["scatter"]["prediction"]) == 7
    assert np.asarray(result["scatter"]["prediction"]) - np.asarray(result["scatter"]["reference"]) == pytest.approx(np.ones(7))
    assert result["scatter"]["scope"] == "sampled_points_from_selected_test_windows"
    assert result["scatter"]["reference_line"]["x"] == result["scatter"]["reference_line"]["y"]


def test_pfi_is_seeded_channel_sensitive_and_reuses_one_mapping_across_spatial_blocks(monkeypatch):
    first, calls = run_synthetic(monkeypatch, seed=17)
    ozone, unused = first["pfi"]["items"]
    assert ozone["channel"] == "TO3" and ozone["status"] == "completed"
    assert ozone["importance_mean"] > 0
    assert len(ozone["repeats"]) == 2
    assert unused["channel"] == "U10M" and unused["importance_mean"] == pytest.approx(0., abs=1e-7)
    mappings = first["pfi"]["source_window_mappings"]
    assert len(mappings) == 2 and all(sorted(row) == [0, 1] and row[0] != 0 and row[1] != 1 for row in mappings)

    # Each spatial tile receives the same source window for each PFI repeat.
    selected = first["scope"]["selected_window_indices"]
    expected_sources = np.asarray([1., 5., 9., 13.])[np.asarray(selected)[np.asarray(mappings[0])]] + 1
    for tile in [(0, 0), (0, 2), (1, 0), (1, 2)]:
        marker = tile[0] * 20. + tile[1] * 2.
        tile_calls = [values[:, 0] - marker for key, values in calls["model_inputs"]
                      if key == tile and len(values) == 2]
        matches = [values for values in tile_calls if np.allclose(values, expected_sources)]
        assert len(matches) == 2
    again, _ = run_synthetic(monkeypatch, seed=17)
    assert again["scope"]["selected_window_indices"] == first["scope"]["selected_window_indices"]
    assert again["scatter"]["reference"] == first["scatter"]["reference"]
    assert again["pfi"]["source_window_mappings"] == mappings
    assert [item["importance_mean"] for item in again["pfi"]["items"]] == pytest.approx(
        [item["importance_mean"] for item in first["pfi"]["items"]])


def test_manifest_legacy_and_custom_task_split_are_read_from_checkpoint(monkeypatch):
    task_split = {"policy": "earth_task_split_v1", "test": {"start_index": 9}}
    result, calls = run_synthetic(monkeypatch, task_split=task_split)
    assert calls["task_split"] == task_split
    assert result["full_test_metrics"]["test_range"] == {"date_start": "2020-01-01", "date_end": "2020-01-02"}
    assert select_window_indices(19, 4, 42) == select_window_indices(19, 4, 42)
    assert select_window_indices(19, 4, 42) != select_window_indices(19, 4, 43)
    mappings = permutation_mappings(4, 5, 42)
    assert mappings == permutation_mappings(4, 5, 42)
    assert all(sorted(row) == [0, 1, 2, 3] and all(i != value for i, value in enumerate(row)) for row in mappings)


def test_legacy_checkpoint_uses_manifest_test_partition(monkeypatch):
    _, task, bundle, dataset, checkpoint, calls = install_synthetic_diagnostics(monkeypatch, task_split=None)
    result = compute_earth_diagnostics(task, bundle.registry, sample_windows=2, scatter_points=4,
                                       histogram_bins=8, seed=42, include_pfi=False,
                                       device="cpu", cache=EarthDiagnosticsCache())
    assert calls["task_split"] is None
    assert result["full_test_metrics"]["split_policy"] == "test_policy"


def test_nonfinite_inputs_and_missing_test_windows_fail_explicitly(monkeypatch):
    service, task, bundle, dataset, checkpoint, calls = install_synthetic_diagnostics(monkeypatch)
    original_read = service._read_tile

    def nonfinite(*args, **kwargs):
        inputs, references = original_read(*args, **kwargs)
        inputs[0, -1, 0, 0, 0] = np.nan
        return inputs, references

    monkeypatch.setattr(service, "_read_tile", nonfinite)
    with pytest.raises(DatasetRequestError, match="non-finite"):
        service.compute_earth_diagnostics(task, bundle.registry, sample_windows=2, scatter_points=4,
                                          histogram_bins=8, seed=42, include_pfi=False, device="cpu",
                                          cache=EarthDiagnosticsCache())
    with pytest.raises(DatasetRequestError, match="no complete windows"):
        select_window_indices(0, 2, 42)


def test_single_test_window_reports_pfi_unavailable_without_fabricating_values(monkeypatch):
    import services.earth_diagnostics as service
    svc, task, bundle, dataset, checkpoint, calls = install_synthetic_diagnostics(monkeypatch, available=1)
    result = svc.compute_earth_diagnostics(task, bundle.registry, sample_windows=1, scatter_points=4,
                                           histogram_bins=8, seed=42, include_pfi=True, pfi_repeats=2,
                                           device="cpu", cache=EarthDiagnosticsCache())
    assert result["pfi"]["status"] == "unavailable"
    assert result["pfi"]["reason_code"] == "earth_pfi_insufficient_windows"
    assert result["pfi"]["items"] == []
    assert result["pfi"]["baseline_value"] is None


def test_cache_hits_are_identity_scoped_and_missing_saved_metrics_stay_missing(monkeypatch):
    import services.earth_diagnostics as service
    svc, task, bundle, dataset, checkpoint, calls = install_synthetic_diagnostics(monkeypatch)
    cache = EarthDiagnosticsCache()
    arguments = dict(sample_windows=2, scatter_points=4, histogram_bins=8, seed=42,
                     pfi_repeats=1, include_pfi=False, device="cpu", cache=cache)
    first = svc.compute_earth_diagnostics(task, bundle.registry, **arguments)
    model_calls = len(calls["model_inputs"])
    second = svc.compute_earth_diagnostics(task, bundle.registry, **arguments)
    assert second["cache"]["hit"] is True and len(calls["model_inputs"]) == model_calls
    monkeypatch.setattr(svc, "_threehour_checkpoint_sha", lambda _: "b" * 64)
    third = svc.compute_earth_diagnostics(task, bundle.registry, **arguments)
    assert third["cache"]["hit"] is False and third["cache"]["key"] != first["cache"]["key"]

    checkpoint.metrics["splits"].pop("test")
    monkeypatch.setattr(svc, "_threehour_checkpoint_sha", lambda _: "c" * 64)
    absent = svc.compute_earth_diagnostics(task, bundle.registry, **arguments)
    assert absent["full_test_metrics"]["status"] == "not_provided"
    assert absent["full_test_metrics"]["metrics"] is None
