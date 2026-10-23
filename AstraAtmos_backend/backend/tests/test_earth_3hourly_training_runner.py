"""Real optimization and strict checkpoint reload on a small synthetic release.

The trusted registry stand-in uses three ten-day splits to keep smoke bounded.
Production split dates remain enforced by the separate package validator.
"""

from __future__ import annotations

import copy
import asyncio
import json
from datetime import date, timedelta
from types import MappingProxyType

import numpy as np
import pytest
import torch

from models.training_scripts.earth_daily import (
    _ArrayDataset, _SpatialTileDataset, run_training,
)
from scripts.build_earth_merra2_3hourly import DATA_FILE, _new_output
from services.dataset_registry import DatasetRegistry
from services.earth_dataset import CHANNELS, UNITS, THREE_HOURLY_SCHEMA
from services.earth_dataset_metadata import VerifiedEarthRelease, package_signature
from services.earth_training_artifact import (
    EarthArtifactError, load_earth_training_artifact,
)
from services.earth_training_contract import build_earth_training_spec
from services.earth_task_split import build_earth_task_split
from services.netcdf_read_lock import netcdf_read_lock
from test_earth_3hourly_training_service import service_environment

DATASET_ID = "earth_merra2_3hourly_v1"


@pytest.fixture(scope="module")
def synthetic_training_release(tmp_path_factory):
    root = tmp_path_factory.mktemp("threehour_runner")
    path = root / DATA_FILE
    days = [date(2020, 1, 1) + timedelta(days=index) for index in range(30)]
    times = np.datetime64("2020-01-01T01:30", "ns") + np.arange(240) * np.timedelta64(3, "h")
    spatial = np.arange(240, dtype="float32")[:, None] * .001 + np.arange(480, dtype="float32")[None, :] * .0005
    with netcdf_read_lock():
        stored = _new_output(path, days, "a" * 64)
        try:
            for index, channel in enumerate(CHANNELS):
                for step in range(240):
                    stored[channel][step] = np.float32(100 + index + step * .01) + spatial
                    stored[f"{channel}_valid_mask"][step] = np.uint8(1)
        finally:
            stored.close()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    splits = {
        name: {"start": str(times[start].astype("datetime64[D]")),
               "end": str(times[start + 79].astype("datetime64[D]")), "steps": 80, "days": 10}
        for name, start in (("train", 0), ("validation", 80), ("test", 160))
    }
    metadata = {
        "dataset_id": DATASET_ID, "dataset_version": "v1", "planet": "earth",
        "schema": THREE_HOURLY_SCHEMA, "dataset_fingerprint": "b" * 64,
        "manifest_sha256": "c" * 64, "data_sha256": "d" * 64,
        "frequency_hours": 3, "step_unit": "hour", "step": 3,
        "time": {"kind": "datetime", "time_zone": "UTC", "label": "interval_center",
                 "start": "2020-01-01T01:30:00Z", "end": "2020-01-30T22:30:00Z", "count": 240},
        "grid": {"shape": [240, 480]}, "grid_shape": [240, 480],
        "channel_order": list(CHANNELS), "variables": [
            {"id": channel, "units": unit} for channel, unit in zip(CHANNELS, UNITS)],
        "splits": splits,
    }
    return VerifiedEarthRelease(
        metadata=metadata, signature=package_signature(root, data_file_name=DATA_FILE),
        dates=times, latitude=-89.625 + np.arange(240) * .75,
        longitude=-179.625 + np.arange(480) * .75, fields=MappingProxyType({}), data_path=path,
    )


def registry_for(release):
    registry = DatasetRegistry(release.data_path.parent / "missing_daily",
                               earth_3hourly_package_dir=release.data_path.parent)

    def get_snapshot(dataset_id, expected_fingerprint=None):
        assert dataset_id == DATASET_ID
        if expected_fingerprint is not None:
            assert expected_fingerprint == release.metadata["dataset_fingerprint"]
        return release

    registry.get_earth_snapshot = get_snapshot
    return registry


@pytest.fixture(scope="module")
def smoke_checkpoint(synthetic_training_release, tmp_path_factory):
    registry = registry_for(synthetic_training_release)
    binding = registry.build_training_binding(DATASET_ID)
    spec = build_earth_training_spec(
        task_id=701, dataset_binding=binding,
        task_split=build_earth_task_split(synthetic_training_release.dates, 56, 24,
                                        {name + "_ratio": 1/3 for name in ("train", "validation", "test")}),
        hyperparameters={**{name + "_ratio": 1/3 for name in ("train", "validation", "test")},
                         "epochs": 1, "batch_size": 8, "seed": 5,
                         "selected_channels": ["U10M", "V10M", "T2M", "SWGDN"]},
    )
    output = tmp_path_factory.mktemp("threehour_smoke_weights") / "task_701.pth"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        result = run_training(spec, output, registry, device="cpu", cache_root=output.parent / "cache")
    finally:
        torch.set_num_threads(previous_threads)
    return output, result, binding, spec


def test_smoke_training_publishes_56_to_24_checkpoint(smoke_checkpoint):
    output, result, binding, spec = smoke_checkpoint
    checkpoint = load_earth_training_artifact(output, expected_binding=binding,
                                              expected_hyperparameters={**spec["hyperparameters"], "_earth_task_split": spec["task_split"]}, expected_task_id=701)
    assert result["split_window_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert checkpoint.model_config["window"] == 56
    assert checkpoint.model_config["horizon"] == 24
    assert [checkpoint.model_config["height"], checkpoint.model_config["width"]] == [240, 480]
    assert checkpoint.training_contract["frequency_hours"] == 3
    assert checkpoint.training_contract["step_unit"] == "hour"
    assert checkpoint.training_contract["timestamp_rule"] == "interval_center"
    assert checkpoint.training_contract["input_channel_order"] == list(CHANNELS)
    assert checkpoint.run["memory_strategy"] == "normalized_memmap_spatial_tiles"
    assert checkpoint.run["spatial_tile_shape"] == [24, 48]
    assert checkpoint.normalization["fit_time_end"] == "2020-01-10T22:30:00Z"
    for split in checkpoint.metrics["splits"].values():
        assert set(split["overall"]) == {"mse", "rmse", "mae", "r2", "mape", "smape"}
        assert len(split["by_lead"]) == 24
        assert [row["lead_hours"] for row in split["by_lead"]] == list(range(3, 73, 3))
        assert [row["horizon_hours"] for row in split["by_horizon"]] == [24, 48, 72]


def test_tensor_wrapper_reads_only_the_requested_window():
    class CountingWindows:
        def __init__(self):
            self.calls = []
        def __len__(self):
            return 10_000
        def __getitem__(self, index):
            self.calls.append(index)
            return np.zeros((56, 1, 2, 2), dtype="float32"), np.zeros((24, 1, 2, 2), dtype="float32")
    windows = CountingWindows()
    wrapper = _ArrayDataset(windows)
    assert windows.calls == []
    wrapper[123]
    assert windows.calls == [123]


def test_spatial_tiles_cover_every_grid_cell_exactly_once():
    wrapper = _SpatialTileDataset([None])
    coverage = np.zeros((240, 480), dtype="int8")
    for latitude, longitude in wrapper.tiles:
        coverage[latitude, longitude] += 1
    assert len(wrapper) == 100
    assert np.all(coverage == 1)


def test_threehour_checkpoint_cannot_bind_to_daily_task(smoke_checkpoint):
    output, _, binding, _ = smoke_checkpoint
    daily_binding = {**copy.deepcopy(binding), "dataset_id": "earth_merra2_daily_v2", "dataset_version": "v2"}
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(output, expected_binding=daily_binding)


@pytest.mark.parametrize("corruption", [None, "daily_schema", "fingerprint", "legacy_metrics", "missing_r2"])
def test_queued_task_completion_uses_strict_threehour_checkpoint(
    service_environment, synthetic_training_release, smoke_checkpoint, monkeypatch, corruption,
):
    """Drive queue creation and the real parent completion gate with trained weights."""
    module, _ = service_environment
    registry = registry_for(synthetic_training_release)
    service = module.TrainingService()
    service._scheduler_started = True
    captured = {}

    class Process:
        pid = 4242
        returncode = 0
        @property
        def stdout(self):
            return self
        def readline(self):
            return ""
        def close(self):
            pass
        def wait(self):
            return self.returncode

    def popen(args, **kwargs):
        captured.update(args=args, env=kwargs["env"])
        spec = json.loads(kwargs["env"]["ASTRAATMOS_EARTH_TRAINING_SPEC"])
        checkpoint = load_earth_training_artifact(smoke_checkpoint[0])
        payload = copy.deepcopy(checkpoint.payload)
        payload["run"]["task_id"] = spec["task_id"]
        if corruption == "daily_schema":
            payload["artifact_schema"] = "astraatmos_earth_forecast_checkpoint_v1"
        elif corruption == "fingerprint":
            payload["dataset_binding"]["dataset_fingerprint"] = "e" * 64
        elif corruption == "legacy_metrics":
            from test_earth_metrics_v2 import legacy_payload
            payload = legacy_payload(payload)
        elif corruption == "missing_r2":
            payload["metrics"]["splits"]["test"]["overall"].pop("r2")
        torch.save(payload, args[-1])
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", popen)

    async def scenario():
        task = await service.start_training(
            user_id=1, custom_model_name="threehour-completion",
            model_script="client-ignored.py", dataset_id=DATASET_ID, dataset_registry=registry,
            hyperparameters=smoke_checkpoint[3]["hyperparameters"],
        )
        plan = await service._prepare_training_execution(task, dataset_registry=registry)
        await service._run_training_subprocess(task.id, **plan)
        return await service.get_task(task.id)

    task = asyncio.run(scenario())
    assert task.status == ("failed" if corruption else "completed")
    assert task.dataset_id == DATASET_ID
    assert task.dataset_fingerprint == synthetic_training_release.metadata["dataset_fingerprint"]
    assert captured["args"][-2] == "--output_path"
    assert "--window" not in captured["args"]
    spec = json.loads(captured["env"]["ASTRAATMOS_EARTH_TRAINING_SPEC"])
    assert spec["training_profile"]["window"] == 56
    assert spec["hyperparameters"]["horizon"] == 24
    if not corruption:
        assert json.loads(task.metrics)["splits"]["test"]["by_horizon"][2]["horizon_hours"] == 72


def test_threehour_prediction_requires_a_completed_artifact():
    from types import SimpleNamespace
    from services.dataset_identity import DatasetRequestError
    from services.earth_prediction_service import _require_completed_earth_task
    with pytest.raises(DatasetRequestError) as error:
        _require_completed_earth_task(SimpleNamespace(dataset_id=DATASET_ID, status="completed"))
    assert (error.value.status_code, error.value.code) == (409, "earth_prediction_task_not_completed")


def test_gridpoint_dlinear_spatial_tiles_preserve_full_field_outputs():
    from services.earth_model_source import ModelSourcePlan, build_earth_model_for_plan
    model, _, _ = build_earth_model_for_plan(ModelSourcePlan(
        model_source="official", input_channel_order=["TO3", "U10M"],
        linear_hidden_layers=2, dataset_id=DATASET_ID,
    ))
    inputs = torch.randn(1, 56, 2, 4, 6)
    with torch.no_grad():
        # Tiling is a property of the shared DLinear weights, so it is asserted on
        # the model itself: the three-hour contract fixes every call to 24x48.
        complete = model(inputs)
        tiled = torch.empty_like(complete)
        for lat in range(0, 4, 2):
            for lon in range(0, 6, 3):
                tiled[..., lat:lat + 2, lon:lon + 3] = model(inputs[..., lat:lat + 2, lon:lon + 3])
    torch.testing.assert_close(tiled, complete)


def test_child_registry_uses_only_the_frozen_server_package_path(tmp_path):
    from models.training_scripts.earth_daily import EarthTrainingError, _build_registry
    path = tmp_path / "package" / "earth_merra2_3hourly.nc"
    spec = {"dataset_binding": {"dataset_id": DATASET_ID}, "data_path": str(path)}
    registry = _build_registry(spec)
    assert registry._earth_specs[DATASET_ID]["path"] == path.parent
    with pytest.raises(EarthTrainingError):
        _build_registry({**spec, "data_path": "relative/earth_merra2_3hourly.nc"})
