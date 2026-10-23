"""Synthetic configurable-horizon training, reload, legacy inference and comparison."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from models.training_scripts.earth_daily import run_training
from services.earth_training_contract import (
    build_earth_training_spec, earth_3hourly_metrics_contract, EARTH_3HOURLY_METRICS_SCHEMA_V2,
)
from services.earth_task_split import build_earth_task_split
from services.earth_training_artifact import load_earth_training_artifact, save_earth_artifact_atomic, FULL_METRIC_KEYS
from services.earth_prediction_service import run_earth_prediction, compare_earth_test_metrics
from services.earth_prediction_cache import EarthPredictionCache
from test_earth_3hourly_training_runner import synthetic_training_release, registry_for
from test_earth_3hourly_uploaded import package_at, source_with_spec, spec as upload_spec
from test_earth_metrics_v2 import legacy_payload


@pytest.mark.parametrize("source", ["official", "uploaded"])
def test_configurable_horizon_pipeline_and_legacy_prediction(synthetic_training_release, tmp_path, monkeypatch, source):
    dataset_id = "earth_merra2_3hourly_v1"
    registry = registry_for(synthetic_training_release)
    binding = registry.build_training_binding(dataset_id)
    ratios = {key + "_ratio": 1/3 for key in ("train", "validation", "test")}
    split = build_earth_task_split(synthetic_training_release.dates, 71, 9, ratios)
    reference = None
    if source == "uploaded":
        declaration = upload_spec()
        declaration["datasets"][dataset_id]["window"] = [71]
        declaration["datasets"][dataset_id]["horizon"] = [9]
        text = source_with_spec(declaration)
        package, validation = package_at(tmp_path, text)
        assert validation.ok
        reference = {"package_id": package.id, "display_name": package.display_name, "version": 1,
                     "content_hash": package.content_hash, "source_path": package.storage_path,
                     "source_text": text, "param_schema": validation.param_schema, "custom_model_params": {"bias": True}}
    spec = build_earth_training_spec(task_id=1801, dataset_binding=binding, task_split=split,
        hyperparameters={**ratios, "window": 71, "horizon": 9, "epochs": 1, "batch_size": 8, "selected_channels": ["U10M"]},
        # This regression compares legacy equal-weight metrics. Generate actual
        # v2 statistics rather than relabeling a new area-weighted checkpoint.
        uploaded_model=reference,
        metrics_contract=earth_3hourly_metrics_contract(EARTH_3HOURLY_METRICS_SCHEMA_V2))
    path = tmp_path / "new.pth"
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        run_training(spec, path, registry, device="cpu", cache_root=tmp_path / "cache")
        checkpoint = load_earth_training_artifact(path)
        for metrics in checkpoint.metrics["splits"].values():
            assert set(metrics["overall"]) == set(FULL_METRIC_KEYS)
            assert len(metrics["by_lead"]) == 9
            assert [row["horizon_hours"] for row in metrics["by_horizon"]] == [24, 27]
        hypers = {**spec["hyperparameters"], "_earth_task_split": split}
        if reference:
            hypers.update(_uploaded_model_id=reference["package_id"], _uploaded_model_version=1,
                          _uploaded_model_content_hash=reference["content_hash"])
        task = SimpleNamespace(id=1801, **binding, user_id=1, status="completed", output_model_path=str(path),
                               hyperparameters=json.dumps(hypers), custom_model_name=source)
        old = legacy_payload(checkpoint.payload)
        old["run"]["task_id"] = 1802
        legacy_path = tmp_path / "old.pth"
        save_earth_artifact_atomic(old, legacy_path)
        legacy = copy.copy(task)
        legacy.id, legacy.output_model_path = 1802, str(legacy_path)
        before = legacy_path.read_bytes()
        import services.earth_prediction_service as service
        monkeypatch.setattr(service, "_field_list", lambda values: [{"shape": list(values.shape)}])
        origin = str(synthetic_training_release.dates[230].astype("datetime64[s]")) + "Z"
        result = run_earth_prediction(legacy, origin, registry, device="cpu", cache=EarthPredictionCache())
        assert result["horizon"] == 9
        assert set(result["metrics"]["overall"]) == set(FULL_METRIC_KEYS)
        assert result["metrics"]["aggregation"] == "user_forecast_origin_lead_grid_uniform"
        monkeypatch.setattr(service, "earth_forward_for_model", lambda *a, **k: pytest.fail("Comparison reran inference"))
        comparison = compare_earth_test_metrics([task, legacy], registry)
        new_metrics, old_metrics = [item["metrics"] for item in comparison["items"]]
        assert set(new_metrics["overall"]) == set(FULL_METRIC_KEYS)
        assert set(old_metrics["overall"]) == {"rmse", "mae"}
        assert legacy_path.read_bytes() == before
    finally:
        torch.set_num_threads(previous)
