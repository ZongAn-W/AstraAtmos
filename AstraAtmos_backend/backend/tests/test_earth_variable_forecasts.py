"""Independent inputs and physical single-target forecasts across the real Earth path."""
import copy
import asyncio
import hashlib
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import torch

from test_earth_5deg_dataset import lowres_release
from test_earth_3hourly_training_service import service_environment
from models.training_scripts.earth_daily import run_training
from services.dataset_identity import DatasetRequestError
from services.dataset_registry import DatasetRegistry
from services.earth_dataset import (EarthThreeHourlyWindows, build_threehour_training_cache,
                                    fit_forecast_normalization, fit_threehour_normalization)
from services.earth_dataset_metadata import package_signature
from services.earth_diagnostics import compute_earth_diagnostics
from services.earth_prediction_service import build_prediction_context, run_earth_prediction, compare_earth_test_metrics
from services.earth_task_split import build_earth_task_split
from services.earth_training_artifact import EarthArtifactError, load_earth_training_artifact
from services.earth_training_contract import (build_earth_training_spec, earth_variable_selection,
    normalize_earth_training_hyperparameters, require_earth_uploaded_forecast_compatibility)
from services.user_model_service import UserModelService
from services.user_model_validator import UserModelValidator
from training_backbones.earth_3hourly_uploaded_contract import LOWRES_FEED, forecast_configurations

LOWRES = "earth_merra2_3hourly_5deg_v1"


@pytest.mark.parametrize("target", ["TO3", "U10M", "V10M", "T2M", "SWGDN"])
def test_all_published_targets_have_independent_explicit_inputs(target):
    hypers = normalize_earth_training_hyperparameters(
        {"target_variable": target, "input_variables": ["SWGDN", "U10M"]}, dataset_id=LOWRES)
    assert earth_variable_selection(hypers) == (target, ["U10M", "SWGDN"])
    assert earth_variable_selection({"selected_channels": []}) == ("TO3", ["TO3"])


@pytest.mark.parametrize("inputs", [[], ["T2M", "T2M"], ["unknown"], "TO3", None])
def test_invalid_explicit_inputs_are_never_replaced_by_defaults(inputs):
    with pytest.raises(DatasetRequestError):
        earth_variable_selection({"input_variables": inputs, "target_variable": "T2M"})


def test_unselected_missing_ozone_does_not_enter_statistics_inputs_or_targets(lowres_release, tmp_path):
    release, _ = lowres_release
    package = tmp_path / "release"
    shutil.copytree(release.data_path.parent, package)
    path = package / release.data_path.name
    with netCDF4.Dataset(str(path), "a") as ds:
        ds["TO3"][:] = np.nan
        ds["TO3_valid_mask"][:] = np.uint8(0)
    release = replace(release, data_path=path, signature=package_signature(package, data_file_name=path.name))
    ratios = {"train_ratio": .5, "validation_ratio": .25, "test_ratio": .25}
    split = build_earth_task_split(release.dates, 2, 1, ratios)
    normalization = fit_forecast_normalization(release, ["U10M"], "T2M", task_split=split)
    assert normalization["channel_order"] == ["U10M"]
    assert normalization["target_channel_index"] == -1
    dataset = EarthThreeHourlyWindows.from_release(release, split="test", window=2, horizon=1,
        input_variables=["U10M"], target_variable="T2M", normalization=normalization, task_split=split)
    raw_inputs, raw_target = dataset[0]
    with netCDF4.Dataset(str(path)) as ds:
        expected = np.asarray(ds["T2M"][dataset._offset + 2:dataset._offset + 3])
    np.testing.assert_allclose(dataset.denormalize_ozone(raw_target[:, 0]), expected, rtol=1e-6)
    assert raw_inputs.shape == (2, 1, 36, 72)
    storage = fit_threehour_normalization(release, ["U10M", "T2M"], task_split=split, exact_inputs=True)
    cache = build_threehour_training_cache(release, storage["channel_order"], storage, tmp_path / "cache", full_grid=True)
    dataset.use_training_cache(cache, storage_normalization=storage)
    cached_inputs, cached_target = dataset[0]
    np.testing.assert_array_equal(cached_inputs, raw_inputs)
    np.testing.assert_array_equal(cached_target, raw_target)
    dataset.close_training_cache()


def _uploaded_source():
    feed = {**LOWRES_FEED, "window": [2], "horizon": [1],
            "supported_targets": ["TO3", "T2M"], "supports_independent_inputs": True,
            "input_variable_sets": [["TO3", "U10M"], ["U10M"]]}
    declaration = {"name": "IndependentEarthForecast", "parameters": {}, "datasets": {LOWRES: feed}}
    return "import torch\nfrom torch import nn\nMODEL_SPEC = " + repr(declaration) + '''
class Forecast(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['target_variable'] == config['target_channel']
        assert config['input_variables'] == config['selected_channels']
        assert config['out_channels'] == 1
        self.projection = nn.Conv2d(config['in_channels'], 1, 1)
        self.horizon = config['horizon']
    def forward(self, x):
        y = self.projection(x.mean(dim=1))
        return y[:, None].expand(-1, self.horizon, -1, -1, -1).contiguous()
def build_model(config):
    return Forecast(config)
'''


def test_upload_requires_exact_executed_target_input_evidence(tmp_path):
    source = _uploaded_source()
    model_file = tmp_path / "model.py"
    model_file.write_bytes(source.encode("utf-8"))
    result = UserModelValidator(timeout_seconds=None).validate_file(model_file)
    assert result.ok, result.errors
    report = result.report_dict()
    require_earth_uploaded_forecast_compatibility(report, "T2M", ["U10M"], dataset_id=LOWRES)
    with pytest.raises(DatasetRequestError):
        require_earth_uploaded_forecast_compatibility(report, "T2M", ["V10M"], dataset_id=LOWRES)
    package = SimpleNamespace(id="variable-model", storage_path=str(model_file), validation_report=json.dumps(report),
        content_hash=hashlib.sha256(source.encode()).hexdigest(), validation_status="valid", display_name="Variable model", version=1)
    compatible = UserModelService.earth_compatibility_for_package(package, dataset_id=LOWRES)
    assert compatible["compatible"]
    assert compatible["supported_targets"] == ["TO3", "T2M"]
    assert compatible["input_variable_sets"] == [["TO3", "U10M"], ["U10M"]]
    damaged = copy.deepcopy(report)
    del damaged["earth_datasets"][LOWRES]["forecast_configurations"]
    package.validation_report = json.dumps(damaged)
    assert not UserModelService.earth_compatibility_for_package(package, dataset_id=LOWRES)["compatible"]
    with pytest.raises(DatasetRequestError):
        require_earth_uploaded_forecast_compatibility(damaged, "T2M", ["U10M"], dataset_id=LOWRES)


def test_upload_admission_matrix_is_bounded():
    with pytest.raises(ValueError, match="at most 32"):
        forecast_configurations({"supported_targets": ["TO3", "T2M"], "supports_independent_inputs": True})


def test_temperature_uploaded_task_admission_and_queue_restore_keep_exact_variables(lowres_release, service_environment, tmp_path, monkeypatch):
    module, _ = service_environment
    release, root = lowres_release
    registry = DatasetRegistry(root / "missing_daily", earth_3hourly_5deg_package_dir=root / "lowres")
    from test_earth_3hourly_uploaded import package_at
    package, result = package_at(tmp_path, _uploaded_source())
    assert result.ok, result.errors
    user_models = UserModelService(storage_root=tmp_path / "models")
    async def get_package(*args):
        return package
    user_models.get_package_for_user = get_package
    service = module.TrainingService()
    service._scheduler_started = True
    task = asyncio.run(service.start_training(user_id=1, custom_model_name="Temperature upload", model_script="ignored.py",
        model_source="uploaded", uploaded_model_id=package.id, dataset_id=LOWRES, dataset_registry=registry,
        hyperparameters={"epochs": 1, "window": 2, "horizon": 1, "batch_size": 1,
                         "train_ratio": .5, "validation_ratio": .25, "test_ratio": .25,
                         "target_variable": "T2M", "input_variables": ["U10M"]}, user_model_service=user_models))
    assert task.status == "queued"
    plan = asyncio.run(service._prepare_training_execution(task, dataset_registry=registry))["earth_training_spec"]
    monkeypatch.setattr(service, "_default_dataset_registry", lambda: registry)
    restored = service._restore_3hourly_training_spec(task)
    assert plan["hyperparameters"]["target_variable"] == restored["hyperparameters"]["target_variable"] == "T2M"
    assert plan["hyperparameters"]["input_variables"] == restored["hyperparameters"]["input_variables"] == ["U10M"]
    assert plan["uploaded_model"] == restored["uploaded_model"]


@pytest.mark.parametrize("uploaded", [False, True])
def test_temperature_training_reload_backtest_diagnostics_in_kelvin(lowres_release, tmp_path, uploaded):
    release, root = lowres_release
    registry = DatasetRegistry(root / "missing_daily", earth_3hourly_5deg_package_dir=root / "lowres")
    binding = registry.build_training_binding(LOWRES)
    ratios = {"train_ratio": .5, "validation_ratio": .25, "test_ratio": .25}
    split = build_earth_task_split(release.dates, 2, 1, ratios)
    reference = None
    if uploaded:
        source = _uploaded_source()
        reference = {"package_id": "independent", "version": 1, "content_hash": hashlib.sha256(source.encode()).hexdigest(),
                     "source_text": source, "display_name": "IndependentEarthForecast", "param_schema": {},
                     "custom_model_params": {}, "contract_schema": LOWRES_FEED["schema"]}
    hypers = {**ratios, "epochs": 1, "batch_size": 2, "window": 2, "horizon": 1,
              "input_variables": ["U10M"], "target_variable": "T2M", "model_source": "uploaded" if uploaded else "official"}
    spec = build_earth_training_spec(task_id=982, dataset_binding=binding, task_split=split,
                                    hyperparameters=hypers, uploaded_model=reference)
    output = tmp_path / "temperature.pth"
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        run_training(spec, output, registry, device="cpu", cache_root=tmp_path / "cache")
        checkpoint = load_earth_training_artifact(output, expected_binding=binding,
            expected_hyperparameters={**spec["hyperparameters"], "_earth_task_split": split})
        assert checkpoint.training_contract["target"] == "T2M"
        assert checkpoint.training_contract["target_unit"] == "K"
        assert checkpoint.metrics["metric_units"]["mse"] == "K^2"
        assert checkpoint.normalization["target_channel_index"] == -1
        task = SimpleNamespace(id=982, status="completed", output_model_path=str(output),
            model_source=hypers["model_source"], hyperparameters=json.dumps({**spec["hyperparameters"], "_earth_task_split": split}), **binding)
        context = build_prediction_context(task, registry)
        assert context.target == "T2M" and context.target_unit == "K"
        prediction = run_earth_prediction(task, context.origins["start"], registry, device="cpu")
        assert prediction["target"] == "T2M" and prediction["target_unit"] == "K"
        assert prediction["input_channel_order"] == ["U10M"]
        assert prediction["metrics"]["metric_units"]["rmse"] == "K"
        with netCDF4.Dataset(str(release.data_path)) as ds:
            first = np.flatnonzero(release.dates == np.datetime64(context.origins["start"].removesuffix("Z")))[0] + 1
            expected_reference = np.asarray(ds["T2M"][first])
        np.testing.assert_allclose(prediction["reference"][0]["field"], expected_reference, rtol=1e-7)
        diagnostic = compute_earth_diagnostics(task, registry, sample_windows=2, scatter_points=16, pfi_repeats=1, device="cpu")
        assert diagnostic["target"] == "T2M" and diagnostic["unit"] == "K"
        assert diagnostic["scatter"]["unit"] == "K"
        assert diagnostic["histogram"]["unit"] == "K"
        assert diagnostic["pfi"]["unit"] == "K"
        with pytest.raises(EarthArtifactError, match="target does not match"):
            load_earth_training_artifact(output, expected_hyperparameters={**spec["hyperparameters"], "target_variable": "TO3"})
    finally:
        torch.set_num_threads(threads)
