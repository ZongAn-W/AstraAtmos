"""Mars runner, checkpoint and inference normalization regressions."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import torch

from services.mars_checkpoint import build_mars_checkpoint, validate_mars_checkpoint
from services.mars_data_service import MCD_VARS_MAP, prepare_scaled_volume
from services.prediction_volume_cache import clear_scaled_volumes


RATIOS = {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}
CHANNEL_CASES = [[], ["U"], ["U", "V"], ["U", "V", "T"]]
BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_SOURCE = '''
import torch
from torch import nn
MODEL_SPEC = {"name": "CheckpointProbe", "parameters": {}}
class Probe(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.horizon = horizon
    def forward(self, x):
        return x[:, -1:, :1].repeat(1, self.horizon, 1, 1, 1) + self.bias
def build_model(config):
    return Probe(config["horizon"])
'''


@pytest.fixture
def mars_data(tmp_path):
    directory = tmp_path / "mcd_overview"
    directory.mkdir()
    path = directory / "MCD_MY27_overview.nc"
    time = np.arange(30, dtype=np.float32)[:, None, None]
    grid = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    fields = {"O3": time * 2 + grid + 5}
    for index, channel in enumerate(("U", "V", "T"), start=1):
        fields[channel] = time ** 2 * index + grid * (index + 1) + 100 * index
    with netCDF4.Dataset(str(path), "w") as dataset:
        for name, count in (("time", 30), ("lat", 2), ("lon", 3)):
            dataset.createDimension(name, count)
        dataset.createVariable("lat", "f4", ("lat",))[:] = [-45, 45]
        dataset.createVariable("lon", "f4", ("lon",))[:] = [-120, 0, 120]
        dataset.createVariable("Ls", "f4", ("time",))[:] = np.arange(30) * 10
        for channel, field in fields.items():
            name = "o3col" if channel == "O3" else MCD_VARS_MAP[channel][0]
            dataset.createVariable(name, "f4", ("time", "lat", "lon"))[:] = field
    mcd_dir = tmp_path / "mcd"
    mcd_dir.mkdir()
    with netCDF4.Dataset(str(mcd_dir / "MCD_MY27.nc"), "w") as dataset:
        for name, count in (("time", 30), ("hour", 1), ("lat", 2), ("lon", 3)):
            dataset.createDimension(name, count)
        dataset.createVariable("Ls", "f4", ("time",))[:] = np.arange(30) * 10
        for channel in ("U", "V", "T"):
            dataset.createVariable(MCD_VARS_MAP[channel][0], "f4", ("time", "hour", "lat", "lon"))[:] = fields[channel][:, None]
    return SimpleNamespace(
        directory=directory, fields=fields,
        kwargs=dict(
            training_dataset="mcd_overview", openmars_dir=tmp_path / "openmars",
            mcd_dir=mcd_dir, raw_dir=directory, window=2, horizon=2,
            split_ratios=RATIOS,
        ),
    )


def _checkpoint(volume, channels):
    return build_mars_checkpoint(
        model_state_dict={"bias": torch.zeros(1)}, volume=volume,
        selected_channels=channels, window=2, horizon=2, split_ratios=RATIOS, seed=11,
    )


@pytest.mark.parametrize("channels", CHANNEL_CASES)
def test_checkpoint_round_trip_matches_model_input(mars_data, channels, tmp_path):
    volume = prepare_scaled_volume(**mars_data.kwargs, selected_channels=channels)
    payload = _checkpoint(volume, channels)
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    validate_mars_checkpoint(loaded, selected_channels=channels, window=2, horizon=2, grid_shape=(2, 3))
    assert loaded["training_contract"]["selected_channels"] == channels
    contract = loaded["training_contract"]
    assert contract["split_policy"] == "strict_timeline_windows_v1"
    assert set(contract["split_window_counts"]) == {"train", "validation", "test"}
    for name in ("train", "validation", "test"):
        assert contract["split_ranges"][name]["window_count"] == contract["split_window_counts"][name]
        assert contract["split_ranges"][name]["window_ranges"]
    normalization = loaded["normalization"]
    assert normalization["input_channel_order"] == ["O3", *channels]
    for key in ("input_mean", "input_scale", "constant_channel_mask"):
        assert len(normalization[key]) == volume.values.shape[-1] == 1 + len(channels)
    reloaded = prepare_scaled_volume(
        **mars_data.kwargs, selected_channels=channels, normalization=normalization,
    )
    np.testing.assert_array_equal(reloaded.values, volume.values)
    np.testing.assert_array_equal(reloaded.y_scaled, volume.y_scaled)


def test_checkpoint_accepts_strict_split_without_validation_windows(mars_data):
    ratios = {"train_ratio": 0.8, "validation_ratio": 0.0, "test_ratio": 0.2}
    volume = prepare_scaled_volume(
        **{**mars_data.kwargs, "split_ratios": ratios}, selected_channels=[]
    )
    payload = build_mars_checkpoint(
        model_state_dict={"bias": torch.zeros(1)}, volume=volume,
        selected_channels=[], window=2, horizon=2, split_ratios=ratios, seed=11,
    )
    validate_mars_checkpoint(payload, selected_channels=[], window=2, horizon=2, grid_shape=(2, 3))
    assert payload["training_contract"]["split_window_counts"]["validation"] == 0
    assert payload["training_contract"]["split_ranges"]["validation"]["window_ranges"] == []


@pytest.mark.parametrize("order", [["U", "V"], ["U", "O3", "V"], ["O3", "V", "U"], ["O3", "U", "T"]])
def test_checkpoint_rejects_wrong_input_channel_order(mars_data, order):
    channels = ["U", "V"]
    payload = _checkpoint(prepare_scaled_volume(**mars_data.kwargs, selected_channels=channels), channels)
    payload["normalization"]["input_channel_order"] = order
    with pytest.raises(ValueError, match="input_channel_order"):
        validate_mars_checkpoint(payload)
    with pytest.raises(ValueError, match="input_channel_order"):
        prepare_scaled_volume(**mars_data.kwargs, selected_channels=channels, normalization=payload["normalization"])


@pytest.mark.parametrize("key", ["input_mean", "input_scale", "constant_channel_mask"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_checkpoint_rejects_wrong_statistic_count(mars_data, key, delta):
    channels = ["U"]
    payload = _checkpoint(prepare_scaled_volume(**mars_data.kwargs, selected_channels=channels), channels)
    values = payload["normalization"][key]
    if delta < 0:
        values.pop()
    else:
        values.append(copy.deepcopy(values[-1]))
    with pytest.raises(ValueError, match="input normalization metadata"):
        validate_mars_checkpoint(payload)
    with pytest.raises(ValueError, match="input normalization metadata"):
        prepare_scaled_volume(**mars_data.kwargs, selected_channels=channels, normalization=payload["normalization"])


@pytest.mark.parametrize("key", ["input_mean", "input_scale"])
@pytest.mark.parametrize("fault", ["shape", "nan", "inf", "nonnumeric"])
def test_checkpoint_rejects_invalid_statistic_arrays(mars_data, key, fault):
    payload = _checkpoint(prepare_scaled_volume(**mars_data.kwargs, selected_channels=[]), [])
    array = payload["normalization"][key][0]
    if fault == "shape":
        array.pop()
    else:
        array[0][0] = {"nan": float("nan"), "inf": float("inf"), "nonnumeric": "invalid"}[fault]
    with pytest.raises(ValueError, match="input normalization"):
        validate_mars_checkpoint(payload)


@pytest.mark.parametrize("scale", [0, -1])
def test_checkpoint_rejects_nonpositive_input_scale(mars_data, scale):
    payload = _checkpoint(prepare_scaled_volume(**mars_data.kwargs, selected_channels=[]), [])
    payload["normalization"]["input_scale"][0][0][0] = scale
    with pytest.raises(ValueError, match="scale must be positive"):
        validate_mars_checkpoint(payload)


@pytest.mark.parametrize("channels", [["O3"], ["V", "U"], ["U", "U"], ["unknown"]])
def test_checkpoint_builder_rejects_ambiguous_auxiliary_channels(mars_data, channels):
    volume = prepare_scaled_volume(**mars_data.kwargs, selected_channels=channels)
    with pytest.raises(ValueError, match="selected_channels"):
        _checkpoint(volume, channels)


@pytest.mark.parametrize("key,value", [("target_mean", float("nan")), ("target_scale", float("inf")), ("target_scale", -1)])
def test_checkpoint_rejects_invalid_target_statistics(mars_data, key, value):
    payload = _checkpoint(prepare_scaled_volume(**mars_data.kwargs, selected_channels=[]), [])
    payload["normalization"][key] = value
    with pytest.raises(ValueError, match="target normalization"):
        validate_mars_checkpoint(payload)


def _demo3():
    spec = importlib.util.spec_from_file_location("mars_checkpoint_demo3", BACKEND_DIR / "models/training_scripts/demo3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("channels", CHANNEL_CASES)
def test_official_and_uploaded_runners_save_same_contract(mars_data, channels, tmp_path, monkeypatch):
    from training_backbones import user_model_runner

    official = _demo3()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MCD_RAW_3H_DIR", str(mars_data.directory))
    source = tmp_path / "probe.source"
    source.write_text(MODEL_SOURCE, encoding="utf-8")
    artifacts = []
    for name, runner in (("official", official), ("uploaded", user_model_runner)):
        path = tmp_path / f"{name}.pth"
        argv = [str(runner.__file__), "--epochs", "1", "--batch_size", "64",
                "--window", "2", "--horizon", "2", "--selected_channels", ",".join(channels),
                "--training_dataset", "mcd_overview", "--output_path", str(path)]
        argv += ["--model_architecture", "dlinear"] if name == "official" else ["--uploaded_model_path", str(source)]
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_mars_checkpoint(payload, selected_channels=channels, window=2, horizon=2, grid_shape=(2, 3))
        artifacts.append(payload)
    assert artifacts[0]["training_contract"] == artifacts[1]["training_contract"]
    assert artifacts[0]["normalization"] == artifacts[1]["normalization"]
    assert artifacts[0]["data_binding"] == artifacts[1]["data_binding"]


@pytest.mark.parametrize("channels", CHANNEL_CASES)
@pytest.mark.parametrize("model_source", ["official", "uploaded"])
def test_inference_applies_full_input_order_and_target_statistics(mars_data, channels, model_source, tmp_path, monkeypatch):
    from services.inference_service import InferenceService

    volume = prepare_scaled_volume(**mars_data.kwargs, selected_channels=channels)
    payload = _checkpoint(volume, channels)
    normalization = payload["normalization"]
    # Distinct constants make index shifts observable, including O3 vs target stats.
    for index in range(1 + len(channels)):
        normalization["input_mean"][index] = np.full((2, 3), 10 * (index + 1)).tolist()
        normalization["input_scale"][index] = np.full((2, 3), index + 2).tolist()
    normalization.update(target_mean=123.0, target_scale=7.0)
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    task = SimpleNamespace(output_model_path=str(path))
    saved = InferenceService._checkpoint_normalization(task)
    service = InferenceService()
    service.openmars_dir = mars_data.kwargs["openmars_dir"]
    service.mcd_dir = mars_data.kwargs["mcd_dir"]
    # Exercise normal caching, including different checkpoints with the same data.
    from services import inference_service
    monkeypatch.setattr(inference_service, "MCD_RAW_3H_DIR", mars_data.directory)
    clear_scaled_volumes()
    def load(norm):
        if model_source == "uploaded":
            _, result = service._prepare_uploaded_task_volume(
                {"selected_channels": channels, "training_dataset": "mcd_overview", **RATIOS},
                2, 2, normalization=norm,
            )
            return result
        return service._load_official_task_volume(
            [MCD_VARS_MAP[channel] for channel in channels], 2, 2,
            split_ratios=RATIOS, normalization=norm,
        )
    if model_source == "official":
        # The same ozone/Ls file also provides the OpenMARS side of this fixture.
        monkeypatch.setattr(service, "openmars_dir", mars_data.directory)
    try:
        loaded = load(saved)
        for index, channel in enumerate(["O3", *channels]):
            expected = (mars_data.fields[channel] - 10 * (index + 1)) / (index + 2 + 1e-6)
            np.testing.assert_allclose(loaded.values[..., index], expected, rtol=1e-6)
        assert loaded.y_mean == 123.0 and loaded.y_std == 7.0
        np.testing.assert_allclose(loaded.y_scaled * (loaded.y_std + 1e-6) + loaded.y_mean, mars_data.fields["O3"], atol=1e-5)
        assert load(saved) is loaded
        other = copy.deepcopy(saved)
        other.update(target_mean=321.0, target_scale=9.0)
        other["input_mean"][0][0][0] += 50
        different = load(other)
        assert different is not loaded
        assert different.y_mean == 321.0 and different.y_std == 9.0
        assert different.values[0, 0, 0, 0] != loaded.values[0, 0, 0, 0]
        wrong = copy.deepcopy(saved)
        wrong["input_channel_order"] = channels
        with pytest.raises(ValueError, match="input_channel_order"):
            load(wrong)
    finally:
        clear_scaled_volumes()


@pytest.mark.parametrize("model_source", ["official", "uploaded"])
def test_prediction_response_denormalizes_o3_with_target_statistics(mars_data, model_source, tmp_path, monkeypatch):
    from services.inference_service import InferenceService
    from training_backbones import user_model_runner

    channels = ["U"]
    volume = prepare_scaled_volume(**mars_data.kwargs, selected_channels=channels)
    payload = _checkpoint(volume, channels)
    payload["normalization"].update(target_mean=123.0, target_scale=7.0)
    # Deliberately different input O3 statistics must never denormalize the target.
    payload["normalization"]["input_mean"][0] = np.full((2, 3), 1000.0).tolist()
    payload["normalization"]["input_scale"][0] = np.full((2, 3), 50.0).tolist()
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    hypers = {"selected_channels": channels, "window": 2, "horizon": 2, **RATIOS,
              "model_architecture": "dlinear", "training_dataset": "mcd_overview",
              "_uploaded_model_path": str(tmp_path / "probe.source")}
    task = SimpleNamespace(id=42, model_source=model_source, custom_model_name="probe",
                           output_model_path=str(path), hyperparameters=json.dumps(hypers),
                           model_script="demo3.py")
    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))
        def forward(self, inputs, ls=None):
            return torch.full((inputs.shape[0], 2, 1, 2, 3), 2.0, device=inputs.device)
    service = InferenceService()
    service.device = torch.device("cpu")
    monkeypatch.setattr(service, "_load_official_task_model", lambda **kwargs: (ConstantModel(), False))
    monkeypatch.setattr(user_model_runner, "load_uploaded_model", lambda *args: ConstantModel())
    result = service._predict_task_with_context(
        task, hypers, ls_start=0, horizon=2,
        data_dirs={"MCD_RAW_3H_DIR": str(mars_data.directory)},
    )
    for step in result["prediction"]:
        np.testing.assert_allclose(step["field"], 137.000002, atol=1e-5)
    for index, step in enumerate(result["ground_truth"]):
        np.testing.assert_allclose(step["field"], mars_data.fields["O3"][2 + index], atol=1e-5)
    assert result["model_info"]["normalization_source"] == "checkpoint"
    assert result["model_info"]["selected_channels"] == channels
