"""Spatial labels follow loader rows/columns for every Mars prediction path."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import netCDF4
import numpy as np
import pytest
import torch

from services import inference_service as inference_module
from services.mars_checkpoint import build_mars_checkpoint
from services.mars_data_service import _coords, load_mars_arrays, prepare_scaled_volume


RATIOS = {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}


def _write_source(path, *, raw=False, latitude=None, longitude=None, bad_axis=None, bad_value=None):
    lat = np.linspace(-90, 90, 37, dtype=np.float32) if raw else np.array([80, 55, 35, 15, -5, -30, -50, -70], dtype=np.float32)
    lon = np.arange(0, 360, 5, dtype=np.float32) if raw else np.array([30, 45, 65, 90, 110, 130, 150, 170], dtype=np.float32)
    if latitude is not None:
        lat = np.asarray(latitude, dtype=np.float32)
    if longitude is not None:
        lon = np.asarray(longitude, dtype=np.float32)
    field = (np.arange(12, dtype=np.float32)[:, None, None] + lat[None, :, None] + 100 + np.arange(len(lon))[None, None, :] * 0.1).astype(np.float32)
    with netCDF4.Dataset(str(path), "w") as dataset:
        for dimension, size in (("time", 12), ("lat", len(lat)), ("lon", len(lon))):
            dataset.createDimension(dimension, size)
        dataset.createVariable("Ls", "f4", ("time",))[:] = np.arange(12) * 5
        target = dataset.createVariable("O3COL" if raw else "o3col", "f4", ("time", "lat", "lon"))
        target[:] = field
        target.units = "um-atm"
        for name, labels in (("lat", lat), ("lon", lon)):
            if name == bad_axis:
                if bad_value == "missing":
                    continue
                labels = labels.copy()
                if bad_value == "nan":
                    labels[0] = np.nan
                elif bad_value == "short":
                    labels = labels[:-1]
            dimension = f"{name}_labels"
            dataset.createDimension(dimension, len(labels))
            dataset.createVariable(name, "f4", (dimension,))[:] = labels
    return field, lat, lon


@pytest.fixture
def mars_grid_sources(tmp_path):
    directories = {name: tmp_path / name for name in ("raw", "openmars", "mcd")}
    for directory in directories.values():
        directory.mkdir()
    raw_field, _, raw_lon = _write_source(directories["raw"] / "MCD_MY27.nc", raw=True)
    openmars_field, lat, lon = _write_source(directories["openmars"] / "openmars_MY27.nc")
    return SimpleNamespace(
        directories=directories,
        expected={
            "mcd_overview": ((raw_field[:, :-1] + raw_field[:, 1:])[:, ::-1] / 2,
                             np.arange(87.5, -90, -5, dtype=np.float32), raw_lon),
            "openmars_mcd": (openmars_field, lat, lon),
        },
    )


def _prepare_case(sources, dataset_id, tmp_path):
    volume = prepare_scaled_volume(
        training_dataset=dataset_id, openmars_dir=sources.directories["openmars"],
        mcd_dir=sources.directories["mcd"], raw_dir=sources.directories["raw"],
        selected_channels=[], window=2, horizon=2, split_ratios=RATIOS,
    )
    payload = build_mars_checkpoint(model_state_dict={"bias": torch.zeros(1)}, volume=volume,
                                    selected_channels=[], window=2, horizon=2, split_ratios=RATIOS, seed=11)
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    return volume, path


@pytest.mark.parametrize("dataset_id", ["mcd_overview", "openmars_mcd"])
@pytest.mark.parametrize("model_source", ["official", "uploaded"])
@pytest.mark.parametrize("version", ["schema", "legacy"])
def test_north_high_fields_keep_source_latitude_and_longitude(mars_grid_sources, dataset_id, model_source, version, tmp_path, monkeypatch):
    from training_backbones import user_model_runner

    volume, path = _prepare_case(mars_grid_sources, dataset_id, tmp_path)
    if version == "legacy":
        torch.save({"bias": torch.zeros(1)}, path)
    field, expected_latitude, expected_longitude = mars_grid_sources.expected[dataset_id]
    expected_truth = field[2:4]
    expected_prediction = expected_truth * 1.5 + 3
    prediction_scaled = (expected_prediction - volume.y_mean) / (volume.y_std + 1e-6)
    class GradientModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))
        def forward(self, inputs, ls=None):
            return torch.as_tensor(prediction_scaled, dtype=torch.float32, device=inputs.device)[None, :, None] + self.bias
    model = GradientModel()
    service = inference_module.InferenceService()
    service.device = torch.device("cpu")
    service.openmars_dir = mars_grid_sources.directories["openmars"]
    service.mcd_dir = mars_grid_sources.directories["mcd"]
    monkeypatch.setattr(service, "_load_official_task_model", lambda **kwargs: (model, False))
    monkeypatch.setattr(user_model_runner, "load_uploaded_model", lambda *args: model)
    if version == "legacy" and model_source == "official":
        monkeypatch.setattr(service, "_is_legacy_official_state_dict", lambda state: True)
    hypers = {"training_dataset": dataset_id, "window": 2, "horizon": 2,
              "selected_channels": [], "model_architecture": "dlinear", **RATIOS,
              "_uploaded_model_path": str(tmp_path / "gradient.source")}
    task = SimpleNamespace(id=42, output_model_path=str(path), model_source=model_source,
                           model_script="demo3.py", custom_model_name="gradient", hyperparameters=json.dumps(hypers))
    data_dirs = {"MCD_RAW_3H_DIR": str(mars_grid_sources.directories["raw"])} if dataset_id == "mcd_overview" else {
        "ASTRAATMOS_OPENMARS_DIR": str(mars_grid_sources.directories["openmars"]),
        "ASTRAATMOS_MCD_DIR": str(mars_grid_sources.directories["mcd"]),
    }
    if version == "legacy" and model_source == "official":
        actual_volume = service._load_official_task_volume([], 2, 2, data_dirs=data_dirs)
        prediction_scaled = (expected_prediction - actual_volume.y_mean) / (actual_volume.y_std + 1e-6)
    result = service._predict_task_with_context(task, hypers, ls_start=0, horizon=2, data_dirs=data_dirs)
    for key, expected_field in (
        ("prediction", expected_prediction), ("ground_truth", expected_truth),
        ("residual", expected_prediction - expected_truth),
    ):
        for index, frame in enumerate(result[key]):
            np.testing.assert_array_equal(frame["lat"], expected_latitude)
            np.testing.assert_array_equal(frame["lon"], expected_longitude)
            np.testing.assert_allclose(frame["field"], expected_field[index], atol=5e-5)
            assert frame["field"][0][0] > frame["field"][-1][0]
        assert result[key][0]["lat"][0] > result[key][0]["lat"][-1]
    if dataset_id == "mcd_overview":
        assert result["prediction"][0]["lat"][0] == 87.5
        assert result["prediction"][0]["lat"][-1] == -87.5
        assert result["prediction"][0]["lon"] == list(range(0, 360, 5))
    else:
        assert result["prediction"][0]["lat"] == [80, 55, 35, 15, -5, -30, -50, -70]
        assert result["prediction"][0]["lon"] == [30, 45, 65, 90, 110, 130, 150, 170]


@pytest.mark.parametrize("latitude", [[80, 35, -5, -70], [-70, -5, 35, 80]])
def test_openmars_array_order_is_retained_even_when_source_axis_is_ascending(tmp_path, latitude):
    directory = tmp_path / "openmars"
    directory.mkdir()
    field, lat, lon = _write_source(directory / "openmars_MY27.nc", latitude=latitude)
    data = load_mars_arrays(training_dataset="openmars_mcd", openmars_dir=directory,
                            mcd_dir=tmp_path / "mcd", raw_dir=tmp_path / "raw", selected_channels=[])
    np.testing.assert_array_equal(data.latitude, lat)
    np.testing.assert_array_equal(data.longitude, lon)
    np.testing.assert_array_equal(data.ozone, field)


@pytest.mark.parametrize("axis", ["latitude", "longitude"])
@pytest.mark.parametrize("fault", [None, [], [1], [1, float("nan")], [1, float("inf")], [[1, 2]]])
def test_coordinate_validation_rejects_missing_empty_mismatched_and_nonfinite_axes(axis, fault):
    coordinates = {"latitude": [80, -70], "longitude": [30, 170]}
    coordinates[axis] = fault
    with pytest.raises(ValueError, match=f"Mars prediction {axis}"):
        inference_module.InferenceService._prediction_coordinates(**coordinates, spatial_shape=(2, 2))


@pytest.mark.parametrize("model_source", ["official", "uploaded"])
@pytest.mark.parametrize("axis", ["latitude", "longitude"])
@pytest.mark.parametrize("fault", [None, [1], [1, float("nan")]])
def test_common_prediction_output_rejects_invalid_coordinates(model_source, axis, fault, monkeypatch):
    service = inference_module.InferenceService.__new__(inference_module.InferenceService)
    monkeypatch.setattr(service, "_load_task_checkpoint", lambda task: None)
    coordinates = {"latitude": [80, -70], "longitude": [30, 170]}
    coordinates[axis] = fault
    window_result = (np.zeros((1, 2, 2)), np.ones((1, 2, 2)), 0, 1,
                     [], [0, 1], [2], {}, coordinates["latitude"], coordinates["longitude"])
    monkeypatch.setattr(service, f"_predict_{model_source}_task_window", lambda **kwargs: window_result)
    output_guard = Mock(side_effect=AssertionError("Do not serialize fields with invalid coordinates"))
    monkeypatch.setattr(service, "_fields_to_dicts", output_guard)
    task = SimpleNamespace(id=42, model_source=model_source, custom_model_name="gradient",
                           output_model_path="unused.pth")
    with pytest.raises(ValueError, match=f"Mars prediction {axis}"):
        service._predict_task_with_context(task, {}, ls_start=0, horizon=1)
    output_guard.assert_not_called()


@pytest.mark.parametrize("model_source", ["official", "uploaded"])
@pytest.mark.parametrize("axis", ["latitude", "longitude"])
def test_invalid_volume_coordinates_fail_before_model_loading(mars_grid_sources, model_source, axis, tmp_path, monkeypatch):
    volume, path = _prepare_case(mars_grid_sources, "openmars_mcd", tmp_path)
    invalid = replace(volume, **{axis: None})
    service = inference_module.InferenceService()
    service.device = torch.device("cpu")
    guard = Mock(side_effect=AssertionError("Do not load models with missing coordinates"))
    task = SimpleNamespace(output_model_path=str(path), model_script="demo3.py", hyperparameters="{}")
    hypers = {"window": 2, "horizon": 2, "selected_channels": [], "_uploaded_model_path": "unused.source"}
    if model_source == "official":
        monkeypatch.setattr(service, "_load_official_task_volume", lambda *args, **kwargs: invalid)
        monkeypatch.setattr(service, "_load_official_task_model", guard)
        predict = service._predict_official_task_window
    else:
        from training_backbones import user_model_runner
        monkeypatch.setattr(service, "_prepare_uploaded_task_volume", lambda *args, **kwargs: ([], invalid))
        monkeypatch.setattr(user_model_runner, "load_uploaded_model", guard)
        predict = service._predict_uploaded_task_window
    with pytest.raises(ValueError, match=f"Mars prediction {axis} coordinates are missing"):
        predict(task, hypers, ls_start=0, horizon=2)
    guard.assert_not_called()


@pytest.mark.parametrize("axis", ["lat", "lon"])
@pytest.mark.parametrize("fault", ["missing", "nan", "short"])
@pytest.mark.parametrize("dataset_id", ["mcd_overview", "openmars_mcd"])
def test_invalid_source_file_coordinates_fail_prediction(tmp_path, dataset_id, axis, fault, monkeypatch):
    directory = tmp_path / dataset_id
    directory.mkdir()
    raw = dataset_id == "mcd_overview"
    _write_source(directory / ("MCD_MY27.nc" if raw else "openmars_MY27.nc"),
                  raw=raw, bad_axis=axis, bad_value=fault)
    service = inference_module.InferenceService()
    service.device = torch.device("cpu")
    service.openmars_dir = directory
    service.mcd_dir = tmp_path / "mcd"
    monkeypatch.setattr(service, "_checkpoint_normalization", lambda task: None)
    monkeypatch.setattr(service, "_load_task_state_dict", lambda task: {})
    guard = Mock(side_effect=AssertionError("Coordinates must be validated before model loading"))
    monkeypatch.setattr(service, "_load_official_task_model", guard)
    task = SimpleNamespace(hyperparameters=json.dumps({"selected_channels": []}), model_script="demo3.py")
    data_dirs = {"MCD_RAW_3H_DIR": str(directory)} if raw else None
    with pytest.raises(ValueError, match="coordinates?|missing lat"):
        service._predict_official_task_window(task, {"window": 2, "horizon": 2}, 0, 2, data_dirs=data_dirs)
    guard.assert_not_called()


@pytest.mark.parametrize("ascending", [False, True])
def test_raw_mcd_36_row_gradient_matches_loader_target_latitude(tmp_path, ascending):
    directory = tmp_path / "raw"
    directory.mkdir()
    latitude = np.arange(87.5, -90, -5, dtype=np.float32)
    input_lat = latitude[::-1] if ascending else latitude
    field, _, lon = _write_source(directory / "MCD_MY27.nc", raw=True, latitude=input_lat)
    data = load_mars_arrays(training_dataset="mcd_overview", openmars_dir=tmp_path / "unused",
                            mcd_dir=tmp_path / "mcd", raw_dir=directory, selected_channels=[])
    np.testing.assert_array_equal(data.latitude, latitude)
    np.testing.assert_array_equal(data.longitude, lon)
    np.testing.assert_array_equal(data.ozone, field[:, ::-1] if ascending else field)


def test_masked_coordinates_are_not_cleaned_to_zero(tmp_path):
    path = tmp_path / "masked.nc"
    with netCDF4.Dataset(str(path), "w") as dataset:
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 2)
        dataset.createVariable("lat", "f4", ("lat",), fill_value=-999)[:] = [80, -999]
        dataset.createVariable("lon", "f4", ("lon",))[:] = [30, 170]
        lat, lon = _coords(dataset)
    assert np.isnan(lat[-1])
    with pytest.raises(ValueError, match="finite"):
        inference_module.InferenceService._prediction_coordinates(lat, lon, (2, 2))


def test_cache_identity_changes_with_mars_output_coordinate_contract(tmp_path, monkeypatch):
    from services import prediction_analysis_cache as cache
    path = tmp_path / "weights.pth"
    path.write_bytes(b"fixture")
    task = SimpleNamespace(id=42, output_model_path=str(path), hyperparameters="{}", dataset_id="mcd_overview")
    original = cache.build_artifact_fingerprint(task)
    monkeypatch.setattr(cache, "MARS_OUTPUT_COORDINATE_POLICY", "old_guessed_coordinates")
    assert cache.build_artifact_fingerprint(task) != original


def test_inference_has_no_guessed_geographic_output_axes():
    source = Path(inference_module.__file__).read_text(encoding="utf-8")
    assert "np.linspace(-87.5, 87.5" not in source
    assert "np.linspace(-180.0, 175.0" not in source


def test_project_openmars_coordinates_when_available():
    from config import OPENMARS_DIR
    from services.netcdf_read_lock import netcdf_read_lock
    path = next(iter(sorted(Path(OPENMARS_DIR).glob("*.nc"))), None)
    if path is None:
        pytest.skip("Project OpenMARS files are unavailable")
    with netcdf_read_lock(), netCDF4.Dataset(str(path)) as dataset:
        lat, lon = _coords(dataset)
        shape = dataset.variables["o3col"].shape[-2:]
    checked_lat, checked_lon = inference_module.InferenceService._prediction_coordinates(lat, lon, shape)
    assert checked_lat[0] > checked_lat[-1]
    np.testing.assert_array_equal(checked_lon, lon)
