import sys
import tempfile
import uuid
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training_backbones import user_model_runner as runner_module  # noqa: E402
from training_backbones.user_model_runner import (  # noqa: E402
    assert_prediction_shape,
    build_uploaded_model_config,
    load_uploaded_model,
    parse_json_arg,
    prepare_tensors,
)
from services.mars_data_service import MarsDataError  # noqa: E402
from training_backbones.uploaded_model_contract import (  # noqa: E402
    attach_uploaded_model_contract,
    expand_topography_batch,
    normalize_auxiliary_inputs,
    run_uploaded_model,
)


MODEL_SOURCE = """
from torch import nn
MODEL_SPEC = {"name":"RunnerTiny", "parameters":{"hidden_dim":{"type":"int","default":8,"min":4,"max":32}}}
class RunnerTiny(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
    def forward(self, x):
        return x[:, -1:, :1].repeat(1, self.horizon, 1, 1, 1)
def build_model(config):
    return RunnerTiny(config["horizon"])
"""

LS_RUNNER_MODEL_SOURCE = """
from torch import nn
MODEL_SPEC = {
    "name": "RunnerLsTiny",
    "auxiliary_inputs": {
        "ls": {
            "required": True,
            "shape": ["batch", "window"],
            "dtype": "float32",
            "unit": "degree",
        }
    },
    "parameters": {},
}
class RunnerLsTiny(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.received_ls = None
    def forward(self, x, ls):
        self.received_ls = ls
        return x[:, -1:, :1].repeat(1, self.horizon, 1, 1, 1)
def build_model(config):
    return RunnerLsTiny(config["horizon"])
"""

BAD_SHAPE_MODEL_SOURCE = """
from torch import nn
MODEL_SPEC = {"name":"BadShapeTiny", "parameters":{}}
class BadShapeTiny(nn.Module):
    def forward(self, x):
        return x[:, -1, 0]
def build_model(config):
    return BadShapeTiny()
"""

LS_MODEL_SPEC = {
    "name": "LsModel",
    "auxiliary_inputs": {
        "ls": {
            "required": True,
            "shape": ["batch", "window"],
            "dtype": "float32",
            "unit": "degree",
        }
    },
}

TOPOGRAPHY_MODEL_SPEC = {
    "name": "TopographyModel",
    "auxiliary_inputs": {
        "topography": {
            "required": True,
            "shape": ["batch", 1, "height", "width"],
            "dtype": "float32",
            "unit": "meter",
        }
    },
}

LS_TOPOGRAPHY_MODEL_SPEC = {
    "name": "LsTopographyModel",
    "auxiliary_inputs": {
        **LS_MODEL_SPEC["auxiliary_inputs"],
        **TOPOGRAPHY_MODEL_SPEC["auxiliary_inputs"],
    },
}


class _LegacyRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.argument_count = 0

    def forward(self, x):
        self.argument_count = 1
        return x[:, -1:, :1]


class _LsRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_ls = None

    def forward(self, x, ls):
        self.received_ls = ls
        return x[:, -1:, :1]


class _TopographyRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_topography = None

    def forward(self, x, topography):
        self.received_topography = topography
        return x[:, -1:, :1]


class _LsTopographyRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arguments = None

    def forward(self, x, ls, topography):
        self.arguments = (x, ls, topography)
        return x[:, -1:, :1]


def test_legacy_uploaded_model_receives_only_x():
    model = attach_uploaded_model_contract(_LegacyRecordingModel(), {"name": "Legacy"})
    x = torch.zeros(2, 3, 1, 8, 16)

    run_uploaded_model(model, x, torch.ones(2, 3), context="legacy test")

    assert model.argument_count == 1


def test_ls_uploaded_model_receives_batch_window_tensor():
    model = attach_uploaded_model_contract(_LsRecordingModel(), LS_MODEL_SPEC)
    x = torch.zeros(2, 3, 1, 8, 16)
    ls = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    run_uploaded_model(model, x, ls, context="Ls test")

    assert torch.equal(model.received_ls, ls)


def test_topography_model_receives_only_x_and_topography():
    model = attach_uploaded_model_contract(
        _TopographyRecordingModel(), TOPOGRAPHY_MODEL_SPEC
    )
    x = torch.zeros(2, 3, 1, 8, 16)
    topography = torch.ones(2, 1, 8, 16, dtype=torch.float32)

    run_uploaded_model(
        model,
        x,
        ls=torch.ones(2, 3),
        topography=topography,
        context="topography test",
    )

    assert model.received_topography is topography


def test_combined_model_argument_order_is_x_ls_topography():
    model = attach_uploaded_model_contract(
        _LsTopographyRecordingModel(), LS_TOPOGRAPHY_MODEL_SPEC
    )
    x = torch.zeros(2, 3, 1, 8, 16)
    ls = torch.ones(2, 3)
    topography = torch.ones(2, 1, 8, 16)

    run_uploaded_model(model, x, ls=ls, topography=topography)

    assert model.arguments[0] is x
    assert model.arguments[1] is ls
    assert model.arguments[2] is topography


def test_undeclared_topography_is_not_passed_to_legacy_or_ls_models():
    x = torch.zeros(2, 3, 1, 8, 16)
    topography = torch.ones(2, 1, 8, 16)
    legacy = attach_uploaded_model_contract(_LegacyRecordingModel(), {"name": "Legacy"})
    ls_model = attach_uploaded_model_contract(_LsRecordingModel(), LS_MODEL_SPEC)

    run_uploaded_model(legacy, x, topography=topography)
    run_uploaded_model(ls_model, x, ls=torch.ones(2, 3), topography=topography)

    assert legacy.argument_count == 1
    assert ls_model.received_ls.shape == (2, 3)


def test_static_topography_expands_without_copy_for_changing_batch_size():
    grid = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    x = torch.zeros(3, 5, 1, 4, 8)

    batch = expand_topography_batch(x, grid, "expand test")

    assert batch.shape == (3, 1, 4, 8)
    assert batch.dtype == torch.float32
    assert batch.stride(0) == 0
    assert batch.untyped_storage().data_ptr() == grid.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    ("auxiliary_inputs", "message"),
    [
        ({"season": {}}, "only ls and topography"),
        ({"ls": None}, "must be a dict.*None"),
        ({"ls": {**LS_MODEL_SPEC["auxiliary_inputs"]["ls"], "required": False}}, "required.*got False"),
        ({"ls": {**LS_MODEL_SPEC["auxiliary_inputs"]["ls"], "required": 1}}, "required.*got 1.*int"),
        ({"ls": {**LS_MODEL_SPEC["auxiliary_inputs"]["ls"], "shape": ["batch", "horizon"]}}, "shape.*horizon"),
        ({"ls": {**LS_MODEL_SPEC["auxiliary_inputs"]["ls"], "dtype": "float64"}}, "dtype.*float64"),
        ({"ls": {**LS_MODEL_SPEC["auxiliary_inputs"]["ls"], "unit": "radian"}}, "unit.*radian"),
        ({"ls": {**LS_MODEL_SPEC["auxiliary_inputs"]["ls"], "extra": True}}, "unexpected fields.*extra"),
        ({"topography": None}, "topography must be a dict.*None"),
        ({"topography": {**TOPOGRAPHY_MODEL_SPEC["auxiliary_inputs"]["topography"], "required": False}}, "topography.required.*False"),
        ({"topography": {**TOPOGRAPHY_MODEL_SPEC["auxiliary_inputs"]["topography"], "shape": ["batch", "height", "width"]}}, "topography.shape.*height"),
        ({"topography": {**TOPOGRAPHY_MODEL_SPEC["auxiliary_inputs"]["topography"], "dtype": "float64"}}, "topography.dtype.*float64"),
        ({"topography": {**TOPOGRAPHY_MODEL_SPEC["auxiliary_inputs"]["topography"], "unit": "kilometer"}}, "topography.unit.*kilometer"),
    ],
)
def test_auxiliary_input_metadata_is_strict(auxiliary_inputs, message):
    with pytest.raises(ValueError, match=message):
        normalize_auxiliary_inputs({"name": "Invalid", "auxiliary_inputs": auxiliary_inputs})


def test_missing_auxiliary_metadata_field_reports_expected_and_actual():
    metadata = dict(LS_MODEL_SPEC["auxiliary_inputs"]["ls"])
    metadata.pop("unit")

    with pytest.raises(ValueError) as exc_info:
        normalize_auxiliary_inputs(
            {"name": "Invalid", "auxiliary_inputs": {"ls": metadata}}
        )

    message = str(exc_info.value)
    assert "MODEL_SPEC.auxiliary_inputs.ls.unit" in message
    assert "expected 'degree'" in message
    assert "got <missing>" in message


def test_unexpected_auxiliary_metadata_field_reports_expected_and_actual():
    metadata = {
        **LS_MODEL_SPEC["auxiliary_inputs"]["ls"],
        "extra": True,
    }

    with pytest.raises(ValueError) as exc_info:
        normalize_auxiliary_inputs(
            {"name": "Invalid", "auxiliary_inputs": {"ls": metadata}}
        )

    message = str(exc_info.value)
    assert "MODEL_SPEC.auxiliary_inputs.ls.extra" in message
    assert "expected <not allowed>" in message
    assert "got True (bool)" in message


@pytest.mark.parametrize(
    ("ls", "message"),
    [
        (None, "requires Ls data"),
        ([[1.0, 2.0, 3.0]], "torch.Tensor"),
        (torch.ones(2, 3, dtype=torch.int64), "floating-point"),
        (torch.ones(2, 3, 1), "2D"),
        (torch.ones(2, 2), "shape mismatch"),
        (torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]]), "finite"),
        (torch.tensor([[1.0, float("inf"), 3.0], [4.0, 5.0, 6.0]]), "finite"),
    ],
)
def test_required_ls_tensor_is_validated(ls, message):
    model = attach_uploaded_model_contract(_LsRecordingModel(), LS_MODEL_SPEC)
    x = torch.zeros(2, 3, 1, 8, 16)

    with pytest.raises(ValueError, match=message):
        run_uploaded_model(model, x, ls, context="validation test")


@pytest.mark.parametrize(
    ("topography", "message"),
    [
        (None, "requires topography data"),
        ([[[1.0]]], "torch.Tensor"),
        (torch.ones(2, 1, 8, 16, dtype=torch.int64), "torch.float32"),
        (torch.ones(2, 8, 16), "4D"),
        (torch.ones(2, 2, 8, 16), "shape mismatch"),
        (torch.ones(1, 1, 8, 16), "shape mismatch"),
        (torch.ones(2, 1, 7, 16), "shape mismatch"),
        (torch.full((2, 1, 8, 16), float("nan")), "finite"),
        (torch.full((2, 1, 8, 16), float("inf")), "finite"),
    ],
)
def test_required_topography_tensor_is_validated(topography, message):
    model = attach_uploaded_model_contract(
        _TopographyRecordingModel(), TOPOGRAPHY_MODEL_SPEC
    )
    x = torch.zeros(2, 3, 1, 8, 16)

    with pytest.raises(ValueError, match=message):
        run_uploaded_model(
            model,
            x,
            topography=topography,
            context="validation test",
        )


def test_uploaded_runner_uses_central_mcd_directory(tmp_path, monkeypatch):
    expected = tmp_path / "mcd"
    captured = {}

    class PreparationObserved(Exception):
        pass

    def capture_prepare_tensors(openmars_dir, mcd_dir, *args, **kwargs):
        captured["mcd_dir"] = mcd_dir
        raise PreparationObserved

    monkeypatch.setattr(runner_module, "MCD_DIR", expected)
    monkeypatch.setattr(runner_module, "prepare_tensors", capture_prepare_tensors)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner_module.__file__),
            "--output_path",
            str(tmp_path / "output.pth"),
            "--uploaded_model_path",
            str(tmp_path / "uploaded.py"),
        ],
    )

    with pytest.raises(PreparationObserved):
        runner_module.main()

    assert captured["mcd_dir"] == expected


def _write_overview_file(
    path: Path,
    offset: float,
    *,
    include_ls: bool = True,
    ls_count: int = 6,
    latitude: object = (-45.0, 45.0),
    longitude: object = (0.0, 120.0, 240.0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 6)
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 3)

        if include_ls:
            ls_dimension = "time"
            if ls_count != 6:
                ds.createDimension("ls_time", ls_count)
                ls_dimension = "ls_time"
            ds.createVariable("Ls", "f4", (ls_dimension,))[:] = np.linspace(
                offset,
                offset + float(ls_count - 1),
                ls_count,
                dtype=np.float32,
            ) % 360.0
        if latitude is not None:
            ds.createVariable("lat", "f4", ("lat",))[:] = np.asarray(latitude, dtype=np.float32)
        if longitude is not None:
            ds.createVariable("lon", "f4", ("lon",))[:] = np.asarray(longitude, dtype=np.float32)

        base = np.arange(36, dtype=np.float32).reshape(6, 2, 3) + offset
        for var_name, delta in {
            "o3col": 0.0,
            "U_Wind": 10.0,
            "V_Wind": 20.0,
            "Dust_Optical_Depth": 30.0,
            "Solar_Flux_DN": 40.0,
            "Temperature": 50.0,
        }.items():
            ds.createVariable(var_name, "f4", ("time", "lat", "lon"))[:] = base + delta


def _write_raw_3h_mcd_file(
    path: Path,
    offset: float,
    *,
    include_longitude: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 10)
        ds.createDimension("lat", 37)
        ds.createDimension("lon", 72)

        ds.createVariable("LS", "f4", ("time",))[:] = np.linspace(
            offset,
            offset + 1.0,
            10,
            dtype=np.float32,
        ) % 360.0
        ds.createVariable("lat", "f4", ("lat",))[:] = np.linspace(90.0, -90.0, 37, dtype=np.float32)
        if include_longitude:
            ds.createVariable("lon", "f4", ("lon",))[:] = np.linspace(
                -180.0,
                175.0,
                72,
                dtype=np.float32,
            )

        base = np.arange(10 * 37 * 72, dtype=np.float32).reshape(10, 37, 72) + offset
        for var_name, delta in {
            "O3COL": 0.0,
            "U": 10.0,
            "T": 20.0,
            "V": 30.0,
            "FSDS": 40.0,
            "DUST": 50.0,
        }.items():
            ds.createVariable(var_name, "f4", ("time", "lat", "lon"))[:] = base + delta
        ds["O3COL"].units = "um-atm"


def test_parse_json_arg_accepts_dict_json_string_and_empty_values():
    assert parse_json_arg({"hidden_dim": 16}) == {"hidden_dim": 16}
    assert parse_json_arg('{"hidden_dim":16}') == {"hidden_dim": 16}
    assert parse_json_arg("") == {}
    assert parse_json_arg(None) == {}


def test_build_uploaded_model_config_merges_core_custom_and_schema_defaults():
    config = build_uploaded_model_config(
        in_channels=3,
        window=4,
        horizon=5,
        height=8,
        width=16,
        selected_channels=["U", "T"],
        custom_model_params={"hidden_dim": 16},
        param_schema={
            "hidden_dim": {"type": "int", "default": 8},
            "dropout": {"type": "float", "default": 0.25},
        },
    )

    assert config == {
        "in_channels": 3,
        "window": 4,
        "horizon": 5,
        "height": 8,
        "width": 16,
        "selected_channels": ["U", "T"],
        "hidden_dim": 16,
        "dropout": 0.25,
    }


def test_load_uploaded_model_imports_build_model_and_returns_module():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "runner_tiny.py"
        model_path.write_text(MODEL_SOURCE, encoding="utf-8")
        config = build_uploaded_model_config(
            in_channels=3,
            window=3,
            horizon=3,
            height=8,
            width=16,
            selected_channels=["U", "V"],
            custom_model_params={},
            param_schema={"hidden_dim": {"type": "int", "default": 8}},
        )

        model = load_uploaded_model(model_path, config)
        output = model(torch.zeros(2, 3, 1, 8, 16))

    assert isinstance(model, torch.nn.Module)
    assert list(output.shape) == [2, 3, 1, 8, 16]


def test_load_uploaded_model_reads_python_source_from_non_py_storage_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "runner_tiny.py.user"
        model_path.write_text(MODEL_SOURCE, encoding="utf-8")
        config = build_uploaded_model_config(
            in_channels=3,
            window=3,
            horizon=3,
            height=8,
            width=16,
            selected_channels=["U", "V"],
            custom_model_params={},
            param_schema={"hidden_dim": {"type": "int", "default": 8}},
        )

        model = load_uploaded_model(model_path, config)
        output = model(torch.zeros(2, 3, 1, 8, 16))

    assert isinstance(model, torch.nn.Module)
    assert list(output.shape) == [2, 3, 1, 8, 16]


def test_load_uploaded_model_attaches_declared_ls_contract():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.py"
        model_path.write_text(LS_RUNNER_MODEL_SOURCE, encoding="utf-8")
        config = build_uploaded_model_config(
            in_channels=1,
            window=3,
            horizon=2,
            height=8,
            width=16,
            selected_channels=[],
            custom_model_params={},
            param_schema={},
        )
        model = load_uploaded_model(model_path, config)
        x = torch.zeros(2, 3, 1, 8, 16)
        ls = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

        prediction = run_uploaded_model(model, x, ls, context="loaded model test")

        assert list(prediction.shape) == [2, 2, 1, 8, 16]
        assert torch.equal(model.received_ls, ls)


def test_split_train_test_carries_ls_only_when_available():
    x = torch.zeros(10, 3, 1, 2, 2)
    y = torch.zeros(10, 2, 1, 2, 2)
    ls = torch.arange(30, dtype=torch.float32).reshape(10, 3)

    train_with_ls, test_with_ls = runner_module._split_train_test(x, y, ls)
    train_legacy, test_legacy = runner_module._split_train_test(x, y)

    assert len(train_with_ls.tensors) == 3
    assert len(test_with_ls.tensors) == 3
    assert torch.equal(train_with_ls.tensors[1], ls[:8])
    assert len(train_legacy.tensors) == 2
    assert len(test_legacy.tensors) == 2


def test_forward_uploaded_batch_passes_ls_and_returns_device_target():
    model = attach_uploaded_model_contract(_LsRecordingModel(), LS_MODEL_SPEC)
    x = torch.zeros(2, 3, 1, 2, 2)
    ls = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.zeros(2, 1, 1, 2, 2)

    prediction, target = runner_module._forward_uploaded_batch(
        model,
        (x, ls, y),
        torch.device("cpu"),
        "training test",
    )

    assert torch.equal(model.received_ls, ls)
    assert torch.equal(prediction, target)


def test_forward_uploaded_batch_keeps_legacy_model_single_input():
    model = attach_uploaded_model_contract(_LegacyRecordingModel(), {"name": "Legacy"})
    x = torch.zeros(2, 3, 1, 2, 2)
    y = torch.zeros(2, 1, 1, 2, 2)

    prediction, target = runner_module._forward_uploaded_batch(
        model,
        (x, y),
        torch.device("cpu"),
        "legacy training test",
    )

    assert model.argument_count == 1
    assert torch.equal(prediction, target)


def test_forward_uploaded_batch_expands_static_topography():
    model = attach_uploaded_model_contract(
        _TopographyRecordingModel(), TOPOGRAPHY_MODEL_SPEC
    )
    x = torch.zeros(2, 3, 1, 2, 2)
    y = torch.zeros(2, 1, 1, 2, 2)
    static_topography = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

    prediction, target = runner_module._forward_uploaded_batch(
        model,
        (x, y),
        torch.device("cpu"),
        "topography training test",
        topography_grid=static_topography,
    )

    assert model.received_topography.shape == (2, 1, 2, 2)
    assert torch.equal(model.received_topography[0], static_topography)
    assert torch.equal(model.received_topography[1], static_topography)
    assert model.received_topography.stride(0) == 0
    assert torch.equal(prediction, target)


def test_runner_main_reuses_topography_in_training_validation_and_metrics(
    monkeypatch,
):
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_topo_{uuid.uuid4().hex}"
    output_path = workspace_tmp / "trained.pth"
    model = attach_uploaded_model_contract(
        _TopographyRecordingModel(), TOPOGRAPHY_MODEL_SPEC
    )
    model.topography_calls = []
    original_forward = model.forward

    def recording_forward(x, topography):
        model.topography_calls.append(topography.detach().clone())
        return original_forward(x, topography).repeat(1, 2, 1, 1, 1)

    model.forward = recording_forward
    x = torch.zeros(5, 3, 1, 2, 3)
    y = torch.zeros(5, 2, 1, 2, 3)
    latitude = np.array([-45.0, 45.0], dtype=np.float32)
    longitude = np.array([0.0, 120.0, 240.0], dtype=np.float32)
    static_topography = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    prepare_topography_calls = []

    def fake_prepare_tensors(*args, **kwargs):
        assert kwargs["return_ls"] is True
        assert kwargs["return_coordinates"] is True
        return x, y, None, 0.0, 1.0, 2, 3, latitude, longitude

    def fake_prepare_topography(target_latitude, target_longitude, **kwargs):
        prepare_topography_calls.append((target_latitude, target_longitude))
        return static_topography

    monkeypatch.setattr(runner_module, "prepare_tensors", fake_prepare_tensors)
    monkeypatch.setattr(runner_module, "prepare_topography_grid", fake_prepare_topography, raising=False)
    monkeypatch.setattr(runner_module, "load_uploaded_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner_module.__file__),
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--window",
            "3",
            "--horizon",
            "2",
            "--output_path",
            str(output_path),
            "--uploaded_model_path",
            str(workspace_tmp / "unused.py"),
        ],
    )

    try:
        runner_module.main()
    finally:
        if output_path.exists():
            output_path.unlink()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    assert len(prepare_topography_calls) == 1
    assert len(model.topography_calls) == 4
    assert [call.shape[0] for call in model.topography_calls] == [2, 2, 1, 1]
    assert all(call.shape[1:] == (1, 2, 3) for call in model.topography_calls)


def test_runner_main_passes_ls_through_training_validation_and_metrics(monkeypatch):
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_main_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    first_file = overview_dir / "MCD_MY24_overview.nc"
    second_file = overview_dir / "MCD_MY25_overview.nc"
    model_path = workspace_tmp / "ls_model.source"
    output_path = workspace_tmp / "trained.pth"
    _write_overview_file(first_file, 0.0, latitude=None, longitude=None)
    _write_overview_file(second_file, 10.0, latitude=None, longitude=None)
    model_path.write_text(LS_RUNNER_MODEL_SOURCE, encoding="utf-8")

    try:
        monkeypatch.setenv("MCD_RAW_3H_DIR", str(overview_dir))
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(runner_module.__file__),
                "--epochs",
                "1",
                "--batch_size",
                "2",
                "--window",
                "3",
                "--horizon",
                "2",
                "--training_dataset",
                "mcd_overview",
                "--output_path",
                str(output_path),
                "--uploaded_model_path",
                str(model_path),
            ],
        )

        runner_module.main()

        assert output_path.is_file()
    finally:
        for file_path in (first_file, second_file, model_path, output_path):
            if file_path.exists():
                file_path.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()


def test_runner_main_trains_legacy_model_when_dataset_has_no_ls(monkeypatch):
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_legacy_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    data_file = overview_dir / "MCD_MY24_overview.nc"
    model_path = workspace_tmp / "legacy_model.source"
    output_path = workspace_tmp / "trained.pth"
    _write_overview_file(
        data_file,
        0.0,
        include_ls=False,
        latitude=None,
        longitude=None,
    )
    model_path.write_text(MODEL_SOURCE, encoding="utf-8")

    try:
        monkeypatch.setenv("MCD_RAW_3H_DIR", str(overview_dir))
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(runner_module.__file__),
                "--epochs",
                "1",
                "--batch_size",
                "2",
                "--window",
                "2",
                "--horizon",
                "2",
                "--training_dataset",
                "mcd_overview",
                "--output_path",
                str(output_path),
                "--uploaded_model_path",
                str(model_path),
            ],
        )

        runner_module.main()

        assert output_path.is_file()
    finally:
        for file_path in (data_file, model_path, output_path):
            if file_path.exists():
                file_path.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()


def test_assert_prediction_shape_accepts_matching_shapes_and_rejects_mismatches():
    prediction = torch.zeros(2, 3, 1, 8, 16)
    target = torch.ones(2, 3, 1, 8, 16)
    assert_prediction_shape(prediction, target, "unit-test")

    try:
        assert_prediction_shape(torch.zeros(2, 1, 8, 16), target, "bad-batch")
    except ValueError as exc:
        message = str(exc)
        assert "bad-batch" in message
        assert "expected" in message
        assert "actual" in message
        assert "shape" in message
    else:
        raise AssertionError("mismatched prediction shape should raise ValueError")


def test_bad_shape_uploaded_model_is_caught_by_prediction_shape_guard():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "bad_shape_tiny.py"
        model_path.write_text(BAD_SHAPE_MODEL_SOURCE, encoding="utf-8")
        config = build_uploaded_model_config(
            in_channels=1,
            window=3,
            horizon=3,
            height=8,
            width=16,
            selected_channels=[],
            custom_model_params={},
            param_schema={},
        )

        model = load_uploaded_model(model_path, config)
        prediction = model(torch.zeros(2, 3, 1, 8, 16))
        target = torch.zeros(2, 3, 1, 8, 16)

    try:
        assert_prediction_shape(prediction, target, "bad-shape-forward")
    except ValueError as exc:
        message = str(exc)
        assert "bad-shape-forward" in message
        assert "expected" in message
        assert "actual" in message
    else:
        raise AssertionError("bad-shape uploaded model output should raise ValueError")


def test_prepare_tensors_builds_uploaded_runner_dataset_from_mcd_overview():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    first_file = overview_dir / "MCD_MY24_overview.nc"
    second_file = overview_dir / "MCD_MY25_overview.nc"
    _write_overview_file(first_file, 0.0)
    _write_overview_file(second_file, 10.0)

    try:
        x_torch, y_torch, _y_mean, _y_std, height, width = prepare_tensors(
            workspace_tmp / "openmars",
            workspace_tmp / "MCD",
            ["U", "T"],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=overview_dir,
        )
    finally:
        if first_file.exists():
            first_file.unlink()
        if second_file.exists():
            second_file.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    assert list(x_torch.shape) == [6, 2, 3, 2, 3]
    assert list(y_torch.shape) == [6, 2, 1, 2, 3]
    assert height == 2
    assert width == 3


def test_prepare_tensors_returns_actual_target_coordinates():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_grid_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    first_file = overview_dir / "MCD_MY24_overview.nc"
    second_file = overview_dir / "MCD_MY25_overview.nc"
    _write_overview_file(first_file, 0.0)
    _write_overview_file(second_file, 10.0)

    try:
        result = prepare_tensors(
            workspace_tmp / "openmars",
            workspace_tmp / "MCD",
            [],
            window=3,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=overview_dir,
            return_ls=True,
            return_coordinates=True,
        )
    finally:
        for file_path in (first_file, second_file):
            if file_path.exists():
                file_path.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    x, y, ls, _mean, _std, height, width, latitude, longitude = result
    assert x.shape[1] == 3
    assert y.shape[1] == 2
    assert ls.shape[1] == 3
    assert (height, width) == (2, 3)
    assert np.array_equal(latitude, np.array([-45.0, 45.0], dtype=np.float32))
    assert np.array_equal(longitude, np.array([0.0, 120.0, 240.0], dtype=np.float32))


def test_prepare_tensors_rejects_mismatched_dataset_grids():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_bad_grid_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    first_file = overview_dir / "MCD_MY24_overview.nc"
    second_file = overview_dir / "MCD_MY25_overview.nc"
    _write_overview_file(first_file, 0.0)
    _write_overview_file(second_file, 10.0, latitude=(45.0, -45.0))

    try:
        with pytest.raises(ValueError, match="Spatial grid mismatch"):
            prepare_tensors(
                workspace_tmp / "openmars",
                workspace_tmp / "MCD",
                [],
                window=3,
                horizon=2,
                training_dataset="mcd_overview",
                mcd_overview_dir=overview_dir,
                return_coordinates=True,
            )
    finally:
        for file_path in (first_file, second_file):
            if file_path.exists():
                file_path.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()


def test_prepare_tensors_rejects_missing_dataset_coordinates():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_no_grid_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    data_file = overview_dir / "MCD_MY24_overview.nc"
    _write_overview_file(data_file, 0.0, longitude=None)

    try:
        with pytest.raises(ValueError) as exc_info:
            prepare_tensors(
                workspace_tmp / "openmars",
                workspace_tmp / "MCD",
                [],
                window=2,
                horizon=2,
                training_dataset="mcd_overview",
                mcd_overview_dir=overview_dir,
                return_coordinates=True,
            )
    finally:
        if data_file.exists():
            data_file.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    assert data_file.name in str(exc_info.value)
    assert "longitude" in str(exc_info.value).lower()


def test_prepare_tensors_can_return_missing_coordinates_when_they_are_optional():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_optional_grid_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    data_file = overview_dir / "MCD_MY24_overview.nc"
    _write_overview_file(data_file, 0.0, latitude=None, longitude=None)

    try:
        result = prepare_tensors(
            workspace_tmp / "openmars",
            workspace_tmp / "MCD",
            [],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=overview_dir,
            return_ls=True,
            return_coordinates=True,
            require_coordinates=False,
        )
    finally:
        if data_file.exists():
            data_file.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    x, y, ls, _mean, _std, height, width, latitude, longitude = result
    assert x.shape == (3, 2, 1, 2, 3)
    assert y.shape == (3, 2, 1, 2, 3)
    assert ls.shape == (3, 2)
    assert (height, width) == (2, 3)
    assert latitude is None
    assert longitude is None


def test_prepare_tensors_aligns_ls_to_history_when_window_differs_from_horizon():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_ls_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    first_file = overview_dir / "MCD_MY24_overview.nc"
    second_file = overview_dir / "MCD_MY25_overview.nc"
    _write_overview_file(first_file, 0.0)
    _write_overview_file(second_file, 10.0)

    try:
        x_torch, y_torch, ls_torch, _y_mean, _y_std, _height, _width = prepare_tensors(
            workspace_tmp / "openmars",
            workspace_tmp / "MCD",
            [],
            window=3,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=overview_dir,
            return_ls=True,
        )
    finally:
        if first_file.exists():
            first_file.unlink()
        if second_file.exists():
            second_file.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    assert list(x_torch.shape) == [4, 3, 1, 2, 3]
    assert list(y_torch.shape) == [4, 2, 1, 2, 3]
    assert list(ls_torch.shape) == [4, 3]
    assert torch.equal(ls_torch[0], torch.tensor([0.0, 1.0, 2.0]))


def test_prepare_tensors_does_not_fabricate_missing_ls_for_legacy_data():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_no_ls_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    data_file = overview_dir / "MCD_MY24_overview.nc"
    _write_overview_file(data_file, 0.0, include_ls=False)

    try:
        x_torch, y_torch, ls_torch, _y_mean, _y_std, _height, _width = prepare_tensors(
            workspace_tmp / "openmars",
            workspace_tmp / "MCD",
            [],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=overview_dir,
            return_ls=True,
        )
    finally:
        if data_file.exists():
            data_file.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    assert list(x_torch.shape) == [3, 2, 1, 2, 3]
    assert list(y_torch.shape) == [3, 2, 1, 2, 3]
    assert ls_torch is None


def test_prepare_tensors_rejects_partial_file_ls_as_unaligned_timeline():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_short_ls_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    first_file = overview_dir / "MCD_MY24_overview.nc"
    second_file = overview_dir / "MCD_MY25_overview.nc"
    _write_overview_file(first_file, 0.0, ls_count=5)
    _write_overview_file(second_file, 10.0)

    try:
        x_torch, y_torch, ls_torch, _y_mean, _y_std, _height, _width = prepare_tensors(
            workspace_tmp / "openmars",
            workspace_tmp / "MCD",
            [],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=overview_dir,
            return_ls=True,
        )
    finally:
        if first_file.exists():
            first_file.unlink()
        if second_file.exists():
            second_file.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    assert list(x_torch.shape) == [6, 2, 1, 2, 3]
    assert list(y_torch.shape) == [6, 2, 1, 2, 3]
    assert ls_torch is None


def test_required_ls_rejects_file_length_mismatch_with_file_context():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_required_short_ls_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    data_file = overview_dir / "MCD_MY24_overview.nc"
    _write_overview_file(data_file, 0.0, ls_count=5)

    try:
        with pytest.raises(ValueError) as exc_info:
            runner_module._load_mcd_overview(overview_dir, [], require_ls=True)
    finally:
        if data_file.exists():
            data_file.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    message = str(exc_info.value)
    assert data_file.name in message
    assert "expected 6" in message
    assert "got 5" in message


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_prepare_tensors_rejects_non_finite_ls_during_data_preparation(bad_value):
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_non_finite_ls_{uuid.uuid4().hex}"
    overview_dir = workspace_tmp / "mcd_overview"
    data_file = overview_dir / "MCD_MY24_overview.nc"
    _write_overview_file(data_file, 0.0)
    with netCDF4.Dataset(str(data_file), "r+") as dataset:
        dataset.variables["Ls"][1] = bad_value

    try:
        with pytest.raises(MarsDataError, match="NaN or Inf|finite"):
            prepare_tensors(
                workspace_tmp / "openmars",
                workspace_tmp / "MCD",
                [],
                window=2,
                horizon=2,
                training_dataset="mcd_overview",
                mcd_overview_dir=overview_dir,
                return_ls=True,
            )
    finally:
        if data_file.exists():
            data_file.unlink()
        if overview_dir.exists():
            overview_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()


def test_prepare_tensors_rejects_masked_mars_values(tmp_path):
    overview_dir = tmp_path / "mcd_overview"
    overview_dir.mkdir()
    path = overview_dir / "MCD_MY24_overview.nc"
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("time", 6)
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        dataset.createVariable("Ls", "f4", ("time",))[:] = np.arange(6, dtype=np.float32)
        ozone = dataset.createVariable("o3col", "f4", ("time", "lat", "lon"), fill_value=-9999.0)
        ozone[:] = 1.0
        ozone[1, 0, 0] = ozone._FillValue
    with pytest.raises(MarsDataError, match="masked|missing"):
        prepare_tensors(
            tmp_path / "openmars",
            tmp_path / "MCD",
            [],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=overview_dir,
            return_ls=True,
        )


def test_prepare_tensors_builds_uploaded_runner_dataset_from_raw_3h_mcd():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_raw_{uuid.uuid4().hex}"
    raw_dir = workspace_tmp / "MCD_Output_global_10m_ls_lst"
    first_file = raw_dir / "MCD_MY24_global_3h_5deg_10m_ls_lst.nc"
    second_file = raw_dir / "MCD_MY25_global_3h_5deg_10m_ls_lst.nc"
    _write_raw_3h_mcd_file(first_file, 0.0)
    _write_raw_3h_mcd_file(second_file, 2.0)

    try:
        x_torch, y_torch, _y_mean, _y_std, height, width = prepare_tensors(
            workspace_tmp / "openmars",
            workspace_tmp / "MCD",
            ["U", "D", "T"],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=raw_dir,
        )
    finally:
        if first_file.exists():
            first_file.unlink()
        if second_file.exists():
            second_file.unlink()
        if raw_dir.exists():
            raw_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    assert list(x_torch.shape) == [14, 2, 4, 36, 72]
    assert list(y_torch.shape) == [14, 2, 1, 36, 72]
    assert height == 36
    assert width == 72


def test_prepare_tensors_allows_raw_mcd_without_longitude_when_coordinates_are_optional():
    workspace_tmp = BACKEND_DIR / ".test_tmp" / f"uploaded_runner_raw_no_lon_{uuid.uuid4().hex}"
    raw_dir = workspace_tmp / "MCD_Output_global_10m_ls_lst"
    data_file = raw_dir / "MCD_MY24_global_3h_5deg_10m_ls_lst.nc"
    _write_raw_3h_mcd_file(data_file, 0.0, include_longitude=False)

    try:
        result = prepare_tensors(
            workspace_tmp / "openmars",
            workspace_tmp / "MCD",
            [],
            window=2,
            horizon=2,
            training_dataset="mcd_overview",
            mcd_overview_dir=raw_dir,
            return_coordinates=True,
            require_coordinates=False,
        )
    finally:
        if data_file.exists():
            data_file.unlink()
        if raw_dir.exists():
            raw_dir.rmdir()
        if workspace_tmp.exists():
            workspace_tmp.rmdir()

    x, y, _mean, _std, height, width, latitude, longitude = result
    assert x.shape == (7, 2, 1, 36, 72)
    assert y.shape == (7, 2, 1, 36, 72)
    assert (height, width) == (36, 72)
    assert latitude is None
    assert longitude is None


@pytest.mark.parametrize("ascending", [False, True])
def test_fit_raw_mcd_36_row_field_matches_declared_target_latitude(ascending):
    latitude = runner_module.RAW_MCD_TARGET_LAT.copy()
    if ascending:
        latitude = latitude[::-1].copy()
    field = np.broadcast_to(latitude[None, :, None], (1, 36, 72)).copy()

    fitted = runner_module._fit_raw_mcd_lat_grid(field, latitude)

    assert np.array_equal(fitted[0, :, 0], runner_module.RAW_MCD_TARGET_LAT)


if __name__ == "__main__":
    test_parse_json_arg_accepts_dict_json_string_and_empty_values()
    test_build_uploaded_model_config_merges_core_custom_and_schema_defaults()
    test_load_uploaded_model_imports_build_model_and_returns_module()
    test_load_uploaded_model_reads_python_source_from_non_py_storage_path()
    test_assert_prediction_shape_accepts_matching_shapes_and_rejects_mismatches()
    test_bad_shape_uploaded_model_is_caught_by_prediction_shape_guard()
    test_prepare_tensors_builds_uploaded_runner_dataset_from_mcd_overview()
    test_prepare_tensors_builds_uploaded_runner_dataset_from_raw_3h_mcd()
    print("uploaded model runner tests passed")
