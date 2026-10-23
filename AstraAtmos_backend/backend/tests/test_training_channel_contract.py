import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_identity import DatasetRequestError  # noqa: E402
from services.training_channels import (  # noqa: E402
    UNIFIED_TRAINING_SCRIPT,
    build_hyperparameter_args,
    get_channels_from_hyperparameters,
    get_task_channel_suffix,
    normalize_training_hyperparameters,
)


def test_channels_from_hyperparameters_normalizes_order_and_filters_unknowns():
    result = get_channels_from_hyperparameters({
        "selected_channels": ["T", "U", "bad", "D", "U"],
    })

    assert result == ["U", "D", "T"]


def test_hyperparameter_args_pass_selected_channels_to_unified_script():
    args = build_hyperparameter_args({
        "epochs": 12,
        "selected_channels": ["T", "U", "bad", "D", "U"],
        "training_dataset": "mcd_overview",
        "model_architecture": "predrnnv2",
        "use_sphere": True,
        "seed": 123,
        "_data_source": "default",
        "stlstm_hidden_dims": [32, 64],
    })

    assert UNIFIED_TRAINING_SCRIPT == "demo3.py"
    assert args == [
        "--epochs", "12",
        "--selected_channels", "U,D,T",
        "--training_dataset", "mcd_overview",
        "--model_architecture", "predrnnv2",
        "--use_sphere", "True",
        "--seed", "123",
        "--stlstm_hidden_dims", "[32, 64]",
    ]


def test_normalize_training_hyperparameters_sanitizes_training_dataset():
    defaulted = normalize_training_hyperparameters({})
    selected = normalize_training_hyperparameters({"training_dataset": "mcd_overview"})

    assert defaulted["training_dataset"] == "openmars_mcd"
    assert selected["training_dataset"] == "mcd_overview"


@pytest.mark.parametrize("value,code", [
    ("unknown", "unknown_dataset"),
    ("earth_merra2_daily_v1", "dataset_training_not_supported"),
    ("", "invalid_dataset_id"),
    (7, "invalid_dataset_id"),
])
def test_normalize_training_hyperparameters_rejects_unsupported_training_dataset(value, code):
    """New training configurations must not silently fall back to openmars_mcd."""
    with pytest.raises(DatasetRequestError) as exc:
        normalize_training_hyperparameters({"training_dataset": value})
    assert exc.value.code == code


def test_normalize_training_hyperparameters_clamps_invalid_numeric_values():
    result = normalize_training_hyperparameters({
        "epochs": -12,
        "batch_size": -3,
        "learning_rate": -0.2,
        "window": -4,
        "horizon": 0,
        "early_stopping_patience": -5,
        "seed": -7,
        "stlstm_hidden_dims": [-16, "", 48],
        "selected_channels": ["T", "D"],
        "model_architecture": "predrnnv2_sphere",
    })

    assert result["epochs"] == 1
    assert result["batch_size"] == 1
    assert result["learning_rate"] == 0.000001
    assert result["window"] == 1
    assert result["horizon"] == 1
    assert result["early_stopping_patience"] == 0
    assert result["seed"] == 0
    assert result["stlstm_hidden_dims"] == [1, 64, 48]
    assert result["selected_channels"] == ["D", "T"]
    assert result["model_architecture"] == "predrnnv2"
    assert result["use_sphere"] is True


def test_normalize_training_hyperparameters_sanitizes_architecture_specific_params():
    result = normalize_training_hyperparameters({
        "model_architecture": "patchtst",
        "patch_len": 0,
        "stride": "2",
        "d_model": "96",
        "n_heads": 4,
        "e_layers": 2.4,
        "d_ff": 192,
        "dropout": 2,
        "spatial_hidden_dim": -8,
    })

    assert result["patch_len"] == 1
    assert result["stride"] == 2
    assert result["d_model"] == 96
    assert result["n_heads"] == 4
    assert result["e_layers"] == 2
    assert result["d_ff"] == 192
    assert result["dropout"] == 0.9
    assert result["spatial_hidden_dim"] == 1
    assert "stlstm_hidden_dims" not in result


def test_normalize_training_hyperparameters_sanitizes_migrated_ablation_params():
    result = normalize_training_hyperparameters({
        "model_architecture": "crossformer",
        "seg_len": 0,
        "win_size": "2",
        "factor": "bad",
        "patch_size": [1, 0, "2"],
        "pooling_sizes": "1,0,3",
        "downsample_factors": ["1", "-2", "4"],
        "mst_spatial_hidden_dim": -8,
        "mst_temporal_depth": 2,
        "gamma": 2.5,
        "dropout": -0.3,
    })

    assert result["model_architecture"] == "crossformer"
    assert result["seg_len"] == 1
    assert result["win_size"] == 2
    assert result["factor"] == 2
    assert result["patch_size"] == [1, 2, 2]
    assert result["pooling_sizes"] == [1, 3]
    assert result["downsample_factors"] == [1, 4]
    assert result["mst_spatial_hidden_dim"] == 1
    assert result["mst_temporal_depth"] == 2
    assert result["gamma"] == 2.5
    assert result["dropout"] == 0


def test_hyperparameter_args_include_architecture_specific_params():
    args = build_hyperparameter_args({
        "model_architecture": "timemixer",
        "d_model": 64,
        "moving_avg": 5,
        "down_sampling_window": 2,
        "down_sampling_layers": 1,
        "dropout": 0.2,
    })

    assert args == [
        "--model_architecture", "timemixer",
        "--d_model", "64",
        "--moving_avg", "5",
        "--down_sampling_window", "2",
        "--down_sampling_layers", "1",
        "--dropout", "0.2",
    ]


def test_normalize_training_hyperparameters_preserves_transfer_settings():
    result = normalize_training_hyperparameters({
        "transfer_learning": "true",
        "transfer_source_type": "task",
        "transfer_source_task_id": "7",
        "transfer_weight_id": "abc",
        "transfer_load_mode": "partial",
        "freeze_mode": "backbone",
        "finetune_learning_rate": "0.00001",
    })

    assert result["transfer_learning"] is True
    assert result["transfer_source_type"] == "task"
    assert result["transfer_source_task_id"] == 7
    assert result["transfer_weight_id"] == "abc"
    assert result["transfer_load_mode"] == "strict"
    assert result["freeze_mode"] == "backbone"
    assert result["finetune_learning_rate"] == 0.00001


def test_hyperparameter_args_include_public_transfer_settings_and_skip_private_path():
    args = build_hyperparameter_args({
        "transfer_learning": True,
        "transfer_source_type": "upload",
        "transfer_weight_id": "weight-1",
        "transfer_load_mode": "strict",
        "freeze_mode": "none",
        "finetune_learning_rate": 0.0001,
        "_transfer_weight_path": "D:/secret/model.pth",
    })

    assert "--transfer_learning" in args
    assert "--transfer_source_type" in args
    assert "--transfer_weight_id" in args
    assert "--finetune_learning_rate" in args
    assert "_transfer_weight_path" not in args
    assert "D:/secret/model.pth" not in args


def test_normalize_training_hyperparameters_defaults_unknown_freeze_mode_to_none():
    result = normalize_training_hyperparameters({
        "transfer_learning": True,
        "freeze_mode": "everything",
    })

    assert result["freeze_mode"] == "none"


def test_normalize_training_hyperparameters_rejects_unknown_architecture():
    try:
        normalize_training_hyperparameters({"model_architecture": "missing_model"})
    except ValueError as exc:
        assert "Unsupported model architecture" in str(exc)
    else:
        raise AssertionError("Unknown architecture should be rejected")


def test_task_channel_suffix_prefers_selected_channels_over_legacy_script_suffix():
    task = type("Task", (), {
        "model_script": "demo3-DT.py",
        "hyperparameters": '{"selected_channels": ["V", "S"]}',
    })()

    assert get_task_channel_suffix(task) == "VS"


def test_task_channel_suffix_falls_back_to_legacy_script_suffix():
    task = type("Task", (), {
        "model_script": "demo3-DT.py",
        "hyperparameters": "{}",
    })()

    assert get_task_channel_suffix(task) == "DT"


if __name__ == "__main__":
    test_channels_from_hyperparameters_normalizes_order_and_filters_unknowns()
    test_hyperparameter_args_pass_selected_channels_to_unified_script()
    test_normalize_training_hyperparameters_clamps_invalid_numeric_values()
    test_normalize_training_hyperparameters_sanitizes_architecture_specific_params()
    test_normalize_training_hyperparameters_sanitizes_migrated_ablation_params()
    test_hyperparameter_args_include_architecture_specific_params()
    test_normalize_training_hyperparameters_preserves_transfer_settings()
    test_hyperparameter_args_include_public_transfer_settings_and_skip_private_path()
    test_normalize_training_hyperparameters_defaults_unknown_freeze_mode_to_none()
    test_normalize_training_hyperparameters_rejects_unknown_architecture()
    test_task_channel_suffix_prefers_selected_channels_over_legacy_script_suffix()
    test_task_channel_suffix_falls_back_to_legacy_script_suffix()
    print("backend training channel contract tests passed")
