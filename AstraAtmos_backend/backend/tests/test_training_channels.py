import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.training_channels import (  # noqa: E402
    UNIFIED_TRAINING_SCRIPT,
    extract_architecture_params,
    get_channels_from_hyperparameters,
    get_task_channel_suffix,
    normalize_training_hyperparameters,
)


def test_unified_script_contract_uses_single_entrypoint():
    assert UNIFIED_TRAINING_SCRIPT == "demo3.py"


def test_available_scripts_exposes_only_unified_script():
    from services.training_service import TrainingService

    assert TrainingService().get_available_scripts() == [UNIFIED_TRAINING_SCRIPT]


def test_channels_from_hyperparameters_normalizes_order_and_filters_unknowns():
    result = get_channels_from_hyperparameters({
        "selected_channels": ["T", "U", "bad", "D", "U"],
    })

    assert result == ["U", "D", "T"]


def test_task_channel_suffix_prefers_selected_channels_over_legacy_script_suffix():
    task = type("Task", (), {
        "model_script": "demo3-DT.py",
        "hyperparameters": json.dumps({"selected_channels": ["V", "S"]}),
    })()

    assert get_task_channel_suffix(task) == "VS"


def test_task_channel_suffix_falls_back_to_legacy_script_suffix():
    task = type("Task", (), {
        "model_script": "demo3-DT.py",
        "hyperparameters": "{}",
    })()

    assert get_task_channel_suffix(task) == "DT"


def test_architecture_param_sanitizing_keeps_gate_probabilities_open_interval():
    normalized = normalize_training_hyperparameters({
        "model_architecture": "convlstm_phase_gated_mst",
        "initial_history_weight": -1,
        "initial_translation_weight": 0,
    })

    assert 0.0 < normalized["initial_history_weight"] < 1.0
    assert 0.0 < normalized["initial_translation_weight"] < 1.0


def test_architecture_param_sanitizing_normalizes_earthformer_three_value_lists():
    normalized = normalize_training_hyperparameters({
        "model_architecture": "earthformer",
        "patch_size": [2],
        "cuboid_size": "1,0,3,4",
    })
    architecture_params = extract_architecture_params(normalized)

    assert architecture_params["patch_size"] == [1, 2, 2]
    assert architecture_params["cuboid_size"] == [1, 3, 4]


if __name__ == "__main__":
    test_unified_script_contract_uses_single_entrypoint()
    test_available_scripts_exposes_only_unified_script()
    test_channels_from_hyperparameters_normalizes_order_and_filters_unknowns()
    test_task_channel_suffix_prefers_selected_channels_over_legacy_script_suffix()
    test_task_channel_suffix_falls_back_to_legacy_script_suffix()
    test_architecture_param_sanitizing_keeps_gate_probabilities_open_interval()
    test_architecture_param_sanitizing_normalizes_earthformer_three_value_lists()
    print("backend unified training channel tests passed")
