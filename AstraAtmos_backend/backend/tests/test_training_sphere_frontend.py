import importlib.util
import sys
import types
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "models" / "training_scripts" / "demo3.py"


def load_demo3_module():
    scipy = types.ModuleType("scipy")
    scipy_interpolate = types.ModuleType("scipy.interpolate")
    scipy_interpolate.interp1d = lambda *args, **kwargs: None
    scipy.interpolate = scipy_interpolate
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.interpolate", scipy_interpolate)

    sklearn = types.ModuleType("sklearn")
    sklearn_metrics = types.ModuleType("sklearn.metrics")
    sklearn_metrics.mean_squared_error = lambda *args, **kwargs: 0.0
    sklearn_metrics.r2_score = lambda *args, **kwargs: 0.0
    sklearn_preprocessing = types.ModuleType("sklearn.preprocessing")
    sklearn_preprocessing.StandardScaler = object
    sklearn.metrics = sklearn_metrics
    sklearn.preprocessing = sklearn_preprocessing
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.metrics", sklearn_metrics)
    sys.modules.setdefault("sklearn.preprocessing", sklearn_preprocessing)

    spec = importlib.util.spec_from_file_location("demo3_training_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sphere_frontend_supports_dust_only_channel_combo():
    demo3 = load_demo3_module()
    frontend = demo3.SpherePhaseWarpFrontEnd(extra_channel_count=1, height=4, width=5)

    x = torch.ones(2, 3, 2, 4, 5)
    ls = torch.tensor([
        [0.0, 90.0, 180.0],
        [45.0, 135.0, 225.0],
    ])

    out = frontend(x, ls)

    assert out.shape == (2, 3, 3, 4, 5)


def test_sphere_input_dim_is_o3_plus_two_features_per_selected_channel():
    demo3 = load_demo3_module()

    assert demo3.get_model_input_dim("predrnnv2", ["U", "D", "T"], use_sphere=False) == 4
    assert demo3.get_model_input_dim("predrnnv2", ["U", "D", "T"], use_sphere=True) == 7
    assert demo3.get_model_input_dim("predrnnv2", [], use_sphere=True) == 1


if __name__ == "__main__":
    test_sphere_frontend_supports_dust_only_channel_combo()
    test_sphere_input_dim_is_o3_plus_two_features_per_selected_channel()
    print("training SPHERE frontend tests passed")
