import sys
import types
from pathlib import Path

import torch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def install_optional_dependency_stubs():
    netcdf4 = types.ModuleType("netCDF4")
    netcdf4.Dataset = object
    sys.modules.setdefault("netCDF4", netcdf4)

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


install_optional_dependency_stubs()

from training_backbones.model_zoo import (  # noqa: E402
    FACTORIES,
    build_forecaster,
    list_model_specs,
    normalize_model_architecture,
    normalize_use_sphere,
)


def test_model_registry_exposes_migrated_raw_backbones_without_sphere_suffix():
    model_ids = [spec["id"] for spec in list_model_specs()]

    for expected in [
        "predrnnv2",
        "predrnnpp",
        "convlstm",
        "simvp",
        "dlinear",
        "informer",
        "autoformer",
        "patchtst",
        "timemixer",
        "timexer",
        "tsmixer",
        "crossformer",
        "earthformer",
        "etsformer",
        "fedformer",
        "itransformer",
        "mau",
        "nbeats",
        "nhits",
        "pyraformer",
        "rnn_cnn_rnn",
        "cnn_rnn_cnn_rnn_cnn",
        "simvp_3dconv",
        "simvp_hybrid3d",
        "convlstm_mst",
        "dlinear_mst",
        "convlstm_phase_gated_mst",
        "convlstm_mst_feature_refiner",
        "convlstm_climatology_anomaly",
    ]:
        assert expected in model_ids

    assert "predrnnv2_sphere" not in model_ids
    assert normalize_model_architecture("crossformer") == "crossformer"
    assert normalize_model_architecture("predrnnv2_sphere") == "predrnnv2"
    assert normalize_use_sphere({"use_sphere": True}) is True
    assert normalize_use_sphere({"model_architecture": "predrnnv2_sphere"}) is True


def test_every_exposed_model_has_a_registered_factory():
    exposed_ids = {spec["id"] for spec in list_model_specs()}

    assert exposed_ids == set(FACTORIES)


def test_unknown_architecture_is_rejected_instead_of_training_predrnnv2():
    try:
        build_forecaster(
            architecture="missing_architecture",
            input_channels=1,
            selected_channels=[],
            hidden_dims=[4, 4, 4],
            height=8,
            width=8,
            window=3,
            horizon=2,
            use_sphere=False,
        )
    except ValueError as exc:
        assert "Unsupported model architecture" in str(exc)
    else:
        raise AssertionError("Unknown architecture should not fall back to PredRNNv2")


def test_architecture_params_override_factory_defaults():
    patchtst_model = build_forecaster(
        architecture="patchtst",
        input_channels=2,
        selected_channels=["D"],
        hidden_dims=[4, 4, 4],
        height=8,
        width=8,
        window=6,
        horizon=2,
        use_sphere=False,
        architecture_params={
            "patch_len": 3,
            "stride": 2,
            "d_model": 12,
            "n_heads": 3,
            "e_layers": 2,
            "d_ff": 24,
            "dropout": 0.25,
        },
    )

    assert patchtst_model.backbone.backbone.patch_len == 3
    assert patchtst_model.backbone.backbone.stride == 2
    assert patchtst_model.backbone.backbone.patch_embed.out_features == 12

    simvp_model = build_forecaster(
        architecture="simvp",
        input_channels=2,
        selected_channels=["D"],
        hidden_dims=[4, 4, 4],
        height=8,
        width=8,
        window=3,
        horizon=2,
        use_sphere=False,
        architecture_params={
            "spatial_hidden_dim": 6,
            "temporal_hidden_dim": 10,
            "temporal_depth": 3,
            "dropout": 0.2,
        },
    )

    assert simvp_model.backbone.encoder.proj.conv.out_channels == 6
    assert len(simvp_model.backbone.translator.blocks) == 3


def test_migrated_ablation_architecture_params_build_expected_backbones():
    crossformer_model = build_forecaster(
        architecture="crossformer",
        input_channels=2,
        selected_channels=["D"],
        hidden_dims=[4, 4, 4],
        height=8,
        width=8,
        window=3,
        horizon=2,
        use_sphere=False,
        architecture_params={
            "seg_len": 1,
            "win_size": 2,
            "factor": 2,
            "d_model": 12,
            "n_heads": 3,
            "e_layers": 1,
            "d_ff": 24,
            "dropout": 0.1,
        },
    )

    assert crossformer_model.backbone.backbone.embedding.value_embedding.out_features == 12

    convlstm_mst_model = build_forecaster(
        architecture="convlstm_mst",
        input_channels=2,
        selected_channels=["D"],
        hidden_dims=[4, 4, 4],
        height=8,
        width=8,
        window=3,
        horizon=2,
        use_sphere=False,
        architecture_params={
            "hidden_dim": 5,
            "mst_spatial_hidden_dim": 6,
            "mst_temporal_hidden_dim": 7,
            "mst_num_downsample": 1,
            "mst_temporal_depth": 2,
            "dropout": 0.1,
        },
    )

    assert convlstm_mst_model.backbone.cells[0].conv.out_channels == 20
    assert len(convlstm_mst_model.backbone.mst_block.translator.blocks) == 2


def test_convlstm_mst_refiner_variants_use_single_hidden_dim_param():
    common = {
        "input_channels": 2,
        "selected_channels": ["D"],
        "hidden_dims": [4, 4, 4],
        "height": 8,
        "width": 8,
        "window": 3,
        "horizon": 2,
        "use_sphere": False,
    }

    feature_refiner = build_forecaster(
        architecture="convlstm_mst_feature_refiner",
        architecture_params={"hidden_dim": 5, "mst_temporal_depth": 1},
        **common,
    )
    climatology_refiner = build_forecaster(
        architecture="convlstm_climatology_anomaly",
        architecture_params={"hidden_dim": 6, "mst_temporal_depth": 1},
        **common,
    )

    assert feature_refiner.backbone.hidden_dims == [5]
    assert feature_refiner.backbone.cells[0].conv.out_channels == 20
    assert climatology_refiner.backbone.hidden_dims == [6]
    assert climatology_refiner.backbone.cells[0].conv.out_channels == 24


def test_selected_backbones_accept_arbitrary_channel_count_with_optional_sphere():
    x = torch.randn(2, 3, 2, 8, 8)
    ls = torch.tensor([[0.0, 45.0, 90.0], [120.0, 180.0, 240.0]])

    for architecture in [spec["id"] for spec in list_model_specs()]:
        model = build_forecaster(
            architecture=architecture,
            input_channels=2,
            selected_channels=["D"],
            hidden_dims=[4, 4, 4],
            height=8,
            width=8,
            window=3,
            horizon=2,
            use_sphere=True,
        )
        model.eval()

        with torch.no_grad():
            output = model(x, ls)

        assert output.shape == (2, 2, 1, 8, 8)


if __name__ == "__main__":
    test_model_registry_exposes_migrated_raw_backbones_without_sphere_suffix()
    test_every_exposed_model_has_a_registered_factory()
    test_unknown_architecture_is_rejected_instead_of_training_predrnnv2()
    test_architecture_params_override_factory_defaults()
    test_migrated_ablation_architecture_params_build_expected_backbones()
    test_selected_backbones_accept_arbitrary_channel_count_with_optional_sphere()
    print("training model zoo tests passed")
