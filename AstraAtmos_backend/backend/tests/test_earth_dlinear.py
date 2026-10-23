"""Earth DLinear construction: channel counts, forward/backward and reload."""

import numpy as np
import pytest
import torch

from training_backbones.earth_daily_model import (
    EARTH_HIDDEN_DIMS,
    create_earth_forecaster,
    earth_forward,
    earth_model_config,
    forecaster_from_config,
    require_earth_channel_order,
    state_dict_to_cpu,
)


def _channel_orders():
    return [
        ["TO3"],
        ["TO3", "U10M"],
        ["TO3", "U10M", "SWGDN"],
        ["TO3", "U10M", "V10M", "T2M"],
        ["TO3", "U10M", "V10M", "T2M", "SWGDN"],
    ]


@pytest.mark.parametrize("order", _channel_orders())
def test_forward_shape_is_exactly_the_target_shape(order):
    model = create_earth_forecaster(order)
    inputs = torch.randn(2, 7, len(order), 36, 72)
    output = earth_forward(model, inputs)
    assert tuple(output.shape) == (2, 3, 1, 36, 72)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("order", _channel_orders())
def test_backward_and_one_optimizer_step_are_finite(order):
    torch.manual_seed(3)
    model = create_earth_forecaster(order)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn(2, 7, len(order), 36, 72)
    targets = torch.randn(2, 3, 1, 36, 72)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    output = earth_forward(model, inputs)
    loss = torch.nn.functional.mse_loss(output, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"
    optimizer.step()
    moved = any(
        not torch.equal(previous, parameter.detach())
        for previous, parameter in zip(before, model.parameters())
    )
    assert moved, "the optimizer step did not update any parameter"


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_batch_size_does_not_change_the_output_contract(batch_size):
    model = create_earth_forecaster(["TO3", "T2M"])
    inputs = torch.randn(batch_size, 7, 2, 36, 72)
    assert tuple(earth_forward(model, inputs).shape) == (batch_size, 3, 1, 36, 72)


def test_channel_projector_is_trainable_and_saved_for_reduced_inputs():
    model = create_earth_forecaster(["TO3", "U10M"])
    # The backbone expects five channels, so a projector bridges 2 -> 5.
    assert isinstance(model.projector.projection, torch.nn.Conv2d)
    state = state_dict_to_cpu(model)
    assert any(key.startswith("projector.projection") for key in state)
    assert any(key.startswith("backbone.") for key in state)

    reloaded = create_earth_forecaster(["TO3", "U10M"])
    reloaded.load_state_dict(state, strict=True)
    inputs = torch.randn(2, 7, 2, 36, 72)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.equal(earth_forward(model, inputs), earth_forward(reloaded, inputs))


def test_five_channel_model_uses_the_identity_projector():
    model = create_earth_forecaster(["TO3", "U10M", "V10M", "T2M", "SWGDN"])
    assert isinstance(model.projector.projection, torch.nn.Identity)
    assert not any(key.startswith("projector.projection") for key in state_dict_to_cpu(model))


def test_sphere_is_never_used_for_earth():
    model = create_earth_forecaster(["TO3"])
    assert model.use_sphere is False
    assert model.sphere is None
    config = earth_model_config(["TO3"])
    assert config["use_sphere"] is False


def test_zero_time_placeholder_is_not_a_saved_feature():
    model = create_earth_forecaster(["TO3"])
    state = state_dict_to_cpu(model)
    # The placeholder only satisfies the shared signature; it must not become state.
    assert not any("ls" in key or "phase" in key for key in state)


def test_hidden_layers_change_the_structure():
    shallow = create_earth_forecaster(["TO3"], linear_hidden_layers=1)
    deep = create_earth_forecaster(["TO3"], linear_hidden_layers=4)
    shallow_count = sum(parameter.numel() for parameter in shallow.parameters())
    deep_count = sum(parameter.numel() for parameter in deep.parameters())
    assert deep_count > shallow_count
    for bad in (0, 5, -1):
        with pytest.raises(ValueError):
            create_earth_forecaster(["TO3"], linear_hidden_layers=bad)


def test_channel_order_validation_rejects_non_canonical_input():
    for bad in ([], ["U10M"], ["TO3", "SWGDN", "U10M"], ["TO3", "TO3"], ["TO3", "NOPE"]):
        with pytest.raises(ValueError):
            require_earth_channel_order(bad)
        with pytest.raises(ValueError):
            create_earth_forecaster(bad)


def test_grid_and_window_mismatch_is_rejected():
    model = create_earth_forecaster(["TO3"])
    with pytest.raises(ValueError):
        earth_forward(model, torch.randn(1, 7, 1, 31, 49))
    # A different window length is rejected by the backbone; wrap it so the failure
    # is reported as a ValueError like every other contract violation.
    with pytest.raises((ValueError, RuntimeError)):
        earth_forward(model, torch.randn(1, 5, 1, 36, 72))


def test_model_config_rebuilds_the_exact_module():
    config = earth_model_config(["TO3", "V10M"], linear_hidden_layers=3)
    assert config["architecture"] == "dlinear"
    assert config["input_channels"] == 2
    assert config["input_channel_order"] == ["TO3", "V10M"]
    assert config["selected_channels"] == ["V10M"]
    assert config["hidden_dims"] == list(EARTH_HIDDEN_DIMS)
    assert (config["height"], config["width"]) == (36, 72)
    assert (config["window"], config["horizon"]) == (7, 3)
    assert config["architecture_params"] == {"linear_hidden_layers": 3}

    rebuilt = forecaster_from_config(config)
    original = create_earth_forecaster(["TO3", "V10M"], 3)
    rebuilt.load_state_dict(state_dict_to_cpu(original), strict=True)
    inputs = torch.randn(2, 7, 2, 36, 72)
    original.eval()
    rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(earth_forward(original, inputs), earth_forward(rebuilt, inputs))


def test_forecaster_from_config_rejects_foreign_configs():
    with pytest.raises(ValueError):
        forecaster_from_config({"architecture": "simvp", "input_channel_order": ["TO3"]})
    with pytest.raises(ValueError):
        forecaster_from_config("nope")


def test_earth_forward_flags_non_finite_output():
    class ConstantNaN(torch.nn.Module):
        def forward(self, x, ls=None):
            return torch.full((x.shape[0], 3, 36, 72), float("nan"))

    with pytest.raises(ValueError):
        earth_forward(ConstantNaN(), torch.randn(1, 7, 1, 36, 72))


def test_model_output_is_deterministic_for_the_same_input():
    model = create_earth_forecaster(["TO3", "T2M"])
    model.eval()
    inputs = torch.randn(1, 7, 2, 36, 72)
    with torch.no_grad():
        first = earth_forward(model, inputs)
        second = earth_forward(model, inputs)
    assert torch.equal(first, second)
    assert np.isfinite(first.numpy()).all()
