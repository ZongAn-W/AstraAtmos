"""Earth DLinear construction and forward pass.

The Earth path reuses the project's existing DLinear backbone through
``build_forecaster`` instead of adding a second implementation. Two things make
that safe:

* the backbone expects five channels, and ``ForecasterAdapter`` already inserts a
  trainable 1x1 ``ChannelProjector`` whenever the input channel count differs, so
  a TO3-only model (1 input) is a real, trainable model rather than an error;
* the backbone's ``forward`` also accepts an Ls sequence, which Earth has no
  equivalent of. :func:`earth_forward` passes a zero placeholder and documents
  that it carries no physical meaning.

``use_sphere`` is always false here: SPHERE is a Mars seasonal front end and is
explicitly not part of the Earth contract.

The historical default remains 7-to-3 on 36x72. Callers select the three-hourly
56-to-24, 240x480 profile explicitly. DLinear shares its temporal weights across
grid points, so its forward pass can process spatial tiles without changing
those weights; callers pass the actual tile height and width to ``earth_forward``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from services.earth_dataset import (
    CHANNELS,
    TARGET_CHANNEL,
    canonical_input_channels,
)
from training_backbones.model_zoo import build_forecaster

EARTH_ARCHITECTURE = "dlinear"
EARTH_HIDDEN_DIMS = [64, 64, 64]
EARTH_HEIGHT = 36
EARTH_WIDTH = 72
EARTH_WINDOW = 7
EARTH_HORIZON = 3


def require_earth_channel_order(input_channel_order: Sequence[str]) -> list[str]:
    """Validate and return the canonical Earth model input order.

    ``TO3`` must be present and first, and every other name must be a published
    Earth channel. A model built for a different order would silently apply the
    wrong normalization vector, so this is a hard error rather than a reorder.
    """
    order = [str(name).strip().upper() for name in (input_channel_order or [])]
    if not order:
        raise ValueError("Earth model requires at least the TO3 input channel")
    if order[0] != TARGET_CHANNEL:
        raise ValueError(f"Earth model input order must start with {TARGET_CHANNEL}")
    if len(set(order)) != len(order):
        raise ValueError("Earth model input order contains duplicates")
    unknown = [name for name in order if name not in CHANNELS]
    if unknown:
        raise ValueError(f"Unsupported Earth input channel: {unknown[0]}")
    canonical = canonical_input_channels(order[1:])
    if canonical != order:
        raise ValueError("Earth model input order is not the canonical order")
    return order


def create_earth_forecaster(
    input_channel_order: Sequence[str],
    linear_hidden_layers: int = 2,
    *,
    window: int = EARTH_WINDOW,
    horizon: int = EARTH_HORIZON,
    height: int = EARTH_HEIGHT,
    width: int = EARTH_WIDTH,
) -> torch.nn.Module:
    """Build the official DLinear; window/horizon are time steps, not days."""
    order = require_earth_channel_order(input_channel_order)
    hidden_layers = int(linear_hidden_layers)
    if hidden_layers < 1 or hidden_layers > 4:
        raise ValueError("linear_hidden_layers must be between 1 and 4")
    return build_forecaster(
        architecture=EARTH_ARCHITECTURE,
        input_channels=len(order),
        selected_channels=list(order[1:]),
        hidden_dims=list(EARTH_HIDDEN_DIMS),
        height=int(height),
        width=int(width),
        window=int(window),
        horizon=int(horizon),
        use_sphere=False,
        architecture_params={"linear_hidden_layers": hidden_layers},
    )


def earth_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    horizon: int = EARTH_HORIZON,
    height: int = EARTH_HEIGHT,
    width: int = EARTH_WIDTH,
) -> torch.Tensor:
    """Run an Earth forecaster and return ``[B, horizon, 1, H, W]``.

    The zero time tensor only satisfies the shared backbone signature, whose
    seasonal front end is disabled for Earth. It is never saved as a feature.
    """
    if inputs.dim() != 5:
        raise ValueError("Earth model input must be [B, window, C, H, W]")
    batch, _, _, input_height, input_width = inputs.shape
    if input_height != int(height) or input_width != int(width):
        raise ValueError(
            f"Earth model expects a {int(height)}x{int(width)} grid, got "
            f"{input_height}x{input_width}"
        )
    time_placeholder = inputs.new_zeros((batch, inputs.shape[1]))
    output = model(inputs, time_placeholder)
    expected = (batch, int(horizon), 1, int(height), int(width))
    if tuple(output.shape) != expected:
        raise ValueError(f"Unexpected Earth output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("Non-finite Earth model output")
    return output


def earth_model_config(
    input_channel_order: Sequence[str],
    linear_hidden_layers: int = 2,
    *,
    window: int = EARTH_WINDOW,
    horizon: int = EARTH_HORIZON,
    height: int = EARTH_HEIGHT,
    width: int = EARTH_WIDTH,
    implementation_id: Optional[str] = None,
) -> dict:
    """Return the complete, serialisable model configuration.

    It must be sufficient on its own to rebuild the exact module through
    :func:`create_earth_forecaster`, including the hidden dims that keep the
    ``build_forecaster`` call contract intact even though DLinear ignores them.
    """
    order = require_earth_channel_order(input_channel_order)
    config = {
        "architecture": EARTH_ARCHITECTURE,
        "input_channels": len(order),
        "input_channel_order": order,
        "selected_channels": list(order[1:]),
        "hidden_dims": list(EARTH_HIDDEN_DIMS),
        "height": int(height),
        "width": int(width),
        "window": int(window),
        "horizon": int(horizon),
        "use_sphere": False,
        "architecture_params": {"linear_hidden_layers": int(linear_hidden_layers)},
    }
    if implementation_id is not None:
        config["implementation_id"] = str(implementation_id)
    return config


def forecaster_from_config(model_config: dict) -> torch.nn.Module:
    """Rebuild a forecaster from a saved model configuration."""
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a mapping")
    if model_config.get("architecture") != EARTH_ARCHITECTURE:
        raise ValueError("model_config is not an Earth DLinear configuration")
    architecture_params = model_config.get("architecture_params") or {}
    return create_earth_forecaster(
        model_config.get("input_channel_order") or [],
        int(architecture_params.get("linear_hidden_layers", 2)),
        window=int(model_config.get("window", EARTH_WINDOW)),
        horizon=int(model_config.get("horizon", EARTH_HORIZON)),
        height=int(model_config.get("height", EARTH_HEIGHT)),
        width=int(model_config.get("width", EARTH_WIDTH)),
    )


def state_dict_to_cpu(model: torch.nn.Module) -> dict:
    """Return a detached CPU copy of a model's state, safe to serialise."""
    return {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}


__all__ = [
    "EARTH_ARCHITECTURE",
    "EARTH_HEIGHT",
    "EARTH_HIDDEN_DIMS",
    "EARTH_HORIZON",
    "EARTH_WIDTH",
    "EARTH_WINDOW",
    "create_earth_forecaster",
    "earth_forward",
    "earth_model_config",
    "forecaster_from_config",
    "require_earth_channel_order",
    "state_dict_to_cpu",
]
