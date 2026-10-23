from __future__ import annotations

from typing import Any

import torch
from torch import nn


LS_AUXILIARY_INPUT = {
    "required": True,
    "shape": ["batch", "window"],
    "dtype": "float32",
    "unit": "degree",
}
TOPOGRAPHY_AUXILIARY_INPUT = {
    "required": True,
    "shape": ["batch", 1, "height", "width"],
    "dtype": "float32",
    "unit": "meter",
}
AUXILIARY_INPUT_SPECS = {
    "ls": LS_AUXILIARY_INPUT,
    "topography": TOPOGRAPHY_AUXILIARY_INPUT,
}
CONTRACT_ATTRIBUTE = "_astraatmos_auxiliary_inputs"


def normalize_auxiliary_inputs(model_spec: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(model_spec, dict):
        raise ValueError("MODEL_SPEC must be a dict")
    if "auxiliary_inputs" not in model_spec:
        return {}

    auxiliary_inputs = model_spec["auxiliary_inputs"]
    if not isinstance(auxiliary_inputs, dict):
        raise ValueError("MODEL_SPEC.auxiliary_inputs must be a dict")
    unsupported = sorted(set(auxiliary_inputs) - set(AUXILIARY_INPUT_SPECS))
    if unsupported:
        raise ValueError(
            "MODEL_SPEC.auxiliary_inputs supports only ls and topography, "
            f"got unsupported inputs: {unsupported}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for name, expected_spec in AUXILIARY_INPUT_SPECS.items():
        if name not in auxiliary_inputs:
            continue
        actual_spec = auxiliary_inputs[name]
        if not isinstance(actual_spec, dict):
            raise ValueError(
                f"MODEL_SPEC.auxiliary_inputs.{name} must be a dict, "
                f"got {actual_spec!r} ({type(actual_spec).__name__})"
            )

        expected_fields = set(expected_spec)
        actual_fields = set(actual_spec)
        missing_fields = sorted(expected_fields - actual_fields)
        unexpected_fields = sorted(actual_fields - expected_fields)
        if missing_fields:
            field = missing_fields[0]
            expected = expected_spec[field]
            raise ValueError(
                f"MODEL_SPEC.auxiliary_inputs.{name} missing fields: "
                f"MODEL_SPEC.auxiliary_inputs.{name}.{field} expected "
                f"{expected!r} ({type(expected).__name__}), got <missing>"
            )
        if unexpected_fields:
            field = unexpected_fields[0]
            actual = actual_spec[field]
            raise ValueError(
                f"MODEL_SPEC.auxiliary_inputs.{name} unexpected fields: "
                f"MODEL_SPEC.auxiliary_inputs.{name}.{field} expected "
                f"<not allowed>, got {actual!r} ({type(actual).__name__})"
            )

        for field, expected in expected_spec.items():
            actual = actual_spec[field]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"MODEL_SPEC.auxiliary_inputs.{name}.{field} expected "
                    f"{expected!r} ({type(expected).__name__}), got "
                    f"{actual!r} ({type(actual).__name__})"
                )
        normalized[name] = dict(expected_spec)
    return normalized


def attach_uploaded_model_contract(model: nn.Module, model_spec: Any) -> nn.Module:
    setattr(model, CONTRACT_ATTRIBUTE, normalize_auxiliary_inputs(model_spec))
    return model


def uploaded_model_requires_ls(model: nn.Module) -> bool:
    return "ls" in getattr(model, CONTRACT_ATTRIBUTE, {})


def uploaded_model_requires_topography(model: nn.Module) -> bool:
    return "topography" in getattr(model, CONTRACT_ATTRIBUTE, {})


def validate_ls_tensor(
    x: torch.Tensor,
    ls: Any,
    context: str,
) -> torch.Tensor:
    if ls is None:
        raise ValueError(f"{context} requires Ls data")
    if not isinstance(ls, torch.Tensor):
        raise ValueError(f"{context} Ls must be a torch.Tensor")
    if not ls.is_floating_point():
        raise ValueError(
            f"{context} Ls must use a floating-point dtype, got {ls.dtype}"
        )
    if ls.ndim != 2:
        raise ValueError(f"{context} Ls must be 2D, got {ls.ndim}D")

    expected_shape = (int(x.shape[0]), int(x.shape[1]))
    actual_shape = tuple(ls.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{context} Ls shape mismatch: "
            f"expected {expected_shape}, got {actual_shape}"
        )
    if not torch.isfinite(ls).all():
        raise ValueError(f"{context} Ls must contain only finite values")
    return ls


def validate_topography_tensor(
    x: torch.Tensor,
    topography: Any,
    context: str,
) -> torch.Tensor:
    if topography is None:
        raise ValueError(f"{context} requires topography data")
    if not isinstance(topography, torch.Tensor):
        raise ValueError(f"{context} topography must be a torch.Tensor")
    if topography.dtype != torch.float32:
        raise ValueError(
            f"{context} topography dtype mismatch: "
            f"expected torch.float32, got {topography.dtype}"
        )
    if topography.ndim != 4:
        raise ValueError(
            f"{context} topography must be 4D, got {topography.ndim}D"
        )

    expected_shape = (
        int(x.shape[0]),
        1,
        int(x.shape[-2]),
        int(x.shape[-1]),
    )
    actual_shape = tuple(topography.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{context} topography shape mismatch: "
            f"expected {expected_shape}, got {actual_shape}"
        )
    if not torch.isfinite(topography).all():
        raise ValueError(
            f"{context} topography must contain only finite values"
        )
    return topography


def expand_topography_batch(
    x: torch.Tensor,
    topography_grid: Any,
    context: str,
) -> torch.Tensor:
    if not isinstance(topography_grid, torch.Tensor):
        raise ValueError(f"{context} static topography must be a torch.Tensor")
    expected_shape = (1, int(x.shape[-2]), int(x.shape[-1]))
    actual_shape = tuple(topography_grid.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{context} static topography shape mismatch: "
            f"expected {expected_shape}, got {actual_shape}"
        )
    topography_batch = topography_grid.unsqueeze(0).expand(
        int(x.shape[0]), -1, -1, -1
    )
    return validate_topography_tensor(x, topography_batch, context)


def run_uploaded_model(
    model: nn.Module,
    x: torch.Tensor,
    ls: Any = None,
    topography: Any = None,
    *,
    context: str = "uploaded model",
):
    requires_ls = uploaded_model_requires_ls(model)
    requires_topography = uploaded_model_requires_topography(model)
    if requires_ls and requires_topography:
        return model(
            x,
            validate_ls_tensor(x, ls, context),
            validate_topography_tensor(x, topography, context),
        )
    if requires_ls:
        return model(x, validate_ls_tensor(x, ls, context))
    if requires_topography:
        return model(x, validate_topography_tensor(x, topography, context))
    return model(x)
