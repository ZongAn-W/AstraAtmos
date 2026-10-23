from __future__ import annotations

from typing import Any

import torch.nn as nn


SUPPORTED_FREEZE_MODES = {"none", "backbone", "head"}
HEAD_MODULE_NAMES = {
    "forecast_head",
    "target_head",
    "head",
    "output_head",
    "output_proj",
    "out_proj",
    "readout",
    "classifier",
    "regressor",
}
HEAD_MODULE_SUFFIXES = ("_head", "_output", "_readout")
HEAD_MODULE_CONTAINS = ("predictor",)


def normalize_freeze_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in SUPPORTED_FREEZE_MODES else "none"


def _is_head_module_name(module_name: str) -> bool:
    if not module_name:
        return False
    last_part = module_name.rsplit(".", 1)[-1].lower()
    if last_part in HEAD_MODULE_NAMES:
        return True
    if any(last_part.endswith(suffix) for suffix in HEAD_MODULE_SUFFIXES):
        return True
    return any(marker in last_part for marker in HEAD_MODULE_CONTAINS)


def _head_parameter_prefixes(model: nn.Module) -> set[str]:
    prefixes: set[str] = set()
    for module_name, module in model.named_modules():
        if module_name and _is_head_module_name(module_name) and any(True for _ in module.parameters()):
            prefixes.add(module_name)
    return prefixes


def _matches_prefix(parameter_name: str, prefixes: set[str]) -> bool:
    return any(parameter_name == prefix or parameter_name.startswith(f"{prefix}.") for prefix in prefixes)


def _has_backbone_module(model: nn.Module) -> bool:
    return any(name == "backbone" for name, _module in model.named_modules())


def _parameter_count(parameters: list[nn.Parameter]) -> int:
    return sum(int(parameter.numel()) for parameter in parameters)


def apply_freeze_strategy(model: nn.Module, freeze_mode: Any) -> dict[str, Any]:
    mode = normalize_freeze_mode(freeze_mode)
    named_parameters = list(model.named_parameters())
    total_count = _parameter_count([parameter for _name, parameter in named_parameters])

    if mode == "none":
        for _name, parameter in named_parameters:
            parameter.requires_grad = True
    else:
        head_prefixes = _head_parameter_prefixes(model)
        has_backbone = _has_backbone_module(model)
        for name, parameter in named_parameters:
            is_head = _matches_prefix(name, head_prefixes)
            if mode == "head":
                parameter.requires_grad = is_head
            elif has_backbone:
                parameter.requires_grad = not name.startswith("backbone.") or is_head
            else:
                parameter.requires_grad = is_head

    trainable_parameters = [
        parameter
        for _name, parameter in named_parameters
        if parameter.requires_grad
    ]
    trainable_count = _parameter_count(trainable_parameters)
    if trainable_count <= 0:
        raise ValueError(f"No trainable parameters remain after applying freeze_mode={mode}")

    return {
        "mode": mode,
        "total_tensor_count": len(named_parameters),
        "trainable_tensor_count": len(trainable_parameters),
        "total_parameter_count": total_count,
        "trainable_parameter_count": trainable_count,
    }
