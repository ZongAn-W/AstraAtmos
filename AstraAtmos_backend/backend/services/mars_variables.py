"""Canonical variables for a single-target Mars forecasting experiment.

Explicit inputs describe the entire model tensor. Historical tasks without
``input_variables`` keep their O3-first auxiliary-channel contract.
"""

from __future__ import annotations

import json
from typing import Any

MARS_VARIABLE_ORDER = ("O3", "U", "V", "D", "S", "T")
MARS_VARIABLE_UNITS = {"O3": "um-atm", "U": "m/s", "V": "m/s", "D": "1", "S": "W/m²", "T": "K"}


def mars_target_variable(value: Any = None) -> str:
    target = "O3" if value is None else value
    if not isinstance(target, str) or target.strip().upper() not in MARS_VARIABLE_ORDER:
        raise ValueError("Mars target_variable must be one of O3, U, V, D, S, T")
    return target.strip().upper()


def mars_input_variables(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            value = value.replace("+", ",").split(",")
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Mars input_variables must contain at least one variable")
    if any(not isinstance(item, str) or item.strip().upper() not in MARS_VARIABLE_ORDER for item in value):
        raise ValueError("Mars input_variables contains an unavailable variable")
    selected = {item.strip().upper() for item in value}
    if len(selected) != len(value):
        raise ValueError("Mars input_variables must not contain duplicate variables")
    return [item for item in MARS_VARIABLE_ORDER if item in selected]


def resolve_mars_variables(hyperparameters: dict | None = None) -> tuple[str, list[str]]:
    hypers = hyperparameters or {}
    target = mars_target_variable(hypers.get("target_variable"))
    if hypers.get("input_variables") is not None:
        inputs = mars_input_variables(hypers["input_variables"])
    else:
        selected = hypers.get("selected_channels") or []
        if isinstance(selected, str):
            try:
                selected = json.loads(selected)
            except ValueError:
                selected = selected.replace("+", ",").split(",")
        if not isinstance(selected, (list, tuple, set)):
            selected = []
        selected = {str(item).strip().upper() for item in selected}
        inputs = ["O3", *[item for item in MARS_VARIABLE_ORDER[1:] if item in selected]]
    return target, inputs


def mars_variable_metadata(hyperparameters: dict | None = None) -> dict:
    target, inputs = resolve_mars_variables(hyperparameters)
    return {"target_variable": target, "target_unit": MARS_VARIABLE_UNITS[target], "input_variables": inputs}
