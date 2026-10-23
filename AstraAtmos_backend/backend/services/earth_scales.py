"""Shared display scales for the Earth 2D APIs.

Both the overview endpoints and the research endpoints must describe the same
variable with the same fixed dataset-scope colour scale, so the rule lives here
once instead of being duplicated per service.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Declared here rather than imported from the overview service so this module
# stays free of import cycles. The two tuples must stay in step with
# ``services.earth_dataset.CHANNELS``.
VARIABLE_IDS = ("TO3", "U10M", "V10M", "T2M", "SWGDN")
WIND_VARIABLE_IDS = ("U10M", "V10M")

COLOR_SCOPE = "dataset"


def color_range(field: Any, variable: str) -> dict:
    """Fixed dataset-scope colour range of one variable.

    The range is taken from the whole release, not the current day, so playback
    does not flicker. Signed wind components are centred on zero, keeping zero in
    the middle of the scale so east/west and north/south read consistently.
    """
    if variable not in VARIABLE_IDS:
        raise ValueError(f"Unknown Earth variable: {variable}")
    values = np.asarray(field, dtype="float32")
    low, high = float(values.min()), float(values.max())
    centered = variable in WIND_VARIABLE_IDS
    if centered:
        bound = max(abs(low), abs(high))
        low, high = -bound, bound
    return {
        "min": low,
        "max": high,
        "scope": COLOR_SCOPE,
        "centered_on_zero": centered,
    }


__all__ = ["COLOR_SCOPE", "color_range"]
