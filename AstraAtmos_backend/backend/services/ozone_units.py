from __future__ import annotations

import numpy as np

KG_M2_PER_UM_ATM_O3 = 2.14e-6
OZONE_COLUMN_UNIT = "um-atm"


def _compact_unit(value: str | None) -> str:
    return (
        str(value or "")
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("micro", "u")
        .replace(" ", "")
        .replace("_", "")
        .replace("^", "")
        .replace("**", "")
    )


def units_indicate_kg_m2(units: str | None) -> bool:
    compact = _compact_unit(units)
    return "kgm-2" in compact or "kg/m2" in compact or "kgm2" in compact


def units_indicate_um_atm(units: str | None) -> bool:
    compact = _compact_unit(units).replace("-", "")
    return "umatm" in compact


def ozone_kg_m2_to_um_atm(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) / KG_M2_PER_UM_ATM_O3).astype(np.float32)


def looks_like_mcd_kg_m2_ozone(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    finite_max = float(np.nanmax(np.abs(finite)))
    return 0.0 < finite_max < 0.01


def normalize_ozone_column_units(
    values: np.ndarray,
    units: str | None = None,
    *,
    allow_mcd_legacy_heuristic: bool = False,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if units_indicate_um_atm(units):
        return arr
    if units_indicate_kg_m2(units) or (allow_mcd_legacy_heuristic and looks_like_mcd_kg_m2_ozone(arr)):
        return ozone_kg_m2_to_um_atm(arr)
    return arr
