"""
时空对齐工具
- unwrap_ls: 处理 Ls 从 360 度回绕到 0 度的不连续问题
- interpolate_mcd_to_openmars: 将 MCD 数据插值到 OpenMARS 的 Ls 栅格点
"""

import numpy as np
from scipy.interpolate import interp1d


def unwrap_ls(ls_array: np.ndarray) -> np.ndarray:
    """
    将 Ls 数组展开为单调递增序列。
    火星太阳黄经 Ls 在 360 度时会回绕到 0 度，导致不连续。
    本函数检测跳变并加 360 度偏移使其单调。

    Args:
        ls_array: 原始 Ls 数组，可能有 360->0 跳变

    Returns:
        单调递增的 Ls 数组
    """
    source = np.asarray(ls_array, dtype=np.float64)
    unwrapped = source.copy()
    offset = 0.0
    for i in range(1, len(source)):
        if source[i] - source[i - 1] < -180:
            offset += 360.0
        unwrapped[i] += offset
    return unwrapped


def interpolate_mcd_to_openmars(
    mcd_data: np.ndarray,
    mcd_ls: np.ndarray,
    target_ls: np.ndarray,
    extrapolate: bool = False,
) -> np.ndarray:
    """
    将 MCD 数据从其原生 Ls 时间栅格点插值到 OpenMARS 的 Ls 栅格点。

    Args:
        mcd_data: MCD 变量数据，shape = (n_mcd_times, lat, lon)
        mcd_ls: MCD 对应的 Ls 数组，shape = (n_mcd_times,)
        target_ls: 目标 Ls 数组，shape = (n_target,)
        extrapolate: 是否开启外推模式

    Returns:
        插值后的数据，shape = (n_target, lat, lon)
    """
    mcd_arr = np.asarray(mcd_data, dtype=np.float64)
    if mcd_arr.ndim != 3:
        raise ValueError("mcd_data must have shape (time, lat, lon)")
    if mcd_arr.shape[0] != len(mcd_ls):
        raise ValueError("mcd_data time dimension must match mcd_ls length")

    n_lat, n_lon = mcd_arr.shape[1], mcd_arr.shape[2]

    mcd_ls_unwrapped = unwrap_ls(mcd_ls)
    target_ls_unwrapped = unwrap_ls(target_ls)

    if not extrapolate:
        valid_mask = (
            (target_ls_unwrapped >= mcd_ls_unwrapped[0]) &
            (target_ls_unwrapped <= mcd_ls_unwrapped[-1])
        )
        target_valid = target_ls_unwrapped[valid_mask]
        fill_val = np.nan
    else:
        valid_mask = slice(None)
        target_valid = target_ls_unwrapped
        fill_val = "extrapolate"

    result = np.full((len(target_ls), n_lat, n_lon), np.nan)
    if target_valid.size == 0:
        return result

    # One vectorized interpolation is much faster than looping over every grid cell.
    interpolator = interp1d(
        mcd_ls_unwrapped,
        mcd_arr,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=fill_val,
    )
    result[valid_mask] = interpolator(target_valid)
    return result


def expand_sol_hour_to_timeline(
    sol_array: np.ndarray,
    n_hours: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 MCD 的 sol 维度展开为 sol×hour 的连续时间索引。

    Args:
        sol_array: sol 数组, shape = (n_sol,)
        n_hours: 每 sol 的小时采样数

    Returns:
        (expanded_sol, expanded_hour) 两个一维数组
    """
    n_sol = len(sol_array)
    hours = np.linspace(0, 24, n_hours, endpoint=False)

    expanded_sol = np.repeat(sol_array, n_hours)
    expanded_hour = np.tile(hours, n_sol)

    return expanded_sol, expanded_hour
