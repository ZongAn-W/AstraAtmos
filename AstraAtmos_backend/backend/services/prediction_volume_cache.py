"""已训练模型预测的「标准化体积」进程内缓存

单次预测只需要一个滑窗样本，但数据准备会先加载全部 OpenMARS/MCD 文件、插值到
OpenMARS 时刻并拟合标准化参数。原始插值体积与 ``window`` / ``horizon`` 无关，
但标准化拟合边界受切分比例、``window`` 和 ``horizon`` 影响，却要
在每个未命中缓存的预测请求里重算一遍（实测约 20 s、并物化数 GB 滑窗）。

本模块缓存**标准化后的连续体积**与其统计量：

- 缓存体积约 ``total_time × height × width × channels × 4B``（实测约 27 MB），
  而不是按样本展开后的数 GB 滑窗张量；
- 调用方拿到体积后只切出自己需要的滑窗，因此不需要为缓存付出大内存代价；
- 键包含目录身份、文件清单、通道、切分比例、``window`` / ``horizon`` 与
  ``(路径, 大小, 修改时间)``，文件被替换即自动失效，
不会返回过期数值。体积同时携带每个 MY 块的合法样本起点，切窗不会跨年度边界。

LRU 上限按条目数控制；单条体积规模取决于数据集，条目数上限取小值以避免内存堆积。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from services.mars_checkpoint import MCD_WINDOW_POLICY, OPENMARS_WINDOW_POLICY

# 体积规模为数百 MB 级时，少量条目即可覆盖多任务复用；超出按 LRU 淘汰。
MAX_CACHED_VOLUMES = 4
_CACHE_HIT = "hit"
_CACHE_MISS = "miss"


@dataclass(frozen=True)
class ScaledVolume:
    """标准化后的连续体积及其统计量。"""

    values: Any  # np.ndarray[total_time, height, width, channels]，float32
    y_scaled: Any  # np.ndarray[total_time, height, width]，float32
    ls: Any  # np.ndarray[total_time] 或 None
    y_mean: float
    y_std: float
    height: int
    width: int
    latitude: Any  # np.ndarray 或 None
    longitude: Any  # np.ndarray 或 None
    split_idx: int
    input_means: tuple[float, ...] = ()
    input_stds: tuple[float, ...] = ()
    dataset_id: str = "openmars_mcd"
    source_type: str = "openmars_mcd"
    data_dir: Path | None = None
    manifest: tuple[dict[str, Any], ...] = ()
    dataset_fingerprint: str | None = None
    sample_starts: Any = None
    sample_mars_years: tuple[int, ...] = ()
    blocks: tuple[Any, ...] = ()
    train_sample_end: int = 0
    data_directories: tuple[Path, ...] = ()
    split_ranges: dict[str, dict[str, Any]] | None = None
    split_window_starts: dict[str, tuple[int, ...]] | None = None
    split_policy: str = "legacy_compatibility"


_cache: "OrderedDict[tuple, ScaledVolume]" = OrderedDict()
_cache_lock = threading.Lock()
_last_lookup: str | None = None


def _list_directory_files(directory: Path) -> list[tuple[str, int, int]]:
    """列出目录内 NetCDF 文件清单与指纹，用于判断缓存是否仍然有效。"""
    try:
        entries = [
            (entry.name, entry.stat().st_size, entry.stat().st_mtime_ns)
            for entry in sorted(Path(directory).glob("*.nc"), key=lambda item: item.name)
        ]
    except OSError:
        # 目录不可读时不做缓存：返回唯一值使签名永不命中。
        return [("<unreadable>", 0, 0)]
    return entries


def volume_signature(
    *,
    openmars_dir: Any,
    mcd_dir: Any,
    selected_channels: Any,
    training_dataset: str,
    mcd_overview_dir: Any = None,
    cache_prefix: str = "",
    split_ratios: Any = None,
    window: int | None = None,
    horizon: int | None = None,
) -> tuple:
    """构造缓存签名；任何影响体积数值的输入都必须进入签名。"""
    signature = (
        cache_prefix,
        str(Path(openmars_dir).resolve()),
        str(Path(mcd_dir).resolve()),
        str(Path(mcd_overview_dir).resolve()) if mcd_overview_dir else "",
        str(training_dataset),
        tuple(selected_channels or ()),
        tuple(sorted((str(key), str(value)) for key, value in (split_ratios or {}).items())),
        None if window is None else int(window),
        None if horizon is None else int(horizon),
        tuple(_list_directory_files(Path(openmars_dir))),
        tuple(_list_directory_files(Path(mcd_dir))),
        tuple(_list_directory_files(Path(mcd_overview_dir))) if mcd_overview_dir else (),
    )
    if training_dataset == "openmars_mcd":
        return signature + (OPENMARS_WINDOW_POLICY,)
    if training_dataset == "mcd_overview":
        return signature + (MCD_WINDOW_POLICY,)
    return signature


def get_scaled_volume(signature: tuple) -> ScaledVolume | None:
    """取缓存的体积；命中时标记本次请求为 ``hit``。"""
    global _last_lookup
    with _cache_lock:
        volume = _cache.get(signature)
        if volume is not None:
            _cache.move_to_end(signature)
        _last_lookup = _CACHE_HIT if volume is not None else _CACHE_MISS
        return volume


def put_scaled_volume(signature: tuple, volume: ScaledVolume) -> None:
    with _cache_lock:
        _cache[signature] = volume
        _cache.move_to_end(signature)
        while len(_cache) > MAX_CACHED_VOLUMES:
            _cache.popitem(last=False)


def last_lookup_status() -> str | None:
    """最近一次 ``get_scaled_volume`` 的结果，供预测结果标注数据来源。"""
    return _last_lookup


def clear_scaled_volumes() -> None:
    global _last_lookup
    with _cache_lock:
        _cache.clear()
        _last_lookup = None
