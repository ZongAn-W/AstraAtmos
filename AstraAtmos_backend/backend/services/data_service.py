"""
底层数据加载服务
负责读取 OpenMARS 和 MCD 的 nc 文件，提供基本的基础 Numpy 张量。
"""

import logging
import glob
import numpy as np
from pathlib import Path

import xarray as xr

from config import (
    OPENMARS_DIR, MCD_DIR, MCD_VARIABLES,
    SUPPORTED_MARS_YEARS,
)
from core.data_align import interpolate_mcd_to_openmars
from services.mcd_file_locator import find_mcd_files

logger = logging.getLogger("astraatmos.data.service")


class DataService:
    """纯粹的 I/O 和内存加载层"""

    def __init__(self):
        self.openmars: dict[int, dict] = {}
        self.mcd: dict[int, dict] = {}
        self.aligned_mcd: dict[int, dict] = {}

        self._load_all()

    def _load_all(self):
        """启动时加载所有可用数据"""
        for my in SUPPORTED_MARS_YEARS:
            self._load_openmars(my)
            self._load_mcd(my)

        for my in self.openmars:
            if my in self.mcd:
                self._align_mcd(my) # 分析模式

        logger.info(
            f"底层数据源加载完成: OpenMARS={list(self.openmars.keys())}, "
            f"MCD={list(self.mcd.keys())}"
        )

    def _load_openmars(self, mars_year: int):
        pattern = str(OPENMARS_DIR / f"*my{mars_year}*ls*.nc")
        files = sorted(glob.glob(pattern))

        if not files:
            logger.warning(f"未找到 MY{mars_year} 的 OpenMARS 文件: {pattern}")
            return

        logger.info(f"加载 OpenMARS MY{mars_year}: {len(files)} 个文件")

        datasets = []
        for f in files:
            try:
                ds = xr.open_dataset(f)
                datasets.append(ds)
            except Exception as e:
                logger.error(f"读取文件失败 {f}: {e}")

        if not datasets:
            return

        combined = xr.concat(datasets, dim="time")

        o3col = combined["o3col"].values
        ls_arr = combined["Ls"].values
        lat_arr = combined["lat"].values
        lon_arr = combined["lon"].values

        if o3col.ndim == 4:
            o3col = np.nanmean(o3col, axis=1)

        sort_idx = np.argsort(ls_arr)
        ls_arr = ls_arr[sort_idx]
        o3col = o3col[sort_idx]

        self.openmars[mars_year] = {
            "o3col": o3col,
            "ls": ls_arr,
            "lat": lat_arr,
            "lon": lon_arr,
        }

        for ds in datasets:
            ds.close()

    def _load_mcd(self, mars_year: int):
        files = find_mcd_files(MCD_DIR, mars_year)

        if not files:
            logger.warning(f"未找到 MCD 数据文件: {MCD_DIR}")
            return

        logger.info(f"加载 MCD MY{mars_year}: {len(files)} 个文件")

        try:
            ds = xr.open_dataset(files[0])
        except Exception as e:
            logger.error(f"读取 MCD 文件失败: {e}")
            return

        mcd_data = {}
        for var in MCD_VARIABLES:
            if var in ds:
                arr = ds[var].values
                if arr.ndim == 4:
                    mcd_data[var] = np.nanmean(arr, axis=1)
                    mcd_data[f"{var}_hourly"] = arr
                else:
                    mcd_data[var] = arr
            else:
                logger.warning(f"  MCD 文件中缺少变量: {var}")

        if "Ls" in ds:
            mcd_data["ls"] = ds["Ls"].values
        elif "ls" in ds:
            mcd_data["ls"] = ds["ls"].values

        if "lat" in ds:
            mcd_data["lat"] = ds["lat"].values
        if "lon" in ds:
            mcd_data["lon"] = ds["lon"].values

        self.mcd[mars_year] = mcd_data
        ds.close()

    def _align_mcd(self, mars_year: int):
        om = self.openmars[mars_year]
        mc = self.mcd[mars_year]
        target_ls = om["ls"]

        if "ls" not in mc:
            logger.warning(f"MCD MY{mars_year} 缺少 Ls 索引，跳过对齐")
            return

        aligned = {"ls": target_ls}

        for var in MCD_VARIABLES:
            if var in mc:
                try:
                    aligned[var] = interpolate_mcd_to_openmars(
                        mc[var], mc["ls"], target_ls,
                        extrapolate=False
                    )
                except Exception as e:
                    logger.error(f"  对齐 {var} 失败: {e}")

        self.aligned_mcd[mars_year] = aligned

    # --- 对外暴露的安全数据访问接口 ---

    def get_openmars_data(self, mars_year: int) -> dict:
        if mars_year not in self.openmars:
            raise ValueError(
                f"MY{mars_year} 数据未加载。可用: {list(self.openmars.keys())}"
            )
        return self.openmars[mars_year]

    def get_mcd_data(self, mars_year: int) -> dict:
        if mars_year not in self.mcd:
            raise ValueError(f"MY{mars_year} 的原始 MCD 数据未加载。")
        return self.mcd[mars_year]

    def get_aligned_mcd_data(self, mars_year: int) -> dict:
        if mars_year not in self.aligned_mcd:
            raise ValueError(f"MY{mars_year} 的对齐 MCD 数据未加载。")
        return self.aligned_mcd[mars_year]

    def get_available_years(self) -> list[int]:
        return sorted(self.openmars.keys())

    def get_ls_range(self, mars_year: int) -> tuple[float, float]:
        om = self.get_openmars_data(mars_year)
        return float(om["ls"][0]), float(om["ls"][-1])

    @staticmethod
    def get_nearest_ls_index(ls_array: np.ndarray, target_ls: float) -> int:
        return int(np.argmin(np.abs(ls_array - target_ls)))
