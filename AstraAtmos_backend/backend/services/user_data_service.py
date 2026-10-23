"""
用户数据服务
-----------
按需读取用户上传的 .nc 文件，提供与 AnalysisService 相同格式的可视化数据。
数据不常驻内存，使用 LRU 缓存（最多缓存 8 个数据集）。

支持两种数据源：
  1. 用户私有数据：通过数据库记录的 file_path 字段定位
  2. 审核通过数据：approved/{record_id}/original.nc
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from cachetools import LRUCache

from config import (
    APPROVED_DIR,
    LATITUDE_BANDS,
    MAX_LS_POINTS,
    MCD_VARIABLES,
)
from database.engine import async_session_maker
from database.models import UploadRecord
from services.overview_upload_contract import DIRECT_SAMPLES_PER_SOL, normalize_overview_upload_dataset
from services.ozone_units import normalize_ozone_column_units

logger = logging.getLogger("astraatmos.user_data")


class UserDataService:
    """按需读取用户上传的 .nc 文件并生成可视化数据"""

    def __init__(self, mcd_cache_service=None):
        # LRU 缓存：key = f"data_{upload_id}", value = 解析后的数据字典
        self._cache: LRUCache = LRUCache(maxsize=8)
        # 审核通过的数据集文件路径索引：{ record_id: Path }
        self._approved_index: dict[int, Path] = {}
        self.mcd_cache_service = mcd_cache_service
        self._scan_approved()

    # ─── 索引管理 ──────────────────────────────────────────────────────────────

    def _scan_approved(self) -> None:
        """扫描 approved/ 目录，建立已通过数据集的路径索引"""
        self._approved_index.clear()
        if not APPROVED_DIR.exists():
            return
        for subdir in APPROVED_DIR.iterdir():
            if subdir.is_dir() and subdir.name.isdigit():
                nc_file = subdir / "original.nc"
                if nc_file.exists():
                    self._approved_index[int(subdir.name)] = nc_file
        logger.info("已扫描 approved 目录: %d 个数据集", len(self._approved_index))

    def reload_approved(self) -> None:
        """热更新：重新扫描 approved 目录（审核通过后调用）"""
        # 清除 approved 数据集的缓存条目
        old_ids = set(self._approved_index.keys())
        self._scan_approved()
        new_ids = set(self._approved_index.keys())
        # 只清除有变化的条目
        for rid in old_ids.symmetric_difference(new_ids):
            for key in (f"data_{rid}", f"mcd_hourly_{rid}"):
                if key in self._cache:
                    del self._cache[key]
        logger.info("approved 数据索引已刷新")

    def get_approved_list(self) -> list[int]:
        """返回所有已通过数据集的 record_id 列表"""
        return sorted(self._approved_index.keys())

    # ─── 文件路径解析 ──────────────────────────────────────────────────────────

    async def _get_record_file_info(self, upload_id: int) -> Optional[dict]:
        async with async_session_maker() as db:
            record = await db.get(UploadRecord, upload_id)
            if not record:
                return None
            return {
                "file_path": record.file_path,
                "data_type": record.data_type,
                "status": record.status,
            }

    async def _get_file_path(self, upload_id: int) -> Optional[str]:
        """
        从 approved 索引或数据库记录中定位 .nc 文件。
        优先检查 approved 目录（已审核通过的在此处），
        然后回退到数据库中记录的原始 file_path。
        """
        # 1. approved 索引
        if upload_id in self._approved_index:
            p = self._approved_index[upload_id]
            if p.exists():
                return str(p)

        # 2. 数据库记录的路径
        info = await self._get_record_file_info(upload_id)
        if info and info.get("data_type") == "mcd" and self.mcd_cache_service is not None:
            ready_path = await self.mcd_cache_service.get_ready_artifact_path(upload_id, "mcd_overview")
            if ready_path is not None:
                return str(ready_path)
            raise ValueError(f"MCD upload {upload_id} overview cache is not ready")

        if info and info.get("file_path") and Path(info["file_path"]).exists():
            return info["file_path"]
        return None

    # ─── NC 文件解析 ───────────────────────────────────────────────────────────

    def _load_nc_file(self, file_path: str) -> dict:
        """
        读取一个 .nc 文件，返回数据总览可直接消费的标准化数据字典。
        """
        ds = xr.open_dataset(file_path, decode_times=False)
        try:
            return normalize_overview_upload_dataset(ds, filename=Path(file_path).name)
        finally:
            ds.close()

    @staticmethod
    def _find_dataset_name(ds: xr.Dataset, aliases: tuple[str, ...]) -> str | None:
        names = list(ds.data_vars) + list(ds.coords)
        lowered = {name.lower(): name for name in names}
        for alias in aliases:
            if alias in names:
                return alias
            found = lowered.get(alias.lower())
            if found:
                return found
        return None

    def _load_mcd_hourly_file(self, file_path: str) -> dict:
        """
        读取用户原始 MCD 3h 文件，只提取昼夜变化所需的小时臭氧数据。
        """
        ds = xr.open_dataset(file_path, decode_times=False)
        try:
            ozone_name = self._find_dataset_name(ds, ("O3COL", "o3col"))
            ls_name = self._find_dataset_name(ds, ("LS", "Ls", "ls"))
            lat_name = self._find_dataset_name(ds, ("lat", "latitude"))
            lon_name = self._find_dataset_name(ds, ("lon", "longitude"))
            if not ozone_name or not ls_name or not lat_name or not lon_name:
                raise ValueError("MCD hourly source missing O3COL/LS/lat/lon fields")

            ozone = normalize_ozone_column_units(
                ds[ozone_name].values,
                ds[ozone_name].attrs.get("units"),
                allow_mcd_legacy_heuristic=True,
            )
            ls_values = np.asarray(ds[ls_name].values, dtype=np.float32).reshape(-1)
            if ozone.ndim != 3 or ls_values.ndim != 1:
                raise ValueError(f"MCD hourly source must be time/lat/lon, got {ozone.shape}")

            sol_count = min(int(ozone.shape[0]), int(ls_values.shape[0])) // DIRECT_SAMPLES_PER_SOL
            if sol_count < 1:
                raise ValueError("MCD hourly source has no complete 3h sol groups")

            trim = sol_count * DIRECT_SAMPLES_PER_SOL
            hourly = np.asarray(ozone[:trim], dtype=np.float32).reshape(
                sol_count,
                DIRECT_SAMPLES_PER_SOL,
                ozone.shape[1],
                ozone.shape[2],
            )
            ls_sol = np.nanmean(
                ls_values[:trim].reshape(sol_count, DIRECT_SAMPLES_PER_SOL),
                axis=1,
            ).astype(np.float32)
            order = np.argsort(ls_sol)

            return {
                "data_type": "mcd",
                "ls": ls_sol[order],
                "lat": np.asarray(ds[lat_name].values, dtype=np.float32).reshape(-1),
                "lon": np.asarray(ds[lon_name].values, dtype=np.float32).reshape(-1),
                "O3COL_hourly": hourly[order],
                "raw_3h_source_file": str(file_path),
            }
        finally:
            ds.close()

    async def _get_mcd_hourly_file_path(self, upload_id: int) -> Optional[str]:
        if upload_id in self._approved_index:
            p = self._approved_index[upload_id]
            if p.exists():
                return str(p)

        if self.mcd_cache_service is not None:
            ready_path = await self.mcd_cache_service.get_ready_artifact_path(upload_id, "mcd_3h")
            if ready_path is not None and Path(ready_path).exists():
                return str(ready_path)

        info = await self._get_record_file_info(upload_id)
        if info and info.get("data_type") == "mcd" and info.get("file_path") and Path(info["file_path"]).exists():
            return info["file_path"]
        return None

    async def _get_data(self, upload_id: int) -> dict:
        """获取（并缓存）指定 upload_id 的解析数据"""
        cache_key = f"data_{upload_id}"
        if cache_key not in self._cache:
            file_path = await self._get_file_path(upload_id)
            if not file_path:
                raise ValueError(f"找不到上传记录 {upload_id} 的数据文件")
            self._cache[cache_key] = await asyncio.to_thread(self._load_nc_file, file_path)
        return self._cache[cache_key]

    # ─── 公共接口 ──────────────────────────────────────────────────────────────

    async def get_loaded_dataset(self, upload_id: int) -> dict:
        """Return the cached parsed payload for a raw uploaded dataset."""
        return await self._get_data(upload_id)

    async def get_loaded_mcd_hourly_dataset(self, upload_id: int) -> dict:
        """Return hourly O3COL data from the original uploaded MCD 3h source."""
        cache_key = f"mcd_hourly_{upload_id}"
        if cache_key not in self._cache:
            file_path = await self._get_mcd_hourly_file_path(upload_id)
            if not file_path:
                raise ValueError(f"找不到上传记录 {upload_id} 的 MCD 3h 原始数据文件")
            self._cache[cache_key] = await asyncio.to_thread(self._load_mcd_hourly_file, file_path)
        return self._cache[cache_key]

    async def get_dataset_summary(self, upload_id: int) -> dict:
        """获取数据集的基本信息摘要"""
        data = await self._get_data(upload_id)
        return {
            "upload_id": upload_id,
            "data_type": data.get("data_type", "unknown"),
            "has_ozone": "o3col" in data,
            "has_mcd_vars": [v for v in MCD_VARIABLES if v in data],
            "lat_points": int(len(data["lat"])) if data.get("lat") is not None else 0,
            "lon_points": int(len(data["lon"])) if data.get("lon") is not None else 0,
            "ls_range": (
                [float(data["ls"][0]), float(data["ls"][-1])]
                if data.get("ls") is not None else None
            ),
            "ls_points": int(len(data["ls"])) if data.get("ls") is not None else 0,
        }

    async def get_globe_data(self, upload_id: int, ls: float) -> dict:
        """获取 3D 点云（与 AnalysisService.get_globe_data 格式一致）"""
        data = await self._get_data(upload_id)

        if "o3col" not in data:
            raise ValueError("该数据集不包含臭氧列浓度数据")

        ls_arr = data["ls"]
        idx = int(np.argmin(np.abs(ls_arr - ls)))
        field = data["o3col"][idx]

        points = []
        for i, lat_v in enumerate(data["lat"]):
            for j, lon_v in enumerate(data["lon"]):
                val = float(field[i, j])
                if not np.isnan(val):
                    points.append({
                        "lat": float(lat_v),
                        "lng": float(lon_v) if lon_v <= 180 else float(lon_v - 360),
                        "val": val,
                    })

        valid_vals = field[~np.isnan(field)]
        return {
            "points": points,
            "minVal": float(np.nanmin(valid_vals)) if len(valid_vals) > 0 else 0.0,
            "maxVal": float(np.nanmax(valid_vals)) if len(valid_vals) > 0 else 1.0,
            "ls": float(ls_arr[idx]),
            "mars_year": None,
        }

    async def get_seasonal_heatmap(self, upload_id: int, variable: str = "o3col") -> dict:
        """获取 Ls-纬度热力图（与 AnalysisService 格式一致）"""
        data = await self._get_data(upload_id)

        if variable not in data:
            raise ValueError(f"该数据集不包含变量 '{variable}'")

        data_3d = data[variable]
        ls_arr = data["ls"]
        lat_arr = data["lat"]

        zonal_mean = np.nanmean(data_3d, axis=2)

        step = max(1, len(ls_arr) // MAX_LS_POINTS)
        ls_ds = ls_arr[::step]
        zm_ds = zonal_mean[::step]

        return {
            "x": [float(v) for v in ls_ds],
            "y": [float(v) for v in lat_arr],
            "z": [
                [float(zm_ds[t, lat_i]) for t in range(len(ls_ds))]
                for lat_i in range(len(lat_arr))
            ],
            "min": float(np.nanmin(zonal_mean)),
            "max": float(np.nanmax(zonal_mean)),
            "variable": variable,
        }

    async def get_seasonal_bands(self, upload_id: int) -> dict:
        """获取纬度带折线图（与 AnalysisService 格式一致）"""
        data = await self._get_data(upload_id)

        if "o3col" not in data:
            raise ValueError("该数据集不包含臭氧数据")

        o3 = data["o3col"]
        ls_arr = data["ls"]
        lat_arr = data["lat"]

        step = max(1, len(ls_arr) // MAX_LS_POINTS)
        ls_ds = ls_arr[::step]
        o3_ds = o3[::step]

        bands = []
        for band_def in LATITUDE_BANDS:
            mask = (lat_arr >= band_def["lat_min"]) & (lat_arr <= band_def["lat_max"])
            if mask.sum() == 0:
                continue
            band_mean = np.nanmean(o3_ds[:, mask, :], axis=(1, 2))
            bands.append({
                "name": band_def["name"],
                "values": [float(v) for v in band_mean],
            })

        return {
            "ls": [float(v) for v in ls_ds],
            "bands": bands,
        }
