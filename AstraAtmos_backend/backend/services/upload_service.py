"""
上传校验服务
-----------
封装 .nc 文件的保存、校验、元信息提取和数据库记录逻辑。
校验函数是同步的（xarray I/O），通过 run_in_executor 在线程池中调用。
"""

import asyncio
import hashlib
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import xarray as xr
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    MAX_UPLOAD_SIZE_MB,
    USER_UPLOADS_DIR,
)
from database.models import UploadRecord
from services.overview_upload_contract import (
    classify_overview_upload_dataset,
    validate_overview_upload_dataset,
)

logger = logging.getLogger("astraatmos.upload")


# ─── 数据类 ───────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    is_valid:   bool            = False
    data_type:  Optional[str]   = None   # 'openmars' | 'mcd' | 'nomad' | None
    mars_year:  Optional[int]   = None
    ls_start:   Optional[float] = None
    ls_end:     Optional[float] = None
    lat_points: int             = 0
    lon_points: int             = 0
    ls_points:  int             = 0
    variables:  list            = field(default_factory=list)
    warnings:   list            = field(default_factory=list)
    error:      Optional[str]   = None


# ─── 服务类 ───────────────────────────────────────────────────────────────────

class UploadService:
    def __init__(self, data_service=None):
        # data_service 预留，后续阶段可用于数据整合
        self.data_service = data_service

    # ── 公共入口 ──────────────────────────────────────────────────────────────

    async def process_upload(
        self,
        file: UploadFile,
        user_id: int,
        db: AsyncSession,
        mars_year: Optional[int] = None,
        description: Optional[str] = None,
    ) -> tuple[UploadRecord, ValidationResult]:
        """完整上传流程：保存 → 校验 → 入库，返回 (record, result)。"""
        upload_id = str(uuid.uuid4())
        upload_dir = USER_UPLOADS_DIR / str(user_id) / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / "original.nc"

        # 读取上传内容（UploadFile.read() 已是 async）
        content = await file.read()
        file_size = len(content)

        # 计算 SHA256 哈希，并检查当前用户是否已上传相同文件
        file_hash = hashlib.sha256(content).hexdigest()
        existing = (await db.execute(
            select(UploadRecord)
            .where(UploadRecord.user_id == user_id)
            .where(UploadRecord.file_hash == file_hash)
        )).scalars().first()
        if existing:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise ValueError(
                f"该文件已上传过（与 {existing.filename} 内容相同）"
            )

        # 写入磁盘（≤200 MB 的一次性写入，可接受在当前线程中执行）
        try:
            save_path.write_bytes(content)
        except OSError as exc:
            logger.error("写入上传文件失败: %s", exc)
            raise

        # 在线程池中运行同步的 xarray 校验，避免阻塞事件循环
        original_filename = file.filename or ""
        loop = asyncio.get_event_loop()
        result: ValidationResult = await loop.run_in_executor(
            None, self.validate_nc_file, save_path, file_size, original_filename
        )

        # 用户手动指定的火星年优先于自动提取结果
        if mars_year is not None:
            result.mars_year = mars_year

        # 入库
        warn_msg = "; ".join(result.warnings) if result.warnings else None
        record = UploadRecord(
            user_id=user_id,
            filename=file.filename or "upload.nc",
            file_path=str(save_path),
            file_size=file_size,
            data_type=result.data_type,
            mars_year=result.mars_year,
            ls_start=result.ls_start,
            ls_end=result.ls_end,
            status="valid" if result.is_valid else "invalid",
            validation_message=warn_msg if result.is_valid else result.error,
            file_hash=file_hash,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return record, result

    # ── 校验逻辑 ──────────────────────────────────────────────────────────────

    def validate_nc_file(self, file_path: Path, file_size: int, filename: str = "") -> ValidationResult:
        """
        校验 .nc 文件。同步函数，应通过 run_in_executor 调用。

        执行顺序：
          Step 1 — 大小 + NetCDF 格式
          Step 2 — 按数据总览上传契约检测并校验（MCD / OpenMARS / NOMAD）
          Step 3 — 提取网格、Ls、火星年等元信息
        """
        result = ValidationResult()

        # Step 1a — 大小检查
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            result.error = (
                f"文件大小 {file_size / 1024 / 1024:.1f} MB "
                f"超过限制 {MAX_UPLOAD_SIZE_MB} MB"
            )
            return result

        # Step 1b — NetCDF 格式检查
        # decode_times=False：跳过时间轴自动解码，避免非标准时间单位导致文件打不开
        try:
            ds = xr.open_dataset(file_path, engine="netcdf4", decode_times=False)
        except Exception:
            result.error = "文件格式无效，请确认上传的是标准 NetCDF（.nc）文件"
            return result

        try:
            return self._validate_dataset(ds, result, filename)
        finally:
            ds.close()

    def _validate_dataset(
        self, ds: xr.Dataset, result: ValidationResult, filename: str = ""
    ) -> ValidationResult:
        """在已打开的 Dataset 上执行数据总览上传契约校验。"""

        contract = validate_overview_upload_dataset(ds, filename, allow_ready_mcd=False)
        result.is_valid = contract.is_valid
        result.data_type = contract.data_type
        result.mars_year = contract.mars_year
        result.ls_start = contract.ls_start
        result.ls_end = contract.ls_end
        result.lat_points = contract.lat_points
        result.lon_points = contract.lon_points
        result.ls_points = contract.ls_points
        result.variables = contract.variables
        result.warnings = contract.warnings
        result.error = contract.error
        return result

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _detect_data_type(self, ds: xr.Dataset) -> Optional[str]:
        """返回 'openmars'、'mcd'、'nomad' 或 None。"""
        return classify_overview_upload_dataset(ds)
