from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sqlalchemy import select

from config import MAX_TRAINING_WEIGHT_SIZE_MB, TRAINING_WEIGHTS_DIR
from database.engine import async_session_maker
from database.models import TrainingWeightFile
from services.earth_training_contract import EARTH_ARTIFACT_SCHEMA, EARTH_3HOURLY_ARTIFACT_SCHEMA


class TrainingWeightService:
    def __init__(self, storage_root: Path | None = None, sessionmaker=None):
        self.storage_root = Path(storage_root or TRAINING_WEIGHTS_DIR)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.sessionmaker = sessionmaker or async_session_maker

    async def create_from_upload(
        self,
        user_id: int,
        original_filename: str,
        content: bytes,
    ) -> TrainingWeightFile:
        suffix = Path(original_filename or "").suffix.lower()
        if suffix not in {".pth", ".pt"}:
            raise ValueError("Training weight file must use a .pth or .pt filename")

        max_size_bytes = MAX_TRAINING_WEIGHT_SIZE_MB * 1024 * 1024
        if len(content) > max_size_bytes:
            raise ValueError(
                f"Training weight file exceeds {MAX_TRAINING_WEIGHT_SIZE_MB} MB size limit"
            )

        weight_id = str(uuid.uuid4())
        safe_name = self._safe_filename(original_filename)
        user_dir = self.storage_root / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        storage_path = user_dir / f"{weight_id}_{safe_name}"
        storage_path.write_bytes(content)

        validation_report = self._validate_weight_file(storage_path)
        record = TrainingWeightFile(
            id=weight_id,
            user_id=user_id,
            original_filename=original_filename,
            storage_path=str(storage_path),
            content_hash=hashlib.sha256(content).hexdigest(),
            file_size=len(content),
            status="ready" if validation_report["ok"] else "invalid",
            validation_report=json.dumps(validation_report, ensure_ascii=False),
        )

        async with self.sessionmaker() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def list_user_weights(self, user_id: int) -> list[TrainingWeightFile]:
        async with self.sessionmaker() as session:
            result = await session.execute(
                select(TrainingWeightFile)
                .where(
                    TrainingWeightFile.user_id == user_id,
                    TrainingWeightFile.deleted_at.is_(None),
                )
                .order_by(TrainingWeightFile.created_at.desc(), TrainingWeightFile.id.desc())
            )
            return list(result.scalars().all())

    async def get_weight_for_user(self, weight_id: str, user_id: int) -> TrainingWeightFile:
        async with self.sessionmaker() as session:
            return await self._get_weight_for_user_in_session(session, weight_id, user_id)

    async def soft_delete_weight(self, weight_id: str, user_id: int) -> None:
        async with self.sessionmaker() as session:
            record = await self._get_weight_for_user_in_session(session, weight_id, user_id)
            record.deleted_at = datetime.now(timezone.utc)
            await session.commit()

    @staticmethod
    async def _get_weight_for_user_in_session(session: Any, weight_id: str, user_id: int) -> TrainingWeightFile:
        result = await session.execute(
            select(TrainingWeightFile).where(TrainingWeightFile.id == weight_id)
        )
        record = result.scalar_one_or_none()
        if record is None or record.deleted_at is not None:
            raise FileNotFoundError("Training weight file not found")
        if record.user_id != user_id:
            raise PermissionError("No permission to access this training weight")
        return record

    @staticmethod
    def _safe_filename(original_filename: str) -> str:
        name = Path(original_filename or "weights.pth").name or "weights.pth"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".pth", ".pt"}:
            safe_name = f"{safe_name}.pth"
        return safe_name

    @staticmethod
    def _validate_weight_file(path: Path) -> dict[str, Any]:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            return {
                "ok": False,
                "errors": [f"Could not load weight file: {exc}"],
                "warnings": [],
            }

        if not isinstance(loaded, dict):
            return {
                "ok": False,
                "errors": ["Weight file must contain a PyTorch state_dict dictionary"],
                "warnings": [],
            }
        # An Earth checkpoint is a *container*, not a bare state dict. It must not
        # be reported as a compatible Mars transfer weight, and its tensors must
        # not be extracted and passed off as one.
        if loaded.get("artifact_schema") in (EARTH_ARTIFACT_SCHEMA, EARTH_3HOURLY_ARTIFACT_SCHEMA):
            return {
                "ok": False,
                "errors": [
                    "This file is an Earth (MERRA-2) forecast checkpoint and is not "
                    "compatible with Mars transfer learning. Upload or select Mars weights instead."
                ],
                "warnings": [],
                "artifact_kind": "earth_forecast_checkpoint",
            }
        return {"ok": True, "errors": [], "warnings": [], "tensor_count": len(loaded)}
