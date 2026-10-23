from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from config import MAX_USER_MODEL_SIZE_KB, USER_MODELS_DIR
from database.engine import async_session_maker
from database.models import UserModelPackage
from services.user_model_validator import (
    UserModelValidator, VALIDATION_TIMEOUT_CODE, validation_report_timed_out,
)


class UserModelService:
    def __init__(
        self,
        storage_root: Path | None = None,
        sessionmaker=None,
        validator: UserModelValidator | None = None,
    ):
        self.storage_root = Path(storage_root or USER_MODELS_DIR)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.sessionmaker = sessionmaker or async_session_maker
        self.validator = validator or UserModelValidator()

    async def create_from_source(
        self,
        user_id: int,
        original_filename: str,
        source: bytes,
    ) -> UserModelPackage:
        if Path(original_filename or "").suffix.lower() != ".py":
            raise ValueError("User model source must use a .py filename")

        max_size_bytes = MAX_USER_MODEL_SIZE_KB * 1024
        if len(source) > max_size_bytes:
            raise ValueError(
                f"User model file exceeds {MAX_USER_MODEL_SIZE_KB} KB size limit"
            )

        content_hash = hashlib.sha256(source).hexdigest()
        safe_name = self._safe_filename(original_filename)
        result = await asyncio.to_thread(self._validate_source_bytes, source, safe_name)

        package_id = str(uuid.uuid4())
        user_dir = self.storage_root / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        storage_path = user_dir / f"{package_id}_{safe_name}.source"
        storage_path.write_bytes(source)

        display_name = result.display_name or Path(original_filename).stem
        version = await self._next_version(user_id, display_name)

        package = UserModelPackage(
            id=package_id,
            user_id=user_id,
            display_name=display_name,
            version=version,
            original_filename=original_filename,
            storage_path=str(storage_path),
            content_hash=content_hash,
            param_schema=json.dumps(result.param_schema, ensure_ascii=False),
            description=result.description,
            validation_status="valid" if result.ok else "invalid",
            validation_report=json.dumps(result.report_dict(), ensure_ascii=False),
        )

        async with self.sessionmaker() as session:
            session.add(package)
            await session.commit()
            await session.refresh(package)
            return package

    async def list_user_packages(self, user_id: int) -> list[UserModelPackage]:
        async with self.sessionmaker() as session:
            result = await session.execute(
                select(UserModelPackage)
                .where(
                    UserModelPackage.user_id == user_id,
                    UserModelPackage.deleted_at.is_(None),
                )
                .order_by(UserModelPackage.created_at.desc(), UserModelPackage.id.desc())
            )
            return list(result.scalars().all())

    async def get_package_for_user(
        self,
        package_id: str,
        user_id: int,
    ) -> UserModelPackage:
        async with self.sessionmaker() as session:
            package = await self._get_package_for_user_in_session(
                session,
                package_id,
                user_id,
            )
            return package

    async def rename_package(
        self, package_id: str, user_id: int, display_name: str
    ) -> UserModelPackage:
        name = display_name.strip()
        if not name or len(name) > 120:
            raise ValueError("Model name must contain 1 to 120 characters")

        async with self.sessionmaker() as session:
            package = await self._get_package_for_user_in_session(session, package_id, user_id)
            if package.display_name != name:
                package.display_name = name
                package.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await session.refresh(package)
            return package

    async def get_source_for_download(
        self, package_id: str, user_id: int
    ) -> tuple[Path, str]:
        package = await self.get_package_for_user(package_id, user_id)
        source_path = Path(package.storage_path).resolve()
        if not source_path.is_relative_to(self.storage_root.resolve()) or not source_path.is_file():
            raise FileNotFoundError("Uploaded model source file not found")
        # Keep Unicode names, but never put an uploaded path or control characters
        # into the attachment filename. The stored .source suffix is private.
        filename = Path(package.original_filename.replace("\\", "/")).name
        filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)
        if Path(filename).suffix.lower() != ".py":
            filename = "model.py"
        return source_path, filename

    async def get_earth_compatibility(
        self, package_id: str, user_id: int, *, dataset_id: str = "earth_merra2"
    ) -> dict:
        """Report whether a stored package may be trained on Earth data.

        The verdict is read from the package's own validation report, which was
        produced by the shared upload validator (including its Earth dry-run). No
        dry-run is repeated per request, so this stays cheap; callers that need a
        fresh proof can use ``revalidate`` first.
        """
        if dataset_id not in {"earth_merra2", "earth_merra2_daily_v1", "earth_merra2_daily_v2", "earth_merra2_3hourly_v1", "earth_merra2_3hourly_5deg_v1"}:
            raise ValueError("Unknown Earth dataset_id")
        package = await self.get_package_for_user(package_id, user_id)
        return self.earth_compatibility_for_package(package, dataset_id=dataset_id)

    @staticmethod
    def earth_compatibility_for_package(package: UserModelPackage, *, dataset_id: str) -> dict:
        """Read the upload proof and verify the source without executing the model."""
        try:
            report = json.loads(package.validation_report or "{}")
        except (TypeError, ValueError):
            report = {}
        if not isinstance(report, dict):
            report = {}
        if dataset_id in ("earth_merra2_3hourly_v1", "earth_merra2_3hourly_5deg_v1"):
            verdicts = report.get("earth_datasets")
            earth = verdicts.get(dataset_id) if isinstance(verdicts, dict) else None
        else:
            earth = report.get("earth")
        earth = earth if isinstance(earth, dict) else {}
        datasets = report.get("datasets") if isinstance(report.get("datasets"), dict) else {}
        declaration = datasets.get(dataset_id) or {}
        available = os.path.isfile(package.storage_path or "")
        compatible = earth.get("compatible") is True and available and package.validation_status == "valid" and report.get("ok") is True
        code = earth.get("code")
        status = "available" if compatible else ("unavailable" if earth else "unknown")
        if dataset_id in ("earth_merra2_3hourly_v1", "earth_merra2_3hourly_5deg_v1") and earth:
            from training_backbones.earth_3hourly_uploaded_contract import FULL_GRID_CONTRACT_SCHEMA, EVAL_BATCH_POLICY, contract_profile
            try:
                execution = contract_profile(earth.get('contract_schema'))
            except ValueError:
                execution = None
            expected_horizon = earth.get('horizon', 24)
            if (execution is None or earth.get("dataset_id") != dataset_id
                    or earth.get('contract_schema') != declaration.get('schema')
                    or (earth.get("compatible") is True and earth.get("eval_batch_policy") != EVAL_BATCH_POLICY)
                    or earth.get("status") == "unknown"
                    or (earth.get("compatible") is True and (earth.get("status") != "available"
                        or earth.get("output_shape") != [2, expected_horizon, 1, *(execution or {}).get('shape', ())]))
                    or (earth.get('contract_schema') in (FULL_GRID_CONTRACT_SCHEMA, 'astraatmos_earth_3hourly_5deg_model_v1')
                        and (earth.get('validation_scope') != 'declared_channels_probe_windows'
                             or earth.get('batch_size') != 1
                             or earth.get('window') not in declaration.get('window', [])
                             or earth.get('horizon') not in declaration.get('horizon', [])))):
                compatible, status, code = False, "unknown", "uploaded_model_compatibility_unknown"
        reasons = list(earth.get("errors") or [])
        if not available:
            compatible = False
            reasons = [
                "the uploaded model file is missing; re-upload or revalidate the model"
            ] + reasons
            status, code = "unavailable", "uploaded_model_missing"
        elif not earth:
            reasons = [
                "this model has no Earth compatibility result yet; revalidate it to check"
            ]
            code = "uploaded_model_compatibility_unknown"
            if package.validation_status == "invalid":
                status, code = (
                    ("unknown", VALIDATION_TIMEOUT_CODE) if validation_report_timed_out(report)
                    else ("unavailable", "uploaded_model_contract_invalid")
                )
                reasons = list(report.get("errors") or reasons)
        if available and hashlib.sha256(Path(package.storage_path).read_bytes()).hexdigest() != package.content_hash:
            compatible, status, code = False, "unavailable", "uploaded_model_tampered"
            reasons = ["The uploaded source no longer matches the verified digest"]
        if not compatible and not reasons:
            reasons = ["Revalidate this model for the selected dataset"]
        warnings = list(report.get("warnings") or [])
        return {
            "package_id": package.id,
            "display_name": package.display_name,
            "version": package.version,
            "validation_status": package.validation_status,
            "source_available": available,
            "compatible": compatible,
            "reasons": reasons,
            "warnings": warnings,
            "datasets": datasets,
            "dataset_id": dataset_id,
            "status": status,
            "code": code,
            "contract_schema": earth.get("contract_schema"),
            "eval_batch_policy": earth.get("eval_batch_policy"),
            "output_shape": earth.get("output_shape"),
            "default_batch_size": 1 if earth.get('contract_schema') in ('astraatmos_earth_3hourly_fullgrid_model_v2', 'astraatmos_earth_3hourly_5deg_model_v1') else 8,
            "max_batch_size": 64,
            "spatial_input_shape": declaration.get('spatial_tile_shape') if dataset_id in ('earth_merra2_3hourly_v1', 'earth_merra2_3hourly_5deg_v1') else None,
        }

    async def revalidate_package(self, package_id: str, user_id: int) -> UserModelPackage:
        async with self.sessionmaker() as session:
            package = await self._get_package_for_user_in_session(
                session,
                package_id,
                user_id,
            )
            source = Path(package.storage_path).read_bytes()
            # Validate under the name the user uploaded. The stored file keeps a
            # ``.source`` suffix so it is never importable directly, and re-using that
            # path here would make the validator reject the extension and report a
            # model as broken for a naming detail.
            result = await asyncio.to_thread(
                self._validate_source_bytes, source, self._upload_filename(package),
            )
            package.validation_status = "valid" if result.ok else "invalid"
            package.validation_report = json.dumps(result.report_dict(), ensure_ascii=False)
            package.param_schema = json.dumps(result.param_schema, ensure_ascii=False)
            package.description = result.description
            package.updated_at = datetime.now(timezone.utc)

            await session.commit()
            await session.refresh(package)
            return package

    async def soft_delete_package(self, package_id: str, user_id: int) -> None:
        async with self.sessionmaker() as session:
            package = await self._get_package_for_user_in_session(
                session,
                package_id,
                user_id,
            )
            now = datetime.now(timezone.utc)
            package.deleted_at = now
            package.updated_at = now
            await session.commit()

    async def _next_version(self, user_id: int, display_name: str) -> int:
        async with self.sessionmaker() as session:
            result = await session.execute(
                select(func.max(UserModelPackage.version)).where(
                    UserModelPackage.user_id == user_id,
                    UserModelPackage.display_name == display_name,
                )
            )
            max_version = result.scalar_one_or_none()
            return int(max_version or 0) + 1

    @staticmethod
    def _safe_filename(original_filename: str) -> str:
        name = Path(original_filename or "model.py").name
        if not name:
            name = "model.py"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        if not safe_name.lower().endswith(".py"):
            safe_name = f"{safe_name}.py"
        return safe_name

    @staticmethod
    def _upload_filename(package: Any) -> str:
        """The ``.py`` name to validate a stored package under.

        ``storage_path`` deliberately carries a ``.source`` suffix (and a uuid
        prefix) so an uploaded file is never importable by accident; validation must
        instead use the user's original filename, or the de-suffixed stored name.
        """
        original = getattr(package, "original_filename", None)
        if not original:
            stored_name = Path(str(getattr(package, "storage_path", ""))).name
            if stored_name.endswith(".source"):
                stored_name = stored_name[: -len(".source")]
            original = stored_name or "model.py"
        return UserModelService._safe_filename(original)

    def _validate_source_bytes(self, source: bytes, safe_name: str):
        with tempfile.TemporaryDirectory(prefix="astraatmos_user_model_") as temp_dir:
            validation_path = Path(temp_dir) / safe_name
            validation_path.write_bytes(source)
            return self.validator.validate_file(validation_path)

    @staticmethod
    async def _get_package_for_user_in_session(
        session,
        package_id: str,
        user_id: int,
    ) -> UserModelPackage:
        result = await session.execute(
            select(UserModelPackage).where(UserModelPackage.id == package_id)
        )
        package = result.scalar_one_or_none()
        if package is None or package.deleted_at is not None:
            raise FileNotFoundError("Uploaded model package not found")
        if package.user_id != user_id:
            raise PermissionError("No permission to access this uploaded model")
        return package
