from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from config import MCD_CACHE_DIR
from database.engine import async_session_maker
from database.models import McdCacheArtifact, McdCacheJob, UploadRecord
from scripts.build_mcd_overview_dataset import build_overview_from_reference_dataset

logger = logging.getLogger("astraatmos.mcd_cache")

MCD_CACHE_SCHEMA_VERSION = "mcd-cache-v1"
REQUIRED_MCD_CACHE_TYPES = ("mcd_overview", "mcd_3h")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class McdCacheService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker = async_session_maker,
        cache_root: Path = MCD_CACHE_DIR,
    ):
        self.sessionmaker = sessionmaker
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    async def enqueue_upload(self, upload_id: int) -> list[McdCacheJob]:
        async with self.sessionmaker() as session:
            record = await session.get(UploadRecord, upload_id)
            if record is None:
                raise ValueError(f"Upload {upload_id} not found")
            if record.data_type != "mcd":
                return []
            if record.mars_year is None:
                raise ValueError(f"Upload {upload_id} has no Mars year")
            if not record.file_hash:
                raise ValueError(f"Upload {upload_id} has no source hash")

            created = []
            for job_type in REQUIRED_MCD_CACHE_TYPES:
                existing = (
                    await session.execute(
                        select(McdCacheJob)
                        .where(McdCacheJob.upload_id == record.id)
                        .where(McdCacheJob.job_type == job_type)
                        .where(McdCacheJob.source_hash == record.file_hash)
                        .where(McdCacheJob.cache_version == MCD_CACHE_SCHEMA_VERSION)
                    )
                ).scalars().first()
                if existing is not None:
                    continue
                job = McdCacheJob(
                    upload_id=record.id,
                    mars_year=int(record.mars_year),
                    job_type=job_type,
                    status="pending",
                    progress=0.0,
                    source_hash=record.file_hash,
                    cache_version=MCD_CACHE_SCHEMA_VERSION,
                )
                session.add(job)
                created.append(job)
            record.status = "cache_building"
            await session.commit()
            return created

    async def run_next_pending_job(self) -> bool:
        async with self.sessionmaker() as session:
            job = (
                await session.execute(
                    select(McdCacheJob)
                    .where(McdCacheJob.status == "pending")
                    .order_by(McdCacheJob.created_at.asc(), McdCacheJob.id.asc())
                    .limit(1)
                )
            ).scalars().first()
            if job is None:
                return False
            job.status = "running"
            job.started_at = _now()
            job.progress = 5.0
            await session.commit()
            job_id = job.id

        try:
            await self._build_job(job_id)
        except Exception as exc:
            logger.exception("MCD cache job failed: %s", job_id)
            await self._mark_job_failed(job_id, str(exc))
        return True

    async def _build_job(self, job_id: int) -> None:
        async with self.sessionmaker() as session:
            job = await session.get(McdCacheJob, job_id)
            if job is None:
                return
            record = await session.get(UploadRecord, job.upload_id)
            if record is None:
                raise ValueError(f"Upload {job.upload_id} not found")
            source_path = Path(record.file_path)
            if not source_path.is_file():
                raise FileNotFoundError(str(source_path))

        try:
            if job.job_type == "mcd_overview":
                artifact_path, metadata = await asyncio.to_thread(
                    self._build_overview, record, source_path
                )
            elif job.job_type == "mcd_3h":
                artifact_path, metadata = await asyncio.to_thread(
                    self._build_3h_pointer, record, source_path
                )
            else:
                raise ValueError(f"Unsupported MCD cache job type: {job.job_type}")
        except Exception as exc:
            await self._mark_job_failed(job_id, str(exc))
            return

        await self._mark_job_completed(job_id, record, artifact_path, metadata)

    def _artifact_dir(self, record: UploadRecord, cache_type: str) -> Path:
        return self.cache_root / "personal" / str(record.id) / MCD_CACHE_SCHEMA_VERSION / cache_type

    def _build_overview(self, record: UploadRecord, source_path: Path) -> tuple[Path, dict]:
        out_dir = self._artifact_dir(record, "mcd_overview")
        staging_dir = out_dir.with_name(f"{out_dir.name}.staging")
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        output_path = staging_dir / f"MCD_MY{record.mars_year}_overview.nc"
        build_overview_from_reference_dataset(source_path, output_path)

        with xr.open_dataset(output_path, decode_times=False) as ds:
            metadata = {
                "mars_year": int(record.mars_year),
                "ls_points": int(ds["Ls"].shape[0]),
                "lat_points": int(ds["lat"].shape[0]),
                "lon_points": int(ds["lon"].shape[0]),
            }

        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / output_path.name
        if final_path.exists():
            final_path.unlink()
        output_path.replace(final_path)
        shutil.rmtree(staging_dir, ignore_errors=True)
        return final_path, metadata

    def _build_3h_pointer(self, record: UploadRecord, source_path: Path) -> tuple[Path, dict]:
        with xr.open_dataset(source_path, decode_times=False) as ds:
            ls_name = next(name for name in ("LS", "Ls", "ls") if name in ds)
            metadata = {
                "mars_year": int(record.mars_year),
                "time_points": int(ds.sizes.get("time", 0)),
                "lat_points": int(ds.sizes.get("lat", 0)),
                "lon_points": int(ds.sizes.get("lon", 0)),
                "ls_min": float(ds[ls_name].min().values),
                "ls_max": float(ds[ls_name].max().values),
            }
        return source_path, metadata

    async def _mark_job_failed(self, job_id: int, error: str) -> None:
        async with self.sessionmaker() as session:
            job = await session.get(McdCacheJob, job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error
            job.progress = 100.0
            job.finished_at = _now()
            await session.flush()
            await self._refresh_upload_status(session, job.upload_id)
            await session.commit()

    async def _mark_job_completed(
        self,
        job_id: int,
        record: UploadRecord,
        artifact_path: Path,
        metadata: dict,
    ) -> None:
        async with self.sessionmaker() as session:
            job = await session.get(McdCacheJob, job_id)
            upload = await session.get(UploadRecord, record.id)
            if job is None or upload is None:
                return

            artifact = McdCacheArtifact(
                upload_id=upload.id,
                mars_year=int(upload.mars_year),
                cache_type=job.job_type,
                status="ready",
                source_hash=job.source_hash,
                cache_version=job.cache_version,
                file_path=str(artifact_path),
                metadata_json=json.dumps(metadata, ensure_ascii=False),
                activated_at=_now(),
            )
            session.add(artifact)
            job.status = "completed"
            job.progress = 100.0
            job.artifact_path = str(artifact_path)
            job.finished_at = _now()
            await session.flush()
            await self._refresh_upload_status(session, upload.id)
            await session.commit()

    async def _refresh_upload_status(self, session, upload_id: int) -> None:
        upload = await session.get(UploadRecord, upload_id)
        if upload is None:
            return
        artifacts = (
            await session.execute(
                select(McdCacheArtifact)
                .where(McdCacheArtifact.upload_id == upload_id)
                .where(McdCacheArtifact.status == "ready")
                .where(McdCacheArtifact.cache_version == MCD_CACHE_SCHEMA_VERSION)
            )
        ).scalars().all()
        ready_types = {artifact.cache_type for artifact in artifacts}
        if set(REQUIRED_MCD_CACHE_TYPES).issubset(ready_types):
            upload.status = "valid"
            return

        jobs = (
            await session.execute(
                select(McdCacheJob)
                .where(McdCacheJob.upload_id == upload_id)
                .where(McdCacheJob.cache_version == MCD_CACHE_SCHEMA_VERSION)
            )
        ).scalars().all()
        failed_jobs = [
            job
            for job in jobs
            if job.job_type in REQUIRED_MCD_CACHE_TYPES and job.status == "failed"
        ]
        if failed_jobs:
            upload.status = "cache_failed"
            first = failed_jobs[0]
            upload.validation_message = f"MCD cache build failed: {first.error or first.job_type}"
            return

        if any(job.job_type in REQUIRED_MCD_CACHE_TYPES and job.status in {"pending", "running"} for job in jobs):
            upload.status = "cache_building"

    async def get_cache_status(self, upload_id: int) -> dict:
        async with self.sessionmaker() as session:
            jobs = (
                await session.execute(
                    select(McdCacheJob)
                    .where(McdCacheJob.upload_id == upload_id)
                    .order_by(McdCacheJob.created_at.asc(), McdCacheJob.id.asc())
                )
            ).scalars().all()
            artifacts = (
                await session.execute(
                    select(McdCacheArtifact)
                    .where(McdCacheArtifact.upload_id == upload_id)
                    .order_by(McdCacheArtifact.created_at.asc(), McdCacheArtifact.id.asc())
                )
            ).scalars().all()
        return {
            "upload_id": int(upload_id),
            "jobs": [
                {
                    "job_type": job.job_type,
                    "status": job.status,
                    "progress": float(job.progress or 0.0),
                    "error": job.error,
                }
                for job in jobs
            ],
            "artifacts": [
                {
                    "cache_type": artifact.cache_type,
                    "status": artifact.status,
                    "cache_version": artifact.cache_version,
                    "metadata": json.loads(artifact.metadata_json or "{}"),
                }
                for artifact in artifacts
            ],
        }

    async def get_ready_artifact_path(self, upload_id: int, cache_type: str) -> Path | None:
        async with self.sessionmaker() as session:
            upload = await session.get(UploadRecord, upload_id)
            if upload is None:
                return None
            artifact = (
                await session.execute(
                    select(McdCacheArtifact)
                    .where(McdCacheArtifact.upload_id == upload_id)
                    .where(McdCacheArtifact.cache_type == cache_type)
                    .where(McdCacheArtifact.status == "ready")
                    .where(McdCacheArtifact.cache_version == MCD_CACHE_SCHEMA_VERSION)
                    .where(McdCacheArtifact.source_hash == upload.file_hash)
                    .order_by(McdCacheArtifact.activated_at.desc(), McdCacheArtifact.id.desc())
                    .limit(1)
                )
            ).scalars().first()
        if artifact is None:
            return None
        path = Path(artifact.file_path)
        return path if path.is_file() else None

    async def _delete_upload_cache_records(self, session, upload_id: int) -> None:
        await session.execute(
            delete(McdCacheArtifact).where(McdCacheArtifact.upload_id == upload_id)
        )
        await session.execute(
            delete(McdCacheJob).where(McdCacheJob.upload_id == upload_id)
        )

    async def delete_upload_cache(self, upload_id: int, session=None) -> None:
        if session is None:
            async with self.sessionmaker() as owned_session:
                await self._delete_upload_cache_records(owned_session, upload_id)
                await owned_session.commit()
        else:
            await self._delete_upload_cache_records(session, upload_id)

        cache_dir = self.cache_root / "personal" / str(upload_id)
        if cache_dir.exists():
            await asyncio.to_thread(shutil.rmtree, cache_dir, ignore_errors=True)
