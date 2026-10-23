import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.engine import Base  # noqa: E402
from database.models import McdCacheArtifact, McdCacheJob, UploadRecord, User  # noqa: E402
from services.mcd_cache_service import MCD_CACHE_SCHEMA_VERSION, McdCacheService  # noqa: E402


def test_mcd_cache_models_can_be_persisted(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'cache.db'}",
            connect_args={"check_same_thread": False},
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with sessionmaker() as session:
            user = User(email="cache@example.com", username="Cache", password_hash="x")
            session.add(user)
            await session.flush()
            upload = UploadRecord(
                user_id=user.id,
                filename="MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
                file_path=str(tmp_path / "original.nc"),
                file_size=100,
                data_type="mcd",
                mars_year=33,
                status="cache_building",
                file_hash="a" * 64,
            )
            session.add(upload)
            await session.flush()
            job = McdCacheJob(
                upload_id=upload.id,
                mars_year=33,
                job_type="mcd_overview",
                status="pending",
                source_hash=upload.file_hash,
                cache_version="v1",
            )
            artifact = McdCacheArtifact(
                upload_id=upload.id,
                mars_year=33,
                cache_type="mcd_3h",
                status="ready",
                source_hash=upload.file_hash,
                cache_version="v1",
                file_path=upload.file_path,
            )
            session.add_all([job, artifact])
            await session.commit()

        async with sessionmaker() as session:
            jobs = (await session.execute(select(McdCacheJob))).scalars().all()
            artifacts = (await session.execute(select(McdCacheArtifact))).scalars().all()
            assert jobs[0].job_type == "mcd_overview"
            assert artifacts[0].cache_type == "mcd_3h"

        await engine.dispose()

    asyncio.run(run())


def _write_raw_mcd(path: Path, mars_year: int = 33):
    shape = (16, 37, 72)
    ds = xr.Dataset(
        data_vars={
            "LS": (("time",), np.linspace(10.0, 20.0, shape[0], dtype=np.float32)),
            "O3COL": (("time", "lat", "lon"), np.ones(shape, dtype=np.float32)),
            "T": (("time", "lat", "lon"), np.full(shape, 180.0, dtype=np.float32)),
            "U": (("time", "lat", "lon"), np.full(shape, 5.0, dtype=np.float32)),
            "V": (("time", "lat", "lon"), np.full(shape, 2.0, dtype=np.float32)),
            "FSDS": (("time", "lat", "lon"), np.full(shape, 90.0, dtype=np.float32)),
            "PS": (("time", "lat", "lon"), np.full(shape, 6.0, dtype=np.float32)),
        },
        coords={
            "time": np.arange(shape[0], dtype=np.int32),
            "lat": np.linspace(90.0, -90.0, shape[1], dtype=np.float32),
            "lon": np.linspace(-180.0, 175.0, shape[2], dtype=np.float32),
        },
        attrs={"mars_year": mars_year},
    )
    ds.to_netcdf(path)
    ds.close()


def test_mcd_cache_service_creates_jobs_and_builds_ready_artifacts(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'cache-build.db'}",
            connect_args={"check_same_thread": False},
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        raw_path = tmp_path / "original.nc"
        _write_raw_mcd(raw_path)

        async with sessionmaker() as session:
            user = User(email="build@example.com", username="Build", password_hash="x")
            session.add(user)
            await session.flush()
            upload = UploadRecord(
                user_id=user.id,
                filename="MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
                file_path=str(raw_path),
                file_size=raw_path.stat().st_size,
                data_type="mcd",
                mars_year=33,
                status="cache_building",
                file_hash="b" * 64,
            )
            session.add(upload)
            await session.commit()
            upload_id = upload.id

        service = McdCacheService(sessionmaker=sessionmaker, cache_root=tmp_path / "mcd_cache")
        await service.enqueue_upload(upload_id)
        await service.run_next_pending_job()
        await service.run_next_pending_job()

        async with sessionmaker() as session:
            upload = await session.get(UploadRecord, upload_id)
            artifacts = (
                await session.execute(
                    select(McdCacheArtifact).where(McdCacheArtifact.upload_id == upload_id)
                )
            ).scalars().all()
            jobs = (
                await session.execute(select(McdCacheJob).where(McdCacheJob.upload_id == upload_id))
            ).scalars().all()

            assert upload.status == "valid"
            assert sorted(job.status for job in jobs) == ["completed", "completed"]
            by_type = {artifact.cache_type: artifact for artifact in artifacts}
            assert set(by_type) == {"mcd_3h", "mcd_overview"}
            assert by_type["mcd_3h"].file_path == str(raw_path)
            assert Path(by_type["mcd_overview"].file_path).is_file()
            assert by_type["mcd_overview"].cache_version == MCD_CACHE_SCHEMA_VERSION
            meta = json.loads(by_type["mcd_overview"].metadata_json)
            assert meta["mars_year"] == 33

        await engine.dispose()

    asyncio.run(run())


def test_mcd_cache_service_marks_missing_source_job_failed(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'cache-failed.db'}",
            connect_args={"check_same_thread": False},
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        missing_path = tmp_path / "missing.nc"
        async with sessionmaker() as session:
            user = User(email="failed@example.com", username="Failed", password_hash="x")
            session.add(user)
            await session.flush()
            upload = UploadRecord(
                user_id=user.id,
                filename="MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
                file_path=str(missing_path),
                file_size=100,
                data_type="mcd",
                mars_year=33,
                status="cache_building",
                file_hash="c" * 64,
            )
            session.add(upload)
            await session.flush()
            session.add(
                McdCacheJob(
                    upload_id=upload.id,
                    mars_year=33,
                    job_type="mcd_overview",
                    status="pending",
                    source_hash=upload.file_hash,
                    cache_version=MCD_CACHE_SCHEMA_VERSION,
                )
            )
            await session.commit()
            upload_id = upload.id

        service = McdCacheService(sessionmaker=sessionmaker, cache_root=tmp_path / "mcd_cache")
        assert await service.run_next_pending_job() is True

        async with sessionmaker() as session:
            upload = await session.get(UploadRecord, upload_id)
            job = (
                await session.execute(select(McdCacheJob).where(McdCacheJob.upload_id == upload_id))
            ).scalars().one()

            assert job.status == "failed"
            assert job.error and "missing.nc" in job.error
            assert upload.status == "cache_failed"
            assert upload.validation_message and "MCD cache build failed" in upload.validation_message

        await engine.dispose()

    asyncio.run(run())


def test_ready_artifact_path_uses_current_cache_version_and_source_hash(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'cache-version.db'}",
            connect_args={"check_same_thread": False},
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        current_path = tmp_path / "current.nc"
        stale_path = tmp_path / "stale.nc"
        _write_raw_mcd(current_path)
        _write_raw_mcd(stale_path)

        async with sessionmaker() as session:
            user = User(email="version@example.com", username="Version", password_hash="x")
            session.add(user)
            await session.flush()
            upload = UploadRecord(
                user_id=user.id,
                filename="MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
                file_path=str(current_path),
                file_size=current_path.stat().st_size,
                data_type="mcd",
                mars_year=33,
                status="valid",
                file_hash="d" * 64,
            )
            session.add(upload)
            await session.flush()
            session.add(
                McdCacheArtifact(
                    upload_id=upload.id,
                    mars_year=33,
                    cache_type="mcd_overview",
                    status="ready",
                    source_hash=upload.file_hash,
                    cache_version=MCD_CACHE_SCHEMA_VERSION,
                    file_path=str(current_path),
                )
            )
            session.add(
                McdCacheArtifact(
                    upload_id=upload.id,
                    mars_year=33,
                    cache_type="mcd_overview",
                    status="ready",
                    source_hash="e" * 64,
                    cache_version="old-version",
                    file_path=str(stale_path),
                )
            )
            await session.commit()
            upload_id = upload.id

        service = McdCacheService(sessionmaker=sessionmaker, cache_root=tmp_path / "mcd_cache")
        ready_path = await service.get_ready_artifact_path(upload_id, "mcd_overview")

        assert ready_path == current_path
        await engine.dispose()

    asyncio.run(run())


def test_delete_upload_cache_removes_metadata_and_cache_directory(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'cache-delete.db'}",
            connect_args={"check_same_thread": False},
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        source_path = tmp_path / "original.nc"
        _write_raw_mcd(source_path)
        cache_root = tmp_path / "mcd_cache"

        async with sessionmaker() as session:
            user = User(email="delete@example.com", username="Delete", password_hash="x")
            session.add(user)
            await session.flush()
            upload = UploadRecord(
                user_id=user.id,
                filename="MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
                file_path=str(source_path),
                file_size=source_path.stat().st_size,
                data_type="mcd",
                mars_year=33,
                status="valid",
                file_hash="f" * 64,
            )
            session.add(upload)
            await session.flush()
            session.add(
                McdCacheJob(
                    upload_id=upload.id,
                    mars_year=33,
                    job_type="mcd_overview",
                    status="completed",
                    source_hash=upload.file_hash,
                    cache_version=MCD_CACHE_SCHEMA_VERSION,
                )
            )
            session.add(
                McdCacheArtifact(
                    upload_id=upload.id,
                    mars_year=33,
                    cache_type="mcd_overview",
                    status="ready",
                    source_hash=upload.file_hash,
                    cache_version=MCD_CACHE_SCHEMA_VERSION,
                    file_path=str(source_path),
                )
            )
            await session.commit()
            upload_id = upload.id

        cache_dir = cache_root / "personal" / str(upload_id)
        cache_dir.mkdir(parents=True)
        (cache_dir / "artifact.nc").write_text("cached")

        service = McdCacheService(sessionmaker=sessionmaker, cache_root=cache_root)
        await service.delete_upload_cache(upload_id)

        async with sessionmaker() as session:
            jobs = (
                await session.execute(select(McdCacheJob).where(McdCacheJob.upload_id == upload_id))
            ).scalars().all()
            artifacts = (
                await session.execute(select(McdCacheArtifact).where(McdCacheArtifact.upload_id == upload_id))
            ).scalars().all()
            assert jobs == []
            assert artifacts == []
        assert not cache_dir.exists()

        await engine.dispose()

    asyncio.run(run())
