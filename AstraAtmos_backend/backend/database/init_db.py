"""
Database bootstrap:
1. create tables if missing
2. patch legacy SQLite schemas
3. create default admin if users table is empty
"""

import logging
from pathlib import Path

from sqlalchemy import select, text

from config import DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD, TRAINING_RESULTS_DIR
from database.dataset_identity_migration import (
    DatasetIdentityMigrationError,
    migrate_dataset_identity,
)
from database.engine import Base, engine, async_session_maker
from database.models import (
    User,
    Notification,
    Feedback,
    UserModelPackage,
    TrainingWeightFile,
    ModelTrainingTask,
    PredictionAnalysisCache,
    DatasetLineageEvent,
    DatasetQualitySnapshot,
    McdCacheJob,
    McdCacheArtifact,
)  # noqa: F401
from services.training_paths import build_task_output_path

logger = logging.getLogger("astraatmos.db")


async def _patch_training_table_columns(conn) -> None:
    """Add missing columns for legacy SQLite databases."""
    result = await conn.execute(text("PRAGMA table_info(model_training_tasks)"))
    existing_columns = {row[1] for row in result.fetchall()}

    columns_to_add = [
        ("pid", "INTEGER"),
        ("custom_model_name", "VARCHAR(255)"),
        ("progress", "FLOAT DEFAULT 0.0"),
        ("current_epoch", "INTEGER DEFAULT 0"),
        ("total_epochs", "INTEGER DEFAULT 0"),
        ("current_loss", "FLOAT"),
        ("eta", "VARCHAR(50)"),
        ("loss_history", "TEXT"),
        ("model_source", "VARCHAR(20) DEFAULT 'official'"),
        ("uploaded_model_id", "VARCHAR(36)"),
        ("uploaded_model_version", "INTEGER"),
        ("queued_at", "DATETIME"),
        ("queue_position", "INTEGER"),
    ]

    for col_name, col_def in columns_to_add:
        if col_name not in existing_columns:
            logger.info("Adding missing column model_training_tasks.%s", col_name)
            await conn.execute(
                text(f"ALTER TABLE model_training_tasks ADD COLUMN {col_name} {col_def}")
            )


async def _patch_personal_source_build_state_columns(conn) -> None:
    """Add missing columns for personal source build states on legacy databases."""
    result = await conn.execute(text("PRAGMA table_info(personal_source_build_states)"))
    existing_columns = {row[1] for row in result.fetchall()}

    columns_to_add = [
        ("stage", "VARCHAR(40) DEFAULT 'idle'"),
        ("progress", "FLOAT DEFAULT 0.0"),
        ("stage_message", "TEXT"),
    ]

    for col_name, col_def in columns_to_add:
        if col_name not in existing_columns:
            logger.info("Adding missing column personal_source_build_states.%s", col_name)
            await conn.execute(
                text(f"ALTER TABLE personal_source_build_states ADD COLUMN {col_name} {col_def}")
            )


async def _patch_notification_table_columns(conn) -> None:
    """Add missing columns for legacy notification tables."""
    result = await conn.execute(text("PRAGMA table_info(notifications)"))
    existing_columns = {row[1] for row in result.fetchall()}
    if "related_training_task_id" not in existing_columns:
        logger.info("Adding missing column notifications.related_training_task_id")
        await conn.execute(
            text(
                "ALTER TABLE notifications "
                "ADD COLUMN related_training_task_id INTEGER"
            )
        )


async def migrate_training_task_output_paths(
    sessionmaker=async_session_maker,
    results_dir: Path = TRAINING_RESULTS_DIR,
) -> int:
    """Move historical weights and rewrite task paths to the English directory."""
    destination_root = Path(results_dir).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    async with sessionmaker() as session:
        result = await session.execute(select(ModelTrainingTask))
        migration_plan = []

        for task in result.scalars().all():
            if not task.output_model_path:
                continue

            source = Path(task.output_model_path).expanduser()
            destination = build_task_output_path(
                task.id,
                task.custom_model_name,
                results_dir=destination_root,
            ).resolve()
            if source.resolve() == destination:
                continue

            if destination.exists():
                raise FileExistsError(f"Training path migration conflict: {destination}")

            migration_plan.append((task, source, destination, source.is_file()))

        moved_files = []
        try:
            for task, source, destination, source_exists in migration_plan:
                if source_exists:
                    source.replace(destination)
                    moved_files.append((source, destination))
                task.output_model_path = str(destination)

            await session.commit()
        except Exception as migration_error:
            rollback_errors = []

            try:
                await session.rollback()
            except Exception as rollback_error:
                rollback_errors.append(
                    f"Database rollback failed: {rollback_error}"
                )

            for source, destination in reversed(moved_files):
                try:
                    if destination.is_file() and not source.exists():
                        source.parent.mkdir(parents=True, exist_ok=True)
                        destination.replace(source)
                    else:
                        rollback_errors.append(
                            f"Cannot restore {destination} to {source}"
                        )
                except Exception as rollback_error:
                    rollback_errors.append(str(rollback_error))

            if rollback_errors:
                details = "; ".join(rollback_errors)
                raise RuntimeError(
                    "Training path migration failed and file rollback was "
                    f"incomplete: {details}"
                ) from migration_error
            raise

    return len(migration_plan)


async def init_database() -> None:
    from auth.security import hash_password

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            try:
                await _patch_training_table_columns(conn)
            except Exception as exc:
                logger.warning("Could not auto-patch model_training_tasks schema: %s", exc)
            try:
                await _patch_personal_source_build_state_columns(conn)
            except Exception as exc:
                logger.warning("Could not auto-patch personal_source_build_states schema: %s", exc)
            try:
                await _patch_notification_table_columns(conn)
            except Exception as exc:
                logger.warning("Could not auto-patch notifications schema: %s", exc)
            # Dataset identity is required for every task query. It runs outside
            # the best-effort legacy patches above: a failure here must stop
            # startup instead of serving a database with an incomplete schema.
            migrated_identities = await migrate_dataset_identity(conn)
            if migrated_identities:
                logger.info(
                    "Backfilled dataset identity for %s historical training tasks",
                    migrated_identities,
                )

        logger.info("Database schema initialization complete")

        migrated_paths = await migrate_training_task_output_paths()
        if migrated_paths:
            logger.info("Migrated %s historical training output paths", migrated_paths)

        async with async_session_maker() as session:
            result = await session.execute(select(User).limit(1))
            if result.scalar_one_or_none() is None:
                admin = User(
                    email=DEFAULT_ADMIN_EMAIL,
                    username="Admin",
                    password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
                    role="admin",
                )
                session.add(admin)
                await session.commit()
                logger.info("Default admin created: %s", DEFAULT_ADMIN_EMAIL)
            else:
                logger.info("Users already exist; skip default admin creation")

    except DatasetIdentityMigrationError:
        logger.exception("Required dataset identity migration failed")
        raise

    except Exception as exc:
        logger.warning("Database initialization failed (startup continues): %s", exc)
