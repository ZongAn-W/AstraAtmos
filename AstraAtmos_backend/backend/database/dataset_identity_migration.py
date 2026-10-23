"""Add and backfill the dataset identity columns on ``model_training_tasks``.

Runs inside the caller's ``engine.begin()`` transaction so a partial migration is
rolled back as a unit. No files are moved, no weights are read, and historical
rows have their hyperparameters, status, weights, metrics and labels left
untouched.
"""

from __future__ import annotations

from sqlalchemy import text

from services.dataset_identity import infer_legacy_identity

TABLE_NAME = "model_training_tasks"

# Column names and SQL types are module constants; only row values are bound.
DATASET_COLUMNS = {
    "dataset_id": "VARCHAR(80)",
    "dataset_version": "VARCHAR(40)",
    "dataset_fingerprint": "VARCHAR(64)",
    "dataset_identity_status": "VARCHAR(32)",
    "dataset_snapshot": "TEXT",
}

_UNBOUND_PREDICATE = (
    "dataset_id IS NULL AND dataset_version IS NULL "
    "AND dataset_fingerprint IS NULL AND dataset_identity_status IS NULL "
    "AND dataset_snapshot IS NULL"
)

_SELECT_UNBOUND = text(
    f"SELECT id, hyperparameters FROM {TABLE_NAME} WHERE {_UNBOUND_PREDICATE}"
)

_UPDATE_IDENTITY = text(
    f"""
    UPDATE {TABLE_NAME}
    SET dataset_id=:dataset_id, dataset_version=:dataset_version,
        dataset_fingerprint=:dataset_fingerprint,
        dataset_identity_status=:dataset_identity_status,
        dataset_snapshot=:dataset_snapshot
    WHERE id=:task_id AND {_UNBOUND_PREDICATE}
    """
)


class DatasetIdentityMigrationError(RuntimeError):
    """Raised when the dataset identity migration cannot be completed."""


async def migrate_dataset_identity(conn) -> int:
    """Add the identity columns if needed and backfill legacy rows.

    Returns the number of rows considered for backfill. The caller owns the
    transaction; on failure nothing is committed.
    """
    try:
        result = await conn.execute(text(f"PRAGMA table_info({TABLE_NAME})"))
        existing = {row[1] for row in result.fetchall()}
        if not existing:
            raise RuntimeError(f"{TABLE_NAME} table is missing")
        for name, sql_type in DATASET_COLUMNS.items():
            if name not in existing:
                await conn.execute(
                    text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {name} {sql_type}")
                )
        rows = (await conn.execute(_SELECT_UNBOUND)).mappings().all()
        for row in rows:
            values = infer_legacy_identity(row["hyperparameters"])
            await conn.execute(_UPDATE_IDENTITY, {**values, "task_id": row["id"]})
        return len(rows)
    except Exception as exc:
        raise DatasetIdentityMigrationError("Dataset identity migration failed") from exc
