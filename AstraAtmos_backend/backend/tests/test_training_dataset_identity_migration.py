"""Legacy SQLite migration contract for dataset identity columns."""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.dataset_identity_migration import (  # noqa: E402
    DATASET_COLUMNS,
    DatasetIdentityMigrationError,
    migrate_dataset_identity,
)

# The historical table before dataset identity existed. It deliberately omits the
# five new columns, so create_all() on the ORM cannot stand in for this test.
LEGACY_SCHEMA = """
CREATE TABLE model_training_tasks (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    model_script VARCHAR(255) NOT NULL,
    model_source VARCHAR(20) NOT NULL DEFAULT 'official',
    uploaded_model_id VARCHAR(36),
    uploaded_model_version INTEGER,
    hyperparameters TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    log_file_path VARCHAR(500),
    output_model_path VARCHAR(500),
    custom_model_name VARCHAR(255),
    pid INTEGER,
    metrics TEXT,
    progress FLOAT DEFAULT 0.0,
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER DEFAULT 0,
    current_loss FLOAT,
    eta VARCHAR(50),
    loss_history TEXT
)
"""

# (name, stored hyperparameters JSON, expected dataset_id, expected status)
LEGACY_ROWS = [
    ("explicit-openmars", '{"training_dataset": "openmars_mcd", "epochs": 5}', "openmars_mcd", "legacy_inferred"),
    ("explicit-mcd", '{"training_dataset": "mcd_overview", "epochs": 5}', "mcd_overview", "legacy_inferred"),
    ("explicit-uppercase", '{"training_dataset": " MCD_Overview "}', "mcd_overview", "legacy_inferred"),
    ("missing-field", '{"epochs": 5}', "openmars_mcd", "legacy_inferred"),
    ("null-field", '{"training_dataset": null}', "openmars_mcd", "legacy_inferred"),
    ("blank-field", '{"training_dataset": "   "}', "openmars_mcd", "legacy_inferred"),
    ("empty-object", "{}", "openmars_mcd", "legacy_inferred"),
    ("broken-json", '{"training_dataset": "openmars_mcd"', None, "legacy_unknown"),
    ("json-array", '[1, 2, 3]', None, "legacy_unknown"),
    ("json-scalar", '7', None, "legacy_unknown"),
    ("unknown-id", '{"training_dataset": "missing"}', None, "legacy_unknown"),
    ("earth-id", '{"training_dataset": "earth_merra2_daily_v1"}', None, "legacy_unknown"),
    ("numeric-id", '{"training_dataset": 7}', None, "legacy_unknown"),
]

def make_engine(db_path):
    return create_async_engine(
        f"sqlite+aiosqlite:///{db_path}", poolclass=StaticPool, future=True
    )


def create_legacy_database(db_path, *, extra_columns=(), rows=LEGACY_ROWS):
    columns = "".join(f", {name} TEXT" for name in extra_columns)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(LEGACY_SCHEMA.rstrip()[:-1] + columns + ")")
        for index, (name, hyperparameters, _dataset_id, _status) in enumerate(rows, start=1):
            connection.execute(
                """
                INSERT INTO model_training_tasks
                    (user_id, model_script, model_source, hyperparameters, status,
                     start_time, output_model_path, custom_model_name, metrics, progress,
                     current_epoch, total_epochs, log_file_path, loss_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    3,
                    "demo3.py",
                    "uploaded" if index % 2 == 0 else "official",
                    hyperparameters,
                    "completed",
                    "2026-01-0%d 00:00:00" % min(index, 9),
                    f"D:/tmp/weights/{name}.pth",
                    name,
                    json.dumps({"rmse": 1.5, "note": name}),
                    100.0,
                    10,
                    10,
                    f"D:/tmp/logs/{name}.log",
                    json.dumps({"train": [1.0], "val": [2.0]}),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def read_rows(db_path, columns="*"):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"SELECT {columns} FROM model_training_tasks ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def table_columns(db_path):
    connection = sqlite3.connect(db_path)
    try:
        return {row[1] for row in connection.execute("PRAGMA table_info(model_training_tasks)")}
    finally:
        connection.close()


def strip_identity(rows):
    """Drop identity columns so pre/post migration field equality is meaningful."""
    return [
        {key: value for key, value in row.items() if not key.startswith("dataset_")}
        for row in rows
    ]


async def run_migration(db_path):
    engine = make_engine(db_path)
    try:
        async with engine.begin() as conn:
            return await migrate_dataset_identity(conn)
    finally:
        await engine.dispose()


def test_legacy_table_is_created_without_identity_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path)

    columns = table_columns(db_path)
    assert "hyperparameters" in columns
    assert not set(DATASET_COLUMNS).intersection(columns)


def test_migration_adds_columns_and_binds_each_legacy_source(tmp_path):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path)

    migrated = asyncio.run(run_migration(db_path))

    assert migrated == len(LEGACY_ROWS)
    assert set(DATASET_COLUMNS).issubset(table_columns(db_path))
    rows = read_rows(db_path, "id, custom_model_name, hyperparameters, dataset_id, "
                              "dataset_version, dataset_fingerprint, "
                              "dataset_identity_status, dataset_snapshot")
    by_name = {row["custom_model_name"]: row for row in rows}
    for name, hyperparameters, dataset_id, status in LEGACY_ROWS:
        row = by_name[name]
        assert row["hyperparameters"] == hyperparameters, name
        assert row["dataset_id"] == dataset_id, name
        assert row["dataset_version"] is None, name
        assert row["dataset_fingerprint"] is None, name
        assert row["dataset_identity_status"] == status, name
        snapshot = json.loads(row["dataset_snapshot"])
        assert snapshot["planet"] == "mars", name
        assert snapshot["version_status"] == "unknown", name
        assert snapshot["binding_basis"], name


def test_migration_uses_explicit_basis_for_named_sources(tmp_path):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path)
    asyncio.run(run_migration(db_path))

    snapshots = {
        row["custom_model_name"]: json.loads(row["dataset_snapshot"])
        for row in read_rows(db_path, "custom_model_name, dataset_snapshot")
    }
    assert snapshots["explicit-openmars"]["binding_basis"] == "explicit_training_dataset"
    assert snapshots["missing-field"]["binding_basis"] == "historical_default"
    assert snapshots["broken-json"]["binding_basis"] == "invalid_hyperparameters"
    assert snapshots["unknown-id"]["binding_basis"] == "unknown_training_dataset"


def test_migration_is_idempotent_and_does_not_rebind_existing_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path)

    assert asyncio.run(run_migration(db_path)) == len(LEGACY_ROWS)
    first_pass = read_rows(db_path, "id, dataset_id, dataset_identity_status, dataset_snapshot")
    assert asyncio.run(run_migration(db_path)) == 0
    second_pass = read_rows(db_path, "id, dataset_id, dataset_identity_status, dataset_snapshot")

    assert first_pass == second_pass


def test_partially_migrated_table_keeps_existing_bindings(tmp_path):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path, extra_columns=("dataset_id", "dataset_identity_status"))
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE model_training_tasks SET dataset_id='mcd_overview', "
            "dataset_identity_status='legacy_inferred' WHERE custom_model_name='unknown-id'"
        )
        connection.commit()
    finally:
        connection.close()

    migrated = asyncio.run(run_migration(db_path))

    assert migrated == len(LEGACY_ROWS) - 1
    rows = {
        row["custom_model_name"]: row
        for row in read_rows(db_path, "custom_model_name, dataset_id, dataset_version, "
                                      "dataset_fingerprint, dataset_identity_status, dataset_snapshot")
    }
    preserved = rows["unknown-id"]
    assert preserved["dataset_id"] == "mcd_overview"
    assert preserved["dataset_identity_status"] == "legacy_inferred"
    assert preserved["dataset_version"] is None
    assert preserved["dataset_fingerprint"] is None
    assert preserved["dataset_snapshot"] is None
    assert set(DATASET_COLUMNS).issubset(table_columns(db_path))


def test_broken_json_row_does_not_abort_the_backfill(tmp_path):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path)

    asyncio.run(run_migration(db_path))

    statuses = {
        row["custom_model_name"]: row["dataset_identity_status"]
        for row in read_rows(db_path, "custom_model_name, dataset_identity_status")
    }
    assert statuses["broken-json"] == "legacy_unknown"
    assert statuses["json-array"] == "legacy_unknown"
    assert statuses["json-scalar"] == "legacy_unknown"
    assert statuses["explicit-openmars"] == "legacy_inferred"


def test_failure_midway_rolls_back_and_can_be_retried(tmp_path):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path)
    before = read_rows(db_path)

    engine = make_engine(db_path)

    class Boom(RuntimeError):
        pass

    class FailingConnection:
        """Proxy that fails on the first identity UPDATE."""

        def __init__(self, conn):
            self._conn = conn

        async def execute(self, statement, parameters=None):
            if "UPDATE model_training_tasks" in str(statement):
                raise Boom("injected failure")
            if parameters is None:
                return await self._conn.execute(statement)
            return await self._conn.execute(statement, parameters)

    async def run_with_failure():
        async with engine.begin() as conn:
            await migrate_dataset_identity(FailingConnection(conn))

    try:
        with pytest.raises(DatasetIdentityMigrationError):
            asyncio.run(run_with_failure())
    finally:
        asyncio.run(engine.dispose())

    after_failure = read_rows(db_path)
    assert strip_identity(after_failure) == strip_identity(before)
    assert all(row["dataset_id"] is None for row in after_failure)
    assert all(row["dataset_snapshot"] is None for row in after_failure)

    # Retrying after the failure completes the migration.
    assert asyncio.run(run_migration(db_path)) == len(LEGACY_ROWS)
    assert all(row["dataset_identity_status"] for row in read_rows(db_path, "dataset_identity_status"))


def test_migration_preserves_unrelated_task_fields_and_weight_files(tmp_path):
    db_path = tmp_path / "legacy.db"
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    create_legacy_database(db_path)

    weight_files = {}
    for name, _hyperparameters, _dataset_id, _status in LEGACY_ROWS:
        path = weights_dir / f"{name}.pth"
        path.write_bytes(b"existing-weights-" + name.encode("utf-8"))
        weight_files[path] = path.read_bytes()

    before = {
        row["custom_model_name"]: {
            key: value for key, value in row.items() if not key.startswith("dataset_")
        }
        for row in read_rows(db_path)
    }
    asyncio.run(run_migration(db_path))
    after = {
        row["custom_model_name"]: {
            key: value for key, value in row.items() if not key.startswith("dataset_")
        }
        for row in read_rows(db_path)
    }

    assert after == before
    for path, content in weight_files.items():
        assert path.read_bytes() == content


def test_missing_legacy_table_fails_loudly(tmp_path):
    db_path = tmp_path / "empty.db"
    sqlite3.connect(db_path).close()

    with pytest.raises(DatasetIdentityMigrationError):
        asyncio.run(run_migration(db_path))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
