"""Report persisted Earth training tasks, their identity and their artifacts.

Run from the backend directory so the backend package is importable:

    cd AstraAtmos_backend/backend
    & (Join-Path $env:CONDA_PREFIX 'python.exe') ../../scripts/audit/report-earth-tasks.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# The backend package root is the current directory when run as documented; add it
# explicitly so the script also works when invoked from the project root.
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "AstraAtmos_backend", "backend"))
for candidate in (os.getcwd(), BACKEND_DIR):
    if os.path.isdir(os.path.join(candidate, "database")) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from sqlalchemy import select  # noqa: E402

from database.engine import async_session_maker  # noqa: E402
from database.models import ModelTrainingTask  # noqa: E402


async def main() -> None:
    async with async_session_maker() as session:
        rows = (
            await session.execute(
                select(ModelTrainingTask)
                .where(ModelTrainingTask.model_script == "earth_daily.py")
                .order_by(ModelTrainingTask.id)
            )
        ).scalars().all()
        print(f"earth_daily.py tasks: {len(rows)}")
        for task in rows:
            metrics = json.loads(task.metrics) if task.metrics else {}
            test_overall = (metrics.get("splits", {}).get("test", {}) or {}).get("overall", {})
            path = task.output_model_path or ""
            exists = os.path.isfile(path)
            size = os.path.getsize(path) if exists else 0
            fingerprint = (task.dataset_fingerprint or "")[:12]
            print(
                f"  id={task.id} status={task.status} dataset={task.dataset_id}@{task.dataset_version} "
                f"identity={task.dataset_identity_status} fp={fingerprint} test_rmse={test_overall.get('rmse')} "
                f"artifact={exists} bytes={size}"
            )
            print(f"       output={path}")
            snapshot = json.loads(task.dataset_snapshot) if task.dataset_snapshot else {}
            print(
                f"       snapshot planet={snapshot.get('planet')} grid={snapshot.get('grid', {}).get('shape')} "
                f"window_counts_ok={'splits' in snapshot}"
            )


if __name__ == "__main__":
    asyncio.run(main())
