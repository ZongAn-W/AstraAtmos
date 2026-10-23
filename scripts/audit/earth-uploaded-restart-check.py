"""Verify a prediction still works from the checkpoint alone, after a restart.

This is the step the HTTP acceptance script cannot do by itself: it must run against
a **freshly started** backend, and it may also remove or tamper with the user's
original model file to prove the checkpoint's embedded copy is what carries the run.

Usage, with the backend freshly restarted::

    python scripts/audit/earth-uploaded-restart-check.py --task-id 32 --model-id <uuid>
    python scripts/audit/earth-uploaded-restart-check.py --task-id 32 --model-id <uuid> --delete-source
    python scripts/audit/earth-uploaded-restart-check.py --task-id 32 --model-id <uuid> --tamper-source
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "AstraAtmos_backend" / "backend"
sys.path.insert(0, str(BACKEND))

BASE = "http://127.0.0.1:8000/api"
ORIGIN = "2021-07-08"


def issue_token() -> str:
    from auth.security import create_access_token

    return create_access_token({"sub": "3", "role": "user"})


def request(method: str, path: str, *, token: str, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            payload = response.read().decode("utf-8")
            return response.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8")
        try:
            return error.code, json.loads(payload)
        except ValueError:
            return error.code, {"detail": payload}


def storage_path_for(model_id: str) -> Path | None:
    from config import DATABASE_URL

    db = Path(DATABASE_URL.split("sqlite")[-1].lstrip(":+/"))
    if not db.is_absolute():
        db = BACKEND / db
    with sqlite3.connect(str(db)) as connection:
        row = connection.execute(
            "select storage_path from user_model_packages where id = ?", (model_id,)
        ).fetchone()
    return Path(row[0]) if row else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--tamper-source", action="store_true")
    args = parser.parse_args()

    token = issue_token()
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"{'OK  ' if ok else 'FAIL'} {label}" + (f" :: {detail}" if detail else ""), flush=True)
        if not ok:
            failures.append(label)

    path = storage_path_for(args.model_id)
    check("uploaded model file located", path is not None and path.is_file(), str(path))

    if args.delete_source and path and path.is_file():
        path.unlink()
        print(f"(deleted {path})", flush=True)
    elif args.tamper_source and path and path.is_file():
        path.write_bytes(path.read_bytes() + b"\n# tampered after training\n")
        print(f"(tampered {path})", flush=True)

    status, context = request(
        "GET", f"/earth/predict/context?training_task_id={args.task_id}", token=token
    )
    check("context loads from the checkpoint alone", status == 200, f"status={status}")
    if status == 200:
        identity = context.get("model") or {}
        check(
            "context still names the pinned uploaded model",
            identity.get("uploaded_model_id") == args.model_id,
            json.dumps(identity, ensure_ascii=False),
        )
        if args.delete_source or args.tamper_source:
            check(
                "context warns that the original file is gone or changed",
                bool(context.get("warnings")),
                json.dumps(context.get("warnings"), ensure_ascii=False),
            )

    status, result = request(
        "POST", "/earth/predict/run", token=token,
        body={"training_task_id": args.task_id, "forecast_origin": ORIGIN},
    )
    check("prediction succeeds after the restart", status == 200, f"status={status}")
    if status == 200:
        identity = result.get("model") or {}
        check(
            "prediction reports the uploaded model identity",
            identity.get("uploaded_model_id") == args.model_id,
            json.dumps(identity, ensure_ascii=False),
        )
        check("prediction unit is DU", result.get("target_unit") == "DU")
        check(
            "prediction dates follow the origin",
            result.get("target_dates") == ["2021-07-09", "2021-07-10", "2021-07-11"],
            json.dumps(result.get("target_dates")),
        )
        fields = {kind: len(result.get(kind) or []) for kind in ("prediction", "reference", "residual")}
        check("prediction/reference/residual all present", all(v == 3 for v in fields.values()),
              json.dumps(fields))
        metrics = (result.get("metrics") or {}).get("overall") or {}
        check(
            "metrics carry a finite DU rmse/mae",
            isinstance(metrics.get("rmse"), (int, float)) and isinstance(metrics.get("mae"), (int, float)),
            json.dumps(metrics),
        )
        day = (result.get("prediction") or [{}])[0]
        grid = day.get("field") or []
        check("grid is 36x72", len(grid) == 36 and bool(grid) and len(grid[0]) == 72,
              f"{len(grid)}x{len(grid[0]) if grid else 0}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
