"""HTTP acceptance for training an uploaded model on Earth data, then predicting.

Runs the real web path against the running backend: register/login, upload a
user model, read its Earth compatibility verdict, start an Earth training task with
`model_source=uploaded`, wait for it, then predict from a historical origin and
confirm the response names the uploaded model and still returns DU
prediction/reference/residual plus metrics.

Usage (from the repository root, backend already served on 127.0.0.1:8000)::

    & (Join-Path $env:CONDA_PREFIX 'python.exe') scripts/audit/earth-uploaded-acceptance.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8000/api"
SAMPLE = Path(__file__).with_name("test_models") / "EarthConvBaseline.py"

RESULTS: list[tuple[str, bool, str]] = []

def check(label: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((label, ok, detail))
    print(f"{'OK  ' if ok else 'FAIL'} {label}" + (f" :: {detail}" if detail else ""), flush=True)
    return ok


def request(method: str, path: str, *, token=None, body=None, raw=None, content_type=None):
    url = f"{BASE}{path}"
    data = None
    headers = {}
    if raw is not None:
        data = raw
        headers["Content-Type"] = content_type or "application/octet-stream"
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = response.read().decode("utf-8")
            return response.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8")
        try:
            return error.code, json.loads(payload)
        except ValueError:
            return error.code, {"detail": payload}


def multipart(field: str, filename: str, content: bytes, boundary: str) -> bytes:
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: text/x-python\r\n\r\n"
    ).encode("utf-8")
    return head + content + f"\r\n--{boundary}--\r\n".encode("utf-8")


def main() -> int:
    # Registration needs an emailed verification code, so this script accepts a
    # bearer token issued by the running app instead of creating an account.
    token = ""
    for index, value in enumerate(sys.argv):
        if value == "--token" and index + 1 < len(sys.argv):
            token = sys.argv[index + 1]
        elif value.startswith("--token="):
            token = value.split("=", 1)[1]
    if not token:
        print("usage: earth-uploaded-acceptance.py --token <bearer token>", flush=True)
        print("issue one with:", flush=True)
        print("  python -c \"import sys;sys.path.insert(0,'.');"
              "from auth.security import create_access_token;"
              "print(create_access_token({'sub':'3','role':'user'}))\"", flush=True)
        return 2
    check("bearer token provided", True, f"{token[:12]}...")

    suffix = uuid.uuid4().hex[:8]
    source = SAMPLE.read_bytes()
    boundary = f"----astraatmos{suffix}"
    status, uploaded = request(
        "POST", "/user-models", token=token,
        raw=multipart("file", SAMPLE.name, source, boundary),
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    if status not in (200, 201) or not uploaded:
        check("upload model", False, f"status={status} body={uploaded}")
        return 1
    model_id = uploaded["id"]
    check("upload model", True, f"id={model_id} status={uploaded.get('validation_status')}")
    report = uploaded.get("validation_report") or {}
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except ValueError:
            report = {}
    earth = report.get("earth") if isinstance(report, dict) else None
    check(
        "upload validation reports the Earth block",
        bool(earth) and earth.get("compatible") is True,
        json.dumps(earth, ensure_ascii=False) if earth else f"keys={sorted(report)}",
    )

    status, compatibility = request("GET", f"/user-models/{model_id}/earth-compatibility", token=token)
    check(
        "GET /user-models/{id}/earth-compatibility",
        status == 200 and compatibility.get("compatible") is True,
        json.dumps(compatibility, ensure_ascii=False) if compatibility else f"status={status}",
    )

    # Revalidation must keep the model valid: the stored file carries a .source
    # suffix, so validating that path instead of the original .py name used to mark
    # every package invalid when the user re-checked it.
    status, revalidated = request("POST", f"/user-models/{model_id}/validate", token=token)
    revalidated_status = revalidated.get("validation_status") if isinstance(revalidated, dict) else None
    check(
        "POST /user-models/{id}/validate keeps the model valid",
        status == 200 and revalidated_status == "valid",
        f"status={status} validation_status={revalidated_status}",
    )

    status, listed = request("GET", "/datasets", token=token)
    earth_rows = [row for row in (listed or {}).get("items", listed if isinstance(listed, list) else [])
                  if isinstance(row, dict) and str(row.get("dataset_id", "")).startswith("earth_merra2")]
    check("datasets catalog exposes Earth", bool(earth_rows), f"{len(earth_rows)} row(s)")

    task_name = f"earth_uploaded_{suffix}"
    payload = {
        "model_script": "unified_training.py",
        "model_name": task_name,
        "model_source": "uploaded",
        "uploaded_model_id": model_id,
        "dataset_id": "earth_merra2_daily_v2",
        "data_source": "default",
        "hyperparameters": {
            "training_dataset": "earth_merra2_daily_v2",
            "model_source": "uploaded",
            "model_architecture": "uploaded",
            "selected_channels": ["U10M", "T2M"],
            "epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.001,
            "seed": 11,
            "custom_model_params": {"hidden_dim": 8, "dropout": 0.0},
        },
    }
    status, task = request("POST", "/training/start", token=token, body=payload)
    if status not in (200, 201) or not task:
        check("POST /training/start (uploaded Earth)", False, f"status={status} body={task}")
        return 1
    task_id = task.get("id")
    check("POST /training/start (uploaded Earth)", True, f"task_id={task_id}")

    # A model that never declared the Earth feed must be refused on the same
    # endpoint: "validated for Mars" is not "usable on Earth".
    mars_only = Path(__file__).with_name("test_models") / "MarsOnlyBaseline.py"
    mars_bytes = mars_only.read_bytes()
    boundary = f"----astraatmos{suffix}b"
    status, mars_uploaded = request(
        "POST", "/user-models", token=token,
        raw=multipart("file", mars_only.name, mars_bytes, boundary),
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    mars_model_id = (mars_uploaded or {}).get("id")
    check(
        "a Mars-only model still validates for Mars",
        status in (200, 201) and (mars_uploaded or {}).get("validation_status") == "valid",
        f"status={status} model_id={mars_model_id}",
    )
    status, mars_compat = request(
        "GET", f"/user-models/{mars_model_id}/earth-compatibility", token=token
    )
    check(
        "Mars-only model is reported as not Earth-compatible",
        status == 200 and mars_compat.get("compatible") is False,
        json.dumps(mars_compat, ensure_ascii=False)[:220] if mars_compat else f"status={status}",
    )
    status, mars_body = request("POST", "/training/start", token=token, body={
        **payload,
        "model_name": f"earth_mars_only_{suffix}",
        "uploaded_model_id": mars_model_id,
    })
    check(
        "POST /training/start refuses the Mars-only model with 422",
        status == 422
        and isinstance(mars_body, dict)
        and (mars_body.get("detail") or {}).get("code") == "uploaded_model_not_earth_compatible",
        f"status={status} detail={json.dumps(mars_body)[:200] if mars_body else None}",
    )

    status_missing = request("POST", "/training/start", token=token, body={
        **payload, "model_name": f"earth_no_id_{suffix}", "uploaded_model_id": None,
    })[0]
    check(
        "uploaded Earth requires an uploaded_model_id",
        status_missing == 422,
        f"status={status_missing}",
    )

    deadline = time.time() + 3600
    last = None
    while time.time() < deadline:
        # There is no single-task GET route; the task list is what the training page
        # itself polls, so it is what this script polls too.
        status, tasks = request("GET", "/training/tasks", token=token)
        if status != 200 or not isinstance(tasks, list):
            check("poll task list", False, f"status={status}")
            return 1
        detail = next((row for row in tasks if row.get("id") == task_id), None)
        if detail is None:
            check("task appears in the task list", False, f"task_id={task_id} not listed")
            return 1
        last = detail
        if detail.get("status") in ("completed", "failed", "stopped"):
            break
        time.sleep(10)
    if not check("uploaded Earth training completed", last.get("status") == "completed",
                 f"status={last.get('status')}"):
        log_status, log = request("GET", f"/training/tasks/{task_id}/logs", token=token)
        tail = json.dumps(log, ensure_ascii=False)[-1200:] if log else ""
        print(f"--- log tail ---\n{tail}", flush=True)
        return 1

    hyper = last.get("hyperparameters") or {}
    if isinstance(hyper, str):
        hyper = json.loads(hyper)
    check(
        "task records the pinned uploaded model identity",
        hyper.get("_uploaded_model_id") == model_id
        and bool(hyper.get("_uploaded_model_version"))
        and len(str(hyper.get("_uploaded_model_content_hash") or "")) == 64,
        json.dumps({k: v for k, v in hyper.items() if k.startswith("_uploaded")}, ensure_ascii=False),
    )

    status, log = request("GET", f"/training/tasks/{task_id}/logs", token=token)
    lines = log.get("lines") if isinstance(log, dict) else log
    text = "\n".join(lines) if isinstance(lines, list) else json.dumps(log, ensure_ascii=False)
    check("run log names the uploaded model", "uploaded:" in text,
          next((line.strip() for line in text.splitlines() if "EarthModel=" in line), ""))

    status, context = request(
        "GET", f"/earth/predict/context?training_task_id={task_id}", token=token
    )
    if status != 200:
        check("GET earth predict context", False, f"status={status} body={context}")
        return 1
    model_block = context.get("model") or {}
    check(
        "context identifies the uploaded model",
        model_block.get("uploaded_model_id") == model_id and model_block.get("model_source") == "uploaded",
        json.dumps(model_block, ensure_ascii=False),
    )
    origins = (context.get("origins") or {})
    origin = "2021-07-08"
    check("context publishes a selectable origin list", bool(origins.get("dates")),
          f"first={origins.get('first')} last={origins.get('last')} count={origins.get('count')}")

    status, result = request("POST", "/earth/predict/run", token=token, body={
        "training_task_id": task_id, "forecast_origin": origin,
    })
    if status != 200 or not result:
        check("POST /earth/predict/run", False, f"status={status} body={result}")
        return 1
    check(
        "prediction returns the uploaded model identity",
        (result.get("model") or {}).get("uploaded_model_id") == model_id,
        json.dumps(result.get("model"), ensure_ascii=False),
    )
    check("prediction target dates follow the origin", result.get("target_dates") ==
          ["2021-07-09", "2021-07-10", "2021-07-11"], json.dumps(result.get("target_dates")))
    check("prediction unit is DU", result.get("target_unit") == "DU")
    fields = {kind: len(result.get(kind) or []) for kind in ("prediction", "reference", "residual")}
    check("prediction/reference/residual all return 3 days", all(v == 3 for v in fields.values()),
          json.dumps(fields))
    metrics = result.get("metrics") or {}
    check("prediction metrics include DU rmse/mae",
          isinstance(metrics.get("overall", {}).get("rmse"), (int, float))
          and isinstance(metrics.get("overall", {}).get("mae"), (int, float)),
          json.dumps(metrics.get("overall")))

    day = (result.get("prediction") or [{}])[0]
    grid = day.get("field") or day.get("values") or []
    check("first prediction day carries a 36x72 grid", len(grid) == 36 and len(grid[0] or []) == 72,
          f"{len(grid)}x{len(grid[0]) if grid else 0}")

    print("\n=== ACCEPTANCE SUMMARY ===")
    failed = [row for row in RESULTS if not row[1]]
    for label, ok, detail in RESULTS:
        print(f"{'OK  ' if ok else 'FAIL'} {label}")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    print(f"task_id={task_id} uploaded_model_id={model_id} forecast_origin={origin}")
    print("Restart the backend, then re-run the prediction above to confirm checkpoint-only reload.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
