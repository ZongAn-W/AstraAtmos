"""Read-only audit by default; --apply performs a verified, backed-up recovery."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from services.earth_training_recovery import recover_task, verify_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--database", type=Path, default=config.DATA_DIR / "astraatmos.db")
    parser.add_argument("--package-dir", type=Path, default=config.EARTH_MERRA2_3HOURLY_DIR)
    parser.add_argument("--audit-root", type=Path, default=config.DATA_DIR / "recovery")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--failure-kind", choices=("metrics-contract", "reload-timeout"),
                        default="metrics-contract")
    args = parser.parse_args()
    options = dict(expected_checkpoint_sha256=args.checkpoint_sha256,
                   failure_kind=args.failure_kind,
                   timeout_seconds=config.USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS,
                   progress=lambda message: print(message, flush=True))
    if args.apply:
        result = recover_task(args.database, args.task_id, args.package_dir,
                              audit_root=args.audit_root, **options)
    else:
        plan = verify_recovery(args.database, args.task_id, args.package_dir, **options)
        result = {"task_id": args.task_id, "verified": True, "database_modified": False,
                  "checkpoint_sha256": plan["checkpoint_sha256"],
                  "test_metrics": plan["metrics"]["splits"]["test"]["overall"]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
