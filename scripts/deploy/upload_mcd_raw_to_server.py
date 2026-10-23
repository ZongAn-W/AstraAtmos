#!/usr/bin/env python3
"""Upload raw 3-hour MCD files used by the Data Overview diurnal chart."""

from __future__ import annotations

import argparse
import os
import posixpath
import socket
import sys
import time
from pathlib import Path

import paramiko


def format_bytes(value: int | float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int | None:
    try:
        return int(sftp.stat(path).st_size)
    except FileNotFoundError:
        return None
    except OSError:
        return None


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    current = ""
    for part in remote_dir.strip("/").split("/"):
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_file(
    sftp: paramiko.SFTPClient,
    local_path: Path,
    remote_dir: str,
    buffer_size: int,
    log_interval: float,
) -> str:
    final_path = posixpath.join(remote_dir, local_path.name)
    part_path = f"{final_path}.part"
    total = local_path.stat().st_size

    final_size = remote_size(sftp, final_path)
    if final_size == total:
        print(f"SKIP {local_path.name}: already complete ({format_bytes(total)})", flush=True)
        return "skipped"

    start_at = remote_size(sftp, part_path) or 0
    if start_at > total:
        print(f"RESET {local_path.name}: remote .part is larger than local", flush=True)
        start_at = 0

    mode = "ab" if start_at else "wb"
    uploaded = start_at
    started = time.monotonic()
    last_log = started

    print(
        f"UPLOAD {local_path.name}: {format_bytes(uploaded)}/{format_bytes(total)}",
        flush=True,
    )

    with local_path.open("rb") as src:
        src.seek(start_at)
        with sftp.open(part_path, mode) as dst:
            dst.set_pipelined(True)
            while True:
                chunk = src.read(buffer_size)
                if not chunk:
                    break
                dst.write(chunk)
                uploaded += len(chunk)
                now = time.monotonic()
                if now - last_log >= log_interval or uploaded == total:
                    elapsed = max(now - started, 0.001)
                    speed = (uploaded - start_at) / elapsed
                    remaining = max(total - uploaded, 0)
                    eta = remaining / speed if speed > 0 else 0
                    percent = uploaded / total * 100 if total else 100
                    print(
                        f"PROGRESS {local_path.name}: {percent:5.1f}% "
                        f"{format_bytes(uploaded)}/{format_bytes(total)} "
                        f"{format_bytes(speed)}/s eta={eta/60:.1f}m",
                        flush=True,
                    )
                    last_log = now

    part_size = remote_size(sftp, part_path)
    if part_size != total:
        raise RuntimeError(
            f"Remote upload size mismatch for {local_path.name}: "
            f"{part_size} != {total}"
        )

    try:
        sftp.posix_rename(part_path, final_path)
    except IOError:
        try:
            sftp.remove(final_path)
        except OSError:
            pass
        sftp.rename(part_path, final_path)

    print(f"DONE {local_path.name}: {format_bytes(total)}", flush=True)
    return "uploaded"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_source = repo_root.parent / "Data" / "MCD_Output_global_10m_ls_lst"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--remote-dir", default="/home/ubuntu/Data/MCD_Output_global_10m_ls_lst")
    parser.add_argument("--password-env", default="ASTRAATMOS_SSH_PASSWORD")
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--buffer-size", type=int, default=1024 * 1024)
    parser.add_argument("--log-interval", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 2

    password = os.environ.get(args.password_env)
    if not password:
        print(f"Missing SSH password environment variable: {args.password_env}", file=sys.stderr)
        return 2

    if args.years:
        files = [
            source_dir / f"MCD_MY{year}_global_3h_5deg_10m_ls_lst.nc"
            for year in args.years
        ]
    else:
        files = sorted(source_dir.glob("MCD_MY*_global_3h_5deg_10m_ls_lst.nc"))

    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        print("Missing local files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2
    if not files:
        print(f"No raw MCD files found in {source_dir}", file=sys.stderr)
        return 2

    total_bytes = sum(path.stat().st_size for path in files)
    print(
        f"Connecting to {args.user}@{args.host}:{args.port}; "
        f"{len(files)} files, {format_bytes(total_bytes)} total",
        flush=True,
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    try:
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(30)
        sftp = client.open_sftp()
        try:
            ensure_remote_dir(sftp, args.remote_dir)
            uploaded = 0
            skipped = 0
            for path in files:
                result = upload_file(
                    sftp,
                    path,
                    args.remote_dir,
                    args.buffer_size,
                    args.log_interval,
                )
                if result == "skipped":
                    skipped += 1
                else:
                    uploaded += 1
            print(f"COMPLETE uploaded={uploaded} skipped={skipped}", flush=True)
        finally:
            sftp.close()
    except (OSError, socket.error, RuntimeError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
