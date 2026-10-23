"""Small shared helpers for Earth preparation artifacts.

The preparation cache is deliberately content addressed.  Model parameters do
not enter its keys, while the verified release, normalization and storage
layout do.  All writers publish a complete directory atomically under a
cross-process lock; readers only accept a ready directory with matching file
identity metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock, Timeout


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def signature_json(signature: Any) -> list[list[Any]]:
    return [list(item) for item in signature]


def normalize_signature(value: Any) -> tuple:
    if not isinstance(value, list):
        raise ValueError("invalid package signature")
    result = []
    for item in value:
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError("invalid package signature")
        result.append(tuple(item))
    return tuple(result)


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return payload if isinstance(payload, dict) else None


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open('xb') as stream:
        stream.write(canonical_json(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def preparation_lock(path: Path, *, timeout: float = 3600.0, progress=None) -> Iterator[None]:
    """Coordinate builders in this and sibling training processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path))
    started = time.perf_counter()
    while True:
        try:
            lock.acquire(timeout=5)
            break
        except Timeout:
            if time.perf_counter() - started >= timeout:
                raise
            if progress is not None:
                progress(f'Earth preparation: waiting for shared builder ({time.perf_counter() - started:.1f}s)')
    try:
        yield
    finally:
        lock.release()


def file_identity(path: Path) -> dict[str, Any]:
    """File identity plus modification/change times; fail closed on Windows.

    Windows st_ctime is creation time. NTFS ChangeTime also changes when a caller
    restores mtime, so a same-size edit or replacement cannot reuse a proof.
    """
    path = Path(path).resolve()
    stat = path.stat()
    changed = int(stat.st_ctime_ns)
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class FileBasicInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in
                        ('CreationTime', 'LastAccessTime', 'LastWriteTime', 'ChangeTime')]
            _fields_.append(('FileAttributes', wintypes.DWORD))
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        query = kernel.GetFileInformationByHandleEx
        query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        query.restype = wintypes.BOOL
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        handle = create(str(path), 0x80, 7, None, 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = FileBasicInfo()
            if not query(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            changed = int(info.ChangeTime) * 100
        finally:
            close(handle)
    return {'path': str(path), 'device': int(stat.st_dev), 'inode': int(stat.st_ino),
            'bytes': int(stat.st_size), 'mtime_ns': int(stat.st_mtime_ns), 'change_ns': changed}


def cache_file_identity(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    identity.pop('path')  # Atomic publication renames the containing directory.
    return identity


def sha256_file(path: Path, *, progress=None, stage='cache integrity') -> str:
    checksum = hashlib.sha256()
    size, done = path.stat().st_size, 0
    last = time.monotonic()
    if progress is not None:
        progress(f'Earth preparation: {stage} 0/{size} bytes')
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum.update(block)
            done += len(block)
            if progress is not None and (done == size or time.monotonic() - last >= 5):
                progress(f'Earth preparation: {stage} {done}/{size} bytes')
                last = time.monotonic()
    return checksum.hexdigest()


_CHECKED_ARRAYS = {}


class ReadOnlyFileGuard:
    """Prevent writes/replacement while Windows keeps read-only maps open.

    NTFS timestamps can remain unchanged until all mmap handles close. Holding
    a shared-read-only handle makes integrity memoization safe during that gap.
    Other readers can still open the same published cache concurrently.
    """

    def __init__(self, path: Path):
        self._handle = None
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            create = kernel.CreateFileW
            create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                               wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            create.restype = wintypes.HANDLE
            close = kernel.CloseHandle
            close.argtypes = [wintypes.HANDLE]
            close.restype = wintypes.BOOL
            self._close_handle = close
            handle = create(str(Path(path).resolve()), 0x80000000, 1, None, 3, 0, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = handle

    def close(self):
        if self._handle is not None:
            self._close_handle(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def verify_array_file(path: Path, expected: dict, checksum: str, *, progress=None) -> bool:
    """Hash once per process/unchanged file and reject edits or partial writes."""
    if not isinstance(checksum, str) or len(checksum) != 64:
        return False
    guard = None
    try:
        guard = ReadOnlyFileGuard(path)
        before = cache_file_identity(path)
        if before != expected:
            return False
        key = (str(path.resolve()), digest_json(before), checksum)
        if key not in _CHECKED_ARRAYS:
            if sha256_file(path, progress=progress) != checksum:
                return False
            if cache_file_identity(path) != before:
                return False
            # Keep Windows write protection as long as the digest is memoized.
            # Releasing it earlier permits an mmap edit with delayed timestamps.
            _CHECKED_ARRAYS[key] = guard
            guard = None
        return cache_file_identity(path) == before
    except OSError:
        return False
    finally:
        if guard is not None:
            guard.close()


def clear_verified_arrays():
    """Release integrity results and their read guards together."""
    for guard in _CHECKED_ARRAYS.values():
        guard.close()
    _CHECKED_ARRAYS.clear()


def seal_payload(payload: dict) -> dict:
    result = dict(payload)
    result.pop('proof_sha256', None)
    result['proof_sha256'] = digest_json(result)
    return result


def valid_seal(payload: dict) -> bool:
    try:
        body = dict(payload)
        checksum = body.pop('proof_sha256', None)
        return checksum == digest_json(body)
    except (TypeError, ValueError):
        return False


def file_identity_matches(path: Path, expected: dict[str, Any]) -> bool:
    try:
        return cache_file_identity(path) == expected
    except OSError:
        return False
