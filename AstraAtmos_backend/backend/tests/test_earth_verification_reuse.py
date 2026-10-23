"""Shared proofs preserve complete release verification without rescanning it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from services import earth_dataset_metadata as metadata
from services.dataset_registry import DatasetRegistry
from services.earth_preparation_cache import digest_json
from test_earth_3hourly_data_contract import make_threehour_release


@pytest.fixture(scope="module")
def verified_fixture(tmp_path_factory):
    return make_threehour_release(tmp_path_factory.mktemp("verification_fixture") / "release")


@pytest.fixture
def package(tmp_path, verified_fixture):
    return shutil.copytree(verified_fixture, tmp_path / "release")


def instrument_scans(monkeypatch):
    calls = []
    original = metadata.validate_earth_3hourly_dataset

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(metadata, "validate_earth_3hourly_dataset", counted)
    return calls


def test_parent_and_child_share_completed_full_verification(package, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    calls = instrument_scans(monkeypatch)
    parent = DatasetRegistry(earth_package_dir=tmp_path / "absent",
                             earth_3hourly_package_dir=package, verification_cache_dir=cache)
    first = parent.get_earth_snapshot("earth_merra2_3hourly_v1")
    child = DatasetRegistry(earth_package_dir=tmp_path / "absent",
                            earth_3hourly_package_dir=package, verification_cache_dir=cache)
    second = child.get_earth_snapshot("earth_merra2_3hourly_v1")
    assert not first.verification_cached and second.verification_cached
    assert calls == [True]
    assert parent.verification_count == 1
    assert child.verification_count == 0 and child.verification_cache_hits == 1
    assert first.metadata == second.metadata
    np.testing.assert_array_equal(first.dates, second.dates)

    script = (
        "import sys; from services.earth_dataset_metadata import read_earth_3hourly_release; "
        "r=read_earth_3hourly_release(sys.argv[1],verification_cache_dir=sys.argv[2]); "
        "assert r.verification_cached; print(r.metadata['dataset_fingerprint'])"
    )
    result = subprocess.run([sys.executable, "-c", script, str(package), str(cache)],
                            capture_output=True, text=True, check=True, timeout=60)
    assert result.stdout.strip() == first.metadata["dataset_fingerprint"]


def test_force_full_and_default_direct_reader_retain_complete_scan(package, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    calls = instrument_scans(monkeypatch)
    metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    assert metadata.read_earth_3hourly_release(package, verification_cache_dir=cache).verification_cached
    assert not metadata.read_earth_3hourly_release(package, verification_cache_dir=cache,
                                                 force_full=True).verification_cached
    assert not metadata.read_earth_3hourly_release(package).verification_cached
    assert len(calls) == 3


def test_registry_force_full_bypasses_in_memory_and_shared_proofs(package, tmp_path, monkeypatch):
    calls = instrument_scans(monkeypatch)
    registry = DatasetRegistry(earth_package_dir=tmp_path / "absent",
                               earth_3hourly_package_dir=package,
                               verification_cache_dir=tmp_path / "cache")
    registry.get_earth_snapshot("earth_merra2_3hourly_v1")
    registry.get_earth_snapshot("earth_merra2_3hourly_v1")
    assert len(calls) == 1
    release = registry.get_earth_snapshot("earth_merra2_3hourly_v1", force_full=True)
    assert not release.verification_cached and len(calls) == 2


@pytest.mark.parametrize("mutation", ["truncate", "metadata", "dates", "coordinates", "version",
                                     "identity", "resigned_metadata", "resigned_dates"])
def test_corrupted_or_old_proof_is_reverified(package, tmp_path, monkeypatch, mutation):
    cache = tmp_path / "cache"
    calls = instrument_scans(monkeypatch)
    original = metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    proof_path = metadata._verification_proof_path(cache, package)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if mutation == "truncate":
        proof_path.write_text('{"metadata":', encoding="utf-8")
    else:
        if mutation in ("metadata", "resigned_metadata"):
            proof["metadata"]["channel_order"].reverse()
        elif mutation in ("dates", "resigned_dates"):
            proof["dates"][0] = proof["dates"][1]
        elif mutation == "coordinates":
            proof["latitude"][0] += 1
        elif mutation == "version":
            proof["schema"] = "previous_verifier"
        elif mutation == "identity":
            proof["metadata"]["data_sha256"] = "a" * 64
        if mutation.startswith("resigned"):
            proof["proof_sha256"] = digest_json({k: v for k, v in proof.items() if k != "proof_sha256"})
        proof_path.write_text(json.dumps(proof), encoding="utf-8")
    recovered = metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    assert not recovered.verification_cached and len(calls) == 2
    assert recovered.metadata == original.metadata


def test_same_size_source_edit_with_restored_mtime_cannot_hit_proof(package, tmp_path):
    cache = tmp_path / "cache"
    metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    path = package / metadata.THREE_HOURLY_DATA_FILE_NAME
    before = metadata.package_signature(package, data_file_name=path.name)
    stat = path.stat()
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        value = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([value[0] ^ 1]))
        stream.flush()
        os.fsync(stream.fileno())
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    assert metadata.package_signature(package, data_file_name=path.name) != before
    with pytest.raises(metadata.EarthPackageError) as error:
        metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    assert error.value.reason == metadata.REASON_DATA_FINGERPRINT_MISMATCH


def test_manifest_edit_forces_new_full_verification(package, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    calls = instrument_scans(monkeypatch)
    first = metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    path = package / "manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    second = metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    assert not second.verification_cached and len(calls) == 2
    assert second.metadata["dataset_fingerprint"] == first.metadata["dataset_fingerprint"]


def test_change_while_reusing_proof_is_rejected(package, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    original = metadata._proof_matches_manifest

    def mutate(*args):
        result = original(*args)
        with (package / "manifest.json").open("ab") as stream:
            stream.write(b"\n")
        return result

    monkeypatch.setattr(metadata, "_proof_matches_manifest", mutate)
    with pytest.raises(metadata.EarthPackageError) as error:
        metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    assert error.value.reason == metadata.REASON_PACKAGE_CHANGED


def test_interrupted_verification_never_publishes_proof(package, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    original = metadata.validate_earth_3hourly_dataset

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("simulated verifier interruption")

    monkeypatch.setattr(metadata, "validate_earth_3hourly_dataset", interrupted)
    with pytest.raises(KeyboardInterrupt):
        metadata.read_earth_3hourly_release(package, verification_cache_dir=cache)
    assert not metadata._verification_proof_path(cache, package).exists()
    monkeypatch.setattr(metadata, "validate_earth_3hourly_dataset", original)
    assert not metadata.read_earth_3hourly_release(package, verification_cache_dir=cache).verification_cached


def test_concurrent_verifiers_publish_once_and_reuse(package, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    calls = instrument_scans(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(metadata.read_earth_3hourly_release, package,
                               verification_cache_dir=cache) for _ in range(2)]
        releases = [future.result(timeout=60) for future in futures]
    assert len(calls) == 1
    assert sorted(r.verification_cached for r in releases) == [False, True]


def test_concurrent_processes_coordinate_full_verification(package, tmp_path):
    cache = tmp_path / "cache"
    marker = tmp_path / "scans.txt"
    script = "\n".join([
        "import sys, time",
        "from pathlib import Path",
        "from services import earth_dataset_metadata as m",
        "original = m.validate_earth_3hourly_dataset",
        "def record(*args, **kwargs):",
        "    with Path(sys.argv[3]).open('a') as stream: stream.write('scan\\n')",
        "    time.sleep(0.2)",
        "    return original(*args, **kwargs)",
        "m.validate_earth_3hourly_dataset = record",
        "r = m.read_earth_3hourly_release(sys.argv[1], verification_cache_dir=sys.argv[2])",
        "print(int(r.verification_cached))",
    ])
    children = [subprocess.Popen([sys.executable, "-c", script, str(package), str(cache), str(marker)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for _ in range(2)]
    results = [child.communicate(timeout=60) for child in children]
    for child, (_, errors) in zip(children, results):
        assert child.returncode == 0, errors
    assert sorted(output.strip() for output, _ in results) == ["0", "1"]
    assert marker.read_text().splitlines() == ["scan"]


def test_validation_progress_reports_scan_and_reuse(package, tmp_path):
    messages = []
    cache = tmp_path / "cache"
    metadata.read_earth_3hourly_release(package, verification_cache_dir=cache, progress=messages.append)
    assert any("source SHA-256" in item and "bytes" in item for item in messages)
    assert any("completed full scan" in item and "elapsed=" in item for item in messages)
    messages.clear()
    metadata.read_earth_3hourly_release(package, verification_cache_dir=cache, progress=messages.append)
    assert any("data validation hit elapsed=" in item for item in messages)
    assert not any("scientific data/mask validation started" in item for item in messages)
