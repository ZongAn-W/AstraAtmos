"""Preparation reuse is content based, complete and safe across processes.

These tests prepare synthetic NetCDF volumes only. They never invoke a model,
start a training task, or interact with the deployed backend.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from models.training_scripts.earth_daily import _build_loaders
from services import earth_dataset as data
from services.earth_dataset_metadata import package_signature
from services.earth_task_split import build_earth_task_split
from services.earth_training_contract import build_earth_training_spec
from services.netcdf_read_lock import netcdf_read_lock
from test_earth_3hourly_training_data import disk_release, mutable_release
from test_earth_3hourly_training_runner import registry_for, synthetic_training_release
from training_backbones.earth_3hourly_uploaded_contract import FULL_GRID_CONTRACT_SCHEMA


def _split(release, *, window=7, horizon=3, train=.5, validation=.25, test=.25):
    return build_earth_task_split(release.dates, window, horizon, {
        "train_ratio": train, "validation_ratio": validation, "test_ratio": test,
    })


def _fit(release, root, *, channels=(), task_split=None, progress=None):
    return data.fit_threehour_normalization(
        release, list(channels), task_split=task_split, cache_root=root, progress=progress,
    )


def _build(release, root, normalization, *, full_grid=True, progress=None):
    return data.build_threehour_training_cache(
        release, normalization["channel_order"], normalization, root,
        full_grid=full_grid, progress=progress,
    )


def _close(dataset):
    dataset.close_training_cache()


def _reject_scan(*args, **kwargs):
    pytest.fail("A reusable preparation artifact must avoid another NetCDF scan")


def test_same_preparation_reuses_statistics_and_volume(disk_release, tmp_path, monkeypatch):
    logs = []
    normalization = _fit(disk_release, tmp_path, progress=logs.append)
    cache = _build(disk_release, tmp_path, normalization, progress=logs.append)
    original = (cache.stat().st_mtime_ns, cache.stat().st_size)
    monkeypatch.setattr(data, "_fit_threehour_normalization_uncached", _reject_scan)
    monkeypatch.setattr(data, "_build_threehour_training_cache_uncached", _reject_scan)
    monkeypatch.setattr(xr, "open_dataset", _reject_scan)
    reused_normalization = _fit(disk_release, tmp_path, progress=logs.append)
    reused = _build(disk_release, tmp_path, reused_normalization, progress=logs.append)
    assert reused_normalization == normalization
    assert reused == cache
    assert original == (cache.stat().st_mtime_ns, cache.stat().st_size)
    assert any("normalization cache hit" in message for message in logs)
    assert any("earth_full_grid_v1 cache hit" in message for message in logs)
    assert any("train frames" in message for message in logs)
    assert any("generated" in message and "s" in message for message in logs)


@pytest.mark.parametrize("full_grid", [False, True])
def test_verified_cache_hash_is_shared_by_builder_and_split_attachments(
    mutable_release, tmp_path, monkeypatch, full_grid,
):
    """A process hashes each unchanged array once, including all three splits."""
    from services import earth_preparation_cache as preparation

    release = mutable_release
    task_split = _split(release)
    normalization = _fit(release, tmp_path, task_split=task_split)
    path = _build(release, tmp_path, normalization, full_grid=full_grid)
    metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    checksum = preparation.sha256_file
    hashes = []
    def counted_hash(candidate, **kwargs):
        hashes.append(Path(candidate).resolve())
        return checksum(candidate, **kwargs)
    monkeypatch.setattr(preparation, "sha256_file", counted_hash)
    monkeypatch.setattr(data, "_fit_threehour_normalization_uncached", _reject_scan)
    monkeypatch.setattr(data, "_build_threehour_training_cache_uncached", _reject_scan)
    assert _build(release, tmp_path, normalization, full_grid=full_grid) == path
    assert hashes == [path.resolve()]
    windows = [data.EarthThreeHourlyWindows.from_release(
        release, split=split, window=7, horizon=3, selected_channels=[],
        normalization=normalization, task_split=task_split,
    ) for split in ("train", "validation", "test")]
    try:
        for dataset in windows:
            dataset.use_training_cache(path)
            inputs, targets = dataset.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))
            assert np.isfinite(inputs).all() and np.isfinite(targets).all()
            assert hashes == [path.resolve()]
        assert _build(release, tmp_path, normalization, full_grid=full_grid) == path
        assert hashes == [path.resolve()]

        # A fresh interpreter has no in-memory integrity result. Its builder
        # must hash the complete published file once before attaching splits.
        child = '''
import json
import sys
from pathlib import Path
from types import MappingProxyType
import numpy as np
from services import earth_dataset as data
from services import earth_preparation_cache as preparation
from services.earth_dataset_metadata import VerifiedEarthRelease, package_signature

payload = json.loads(sys.stdin.read())
source = Path(payload['data_path'])
release = VerifiedEarthRelease(
    metadata=payload['metadata'], signature=package_signature(source.parent, data_file_name=source.name),
    dates=np.array(payload['dates'], dtype='datetime64[ns]'),
    latitude=np.array(payload['latitude']), longitude=np.array(payload['longitude']),
    fields=MappingProxyType({}), data_path=source,
)
assert not preparation._CHECKED_ARRAYS
original_hash = preparation.sha256_file
hashes = []
def counted_hash(path, **kwargs):
    hashes.append(str(Path(path).resolve()))
    return original_hash(path, **kwargs)
preparation.sha256_file = counted_hash
def reject_build(*args, **kwargs):
    raise AssertionError('Existing cache must not be generated again')
data._build_threehour_training_cache_uncached = reject_build
normalization = payload['normalization']
path = data.build_threehour_training_cache(
    release, ['TO3'], normalization, payload['cache_root'], full_grid=payload['full_grid'],
)
assert path == Path(payload['cache_path'])
counts = [len(hashes)]
for split in ('train', 'validation', 'test'):
    windows = data.EarthThreeHourlyWindows.from_release(
        release, split=split, window=7, horizon=3, selected_channels=[],
        normalization=normalization, task_split=payload['task_split'],
    )
    try:
        windows.use_training_cache(path)
        inputs, targets = windows.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))
        assert np.isfinite(inputs).all() and np.isfinite(targets).all()
        counts.append(len(hashes))
    finally:
        if windows._normalized_map is not None:
            windows._normalized_map._mmap.close()
print(json.dumps({'path': str(path), 'hashes': hashes, 'counts': counts}))
'''
        payload = {
            "data_path": str(release.data_path), "metadata": release.metadata,
            "dates": release.dates.astype(str).tolist(),
            "latitude": release.latitude.tolist(), "longitude": release.longitude.tolist(),
            "normalization": normalization, "task_split": task_split,
            "cache_root": str(tmp_path), "cache_path": str(path), "full_grid": full_grid,
        }
        backend = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(backend)}
        completed = subprocess.run(
            [sys.executable, "-c", child], input=json.dumps(payload),
            cwd=backend, env=env, capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        independent = json.loads(completed.stdout)
        assert independent["hashes"] == [str(path.resolve())]
        assert independent["counts"] == [1, 1, 1, 1]

        # Windows defers mmap timestamps until reader handles close. The read
        # guard must forbid writes while the certified maps are alive.
        if os.name == 'nt':
            with pytest.raises(OSError):
                np.load(path, mmap_mode='r+', allow_pickle=False)
            with pytest.raises(OSError):
                path.replace(path.with_name('replaced.npy'))
            for dataset in windows:
                _close(dataset)
            preparation.clear_verified_arrays()

        # Once readers close, a finite edit invalidates the cached digest.
        modified = np.load(path, mmap_mode="r+", allow_pickle=False)
        modified.flat[0] += np.float32(.125)
        modified.flush()
        modified._mmap.close()
        assert not preparation.verify_array_file(
            path, metadata["array_identity"], metadata["array_sha256"],
        )
        if os.name != 'nt':
            with pytest.raises(ValueError, match="cache file changed after publication"):
                windows[0].read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))
        with pytest.raises(ValueError, match="changed after publication|integrity check failed"):
            windows[1].use_training_cache(path)
        assert hashes[-1] == path.resolve()

        with netcdf_read_lock(), netCDF4.Dataset(release.data_path, "r+") as stored:
            stored["TO3"][0, 0, 0] += np.float32(1)
        with pytest.raises(ValueError, match="changed since verification"):
            windows[0].read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))
        with pytest.raises(ValueError, match="changed since verification"):
            windows[2].use_training_cache(path)
    finally:
        for dataset in windows:
            _close(dataset)


@pytest.mark.parametrize("full_grid", [False, True])
def test_model_batch_and_training_parameters_do_not_rebuild_loaders_cache(
    synthetic_training_release, tmp_path, monkeypatch, full_grid,
):
    """Exercise the real preparation caller without running an optimizer."""
    release = synthetic_training_release
    registry = registry_for(release)
    binding = registry.build_training_binding("earth_merra2_3hourly_v1")
    ratios = {name + "_ratio": 1 / 3 for name in ("train", "validation", "test")}
    original = build_earth_training_spec(
        task_id=901, dataset_binding=binding,
        task_split=build_earth_task_split(release.dates, 56, 24, ratios),
        hyperparameters={**ratios, "selected_channels": [], "batch_size": 2},
    )
    if full_grid:
        original["hyperparameters"].update(model_source="uploaded", model_architecture="uploaded")
        original["uploaded_model"] = {"contract_schema": FULL_GRID_CONTRACT_SCHEMA}
    first = _build_loaders(original, registry, 2, 42, cache_root=tmp_path)
    try:
        first_path = Path(first["splits"]["train"]._normalized_map.filename)
        monkeypatch.setattr(data, "_fit_threehour_normalization_uncached", _reject_scan)
        monkeypatch.setattr(data, "_build_threehour_training_cache_uncached", _reject_scan)
        changed = copy.deepcopy(original)
        changed["task_id"] = 902
        changed["hyperparameters"].update(
            batch_size=5, epochs=4, learning_rate=.002, seed=6, linear_hidden_layers=3,
            custom_model_params={"width": 64, "dropout": .1},
            model_source="uploaded", model_architecture="uploaded",
        )
        changed["uploaded_model"] = {
            "contract_schema": FULL_GRID_CONTRACT_SCHEMA if full_grid else "astraatmos_earth_3hourly_model_v1",
        }
        second = _build_loaders(changed, registry, 5, 6, cache_root=tmp_path)
        try:
            assert Path(second["splits"]["train"]._normalized_map.filename) == first_path
            assert second["normalization"] == first["normalization"]
            assert second["train_loader"].batch_size == 5
            assert second["counts"] == first["counts"]
        finally:
            for dataset in second["splits"].values():
                _close(dataset)
    finally:
        for dataset in first["splits"].values():
            _close(dataset)


def test_window_and_heldout_partition_changes_rebind_metadata_without_refitting(
    disk_release, tmp_path, monkeypatch,
):
    first_split = _split(disk_release)
    first = _fit(disk_release, tmp_path, task_split=first_split)
    first_cache = _build(disk_release, tmp_path, first)
    # The raw train boundary remains exactly 44 frames. Windows and held-out
    # partitions affect consumers, but they do not change normalized bytes.
    changed_split = _split(disk_release, window=5, horizon=2, validation=.3, test=.2)
    assert changed_split["ranges"]["train"]["raw_end"] == first_split["ranges"]["train"]["raw_end"]
    monkeypatch.setattr(data, "_fit_threehour_normalization_uncached", _reject_scan)
    monkeypatch.setattr(data, "_build_threehour_training_cache_uncached", _reject_scan)
    changed = _fit(disk_release, tmp_path, task_split=changed_split)
    assert changed["task_split"] == changed_split
    assert changed["mean"] == first["mean"] and changed["scale"] == first["scale"]
    assert _build(disk_release, tmp_path, changed) == first_cache
    for name in ("train", "validation", "test"):
        windows = data.EarthThreeHourlyWindows.from_release(
            disk_release, split=name, window=5, horizon=2, selected_channels=[],
            task_split=changed_split, normalization=changed,
        )
        try:
            windows.use_training_cache(first_cache)
            assert len(windows) == changed_split["ranges"][name]["window_count"]
        finally:
            _close(windows)


@pytest.mark.parametrize("change", ["source", "fingerprint", "channels", "train", "method_version"])
def test_normalization_identity_changes_force_train_only_refit(
    mutable_release, tmp_path, monkeypatch, change,
):
    release = mutable_release
    split = _split(release)
    first = _fit(release, tmp_path, task_split=split)
    selected = []
    if change == "source":
        with netcdf_read_lock(), netCDF4.Dataset(release.data_path, "r+") as stored:
            stored["TO3"][0, 0, 0] += np.float32(2)
        release = replace(release, signature=package_signature(
            release.data_path.parent, data_file_name=release.data_path.name,
        ))
    elif change == "fingerprint":
        metadata = copy.deepcopy(release.metadata)
        metadata["dataset_fingerprint"] = "e" * 64
        release = replace(release, metadata=metadata)
    elif change == "channels":
        selected = ["U10M"]
    elif change == "train":
        split = _split(release, train=.4, validation=.3, test=.3)
    else:
        monkeypatch.setattr(data, "NORMALIZATION_CACHE_VERSION", "earth_test_next_method_v3")
    original = data._fit_threehour_normalization_uncached
    calls = []
    def fit(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(data, "_fit_threehour_normalization_uncached", fit)
    second = _fit(release, tmp_path, channels=selected, task_split=split)
    assert calls == [True]
    assert len(list((tmp_path / "normalization").glob("*.json"))) == 2
    assert second["fit_time_end"] == split["ranges"]["train"]["date_end"]
    if change == "channels":
        assert second["channel_order"] == ["TO3", "U10M"]
    if change == "train":
        assert second["mean"] != first["mean"]


@pytest.mark.parametrize("corruption", ["json", "version", "identity", "statistics"])
def test_damaged_normalization_is_refitted(disk_release, tmp_path, monkeypatch, corruption):
    first = _fit(disk_release, tmp_path)
    path, = (tmp_path / "normalization").glob("*.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "json":
        path.write_text("{interrupted", encoding="utf-8")
    else:
        if corruption == "version":
            payload["schema"] = "earth_3hourly_normalization_v0"
        elif corruption == "identity":
            payload["identity"]["fit_time_end"] = "2030-01-01T01:30:00Z"
        else:
            payload["normalization"]["mean"][0] += 100
        path.write_text(json.dumps(payload), encoding="utf-8")
    original = data._fit_threehour_normalization_uncached
    calls = []
    def fit(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(data, "_fit_threehour_normalization_uncached", fit)
    assert _fit(disk_release, tmp_path) == first
    assert calls == [True]


def test_full_grid_and_spatial_tiles_reuse_and_read_identical_float32_windows(
    disk_release, tmp_path, monkeypatch,
):
    normalization = _fit(disk_release, tmp_path, channels=["U10M"])
    raw = data.EarthThreeHourlyWindows.from_release(
        disk_release, window=7, horizon=3, selected_channels=["U10M"], normalization=normalization,
    )
    expected = raw.read_window(2, lat_slice=slice(22, 27), lon_slice=slice(46, 51))
    paths = {full_grid: _build(disk_release, tmp_path, normalization, full_grid=full_grid)
             for full_grid in (False, True)}
    assert paths[False] != paths[True]
    monkeypatch.setattr(data, "_build_threehour_training_cache_uncached", _reject_scan)
    for full_grid, path in paths.items():
        assert _build(disk_release, tmp_path, normalization, full_grid=full_grid) == path
        windows = data.EarthThreeHourlyWindows.from_release(
            disk_release, window=7, horizon=3, selected_channels=["U10M"], normalization=normalization,
        )
        try:
            windows.use_training_cache(path)
            actual = windows.read_window(2, lat_slice=slice(22, 27), lon_slice=slice(46, 51))
            assert all(value.dtype == np.dtype("float32") for value in actual)
            for current, reference in zip(actual, expected):
                np.testing.assert_allclose(current, reference, rtol=2e-6, atol=2e-6)
        finally:
            _close(windows)


@pytest.mark.parametrize("change", ["fingerprint", "channels", "train", "layout"])
def test_training_cache_content_identity_changes_do_not_reuse(disk_release, tmp_path, change):
    release = disk_release
    split = _split(release)
    normalization = _fit(release, tmp_path, task_split=split)
    first = _build(release, tmp_path, normalization)
    full_grid, channels = True, []
    if change == "fingerprint":
        metadata = copy.deepcopy(release.metadata)
        metadata["dataset_fingerprint"] = "f" * 64
        release = replace(release, metadata=metadata)
    elif change == "channels":
        channels = ["SWGDN"]
    elif change == "train":
        split = _split(release, train=.4, validation=.3, test=.3)
    else:
        full_grid = False
    normalization = _fit(release, tmp_path, channels=channels, task_split=split)
    second = _build(release, tmp_path, normalization, full_grid=full_grid)
    assert second != first
    assert first.is_file()


@pytest.mark.parametrize("corruption", ["payload", "truncate", "metadata", "version", "identity", "incomplete"])
def test_corrupted_and_unready_cache_is_rejected_and_retained(
    disk_release, tmp_path, corruption,
):
    normalization = _fit(disk_release, tmp_path)
    path = _build(disk_release, tmp_path, normalization)
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if corruption == "payload":
        # A finite edit must be rejected too, even if a reader cannot spot NaN.
        modified = np.load(path, mmap_mode="r+", allow_pickle=False)
        modified.flat[0] += np.float32(.125)
        modified.flush()
        modified._mmap.close()
    elif corruption == "truncate":
        with path.open("r+b") as stream:
            stream.truncate(512)
    elif corruption == "metadata":
        metadata_path.write_text("{partial", encoding="utf-8")
    else:
        if corruption == "version":
            metadata["cache_version"] = 0
        elif corruption == "identity":
            metadata["dataset_fingerprint"] = "wrong-package"
        else:
            metadata["cache_status"] = "building"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    windows = data.EarthThreeHourlyWindows.from_release(
        disk_release, window=7, horizon=3, selected_channels=[], normalization=normalization,
    )
    with pytest.raises((ValueError, OSError)):
        windows.use_training_cache(path)
    repaired = _build(disk_release, tmp_path, normalization)
    # Regeneration may rename the rejected directory, but never delete its bytes.
    assert len(list(tmp_path.rglob("normalized.npy"))) == 2
    windows.use_training_cache(repaired)
    try:
        assert np.isfinite(windows.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))[0]).all()
    finally:
        _close(windows)


def test_interrupted_generation_never_publishes_ready_volume(disk_release, tmp_path, monkeypatch):
    normalization = _fit(disk_release, tmp_path)
    original = data._threehour_values
    reads = []
    def interrupted(*args, **kwargs):
        reads.append(True)
        if len(reads) == 2:
            raise RuntimeError("synthetic interrupted preparation")
        return original(*args, **kwargs)
    monkeypatch.setattr(data, "_threehour_values", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        _build(disk_release, tmp_path, normalization)
    abandoned = list(tmp_path.rglob("normalized.npy"))
    assert abandoned
    assert not any((path.parent / "ready.json").exists() for path in abandoned)
    monkeypatch.setattr(data, "_threehour_values", original)
    published = _build(disk_release, tmp_path, normalization)
    assert published not in abandoned
    assert all(path.is_file() for path in abandoned)


@pytest.mark.parametrize("operation", ["normalization", "build", "attach", "read"])
def test_source_changes_are_rejected_even_with_prepared_artifacts(
    mutable_release, tmp_path, operation,
):
    release = mutable_release
    normalization = _fit(release, tmp_path)
    path = _build(release, tmp_path, normalization)
    windows = data.EarthThreeHourlyWindows.from_release(
        release, window=7, horizon=3, selected_channels=[], normalization=normalization,
    )
    if operation == "read":
        windows.use_training_cache(path)
    with netcdf_read_lock(), netCDF4.Dataset(release.data_path, "r+") as stored:
        stored["TO3"][0, 0, 0] += np.float32(1)
    try:
        with pytest.raises(ValueError, match="changed since verification"):
            if operation == "normalization":
                _fit(release, tmp_path)
            elif operation == "build":
                _build(release, tmp_path, normalization)
            elif operation == "attach":
                windows.use_training_cache(path)
            else:
                windows.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))
    finally:
        _close(windows)


def test_existing_complete_full_grid_cache_is_validated_and_reused(disk_release, tmp_path, monkeypatch):
    """Migration scans the old artifact once instead of generating 12.55 GB again."""
    normalization = data.fit_threehour_normalization(disk_release, [])
    existing = _build(disk_release, tmp_path, normalization)
    # Reconstruct the metadata emitted before shared preparation certificates;
    # the complete numeric payload is the same float32 time-major array.
    metadata_path = existing.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("cache_identity_version", "cache_status", "cache_key", "identity",
                "array_identity", "array_sha256", "metadata_sha256", "package_signature"):
        metadata.pop(key, None)
    metadata.pop("proof_sha256", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(data, "_build_threehour_training_cache_uncached", _reject_scan)
    current = _fit(disk_release, tmp_path)
    found = _build(disk_release, tmp_path, current)
    assert found == existing
    windows = data.EarthThreeHourlyWindows.from_release(
        disk_release, window=7, horizon=3, selected_channels=[], normalization=current,
    )
    try:
        windows.use_training_cache(found)
        assert windows.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))[0].dtype == np.dtype("float32")
    finally:
        _close(windows)


def test_two_processes_publish_one_complete_cache(disk_release, tmp_path):
    """Two independent Python interpreters contend for the same first preparation."""
    helper = tmp_path / "prepare_child.py"
    helper.write_text('''
import json
import sys
import time
from pathlib import Path
from types import MappingProxyType
import numpy as np
from services import earth_dataset as data
from services.earth_dataset_metadata import VerifiedEarthRelease, package_signature

payload = json.loads(Path(sys.argv[1]).read_text())
root = Path(payload['root'])
source = Path(payload['data_path'])
release = VerifiedEarthRelease(
    metadata=payload['metadata'], signature=package_signature(source.parent, data_file_name=source.name),
    dates=np.array(payload['dates'], dtype='datetime64[ns]'),
    latitude=np.array(payload['latitude']), longitude=np.array(payload['longitude']),
    fields=MappingProxyType({}), data_path=source,
)
number = sys.argv[2]
for name in ('_fit_threehour_normalization_uncached', '_build_threehour_training_cache_uncached'):
    original = getattr(data, name)
    def wrap(*args, _original=original, _name=name, **kwargs):
        with (root / 'builders.txt').open('a') as log:
            log.write(number + ' ' + _name + '\\n')
        return _original(*args, **kwargs)
    setattr(data, name, wrap)
(root / ('child-' + number + '.ready')).write_text('ready')
deadline = time.monotonic() + 30
while not (root / 'go').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('concurrent test barrier timed out')
    time.sleep(.02)
normalization = data.fit_threehour_normalization(release, [], cache_root=root / 'cache')
path = data.build_threehour_training_cache(release, ['TO3'], normalization, root / 'cache', full_grid=True)
windows = data.EarthThreeHourlyWindows.from_release(
    release, window=7, horizon=3, selected_channels=[], normalization=normalization,
)
windows.use_training_cache(path)
inputs, targets = windows.read_window(0, lat_slice=slice(0, 1), lon_slice=slice(0, 1))
windows._normalized_map._mmap.close()
(root / ('child-' + number + '.result.json')).write_text(json.dumps({
    'path': str(path), 'finite': bool(np.isfinite(inputs).all() and np.isfinite(targets).all()),
}))
''', encoding="utf-8")
    payload = tmp_path / "release.json"
    payload.write_text(json.dumps({
        "root": str(tmp_path), "data_path": str(disk_release.data_path),
        "metadata": disk_release.metadata, "dates": disk_release.dates.astype(str).tolist(),
        "latitude": disk_release.latitude.tolist(), "longitude": disk_release.longitude.tolist(),
    }), encoding="utf-8")
    env = dict(os.environ)
    backend = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(backend)
    processes = [subprocess.Popen(
        [sys.executable, str(helper), str(payload), str(number)], cwd=backend,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    ) for number in (1, 2)]
    try:
        deadline = time.monotonic() + 45
        while not all((tmp_path / f"child-{number}.ready").exists() for number in (1, 2)):
            if any(process.poll() is not None for process in processes):
                break
            if time.monotonic() > deadline:
                pytest.fail("Preparation subprocess did not reach the barrier")
            time.sleep(.02)
        (tmp_path / "go").write_text("go", encoding="utf-8")
        for process in processes:
            stdout, stderr = process.communicate(timeout=120)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=15)
    results = [json.loads((tmp_path / f"child-{number}.result.json").read_text()) for number in (1, 2)]
    assert results[0]["path"] == results[1]["path"]
    assert all(result["finite"] for result in results)
    builders = (tmp_path / "builders.txt").read_text().splitlines()
    assert sum("_fit_threehour_normalization_uncached" in line for line in builders) == 1
    assert sum("_build_threehour_training_cache_uncached" in line for line in builders) == 1
    assert len(list((tmp_path / "cache").rglob("normalized.npy"))) == 1


def test_cache_changed_between_validation_and_mapping_is_rejected(disk_release, tmp_path, monkeypatch):
    normalization = _fit(disk_release, tmp_path)
    path = _build(disk_release, tmp_path, normalization)
    windows = data.EarthThreeHourlyWindows.from_release(
        disk_release, window=7, horizon=3, selected_channels=[], normalization=normalization,
    )
    candidate = data._training_cache_candidate
    def replace_after_validation(*args, **kwargs):
        result = candidate(*args, **kwargs)
        if result is not None:
            mapped = np.load(path, mmap_mode='r+')
            mapped.flat[0] += np.float32(.5)
            mapped.flush()
            mapped._mmap.close()
        return result
    monkeypatch.setattr(data, '_training_cache_candidate', replace_after_validation)
    with pytest.raises((ValueError, OSError)):
        windows.use_training_cache(path)
    assert windows._normalized_map is None


def test_direct_legacy_cache_attachment_requires_full_source_certification(disk_release, tmp_path):
    normalization = _fit(disk_release, tmp_path)
    path = _build(disk_release, tmp_path, normalization)
    metadata_path = path.with_name('metadata.json')
    metadata = json.loads(metadata_path.read_text())
    for key in ('cache_identity_version', 'cache_status', 'cache_key', 'identity',
                'array_identity', 'array_sha256', 'proof_sha256', 'package_signature'):
        metadata.pop(key, None)
    metadata_path.write_text(json.dumps(metadata))
    mapped = np.load(path, mmap_mode='r+')
    mapped.flat[0] += np.float32(.5)
    mapped.flush()
    mapped._mmap.close()
    windows = data.EarthThreeHourlyWindows.from_release(
        disk_release, window=7, horizon=3, selected_channels=[], normalization=normalization,
    )
    with pytest.raises(ValueError, match='integrity check failed'):
        windows.use_training_cache(path)
    assert windows._normalized_map is None
