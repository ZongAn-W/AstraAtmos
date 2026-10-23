"""Bounded Earth backtest cache, independent of all Mars cache identities."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict

import numpy as np

ARRAY_KEYS = ("_prediction_du", "_reference_du", "_residual_du")
TASK_SPLIT_BOUNDARY_VERSION = "earth_task_split_complete_window_v1"


def build_earth_prediction_cache_key(*, planet, dataset_id, dataset_version,
                                     dataset_fingerprint, task_id, forecast_origin,
                                     target_timestamps, checkpoint_sha256="",
                                     split_boundary_policy="legacy_compatibility", topography_identity=None):
    identity = {
        "schema": "earth_backtest_cache_v1", "planet": planet,
        "dataset_id": dataset_id, "dataset_version": dataset_version,
        "dataset_fingerprint": dataset_fingerprint, "task_id": int(task_id),
        "forecast_origin": forecast_origin, "target_timestamps": list(target_timestamps),
        "checkpoint_sha256": checkpoint_sha256,
        "split_boundary_policy": split_boundary_policy,
    }
    if topography_identity is not None:
        identity["topography"] = topography_identity
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class EarthPredictionCache:
    """Keep immutable compact arrays; each public result is serialized afresh."""

    def __init__(self, max_bytes=128 * 1024 * 1024, max_entries=4):
        self.max_bytes, self.max_entries = int(max_bytes), int(max_entries)
        self._entries, self._bytes = OrderedDict(), 0
        self._lock = threading.RLock()

    def get(self, key):
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            self._entries.move_to_end(key)
            result = copy.deepcopy({name: value for name, value in item[0].items() if name not in ARRAY_KEYS})
            result.update({name: item[0][name] for name in ARRAY_KEYS})
            return result

    def put(self, key, result):
        arrays = {name: np.array(result[name], dtype="float32", copy=True) for name in ARRAY_KEYS}
        for value in arrays.values():
            value.flags.writeable = False
        size = sum(value.nbytes for value in arrays.values())
        if size > self.max_bytes or self.max_entries < 1:
            return
        metadata = copy.deepcopy({name: value for name, value in result.items() if name not in ARRAY_KEYS})
        size += len(json.dumps(metadata, allow_nan=False).encode())
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            self._entries[key] = ({**metadata, **arrays}, size)
            self._bytes += size
            while self._bytes > self.max_bytes or len(self._entries) > self.max_entries:
                _, (_, removed_size) = self._entries.popitem(last=False)
                self._bytes -= removed_size


EARTH_PREDICTION_CACHE = EarthPredictionCache()
