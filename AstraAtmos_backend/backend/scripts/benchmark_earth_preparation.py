"""Measure Earth preparation only; never construct or train a model.

Use a fresh cache directory for --mode cold, then run --mode reuse in a new
process against the same directory. --mode existing certifies matching current
layout caches in place once, without recreating the volume. Reports include
source identity and stage times, not model throughput or accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from services.earth_dataset_metadata import read_earth_3hourly_release
from services.earth_dataset import (canonical_input_channels, fit_threehour_normalization,
                                    build_threehour_training_cache, EarthThreeHourlyWindows)
from services.earth_task_split import build_earth_task_split
from services.earth_preparation_cache import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['cold', 'reuse', 'existing'], default='reuse')
    parser.add_argument('--layout', choices=['full', 'tiles'], default='full')
    parser.add_argument('--channels', nargs='+', default=['TO3', 'U10M', 'V10M', 'T2M', 'SWGDN'])
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--train-ratio', type=float, default=.7)
    parser.add_argument('--validation-ratio', type=float, default=.15)
    parser.add_argument('--test-ratio', type=float, default=.15)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    progress = lambda message: print(message, flush=True)
    total = time.perf_counter()
    stage = time.perf_counter()
    release = read_earth_3hourly_release(
        args.package_dir, verification_cache_dir=args.cache_root,
        force_full=args.mode == 'cold', progress=progress,
    )
    release.metadata.setdefault('dataset_id', 'earth_merra2_3hourly_v1')
    release.metadata.setdefault('dataset_version', 'v1')
    validation = time.perf_counter() - stage
    ratios = {name + '_ratio': getattr(args, name + '_ratio')
              for name in ('train', 'validation', 'test')}
    split = build_earth_task_split(release.dates, args.window, args.horizon, ratios)
    channels = canonical_input_channels(args.channels)
    stage = time.perf_counter()
    normalization = fit_threehour_normalization(
        release, channels, task_split=split, cache_root=args.cache_root, progress=progress,
    )
    normalization_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    cache = build_threehour_training_cache(release, channels, normalization, args.cache_root,
                                          full_grid=args.layout == 'full', progress=progress)
    cache_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    datasets = []
    try:
        for name in ('train', 'validation', 'test'):
            dataset = EarthThreeHourlyWindows.from_release(
                release, split=name, window=args.window, horizon=args.horizon,
                selected_channels=channels[1:], normalization=normalization, task_split=split,
            )
            dataset.use_training_cache(cache, progress=progress)
            datasets.append(dataset)
        # One bounded read proves the map is usable, without model execution.
        datasets[0].read_window(0, lat_slice=slice(0, 24), lon_slice=slice(0, 48))
    finally:
        for dataset in datasets:
            dataset.close_training_cache()
    result = {
        'mode': args.mode, 'layout': args.layout, 'channel_order': channels,
        'dataset_fingerprint': release.metadata['dataset_fingerprint'],
        'source_bytes': release.data_path.stat().st_size,
        'cache_path': str(cache), 'cache_bytes': cache.stat().st_size,
        'validation_hit': release.verification_cached, 'task_split': split,
        'validation_seconds': validation, 'normalization_seconds': normalization_seconds,
        'cache_seconds': cache_seconds, 'attach_read_seconds': time.perf_counter() - stage,
        'total_seconds': time.perf_counter() - total,
        'scope': 'preparation only; no model construction, training, upload, or formal service reload',
    }
    atomic_write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
