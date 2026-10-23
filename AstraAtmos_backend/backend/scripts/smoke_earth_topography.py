"""Bounded real MERRA-2 + ETOPO upload/train/reload/backtest smoke.

Reads three complete days from the configured, verified release into a new
checkout-external package. Uses the ordinary registry, validator and runner;
never writes the application database, original release or existing weights.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import xarray as xr

from config import EARTH_MERRA2_3HOURLY_DIR, EARTH_TRAINING_CACHE_DIR
from models.training_scripts.earth_daily import run_training
from scripts.build_earth_merra2_3hourly import (
    DATA_FILE, _new_output, dataset_fingerprint, source_inventory_sha256,
)
from services.dataset_registry import DatasetRegistry
from services.earth_dataset import CHANNELS, EarthThreeHourlyWindows, validate_earth_3hourly_dataset
from services.earth_dataset_metadata import (
    file_sha256, package_signature, read_earth_3hourly_release,
)
from services.earth_derived_inputs import freeze_topography_for_reference
from services.earth_diagnostics import compute_earth_diagnostics, EarthDiagnosticsCache
from services.earth_model_source import earth_forward_for_model
from services.earth_prediction_cache import EarthPredictionCache
from services.earth_prediction_service import run_earth_prediction
from services.earth_task_split import build_earth_task_split, utc
from services.earth_topography import load_earth_topography, prepare_topography_batch
from services.earth_training_artifact import (
    build_earth_model_from_checkpoint, load_earth_training_artifact,
    verify_earth_model_reload_isolated,
)
from services.earth_training_contract import build_earth_training_spec
from services.netcdf_read_lock import netcdf_read_lock
from services.user_model_validator import UserModelValidator

DATASET_ID = "earth_merra2_3hourly_v1"


def create_subset(release, output):
    """Publish true source values/masks and a independently verified identity."""
    output.mkdir()
    manifest = copy.deepcopy(release.metadata["manifest"])
    first = date.fromisoformat(str(release.dates[0].astype("datetime64[D]")))
    days = [first + timedelta(days=index) for index in range(3)]
    day_names = {day.isoformat() for day in days}
    inventory = [entry for entry in manifest["source_files"] if entry["date"] in day_names]
    if len(inventory) != 6:
        raise ValueError("Three real days require six verified SLV/RAD source inventory entries")
    source_sha = source_inventory_sha256(inventory, 3)
    path = output / DATA_FILE
    with netcdf_read_lock(), xr.open_dataset(release.data_path, engine="netcdf4", mask_and_scale=False) as source:
        target = _new_output(path, days, source_sha)
        try:
            for name in ("lat", "lon", "lat_bounds", "lon_bounds"):
                target[name][:] = source[name].values
            for start in range(0, 24, 8):
                for name in CHANNELS:
                    for variable in (name, name + "_valid_mask"):
                        values = source[variable].isel(time=slice(start, start + 8)).values
                        target[variable][start:start + 8] = values
        finally:
            target.close()
    with netcdf_read_lock(), xr.open_dataset(path, engine="netcdf4", mask_and_scale=False) as subset:
        statistics = validate_earth_3hourly_dataset(subset)
        dimensions = dict(subset.sizes)
        start_time, end_time = utc(subset.time.values[0]), utc(subset.time.values[-1])
        interval = [utc(subset.time_bounds.values[0, 0]), utc(subset.time_bounds.values[-1, 1])]
    manifest.update({
        "build_mode": "subset", "fingerprint_rule": 2, "source_inventory_version": 3,
        "source_files": inventory, "source_sha256": source_sha,
        "source_fingerprint_basis": "sha256 of canonical source content identity inventory",
        "dimensions": dimensions, "time_start": start_time, "time_end": end_time,
        "variables": statistics, "data_bytes": path.stat().st_size, "data_sha256": file_sha256(path),
        "subset_parent": {"dataset_fingerprint": release.metadata["dataset_fingerprint"],
                          "data_sha256": release.metadata["data_sha256"],
                          "manifest_sha256": release.metadata["manifest_sha256"],
                          "source_time_indices": [0, 24]},
        "limitations": ["Isolated three-day real-data smoke subset; no forecast accuracy acceptance"],
    })
    manifest["source"].update(date_start=days[0].isoformat(), date_end=days[-1].isoformat(), daily_file_count=3)
    if "source_mtimes" in manifest:
        manifest["source_mtimes"] = {entry["path"]: manifest["source_mtimes"][entry["path"]] for entry in inventory}
    manifest["time_rule"].update(interval_start=interval[0], interval_end_exclusive=interval[1])
    manifest["splits"] = {"train": {"start": days[0].isoformat(), "end": days[-1].isoformat(), "days": 3, "steps": 24},
                          "validation": {"start": None, "end": None, "days": 0, "steps": 0},
                          "test": {"start": None, "end": None, "days": 0, "steps": 0}}
    manifest["dataset_fingerprint"] = dataset_fingerprint(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return read_earth_3hourly_release(output)


def run_model(root, registry, release, template, task_id):
    folder = root / template.removesuffix(".py")
    folder.mkdir()
    source = (Path(__file__).resolve().parents[3] / "docs" / template).read_bytes()
    model_path = folder / "model.py"
    model_path.write_bytes(source)
    validation = UserModelValidator().validate_file(model_path)
    if not validation.ok:
        raise ValueError(f"Real-asset isolated upload validation failed: {validation.errors}")
    proof = validation.earth_compatibilities[DATASET_ID]
    reference = {"package_id": "isolated-real-terrain-smoke", "display_name": template, "version": 1,
                 "source_text": source.decode("utf-8"), "source_path": str(model_path),
                 "content_hash": hashlib.sha256(source).hexdigest(), "param_schema": {},
                 "custom_model_params": {}, "contract_schema": proof["contract_schema"]}
    frozen = freeze_topography_for_reference(reference, release)
    if proof.get("topography_identity") != frozen["identity"]:
        raise ValueError("Upload proof and frozen task must use identical real terrain")
    full_grid = "fullgrid" in template
    ratios = {name + "_ratio": 1 / 3 for name in ("train", "validation", "test")}
    split = build_earth_task_split(release.dates, 7, 1, ratios)
    binding = registry.build_training_binding(DATASET_ID)
    spec = build_earth_training_spec(task_id=task_id, dataset_binding=binding,
        hyperparameters={**ratios, "window": 7, "horizon": 1, "epochs": 1, "seed": 42,
                         "batch_size": 1 if full_grid else 7, "selected_channels": ["U10M"]},
        task_split=split, uploaded_model=reference, topography=frozen)
    weights = folder / "trained.pth"
    result = run_training(spec, weights, registry, device="cpu", cache_root=root / "training-cache")
    task_hypers = {**spec["hyperparameters"], "_earth_task_split": split, "_earth_topography": frozen}
    checkpoint = load_earth_training_artifact(weights, expected_binding=binding,
        expected_hyperparameters=task_hypers, expected_task_id=task_id)
    verify_earth_model_reload_isolated(weights)
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    # A real-data forward must change when the same input receives zero terrain.
    dataset = EarthThreeHourlyWindows.from_release(release, split="test", window=7, horizon=1,
        selected_channels=["U10M"], normalization=checkpoint.normalization, task_split=split)
    lat_slice, lon_slice = (slice(None), slice(None)) if full_grid else (slice(120, 144), slice(300, 348))
    inputs, _ = dataset.read_window(0, lat_slice=lat_slice, lon_slice=lon_slice)
    tensor = torch.from_numpy(inputs[None])
    asset = load_earth_topography(release=release, expected_identity=frozen["identity"])
    terrain = prepare_topography_batch(asset, tensor, None if full_grid else [(lat_slice, lon_slice)])
    extra = {"longitude": torch.from_numpy(dataset.window_longitudes([0]))} if full_grid else {}
    with torch.no_grad():
        physical = earth_forward_for_model(model, tensor, model_source="uploaded", horizon=1,
                                          topography=terrain, **extra)
        zero = earth_forward_for_model(model, tensor, model_source="uploaded", horizon=1,
                                      topography=torch.zeros_like(terrain), **extra)
    difference = float((physical - zero).abs().max())
    gain = float(model.terrain_scale.detach())
    if difference <= 0 or gain == float(np.float32(.1)):
        raise ValueError("Real terrain must affect output and its trainable coefficient must update")
    task = SimpleNamespace(id=task_id, user_id=1, status="completed", output_model_path=str(weights),
                           hyperparameters=json.dumps(task_hypers), **binding)
    origin = utc(dataset.dates[6])
    forecast = run_earth_prediction(task, origin, registry, device="cpu", cache=EarthPredictionCache())
    diagnostic = compute_earth_diagnostics(task, registry, sample_windows=1, scatter_points=32,
        include_pfi=False, device="cpu", cache=EarthDiagnosticsCache())
    if forecast["origin_split"] != "test" or diagnostic["status"] != "completed":
        raise ValueError("Real historical forecast/diagnostic failed")
    record = {"template": template, "upload": "passed_spawn_isolation", "training": result,
              "checkpoint": str(weights), "strict_reload": "passed_including_spawn",
              "origin": origin, "backtest_horizon": forecast["horizon"],
              "diagnostic": diagnostic["status"], "terrain_gain": gain,
              "terrain_output_max_absolute_effect_normalized": difference,
              "topography_identity": frozen["identity"],
              "season_algorithm": checkpoint.model_config.get("season_algorithm"),
              "pfi": "not_run_one_test_window; covered_by_synthetic_position_regression"}
    (folder / "report.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=EARTH_MERRA2_3HOURLY_DIR)
    parser.add_argument("--output-dir", type=Path, required=True, help="new checkout-external directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parents[3]
    if output.exists() or output.is_relative_to(checkout) or output.is_relative_to(args.release.resolve()):
        parser.error("--output-dir must be a new directory outside checkout and the source release")
    output.mkdir(parents=True)
    before = package_signature(args.release, data_file_name=DATA_FILE)
    print("Verifying actual source release using the configured shared proof", flush=True)
    release = read_earth_3hourly_release(args.release, verification_cache_dir=EARTH_TRAINING_CACHE_DIR, progress=print)
    subset = create_subset(release, output / "release")
    registry = DatasetRegistry(output / "missing_daily", earth_3hourly_package_dir=subset.data_path.parent,
                               verification_cache_dir=output / "verification-cache")
    torch.set_num_threads(2)
    records = [run_model(output, registry, subset, template, index + 1) for index, template in enumerate((
        "earth-3hourly-topography-model-template.py", "earth-3hourly-fullgrid-topography-model-template.py"))]
    if before != package_signature(args.release, data_file_name=DATA_FILE):
        raise ValueError("Original published release changed during smoke")
    report = {"status": "passed", "date": "2026-10-13", "data": "real_MERRA2_values_and_real_ETOPO",
              "subset": {"shape": [24, 240, 480], "dataset_fingerprint": subset.metadata["dataset_fingerprint"],
                         "parent": release.metadata["dataset_fingerprint"]},
              "source_release_unchanged": True, "models": records,
              "limits": "Three days, 7->1 steps, 1 epoch, CPU; no prediction accuracy claim"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Real terrain smoke passed: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
