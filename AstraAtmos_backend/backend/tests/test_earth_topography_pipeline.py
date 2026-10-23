"""Actual terrain-consuming optimization, strict reload, backtest and diagnostics."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.training_scripts import earth_daily as runner
from services.earth_derived_inputs import freeze_topography_for_reference, topography_for_reference
from services.earth_diagnostics import compute_earth_diagnostics, EarthDiagnosticsCache
from services.earth_prediction_service import run_earth_prediction
from services.earth_prediction_cache import EarthPredictionCache
from services.earth_task_split import build_earth_task_split
from services.earth_training_contract import build_earth_training_spec
from services.earth_training_artifact import load_earth_training_artifact, build_earth_model_from_checkpoint
from test_earth_3hourly_training_runner import synthetic_training_release, registry_for
from test_earth_3hourly_uploaded import package_at
from test_earth_topography_checkpoint import terrain_asset


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def template_reference(tmp_path, template):
    source = (Path(__file__).resolve().parents[3] / "docs" / template).read_text(encoding="utf-8")
    package, validation = package_at(tmp_path, source)
    assert validation.ok, validation.errors
    report = validation.earth_compatibilities["earth_merra2_3hourly_v1"]
    reference = {"package_id": package.id, "display_name": package.display_name, "version": 1,
                 "content_hash": package.content_hash, "source_path": package.storage_path,
                 "source_text": source, "param_schema": {}, "custom_model_params": {},
                 "contract_schema": report["contract_schema"]}
    return reference, report


@pytest.mark.parametrize("template", ["earth-3hourly-topography-model-template.py",
                                     "earth-3hourly-fullgrid-topography-model-template.py"])
def test_upload_train_save_reload_backtest_diagnostic_chain(
    terrain_asset, synthetic_training_release, tmp_path, monkeypatch, template,
):
    asset, _ = terrain_asset
    import services.earth_topography as terrain
    monkeypatch.setattr(terrain, "EARTH_MERRA2_3HOURLY_DIR", synthetic_training_release.data_path.parent)
    reference, report = template_reference(tmp_path, template)
    assert report["topography_identity"] == asset.identity
    registry = registry_for(synthetic_training_release)
    binding = registry.build_training_binding("earth_merra2_3hourly_v1")
    ratios = {name + "_ratio": 1 / 3 for name in ("train", "validation", "test")}
    split = build_earth_task_split(synthetic_training_release.dates, 56, 24, ratios)
    frozen = freeze_topography_for_reference(reference, synthetic_training_release)
    spec = build_earth_training_spec(task_id=6901, dataset_binding=binding,
        hyperparameters={**ratios, "epochs": 1, "batch_size": 1 if "fullgrid" in template else 7,
                         "seed": 42, "selected_channels": ["U10M"]},
        task_split=split, uploaded_model=reference, topography=frozen)
    path = tmp_path / "terrain-trained.pth"
    result = runner.run_training(spec, path, registry, device="cpu", cache_root=tmp_path / "cache")
    assert result["epochs_completed"] == 1
    checkpoint = load_earth_training_artifact(path)
    assert checkpoint.model_config["topography_identity"] == frozen["identity"]
    assert checkpoint.training_contract["topography_identity"] == frozen["identity"]
    assert checkpoint.normalization["channel_order"] == ["TO3", "U10M"]
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    assert model.terrain_scale.requires_grad
    task = SimpleNamespace(id=6901, status="completed", user_id=1, output_model_path=str(path), **binding,
        hyperparameters=json.dumps({**spec["hyperparameters"], "_earth_task_split": split,
                                   "_earth_topography": frozen}))
    import services.earth_prediction_service as prediction
    observed = []
    real_forward = prediction.earth_forward_for_model
    def record(model, inputs, **kwargs):
        observed.append(kwargs["topography"].detach().cpu().numpy().copy())
        return real_forward(model, inputs, **kwargs)
    monkeypatch.setattr(prediction, "earth_forward_for_model", record)
    monkeypatch.setattr(prediction, "_field_list", lambda value: [{"shape": list(value.shape)}])
    cache = EarthPredictionCache()
    forecast = run_earth_prediction(task, "2020-01-27T22:30:00Z", registry, device="cpu", cache=cache)
    assert forecast["origin_split"] == "test" and forecast["horizon"] == 24
    if "fullgrid" in template:
        assert len(observed) == 1
        np.testing.assert_array_equal(observed[0][0, 0], asset.elevation)
    else:
        assert len(observed) == 100
        for index, tensor in enumerate(observed):
            row, column = divmod(index, 10)
            np.testing.assert_array_equal(tensor[0, 0], asset.elevation[row*24:(row+1)*24, column*48:(column+1)*48])
    diagnostic = compute_earth_diagnostics(task, registry, sample_windows=1, scatter_points=7,
        include_pfi=True, device="cpu", cache=EarthDiagnosticsCache())
    assert diagnostic["status"] == "completed"
    assert diagnostic["histogram"]["valid_points"] == 24 * 240 * 480
    assert diagnostic["pfi"]["reason_code"] == "earth_pfi_insufficient_windows"
    # A corrupt asset must fail even when a valid prior prediction is cached.
    broken = copy.deepcopy(frozen)
    broken["identity"]["source_files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frozen identity"):
        topography_for_reference(reference, synthetic_training_release, broken)


def test_loader_shuffled_mixed_locations_partial_batch_and_repeated_indices(terrain_asset):
    asset, _ = terrain_asset
    class Windows:
        def __len__(self):
            return 1
        def read_window(self, index, lat_slice=None, lon_slice=None):
            return np.zeros((2, 1, 24, 48), np.float32), np.zeros((1, 1, 24, 48), np.float32)
        def read_windows(self, requests):
            return [self.read_window(index, lat, lon) for index, lat, lon in requests]
    dataset = runner._SpatialTileDataset(Windows(), topography_asset=asset)
    indices = [91, 2, 46, 2, 99]
    samples = dataset.__getitems__(indices)
    for index, sample in zip(indices, samples):
        row, column = divmod(index, 10)
        np.testing.assert_array_equal(sample["topography"][0].numpy(),
            asset.elevation[row*24:(row+1)*24, column*48:(column+1)*48])
    loader = torch.utils.data.DataLoader(dataset, batch_size=7, shuffle=True,
                                        generator=torch.Generator().manual_seed(9))
    batches = list(loader)
    assert len(batches[-1]["topography"]) == 2
    fingerprints = sorted(float(value) for batch in batches for value in batch["topography"][:, 0, 0, 0])
    assert fingerprints == sorted(float(asset.elevation[row, column])
        for row in range(0, 240, 24) for column in range(0, 480, 48))


@pytest.mark.parametrize("full_grid", [False, True])
def test_pfi_keeps_recipient_location_terrain_and_season(terrain_asset, monkeypatch, full_grid):
    """Permutations change only the chosen weather channel, including v2 reads."""
    from test_earth_diagnostics import install_synthetic_diagnostics
    from training_backbones.earth_3hourly_uploaded_contract import FULL_GRID_CONTRACT_SCHEMA
    asset, _ = terrain_asset
    service, task, bundle, dataset, checkpoint, _ = install_synthetic_diagnostics(monkeypatch)
    subset = SimpleNamespace(elevation=asset.elevation[:2, :4])
    checkpoint.model_config["derived_inputs"] = ["ls", "topography"]
    dataset.window_longitudes = lambda indices: np.repeat(np.asarray(indices, np.float32)[:, None], 2, axis=1)
    monkeypatch.setattr(service, "topography_for_checkpoint", lambda *_: subset)
    if full_grid:
        checkpoint.training_contract["uploaded_model_schema"] = FULL_GRID_CONTRACT_SCHEMA
        tile_read = service._read_tile
        def global_read(ds, dataset, indices):
            pairs = [tile_read(ds, dataset, indices, lat, lon) for lat in range(2) for lon in (0, 2)]
            inputs = np.concatenate([np.concatenate([pairs[row*2+col][0] for col in range(2)], -1)
                                     for row in range(2)], -2)
            references = np.concatenate([np.concatenate([pairs[row*2+col][1] for col in range(2)], -1)
                                        for row in range(2)], -2)
            return inputs, references
        monkeypatch.setattr(service, "_read_full_grid", global_read)
    observed = []
    def checked(model, inputs, cp, device, longitude=None, topography=None):
        assert longitude is not None and topography is not None
        # U10M encodes original recipient index and tile position. For TO3
        # permutations it remains untouched and identifies the receiver.
        for row in range(len(inputs)):
            season = float(longitude[row, 0])
            marker = float(inputs[row, -1, 1, 0, 0]) - (100 + season * 100)
            if full_grid:
                np.testing.assert_array_equal(topography[row, 0].numpy(), subset.elevation)
            elif marker in (0, 4, 20, 24):
                lat, lon = int(marker // 20), int((marker % 20) / 2)
                np.testing.assert_array_equal(topography[row, 0].numpy(), subset.elevation[lat:lat+1, lon:lon+2])
        observed.append((inputs.copy(), longitude.clone(), topography.clone()))
        return inputs[:, -1:, :1].astype("float64") + topography[:, None].numpy() / 1000
    monkeypatch.setattr(service, "_predict_tile", checked)
    result = compute_earth_diagnostics(task, bundle.registry, sample_windows=2,
        scatter_points=4, include_pfi=True, pfi_repeats=1, seed=17,
        device="cpu", cache=EarthDiagnosticsCache())
    assert result["pfi"]["status"] == "completed"
    assert [item["channel"] for item in result["pfi"]["items"]] == ["TO3", "U10M"]
    assert len(observed) >= 8
