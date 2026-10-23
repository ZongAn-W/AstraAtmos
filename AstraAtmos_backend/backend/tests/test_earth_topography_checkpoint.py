"""Terrain checkpoint freeze, scientific identity guards and executable reload.

All fixtures use a non-symmetric non-constant synthetic field. The metadata names
the production protocol, but the synthetic source hashes never claim that a real
ETOPO source was downloaded or a real forecast was validated.
"""

import copy
import hashlib

import netCDF4
import numpy as np
import pytest
import torch

import services.earth_topography as terrain
from services.earth_model_source import build_uploaded_earth_model
from services.earth_season import SEASON_ALGORITHM
from services.earth_training_artifact import (
    EarthArtifactError, build_checkpoint_payload, build_earth_model_from_checkpoint,
    clear_earth_runtime_model_cache, load_earth_training_artifact,
    save_earth_artifact_atomic, validate_checkpoint_payload,
    verify_earth_model_reload_isolated,
)
from services.netcdf_read_lock import netcdf_read_lock
from test_earth_3hourly_artifact import _binding, _payload
from training_backbones.earth_3hourly_uploaded_contract import (
    CONTRACT_SCHEMA, DATASET_ID, DERIVED_INPUT_SPECS, FEED,
    FULL_GRID_CONTRACT_SCHEMA, FULL_GRID_FEED,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def terrain_asset(tmp_path, monkeypatch):
    lat = -89.625 + np.arange(240) * .75
    lon = -179.625 + np.arange(480) * .75
    lb = np.stack((lat - .375, lat + .375), axis=1)
    ob = np.stack((lon - .375, lon + .375), axis=1)
    elevation = np.asarray(100 + np.arange(240)[:, None] * 4.1
                           + np.arange(480)[None, :] * .7, dtype=np.float32)
    elevation[0, 0] = -17.5
    grid = (lat, lon, lb, ob)
    metadata = {
        "schema": terrain.ASSET_SCHEMA, "source": "NOAA NCEI ETOPO 2022",
        "product": "ETOPO_2022_v1", "version": "2022_v1", "variant": "ice_surface",
        "source_resolution_arc_seconds": 60,
        "source_files": [{"name": "synthetic-etopo.nc", "bytes": 124,
                          "sha256": "b" * 64, "url": "https://example.com/synthetic-etopo.nc"}],
        "land_mask": {"source": "synthetic independent land mask", "version": "test_v1",
                      "sha256": "c" * 64, "url": "https://example.com/synthetic-land.nc",
                      "method": "independent_binary_land_at_source_pixel_centres_v1"},
        "horizontal_datum": "WGS84 EPSG:4326", "vertical_datum": "EGM2008 geoid EPSG:3855",
        "ocean_policy": terrain.OCEAN_POLICY, "ice_policy": terrain.ICE_POLICY,
        "regrid_algorithm": terrain.REGRID_ALGORITHM,
    }
    path = tmp_path / "synthetic-terrain.nc"
    with netcdf_read_lock(), netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("latitude", 240)
        ds.createDimension("longitude", 480)
        ds.createDimension("bounds", 2)
        for name, dims, values, units in (
            ("latitude", ("latitude",), lat, "degrees_north"),
            ("longitude", ("longitude",), lon, "degrees_east"),
            ("lat_bounds", ("latitude", "bounds"), lb, "degrees_north"),
            ("lon_bounds", ("longitude", "bounds"), ob, "degrees_east"),
            ("elevation", ("latitude", "longitude"), elevation, "meter"),
        ):
            var = ds.createVariable(name, "f4" if name == "elevation" else "f8", dims)
            var.units = units
            var[:] = values
        ds.schema = terrain.ASSET_SCHEMA
        ds.provenance_json = terrain.canonical_json(metadata)
        ds.asset_content_sha256 = terrain.asset_content_identity(elevation, grid, metadata)
        ds.target_grid_sha256 = terrain.grid_identity(*grid)
    monkeypatch.setattr(terrain, "EARTH_TOPOGRAPHY_PATH", path)
    monkeypatch.setenv("ASTRAATMOS_EARTH_TOPOGRAPHY_PATH", str(path))
    asset = terrain.load_earth_topography(*grid)
    return asset, path


def _source(declared, schema):
    spec = {"name": "CheckpointTerrain", "parameters": {},
            "derived_inputs": {name: copy.deepcopy(DERIVED_INPUT_SPECS[name]) for name in declared},
            "datasets": {DATASET_ID: copy.deepcopy(
                FULL_GRID_FEED if schema == FULL_GRID_CONTRACT_SCHEMA else FEED)}}
    arguments = ", ".join(["x", *declared])
    seasonal = " + ls[:, -1:, None, None, None] * 0.001" if "ls" in declared else ""
    return (
        "import torch\nfrom torch import nn\n"
        f"MODEL_SPEC = {spec!r}\n"
        "class TerrainModel(nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        self.gain = nn.Parameter(torch.tensor(0.02))\n"
        "        self.horizon = config['horizon']\n"
        f"    def forward(self, {arguments}):\n"
        f"        output = x[:, -1:, :1] + self.gain * topography[:, None]{seasonal}\n"
        "        return output.repeat(1, self.horizon, 1, 1, 1)\n"
        "def build_model(config):\n    return TerrainModel(config)\n"
    )


def _terrain_payload(asset, *, combined=False, full_grid=False):
    base, _, _ = _payload(order=("TO3",))
    declared = ["ls", "topography"] if combined else ["topography"]
    schema = FULL_GRID_CONTRACT_SCHEMA if full_grid else CONTRACT_SCHEMA
    source = _source(declared, schema)
    reference = {"package_id": "synthetic-terrain-model", "display_name": "Checkpoint Terrain",
                 "version": 1, "content_hash": hashlib.sha256(source.encode()).hexdigest(),
                 "source_text": source, "param_schema": {}, "custom_model_params": {},
                 "contract_schema": schema}
    model, config, _ = build_uploaded_earth_model(
        reference=reference, input_channel_order=["TO3"], window=56, horizon=24,
        height=240, width=480, dataset_id=DATASET_ID, contract_schema=schema,
    )
    reference.update(build_config=config, input_channel_order=["TO3"])
    run = copy.deepcopy(base["run"])
    run["hyperparameters"] = {"batch_size": 1 if full_grid else 8}
    payload = build_checkpoint_payload(
        model=model, input_channel_order=["TO3"], linear_hidden_layers=2,
        dataset_binding=_binding(), normalization=base["normalization"], run=run,
        metrics=base["metrics"], split_window_counts=base["training_contract"]["split_window_counts"],
        split_ranges=base["training_contract"]["split_ranges"], task_id=22,
        model_source="uploaded", uploaded_model=reference,
        topography_identity=asset.identity,
        topography_grid={name: getattr(asset, name).tolist()
                         for name in ("latitude", "longitude", "lat_bounds", "lon_bounds")},
    )
    return payload


@pytest.mark.parametrize("combined", [False, True])
def test_declared_terrain_freezes_identity_grid_and_reloads_actual_consuming_model(
        terrain_asset, tmp_path, combined):
    asset, _ = terrain_asset
    payload = _terrain_payload(asset, combined=combined)
    assert payload["model_config"]["topography_identity"] == asset.identity
    assert payload["training_contract"]["topography_identity"] == asset.identity
    if combined:
        assert payload["model_config"]["season_algorithm"] == SEASON_ALGORITHM
    path = tmp_path / "terrain.pth"
    save_earth_artifact_atomic(payload, path)
    checkpoint = load_earth_training_artifact(path)
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    inputs = torch.zeros((2, 56, 1, 24, 48), dtype=torch.float32)
    geo = terrain.prepare_topography_batch(asset, inputs, [
        (slice(0, 24), slice(0, 48)), (slice(120, 144), slice(240, 288))])
    args = [torch.zeros(2, 56)] if combined else []
    first = model(inputs, *args, geo)
    second = model(inputs, *args, geo + 1)
    assert not torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_terrain_fullgrid_checkpoint_roundtrip(terrain_asset, tmp_path):
    asset, _ = terrain_asset
    payload = _terrain_payload(asset, full_grid=True)
    save_earth_artifact_atomic(payload, tmp_path / "fullgrid.pth")
    loaded = load_earth_training_artifact(tmp_path / "fullgrid.pth")
    assert loaded.model_config["spatial_tile_shape"] == [240, 480]


@pytest.mark.parametrize("kind", ["missing", "unknown", "algorithm", "mirror", "grid", "grid_missing"])
def test_terrain_identity_and_grid_tampering_fail_closed(terrain_asset, kind):
    asset, _ = terrain_asset
    payload = _terrain_payload(asset)
    if kind == "missing":
        payload["model_config"]["topography_identity"].pop("vertical_datum")
    elif kind == "unknown":
        payload["model_config"]["topography_identity"]["unknown"] = True
    elif kind == "algorithm":
        payload["model_config"]["topography_identity"]["regrid_algorithm"] = "invented_v9"
    elif kind == "mirror":
        payload["training_contract"]["topography_identity"]["source_files"][0]["sha256"] = "d" * 64
    elif kind == "grid":
        payload["model_config"]["topography_grid"]["longitude"][0] += .375
    elif kind == "grid_missing":
        payload["training_contract"].pop("topography_grid")
    with pytest.raises(EarthArtifactError):
        validate_checkpoint_payload(payload)


def test_current_asset_replacement_is_checked_even_before_cached_model_return(
        terrain_asset, tmp_path):
    asset, path = terrain_asset
    payload = _terrain_payload(asset)
    output = tmp_path / "terrain.pth"
    save_earth_artifact_atomic(payload, output)
    checkpoint = load_earth_training_artifact(output)
    clear_earth_runtime_model_cache()
    build_earth_model_from_checkpoint(checkpoint)
    with netcdf_read_lock(), netCDF4.Dataset(str(path), "r+") as ds:
        ds.variables["elevation"][0, 0] += np.float32(1)
    with pytest.raises(EarthArtifactError):
        build_earth_model_from_checkpoint(checkpoint)
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(output)


def test_task_terrain_identity_must_match_checkpoint(terrain_asset, tmp_path):
    asset, _ = terrain_asset
    payload = _terrain_payload(asset)
    path = tmp_path / "terrain.pth"
    save_earth_artifact_atomic(payload, path)
    hypers = {"training_dataset": DATASET_ID, "model_source": "uploaded",
              "_earth_topography": {"declaration": {"topography": DERIVED_INPUT_SPECS["topography"]},
                                    "identity": copy.deepcopy(asset.identity)}}
    load_earth_training_artifact(path, expected_hyperparameters=hypers)
    hypers["_earth_topography"]["identity"]["source_files"][0]["sha256"] = "d" * 64
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(path, expected_hyperparameters=hypers)
    with pytest.raises(EarthArtifactError):
        load_earth_training_artifact(path, expected_hyperparameters={"model_source": "uploaded"})


def test_source_declaration_cannot_disappear_from_frozen_checkpoint(terrain_asset, tmp_path):
    asset, _ = terrain_asset
    payload = _terrain_payload(asset)
    for block in ("model_config", "training_contract"):
        for key in ("derived_inputs", "topography_identity", "topography_grid"):
            payload[block].pop(key)
    path = tmp_path / "declaration-erased.pth"
    torch.save(payload, path)
    # Structural validation is followed by executed-source declaration proof.
    loaded = load_earth_training_artifact(path)
    with pytest.raises(EarthArtifactError, match="derived inputs"):
        build_earth_model_from_checkpoint(loaded)


def test_missing_terrain_does_not_change_undeclared_legacy_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(terrain, "EARTH_TOPOGRAPHY_PATH", tmp_path / "does-not-exist.nc")
    payload, _, _ = _payload()
    assert "topography_identity" not in payload["model_config"]
    assert "topography_grid" not in payload["training_contract"]
    save_earth_artifact_atomic(payload, tmp_path / "legacy.pth")
    build_earth_model_from_checkpoint(load_earth_training_artifact(tmp_path / "legacy.pth"))


def test_isolated_completion_gate_has_real_terrain_input(terrain_asset, tmp_path):
    asset, _ = terrain_asset
    path = tmp_path / "terrain-isolated.pth"
    save_earth_artifact_atomic(_terrain_payload(asset, combined=True), path)
    verify_earth_model_reload_isolated(path, timeout_seconds=45)
