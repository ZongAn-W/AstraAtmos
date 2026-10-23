"""OpenMARS files can contain multiple actual Mars-year segments."""

import json
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import torch

from services.mars_checkpoint import build_mars_checkpoint
from services.mars_data_service import (
    MarsDataError, _discover_year_segments, _openmars_year_segments,
    identity_snapshot, load_mars_arrays, prepare_scaled_volume, resolve_forecast_window,
)


RATIOS = {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}
CROSS_YEAR_NAME = "openmars_ozo_my27_ls358_my28_ls13.nc"


def _write_openmars(path, ls, *, ozone_count=None):
    ls = np.asarray(ls, dtype=np.float32)
    count = len(ls) if ozone_count is None else ozone_count
    field = np.arange(count * 6, dtype=np.float32).reshape(count, 2, 3)
    with netCDF4.Dataset(str(path), "w") as dataset:
        for name, size in (("time", count), ("ls_time", len(ls)), ("lat", 2), ("lon", 3)):
            dataset.createDimension(name, size)
        dataset.createVariable("lat", "f4", ("lat",))[:] = [-45, 45]
        dataset.createVariable("lon", "f4", ("lon",))[:] = [-120, 0, 120]
        dataset.createVariable("Ls", "f4", ("ls_time",))[:] = ls
        dataset.createVariable("o3col", "f4", ("time", "lat", "lon"))[:] = field
    return field


@pytest.fixture
def openmars_data(tmp_path):
    directory = tmp_path / "openmars"
    directory.mkdir()
    mcd_dir = tmp_path / "mcd"
    mcd_dir.mkdir()
    ls = np.array([358, 359, *range(10)], dtype=np.float32)
    source = directory / CROSS_YEAR_NAME
    field = _write_openmars(source, ls)
    return SimpleNamespace(
        directory=directory, source=source, ls=ls, field=field,
        kwargs=dict(training_dataset="openmars_mcd", openmars_dir=directory,
                    mcd_dir=mcd_dir, raw_dir=tmp_path / "unused_raw", selected_channels=[]),
    )


def _volume(data, window=1, horizon=1):
    return prepare_scaled_volume(**data.kwargs, window=window, horizon=horizon, split_ratios=RATIOS)


def test_cross_year_file_retains_both_global_and_file_ranges(openmars_data):
    segments = _discover_year_segments(openmars_data.directory, "openmars")
    assert [(s.mars_year, s.start, s.end, s.file_start, s.file_end) for s in segments] == [
        (27, 0, 2, 0, 2), (28, 2, 12, 2, 12),
    ]
    assert all(segment.source_file == openmars_data.source for segment in segments)
    data = load_mars_arrays(**openmars_data.kwargs)
    assert [(b.mars_year, b.start, b.end) for b in data.blocks] == [(27, 0, 2), (28, 2, 12)]
    assert [block.segments[0] for block in data.blocks] == list(segments)
    assert all(block.source_files == (CROSS_YEAR_NAME,) for block in data.blocks)
    np.testing.assert_array_equal(data.ls, openmars_data.ls)
    np.testing.assert_array_equal(data.ozone, openmars_data.field)
    np.testing.assert_array_equal(np.concatenate([b.values["ozone"] for b in data.blocks]), openmars_data.field)


@pytest.mark.parametrize("window,horizon", [(1, 1), (2, 1), (1, 3), (2, 2), (3, 2)])
def test_complete_windows_are_generated_on_the_merged_timeline(openmars_data, window, horizon):
    volume = _volume(openmars_data, window, horizon)
    expected = list(range(len(openmars_data.ls) - window - horizon + 1))
    assert volume.sample_starts.tolist() == expected
    assert volume.sample_mars_years == tuple(27 if start < 2 else 28 for start in expected)
    assert [(b.mars_year, b.start, b.end) for b in volume.blocks] == [(27, 0, 2), (28, 2, 12)]
    assert volume.values.shape[0] == len(openmars_data.ls)
    np.testing.assert_array_equal(volume.ls, openmars_data.ls)
    for start, year in zip(volume.sample_starts, volume.sample_mars_years):
        assert year == (27 if start < 2 else 28)
        assert len(volume.ls[start:start + window + horizon]) == window + horizon


@pytest.mark.parametrize("requested,window,horizon,index,year,input_ls,target_ls", [
    (358, 1, 1, 0, 27, [358], [359]),
    (359, 1, 1, 1, 27, [359], [0]),
    (0, 1, 1, 2, 28, [0], [1]),
    (360, 1, 1, 2, 28, [0], [1]),
    (359, 2, 2, 1, 27, [359, 0], [1, 2]),
])
def test_resolver_selects_actual_year_and_crossing_labels(openmars_data, requested, window, horizon, index, year, input_ls, target_ls):
    volume = _volume(openmars_data, window, horizon)
    result = resolve_forecast_window(
        volume.ls, requested, window, horizon,
        candidate_starts=volume.sample_starts, candidate_mars_years=volume.sample_mars_years,
        candidate_blocks=volume.blocks,
    )
    assert result == dict(sample_index=index, block_index=0 if year == 27 else 1,
                          mars_year=year, input_ls=input_ls, target_ls=target_ls)


def test_real_block_indices_are_preserved_for_multiple_files_of_one_year(openmars_data):
    _write_openmars(openmars_data.directory / "openmars_ozo_my27_ls350_my27_ls357.nc", range(350, 358))
    volume = _volume(openmars_data)
    assert [(b.mars_year, b.start, b.end) for b in volume.blocks] == [(27, 0, 8), (27, 8, 10), (28, 10, 20)]
    result = resolve_forecast_window(volume.ls, 358, 1, 1, volume.sample_starts,
                                     volume.sample_mars_years, volume.blocks)
    assert result["mars_year"] == 27 and result["block_index"] == 1
    assert result["sample_index"] == 8
    segments = _discover_year_segments(openmars_data.directory, "openmars")
    assert [(s.start, s.end, s.file_start, s.file_end) for s in segments[1:]] == [(8, 10, 0, 2), (10, 20, 2, 12)]
    snapshot = identity_snapshot(**openmars_data.kwargs, window=1, horizon=1,
                                 split_ratios=RATIOS, normalization=volume)
    assert [b["window_count"] for b in snapshot["year_blocks"]] == [8, 2, 9]


def test_resolver_accepts_an_explicit_cross_year_candidate(openmars_data):
    volume = _volume(openmars_data)
    result = resolve_forecast_window(volume.ls, 359, 1, 1, [1], [27], volume.blocks)
    assert result == dict(sample_index=1, block_index=0, mars_year=27,
                          input_ls=[359.0], target_ls=[0.0])


def test_resolver_accepts_a_candidate_crossing_files_within_the_same_year(openmars_data):
    _write_openmars(openmars_data.directory / "openmars_ozo_my27_ls350_my27_ls357.nc", range(350, 358))
    volume = _volume(openmars_data)
    result = resolve_forecast_window(volume.ls, 357, 1, 1, [7], [27], volume.blocks)
    assert result == dict(sample_index=7, block_index=0, mars_year=27,
                          input_ls=[357.0], target_ls=[358.0])


def test_resolver_rejects_year_labels_that_disagree_with_blocks(openmars_data):
    volume = _volume(openmars_data)
    with pytest.raises(MarsDataError, match="actual data blocks"):
        resolve_forecast_window(volume.ls, 0, 1, 1, volume.sample_starts,
                                 [27] * len(volume.sample_starts), volume.blocks)


@pytest.mark.parametrize("name,ls,reason", [
    ("openmars_my27_my29.nc", [358, 359, 0, 1], "filename Mars years disagree"),
    ("openmars_my27_my28.nc", [0, 1, 2, 3], "filename Mars years disagree"),
    ("openmars_my27_my27.nc", [358, 359, 0, 1], "filename Mars years disagree"),
    ("openmars_my27.nc", [0, float("nan"), 2], "finite"),
    ("openmars_my27.nc", [0, 361, 2], "0..360"),
    ("openmars_my27.nc", [100, 99, 101], "backwards without a year wrap"),
])
def test_ambiguous_or_invalid_openmars_year_metadata_fails(tmp_path, name, ls, reason):
    _write_openmars(tmp_path / name, ls)
    with pytest.raises(MarsDataError, match=reason):
        _discover_year_segments(tmp_path, "openmars")


def test_single_year_name_can_anchor_an_observed_wrap(tmp_path):
    _write_openmars(tmp_path / "openmars_my27.nc", [358, 359, 0, 1])
    assert [s.mars_year for s in _discover_year_segments(tmp_path, "openmars")] == [27, 28]


def test_multiple_observed_wraps_create_multiple_segments(tmp_path):
    _write_openmars(tmp_path / "openmars_my27_my29.nc", [358, 359, 0, 1, 358, 359, 0, 1])
    assert [(s.mars_year, s.file_start, s.file_end) for s in _discover_year_segments(tmp_path, "openmars")] == [
        (27, 0, 2), (28, 2, 6), (29, 6, 8),
    ]


def test_each_openmars_file_must_have_matching_ls_and_ozone_lengths(tmp_path):
    _write_openmars(tmp_path / "openmars_my27.nc", [0, 1, 2], ozone_count=4)
    with pytest.raises(MarsDataError, match="Ls length"):
        _discover_year_segments(tmp_path, "openmars")


@pytest.mark.parametrize("dataset_name,variable_name", [("mcd_overview", "o3col"), ("mcd_overview_raw", "O3COL")])
def test_mcd_segments_continue_to_use_one_year_per_file(tmp_path, dataset_name, variable_name):
    path = tmp_path / "MCD_MY24.nc"
    with netCDF4.Dataset(str(path), "w") as dataset:
        dataset.createDimension("time", 2)
        dataset.createDimension("hour", 3)
        dataset.createDimension("lat", 1)
        dataset.createDimension("lon", 1)
        dimensions = ("time", "hour", "lat", "lon") if variable_name == "O3COL" else ("time", "lat", "lon")
        dataset.createVariable(variable_name, "f4", dimensions)[:] = 1
    segments = _discover_year_segments(tmp_path, dataset_name)
    count = 6 if variable_name == "O3COL" else 2
    assert len(segments) == 1
    assert (segments[0].mars_year, segments[0].start, segments[0].end, segments[0].file_end) == (24, 0, count, count)


def test_checkpoint_and_snapshot_preserve_source_segment_provenance(openmars_data):
    volume = _volume(openmars_data)
    payload = build_mars_checkpoint(model_state_dict={"bias": torch.zeros(1)}, volume=volume,
                                    selected_channels=[], window=1, horizon=1, split_ratios=RATIOS, seed=11)
    snapshot = identity_snapshot(**openmars_data.kwargs, window=1, horizon=1,
                                 split_ratios=RATIOS, normalization=volume)
    assert [block["segments"] for block in payload["data_binding"]["year_blocks"]] == [
        block["segments"] for block in snapshot["year_blocks"]
    ]
    assert [block["segments"][0]["file_start"] for block in snapshot["year_blocks"]] == [0, 2]
    assert [block["window_count"] for block in snapshot["year_blocks"]] == [2, 9]


@pytest.mark.parametrize("model_source", ["official", "uploaded", "legacy_official"])
@pytest.mark.parametrize("window,horizon,requested,year,block,index,input_ls,target_ls", [
    (1, 1, 358, 27, 0, 0, [358], [359]),
    (1, 1, 0, 28, 1, 2, [0], [1]),
    (2, 2, 359, 27, 0, 1, [359, 0], [1, 2]),
])
def test_inference_paths_use_split_blocks_and_real_ls(openmars_data, tmp_path, monkeypatch, model_source, window, horizon, requested, year, block, index, input_ls, target_ls):
    from services.inference_service import InferenceService

    volume = _volume(openmars_data, window, horizon)
    state = {"bias": torch.zeros(1)}
    payload = build_mars_checkpoint(model_state_dict=state, volume=volume, selected_channels=[],
                                    window=window, horizon=horizon, split_ratios=RATIOS, seed=11)
    path = tmp_path / "checkpoint.pth"
    torch.save(state if model_source == "legacy_official" else payload, path)
    source = tmp_path / "probe.source"
    source.write_text('''
import torch
from torch import nn
MODEL_SPEC = {"name": "YearBoundaryProbe", "parameters": {}}
class Probe(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.horizon = horizon
    def forward(self, x):
        return x[:, -1:, :1].repeat(1, self.horizon, 1, 1, 1) + self.bias
def build_model(config):
    return Probe(config["horizon"])
''', encoding="utf-8")
    hypers = dict(window=window, horizon=horizon, selected_channels=[], model_architecture="dlinear",
                  training_dataset="openmars_mcd", _uploaded_model_path=str(source), **RATIOS)
    task = SimpleNamespace(id=42, output_model_path=str(path), custom_model_name="probe",
                           model_source="uploaded" if model_source == "uploaded" else "official",
                           model_script="demo3.py", hyperparameters=json.dumps(hypers))
    class OfficialProbe(torch.nn.Module):
        def forward(self, inputs, ls):
            return inputs[:, -1:, :1].repeat(1, horizon, 1, 1, 1)
    service = InferenceService()
    service.device = torch.device("cpu")
    monkeypatch.setattr(service, "_load_official_task_model", lambda **kwargs: (OfficialProbe(), False))
    if model_source == "legacy_official":
        monkeypatch.setattr(service, "_is_legacy_official_state_dict", lambda state: True)
    result = service._predict_task_with_context(task, hypers, requested, horizon, data_dirs={
        "ASTRAATMOS_OPENMARS_DIR": str(openmars_data.directory),
        "ASTRAATMOS_MCD_DIR": str(openmars_data.kwargs["mcd_dir"]),
    })
    assert result["model_info"]["mars_year"] == year
    assert result["model_info"]["block_index"] == block
    assert result["model_info"]["sample_index"] == index
    assert result["input_ls_values"] == input_ls
    assert result["ls_values"] == target_ls
    np.testing.assert_allclose(result["ground_truth"][0]["field"], openmars_data.field[index + window], atol=1e-5)


@pytest.mark.parametrize("dataset_id", ["openmars_mcd", "mcd_overview"])
def test_window_policy_invalidates_mars_caches(openmars_data, tmp_path, monkeypatch, dataset_id):
    from services import prediction_analysis_cache as analysis_cache
    from services import prediction_volume_cache as volume_cache

    path = tmp_path / "weight.pth"
    path.write_bytes(b"fixture")
    task = SimpleNamespace(id=42, output_model_path=str(path), dataset_id=dataset_id,
                           hyperparameters=json.dumps({"training_dataset": dataset_id}))
    directories = {"ASTRAATMOS_OPENMARS_DIR": str(openmars_data.directory),
                   "ASTRAATMOS_MCD_DIR": str(openmars_data.kwargs["mcd_dir"])}
    signature_kwargs = dict(openmars_dir=openmars_data.directory, mcd_dir=openmars_data.kwargs["mcd_dir"],
                            selected_channels=[], training_dataset=dataset_id)
    current_fingerprint = analysis_cache.build_artifact_fingerprint(task, directories)
    current_signature = volume_cache.volume_signature(**signature_kwargs)
    monkeypatch.setattr(analysis_cache, "OPENMARS_WINDOW_POLICY", "old_file_year_policy")
    monkeypatch.setattr(analysis_cache, "MCD_WINDOW_POLICY", "old_file_year_policy")
    monkeypatch.setattr(volume_cache, "OPENMARS_WINDOW_POLICY", "old_file_year_policy")
    monkeypatch.setattr(volume_cache, "MCD_WINDOW_POLICY", "old_file_year_policy")
    previous_fingerprint = analysis_cache.build_artifact_fingerprint(task, directories)
    previous_signature = volume_cache.volume_signature(**signature_kwargs)
    assert current_fingerprint != previous_fingerprint
    assert current_signature != previous_signature


def test_real_cross_year_openmars_file_metadata_when_available():
    from config import OPENMARS_DIR

    path = Path(OPENMARS_DIR) / CROSS_YEAR_NAME
    if not path.is_file():
        pytest.skip("Project OpenMARS boundary file is unavailable")
    from services.netcdf_read_lock import netcdf_read_lock
    with netcdf_read_lock(), netCDF4.Dataset(str(path)) as dataset:
        ls = np.asarray(dataset.variables["Ls"][:], dtype=np.float32).reshape(-1)
        count = dataset.variables["o3col"].shape[0]
    assert len(ls) == count
    segments = _openmars_year_segments(path, ls, 0)
    assert [segment.mars_year for segment in segments] == [27, 28]
    assert segments[0].file_start == 0 and segments[-1].file_end == count
    boundary = segments[1].file_start
    assert ls[boundary - 1] > 350 and ls[boundary] < 10
    assert sum(segment.end - segment.start for segment in segments) == count
