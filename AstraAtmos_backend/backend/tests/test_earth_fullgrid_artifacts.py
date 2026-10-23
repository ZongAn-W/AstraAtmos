"""Full-grid uploads keep checkpoint identity and never tile their model calls."""

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services.earth_model_source import EarthModelBuildError, build_uploaded_earth_model
from services.earth_training_artifact import (
    EarthArtifactError, build_checkpoint_payload, build_earth_model_from_checkpoint,
    build_metrics_block, compute_earth_metrics, load_earth_training_artifact,
    save_earth_artifact_atomic, validate_checkpoint_payload,
)
from test_earth_3hourly_artifact import DATASET_ID, SPLITS, _payload
from test_earth_3hourly_training_runner import synthetic_training_release
from training_backbones.earth_3hourly_uploaded_contract import (
    CONTRACT_SCHEMA, FEED, FULL_GRID_CONTRACT_SCHEMA, FULL_GRID_FEED,
    FULL_GRID_IMPLEMENTATION_ID, build_config,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _reference(schema=FULL_GRID_CONTRACT_SCHEMA):
    feed = copy.deepcopy(FULL_GRID_FEED if schema == FULL_GRID_CONTRACT_SCHEMA else FEED)
    feed.update(window=[2], horizon=[1])
    source = "import torch\nfrom torch import nn\nMODEL_SPEC = " + repr({
        "name": "GlobalContext", "parameters": {}, "datasets": {DATASET_ID: feed},
    }) + """
class GlobalContext(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))
        self.horizon = config['horizon']
        self.height, self.width = config['height'], config['width']
    def forward(self, x):
        if tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError('The full model received a cropped grid')
        latest = x[:, -1:, :1]
        return (latest + self.gain * latest.mean(dim=(-2, -1), keepdim=True)).expand(-1, self.horizon, -1, -1, -1)
def build_model(config):
    return GlobalContext(config)
"""
    return {
        "package_id": "global-model", "display_name": "GlobalContext", "version": 1,
        "content_hash": hashlib.sha256(source.encode()).hexdigest(), "source_text": source,
        "source_path": None, "param_schema": {}, "custom_model_params": {},
        "input_channel_order": ["TO3"],
        "build_config": build_config(["TO3"], window=2, horizon=1, contract_schema=schema),
    }


def _uploaded_payload(schema=FULL_GRID_CONTRACT_SCHEMA, *, batch_size=1):
    base, _, _ = _payload(("TO3",))
    reference = _reference(schema)
    model, _, _ = build_uploaded_earth_model(
        reference=reference, input_channel_order=["TO3"], window=2, horizon=1,
        height=240, width=480, dataset_id=DATASET_ID,
    )
    counts = {name: split["steps"] - 2 for name, split in SPLITS.items()}
    ranges = copy.deepcopy(base["training_contract"]["split_ranges"])
    for name in ranges:
        ranges[name]["window_count"] = counts[name]
    summaries = {}
    for name in ("validation", "test"):
        metrics = compute_earth_metrics(np.ones((1, 1, 1, 1, 1)), np.zeros((1, 1, 1, 1, 1)),
                                        dataset_id=DATASET_ID)
        metrics["by_lead"][0]["target_statistics"]["count"] = counts[name] * 240 * 480
        summaries[name] = metrics
    payload = build_checkpoint_payload(
        model=model, input_channel_order=["TO3"], linear_hidden_layers=2,
        dataset_binding=base["dataset_binding"], normalization=base["normalization"],
        run={**base["run"], "hyperparameters": {"window": 2, "horizon": 1, "batch_size": batch_size}},
        metrics=build_metrics_block(validation=summaries["validation"], test=summaries["test"],
            validation_window_count=counts["validation"], test_window_count=counts["test"],
            dataset_id=DATASET_ID, horizon=1),
        split_window_counts=counts, split_ranges=ranges, model_source="uploaded", uploaded_model=reference,
    )
    return payload


@pytest.mark.parametrize('batch_size', [2, 4])
def test_full_grid_checkpoint_roundtrip_uses_entire_saved_batch(tmp_path, monkeypatch, batch_size):
    import services.earth_training_artifact as artifact
    payload = _uploaded_payload(batch_size=batch_size)
    calls = []
    original = artifact._forward_payload_model

    def record(model, descriptor, probe):
        calls.append(tuple(probe.shape))
        return original(model, descriptor, probe)

    monkeypatch.setattr(artifact, "_forward_payload_model", record)
    path = tmp_path / "fullgrid.pth"
    save_earth_artifact_atomic(payload, path)
    checkpoint = load_earth_training_artifact(path)
    assert checkpoint.model_config["contract_schema"] == FULL_GRID_CONTRACT_SCHEMA
    assert checkpoint.model_config["implementation_id"] == FULL_GRID_IMPLEMENTATION_ID
    assert checkpoint.training_contract["spatial_tile_shape"] == [240, 480]
    assert checkpoint.uploaded_model["build_config"]["height"] == 240
    assert calls == [(batch_size, 2, 1, 240, 480)] * 2
    rebuilt, _ = build_earth_model_from_checkpoint(checkpoint)
    with torch.no_grad():
        output = rebuilt(torch.ones(1, 2, 1, 240, 480))
    assert output.shape == (1, 1, 1, 240, 480)
    torch.testing.assert_close(output, torch.full_like(output, 2))


@pytest.mark.parametrize("schema", [CONTRACT_SCHEMA, FULL_GRID_CONTRACT_SCHEMA])
@pytest.mark.parametrize("declaration_style", ["constant", "composed_dict"])
def test_nonliteral_declarations_survive_checkpoint_reload(tmp_path, schema, declaration_style):
    payload = _uploaded_payload(schema)
    reference = payload["model_ref"]["uploaded_model"]
    source = reference["source_text"]
    if declaration_style == "constant":
        source = source.replace("MODEL_SPEC = ", "SCHEMA = " + repr(schema) + "\nMODEL_SPEC = ", 1)
        source = source.replace("'schema': " + repr(schema), "'schema': SCHEMA", 1)
    else:
        source = source.replace("MODEL_SPEC = ", "DECLARED_SPEC = ", 1)
        source = source.replace("\nclass GlobalContext", "\nMODEL_SPEC = dict(DECLARED_SPEC)\nclass GlobalContext", 1)
    reference.update(source_text=source, content_hash=hashlib.sha256(source.encode()).hexdigest())
    assert reference["contract_schema"] == schema
    if schema == CONTRACT_SCHEMA:
        reference.pop("contract_schema")
    path = tmp_path / "nonliteral.pth"
    save_earth_artifact_atomic(payload, path)
    checkpoint = load_earth_training_artifact(path)
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    shape = (240, 480) if schema == FULL_GRID_CONTRACT_SCHEMA else (24, 48)
    with torch.no_grad():
        output = model(torch.ones(1, 2, 1, *shape))
    assert output.shape == (1, 1, 1, *shape)
    torch.testing.assert_close(output, torch.full_like(output, 2))


@pytest.mark.parametrize("field", ["shape", "implementation", "training_schema", "build_shape", "source_schema"])
def test_full_grid_checkpoint_rejects_contract_mismatches(field):
    payload = _uploaded_payload()
    if field == "shape":
        payload["model_config"]["spatial_tile_shape"] = [24, 48]
    elif field == "implementation":
        payload["model_ref"]["implementation_id"] = "astraatmos_earth_3hourly_uploaded_runner_v1"
    elif field == "training_schema":
        payload["training_contract"]["uploaded_model_schema"] = CONTRACT_SCHEMA
    elif field == "build_shape":
        payload["model_ref"]["uploaded_model"]["build_config"]["height"] = 24
    else:
        source = _reference(CONTRACT_SCHEMA)
        payload["model_ref"]["uploaded_model"].update(source_text=source["source_text"], content_hash=source["content_hash"])
    with pytest.raises(EarthArtifactError):
        validate_checkpoint_payload(payload)


def test_frozen_source_cannot_change_the_requested_spatial_contract():
    with pytest.raises(EarthModelBuildError, match="schema disagrees"):
        build_uploaded_earth_model(reference=_reference(CONTRACT_SCHEMA), input_channel_order=["TO3"],
            window=2, horizon=1, height=240, width=480, dataset_id=DATASET_ID,
            contract_schema=FULL_GRID_CONTRACT_SCHEMA)
    reference = _reference(CONTRACT_SCHEMA)
    reference["contract_schema"] = FULL_GRID_CONTRACT_SCHEMA
    with pytest.raises(EarthModelBuildError, match="schema disagrees"):
        build_uploaded_earth_model(reference=reference, input_channel_order=["TO3"],
            window=2, horizon=1, height=240, width=480, dataset_id=DATASET_ID)


@pytest.mark.parametrize("batch_size", [None, 65, True, 1.5])
def test_full_grid_checkpoint_requires_the_actual_supported_batch_size(batch_size):
    payload = _uploaded_payload()
    payload["run"]["hyperparameters"]["batch_size"] = batch_size
    with pytest.raises(EarthArtifactError, match="training batch size"):
        validate_checkpoint_payload(payload)


def test_full_grid_checkpoint_preserves_frozen_task_contract(tmp_path):
    payload = _uploaded_payload()
    path = tmp_path / "frozen-fullgrid.pth"
    torch.save(payload, path)
    with pytest.raises(EarthArtifactError, match="spatial execution contract"):
        load_earth_training_artifact(path, expected_hyperparameters={"model_source": "uploaded",
            "_earth_uploaded_contract_schema": CONTRACT_SCHEMA})
    with pytest.raises(EarthArtifactError, match="training batch size"):
        load_earth_training_artifact(path, expected_hyperparameters={"model_source": "uploaded", "batch_size": 2})


def test_v1_uploaded_checkpoint_keeps_tile_probe_and_dimensions(tmp_path, monkeypatch):
    import services.earth_training_artifact as artifact
    payload = _uploaded_payload(CONTRACT_SCHEMA)
    calls = []
    original = artifact._forward_payload_model
    monkeypatch.setattr(artifact, "_forward_payload_model",
        lambda model, descriptor, probe: (calls.append(tuple(probe.shape)), original(model, descriptor, probe))[1])
    save_earth_artifact_atomic(payload, tmp_path / "v1.pth")
    assert payload["training_contract"]["spatial_tile_shape"] == [24, 48]
    assert calls == [(2, 2, 1, 24, 48)] * 2


def test_full_grid_prediction_uses_one_model_call_and_global_context(monkeypatch):
    import services.earth_prediction_service as prediction
    from services.earth_prediction_cache import EarthPredictionCache
    payload = _uploaded_payload()
    checkpoint = SimpleNamespace(
        **{key: payload[key] for key in ("model_config", "model_ref", "dataset_binding", "training_contract", "normalization")},
        model_source="uploaded", uploaded_model=payload["model_ref"]["uploaded_model"],
        model_identity=lambda: {"model_source": "uploaded"},
    )
    dates = np.datetime64("2020-01-01T01:30") + np.arange(5) * np.timedelta64(3, "h")
    release = SimpleNamespace(dates=dates, latitude=np.linspace(-89.625, 89.625, 240),
                              longitude=np.linspace(-179.625, 179.625, 480))
    reference = _reference()
    model, _, _ = build_uploaded_earth_model(reference=reference, input_channel_order=["TO3"],
        window=2, horizon=1, height=240, width=480, dataset_id=DATASET_ID)
    inputs = np.ones((2, 1, 240, 480), dtype="float32")
    inputs[:, :, 0, 0] = 115201
    calls = []
    original_forward = prediction.earth_forward_for_model

    def record_forward(model, values, **kwargs):
        calls.append(tuple(values.shape))
        return original_forward(model, values, **kwargs)

    monkeypatch.setattr(prediction, "_threehour_checkpoint_sha", lambda _: "a" * 64)
    monkeypatch.setattr(prediction, "_load_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(prediction, "_sync_release", lambda *_: release)
    monkeypatch.setattr(prediction, "_validate_checkpoint_split_ranges", lambda *_: None)
    monkeypatch.setattr(prediction, "threehour_origin_index", lambda *_args, **_kw: 1)
    monkeypatch.setattr(prediction, "threehour_forecast_timestamps", lambda *_args, **_kw: ["2020-01-01T07:30:00Z"])
    monkeypatch.setattr(prediction, "threehour_origin_split", lambda *_args, **_kw: "test")
    monkeypatch.setattr(prediction, "_read_threehour_forecast_block", lambda *_args, **_kw: (inputs, np.zeros((1, 240, 480), dtype="float32")))
    monkeypatch.setattr(prediction, "build_earth_model_from_checkpoint", lambda _: (model, []))
    monkeypatch.setattr(prediction, "earth_forward_for_model", record_forward)
    monkeypatch.setattr(prediction, "_field_list", lambda values: np.asarray(values))
    result = prediction._run_threehour_prediction(SimpleNamespace(id=22), "origin", object(), device="cpu", cache=EarthPredictionCache())
    assert calls == [(1, 2, 1, 240, 480)]
    assert result["prediction"][0, -1, -1] == pytest.approx(310.)
    assert result["metrics"]["by_lead"][0]["target_statistics"]["count"] == 240 * 480


def test_full_grid_diagnostics_and_pfi_use_one_global_window_per_call(monkeypatch):
    from test_earth_diagnostics import install_synthetic_diagnostics
    from services.earth_diagnostics import EarthDiagnosticsCache
    service, task, bundle, dataset, checkpoint, calls = install_synthetic_diagnostics(monkeypatch)
    checkpoint.model_source = "uploaded"
    checkpoint.training_contract["uploaded_model_schema"] = FULL_GRID_CONTRACT_SCHEMA
    observed = []

    def read_full(_ds, _dataset, selected):
        assert len(selected) == 1
        values = np.array([1., 5., 9., 13.])
        inputs = np.zeros((1, 2, 2, 2, 4), dtype="float32")
        reference = np.full((1, 1, 1, 2, 4), values[selected[0]], dtype="float32")
        inputs[:, :, :1] = reference + 1.
        inputs[:, :, 1] = 100. + selected[0] * 100.
        return inputs, reference

    original_predict = service._predict_tile
    monkeypatch.setattr(service, "_read_full_grid", read_full)
    monkeypatch.setattr(service, "_read_tile", lambda *_: pytest.fail("Full-grid diagnostics must not read spatial tiles"))
    monkeypatch.setattr(service, "_predict_tile",
        lambda model, inputs, cp, device: (observed.append(tuple(inputs.shape)), original_predict(model, inputs, cp, device))[1])
    result = service.compute_earth_diagnostics(task, bundle.registry, sample_windows=2, scatter_points=7,
        histogram_bins=8, pfi_repeats=2, include_pfi=True, device="cpu", cache=EarthDiagnosticsCache())
    assert observed and all(shape == (1, 2, 2, 2, 4) for shape in observed)
    assert len(observed) == 4 + 2 * 2 * 2
    assert result["sample_metrics"]["overall"]["mae"] == pytest.approx(1.)
    assert sum(result["histogram"]["counts"]) == 16
    assert result["pfi"]["items"][0]["importance_mean"] > 0
    assert result["pfi"]["items"][1]["importance_mean"] == pytest.approx(0.)
    assert result["pfi"]["permutation"] == "whole_global_window_channel_derangement"
    assert result["scatter"]["sampling"]["stream_order"] == "window_lead_lat_lon"


def test_full_grid_diagnostic_read_matches_real_global_window(synthetic_training_release):
    from services.earth_dataset import EarthThreeHourlyWindows, fit_threehour_normalization
    from services.earth_diagnostics import _open_verified_dataset, _read_full_grid
    release = synthetic_training_release
    normalization = fit_threehour_normalization(release, ["TO3", "U10M"])
    dataset = EarthThreeHourlyWindows.from_release(release, split="test", window=2, horizon=1,
        selected_channels=["U10M"], normalization=normalization)
    expected_inputs, expected_reference = dataset[1]
    with _open_verified_dataset(release) as ds:
        inputs, reference = _read_full_grid(ds, dataset, [1])
    assert inputs.shape == (1, 2, 2, 240, 480)
    assert reference.shape == (1, 1, 1, 240, 480)
    np.testing.assert_array_equal(inputs[0], expected_inputs)
    np.testing.assert_allclose(reference[0],
        expected_reference * normalization["scale"][0] + normalization["mean"][0], rtol=1e-6, atol=1e-5)
