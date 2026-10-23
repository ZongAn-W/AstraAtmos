"""Independent upload contract, cached task admission, synthetic training/backtesting."""

import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services.dataset_identity import DatasetRequestError
from services.earth_model_source import build_uploaded_earth_model, EarthModelBuildError
from services.earth_prediction_service import build_prediction_context, run_earth_prediction
from services.earth_training_artifact import (
    EarthArtifactError, build_earth_model_from_checkpoint, load_earth_training_artifact,
    validate_checkpoint_payload,
    verify_earth_model_reload_isolated,
)
from services.earth_training_contract import build_earth_training_spec
from services.earth_task_split import build_earth_task_split
from services.user_model_validator import UserModelValidator, UserModelValidationResult, VALIDATION_TIMEOUT_CODE
from services.user_model_service import UserModelService
from training_backbones.earth_3hourly_uploaded_contract import (
    DATASET_ID, CONTRACT_SCHEMA, EVAL_BATCH_POLICY, FEED, channel_orders, forward,
)
from training_backbones.uploaded_model_dataset_spec import normalize_dataset_declarations, DatasetCapabilityError
from training_backbones.uploaded_model_earth_gate import evaluate_package_earth_compatibility
from models.training_scripts.earth_daily import run_training
from test_earth_3hourly_training_runner import synthetic_training_release, registry_for
from test_earth_3hourly_training_service import service_environment

TEMPLATE = Path(__file__).resolve().parents[3] / "docs" / "earth-3hourly-uploaded-model-template.py"


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def source_with_spec(spec):
    tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODEL_SPEC" for t in node.targets):
            node.value = ast.parse(repr(spec), mode="eval").body
    return ast.unparse(tree)


def spec():
    return {"name": "ThreeHour", "parameters": {"bias": {"type": "bool", "default": True}},
            "datasets": {DATASET_ID: copy.deepcopy(FEED)}}


def package_at(root, source=None, *, isolated=False):
    source = source or TEMPLATE.read_text(encoding="utf-8")
    path = root / "model.py"
    path.write_bytes(source.encode("utf-8"))
    validator = UserModelValidator() if isolated else UserModelValidator(timeout_seconds=None)
    result = validator.validate_file(path)
    package = SimpleNamespace(id="three-hour-model", user_id=1, version=1, display_name="ThreeHour",
                              original_filename="model.py", storage_path=str(path),
                              content_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                              validation_status="valid" if result.ok else "invalid",
                              param_schema=json.dumps(result.param_schema), validation_report=json.dumps(result.report_dict()))
    return package, result


@pytest.mark.parametrize("key", list(FEED))
def test_every_three_hour_spec_field_is_required_and_exact(key):
    declaration = spec()
    del declaration["datasets"][DATASET_ID][key]
    with pytest.raises(DatasetCapabilityError):
        normalize_dataset_declarations(declaration)
    declaration = spec()
    declaration["datasets"][DATASET_ID][key] = None
    with pytest.raises(DatasetCapabilityError):
        normalize_dataset_declarations(declaration)


@pytest.mark.parametrize("key,value", [("window", [56.0]), ("step", 3.0), ("grid", [[240.0, 480]]),
    ("target", "O3"), ("target_unit", "Dobsons"), ("frequency_hours", True), ("tensor_layout", "BCTHW"),
    ("input_channels", ["TO3", "T2M"]), ("time_zone", "local"), ("spatial_tile_shape", [240, 480])])
def test_spec_refuses_wrong_types_units_axes_and_dimensions(key, value):
    declaration = spec()
    declaration["datasets"][DATASET_ID][key] = value
    with pytest.raises(DatasetCapabilityError):
        normalize_dataset_declarations(declaration)


@pytest.mark.parametrize("key", ["data_dir", "data_path", "dataset_version", "dataset_fingerprint", "fingerprint", "snapshot", "dataset_snapshot"])
def test_spec_cannot_supply_dataset_identity_or_paths(key):
    declaration = spec()
    declaration["datasets"][DATASET_ID][key] = "client-supplied"
    with pytest.raises(DatasetCapabilityError):
        normalize_dataset_declarations(declaration)


def test_template_passes_spawn_isolation_and_does_not_claim_daily_or_mars(tmp_path):
    _, result = package_at(tmp_path, isolated=True)
    assert result.ok, result.errors
    assert not result.earth_ok and result.mars_ok is False
    verdict = result.report_dict()["earth_datasets"][DATASET_ID]
    assert verdict["contract_schema"] == CONTRACT_SCHEMA and verdict["status"] == "available"
    assert verdict["eval_batch_policy"] == EVAL_BATCH_POLICY
    assert verdict["output_shape"] == [2, 1, 1, 24, 48]
    assert len(channel_orders()) == 16


@pytest.mark.parametrize("windows,horizons,expected", [
    ([56, 2], [24, 1], (2, 1)),
    ([56], [24], (56, 24)),
])
def test_tile_admission_uses_declared_probe_windows_for_all_channels(
    tmp_path, monkeypatch, windows, horizons, expected,
):
    import training_backbones.earth_3hourly_uploaded_contract as contract

    declaration = spec()
    declaration["datasets"][DATASET_ID].update(window=windows, horizon=horizons)
    calls = []
    original = contract.dry_run

    def observed(build_model, order, params, **kwargs):
        calls.append((order, kwargs["window"], kwargs["horizon"], kwargs["batch_size"]))
        return original(build_model, order, params, **kwargs)

    monkeypatch.setattr(contract, "dry_run", observed)
    package, result = package_at(tmp_path, source_with_spec(declaration))
    assert result.ok, result.errors
    assert calls == [(order, *expected, 8) for order in channel_orders()]
    verdict = result.earth_compatibilities[DATASET_ID]
    assert (verdict["window"], verdict["horizon"]) == expected
    assert verdict["output_shape"] == [2, expected[1], 1, 24, 48]
    assert verdict["validation_scope"] == "declared_channels_probe_windows"
    assert UserModelService.earth_compatibility_for_package(
        package, dataset_id=DATASET_ID,
    )["compatible"] is True

    # Explicit configuration probes still execute the requested long windows,
    # channels and batch, even when admission preferred short windows.
    calls.clear()
    checked = evaluate_package_earth_compatibility(
        package, UserModelValidator(timeout_seconds=None), dataset_id=DATASET_ID,
        earth_probe={"input_channel_order": ["TO3", "T2M"], "window": 56,
                     "horizon": 24, "batch_size": 4, "custom_model_params": {}},
    )
    assert checked.compatible, checked.reasons
    assert checked.output_shape == [2, 24, 1, 24, 48]
    assert calls == [(["TO3", "T2M"], 56, 24, 4)]


@pytest.mark.parametrize("replacement", [
    'return x[:, :24, :1]',
    'return self.temporal(mixed).permute(0, 3, 1, 2).unsqueeze(2).double()',
    'return self.temporal(mixed).permute(0, 3, 1, 2).unsqueeze(2) * float("nan")',
    'return self.temporal(mixed).permute(0, 3, 1, 2).unsqueeze(2).detach()',
])
def test_execution_must_match_declaration_and_have_gradients(tmp_path, replacement):
    source = TEMPLATE.read_text(encoding="utf-8").replace(
        'return self.temporal(mixed).permute(0, 3, 1, 2).unsqueeze(2)', replacement)
    _, result = package_at(tmp_path, source)
    assert not result.ok
    verdict = result.earth_compatibilities[DATASET_ID]
    assert verdict["code"] == "uploaded_model_earth_3hourly_dry_run_failed"


def batch_dependent_source(kind):
    source = TEMPLATE.read_text(encoding="utf-8")
    if kind == "batch_norm":
        source = source.replace('batch, time, channels, height, width = x.shape',
            'x = self.batch_norm(x)\n        batch, time, channels, height, width = x.shape')
        return source.replace('self.mix = nn.Conv2d',
            'self.batch_norm = nn.BatchNorm3d(config["window"], track_running_stats=False)\n        self.mix = nn.Conv2d')
    if kind == "batch_size":
        return source.replace('batch, time, channels, height, width = x.shape',
            'x = x + x.shape[0]\n        batch, time, channels, height, width = x.shape')
    if kind in ("batch_gt_two", "batch_gt_eight"):
        threshold = 2 if kind == "batch_gt_two" else 8
        return source.replace('batch, time, channels, height, width = x.shape',
            f'if x.shape[0] > {threshold}:\n            x = x - x.mean(dim=0, keepdim=True)\n        batch, time, channels, height, width = x.shape')
    return source.replace('batch, time, channels, height, width = x.shape',
        'x = x - x.mean(dim=0, keepdim=True)\n        batch, time, channels, height, width = x.shape')


@pytest.mark.parametrize("kind", ["batch_mean", "batch_norm", "batch_size", "batch_gt_two"])
def test_admission_rejects_batch_dependent_eval_models(tmp_path, kind):
    package, validation = package_at(tmp_path, batch_dependent_source(kind))
    assert not validation.ok
    verdict = validation.earth_compatibilities[DATASET_ID]
    assert verdict["code"] == "uploaded_model_earth_3hourly_dry_run_failed"
    assert "sample-independent" in verdict["errors"][0]
    refused = evaluate_package_earth_compatibility(package, UserModelValidator(timeout_seconds=None), dataset_id=DATASET_ID)
    assert not refused.compatible and refused.code == verdict["code"]


def test_standard_batch_norm_with_running_statistics_remains_compatible(tmp_path):
    source = batch_dependent_source("batch_norm").replace('track_running_stats=False', 'track_running_stats=True')
    _, validation = package_at(tmp_path, source)
    assert validation.ok, validation.errors


def test_task_creation_does_not_repeat_validation_for_the_requested_batch(service_environment, tmp_path):
    module, registry = service_environment
    package, validation = package_at(tmp_path, batch_dependent_source("batch_gt_eight"))
    assert validation.ok
    class Packages:
        async def get_package_for_user(self, *args):
            return package
    service = module.TrainingService()
    service._scheduler_started = True
    class UnexpectedValidator:
        def validate_file(self, *args, **kwargs):
            pytest.fail("task creation must use the stored upload verdict")
    service._earth_model_validator = UnexpectedValidator()
    task = asyncio.run(service.start_training(user_id=1, custom_model_name="batch 12 queued", model_script="ignored.py",
        model_source="uploaded", uploaded_model_id=package.id, dataset_id=DATASET_ID, dataset_registry=registry,
        hyperparameters={"batch_size": 12}, user_model_service=Packages()))
    assert task.status == "queued"
    assert json.loads(task.hyperparameters)["batch_size"] == 12


@pytest.mark.parametrize("mutation", ["calibration", "counter", "parameter"])
def test_eval_state_mutation_is_rejected_and_probe_restores_state(mutation):
    from training_backbones.earth_3hourly_uploaded_contract import validate_eval_batch_independence
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.register_buffer("shift", torch.zeros(()))
            self.register_buffer("ready", torch.tensor(False))
        def forward(self, inputs):
            if mutation == "calibration" and not self.ready:
                self.shift.copy_(inputs.mean())
                self.ready.fill_(True)
            elif mutation == "counter":
                self.shift.add_(1)
            elif mutation == "parameter":
                self.weight.add_(1)
            return inputs[:, -1:, :1] * self.weight + self.shift
    model = Stateful()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="must not mutate"):
        validate_eval_batch_independence(model, window=2, channels=1, horizon=1)
    assert model.training
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_channels_not_just_boundary_counts_are_dry_run(tmp_path):
    source = TEMPLATE.read_text(encoding="utf-8").replace(
        'return EarthThreeHourLinear(config)',
        'if config["selected_channels"] == ["TO3", "V10M"]:\n        raise ValueError("specific combination")\n    return EarthThreeHourLinear(config)')
    _, result = package_at(tmp_path, source)
    assert not result.ok and "specific combination" in result.earth_compatibilities[DATASET_ID]["errors"][0]


def test_daily_mars_and_legacy_reports_cannot_prove_three_hour_compatibility(tmp_path):
    from test_earth_uploaded_upload_flow import EARTH_MODEL_SOURCE
    package, result = package_at(tmp_path, EARTH_MODEL_SOURCE)
    assert result.ok and result.earth_ok
    verdict = evaluate_package_earth_compatibility(package, UserModelValidator(timeout_seconds=None), dataset_id=DATASET_ID)
    assert not verdict.compatible and verdict.code == "uploaded_model_not_earth_3hourly_compatible"
    service = UserModelService(storage_root=tmp_path / "storage")
    async def get_package(*args):
        return package
    service.get_package_for_user = get_package
    cached = asyncio.run(service.get_earth_compatibility(package.id, 1, dataset_id=DATASET_ID))
    assert cached["status"] == "unknown" and cached["code"] == "uploaded_model_compatibility_unknown"
    assert asyncio.run(service.get_earth_compatibility(package.id, 1, dataset_id="earth_merra2_daily_v2"))["compatible"]


def test_failed_daily_build_does_not_skip_three_hour_verdict(tmp_path):
    declaration = spec()
    declaration["datasets"]["earth_merra2"] = {"window": [7], "horizon": [3], "grid": [[36, 72]]}
    source = source_with_spec(declaration).replace(
        "return EarthThreeHourLinear(config)",
        'if config["window"] != 56:\n        raise ValueError("daily unsupported")\n    return EarthThreeHourLinear(config)')
    _, result = package_at(tmp_path, source)
    assert result.ok and not result.earth_ok and result.earth_compatibilities[DATASET_ID]["compatible"]


def test_source_integrity_and_unknown_verdict_codes(tmp_path):
    package, _ = package_at(tmp_path)
    class UnknownValidator:
        def validate_file(self, *args, **kwargs):
            return UserModelValidationResult(ok=True)
    verdict = evaluate_package_earth_compatibility(package, UnknownValidator(), dataset_id=DATASET_ID)
    assert verdict.status == "unknown" and verdict.code == "uploaded_model_compatibility_unknown"
    Path(package.storage_path).write_text("changed", encoding="utf-8")
    verdict = evaluate_package_earth_compatibility(package, dataset_id=DATASET_ID)
    assert verdict.code == "uploaded_model_tampered"


def test_top_level_timeout_never_executes_in_parent(tmp_path):
    source = TEMPLATE.read_text(encoding="utf-8") + "\nwhile True:\n    pass\n"
    package = SimpleNamespace(storage_path=str(tmp_path / "loop.source"), display_name="loop", version=1,
                              content_hash=hashlib.sha256(source.encode()).hexdigest())
    Path(package.storage_path).write_bytes(source.encode("utf-8"))
    verdict = evaluate_package_earth_compatibility(package, UserModelValidator(timeout_seconds=4), dataset_id=DATASET_ID)
    assert not verdict.compatible and verdict.status == "unknown"
    assert verdict.code == VALIDATION_TIMEOUT_CODE
    assert "timed out after 4 seconds" in verdict.reasons[0]


@pytest.mark.parametrize("errors,result_code,status,expected", [
    (["budget exceeded"], VALIDATION_TIMEOUT_CODE, "unknown", VALIDATION_TIMEOUT_CODE),
    (["User model validation timed out after 30.0 seconds"], None, "unknown", VALIDATION_TIMEOUT_CODE),
    (["User model validation process exited without a result: 1"], None, "unknown", "uploaded_model_compatibility_unknown"),
    (["model forward timeout is unsupported"], None, "unavailable", "uploaded_model_contract_invalid"),
])
def test_three_hour_gate_keeps_timeout_and_abnormal_exit_blocked(tmp_path, errors, result_code, status, expected):
    source = TEMPLATE.read_bytes()
    path = tmp_path / "uploaded.source"
    path.write_bytes(source)
    package = SimpleNamespace(storage_path=str(path), display_name="ThreeHour", version=1,
                              content_hash=hashlib.sha256(source).hexdigest())
    class Validator:
        def validate_file(self, *args, **kwargs):
            return UserModelValidationResult(ok=False, errors=errors, code=result_code)
    verdict = evaluate_package_earth_compatibility(package, Validator(), dataset_id=DATASET_ID)
    assert verdict.compatible is False
    assert verdict.status == status and verdict.code == expected
    assert verdict.reasons == errors


@pytest.mark.parametrize("source_state,expected", [
    ("missing", "uploaded_model_missing"), ("tampered", "uploaded_model_tampered"),
])
def test_three_hour_gate_source_errors_take_priority_over_validation(tmp_path, source_state, expected):
    source = TEMPLATE.read_bytes()
    path = tmp_path / "uploaded.source"
    if source_state == "tampered":
        path.write_bytes(source + b"\n# changed\n")
    package = SimpleNamespace(storage_path=str(path), display_name="ThreeHour", version=1,
                              content_hash=hashlib.sha256(source).hexdigest())
    class Validator:
        def validate_file(self, *args, **kwargs):
            raise AssertionError("source integrity must be checked before validation")
    verdict = evaluate_package_earth_compatibility(package, Validator(), dataset_id=DATASET_ID)
    assert verdict.compatible is False and verdict.code == expected


@pytest.fixture(scope="module")
def uploaded_smoke(synthetic_training_release, tmp_path_factory):
    root = tmp_path_factory.mktemp("uploaded-three-hour-smoke")
    package, validation = package_at(root)
    assert validation.ok
    registry = registry_for(synthetic_training_release)
    binding = registry.build_training_binding(DATASET_ID)
    reference = {"package_id": package.id, "display_name": package.display_name, "version": 1,
                 "content_hash": package.content_hash, "source_path": package.storage_path,
                 "source_text": TEMPLATE.read_text(encoding="utf-8"), "param_schema": validation.param_schema,
                 "custom_model_params": {"bias": True}}
    spec = build_earth_training_spec(task_id=902, dataset_binding=binding, uploaded_model=reference,
                                    task_split=build_earth_task_split(synthetic_training_release.dates, 56, 24,
                                        {name + "_ratio": 1/3 for name in ("train", "validation", "test")}),
                                    hyperparameters={**{name + "_ratio": 1/3 for name in ("train", "validation", "test")},
                                                     "epochs": 1, "batch_size": 8, "selected_channels": ["U10M"]})
    output = root / "new-uploaded-checkpoint.pth"
    result = run_training(spec, output, registry, device="cpu", cache_root=root / "cache")
    task = SimpleNamespace(id=902, **binding, user_id=1, status="completed", output_model_path=str(output),
                           hyperparameters=json.dumps({**spec["hyperparameters"], "_earth_task_split": spec["task_split"], "_uploaded_model_id": package.id,
                                                       "_uploaded_model_version": 1, "_uploaded_model_content_hash": package.content_hash}))
    return SimpleNamespace(task=task, registry=registry, output=output, result=result, reference=reference)


def test_uploaded_synthetic_training_evaluates_and_strictly_reloads(uploaded_smoke):
    checkpoint = load_earth_training_artifact(uploaded_smoke.output)
    assert checkpoint.model_source == "uploaded"
    assert checkpoint.model_config["contract_schema"] == CONTRACT_SCHEMA
    assert checkpoint.uploaded_model["build_config"]["height"] == 24
    assert checkpoint.training_contract["strict_split_windows"] is True
    assert checkpoint.normalization["fit_time_end"] == "2020-01-10T22:30:00Z"
    for split in ("validation", "test"):
        assert len(checkpoint.metrics["splits"][split]["by_lead"]) == 24
        assert set(checkpoint.metrics["splits"][split]["overall"]) == {"mse", "rmse", "mae", "r2", "mape", "smape"}
    assert checkpoint.metrics["schema"] == "earth_training_metrics_3hourly_v3"
    assert checkpoint.metrics["aggregation"] == "forecast_origin_lead_cell_area_weighted"
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    assert forward(model, torch.randn(2, 56, 2, 24, 48)).shape == (2, 24, 1, 24, 48)


def test_double_factory_uses_float32_in_training_reload_and_backtest(synthetic_training_release, tmp_path, monkeypatch):
    source = TEMPLATE.read_text(encoding="utf-8").replace(
        'return EarthThreeHourLinear(config)', 'return EarthThreeHourLinear(config).double()')
    package, validation = package_at(tmp_path, source)
    assert validation.ok, validation.errors
    reference = {"package_id": package.id, "display_name": package.display_name, "version": 1,
                 "content_hash": package.content_hash, "source_path": package.storage_path,
                 "source_text": source, "param_schema": validation.param_schema,
                 "custom_model_params": {"bias": True}}
    registry = registry_for(synthetic_training_release)
    binding = registry.build_training_binding(DATASET_ID)
    ratios = {name + "_ratio": 1/3 for name in ("train", "validation", "test")}
    split = build_earth_task_split(synthetic_training_release.dates, 56, 24, ratios)
    spec = build_earth_training_spec(task_id=1902, dataset_binding=binding, uploaded_model=reference,
                                   task_split=split, hyperparameters={**ratios, "epochs": 1,
                                   "batch_size": 8, "selected_channels": ["U10M"]})
    path = tmp_path / "double-factory.pth"
    run_training(spec, path, registry, device="cpu", cache_root=tmp_path / "cache")
    checkpoint = load_earth_training_artifact(path)
    assert all(t.dtype == torch.float32 for t in checkpoint.payload["model_state_dict"].values() if t.is_floating_point())
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    inputs = torch.randn(2, 56, 2, 24, 48)
    torch.testing.assert_close(forward(model, inputs), torch.cat([forward(model, x[None]) for x in inputs]))
    if torch.cuda.is_available():
        assert forward(model.to("cuda"), inputs.to("cuda")).dtype == torch.float32
    task = SimpleNamespace(id=1902, **binding, user_id=1, status="completed", output_model_path=str(path),
                           hyperparameters=json.dumps({**spec["hyperparameters"], "_earth_task_split": split}))
    import services.earth_prediction_service as prediction
    monkeypatch.setattr(prediction, "_field_list", lambda values: [{"shape": list(values.shape)}])
    result = run_earth_prediction(task, "2020-01-27T22:30:00Z", registry, device="cpu")
    assert result["origin_split"] == "test" and set(result["metrics"]["overall"]) == {"mse", "rmse", "mae", "r2", "mape", "smape"}


@pytest.mark.parametrize("kind,batch_size", [("batch_mean", 8), ("batch_gt_two", 12)])
def test_old_batch_dependent_checkpoint_is_rejected_on_reload(uploaded_smoke, tmp_path, kind, batch_size):
    payload = copy.deepcopy(load_earth_training_artifact(uploaded_smoke.output).payload)
    source = batch_dependent_source(kind)
    payload["run"]["hyperparameters"]["batch_size"] = batch_size
    reference = payload["model_ref"]["uploaded_model"]
    reference.update(source_text=source, content_hash=hashlib.sha256(source.encode()).hexdigest(),
                     source_path=str(tmp_path / "unavailable.source"))
    path = tmp_path / "batch-dependent.pth"
    torch.save(payload, path)
    checkpoint = load_earth_training_artifact(path)
    with pytest.raises(EarthArtifactError, match="sample-independent"):
        build_earth_model_from_checkpoint(checkpoint)
    with pytest.raises(EarthArtifactError, match="sample-independent"):
        verify_earth_model_reload_isolated(path)


@pytest.mark.parametrize("training_batch_size", [12, 64])
def test_bounded_reload_probe_does_not_prove_eval_independence_above_eight(
    uploaded_smoke, tmp_path, monkeypatch, training_batch_size,
):
    """Large training batches never enlarge the checkpoint's runtime probe."""
    from training_backbones import earth_3hourly_uploaded_contract as contract

    payload = copy.deepcopy(load_earth_training_artifact(uploaded_smoke.output).payload)
    source = batch_dependent_source("batch_gt_eight")
    payload["run"]["hyperparameters"]["batch_size"] = training_batch_size
    payload["model_ref"]["uploaded_model"].update(
        source_text=source, content_hash=hashlib.sha256(source.encode()).hexdigest(),
        source_path=str(tmp_path / "unavailable.source"),
    )
    path = tmp_path / "above-eight-boundary.pth"
    torch.save(payload, path)
    checkpoint = load_earth_training_artifact(path)
    observed_batches = []
    actual_forward = contract.forward

    def record_forward(model, inputs, **kwargs):
        observed_batches.append(inputs.shape[0])
        return actual_forward(model, inputs, **kwargs)

    monkeypatch.setattr(contract, "forward", record_forward)
    model, _ = build_earth_model_from_checkpoint(checkpoint)
    assert model is not None
    assert checkpoint.run["hyperparameters"]["batch_size"] == training_batch_size
    assert set(observed_batches) == {1, 2, 3}
    # This model changes its behaviour only at B > 8. Passing the bounded
    # B=2/3 probe preserves its known limitation; it is not proof for B=12/64.
    verify_earth_model_reload_isolated(path)


@pytest.mark.parametrize("mutation", ["daily_schema", "implementation", "source", "tile", "build", "unit", "fingerprint", "normalization"])
def test_checkpoint_isolation_rejects_corrupt_contract(uploaded_smoke, mutation):
    payload = copy.deepcopy(load_earth_training_artifact(uploaded_smoke.output).payload)
    if mutation == "daily_schema":
        payload["artifact_schema"] = "astraatmos_earth_forecast_checkpoint_v1"
    elif mutation == "implementation":
        payload["model_config"]["implementation_id"] = "astraatmos_gridpoint_dlinear_3hourly_v1"
    elif mutation == "source":
        payload["model_ref"]["uploaded_model"]["source_text"] += "\n# changed\n"
    elif mutation == "tile":
        payload["model_config"]["spatial_tile_shape"] = [240, 480]
    elif mutation == "build":
        payload["model_ref"]["uploaded_model"]["build_config"]["window"] = 7
    elif mutation == "unit":
        payload["training_contract"]["target_unit"] = "kg m-2"
    elif mutation == "fingerprint":
        payload["dataset_binding"]["dataset_fingerprint"] = "e" * 64
    else:
        payload["normalization"]["fit_step_count"] = 240
    with pytest.raises(EarthArtifactError):
        validate_checkpoint_payload(payload)


def test_uploaded_backtest_reads_full_global_field_and_disallows_cross_split(uploaded_smoke, monkeypatch):
    import services.earth_prediction_service as service
    from services.earth_prediction_cache import EarthPredictionCache
    monkeypatch.setattr(service, "EARTH_PREDICTION_CACHE", EarthPredictionCache())
    context = build_prediction_context(uploaded_smoke.task, uploaded_smoke.registry)
    assert context.origins["count"] == 3
    assert context.origins["timestamps"] == ["2020-01-07T22:30:00Z", "2020-01-17T22:30:00Z", "2020-01-27T22:30:00Z"]
    result = run_earth_prediction(uploaded_smoke.task, "2020-01-27T22:30:00Z", uploaded_smoke.registry, device="cpu")
    assert result["model_source"] == "uploaded" and result["origin_split"] == "test"
    assert result["target_unit"] == "DU" and len(result["target_timestamps"]) == 24
    fields = np.asarray([row["field"] for row in result["prediction"]], dtype="float32")
    assert fields.shape == (24, 240, 480) and np.isfinite(fields).all()
    with pytest.raises(DatasetRequestError) as error:
        run_earth_prediction(uploaded_smoke.task, "2020-01-21T01:30:00Z", uploaded_smoke.registry, device="cpu")
    assert error.value.code == "earth_prediction_origin_out_of_range"


def test_task_creation_and_restart_pin_source_and_custom_parameters(service_environment, tmp_path, monkeypatch):
    module, registry = service_environment
    package, _ = package_at(tmp_path)
    service = module.TrainingService()
    service._scheduler_started = True
    class UnexpectedValidator:
        def validate_file(self, *args, **kwargs):
            pytest.fail("task creation and recovery must not repeat upload validation")
    service._earth_model_validator = UnexpectedValidator()
    user_models = UserModelService(storage_root=tmp_path / "storage", validator=UserModelValidator(timeout_seconds=None))
    async def get_package(*args):
        return package
    user_models.get_package_for_user = get_package
    async def scenario():
        return await service.start_training(user_id=1, custom_model_name="uploaded three hour", model_script="ignored.py",
            model_source="uploaded", uploaded_model_id=package.id, dataset_id=DATASET_ID, dataset_registry=registry,
            hyperparameters={"epochs": 1, "selected_channels": ["U10M"], "custom_model_params": {"bias": False}},
            user_model_service=user_models)
    task = asyncio.run(scenario())
    assert task.status == "queued" and task.uploaded_model_id == package.id
    plan = asyncio.run(service._prepare_training_execution(task, dataset_registry=registry))
    queued = plan["earth_training_spec"]
    assert queued["uploaded_model"]["custom_model_params"] == {"bias": False}
    assert queued["uploaded_model"]["content_hash"] == package.content_hash
    assert queued["uploaded_model"]["source_text"] == TEMPLATE.read_text(encoding="utf-8")
    assert queued["uploaded_model"]["param_schema"] == json.loads(package.param_schema)
    assert queued["uploaded_model"]["contract_schema"] == CONTRACT_SCHEMA
    monkeypatch.setattr(service, "_default_dataset_registry", lambda: registry)
    restored = service._restore_3hourly_training_spec(task)
    assert restored["uploaded_model"] == queued["uploaded_model"]
    assert restored["dataset_binding"] == queued["dataset_binding"]


def test_reserved_custom_parameters_are_rejected_before_a_task(service_environment, tmp_path):
    module, registry = service_environment
    package, _ = package_at(tmp_path)
    class Packages:
        async def get_package_for_user(self, *args):
            return package
    service = module.TrainingService()
    with pytest.raises(DatasetRequestError) as error:
        asyncio.run(service.start_training(user_id=1, custom_model_name="rejected upload", model_script="ignored.py",
            model_source="uploaded", uploaded_model_id=package.id, dataset_id=DATASET_ID, dataset_registry=registry,
            hyperparameters={"custom_model_params": {"data_dir": "injected"}}, user_model_service=Packages()))
    assert error.value.code == "invalid_earth_training_parameters"
    assert asyncio.run(service.get_all_tasks()) == []


def test_actual_custom_parameters_are_frozen_without_a_training_probe(service_environment, tmp_path):
    module, registry = service_environment
    source = TEMPLATE.read_text(encoding="utf-8").replace('return EarthThreeHourLinear(config)',
        'if not config["bias"]:\n        raise ValueError("selected bias rejected")\n    return EarthThreeHourLinear(config)')
    package, validation = package_at(tmp_path, source)
    assert validation.ok
    class Packages:
        async def get_package_for_user(self, *args):
            return package
    service = module.TrainingService()
    service._scheduler_started = True
    task = asyncio.run(service.start_training(user_id=1, custom_model_name="stored actual parameters", model_script="ignored.py",
        model_source="uploaded", uploaded_model_id=package.id, dataset_id=DATASET_ID, dataset_registry=registry,
        hyperparameters={"custom_model_params": {"bias": False}}, user_model_service=Packages()))
    hypers = json.loads(task.hyperparameters)
    assert task.status == "queued" and hypers["custom_model_params"] == {"bias": False}
    reference = hypers["_earth_uploaded_reference"]
    assert reference["custom_model_params"] == {"bias": False}
    assert reference["source_text"] == source
    assert reference["content_hash"] == package.content_hash
    assert reference["contract_schema"] == CONTRACT_SCHEMA


@pytest.mark.parametrize("mutation,expected", [
    ("missing", "uploaded_model_missing"),
    ("tampered", "uploaded_model_tampered"),
    ("unproven", "uploaded_model_compatibility_unknown"),
    ("wrong_dataset", "uploaded_model_compatibility_unknown"),
])
def test_task_creation_still_requires_matching_source_and_upload_evidence(service_environment, tmp_path, mutation, expected):
    module, registry = service_environment
    package, validation = package_at(tmp_path)
    if mutation == "missing":
        package.storage_path = str(tmp_path / "missing.source")
    elif mutation == "tampered":
        Path(package.storage_path).write_text("changed", encoding="utf-8")
    else:
        report = validation.report_dict()
        if mutation == "unproven":
            report["earth_datasets"][DATASET_ID].pop("eval_batch_policy")
        else:
            report["earth_datasets"].pop(DATASET_ID)
        package.validation_report = json.dumps(report)
    class Packages:
        async def get_package_for_user(self, *args):
            return package
    class UnexpectedValidator:
        def validate_file(self, *args, **kwargs):
            pytest.fail("invalid cached evidence must not trigger automatic model validation")
    service = module.TrainingService(earth_model_validator=UnexpectedValidator())
    with pytest.raises(DatasetRequestError) as error:
        asyncio.run(service.start_training(user_id=1, custom_model_name="invalid cached upload", model_script="ignored.py",
            model_source="uploaded", uploaded_model_id=package.id, dataset_id=DATASET_ID, dataset_registry=registry,
            hyperparameters={}, user_model_service=Packages()))
    assert error.value.code == expected
    assert asyncio.run(service.get_all_tasks()) == []


@pytest.mark.parametrize("key", ["data_dir", "dataset_version", "fingerprint", "snapshot", "contract_schema", "device", "normalization",
                                "data_directories", "dataset_binding", "version", "dataset_identity_status", "training_dataset"])
def test_custom_parameter_schema_cannot_override_contract(tmp_path, key):
    declaration = spec()
    declaration["parameters"][key] = {"type": "select", "options": ["injected"], "default": "injected"}
    _, result = package_at(tmp_path, source_with_spec(declaration))
    assert not result.ok and result.earth_compatibilities[DATASET_ID]["code"] == "uploaded_model_contract_invalid"


def test_embedded_source_is_hash_checked_even_when_original_is_missing(uploaded_smoke, tmp_path):
    checkpoint = load_earth_training_artifact(uploaded_smoke.output)
    reference = copy.deepcopy(checkpoint.uploaded_model)
    reference["source_path"] = str(tmp_path / "missing-original.source")
    kwargs = dict(input_channel_order=["TO3", "U10M"], window=56, horizon=24, height=240, width=480, dataset_id=DATASET_ID)
    model, _, _ = build_uploaded_earth_model(reference=reference, **kwargs)
    assert forward(model, torch.randn(1, 56, 2, 24, 48)).dtype == torch.float32
    reference["source_text"] += "\n# changed\n"
    with pytest.raises(EarthModelBuildError) as error:
        build_uploaded_earth_model(reference=reference, **kwargs)
    assert error.value.code == "uploaded_model_tampered"


def test_uploaded_weights_cannot_load_with_wrong_shape_or_as_mars(uploaded_smoke, tmp_path):
    from services.training_weight_service import TrainingWeightService
    checkpoint = load_earth_training_artifact(uploaded_smoke.output)
    payload = copy.deepcopy(checkpoint.payload)
    key = next(iter(payload["model_state_dict"]))
    payload["model_state_dict"][key] = torch.zeros(1)
    path = tmp_path / "new-corrupt-upload-checkpoint.pth"
    torch.save(payload, path)
    with pytest.raises(EarthArtifactError):
        build_earth_model_from_checkpoint(load_earth_training_artifact(path))
    assert TrainingWeightService._validate_weight_file(uploaded_smoke.output)["ok"] is False


def test_template_device_and_input_dtype_contract(tmp_path):
    package, validation = package_at(tmp_path)
    reference = {"source_path": package.storage_path, "content_hash": package.content_hash,
                 "param_schema": validation.param_schema, "custom_model_params": {"bias": True}}
    model, _, _ = build_uploaded_earth_model(reference=reference, input_channel_order=["TO3"],
                                           window=56, horizon=24, height=240, width=480, dataset_id=DATASET_ID)
    inputs = torch.randn(1, 56, 1, 24, 48)
    assert forward(model, inputs).device == inputs.device
    with pytest.raises(ValueError):
        forward(model, inputs.double())
    if torch.cuda.is_available():
        cuda_inputs = inputs.to("cuda")
        assert forward(model.to("cuda"), cuda_inputs).device == cuda_inputs.device


@pytest.mark.parametrize("status,shape", [("unknown", [2, 24, 1, 24, 48]), ("available", [1, 24, 1, 24, 48])])
def test_incomplete_execution_evidence_cannot_be_available(tmp_path, status, shape):
    package, result = package_at(tmp_path)
    result.earth_compatibilities[DATASET_ID].update(status=status, output_shape=shape)
    package.validation_report = json.dumps(result.report_dict())
    service = UserModelService(storage_root=tmp_path / "storage")
    async def get_package(*args):
        return package
    service.get_package_for_user = get_package
    cached = asyncio.run(service.get_earth_compatibility(package.id, 1, dataset_id=DATASET_ID))
    assert cached["status"] == "unknown" and not cached["compatible"]
    class Validator:
        def validate_file(self, *args, **kwargs):
            return result
    gate = evaluate_package_earth_compatibility(package, Validator(), dataset_id=DATASET_ID)
    assert not gate.compatible and gate.code == "uploaded_model_compatibility_unknown"


def test_old_validation_report_requires_sample_independence_revalidation(tmp_path):
    package, result = package_at(tmp_path)
    result.earth_compatibilities[DATASET_ID].pop("eval_batch_policy")
    package.validation_report = json.dumps(result.report_dict())
    service = UserModelService(storage_root=tmp_path / "storage")
    async def get_package(*args):
        return package
    service.get_package_for_user = get_package
    verdict = asyncio.run(service.get_earth_compatibility(package.id, 1, dataset_id=DATASET_ID))
    assert not verdict["compatible"] and verdict["status"] == "unknown"
    class OldValidator:
        def validate_file(self, *args, **kwargs):
            return result
    assert not evaluate_package_earth_compatibility(package, OldValidator(), dataset_id=DATASET_ID).compatible
    current = evaluate_package_earth_compatibility(package, UserModelValidator(timeout_seconds=None), dataset_id=DATASET_ID)
    assert current.compatible


@pytest.mark.parametrize("corruption", [None, "weights", "package"])
def test_parent_completion_checks_uploaded_identity_and_strict_reload(
        uploaded_smoke, service_environment, tmp_path, monkeypatch, corruption):
    module, _ = service_environment
    from test_earth_training_service import StubProcess
    package, _ = package_at(tmp_path)
    class Packages:
        async def get_package_for_user(self, *args):
            return package
    service = module.TrainingService()
    service._scheduler_started = True
    service._earth_model_validator = UserModelValidator(timeout_seconds=None)
    task = asyncio.run(service.start_training(user_id=1, custom_model_name="parent uploaded completion",
        model_script="ignored.py", model_source="uploaded", uploaded_model_id=package.id,
        dataset_id=DATASET_ID, dataset_registry=uploaded_smoke.registry,
        hyperparameters={**{name + "_ratio": 1/3 for name in ("train", "validation", "test")},
                             "epochs": 1, "selected_channels": ["U10M"]}, user_model_service=Packages()))
    plan = asyncio.run(service._prepare_training_execution(task, dataset_registry=uploaded_smoke.registry))
    spec = plan["earth_training_spec"]
    output = tmp_path / "new-parent-checkpoint.pth"
    def popen(*args, **kwargs):
        payload = copy.deepcopy(load_earth_training_artifact(uploaded_smoke.output).payload)
        payload["run"]["task_id"] = task.id
        if corruption == "weights":
            key = next(iter(payload["model_state_dict"]))
            payload["model_state_dict"][key] = torch.zeros(1)
        if corruption == "package":
            payload["model_ref"]["uploaded_model"]["package_id"] = "different-package"
        torch.save(payload, output)
        return StubProcess(lambda: None)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    asyncio.run(service._run_training_subprocess(task.id, "earth_daily.py", spec["hyperparameters"],
        tmp_path / "parent.log", output, earth_training_spec=spec))
    completed = asyncio.run(service.get_task(task.id))
    assert completed.status == ("failed" if corruption else "completed")
    if corruption:
        assert json.loads(completed.metrics)["error_code"] == "invalid_earth_training_artifact"
    else:
        assert json.loads(completed.metrics)["unit"] == "DU"


def test_strict_reload_has_a_process_timeout(uploaded_smoke):
    with pytest.raises(EarthArtifactError, match="timed out"):
        verify_earth_model_reload_isolated(uploaded_smoke.output, timeout_seconds=.01)


def test_http_upload_verdict_revalidation_and_training_errors(service_environment, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.dependencies import get_current_user
    from routers import training, user_models
    module, registry = service_environment
    service = module.TrainingService()
    service._scheduler_started = True
    service._earth_model_validator = UserModelValidator(timeout_seconds=None)
    user_service = UserModelService(storage_root=tmp_path / "http-models", sessionmaker=module.async_session_maker,
                                   validator=UserModelValidator())
    monkeypatch.setattr(training, "training_service", service)
    class Tags:
        async def get_task_tags(self, ids, user_id):
            return {task_id: [] for task_id in ids}
    monkeypatch.setattr(training, "training_tag_service", Tags())
    app = FastAPI()
    app.state.user_model_service, app.state.dataset_registry = user_service, registry
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, role="user")
    app.include_router(user_models.router, prefix="/api")
    app.include_router(training.router, prefix="/api")
    with TestClient(app) as client:
        for kind in ("earth-3hourly-template", "earth-3hourly-guide", "template", "guide"):
            response = client.get(f"/api/user-models/downloads/{kind}")
            assert response.status_code == 200 and len(response.content) > 100
            assert "attachment" in response.headers["content-disposition"]
        uploaded = client.post("/api/user-models", files={"file": ("model.py", TEMPLATE.read_bytes(), "text/x-python")})
        assert uploaded.status_code == 200, uploaded.text
        package = uploaded.json()
        verdict = package["validation_report"]["earth_datasets"][DATASET_ID]
        assert verdict["status"] == "available" and verdict["contract_schema"] == CONTRACT_SCHEMA
        endpoint = f'/api/user-models/{package["id"]}/earth-compatibility'
        assert client.get(endpoint, params={"dataset_id": DATASET_ID}).json()["compatible"]
        assert not client.get(endpoint).json()["compatible"]
        assert client.get(endpoint, params={"dataset_id": "bogus"}).json()["detail"]["code"] == "unknown_dataset_id"
        rechecked = client.post(f'/api/user-models/{package["id"]}/validate')
        assert rechecked.status_code == 200 and rechecked.json()["validation_status"] == "valid", rechecked.text
        request = {"model_script": "ignored.py", "model_name": "HTTP upload task", "model_source": "uploaded",
                   "uploaded_model_id": package["id"], "dataset_id": DATASET_ID,
                   "hyperparameters": {"epochs": 1, "custom_model_params": {"bias": False}}}
        invalid = client.post("/api/training/start", json={**request, "hyperparameters": {"validation_ratio": 0}})
        assert invalid.status_code == 422 and invalid.json()["detail"]["code"] == "invalid_earth_training_parameters"
        assert asyncio.run(service.get_all_tasks()) == []
        queued = client.post("/api/training/start", json=request)
        assert queued.status_code == 200 and queued.json()["status"] == "queued", queued.text
        assert queued.json()["dataset_id"] == DATASET_ID
        from test_earth_uploaded_upload_flow import EARTH_MODEL_SOURCE
        old = client.post("/api/user-models", files={"file": ("daily.py", EARTH_MODEL_SOURCE.encode(), "text/x-python")}).json()
        assert old["validation_report"]["earth"]["compatible"]
        refused = client.post("/api/training/start", json={**request, "model_name": "reject daily as threehour",
                              "uploaded_model_id": old["id"], "hyperparameters": {"epochs": 1}})
        assert refused.status_code == 422, refused.text
        assert refused.json()["detail"]["code"] == "uploaded_model_not_earth_3hourly_compatible"
