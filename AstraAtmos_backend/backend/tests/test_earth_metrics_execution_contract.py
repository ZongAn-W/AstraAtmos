"""Frozen v2/v3 contracts survive queue restore, optimization and completion."""

import asyncio
import copy
import json

import pytest
import torch

from models.training_scripts import earth_daily as runner
from services.dataset_identity import DatasetRequestError, EARTH_DATASET_3HOURLY_ID
from services.earth_task_split import build_earth_task_split
from services.earth_training_artifact import load_earth_training_artifact
from services.earth_training_contract import (
    CELL_AREA_WEIGHTED_AGGREGATION, GRID_UNIFORM_AGGREGATION,
    EARTH_3HOURLY_METRICS_SCHEMA_V2, EARTH_3HOURLY_METRICS_SCHEMA_V3,
    build_earth_training_spec, earth_3hourly_metrics_contract,
)
from test_earth_3hourly_training_runner import synthetic_training_release, registry_for
from test_earth_3hourly_training_service import service_environment


@pytest.fixture(scope="module", params=[EARTH_3HOURLY_METRICS_SCHEMA_V2, EARTH_3HOURLY_METRICS_SCHEMA_V3])
def executed_contract(request, synthetic_training_release, tmp_path_factory):
    registry = registry_for(synthetic_training_release)
    binding = registry.build_training_binding(EARTH_DATASET_3HOURLY_ID)
    ratios = {name + "_ratio": 1 / 3 for name in ("train", "validation", "test")}
    split = build_earth_task_split(synthetic_training_release.dates, 79, 1, ratios)
    contract = earth_3hourly_metrics_contract(request.param, executable=True)
    spec = build_earth_training_spec(
        task_id=2701, dataset_binding=binding, task_split=split, metrics_contract=contract,
        hyperparameters={**ratios, "window": 79, "horizon": 1, "epochs": 1,
                         "batch_size": 16, "selected_channels": [], "seed": 5},
    )
    root = tmp_path_factory.mktemp("metrics_execution_" + request.param[-2:])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with pytest.MonkeyPatch.context() as contract_patch:
            if request.param == EARTH_3HOURLY_METRICS_SCHEMA_V2:
                def no_area_weights(*args, **kwargs):
                    pytest.fail("Frozen v2 optimization/evaluation must not request area weights")
                contract_patch.setattr(runner, "threehour_global_cell_area_weights", no_area_weights)
            result = runner.run_training(spec, root / "model.pth", registry, device="cpu", cache_root=root / "cache")
    finally:
        torch.set_num_threads(previous_threads)
    return registry, binding, spec, result


def test_frozen_contract_controls_actual_loss_and_saved_statistics(executed_contract):
    registry, binding, spec, result = executed_contract
    contract = spec["metrics_contract"]
    checkpoint = load_earth_training_artifact(
        result["output_path"], expected_binding=binding, expected_task_id=2701,
        expected_hyperparameters={**spec["hyperparameters"], "_earth_task_split": spec["task_split"],
                                 "_earth_metrics_schema": contract["schema"],
                                 "_earth_metrics_aggregation": contract["aggregation"]},
    )
    assert {key: checkpoint.metrics[key] for key in contract} == contract
    weighted = contract["aggregation"] == CELL_AREA_WEIGHTED_AGGREGATION
    assert checkpoint.run["loss"] == ("normalized_mse_cell_area_weighted" if weighted else "normalized_mse_grid_uniform")
    stats = checkpoint.metrics["splits"]["test"]["by_lead"][0]["target_statistics"]
    assert ("weight_sum" in stats) is weighted
    assert checkpoint.metrics["splits"]["test"]["window_count"] == 1
    assert checkpoint.run["epochs_completed"] == 1


def test_queue_restore_and_completion_retain_frozen_contract(executed_contract, service_environment, monkeypatch):
    registry, binding, trained_spec, result = executed_contract
    module, _ = service_environment
    service = module.TrainingService()
    service._scheduler_started = True
    contract = trained_spec["metrics_contract"]
    payload = load_earth_training_artifact(result["output_path"]).payload

    class Process:
        pid = 9471
        returncode = 0
        def __init__(self):
            self.stdout = self
        def readline(self):
            return ""
        def close(self):
            pass
        def wait(self):
            return 0

    def popen(args, **kwargs):
        spec = json.loads(kwargs["env"]["ASTRAATMOS_EARTH_TRAINING_SPEC"])
        assert spec["metrics_contract"] == contract
        output = copy.deepcopy(payload)
        output["run"]["task_id"] = spec["task_id"]
        torch.save(output, args[-1])
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", popen)

    async def scenario():
        task = await service.start_training(
            user_id=1, custom_model_name="contract " + contract["schema"], model_script="ignored.py",
            hyperparameters=trained_spec["hyperparameters"], dataset_id=EARTH_DATASET_3HOURLY_ID,
            dataset_registry=registry,
        )
        raw = json.loads(task.hyperparameters)
        assert raw["_earth_metrics_schema"] == EARTH_3HOURLY_METRICS_SCHEMA_V3
        assert raw["_earth_metrics_aggregation"] == CELL_AREA_WEIGHTED_AGGREGATION
        if contract["schema"] == EARTH_3HOURLY_METRICS_SCHEMA_V2:
            # Represent a previously queued v2 row without converting its meaning.
            raw["_earth_metrics_schema"] = EARTH_3HOURLY_METRICS_SCHEMA_V2
            raw.pop("_earth_metrics_aggregation")
            async with module.async_session_maker() as session:
                row = await session.get(module.ModelTrainingTask, task.id)
                row.hyperparameters = json.dumps(raw)
                await session.commit()
            task = await service.get_task(task.id)
        plan = await service._prepare_training_execution(task, dataset_registry=registry)
        assert plan["earth_training_spec"]["metrics_contract"] == contract
        await service._run_training_subprocess(task.id, **plan)
        return await service.get_task(task.id)

    completed = asyncio.run(scenario())
    assert completed.status == "completed", completed.metrics
    assert completed.progress == 100
    assert json.loads(completed.metrics) == payload["metrics"]


@pytest.mark.parametrize("contract", [
    {"schema": "earth_training_metrics_3hourly_v99", "aggregation": GRID_UNIFORM_AGGREGATION},
    {"schema": EARTH_3HOURLY_METRICS_SCHEMA_V2, "aggregation": CELL_AREA_WEIGHTED_AGGREGATION},
    {"schema": EARTH_3HOURLY_METRICS_SCHEMA_V3, "aggregation": GRID_UNIFORM_AGGREGATION},
    {"schema": EARTH_3HOURLY_METRICS_SCHEMA_V3},
    None,
])
def test_invalid_execution_contract_stops_before_data_preparation(tmp_path, contract):
    spec = build_earth_training_spec(task_id=2702, dataset_binding={"dataset_id": EARTH_DATASET_3HOURLY_ID}, hyperparameters={})
    spec["metrics_contract"] = contract

    class NoDataReads:
        def get_earth_snapshot(self, *args, **kwargs):
            pytest.fail("Invalid contract reached data preparation")

    with pytest.raises(DatasetRequestError, match="metrics|aggregation"):
        runner.run_training(spec, tmp_path / "never.pth", NoDataReads(), device="cpu")
    assert not (tmp_path / "never.pth").exists()
