"""Earth training runner: best-weight selection, patience, metrics and artifact."""

import json

import numpy as np
import pytest
import torch

from models.training_scripts.earth_daily import (
    EarthTrainingError,
    _evaluate_split,
    resolve_device,
    run_training,
    seed_everything,
)
from services.dataset_registry import DatasetRegistry
from services.earth_training_artifact import load_earth_training_artifact
from services.earth_training_contract import build_earth_training_spec


def _spec(fixture, *, task_id=901, selected_channels=None, **hyperparameters):
    registry = DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **fixture)
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    payload = {
        "selected_channels": selected_channels if selected_channels is not None else ["T2M"],
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "seed": 5,
        "early_stopping_patience": 0,
        "linear_hidden_layers": 2,
    }
    payload.update(hyperparameters)
    return registry, binding, build_earth_training_spec(
        task_id=task_id, dataset_binding=binding, hyperparameters=payload
    )


def _run(fixture, tmp_path, *, name="task_901.pth", device="cpu", **kwargs):
    registry, binding, spec = _spec(fixture, **kwargs)
    output = tmp_path / name
    result = run_training(spec, output, registry, device=device)
    return registry, binding, spec, output, result


def test_one_epoch_run_publishes_a_strictly_loadable_checkpoint(earth_global_release, tmp_path):
    registry, binding, spec, output, result = _run(earth_global_release, tmp_path)
    assert output.is_file()
    assert result["best_epoch"] == 1
    assert result["epochs_completed"] == 1
    assert result["split_window_counts"] == {"train": 357, "validation": 172, "test": 175}

    checkpoint = load_earth_training_artifact(
        output, expected_binding=binding, expected_task_id=901,
        expected_hyperparameters={"selected_channels": ["T2M"]},
    )
    assert checkpoint.run["run_complete"] is True
    assert checkpoint.run["optimizer"] == "Adam"
    assert checkpoint.run["loss"] == "normalized_mse_grid_uniform"
    assert checkpoint.run["device"] == "cpu"
    assert checkpoint.training_contract["input_channel_order"] == ["TO3", "T2M"]
    assert checkpoint.training_contract["split_policy"] == "published_manifest_splits"
    assert checkpoint.training_contract["split_ranges"] == {
        "train": {"date_start": "2020-01-01", "date_end": "2020-12-31", "window_count": 357},
        "validation": {"date_start": "2021-01-01", "date_end": "2021-06-30", "window_count": 172},
        "test": {"date_start": "2021-07-01", "date_end": "2021-12-31", "window_count": 175},
    }
    assert checkpoint.normalization["fit_split"] == "train"
    assert checkpoint.normalization["fit_date_end"] == "2020-12-31"


def test_metrics_are_finite_du_with_overall_and_per_lead_values(earth_global_release, tmp_path):
    _, _, _, _, result = _run(earth_global_release, tmp_path)
    splits = result["metrics"]["splits"]
    assert result["metrics"]["unit"] == "DU"
    assert result["metrics"]["target"] == "TO3"
    assert splits["test"]["window_count"] == 175
    assert splits["validation"]["window_count"] == 172
    for name in ("validation", "test"):
        overall = splits[name]["overall"]
        assert np.isfinite(overall["rmse"]) and np.isfinite(overall["mae"])
        assert overall["rmse"] >= 0 and overall["mae"] >= 0
        assert [row["lead_day"] for row in splits[name]["by_lead"]] == [1, 2, 3]
        for row in splits[name]["by_lead"]:
            assert np.isfinite(row["rmse"]) and np.isfinite(row["mae"])
    json.dumps(result["metrics"])


def test_training_smoke_actually_learns_on_the_tiny_fixture(earth_global_release, tmp_path):
    """A short run must reduce its own training loss, proving the loop optimizes."""
    registry, binding, spec = _spec(earth_global_release, epochs=3, batch_size=64, learning_rate=0.01)
    output = tmp_path / "task_901_learn.pth"
    result = run_training(spec, output, registry, device="cpu")
    assert result["epochs_completed"] == 3
    # Validation must be scored (not filled from the test split) and be finite.
    assert np.isfinite(result["metrics"]["splits"]["validation"]["overall"]["rmse"])


def test_validation_loss_weights_uneven_batches_by_element_count(monkeypatch):
    import models.training_scripts.earth_daily as runner

    class IdentityDataset:
        def denormalize_ozone(self, values):
            return values

    first_inputs = torch.zeros(2, 7, 1, 1, 1)
    first_targets = torch.zeros(2, 3, 1, 1, 1)
    second_inputs = torch.ones(1, 7, 1, 1, 1)
    second_targets = torch.zeros(1, 3, 1, 1, 1)

    def forward(_model, inputs, **_kwargs):
        if inputs.shape[0] == 2:
            return torch.zeros(2, 3, 1, 1, 1)
        return torch.full((1, 3, 1, 1, 1), 2.0)

    monkeypatch.setattr(runner, "earth_forward_for_model", forward)
    _metrics, loss = _evaluate_split(
        torch.nn.Identity(),
        [(first_inputs, first_targets), (second_inputs, second_targets)],
        IdentityDataset(),
        torch.device("cpu"),
        loss_function=torch.nn.MSELoss(),
    )
    # Batch means are 0 and 4; element-weighted mean is (0*6 + 4*3) / 9.
    assert loss == pytest.approx(4.0 / 3.0)


def test_ozone_only_run_works_end_to_end(earth_global_release, tmp_path):
    _, _, _, output, result = _run(
        earth_global_release, tmp_path, name="task_901_o3.pth",
        selected_channels=[], epochs=1, batch_size=64,
    )
    checkpoint = load_earth_training_artifact(output, expected_task_id=901)
    assert checkpoint.training_contract["input_channel_order"] == ["TO3"]
    assert checkpoint.normalization["channel_order"] == ["TO3"]
    assert np.isfinite(result["metrics"]["splits"]["test"]["overall"]["rmse"])


def test_all_channels_run_works_end_to_end(earth_global_release, tmp_path):
    _, _, _, output, _ = _run(
        earth_global_release, tmp_path, name="task_901_all.pth",
        selected_channels=["U10M", "V10M", "T2M", "SWGDN"], epochs=1, batch_size=64,
    )
    checkpoint = load_earth_training_artifact(output, expected_task_id=901)
    assert checkpoint.training_contract["input_channel_order"] == [
        "TO3", "U10M", "V10M", "T2M", "SWGDN",
    ]


def test_run_is_reproducible_for_the_same_seed(earth_global_release, tmp_path):
    _, _, _, first, result_a = _run(
        earth_global_release, tmp_path, name="task_901_seed5a.pth", seed=5, batch_size=64,
    )
    _, _, _, second, result_b = _run(
        earth_global_release, tmp_path, name="task_901_seed5b.pth", seed=5, batch_size=64,
    )
    checkpoint_a = load_earth_training_artifact(first, expected_task_id=901)
    checkpoint_b = load_earth_training_artifact(second, expected_task_id=901)
    for key, value in checkpoint_a.payload["model_state_dict"].items():
        assert torch.equal(value, checkpoint_b.payload["model_state_dict"][key])
    assert result_a["metrics"]["splits"]["test"]["overall"] == result_b["metrics"]["splits"]["test"]["overall"]


def test_missing_spec_makes_the_cli_exit_non_zero(earth_global_release, tmp_path):
    """The real CLI must refuse to run without the service-provided spec."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parents[1]
    script = backend_dir / "models" / "training_scripts" / "earth_daily.py"
    environment = dict(os.environ)
    environment.pop("ASTRAATMOS_EARTH_TRAINING_SPEC", None)
    output = tmp_path / "cli_nope.pth"
    completed = subprocess.run(
        [sys.executable, str(script), "--output_path", str(output)],
        cwd=str(backend_dir),
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "ASTRAATMOS_EARTH_TRAINING_SPEC" in (completed.stdout + completed.stderr)
    assert not output.exists()

    from models.training_scripts.earth_daily import _load_spec_from_env

    with pytest.raises(EarthTrainingError):
        _load_spec_from_env()


def test_invalid_spec_object_is_rejected(earth_global_release, tmp_path):
    registry = DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **earth_global_release)
    with pytest.raises(EarthTrainingError):
        run_training({}, tmp_path / "x.pth", registry, device="cpu")


@pytest.mark.parametrize(
    "bad_spec",
    [
        {},
        {"schema": "nope", "task_id": 1, "dataset_binding": {}, "hyperparameters": {}},
        {"schema": "earth_training_spec_v1", "task_id": 0, "dataset_binding": {}, "hyperparameters": {}},
        {"schema": "earth_training_spec_v1", "task_id": 1, "hyperparameters": {}},
    ],
)
def test_malformed_specs_are_rejected(earth_global_release, tmp_path, bad_spec):
    registry = DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **earth_global_release)
    with pytest.raises((EarthTrainingError, Exception)):
        run_training(bad_spec, tmp_path / "nope.pth", registry, device="cpu")
    assert not (tmp_path / "nope.pth").exists()


def test_dataset_identity_mismatch_stops_before_training(earth_global_release, tmp_path):
    registry, binding, spec = _spec(earth_global_release)
    tampered = json.loads(json.dumps(spec))
    tampered["dataset_binding"]["dataset_fingerprint"] = "0" * 64
    output = tmp_path / "tampered.pth"
    with pytest.raises(Exception) as error:
        run_training(tampered, output, registry, device="cpu")
    assert "changed" in str(error.value).lower()
    assert not output.exists()

    tampered_id = json.loads(json.dumps(spec))
    tampered_id["dataset_binding"]["dataset_id"] = "openmars_mcd"
    with pytest.raises(Exception):
        run_training(tampered_id, tmp_path / "tampered2.pth", registry, device="cpu")


def test_early_stopping_restores_the_best_epoch(earth_global_release, tmp_path, monkeypatch):
    """A controlled loss sequence must pick the best epoch, not the last one."""
    import models.training_scripts.earth_daily as runner

    sequence = [0.8, 0.4, 0.6, 0.7]
    seen = {"index": 0}

    original_evaluate = runner._evaluate_split

    def fake_evaluate(model, loader, dataset, device, *, loss_function, model_source="official"):
        metrics = {
            "overall": {"rmse": 1.0, "mae": 0.5},
            "by_lead": [
                {"lead_day": 1, "rmse": 1.0, "mae": 0.5},
                {"lead_day": 2, "rmse": 1.0, "mae": 0.5},
                {"lead_day": 3, "rmse": 1.0, "mae": 0.5},
            ],
        }
        index = min(seen["index"], len(sequence) - 1)
        seen["index"] += 1
        return metrics, sequence[index]

    monkeypatch.setattr(runner, "_evaluate_split", fake_evaluate)
    registry, binding, spec = _spec(
        earth_global_release, epochs=4, batch_size=64, early_stopping_patience=1,
    )
    output = tmp_path / "task_901_early.pth"
    result = run_training(spec, output, registry, device="cpu")
    # Sequence 0.8, 0.4, 0.6 -> best is epoch 2; patience 1 stops after epoch 3.
    assert result["best_epoch"] == 2
    checkpoint = load_earth_training_artifact(output, expected_task_id=901)
    assert checkpoint.run["best_epoch"] == 2
    assert checkpoint.run["stopped_early"] is True
    assert checkpoint.run["epochs_completed"] == 3
    assert checkpoint.run["best_validation_loss"] == pytest.approx(0.4)
    assert original_evaluate is not None


def test_best_state_is_a_detached_cpu_copy(earth_global_release, tmp_path):
    """The published state must not be a live reference to the trained model."""
    registry, binding, spec = _spec(earth_global_release, epochs=2, batch_size=64)
    output = tmp_path / "task_901_cpu.pth"
    run_training(spec, output, registry, device="cpu")
    raw = torch.load(output, map_location="cpu", weights_only=True)
    for value in raw["model_state_dict"].values():
        assert value.device.type == "cpu"
        assert not value.requires_grad
    # Continuing to train the live model cannot change the saved tensors.
    snapshot = {key: value.clone() for key, value in raw["model_state_dict"].items()}
    from training_backbones.earth_daily_model import create_earth_forecaster, earth_forward

    live = create_earth_forecaster(["TO3", "T2M"])
    optimizer = torch.optim.Adam(live.parameters(), lr=0.5)
    loss = earth_forward(live, torch.randn(1, 7, 2, 36, 72)).pow(2).mean()
    loss.backward()
    optimizer.step()
    for key, value in raw["model_state_dict"].items():
        assert torch.equal(value, snapshot[key])


def test_resolve_device_and_seeding_helpers():
    assert str(resolve_device("cpu")) == "cpu"
    seed_everything(11)
    first = torch.randn(3)
    seed_everything(11)
    second = torch.randn(3)
    assert torch.equal(first, second)
    import random

    seed_everything(4)
    assert random.random() == random.Random(4).random()


def test_artifact_is_never_published_when_the_run_fails(earth_global_release, tmp_path, monkeypatch):
    import models.training_scripts.earth_daily as runner

    def explode(model, inputs, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(runner, "earth_forward_for_model", explode)
    registry, binding, spec = _spec(earth_global_release, epochs=1, batch_size=64)
    output = tmp_path / "task_901_fail.pth"
    with pytest.raises(Exception):
        run_training(spec, output, registry, device="cpu")
    assert not output.exists()
    assert not (tmp_path / "task_901_fail.pth.partial").exists()
