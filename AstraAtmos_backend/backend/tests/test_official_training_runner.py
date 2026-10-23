import importlib.util
import sys
import uuid
from pathlib import Path

import pytest
import torch
import torch.nn as nn


BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BACKEND_DIR / "models" / "training_scripts" / "demo3.py"


def load_demo3_module():
    module_name = f"astraatmos_official_runner_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyOfficialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(4, 4)
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.forecast_head = nn.Linear(4, 1)


def test_official_runner_uses_central_mcd_directory(tmp_path, monkeypatch):
    demo3 = load_demo3_module()
    expected_mcd_dir = tmp_path / "resolved-mcd"
    captured = {}

    class PreparationObserved(Exception):
        pass

    def capture_training_data(**kwargs):
        captured.update(kwargs)
        raise PreparationObserved

    monkeypatch.setattr(demo3, "MCD_DIR", expected_mcd_dir)
    monkeypatch.setattr(demo3, "_prepare_training_data", capture_training_data)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), "--output_path", str(tmp_path / "model.pth")],
    )

    with pytest.raises(PreparationObserved):
        demo3.main()

    assert captured["mcd_dir"] == expected_mcd_dir


def test_official_runner_strictly_loads_transfer_weights_and_freezes_backbone(tmp_path, monkeypatch):
    demo3 = load_demo3_module()
    source_model = TinyOfficialModel()
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.fill_(0.25)

    weight_path = tmp_path / "official_source.pth"
    torch.save(source_model.state_dict(), weight_path)
    monkeypatch.setenv("ASTRAATMOS_TRANSFER_WEIGHT_PATH", str(weight_path))

    target_model = TinyOfficialModel()
    demo3.apply_transfer_learning(
        target_model,
        {
            "transfer_learning": True,
            "transfer_load_mode": "strict",
            "freeze_mode": "backbone",
        },
        torch.device("cpu"),
    )

    for name, tensor in target_model.state_dict().items():
        assert torch.equal(tensor, source_model.state_dict()[name])
    assert any(not parameter.requires_grad for parameter in target_model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in target_model.parameters())


def test_official_runner_rejects_incompatible_transfer_weights(tmp_path, monkeypatch):
    demo3 = load_demo3_module()
    source_model = TinyOfficialModel()
    state_dict = source_model.state_dict()
    state_dict.pop(next(iter(state_dict)))

    weight_path = tmp_path / "incompatible.pth"
    torch.save(state_dict, weight_path)
    monkeypatch.setenv("ASTRAATMOS_TRANSFER_WEIGHT_PATH", str(weight_path))

    with pytest.raises(RuntimeError, match="Missing key"):
        demo3.apply_transfer_learning(
            TinyOfficialModel(),
            {
                "transfer_learning": True,
                "transfer_load_mode": "strict",
                "freeze_mode": "none",
            },
            torch.device("cpu"),
        )
