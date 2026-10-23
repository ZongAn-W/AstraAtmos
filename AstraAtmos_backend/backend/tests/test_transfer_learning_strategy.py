import sys
from pathlib import Path

import torch.nn as nn

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.transfer_learning_strategy import apply_freeze_strategy  # noqa: E402


class TinyOfficialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Conv2d(1, 2, kernel_size=1)
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=1),
            nn.ReLU(),
        )
        self.forecast_head = nn.Conv2d(4, 1, kernel_size=1)


class TinyUploadedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.head = nn.Linear(8, 1)


class TinySequentialHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 1),
        )


def _trainable_names(model):
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def test_freeze_none_leaves_all_parameters_trainable():
    model = TinyOfficialModel()

    report = apply_freeze_strategy(model, "none")

    assert report["mode"] == "none"
    assert report["trainable_parameter_count"] == report["total_parameter_count"]
    assert _trainable_names(model) == {name for name, _ in model.named_parameters()}


def test_freeze_backbone_keeps_adapter_and_head_trainable():
    model = TinyOfficialModel()

    report = apply_freeze_strategy(model, "backbone")

    assert report["mode"] == "backbone"
    assert _trainable_names(model) == {
        "projector.weight",
        "projector.bias",
        "forecast_head.weight",
        "forecast_head.bias",
    }


def test_freeze_head_keeps_only_output_head_trainable():
    model = TinyUploadedModel()

    report = apply_freeze_strategy(model, "head")

    assert report["mode"] == "head"
    assert _trainable_names(model) == {"head.weight", "head.bias"}


def test_freeze_head_supports_sequential_output_heads():
    model = TinySequentialHeadModel()

    report = apply_freeze_strategy(model, "head")

    assert report["mode"] == "head"
    assert _trainable_names(model) == {"head.1.weight", "head.1.bias"}


def test_freeze_head_fails_when_no_head_is_detected():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    try:
        apply_freeze_strategy(model, "head")
    except ValueError as exc:
        assert "No trainable parameters" in str(exc)
    else:
        raise AssertionError("Expected missing head modules to fail")


if __name__ == "__main__":
    test_freeze_none_leaves_all_parameters_trainable()
    test_freeze_backbone_keeps_adapter_and_head_trainable()
    test_freeze_head_keeps_only_output_head_trainable()
    test_freeze_head_supports_sequential_output_heads()
    test_freeze_head_fails_when_no_head_is_detected()
    print("transfer learning strategy tests passed")
