import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.prediction_horizon import (  # noqa: E402
    MAX_PREDICTION_HORIZON,
    get_trained_horizon,
    validate_prediction_horizon,
)


def test_reads_supported_trained_horizons():
    assert MAX_PREDICTION_HORIZON == 30
    assert get_trained_horizon({"horizon": 1}) == 1
    assert get_trained_horizon({"horizon": 30}) == 30


@pytest.mark.parametrize("hyperparameters", [
    {},
    {"horizon": None},
    {"horizon": 0},
    {"horizon": 31},
    {"horizon": 2.5},
    {"horizon": True},
    {"horizon": "not-a-number"},
])
def test_rejects_invalid_trained_horizon_metadata(hyperparameters):
    with pytest.raises(ValueError, match="valid trained output horizon"):
        get_trained_horizon(hyperparameters)


def test_accepts_request_at_the_trained_model_horizon():
    assert validate_prediction_horizon({"horizon": 10}, 10) == 10


def test_rejects_request_above_the_trained_model_horizon():
    with pytest.raises(ValueError, match="requested prediction horizon 11 exceeds trained model horizon 10"):
        validate_prediction_horizon({"horizon": 10}, 11)


@pytest.mark.parametrize("requested", [0, 31, 2.5, True])
def test_rejects_requests_outside_the_project_range(requested):
    with pytest.raises(ValueError, match="between 1 and 30"):
        validate_prediction_horizon({"horizon": 30}, requested)


def test_comparison_contract_is_the_smallest_selected_model_horizon():
    trained_horizons = [
        get_trained_horizon({"horizon": 10}),
        get_trained_horizon({"horizon": 6}),
        get_trained_horizon({"horizon": 18}),
    ]

    assert min(trained_horizons) == 6
    for trained_horizon in trained_horizons:
        assert validate_prediction_horizon({"horizon": trained_horizon}, 6) == 6
