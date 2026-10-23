"""The official three-hour Earth model runs the uploaded model's contract.

Both sources now share one invocation and one validation path
(``earth_3hourly_uploaded_contract.forward``); only the spatial profile still
differs between them. These tests pin that unification: the official model is
contract-prepared, callable with a single argument, and subject to the same
input/output checks. The retired daily profile keeps its own invocation.
"""

import pytest
import torch

from services.earth_model_source import (
    EarthModelBuildError,
    ModelSourcePlan,
    build_earth_model_for_plan,
    earth_forward_for_model,
)
from training_backbones.earth_3hourly_uploaded_contract import CONTRACT_SCHEMA

DATASET_ID = "earth_merra2_3hourly_v1"
DAILY_DATASET_ID = "earth_merra2_daily_v2"


def _official_plan(dataset_id=DATASET_ID, window=56, horizon=24):
    return ModelSourcePlan(
        model_source="official", input_channel_order=["TO3"], linear_hidden_layers=2,
        uploaded_model=None, dataset_id=dataset_id, window=window, horizon=horizon,
    )


def test_three_hour_official_model_is_contract_prepared_and_single_argument():
    model, config, warnings = build_earth_model_for_plan(_official_plan())
    assert warnings == ()
    assert getattr(model, "_astraatmos_earth_3hourly_contract", None) == CONTRACT_SCHEMA
    assert model._astraatmos_earth_window == 56
    assert tuple(model._astraatmos_earth_spatial_shape) == (24, 48)
    assert not getattr(model, "_astraatmos_earth_derived_inputs", {})

    inputs = torch.zeros(2, 56, 1, 24, 48)
    with torch.no_grad():
        assert model(inputs).shape == (2, 24, 1, 24, 48)
    assert config["implementation_id"] == "astraatmos_gridpoint_dlinear_3hourly_v1"


def test_thre_hour_official_forward_takes_the_shared_contract_path():
    model, _config, _warnings = build_earth_model_for_plan(_official_plan())
    inputs = torch.zeros(2, 56, 1, 24, 48)
    with torch.no_grad():
        output = earth_forward_for_model(
            model, inputs, model_source="official", horizon=24, height=24, width=48,
        )
    assert output.shape == (2, 24, 1, 24, 48) and output.dtype == torch.float32


@pytest.mark.parametrize("bad_inputs,reason", [
    (torch.zeros(2, 56, 1, 24, 48, dtype=torch.float64), "float64 input"),
    (torch.zeros(2, 56, 1, 12, 24), "wrong spatial tile"),
    (torch.zeros(2, 32, 1, 24, 48), "window that disagrees with the model"),
    (torch.full((2, 56, 1, 24, 48), float("nan")), "non-finite input"),
])
def test_shared_contract_validation_now_applies_to_the_official_model(bad_inputs, reason):
    model, _config, _warnings = build_earth_model_for_plan(_official_plan())
    with pytest.raises((EarthModelBuildError, ValueError)):
        with torch.no_grad():
            earth_forward_for_model(
                model, bad_inputs, model_source="official", horizon=24, height=24, width=48,
            )


def test_retired_daily_profile_keeps_its_own_invocation():
    model, _config, _warnings = build_earth_model_for_plan(_official_plan(DAILY_DATASET_ID, window=7, horizon=3))
    assert getattr(model, "_astraatmos_earth_3hourly_contract", None) is None
    inputs = torch.zeros(2, 7, 1, 36, 72)
    with torch.no_grad():
        assert earth_forward_for_model(
            model, inputs, model_source="official", horizon=3, height=36, width=72,
        ).shape == (2, 3, 1, 36, 72)
