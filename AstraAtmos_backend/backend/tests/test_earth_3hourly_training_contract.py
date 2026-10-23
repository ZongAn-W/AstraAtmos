"""Dataset selection fixes cadence, windows and supported model sources."""

import copy

import pytest

from services.dataset_identity import DatasetRequestError, EARTH_DATASET_3HOURLY_ID
from services.earth_training_contract import (
    EARTH_3HOURLY_IMPLEMENTATION_ID,
    EARTH_IMPLEMENTATION_ID,
    build_earth_training_spec,
    canonical_channel_order,
    earth_training_profile,
    normalize_earth_training_hyperparameters,
    require_earth_training_configuration,
    split_window_counts,
)


def test_profile_selection_preserves_the_legacy_daily_profile():
    daily = earth_training_profile()
    assert daily == earth_training_profile("earth_merra2_daily_v2")
    assert daily == earth_training_profile("earth_merra2_daily_v1")
    assert (daily["window"], daily["horizon"], daily["step_unit"]) == (7, 3, "day")
    assert daily["implementation_id"] == EARTH_IMPLEMENTATION_ID
    assert "frequency_hours" not in daily
    profile = earth_training_profile(EARTH_DATASET_3HOURLY_ID)
    assert (profile["window"], profile["horizon"]) == (56, 24)
    assert (profile["step_unit"], profile["step"], profile["frequency_hours"]) == ("hour", 3, 3)
    assert profile["grid_shape"] == [240, 480]
    assert (profile["target"], profile["target_unit"]) == ("TO3", "DU")
    assert profile["model_architectures"] == ["dlinear", "uploaded"]
    assert profile["model_sources"] == ["official", "uploaded"]
    assert profile["implementation_id"] == EARTH_3HOURLY_IMPLEMENTATION_ID
    profile["optional_channels"].clear()
    assert earth_training_profile(EARTH_DATASET_3HOURLY_ID)["optional_channels"]


@pytest.mark.parametrize("entry", ["explicit", "legacy", "both"])
def test_normalizer_selects_new_windows_without_changing_channel_order(entry):
    hypers = {"selected_channels": ["SWGDN", "U10M"], "epochs": 1}
    kwargs = {}
    if entry in ("explicit", "both"):
        kwargs["dataset_id"] = EARTH_DATASET_3HOURLY_ID
    if entry in ("legacy", "both"):
        hypers["training_dataset"] = EARTH_DATASET_3HOURLY_ID.upper()
    normalized = normalize_earth_training_hyperparameters(hypers, **kwargs)
    assert (normalized["window"], normalized["horizon"]) == (56, 24)
    assert normalized["training_dataset"] == EARTH_DATASET_3HOURLY_ID
    assert canonical_channel_order(normalized["selected_channels"]) == ["TO3", "U10M", "SWGDN"]
    assert normalized["model_source"] == "official"


@pytest.mark.parametrize("hypers", [
    {"window": 0}, {"horizon": -1}, {"window": 56.0}, {"horizon": True},
    {"window": 241}, {"horizon": 241},
])
def test_new_task_never_coerces_or_downgrades_wrong_windows(hypers):
    with pytest.raises(DatasetRequestError) as error:
        normalize_earth_training_hyperparameters(hypers, dataset_id=EARTH_DATASET_3HOURLY_ID)
    assert (error.value.status_code, error.value.code) == (422, "invalid_earth_training_parameters")


@pytest.mark.parametrize('window,horizon', [(16, 8), (55, 25), (240, 1), (1, 240)])
def test_custom_threehour_windows_survive_normalization_and_spec(window, horizon):
    normalized = normalize_earth_training_hyperparameters(
        {'window': window, 'horizon': horizon}, dataset_id=EARTH_DATASET_3HOURLY_ID)
    assert (normalized['window'], normalized['horizon']) == (window, horizon)
    profile = earth_training_profile(EARTH_DATASET_3HOURLY_ID, normalized)
    assert (profile['window'], profile['horizon']) == (window, horizon)
    assert profile['frequency_hours'] == 3


@pytest.mark.parametrize("daily_id", ["earth_merra2_daily_v1", "earth_merra2_daily_v2"])
def test_daily_tasks_refuse_new_windows(daily_id):
    with pytest.raises(DatasetRequestError):
        normalize_earth_training_hyperparameters({"window": 56, "horizon": 24}, dataset_id=daily_id)
    normalized = normalize_earth_training_hyperparameters({}, dataset_id=daily_id)
    assert (normalized["window"], normalized["horizon"]) == (7, 3)


@pytest.mark.parametrize("hypers", [
    {"model_architecture": "simvp"}, {"use_sphere": True}, {"transfer_learning": True},
])
def test_new_task_rejects_unsupported_models_with_a_stable_409(hypers):
    with pytest.raises(DatasetRequestError) as error:
        normalize_earth_training_hyperparameters(hypers, dataset_id=EARTH_DATASET_3HOURLY_ID)
    assert (error.value.status_code, error.value.code) == (409, "dataset_training_configuration_not_supported")


def test_uploaded_configuration_requires_an_id_before_bad_dimensions():
    with pytest.raises(DatasetRequestError) as error:
        require_earth_training_configuration(
            model_source="uploaded", uploaded_model_id=None,
            hyperparameters={"window": 7}, dataset_id=EARTH_DATASET_3HOURLY_ID,
        )
    assert (error.value.status_code, error.value.code) == (422, "invalid_earth_training_parameters")


@pytest.mark.parametrize("operation", ["normalize", "configuration", "spec"])
def test_dataset_conflicts_are_rejected_before_profile_selection(operation):
    hypers = {"training_dataset": "earth_merra2_daily_v2"}
    with pytest.raises(DatasetRequestError) as error:
        if operation == "normalize":
            normalize_earth_training_hyperparameters(hypers, dataset_id=EARTH_DATASET_3HOURLY_ID)
        elif operation == "configuration":
            require_earth_training_configuration(
                model_source="official", uploaded_model_id=None,
                hyperparameters=hypers, dataset_id=EARTH_DATASET_3HOURLY_ID,
            )
        else:
            build_earth_training_spec(
                task_id=1, dataset_binding={"dataset_id": EARTH_DATASET_3HOURLY_ID}, hyperparameters=hypers,
            )
    assert (error.value.status_code, error.value.code) == (400, "dataset_id_conflict")


def test_internal_spec_freezes_new_identity_and_training_profile():
    binding = {
        "dataset_id": EARTH_DATASET_3HOURLY_ID, "dataset_version": "v1",
        "dataset_fingerprint": "a" * 64, "dataset_identity_status": "verified",
        "dataset_snapshot": {"planet": "earth", "time": {"kind": "datetime", "step": 3}},
    }
    before = copy.deepcopy(binding)
    spec = build_earth_training_spec(task_id=2, dataset_binding=binding, hyperparameters={"selected_channels": []})
    assert spec["hyperparameters"]["training_dataset"] == EARTH_DATASET_3HOURLY_ID
    assert (spec["hyperparameters"]["window"], spec["hyperparameters"]["horizon"]) == (56, 24)
    assert spec["training_profile"] == earth_training_profile(EARTH_DATASET_3HOURLY_ID)
    spec["dataset_binding"]["dataset_snapshot"]["time"]["step"] = 24
    assert binding == before
    with pytest.raises(DatasetRequestError) as error:
        build_earth_training_spec(task_id=2, dataset_binding=binding, hyperparameters={}, uploaded_model={"package_id": 3})
    assert error.value.status_code == 409


def test_full_release_sample_counts_use_steps_with_no_split_overlap():
    assert split_window_counts(
        {"train": 366 * 8, "validation": 181 * 8, "test": 184 * 8}, window=56, horizon=24,
    ) == {"train": 2849, "validation": 1369, "test": 1393}


@pytest.mark.parametrize("field", ["dataset_fingerprint", "dataset_version", "dataset_snapshot", "fingerprint", "snapshot"])
def test_new_normalizer_refuses_client_identity(field):
    with pytest.raises(DatasetRequestError) as error:
        normalize_earth_training_hyperparameters({field: "forged"}, dataset_id=EARTH_DATASET_3HOURLY_ID)
    assert error.value.code == "client_identity_not_allowed"
