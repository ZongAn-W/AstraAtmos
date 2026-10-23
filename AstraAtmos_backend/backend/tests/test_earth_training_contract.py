"""Earth training contract: strict parameters, channel order and profile."""

import copy

import pytest

from services.dataset_identity import DatasetRequestError
from services.earth_training_contract import (
    EARTH_ALLOWED_PARAMETER_KEYS,
    EARTH_GRID_SHAPE,
    EARTH_HORIZON,
    EARTH_OPTIONAL_CHANNELS,
    EARTH_TARGET_CHANNEL,
    EARTH_TARGET_UNIT,
    EARTH_TRAINING_SPEC_SCHEMA,
    EARTH_WINDOW,
    build_earth_training_spec,
    canonical_channel_order,
    earth_training_profile,
    input_units_for,
    is_earth_dataset_id,
    normalize_earth_training_hyperparameters,
    require_earth_training_configuration,
    split_window_counts,
)


def test_channel_order_and_ozone_only_selection():
    one = normalize_earth_training_hyperparameters({"selected_channels": []})
    assert one["selected_channels"] == []
    assert (one["window"], one["horizon"]) == (EARTH_WINDOW, EARTH_HORIZON)

    two = normalize_earth_training_hyperparameters({"selected_channels": ["SWGDN", "U10M"]})
    # Canonical order is TO3 first then U10M, V10M, T2M, SWGDN - never request order.
    assert two["selected_channels"] == ["U10M", "SWGDN"]


def test_missing_selection_defaults_to_every_auxiliary_channel():
    normalized = normalize_earth_training_hyperparameters({})
    assert normalized["selected_channels"] == list(EARTH_OPTIONAL_CHANNELS)
    assert (normalized["train_ratio"], normalized["validation_ratio"], normalized["test_ratio"]) == (0.7, 0.2, 0.1)


def test_custom_split_ratios_are_rejected_for_published_earth_splits():
    with pytest.raises(DatasetRequestError, match="published manifest splits"):
        normalize_earth_training_hyperparameters({
            "train_ratio": 0.6,
            "validation_ratio": 0.3,
            "test_ratio": 0.1,
        })


def test_explicit_to3_in_the_selection_is_ignored_not_duplicated():
    normalized = normalize_earth_training_hyperparameters({"selected_channels": ["TO3", "T2M"]})
    assert normalized["selected_channels"] == ["T2M"]
    assert canonical_channel_order(["TO3", "T2M"]) == [EARTH_TARGET_CHANNEL, "T2M"]


@pytest.mark.parametrize(
    "bad",
    [
        {"window": 3},
        {"window": 7.0},
        {"window": True},
        {"horizon": 5},
        {"epochs": 1.5},
        {"epochs": True},
        {"epochs": 0},
        {"epochs": 1001},
        {"batch_size": 65},
        {"batch_size": "8"},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
        {"learning_rate": 1.5},
        {"seed": -1},
        {"seed": 2**32},
        {"early_stopping_patience": 201},
        {"linear_hidden_layers": 5},
        {"selected_channels": ["U"]},
        {"selected_channels": ["U10M", "U10M"]},
        {"selected_channels": "U10M"},
        {"selected_channels": ["TO3", "TO3"]},
        {"_data_source": "personal"},
        {"device": "cuda:1"},
        {"normalization": {"mean": [1]}},
        {"target": "T2M"},
        {"custom_split": "train"},
    ],
)
def test_invalid_parameters_are_rejected_with_422(bad):
    with pytest.raises(DatasetRequestError) as error:
        normalize_earth_training_hyperparameters(bad)
    assert error.value.status_code == 422


@pytest.mark.parametrize(
    "field",
    ["dataset_version", "dataset_fingerprint", "dataset_identity_status", "dataset_snapshot"],
)
def test_client_supplied_identity_is_rejected_with_the_identity_code(field):
    """Identity stays server generated; the first-stage identity error is reused."""
    with pytest.raises(DatasetRequestError) as error:
        normalize_earth_training_hyperparameters({field: "x"})
    assert error.value.code == "client_identity_not_allowed"
    assert error.value.status_code == 400


def test_unsupported_configurations_are_rejected_with_409():
    with pytest.raises(DatasetRequestError) as architecture:
        normalize_earth_training_hyperparameters({"model_architecture": "simvp"})
    assert architecture.value.status_code == 409

    with pytest.raises(DatasetRequestError) as sphere:
        normalize_earth_training_hyperparameters({"use_sphere": True})
    assert sphere.value.status_code == 409

    with pytest.raises(DatasetRequestError) as transfer:
        normalize_earth_training_hyperparameters({"transfer_learning": True})
    assert transfer.value.status_code == 409

    # An unknown source is refused; 'uploaded' is now a supported source and is
    # covered by the uploaded-model contract tests.
    with pytest.raises(DatasetRequestError) as source:
        normalize_earth_training_hyperparameters({"model_source": "carrier-pigeon"})
    assert source.value.status_code == 409


def test_require_configuration_rejects_unsupported_before_parameters():
    # A genuine capability failure wins over parameter validation: an unknown model
    # source is refused with the capability code even though window=3 is also wrong.
    with pytest.raises(DatasetRequestError) as error:
        require_earth_training_configuration(
            model_source="carrier-pigeon",
            uploaded_model_id=None,
            hyperparameters={"window": 3},
        )
    assert error.value.status_code == 409
    assert error.value.code == "dataset_training_configuration_not_supported"

    # An uploaded-model request with a bad window fails on the parameter instead,
    # which is the documented precedence for supported sources.
    with pytest.raises(DatasetRequestError) as parameter_error:
        require_earth_training_configuration(
            model_source="uploaded",
            uploaded_model_id="model-1",
            hyperparameters={"window": 3},
        )
    assert parameter_error.value.status_code == 422

    accepted = require_earth_training_configuration(
        model_source="official",
        uploaded_model_id=None,
        hyperparameters={"selected_channels": ["U10M"], "epochs": 3},
    )
    assert accepted["epochs"] == 3
    assert accepted["selected_channels"] == ["U10M"]


def test_false_like_compatibility_flags_are_accepted():
    normalized = normalize_earth_training_hyperparameters({
        "use_sphere": False,
        "transfer_learning": False,
        "model_source": "official",
        "model_architecture": "dlinear",
    })
    assert normalized["use_sphere"] is False
    assert normalized["model_architecture"] == "dlinear"
    assert normalized["model_source"] == "official"


def test_normalization_never_sends_identity_or_internal_fields():
    normalized = normalize_earth_training_hyperparameters({"selected_channels": ["U10M"]})
    for key in normalized:
        assert key in EARTH_ALLOWED_PARAMETER_KEYS or key == "training_dataset"
        assert not key.startswith("_")
    assert "dataset_version" not in normalized
    assert "dataset_fingerprint" not in normalized
    assert "dataset_snapshot" not in normalized


def test_units_and_planet_identity():
    assert input_units_for(["TO3", "U10M", "SWGDN"]) == ["DU", "m s-1", "W m-2"]
    assert is_earth_dataset_id("earth_merra2_daily_v2")
    assert is_earth_dataset_id(" EARTH_MERRA2_DAILY_V1 ")
    assert not is_earth_dataset_id("openmars_mcd")
    assert not is_earth_dataset_id(None)


def test_profile_publishes_the_fixed_contract():
    profile = earth_training_profile()
    assert profile["profile_id"] == "earth_daily_dlinear_v1"
    assert profile["model_architectures"] == ["dlinear", "uploaded"]
    assert profile["model_sources"] == ["official", "uploaded"]
    assert profile["target"] == EARTH_TARGET_CHANNEL
    assert profile["target_unit"] == EARTH_TARGET_UNIT
    assert (profile["window"], profile["horizon"]) == (EARTH_WINDOW, EARTH_HORIZON)
    assert profile["supported_planet"] == "earth"
    assert profile["supports_sphere"] is False
    assert profile["supports_transfer_learning"] is False
    assert profile["grid_shape"] == list(EARTH_GRID_SHAPE)
    assert profile["default_selected_channels"] == list(EARTH_OPTIONAL_CHANNELS)

    # A caller must not be able to mutate the published profile in place.
    profile["window"] = 99
    assert earth_training_profile()["window"] == EARTH_WINDOW


def test_split_window_counts_follow_the_day_count():
    counts = split_window_counts({"train": 366, "validation": 181, "test": 184})
    assert counts == {"train": 357, "validation": 172, "test": 175}

    with pytest.raises(DatasetRequestError):
        split_window_counts({"train": 9})


def test_internal_spec_carries_identity_and_canonical_parameters():
    binding = {
        "dataset_id": "earth_merra2_daily_v2",
        "dataset_version": "v2",
        "dataset_fingerprint": "a" * 64,
        "dataset_identity_status": "verified",
        "dataset_snapshot": {"planet": "earth", "dataset_id": "earth_merra2_daily_v2"},
    }
    spec = build_earth_training_spec(
        task_id=7,
        dataset_binding=binding,
        hyperparameters={"selected_channels": ["SWGDN"], "epochs": 2},
    )
    assert spec["schema"] == EARTH_TRAINING_SPEC_SCHEMA
    assert spec["task_id"] == 7
    assert spec["dataset_binding"]["dataset_fingerprint"] == "a" * 64
    assert spec["hyperparameters"]["selected_channels"] == ["SWGDN"]
    assert spec["hyperparameters"]["window"] == EARTH_WINDOW

    # The spec is a copy: mutating it must not touch the caller's binding.
    frozen = copy.deepcopy(binding)
    spec["dataset_binding"]["dataset_snapshot"]["planet"] = "mars"
    assert binding == frozen
