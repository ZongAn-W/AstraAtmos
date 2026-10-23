import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_identity import (  # noqa: E402
    DATASET_IDS,
    DatasetRequestError,
    build_dataset_fingerprint,
    infer_legacy_identity,
    require_training_dataset,
    resolve_dataset_id,
)


@pytest.mark.parametrize("top,hypers,expected", [
    (None, {}, "openmars_mcd"),
    (None, {"training_dataset": "mcd_overview"}, "mcd_overview"),
    ("mcd_overview", {}, "mcd_overview"),
    (" MCD_OVERVIEW ", {"training_dataset": "mcd_overview"}, "mcd_overview"),
    ("earth_merra2_3hourly_v1", {}, "earth_merra2_3hourly_v1"),
])
def test_resolve_dataset_id(top, hypers, expected):
    assert resolve_dataset_id(top, hypers) == expected


@pytest.mark.parametrize("top,hypers,code", [
    ("", {}, "invalid_dataset_id"),
    (None, {"training_dataset": 7}, "invalid_dataset_id"),
    ("missing", {}, "unknown_dataset"),
    ("openmars_mcd", {"training_dataset": "mcd_overview"}, "dataset_id_conflict"),
    (None, {"dataset_fingerprint": "forged"}, "client_identity_not_allowed"),
])
def test_rejects_bad_dataset_requests(top, hypers, code):
    with pytest.raises(DatasetRequestError) as exc:
        resolve_dataset_id(top, hypers)
    assert exc.value.code == code


@pytest.mark.parametrize("field", [
    "dataset_id",
    "dataset_version",
    "dataset_fingerprint",
    "dataset_identity_status",
    "dataset_snapshot",
])
def test_rejects_every_server_owned_identity_field_in_hyperparameters(field):
    with pytest.raises(DatasetRequestError) as exc:
        resolve_dataset_id(None, {field: "anything"})
    assert exc.value.code == "client_identity_not_allowed"
    assert exc.value.status_code == 400


def test_request_error_is_value_error_with_status():
    error = DatasetRequestError("unknown_dataset", "Unknown dataset id")
    assert isinstance(error, ValueError)
    assert error.status_code == 400

    unsupported = DatasetRequestError(
        "dataset_training_not_supported", "Earth training is not connected yet", status_code=409
    )
    assert isinstance(unsupported, ValueError)
    assert unsupported.status_code == 409


def test_registered_ids_are_stable_and_ordered():
    assert DATASET_IDS == (
        "openmars_mcd", "mcd_overview", "earth_merra2_daily_v1", "earth_merra2_daily_v2",
        "earth_merra2_3hourly_v1",
        "earth_merra2_3hourly_5deg_v1",
    )


@pytest.mark.parametrize("value,expected", [
    (None, "openmars_mcd"),
    ("openmars_mcd", "openmars_mcd"),
    (" MCD_OVERVIEW ", "mcd_overview"),
])
def test_require_training_dataset_accepts_mars_ids(value, expected):
    assert require_training_dataset(value) == expected


@pytest.mark.parametrize("value,code,status", [
    ("earth_merra2_daily_v1", "dataset_training_not_supported", 409),
    ("missing", "unknown_dataset", 400),
    (7, "invalid_dataset_id", 400),
    # A blank id is an explicit bad request, not a request to use the default.
    ("", "invalid_dataset_id", 400),
])
def test_require_training_dataset_rejects_unsupported_ids(value, code, status):
    with pytest.raises(DatasetRequestError) as exc:
        require_training_dataset(value)
    assert exc.value.code == code
    assert exc.value.status_code == status


def test_fingerprint_depends_on_every_identity_component():
    base = build_dataset_fingerprint("earth_merra2_daily_v1", "v1", "a" * 64, "b" * 64)
    assert len(base) == 64
    assert base == base.lower()
    assert base == build_dataset_fingerprint(
        "earth_merra2_daily_v1", "v1", "a" * 64, "b" * 64
    )

    variants = [
        build_dataset_fingerprint("mcd_overview", "v1", "a" * 64, "b" * 64),
        build_dataset_fingerprint("earth_merra2_daily_v1", "v2", "a" * 64, "b" * 64),
        build_dataset_fingerprint("earth_merra2_daily_v1", "v1", "c" * 64, "b" * 64),
        build_dataset_fingerprint("earth_merra2_daily_v1", "v1", "a" * 64, "d" * 64),
        build_dataset_fingerprint("earth_merra2_daily_v1", None, "a" * 64, "b" * 64),
    ]
    assert base not in variants
    assert len(set(variants)) == len(variants)


def test_fingerprint_matches_locked_earth_release_combination():
    assert build_dataset_fingerprint(
        "earth_merra2_daily_v1",
        "v1",
        "1d346a059aff57964cb726dde7113fc495adb4b8245f533911dc6f3d63e52d7e",
        "c21c4ddcb03a69d2eb504f746c86b3a15cf33a06929044f50da768bdfc45cf74",
    ) == "74ce14752cb83932b29b458d9c19a61804861c270e6fec33f55316778ad64d31"


@pytest.mark.parametrize("raw,expected_id,basis,status", [
    ('{"training_dataset": "openmars_mcd"}', "openmars_mcd", "explicit_training_dataset", "legacy_inferred"),
    ('{"training_dataset": "mcd_overview"}', "mcd_overview", "explicit_training_dataset", "legacy_inferred"),
    ('{"training_dataset": " MCD_Overview "}', "mcd_overview", "explicit_training_dataset", "legacy_inferred"),
    ('{"epochs": 5}', "openmars_mcd", "historical_default", "legacy_inferred"),
    ('{"training_dataset": null}', "openmars_mcd", "historical_default", "legacy_inferred"),
    ('{"training_dataset": ""}', "openmars_mcd", "historical_default", "legacy_inferred"),
    ('{"training_dataset": "   "}', "openmars_mcd", "historical_default", "legacy_inferred"),
    ('{}', "openmars_mcd", "historical_default", "legacy_inferred"),
    ("", "openmars_mcd", "historical_default", "legacy_inferred"),
    (None, "openmars_mcd", "historical_default", "legacy_inferred"),
    ('{"training_dataset": "earth_merra2_daily_v1"}', None, "unknown_training_dataset", "legacy_unknown"),
    ('{"training_dataset": "missing"}', None, "unknown_training_dataset", "legacy_unknown"),
    ('{"training_dataset": 7}', None, "unknown_training_dataset", "legacy_unknown"),
    ('{"training_dataset": ["openmars_mcd"]}', None, "unknown_training_dataset", "legacy_unknown"),
    ("not-json", None, "invalid_hyperparameters", "legacy_unknown"),
    ("[1, 2]", None, "invalid_hyperparameters", "legacy_unknown"),
    ("7", None, "invalid_hyperparameters", "legacy_unknown"),
])
def test_infer_legacy_identity_matrix(raw, expected_id, basis, status):
    result = infer_legacy_identity(raw)

    assert result["dataset_id"] == expected_id
    assert result["dataset_version"] is None
    assert result["dataset_fingerprint"] is None
    assert result["dataset_identity_status"] == status
    snapshot = json.loads(result["dataset_snapshot"])
    assert snapshot == {
        "planet": "mars",
        "binding_basis": basis,
        "version_status": "unknown",
    }
    assert result["dataset_snapshot"] == json.dumps(snapshot, sort_keys=True)


def test_infer_legacy_identity_never_raises_and_never_defaults_unknown_sources():
    # Earth is a registered id but not a legacy Mars source: it must stay unknown.
    assert infer_legacy_identity('{"training_dataset": "earth_merra2_daily_v1"}')["dataset_id"] is None
    # Truncated JSON must not be guessed as the historical default.
    assert infer_legacy_identity('{"training_dataset": "openmars_mcd"')["dataset_id"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
