import json
from types import SimpleNamespace

import pytest

from services.prediction_analysis_cache import (
    CACHE_SCHEMA_VERSION,
    CachePayloadError,
    build_artifact_fingerprint,
    build_request_hash,
    decode_payload,
    encode_payload,
)


def test_request_hash_normalizes_effective_inputs():
    assert build_request_hash(
        "prediction", {"ls_start": 0, "horizon": 3}
    ) == build_request_hash(
        "prediction", {"ls_start": 360, "horizon": 3}
    )
    assert build_request_hash(
        "prediction",
        {
            "ls_start": 90,
            "horizon": 3,
            "selected_variables": ["Temperature"],
        },
    ) == build_request_hash(
        "prediction",
        {
            "ls_start": 90.0,
            "horizon": 3,
            "selected_variables": ["U_Wind"],
        },
    )
    assert build_request_hash(
        "pfi",
        {
            "horizon": 3,
            "selected_variables": [" U_Wind ", "Temperature", "U_Wind"],
            "ls_start": 90,
        },
    ) == build_request_hash(
        "pfi",
        {
            "horizon": 3,
            "selected_variables": ["Temperature", "U_Wind"],
        },
    )
    assert build_request_hash("metrics", {"horizon": 2}) != build_request_hash(
        "metrics", {"horizon": 3}
    )
    assert build_request_hash(
        "prediction", {"ls_start": 90, "horizon": 3, "mars_year": 27}
    ) == build_request_hash(
        "prediction", {"ls_start": 90, "horizon": 3, "mars_year": 28}
    )


def test_artifact_fingerprint_changes_with_model_training_data_and_version(tmp_path):
    model_path = tmp_path / "model.pth"
    openmars = tmp_path / "openmars"
    mcd = tmp_path / "mcd"
    openmars.mkdir()
    mcd.mkdir()
    model_path.write_bytes(b"weights-v1")
    (openmars / "MY27.nc").write_bytes(b"openmars-v1")
    (mcd / "MY27.nc").write_bytes(b"mcd-v1")
    task = SimpleNamespace(
        id=9,
        model_source="official",
        uploaded_model_id=None,
        uploaded_model_version=None,
        output_model_path=str(model_path),
        hyperparameters=json.dumps({"horizon": 3, "window": 3}),
    )
    data_dirs = {
        "ASTRAATMOS_OPENMARS_DIR": str(openmars),
        "ASTRAATMOS_MCD_DIR": str(mcd),
    }

    original = build_artifact_fingerprint(
        task, data_dirs, schema_version=CACHE_SCHEMA_VERSION
    )
    same = build_artifact_fingerprint(
        task, data_dirs, schema_version=CACHE_SCHEMA_VERSION
    )
    assert same == original

    model_path.write_bytes(b"weights-version-2")
    assert (
        build_artifact_fingerprint(
            task, data_dirs, schema_version=CACHE_SCHEMA_VERSION
        )
        != original
    )

    model_path.write_bytes(b"weights-v1")
    task.hyperparameters = json.dumps({"horizon": 4, "window": 3})
    assert (
        build_artifact_fingerprint(
            task, data_dirs, schema_version=CACHE_SCHEMA_VERSION
        )
        != original
    )

    task.hyperparameters = json.dumps({"horizon": 3, "window": 3})
    (mcd / "MY27.nc").write_bytes(b"mcd-version-2")
    assert (
        build_artifact_fingerprint(
            task, data_dirs, schema_version=CACHE_SCHEMA_VERSION
        )
        != original
    )
    assert (
        build_artifact_fingerprint(
            task, data_dirs, schema_version=CACHE_SCHEMA_VERSION + 1
        )
        != original
    )


def test_payload_codec_is_versioned_deterministic_and_rejects_wrong_type():
    result = {"overall": {"rmse": 1.25}, "per_step": []}
    encoded = encode_payload("metrics", result)

    assert encoded == encode_payload("metrics", result)
    assert decode_payload("metrics", encoded) == result
    with pytest.raises(CachePayloadError, match="analysis type"):
        decode_payload("pfi", encoded)
    with pytest.raises(CachePayloadError):
        decode_payload("metrics", b"not-gzip")
