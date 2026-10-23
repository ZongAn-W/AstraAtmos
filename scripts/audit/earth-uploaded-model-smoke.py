"""Exercise the full uploaded-Earth path against the real v2 package.

This is a developer smoke check, not part of the pytest suite: it validates the
sample model, trains it on the real Earth release through the same runner the web
task uses, then rebuilds it and predicts from the checkpoint alone.

Run from ``AstraAtmos_backend/backend``::

    & (Join-Path $env:CONDA_PREFIX 'python.exe') ../../scripts/audit/earth-uploaded-model-smoke.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2] / "AstraAtmos_backend" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402

from config import EARTH_MERRA2_DIR, EARTH_MERRA2_V1_DIR  # noqa: E402
from models.training_scripts.earth_daily import run_training  # noqa: E402
from services.dataset_registry import DatasetRegistry  # noqa: E402
from services.earth_training_artifact import (  # noqa: E402
    build_earth_model_from_checkpoint,
    load_earth_training_artifact,
)
from services.earth_training_contract import build_earth_training_spec  # noqa: E402
from services.uploaded_model_source import verify_reference_source  # noqa: E402
from services.user_model_validator import UserModelValidator  # noqa: E402


class FakePackage:
    """Stands in for a UserModelPackage row without touching the database."""

    def __init__(self, model_id, name, version, path, source_bytes):
        self.id = model_id
        self.display_name = name
        self.version = version
        self.storage_path = str(path)
        self.original_filename = path.name
        self.content_hash = hashlib.sha256(source_bytes).hexdigest()
        self.validation_status = "valid"


def main() -> int:
    sample = Path(__file__).with_name("test_models") / "EarthConvBaseline.py"
    source_bytes = sample.read_bytes()
    workdir = Path(tempfile.mkdtemp(prefix="earth-uploaded-smoke-"))
    model_path = workdir / sample.name
    model_path.write_bytes(source_bytes)

    print("[1] 校验上传模型（含 Earth dry-run）")
    validator = UserModelValidator()
    result = validator.validate_file(model_path)
    print(f"    mars-ok={result.ok} earth-ok={result.earth_ok} shape={result.earth_output_shape}")
    print(f"    datasets={json.dumps(result.datasets, ensure_ascii=False)}")
    if result.earth_errors:
        print(f"    earth_errors={result.earth_errors}")
    assert result.ok, result.errors
    assert result.earth_ok, result.earth_errors
    assert result.earth_output_shape == [1, 3, 1, 36, 72]

    print("[2] 地球兼容性判定（非 Earth 模型应被拒）")
    from training_backbones.uploaded_model_earth_gate import evaluate_package_earth_compatibility

    package = FakePackage("smoke-pkg", "EarthConvBaseline", 1, model_path, source_bytes)
    verdict = evaluate_package_earth_compatibility(package, validator=validator)
    print(f"    compatible={verdict.compatible} reasons={verdict.reasons}")
    assert verdict.compatible, verdict.reasons

    mars_only = workdir / "mars_only.source"
    mars_only.write_text(
        "import torch\nfrom torch import nn\n"
        "MODEL_SPEC = {'name': 'MarsOnly', 'parameters': {}}\n"
        "class M(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x[:, -1:]\n"
        "def build_model(config):\n"
        "    return M()\n",
        encoding="utf-8",
    )
    mars_verdict = evaluate_package_earth_compatibility(
        FakePackage("mars-pkg", "MarsOnly", 1, mars_only, mars_only.read_bytes()),
        validator=validator,
    )
    print(f"    mars-only compatible={mars_verdict.compatible} reasons={mars_verdict.reasons}")
    assert not mars_verdict.compatible

    print("[3] 用真实 v2 小包训练上传模型（1 epoch）")
    registry = DatasetRegistry(
        EARTH_MERRA2_DIR,
        earth_dataset_id="earth_merra2_daily_v2",
        legacy_earth_package_dir=EARTH_MERRA2_V1_DIR,
    )
    binding = registry.build_training_binding("earth_merra2_daily_v2")
    reference = {
        "package_id": package.id,
        "display_name": package.display_name,
        "version": package.version,
        "content_hash": package.content_hash,
        "source_path": package.storage_path,
        "source_text": source_bytes.decode("utf-8"),
        "param_schema": result.param_schema,
        "custom_model_params": {"hidden_dim": 8, "dropout": 0.0},
    }
    spec = build_earth_training_spec(
        task_id=900001,
        dataset_binding=binding,
        hyperparameters={
            "training_dataset": "earth_merra2_daily_v2",
            "model_source": "uploaded",
            "model_architecture": "uploaded",
            "selected_channels": ["U10M", "T2M"],
            "epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.001,
            "seed": 11,
        },
        uploaded_model=reference,
    )
    output = workdir / "task_900001_uploaded.pth"
    summary = run_training(spec, output, registry, device="cpu")
    print(f"    best_epoch={summary['best_epoch']} model_source={summary['model_source']}")
    print(f"    identity={json.dumps(summary['model_identity'], ensure_ascii=False)}")
    print(f"    test={json.dumps(summary['metrics']['splits']['test']['overall'])}")
    assert summary["model_source"] == "uploaded"
    assert output.is_file()

    print("[4] 独立重载并预测（模拟服务重启）")
    checkpoint = load_earth_training_artifact(output, expected_binding=binding, expected_task_id=900001)
    print(f"    model_ref.source={checkpoint.model_source} arch={checkpoint.model_ref.get('architecture')}")
    print(f"    source_available={verify_reference_source(checkpoint.uploaded_model)['status']}")
    model, warnings = build_earth_model_from_checkpoint(checkpoint)
    print(f"    rebuild warnings={list(warnings)}")
    assert not warnings, "the on-disk source should still be intact at this point"

    from services.earth_prediction_service import run_earth_prediction

    class FakeTask:
        pass

    task = FakeTask()
    task.id = 900001
    task.dataset_id = "earth_merra2_daily_v2"
    task.dataset_version = "v2"
    task.dataset_fingerprint = binding["dataset_fingerprint"]
    task.dataset_identity_status = "verified"
    task.dataset_snapshot = binding["dataset_snapshot"]
    task.status = "completed"
    task.output_model_path = str(output)
    task.hyperparameters = json.dumps(
        {"training_dataset": "earth_merra2_daily_v2", "selected_channels": ["U10M", "T2M"]}
    )
    prediction = run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    print(f"    target_dates={prediction['target_dates']}")
    print(f"    model={json.dumps(prediction['model'], ensure_ascii=False)}")
    print(f"    metrics={json.dumps(prediction['metrics']['overall'])}")
    assert prediction["target_unit"] == "DU"
    assert prediction["model"]["model_source"] == "uploaded"
    assert len(prediction["prediction"]) == 3
    assert np.isfinite(np.asarray(prediction["prediction"][0]["field"], dtype="float64")).all()

    print("[5] 原始模型文件删除后仍可预测（使用 checkpoint 内嵌副本）")
    model_path.unlink()
    reloaded = load_earth_training_artifact(output, expected_binding=binding, expected_task_id=900001)
    model_after, warnings_after = build_earth_model_from_checkpoint(reloaded)
    print(f"    warnings={list(warnings_after)}")
    assert warnings_after, "deleting the file must be reported"
    prediction_after = run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    same = np.allclose(
        np.asarray(prediction["prediction"][0]["field"], dtype="float64"),
        np.asarray(prediction_after["prediction"][0]["field"], dtype="float64"),
    )
    print(f"    prediction identical to the pre-deletion run: {same}")
    assert same
    del model, model_after

    print("[6] 篡改模型文件后使用 checkpoint 内嵌副本并给出警告")
    model_path.write_bytes(source_bytes + b"\n# tampered\n")
    reloaded = load_earth_training_artifact(output, expected_binding=binding, expected_task_id=900001)
    _, warnings_tampered = build_earth_model_from_checkpoint(reloaded)
    print(f"    warnings={list(warnings_tampered)}")
    assert warnings_tampered and "no longer matches" in warnings_tampered[0]
    prediction_tampered = run_earth_prediction(task, "2021-07-08", registry, device="cpu")
    assert np.allclose(
        np.asarray(prediction["prediction"][0]["field"], dtype="float64"),
        np.asarray(prediction_tampered["prediction"][0]["field"], dtype="float64"),
    )

    print("\n=== SMOKE OK ===")
    print(f"artifacts under {workdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
