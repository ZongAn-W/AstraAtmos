"""Is this uploaded model usable for the Earth MERRA-2 feed?

A model is Earth-compatible only when **all** of these hold:

1. its source file is present and hashes to the recorded digest (nothing drifted);
2. it passes the shared safety gate (torch/numpy imports only, no unsafe calls);
3. its ``MODEL_SPEC`` declares the ``earth_merra2`` feed;
4. that declaration does not ask for Mars-only auxiliary inputs (Ls, MOLA topography);
5. it actually builds and runs on the Earth tensor contract
   ``[B, 7, C, 36, 72] -> [B, 3, 1, 36, 72]`` for both boundary channel counts.

"Validated for Mars" is deliberately **not** part of this: a Mars-capable model is
never treated as Earth-capable by default, which is exactly the assumption this
gate exists to break.

The real work is delegated to :class:`services.user_model_validator.UserModelValidator`,
whose Earth dry-run builds and executes the model on that exact contract. This
module only adds the file-integrity and dataset-identity framing, so the upload
page, the compatibility endpoint and the training start gate all read the same
verdict.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.uploaded_model_source import hash_source_bytes
from training_backbones.uploaded_model_dataset_spec import (
    DatasetCapabilityError,
    declares_earth_feed,
    earth_incompatibility_reasons,
    normalize_dataset_declarations,
    EARTH_3HOURLY_FEED_KEY,
)
from training_backbones.uploaded_model_source_check import validate_uploaded_model_source


def build_validator() -> Any:
    """Return the shared upload validator (imported lazily to avoid cycles)."""
    from services.user_model_validator import UserModelValidator

    return UserModelValidator()


@dataclass
class EarthCompatibility:
    """The verdict for one uploaded model."""

    compatible: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    declares_earth: bool = False
    datasets: dict[str, Any] = field(default_factory=dict)
    output_shape: list[int] | None = None
    display_name: str | None = None
    version: int | None = None
    param_schema: dict[str, Any] = field(default_factory=dict)
    dataset_id: str = "earth_merra2"
    status: str = "unavailable"
    code: str | None = None

    def report(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "declares_earth_feed": self.declares_earth,
            "datasets": self.datasets,
            "output_shape": self.output_shape,
            "display_name": self.display_name,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "status": "available" if self.compatible else self.status,
            "code": self.code,
        }


def _static_declaration_check(
    source_text: str, filename: str, *, dataset_id: str = "earth_merra2"
) -> EarthCompatibility:
    """Cheap check with no runtime: syntax, safety and the dataset declaration.

    This is what explains *why* a model cannot serve Earth before paying for a
    build, and it is the only check available when no validator is supplied.
    """
    safety_errors = validate_uploaded_model_source(source_text, filename)
    if safety_errors:
        return EarthCompatibility(compatible=False, reasons=safety_errors)

    namespace: dict[str, Any] = {}
    try:
        exec(compile(source_text, filename, "exec"), namespace)  # noqa: S102 - AST-checked above
    except Exception as exc:  # noqa: BLE001 - reported as an incompatibility reason
        return EarthCompatibility(
            compatible=False,
            reasons=[f"the uploaded model source could not be executed: {exc}"],
        )

    model_spec = namespace.get("MODEL_SPEC")
    if not isinstance(model_spec, dict):
        return EarthCompatibility(compatible=False, reasons=["MODEL_SPEC must be exported as a dict"])
    if not callable(namespace.get("build_model")):
        return EarthCompatibility(
            compatible=False, reasons=["the model must export build_model(config)"]
        )

    try:
        datasets = normalize_dataset_declarations(model_spec)
    except DatasetCapabilityError as exc:
        return EarthCompatibility(compatible=False, reasons=[str(exc)])

    try:
        from training_backbones.uploaded_model_contract import normalize_auxiliary_inputs

        auxiliary_inputs = normalize_auxiliary_inputs(model_spec)
    except ValueError as exc:
        return EarthCompatibility(compatible=False, reasons=[str(exc)], datasets=datasets)

    declares = declares_earth_feed(model_spec)
    if dataset_id == EARTH_3HOURLY_FEED_KEY:
        try:
            feed = normalize_dataset_declarations(model_spec).get(dataset_id)
        except DatasetCapabilityError as exc:
            return EarthCompatibility(compatible=False, reasons=[str(exc)], datasets=datasets, dataset_id=dataset_id)
        if feed is None:
            return EarthCompatibility(
                compatible=False,
                reasons=[f"MODEL_SPEC must declare the {dataset_id} dataset feed"],
                declares_earth=False,
                datasets=datasets,
                dataset_id=dataset_id,
            )
        return EarthCompatibility(
            compatible=False,
            reasons=[],
            declares_earth=True,
            datasets=datasets,
            dataset_id=dataset_id,
        )
    # The channel count the platform would feed is 1..5; declaring the maximum makes
    # this a necessary condition rather than a guess.
    reasons = earth_incompatibility_reasons(
        model_spec,
        auxiliary_inputs,
        height=36,
        width=72,
        channel_count=5,
        window=7,
        horizon=3,
    )
    return EarthCompatibility(
        compatible=False,  # the tensor contract has not been exercised yet
        reasons=reasons,
        declares_earth=declares,
        datasets=datasets,
    )


def evaluate_earth_compatibility(
    *,
    source_text: str,
    filename: str = "uploaded_model.py",
) -> EarthCompatibility:
    """Report whether source text declares a usable Earth feed.

    This is the static half of the verdict: syntax, safety and the dataset
    declaration. Callers that need the executed proof should use
    :func:`evaluate_package_earth_compatibility`, which runs the validator's Earth
    dry-run and returns its verdict.
    """
    static = _static_declaration_check(source_text, filename)
    if not static.reasons:
        # The declaration is consistent, but the tensor contract has not been
        # exercised, so this is not by itself a green light.
        static.reasons = [
            "the Earth tensor contract was not exercised; revalidate the model so the "
            "platform can build it with Earth's 7 -> 3 window on the 36x72 grid"
        ]
    return static


def evaluate_package_earth_compatibility(
    package: Any, validator: Any = None, *, dataset_id: str = "earth_merra2", earth_probe=None
) -> EarthCompatibility:
    """Run the Earth check for a stored ``UserModelPackage`` row.

    The file is read and hashed first, so a drifted upload is reported as such
    instead of being dry-run and reported as a model problem.
    """
    storage_path = str(getattr(package, "storage_path", "") or "")
    display_name = getattr(package, "display_name", None)
    version = getattr(package, "version", None)
    if not storage_path or not Path(storage_path).is_file():
        return EarthCompatibility(
            compatible=False,
            reasons=["the uploaded model file is missing; re-upload or revalidate the model"],
            display_name=display_name,
            version=version,
            dataset_id=dataset_id,
            code="uploaded_model_missing",
        )

    raw = Path(storage_path).read_bytes()
    expected_hash = str(getattr(package, "content_hash", "") or "")
    if expected_hash and hash_source_bytes(raw) != expected_hash:
        return EarthCompatibility(
            compatible=False,
            reasons=[
                "the uploaded model file no longer matches the verified version; "
                "re-upload or revalidate the model"
            ],
            display_name=display_name,
            version=version,
            dataset_id=dataset_id,
            code="uploaded_model_tampered",
        )

    if dataset_id == EARTH_3HOURLY_FEED_KEY:
        return _evaluate_three_hour_package(package, raw, validator, earth_probe=earth_probe)

    source_text = raw.decode("utf-8")
    filename = f"{display_name or 'uploaded_model'}.py"
    static = _static_declaration_check(source_text, filename, dataset_id=dataset_id)
    if static.reasons:
        static.display_name = display_name
        static.version = version
        return static

    active_validator = validator or build_validator()
    # The dry-run must use a ``.py`` name: the stored file carries a ``.source``
    # suffix so it is never importable by accident, and the validator rejects that
    # extension - which would report a perfectly good model as unproven.
    with tempfile.TemporaryDirectory(prefix="astraatmos_earth_gate_") as temp_dir:
        probe_path = Path(temp_dir) / filename
        probe_path.write_bytes(raw)
        result = active_validator.validate_file(probe_path)
    if dataset_id == EARTH_3HOURLY_FEED_KEY:
        block = result.earth_compatibilities.get(dataset_id) if result.earth_compatibilities else None
        if not isinstance(block, dict):
            block = {"compatible": False, "errors": [
                "this model has no verified Earth three-hourly compatibility result"
            ], "output_shape": None}
        compatible = bool(block.get("compatible"))
        reasons = list(block.get("errors") or [])
        output_shape = block.get("output_shape")
    else:
        compatible = bool(result.earth_ok)
        reasons = list(result.earth_errors)
        output_shape = result.earth_output_shape
    verdict = EarthCompatibility(
        compatible=compatible,
        reasons=reasons,
        warnings=list(result.warnings),
        declares_earth=dataset_id in (result.datasets or {}),
        datasets=result.datasets,
        output_shape=output_shape,
        display_name=display_name,
        version=version,
        param_schema=dict(result.param_schema),
        dataset_id=dataset_id,
    )
    if not verdict.compatible and not verdict.reasons:
        verdict.reasons = [
            "the model did not pass the Earth dry-run and reported no specific reason"
        ]
    return verdict


def _evaluate_three_hour_package(package, raw, validator, *, earth_probe=None):
    from services.user_model_validator import VALIDATION_TIMEOUT_CODE, validation_report_timed_out
    from training_backbones.earth_3hourly_uploaded_contract import EVAL_BATCH_POLICY, contract_profile

    identity = dict(dataset_id=EARTH_3HOURLY_FEED_KEY,
                    display_name=getattr(package, "display_name", None), version=getattr(package, "version", None))
    if hash_source_bytes(raw) != getattr(package, "content_hash", None):
        return EarthCompatibility(False, ["The source has no verified matching digest"],
                                  code="uploaded_model_tampered", **identity)
    # Execute only inside the validator's bounded child process.
    with tempfile.TemporaryDirectory(prefix="astraatmos_earth_3hour_gate_") as temp_dir:
        path = Path(temp_dir) / "earth_model.py"
        path.write_bytes(raw)
        result = (validator or build_validator()).validate_file(path, earth_probe=earth_probe)
    block = result.earth_compatibilities.get(EARTH_3HOURLY_FEED_KEY)
    if not block:
        declared = EARTH_3HOURLY_FEED_KEY in result.datasets
        unknown = not result.datasets and not result.errors
        validation_timeout = validation_report_timed_out(result.report_dict())
        unknown = unknown or validation_timeout or any("without a result" in e for e in result.errors)
        if validation_timeout:
            code = VALIDATION_TIMEOUT_CODE
        elif unknown:
            code = "uploaded_model_compatibility_unknown"
        else:
            code = "uploaded_model_contract_invalid" if declared or result.errors else "uploaded_model_not_earth_3hourly_compatible"
        return EarthCompatibility(
            False, list(result.errors) or ["No verified three-hour declaration and dry-run result"],
            status="unknown" if unknown else "unavailable", code=code,
            declares_earth=declared, datasets=result.datasets, **identity,
        )
    declaration = result.datasets.get(EARTH_3HOURLY_FEED_KEY) or {}
    try:
        execution = contract_profile(declaration.get('schema'))
    except ValueError:
        execution = None
    declared_horizons = declaration.get('horizon', [24])
    default_horizon = (execution or {}).get('probe_horizon', 24)
    probe_horizon = (earth_probe or {}).get('horizon', default_horizon if default_horizon in declared_horizons else declared_horizons[0])
    proven = (result.ok is True and block.get("compatible") is True and block.get("status") == "available"
              and block.get("dataset_id") == EARTH_3HOURLY_FEED_KEY
              and execution is not None and block.get("contract_schema") == execution['schema']
              and block.get("eval_batch_policy") == EVAL_BATCH_POLICY
              and block.get("output_shape") == [2, probe_horizon, 1, *(execution or {}).get('shape', ())])
    return EarthCompatibility(
        proven, list(block.get("errors") or []), warnings=list(result.warnings),
        declares_earth=EARTH_3HOURLY_FEED_KEY in result.datasets, datasets=result.datasets,
        output_shape=block.get("output_shape"), param_schema=dict(result.param_schema),
        status="available" if proven else ("unknown" if block.get("status") == "unknown" else "unavailable"),
        code=None if proven else block.get("code") or "uploaded_model_compatibility_unknown", **identity,
    )


__all__ = [
    "EarthCompatibility",
    "build_validator",
    "evaluate_earth_compatibility",
    "evaluate_package_earth_compatibility",
]
