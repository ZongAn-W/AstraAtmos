"""Build the Earth forecaster for either model source.

Earth supports the official DLinear and uploaded single-file models. Uploaded
three-hour models use a versioned BTCHW interface: v1 spatial tiles or v2 complete
global maps. The hash-verified source declaration chooses the build dimensions.

The uploaded branch:

* executes the **verified** source only (hash-checked, AST-checked the same way as
  at upload) and never a bare pickle;
* refuses Mars-only auxiliary inputs; three-hour models separately declare
  UTC-derived ls and verified ETOPO orography through derived_inputs;
* records the exact build configuration so a reload rebuilds the same module.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from services.uploaded_model_source import (
    UploadedModelSourceError,
    resolve_source_text,
    hash_source_bytes,
)
from services.dataset_identity import EARTH_THREE_HOURLY_DATASET_IDS
from services.earth_training_contract import earth_training_profile
from training_backbones.earth_daily_model import (
    earth_model_config,
    create_earth_forecaster,
    require_earth_channel_order,
    state_dict_to_cpu,
)
from training_backbones.uploaded_model_contract import (
    attach_uploaded_model_contract,
    normalize_auxiliary_inputs,
    run_uploaded_model,
    uploaded_model_requires_ls,
    uploaded_model_requires_topography,
)
from training_backbones.uploaded_model_source_check import assert_source_is_safe

MODEL_SOURCE_OFFICIAL = "official"
MODEL_SOURCE_UPLOADED = "uploaded"
EARTH_MODEL_SOURCES = (MODEL_SOURCE_OFFICIAL, MODEL_SOURCE_UPLOADED)


class EarthModelBuildError(ValueError):
    """The Earth model could not be built for the requested source."""

    def __init__(self, message: str, *, code: str = "invalid_earth_training_artifact"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ModelSourcePlan:
    """What the platform decided to train, in a form both runner and checkpoint keep."""

    model_source: str
    input_channel_order: list[str]
    linear_hidden_layers: int
    uploaded_model: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    dataset_id: str | None = None
    window: int | None = None
    horizon: int | None = None


def _load_uploaded_module(source_text: str, filename: str):
    """Import uploaded source text as a throwaway module.

    The text is the hash-verified copy, and the shared safety gate is re-applied
    here so a file that slipped past the hash check still cannot import anything
    outside torch/numpy or call an unsafe builtin.
    """
    try:
        tree = assert_source_is_safe(source_text, filename)
    except ValueError as exc:
        raise EarthModelBuildError(str(exc), code="uploaded_model_source_invalid") from exc
    del tree

    module_name = f"_astraatmos_earth_uploaded_{uuid.uuid4().hex}"
    loader = importlib.machinery.SourceFileLoader(module_name, filename)
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None or spec.loader is None:
        raise EarthModelBuildError(
            "The uploaded model source could not be loaded",
            code="uploaded_model_source_invalid",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        # exec of hash-verified, AST-checked user source: the documented upload
        # protocol executes build_model(config), not a serialized object.
        exec(compile(source_text, filename, "exec"), module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    return module


def uploaded_model_config(
    *,
    input_channel_order: Sequence[str],
    window: int,
    horizon: int,
    height: int,
    width: int,
    param_schema: Any,
    custom_model_params: Any,
    declared_parameters: Any = None,
) -> dict[str, Any]:
    """The build config handed to ``build_model`` for the Earth feed.

    ``in_channels`` is the real number of channels the platform will feed, and
    ``selected_channels`` starts with the mandatory ``TO3`` target so a model never
    has to guess which channel is the prediction target.

    Parameter defaults come from the model's own ``MODEL_SPEC.parameters`` merged
    with the schema stored at upload time. The model's declaration is the fallback
    when a stored schema is incomplete, so a model that reads
    ``config["hidden_dim"]`` still builds instead of failing with a KeyError.
    """
    order = require_earth_channel_order(input_channel_order)
    schema: dict[str, Any] = {}
    if isinstance(declared_parameters, dict):
        schema.update({key: value for key, value in declared_parameters.items() if isinstance(value, dict)})
    if isinstance(param_schema, dict):
        # The stored schema is authoritative where both exist.
        schema.update({key: value for key, value in param_schema.items() if isinstance(value, dict)})
    params = custom_model_params if isinstance(custom_model_params, dict) else {}

    config: dict[str, Any] = {
        "in_channels": len(order),
        "window": int(window),
        "horizon": int(horizon),
        "height": int(height),
        "width": int(width),
        "selected_channels": list(order),
        "target_channel": order[0],
    }
    for name, field in schema.items():
        config[name] = params.get(name, field.get("default"))
    # A value the user supplied for a parameter the schema does not describe is
    # still passed through rather than silently dropped.
    for name, value in params.items():
        config.setdefault(name, value)
    return config


def build_uploaded_earth_model(
    *,
    reference: dict[str, Any],
    input_channel_order: Sequence[str],
    window: int,
    horizon: int,
    height: int,
    width: int,
    dataset_id: str | None = None,
    contract_schema: str | None = None,
    derived_inputs: Sequence[str] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], tuple[str, ...]]:
    """Build the uploaded model for the Earth feed.

    Returns ``(model, build_config, warnings)``. Raises for a tampered source, a
    missing source without an embedded copy, or a Mars-only auxiliary input.
    ``derived_inputs`` is the declaration frozen in a checkpoint; when given, the
    rebuilt source must declare exactly the same tensors.
    """
    order = require_earth_channel_order(input_channel_order)
    if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS and [height, width] != earth_training_profile(dataset_id)["grid_shape"]:
        raise EarthModelBuildError("Three-hour uploaded model requires the dataset's exact global grid",
                                   code="uploaded_model_contract_invalid")
    try:
        source_text, source_report = resolve_source_text(reference)
    except UploadedModelSourceError as exc:
        raise EarthModelBuildError(str(exc), code=exc.code) from exc
    if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
        embedded = reference.get("source_text")
        if (not reference.get("content_hash")
                or hash_source_bytes(source_text.encode("utf-8")) != reference["content_hash"]
                or (isinstance(embedded, str) and hash_source_bytes(embedded.encode("utf-8")) != reference["content_hash"])):
            raise EarthModelBuildError("Three-hour uploaded source digest is invalid", code="uploaded_model_tampered")
    filename = str(reference.get("display_name") or "uploaded_model") + ".py"

    module = _load_uploaded_module(source_text, filename)
    model_spec = getattr(module, "MODEL_SPEC", None)
    build_model = getattr(module, "build_model", None)
    if not isinstance(model_spec, dict):
        raise EarthModelBuildError(
            "The uploaded model no longer exports MODEL_SPEC as a dict",
            code="uploaded_model_source_invalid",
        )
    if not callable(build_model):
        raise EarthModelBuildError(
            "The uploaded model no longer exports build_model(config)",
            code="uploaded_model_source_invalid",
        )

    if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
        from training_backbones.uploaded_model_dataset_spec import (
            EARTH_3HOURLY_FEED_KEY,
            earth_3hourly_feed_from_spec,
        )
        try:
            feed = earth_3hourly_feed_from_spec(model_spec, dataset_id=dataset_id)
        except Exception as exc:
            raise EarthModelBuildError(
                f"The uploaded model has an invalid Earth three-hourly spec: {exc}",
                code="uploaded_model_contract_invalid",
            ) from exc
        if feed is None:
            raise EarthModelBuildError(
                f"The uploaded model must declare MODEL_SPEC.datasets.{EARTH_3HOURLY_FEED_KEY}",
                code="uploaded_model_not_earth_3hourly_compatible",
            )
        if window not in feed['window'] or horizon not in feed['horizon']:
            raise EarthModelBuildError(
                f"Uploaded model does not declare window={window}, horizon={horizon}",
                code="uploaded_model_contract_invalid",
            )
        from services.user_model_validator import UserModelValidator
        from training_backbones.earth_3hourly_uploaded_contract import (
            build_config, contract_schema_from_spec, normalize_derived_inputs,
            prepare_model, RESERVED_PARAMETERS,
        )
        try:
            source_contract_schema = contract_schema_from_spec(model_spec)
        except ValueError as exc:
            raise EarthModelBuildError(str(exc), code="uploaded_model_contract_invalid") from exc
        from training_backbones.earth_3hourly_uploaded_contract import contract_profile
        if contract_profile(source_contract_schema)["dataset_id"] != dataset_id:
            raise EarthModelBuildError("Uploaded spatial schema belongs to another Earth dataset", code="uploaded_model_contract_invalid")
        if any(expected is not None and source_contract_schema != expected
               for expected in (contract_schema, reference.get("contract_schema"))):
            raise EarthModelBuildError("Stored Earth contract schema disagrees with verified source",
                                       code="uploaded_model_contract_invalid")
        schema, errors = UserModelValidator._normalize_parameters(model_spec.get("parameters", {}))
        if errors or any(k in RESERVED_PARAMETERS or k.startswith("_") for k in schema):
            raise EarthModelBuildError("Invalid three-hour uploaded parameters",
                                       code="uploaded_model_contract_invalid")
        if model_spec.get("auxiliary_inputs"):
            # Earth keeps meteorological channel choices separate from the
            # server-provided seasonal coordinate and ETOPO resource.
            raise EarthModelBuildError(
                "The Mars auxiliary_inputs block is not available for Earth data; "
                "declare MODEL_SPEC.derived_inputs for ls and/or ETOPO topography",
                code="uploaded_model_contract_invalid",
            )
        try:
            source_derived_inputs = normalize_derived_inputs(model_spec)
        except ValueError as exc:
            raise EarthModelBuildError(str(exc), code="uploaded_model_contract_invalid") from exc
        if reference.get("param_schema") != schema:
            raise EarthModelBuildError("Stored parameter schema disagrees with verified source",
                                       code="uploaded_model_contract_invalid")
        declared_names = sorted(source_derived_inputs)
        # The frozen declaration comes from the checkpoint. A rebuilt model must
        # receive exactly the tensors it was trained with; changing the block
        # would silently shift the model's inputs.
        if derived_inputs is not None and list(derived_inputs) != declared_names:
            raise EarthModelBuildError("Stored derived inputs disagree with verified source",
                                       code="uploaded_model_contract_invalid")
        params, errors = UserModelValidator.normalize_custom_params(schema, reference.get("custom_model_params"))
        if errors:
            raise EarthModelBuildError("; ".join(errors), code="invalid_earth_training_parameters")
        config = build_config(order, params, window=window, horizon=horizon,
                              contract_schema=source_contract_schema)
        if reference.get("build_config") and reference["build_config"] != config:
            raise EarthModelBuildError("Stored three-hour build config disagrees with verified source",
                                       code="uploaded_model_contract_invalid")
        try:
            model = prepare_model(build_model(config), window=window,
                                  contract_schema=source_contract_schema,
                                  derived_inputs=source_derived_inputs)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise EarthModelBuildError(str(exc), code="uploaded_model_contract_invalid") from exc
        return model, config, ()

    try:
        auxiliary_inputs = normalize_auxiliary_inputs(model_spec)
    except ValueError as exc:
        raise EarthModelBuildError(str(exc), code="uploaded_model_source_invalid") from exc

    config = uploaded_model_config(
        input_channel_order=order,
        window=window,
        horizon=horizon,
        height=height,
        width=width,
        param_schema=reference.get("param_schema"),
        custom_model_params=reference.get("custom_model_params"),
        declared_parameters=model_spec.get("parameters"),
    )
    try:
        model = build_model(config)
    except Exception as exc:  # noqa: BLE001 - report as a build failure
        raise EarthModelBuildError(
            f"The uploaded model could not be built for Earth data: {exc}"
        ) from exc
    if not isinstance(model, torch.nn.Module):
        raise EarthModelBuildError("build_model(config) must return torch.nn.Module")

    attach_uploaded_model_contract(model, model_spec)
    if uploaded_model_requires_ls(model):
        raise EarthModelBuildError(
            "The uploaded model requires Ls, which is a Mars-only input and is not "
            "available for Earth data"
        )
    if uploaded_model_requires_topography(model):
        raise EarthModelBuildError(
            "The uploaded model requires MOLA topography, which is a Mars-only input "
            "and is not available for Earth data"
        )

    warnings: list[str] = []
    if source_report.get("used_embedded_source"):
        if source_report.get("status") == "tampered":
            warnings.append(
                "the uploaded model file no longer matches the trained version "
                f"({source_report.get('detail')}); the model was rebuilt from the "
                "verified copy stored inside the checkpoint"
            )
        else:
            warnings.append(
                "the original uploaded model file is unavailable; the model was "
                "rebuilt from the verified copy stored inside the checkpoint"
            )
    return model, config, tuple(warnings)


def prepare_official_three_hour_model(model: torch.nn.Module, model_config: Mapping[str, Any]) -> torch.nn.Module:
    """Give a three-hourly official model the shared Earth contract attributes.

    The official DLinear is then invoked through exactly the same
    ``earth_3hourly_uploaded_contract.forward`` (and therefore the same shape,
    dtype, finiteness and output validation) as an uploaded model. The spatial
    profile is unchanged: it stays the v1 24x48 tile the caller passes in.
    """
    if (model_config.get("height"), model_config.get("width")) == (36, 72):
        return model

    from training_backbones.earth_3hourly_uploaded_contract import CONTRACT_SCHEMA, prepare_model

    try:
        window = int(model_config["window"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EarthModelBuildError("Earth model config has no usable window",
                                   code="invalid_earth_model_config") from exc
    return prepare_model(model, window=window, contract_schema=CONTRACT_SCHEMA)


def build_earth_model_for_plan(plan: ModelSourcePlan) -> tuple[torch.nn.Module, dict[str, Any], tuple[str, ...]]:
    """Build the Earth model described by a plan (official or uploaded)."""
    profile = earth_training_profile(plan.dataset_id, {'window': plan.window, 'horizon': plan.horizon})
    height, width = profile["grid_shape"]
    if plan.model_source == MODEL_SOURCE_OFFICIAL:
        model = create_earth_forecaster(
            plan.input_channel_order, plan.linear_hidden_layers,
            window=profile["window"], horizon=profile["horizon"],
            height=height, width=width,
        )
        config = earth_model_config(
            plan.input_channel_order, plan.linear_hidden_layers,
            window=profile["window"], horizon=profile["horizon"],
            height=height, width=width,
        )
        if plan.dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            config["implementation_id"] = profile["implementation_id"]
            model = prepare_official_three_hour_model(model, config)
        return model, config, ()
    if plan.model_source == MODEL_SOURCE_UPLOADED:
        if not isinstance(plan.uploaded_model, dict):
            raise EarthModelBuildError(
                "The uploaded model plan has no model reference",
                code="uploaded_model_reference_missing",
            )
        return build_uploaded_earth_model(
            reference=plan.uploaded_model,
            input_channel_order=plan.input_channel_order,
            window=profile["window"],
            horizon=profile["horizon"],
            height=height,
            width=width,
            dataset_id=plan.dataset_id,
        )
    raise EarthModelBuildError(f"Unsupported Earth model source: {plan.model_source}")


def earth_forward_for_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    model_source: str,
    horizon: int = 3,
    height: int = 36,
    width: int = 72,
    longitude: Any = None,
    topography: Any = None,
) -> torch.Tensor:
    """Run either model type and normalize the output contract.

    Uploaded models are called through their declared contract runner, which
    enforces the spatial shape; the official DLinear keeps its zero time
    placeholder (see :func:`training_backbones.earth_daily_model.earth_forward`).
    ``longitude`` is the seasonal coordinate. It is only ever consumed by a
    three-hour uploaded model that declared it in ``MODEL_SPEC.derived_inputs``;
    every other model is called exactly as before.
    """
    if model_source == MODEL_SOURCE_OFFICIAL:
        if getattr(model, "_astraatmos_earth_3hourly_contract", None):
            # A three-hourly official model now runs the same contract as an
            # uploaded one. Only the retired daily profile keeps its own
            # invocation, because its grid and window policy differ.
            from training_backbones.earth_3hourly_uploaded_contract import forward

            try:
                return forward(model, inputs, horizon=horizon, ls=longitude, topography=topography)
            except ValueError as exc:
                raise EarthModelBuildError(str(exc), code="invalid_earth_model_contract") from exc
        from training_backbones.earth_daily_model import earth_forward

        return earth_forward(model, inputs, horizon=horizon, height=height, width=width)
    if model_source != MODEL_SOURCE_UPLOADED:
        raise EarthModelBuildError(f"Unsupported Earth model source: {model_source}")
    if getattr(model, "_astraatmos_earth_3hourly_contract", None):
        from training_backbones.earth_3hourly_uploaded_contract import forward
        try:
            return forward(model, inputs, horizon=horizon, ls=longitude, topography=topography)
        except ValueError as exc:
            raise EarthModelBuildError(str(exc), code="uploaded_model_contract_invalid") from exc
    if inputs.dim() != 5:
        raise EarthModelBuildError("Earth model input must be [B, window, C, H, W]")
    batch, _, _, input_height, input_width = inputs.shape
    if int(input_height) != int(height) or int(input_width) != int(width):
        raise EarthModelBuildError(
            f"Earth model expects a {int(height)}x{int(width)} grid, got "
            f"{int(input_height)}x{int(input_width)}"
        )
    output = run_uploaded_model(model, inputs, context="Earth uploaded model")
    expected = (batch, int(horizon), 1, int(height), int(width))
    if tuple(output.shape) != expected:
        raise EarthModelBuildError(
            f"Unexpected Earth output shape: {tuple(output.shape)}"
        )
    if not torch.isfinite(output).all():
        raise EarthModelBuildError("Non-finite Earth model output")
    return output


__all__ = [
    "EARTH_MODEL_SOURCES",
    "EarthModelBuildError",
    "MODEL_SOURCE_OFFICIAL",
    "MODEL_SOURCE_UPLOADED",
    "ModelSourcePlan",
    "build_earth_model_for_plan",
    "build_uploaded_earth_model",
    "earth_forward_for_model",
    "state_dict_to_cpu",
    "uploaded_model_config",
]
