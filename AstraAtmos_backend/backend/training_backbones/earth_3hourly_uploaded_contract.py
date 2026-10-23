"""Independent, versioned contract for server-fed Earth three-hour models."""

from itertools import combinations

from training_backbones.uploaded_model_contract import (
    LS_AUXILIARY_INPUT,
    TOPOGRAPHY_AUXILIARY_INPUT,
    normalize_auxiliary_inputs,
    validate_ls_tensor,
    validate_topography_tensor,
)

DATASET_ID = "earth_merra2_3hourly_v1"
CONTRACT_SCHEMA = "astraatmos_earth_3hourly_uploaded_model_v1"
IMPLEMENTATION_ID = "astraatmos_earth_3hourly_uploaded_runner_v1"
TILE_SHAPE = (24, 48)
FULL_GRID_CONTRACT_SCHEMA = "astraatmos_earth_3hourly_fullgrid_model_v2"
FULL_GRID_IMPLEMENTATION_ID = "astraatmos_earth_3hourly_fullgrid_runner_v2"
FULL_GRID_SHAPE = (240, 480)
LOWRES_DATASET_ID = "earth_merra2_3hourly_5deg_v1"
LOWRES_CONTRACT_SCHEMA = "astraatmos_earth_3hourly_5deg_model_v1"
SUPPORTED_CONTRACT_SCHEMAS = frozenset({CONTRACT_SCHEMA, FULL_GRID_CONTRACT_SCHEMA, LOWRES_CONTRACT_SCHEMA})
CHANNELS = ("TO3", "U10M", "V10M", "T2M", "SWGDN")
UNITS = ("DU", "m s-1", "m s-1", "K", "W m-2")
FEED = {
    "schema": CONTRACT_SCHEMA,
    "frequency_hours": 3, "step_unit": "hour", "step": 3, "time_zone": "UTC",
    "window": [56], "horizon": [24], "grid": [[240, 480]],
    "target": "TO3", "target_unit": "DU",
    "input_channels": list(CHANNELS), "input_units": list(UNITS),
    "dtype": "float32", "tensor_layout": "BTCHW",
    "auxiliary_inputs": list(CHANNELS[1:]), "spatial_tile_shape": list(TILE_SHAPE),
    "output_channels": ["TO3"], "output_units": ["DU"],
}
FULL_GRID_FEED = {**FEED, "schema": FULL_GRID_CONTRACT_SCHEMA,
                  "spatial_tile_shape": list(FULL_GRID_SHAPE)}
LOWRES_FEED = {**FEED, "schema": LOWRES_CONTRACT_SCHEMA,
               "grid": [[36, 72]], "spatial_tile_shape": [36, 72]}
RESERVED_PARAMETERS = frozenset({
    *FEED, "dataset_id", "dataset_version", "dataset_fingerprint", "dataset_snapshot",
    "fingerprint", "snapshot", "version", "data_dir", "data_path", "dataset_dir", "dataset_path",
    "data_directories", "data_binding", "dataset_binding", "dataset_identity_status",
    "training_dataset", "planet", "data_source",
    "in_channels", "height", "width", "selected_channels", "target_channel",
    "global_grid_shape", "device", "normalization", "contract_schema", "implementation_id",
    "derived_inputs", "topography", "topography_path", "topography_identity", "season_algorithm",
})
EVAL_BATCH_POLICY = "earth_eval_sample_independent_v1"
# Runtime checkpoint reads use a separate, bounded probe budget.  The training
# batch remains part of the frozen contract, but is never used to size these
# safety probes (a full-grid training batch of 64 would allocate several GiB).
RUNTIME_PROBE_VERSION = "earth_eval_runtime_probe_v2"
RUNTIME_PROBE_BATCH_SIZE = 2
RUNTIME_PROBE_MAX_INPUT_ELEMENTS = 70_000_000

#: ``MODEL_SPEC`` key for tensors the server *derives* for the model. It is
#: deliberately not named ``auxiliary_inputs``: in this feed that name already
#: means the four optional observation channels (``U10M``/``V10M``/``T2M``/
#: ``SWGDN``), and the Mars contract uses it for its own non-channel tensors.
#: Reusing it would make one key mean two different things.
DERIVED_INPUT_KEY = "derived_inputs"
#: UTC-derived solar longitude and independently verified ETOPO orography.
#: The declaration shares tensor fields with Mars, never its resource loader.
DERIVED_INPUT_SPECS = {"ls": LS_AUXILIARY_INPUT, "topography": TOPOGRAPHY_AUXILIARY_INPUT}
#: Attribute stamped on a prepared model; distinct from the Mars contract's
#: ``_astraatmos_auxiliary_inputs`` because the two feeds mean different things.
DERIVED_INPUT_ATTRIBUTE = "_astraatmos_earth_derived_inputs"


def contract_profile(schema=CONTRACT_SCHEMA):
    if schema not in SUPPORTED_CONTRACT_SCHEMAS:
        raise ValueError("Unsupported Earth three-hour uploaded model schema")
    lowres = schema == LOWRES_CONTRACT_SCHEMA
    full_grid = schema in (FULL_GRID_CONTRACT_SCHEMA, LOWRES_CONTRACT_SCHEMA)
    return {"schema": schema, "shape": (36, 72) if lowres else FULL_GRID_SHAPE if full_grid else TILE_SHAPE,
            "dataset_id": LOWRES_DATASET_ID if lowres else DATASET_ID,
            "implementation_id": "astraatmos_earth_3hourly_5deg_runner_v1" if lowres else FULL_GRID_IMPLEMENTATION_ID if full_grid else IMPLEMENTATION_ID,
            "full_grid": full_grid, "default_batch_size": 1 if full_grid else 8,
            "max_batch_size": 64,
            # Admission exercises every channel subset and repeated eval/train
            # calls. Prefer short declared windows for both spatial contracts;
            # actual task windows remain frozen and checked during execution.
            "probe_window": 2, "probe_horizon": 1}


def contract_schema_from_spec(model_spec):
    declarations = model_spec.get("datasets") or {}
    feeds = [declarations[key] for key in (DATASET_ID, LOWRES_DATASET_ID) if key in declarations]
    if len(feeds) != 1:
        raise ValueError("An Earth model must declare exactly one three-hour spatial feed")
    feed = feeds[0]
    schema = feed.get("schema")
    contract_profile(schema)
    return schema


def contract_schema_from_source(source):
    """Read only a literal schema declaration; never execute uploaded code here."""
    import ast

    def field(node, key):
        if not isinstance(node, ast.Dict):
            raise ValueError("Earth MODEL_SPEC must declare its dataset schema literally")
        matches = [value for name, value in zip(node.keys, node.values)
                   if isinstance(name, ast.Constant) and name.value == key]
        if len(matches) != 1:
            raise ValueError(f"Earth MODEL_SPEC must declare one {key!r} field")
        return matches[0]

    assignments = [node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and any(isinstance(name, ast.Name) and name.id == "MODEL_SPEC" for name in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Earth source must export one MODEL_SPEC assignment")
    datasets = field(assignments[0], "datasets")
    keys = [name.value for name in datasets.keys if isinstance(name, ast.Constant)]
    dataset_ids = [key for key in (DATASET_ID, LOWRES_DATASET_ID) if key in keys]
    if len(dataset_ids) != 1:
        raise ValueError("An Earth model must declare exactly one three-hour spatial feed")
    schema = ast.literal_eval(field(field(datasets, dataset_ids[0]), "schema"))
    contract_profile(schema)
    return schema


def contract_schema_for_reference(reference):
    """Use the isolated validator's frozen schema, with v1 legacy fallback."""
    schema = reference.get("contract_schema") or CONTRACT_SCHEMA
    contract_profile(schema)
    try:
        declared = contract_schema_from_source(reference.get("source_text"))
    except (ValueError, TypeError, SyntaxError):
        # Valid MODEL_SPEC declarations may use constants or compose dictionaries.
        # The hash-verified model build checks the executed declaration again.
        declared = None
    if declared is not None and declared != schema:
        raise ValueError("Frozen model schema disagrees with its source")
    return schema


def probe_longitude(batch, window, *, device=None):
    """Return a deterministic, contract-valid seasonal coordinate for probes.

    Admission never reads data, so the probe needs a plausible value rather than
    the real one: it must only be finite, ``float32`` and shaped ``[batch,
    window]``. It is built from the published three-hour axis so the numbers are
    in the same range a real run would supply.
    """
    import numpy as np
    import torch

    from services.earth_season import solar_longitude

    axis = np.datetime64("2020-01-01T01:30", "s") + np.arange(int(window)) * np.timedelta64(3, "h")
    values = np.repeat(solar_longitude(axis)[None, :], int(batch), axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype="float32"))
    return tensor if device is None else tensor.to(device)


def prepare_model(model, *, window, contract_schema=CONTRACT_SCHEMA, derived_inputs=None):
    """Use the same floating-point/device setup for admission and reconstruction."""
    import torch

    if not isinstance(model, torch.nn.Module):
        raise ValueError("build_model(config) must return torch.nn.Module")
    model.to(device="cpu", dtype=torch.float32)
    profile = contract_profile(contract_schema)
    model._astraatmos_earth_3hourly_contract = contract_schema
    model._astraatmos_earth_spatial_shape = profile["shape"]
    model._astraatmos_earth_window = window
    # An absent block means the model was built before derived inputs existed and
    # must keep being called with inputs only.
    model._astraatmos_earth_derived_inputs = normalize_derived_inputs(
        {DERIVED_INPUT_KEY: derived_inputs} if derived_inputs is not None else {})
    validate_forward_signature(model)
    return model


def validate_forward_signature(model):
    """A declaration has one explicit positional argument per derived tensor."""
    import inspect

    names = derived_input_names(model)
    try:
        signature = inspect.signature(model.forward)
        signature.bind(*([object()] * (1 + len(names))))
    except (TypeError, ValueError) as exc:
        raise ValueError("Earth forward signature disagrees with derived_inputs") from exc
    if names:
        parameters = list(signature.parameters.values())
        positional = [parameter for parameter in parameters
                      if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
        if (len(parameters) != 1 + len(names) or len(positional) != 1 + len(names)
                or [parameter.name for parameter in positional[1:]] != names
                or any(parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
                       for parameter in parameters)):
            raise ValueError("Earth forward must explicitly accept x followed by " + ", ".join(names))


def build_config(order, params=None, *, window=56, horizon=24, contract_schema=CONTRACT_SCHEMA):
    from services.earth_training_contract import earth_training_profile
    execution = contract_profile(contract_schema)
    profile = earth_training_profile(execution['dataset_id'], {'window': window, 'horizon': horizon})
    order = list(order)
    if not order or order[0] != "TO3" or order != [c for c in CHANNELS if c in order]:
        raise ValueError("Earth three-hour channels must be unique and canonical with TO3 first")
    if len(order) != len(set(order)) or any(c not in CHANNELS for c in order):
        raise ValueError("Unsupported Earth three-hour channel order")
    params = dict(params or {})
    execution = contract_profile(contract_schema)
    if any(key in RESERVED_PARAMETERS or key.startswith("_") for key in params):
        raise ValueError("Custom parameters cannot override the server Earth contract")
    return {
        **params, "dataset_id": execution['dataset_id'], "contract_schema": contract_schema,
        "in_channels": len(order), "selected_channels": order, "target_channel": "TO3",
        "window": profile['window'], "horizon": profile['horizon'],
        "height": execution["shape"][0], "width": execution["shape"][1],
        "global_grid_shape": list(profile['grid_shape']), "spatial_tile_shape": list(execution["shape"]),
    }


def channel_orders():
    return [["TO3", *subset] for count in range(5) for subset in combinations(CHANNELS[1:], count)]


def normalize_derived_inputs(model_spec):
    """Validate ``MODEL_SPEC.derived_inputs`` and return the declared tensors.

    Tensor field specifications are shared with Mars, while Earth resource and
    producer identities remain separate. A missing block means "no derived
    input", preserving the call used by models trained before this contract.
    """
    if not isinstance(model_spec, dict):
        raise ValueError("MODEL_SPEC must be a dict")
    if DERIVED_INPUT_KEY not in model_spec:
        return {}
    declared = model_spec[DERIVED_INPUT_KEY]
    try:
        normalized = normalize_auxiliary_inputs({"auxiliary_inputs": declared})
    except ValueError as exc:
        raise ValueError(f"MODEL_SPEC.{DERIVED_INPUT_KEY}: {exc}") from exc
    # Python list equality considers True == 1; declarations must preserve the
    # literal integer channel dimension rather than accepting boolean shapes.
    for name, specification in normalized.items():
        shape = declared[name]["shape"]
        if any(type(actual) is not type(expected)
               for actual, expected in zip(shape, specification["shape"])):
            raise ValueError(f"MODEL_SPEC.{DERIVED_INPUT_KEY}.{name}.shape has invalid dimension types")
    unsupported = sorted(set(normalized) - set(DERIVED_INPUT_SPECS))
    if unsupported:
        raise ValueError(
            f"MODEL_SPEC.{DERIVED_INPUT_KEY} supports only {sorted(DERIVED_INPUT_SPECS)} "
            f"for the Earth feed, got {unsupported}"
        )
    return normalized


def derived_input_names(model):
    """Return tensor arguments in the fixed ls, topography call order."""
    declared = getattr(model, DERIVED_INPUT_ATTRIBUTE, {}) or {}
    return [name for name in DERIVED_INPUT_SPECS if name in declared]


def requires_longitude(model):
    return "ls" in (getattr(model, DERIVED_INPUT_ATTRIBUTE, {}) or {})


def requires_topography(model):
    return "topography" in (getattr(model, DERIVED_INPUT_ATTRIBUTE, {}) or {})


def forward(model, inputs, *, horizon=24, ls=None, topography=None):
    import torch

    shape = contract_profile(getattr(model, '_astraatmos_earth_3hourly_contract', CONTRACT_SCHEMA))["shape"]
    if (not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.float32
            or inputs.ndim != 5 or inputs.shape[0] < 1
            or inputs.shape[1] != getattr(model, '_astraatmos_earth_window', inputs.shape[1])
            or not 1 <= inputs.shape[1] <= 240 or not 1 <= inputs.shape[2] <= 5
            or tuple(inputs.shape[-2:]) != shape or not torch.isfinite(inputs).all()):
        raise ValueError(f"Earth three-hour input must be finite float32 [B,window,C,{shape[0]},{shape[1]}]")
    arguments = [inputs]
    if requires_longitude(model):
        if isinstance(ls, torch.Tensor) and (ls.dtype != torch.float32 or ls.device != inputs.device):
            raise ValueError("Earth ls must be float32 on the input device")
        value = validate_ls_tensor(inputs, ls, "Earth three-hour model")
        arguments.append(value.clone())
    if requires_topography(model):
        if isinstance(topography, torch.Tensor) and topography.device != inputs.device:
            raise ValueError("Earth topography must be on the input device")
        value = validate_topography_tensor(inputs, topography, "Earth three-hour model")
        # Protect caller-owned static assets and expanded batch views from an
        # uploaded model's in-place operations, in training as well as eval.
        arguments.append(value.clone())
    output = model(*arguments)
    # The runner may use CUDA autocast while keeping the public Earth contract
    # float32. Casting only at the contract boundary preserves strict external
    # dtype semantics and keeps the model's internal Tensor Core path enabled.
    if (inputs.device.type == "cuda" and torch.is_autocast_enabled()
            and isinstance(output, torch.Tensor)
            and output.dtype in (torch.float16, torch.bfloat16)):
        output = output.float()
    expected = (inputs.shape[0], horizon, 1, *shape)
    if (not isinstance(output, torch.Tensor) or tuple(output.shape) != expected
            or output.dtype != torch.float32 or output.device != inputs.device
            or not torch.isfinite(output).all()):
        raise ValueError(f"Earth three-hour output must be finite float32 {expected} on the input device")
    return output


def validate_eval_batch_independence(model, *, window, channels, horizon, batch_size=8,
                                     probe_batch_size=None, max_input_elements=None,
                                     topography=None):
    """Reject eval outputs that depend on batch neighbours, order or call state."""
    import torch

    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("Earth eval batch size must be an integer between 1 and 64")
    generator = torch.Generator().manual_seed(1107)
    execution = contract_profile(getattr(model, '_astraatmos_earth_3hourly_contract', CONTRACT_SCHEMA))
    if batch_size > execution["max_batch_size"]:
        raise ValueError(f"Full-grid Earth batch size cannot exceed {execution['max_batch_size']}")
    modes = [(module, module.training) for module in model.modules()]
    registries = [(module, group, dict(getattr(module, group)))
                  for module in model.modules() for group in ("_parameters", "_buffers")]
    state = [(module, group, name, value, value.detach().clone())
             for module, group, entries in registries for name, value in entries.items() if value is not None]

    terrain = _probe_topography_asset(model, topography)

    def probe(inputs, probe_terrain=None):
        probe_ls = probe_longitude(inputs.shape[0], inputs.shape[1], device=inputs.device) if requires_longitude(model) else None
        output = forward(model, inputs, horizon=horizon, ls=probe_ls, topography=probe_terrain).clone()
        if (any(getattr(module, group).keys() != entries.keys()
                or any(getattr(module, group)[name] is not value for name, value in entries.items())
                for module, group, entries in registries)
                or any(not torch.equal(value, saved) for _, _, _, value, saved in state)):
            raise ValueError("Earth eval must not mutate model parameters or buffers")
        return output

    model.eval()
    try:
        with torch.no_grad():
            # ``probe_batch_size`` is deliberately independent of the training
            # batch.  Legacy callers omit it and retain the historical probe;
            # checkpoint/runtime callers pass the bounded value explicitly.
            effective_batch = batch_size if probe_batch_size is None else probe_batch_size
            if type(effective_batch) is not int or not 1 <= effective_batch <= execution["max_batch_size"]:
                raise ValueError("Earth runtime probe batch size is invalid")
            batches = ({1, 2, effective_batch} if execution["full_grid"]
                       else {2, 3, effective_batch})
            max_elements = max_input_elements
            if max_elements is not None and (type(max_elements) is not int or max_elements < 1):
                raise ValueError("Earth runtime probe input budget is invalid")
            for batch in sorted(batches):
                if max_elements is not None and batch * window * channels * execution["shape"][0] * execution["shape"][1] > max_elements:
                    raise ValueError("Earth runtime probe exceeds its input budget")
                inputs = torch.randn((batch, window, channels, *execution["shape"]), generator=generator, dtype=torch.float32)
                inputs[1:] = inputs[1:] * 0.5 + 2.0
                probe_terrain = _probe_topography_batch(model, terrain, inputs)
                together = probe(inputs, probe_terrain)
                separate = torch.cat([probe(inputs[index:index + 1],
                    None if probe_terrain is None else probe_terrain[index:index + 1]) for index in range(batch)])
                reordered = probe(inputs.flip(0), None if probe_terrain is None else probe_terrain.flip(0)).flip(0)
                changed = inputs.clone()
                changed[1:] = changed[1:] * -2.0 - 3.0
                changed_terrain = None if probe_terrain is None else probe_terrain.clone()
                if changed_terrain is not None:
                    changed_terrain[1:] = changed_terrain[1:].flip(-1)
                changed_output = probe(changed, changed_terrain)
                repeated = probe(inputs, probe_terrain)
                comparisons = ((together, separate), (together, reordered),
                               (together[:1], changed_output[:1]), (together, repeated))
                if any(not torch.allclose(left, right, rtol=1e-5, atol=1e-5) for left, right in comparisons):
                    raise ValueError("Earth eval predictions must be sample-independent across batch size, "
                                     "neighbours, order and repeated calls")
    finally:
        with torch.no_grad():
            for module, group, name, value, saved in state:
                if value.shape == saved.shape and value.dtype == saved.dtype and value.device == saved.device:
                    value.copy_(saved)
                else:
                    value.data = saved
            for module, group, entries in registries:
                getattr(module, group).clear()
                getattr(module, group).update(entries)
        for module, training in modes:
            module.training = training


def dry_run(build_model, order, params, *, window=56, horizon=24, batch_size=8,
            contract_schema=CONTRACT_SCHEMA, derived_inputs=None, topography=None):
    import torch

    execution = contract_profile(contract_schema)
    model = prepare_model(build_model(build_config(order, params, window=window, horizon=horizon,
                                                   contract_schema=contract_schema)),
                          window=window, contract_schema=contract_schema,
                          derived_inputs=derived_inputs)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("Earth training requires trainable model parameters")
    terrain = _probe_topography_asset(model, topography)
    validate_eval_batch_independence(model, window=window, channels=len(order), horizon=horizon,
                                    batch_size=batch_size, topography=terrain)
    generator = torch.Generator().manual_seed(1107)
    batches = tuple(dict.fromkeys((1, batch_size, 2))) if execution["full_grid"] else (1, 2)
    for batch in batches:
        inputs = torch.randn((batch, window, len(order), *execution["shape"]), generator=generator, dtype=torch.float32)
        probe_ls = probe_longitude(batch, window, device=inputs.device) if requires_longitude(model) else None
        probe_terrain = _probe_topography_batch(model, terrain, inputs)
        model.eval()
        with torch.no_grad():
            output = forward(model, inputs, horizon=horizon, ls=probe_ls, topography=probe_terrain)
        model.train()
        model.zero_grad(set_to_none=True)
        loss = forward(model, inputs, horizon=horizon, ls=probe_ls, topography=probe_terrain).square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("Earth model must produce finite training gradients")
    validate_eval_batch_independence(model, window=window, channels=len(order), horizon=horizon,
                                    batch_size=batch_size, topography=terrain)
    return list(output.shape)


def _probe_topography_asset(model, asset):
    if not requires_topography(model):
        return None
    if asset is None:
        from services.earth_topography import load_earth_topography
        dataset_id = contract_profile(getattr(model, '_astraatmos_earth_3hourly_contract', CONTRACT_SCHEMA))['dataset_id']
        asset = load_earth_topography(dataset_id=dataset_id) if dataset_id == LOWRES_DATASET_ID else load_earth_topography()
    import numpy as np
    if np.ptp(asset.elevation) <= 0:
        raise ValueError("Earth topography probes require a nonconstant verified terrain asset")
    return asset


def _probe_topography_batch(model, asset, inputs):
    """Probe distinct v1 locations and reuse the real global v2 asset."""
    if not requires_topography(model):
        return None
    from services.earth_topography import prepare_topography_batch
    execution = contract_profile(getattr(model, '_astraatmos_earth_3hourly_contract', CONTRACT_SCHEMA))
    spatial_slices = None if execution["full_grid"] else [
        (slice((index * 7 % 10) * 24, (index * 7 % 10 + 1) * 24),
         slice((index * 3 % 10) * 48, (index * 3 % 10 + 1) * 48))
        for index in range(inputs.shape[0])]
    return prepare_topography_batch(asset, inputs, spatial_slices=spatial_slices)
