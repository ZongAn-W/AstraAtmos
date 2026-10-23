import json
from typing import Any, Optional

from services.dataset_identity import resolve_dataset_id, require_training_dataset
from training_backbones.model_zoo import normalize_model_architecture, normalize_use_sphere
from services.training_split import normalize_split_ratios


UNIFIED_TRAINING_SCRIPT = "demo3.py"
CHANNEL_ORDER = ["U", "V", "D", "S", "T"]
TRAINING_DATASET_OPENMARS_MCD = "openmars_mcd"
TRAINING_DATASET_MCD_OVERVIEW = "mcd_overview"
TRAINING_DATASET_IDS = {
    TRAINING_DATASET_OPENMARS_MCD,
    TRAINING_DATASET_MCD_OVERVIEW,
}
MIN_POSITIVE_FLOAT = 0.000001
ARCHITECTURE_INTEGER_PARAMS = {
    "spatial_hidden_dim": 64,
    "temporal_hidden_dim": 128,
    "temporal_depth": 2,
    "linear_hidden_layers": 2,
    "d_model": 64,
    "n_heads": 2,
    "e_layers": 1,
    "d_ff": 128,
    "patch_len": 2,
    "stride": 1,
    "moving_avg": 3,
    "down_sampling_window": 2,
    "down_sampling_layers": 1,
    "hidden_dim": 64,
    "seg_len": 1,
    "win_size": 2,
    "factor": 2,
    "num_downsample": 1,
    "cnn_depth": 1,
    "mst_spatial_hidden_dim": 64,
    "mst_temporal_hidden_dim": 128,
    "mst_num_downsample": 1,
    "mst_temporal_depth": 1,
    "phase_context_dim": 8,
    "climatology_hidden_dim": 8,
    "kernel_size": 3,
    "tau": 2,
    "stack_count": 1,
    "blocks_per_stack": 1,
    "top_k_freq": 2,
    "modes": 4,
    "mlp_ratio": 2,
}
ARCHITECTURE_FLOAT_PARAMS = {
    "dropout": 0.1,
    "gamma": 1.0,
    "initial_history_weight": 0.7,
    "initial_translation_weight": 0.7,
}
OPEN_INTERVAL_FLOAT_PARAMS = {"initial_history_weight", "initial_translation_weight"}
ZERO_TO_POINT_NINE_FLOAT_PARAMS = {"dropout", *OPEN_INTERVAL_FLOAT_PARAMS}
ARCHITECTURE_INTEGER_LIST_PARAMS = {
    "patch_size": [1, 2, 2],
    "cuboid_size": [1, 2, 2],
    "pooling_sizes": [1, 2],
    "downsample_factors": [1],
}
THREE_VALUE_INTEGER_LIST_PARAMS = {"patch_size", "cuboid_size"}
ARCHITECTURE_PARAM_KEYS = (
    tuple(ARCHITECTURE_INTEGER_PARAMS)
    + tuple(ARCHITECTURE_FLOAT_PARAMS)
    + tuple(ARCHITECTURE_INTEGER_LIST_PARAMS)
)
RECURRENT_MODEL_ARCHITECTURES = {"predrnnv2", "predrnnpp", "convlstm"}
TRANSFER_FREEZE_MODES = {"none", "backbone", "head"}


def _is_blank(value: Any) -> bool:
    return value is None or value == ""


def _positive_int(value: Any, fallback: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
    if _is_blank(value):
        return fallback
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    if parsed < minimum:
        return minimum
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def _non_negative_int(value: Any, fallback: int, maximum: Optional[int] = None) -> int:
    if _is_blank(value):
        return fallback
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    if parsed < 0:
        return 0
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def _bool_value(value: Any, fallback: bool = False) -> bool:
    if _is_blank(value):
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return fallback


def _safe_string(value: Any, fallback: str = "") -> str:
    if _is_blank(value):
        return fallback
    return str(value).strip()


def _freeze_mode(value: Any) -> str:
    mode = _safe_string(value, "none").lower()
    return mode if mode in TRANSFER_FREEZE_MODES else "none"


def _training_dataset(value: Any) -> str:
    # New training configurations are parsed strictly: unknown or not yet
    # supported datasets are rejected instead of silently falling back.
    dataset_id = resolve_dataset_id(None, {"training_dataset": value})
    return require_training_dataset(dataset_id)


def _positive_float(
    value: Any,
    fallback: float,
    minimum: float = MIN_POSITIVE_FLOAT,
    maximum: Optional[float] = None,
) -> float:
    if _is_blank(value):
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if parsed < minimum:
        return minimum
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def _positive_int_list(value: Any, fallback: list[int]) -> list[int]:
    if _is_blank(value):
        return list(fallback)
    if isinstance(value, str):
        items = value.strip().strip("[]").replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        return list(fallback)

    parsed: list[int] = []
    for item in items:
        if _is_blank(item):
            continue
        try:
            next_value = int(round(float(item)))
        except (TypeError, ValueError):
            continue
        if next_value >= 1:
            parsed.append(next_value)
    return parsed or list(fallback)


def _fixed_length_positive_int_list(value: Any, fallback: list[int], length: int) -> list[int]:
    parsed = _positive_int_list(value, fallback)
    padded = parsed[:length]
    while len(padded) < length:
        padded.append(fallback[len(padded)] if len(padded) < len(fallback) else fallback[-1])
    return padded


def _earthformer_size_list(value: Any, fallback: list[int]) -> list[int]:
    normalized = _fixed_length_positive_int_list(value, fallback, 3)
    normalized[0] = 1
    return normalized


def get_channels_from_hyperparameters(hyperparameters: Optional[dict[str, Any]]) -> list[str]:
    raw_channels = (hyperparameters or {}).get("selected_channels", [])
    if isinstance(raw_channels, str):
        raw_items = raw_channels.replace("+", ",").split(",")
    elif isinstance(raw_channels, (list, tuple, set)):
        raw_items = raw_channels
    else:
        raw_items = []

    selected = {str(item).strip().upper() for item in raw_items}
    return [channel for channel in CHANNEL_ORDER if channel in selected]


def extract_architecture_params(hyperparameters: Optional[dict[str, Any]]) -> dict[str, Any]:
    hypers = hyperparameters or {}
    return {
        key: hypers[key]
        for key in ARCHITECTURE_PARAM_KEYS
        if key in hypers and not _is_blank(hypers.get(key))
    }


def normalize_training_hyperparameters(hyperparameters: Optional[dict[str, Any]]) -> dict[str, Any]:
    normalized = dict(hyperparameters or {})
    normalized.update(normalize_split_ratios(normalized))

    normalized["epochs"] = _positive_int(normalized.get("epochs"), 10)
    normalized["batch_size"] = _positive_int(normalized.get("batch_size"), 32)
    normalized["learning_rate"] = _positive_float(normalized.get("learning_rate"), 0.001)
    normalized["window"] = _positive_int(normalized.get("window"), 3)
    normalized["horizon"] = _positive_int(normalized.get("horizon"), 3)
    normalized["training_dataset"] = _training_dataset(normalized.get("training_dataset"))
    normalized["early_stopping_patience"] = _non_negative_int(
        normalized.get("early_stopping_patience"), 0, maximum=200
    )
    normalized["seed"] = _non_negative_int(
        normalized.get("seed"), 11, maximum=2147483647
    )
    use_sphere = normalize_use_sphere(normalized)
    normalized["model_architecture"] = normalize_model_architecture(normalized.get("model_architecture"))

    if normalized["model_architecture"] in RECURRENT_MODEL_ARCHITECTURES:
        raw_hidden_dims = normalized.get("stlstm_hidden_dims", [64, 64, 64])
        if not isinstance(raw_hidden_dims, (list, tuple)):
            raw_hidden_dims = [raw_hidden_dims]
        normalized["stlstm_hidden_dims"] = [
            _positive_int(dim, 64) for dim in raw_hidden_dims
        ] or [64]
    else:
        normalized.pop("stlstm_hidden_dims", None)

    for key, fallback in ARCHITECTURE_INTEGER_PARAMS.items():
        if key in normalized:
            normalized[key] = _positive_int(normalized.get(key), fallback)

    for key, fallback in ARCHITECTURE_FLOAT_PARAMS.items():
        if key in normalized:
            minimum = MIN_POSITIVE_FLOAT if key in OPEN_INTERVAL_FLOAT_PARAMS else 0.0
            maximum = 0.9 if key in ZERO_TO_POINT_NINE_FLOAT_PARAMS else None
            normalized[key] = _positive_float(normalized.get(key), fallback, minimum=minimum, maximum=maximum)

    for key, fallback in ARCHITECTURE_INTEGER_LIST_PARAMS.items():
        if key in normalized:
            if key in THREE_VALUE_INTEGER_LIST_PARAMS:
                normalized[key] = _earthformer_size_list(normalized.get(key), fallback)
            else:
                normalized[key] = _positive_int_list(normalized.get(key), fallback)

    normalized["selected_channels"] = get_channels_from_hyperparameters(normalized)
    normalized["use_sphere"] = use_sphere
    transfer_enabled = _bool_value(normalized.get("transfer_learning"), False)
    if transfer_enabled:
        normalized["transfer_learning"] = True
        source_type = _safe_string(normalized.get("transfer_source_type"), "task").lower()
        normalized["transfer_source_type"] = source_type if source_type in {"task", "upload"} else "task"
        normalized["transfer_source_task_id"] = _non_negative_int(
            normalized.get("transfer_source_task_id"),
            0,
        )
        normalized["transfer_weight_id"] = _safe_string(normalized.get("transfer_weight_id"), "")
        normalized["transfer_load_mode"] = "strict"
        normalized["freeze_mode"] = _freeze_mode(normalized.get("freeze_mode"))
        normalized["finetune_learning_rate"] = _positive_float(
            normalized.get("finetune_learning_rate"),
            normalized["learning_rate"],
        )
    else:
        for key in (
            "transfer_learning",
            "transfer_source_type",
            "transfer_source_task_id",
            "transfer_weight_id",
            "transfer_load_mode",
            "freeze_mode",
            "finetune_learning_rate",
        ):
            normalized.pop(key, None)
    return normalized


def get_task_channel_suffix(task: Any) -> str:
    try:
        hyperparameters = json.loads(getattr(task, "hyperparameters", "") or "{}")
    except Exception:
        hyperparameters = {}

    channels = get_channels_from_hyperparameters(hyperparameters)
    if channels:
        return "".join(channels)

    script_name = getattr(task, "model_script", "") or ""
    if script_name.startswith("demo3-") and script_name.endswith(".py"):
        return script_name.replace("demo3-", "", 1).replace(".py", "", 1)
    return ""


def serialize_hyperparameter_value(key: str, value: Any) -> Optional[str]:
    if key.startswith("_"):
        return None
    if key == "selected_channels":
        return ",".join(get_channels_from_hyperparameters({key: value}))
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def build_hyperparameter_args(hyperparameters: Optional[dict[str, Any]]) -> list[str]:
    args: list[str] = []
    for key, value in (hyperparameters or {}).items():
        serialized = serialize_hyperparameter_value(key, value)
        if serialized is None:
            continue
        args.extend([f"--{key}", serialized])
    return args
