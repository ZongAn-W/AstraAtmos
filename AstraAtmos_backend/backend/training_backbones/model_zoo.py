from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


MODEL_LABELS = {
    "predrnnv2": "PredRNNv2",
    "predrnnpp": "PredRNN++",
    "convlstm": "ConvLSTM",
    "simvp": "SimVP",
    "dlinear": "DLinear",
    "informer": "Informer",
    "autoformer": "Autoformer",
    "patchtst": "PatchTST",
    "timemixer": "TimeMixer",
    "timexer": "TimeXer",
    "tsmixer": "TSMixer",
    "crossformer": "Crossformer",
    "earthformer": "Earthformer",
    "etsformer": "ETSformer",
    "fedformer": "FEDformer",
    "itransformer": "iTransformer",
    "mau": "MAU",
    "nbeats": "N-BEATS",
    "nhits": "N-HiTS",
    "pyraformer": "Pyraformer",
    "rnn_cnn_rnn": "RNN-CNN-RNN",
    "cnn_rnn_cnn_rnn_cnn": "CNN-RNN-CNN-RNN-CNN",
    "simvp_3dconv": "SimVP-3DConv",
    "simvp_hybrid3d": "SimVP-Hybrid3D",
    "convlstm_mst": "ConvLSTM-MST",
    "dlinear_mst": "DLinear-MST",
    "convlstm_phase_gated_mst": "ConvLSTM-PhaseGated-MST",
    "convlstm_mst_feature_refiner": "ConvLSTM-MST-Feature",
    "convlstm_climatology_anomaly": "ConvLSTM-Climatology-Anomaly",
}

MIGRATED_MODEL_SPECS = [
    {"id": "predrnnv2", "label": "PredRNNv2"},
    {"id": "predrnnpp", "label": "PredRNN++"},
    {"id": "convlstm", "label": "ConvLSTM"},
    {"id": "simvp", "label": "SimVP"},
    {"id": "dlinear", "label": "DLinear"},
    {"id": "informer", "label": "Informer"},
    {"id": "autoformer", "label": "Autoformer"},
    {"id": "patchtst", "label": "PatchTST"},
    {"id": "timemixer", "label": "TimeMixer"},
    {"id": "timexer", "label": "TimeXer"},
    {"id": "tsmixer", "label": "TSMixer"},
    {"id": "crossformer", "label": "Crossformer"},
    {"id": "earthformer", "label": "Earthformer"},
    {"id": "etsformer", "label": "ETSformer"},
    {"id": "fedformer", "label": "FEDformer"},
    {"id": "itransformer", "label": "iTransformer"},
    {"id": "mau", "label": "MAU"},
    {"id": "nbeats", "label": "N-BEATS"},
    {"id": "nhits", "label": "N-HiTS"},
    {"id": "pyraformer", "label": "Pyraformer"},
    {"id": "rnn_cnn_rnn", "label": "RNN-CNN-RNN"},
    {"id": "cnn_rnn_cnn_rnn_cnn", "label": "CNN-RNN-CNN-RNN-CNN"},
    {"id": "simvp_3dconv", "label": "SimVP-3DConv"},
    {"id": "simvp_hybrid3d", "label": "SimVP-Hybrid3D"},
    {"id": "convlstm_mst", "label": "ConvLSTM-MST"},
    {"id": "dlinear_mst", "label": "DLinear-MST"},
    {"id": "convlstm_phase_gated_mst", "label": "ConvLSTM-PhaseGated-MST"},
    {"id": "convlstm_mst_feature_refiner", "label": "ConvLSTM-MST-Feature"},
    {"id": "convlstm_climatology_anomaly", "label": "ConvLSTM-Climatology-Anomaly"},
]

MODEL_IDS = tuple(spec["id"] for spec in MIGRATED_MODEL_SPECS)


def list_model_specs() -> list[dict[str, str]]:
    return [{"id": model_id, "label": MODEL_LABELS[model_id]} for model_id in FACTORIES]


def normalize_model_architecture(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "predrnnv2"
    if normalized == "predrnnv2_sphere":
        return "predrnnv2"
    if normalized in MODEL_IDS:
        return normalized
    raise ValueError(f"Unsupported model architecture: {value}")


def normalize_use_sphere(hyperparameters: Optional[dict[str, Any]]) -> bool:
    hypers = hyperparameters or {}
    raw = hypers.get("use_sphere", False)
    if isinstance(raw, str):
        raw = raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw) or str(hypers.get("model_architecture", "")).strip().lower().endswith("_sphere")


class SpherePhaseWarpFrontEnd(nn.Module):
    def __init__(self, extra_channel_count: int, height: int, width: int):
        super().__init__()
        self.extra_channel_count = int(extra_channel_count)
        self.height = int(height)
        self.width = int(width)
        spatial_shape = (1, 1, 1, self.height, self.width)
        pair_shape = (1, 1, max(1, self.extra_channel_count), self.height, self.width)

        self.o3_weight = nn.Parameter(torch.zeros(spatial_shape))
        self.o3_bias = nn.Parameter(torch.zeros(spatial_shape))
        self.o3_warp = nn.Parameter(torch.zeros(spatial_shape))

        self.sin_weight = nn.Parameter(torch.ones(pair_shape))
        self.cos_weight = nn.Parameter(torch.ones(pair_shape))
        self.sin_bias = nn.Parameter(torch.zeros(pair_shape))
        self.cos_bias = nn.Parameter(torch.zeros(pair_shape))
        self.sin_warp = nn.Parameter(torch.zeros(pair_shape))
        self.cos_warp = nn.Parameter(torch.zeros(pair_shape))

    def _ls_to_radians(self, ls: torch.Tensor) -> torch.Tensor:
        if ls.dim() != 2:
            raise ValueError(f"Expected ls with shape [B, T], got {tuple(ls.shape)}")
        return (ls * (torch.pi / 180.0)).view(ls.size(0), ls.size(1), 1, 1, 1)

    def forward(self, x: torch.Tensor, ls: torch.Tensor) -> torch.Tensor:
        expected_channels = 1 + self.extra_channel_count
        if x.size(2) != expected_channels:
            raise ValueError(f"Expected {expected_channels} input channels, got {x.size(2)}")

        ls_rad = self._ls_to_radians(ls)
        o3 = x[:, :, 0:1, ...]
        o3_fused = o3 + self.o3_weight * torch.sin(
            ls_rad + torch.tanh(self.o3_warp) * torch.cos(ls_rad) + self.o3_bias
        )

        if self.extra_channel_count <= 0:
            return o3_fused

        extra = x[:, :, 1:, ...]
        sin_weight = self.sin_weight[:, :, :self.extra_channel_count, ...]
        cos_weight = self.cos_weight[:, :, :self.extra_channel_count, ...]
        sin_bias = self.sin_bias[:, :, :self.extra_channel_count, ...]
        cos_bias = self.cos_bias[:, :, :self.extra_channel_count, ...]
        sin_warp = self.sin_warp[:, :, :self.extra_channel_count, ...]
        cos_warp = self.cos_warp[:, :, :self.extra_channel_count, ...]

        extra_sin = extra * sin_weight * torch.sin(
            ls_rad + torch.tanh(sin_warp) * torch.cos(ls_rad) + sin_bias
        )
        extra_cos = extra * cos_weight * torch.cos(
            ls_rad + torch.tanh(cos_warp) * torch.sin(ls_rad) + cos_bias
        )
        return torch.cat([o3_fused, extra_sin, extra_cos], dim=2)


class ChannelProjector(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        if self.input_channels == self.output_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")
        projected = self.projection(x.reshape(batch_size * seq_len, channels, height, width))
        return projected.reshape(batch_size, seq_len, self.output_channels, height, width)


class ForecasterAdapter(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        input_channels: int,
        backbone_channels: int,
        selected_channel_count: int,
        height: int,
        width: int,
        use_sphere: bool,
    ):
        super().__init__()
        self.use_sphere = bool(use_sphere)
        self.sphere = SpherePhaseWarpFrontEnd(selected_channel_count, height, width) if self.use_sphere else None
        self.projector = ChannelProjector(
            input_channels=1 + (2 * selected_channel_count) if self.use_sphere else input_channels,
            output_channels=backbone_channels,
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor, ls: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.sphere is not None:
            if ls is None:
                raise ValueError("SPHERE model requires Ls sequence input")
            x = self.sphere(x, ls)
        x = self.projector(x)
        try:
            output = self.backbone(x, ls)
        except TypeError:
            output = self.backbone(x)

        if output.dim() == 4:
            output = output.unsqueeze(2)
        return output


@dataclass(frozen=True)
class BackboneFactory:
    module_name: str
    class_name: str
    input_channels: int
    builder: Callable[[type[nn.Module], dict[str, Any]], nn.Module]


def _import_backbone_class(module_name: str, class_name: str) -> type[nn.Module]:
    module = import_module(f"training_backbones.ablation.{module_name}")
    return getattr(module, class_name)


def _hidden_dims(params: dict[str, Any], minimum_layers: int = 1) -> list[int]:
    dims = list(params["hidden_dims"] or [64, 64, 64])
    while len(dims) < minimum_layers:
        dims.append(dims[-1] if dims else 64)
    return dims


def _hidden_dims_or_single(params: dict[str, Any], minimum_layers: int = 1) -> list[int]:
    if params.get("hidden_dim") not in (None, ""):
        dims = [_param_int(params, "hidden_dim", _small_dim(params))]
    else:
        dims = _hidden_dims(params, minimum_layers)
    while len(dims) < minimum_layers:
        dims.append(dims[-1] if dims else 64)
    return dims


def _small_dim(params: dict[str, Any], fallback: int = 32) -> int:
    dims = params.get("hidden_dims") or []
    return int(dims[0]) if dims else fallback


def _param_int(params: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(params.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _param_float(params: dict[str, Any], key: str, fallback: float) -> float:
    try:
        return float(params.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _param_list_int(params: dict[str, Any], key: str, fallback: list[int]) -> list[int]:
    value = params.get(key, fallback)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return list(fallback)
        value = value.strip("[]")
        items = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return list(fallback)
    parsed: list[int] = []
    for item in items:
        try:
            parsed_item = int(round(float(item)))
        except (TypeError, ValueError):
            continue
        if parsed_item > 0:
            parsed.append(parsed_item)
    return parsed or list(fallback)


def _fixed_param_list_int(params: dict[str, Any], key: str, fallback: list[int], length: int) -> list[int]:
    values = _param_list_int(params, key, fallback)[:length]
    while len(values) < length:
        values.append(fallback[len(values)] if len(values) < len(fallback) else fallback[-1])
    return values


def _earthformer_size(params: dict[str, Any], key: str) -> tuple[int, int, int]:
    values = _fixed_param_list_int(params, key, [1, 2, 2], 3)
    values[0] = 1
    return tuple(values)


def _spatial_dim(params: dict[str, Any]) -> int:
    return _param_int(params, "spatial_hidden_dim", max(8, _small_dim(params)))


def _temporal_dim(params: dict[str, Any]) -> int:
    return _param_int(params, "temporal_hidden_dim", max(16, _small_dim(params) * 2))


def _mst_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "mst_spatial_hidden_dim": _param_int(params, "mst_spatial_hidden_dim", _spatial_dim(params)),
        "mst_temporal_hidden_dim": _param_int(params, "mst_temporal_hidden_dim", _temporal_dim(params)),
        "mst_num_downsample": _param_int(params, "mst_num_downsample", 1),
        "mst_temporal_depth": _param_int(params, "mst_temporal_depth", _param_int(params, "temporal_depth", 1)),
        "dropout": _param_float(params, "dropout", 0.1),
    }


def _grid_builder(class_obj: type[nn.Module], **kwargs: Any) -> nn.Module:
    return class_obj(**kwargs)


FACTORIES: dict[str, BackboneFactory] = {
    "predrnnv2": BackboneFactory(
        "predrnnv2_phasewarp_compare",
        "PredRNNv2Forecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dims=_hidden_dims_or_single(p),
        ),
    ),
    "predrnnpp": BackboneFactory(
        "predrnnpp_phasewarp_compare",
        "PredRNNPPForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dims=_hidden_dims(p, minimum_layers=2),
        ),
    ),
    "convlstm": BackboneFactory(
        "convlstm_phasewarp_compare",
        "ConvLSTMForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dims=_hidden_dims_or_single(p),
        ),
    ),
    "simvp": BackboneFactory(
        "simvp_phasewarp_compare",
        "SimVPForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            spatial_hidden_dim=_param_int(p, "spatial_hidden_dim", max(8, _small_dim(p))),
            temporal_hidden_dim=_param_int(p, "temporal_hidden_dim", max(16, _small_dim(p) * 2)),
            num_downsample=1,
            temporal_depth=_param_int(p, "temporal_depth", 2),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "dlinear": BackboneFactory(
        "dlinear_phasewarp_compare",
        "GridPointDLinearO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            linear_hidden_layers=_param_int(p, "linear_hidden_layers", 2),
        ),
    ),
    "informer": BackboneFactory(
        "informer_phasewarp_compare",
        "GridPointInformerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            n_heads=_param_int(p, "n_heads", 2),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "autoformer": BackboneFactory(
        "autoformer_phasewarp_compare",
        "GridPointAutoformerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            n_heads=_param_int(p, "n_heads", 2),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            moving_avg=_param_int(p, "moving_avg", 3),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "patchtst": BackboneFactory(
        "patchtst_phasewarp_compare",
        "GridPointPatchTSTO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            patch_len=_param_int(p, "patch_len", max(1, min(2, p["window"]))),
            stride=_param_int(p, "stride", 1),
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            n_heads=_param_int(p, "n_heads", 2),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "timemixer": BackboneFactory(
        "timemixer_phasewarp_compare",
        "GridPointTimeMixerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            e_layers=_param_int(p, "e_layers", 1),
            moving_avg=_param_int(p, "moving_avg", 3),
            down_sampling_window=_param_int(p, "down_sampling_window", 2),
            down_sampling_layers=_param_int(p, "down_sampling_layers", 1),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "timexer": BackboneFactory(
        "timexer_phasewarp_compare",
        "GridPointTimeXerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            patch_len=_param_int(p, "patch_len", max(1, min(2, p["window"]))),
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            n_heads=_param_int(p, "n_heads", 2),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "tsmixer": BackboneFactory(
        "tsmixer_phasewarp_compare",
        "GridPointTSMixerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dim=_param_int(p, "hidden_dim", max(8, _small_dim(p))),
            e_layers=_param_int(p, "e_layers", 1),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "crossformer": BackboneFactory(
        "crossformer_phasewarp_compare",
        "GridPointCrossformerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            seg_len=_param_int(p, "seg_len", 1),
            win_size=_param_int(p, "win_size", 2),
            factor=_param_int(p, "factor", 2),
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            n_heads=_param_int(p, "n_heads", 2),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "earthformer": BackboneFactory(
        "earthformer_phasewarp_compare",
        "EarthformerForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            num_heads=_param_int(p, "n_heads", 2),
            num_layers=_param_int(p, "e_layers", 1),
            patch_size=_earthformer_size(p, "patch_size"),
            cuboid_size=_earthformer_size(p, "cuboid_size"),
            mlp_ratio=_param_int(p, "mlp_ratio", 2),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "etsformer": BackboneFactory(
        "etsformer_phasewarp_compare",
        "GridPointETSformerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            dropout=_param_float(p, "dropout", 0.1),
            top_k_freq=_param_int(p, "top_k_freq", 2),
        ),
    ),
    "fedformer": BackboneFactory(
        "fedformer_phasewarp_compare",
        "GridPointFEDformerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            moving_avg=_param_int(p, "moving_avg", 3),
            dropout=_param_float(p, "dropout", 0.1),
            modes=_param_int(p, "modes", 4),
        ),
    ),
    "itransformer": BackboneFactory(
        "itransformer_phasewarp_compare",
        "GridPointITransformerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            n_heads=_param_int(p, "n_heads", 2),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "mau": BackboneFactory(
        "mau_phasewarp_compare",
        "MAUForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dim=_param_int(p, "hidden_dim", max(8, _small_dim(p))),
            num_layers=_param_int(p, "e_layers", 1),
            tau=_param_int(p, "tau", 2),
            kernel_size=_param_int(p, "kernel_size", 3),
            gamma=_param_float(p, "gamma", 1.0),
        ),
    ),
    "nbeats": BackboneFactory(
        "nbeats_phasewarp_compare",
        "GridPointNBeatsO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            stack_count=_param_int(p, "stack_count", 1),
            blocks_per_stack=_param_int(p, "blocks_per_stack", 1),
            hidden_dim=_param_int(p, "hidden_dim", max(8, _small_dim(p))),
            n_layers=_param_int(p, "e_layers", 1),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "nhits": BackboneFactory(
        "nhits_phasewarp_compare",
        "GridPointNHitsO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            stack_count=_param_int(p, "stack_count", 1),
            blocks_per_stack=_param_int(p, "blocks_per_stack", 1),
            hidden_dim=_param_int(p, "hidden_dim", max(8, _small_dim(p))),
            n_layers=_param_int(p, "e_layers", 1),
            pooling_sizes=_param_list_int(p, "pooling_sizes", [1]),
            downsample_factors=_param_list_int(p, "downsample_factors", [1]),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "pyraformer": BackboneFactory(
        "pyraformer_phasewarp_compare",
        "GridPointPyraformerO3",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            d_model=_param_int(p, "d_model", max(8, _small_dim(p))),
            n_heads=_param_int(p, "n_heads", 2),
            e_layers=_param_int(p, "e_layers", 1),
            d_ff=_param_int(p, "d_ff", max(16, _small_dim(p) * 2)),
            dropout=_param_float(p, "dropout", 0.1),
            pooling_sizes=_param_list_int(p, "pooling_sizes", [1, 2]),
        ),
    ),
    "rnn_cnn_rnn": BackboneFactory(
        "rnn_cnn_rnn_phasewarp_compare",
        "RNNCNNRNNForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            temporal_hidden_dim=_temporal_dim(p),
            spatial_hidden_dim=_spatial_dim(p),
            cnn_depth=_param_int(p, "cnn_depth", 1),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "cnn_rnn_cnn_rnn_cnn": BackboneFactory(
        "cnn_rnn_cnn_rnn_cnn_phasewarp_compare",
        "CNNRNNCNNRNNCNNForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            spatial_hidden_dim=_spatial_dim(p),
            temporal_hidden_dim=_temporal_dim(p),
            cnn_depth=_param_int(p, "cnn_depth", 1),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "simvp_3dconv": BackboneFactory(
        "simvp_3dconv_phasewarp_compare",
        "SimVP3DConvForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            spatial_hidden_dim=_spatial_dim(p),
            temporal_hidden_dim=_temporal_dim(p),
            num_downsample=_param_int(p, "num_downsample", 1),
            temporal_depth=_param_int(p, "temporal_depth", 1),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "simvp_hybrid3d": BackboneFactory(
        "simvp_hybrid3d_phasewarp_compare",
        "SimVPHybrid3DForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            spatial_hidden_dim=_spatial_dim(p),
            temporal_hidden_dim=_temporal_dim(p),
            num_downsample=_param_int(p, "num_downsample", 1),
            temporal_depth=_param_int(p, "temporal_depth", 1),
            dropout=_param_float(p, "dropout", 0.1),
        ),
    ),
    "convlstm_mst": BackboneFactory(
        "convlstm_mst_phasewarp_compare",
        "ConvLSTMMSTForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dims=_hidden_dims_or_single(p),
            kernel_size=_param_int(p, "kernel_size", 3),
            **_mst_kwargs(p),
        ),
    ),
    "dlinear_mst": BackboneFactory(
        "dlinear_mst_raw",
        "DLinearMSTForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            input_channels=5,
            linear_hidden_layers=_param_int(p, "linear_hidden_layers", 2),
            **_mst_kwargs(p),
        ),
    ),
    "convlstm_phase_gated_mst": BackboneFactory(
        "convlstm_phase_gated_mst",
        "ConvLSTMPhaseGatedMSTForecaster",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dims=_hidden_dims_or_single(p),
            phase_context_dim=_param_int(p, "phase_context_dim", 8),
            kernel_size=_param_int(p, "kernel_size", 3),
            initial_history_weight=_param_float(p, "initial_history_weight", 0.7),
            initial_translation_weight=_param_float(p, "initial_translation_weight", 0.7),
            **_mst_kwargs(p),
        ),
    ),
    "convlstm_mst_feature_refiner": BackboneFactory(
        "convlstm_mst_feature_refiner",
        "ConvLSTMMSTFeatureRefiner",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dims=_hidden_dims_or_single(p),
            kernel_size=_param_int(p, "kernel_size", 3),
            **_mst_kwargs(p),
        ),
    ),
    "convlstm_climatology_anomaly": BackboneFactory(
        "convlstm_climatology_anomaly_refiner",
        "ConvLSTMClimatologyAnomalyRefiner",
        5,
        lambda cls, p: _grid_builder(
            cls,
            seq_len=p["window"],
            pred_len=p["horizon"],
            lat_size=p["height"],
            lon_size=p["width"],
            use_phase_warp=False,
            hidden_dims=_hidden_dims_or_single(p),
            climatology_hidden_dim=_param_int(p, "climatology_hidden_dim", 8),
            kernel_size=_param_int(p, "kernel_size", 3),
            **_mst_kwargs(p),
        ),
    ),
}


def build_forecaster(
    architecture: Any,
    input_channels: int,
    selected_channels: list[str],
    hidden_dims: list[int],
    height: int,
    width: int,
    window: int,
    horizon: int,
    use_sphere: bool = False,
    architecture_params: Optional[dict[str, Any]] = None,
) -> nn.Module:
    normalized = normalize_model_architecture(architecture)
    if normalized not in FACTORIES:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    factory = FACTORIES[normalized]
    class_obj = _import_backbone_class(factory.module_name, factory.class_name)
    params = {
        "input_channels": int(input_channels),
        "selected_channels": list(selected_channels or []),
        "hidden_dims": list(hidden_dims or [64, 64, 64]),
        "height": int(height),
        "width": int(width),
        "window": int(window),
        "horizon": int(horizon),
    }
    params.update(architecture_params or {})
    backbone = factory.builder(class_obj, params)
    return ForecasterAdapter(
        backbone=backbone,
        input_channels=int(input_channels),
        backbone_channels=factory.input_channels,
        selected_channel_count=max(0, int(input_channels) - 1),
        height=int(height),
        width=int(width),
        use_sphere=bool(use_sphere),
    )
