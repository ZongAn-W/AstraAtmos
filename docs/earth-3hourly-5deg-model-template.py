"""Standalone SimVP for complete normalized Earth three-hour global fields.

Standalone 5 degree (36 x 72) model. Declare the independent low-resolution
feed; time cadence remains three hours. This template declares no derived inputs.
"""

import torch
from torch import nn
from torch.nn import functional as F


MODEL_SPEC = {
    "name": "EarthThreeHour5DegreeSimVP",
    "description": (
        "Per-frame spatial encoder, multiscale SimVP temporal translator, "
        "and spatial decoder for normalized TO3 forecasts."
    ),
    "parameters": {
        "spatial_hidden_dim": {"type": "int", "default": 8, "min": 4, "max": 256},
        "temporal_hidden_dim": {"type": "int", "default": 16, "min": 8, "max": 512},
        "num_temporal_blocks": {"type": "int", "default": 2, "min": 1, "max": 8},
        "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
    },
    "datasets": {
        "earth_merra2_3hourly_5deg_v1": {
            "schema": "astraatmos_earth_3hourly_5deg_model_v1",
            "frequency_hours": 3,
            "step_unit": "hour",
            "step": 3,
            "time_zone": "UTC",
            "window": list(range(1, 241)),
            "horizon": list(range(1, 241)),
            "grid": [[36, 72]],
            "target": "TO3",
            "target_unit": "DU",
            "input_channels": ["TO3", "U10M", "V10M", "T2M", "SWGDN"],
            "input_units": ["DU", "m s-1", "m s-1", "K", "W m-2"],
            "dtype": "float32",
            "tensor_layout": "BTCHW",
            "auxiliary_inputs": ["U10M", "V10M", "T2M", "SWGDN"],
            "spatial_tile_shape": [36, 72],
            "output_channels": ["TO3"],
            "output_units": ["DU"],
        },
    },
}


class LongitudeConv2d(nn.Conv2d):
    def _conv_forward(self, inputs, weight, bias):
        latitude_padding, longitude_padding = self.padding
        if longitude_padding:
            inputs = F.pad(inputs, (longitude_padding, longitude_padding, 0, 0), mode="circular")
        return F.conv2d(
            inputs, weight, bias, self.stride, (latitude_padding, 0),
            self.dilation, self.groups,
        )


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            LongitudeConv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            LongitudeConv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class TemporalInceptionBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.branch_3x3 = LongitudeConv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.branch_5x5 = LongitudeConv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels
        )
        self.merge = LongitudeConv2d(2 * channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        normalized = self.norm(x)
        multiscale = torch.cat(
            (self.branch_3x3(normalized), self.branch_5x5(normalized)), dim=1
        )
        return x + self.dropout(self.activation(self.merge(multiscale)))


class TemporalTranslator(nn.Module):
    def __init__(
        self, window, horizon, spatial_hidden_dim, temporal_hidden_dim, num_blocks, dropout
    ):
        super().__init__()
        self.input_projection = LongitudeConv2d(
            window * spatial_hidden_dim, temporal_hidden_dim, kernel_size=1
        )
        self.blocks = nn.Sequential(
            *[
                TemporalInceptionBlock(temporal_hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_projection = LongitudeConv2d(
            temporal_hidden_dim, horizon * spatial_hidden_dim, kernel_size=1
        )

    def forward(self, x):
        return self.output_projection(self.blocks(self.input_projection(x)))


class SpatialDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.GELU(),
            LongitudeConv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class EarthThreeHour5DegreeSimVP(nn.Module):
    def __init__(
        self,
        in_channels,
        window,
        horizon,
        spatial_hidden_dim,
        temporal_hidden_dim,
        num_temporal_blocks,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.spatial_hidden_dim = spatial_hidden_dim
        self.spatial_encoder = SpatialEncoder(in_channels, spatial_hidden_dim)
        self.temporal_translator = TemporalTranslator(
            window,
            horizon,
            spatial_hidden_dim,
            temporal_hidden_dim,
            num_temporal_blocks,
            dropout,
        )
        self.spatial_decoder = SpatialDecoder(spatial_hidden_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 5:
            raise ValueError("Expected [batch, window, channels, 36, 72] input.")
        if x.dtype != torch.float32:
            raise ValueError("x must use float32, as required by the Earth contract.")

        batch, window, channels, height, width = x.shape
        if batch < 1:
            raise ValueError("batch must contain at least one sample.")
        if window != self.window:
            raise ValueError(f"Expected window={self.window}, received {window}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, received {channels}.")
        if (height, width) != (36, 72):
            raise ValueError("Expected the Earth spatial tile shape [36, 72].")

        encoded = self.spatial_encoder(
            x.reshape(batch * window, channels, height, width)
        )
        encoded_height, encoded_width = encoded.shape[-2:]
        # Folding time into channels lets the translator mix the entire history.
        history = encoded.reshape(
            batch, window * self.spatial_hidden_dim, encoded_height, encoded_width
        )
        future = self.temporal_translator(history).reshape(
            batch * self.horizon,
            self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        decoded = self.spatial_decoder(future)
        return decoded.reshape(batch, self.horizon, 1, height, width)


def _bounded_int(config, key, minimum, maximum, default=None):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer in [{minimum}, {maximum}].")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be an integer in [{minimum}, {maximum}].")
    return value


def build_model(config):
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary.")
    in_channels = _bounded_int(config, "in_channels", 1, 5)
    window = _bounded_int(config, "window", 1, 240)
    horizon = _bounded_int(config, "horizon", 1, 240)
    _bounded_int(config, "height", 36, 36, 36)
    _bounded_int(config, "width", 72, 72, 72)

    expected_metadata = {
        "dataset_id": "earth_merra2_3hourly_5deg_v1",
        "contract_schema": "astraatmos_earth_3hourly_5deg_model_v1",
        "global_grid_shape": [36, 72],
        "spatial_tile_shape": [36, 72],
        "target_channel": "TO3",
    }
    for key, expected in expected_metadata.items():
        if key not in config:
            continue
        value = config[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"{key} must be {expected!r}.")
        if isinstance(expected, list) and any(type(item) is not int for item in value):
            raise ValueError(f"{key} must contain integer dimensions.")

    if "selected_channels" in config:
        selected = config["selected_channels"]
        channel_order = ["TO3", "U10M", "V10M", "T2M", "SWGDN"]
        if not isinstance(selected, list) or len(selected) != in_channels:
            raise ValueError("selected_channels must be a list matching in_channels.")
        ordered = [channel for channel in channel_order if channel in selected]
        if not selected or selected[0] != "TO3" or selected != ordered:
            raise ValueError("selected_channels must contain TO3 then ordered auxiliaries.")

    parameters = {}
    for key in ("spatial_hidden_dim", "temporal_hidden_dim", "num_temporal_blocks"):
        schema = MODEL_SPEC["parameters"][key]
        parameters[key] = _bounded_int(
            config, key, schema["min"], schema["max"], schema["default"]
        )
    dropout_schema = MODEL_SPEC["parameters"]["dropout"]
    dropout = config.get("dropout", dropout_schema["default"])
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not dropout_schema["min"] <= dropout <= dropout_schema["max"]
    ):
        raise ValueError("dropout must be a finite number in [0.0, 0.9].")

    return EarthThreeHour5DegreeSimVP(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        dropout=float(dropout),
        **parameters,
    )
