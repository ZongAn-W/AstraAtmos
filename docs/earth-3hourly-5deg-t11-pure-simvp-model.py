"""Pure SimVP control extracted from the frozen T11 model (task 90).

T11 source SHA-256:
38b4e6dd45318d3d364562e2e844f6ece4e9dfe1d2776c8598e1a89d2d019c4e

The SpatialEncoder, TemporalInceptionBlock, TemporalTranslator and SpatialDecoder
definitions are copied unchanged from T11. This control removes ConvLSTM and all
seasonal/terrain gates. The spatial encoder consequently takes the observation
channels directly, instead of the 16 ConvLSTM feature channels. There is no extra
input adapter, terrain encoder or Ls encoder. All other SimVP layers, padding and
default widths are retained: spatial 64, temporal 128, two blocks, dropout 0.1.

Use Earth three-hour 5 degree data, five channels, window=20, horizon=20, batch=32,
epochs=20, learning_rate=0.001, seed=11, patience=5 and split=70/15/15 to match T11.
The platform supplies training and checkpoint handling; this file exports only
MODEL_SPEC and build_model(config). Forward takes x alone and returns normalized
TO3 [B,horizon,1,36,72]. No derived-input assets are required.

Large CUDA batches use the same sample chunking and non-reentrant activation
checkpoint policy as T11. These do not add layers or alter the optimizer batch.
This is T11's SimVP backbone control, not a claim of reproducing another published
SimVP implementation or having the same total parameter count as the full T11.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


MODEL_SPEC = {
    "name": "EarthThreeHour5DegreePureSimVPT11Control",
    "description": "Pure SimVP control using T11's spatial encoder, temporal blocks and decoder, without ConvLSTM or seasonal/terrain gates.",
    "parameters": {
        "spatial_hidden_dim": {"type": "int", "default": 64, "min": 4, "max": 256},
        "temporal_hidden_dim": {"type": "int", "default": 128, "min": 8, "max": 512},
        "num_temporal_blocks": {"type": "int", "default": 2, "min": 2, "max": 8},
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


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class TemporalInceptionBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.branch_3x3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.branch_5x5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
        )
        self.merge = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        normalized = self.norm(x)
        mixed = torch.cat(
            (self.branch_3x3(normalized), self.branch_5x5(normalized)), dim=1
        )
        return x + self.dropout(self.activation(self.merge(mixed)))


class TemporalTranslator(nn.Module):
    def __init__(
        self,
        window,
        horizon,
        spatial_hidden_dim,
        temporal_hidden_dim,
        num_blocks,
        dropout,
    ):
        super().__init__()
        self.horizon = horizon
        self.spatial_hidden_dim = spatial_hidden_dim
        self.input_projection = nn.Conv2d(
            window * spatial_hidden_dim, temporal_hidden_dim, kernel_size=1
        )
        self.blocks = nn.Sequential(
            *[
                TemporalInceptionBlock(temporal_hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_projection = nn.Conv2d(
            temporal_hidden_dim, horizon * spatial_hidden_dim, kernel_size=1
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.output_projection(x)


class SpatialDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, output_size):
        x = self.layers(x)
        if x.shape[-2:] != output_size:
            x = torch.nn.functional.interpolate(
                x, size=output_size, mode="bilinear", align_corners=False
            )
        return x


class EarthThreeHour5DegreePureSimVPT11Control(nn.Module):
    def __init__(self, in_channels, window, horizon, spatial_hidden_dim,
                 temporal_hidden_dim, num_temporal_blocks, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.spatial_hidden_dim = spatial_hidden_dim
        self.spatial_encoder = SpatialEncoder(in_channels, spatial_hidden_dim)
        self.temporal_translator = TemporalTranslator(
            window, horizon, spatial_hidden_dim, temporal_hidden_dim,
            num_temporal_blocks, dropout,
        )
        self.spatial_decoder = SpatialDecoder(spatial_hidden_dim)

    def _validate_inputs(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("x must have shape [batch, window, channels, height, width].")
        parameter = next(self.parameters())
        if not torch.is_floating_point(x) or x.dtype != parameter.dtype:
            raise ValueError("x and model parameters must use the same floating dtype.")
        if x.device != parameter.device:
            raise ValueError("x and model parameter device must match.")
        if not torch.isfinite(x).all():
            raise ValueError("x values must all be finite; NaN and Inf are invalid.")
        batch, window, channels, height, width = x.shape
        if (height, width) != (36, 72):
            raise ValueError("Earth 5 degree input must have spatial shape [36, 72].")
        if batch <= 0:
            raise ValueError("x batch must be positive.")
        if window != self.window:
            raise ValueError(f"x window must be {self.window}, but received {window}.")
        if channels != self.in_channels:
            raise ValueError(f"x channels must be {self.in_channels}, but received {channels}.")

    def forward(self, x):
        self._validate_inputs(x)
        if x.is_cuda and x.shape[0] * self.window >= 256:
            chunk_size = max(1, 128 // self.window)
            predictions = []
            for start in range(0, x.shape[0], chunk_size):
                inputs = x[start:start + chunk_size]
                if self.training and torch.is_grad_enabled():
                    prediction = checkpoint(self._forward_impl, inputs, use_reentrant=False)
                else:
                    prediction = self._forward_impl(inputs)
                predictions.append(prediction)
            return torch.cat(predictions, dim=0)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        batch, window, channels, height, width = x.shape
        encoded = self.spatial_encoder(x.reshape(batch * window, channels, height, width))
        encoded_height, encoded_width = encoded.shape[-2:]
        history = encoded.reshape(
            batch, window * self.spatial_hidden_dim, encoded_height, encoded_width,
        )
        translated = self.temporal_translator(history)
        decoded = self.spatial_decoder(
            translated.reshape(
                batch * self.horizon, self.spatial_hidden_dim, encoded_height, encoded_width,
            ),
            (height, width),
        )
        return decoded.reshape(batch, self.horizon, 1, height, width)


def _positive_int(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def build_model(config):
    values = dict(config)
    expected_contract = {
        "dataset_id": "earth_merra2_3hourly_5deg_v1",
        "contract_schema": "astraatmos_earth_3hourly_5deg_model_v1",
        "height": 36,
        "width": 72,
        "global_grid_shape": [36, 72],
        "spatial_tile_shape": [36, 72],
    }
    for key, expected in expected_contract.items():
        if key not in values:
            continue
        value = values[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"{key} must equal {expected!r} for the 36x72 Earth contract.")
        if isinstance(expected, list) and any(type(item) is not int for item in value):
            raise ValueError(f"{key} must contain integer dimensions.")
    for key, schema in MODEL_SPEC["parameters"].items():
        value = values.setdefault(key, schema["default"])
        if schema["type"] == "int":
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid_type or not schema["min"] <= value <= schema["max"]:
            raise ValueError(
                f"{key} must be {schema['type']} in "
                f"[{schema['min']}, {schema['max']}]."
            )

    return EarthThreeHour5DegreePureSimVPT11Control(
        in_channels=_positive_int(values, "in_channels"),
        window=_positive_int(values, "window"),
        horizon=_positive_int(values, "horizon"),
        spatial_hidden_dim=values["spatial_hidden_dim"],
        temporal_hidden_dim=values["temporal_hidden_dim"],
        num_temporal_blocks=values["num_temporal_blocks"],
        dropout=float(values["dropout"]),
    )
