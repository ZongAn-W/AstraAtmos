"""Gated SimVP control extracted from the frozen T11 model (task 90).

T11 source SHA-256:
38b4e6dd45318d3d364562e2e844f6ece4e9dfe1d2776c8598e1a89d2d019c4e

This control removes ConvLSTM and its hidden-dimension parameter. The spatial
encoder takes observation channels directly (five in the T11 experiment), so its
first convolution changes from 16->64 to 5->64. All SimVP component definitions,
seasonal/terrain encoders, the shared gate and the gated translator are retained.
All five gate calls remain for the default two temporal blocks: the gates that
surrounded ConvLSTM now run consecutively on the observations before the spatial
encoder. The gate is then applied after the encoder, between temporal blocks,
and before the decoder. Extra temporal blocks retain a gate between every pair.
The first three calls use per-frame historical Ls; the last two pool its encoding.
All calls share gate weights and the learnable clamped strength.

Input is normalized Earth three-hour history [B,window,C,36,72], historical solar
longitude [B,window], and matching ETOPO terrain [B,1,36,72] in physical metres.
Output is normalized TO3 [B,horizon,1,36,72], without spatial tiling or recurrence.
The independent dataset is earth_merra2_3hourly_5deg_v1 and the contract is
astraatmos_earth_3hourly_5deg_model_v1. T11 defaults are spatial 64, temporal 128,
two temporal blocks, dropout 0.1, gate hidden 32 and initial gate strength 1.0.
Use five channels, 20->20, batch 32, 20 epochs, seed 11, learning rate 0.001,
patience 5 and a 70/15/15 split to compare against T11.

T11 forward-local encoding reuse, CUDA sample chunking and non-reentrant
activation checkpointing are retained. The optimizer batch and time windows do
not change. This is a new standalone model; it cannot load a T11 checkpoint
strictly because the recurrent state is removed and the first convolution differs.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


MODEL_SPEC = {
    "name": "EarthThreeHour5DegreeGatedSimVPNoConvLSTMT11Control",
    "description": "T11 gated SimVP control without ConvLSTM, retaining all five shared seasonal/terrain gate calls at the default two temporal blocks.",
    "parameters": {
        "spatial_hidden_dim": {"type": "int", "default": 64, "min": 4, "max": 256},
        "temporal_hidden_dim": {"type": "int", "default": 128, "min": 8, "max": 512},
        "num_temporal_blocks": {"type": "int", "default": 2, "min": 2, "max": 8},
        "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
        "gate_hidden_dim": {"type": "int", "default": 32, "min": 4, "max": 256},
        "initial_gate_strength": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
    },
    "derived_inputs": {
        "ls": {
            "required": True,
            "shape": ["batch", "window"],
            "dtype": "float32",
            "unit": "degree",
        },
        "topography": {
            "required": True,
            "shape": ["batch", 1, "height", "width"],
            "dtype": "float32",
            "unit": "meter",
        },
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


class LsHarmonicEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def build_harmonic_features(self, ls):
        if not isinstance(ls, torch.Tensor) or ls.ndim != 2:
            raise ValueError("ls must be a rank-2 tensor [batch, window].")
        if not torch.is_floating_point(ls):
            raise TypeError("ls must use a floating-point dtype.")

        radians = torch.deg2rad(ls)
        return torch.stack(
            (
                torch.sin(radians),
                torch.cos(radians),
                torch.sin(2 * radians),
                torch.cos(2 * radians),
            ),
            dim=-1,
        )

    def forward(self, ls):
        harmonics = self.build_harmonic_features(ls)
        harmonics = harmonics.to(dtype=self.layers[0].weight.dtype)
        return self.layers(harmonics)


class TopographyEncoder(nn.Module):
    ELEVATION_SCALE_METERS = 10000.0

    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def scale_elevation(self, topography):
        return topography / self.ELEVATION_SCALE_METERS

    def encode(self, topography):
        topography = topography.to(dtype=self.layers[0].weight.dtype)
        return self.layers(self.scale_elevation(topography))

    def forward(self, topography, output_size):
        encoded = self.encode(topography)
        if encoded.shape[-2:] != output_size:
            encoded = torch.nn.functional.interpolate(
                encoded, size=output_size, mode="bilinear", align_corners=False
            )
        return encoded


class SharedTerrainAffineGate(nn.Module):
    def __init__(self, channels, hidden_dim, initial_gate_strength):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.topography_encoder = TopographyEncoder(hidden_dim)
        self.terrain_multiplier = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.terrain_bias = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.fusion_cnn = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.ls_encoder = LsHarmonicEncoder(hidden_dim)
        self.output_projection = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.gate_strength = nn.Parameter(torch.tensor(float(initial_gate_strength)))
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        nn.init.zeros_(self.output_projection.bias)

    def current_strength(self):
        bounded = self.gate_strength.clamp(0.0, 1.0)
        return self.gate_strength + (bounded - self.gate_strength).detach()

    def prepare_context(self, ls, topography):
        return {"terrain": self.topography_encoder.encode(topography),
                "seasonal": self.ls_encoder(ls), "spatial": {}}

    def gate_values(self, encoded, ls, topography, pool_ls=False, context=None):
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5:
            raise ValueError(
                "encoded must have shape [batch, steps, channels, height, width]."
            )
        batch, steps, channels, height, width = encoded.shape
        if not 0 < channels <= self.channels:
            raise ValueError(f"encoded channels must be in [1, {self.channels}].")
        if not isinstance(ls, torch.Tensor) or ls.ndim != 2:
            raise ValueError("ls must be a tensor with shape [batch, window].")
        if ls.shape[0] != batch or ls.shape[1] == 0:
            raise ValueError("ls must have the same batch and a nonempty history window.")
        if not pool_ls and tuple(ls.shape) != (batch, steps):
            raise ValueError(f"ls shape must be ({batch}, {steps}).")
        if not torch.is_floating_point(ls):
            raise ValueError("ls dtype must be floating point.")
        if ls.device != encoded.device:
            raise ValueError("encoded and ls device must match.")
        if not torch.isfinite(ls).all():
            raise ValueError("ls values must all be finite; NaN and Inf are invalid.")

        # Every position uses the same channel slots and the same convolution weights.
        padded = torch.nn.functional.pad(
            encoded, (0, 0, 0, 0, 0, self.channels - channels)
        )
        if context is None:
            context = self.prepare_context(ls, topography)
        key = (height, width, encoded.dtype)
        if key not in context["spatial"]:
            terrain = context["terrain"]
            if terrain.shape[-2:] != (height, width):
                terrain = torch.nn.functional.interpolate(
                    terrain, size=(height, width), mode="bilinear", align_corners=False
                )
            context["spatial"][key] = (
                self.terrain_multiplier(terrain).to(dtype=encoded.dtype).unsqueeze(1),
                self.terrain_bias(terrain).to(dtype=encoded.dtype).unsqueeze(1),
            )
        multiplier, bias = context["spatial"][key]
        affine = torch.nn.functional.gelu(multiplier * padded + bias)
        dynamic = self.fusion_cnn(
            affine.reshape(batch * steps, self.channels, height, width)
        ).reshape(batch, steps, self.hidden_dim, height, width)
        seasonal = context["seasonal"].to(dtype=dynamic.dtype)
        if pool_ls:
            seasonal = seasonal.mean(dim=1, keepdim=True)
        seasonal = seasonal.unsqueeze(-1).unsqueeze(-1)
        joint = torch.nn.functional.gelu(dynamic + seasonal)
        gate = torch.tanh(self.output_projection(
            joint.reshape(batch * steps, self.hidden_dim, height, width)
        ).reshape(batch, steps, self.channels, height, width))
        return gate[:, :, :channels]

    def forward(self, encoded, ls, topography, pool_ls=False, context=None):
        gate = self.gate_values(encoded, ls, topography, pool_ls=pool_ls, context=context)
        scales = 1.0 + self.current_strength() * gate
        return encoded * scales, gate, scales


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


class SharedGateTranslator(TemporalTranslator):
    def forward(self, x, ls, topography, shared_gate, context=None):
        x = self.input_projection(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index < len(self.blocks) - 1:
                x, _, _ = shared_gate(x.unsqueeze(1), ls, topography, pool_ls=True, context=context)
                x = x.squeeze(1)
        return self.output_projection(x)


class EarthThreeHour5DegreeGatedSimVPNoConvLSTMT11Control(nn.Module):
    def __init__(
        self, in_channels, window, horizon,
        spatial_hidden_dim, temporal_hidden_dim, num_temporal_blocks,
        dropout, gate_hidden_dim, initial_gate_strength,
    ):
        super().__init__()
        if num_temporal_blocks < 2:
            raise ValueError("multi-position gating requires num_temporal_blocks >= 2.")
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.spatial_hidden_dim = spatial_hidden_dim
        self.spatial_encoder = SpatialEncoder(in_channels, spatial_hidden_dim)
        self.temporal_translator = SharedGateTranslator(
            window, horizon, spatial_hidden_dim, temporal_hidden_dim,
            num_temporal_blocks, dropout,
        )
        self.spatial_decoder = SpatialDecoder(spatial_hidden_dim)
        # One registered gate is reused at every modulation site.
        self.shared_gate = SharedTerrainAffineGate(
            channels=max(in_channels, spatial_hidden_dim, temporal_hidden_dim),
            hidden_dim=gate_hidden_dim,
            initial_gate_strength=initial_gate_strength,
        )

    def _validate_inputs(self, x, ls, topography):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError(
                "x must have shape [batch, window, channels, height, width]."
            )

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
        if batch <= 0 or height <= 0 or width <= 0:
            raise ValueError("x batch and spatial dimensions must be positive.")
        if window != self.window:
            raise ValueError(f"x window must be {self.window}, but received {window}.")
        if channels != self.in_channels:
            raise ValueError(f"x channels must be {self.in_channels}, but received {channels}.")
        if ls is None or not isinstance(ls, torch.Tensor) or ls.ndim != 2:
            raise ValueError("ls must be a tensor with shape [batch, window].")
        if tuple(ls.shape) != (batch, self.window):
            raise ValueError(f"ls shape must be ({batch}, {self.window}).")
        if not torch.is_floating_point(ls):
            raise ValueError("ls dtype must be floating point.")
        if ls.device != x.device:
            raise ValueError("x and ls device must match.")
        if not torch.isfinite(ls).all():
            raise ValueError("ls values must all be finite; NaN and Inf are invalid.")
        if (
            topography is None
            or not isinstance(topography, torch.Tensor)
            or topography.ndim != 4
        ):
            raise ValueError("topography must have shape [batch, 1, height, width].")
        if tuple(topography.shape) != (batch, 1, height, width):
            raise ValueError(f"topography shape must be ({batch}, 1, {height}, {width}).")
        if not torch.is_floating_point(topography):
            raise ValueError("topography dtype must be floating point.")
        if topography.device != x.device:
            raise ValueError("x and topography device must match.")
        if not torch.isfinite(topography).all():
            raise ValueError("topography values must all be finite; NaN and Inf are invalid.")

    def forward(self, x, ls, topography):
        self._validate_inputs(x, ls, topography)
        if x.is_cuda and x.shape[0] * self.window >= 256:
            # All normalization is per sample. Keep the optimizer batch intact
            # while bounding each replay's full-resolution gate intermediates.
            chunk_size = max(1, 128 // self.window)
            predictions = []
            for start in range(0, x.shape[0], chunk_size):
                stop = start + chunk_size
                args = (x[start:stop], ls[start:stop], topography[start:stop])
                if self.training and torch.is_grad_enabled():
                    prediction = checkpoint(self._forward_impl, *args, use_reentrant=False)
                else:
                    prediction = self._forward_impl(*args)
                predictions.append(prediction)
            return torch.cat(predictions, dim=0)
        return self._forward_impl(x, ls, topography)

    def _forward_impl(self, x, ls, topography):
        batch, window, _, height, width = x.shape
        context = self.shared_gate.prepare_context(ls, topography)
        x, _, _ = self.shared_gate(x, ls, topography, context=context)
        # Retain both gate calls formerly separated by ConvLSTM.
        x, _, _ = self.shared_gate(x, ls, topography, context=context)
        encoded = self.spatial_encoder(
            x.reshape(batch * window, self.in_channels, height, width)
        )
        encoded_height, encoded_width = encoded.shape[-2:]
        encoded = encoded.reshape(
            batch, window, self.spatial_hidden_dim, encoded_height, encoded_width,
        )
        encoded, _, _ = self.shared_gate(encoded, ls, topography, context=context)
        encoded = encoded.reshape(
            batch, window * self.spatial_hidden_dim, encoded_height, encoded_width,
        )
        translated = self.temporal_translator(encoded, ls, topography, self.shared_gate, context=context)
        translated = translated.reshape(
            batch, self.horizon, self.spatial_hidden_dim, encoded_height, encoded_width,
        )
        translated, _, _ = self.shared_gate(translated, ls, topography, pool_ls=True, context=context)
        decoded = self.spatial_decoder(
            translated.reshape(
                batch * self.horizon, self.spatial_hidden_dim, encoded_height, encoded_width,
            ),
            (height, width),
        )
        prediction = decoded.reshape(batch, self.horizon, 1, height, width)
        return prediction


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

    return EarthThreeHour5DegreeGatedSimVPNoConvLSTMT11Control(
        in_channels=_positive_int(values, "in_channels"),
        window=_positive_int(values, "window"),
        horizon=_positive_int(values, "horizon"),
        spatial_hidden_dim=values["spatial_hidden_dim"],
        temporal_hidden_dim=values["temporal_hidden_dim"],
        num_temporal_blocks=values["num_temporal_blocks"],
        dropout=float(values["dropout"]),
        gate_hidden_dim=values["gate_hidden_dim"],
        initial_gate_strength=float(values["initial_gate_strength"]),
    )
