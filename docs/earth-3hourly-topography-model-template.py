"""Earth v1 tile model consuming physical ETOPO orography in metres."""

import torch
from torch import nn

MODEL_SPEC = {
    "name": "EarthThreeHourTopographyLinear",
    "description": "Per-sample tile orography conditions a temporal TO3 forecast",
    "parameters": {},
    "derived_inputs": {
        "topography": {"required": True, "shape": ["batch", 1, "height", "width"],
                       "dtype": "float32", "unit": "meter"},
    },
    "datasets": {
        "earth_merra2_3hourly_v1": {
            "schema": "astraatmos_earth_3hourly_uploaded_model_v1",
            "frequency_hours": 3, "step_unit": "hour", "step": 3, "time_zone": "UTC",
            "window": list(range(1, 241)), "horizon": list(range(1, 241)), "grid": [[240, 480]],
            "target": "TO3", "target_unit": "DU",
            "input_channels": ["TO3", "U10M", "V10M", "T2M", "SWGDN"],
            "input_units": ["DU", "m s-1", "m s-1", "K", "W m-2"],
            "dtype": "float32", "tensor_layout": "BTCHW",
            "auxiliary_inputs": ["U10M", "V10M", "T2M", "SWGDN"],
            "spatial_tile_shape": [24, 48], "output_channels": ["TO3"], "output_units": ["DU"],
        },
    },
}


class EarthTopographyLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mix = nn.Conv2d(config["in_channels"], 1, 1)
        self.terrain_scale = nn.Parameter(torch.tensor(0.1))
        self.temporal = nn.Linear(config["window"], config["horizon"])

    def forward(self, x, topography):
        batch, time, channels, height, width = x.shape
        mixed = self.mix(x.reshape(batch * time, channels, height, width))
        mixed = mixed.reshape(batch, time, height, width)
        # This model chooses its own explicit metre-to-kilometre scaling;
        # terrain is independent of the platform's meteorological statistics.
        mixed = mixed + self.terrain_scale * topography[:, 0, None] / 1000.0
        return self.temporal(mixed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).unsqueeze(2)


def build_model(config):
    return EarthTopographyLinear(config)
