"""Earth three-hour v1: normalized float32 BTCHW spatial tiles, CPU or CUDA."""

import torch
from torch import nn

MODEL_SPEC = {
    "name": "EarthThreeHourLinear",
    "description": "Channel mixing and temporal linear forecast for TO3",
    "parameters": {"bias": {"type": "bool", "default": True}},
    # Optional seasonal coordinate. Uncomment this block *and* change
    # ``def forward(self, x):`` to ``def forward(self, x, ls):`` to receive the
    # server-derived solar longitude of each input window. Leaving it out keeps
    # the single-argument call. See docs/earth-3hourly-uploaded-model.md.
    # For a model that actually consumes ETOPO orography, use the separate
    # earth-3hourly-topography-model-template.py. This template stays compatible
    # with deployments without a configured terrain asset.
    # "derived_inputs": {"ls": {"required": True, "shape": ["batch", "window"],
    #                           "dtype": "float32", "unit": "degree"}},
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


class EarthThreeHourLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mix = nn.Conv2d(config["in_channels"], 1, 1, bias=config["bias"])
        self.temporal = nn.Linear(config["window"], config["horizon"], bias=config["bias"])

    def forward(self, x):
        batch, time, channels, height, width = x.shape
        mixed = self.mix(x.reshape(batch * time, channels, height, width))
        mixed = mixed.reshape(batch, time, height, width).permute(0, 2, 3, 1)
        return self.temporal(mixed).permute(0, 3, 1, 2).unsqueeze(2)


def build_model(config):
    return EarthThreeHourLinear(config)
