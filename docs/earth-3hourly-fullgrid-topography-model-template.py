"""Earth v2 full-grid model consuming solar longitude and ETOPO orography."""

import torch
from torch import nn

MODEL_SPEC = {
    "name": "EarthThreeHourSeasonTopographyLinear",
    "description": "Whole-grid TO3 forecast conditioned on season and physical orography",
    "parameters": {},
    "derived_inputs": {
        "ls": {"required": True, "shape": ["batch", "window"],
               "dtype": "float32", "unit": "degree"},
        "topography": {"required": True, "shape": ["batch", 1, "height", "width"],
                       "dtype": "float32", "unit": "meter"},
    },
    "datasets": {
        "earth_merra2_3hourly_v1": {
            "schema": "astraatmos_earth_3hourly_fullgrid_model_v2",
            "frequency_hours": 3, "step_unit": "hour", "step": 3, "time_zone": "UTC",
            "window": list(range(1, 241)), "horizon": list(range(1, 241)), "grid": [[240, 480]],
            "target": "TO3", "target_unit": "DU",
            "input_channels": ["TO3", "U10M", "V10M", "T2M", "SWGDN"],
            "input_units": ["DU", "m s-1", "m s-1", "K", "W m-2"],
            "dtype": "float32", "tensor_layout": "BTCHW",
            "auxiliary_inputs": ["U10M", "V10M", "T2M", "SWGDN"],
            "spatial_tile_shape": [240, 480], "output_channels": ["TO3"], "output_units": ["DU"],
        },
    },
}


class EarthSeasonTopographyLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mix = nn.Conv2d(config["in_channels"], 1, 1)
        self.terrain_scale = nn.Parameter(torch.tensor(0.1))
        self.season_scale = nn.Parameter(torch.tensor(0.1))
        self.temporal = nn.Linear(config["window"], config["horizon"])

    def forward(self, x, ls, topography):
        batch, time, channels, height, width = x.shape
        mixed = self.mix(x.reshape(batch * time, channels, height, width))
        mixed = mixed.reshape(batch, time, height, width)
        mixed = mixed + self.terrain_scale * topography[:, 0, None] / 1000.0
        mixed = mixed + self.season_scale * torch.sin(ls[:, :, None, None] * (torch.pi / 180.0))
        return self.temporal(mixed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).unsqueeze(2)


def build_model(config):
    return EarthSeasonTopographyLinear(config)
