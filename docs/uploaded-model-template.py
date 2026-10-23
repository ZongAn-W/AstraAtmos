import torch
from torch import nn


MODEL_SPEC = {
    "name": "ExampleUploadedModel",
    "description": "Minimal model definition for AstraAtmos uploaded-model training.",
    "parameters": {
        "hidden_dim": {"type": "int", "default": 16, "min": 4, "max": 128},
        "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
    },
}

# Earth MERRA-2 training requires an explicit opt-in. Without this block the model is
# treated as Mars-only: it still trains on Mars, but the platform refuses Earth
# training and says the model "does not declare the earth_merra2 dataset feed".
# "Valid for Mars" is never treated as "valid for Earth".
#
# Earth passes [batch, 7, channels, 36, 72] and expects [batch, 3, 1, 36, 72].
# Earth never supplies Ls or MOLA topography, so a model that declares either of
# them as required is rejected for Earth.
#
# MODEL_SPEC_WITH_EARTH = {
#     **MODEL_SPEC,
#     "datasets": {
#         "earth_merra2": {
#             "grid": [[36, 72]],
#             "window": [7],
#             "horizon": [3],
#             "auxiliary_inputs": [],
#             "target_leading_channels": 1,
#         }
#     },
# }

# Models that need historical solar longitude can opt in with this metadata:
# MODEL_SPEC_WITH_LS = {
#     **MODEL_SPEC,
#     "auxiliary_inputs": {
#         "ls": {
#             "required": True,
#             "shape": ["batch", "window"],
#             "dtype": "float32",
#             "unit": "degree",
#         }
#     },
# }
# The corresponding model method is:
# def forward(self, x, ls):
#     ...

# Models that need static MOLA elevation can opt in with this metadata:
# MODEL_SPEC_WITH_TOPOGRAPHY = {
#     **MODEL_SPEC,
#     "auxiliary_inputs": {
#         "topography": {
#             "required": True,
#             "shape": ["batch", 1, "height", "width"],
#             "dtype": "float32",
#             "unit": "meter",
#         }
#     },
# }
# topography is static and has no window/time axis. It is not part of
# config["in_channels"]. A terrain encoder may fuse it with the last frame:
# def forward(self, x, topography):
#     terrain_features = self.terrain_encoder(topography)
#     temporal_features = self.encoder(x[:, -1])
#     fused = temporal_features + terrain_features
#     return self.head(fused).unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)

# Models may declare both inputs. Positional order is always x, ls, topography:
# MODEL_SPEC_WITH_LS_AND_TOPOGRAPHY = {
#     **MODEL_SPEC,
#     "auxiliary_inputs": {
#         "ls": {
#             "required": True,
#             "shape": ["batch", "window"],
#             "dtype": "float32",
#             "unit": "degree",
#         },
#         "topography": {
#             "required": True,
#             "shape": ["batch", 1, "height", "width"],
#             "dtype": "float32",
#             "unit": "meter",
#         },
#     },
# }
# def forward(self, x, ls, topography):
#     terrain_features = self.terrain_encoder(topography)
#     temporal_features = self.encoder(x[:, -1])
#     phase_bias = self.phase_encoder(ls).unsqueeze(-1).unsqueeze(-1)
#     fused = temporal_features + terrain_features + phase_bias
#     return self.head(fused).unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)


class ExampleUploadedModel(nn.Module):
    def __init__(self, in_channels, horizon, hidden_dim, dropout):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        # x shape: [batch, window, channels, height, width]
        last_frame = x[:, -1]
        prediction = self.encoder(last_frame)
        return prediction.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)


def build_model(config):
    return ExampleUploadedModel(
        in_channels=config["in_channels"],
        horizon=config["horizon"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )
