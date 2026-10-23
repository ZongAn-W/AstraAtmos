"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import torch
import torch.nn as nn

class PhaseWarpFrontEnd(nn.Module):

    def __init__(self, spatial_shape=()):
        super().__init__()
        self.spatial_shape = tuple(spatial_shape)
        param_shape = (1, 1, 1, *self.spatial_shape)
        self.w1_o3 = nn.Parameter(torch.zeros(param_shape))
        self.b1_o3 = nn.Parameter(torch.zeros(param_shape))
        self.k1_o3 = nn.Parameter(torch.zeros(param_shape))
        self.w1_u = nn.Parameter(torch.ones(param_shape))
        self.w2_u = nn.Parameter(torch.ones(param_shape))
        self.b1_u = nn.Parameter(torch.zeros(param_shape))
        self.b2_u = nn.Parameter(torch.zeros(param_shape))
        self.k1_u = nn.Parameter(torch.zeros(param_shape))
        self.k2_u = nn.Parameter(torch.zeros(param_shape))
        self.w1_v = nn.Parameter(torch.ones(param_shape))
        self.w2_v = nn.Parameter(torch.ones(param_shape))
        self.b1_v = nn.Parameter(torch.zeros(param_shape))
        self.b2_v = nn.Parameter(torch.zeros(param_shape))
        self.k1_v = nn.Parameter(torch.zeros(param_shape))
        self.k2_v = nn.Parameter(torch.zeros(param_shape))
        self.w1_t = nn.Parameter(torch.ones(param_shape))
        self.w2_t = nn.Parameter(torch.ones(param_shape))
        self.b1_t = nn.Parameter(torch.zeros(param_shape))
        self.b2_t = nn.Parameter(torch.zeros(param_shape))
        self.k1_t = nn.Parameter(torch.zeros(param_shape))
        self.k2_t = nn.Parameter(torch.zeros(param_shape))
        self.w1_f = nn.Parameter(torch.ones(param_shape))
        self.w2_f = nn.Parameter(torch.ones(param_shape))
        self.b1_f = nn.Parameter(torch.zeros(param_shape))
        self.b2_f = nn.Parameter(torch.zeros(param_shape))
        self.k1_f = nn.Parameter(torch.zeros(param_shape))
        self.k2_f = nn.Parameter(torch.zeros(param_shape))

    def _ls_to_radians(self, ls):
        if ls.dim() != 2:
            raise ValueError(f'Expected ls with shape [B, T], got {tuple(ls.shape)}')
        extra_dims = [1] * len(self.spatial_shape)
        return (ls * (torch.pi / 180.0)).view(ls.size(0), ls.size(1), 1, *extra_dims)

    def _modulate_pair(self, x_var, ls_rad, w_sin, w_cos, b_sin, b_cos, k_sin, k_cos):
        k_safe_sin = torch.tanh(k_sin)
        k_safe_cos = torch.tanh(k_cos)
        warp_sin = k_safe_sin * torch.cos(ls_rad)
        warp_cos = k_safe_cos * torch.sin(ls_rad)
        x_sin = x_var * (w_sin * torch.sin(ls_rad + warp_sin + b_sin))
        x_cos = x_var * (w_cos * torch.cos(ls_rad + warp_cos + b_cos))
        return (x_sin, x_cos)

    def fuse_o3(self, o3, ls):
        ls_rad = self._ls_to_radians(ls) if ls.dim() == 2 else ls
        k_safe_o3 = torch.tanh(self.k1_o3)
        warp_o3 = k_safe_o3 * torch.cos(ls_rad)
        return o3 + self.w1_o3 * torch.sin(ls_rad + warp_o3 + self.b1_o3)

    def forward(self, x, ls):
        if x.size(2) != 5:
            raise ValueError(f'Expected 5 raw input channels, got {x.size(2)}')
        ls_rad = self._ls_to_radians(ls)
        o3 = x[:, :, 0:1, ...]
        u = x[:, :, 1:2, ...]
        v = x[:, :, 2:3, ...]
        temp = x[:, :, 3:4, ...]
        flux = x[:, :, 4:5, ...]
        o3_fused = self.fuse_o3(o3, ls_rad)
        (u_sin, u_cos) = self._modulate_pair(u, ls_rad, self.w1_u, self.w2_u, self.b1_u, self.b2_u, self.k1_u, self.k2_u)
        (v_sin, v_cos) = self._modulate_pair(v, ls_rad, self.w1_v, self.w2_v, self.b1_v, self.b2_v, self.k1_v, self.k2_v)
        (t_sin, t_cos) = self._modulate_pair(temp, ls_rad, self.w1_t, self.w2_t, self.b1_t, self.b2_t, self.k1_t, self.k2_t)
        (f_sin, f_cos) = self._modulate_pair(flux, ls_rad, self.w1_f, self.w2_f, self.b1_f, self.b2_f, self.k1_f, self.k2_f)
        return torch.cat([o3_fused, u_sin, u_cos, v_sin, v_cos, t_sin, t_cos, f_sin, f_cos], dim=2)

    def get_plot_configs(self):
        return [('O3', self.w1_o3, self.b1_o3, self.k1_o3), ('U_Wind_Sin', self.w1_u, self.b1_u, self.k1_u), ('U_Wind_Cos', self.w2_u, self.b2_u, self.k2_u), ('V_Wind_Sin', self.w1_v, self.b1_v, self.k1_v), ('V_Wind_Cos', self.w2_v, self.b2_v, self.k2_v), ('Temperature_Sin', self.w1_t, self.b1_t, self.k1_t), ('Temperature_Cos', self.w2_t, self.b2_t, self.k2_t), ('Solar_Flux_Sin', self.w1_f, self.b1_f, self.k1_f), ('Solar_Flux_Cos', self.w2_f, self.b2_f, self.k2_f)]
