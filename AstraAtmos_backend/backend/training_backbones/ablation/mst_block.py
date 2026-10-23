"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    """基础 2D 卷积块，支持普通卷积和反卷积上采样。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, transpose=False, activate=True):
        super().__init__()
        padding = kernel_size // 2
        if transpose:
            if stride == 2:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            else:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activate = activate
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activate:
            x = self.act(x)
        return x

class InceptionTemporalBlock(nn.Module):
    """多尺度 latent translator 块，保留 SimVP 最有效的 3/5/7 空间尺度。"""

    def __init__(self, channels, hidden_channels, dropout):
        super().__init__()
        self.pre = BasicConv2d(channels, hidden_channels, kernel_size=1, stride=1, transpose=False, activate=True)
        self.branch3 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, transpose=False, activate=True)
        self.branch5 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, transpose=False, activate=True)
        self.branch7 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=7, stride=1, transpose=False, activate=True)
        self.proj = nn.Sequential(nn.Conv2d(hidden_channels * 3, channels, kernel_size=1, stride=1, padding=0), nn.GroupNorm(1, channels), nn.Dropout(dropout))
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.pre(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.proj(x)
        return self.act(x + residual)

class SpatialEncoder(nn.Module):
    """把每个历史时间步编码到低分辨率 latent grid。"""

    def __init__(self, in_channels, hidden_channels, num_downsample):
        super().__init__()
        self.proj = BasicConv2d(in_channels, hidden_channels, kernel_size=3, stride=1, transpose=False, activate=True)
        self.down_blocks = nn.ModuleList([BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, transpose=False, activate=True) for _ in range(num_downsample)])

    def forward(self, x):
        x = self.proj(x)
        for block in self.down_blocks:
            x = block(x)
        return x

class LatentTranslator(nn.Module):
    """把历史 latent sequence 翻译成未来 latent sequence。"""

    def __init__(self, seq_len, pred_len, spatial_hidden_dim, temporal_hidden_dim, temporal_depth, dropout):
        super().__init__()
        self.pred_len = pred_len
        self.spatial_hidden_dim = spatial_hidden_dim
        in_channels = seq_len * spatial_hidden_dim
        self.in_proj = BasicConv2d(in_channels, temporal_hidden_dim, kernel_size=1, stride=1, transpose=False, activate=True)
        self.blocks = nn.ModuleList([InceptionTemporalBlock(channels=temporal_hidden_dim, hidden_channels=temporal_hidden_dim, dropout=dropout) for _ in range(temporal_depth)])
        self.out_proj = nn.Sequential(nn.Conv2d(temporal_hidden_dim, pred_len * spatial_hidden_dim, kernel_size=1, stride=1, padding=0), nn.GroupNorm(1, pred_len * spatial_hidden_dim))

    def forward(self, z):
        (batch_size, seq_len, hidden_dim, height, width) = z.shape
        x = z.reshape(batch_size, seq_len * hidden_dim, height, width)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)
        return x.view(batch_size, self.pred_len, self.spatial_hidden_dim, height, width)

class SpatialDecoder(nn.Module):
    """把未来 latent grid 解码到目标网格分辨率。"""

    def __init__(self, hidden_channels, out_channels, num_upsample):
        super().__init__()
        self.up_blocks = nn.ModuleList([BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, transpose=True, activate=True) for _ in range(num_upsample)])
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, out_height, out_width):
        for block in self.up_blocks:
            x = block(x)
        x = self.head(x)
        if x.shape[-2] != out_height or x.shape[-1] != out_width:
            x = F.interpolate(x, size=(out_height, out_width), mode='bilinear', align_corners=False)
        return x

class MSTBlock(nn.Module):
    """
    即插即用的多尺度隐空间时序翻译插件。

    参数：
    - return_residual_gate=False: 直接返回 MST 预测；
    - return_residual_gate=True: 需要传入 baseline，并返回 gate 融合结果。

    输入：
    - x: [B, T, C, H, W]
    - baseline: [B, pred_len, out_channels, H, W]，仅 residual gate 模式需要。

    输出：
    - [B, pred_len, out_channels, H, W]
    """

    def __init__(self, seq_len, pred_len, in_channels, out_channels, spatial_hidden_dim=32, temporal_hidden_dim=128, num_downsample=2, temporal_depth=4, dropout=0.1, return_residual_gate=False, initial_baseline_weight=0.5):
        super().__init__()
        if not 0.0 < initial_baseline_weight < 1.0:
            raise ValueError('initial_baseline_weight must be between 0 and 1.')
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_residual_gate = return_residual_gate
        self.encoder = SpatialEncoder(in_channels=in_channels, hidden_channels=spatial_hidden_dim, num_downsample=num_downsample)
        self.translator = LatentTranslator(seq_len=seq_len, pred_len=pred_len, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, temporal_depth=temporal_depth, dropout=dropout)
        self.decoder = SpatialDecoder(hidden_channels=spatial_hidden_dim, out_channels=out_channels, num_upsample=num_downsample)
        if return_residual_gate:
            logit_value = torch.logit(torch.tensor(float(initial_baseline_weight)))
            self.residual_logit = nn.Parameter(logit_value)
        else:
            self.register_parameter('residual_logit', None)

    def forward(self, x, baseline=None):
        (batch_size, seq_len, channels, height, width) = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {seq_len}')
        if channels != self.in_channels:
            raise ValueError(f'Expected in_channels={self.in_channels}, got {channels}')
        z = x.reshape(batch_size * seq_len, channels, height, width)
        z = self.encoder(z)
        (_, hidden_dim, hid_h, hid_w) = z.shape
        z = z.view(batch_size, seq_len, hidden_dim, hid_h, hid_w)
        future_z = self.translator(z)
        future_z = future_z.reshape(batch_size * self.pred_len, hidden_dim, hid_h, hid_w)
        mst_out = self.decoder(future_z, out_height=height, out_width=width)
        mst_out = mst_out.view(batch_size, self.pred_len, self.out_channels, height, width)
        if not self.return_residual_gate:
            return mst_out
        if baseline is None:
            raise ValueError('baseline is required when return_residual_gate=True.')
        if baseline.shape != mst_out.shape:
            raise ValueError(f'Expected baseline shape {tuple(mst_out.shape)}, got {tuple(baseline.shape)}')
        baseline_weight = torch.sigmoid(self.residual_logit)
        return baseline_weight * baseline + (1.0 - baseline_weight) * mst_out
