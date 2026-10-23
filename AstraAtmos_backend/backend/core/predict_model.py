"""
PredRNNv2 模型定义
与训练脚本 demo3.py 完全一致的结构，确保权重加载正确。
"""

import torch
import torch.nn as nn


class SpatioTemporalLSTMCellv2(nn.Module):
    """PredRNNv2 的核心单元：时空 LSTM Cell"""

    def __init__(self, in_channel, num_hidden, height, width, filter_size):
        super().__init__()
        padding = filter_size // 2
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, filter_size, padding=padding)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, filter_size, padding=padding)
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, filter_size, padding=padding)
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, filter_size, padding=padding)
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, 1)
        self.num_hidden = num_hidden

    def forward(self, x, h, c, m):
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)
        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_concat, self.num_hidden, 1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, 1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, 1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + 1.0)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c + i_t * g_t

        i_tp = torch.sigmoid(i_xp + i_m)
        f_tp = torch.sigmoid(f_xp + f_m + 1.0)
        g_tp = torch.tanh(g_xp + g_m)
        m_new = f_tp * m + i_tp * g_tp

        mem = torch.cat([c_new, m_new], dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class PredRNNv2(nn.Module):
    """
    PredRNNv2 完整网络

    训练参数（与 predrnn_highlat_gpu.pth 权重匹配）：
    - input_dim=7（O3 + 6 个环境变量）
    - hidden_dims=[64, 64, 64]（3 层）
    - filter_size=3
    - height=36, width=72
    - horizon=3
    """

    def __init__(self, input_dim=7, hidden_dims=None, height=36, width=72, horizon=3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_ch = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(
                SpatioTemporalLSTMCellv2(in_ch, hidden_dims[i], height, width, 3)
            )
        self.conv_last = nn.Conv2d(hidden_dims[-1], 1, 1)
        self.horizon = horizon
        self.hidden_dims = hidden_dims

    def forward(self, x):
        """
        Args:
            x: (batch, window, channels, height, width)
               channels=7: [O3, u, v, ps, temp, dust, flux]

        Returns:
            preds: (batch, horizon, 1, height, width)
        """
        B, T, C, H, W = x.shape
        h = [torch.zeros(B, d, H, W, device=x.device) for d in self.hidden_dims]
        c = [torch.zeros_like(h[i]) for i in range(len(h))]
        m = torch.zeros_like(h[0])

        # Encoder：逐时间步输入
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]

        # Decoder：自回归预测（用最后一个完整输入，不用上一步预测）
        preds = []
        dec_inp = x[:, -1]  # 最后一个完整 7 通道输入
        for _ in range(self.horizon):
            inp = dec_inp
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]
            pred = self.conv_last(h[-1])
            preds.append(pred)

        return torch.stack(preds, dim=1)
