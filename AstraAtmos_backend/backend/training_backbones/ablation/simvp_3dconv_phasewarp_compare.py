"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .phase_warp_frontend import PhaseWarpFrontEnd
from .seed_utils import set_experiment_seed
from .rnn_cnn_rnn_phasewarp_compare import EarlyStopping, Logger, build_grid_dataloaders, evaluate_metrics, load_aligned_cube

class BasicConv2d(nn.Module):
    """SimVP 风格基础 2D 卷积模块。"""

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

class BasicConv3d(nn.Module):
    """用于 latent translator 的 3D 时空卷积模块。"""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), dropout=0.0):
        super().__init__()
        padding = tuple((k // 2 for k in kernel_size))
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding), nn.GroupNorm(1, out_channels), nn.GELU(), nn.Dropout3d(dropout))

    def forward(self, x):
        return self.block(x)

class Residual3DConvBlock(nn.Module):
    """在低分辨率隐空间上做局部时间-空间联合建模。"""

    def __init__(self, channels, hidden_channels, dropout):
        super().__init__()
        self.pre = BasicConv3d(channels, hidden_channels, kernel_size=(1, 1, 1), dropout=dropout)
        self.conv = BasicConv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 3), dropout=dropout)
        self.proj = nn.Sequential(nn.Conv3d(hidden_channels, channels, kernel_size=(1, 1, 1), padding=0), nn.GroupNorm(1, channels), nn.Dropout3d(dropout))
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.pre(x)
        x = self.conv(x)
        x = self.proj(x)
        return self.act(x + residual)

class SpatialEncoder(nn.Module):
    """SimVP 空间编码器，逐帧把原始网格压缩到低分辨率隐空间。"""

    def __init__(self, in_channels, hid_S, num_downsample):
        super().__init__()
        self.proj = BasicConv2d(in_channels, hid_S, kernel_size=3, stride=1, transpose=False, activate=True)
        self.down_blocks = nn.ModuleList([BasicConv2d(hid_S, hid_S, kernel_size=3, stride=2, transpose=False, activate=True) for _ in range(num_downsample)])

    def forward(self, x):
        x = self.proj(x)
        for block in self.down_blocks:
            x = block(x)
        return x

class Spatiotemporal3DTranslator(nn.Module):
    """
    用 3D 卷积替换 SimVP 原来的 2D temporal translator。

    输入：
    - [B, seq_len, hid_S, H', W']

    输出：
    - [B, pred_len, hid_S, H', W']
    """

    def __init__(self, seq_len, pred_len, hid_S, hid_T, temporal_depth, dropout):
        super().__init__()
        self.pred_len = pred_len
        self.hid_S = hid_S
        self.in_proj = BasicConv3d(hid_S, hid_T, kernel_size=(1, 1, 1), dropout=dropout)
        self.blocks = nn.ModuleList([Residual3DConvBlock(channels=hid_T, hidden_channels=hid_T, dropout=dropout) for _ in range(temporal_depth)])
        self.time_pool = nn.Conv3d(hid_T, hid_T, kernel_size=(seq_len, 1, 1), padding=0)
        self.out_proj = nn.Sequential(nn.Conv2d(hid_T, pred_len * hid_S, kernel_size=1, stride=1, padding=0), nn.GroupNorm(1, pred_len * hid_S))

    def forward(self, z):
        (batch_size, seq_len, hid_S, height, width) = z.shape
        x = z.permute(0, 2, 1, 3, 4).contiguous()
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.time_pool(x).squeeze(2)
        x = self.out_proj(x)
        return x.view(batch_size, self.pred_len, self.hid_S, height, width)

class SpatialDecoder(nn.Module):
    """SimVP 空间解码器，把未来隐状态还原到原始网格大小。"""

    def __init__(self, hid_S, num_upsample):
        super().__init__()
        self.up_blocks = nn.ModuleList([BasicConv2d(hid_S, hid_S, kernel_size=3, stride=2, transpose=True, activate=True) for _ in range(num_upsample)])
        self.head = nn.Conv2d(hid_S, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, out_height, out_width):
        for block in self.up_blocks:
            x = block(x)
        x = self.head(x)
        if x.shape[-2] != out_height or x.shape[-1] != out_width:
            x = F.interpolate(x, size=(out_height, out_width), mode='bilinear', align_corners=False)
        return x

class SimVP3DConvForecaster(nn.Module):
    """
    SimVP-3DConv 臭氧预测模型。

    与原 SimVP 风格脚本的差别：
    - 空间编码器不变；
    - 空间解码器不变；
    - 中间 translator 从 2D Inception translator 换成 3D spatiotemporal translator。
    """

    def __init__(self, seq_len, pred_len, lat_size, lon_size, use_phase_warp, spatial_hidden_dim, temporal_hidden_dim, num_downsample, temporal_depth, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.use_phase_warp = use_phase_warp
        if use_phase_warp:
            self.phase_warp = PhaseWarpFrontEnd(spatial_shape=(lat_size, lon_size))
            input_dim = 9
        else:
            self.phase_warp = None
            input_dim = 5
        self.encoder = SpatialEncoder(in_channels=input_dim, hid_S=spatial_hidden_dim, num_downsample=num_downsample)
        self.translator = Spatiotemporal3DTranslator(seq_len=seq_len, pred_len=pred_len, hid_S=spatial_hidden_dim, hid_T=temporal_hidden_dim, temporal_depth=temporal_depth, dropout=dropout)
        self.decoder = SpatialDecoder(hid_S=spatial_hidden_dim, num_upsample=num_downsample)

    def forward(self, x, ls):
        if self.phase_warp is not None:
            features = self.phase_warp(x, ls)
        else:
            features = x
        (batch_size, seq_len, channels, height, width) = features.shape
        if seq_len != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {seq_len}')
        z = features.reshape(batch_size * seq_len, channels, height, width)
        z = self.encoder(z)
        (_, hid_S, hid_h, hid_w) = z.shape
        z = z.view(batch_size, seq_len, hid_S, hid_h, hid_w)
        future_z = self.translator(z)
        future_z = future_z.reshape(batch_size * self.pred_len, hid_S, hid_h, hid_w)
        out = self.decoder(future_z, out_height=height, out_width=width)
        return out.view(batch_size, self.pred_len, height, width)

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, horizon, window, spatial_hidden_dim, temporal_hidden_dim, num_downsample, temporal_depth, dropout, epochs, learning_rate, early_stopping_patience, base_dir):
    """训练一组 SimVP-3DConv 实验，并返回评估指标。"""
    print(f'\n[Experiment] {label}')
    model = SimVP3DConvForecaster(seq_len=window, pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, num_downsample=num_downsample, temporal_depth=temporal_depth, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    checkpoint_path = os.path.join(base_dir, 'models', 'training_results', f'{label.lower()}_checkpoint.pth')
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint_path)
    for epoch_idx in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for (xb, lsb, yb) in train_loader:
            xb = xb.to(device)
            lsb = lsb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, lsb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        avg_train_loss = train_loss_sum / len(train_loader)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for (xb, lsb, yb) in test_loader:
                xb = xb.to(device)
                lsb = lsb.to(device)
                yb = yb.to(device)
                pred = model(xb, lsb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item()
        avg_val_loss = val_loss_sum / len(test_loader)
        print(f'{label} | Epoch {epoch_idx + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f'{label} triggered early stopping.')
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metrics = evaluate_metrics(model, test_loader, device, y_std, y_mean)
    save_path = os.path.join(base_dir, 'models', 'training_results', f'{label.lower()}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'{label} weights saved to: {save_path}')
    print(f"{label} Metrics | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R^2: {metrics['r2']:.4f} | SMAPE: {metrics['smape']:.2%}")
    return metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(base_dir, 'models', 'training_logs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models', 'training_results'), exist_ok=True)
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'SimVP_3DConv_PhaseWarp_Compare.txt'))
    seed = 11
    set_experiment_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    print(f'Single Seed: {seed}')
    window = 3
    horizon = 3
    batch_size = 4
    spatial_hidden_dim = 32
    temporal_hidden_dim = 128
    num_downsample = 2
    temporal_depth = 4
    dropout = 0.1
    epochs = 20
    learning_rate = 0.001
    early_stopping_patience = 5
    (x_raw, y_raw, ls_raw) = load_aligned_cube(base_dir)
    (lat_size, lon_size) = (y_raw.shape[1], y_raw.shape[2])
    (train_loader, test_loader, y_mean, y_std) = build_grid_dataloaders(x_raw=x_raw, y_raw=y_raw, ls_raw=ls_raw, window=window, horizon=horizon, batch_size=batch_size)
    raw_metrics = train_and_evaluate(label=f'SimVP_3DConv_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, window=window, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, num_downsample=num_downsample, temporal_depth=temporal_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
    set_experiment_seed(seed)
    phase_metrics = train_and_evaluate(label=f'SimVP_3DConv_PhaseWarp_seed{seed}', use_phase_warp=True, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, window=window, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, num_downsample=num_downsample, temporal_depth=temporal_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
    print('\n[Comparison Summary]')
    print(f"SimVP_3DConv_Raw Metrics | RMSE: {raw_metrics['rmse']:.4f} | MAE: {raw_metrics['mae']:.4f} | R^2: {raw_metrics['r2']:.4f} | SMAPE: {raw_metrics['smape']:.2%}")
    print(f"SimVP_3DConv_PhaseWarp Metrics | RMSE: {phase_metrics['rmse']:.4f} | MAE: {phase_metrics['mae']:.4f} | R^2: {phase_metrics['r2']:.4f} | SMAPE: {phase_metrics['smape']:.2%}")
    print(f"RMSE improvement: {raw_metrics['rmse'] - phase_metrics['rmse']:.4f}")
    print(f"MAE improvement : {raw_metrics['mae'] - phase_metrics['mae']:.4f}")
    print(f"R^2 gain        : {phase_metrics['r2'] - raw_metrics['r2']:.4f}")
    print(f"SMAPE gain      : {raw_metrics['smape'] - phase_metrics['smape']:.2%}")
