"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import os
import sys
import torch
import torch.nn as nn
from .phase_warp_frontend import PhaseWarpFrontEnd
from .seed_utils import set_experiment_seed
from .rnn_cnn_rnn_phasewarp_compare import EarlyStopping, Logger, build_grid_dataloaders, evaluate_metrics, load_aligned_cube

class ConvBlock(nn.Module):
    """轻量卷积块，用于空间编码、空间融合和空间解码。"""

    def __init__(self, in_channels, out_channels, dropout, activate=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.GroupNorm(1, out_channels)]
        if activate:
            layers.extend([nn.GELU(), nn.Dropout(dropout)])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SpatialStack(nn.Module):
    """保持网格分辨率不变的卷积堆叠。"""

    def __init__(self, in_channels, hidden_channels, depth, dropout):
        super().__init__()
        if depth < 1:
            raise ValueError('cnn_depth must be at least 1.')
        blocks = [ConvBlock(in_channels, hidden_channels, dropout)]
        for _ in range(depth - 1):
            blocks.append(ConvBlock(hidden_channels, hidden_channels, dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class SpatialDecoder(nn.Module):
    """把未来隐状态解码成单通道臭氧场。"""

    def __init__(self, in_channels, hidden_channels, depth, dropout):
        super().__init__()
        if depth < 1:
            raise ValueError('cnn_depth must be at least 1.')
        blocks = [ConvBlock(in_channels, hidden_channels, dropout)]
        for _ in range(depth - 1):
            blocks.append(ConvBlock(hidden_channels, hidden_channels, dropout))
        blocks.append(nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class CNNRNNCNNRNNCNNForecaster(nn.Module):
    """
    空间优先的 CNN-RNN-CNN-RNN-CNN 臭氧预测模型。

    输入：
    - x: [B, T, 5, H, W]
    - ls: [B, T]

    输出：
    - [B, pred_len, H, W]
    """

    def __init__(self, seq_len, pred_len, lat_size, lon_size, use_phase_warp, spatial_hidden_dim, temporal_hidden_dim, cnn_depth, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.use_phase_warp = use_phase_warp
        self.spatial_hidden_dim = spatial_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        if use_phase_warp:
            self.phase_warp = PhaseWarpFrontEnd(spatial_shape=(lat_size, lon_size))
            input_dim = 9
        else:
            self.phase_warp = None
            input_dim = 5
        self.spatial_encoder = SpatialStack(in_channels=input_dim, hidden_channels=spatial_hidden_dim, depth=cnn_depth, dropout=dropout)
        self.temporal_encoder = nn.GRU(input_size=spatial_hidden_dim, hidden_size=temporal_hidden_dim, batch_first=True)
        self.spatial_fusion = SpatialStack(in_channels=temporal_hidden_dim, hidden_channels=temporal_hidden_dim, depth=cnn_depth, dropout=dropout)
        self.temporal_decoder = nn.GRU(input_size=temporal_hidden_dim, hidden_size=temporal_hidden_dim, batch_first=True)
        self.decoder_token = nn.Parameter(torch.zeros(1, 1, temporal_hidden_dim))
        self.spatial_decoder = SpatialDecoder(in_channels=temporal_hidden_dim, hidden_channels=spatial_hidden_dim, depth=cnn_depth, dropout=dropout)

    def forward(self, x, ls):
        if self.phase_warp is not None:
            features = self.phase_warp(x, ls)
        else:
            features = x
        (batch_size, seq_len, channels, height, width) = features.shape
        if seq_len != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {seq_len}')
        frames = features.reshape(batch_size * seq_len, channels, height, width)
        encoded_frames = self.spatial_encoder(frames)
        encoded_frames = encoded_frames.view(batch_size, seq_len, self.spatial_hidden_dim, height, width)
        point_sequences = encoded_frames.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, seq_len, self.spatial_hidden_dim)
        (_, encoder_hidden) = self.temporal_encoder(point_sequences)
        point_hidden = encoder_hidden[-1]
        hidden_grid = point_hidden.view(batch_size, height, width, self.temporal_hidden_dim).permute(0, 3, 1, 2)
        fused_grid = self.spatial_fusion(hidden_grid)
        decoder_h0 = fused_grid.permute(0, 2, 3, 1).reshape(batch_size * height * width, self.temporal_hidden_dim)
        decoder_h0 = decoder_h0.unsqueeze(0).contiguous()
        decoder_input = self.decoder_token.expand(batch_size * height * width, self.pred_len, -1)
        (decoded, _) = self.temporal_decoder(decoder_input, decoder_h0)
        future_grid = decoded.view(batch_size, height, width, self.pred_len, self.temporal_hidden_dim)
        future_grid = future_grid.permute(0, 3, 4, 1, 2).reshape(batch_size * self.pred_len, self.temporal_hidden_dim, height, width)
        out = self.spatial_decoder(future_grid)
        return out.view(batch_size, self.pred_len, height, width)

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, horizon, window, spatial_hidden_dim, temporal_hidden_dim, cnn_depth, dropout, epochs, learning_rate, early_stopping_patience, base_dir):
    """训练一组 CNN-RNN-CNN-RNN-CNN 实验，并返回评估指标。"""
    print(f'\n[Experiment] {label}')
    model = CNNRNNCNNRNNCNNForecaster(seq_len=window, pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, cnn_depth=cnn_depth, dropout=dropout).to(device)
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
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'CNN_RNN_CNN_RNN_CNN_PhaseWarp_Compare.txt'))
    seed = 11
    set_experiment_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    print(f'Single Seed: {seed}')
    window = 3
    horizon = 3
    batch_size = 4
    spatial_hidden_dim = 32
    temporal_hidden_dim = 64
    cnn_depth = 2
    dropout = 0.1
    epochs = 20
    learning_rate = 0.001
    early_stopping_patience = 5
    (x_raw, y_raw, ls_raw) = load_aligned_cube(base_dir)
    (lat_size, lon_size) = (y_raw.shape[1], y_raw.shape[2])
    (train_loader, test_loader, y_mean, y_std) = build_grid_dataloaders(x_raw=x_raw, y_raw=y_raw, ls_raw=ls_raw, window=window, horizon=horizon, batch_size=batch_size)
    raw_metrics = train_and_evaluate(label=f'CNN_RNN_CNN_RNN_CNN_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, window=window, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, cnn_depth=cnn_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
    set_experiment_seed(seed)
    phase_metrics = train_and_evaluate(label=f'CNN_RNN_CNN_RNN_CNN_PhaseWarp_seed{seed}', use_phase_warp=True, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, window=window, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, cnn_depth=cnn_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
    print('\n[Comparison Summary]')
    print(f"CNN_RNN_CNN_RNN_CNN_Raw Metrics | RMSE: {raw_metrics['rmse']:.4f} | MAE: {raw_metrics['mae']:.4f} | R^2: {raw_metrics['r2']:.4f} | SMAPE: {raw_metrics['smape']:.2%}")
    print(f"CNN_RNN_CNN_RNN_CNN_PhaseWarp Metrics | RMSE: {phase_metrics['rmse']:.4f} | MAE: {phase_metrics['mae']:.4f} | R^2: {phase_metrics['r2']:.4f} | SMAPE: {phase_metrics['smape']:.2%}")
    print(f"RMSE improvement: {raw_metrics['rmse'] - phase_metrics['rmse']:.4f}")
    print(f"MAE improvement : {raw_metrics['mae'] - phase_metrics['mae']:.4f}")
    print(f"R^2 gain        : {phase_metrics['r2'] - raw_metrics['r2']:.4f}")
    print(f"SMAPE gain      : {raw_metrics['smape'] - phase_metrics['smape']:.2%}")
