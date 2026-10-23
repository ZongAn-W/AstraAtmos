"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import os
import sys
import torch
import torch.nn as nn
from .mst_block import MSTBlock
from .rnn_cnn_rnn_phasewarp_compare import EarlyStopping, Logger, build_grid_dataloaders, evaluate_metrics, load_aligned_cube
from .seed_utils import set_experiment_seed

class MovingAverage(nn.Module):
    """Moving average used by the DLinear-style decomposition."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.kernel_size <= 1:
            return x
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class SeriesDecomposition(nn.Module):
    """Split a sequence into seasonal residual and trend components."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return (seasonal, trend)

class LinearLayer(nn.Module):
    """Small configurable MLP used for seq_len -> pred_len projection."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_hidden_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class DLinearPointBackbone(nn.Module):
    """
    DLinear-style temporal projection for point-wise multivariate sequences.

    Input:  [N, seq_len, feature_dim]
    Output: [N, pred_len, feature_dim]
    """

    def __init__(self, seq_len, pred_len, feature_dim, linear_hidden_layers=2):
        super().__init__()
        kernel_size = seq_len if seq_len % 2 == 1 else max(1, seq_len - 1)
        hidden_dim = max(4, feature_dim * 2)
        self.decomposition = SeriesDecomposition(kernel_size)
        self.linear_seasonal = LinearLayer(seq_len, hidden_dim, pred_len, num_hidden_layers=linear_hidden_layers)
        self.linear_trend = LinearLayer(seq_len, hidden_dim, pred_len, num_hidden_layers=linear_hidden_layers)

    def forward(self, x):
        (seasonal_init, trend_init) = self.decomposition(x)
        seasonal_init = seasonal_init.transpose(1, 2)
        trend_init = trend_init.transpose(1, 2)
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        return (seasonal_output + trend_output).transpose(1, 2)

class GridPointDLinearBaseline(nn.Module):
    """Apply one DLinear point model to every grid point with shared weights."""

    def __init__(self, seq_len, pred_len, lat_size, lon_size, input_channels=5, linear_hidden_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.input_channels = input_channels
        self.backbone = DLinearPointBackbone(seq_len=seq_len, pred_len=pred_len, feature_dim=input_channels, linear_hidden_layers=linear_hidden_layers)
        self.target_head = nn.Linear(input_channels, 1)

    def forward(self, x):
        (batch_size, seq_len, channels, lat_size, lon_size) = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {seq_len}')
        if channels != self.input_channels:
            raise ValueError(f'Expected input_channels={self.input_channels}, got {channels}')
        x_point = x.permute(0, 3, 4, 1, 2).reshape(batch_size * lat_size * lon_size, seq_len, channels)
        pred_features = self.backbone(x_point)
        pred_o3 = self.target_head(pred_features).squeeze(-1)
        return pred_o3.view(batch_size, lat_size, lon_size, self.pred_len).permute(0, 3, 1, 2)

class DLinearMSTForecaster(nn.Module):
    """
    DLinear baseline plus an MST residual plugin.

    Input:
    - x:  [B, seq_len, 5, H, W]
    - ls: [B, seq_len], kept for API compatibility with other scripts.

    Output:
    - [B, pred_len, H, W]
    """

    def __init__(self, seq_len, pred_len, lat_size, lon_size, input_channels=5, linear_hidden_layers=2, mst_spatial_hidden_dim=32, mst_temporal_hidden_dim=128, mst_num_downsample=2, mst_temporal_depth=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.input_channels = input_channels
        self.dlinear = GridPointDLinearBaseline(seq_len=seq_len, pred_len=pred_len, lat_size=lat_size, lon_size=lon_size, input_channels=input_channels, linear_hidden_layers=linear_hidden_layers)
        self.mst_block = MSTBlock(seq_len=seq_len, pred_len=pred_len, in_channels=input_channels, out_channels=1, spatial_hidden_dim=mst_spatial_hidden_dim, temporal_hidden_dim=mst_temporal_hidden_dim, num_downsample=mst_num_downsample, temporal_depth=mst_temporal_depth, dropout=dropout, return_residual_gate=True, initial_baseline_weight=0.7)

    def forward(self, x, ls):
        _ = ls
        if x.shape[1] != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {x.shape[1]}')
        baseline = self.dlinear(x).unsqueeze(2)
        fused = self.mst_block(x, baseline=baseline)
        return fused.squeeze(2)

def train_and_evaluate(label, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, window, horizon, linear_hidden_layers, mst_spatial_hidden_dim, mst_temporal_hidden_dim, mst_num_downsample, mst_temporal_depth, dropout, epochs, learning_rate, early_stopping_patience, base_dir):
    """Train one DLinear_MST_Raw run and return restored-scale metrics."""
    print(f'\n[Experiment] {label}')
    model = DLinearMSTForecaster(seq_len=window, pred_len=horizon, lat_size=lat_size, lon_size=lon_size, input_channels=5, linear_hidden_layers=linear_hidden_layers, mst_spatial_hidden_dim=mst_spatial_hidden_dim, mst_temporal_hidden_dim=mst_temporal_hidden_dim, mst_num_downsample=mst_num_downsample, mst_temporal_depth=mst_temporal_depth, dropout=dropout).to(device)
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
    baseline_weight = torch.sigmoid(model.mst_block.residual_logit).item()
    save_path = os.path.join(base_dir, 'models', 'training_results', f'{label.lower()}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'{label} weights saved to: {save_path}')
    print(f'{label} learned DLinear baseline fusion weight: {baseline_weight:.4f}')
    print(f"{label} Metrics | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R^2: {metrics['r2']:.4f} | SMAPE: {metrics['smape']:.2%}")
    return metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(base_dir, 'models', 'training_logs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models', 'training_results'), exist_ok=True)
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'DLinear_MST_Raw.txt'))
    seed = 11
    set_experiment_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    print(f'Single Seed: {seed}')
    print('Model: DLinear_MST_Raw')
    window = 3
    horizon = 3
    batch_size = 4
    linear_hidden_layers = 2
    mst_spatial_hidden_dim = 32
    mst_temporal_hidden_dim = 128
    mst_num_downsample = 2
    mst_temporal_depth = 4
    dropout = 0.1
    epochs = 20
    learning_rate = 0.001
    early_stopping_patience = 5
    (x_raw, y_raw, ls_raw) = load_aligned_cube(base_dir)
    (lat_size, lon_size) = (y_raw.shape[1], y_raw.shape[2])
    (train_loader, test_loader, y_mean, y_std) = build_grid_dataloaders(x_raw=x_raw, y_raw=y_raw, ls_raw=ls_raw, window=window, horizon=horizon, batch_size=batch_size)
    metrics = train_and_evaluate(label=f'DLinear_MST_Raw_seed{seed}', train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, window=window, horizon=horizon, linear_hidden_layers=linear_hidden_layers, mst_spatial_hidden_dim=mst_spatial_hidden_dim, mst_temporal_hidden_dim=mst_temporal_hidden_dim, mst_num_downsample=mst_num_downsample, mst_temporal_depth=mst_temporal_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
    print('\n[Summary]')
    print(f"DLinear_MST_Raw Metrics | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R^2: {metrics['r2']:.4f} | SMAPE: {metrics['smape']:.2%}")
