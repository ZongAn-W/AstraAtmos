"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import os
import sys
import torch
import torch.nn as nn
from .mst_block import MSTBlock
from .phase_warp_frontend import PhaseWarpFrontEnd
from .rnn_cnn_rnn_phasewarp_compare import EarlyStopping, Logger, build_grid_dataloaders, evaluate_metrics, load_aligned_cube
from .seed_utils import set_experiment_seed

class ConvLSTMCell(nn.Module):
    """One ConvLSTM cell for a spatial grid at a single time step."""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        (i_gate, f_gate, o_gate, g_gate) = torch.chunk(gates, 4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)
        c_next = f_gate * c_cur + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        return (h_next, c_next)

class ConvLSTMMSTFeatureRefiner(nn.Module):
    """
    ConvLSTM encoder with MST inserted at the hidden-sequence level.

    Output shape:
    - [B, pred_len, H, W]
    """

    def __init__(self, seq_len, pred_len, lat_size, lon_size, use_phase_warp, hidden_dims, mst_spatial_hidden_dim, mst_temporal_hidden_dim, mst_num_downsample, mst_temporal_depth, dropout, kernel_size=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.use_phase_warp = use_phase_warp
        self.hidden_dims = list(hidden_dims)
        if use_phase_warp:
            self.phase_warp = PhaseWarpFrontEnd(spatial_shape=(lat_size, lon_size))
            input_dim = 9
        else:
            self.phase_warp = None
            input_dim = 5
        self.cells = nn.ModuleList()
        for (idx, hidden_dim) in enumerate(self.hidden_dims):
            cur_input_dim = input_dim if idx == 0 else self.hidden_dims[idx - 1]
            self.cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size=kernel_size))
        hidden_dim = self.hidden_dims[-1]
        self.mst_block = MSTBlock(seq_len=seq_len, pred_len=1, in_channels=hidden_dim, out_channels=hidden_dim, spatial_hidden_dim=mst_spatial_hidden_dim, temporal_hidden_dim=mst_temporal_hidden_dim, num_downsample=mst_num_downsample, temporal_depth=mst_temporal_depth, dropout=dropout, return_residual_gate=False)
        self.forecast_head = nn.Conv2d(hidden_dim, pred_len, kernel_size=1)

    def _init_states(self, batch_size, device):
        states = []
        for hidden_dim in self.hidden_dims:
            h_state = torch.zeros(batch_size, hidden_dim, self.lat_size, self.lon_size, device=device)
            c_state = torch.zeros_like(h_state)
            states.append([h_state, c_state])
        return states

    def _prepare_features(self, x, ls):
        if x.shape[1] != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {x.shape[1]}')
        if self.phase_warp is None:
            return x
        return self.phase_warp(x, ls)

    def encode_hidden_sequence(self, x, ls):
        """Return final-layer ConvLSTM hidden states for every input step."""
        features = self._prepare_features(x, ls)
        (batch_size, seq_len, _, _, _) = features.shape
        states = self._init_states(batch_size, features.device)
        hidden_sequence = []
        for t in range(seq_len):
            current = features[:, t]
            for (layer_idx, cell) in enumerate(self.cells):
                (h_cur, c_cur) = states[layer_idx]
                (h_next, c_next) = cell(current, h_cur, c_cur)
                states[layer_idx] = [h_next, c_next]
                current = h_next
            hidden_sequence.append(current)
        return torch.stack(hidden_sequence, dim=1)

    def refine_hidden_sequence(self, hidden_sequence):
        """Translate historical hidden states into one refined future hidden."""
        refined_hidden = self.mst_block(hidden_sequence)
        return refined_hidden.squeeze(1)

    def forward(self, x, ls):
        hidden_sequence = self.encode_hidden_sequence(x, ls)
        refined_hidden = self.refine_hidden_sequence(hidden_sequence)
        return self.forecast_head(refined_hidden)

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, window, horizon, hidden_dims, mst_spatial_hidden_dim, mst_temporal_hidden_dim, mst_num_downsample, mst_temporal_depth, dropout, epochs, learning_rate, early_stopping_patience, base_dir):
    """Train one ConvLSTM_MST_FeatureRefiner run and return metrics."""
    print(f'\n[Experiment] {label}')
    model = ConvLSTMMSTFeatureRefiner(seq_len=window, pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, hidden_dims=hidden_dims, mst_spatial_hidden_dim=mst_spatial_hidden_dim, mst_temporal_hidden_dim=mst_temporal_hidden_dim, mst_num_downsample=mst_num_downsample, mst_temporal_depth=mst_temporal_depth, dropout=dropout).to(device)
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
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'ConvLSTM_MST_FeatureRefiner.txt'))
    seed = 11
    set_experiment_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    print(f'Single Seed: {seed}')
    print('Model: ConvLSTM_MST_FeatureRefiner')
    window = 3
    horizon = 3
    batch_size = 4
    hidden_dim = 32
    num_layers = 2
    hidden_dims = [hidden_dim] * num_layers
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
    raw_metrics = train_and_evaluate(label=f'ConvLSTM_MSTRefiner_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, window=window, horizon=horizon, hidden_dims=hidden_dims, mst_spatial_hidden_dim=mst_spatial_hidden_dim, mst_temporal_hidden_dim=mst_temporal_hidden_dim, mst_num_downsample=mst_num_downsample, mst_temporal_depth=mst_temporal_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
    set_experiment_seed(seed)
    phase_metrics = train_and_evaluate(label=f'ConvLSTM_MSTRefiner_PhaseWarp_seed{seed}', use_phase_warp=True, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, window=window, horizon=horizon, hidden_dims=hidden_dims, mst_spatial_hidden_dim=mst_spatial_hidden_dim, mst_temporal_hidden_dim=mst_temporal_hidden_dim, mst_num_downsample=mst_num_downsample, mst_temporal_depth=mst_temporal_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
    print('\n[Comparison Summary]')
    print(f"ConvLSTM_MSTRefiner_Raw Metrics | RMSE: {raw_metrics['rmse']:.4f} | MAE: {raw_metrics['mae']:.4f} | R^2: {raw_metrics['r2']:.4f} | SMAPE: {raw_metrics['smape']:.2%}")
    print(f"ConvLSTM_MSTRefiner_PhaseWarp Metrics | RMSE: {phase_metrics['rmse']:.4f} | MAE: {phase_metrics['mae']:.4f} | R^2: {phase_metrics['r2']:.4f} | SMAPE: {phase_metrics['smape']:.2%}")
    print(f"RMSE improvement: {raw_metrics['rmse'] - phase_metrics['rmse']:.4f}")
    print(f"MAE improvement : {raw_metrics['mae'] - phase_metrics['mae']:.4f}")
    print(f"R^2 gain        : {phase_metrics['r2'] - raw_metrics['r2']:.4f}")
    print(f"SMAPE gain      : {raw_metrics['smape'] - phase_metrics['smape']:.2%}")
