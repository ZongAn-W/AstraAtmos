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

def _initial_logit(probability):
    if not 0.0 < probability < 1.0:
        raise ValueError('initial gate probability must be between 0 and 1.')
    return torch.logit(torch.tensor(float(probability)))

class PhaseContextEncoder(nn.Module):
    """Encode historical solar longitude using first and second harmonics."""

    def __init__(self, seq_len, context_dim):
        super().__init__()
        self.seq_len = seq_len
        self.context_dim = context_dim
        self.net = nn.Sequential(nn.Linear(seq_len * 4, context_dim), nn.GELU(), nn.Linear(context_dim, context_dim), nn.LayerNorm(context_dim))

    def build_harmonic_features(self, ls):
        if ls.dim() != 2:
            raise ValueError(f'Expected ls shape [B, T], got {tuple(ls.shape)}')
        if ls.shape[1] != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {ls.shape[1]}')
        ls_rad = ls * (torch.pi / 180.0)
        return torch.stack([torch.sin(ls_rad), torch.cos(ls_rad), torch.sin(2.0 * ls_rad), torch.cos(2.0 * ls_rad)], dim=-1)

    def forward(self, ls):
        harmonic_features = self.build_harmonic_features(ls)
        return self.net(harmonic_features.reshape(ls.shape[0], self.seq_len * 4))

class PhaseConditionedMemoryGate(nn.Module):
    """
    Gate which ConvLSTM historical memories are admitted into the MST translator.

    The gate weight multiplies each historical hidden state. The complement keeps
    the final ConvLSTM memory as a conservative fallback. Phase context acts as a
    direct sigmoid-logit bias so seasonal phase can change the default gate
    openness without competing with spatial hidden features as another channel.
    """

    def __init__(self, hidden_dim, context_dim, initial_history_weight=0.7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phase_to_bias = nn.Linear(context_dim, hidden_dim)
        self.gate = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.reset_parameters(initial_history_weight)

    def reset_parameters(self, initial_history_weight):
        nn.init.xavier_uniform_(self.phase_to_bias.weight, gain=0.1)
        nn.init.zeros_(self.phase_to_bias.bias)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        with torch.no_grad():
            self.gate.bias.fill_(_initial_logit(initial_history_weight))

    def forward(self, hidden_sequence, phase_context):
        if hidden_sequence.dim() != 5:
            raise ValueError(f'Expected hidden_sequence [B, T, C, H, W], got {tuple(hidden_sequence.shape)}')
        (batch_size, seq_len, channels, height, width) = hidden_sequence.shape
        if channels != self.hidden_dim:
            raise ValueError(f'Expected hidden_dim={self.hidden_dim}, got {channels}')
        gate_input = hidden_sequence.reshape(batch_size * seq_len, channels, height, width)
        gate_logits = self.gate(gate_input).view(batch_size, seq_len, channels, height, width)
        phase_bias = self.phase_to_bias(phase_context).view(batch_size, 1, channels, 1, 1)
        gate_map = torch.sigmoid(gate_logits + phase_bias)
        last_hidden = hidden_sequence[:, -1:].expand_as(hidden_sequence)
        admitted = gate_map * hidden_sequence + (1.0 - gate_map) * last_hidden
        return (admitted, gate_map)

class PhaseConditionedResidualGate(nn.Module):
    """
    Gate translated MST future memories against the last ConvLSTM memory.

    The gate weight multiplies the translated future hidden state. The complement
    preserves the last recurrent hidden state. Phase context acts as a direct
    sigmoid-logit bias on the translated-vs-recurrent residual write.
    """

    def __init__(self, hidden_dim, context_dim, initial_translation_weight=0.7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phase_to_bias = nn.Linear(context_dim, hidden_dim)
        self.gate = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.reset_parameters(initial_translation_weight)

    def reset_parameters(self, initial_translation_weight):
        nn.init.xavier_uniform_(self.phase_to_bias.weight, gain=0.1)
        nn.init.zeros_(self.phase_to_bias.bias)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        with torch.no_grad():
            self.gate.bias.fill_(_initial_logit(initial_translation_weight))

    def forward(self, future_hidden, last_hidden, phase_context):
        if future_hidden.dim() != 5:
            raise ValueError(f'Expected future_hidden [B, P, C, H, W], got {tuple(future_hidden.shape)}')
        if last_hidden.dim() != 4:
            raise ValueError(f'Expected last_hidden [B, C, H, W], got {tuple(last_hidden.shape)}')
        (batch_size, pred_len, channels, height, width) = future_hidden.shape
        if channels != self.hidden_dim:
            raise ValueError(f'Expected hidden_dim={self.hidden_dim}, got {channels}')
        last_hidden_seq = last_hidden.unsqueeze(1).expand(batch_size, pred_len, channels, height, width)
        gate_input = torch.cat([future_hidden, last_hidden_seq], dim=2)
        gate_input = gate_input.reshape(batch_size * pred_len, channels * 2, height, width)
        gate_logits = self.gate(gate_input).view(batch_size, pred_len, channels, height, width)
        phase_bias = self.phase_to_bias(phase_context).view(batch_size, 1, channels, 1, 1)
        gate_map = torch.sigmoid(gate_logits + phase_bias)
        final_hidden = gate_map * future_hidden + (1.0 - gate_map) * last_hidden_seq
        return (final_hidden, gate_map)

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

class ConvLSTMPhaseGatedMSTForecaster(nn.Module):
    """ConvLSTM encoder with phase-gated serial MST hidden translation."""

    def __init__(self, seq_len, pred_len, lat_size, lon_size, use_phase_warp, hidden_dims, phase_context_dim, mst_spatial_hidden_dim, mst_temporal_hidden_dim, mst_num_downsample, mst_temporal_depth, dropout, kernel_size=3, initial_history_weight=0.7, initial_translation_weight=0.7):
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
        self.phase_context = PhaseContextEncoder(seq_len=seq_len, context_dim=phase_context_dim)
        self.memory_gate = PhaseConditionedMemoryGate(hidden_dim=hidden_dim, context_dim=phase_context_dim, initial_history_weight=initial_history_weight)
        self.mst_block = MSTBlock(seq_len=seq_len, pred_len=pred_len, in_channels=hidden_dim, out_channels=hidden_dim, spatial_hidden_dim=mst_spatial_hidden_dim, temporal_hidden_dim=mst_temporal_hidden_dim, num_downsample=mst_num_downsample, temporal_depth=mst_temporal_depth, dropout=dropout, return_residual_gate=False)
        self.residual_gate = PhaseConditionedResidualGate(hidden_dim=hidden_dim, context_dim=phase_context_dim, initial_translation_weight=initial_translation_weight)
        self.forecast_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

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

    def forecast_from_hidden(self, hidden_sequence):
        (batch_size, pred_len, channels, height, width) = hidden_sequence.shape
        flat_hidden = hidden_sequence.reshape(batch_size * pred_len, channels, height, width)
        flat_pred = self.forecast_head(flat_hidden)
        return flat_pred.view(batch_size, pred_len, height, width)

    def forward(self, x, ls, return_diagnostics=False):
        hidden_sequence = self.encode_hidden_sequence(x, ls)
        phase_context = self.phase_context(ls)
        (admitted_hidden, memory_gate) = self.memory_gate(hidden_sequence, phase_context)
        future_hidden = self.mst_block(admitted_hidden)
        (final_hidden, residual_gate) = self.residual_gate(future_hidden, hidden_sequence[:, -1], phase_context)
        prediction = self.forecast_from_hidden(final_hidden)
        if not return_diagnostics:
            return prediction
        diagnostics = {'phase_context': phase_context, 'memory_gate': memory_gate, 'future_hidden': future_hidden, 'residual_gate': residual_gate}
        return (prediction, diagnostics)

def evaluate_gate_diagnostics(model, loader, device):
    model.eval()
    memory_values = []
    residual_values = []
    with torch.no_grad():
        for (xb, lsb, _) in loader:
            xb = xb.to(device)
            lsb = lsb.to(device)
            (_, diagnostics) = model(xb, lsb, return_diagnostics=True)
            memory_values.append(diagnostics['memory_gate'].mean().item())
            residual_values.append(diagnostics['residual_gate'].mean().item())
    memory_mean = sum(memory_values) / max(1, len(memory_values))
    residual_mean = sum(residual_values) / max(1, len(residual_values))
    return {'memory_gate_mean': memory_mean, 'residual_gate_mean': residual_mean}

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, window, horizon, hidden_dims, phase_context_dim, mst_spatial_hidden_dim, mst_temporal_hidden_dim, mst_num_downsample, mst_temporal_depth, dropout, epochs, learning_rate, early_stopping_patience, base_dir):
    print(f'\n[Experiment] {label}')
    model = ConvLSTMPhaseGatedMSTForecaster(seq_len=window, pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, hidden_dims=hidden_dims, phase_context_dim=phase_context_dim, mst_spatial_hidden_dim=mst_spatial_hidden_dim, mst_temporal_hidden_dim=mst_temporal_hidden_dim, mst_num_downsample=mst_num_downsample, mst_temporal_depth=mst_temporal_depth, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    checkpoint_path = os.path.join(base_dir, 'models', TRAIN_RESULT_DIR, f'{label.lower()}_checkpoint.pth')
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
    gate_stats = evaluate_gate_diagnostics(model, test_loader, device)
    save_path = os.path.join(base_dir, 'models', TRAIN_RESULT_DIR, f'{label.lower()}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'{label} weights saved to: {save_path}')
    print(f"{label} Gate Diagnostics | memory_gate_mean: {gate_stats['memory_gate_mean']:.4f} | residual_gate_mean: {gate_stats['residual_gate_mean']:.4f}")
    print(f"{label} Metrics | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R^2: {metrics['r2']:.4f} | SMAPE: {metrics['smape']:.2%}")
    return metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(base_dir, 'models', TRAIN_LOG_DIR), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models', TRAIN_RESULT_DIR), exist_ok=True)
    sys.stdout = Logger(os.path.join(base_dir, 'models', TRAIN_LOG_DIR, 'ConvLSTM_PhaseGatedMST.txt'))
    seed = 11
    set_experiment_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    print(f'Single Seed: {seed}')
    print('Model: ConvLSTM_PhaseGatedMST')
    window = 3
    horizon = 3
    batch_size = 4
    hidden_dim = 32
    num_layers = 2
    hidden_dims = [hidden_dim] * num_layers
    phase_context_dim = 32
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
    raw_metrics = train_and_evaluate(label=f'ConvLSTM_PhaseGatedMST_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, window=window, horizon=horizon, hidden_dims=hidden_dims, phase_context_dim=phase_context_dim, mst_spatial_hidden_dim=mst_spatial_hidden_dim, mst_temporal_hidden_dim=mst_temporal_hidden_dim, mst_num_downsample=mst_num_downsample, mst_temporal_depth=mst_temporal_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
    print('\n[Raw Summary]')
    print(f"ConvLSTM_PhaseGatedMST_Raw Metrics | RMSE: {raw_metrics['rmse']:.4f} | MAE: {raw_metrics['mae']:.4f} | R^2: {raw_metrics['r2']:.4f} | SMAPE: {raw_metrics['smape']:.2%}")
