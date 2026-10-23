"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import glob
import os
import re
import sys
import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from .phase_warp_frontend import PhaseWarpFrontEnd
from .legacy_data_paths import resolve_legacy_data_dirs
from .seed_utils import RUN_SEEDS, print_seed_summary, set_experiment_seed, summarize_seed_metrics

class Logger(object):
    """把终端输出同步写入日志文件，方便之后回看训练过程。"""

    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class EarlyStopping:
    """
    基于验证集损失的简单早停器。
    compare 脚本的目标是做受控实验，因此实现保持尽量朴素透明。
    """

    def __init__(self, patience=5, verbose=False, delta=0.0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def natural_sort_key(s):
    """按人类习惯排序文件名，例如 2 会排在 10 前面。"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def merge_sol_hour(x):
    """把 MCD 数据从 [sol, hour, lat, lon] 展平为 [time, lat, lon]。"""
    (sols, hours, lat_size, lon_size) = x.shape
    return x.reshape(sols * hours, lat_size, lon_size)

def clean_invalid(x, name):
    """把 NaN / inf / 极端异常值清洗掉，避免训练时数值不稳定。"""
    x = np.array(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 10000000000.0)
    if np.any(bad):
        print(f'{name}: cleaned {bad.sum()} invalid values')
        x[bad] = np.nan
    return np.nan_to_num(x, nan=0.0)

def unwrap_ls(ls_array):
    """
    把循环的 Ls 序列展开成连续轴，用于后续插值。
    例如：
    350, 355, 2, 7
    会被展开为：
    350, 355, 362, 367
    """
    ls_unwrapped = np.copy(ls_array)
    year_offset = 0
    for idx in range(1, len(ls_unwrapped)):
        if ls_array[idx] < ls_array[idx - 1] - 180:
            year_offset += 360
        ls_unwrapped[idx] += year_offset
    return ls_unwrapped

def load_aligned_cube(base_dir):
    """
    读取臭氧和气象变量，并把它们统一对齐到 OpenMars 时间轴上。

    返回：
    - x_raw: [T, H, W, 5]
      通道顺序 [O3, U, V, Temperature, Solar_Flux]
    - y_raw: [T, H, W]
      目标臭氧场
    - om_ls_continuous: [T]
      与 x_raw / y_raw 对齐的连续 Ls
    """
    (openmars_dir, mcd_dir) = resolve_legacy_data_dirs(base_dir)
    print('\n[Step 1] Loading OpenMars Data...')
    file_list = sorted(glob.glob(os.path.join(openmars_dir, '*.nc')), key=natural_sort_key)
    if not file_list:
        raise FileNotFoundError('OpenMars files were not found.')
    o3_list = []
    om_ls_list = []
    for file_path in file_list:
        ds = nc.Dataset(file_path)
        o3_list.append(ds.variables['o3col'][:])
        if 'Ls' in ds.variables:
            om_ls_list.append(ds.variables['Ls'][:])
        elif 'ls' in ds.variables:
            om_ls_list.append(ds.variables['ls'][:])
        else:
            raise ValueError(f'Missing Ls variable in {file_path}')
        ds.close()
    y_raw = clean_invalid(np.concatenate(o3_list, axis=0), 'OpenMars O3')
    om_ls_raw = np.concatenate(om_ls_list, axis=0)
    print(f'OpenMars final shape: {y_raw.shape}')
    print('\n[Step 2] Loading MCD Data...')
    target_files = [os.path.join(mcd_dir, 'MCD_MY27_Lat-90-90_real.nc'), os.path.join(mcd_dir, 'MCD_MY28_Lat-90-90_real.nc')]
    short_names = ['u', 'v', 'temp', 'fluxsurf_dn_sw']
    mcd_data_list = {key: [] for key in short_names}
    mcd_ls_list = []
    for file_path in target_files:
        if not os.path.exists(file_path):
            continue
        print(f'Loading: {os.path.basename(file_path)}')
        ds = nc.Dataset(file_path)
        mcd_data_list['u'].append(merge_sol_hour(ds.variables['U_Wind'][:]))
        mcd_data_list['v'].append(merge_sol_hour(ds.variables['V_Wind'][:]))
        mcd_data_list['temp'].append(merge_sol_hour(ds.variables['Temperature'][:]))
        mcd_data_list['fluxsurf_dn_sw'].append(merge_sol_hour(ds.variables['Solar_Flux_DN'][:]))
        ls_tmp = ds.variables['Ls'][:] if 'Ls' in ds.variables else ds.variables['ls'][:]
        (sols, hours) = ds.variables['U_Wind'].shape[:2]
        if ls_tmp.ndim == 1 and len(ls_tmp) == sols:
            ls_expanded = np.zeros(sols * hours)
            for idx in range(sols):
                ls_start = ls_tmp[idx]
                if idx < sols - 1:
                    ls_end = ls_tmp[idx + 1]
                    if ls_end < ls_start:
                        ls_end += 360.0
                else:
                    ls_end = ls_start + (ls_tmp[1] - ls_tmp[0] if sols > 1 else 0.5)
                ls_expanded[idx * hours:(idx + 1) * hours] = np.linspace(ls_start, ls_end, hours, endpoint=False)
            mcd_ls_list.append(ls_expanded % 360.0)
        else:
            mcd_ls_list.append(ls_tmp.flatten())
        ds.close()
    vars_dict = {key: clean_invalid(np.concatenate(value, axis=0), key) for (key, value) in mcd_data_list.items()}
    if 'fluxsurf_dn_sw' in vars_dict:
        vars_dict['fluxsurf_dn_sw'] /= np.max(vars_dict['fluxsurf_dn_sw']) + 1e-06
    print('\n[Step 3] Aligning MCD to OpenMars time axis...')
    om_ls_continuous = unwrap_ls(om_ls_raw)
    mcd_ls_continuous = unwrap_ls(np.concatenate(mcd_ls_list, axis=0))
    for key in vars_dict:
        interpolator = interp1d(mcd_ls_continuous, vars_dict[key], axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
        vars_dict[key] = interpolator(om_ls_continuous)
    x_raw = np.stack([y_raw, vars_dict['u'], vars_dict['v'], vars_dict['temp'], vars_dict['fluxsurf_dn_sw']], axis=-1)
    return (x_raw, y_raw, om_ls_continuous)

def build_grid_dataloaders(x_raw, y_raw, ls_raw, window, horizon, batch_size):
    """
    在严格历史输入设定下构造样本：
    - 输入：x[t : t + window] 与 ls[t : t + window]
    - 目标：y[t + window : t + window + horizon]
    """
    (time_steps, lat_size, lon_size, channels) = x_raw.shape
    split_time_idx = int(0.8 * time_steps)
    x_train_raw = x_raw[:split_time_idx]
    y_train_raw = y_raw[:split_time_idx]
    x_scaled = np.zeros_like(x_raw, dtype=np.float32)
    for channel_idx in range(channels):
        scaler = StandardScaler()
        scaler.fit(x_train_raw[..., channel_idx].reshape(split_time_idx, -1))
        x_scaled[..., channel_idx] = scaler.transform(x_raw[..., channel_idx].reshape(time_steps, -1)).reshape(time_steps, lat_size, lon_size)
    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std()
    y_scaled = (y_raw - y_mean) / (y_std + 1e-06)
    x_seq = []
    y_seq = []
    ls_seq = []
    for idx in range(time_steps - window - horizon + 1):
        x_seq.append(x_scaled[idx:idx + window])
        y_seq.append(y_scaled[idx + window:idx + window + horizon])
        ls_seq.append(ls_raw[idx:idx + window])
    x_torch = torch.tensor(np.array(x_seq)).permute(0, 1, 4, 2, 3).float()
    y_torch = torch.tensor(np.array(y_seq)).float()
    ls_torch = torch.tensor(np.array(ls_seq)).float()
    split_sample_idx = max(0, min(split_time_idx - window - horizon + 1, len(x_torch)))
    train_dataset = TensorDataset(x_torch[:split_sample_idx], ls_torch[:split_sample_idx], y_torch[:split_sample_idx])
    test_dataset = TensorDataset(x_torch[split_sample_idx:], ls_torch[split_sample_idx:], y_torch[split_sample_idx:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (train_loader, test_loader, y_mean, y_std)

class ExponentialSmoothingLayer(nn.Module):
    """
    可学习指数平滑层。

    这层用一个可学习平滑系数 alpha，在时间维递推得到平滑表示：
    s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    ETSformer 的直觉之一，就是把这种经典统计学平滑思想融入深度模型。
    """

    def __init__(self, d_model):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        states = []
        prev = x[:, 0:1, :]
        states.append(prev)
        for time_idx in range(1, x.size(1)):
            current = alpha * x[:, time_idx:time_idx + 1, :] + (1.0 - alpha) * prev
            states.append(current)
            prev = current
        return torch.cat(states, dim=1)

class GrowthLayer(nn.Module):
    """
    对增长项做建模。

    做法：
    1. 先计算相邻时刻差分，近似局部增长速度；
    2. 再用指数平滑过滤掉高噪声波动；
    3. 输出平滑后的 growth 表示。
    """

    def __init__(self, d_model):
        super().__init__()
        self.smoothing = ExponentialSmoothingLayer(d_model=d_model)

    def forward(self, x):
        growth = torch.zeros_like(x)
        growth[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return self.smoothing(growth)

class FrequencySeasonalLayer(nn.Module):
    """
    对季节项做频域建模。

    思路：
    - 在时间维做 FFT；
    - 只保留一部分主导频率；
    - 对这些频率施加可学习复数权重；
    - 再通过 iFFT 回到时间域。

    这和 ETSformer 中用 frequency attention 建模 seasonality 的思想一致。
    """

    def __init__(self, d_model, top_k=4):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.weight_real = nn.Parameter(torch.randn(top_k, d_model) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(top_k, d_model) * 0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = torch.zeros_like(x_ft)
        used_k = min(self.top_k, x_ft.size(1))
        weight = torch.complex(self.weight_real[:used_k], self.weight_imag[:used_k]).unsqueeze(0)
        out_ft[:, :used_k, :] = x_ft[:, :used_k, :] * weight
        return torch.fft.irfft(out_ft, n=seq_len, dim=1)

class ETSformerEncoderLayer(nn.Module):
    """
    轻量 ETSformer 风格编码层。

    每一层把输入拆成三类语义：
    - level：平滑基线
    - growth：变化趋势
    - seasonality：周期波动

    然后把三者重新融合，形成新的隐藏表示。
    """

    def __init__(self, d_model, d_ff, dropout, top_k_freq):
        super().__init__()
        self.level_layer = ExponentialSmoothingLayer(d_model=d_model)
        self.growth_layer = GrowthLayer(d_model=d_model)
        self.seasonal_layer = FrequencySeasonalLayer(d_model=d_model, top_k=top_k_freq)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        level = self.level_layer(x)
        growth = self.growth_layer(x)
        seasonal = self.seasonal_layer(x - level)
        fused = level + growth + seasonal
        x = self.norm(fused + self.ffn(fused))
        return (x, level, growth, seasonal)

class ETSformerBackbone(nn.Module):
    """
    适合当前项目的轻量 ETSformer 风格主干。

    保留的核心直觉：
    - 用 exponential smoothing 建模 level 与 growth；
    - 用 frequency block 建模 seasonality；
    - 最后把三部分分别投影到未来，再叠加成未来多变量预测。
    """

    def __init__(self, seq_len, pred_len, n_vars, d_model, e_layers, d_ff, dropout, top_k_freq):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.value_embedding = nn.Linear(n_vars, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ETSformerEncoderLayer(d_model=d_model, d_ff=d_ff, dropout=dropout, top_k_freq=top_k_freq) for _ in range(e_layers)])
        self.level_proj = nn.Linear(d_model, n_vars)
        self.growth_proj = nn.Linear(d_model, n_vars)
        self.seasonal_time_proj = nn.Linear(seq_len, pred_len)
        self.seasonal_feature_proj = nn.Linear(d_model, n_vars)

    def forward(self, x):
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-05)
        x = x / stdev
        hidden = self.dropout(self.value_embedding(x))
        last_level = None
        last_growth = None
        last_seasonal = None
        for layer in self.layers:
            (hidden, last_level, last_growth, last_seasonal) = layer(hidden)
        level_base = self.level_proj(last_level[:, -1, :]).unsqueeze(1).repeat(1, self.pred_len, 1)
        growth_step = self.growth_proj(last_growth[:, -1, :]).unsqueeze(1)
        horizon_index = torch.arange(1, self.pred_len + 1, device=x.device, dtype=x.dtype).view(1, self.pred_len, 1)
        growth_forecast = growth_step * horizon_index
        seasonal_forecast = self.seasonal_time_proj(last_seasonal.transpose(1, 2)).transpose(1, 2)
        seasonal_forecast = self.seasonal_feature_proj(seasonal_forecast)
        dec_out = level_base + growth_forecast + seasonal_forecast
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

class GridPointETSformerO3(nn.Module):
    """
    把每个经纬格点看成一条多变量时间序列，并共享同一个 ETSformer 模型。

    输入：
    - x: [B, T, C, H, W]
    - ls: [B, T]

    中间：
    - reshape 为 [B * H * W, T, C]
    - 可选经过 PhaseWarpFrontEnd
    - ETSformer 预测未来每个特征的轨迹
    - 再用 target_head 把未来特征压成未来 O3

    输出：
    - [B, horizon, H, W]
    """

    def __init__(self, seq_len, pred_len, lat_size, lon_size, use_phase_warp, d_model, e_layers, d_ff, dropout, top_k_freq):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.use_phase_warp = use_phase_warp
        if use_phase_warp:
            self.phase_warp = PhaseWarpFrontEnd()
            n_vars = 9
        else:
            self.phase_warp = None
            n_vars = 5
        self.backbone = ETSformerBackbone(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars, d_model=d_model, e_layers=e_layers, d_ff=d_ff, dropout=dropout, top_k_freq=top_k_freq)
        self.target_head = nn.Linear(n_vars, 1)

    def forward(self, x, ls):
        (batch_size, seq_len, channels, lat_size, lon_size) = x.shape
        x_point = x.permute(0, 3, 4, 1, 2).reshape(batch_size * lat_size * lon_size, seq_len, channels)
        ls_point = ls[:, None, None, :].expand(batch_size, lat_size, lon_size, seq_len).reshape(batch_size * lat_size * lon_size, seq_len)
        if self.phase_warp is not None:
            features = self.phase_warp(x_point, ls_point)
        else:
            features = x_point
        pred_features = self.backbone(features)
        pred_o3 = self.target_head(pred_features).squeeze(-1)
        return pred_o3.view(batch_size, lat_size, lon_size, self.pred_len).permute(0, 3, 1, 2)

def evaluate_metrics(model, loader, device, y_std, y_mean):
    """把标准化空间里的预测还原到物理臭氧单位后再计算指标。"""
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for (xb, lsb, yb) in loader:
            xb = xb.to(device)
            lsb = lsb.to(device)
            pred = model(xb, lsb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    y_pred_phys = preds * y_std + y_mean
    y_true_phys = trues * y_std + y_mean
    pred_flat = y_pred_phys.reshape(-1)
    true_flat = y_true_phys.reshape(-1)
    mse = np.mean((pred_flat - true_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - true_flat))
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    smape = np.mean(2.0 * np.abs(pred_flat - true_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-06))
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'smape': smape}

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, window, horizon, d_model, e_layers, d_ff, dropout, top_k_freq, epochs, learning_rate, early_stopping_patience, base_dir):
    """
    训练一组 ETSformer 实验，并返回评估指标。
    受控变量只有一个：是否启用 phase-warp 前端。
    """
    print(f'\n[Experiment] {label}')
    model = GridPointETSformerO3(seq_len=window, pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, d_model=d_model, e_layers=e_layers, d_ff=d_ff, dropout=dropout, top_k_freq=top_k_freq).to(device)
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
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'ETSformer_PhaseWarp_Compare.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    window = 3
    horizon = 3
    batch_size = 4
    d_model = 64
    e_layers = 2
    d_ff = 128
    dropout = 0.1
    top_k_freq = 4
    epochs = 15
    learning_rate = 0.001
    early_stopping_patience = 5
    (x_raw, y_raw, ls_raw) = load_aligned_cube(base_dir)
    (lat_size, lon_size) = (y_raw.shape[1], y_raw.shape[2])
    (train_loader, test_loader, y_mean, y_std) = build_grid_dataloaders(x_raw=x_raw, y_raw=y_raw, ls_raw=ls_raw, window=window, horizon=horizon, batch_size=batch_size)
    raw_seed_metrics = []
    phase_seed_metrics = []
    for seed in RUN_SEEDS:
        print(f'\n[Seed] {seed}')
        set_experiment_seed(seed)
        raw_metrics = train_and_evaluate(label=f'ETSformer_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, window=window, horizon=horizon, d_model=d_model, e_layers=e_layers, d_ff=d_ff, dropout=dropout, top_k_freq=top_k_freq, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        raw_seed_metrics.append(raw_metrics)
        set_experiment_seed(seed)
        phase_metrics = train_and_evaluate(label=f'ETSformer_PhaseWarp_seed{seed}', use_phase_warp=True, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, window=window, horizon=horizon, d_model=d_model, e_layers=e_layers, d_ff=d_ff, dropout=dropout, top_k_freq=top_k_freq, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        phase_seed_metrics.append(phase_metrics)
    raw_summary = summarize_seed_metrics(raw_seed_metrics)
    phase_summary = summarize_seed_metrics(phase_seed_metrics)
    print('\n[Comparison Summary]')
    print_seed_summary('ETSformer_Raw', raw_summary)
    print_seed_summary('ETSformer_PhaseWarp', phase_summary)
    print(f"RMSE improvement: {raw_summary['rmse']['mean'] - phase_summary['rmse']['mean']:.4f}")
    print(f"MAE improvement : {raw_summary['mae']['mean'] - phase_summary['mae']['mean']:.4f}")
    print(f"R^2 gain        : {phase_summary['r2']['mean'] - raw_summary['r2']['mean']:.4f}")
    print(f"SMAPE gain      : {raw_summary['smape']['mean'] - phase_summary['smape']['mean']:.2%}")
