"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import glob
import os
import re
import sys
from collections import deque
import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from .phase_warp_frontend import PhaseWarpFrontEnd
from .legacy_data_paths import resolve_legacy_data_dirs
from .seed_utils import RUN_SEEDS, print_seed_summary, set_experiment_seed, summarize_seed_metrics

class Logger(object):
    """把终端输出同步写入日志文件，方便复现实验过程。"""

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
    这里保持实现尽量朴素，目的是做“是否加入相位扭曲模块”的受控对比。
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
    """把 NaN / inf / 异常填充值替换成 0，避免训练阶段数值不稳定。"""
    x = np.array(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 10000000000.0)
    if np.any(bad):
        print(f'{name}: cleaned {bad.sum()} invalid values')
        x[bad] = np.nan
    return np.nan_to_num(x, nan=0.0)

def unwrap_ls(ls_array):
    """
    将循环的 Ls 序列展开成连续轴，用于后续插值。
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
    读取臭氧和气象变量，并将它们统一对齐到 OpenMars 的时间轴上。

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
    在“严格历史输入”的设定下构造样本序列。

    输入：
    - x[t : t + window]
    - ls[t : t + window]

    预测目标：
    - y[t + window : t + window + horizon]
    """
    (time_steps, lat_size, lon_size, channels) = x_raw.shape
    split_time_idx = int(0.8 * time_steps)
    x_train_raw = x_raw[:split_time_idx]
    y_train_raw = y_raw[:split_time_idx]
    x_scaled = np.zeros_like(x_raw)
    for channel_idx in range(channels):
        scaler = StandardScaler()
        scaler.fit(x_train_raw[..., channel_idx].reshape(split_time_idx, -1))
        x_scaled[..., channel_idx] = scaler.transform(x_raw[..., channel_idx].reshape(time_steps, -1)).reshape(time_steps, lat_size, lon_size)
    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std()
    y_scaled = (y_raw - y_mean) / y_std
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

class MAUCell(nn.Module):
    """
    MAU 的核心单元。

    根据论文中的思路，这里维护两类状态：
    - temporal state T：偏向表示运动 / 时序变化
    - spatial state S：偏向表示当前时刻的外观 / 空间结构

    在时间步 t、层 k：
    1. 先根据当前 spatial state 与过去 tau 步的 spatial states 的相似性，
       给过去 tau 步的 temporal states 分配注意力权重；
    2. 得到聚合后的长期运动信息 T_att；
    3. 再与最近一步 temporal state 融合成增强运动信息 T_AMI；
    4. 最后用两个更新门，把 motion / appearance 融合成新的 T_t^k 和 S_t^k。
    """

    def __init__(self, hidden_dim, tau, kernel_size=5, gamma=1.0):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.gamma = gamma
        self.ws = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.wf = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.wtu = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.wsu = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.wtt = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.wst = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.wss = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.wts = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.norm_t = nn.GroupNorm(1, hidden_dim)
        self.norm_s = nn.GroupNorm(1, hidden_dim)

    def _pad_history(self, history_list, template):
        """
        把早期时间步不足 tau 的历史用全零状态补齐。
        这样同一套公式可以稳定用于整个时间序列。
        """
        padded = list(history_list)[-self.tau:]
        while len(padded) < self.tau:
            padded.insert(0, torch.zeros_like(template))
        return padded

    def forward(self, s_cur, s_history, t_history):
        """
        输入：
        - s_cur: 当前层输入的当前 spatial state，形状 [B, C, H, W]
        - s_history: 当前层输入在过去若干步的 spatial states 列表
        - t_history: 当前层自身在过去若干步的 temporal states 列表

        输出：
        - t_new: 当前时间步更新后的 temporal state
        - s_new: 当前时间步更新后的 spatial state
        """
        s_hist = self._pad_history(s_history, s_cur)
        t_hist = self._pad_history(t_history, s_cur)
        s_prime = self.ws(s_cur)
        score_list = []
        for s_past in s_hist:
            score = torch.sum(s_past * s_prime, dim=(1, 2, 3))
            score_list.append(score)
        attn_scores = torch.stack(score_list, dim=1)
        attn_weights = F.softmax(attn_scores, dim=1)
        t_stack = torch.stack(t_hist, dim=1)
        t_att = torch.sum(t_stack * attn_weights[:, :, None, None, None], dim=1)
        t_prev = t_hist[-1]
        u_f = torch.sigmoid(self.wf(t_prev))
        t_ami = u_f * t_prev + (1.0 - u_f) * t_att
        u_t = torch.sigmoid(self.wtu(t_ami))
        u_s = torch.sigmoid(self.wsu(s_cur))
        t_new = u_t * self.wtt(t_ami) + (1.0 - u_t) * self.wst(s_cur)
        s_new = u_s * self.wss(s_cur) + (1.0 - u_s) * self.wts(t_ami) + self.gamma * s_cur
        t_new = self.norm_t(t_new)
        s_new = self.norm_s(s_new)
        return (t_new, s_new)

class MAUForecaster(nn.Module):
    """
    一个适合当前项目的 MAU 风格时空预测主干。

    与论文中的逐帧自回归解码不同，这里为了适配“只用历史输入、直接输出未来 O3”
    的实验设定，采用如下改造：
    1. 历史多变量网格先经过共享卷积编码；
    2. 多层 MAU 沿时间轴编码整个历史序列；
    3. 用最后一个时间步最顶层的 spatial / temporal 状态联合回归未来 horizon 张 O3 图。

    这样可以保留 MAU 的关键建模思想，同时又和你现在这套历史窗口预测框架一致。
    """

    def __init__(self, pred_len, lat_size, lon_size, use_phase_warp, hidden_dim, num_layers, tau, kernel_size=5, gamma=1.0):
        super().__init__()
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.use_phase_warp = use_phase_warp
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tau = tau
        if use_phase_warp:
            self.phase_warp = PhaseWarpFrontEnd(spatial_shape=(lat_size, lon_size))
            input_dim = 9
        else:
            self.phase_warp = None
            input_dim = 5
        self.frame_encoder = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU(), nn.GroupNorm(1, hidden_dim))
        self.cells = nn.ModuleList([MAUCell(hidden_dim=hidden_dim, tau=tau, kernel_size=kernel_size, gamma=gamma) for _ in range(num_layers)])
        self.forecast_head = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, pred_len, kernel_size=1))

    def forward(self, x, ls):
        if self.phase_warp is not None:
            features = self.phase_warp(x, ls)
        else:
            features = x
        (batch_size, seq_len, _, _, _) = features.shape
        temporal_histories = [deque(maxlen=self.tau) for _ in range(self.num_layers)]
        spatial_histories = [deque(maxlen=self.tau) for _ in range(self.num_layers)]
        encoded_histories = deque(maxlen=self.tau)
        top_t_last = None
        top_s_last = None
        for t in range(seq_len):
            encoded_cur = self.frame_encoder(features[:, t])
            current_input = encoded_cur
            current_t_states = []
            current_s_states = []
            for (layer_idx, cell) in enumerate(self.cells):
                if layer_idx == 0:
                    s_history = list(encoded_histories)
                else:
                    s_history = list(spatial_histories[layer_idx - 1])
                t_history = list(temporal_histories[layer_idx])
                (t_new, s_new) = cell(current_input, s_history, t_history)
                current_t_states.append(t_new)
                current_s_states.append(s_new)
                current_input = s_new
            encoded_histories.append(encoded_cur)
            for layer_idx in range(self.num_layers):
                temporal_histories[layer_idx].append(current_t_states[layer_idx])
                spatial_histories[layer_idx].append(current_s_states[layer_idx])
            top_t_last = current_t_states[-1]
            top_s_last = current_s_states[-1]
        final_feature = torch.cat([top_s_last, top_t_last], dim=1)
        return self.forecast_head(final_feature)

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
    r2 = 1 - ss_res / ss_tot
    smape = np.mean(2.0 * np.abs(pred_flat - true_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-06))
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'smape': smape}

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, horizon, hidden_dim, num_layers, tau, kernel_size, gamma, epochs, learning_rate, early_stopping_patience, base_dir):
    """
    训练一组 MAU 风格实验，并返回评估指标。
    受控变量只有一个：是否启用 phase-warp 前端。
    """
    print(f'\n[Experiment] {label}')
    model = MAUForecaster(pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, hidden_dim=hidden_dim, num_layers=num_layers, tau=tau, kernel_size=kernel_size, gamma=gamma).to(device)
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
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'MAU_PhaseWarp_Compare.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    window = 3
    horizon = 3
    batch_size = 4
    hidden_dim = 32
    num_layers = 2
    tau = 3
    kernel_size = 5
    gamma = 1.0
    epochs = 20
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
        raw_metrics = train_and_evaluate(label=f'MAU_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, hidden_dim=hidden_dim, num_layers=num_layers, tau=tau, kernel_size=kernel_size, gamma=gamma, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        raw_seed_metrics.append(raw_metrics)
        set_experiment_seed(seed)
        phase_metrics = train_and_evaluate(label=f'MAU_PhaseWarp_seed{seed}', use_phase_warp=True, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, hidden_dim=hidden_dim, num_layers=num_layers, tau=tau, kernel_size=kernel_size, gamma=gamma, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        phase_seed_metrics.append(phase_metrics)
    raw_summary = summarize_seed_metrics(raw_seed_metrics)
    phase_summary = summarize_seed_metrics(phase_seed_metrics)
    print('\n[Comparison Summary]')
    print_seed_summary('MAU_Raw', raw_summary)
    print_seed_summary('MAU_PhaseWarp', phase_summary)
    print(f"RMSE improvement: {raw_summary['rmse']['mean'] - phase_summary['rmse']['mean']:.4f}")
    print(f"MAE improvement : {raw_summary['mae']['mean'] - phase_summary['mae']['mean']:.4f}")
    print(f"R^2 gain        : {phase_summary['r2']['mean'] - raw_summary['r2']['mean']:.4f}")
    print(f"SMAPE gain      : {raw_summary['smape']['mean'] - phase_summary['smape']['mean']:.2%}")
