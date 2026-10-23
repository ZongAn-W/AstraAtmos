"""Sanitized backbone definitions imported from the ablation experiment folder.
Generated for AstraAtmos training; top-level experiment runners are intentionally omitted.
"""
import glob
import math
import os
import re
import sys
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

class SeriesDecomposition(nn.Module):
    """
    时间序列分解模块。

    做法：
    - 用 moving average 提取平滑趋势项 trend
    - 用原序列减去 trend 得到季节/残差项 season

    这是 TimeMixer 中“分解后再混合”的基础步骤。
    """

    def __init__(self, moving_avg):
        super().__init__()
        self.moving_avg = moving_avg
        self.avg = nn.AvgPool1d(kernel_size=moving_avg, stride=1)

    def forward(self, x):
        if self.moving_avg == 1:
            trend = x
        else:
            front = x[:, 0:1, :].repeat(1, (self.moving_avg - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, self.moving_avg // 2, 1)
            x_pad = torch.cat([front, x, end], dim=1)
            trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        season = x - trend
        return (season, trend)

def scale_lengths_from_seq(seq_len, down_sampling_layers, down_sampling_window):
    """
    根据原始长度与下采样倍率，构造多尺度长度列表。

    例如：
    - seq_len=12, down_sampling_window=2, down_sampling_layers=2
      -> [12, 6, 3]

    这里允许长度最终降到 1，但不会再继续产生重复尺度。
    """
    lengths = [seq_len]
    current = seq_len
    for _ in range(down_sampling_layers):
        next_len = max(1, math.ceil(current / down_sampling_window))
        if next_len == current:
            break
        lengths.append(next_len)
        current = next_len
    return lengths

def resize_seq_length(x, target_len):
    """
    把序列长度从 L 调整到 target_len。

    输入：
    - x: [B, L, C]

    这里统一采用自适应平均池化完成长度缩放：
    - 细尺度 -> 粗尺度：自然相当于 downsample
    - 粗尺度 -> 细尺度：相当于平滑插值式上采样

    这样实现简单、稳定，也很符合 TimeMixer 的多尺度聚合精神。
    """
    if x.shape[1] == target_len:
        return x
    return F.adaptive_avg_pool1d(x.permute(0, 2, 1), target_len).permute(0, 2, 1)

class ScaleMixerMLP(nn.Module):
    """
    在单个尺度上做时间维 mixing 的 MLP。

    TimeMixer 的一个核心思想是：
    - 先分解，再分别对趋势项和季节项建模；
    - 每个尺度都要有自己的时序变换器。

    这里的实现方式是：
    - 对 [B, L, C] 先转为 [B, C, L]
    - 在长度维 L 上做两层 MLP
    """

    def __init__(self, seq_len, dropout):
        super().__init__()
        hidden_len = max(seq_len, 2)
        self.net = nn.Sequential(nn.Linear(seq_len, hidden_len), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_len, seq_len), nn.Dropout(dropout))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)

class TimeMixerBlock(nn.Module):
    """
    一个 TimeMixer 风格的 Past-Decomposable-Mixing 块。

    流程：
    1. 对每个尺度的序列分别分解成 season / trend；
    2. 在各自尺度上做时间维 MLP mixing；
    3. season 分支做细 -> 粗的 bottom-up mixing；
    4. trend 分支做粗 -> 细的 top-down mixing；
    5. 再把两部分相加，并做一次特征维变换与残差连接。

    这种“同尺度建模 + 跨尺度信息流动”的结构，是 TimeMixer 的关键特征。
    """

    def __init__(self, scale_lengths, d_model, moving_avg, dropout):
        super().__init__()
        self.scale_lengths = list(scale_lengths)
        self.decomp = SeriesDecomposition(moving_avg=moving_avg)
        self.season_mixers = nn.ModuleList([ScaleMixerMLP(seq_len=length, dropout=dropout) for length in scale_lengths])
        self.trend_mixers = nn.ModuleList([ScaleMixerMLP(seq_len=length, dropout=dropout) for length in scale_lengths])
        self.feature_mixers = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.Dropout(dropout)) for _ in scale_lengths])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in scale_lengths])

    def forward(self, x_list):
        season_list = []
        trend_list = []
        for (idx, x) in enumerate(x_list):
            (season, trend) = self.decomp(x)
            season = self.season_mixers[idx](season)
            trend = self.trend_mixers[idx](trend)
            season_list.append(season)
            trend_list.append(trend)
        mixed_season = [season_list[0]]
        for idx in range(1, len(season_list)):
            fine_to_coarse = resize_seq_length(mixed_season[idx - 1], self.scale_lengths[idx])
            mixed_season.append(season_list[idx] + fine_to_coarse)
        mixed_trend = [None] * len(trend_list)
        mixed_trend[-1] = trend_list[-1]
        for idx in range(len(trend_list) - 2, -1, -1):
            coarse_to_fine = resize_seq_length(mixed_trend[idx + 1], self.scale_lengths[idx])
            mixed_trend[idx] = trend_list[idx] + coarse_to_fine
        outputs = []
        for (idx, x) in enumerate(x_list):
            y = mixed_season[idx] + mixed_trend[idx]
            y = y + self.feature_mixers[idx](y)
            y = self.norms[idx](x + y)
            outputs.append(y)
        return outputs

class TimeMixerBackbone(nn.Module):
    """
    一个适合当前项目的轻量 TimeMixer 主干。

    主流程：
    1. 输入 [B_like, seq_len, n_vars] 先做逐样本归一化；
    2. 用线性层把变量维映射到 d_model；
    3. 构造多尺度序列列表；
    4. 经过若干个 TimeMixerBlock 做分解与混合；
    5. 每个尺度都投影到 pred_len，再汇总多尺度预测；
    6. 投影回变量维，得到 [B_like, pred_len, n_vars]。
    """

    def __init__(self, seq_len, pred_len, n_vars, d_model, e_layers, moving_avg, down_sampling_window, down_sampling_layers, dropout, use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model
        self.use_norm = use_norm
        self.scale_lengths = scale_lengths_from_seq(seq_len=seq_len, down_sampling_layers=down_sampling_layers, down_sampling_window=down_sampling_window)
        self.input_proj = nn.Linear(n_vars, d_model)
        self.blocks = nn.ModuleList([TimeMixerBlock(scale_lengths=self.scale_lengths, d_model=d_model, moving_avg=moving_avg, dropout=dropout) for _ in range(e_layers)])
        self.predictors = nn.ModuleList([nn.Linear(length, pred_len) for length in self.scale_lengths])
        self.output_proj = nn.Linear(d_model, n_vars)

    def build_multiscale(self, x):
        outputs = [x]
        for target_len in self.scale_lengths[1:]:
            outputs.append(resize_seq_length(x, target_len))
        return outputs

    def forward(self, x):
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-05)
            x = x / stdev
        else:
            means = None
            stdev = None
        x = self.input_proj(x)
        x_list = self.build_multiscale(x)
        for block in self.blocks:
            x_list = block(x_list)
        pred_list = []
        for (idx, x_scale) in enumerate(x_list):
            pred = self.predictors[idx](x_scale.permute(0, 2, 1)).permute(0, 2, 1)
            pred_list.append(pred)
        x = torch.stack(pred_list, dim=0).mean(dim=0)
        x = self.output_proj(x)
        if self.use_norm:
            x = x * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            x = x + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return x

class GridPointTimeMixerO3(nn.Module):
    """
    把每个经纬度格点看成一条多变量时间序列，并共享同一个 TimeMixer 模型。

    输入：
    - x: [B, T, C, H, W]
    - ls: [B, T]

    中间：
    - reshape 为 [B * H * W, T, C]
    - 可选经过 PhaseWarpFrontEnd
    - TimeMixer 预测未来每个特征的轨迹
    - 用 target_head 把未来特征压成未来 O3

    输出：
    - [B, horizon, H, W]
    """

    def __init__(self, seq_len, pred_len, lat_size, lon_size, use_phase_warp, d_model, e_layers, moving_avg, down_sampling_window, down_sampling_layers, dropout):
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
        self.backbone = TimeMixerBackbone(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars, d_model=d_model, e_layers=e_layers, moving_avg=moving_avg, down_sampling_window=down_sampling_window, down_sampling_layers=down_sampling_layers, dropout=dropout, use_norm=True)
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
    r2 = 1 - ss_res / ss_tot
    smape = np.mean(2.0 * np.abs(pred_flat - true_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-06))
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'smape': smape}

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, window, horizon, d_model, e_layers, moving_avg, down_sampling_window, down_sampling_layers, dropout, epochs, learning_rate, early_stopping_patience, base_dir):
    """
    训练一组 TimeMixer 实验，并返回评估指标。
    受控变量只有一个：是否启用 phase-warp 前端。
    """
    print(f'\n[Experiment] {label}')
    model = GridPointTimeMixerO3(seq_len=window, pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, d_model=d_model, e_layers=e_layers, moving_avg=moving_avg, down_sampling_window=down_sampling_window, down_sampling_layers=down_sampling_layers, dropout=dropout).to(device)
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
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'TimeMixer_PhaseWarp_Compare.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    window = 3
    horizon = 3
    batch_size = 4
    d_model = 64
    e_layers = 2
    moving_avg = 3
    down_sampling_window = 2
    down_sampling_layers = 1
    dropout = 0.1
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
        raw_metrics = train_and_evaluate(label=f'TimeMixer_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, window=window, horizon=horizon, d_model=d_model, e_layers=e_layers, moving_avg=moving_avg, down_sampling_window=down_sampling_window, down_sampling_layers=down_sampling_layers, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        raw_seed_metrics.append(raw_metrics)
        set_experiment_seed(seed)
        phase_metrics = train_and_evaluate(label=f'TimeMixer_PhaseWarp_seed{seed}', use_phase_warp=True, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, window=window, horizon=horizon, d_model=d_model, e_layers=e_layers, moving_avg=moving_avg, down_sampling_window=down_sampling_window, down_sampling_layers=down_sampling_layers, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        phase_seed_metrics.append(phase_metrics)
    raw_summary = summarize_seed_metrics(raw_seed_metrics)
    phase_summary = summarize_seed_metrics(phase_seed_metrics)
    print('\n[Comparison Summary]')
    print_seed_summary('TimeMixer_Raw', raw_summary)
    print_seed_summary('TimeMixer_PhaseWarp', phase_summary)
    print(f"RMSE improvement: {raw_summary['rmse']['mean'] - phase_summary['rmse']['mean']:.4f}")
    print(f"MAE improvement : {raw_summary['mae']['mean'] - phase_summary['mae']['mean']:.4f}")
    print(f"R^2 gain        : {phase_summary['r2']['mean'] - raw_summary['r2']['mean']:.4f}")
    print(f"SMAPE gain      : {raw_summary['smape']['mean'] - phase_summary['smape']['mean']:.2%}")
