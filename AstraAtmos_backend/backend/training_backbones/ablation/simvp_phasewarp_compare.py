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

class BasicConv2d(nn.Module):
    """
    SimVP 风格基础卷积模块。

    用途：
    - 普通卷积：用于编码、时序建模器内部的通道变换
    - 反卷积：用于解码阶段的上采样
    """

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
    """
    SimVP 风格的隐空间时序块。

    这里参考官方模型里“Inception 风格多尺度卷积”的思想：
    - 用多个不同大小的卷积核并行抽取隐空间模式；
    - 再把它们拼接后投影回原通道数；
    - 最后做残差连接。

    注意：
    - 这里的输入已经把时间维与通道维合并，因此块内部主要处理的是
      “压缩后的时空隐表示”。
    """

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
    """
    SimVP 的空间编码器。

    输入：
    - [B*T, C, H, W]

    输出：
    - [B*T, hid_S, H', W']

    作用：
    - 先把原始高分辨率网格压缩到更紧凑的隐空间；
    - 后续时序建模就在这个低分辨率隐空间里进行，以降低计算量。
    """

    def __init__(self, in_channels, hid_S, num_downsample):
        super().__init__()
        self.proj = BasicConv2d(in_channels, hid_S, kernel_size=3, stride=1, transpose=False, activate=True)
        self.down_blocks = nn.ModuleList([BasicConv2d(hid_S, hid_S, kernel_size=3, stride=2, transpose=False, activate=True) for _ in range(num_downsample)])

    def forward(self, x):
        x = self.proj(x)
        for block in self.down_blocks:
            x = block(x)
        return x

class TemporalTranslator(nn.Module):
    """
    SimVP 的隐空间时序建模器。

    输入：
    - z: [B, seq_len, hid_S, H', W']

    做法：
    1. 把时间维和通道维合并成 [B, seq_len * hid_S, H', W']；
    2. 在隐空间上堆叠多个 InceptionTemporalBlock；
    3. 最后投影成 [B, pred_len * hid_S, H', W']；
    4. 再还原为 [B, pred_len, hid_S, H', W']。

    这对应 SimVP 中“在编码后的隐空间里做时序演化建模”的核心思路。
    """

    def __init__(self, seq_len, pred_len, hid_S, hid_T, temporal_depth, dropout):
        super().__init__()
        self.pred_len = pred_len
        self.hid_S = hid_S
        in_channels = seq_len * hid_S
        self.in_proj = BasicConv2d(in_channels, hid_T, kernel_size=1, stride=1, transpose=False, activate=True)
        self.blocks = nn.ModuleList([InceptionTemporalBlock(channels=hid_T, hidden_channels=hid_T, dropout=dropout) for _ in range(temporal_depth)])
        self.out_proj = nn.Sequential(nn.Conv2d(hid_T, pred_len * hid_S, kernel_size=1, stride=1, padding=0), nn.GroupNorm(1, pred_len * hid_S))

    def forward(self, z):
        (batch_size, seq_len, hid_S, height, width) = z.shape
        x = z.reshape(batch_size, seq_len * hid_S, height, width)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)
        return x.view(batch_size, self.pred_len, self.hid_S, height, width)

class SpatialDecoder(nn.Module):
    """
    SimVP 的空间解码器。

    输入：
    - [B*pred_len, hid_S, H', W']

    输出：
    - [B*pred_len, 1, H, W]

    做法：
    - 用与编码器对称的上采样模块把隐空间特征还原回原始网格大小；
    - 最后用 1x1 卷积映射成单通道臭氧图。
    """

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

class SimVPForecaster(nn.Module):
    """
    一个适合当前项目的 SimVP 风格臭氧预测模型。

    流程：
    1. 可选经过 PhaseWarpFrontEnd，把 5 通道扩展成 9 通道；
    2. 对每个历史时间步分别做空间编码；
    3. 在低分辨率隐空间里做时序建模；
    4. 再把未来隐状态解码回未来 O3 网格图。

    输出：
    - [B, pred_len, H, W]
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
        self.translator = TemporalTranslator(seq_len=seq_len, pred_len=pred_len, hid_S=spatial_hidden_dim, hid_T=temporal_hidden_dim, temporal_depth=temporal_depth, dropout=dropout)
        self.decoder = SpatialDecoder(hid_S=spatial_hidden_dim, num_upsample=num_downsample)

    def forward(self, x, ls):
        if self.phase_warp is not None:
            features = self.phase_warp(x, ls)
        else:
            features = x
        (batch_size, seq_len, channels, height, width) = features.shape
        z = features.reshape(batch_size * seq_len, channels, height, width)
        z = self.encoder(z)
        (_, hid_S, hid_h, hid_w) = z.shape
        z = z.view(batch_size, seq_len, hid_S, hid_h, hid_w)
        future_z = self.translator(z)
        future_z = future_z.reshape(batch_size * self.pred_len, hid_S, hid_h, hid_w)
        out = self.decoder(future_z, out_height=height, out_width=width)
        return out.view(batch_size, self.pred_len, height, width)

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

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, horizon, window, spatial_hidden_dim, temporal_hidden_dim, num_downsample, temporal_depth, dropout, epochs, learning_rate, early_stopping_patience, base_dir):
    """
    训练一组 SimVP 实验，并返回评估指标。
    受控变量只有一个：是否启用 phase-warp 前端。
    """
    print(f'\n[Experiment] {label}')
    model = SimVPForecaster(seq_len=window, pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, num_downsample=num_downsample, temporal_depth=temporal_depth, dropout=dropout).to(device)
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
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'SimVP_PhaseWarp_Compare.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
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
    raw_seed_metrics = []
    phase_seed_metrics = []
    for seed in RUN_SEEDS:
        print(f'\n[Seed] {seed}')
        set_experiment_seed(seed)
        raw_metrics = train_and_evaluate(label=f'SimVP_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, window=window, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, num_downsample=num_downsample, temporal_depth=temporal_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        raw_seed_metrics.append(raw_metrics)
        set_experiment_seed(seed)
        phase_metrics = train_and_evaluate(label=f'SimVP_PhaseWarp_seed{seed}', use_phase_warp=True, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, window=window, spatial_hidden_dim=spatial_hidden_dim, temporal_hidden_dim=temporal_hidden_dim, num_downsample=num_downsample, temporal_depth=temporal_depth, dropout=dropout, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        phase_seed_metrics.append(phase_metrics)
    raw_summary = summarize_seed_metrics(raw_seed_metrics)
    phase_summary = summarize_seed_metrics(phase_seed_metrics)
    print('\n[Comparison Summary]')
    print_seed_summary('SimVP_Raw', raw_summary)
    print_seed_summary('SimVP_PhaseWarp', phase_summary)
    print(f"RMSE improvement: {raw_summary['rmse']['mean'] - phase_summary['rmse']['mean']:.4f}")
    print(f"MAE improvement : {raw_summary['mae']['mean'] - phase_summary['mae']['mean']:.4f}")
    print(f"R^2 gain        : {phase_summary['r2']['mean'] - raw_summary['r2']['mean']:.4f}")
    print(f"SMAPE gain      : {raw_summary['smape']['mean'] - phase_summary['smape']['mean']:.2%}")
