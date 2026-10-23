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
    这类 compare 脚本主要目的是做受控实验，因此早停逻辑保持简洁透明即可。
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
    """
    把 MCD 数据从 [sol, hour, lat, lon] 展平为 [time, lat, lon]。
    后面需要和 OpenMars 共用一条时间轴，因此先统一时间维组织形式。
    """
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
    把循环的 Ls 序列展开成连续轴，方便后续插值。
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
    读取 OpenMars 臭氧与 MCD 气象变量，并把它们对齐到同一条 OpenMars 时间轴上。

    返回：
    - x_raw: [T, H, W, 5]
      通道顺序为 [O3, U, V, Temperature, Solar_Flux]
    - y_raw: [T, H, W]
      目标臭氧场
    - om_ls_continuous: [T]
      与 x_raw / y_raw 完整对齐的连续 Ls 序列
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
            ls_expanded = np.zeros(sols * hours, dtype=np.float32)
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

    这样所有模型都只看历史，不偷看未来气象量，便于做跨 backbone 的公平比较。
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

class GradientHighwayUnit(nn.Module):
    """
    GHU 是 PredRNN++ 里专门用来改善长时间梯度传播的模块。

    可以把它理解成：
    - 一条额外的“高速通道” z；
    - 由当前输入表征 x 和已有 z 共同决定新的门控更新；
    - 这样时间深度很大时，梯度不必完全依赖层层递归单元硬传回去。
    """

    def __init__(self, input_dim, hidden_dim, filter_size=3):
        super().__init__()
        padding = filter_size // 2
        self.conv_x = nn.Conv2d(input_dim, hidden_dim * 2, filter_size, padding=padding)
        self.conv_z = nn.Conv2d(hidden_dim, hidden_dim * 2, filter_size, padding=padding)

    def forward(self, x, z):
        x_concat = self.conv_x(x)
        z_concat = self.conv_z(z)
        (p, u) = torch.chunk(x_concat + z_concat, 2, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1.0 - u) * z
        return z_new

class CausalLSTMCell(nn.Module):
    """
    PredRNN++ 的核心递归单元：Causal LSTM。

    与普通 ConvLSTM 相比，它更强调两类记忆的层级传递：
    - c：层内的时序记忆；
    - m：跨层、跨时间共同流动的时空记忆。

    PredRNN++ 正是通过这种“更深的记忆路径”来增强复杂时空依赖建模能力。
    """

    def __init__(self, input_dim, hidden_dim, memory_dim, filter_size=3):
        super().__init__()
        padding = filter_size // 2
        self.hidden_dim = hidden_dim
        self.conv_x = nn.Conv2d(input_dim, hidden_dim * 7, filter_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_dim, hidden_dim * 4, filter_size, padding=padding)
        self.conv_c = nn.Conv2d(hidden_dim, hidden_dim * 3, filter_size, padding=padding)
        self.conv_m = nn.Conv2d(memory_dim, hidden_dim * 3, filter_size, padding=padding)
        self.conv_c2m = nn.Conv2d(hidden_dim, hidden_dim * 4, filter_size, padding=padding)
        self.conv_m2o = nn.Conv2d(hidden_dim, hidden_dim, filter_size, padding=padding)
        self.conv_mem = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

    def forward(self, x, h, c, m):
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        c_concat = self.conv_c(c)
        m_concat = self.conv_m(m)
        (i_x, g_x, f_x, o_x, i_xp, g_xp, f_xp) = torch.split(x_concat, self.hidden_dim, dim=1)
        (i_h, g_h, f_h, o_h) = torch.split(h_concat, self.hidden_dim, dim=1)
        (i_c, g_c, f_c) = torch.split(c_concat, self.hidden_dim, dim=1)
        (i_m, f_m, m_m) = torch.split(m_concat, self.hidden_dim, dim=1)
        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + 1.0)
        g_t = torch.tanh(g_x + g_h + g_c)
        c_new = f_t * c + i_t * g_t
        c2m_concat = self.conv_c2m(c_new)
        (i_c2m, g_c2m, f_c2m, o_c) = torch.split(c2m_concat, self.hidden_dim, dim=1)
        i_tp = torch.sigmoid(i_c2m + i_xp + i_m)
        f_tp = torch.sigmoid(f_c2m + f_xp + f_m + 1.0)
        g_tp = torch.tanh(g_c2m + g_xp)
        m_new = f_tp * torch.tanh(m_m) + i_tp * g_tp
        o_m = self.conv_m2o(m_new)
        o_t = torch.tanh(o_x + o_h + o_c + o_m)
        merged_memory = torch.cat([c_new, m_new], dim=1)
        cell = self.conv_mem(merged_memory)
        h_new = o_t * torch.tanh(cell)
        return (h_new, c_new, m_new)

class PredRNNPPForecaster(nn.Module):
    """
    面向当前火星臭氧任务的 PredRNN++ 风格预测器。

    整体流程：
    1. 若启用 Phase Warp，则先把 5 通道输入变成 9 通道相位扭曲特征；
    2. 历史序列先通过第 1 层 Causal LSTM；
    3. 然后进入 GHU，形成更利于长时梯度传播的高速通道 z；
    4. 再交给后续 Causal LSTM 层继续编码；
    5. 用最后一层最后时刻的隐藏状态，直接映射出未来 pred_len 张臭氧图。

    这仍然是“PredRNN++ 风格 compare 版”，不是逐帧自回归视频预测原版。
    这么做的原因是：
    - 你当前所有对比脚本都统一采用“仅看历史窗口”的协议；
    - 这样能避免未来气象变量泄露；
    - 也更便于把性能增益归因到相位扭曲模块，而不是解码策略差异。
    """

    def __init__(self, pred_len, lat_size, lon_size, use_phase_warp, hidden_dims, filter_size=3):
        super().__init__()
        if len(hidden_dims) < 2:
            raise ValueError('PredRNN++ 至少需要 2 层 CausalLSTM，才能在第 1 层和第 2 层之间插入 GHU。')
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
        for (layer_idx, hidden_dim) in enumerate(self.hidden_dims):
            current_input_dim = input_dim if layer_idx == 0 else self.hidden_dims[layer_idx - 1]
            current_memory_dim = self.hidden_dims[-1] if layer_idx == 0 else self.hidden_dims[layer_idx - 1]
            self.cells.append(CausalLSTMCell(input_dim=current_input_dim, hidden_dim=hidden_dim, memory_dim=current_memory_dim, filter_size=filter_size))
        self.ghu = GradientHighwayUnit(self.hidden_dims[0], self.hidden_dims[0], filter_size=filter_size)
        self.forecast_head = nn.Conv2d(self.hidden_dims[-1], pred_len, kernel_size=1)

    def _init_states(self, batch_size, device):
        """初始化所有层的 h / c，以及 GHU 状态 z 和跨层记忆 m。"""
        h_states = []
        c_states = []
        for hidden_dim in self.hidden_dims:
            h_state = torch.zeros(batch_size, hidden_dim, self.lat_size, self.lon_size, device=device)
            c_state = torch.zeros_like(h_state)
            h_states.append(h_state)
            c_states.append(c_state)
        memory = torch.zeros(batch_size, self.hidden_dims[-1], self.lat_size, self.lon_size, device=device)
        z_state = torch.zeros(batch_size, self.hidden_dims[0], self.lat_size, self.lon_size, device=device)
        return (h_states, c_states, memory, z_state)

    def forward(self, x, ls):
        if self.phase_warp is not None:
            features = self.phase_warp(x, ls)
        else:
            features = x
        (batch_size, seq_len, _, _, _) = features.shape
        (h_states, c_states, memory, z_state) = self._init_states(batch_size, features.device)
        for time_idx in range(seq_len):
            current = features[:, time_idx]
            (h_next, c_next, memory) = self.cells[0](current, h_states[0], c_states[0], memory)
            h_states[0] = h_next
            c_states[0] = c_next
            z_state = self.ghu(h_states[0], z_state)
            current = z_state
            (h_next, c_next, memory) = self.cells[1](current, h_states[1], c_states[1], memory)
            h_states[1] = h_next
            c_states[1] = c_next
            current = h_next
            for layer_idx in range(2, len(self.cells)):
                (h_next, c_next, memory) = self.cells[layer_idx](current, h_states[layer_idx], c_states[layer_idx], memory)
                h_states[layer_idx] = h_next
                c_states[layer_idx] = c_next
                current = h_next
        last_hidden = h_states[-1]
        return self.forecast_head(last_hidden)

def evaluate_metrics(model, loader, device, y_std, y_mean):
    """把标准化空间里的预测还原到物理量空间后，再计算 RMSE / MAE / R^2 / SMAPE。"""
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
    y_pred_phys = preds * (y_std + 1e-06) + y_mean
    y_true_phys = trues * (y_std + 1e-06) + y_mean
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

def train_and_evaluate(label, use_phase_warp, train_loader, test_loader, device, y_std, y_mean, lat_size, lon_size, horizon, hidden_dims, filter_size, epochs, learning_rate, early_stopping_patience, base_dir):
    """
    训练一组 PredRNN++ 对比实验，并返回评估指标。
    两组实验之间唯一改变的是：是否启用相位扭曲前端。
    """
    print(f'\n[Experiment] {label}')
    model = PredRNNPPForecaster(pred_len=horizon, lat_size=lat_size, lon_size=lon_size, use_phase_warp=use_phase_warp, hidden_dims=hidden_dims, filter_size=filter_size).to(device)
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
    sys.stdout = Logger(os.path.join(base_dir, 'models', 'training_logs', 'PredRNNPP_PhaseWarp_Compare.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Device: {device}')
    window = 3
    horizon = 3
    batch_size = 4
    hidden_dim = 32
    num_layers = 3
    if num_layers < 2:
        raise ValueError('PredRNN++ 需要至少 2 层。')
    hidden_dims = [hidden_dim] * num_layers
    filter_size = 3
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
        raw_metrics = train_and_evaluate(label=f'PredRNNPP_Raw_seed{seed}', use_phase_warp=False, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, hidden_dims=hidden_dims, filter_size=filter_size, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        raw_seed_metrics.append(raw_metrics)
        set_experiment_seed(seed)
        phase_metrics = train_and_evaluate(label=f'PredRNNPP_PhaseWarp_seed{seed}', use_phase_warp=True, train_loader=train_loader, test_loader=test_loader, device=device, y_std=y_std, y_mean=y_mean, lat_size=lat_size, lon_size=lon_size, horizon=horizon, hidden_dims=hidden_dims, filter_size=filter_size, epochs=epochs, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, base_dir=base_dir)
        phase_seed_metrics.append(phase_metrics)
    raw_summary = summarize_seed_metrics(raw_seed_metrics)
    phase_summary = summarize_seed_metrics(phase_seed_metrics)
    print('\n[Comparison Summary]')
    print_seed_summary('PredRNNPP_Raw', raw_summary)
    print_seed_summary('PredRNNPP_PhaseWarp', phase_summary)
    print(f"RMSE improvement: {raw_summary['rmse']['mean'] - phase_summary['rmse']['mean']:.4f}")
    print(f"MAE improvement : {raw_summary['mae']['mean'] - phase_summary['mae']['mean']:.4f}")
    print(f"R^2 gain        : {phase_summary['r2']['mean'] - raw_summary['r2']['mean']:.4f}")
    print(f"SMAPE gain      : {raw_summary['smape']['mean'] - phase_summary['smape']['mean']:.2%}")
