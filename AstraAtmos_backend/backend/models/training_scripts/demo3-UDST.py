import argparse
import json
import sys
import os
import glob
import re
import numpy as np
import netCDF4 as nc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self): pass

try:
    import matplotlib
    matplotlib.use('Agg')
except: pass

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--stlstm_hidden_dims", type=str, default="[64, 64, 64]")
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--window", type=int, default=3)
parser.add_argument("--horizon", type=int, default=3)
parser.add_argument("--early_stopping_patience", type=int, default=0)
args, unknown = parser.parse_known_args()

epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
try:
    hidden_dims = json.loads(args.stlstm_hidden_dims.replace("'", "\""))
except:
    hidden_dims = [64, 64, 64]
early_stopping_patience = args.early_stopping_patience

os.makedirs(os.path.join(base_dir, "models", "training_logs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "models", "training_results"), exist_ok=True)
sys.stdout = Logger(os.path.join(base_dir, "models", "training_logs", "UDST.txt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Device: {device}")
print(f"Config: Epochs={epochs}, BatchSize={batch_size}, LR={learning_rate}, HiddenDims={hidden_dims}")

openmars_dir = os.environ.get("ASTRAATMOS_OPENMARS_DIR", os.path.join(base_dir, "data", "openmars"))
mcd_dir = os.environ.get("ASTRAATMOS_MCD_DIR", os.path.join(base_dir, "data", "MCD"))
window, horizon = args.window, args.horizon
ST_H, ST_W = 36, 72 # 使用 ST_ 前缀避免冲突

print("\n[Step 1] Loading OpenMars Data (Global)...")
o3_list, om_ls_list = [], []
def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
file_list = sorted(glob.glob(os.path.join(openmars_dir, "*.nc")), key=natural_sort_key)
ref_ds = nc.Dataset(file_list[0])
for f in file_list:
    ds = nc.Dataset(f); o3_list.append(ds.variables['o3col'][:])
    ls_v = ds.variables['Ls'][:] if 'Ls' in ds.variables else ds.variables['ls'][:]
    om_ls_list.append(ls_v); ds.close()
y_raw = np.concatenate(o3_list, axis=0)
om_ls_raw = np.concatenate(om_ls_list, axis=0)

print("\n[Step 2] Loading MCD Data...")
mcd_vars_map = {'U':('U_Wind','u'), 'V':('V_Wind','v'), 'D':('Dust_Optical_Depth','dustq'), 'S':('Solar_Flux_DN','fluxsurf_dn_sw'), 'T':('Temperature','temp')}
active_vars = [c for c in "UDST" if c in mcd_vars_map]
short_names = [mcd_vars_map[c][1] for c in active_vars]
mcd_data = {sn: [] for sn in short_names}; mcd_ls = []
for f_nc in [os.path.join(mcd_dir, "MCD_MY27_Lat-90-90_real.nc"), os.path.join(mcd_dir, "MCD_MY28_Lat-90-90_real.nc")]:
    if not os.path.exists(f_nc): continue
    ds = nc.Dataset(f_nc)
    for c_v in active_vars:
        mv, sn = mcd_vars_map[c_v]; d = ds.variables[mv][:]
        mcd_data[sn].append(d.reshape(d.shape[0]*d.shape[1], d.shape[2], d.shape[3]))
    ls_t = ds.variables['Ls'][:] if 'Ls' in ds.variables else ds.variables['ls'][:]
    s_d, h_d = ds.variables[mcd_vars_map[active_vars[0]][0]].shape[:2]
    ls_e = np.zeros(s_d * h_d)
    for i in range(s_d):
        l_s = ls_t[i]; l_e = ls_t[i+1] if i < s_d-1 else l_s + 0.5
        ls_e[i*h_d:(i+1)*h_d] = np.linspace(l_s, l_e, h_d, endpoint=False)
    mcd_ls.append(ls_e % 360.0); ds.close()
vars_dict = {k: np.nan_to_num(np.concatenate(mcd_data[k], axis=0), nan=0.0) for k in short_names}
def unwrap(ls_in):
    out = np.copy(ls_in); off = 0
    for j in range(1, len(out)):
        if ls_in[j] < ls_in[j-1] - 180: off += 360
        out[j] += off
    return out
mcd_ls_c = unwrap(np.concatenate(mcd_ls)); om_ls_c = unwrap(om_ls_raw)
for k in vars_dict:
    vars_dict[k] = interp1d(mcd_ls_c, vars_dict[k], axis=0, bounds_error=False, fill_value="extrapolate")(om_ls_c)
            
# ========================================
# 5. 构建 X, y 数据集
# ========================================
X_raw = np.stack([y_raw, vars_dict['u'], vars_dict['dustq'], vars_dict['fluxsurf_dn_sw'], vars_dict['temp']], axis=-1)
T, H_raw, W_raw, C_feat = X_raw.shape
print(f"最终数据集 X_raw: {X_raw.shape}")

split_idx = int(0.8 * (T - window - horizon + 1)) + window
X_scaled = np.zeros_like(X_raw)
for c in range(C_feat):
    scaler = StandardScaler()
    scaler.fit(X_raw[:split_idx, ..., c].reshape(split_idx, -1))
    X_scaled[..., c] = scaler.transform(X_raw[..., c].reshape(T, -1)).reshape(T, H_raw, W_raw)
y_train_part = y_raw[:split_idx]
y_mean, y_std = y_train_part.mean(), y_train_part.std()
y_scaled = (y_raw - y_mean) / (y_std + 1e-6)

X_seq, y_seq = [], []
for i in range(T - window - horizon + 1):
    X_seq.append(X_scaled[i: i + window])
    y_seq.append(y_scaled[i + window: i + window + horizon])

X_torch = torch.tensor(np.array(X_seq)).permute(0, 1, 4, 2, 3).float()
y_torch = torch.tensor(np.array(y_seq)).unsqueeze(2).float()
split = int(0.8 * len(X_torch))
train_loader = DataLoader(TensorDataset(X_torch[:split], y_torch[:split]), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_torch[split:], y_torch[split:]), batch_size=batch_size, shuffle=False)

# ========================================
# 6. 模型定义 (PredRNNv2 Optimized)
# ========================================
class SpatioTemporalLSTMCellv2(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size=3):
        super().__init__()
        padding = filter_size // 2
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, filter_size, padding=padding)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, filter_size, padding=padding)
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, filter_size, padding=padding)
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, filter_size, padding=padding)
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, 1)
        self.num_hidden = num_hidden

    def forward(self, x, h, c, m):
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)
        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_concat, self.num_hidden, 1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, 1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, 1)
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + 1.0)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c + i_t * g_t
        i_tp = torch.sigmoid(i_xp + i_m)
        f_tp = torch.sigmoid(f_xp + f_m + 1.0)
        g_tp = torch.tanh(g_xp + g_m)
        m_new = f_tp * m + i_tp * g_tp
        mem = torch.cat([c_new, m_new], dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new

class PredRNNv2(nn.Module):
    def __init__(self, input_dim, hidden_dims, height, width, horizon=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_ch = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(SpatioTemporalLSTMCellv2(in_ch, hidden_dims[i], height, width))
        self.conv_last = nn.Conv2d(hidden_dims[-1], 1, 1)
        self.horizon = horizon
        self.hidden_dims = hidden_dims

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = [torch.zeros(B, d, H, W, device=x.device) for d in self.hidden_dims]
        c = [torch.zeros_like(h[i]) for i in range(len(h))]
        m = [torch.zeros_like(h[i]) for i in range(len(h))] # 每层独立的 M
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i], m[i] = cell(inp, h[i], c[i], m[i])
                inp = h[i]
        preds = []
        dec_inp = x[:, -1]
        for _ in range(self.horizon):
            inp = dec_inp
            for i, cell in enumerate(self.layers):
                h[i], c[i], m[i] = cell(inp, h[i], c[i], m[i])
                inp = h[i]
            preds.append(self.conv_last(h[-1]))
        return torch.stack(preds, dim=1)

model = PredRNNv2(input_dim=C_feat, hidden_dims=hidden_dims, height=ST_H, width=ST_W, horizon=horizon).to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.SmoothL1Loss()

print("\n[Step 3] Start Training...")
best_val_loss = float('inf')
patience_counter = 0
for ep in range(epochs):
    model.train()
    loss_sum = 0
    for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); pred = model(xb); loss = criterion(pred, yb)
        loss.backward(); opt.step(); loss_sum += loss.item()
        if batch_idx % 20 == 0 or batch_idx == len(train_loader):
            print(f"Epoch {ep+1}/{epochs} Batch {batch_idx}/{len(train_loader)} Loss={loss.item():.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xv, yv in test_loader:
            xv, yv = xv.to(device), yv.to(device)
            p = model(xv); l = criterion(p, yv)
            val_loss += l.item()
    
    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch {ep+1}/{epochs} Loss={loss_sum/len(train_loader):.4f} Val Loss={avg_val_loss:.4f}")

    # Early stopping
    if early_stopping_patience > 0:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[Early Stopping] Val loss did not improve for {early_stopping_patience} epochs. Stopped at epoch {ep+1}.")
                break

model.eval()
trues, preds_all = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device); pred = model(xb)
        trues.append(yb.numpy()); preds_all.append(pred.cpu().numpy())
trues = np.concatenate(trues, axis=0); preds = np.concatenate(preds_all, axis=0)
y_true = trues.flatten()
y_pred = preds.flatten()
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

print(f"\nMetrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-Squared: {r2:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"SMAPE: {smape:.4f}%")
save_path = args.output_path or os.path.join(base_dir, "models", "training_results", "predrnn_UDST.pth")
torch.save(model.state_dict(), save_path)
print(f"模型已保存: {save_path}")
