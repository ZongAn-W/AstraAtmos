/**
 * 实验中心验收夹具：建立 / 清理一次性验收档案。
 *
 * 训练页的浏览器验收需要「一个登录账号 + 一个通过校验的上传模型 + 若干训练任务」。
 * 本脚本只使用既有接口与数据库模型，不使用任何测试专用后门：
 *
 *   prepare  在本地开发库里创建临时验收账号（邮箱后缀固定，便于识别），
 *            登录取得 JWT，并生成 + 上传一个通过校验的探针模型，
 *            可选注入一条 running 与一条 completed 任务（含真实日志文件）。
 *   cleanup  删除该账号及其上传模型、训练任务、日志文件与任务标签。
 *
 * 用法（后端工作目录为 AstraAtmos_backend/backend）：
 *   & '<conda env python>' scripts/audit/experiment-console/seed-probe-profile.mjs --help
 *
 * 注意：本脚本面向本地开发库，不要对生产库运行。验证完成后请执行 cleanup。
 */

import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
// scripts/audit/experiment-console -> 仓库根
const REPO_ROOT = resolve(HERE, '..', '..', '..');
const BACKEND_DIR = join(REPO_ROOT, 'AstraAtmos_backend', 'backend');
const STATE_FILE = join(HERE, '.probe-profile.json');
const MODEL_FILE = join(HERE, '.probe-surface-net.py');
const LOG_DIR = join(BACKEND_DIR, 'data', 'probe_logs');

const EMAIL = 'experiment.console.probe@astraatmos.local';
const PASSWORD = 'ProbeConsole#2026';
const RUNNING_NAME = 'PROBE 控制台监控验收';
const DONE_NAME = 'PROBE 结果工作区验收';

const argv = process.argv.slice(2);
const command = argv[0] || 'prepare';
const hasFlag = (name) => argv.includes(`--${name}`);
const valueOf = (name, fallback) => {
  const index = argv.indexOf(`--${name}`);
  return index >= 0 && argv[index + 1] ? argv[index + 1] : fallback;
};
const PYTHON = valueOf('python', process.env.ASTRAATMOS_PYTHON || '$env:CONDA_PREFIX\python.exe');
const API = valueOf('api', process.env.ASTRAATMOS_API || 'http://127.0.0.1:8000/api');

if (hasFlag('help') || command === 'help') {
  console.log(`实验中心验收夹具

用法：
  node scripts/audit/experiment-console/seed-probe-profile.mjs prepare [--python <exe>] [--api <url>] [--no-tasks]
  node scripts/audit/experiment-console/seed-probe-profile.mjs cleanup [--python <exe>]

prepare 会写出 ${STATE_FILE}（含 JWT 与任务 id，已在 .gitignore 之外，请勿提交）；
cleanup 会删除该文件、探针模型文件、日志目录与数据库记录。
`);
  process.exit(0);
}

const MODEL_SOURCE = `import torch
from torch import nn

MODEL_SPEC = {
    "name": "ProbeSurfaceNet",
    "description": "Verification probe model for the experiment center console audit.",
    "parameters": {
        "probe_alpha": {"type": "int", "default": 23, "min": 1, "max": 128},
        "probe_ratio": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0},
        "probe_flag": {"type": "bool", "default": True},
    },
}


class ProbeSurfaceNet(nn.Module):
    """逐像素持久性基线：把最后一帧的臭氧通道复制到整个预测步长。"""

    def __init__(self, horizon, probe_alpha=23, probe_ratio=0.35, probe_flag=True):
        super().__init__()
        self.horizon = horizon
        self.probe_alpha = probe_alpha
        self.probe_ratio = probe_ratio
        self.probe_flag = probe_flag

    def forward(self, x):
        last = x[:, -1, :1]
        return last.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)


def build_model(config):
    return ProbeSurfaceNet(
        config["horizon"],
        probe_alpha=config.get("probe_alpha", 23),
        probe_ratio=config.get("probe_ratio", 0.35),
        probe_flag=config.get("probe_flag", True),
    )
`;

const RUNNING_LOG = [
  '[2026-01-01 10:00:01] device: cuda:0 | torch 2.5.1',
  '[2026-01-01 10:00:02] dataset=openmars_mcd window=3 horizon=3 channels=O3,U',
  '[2026-01-01 10:00:03] model_source=uploaded uploaded_model=ProbeSurfaceNet v1',
  '[2026-01-01 10:00:05] epoch 1/8 step 10 loss=0.4821 val_loss=0.4510 lr=0.00100',
  '[2026-01-01 10:00:12] epoch 1/8 step 20 loss=0.4337 val_loss=0.4218 lr=0.00100',
  '[2026-01-01 10:00:20] epoch 1/8 done loss=0.4102 val_loss=0.4021',
  '[2026-01-01 10:00:28] epoch 2/8 step 10 loss=0.3980 val_loss=0.3902 lr=0.00100',
  '[2026-01-01 10:00:36] epoch 2/8 step 20 loss=0.3811 val_loss=0.3790 lr=0.00100',
  '[2026-01-01 10:00:44] epoch 2/8 done loss=0.3702 val_loss=0.3711',
  '[2026-01-01 10:00:52] epoch 3/8 step 10 loss=0.3615 val_loss=0.3660 lr=0.00100',
  '[2026-01-01 11:00:00] epoch 3/8 step 20 loss=0.3520 val_loss=0.3588 lr=0.00100',
  '[2026-01-01 11:00:08] epoch 3/8 done loss=0.3441 val_loss=0.3510',
  '[2026-01-01 11:00:16] epoch 4/8 step 10 loss=0.3388 val_loss=0.3470 lr=0.00100',
  'warning: val loss plateaued for 2 epochs, early stopping patience is 5',
  '[2026-01-01 11:00:24] epoch 4/8 done loss=0.3302 val_loss=0.3421',
];

const DONE_LOG = RUNNING_LOG.concat([
  '[2026-01-01 11:00:32] epoch 5/8 done loss=0.3241 val_loss=0.3388',
  '[2026-01-01 11:00:40] epoch 6/8 done loss=0.3180 val_loss=0.3350',
  '[2026-01-01 11:00:48] epoch 7/8 done loss=0.3122 val_loss=0.3321',
  '[2026-01-01 11:00:56] epoch 8/8 done loss=0.3088 val_loss=0.3301',
  '[2026-01-01 11:01:00] best checkpoint saved: models/checkpoints/probe_surface_net_best.pth',
  '[2026-01-01 11:01:02] evaluation RMSE=0.0412 MAE=0.0288 R2=0.9312',
]);

function runPython(script) {
  const result = spawnSync(PYTHON, ['-'], {
    cwd: BACKEND_DIR,
    input: script,
    encoding: 'utf8',
    stdio: ['pipe', 'inherit', 'inherit'],
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`python 退出码 ${result.status}`);
}

function prepareDatabaseScript({ uploadedModelId, withTasks }) {
  const payload = JSON.stringify({
    email: EMAIL,
    password: PASSWORD,
    runningName: RUNNING_NAME,
    doneName: DONE_NAME,
    runningLog: join(LOG_DIR, 'probe_running.log'),
    doneLog: join(LOG_DIR, 'probe_done.log'),
    uploadedModelId,
    withTasks,
  });
  return `
import asyncio, json, os, sys
from datetime import datetime, timedelta
from pathlib import Path

BACKEND = Path(${JSON.stringify(BACKEND_DIR)})
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from sqlalchemy import text
from database.engine import async_session_maker

CONF = json.loads(${JSON.stringify(payload)})

RUNNING_HYPERS = {
    "epochs": 8, "batch_size": 16, "learning_rate": 0.001, "window": 3, "horizon": 3,
    "seed": 11, "early_stopping_patience": 5, "selected_channels": ["U"],
    "model_source": "uploaded", "model_architecture": "predrnnv2", "use_sphere": False,
    "training_dataset": "openmars_mcd",
    "custom_model_params": {"probe_alpha": 23, "probe_ratio": 0.35, "probe_flag": True},
}
RUNNING_LOSS = {"train": [0.4821, 0.4102, 0.3702, 0.3441, 0.3302], "val": [0.4510, 0.4021, 0.3711, 0.3510, 0.3421]}
DONE_LOSS = {"train": [0.4821, 0.4102, 0.3702, 0.3441, 0.3302, 0.3241, 0.3180, 0.3088],
             "val": [0.4510, 0.4021, 0.3711, 0.3510, 0.3421, 0.3388, 0.3350, 0.3301]}

def pack(hypers, channels):
    merged = dict(hypers)
    merged["selected_channels"] = channels
    merged["_uploaded_model_id"] = CONF["uploadedModelId"]
    merged["_uploaded_model_name"] = "ProbeSurfaceNet"
    merged["_uploaded_model_version"] = 1
    return json.dumps(merged, ensure_ascii=False)

async def cleanup_tasks(session, user_id):
    rows = (await session.execute(text(
        "SELECT id, log_file_path FROM model_training_tasks WHERE user_id = :uid AND custom_model_name LIKE 'PROBE%'"
    ), {"uid": user_id})).all()
    for _, path in rows:
        if path and os.path.exists(path):
            os.remove(path)
    await session.execute(text(
        "DELETE FROM training_task_tags WHERE task_id IN "
        "(SELECT id FROM model_training_tasks WHERE user_id = :uid AND custom_model_name LIKE 'PROBE%')"
    ), {"uid": user_id})
    await session.execute(text(
        "DELETE FROM model_training_tasks WHERE user_id = :uid AND custom_model_name LIKE 'PROBE%'"
    ), {"uid": user_id})
    return len(rows)

async def main():
    os.makedirs(os.path.dirname(CONF["runningLog"]), exist_ok=True)
    async with async_session_maker() as session:
        user_id = (await session.execute(text("SELECT id FROM users WHERE email = :email"), {"email": CONF["email"]})).scalar_one_or_none()
        removed = await cleanup_tasks(session, user_id) if user_id else 0
        await session.commit()

        if not CONF["withTasks"]:
            print(json.dumps({"removed": removed, "withTasks": False}, ensure_ascii=False))
            return

        # 任务清理完成后再写日志文件，避免刚写好的日志被上一步删掉。
        with open(CONF["runningLog"], "w", encoding="utf-8") as fh:
            fh.write("\\n".join(${JSON.stringify(RUNNING_LOG)}) + "\\n")
        with open(CONF["doneLog"], "w", encoding="utf-8") as fh:
            fh.write("\\n".join(${JSON.stringify(DONE_LOG)}) + "\\n")

        columns = ("user_id, model_script, model_source, uploaded_model_id, uploaded_model_version, "
                   "hyperparameters, status, start_time, log_file_path, output_model_path, custom_model_name, pid, "
                   "progress, current_epoch, total_epochs, current_loss, eta, loss_history, metrics, "
                   "dataset_id, dataset_version, dataset_identity_status")
        now = datetime.now()
        running_values = {
            "uid": user_id, "script": "demo3.py", "status": "running",
            "start": now - timedelta(minutes=6), "log": CONF["runningLog"], "out": None,
            "name": CONF["runningName"], "pid": 999999, "progress": 42.5, "epoch": 4, "total": 8,
            "loss": 0.3302, "eta": "00:03:12", "history": json.dumps(RUNNING_LOSS),
            "metrics": None, "hypers": pack(RUNNING_HYPERS, ["U"]),
        }
        done_values = dict(running_values)
        done_values.update({
            "script": "demo3.py", "status": "completed", "start": now - timedelta(hours=2),
            "log": CONF["doneLog"], "out": "models/checkpoints/probe_surface_net_best.pth",
            "name": CONF["doneName"], "pid": None, "progress": 100.0, "epoch": 8, "total": 8,
            "loss": 0.3088, "eta": "00:00:00", "history": json.dumps(DONE_LOSS),
            "metrics": json.dumps({"rmse": 0.0412, "mae": 0.0288, "mse": 0.0017, "r2": 0.9312}),
            "hypers": pack(RUNNING_HYPERS, ["U", "T"]),
        })

        inserted = {}
        for key, values in (("running", running_values), ("done", done_values)):
            result = await session.execute(text(
                "INSERT INTO model_training_tasks (" + columns + ") VALUES ("
                ":uid, :script, 'uploaded', :model_id, 1, :hypers, :status, :start, :log, :out, :name, :pid, "
                ":progress, :epoch, :total, :loss, :eta, :history, :metrics, 'openmars_mcd', 'v1', 'verified')"
            ), {
                **values,
                "model_id": CONF["uploadedModelId"],
            })
            inserted[key] = result.lastrowid
        await session.commit()
        print(json.dumps({"removed": removed, "withTasks": True, **inserted}, ensure_ascii=False))

asyncio.run(main())
`;
}

function cleanupDatabaseScript() {
  return `
import asyncio, json, os, sys
from pathlib import Path

BACKEND = Path(${JSON.stringify(BACKEND_DIR)})
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from sqlalchemy import text
from database.engine import async_session_maker

EMAIL = ${JSON.stringify(EMAIL)}
PREFIX = 'PROBE'
LOG_DIR = Path(${JSON.stringify(LOG_DIR)})

async def main():
    async with async_session_maker() as session:
        user_id = (await session.execute(text("SELECT id FROM users WHERE email = :email"), {"email": EMAIL})).scalar_one_or_none()
        if user_id is None:
            print(json.dumps({"removed": False}, ensure_ascii=False))
            return
        models = (await session.execute(text("SELECT id, storage_path FROM user_model_packages WHERE user_id = :uid"), {"uid": user_id})).all()
        tasks = (await session.execute(text(
            "SELECT id, log_file_path, output_model_path FROM model_training_tasks WHERE user_id = :uid OR custom_model_name LIKE :prefix"
        ), {"uid": user_id, "prefix": PREFIX + '%'})).all()

        for _, path in models:
            if path and os.path.exists(path):
                os.remove(path)
        for _, log_path, output_path in tasks:
            for path in (log_path, output_path):
                if path and os.path.exists(path):
                    os.remove(path)

        await session.execute(text("DELETE FROM training_task_tags WHERE task_id IN (SELECT id FROM model_training_tasks WHERE user_id = :uid)"), {"uid": user_id})
        await session.execute(text("DELETE FROM model_training_tasks WHERE user_id = :uid"), {"uid": user_id})
        await session.execute(text("DELETE FROM training_model_tags WHERE user_id = :uid"), {"uid": user_id})
        await session.execute(text("DELETE FROM user_model_packages WHERE user_id = :uid"), {"uid": user_id})
        await session.execute(text("DELETE FROM notifications WHERE user_id = :uid"), {"uid": user_id})
        await session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})
        await session.commit()
        print(json.dumps({"removed": True, "models": len(models), "tasks": len(tasks)}, ensure_ascii=False))

    if LOG_DIR.exists():
        for item in LOG_DIR.iterdir():
            if item.is_file():
                item.unlink()
        try:
            LOG_DIR.rmdir()
        except OSError:
            pass

asyncio.run(main())
`;
}

async function apiJson(path, { method = 'GET', body, token } = {}) {
  const response = await fetch(`${API}${path}`, {
    method,
    headers: {
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let parsed = null;
  try { parsed = text ? JSON.parse(text) : null; } catch { parsed = null; }
  if (!response.ok) {
    throw new Error(`${method} ${path} -> ${response.status} ${text.slice(0, 200)}`);
  }
  return parsed;
}

async function uploadProbeModel(token) {
  const boundary = `----probe${Date.now().toString(16)}`;
  const file = readFileSync(MODEL_FILE);
  const body = Buffer.concat([
    Buffer.from(`--${boundary}\r\n`),
    Buffer.from('Content-Disposition: form-data; name="file"; filename="probe_surface_net.py"\r\n'),
    Buffer.from('Content-Type: text/x-python\r\n\r\n'),
    file,
    Buffer.from(`\r\n--${boundary}--\r\n`),
  ]);
  const response = await fetch(`${API}/user-models`, {
    method: 'POST',
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
      Authorization: `Bearer ${token}`,
    },
    body,
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`上传探针模型失败: ${response.status} ${text.slice(0, 300)}`);
  return JSON.parse(text);
}

function ensureProbeUser() {
  runPython(`
import asyncio, json, os, sys
from pathlib import Path
BACKEND = Path(${JSON.stringify(BACKEND_DIR)})
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
from sqlalchemy import text
from database.engine import async_session_maker
from auth.security import hash_password
EMAIL = ${JSON.stringify(EMAIL)}
PASSWORD = ${JSON.stringify(PASSWORD)}
async def main():
    async with async_session_maker() as session:
        row = (await session.execute(text("SELECT id FROM users WHERE email = :email"), {"email": EMAIL})).scalar_one_or_none()
        if row:
            await session.execute(text("UPDATE users SET password_hash = :hash, username = 'Probe Console' WHERE id = :uid"),
                                  {"hash": hash_password(PASSWORD), "uid": row})
            await session.commit()
            print(json.dumps({"created": False, "id": row}))
            return
        await session.execute(text(
            "INSERT INTO users (email, username, password_hash, role, created_at, is_active) "
            "VALUES (:email, 'Probe Console', :hash, 'user', :now, 1)"
        ), {"email": EMAIL, "hash": hash_password(PASSWORD), "now": __import__('datetime').datetime.now().isoformat(sep=' ', timespec='seconds')})
        await session.commit()
        row = (await session.execute(text("SELECT id FROM users WHERE email = :email"), {"email": EMAIL})).scalar_one()
        print(json.dumps({"created": True, "id": row}))
asyncio.run(main())
`);
}

async function prepare() {
  if (!existsSync(BACKEND_DIR)) throw new Error(`未找到后端目录 ${BACKEND_DIR}`);
  mkdirSync(HERE, { recursive: true });
  mkdirSync(LOG_DIR, { recursive: true });
  writeFileSync(MODEL_FILE, MODEL_SOURCE, 'utf8');
  ensureProbeUser();
  const login = await apiJson('/auth/login', { method: 'POST', body: { email: EMAIL, password: PASSWORD } });
  const token = login?.token;
  if (!token) throw new Error('登录未返回 token');
  const models = await apiJson('/user-models', { token });
  const existing = (models?.items || []).find((item) => item.original_filename === 'probe_surface_net.py');
  const uploaded = existing || await uploadProbeModel(token);
  const withTasks = !hasFlag('no-tasks');
  runPython(prepareDatabaseScript({ uploadedModelId: uploaded.id, withTasks }));
  const state = {
    email: EMAIL,
    password: PASSWORD,
    token,
    uploadedModelId: uploaded.id,
    uploadedModelName: uploaded.display_name,
    validationStatus: uploaded.validation_status,
    paramSchema: uploaded.param_schema,
  };
  writeFileSync(STATE_FILE, `${JSON.stringify(state, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({
    prepared: true,
    account: EMAIL,
    uploadedModelId: state.uploadedModelId,
    validationStatus: state.validationStatus,
    tokenFile: STATE_FILE,
    withTasks,
  }, null, 2));
}

function cleanup() {
  runPython(cleanupDatabaseScript());
  rmSync(MODEL_FILE, { force: true });
  rmSync(STATE_FILE, { force: true });
  console.log(JSON.stringify({ cleaned: true }, null, 2));
}

if (command === 'prepare') {
  await prepare();
} else if (command === 'cleanup') {
  cleanup();
} else {
  console.error(`未知命令 ${command}，可用：prepare / cleanup / help`);
  process.exit(2);
}
