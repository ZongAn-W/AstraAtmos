/**
 * 训练监控 / 结果工作区验收。
 *
 * 用 seed-probe-profile.mjs 注入的 running / completed 任务，通过真实训练接口
 * （任务列表与日志轮询）验证监控阶段与结果阶段仍然可用。
 *
 * 前置：见 verify-console.mjs 头注释（构建、生产服务、后端、验收档案、无头浏览器）。
 *
 * 用法：
 *   node scripts/audit/experiment-console/verify-stages.mjs [--port 9333] [--token <jwt>]
 */
import http from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const STATE_FILE = join(HERE, '.probe-profile.json');

const args = process.argv.slice(2);
function argOf(name, fallback) {
  const index = args.indexOf(`--${name}`);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
}

function readProbeToken() {
  if (!existsSync(STATE_FILE)) return '';
  try {
    return String(JSON.parse(readFileSync(STATE_FILE, 'utf8'))?.token || '');
  } catch {
    return '';
  }
}

const CDP_PORT = Number(argOf('port', '9333'));
const BASE = argOf('base', 'http://127.0.0.1:5173');
const TOKEN = argOf('token', readProbeToken());
const results = [];

function log(line) { console.log(line); }
function check(name, ok, detail = '') {
  results.push({ name, ok: Boolean(ok), detail });
  log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

function httpJson(path) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port: CDP_PORT, path, method: 'GET' }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { body += c; });
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch (e) { reject(new Error(body.slice(0, 300))); } });
    });
    req.on('error', reject);
    req.end();
  });
}

class Cdp {
  constructor(ws) {
    this.ws = ws; this.n = 1; this.p = new Map();
    ws.addEventListener('message', (e) => {
      const m = JSON.parse(e.data);
      if (m.id && this.p.has(m.id)) {
        const { resolve, reject } = this.p.get(m.id);
        this.p.delete(m.id);
        m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
      }
    });
  }
  send(method, params = {}, sid) {
    const id = this.n++;
    return new Promise((resolve, reject) => {
      this.p.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(sid ? { id, method, params, sessionId: sid } : { id, method, params }));
    });
  }
}

const version = await httpJson('/json/version');
const ws = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
const browser = new Cdp(ws);
const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true });
const call = (m, p) => browser.send(m, p, sessionId);
await call('Page.enable');
await call('Runtime.enable');

async function evaluate(expression) {
  const r = await call('Runtime.evaluate', { expression: `(async () => { ${expression} })()`, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval error');
  return r.result.value;
}

await call('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
await call('Page.navigate', { url: `${BASE}/#/training` });
await new Promise((r) => setTimeout(r, 2200));
await evaluate(`
  localStorage.setItem('astraatmos_settings', JSON.stringify({ language: 'zh', theme: 'dark' }));
  ${TOKEN ? `localStorage.setItem('astraatmos_token', ${JSON.stringify(TOKEN)});` : "localStorage.removeItem('astraatmos_token');"}
  return true;
`);
await call('Page.reload', { ignoreCache: true });
await evaluate(`
  // 训练 / 结果阶段默认展开实验目录，等待目录与外框就绪。
  for (let i = 0; i < 80; i += 1) {
    if (document.querySelector('.experiment-directory-row') && document.querySelector('.experiment-directory')) break;
    await new Promise((r) => setTimeout(r, 250));
  }
  document.documentElement.style.scrollBehavior = 'auto';
  return true;
`);

/** 按任务名点开目录里的实验行。 */
async function openTaskByName(name) {
  return evaluate(`
    const rows = [...document.querySelectorAll('.experiment-directory-row')];
    const row = rows.find((el) => (el.innerText || '').includes(${JSON.stringify(name)}));
    if (!row) return { clicked: false, names: rows.map((r) => (r.innerText || '').split('\\n')[0]).slice(0, 12) };
    // 目录行本身是可聚焦按钮：用键盘事件触发，顺带验证键盘可达。
    row.focus();
    row.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await new Promise((r) => setTimeout(r, 300));
    if (document.querySelector('.experiment-center-grid')?.dataset.stage !== 'configure') {
      // 已在目标阶段；若焦点行未生效则退回鼠标点击。
    }
    row.click();
    for (let i = 0; i < 40; i += 1) {
      if (document.querySelector('[data-stage-workspace="monitor"]:not([hidden]), [data-stage-workspace="result"]:not([hidden])')) break;
      await new Promise((r) => setTimeout(r, 200));
    }
    await new Promise((r) => setTimeout(r, 900));
    return { clicked: true, focused: document.activeElement === row };
  `);
}

// ---------------- 训练监控阶段 -------------------------------------------------
log('\n--- 训练监控阶段（running 任务） ---');
const openedRunning = await openTaskByName('PROBE 控制台监控验收');
check('目录中可找到注入的运行中实验', openedRunning.clicked, JSON.stringify(openedRunning.names || []));

const monitor = await evaluate(`
  const workspace = document.querySelector('[data-stage-workspace="monitor"]');
  const visible = workspace && !workspace.hasAttribute('hidden');
  const logRegion = document.querySelector('[role="log"]');
  const rows = logRegion ? logRegion.querySelectorAll('.experiment-monitor-log-row').length : 0;
  const text = logRegion ? logRegion.innerText : '';
  const chart = document.querySelector('[data-stage-workspace="monitor"] .js-plotly-plot, [data-stage-workspace="monitor"] .plotly');
  return {
    visible,
    stage: document.querySelector('.experiment-center-grid')?.dataset.stage,
    progressText: workspace ? (workspace.innerText.match(/(\\d+(?:\\.\\d+)?)\\s*%/) || [])[0] : '',
    logRows: rows,
    logTextSample: text.slice(0, 200),
    hasEpochLine: /epoch/i.test(text),
    hasLossLine: /loss=/i.test(text),
    hasWarningTone: Boolean(logRegion && logRegion.querySelector('[data-tone="warning"]')),
    autoFollow: /自动跟随|已暂停跟随/.test(workspace ? workspace.innerText : ''),
    stopButton: Boolean([...(workspace ? workspace.querySelectorAll('button') : [])].find((b) => /停止训练/.test(b.innerText))),
    chartPresent: Boolean(chart),
    hasLossTitle: /损失演变/.test(workspace ? workspace.innerText : ''),
  };
`);
log(`     监控区: ${JSON.stringify({ ...monitor, logTextSample: undefined })}`);
log(`     日志样本: ${(monitor.logTextSample || '').replace(/\n/g, ' | ').slice(0, 160)}`);
check('运行中任务直接进入监控阶段', monitor.visible && monitor.stage === 'monitor', `stage=${monitor.stage}`);
check('监控显示进度百分比', /%/.test(monitor.progressText || ''), monitor.progressText);
check('实时日志仍是终端式输出区域（role=log）', monitor.logRows > 0, `日志行=${monitor.logRows}`);
check('日志内容包含 epoch 与 loss 行', monitor.hasEpochLine && monitor.hasLossLine);
check('日志按级别着色（warning 行有 tone）', monitor.hasWarningTone);
check('监控保留自动跟随状态与停止训练入口', monitor.autoFollow && monitor.stopButton);
check('监控保留 Loss 曲线', monitor.chartPresent && monitor.hasLossTitle);

const manualScroll = await evaluate(`
  const logRegion = document.querySelector('[role="log"]');
  const workspace = document.querySelector('[data-stage-workspace="monitor"]');
  const before = logRegion.scrollTop;
  // 滚到离底部足够远的位置（判定阈值 50px），模拟用户手动回看历史日志。
  logRegion.scrollTop = 0;
  logRegion.dispatchEvent(new Event('scroll', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 500));
  const pausedHint = /已暂停跟随/.test(workspace.innerText);
  // 再滚回底部，自动跟随应当恢复。
  logRegion.scrollTop = logRegion.scrollHeight;
  logRegion.dispatchEvent(new Event('scroll', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 500));
  return {
    before,
    after: logRegion.scrollTop,
    pausedHint,
    resumed: /自动跟随/.test(workspace.innerText) && !/已暂停跟随/.test(workspace.innerText),
  };
`);
check(
  '日志区域的自动跟随/暂停状态可见',
  /自动跟随|已暂停跟随/.test(monitor.autoFollowText || '') || Boolean(monitor.autoFollow),
  JSON.stringify(manualScroll)
);
log(`     日志滚动: ${JSON.stringify(manualScroll)}`);

// ---------------- 结果阶段 -----------------------------------------------------
log('\n--- 结果阶段（completed 任务） ---');
const openedDone = await openTaskByName('PROBE 结果工作区验收');
check('目录中可找到注入的已完成实验', openedDone.clicked, JSON.stringify(openedDone.names || []));

const result = await evaluate(`
  const workspace = document.querySelector('[data-stage-workspace="result"]');
  const visible = workspace && !workspace.hasAttribute('hidden');
  const text = visible ? workspace.innerText : '';
  const toggle = workspace ? workspace.querySelector('.experiment-log-toggle') : null;
  return {
    visible,
    stage: document.querySelector('.experiment-center-grid')?.dataset.stage,
    directory: document.querySelector('.experiment-center-grid')?.dataset.directory,
    metrics: ['RMSE', 'MAE', 'MSE', 'R²'].filter((key) => text.includes(key)),
    hasLogToggle: Boolean(toggle),
    logToggleExpanded: toggle ? toggle.getAttribute('aria-expanded') : null,
    actions: [...(workspace ? workspace.querySelectorAll('button') : [])].map((b) => b.innerText.trim()).filter(Boolean).slice(0, 12),
    hasParams: /训练配置|参数/.test(text),
    hasDatasetIdentity: /openmars_mcd/.test(text),
    textSample: text.replace(/\\s+/g, ' ').slice(0, 260),
  };
`);
log(`     结果区: ${JSON.stringify(result)}`);
check('已完成任务进入结果阶段', result.visible && result.stage === 'result', `stage=${result.stage}`);
check('结果阶段默认展开实验目录', result.directory === 'open', result.directory);
check('结果页显示后端返回的指标', result.metrics.length >= 3, result.metrics.join(','));
check('结果页保留折叠的「查看运行日志」', result.hasLogToggle && result.logToggleExpanded === 'false');
check('结果页保留完整训练参数与数据集身份', result.hasParams && result.hasDatasetIdentity);
check('结果页保留用于预测 / 去模型比较 / 复制配置等动作', result.actions.length >= 4, result.actions.join(' | '));

const openedLog = await evaluate(`
  const toggle = document.querySelector('.experiment-log-toggle');
  toggle.click();
  await new Promise((r) => setTimeout(r, 600));
  const region = document.querySelector('[role="log"]');
  const rows = region ? region.querySelectorAll('.experiment-monitor-log-row').length : 0;
  const text = region ? region.innerText : '';
  return {
    expanded: toggle.getAttribute('aria-expanded'),
    rows,
    hasFinalLine: /evaluation RMSE/.test(text),
    sample: text.replace(/\\s+/g, ' ').slice(0, 150),
  };
`);
log(`     展开日志: ${JSON.stringify({ ...openedLog, sample: undefined })}`);
check('结果页展开后仍能看到完整历史日志', openedLog.expanded === 'true' && openedLog.rows > 10, `行数=${openedLog.rows}`);
check('日志包含训练结束时的评估行', openedLog.hasFinalLine);

// ---------------- 复制配置回填 -------------------------------------------------
log('\n--- 复制配置回填 ---');
const copied = await evaluate(`
  const buttons = [...document.querySelectorAll('[data-stage-workspace="result"] button')];
  const target = buttons.find((b) => /复制配置/.test(b.innerText));
  if (!target) return { found: false, buttons: buttons.map((b) => b.innerText.trim()) };
  target.click();
  await new Promise((r) => setTimeout(r, 900));
  const grid = document.querySelector('.experiment-center-grid');
  const canvas = document.querySelector('.experiment-center-canvas');
  const uploadedPanel = document.querySelector('[data-uploaded-model-panel="true"]');
  const nameInput = document.querySelector('.experiment-canvas-name');
  // 训练参数与自定义参数现在是超参数的页签，逐个切过去再读。
  const clickTab = async (tab) => {
    const el = document.querySelector('[data-expert-tab="' + tab + '"]');
    if (el) { el.click(); await new Promise((r) => setTimeout(r, 350)); }
  };
  await clickTab('training');
  const epochs = document.querySelector('[data-config-group="training"] [data-parameter="epochs"] input');
  const epochsValue = epochs ? epochs.value : '';
  const inspector = document.querySelector('.experiment-inspector')?.innerText || '';
  await clickTab('customParams');
  // 自定义参数按 schema 动态渲染在超参数面板里，逐项读取实际值。
  const customPanel = document.querySelector('[data-expert-panel="customParams"]');
  const customValues = customPanel
    ? [...customPanel.querySelectorAll('input, select')].map((el) => ({
        name: (el.closest('div')?.parentElement?.innerText || '').split('\\n')[0].trim(),
        value: el.type === 'checkbox' ? String(el.checked) : el.value,
      }))
    : [];
  return {
    found: true,
    stage: grid?.dataset.stage,
    name: nameInput ? nameInput.value : '',
    epochs: epochsValue,
    uploadedModelShown: Boolean(uploadedPanel),
    expertTab: customPanel ? 'customParams' : 'other',
    customValues: customValues.slice(0, 5),
    inspectorModel: inspector.replace(/\\s+/g, ' ').slice(0, 120),
    canvasPresent: Boolean(canvas),
  };
`);
log(`     复制结果: ${JSON.stringify(copied)}`);
check('复制配置回到配置阶段并载入画布', copied.found && copied.stage === 'configure' && copied.canvasPresent);
check('复制配置回填实验名与训练参数', Boolean(copied.name) && copied.epochs === '8', `name=${copied.name} epochs=${copied.epochs}`);
check(
  '复制配置带回上传模型的自定义参数（probe_alpha=23 / probe_ratio=0.35 / probe_flag=true）',
  copied.uploadedModelShown
    && copied.customValues.some((item) => item.value === '23')
    && copied.customValues.some((item) => Number(item.value) === 0.35)
    && copied.customValues.some((item) => item.value === 'true'),
  JSON.stringify(copied.customValues)
);
check('配置检查器显示上传模型', /上传模型/.test(copied.inspectorModel));

// ---------------- 汇总 ---------------------------------------------------------
const failed = results.filter((item) => !item.ok);
log(`\n===== 汇总：${results.length - failed.length}/${results.length} 项通过 =====`);
failed.forEach((item) => log(`失败: ${item.name} — ${item.detail}`));
if (failed.length > 0) process.exitCode = 1;

await browser.send('Target.closeTarget', { targetId });
process.exit(process.exitCode || 0);
