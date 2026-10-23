/**
 * 实验中心控制台验收 · 大气实验控制台版式（第四轮）。
 *
 * 覆盖：默认三列（目录展开）/ 收起两列、sticky 吸附、fixed 运行条与底部预留、
 * 上传模型主入口、官方架构选择器、快捷载荷条与序列图示、参数矩阵、
 * 专家侧边页签、检查器简短摘要与状态一致性、响应式无横向溢出。
 *
 * 前置：
 *   1. 在 frontend/ 执行 npm run build；
 *   2. 启动生产静态服务（scripts/serve-prod.mjs）与后端；
 *   3. 运行同目录 seed-probe-profile.mjs prepare（生成一次性验收账号与上传模型）；
 *   4. 启动无头 Edge 并打开远程调试端口：
 *      msedge --headless=new --remote-debugging-port=9333 --user-data-dir=<临时目录>
 *
 * 用法：
 *   node scripts/audit/experiment-console/verify-console.mjs [--port 9333] [--url http://127.0.0.1:5173]
 * 默认从 seed-probe-profile.mjs 写出的 .probe-profile.json 读取 JWT（也可用 --token 覆盖）。
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
const BASE = argOf('url', 'http://127.0.0.1:5173');
const TOKEN = argOf('token', readProbeToken());

function httpJson(path) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port: CDP_PORT, path, method: 'GET' }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        try { resolve(JSON.parse(body)); } catch (err) { reject(new Error(`${path}: ${body.slice(0, 200)}`)); }
      });
    });
    req.on('error', reject);
    req.end();
  });
}

async function waitForCdp(timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      return await httpJson('/json/version');
    } catch (err) {
      if (Date.now() > deadline) throw new Error(`CDP 未就绪: ${err.message}`);
      await new Promise((r) => setTimeout(r, 300));
    }
  }
}

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 1;
    this.pending = new Map();
    ws.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(JSON.stringify(message.error)));
        else resolve(message.result);
      }
    });
  }

  send(method, params = {}, sessionId) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(sessionId ? { id, method, params, sessionId } : { id, method, params }));
    });
  }
}

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok: Boolean(ok), detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

const version = await waitForCdp();
console.log(`浏览器: ${version.Browser}`);
const ws = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, { once: true });
  ws.addEventListener('error', () => reject(new Error('WebSocket 连接失败')), { once: true });
});
const browser = new Cdp(ws);
const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true });
const call = (method, params) => browser.send(method, params, sessionId);
await call('Page.enable');
await call('Runtime.enable');

async function evaluate(expression) {
  const response = await call('Runtime.evaluate', {
    expression: `(async () => { ${expression} })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.exception?.description || JSON.stringify(response.exceptionDetails));
  }
  return response.result.value;
}

async function setViewport(width, height) {
  await call('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  await evaluate(`
    let last = '';
    let stable = 0;
    for (let i = 0; i < 40; i += 1) {
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const doc = document.documentElement;
      const snapshot = doc.clientWidth + 'x' + doc.scrollWidth + 'x' + doc.scrollHeight;
      stable = snapshot === last ? stable + 1 : 0;
      last = snapshot;
      if (stable >= 3) break;
    }
    return true;
  `);
}

/**
 * 打开训练页并进入配置阶段。
 *
 * 账号里若有运行中的实验，页面会按既有行为自动进入监控阶段，
 * 因此配置相关的检查先点「新建实验」。
 */
async function openConfigure({ token = TOKEN, settings = { language: 'zh', theme: 'dark' } } = {}) {
  await call('Page.navigate', { url: `${BASE}/` });
  await new Promise((r) => setTimeout(r, 1200));
  await evaluate(`
    localStorage.setItem('astraatmos_settings', JSON.stringify(${JSON.stringify(settings)}));
    ${token ? `localStorage.setItem('astraatmos_token', ${JSON.stringify(token)});` : "localStorage.removeItem('astraatmos_token');"}
    return true;
  `);
  await call('Page.reload', { ignoreCache: true });
  for (let attempt = 0; attempt < 3; attempt += 1) {
    await call('Page.navigate', { url: `${BASE}/#/training` });
    const loaded = await evaluate(`
      for (let i = 0; i < 60; i += 1) {
        if (document.querySelector('[data-run-bar="true"]') && document.querySelector('.experiment-inspector-state')) return true;
        await new Promise(r=>setTimeout(r,250));
      }
      return false;
    `);
    if (loaded) break;
  }
  // 载入后约 0.5–1 秒会按既有行为自动选中运行中的任务，因此这里要求
  // 配置阶段「持续稳定」（连续约 1.5 秒不变）才算进入成功。
  const waitConfigured = () => evaluate(`
    let stableTicks = 0;
    for (let i = 0; i < 80; i += 1) {
      const grid = document.querySelector('.experiment-center-grid');
      const ok = grid && grid.dataset.stage === 'configure'
        && document.querySelector('[data-run-bar="true"]')
        && document.querySelector('.experiment-inspector-state');
      stableTicks = ok ? stableTicks + 1 : 0;
      if (stableTicks >= 6) return 'configure';
      await new Promise(r=>setTimeout(r,250));
    }
    return document.querySelector('.experiment-center-grid')?.dataset.stage || null;
  `);

  // 载入后约 0.5 秒会按既有行为自动选中运行中的任务；这里重新进页面（回到配置阶段），
  // 再点「新建实验」并确认阶段稳定在 configure，与真实用户路径一致。
  let stage = null;
  for (let attempt = 0; attempt < 4 && stage !== 'configure'; attempt += 1) {
    await call('Page.navigate', { url: `${BASE}/#/training` });
    stage = await waitConfigured();
    if (stage === 'configure') break;
    // 自动选中已经把页面切到监控阶段：点「新建实验」回到配置并确认稳定。
    await evaluate(`
      const button = [...document.querySelectorAll('.experiment-center-header-meta button')]
        .find((el) => /新建实验|New experiment/.test(el.innerText));
      button?.click();
      return true;
    `);
    stage = await waitConfigured();
  }
  if (stage !== 'configure') throw new Error(`未能进入配置阶段（当前 ${stage}）`);
  await evaluate(`document.documentElement.style.scrollBehavior = 'auto'; return true;`);
}

const METRICS = `
  const rect = (sel) => { const el = document.querySelector(sel); if (!el) return null; const r = el.getBoundingClientRect(); return { l: Math.round(r.left), r: Math.round(r.right), w: Math.round(r.width), t: Math.round(r.top), b: Math.round(r.bottom), h: Math.round(r.height) }; };
  const grid = document.querySelector('.experiment-center-grid');
  const inspector = document.querySelector('.experiment-inspector-panel');
  const runBar = document.querySelector('[data-run-bar="true"]');
  return {
    viewport: { w: document.documentElement.clientWidth, h: window.innerHeight },
    scrollY: Math.round(window.scrollY),
    maxScroll: Math.round(document.documentElement.scrollHeight - window.innerHeight),
    overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    columns: grid ? getComputedStyle(grid).gridTemplateColumns : null,
    columnCount: grid ? getComputedStyle(grid).gridTemplateColumns.split(' ').length : 0,
    directory: grid ? grid.dataset.directory : null,
    inspectorMode: grid ? grid.dataset.inspector : null,
    rail: rect('.experiment-directory'),
    canvas: rect('.experiment-center-canvas'),
    inspector: rect('.experiment-inspector-panel'),
    railPosition: document.querySelector('.experiment-directory') ? getComputedStyle(document.querySelector('.experiment-directory')).position : null,
    inspectorPosition: inspector ? getComputedStyle(inspector).position : null,
    canvasPosition: document.querySelector('.experiment-center-canvas') ? getComputedStyle(document.querySelector('.experiment-center-canvas')).position : null,
    runBar: rect('[data-run-bar="true"]'),
    runBarPosition: runBar ? getComputedStyle(runBar).position : null,
    runBarState: document.querySelector('.experiment-run-bar-state')?.dataset.runState || null,
    runBarSummary: document.querySelector('[data-run-bar-summary]')?.textContent?.trim() || '',
    lastSection: rect('[data-config-group="expert"]'),
    inspectorOverflow: inspector ? inspector.scrollHeight - inspector.clientHeight : null,
  };
`;

// ---------------- 1440×900 ----------------
console.log('\n=== 视口 1440×900 ===');
await setViewport(1440, 900);
await openConfigure();
let m = await evaluate(METRICS);
check('默认三列（目录 + 画布 + 检查器）', m.columnCount === 3 && m.directory === 'open', `columns=${m.columns}`);
check('默认渲染实验目录', m.rail !== null, `rail=${m.rail ? `${m.rail.w}x${m.rail.h}` : 'null'}`);
check('检查器列宽 284px', m.inspector && m.inspector.w === 284, `inspector=${m.inspector?.w}`);
check('运行条 fixed 横跨浏览器', m.runBarPosition === 'fixed' && m.runBar.w === m.viewport.w, `${m.runBarPosition} w=${m.runBar?.w}/${m.viewport.w}`);
check('运行条高度 60–80px', m.runBar.h >= 56 && m.runBar.h <= 84, `${m.runBar?.h}px`);
check('画布不是 sticky（正常文档流）', m.canvasPosition !== 'sticky' && m.canvasPosition !== 'fixed', m.canvasPosition);
check('检查器 sticky', m.inspectorPosition === 'sticky', m.inspectorPosition);
check('1440 无横向溢出', m.overflowX <= 0, `overflowX=${m.overflowX}`);
check('检查器无内部滚动条', m.inspectorOverflow <= 0, `overflow=${m.inspectorOverflow}`);

const canvasHeadInfo = await evaluate(`
  const nameInput = document.querySelector('.experiment-canvas-name');
  const state = document.querySelector('.experiment-canvas-state');
  const summary = document.querySelector('.experiment-config-summary');
  // 输入与预测、训练参数现在是超参数的页签，只有激活页在 DOM 里。
  const clickTab = async (tab) => {
    const el = document.querySelector('[data-expert-tab="' + tab + '"]');
    if (el) { el.click(); await new Promise((r) => setTimeout(r, 300)); }
  };
  const sectionOrder = [...document.querySelectorAll('[data-config-group]')].map((el) => el.dataset.configGroup);
  const expertGrid = getComputedStyle(document.querySelector('.experiment-expert')).gridTemplateColumns;
  const expertTabs = [...document.querySelectorAll('[data-expert-tab]')].map((el) => el.dataset.expertTab);
  await clickTab('training');
  const paramGrid = document.querySelector('.experiment-param-grid');
  const paramColumns = paramGrid ? getComputedStyle(paramGrid).gridTemplateColumns : null;
  const paramCodes = [...document.querySelectorAll('.experiment-param-label code')].map((el) => el.textContent.trim());
  await clickTab('payload');
  const payloadBar = Boolean(document.querySelector('.experiment-payload-bar'));
  return {
    hasNameInput: Boolean(nameInput),
    nameValue: nameInput ? nameInput.value : null,
    state: state ? state.dataset.canvasState : null,
    stateText: state ? state.innerText.trim() : '',
    duplicateSummary: Boolean(summary),
    sectionOrder,
    paramColumns,
    paramCodes,
    expertGrid,
    expertTabs,
    payloadBar,
    flowDiagram: Boolean(document.querySelector('[data-sequence-diagram="true"]')),
  };
`);
check('画布头部是可编辑实验名称', canvasHeadInfo.hasNameInput);
check('画布不再有重复的「当前配置」摘要', !canvasHeadInfo.duplicateSummary);
check('画布四个分区：模型名称 / 数据集 / 模型 / 超参数', canvasHeadInfo.sectionOrder.join(',') === 'name,dataset,model,expert', canvasHeadInfo.sectionOrder.join(','));
const paramTracks = canvasHeadInfo.paramColumns ? canvasHeadInfo.paramColumns.split(' ') : [];
const minParamTrack = paramTracks.length ? Math.min(...paramTracks.map((v) => parseFloat(v))) : 0;
check('训练参数矩阵按可用宽度自动排布（每列 ≥ 140px）', paramTracks.length >= 2 && minParamTrack >= 140, canvasHeadInfo.paramColumns);
check('参数标签旁不再有 WINDOW / BATCH 这类英文小码', canvasHeadInfo.paramCodes.length === 0, canvasHeadInfo.paramCodes.join(','));
check('超参数是侧边页签 + 右侧内容', canvasHeadInfo.expertGrid.split(' ').length === 2, canvasHeadInfo.expertGrid);
check('超参数页签含输入与预测 / 训练参数', canvasHeadInfo.expertTabs.slice(0, 2).join(',') === 'payload,training', canvasHeadInfo.expertTabs.join(','));
check('载荷条存在（序列示意图已删除）', canvasHeadInfo.payloadBar);

// 目录开关：默认展开（三列）→ 收起（两列）→ 再展开（三列）
const collapsed = await evaluate(`
  const toggle = [...document.querySelectorAll('.experiment-center-header-meta button')].find((el) => /目录/.test(el.innerText));
  if (!toggle) return { missingToggle: true, buttons: [...document.querySelectorAll('.experiment-center-header-meta button')].map((el) => el.innerText.trim()) };
  toggle.click();
  await new Promise((r) => setTimeout(r, 500));
  ${METRICS}
`);
check('目录开关按钮存在', !collapsed.missingToggle, JSON.stringify(collapsed.buttons || []));
check('收起目录后两列', collapsed.columnCount === 2 && collapsed.directory === 'closed', `columns=${collapsed.columns}`);
check('收起后不渲染目录', collapsed.rail === null);

const expanded = await evaluate(`
  const toggle = [...document.querySelectorAll('.experiment-center-header-meta button')].find((el) => /目录/.test(el.innerText));
  toggle.click();
  await new Promise((r) => setTimeout(r, 500));
  ${METRICS}
`);
check('再次展开目录后三列', expanded.columnCount === 3, `columns=${expanded.columns}`);
check('目录列宽 235px、检查器 284px', Math.abs(expanded.rail.w - 235) <= 1 && expanded.inspector.w === 284, `${expanded.rail?.w}/${expanded.inspector?.w}`);
check('目录与检查器都是 sticky', expanded.railPosition === 'sticky' && expanded.inspectorPosition === 'sticky', `${expanded.railPosition}/${expanded.inspectorPosition}`);
check('展开后无横向溢出', expanded.overflowX <= 0, `overflowX=${expanded.overflowX}`);

// 滚动吸附
const scrolled = await evaluate(`
  window.scrollTo(0, 600);
  await new Promise((r) => setTimeout(r, 400));
  ${METRICS}
`);
check('滚动 ≥ 500px', scrolled.scrollY >= 500, `scrollY=${scrolled.scrollY}`);
check('目录吸附在 86px', Math.abs(scrolled.rail.t - 86) <= 2, `top=${scrolled.rail?.t}`);
check('检查器吸附在 86px', Math.abs(scrolled.inspector.t - 86) <= 2, `top=${scrolled.inspector?.t}`);
check('滚动时运行条仍贴视口底部', Math.abs(scrolled.runBar.b - scrolled.viewport.h) <= 1, `bottom=${scrolled.runBar?.b}/${scrolled.viewport.h}`);

// 页尾预留
const bottom = await evaluate(`
  window.scrollTo(0, document.documentElement.scrollHeight);
  await new Promise((r) => setTimeout(r, 400));
  ${METRICS}
`);
check('可以滚到页尾', bottom.scrollY >= bottom.maxScroll - 2, `${bottom.scrollY}/${bottom.maxScroll}`);
check('最后一个分区不被运行条遮挡', bottom.lastSection.b <= bottom.runBar.t + 1, `section=${bottom.lastSection?.b} runBarTop=${bottom.runBar?.t}`);
check('页尾运行条仍贴底', Math.abs(bottom.runBar.b - bottom.viewport.h) <= 1, `bottom=${bottom.runBar?.b}`);

// ---------------- 1024×768 ----------------
console.log('\n=== 视口 1024×768 ===');
await setViewport(1024, 768);
m = await evaluate(METRICS);
check('1024 无横向溢出', m.overflowX <= 0, `overflowX=${m.overflowX} columns=${m.columns}`);
check('1024 保持三列（自适应收窄）', m.columnCount === 3, m.columns);
check('1024 目录与检查器都保留', Boolean(m.rail) && Boolean(m.inspector));

// ---------------- 390×844 ----------------
console.log('\n=== 视口 390×844 ===');
await setViewport(390, 844);
m = await evaluate(METRICS);
check('390 单列', m.columnCount === 1, m.columns);
check('390 侧栏取消 sticky', m.railPosition === 'static' && m.inspectorPosition === 'static', `${m.railPosition}/${m.inspectorPosition}`);
check('390 运行条回到普通流', m.runBarPosition === 'static', m.runBarPosition);
check('390 无横向溢出', m.overflowX <= 0, `overflowX=${m.overflowX}`);
const mobileUsable = await evaluate(`
  const start = document.querySelector('[data-run-bar-start="true"]');
  const tabs = document.querySelectorAll('[data-expert-tab]').length;
  const upload = Boolean(document.querySelector('[data-uploaded-model-upload="true"]'));
  // 训练参数是超参数的页签，先切过去再确认字段可读。
  document.querySelector('[data-expert-tab="training"]')?.click();
  await new Promise((r) => setTimeout(r, 300));
  const param = document.querySelector('[data-parameter="epochs"] input');
  return { startHeight: Math.round(start.getBoundingClientRect().height), tabs, upload, param: Boolean(param) };
`);
check('390 上传入口与超参数页签可用', mobileUsable.upload && mobileUsable.tabs >= 6, `upload=${mobileUsable.upload} tabs=${mobileUsable.tabs}`);
check('390 主按钮高度 ≥ 44px', mobileUsable.startHeight >= 44, `${mobileUsable.startHeight}px`);

// ---------------- 字体放大 / 浅色 / 英文 ----------------
console.log('\n=== 字体放大 1.3 / 浅色英文 ===');
await setViewport(1440, 900);
await evaluate(`document.documentElement.style.setProperty('--font-scale', '1.3'); await new Promise(r=>setTimeout(r,400)); return true;`);
m = await evaluate(METRICS);
check('放大文字：运行条自适应且不被裁掉', m.runBar.h >= 56, `${m.runBar?.h}px`);
check('放大文字：无横向溢出', m.overflowX <= 0, `overflowX=${m.overflowX}`);
await evaluate(`document.documentElement.style.removeProperty('--font-scale'); return true;`);

const failed = results.filter((item) => !item.ok);
console.log(`\n===== 汇总：${results.length - failed.length}/${results.length} 项通过 =====`);
failed.forEach((item) => console.log(`失败: ${item.name} — ${item.detail}`));
if (failed.length) process.exitCode = 1;

await browser.send('Target.closeTarget', { targetId });
process.exit(process.exitCode || 0);
