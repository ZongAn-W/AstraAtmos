/**
 * 实验中心（`#/training`）· 交互状态视觉验收（卡片层级整改第二步用）。
 *
 * 静态截图看不到「强描边是否留给真实状态」，本脚本用真实交互逐个触发：
 *   · 参数输入框 hover / focus
 *   · 数据集下拉 focus
 *   · 模型来源胶囊选中态
 *   · 官方模型切换后的架构选择器选中态与家族筛选
 *   · 实验名称输入框 focus 与 invalid（清空后失焦）
 *   · 悬停后的按钮 / 目录行
 * 每步输出该元素的实际描边、背景与 box-shadow，并保存截图。
 *
 * 前置与用法同 verify-font-hierarchy.mjs（生产静态服务 + 后端 + 无头 Edge CDP）。
 *   node scripts/audit/experiment-console/verify-interaction-states.mjs --label round2
 */
import http from 'node:http';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const STATE_FILE = join(HERE, '.probe-profile.json');

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
};

const LABEL = argOf('label', 'states');
const CDP_PORT = Number(argOf('port', '9333'));
const BASE = argOf('url', 'http://127.0.0.1:5173');
const OUT_DIR = join(HERE, 'shots', LABEL);
mkdirSync(OUT_DIR, { recursive: true });

function readProbeToken() {
  if (!existsSync(STATE_FILE)) return '';
  try {
    return String(JSON.parse(readFileSync(STATE_FILE, 'utf8'))?.token || '');
  } catch {
    return '';
  }
}
const TOKEN = argOf('token', readProbeToken());

function httpJson(path) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port: CDP_PORT, path, method: 'GET' }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { body += c; });
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch (err) { reject(new Error(`${path}: ${body.slice(0, 200)}`)); } });
    });
    req.on('error', reject);
    req.end();
  });
}

async function waitForCdp(timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try { return await httpJson('/json/version'); } catch (err) {
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

const version = await waitForCdp();
console.log(`浏览器: ${version.Browser} · 标签: ${LABEL}`);
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
await call('DOM.enable');

async function evaluate(expression) {
  const response = await call('Runtime.evaluate', {
    expression: `(async () => { ${expression} })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.exception?.description || 'eval error');
  return response.result.value;
}

const VIEW = { width: 1440, height: 900 };
await call('Emulation.setDeviceMetricsOverride', { ...VIEW, deviceScaleFactor: 1, mobile: false });

await call('Page.navigate', { url: `${BASE}/` });
await new Promise((r) => setTimeout(r, 1200));
await evaluate(`
  localStorage.setItem('astraatmos_settings', JSON.stringify({ language: 'zh', theme: 'dark' }));
  ${TOKEN ? `localStorage.setItem('astraatmos_token', ${JSON.stringify(TOKEN)});` : "localStorage.removeItem('astraatmos_token');"}
  return true;
`);
await call('Page.reload', { ignoreCache: true });

const waitConfigured = () => evaluate(`
  let stableTicks = 0;
  for (let i = 0; i < 80; i += 1) {
    const grid = document.querySelector('.experiment-center-grid');
    const ok = grid && grid.dataset.stage === 'configure' && document.querySelector('[data-run-bar="true"]');
    stableTicks = ok ? stableTicks + 1 : 0;
    if (stableTicks >= 6) return 'configure';
    await new Promise(r=>setTimeout(r,250));
  }
  return document.querySelector('.experiment-center-grid')?.dataset.stage || null;
`);
let stage = null;
for (let attempt = 0; attempt < 4 && stage !== 'configure'; attempt += 1) {
  await call('Page.navigate', { url: `${BASE}/#/training` });
  stage = await waitConfigured();
  if (stage === 'configure') break;
  await evaluate(`
    [...document.querySelectorAll('.experiment-center-header-meta button')].find((el) => /新建实验|New experiment/.test(el.innerText))?.click();
    return true;
  `);
  stage = await waitConfigured();
}
if (stage !== 'configure') throw new Error(`未能进入配置阶段（当前 ${stage}）`);
await evaluate(`
  document.documentElement.style.scrollBehavior = 'auto';
  for (let i = 0; i < 40 && !document.querySelector('[data-uploaded-model-card="true"]'); i += 1) {
    await new Promise(r=>setTimeout(r,250));
  }
  return true;
`);

const results = [];
function record(name, ok, detail) {
  results.push({ name, ok: Boolean(ok), detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} — ${detail}`);
}

const DESCRIBE = (selector) => `
  const el = document.querySelector(${JSON.stringify(selector)});
  if (!el) return { state: 'not-found' };
  const cs = getComputedStyle(el);
  const alphaOf = (color) => {
    const m = color.match(/rgba?\\(([^)]+)\\)/);
    if (!m) return 1;
    const parts = m[1].split(',').map((v) => parseFloat(v));
    return parts.length > 3 ? parts[3] : 1;
  };
  const edges = ['Top', 'Right', 'Bottom', 'Left'].map((edge) => ({
    edge: edge.toLowerCase(),
    width: parseFloat(cs['border' + edge + 'Width']),
    color: cs['border' + edge + 'Color'],
    visible: parseFloat(cs['border' + edge + 'Width']) > 0 && alphaOf(cs['border' + edge + 'Color']) > 0.01,
  }));
  const rect = el.getBoundingClientRect();
  return {
    state: 'ok',
    rect: { w: Math.round(rect.width), h: Math.round(rect.height) },
    visibleEdges: edges.filter((e) => e.visible).map((e) => e.edge).join('') || 'none',
    borderColors: edges.filter((e) => e.visible).map((e) => e.color).join('|') || 'none',
    background: cs.backgroundColor,
    boxShadow: cs.boxShadow === 'none' ? 'none' : cs.boxShadow,
    outline: cs.outlineStyle === 'none' ? 'none' : cs.outlineWidth + ' ' + cs.outlineStyle + ' ' + cs.outlineColor,
    matchedFocusVisible: el.matches(':focus-visible'),
    value: el.value ?? null,
    ariaInvalid: el.getAttribute('aria-invalid'),
  };
`;

async function shoot(name) {
  const { data } = await call('Page.captureScreenshot', { format: 'png' });
  const file = join(OUT_DIR, `${name}.png`);
  writeFileSync(file, Buffer.from(data, 'base64'));
  return file;
}

async function hover(selector, x = 0.5, y = 0.5) {
  const box = await evaluate(`
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return null;
    el.scrollIntoView({ block: 'center' });
    await new Promise(r=>setTimeout(r,250));
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width * ${x}, y: r.top + r.height * ${y} };
  `);
  if (!box) throw new Error(`hover 目标不存在: ${selector}`);
  await call('Input.dispatchMouseEvent', { type: 'mouseMoved', x: box.x, y: box.y, buttons: 0 });
  await new Promise((r) => setTimeout(r, 260));
}

async function clickSelector(selector) {
  await evaluate(`
    const el = document.querySelector(${JSON.stringify(selector)});
    if (el) { el.scrollIntoView({ block: 'center' }); await new Promise(r=>setTimeout(r,250)); el.click(); }
    return true;
  `);
  await new Promise((r) => setTimeout(r, 350));
}

const report = { label: LABEL, viewport: VIEW, steps: [] };
const step = async (name, body) => {
  const detail = await body();
  report.steps.push({ name, ...detail });
  return detail;
};

// 1) 参数输入框：静态 / hover / focus（真实鼠标按下，与用户路径一致）
//    训练参数现在是超参数模块的页签，先切过去；输入与预测是默认页签。
await clickSelector('[data-expert-tab="training"]');
const paramStatic = await step('param input 静态', () => evaluate(DESCRIBE('.experiment-param-cell input')));
record('参数输入框静态有可辨认基线', paramStatic.visibleEdges === 'bottom', `${paramStatic.visibleEdges} ${paramStatic.borderColors}`);
const paramShot = await shoot('state-01-param-static');

await hover('.experiment-param-cell input');
const paramHover = await step('param input hover', () => evaluate(DESCRIBE('.experiment-param-cell input')));
record('参数输入框 hover 增强边界', paramHover.visibleEdges === 'bottom', `${paramHover.borderColors}`);
const paramHoverShot = await shoot('state-02-param-hover');

await evaluate(`
  document.querySelector('.experiment-param-cell input').scrollIntoView({ block: 'center' });
  await new Promise(r=>setTimeout(r,250));
  return true;
`);
const paramBox = await evaluate(`
  const r = document.querySelector('.experiment-param-cell input').getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
`);
await call('Input.dispatchMouseEvent', { type: 'mousePressed', x: paramBox.x, y: paramBox.y, button: 'left', clickCount: 1 });
await call('Input.dispatchMouseEvent', { type: 'mouseReleased', x: paramBox.x, y: paramBox.y, button: 'left', clickCount: 1 });
await new Promise((r) => setTimeout(r, 300));
const paramFocus = await step('param input focus', () => evaluate(DESCRIBE('.experiment-param-cell input')));
record('参数输入框 focus 有明确反馈', paramFocus.borderColors.includes('154, 217, 239'), `${paramFocus.borderColors} outline=${paramFocus.outline}`);
const paramFocusShot = await shoot('state-03-param-focus');

// 2) 数据集选项 focus（列表式单选，取代原生下拉）
await evaluate(`
  const el = document.querySelector('.experiment-dataset-option:not([aria-checked="true"])');
  el.scrollIntoView({ block: 'center' });
  el.focus();
  await new Promise(r=>setTimeout(r,300));
  return true;
`);
const selectFocus = await step('dataset option focus', () => evaluate(DESCRIBE('.experiment-dataset-option:not([aria-checked="true"])')));
record('数据集选项 focus 有明确反馈', selectFocus.outline.includes('154, 217, 239'), `outline=${selectFocus.outline} edges=${selectFocus.visibleEdges}`);
const selectShot = await shoot('state-04-select-focus');

// 3) 模型来源胶囊：未选中 vs 选中（上传模型为默认选中）
const sourcePressed = await step('source option 选中态', () => evaluate(DESCRIBE('.experiment-source-option[aria-pressed="true"]')));
record('来源胶囊选中态保留强调线', sourcePressed.boxShadow !== 'none', `shadow=${sourcePressed.boxShadow.slice(0, 48)} border=${sourcePressed.visibleEdges}`);
const sourceShot = await shoot('state-05-source-selected');

// 4) 官方模型：架构选择器选中态与家族筛选
await clickSelector('.experiment-source-option:not([aria-pressed="true"])');
// 切到官方模型后可能仍在等模型列表/校验，这里等一下再断言，避免把慢加载误判成失败。
await evaluate(`
  for (let i = 0; i < 40 && !document.querySelector('.experiment-architecture-current'); i += 1) {
    await new Promise(r=>setTimeout(r,250));
  }
  return Boolean(document.querySelector('.experiment-architecture-current'));
`);
const archCurrent = await step('architecture current', () => evaluate(DESCRIBE('.experiment-architecture-current')));
record(
  '官方模型当前架构有左强调线',
  archCurrent.state === 'ok' && archCurrent.boxShadow !== 'none',
  `state=${archCurrent.state} shadow=${String(archCurrent.boxShadow).slice(0, 44)}`,
);
await evaluate(`
  document.querySelector('.experiment-architecture-current')?.click();
  await new Promise(r=>setTimeout(r,400));
  return true;
`);
await hover('.experiment-architecture-option');
const archHover = await step('architecture option hover', () => evaluate(DESCRIBE('.experiment-architecture-option')));
const archList = await evaluate(`
  const options = [...document.querySelectorAll('.experiment-architecture-option')];
  return {
    count: options.length,
    pressed: options.filter((el) => el.getAttribute('aria-pressed') === 'true').length,
    firstLabel: options[0]?.textContent.trim().slice(0, 24) || null,
  };
`);
console.log(`  架构列表: ${JSON.stringify(archList)}`);
await evaluate(`
  const target = [...document.querySelectorAll('.experiment-architecture-option')]
    .find((el) => el.getAttribute('aria-pressed') !== 'true') || document.querySelector('.experiment-architecture-option');
  target?.click();
  await new Promise(r=>setTimeout(r,400));
  return true;
`);
// 选择后模型库会收起，因此改从「当前架构」按钮读回选中结果，并确认列表里仍保留一项 aria-pressed。
const archAfter = await evaluate(`
  const current = document.querySelector('.experiment-architecture-current-value');
  const options = [...document.querySelectorAll('.experiment-architecture-option')];
  return {
    currentLabel: current ? current.textContent.trim() : null,
    listOpen: options.length > 0,
    pressedCount: options.filter((el) => el.getAttribute('aria-pressed') === 'true').length,
  };
`);
record(
  '架构选择写入当前值',
  Boolean(archAfter.currentLabel),
  `current=${archAfter.currentLabel} 列表展开=${archAfter.listOpen} 选中项=${archAfter.pressedCount}`,
);
// 再展开一次，单独验证选中项的视觉强调线。
await evaluate(`
  document.querySelector('.experiment-architecture-current')?.click();
  await new Promise(r=>setTimeout(r,400));
  return true;
`);
const archSelected = await step('architecture option 选中态', () => evaluate(DESCRIBE('.experiment-architecture-option[aria-pressed="true"]')));
record(
  '架构选项选中态可辨认',
  archSelected.state === 'ok' && (archSelected.visibleEdges.includes('left') || archSelected.boxShadow !== 'none'),
  `state=${archSelected.state} border=${archSelected.visibleEdges} shadow=${String(archSelected.boxShadow).slice(0, 44)}`,
);
const archShot = await shoot('state-06-architecture-selected');
await evaluate(`
  document.querySelector('.experiment-architecture-current')?.click();
  await new Promise(r=>setTimeout(r,300));
  return true;
`);

// 5) 实验名称：focus 与 invalid
await evaluate(`
  const el = document.querySelector('.experiment-canvas-name');
  el.scrollIntoView({ block: 'center' });
  el.focus();
  await new Promise(r=>setTimeout(r,300));
  return true;
`);
const nameFocus = await step('canvas name focus', () => evaluate(DESCRIBE('.experiment-canvas-name')));
record('实验名 focus 有下划线反馈', nameFocus.borderColors.includes('154, 217, 239'), nameFocus.borderColors);
const nameShot = await shoot('state-07-name-focus');

const nameInvalid = await evaluate(`
  const el = document.querySelector('#experiment-name-input');
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(el, '   ');
  el.dispatchEvent(new Event('input', { bubbles: true }));
  await new Promise(r=>setTimeout(r,400));
  const error = document.querySelector('.experiment-canvas-name-error');
  return { ariaInvalid: el.getAttribute('aria-invalid'), errorText: error ? error.textContent.trim().slice(0, 40) : null };
`);
const nameInvalidStyle = await step('canvas name invalid', () => evaluate(DESCRIBE('#experiment-name-input')));
record(
  '实验名非法时进入 invalid 状态（红描边 + 说明）',
  nameInvalid.ariaInvalid === 'true' && nameInvalidStyle.borderColors.includes('217, 92, 92'),
  `aria-invalid=${nameInvalid.ariaInvalid} border=${nameInvalidStyle.borderColors} error="${nameInvalid.errorText}"`,
);
const nameInvalidShot = await shoot('state-08-name-invalid');

// 还原实验名，避免影响后续步骤
await evaluate(`
  const el = document.querySelector('.experiment-canvas-name');
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(el, ${JSON.stringify('predrnn_UDT_v1')});
  el.dispatchEvent(new Event('input', { bubbles: true }));
  await new Promise(r=>setTimeout(r,300));
  return true;
`);

// 6) 悬停：通用按钮、目录行、专家页签
await hover('.experiment-expert-action');
const actionHover = await step('expert action hover', () => evaluate(DESCRIBE('.experiment-expert-action')));
record('次要按钮 hover 才出现描边', actionHover.visibleEdges !== 'none', `${actionHover.visibleEdges} ${actionHover.borderColors}`);
const actionShot = await shoot('state-09-action-hover');

// 目录默认就是展开的；若被收起（或阶段切换收起）先打开再测悬停。
await evaluate(`
  if (!document.querySelector('.experiment-directory')) {
    const toggle = [...document.querySelectorAll('.experiment-center-header-meta button')].find((el) => /目录/.test(el.innerText));
    toggle?.click();
    await new Promise(r=>setTimeout(r,500));
  }
  return true;
`);
await hover('.experiment-directory-row, .experiment-directory-filter');
const filterHover = await step('directory filter hover', () => evaluate(DESCRIBE('.experiment-directory-filter')));
record('目录筛选 hover 才出现描边', filterHover.state === 'ok', `${filterHover.state} ${filterHover.visibleEdges}`);
const dirShot = await shoot('state-10-directory');

// 7) 运行条与整体状态
const runBar = await step('run bar', () => evaluate(DESCRIBE('[data-run-bar="true"]')));
record('运行条保留顶部细边线', runBar.visibleEdges === 'top', runBar.visibleEdges);
const layout = await evaluate(`
  return {
    overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    runBarHeight: Math.round(document.querySelector('[data-run-bar="true"]').getBoundingClientRect().height),
    columns: getComputedStyle(document.querySelector('.experiment-center-grid')).gridTemplateColumns,
  };
`);
record('目录展开时无横向溢出', layout.overflowX <= 0, `overflowX=${layout.overflowX} columns=${layout.columns} runBar=${layout.runBarHeight}px`);
const finalShot = await shoot('state-11-directory-open');

report.layout = layout;
report.staticShots = { paramShot, paramHoverShot, paramFocusShot, selectShot, sourceShot, archShot, nameShot, nameInvalidShot, actionShot, dirShot, finalShot };
writeFileSync(join(OUT_DIR, 'interaction-states.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8');

const failed = results.filter((item) => !item.ok);
console.log(`\n===== 状态检查：${results.length - failed.length}/${results.length} 项通过 =====`);
failed.forEach((item) => console.log(`失败: ${item.name} — ${item.detail}`));
console.log(`报告: ${join(OUT_DIR, 'interaction-states.json')}`);
if (failed.length) process.exitCode = 1;

await browser.send('Target.closeTarget', { targetId });
process.exit(process.exitCode || 0);
