/**
 * 实验中心（`#/training`）桌面端字体层级验收。
 *
 * 覆盖：1440×900 与 1920×1080 两个桌面视口下的
 *   · 关键文字节点的 computed font-size / 颜色（三级层级对照）
 *   · 文字裁切检测（scrollWidth/Height 超出 clientWidth/Height）
 *   · 横向溢出、运行条高度、页尾预留
 *   · 每个视口的运行条、参数区、检查器截图与整页对照图
 *
 * 前置：
 *   1. 在 frontend/ 执行 npm run build（脚本读 dist 产物，不读源码）；
 *   2. 生产静态服务在 http://127.0.0.1:5173，后端在 http://127.0.0.1:8000；
 *   3. node scripts/audit/experiment-console/seed-probe-profile.mjs prepare --no-tasks；
 *   4. 启动带远程调试端口的无头 Edge：
 *      msedge --headless=new --remote-debugging-port=9333 --user-data-dir=<临时目录>
 *
 * 用法：
 *   node scripts/audit/experiment-console/verify-font-hierarchy.mjs --label before
 *   node scripts/audit/experiment-console/verify-font-hierarchy.mjs --label after
 *   node scripts/audit/experiment-console/verify-font-hierarchy.mjs --label mobile --viewports 1024x768,390x844
 * 输出：脚本文本报告 + shots/<label>/ 下的 PNG 截图与 font-metrics.json。
 */
import http from 'node:http';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const STATE_FILE = join(HERE, '.probe-profile.json');

const args = process.argv.slice(2);
function argOf(name, fallback) {
  const index = args.indexOf(`--${name}`);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
}

const LABEL = argOf('label', 'run');
const CDP_PORT = Number(argOf('port', '9333'));
const BASE = argOf('url', 'http://127.0.0.1:5173');
const THEME = argOf('theme', 'dark');
const OUT_DIR = join(HERE, 'shots', LABEL);
mkdirSync(OUT_DIR, { recursive: true });

/** 默认检查两个桌面视口；可用 --viewports 1440x900,1024x768,390x844 追加回归视口。 */
const VIEWPORTS = argOf('viewports', '1440x900,1920x1080')
  .split(',')
  .map((item) => item.trim())
  .filter(Boolean)
  .map((item) => item.split('x').map(Number))
  .map(([w, h]) => [w, h]);

/** 已在信息减法中删除的装饰 / 重复节点：保留 key 作为「确认不在页面上」的证据。 */
const REMOVED = '__removed__';

const CSS_OVERRIDE = argOf('css', '');

function readProbeToken() {
  if (!existsSync(STATE_FILE)) return '';
  try {
    return String(JSON.parse(readFileSync(STATE_FILE, 'utf8'))?.token || '');
  } catch {
    return '';
  }
}
/** `--guest` 用未登录状态跑一遍：检查器的错误 / 待处理状态需要单独核对。 */
const GUEST = args.includes('--guest');
/** `--ready` 先把实验名填成唯一值，用于核对「配置就绪」下的运行条与检查器。 */
const READY = args.includes('--ready');
/** `--official` 先切到官方模型，用于核对官方模型态的画布与检查器。 */
const OFFICIAL = args.includes('--official');
const TOKEN = GUEST ? '' : argOf('token', readProbeToken());

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
 * 用 `--css <文件或 URL>` 提供的样式表替换构建产物里的样式表。
 * 页面结构、登录状态与交互逻辑都来自当前构建，只有 CSS 被替换，
 * 因此可以在一份构建上对比「改前 / 改后」两套字号，避免构建间的 DOM 漂移。
 */
async function applyCssOverride() {
  if (!CSS_OVERRIDE) return;
  const source = /^https?:/.test(CSS_OVERRIDE)
    ? await (await fetch(CSS_OVERRIDE)).text()
    : readFileSync(CSS_OVERRIDE, 'utf8');
  await evaluate(`
    for (const link of document.querySelectorAll('link[rel="stylesheet"]')) link.disabled = true;
    let style = document.getElementById('audit-css-override');
    if (!style) { style = document.createElement('style'); style.id = 'audit-css-override'; document.head.appendChild(style); }
    style.textContent = ${JSON.stringify(source)};
    await new Promise(r=>setTimeout(r,500));
    return true;
  `);
  console.log(`已替换样式表: ${CSS_OVERRIDE}`);
}

/** 打开训练页并停在配置阶段（与 verify-console.mjs 同一条用户路径）。 */
async function openConfigure() {
  await call('Page.navigate', { url: `${BASE}/` });
  await new Promise((r) => setTimeout(r, 1200));
  await evaluate(`
    localStorage.setItem('astraatmos_settings', JSON.stringify({ language: 'zh', theme: ${JSON.stringify(THEME)} }));
    ${TOKEN ? `localStorage.setItem('astraatmos_token', ${JSON.stringify(TOKEN)});` : "localStorage.removeItem('astraatmos_token');"}
    return true;
  `);
  await call('Page.reload', { ignoreCache: true });
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
  let stage = null;
  for (let attempt = 0; attempt < 4 && stage !== 'configure'; attempt += 1) {
    await call('Page.navigate', { url: `${BASE}/#/training` });
    stage = await waitConfigured();
    if (stage === 'configure') break;
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

  // 官方模型态：切到官方模型并等架构选择器出现。
  if (OFFICIAL) {
    const switched = await evaluate(`
      const official = document.querySelector('[data-model-source-option="official"]');
      if (official && official.getAttribute('aria-pressed') !== 'true') official.click();
      for (let i = 0; i < 40 && !document.querySelector('.experiment-architecture-current'); i += 1) {
        await new Promise(r=>setTimeout(r,250));
      }
      return Boolean(document.querySelector('.experiment-architecture-current'));
    `);
    console.log(`  [official] 架构选择器就绪=${switched}`);
  }

  // 配置就绪状态：填一个唯一实验名称（默认名与占位符同名，登录后会被判重名）。
  if (READY) {
    const named = await evaluate(`
      for (let i = 0; i < 40 && !document.querySelector('#experiment-name-input'); i += 1) await new Promise(r=>setTimeout(r,250));
      const el = document.querySelector('#experiment-name-input');
      if (!el) return { ok: false };
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(el, 'final_visual_audit');
      el.dispatchEvent(new Event('input', { bubbles: true }));
      for (let i = 0; i < 40; i += 1) {
        const capsule = document.querySelector('.experiment-inspector-state');
        if (capsule && capsule.dataset.inspectorState === 'ready') break;
        await new Promise(r=>setTimeout(r,250));
      }
      const capsule = document.querySelector('.experiment-inspector-state');
      return { ok: true, value: el.value, state: capsule ? capsule.dataset.inspectorState : null };
    `);
    console.log(`  [ready] 实验名="${named.value}" 检查器状态=${named.state}`);
  }

  // 展开「管理模型」，让卡片、列表与格式说明的节点在采集前就存在且稳定
  // （模型列表是登录后异步加载的，收起状态下卡片节点会缺失）。
  await evaluate(`
    for (let i = 0; i < 40 && !document.querySelector('[data-uploaded-model-card="true"]'); i += 1) {
      await new Promise(r=>setTimeout(r,250));
    }
    const manage = [...document.querySelectorAll('button')].find((el) => /管理模型|Manage models/.test(el.innerText));
    if (manage && manage.getAttribute('aria-expanded') !== 'true') manage.click();
    await new Promise(r=>setTimeout(r,400));
    return true;
  `);
}

/**
 * 展开 / 收起「管理模型」。
 * 展开会多出模型列表、格式说明与校验详情（约 220px），会改变首屏几何，
 * 因此首屏与参数区采集前必须收起，只有需要卡片 / 列表节点时才展开。
 */
async function setUploadedManage(open) {
  await evaluate(`
    const manage = [...document.querySelectorAll('button')].find((el) => /管理模型|Manage models/.test(el.innerText));
    if (manage) {
      const isOpen = manage.getAttribute('aria-expanded') === 'true';
      if (isOpen !== ${open ? 'true' : 'false'}) {
        manage.click();
        await new Promise(r=>setTimeout(r,400));
      }
    }
    return true;
  `);
}

/** 采集指定选择器的 computed 样式（缺失时标记 not-found）。 */
const PROBES = {
  // 页头 / 阶段指示
  'header.eyebrow': '.experiment-center-eyebrow',
  'header.title': '.experiment-center-title',
  'header.stageTab': '.experiment-center-stage-tab',
  'header.stageIndex': REMOVED,
  // 画布头部
  'canvas.name': '.experiment-canvas-name',
  'canvas.meta': REMOVED,
  'canvas.state': REMOVED,
  // 分区标题与序号
  'section.index': '.experiment-section-index',
  'section.title': '.experiment-section-title',
  'section.hint': REMOVED,
  // 02 数据集 / 03 模型
  'task.choiceLabel': '.experiment-choice-label',
  'task.choiceLabelEm': REMOVED,
  'task.datasetOption': '.experiment-dataset-option strong',
  'task.datasetOptionHint': REMOVED,
  'task.select': REMOVED,
  'task.choiceMeta': REMOVED,
  'task.sourceOptionStrong': '.experiment-source-option strong',
  'task.sourceOptionSmall': REMOVED,
  'task.uploadedName': '.experiment-uploaded-name',
  'task.uploadedMeta': '.experiment-uploaded-meta',
  'task.uploadedStatus': '.experiment-uploaded-status',
  'task.uploadedIcon': REMOVED,
  'task.uploadedEmpty': '.experiment-uploaded-empty',
  'task.uploadedLink': '.experiment-uploaded-link',
  // 02 载荷条与序列图示
  'payload.lock': '.experiment-payload-lock',
  'payload.lockSpan': '.experiment-payload-lock span',
  'payload.chip': '.experiment-channel-chip',
  'payload.chipB': '.experiment-channel-chip b',
  'payload.count': '.experiment-payload-count',
  'flow.caption': REMOVED,
  'flow.frame': REMOVED,
  'flow.model': REMOVED,
  'flow.modelSmall': REMOVED,
  // 03 参数矩阵
  'params.label': '.experiment-param-label',
  'params.labelCode': REMOVED,
  'params.input': '.experiment-param-cell input',
  'params.unit': REMOVED,
  // 04 专家参数
  'expert.tab': '.experiment-expert-tab',
  'expert.tabCount': '.experiment-expert-tab-count',
  'expert.heading': '.experiment-expert-heading',
  'expert.note': '.experiment-expert-note',
  'expert.fieldLabel': '.experiment-expert-field > span',
  'expert.fieldInput': '.experiment-expert-field input',
  'expert.ok': '.experiment-expert-ok',
  'expert.action': '.experiment-expert-action',
  // 检查器
  'inspector.title': '.experiment-inspector-title',
  'inspector.state': '.experiment-inspector-state',
  'inspector.label': '.experiment-inspector-label',
  'inspector.value': '.experiment-inspector-value',
  'inspector.sub': '.experiment-inspector-sub',
  'inspector.row': '.experiment-inspector-readiness-row',
  'inspector.check': '.experiment-inspector-check',
  'inspector.cellSmall': REMOVED,
  'inspector.cellStrong': REMOVED,
  'inspector.detailsRow': REMOVED,
  'inspector.note': '.experiment-inspector-note',
  'inspector.link': '.experiment-inspector-link',
  // 运行条
  'runbar.state': '.experiment-run-bar-state',
  'runbar.meta': '[data-run-bar-summary]',
  'runbar.note': '.experiment-run-bar-note',
  'runbar.start': '.experiment-run-bar-start',
  // 目录（展开后）
  'directory.title': '.experiment-directory-title',
  'directory.filter': '.experiment-directory-filter',
  'directory.name': '.experiment-directory-name',
  'directory.id': '.experiment-directory-id',
  'directory.meta': '.experiment-directory-meta',
  'directory.badge': '.experiment-directory-badge',
};

/** 需要统计「静态描边 / 面层」的分区容器（第二轮卡片层级整改用）。 */
const FRAME_PROBES = [
  ['task.choiceBlock', '.experiment-choice-block'],
  ['task.sourceOption', '.experiment-source-option'],
  ['task.uploadedCard', '.experiment-uploaded-card'],
  ['task.uploadedItem', '.experiment-uploaded-item'],
  ['task.archOption', '.experiment-architecture-option'],
  ['payload.bar', '.experiment-payload-bar'],
  ['payload.chip', '.experiment-channel-chip'],
  ['params.cell', '.experiment-param-cell'],
  ['params.input', '.experiment-param-cell input'],
  ['expert.shell', '.experiment-expert'],
  ['expert.field', '.experiment-expert-field'],
  ['expert.fieldInput', '.experiment-expert-field input'],
  ['inspector.cell', '.experiment-inspector-cell'],
  ['inspector.block', '.experiment-inspector-block'],
  ['runbar', '[data-run-bar="true"]'],
  ['directory.row', '.experiment-directory-row'],
  ['directory.filter', '.experiment-directory-filter'],
  ['inspector.head', '.experiment-inspector-head'],
  ['inspector.issue.mark', '.experiment-inspector-issue-mark'],
  ['inspector.issue.action', '.experiment-inspector-issue-action'],
  ['inspector.check.ok', '.experiment-inspector-readiness-row[data-ok="true"] .experiment-inspector-check'],
  ['inspector.check.bad', '.experiment-inspector-readiness-row[data-ok="false"] .experiment-inspector-check'],
  ['inspector.link', '.experiment-inspector-link'],
];

/** 检查器专属读数（第四步）：区块几何、静态分隔线与状态色。 */
const INSPECTOR_METRICS = `
  const inspector = document.querySelector('.experiment-inspector-panel');
  const blocks = [...document.querySelectorAll('.experiment-inspector-block')];
  const rectOf = (el) => { const r = el.getBoundingClientRect(); return { t: Math.round(r.top), b: Math.round(r.bottom), h: Math.round(r.height) }; };
  const alphaOf = (color) => {
    const m = String(color).match(/rgba?\\(([^)]+)\\)/);
    if (!m) return 1;
    const parts = m[1].split(',').map((v) => parseFloat(v));
    return parts.length > 3 ? parts[3] : 1;
  };
  const dividers = [];
  for (const el of document.querySelectorAll('.experiment-inspector-head, .experiment-inspector-block')) {
    const cs = getComputedStyle(el);
    const width = parseFloat(cs.borderBottomWidth);
    if (width > 0 && alphaOf(cs.borderBottomColor) > 0.01) {
      dividers.push({ cls: String(el.className).split(' ')[0], color: cs.borderBottomColor.replace(/\\s+/g, '') });
    }
  }
  const issueItems = [...document.querySelectorAll('[data-inspector-issue]')];
  return {
    panel: inspector ? rectOf(inspector) : null,
    panelOverflow: inspector ? inspector.scrollHeight - inspector.clientHeight : null,
    head: rectOf(document.querySelector('.experiment-inspector-head')),
    blockCount: blocks.length,
    blockGaps: blocks.slice(1).map((el, i) => Math.round(el.getBoundingClientRect().top - blocks[i].getBoundingClientRect().bottom)),
    blocks: blocks.map((el) => ({ ...rectOf(el), label: el.querySelector('.experiment-inspector-label')?.textContent.trim() || '' })),
    dividerCount: dividers.length,
    dividers,
    // 信息减法后：正常态只有一行结论，问题态逐条列出可定位的原因。
    allClear: Boolean(document.querySelector('[data-inspector-clear="true"]')),
    issueCount: issueItems.length,
    issues: issueItems.map((el) => ({
      key: el.dataset.inspectorIssue,
      text: (el.querySelector('.experiment-inspector-issue-text')?.textContent || '').trim(),
      action: el.querySelector('.experiment-inspector-issue-action')?.textContent.trim() || null,
    })),
    statePill: (() => {
      const el = document.querySelector('.experiment-inspector-state');
      if (!el) return null;
      const cs = getComputedStyle(el);
      return { state: el.dataset.inspectorState, ready: el.dataset.inspectorReady, text: el.textContent.trim(), color: cs.color, background: cs.backgroundColor, border: cs.borderTopColor };
    })(),
    titleWeight: (() => { const el = document.querySelector('.experiment-inspector-title'); return el ? getComputedStyle(el).fontWeight : null; })(),
    labelColor: (() => { const el = document.querySelector('.experiment-inspector-label'); return el ? getComputedStyle(el).color : null; })(),
    valueColor: (() => { const el = document.querySelector('.experiment-inspector-value'); return el ? getComputedStyle(el).color : null; })(),
    subColor: (() => { const el = document.querySelector('.experiment-inspector-sub'); return el ? getComputedStyle(el).color : null; })(),
    blockers: document.querySelectorAll('.experiment-inspector-blockers li').length,
  };
`;

const collectMetrics = (probeEntries) => `
  const rect = (sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { l: Math.round(r.left), r: Math.round(r.right), w: Math.round(r.width), t: Math.round(r.top), b: Math.round(r.bottom), h: Math.round(r.height) };
  };
  const probes = ${JSON.stringify(probeEntries)};
  const styles = {};
  for (const [key, selector] of Object.entries(probes)) {
    if (selector === '__removed__') { styles[key] = { state: 'removed' }; continue; }
    const el = document.querySelector(selector);
    if (!el) { styles[key] = { state: 'not-found', selector }; continue; }
    const cs = getComputedStyle(el);
    const text = (el.innerText || el.value || el.getAttribute('placeholder') || '').trim().replace(/\\s+/g, ' ').slice(0, 44);
    const clipped = el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1;
    styles[key] = {
      state: 'ok', selector, text,
      size: parseFloat(cs.fontSize), weight: cs.fontWeight, color: cs.color,
      family: cs.fontFamily.split(',')[0].replace(/["']/g, ''),
      clipped,
    };
  }

  // 描边 / 面层：静态渲染里某一个元素究竟靠什么划出边界。
  const frameProbes = ${JSON.stringify(FRAME_PROBES)};
  const frames = {};
  for (const [key, selector] of frameProbes) {
    const el = document.querySelector(selector);
    if (!el) { frames[key] = { state: 'not-found' }; continue; }
    const cs = getComputedStyle(el);
    const widths = [cs.borderTopWidth, cs.borderRightWidth, cs.borderBottomWidth, cs.borderLeftWidth];
    const colors = [cs.borderTopColor, cs.borderRightColor, cs.borderBottomColor, cs.borderLeftColor];
    const visible = (name, width, color) => {
      const px = parseFloat(width);
      if (!(px > 0) || px !== px) return false;
      const m = color.match(/rgba?\\(([^)]+)\\)/);
      if (!m) return true;
      const parts = m[1].split(',').map((v) => parseFloat(v));
      const alpha = parts.length > 3 ? parts[3] : 1;
      return alpha > 0.01;
    };
    const visibleEdges = ['top', 'right', 'bottom', 'left'].filter((edge, i) => visible(edge, widths[i], colors[i]));
    frames[key] = {
      state: 'ok',
      borderWidths: widths.map((w) => Math.round(parseFloat(w) * 100) / 100).join('/'),
      borderColors: colors.map((c) => c.replace(/\\s+/g, '')).join('|'),
      visibleEdges: visibleEdges.join('') || 'none',
      fullStroke: visibleEdges.length === 4,
      background: cs.backgroundColor,
      backgroundImage: cs.backgroundImage === 'none' ? 'none' : 'gradient',
      boxShadow: cs.boxShadow === 'none' ? 'none' : cs.boxShadow.slice(0, 40),
      radius: cs.borderTopLeftRadius,
    };
  }

  // 文字裁切扫描：只统计确实含文字、且样式会隐藏溢出的叶子节点。
  const clips = [];
  for (const el of document.querySelectorAll('.experiment-center *')) {
    if (el.classList.contains('sr-only')) continue;
    if (!el.childElementCount && (el.textContent || '').trim()) {
      const cs = getComputedStyle(el);
      const overflowHidden = cs.overflow !== 'visible' || cs.textOverflow === 'ellipsis';
      if (!overflowHidden) continue;
      const dw = el.scrollWidth - el.clientWidth;
      const dh = el.scrollHeight - el.clientHeight;
      if (dw > 1 || dh > 1) {
        clips.push({
          cls: typeof el.className === 'string' ? el.className.split(' ')[0] : '',
          text: (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 40),
          dw, dh,
        });
      }
    }
  }

  const grid = document.querySelector('.experiment-center-grid');
  const inspector = document.querySelector('.experiment-inspector-panel');
  const runBar = document.querySelector('[data-run-bar="true"]');
  const sectionRects = {};
  for (const group of ['name', 'dataset', 'model', 'expert']) {
    const el = document.querySelector('[data-config-group="' + group + '"]');
    if (el) {
      const r = el.getBoundingClientRect();
      sectionRects[group] = {
        t: Math.round(r.top), b: Math.round(r.bottom), h: Math.round(r.height),
        fullyVisible: r.bottom <= window.innerHeight,
        partiallyVisible: r.top < window.innerHeight && r.bottom > window.innerHeight,
      };
    }
  }
  // 首屏（运行条上沿以上）能看到多少内容，用于核对信息密度。
  const runBarTop = runBar ? runBar.getBoundingClientRect().top : window.innerHeight;
  const firstScreenOf = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      t: Math.round(r.top), b: Math.round(r.bottom),
      insideRunBar: r.bottom <= runBarTop + 1,
      overlapsRunBar: r.top < runBarTop && r.bottom > runBarTop,
    };
  };
  const groupFirstScreen = {};
  for (const group of ['name', 'dataset', 'model', 'expert']) {
    const el = document.querySelector('[data-config-group="' + group + '"]');
    if (!el) continue;
    const r = el.getBoundingClientRect();
    groupFirstScreen[group] = r.top < runBarTop
      ? (r.bottom <= runBarTop ? 'full' : Math.round(runBarTop - r.top))
      : 'none';
  }
  return {
    viewport: { w: document.documentElement.clientWidth, h: window.innerHeight },
    scrollY: Math.round(window.scrollY),
    scrollHeight: document.documentElement.scrollHeight,
    maxScroll: Math.round(document.documentElement.scrollHeight - window.innerHeight),
    overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    columns: grid ? getComputedStyle(grid).gridTemplateColumns : null,
    directory: grid ? grid.dataset.directory : null,
    inspectorMode: grid ? grid.dataset.inspector : null,
    rail: rect('.experiment-directory'),
    canvas: rect('.experiment-center-canvas'),
    inspector: rect('.experiment-inspector-panel'),
    runBar: rect('[data-run-bar="true"]'),
    runBarPosition: runBar ? getComputedStyle(runBar).position : null,
    runBarState: document.querySelector('.experiment-run-bar-state')?.dataset.runState || null,
    runBarSummary: document.querySelector('[data-run-bar-summary]')?.textContent?.trim() || '',
    inspectorOverflow: inspector ? inspector.scrollHeight - inspector.clientHeight : null,
    sections: sectionRects,
    groupFirstScreen,
    datasetList: firstScreenOf('.experiment-dataset-list'),
    datasetBlock: firstScreenOf('.experiment-choice-block:not([data-model-block])'),
    modelBlock: firstScreenOf('.experiment-choice-block[data-model-block]'),
    payloadBar: firstScreenOf('.experiment-payload-bar'),
    paramGrid: firstScreenOf('.experiment-param-grid'),
    runBarTop: Math.round(runBarTop),
    inspectorReport: (() => { ${INSPECTOR_METRICS} })(),
    clips,
    styles,
    frames,
  };
`;

async function collect() {
  return evaluate(collectMetrics(PROBES));
}

/** 输入与预测、训练参数现在是超参数模块的页签，只有激活的那一页才在 DOM 里。 */
async function activateExpertTab(tab) {
  return evaluate(`
    const target = document.querySelector('[data-expert-tab="${tab}"]');
    if (!target) return false;
    target.click();
    await new Promise((r) => setTimeout(r, 300));
    return true;
  `);
}

async function capture(name, { fullPage = false } = {}) {
  const params = { format: 'png' };
  if (fullPage) params.captureBeyondViewport = true;
  const { data } = await call('Page.captureScreenshot', params);
  const file = join(OUT_DIR, `${name}.png`);
  writeFileSync(file, Buffer.from(data, 'base64'));
  return file;
}

const report = { label: LABEL, browser: version.Browser, viewports: {} };
const SIZE_RE = /^(9|10|11)px$/;

for (const [width, height] of VIEWPORTS) {
  const tag = `${width}x${height}`;
  console.log(`\n=== 视口 ${tag} ===`);
  await setViewport(width, height);
  await openConfigure();
  await applyCssOverride();
  await setViewport(width, height);

  // 1) 首屏：默认展开目录（三列）+ 收起「管理模型」，等布局稳定再采集，
  //   避免异步加载（模型列表、任务列表）改变高度后读到中间态。
  await setUploadedManage(false);
  await evaluate(`
    window.scrollTo(0, 0);
    let last = '';
    let stable = 0;
    for (let i = 0; i < 60; i += 1) {
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const task = document.querySelector('[data-config-group="name"]');
      const snapshot = document.documentElement.scrollHeight + ':' + (task ? Math.round(task.getBoundingClientRect().height) : 0);
      stable = snapshot === last ? stable + 1 : 0;
      last = snapshot;
      if (stable >= 5) break;
    }
    window.scrollTo(0, 0);
    await new Promise((r) => setTimeout(r, 150));
    return true;
  `);
  const first = await collect();
  const firstShot = await capture(`${tag}-01-first-screen`);

  // 1b) 展开「管理模型」，采集模型列表 / 格式说明 / 校验详情这些只在展开态存在的节点。
  await setUploadedManage(true);
  const manageOpen = await collect();
  const manageShot = await capture(`${tag}-01b-uploaded-manage`);
  await setUploadedManage(false);

  // 2) 参数区：训练参数与专家字段都是超参数的页签，先切到对应页签再采集。
  await activateExpertTab('training');
  const params = await evaluate(`
    const target = document.querySelector('[data-config-group="training"]');
    if (target) target.scrollIntoView({ block: 'center' });
    await new Promise(r=>setTimeout(r,400));
    ${collectMetrics(PROBES)}
  `);
  const paramsShot = await capture(`${tag}-02-params`);

  await activateExpertTab('customParams');
  const expert = await evaluate(`
    const target = document.querySelector('[data-config-group="expert"]');
    if (target) target.scrollIntoView({ block: 'start' });
    await new Promise(r=>setTimeout(r,400));
    ${collectMetrics(PROBES)}
  `);
  const expertShot = await capture(`${tag}-03-expert`);

  // 3) 检查器：滚动后仍吸附在视口内，单独截取检查器元素
  const inspector = await evaluate(`
    window.scrollTo(0, ${Math.round(height / 2)});
    await new Promise(r=>setTimeout(r,400));
    ${collectMetrics(PROBES)}
  `);
  const inspectorShot = await capture(`${tag}-04-inspector`);

  // 4) 收起实验目录（两列），确认目录开关后的画布字号与不溢出
  const expanded = await evaluate(`
    const toggle = [...document.querySelectorAll('.experiment-center-header-meta button')].find((el) => /目录/.test(el.innerText));
    toggle?.click();
    await new Promise(r=>setTimeout(r,600));
    ${collectMetrics(PROBES)}
  `);
  const expandedShot = await capture(`${tag}-05-directory-collapsed`);

  // 5) 页尾运行条
  const bottom = await evaluate(`
    window.scrollTo(0, document.documentElement.scrollHeight);
    await new Promise(r=>setTimeout(r,400));
    ${collectMetrics(PROBES)}
  `);
  const bottomShot = await capture(`${tag}-06-page-bottom`);

  // 6) 整页对照图（运行条临时隐藏：fixed 元素在 captureBeyondViewport 下会遮挡中部内容）
  await evaluate(`
    const bar = document.querySelector('[data-run-bar="true"]');
    if (bar) bar.setAttribute('data-audit-hidden', 'true');
    const style = document.createElement('style');
    style.id = 'audit-hide-runbar';
    style.textContent = '[data-audit-hidden="true"]{visibility:hidden !important;}';
    document.head.appendChild(style);
    window.scrollTo(0, 0);
    await new Promise(r=>setTimeout(r,300));
    return true;
  `);
  const fullPageFile = await capture(`${tag}-07-full-page`, { fullPage: true });
  await evaluate(`
    document.getElementById('audit-hide-runbar')?.remove();
    document.querySelector('[data-run-bar="true"]')?.removeAttribute('data-audit-hidden');
    return true;
  `);

  report.viewports[tag] = {
    firstScreen: { shot: firstShot, ...first },
    uploadedManage: { shot: manageShot, ...manageOpen },
    params: { shot: paramsShot, ...params },
    expert: { shot: expertShot, ...expert },
    inspector: { shot: inspectorShot, ...inspector },
    directoryOpen: { shot: expandedShot, ...expanded },
    pageBottom: { shot: bottomShot, ...bottom },
    fullPage: { shot: fullPageFile },
  };

  // 控制台摘要
  const fontDump = (metrics) => Object.entries(metrics.styles)
    .map(([key, value]) => `${key}=${value.state === 'ok' ? `${value.size}px` : 'n/a'}`)
    .join(' ');
  console.log(`首屏 columns=${first.columns} overflowX=${first.overflowX} 运行条=${first.runBar?.h}px`);
  console.log(`参数区: ${fontDump(params)}`);
  console.log(`检查器: ${fontDump(inspector)}`);
  console.log(`运行条: 状态=${inspector.styles['runbar.state']?.size}px 摘要=${inspector.styles['runbar.meta']?.size}px 主按钮=${inspector.styles['runbar.start']?.size}px 高=${bottom.runBar?.h}px`);
  const small = Object.entries(inspector.styles)
    .filter(([, v]) => v.state === 'ok' && SIZE_RE.test(`${v.size}px`))
    .map(([k, v]) => `${k}(${v.size}px)`);
  console.log(`检查器/运行条命中 9–11px 的节点: ${small.length ? small.join(', ') : '无'}`);
  console.log(`裁切节点: ${inspector.clips.length ? JSON.stringify(inspector.clips) : '无'}`);
  console.log(`收起目录 columns=${expanded.columns} overflowX=${expanded.overflowX}`);
  console.log(`页尾 scrollY=${bottom.scrollY}/${bottom.maxScroll} 运行条 bottom=${bottom.runBar?.b}/${bottom.viewport.h} 溢出=${bottom.overflowX}`);
  console.log(`首屏可见（scrollY=${first.scrollY} 运行条上沿 ${first.runBarTop}px）：${JSON.stringify(first.groupFirstScreen)}`);
  console.log(`  模型名称 h=${first.sections?.name?.h}px 数据集 h=${first.sections?.dataset?.h}px 数据集块 h=${first.datasetBlock?.h}px 模型块 h=${first.modelBlock?.h}px 载荷条 ${first.payloadBar?.t}–${first.payloadBar?.b}`);
  const ins = first.inspectorReport;
  if (ins) {
    console.log(`检查器: 面板 ${ins.panel?.h}px 内部滚动=${ins.panelOverflow} 区块 ${ins.blockCount} 个 区块间距=[${ins.blockGaps.join(',')}] 静态分隔线 ${ins.dividerCount} 条`);
    console.log(`  结论: ${ins.allClear ? '一行「通过」' : `${ins.issueCount} 条待处理`} ${ins.issues.map((i) => `${i.key}${i.action ? `[${i.action}]` : ''}`).join(' ')}`);
    console.log(`  状态胶囊: ${ins.statePill?.state} "${ins.statePill?.text}" 颜色=${ins.statePill?.color}`);
    console.log(`  层级色: 标签=${ins.labelColor} 值=${ins.valueColor} 说明=${ins.subColor} 标题字重=${ins.titleWeight}`);
  }
  const removedPresent = Object.entries(first.styles).filter(([, v]) => v.state === 'removed').length;
  console.log(`信息减法：${removedPresent} 个已删除的装饰 / 重复节点确认不在页面上`);

  // 第二轮：静态描边统计（整圈描边的容器数 = 视觉噪声的主要来源）
  const frameDump = (metrics) => Object.entries(metrics.frames)
    .filter(([, v]) => v.state === 'ok')
    .map(([k, v]) => `${k}=${v.fullStroke ? '整圈' : v.visibleEdges === 'none' ? '无边' : v.visibleEdges}`)
    .join(' ');
  const fullStrokes = (metrics) => Object.values(metrics.frames).filter((v) => v.state === 'ok' && v.fullStroke).length;
  console.log(`静态整圈描边容器数: 首屏=${fullStrokes(first)} 参数区=${fullStrokes(params)} 检查器=${fullStrokes(inspector)} 目录展开=${fullStrokes(expanded)}`);
  console.log(`描边明细（参数区）: ${frameDump(params)}`);
  console.log(`描边明细（检查器）: ${frameDump(inspector)}`);
}

writeFileSync(join(OUT_DIR, 'font-metrics.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(`\n报告: ${join(OUT_DIR, 'font-metrics.json')}`);
console.log(`截图: ${OUT_DIR}`);

await browser.send('Target.closeTarget', { targetId });
process.exit(0);
