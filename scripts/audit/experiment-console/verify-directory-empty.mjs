/**
 * 实验中心（`#/training`）· 左侧实验目录的空状态验收（第五步用）。
 *
 * 每个场景用**独立页面会话**（不复用 CDP session），避免注入的 fetch 包装残留到下一场景。
 * 覆盖：
 *   empty     确实没有实验（登录 + 0 个任务）
 *   error     取任务列表失败（注入 500；不改仓库代码）
 *   no-match  有实验但搜索 / 筛选无匹配（仅当验收账号里已有任务）
 *   ready     目录有内容（仅当验收账号里已有任务）
 *
 * 前置：构建、后端 8000、生产静态服务 5173、带 CDP 的无头 Edge、seed-probe-profile prepare。
 * 用法：node scripts/audit/experiment-console/verify-directory-empty.mjs --label step5
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
const LABEL = argOf('label', 'directory');
const CDP_PORT = Number(argOf('port', '9333'));
const BASE = argOf('url', 'http://127.0.0.1:5173');
const OUT_DIR = join(HERE, 'shots', LABEL);
mkdirSync(OUT_DIR, { recursive: true });

function readProbeToken() {
  if (!existsSync(STATE_FILE)) return '';
  try { return String(JSON.parse(readFileSync(STATE_FILE, 'utf8'))?.token || ''); } catch { return ''; }
}
const TOKEN = argOf('token', readProbeToken());

function httpJson(path) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port: CDP_PORT, path, method: 'GET' }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { body += c; });
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch (e) { reject(e); } });
    });
    req.on('error', reject);
    req.end();
  });
}

const version = await httpJson('/json/version');
const ws = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
let nextId = 1;
const pending = new Map();
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    const p = pending.get(m.id);
    pending.delete(m.id);
    m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result);
  }
});
const browserSend = (method, params = {}) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject });
  ws.send(JSON.stringify({ id, method, params }));
});

/** 每个场景开一个独立页面会话，互不残留。 */
let session = null;
async function newPage(width, height) {
  if (session) {
    await browserSend('Target.closeTarget', { targetId: session.targetId }).catch(() => {});
    session = null;
  }
  const { targetId } = await browserSend('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await browserSend('Target.attachToTarget', { targetId, flatten: true });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params, sessionId }));
  });
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  session = { targetId, send };
  return session;
}

async function evaluate(expression) {
  const r = await session.send('Runtime.evaluate', {
    expression: `(async () => { ${expression} })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval error');
  return r.result.value;
}

const waitConfigured = () => evaluate(`
  let stable = 0;
  for (let i = 0; i < 80; i += 1) {
    const grid = document.querySelector('.experiment-center-grid');
    const ok = grid && grid.dataset.stage === 'configure' && document.querySelector('[data-run-bar="true"]');
    stable = ok ? stable + 1 : 0;
    if (stable >= 6) return 'configure';
    await new Promise(r=>setTimeout(r,250));
  }
  return document.querySelector('.experiment-center-grid')?.dataset.stage || null;
`);

/** 打开训练页、进入配置阶段并展开目录；failTasks 时只让取任务列表失败。 */
async function openDirectory({ token, failTasks = false }) {
  if (failTasks) {
    await session.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `
        (function () {
          const original = window.fetch;
          window.fetch = function (input, init) {
            const url = typeof input === 'string' ? input : (input && input.url) || '';
            if (url.includes('/api/training/tasks') && (!init || !init.method || init.method === 'GET')) {
              return Promise.resolve(new Response(JSON.stringify({ detail: 'audit injected failure' }), { status: 500, headers: { 'Content-Type': 'application/json' } }));
            }
            return original.apply(this, arguments);
          };
        })();
      `,
    });
  }
  await session.send('Page.navigate', { url: `${BASE}/` });
  await new Promise((r) => setTimeout(r, 1200));
  const stored = await evaluate(`
    try {
      localStorage.removeItem('astraatmos_token');
      localStorage.setItem('astraatmos_settings', JSON.stringify({ language: 'zh', theme: 'dark' }));
      ${token ? `localStorage.setItem('astraatmos_token', ${JSON.stringify(token)});` : "localStorage.removeItem('astraatmos_token');"}
      return (localStorage.getItem('astraatmos_token') || '').length;
    } catch (e) { return -1; }
  `);
  if (stored < 0) throw new Error('无法写入 localStorage');
  await session.send('Page.reload', { ignoreCache: true });
  let stage = null;
  for (let attempt = 0; attempt < 4 && stage !== 'configure'; attempt += 1) {
    await session.send('Page.navigate', { url: `${BASE}/#/training` });
    stage = await waitConfigured();
    if (stage === 'configure') break;
    await evaluate(`[...document.querySelectorAll('.experiment-center-header-meta button')].find((el) => /新建实验/.test(el.innerText))?.click(); return true;`);
    stage = await waitConfigured();
  }
  if (stage !== 'configure') throw new Error(`未进入配置阶段: ${stage}`);
  await evaluate(`
    const toggle = [...document.querySelectorAll('.experiment-center-header-meta button')].find((el) => /目录/.test(el.innerText));
    if (toggle && toggle.innerText.includes('显示')) toggle.click();
    await new Promise(r=>setTimeout(r,500));
    let last = '';
    let stable = 0;
    for (let i = 0; i < 60; i += 1) {
      const dir = document.querySelector('.experiment-directory[data-directory-state]');
      const state = dir ? dir.dataset.directoryState : 'none';
      // 等状态离开 loading（首次取任务列表期间会短暂为 loading）。
      stable = (state === last && state !== 'loading') ? stable + 1 : 0;
      last = state;
      if (stable >= 4) break;
      await new Promise(r=>setTimeout(r,250));
    }
    document.documentElement.style.scrollBehavior = 'auto';
    return true;
  `);
}

const READ = `
  const dir = document.querySelector('.experiment-directory[data-directory-state]');
  const rectOf = (el) => { if (!el) return null; const r = el.getBoundingClientRect(); return { t: Math.round(r.top), b: Math.round(r.bottom), h: Math.round(r.height), w: Math.round(r.width) }; };
  const training = document.querySelector('.training-tag-directory');
  const filters = document.querySelector('.experiment-directory-filters');
  const empty = document.querySelector('.experiment-directory-empty');
  const list = document.querySelector('.experiment-directory-list');
  const search = document.querySelector('.training-tag-directory .training-tag-input');
  const manage = [...document.querySelectorAll('.training-tag-directory .training-tag-button')].find((el) => /管理标签/.test(el.innerText));
  const createAction = document.querySelector('[data-empty-action="create"]');
  const retryAction = document.querySelector('[data-empty-action="retry"]');
  const cs = (el, prop) => (el ? getComputedStyle(el)[prop] : null);
  const clips = [];
  for (const el of document.querySelectorAll('.experiment-directory *')) {
    // 实验名是设计上的单行省略号，不算裁切。
    if (el.classList.contains('experiment-directory-name') || el.classList.contains('sr-only')) continue;
    if (!el.childElementCount && (el.textContent || '').trim()) {
      const s = getComputedStyle(el);
      if (s.overflow === 'visible' && s.textOverflow !== 'ellipsis') continue;
      const dw = el.scrollWidth - el.clientWidth;
      const dh = el.scrollHeight - el.clientHeight;
      if (dw > 1 || dh > 1) clips.push({ cls: String(el.className).split(' ')[0], text: (el.textContent || '').trim().slice(0, 24), dw, dh });
    }
  }
  return {
    directoryState: dir ? dir.dataset.directoryState : null,
    trainingClass: training ? training.className : null,
    directory: rectOf(dir),
    canvas: rectOf(document.querySelector('.experiment-center-canvas')),
    inspector: rectOf(document.querySelector('.experiment-inspector-panel')),
    runBarTop: (() => { const el = document.querySelector('[data-run-bar="true"]'); return el ? Math.round(el.getBoundingClientRect().top) : null; })(),
    head: rectOf(document.querySelector('.experiment-directory-head')),
    filters: { rect: rectOf(filters), display: cs(filters, 'display'), buttons: document.querySelectorAll('.experiment-directory-filter').length },
    toolbar: rectOf(document.querySelector('.training-tag-directory .training-tag-toolbar')),
    search: { rect: rectOf(search), minHeight: cs(search, 'minHeight'), background: cs(search, 'backgroundColor'), visible: Boolean(search && search.offsetParent) },
    manage: { rect: rectOf(manage), visible: Boolean(manage && manage.offsetParent) },
    list: { rect: rectOf(list), flex: cs(list, 'flex') },
    empty: {
      rect: rectOf(empty),
      reason: empty ? empty.dataset.emptyReason : null,
      text: empty ? empty.querySelector('.experiment-directory-empty-text')?.textContent.trim() : null,
      role: empty ? empty.getAttribute('role') : null,
      color: empty ? cs(empty.querySelector('.experiment-directory-empty-text'), 'color') : null,
      fontSize: empty ? cs(empty.querySelector('.experiment-directory-empty-text'), 'fontSize') : null,
      hasCreate: Boolean(createAction),
      hasRetry: Boolean(retryAction),
    },
    tagFilterVisible: Boolean(document.querySelector('.training-tag-filter') && document.querySelector('.training-tag-filter').offsetParent),
    rows: document.querySelectorAll('.experiment-directory-row').length,
    overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    clips,
  };
`;

const shoot = async (name) => {
  const { data } = await session.send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(join(OUT_DIR, `${name}.png`), Buffer.from(data, 'base64'));
};

const report = { label: LABEL, viewports: {} };
const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok: Boolean(ok), detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} — ${detail}`);
};

for (const [width, height] of [[1440, 900], [1920, 1080]]) {
  const tag = `${width}x${height}`;

  // --- 目录状态（取决于验收账号里有没有任务）------------------------------
  console.log(`\n=== ${tag} · 目录状态 ===`);
  await newPage(width, height);
  await openDirectory({ token: TOKEN });
  const empty = await evaluate(READ);
  await shoot(`${tag}-A-directory`);
  report.viewports[`${tag}-empty`] = empty;
  const accountHasTasks = empty.directoryState === 'ready';
  console.log(`  状态=${empty.directoryState} 目录高=${empty.directory?.h}px 筛选栏=${empty.filters.display} 行数=${empty.rows} 标签面板=${empty.tagFilterVisible}`);
  console.log(`  空态: reason=${empty.empty.reason} "${empty.empty.text}" 新建按钮=${empty.empty.hasCreate} 重试=${empty.empty.hasRetry}`);
  console.log(`  搜索框高=${empty.search.rect?.h}px 标签管理可见=${empty.manage.visible} 溢出X=${empty.overflowX} 裁切=${JSON.stringify(empty.clips)}`);

  if (!accountHasTasks) {
    check(`${tag} 空目录标记为 empty`, empty.directoryState === 'empty', String(empty.directoryState));
    check(`${tag} 空目录隐藏状态筛选栏（DOM 仍保留 4 个按钮）`, empty.filters.display === 'none' && empty.filters.buttons === 4, `${empty.filters.display} buttons=${empty.filters.buttons}`);
    check(`${tag} 空目录只占内容高度（< 420px）`, empty.directory?.h > 0 && empty.directory.h < 420, `${empty.directory?.h}px`);
    check(`${tag} 空态给出「新建实验」动作`, empty.empty.hasCreate && empty.empty.reason === 'no-experiments', `reason=${empty.empty.reason} create=${empty.empty.hasCreate}`);
    check(`${tag} 空目录无裁切与横向溢出`, empty.clips.length === 0 && empty.overflowX <= 0, `clips=${empty.clips.length} overflowX=${empty.overflowX}`);
    check(`${tag} 搜索与标签管理仍可用`, empty.search.visible && empty.manage.visible, `search=${empty.search.visible} manage=${empty.manage.visible}`);
    if (width === 1440) {
      const before = await evaluate(`return document.querySelector('.experiment-center-grid')?.dataset.stage || null;`);
      await evaluate(`document.querySelector('[data-empty-action="create"]')?.click(); await new Promise(r=>setTimeout(r,500)); return true;`);
      const after = await evaluate(`return document.querySelector('.experiment-center-grid')?.dataset.stage || null;`);
      check(`${tag} 空态「新建实验」可点击并保持配置阶段`, before === 'configure' && after === 'configure', `${before} -> ${after}`);
    }
  } else {
    console.log(`  （账号已有 ${empty.rows} 个实验：核对有内容时的筛选栏、行渲染与筛选无匹配）`);
    check(`${tag} 有内容时筛选栏保留且行渲染`, empty.filters.display !== 'none' && empty.filters.buttons === 4 && empty.rows > 0, `rows=${empty.rows} filters=${empty.filters.display}`);
    check(`${tag} 有内容时目录撑满侧栏（列表可滚动）`, empty.directory?.h > 400 && empty.list.flex.includes('1 1 auto'), `dir=${empty.directory?.h}px listFlex=${empty.list.flex}`);
    check(`${tag} 有内容时无裁切与横向溢出`, empty.clips.length === 0 && empty.overflowX <= 0, `clips=${empty.clips.length} overflowX=${empty.overflowX}`);
    const noMatch = await evaluate(`
      const input = document.querySelector('.training-tag-directory .training-tag-input');
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(input, 'zzz-no-such-experiment');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise(r=>setTimeout(r,500));
      ${READ}
    `);
    await shoot(`${tag}-A2-no-match`);
    report.viewports[`${tag}-no-match`] = noMatch;
    check(`${tag} 搜索无匹配标记 no-match 且筛选栏保留`, noMatch.empty.reason === 'no-match' && noMatch.filters.display !== 'none', `reason=${noMatch.empty.reason} filters=${noMatch.filters.display}`);
    check(`${tag} 搜索无匹配时不显示「新建实验」动作`, noMatch.empty.hasCreate === false, `create=${noMatch.empty.hasCreate}`);
  }
  await session.send('Target.closeTarget', { targetId: session.targetId });

  // --- 取任务列表失败 -----------------------------------------------------
  console.log(`\n=== ${tag} · 加载失败（注入 500）===`);
  await newPage(width, height);
  await openDirectory({ token: TOKEN, failTasks: true });
  const failed = await evaluate(READ);
  await shoot(`${tag}-B-error`);
  report.viewports[`${tag}-error`] = failed;
  console.log(`  状态=${failed.directoryState} 空态 reason=${failed.empty.reason} "${failed.empty.text}" 重试=${failed.empty.hasRetry} role=${failed.empty.role} 色=${failed.empty.color}`);
  check(`${tag} 取任务失败标记 error`, failed.directoryState === 'error' && failed.empty.reason === 'error', `${failed.directoryState}/${failed.empty.reason}`);
  check(`${tag} 失败态提供重试且为 alert`, failed.empty.hasRetry && failed.empty.role === 'alert', `retry=${failed.empty.hasRetry} role=${failed.empty.role}`);
  check(`${tag} 失败态不误报「暂无实验」`, !/暂无实验/.test(failed.empty.text || ''), String(failed.empty.text));
  await session.send('Target.closeTarget', { targetId: session.targetId });
}

writeFileSync(join(OUT_DIR, 'directory-states.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8');
const failedFinal = results.filter((item) => !item.ok);
console.log(`\n===== 目录空状态检查：${results.length - failedFinal.length}/${results.length} 项通过 =====`);
failedFinal.forEach((item) => console.log(`失败: ${item.name} — ${item.detail}`));
console.log(`报告: ${join(OUT_DIR, 'directory-states.json')}`);
if (failedFinal.length) process.exitCode = 1;

process.exit(process.exitCode || 0);
