/**
 * 实验中心（`#/training`）· 检查器状态视觉验收（第四步用）。
 *
 * 用真实交互分别构造两种状态并记录读数 + 截图：
 *   A. 全部就绪：登录 + 有效上传模型 + 填写唯一实验名称
 *   B. 存在错误 / 待处理：未登录（或清空实验名称）
 * 记录：状态胶囊、就绪行记号与颜色、静态分隔线数、区块几何、标签/值/说明色、
 *       面板内部滚动、文字裁切、横向溢出。
 *
 * 用法（前置同 verify-font-hierarchy.mjs）：
 *   node scripts/audit/experiment-console/verify-inspector-states.mjs --label step4
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
const LABEL = argOf('label', 'inspector');
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
const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject });
  ws.send(JSON.stringify(sessionId ? { id, method, params, sessionId } : { id, method, params }));
});
const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
const call = (m, p) => send(m, p, sessionId);
await call('Page.enable');
await call('Runtime.enable');

async function evaluate(expression) {
  const r = await call('Runtime.evaluate', { expression: `(async () => { ${expression} })()`, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval error');
  return r.result.value;
}

const READ = `
  const alphaOf = (color) => {
    const m = String(color).match(/rgba?\\(([^)]+)\\)/);
    if (!m) return 1;
    const parts = m[1].split(',').map((v) => parseFloat(v));
    return parts.length > 3 ? parts[3] : 1;
  };
  const inspector = document.querySelector('.experiment-inspector-panel');
  const blocks = [...document.querySelectorAll('.experiment-inspector-block')];
  const dividers = [...document.querySelectorAll('.experiment-inspector-head, .experiment-inspector-block')]
    .filter((el) => parseFloat(getComputedStyle(el).borderBottomWidth) > 0 && alphaOf(getComputedStyle(el).borderBottomColor) > 0.01)
    .length;
  const issueItems = [...document.querySelectorAll('[data-inspector-issue]')];
  const clips = [];
  for (const el of document.querySelectorAll('.experiment-inspector *')) {
    if (el.classList.contains('sr-only')) continue;
    if (!el.childElementCount && (el.textContent || '').trim()) {
      const cs = getComputedStyle(el);
      if (cs.overflow === 'visible' && cs.textOverflow !== 'ellipsis') continue;
      const dw = el.scrollWidth - el.clientWidth;
      const dh = el.scrollHeight - el.clientHeight;
      if (dw > 1 || dh > 1) clips.push({ cls: String(el.className).split(' ')[0], text: (el.textContent || '').trim().slice(0, 30), dw, dh });
    }
  }
  return {
    panelWidth: inspector ? Math.round(inspector.getBoundingClientRect().width) : null,
    panelHeight: inspector ? Math.round(inspector.getBoundingClientRect().height) : null,
    panelOverflow: inspector ? inspector.scrollHeight - inspector.clientHeight : null,
    dividerCount: dividers,
    blockCount: blocks.length,
    blockGaps: blocks.slice(1).map((el, i) => Math.round(el.getBoundingClientRect().top - blocks[i].getBoundingClientRect().bottom)),
    capsule: (() => {
      const el = document.querySelector('.experiment-inspector-state');
      if (!el) return null;
      const cs = getComputedStyle(el);
      return { state: el.dataset.inspectorState, ready: el.dataset.inspectorReady, text: el.textContent.trim(), color: cs.color, background: cs.backgroundColor };
    })(),
    currentModel: document.querySelector('[data-inspector-field="current-model"]')?.textContent.trim() || null,
    modelSource: document.querySelector('[data-inspector-field="model-source"]')?.textContent.trim() || null,
    allClear: Boolean(document.querySelector('[data-inspector-clear="true"]')),
    allClearText: document.querySelector('[data-inspector-clear="true"]')?.textContent.trim() || null,
    issues: issueItems.map((el) => ({
      key: el.dataset.inspectorIssue,
      text: (el.querySelector('.experiment-inspector-issue-text')?.textContent || '').trim(),
      mark: (el.querySelector('.experiment-inspector-issue-mark')?.textContent || '').trim(),
      markBg: (() => { const m = el.querySelector('.experiment-inspector-issue-mark'); return m ? getComputedStyle(m).backgroundColor : null; })(),
      action: (el.querySelector('.experiment-inspector-issue-action')?.textContent || '').trim() || null,
    })),
    typography: {
      labelColor: (() => { const el = document.querySelector('.experiment-inspector-label'); return el ? getComputedStyle(el).color : null; })(),
      labelSize: (() => { const el = document.querySelector('.experiment-inspector-label'); return el ? getComputedStyle(el).fontSize : null; })(),
      valueColor: (() => { const el = document.querySelector('.experiment-inspector-value'); return el ? getComputedStyle(el).color : null; })(),
      valueSize: (() => { const el = document.querySelector('.experiment-inspector-value'); return el ? getComputedStyle(el).fontSize : null; })(),
      subColor: (() => { const el = document.querySelector('.experiment-inspector-sub'); return el ? getComputedStyle(el).color : null; })(),
      subSize: (() => { const el = document.querySelector('.experiment-inspector-sub'); return el ? getComputedStyle(el).fontSize : null; })(),
      issueColor: (() => { const el = document.querySelector('.experiment-inspector-blockers li'); return el ? getComputedStyle(el).color : null; })(),
    },
    // 信息减法后不应再出现的节点
    removedNodes: ['experiment-inspector-telemetry', 'experiment-inspector-details', 'experiment-inspector-cell']
      .filter((cls) => document.querySelector('.' + cls)),
    overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    clips,
    runBarState: document.querySelector('.experiment-run-bar-state')?.dataset.runState || null,
    blockedCount: issueItems.length,
  };
`;

const OPEN = async (width, height, { token }) => {
  await call('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  // 与 verify-console / verify-font-hierarchy 相同的登录路径：先到首页，再写 localStorage。
  // 每轮先清掉上一轮残留（未登录 / 已登录），并等文档真正落到 BASE 源后再写。
  await call('Page.navigate', { url: `${BASE}/` });
  await new Promise((r) => setTimeout(r, 1200));
  const stored = await evaluate(`
    for (let i = 0; i < 20; i += 1) {
      const ready = document.readyState !== 'loading' && location.origin === ${JSON.stringify(new URL(BASE).origin)};
      if (ready) {
        try {
          localStorage.removeItem('astraatmos_token');
          localStorage.setItem('astraatmos_settings', JSON.stringify({ language: 'zh', theme: 'dark' }));
          ${token ? `localStorage.setItem('astraatmos_token', ${JSON.stringify(token)});` : "localStorage.removeItem('astraatmos_token');"}
          return { ok: true, tokenLen: (localStorage.getItem('astraatmos_token') || '').length };
        } catch (e) { /* 文档仍在切换，重试 */ }
      }
      await new Promise(r=>setTimeout(r,250));
    }
    return { ok: false, tokenLen: -1 };
  `);
  if (!stored.ok) throw new Error(`未能写入 ${BASE} 的 localStorage`);
  console.log(`  [debug] 写入 localStorage token=${stored.tokenLen}`);
  await call('Page.reload', { ignoreCache: true });
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
  let stage = null;
  for (let attempt = 0; attempt < 4 && stage !== 'configure'; attempt += 1) {
    await call('Page.navigate', { url: `${BASE}/#/training` });
    stage = await waitConfigured();
    if (stage === 'configure') break;
    await evaluate(`[...document.querySelectorAll('.experiment-center-header-meta button')].find((el) => /新建实验/.test(el.innerText))?.click(); return true;`);
    stage = await waitConfigured();
  }
  if (stage !== 'configure') throw new Error(`未进入配置阶段: ${stage}`);
  // 等模型列表 / 任务列表与登录态都加载完成，避免读到中间态。
  const settled = await evaluate(`
    let sawModels = false;
    for (let i = 0; i < 160; i += 1) {
      const empty = document.querySelector('[data-uploaded-model-empty="true"]');
      const card = document.querySelector('[data-uploaded-model-card="true"]');
      const loginRow = Boolean(document.querySelector('[data-inspector-issue="login"]'));
      sawModels = sawModels || Boolean(card) || Boolean(empty);
      const want = ${token ? 'true' : 'false'};
      if (sawModels && loginRow !== want) break;
      await new Promise(r=>setTimeout(r,250));
    }
    document.documentElement.style.scrollBehavior = 'auto';
    await new Promise(r=>setTimeout(r,400));
      const loginRow = Boolean(document.querySelector('[data-inspector-issue="login"]'));
    return { tokenLen: (localStorage.getItem('astraatmos_token') || '').length, loginRow, card: Boolean(document.querySelector('[data-uploaded-model-card="true"]')) };
  `);
  console.log(`  [debug] 载入后 token=${settled.tokenLen} 未登录提示行=${settled.loginRow} 上传卡=${settled.card}`);
  if (token && settled.loginRow) {
    // 登录态偶尔慢于模型列表：给 AuthContext 的 /auth/me 再留时间；若仍未登录，
    // 整页刷新一次让它重新初始化（不改变任何应用状态，只是重试加载）。
    for (let round = 0; round < 6; round += 1) {
      await evaluate(`
        for (let i = 0; i < 24; i += 1) {
      const loginRow = Boolean(document.querySelector('[data-inspector-issue="login"]'));
          if (!loginRow) break;
          await new Promise(r=>setTimeout(r,250));
        }
        return true;
      `);
      const stillGuest = await evaluate(`
        return Boolean(document.querySelector('[data-inspector-issue="login"]'));
      `);
      if (!stillGuest) break;
      await call('Page.reload', { ignoreCache: true });
      await waitConfigured();
      await new Promise((r) => setTimeout(r, 1200));
    }
  }
};

const shoot = async (name) => {
  const { data } = await call('Page.captureScreenshot', { format: 'png' });
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
  console.log(`\n=== ${tag} · 存在错误 / 待处理（未登录）===`);
  await OPEN(width, height, { token: '' });
  const guest = await evaluate(READ);
  await evaluate(`window.scrollTo(0, Math.round(window.innerHeight / 2)); await new Promise(r=>setTimeout(r,400)); return true;`);
  await shoot(`${tag}-A-blocked`);
  report.viewports[`${tag}-blocked`] = guest;
  console.log(`  胶囊=${guest.capsule?.state} "${guest.capsule?.text}" 待处理=${guest.blockedCount} 分隔线=${guest.dividerCount} 面板=${guest.panelHeight}px`);
  console.log(`  问题: ${guest.issues.map((i) => `${i.mark}${i.text}${i.action ? `[${i.action}]` : ''}`).join(' | ')}`);
  check(`${tag} 未登录时列出来源缺失与登录要求`, guest.blockedCount >= 1 && guest.issues.some((i) => i.mark === '!' && i.key === 'login'), `issues=${guest.issues.map((i) => i.key).join(',')}`);
  check(`${tag} 未登录时胶囊为 guest`, guest.capsule?.state === 'guest', String(guest.capsule?.state));
  check(`${tag} 登录要求带可定位入口`, Boolean(guest.issues.find((i) => i.key === 'login')?.action), String(guest.issues.find((i) => i.key === 'login')?.action));

  console.log(`\n=== ${tag} · 全部就绪（登录 + 有效上传模型 + 唯一实验名称）===`);
  await OPEN(width, height, { token: TOKEN });
  const prepared = await evaluate(`
    const el = document.querySelector('#experiment-name-input');
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, 'inspector_ready_probe');
    el.dispatchEvent(new Event('input', { bubbles: true }));
    await new Promise(r=>setTimeout(r,600));
    return { value: el.value, ariaInvalid: el.getAttribute('aria-invalid') };
  `);
  const ready = await evaluate(READ);
  await evaluate(`window.scrollTo(0, Math.round(window.innerHeight / 2)); await new Promise(r=>setTimeout(r,400)); return true;`);
  await shoot(`${tag}-B-ready`);
  report.viewports[`${tag}-ready`] = { ...ready, prepared };
  console.log(`  实验名="${prepared.value}" aria-invalid=${prepared.ariaInvalid}`);
  console.log(`  胶囊=${ready.capsule?.state} ready=${ready.capsule?.ready} "${ready.capsule?.text}" 待处理=${ready.blockedCount}`);
  console.log(`  结论: ${ready.allClear ? `一行「${ready.allClearText}」` : '仍有问题'}`);
  console.log(`  当前模型: ${ready.currentModel} / ${ready.modelSource}`);
  console.log(`  分隔线=${ready.dividerCount} 区块=${ready.blockCount} 间距=[${ready.blockGaps.join(',')}] 面板=${ready.panelHeight}px 溢出X=${ready.overflowX} 裁切=${JSON.stringify(ready.clips)}`);
  console.log(`  层级: 标签 ${ready.typography.labelSize}/${ready.typography.labelColor} · 值 ${ready.typography.valueSize}/${ready.typography.valueColor} · 说明 ${ready.typography.subSize}/${ready.typography.subColor}`);
  check(`${tag} 全部就绪时胶囊为 ready 且文案正确`, ready.capsule?.state === 'ready' && ready.capsule?.ready === 'true', `${ready.capsule?.state} "${ready.capsule?.text}"`);
  check(`${tag} 全部就绪时用一行结论而不是逐条已通过`, ready.allClear === true && ready.blockedCount === 0, `allClear=${ready.allClear} issues=${ready.blockedCount} text="${ready.allClearText}"`);
  check(`${tag} 检查器保留当前模型身份`, Boolean(ready.currentModel), String(ready.currentModel));
  check(`${tag} 常态摘要与折叠详情已移除`, ready.removedNodes.length === 0, `removed=${ready.removedNodes.join(',') || 'none'}`);
  check(`${tag} 检查器无内部滚动且无裁切`, ready.panelOverflow <= 0 && ready.clips.length === 0 && ready.overflowX <= 0, `overflow=${ready.panelOverflow} clips=${ready.clips.length} overflowX=${ready.overflowX}`);
  check(`${tag} 检查器宽度仍为 284px`, ready.panelWidth === 284, String(ready.panelWidth));
}

writeFileSync(join(OUT_DIR, 'inspector-states.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8');
const failed = results.filter((item) => !item.ok);
console.log(`\n===== 检查器状态检查：${results.length - failed.length}/${results.length} 项通过 =====`);
failed.forEach((item) => console.log(`失败: ${item.name} — ${item.detail}`));
console.log(`报告: ${join(OUT_DIR, 'inspector-states.json')}`);
if (failed.length) process.exitCode = 1;

await send('Target.closeTarget', { targetId });
process.exit(process.exitCode || 0);
