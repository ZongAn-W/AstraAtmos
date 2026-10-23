// Run with playwright-cli run-code --filename scripts/audit/experiment-matrix/browser-check.js.
// All API requests and writes stay in this browser fixture, never in the database.
async (page) => {
  const checks = [];
  const errors = [];
  const consoleErrors = [];
  const writes = [];
  const userId = 99001;
  const tags = Array.from({ length: 40 }, (_, i) => ({ id: 9000 + i, name: `验证标签 ${String(i + 1).padStart(2, '0')}`, user_id: userId }));
  const tasks = Array.from({ length: 36 }, (_, i) => ({
    id: 99000 + i, user_id: userId, custom_model_name: `矩阵验证实验 ${String(i + 1).padStart(2, '0')}`,
    status: 'completed', progress: 100, current_epoch: 10, total_epochs: 10,
    dataset_id: 'openmars_mcd', model_source: 'official',
    start_time: new Date(Date.UTC(2026, 9, 8, 0, i)).toISOString(),
    end_time: new Date(Date.UTC(2026, 9, 8, 1, i)).toISOString(),
    hyperparameters: JSON.stringify({ model_architecture: 'convlstm', model_source: 'official', training_dataset: 'openmars_mcd', window: 3, horizon: 3, epochs: 10, batch_size: 16, learning_rate: 0.001, seed: i, selected_channels: ['u', 'v'] }),
    metrics: JSON.stringify({ rmse: 0.12 + i * 0.01, mae: 0.1, r2: 0.9 }),
    tags: [tags[i % 4], tags[(i % 4) + 4], tags[(i % 4) + 8]],
  }));
  let rejectSave = false;
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error' && !message.text().includes('403 (Forbidden)')) consoleErrors.push(message.text()); });
  await page.unroute('**/api/**');
  await page.route('**/api/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    let body = {};
    let status = 200;
    if (path === '/api/auth/me') body = { id: userId, username: 'matrix-audit', email: 'matrix-audit@example.invalid', role: 'user', is_admin: false };
    else if (path === '/api/training/tasks') body = tasks;
    else if (path === '/api/training/tags' && request.method() === 'GET') body = tags;
    else if (/\/training\/tasks\/\d+\/tags$/.test(path) && request.method() === 'PUT') {
      if (rejectSave) { status = 403; body = { detail: '隔离夹具：无权保存标签' }; }
      else {
        const id = Number(path.split('/').at(-2));
        const ids = request.postDataJSON().tag_ids;
        const task = tasks.find(item => item.id === id);
        task.tags = tags.filter(tag => ids.includes(tag.id));
        writes.push({ id, ids });
        body = task;
      }
    } else if (path === '/api/training/tags' && request.method() === 'POST') {
      body = { id: 9000 + tags.length, name: request.postDataJSON().name, user_id: userId };
      tags.push(body);
    } else if (request.method() !== 'GET') throw new Error(`Unexpected mutation: ${request.method()} ${path}`);
    else if (path.endsWith('/logs')) body = { lines: [] };
    else if (path === '/api/notifications') body = { items: [], unread_count: 0 };
    else if (path === '/api/datasets') body = { datasets: [{ dataset_id: 'openmars_mcd', id: 'openmars_mcd', planet: 'mars', display_name: 'OpenMARS + MCD', status: 'available', capabilities: { training: true } }], default_earth_dataset_id: 'earth_merra2_3hourly_v1' };
    else if (['/api/training/scripts', '/api/user-models', '/api/training/weights'].includes(path)) body = [];
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  });
  await page.routeWebSocket('**/ws/**', () => {});
  await page.addInitScript(({ userId }) => {
    localStorage.setItem('astraatmos_token', 'isolated-browser-fixture');
    localStorage.setItem(`astraatmos_experiment_matrix_columns:${userId}`, JSON.stringify({ columns: ['model_name', 'tags', 'architecture', 'dataset', 'status', 'progress', 'start_time', 'training:window', 'training:horizon', 'training:epochs', 'training:batch_size', 'training:learning_rate', 'training:seed', 'metric:rmse', 'metric:mae', 'metric:r2'] }));
  }, { userId });
  const assert = (condition, message) => { if (!condition) throw new Error(message); checks.push(message); };
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('http://127.0.0.1:5173/#/training');
  await page.reload();
  await page.getByRole('button', { name: '实验矩阵', exact: true }).click();
  await page.locator('.experiment-matrix[data-matrix-state="ready"]').waitFor();
  assert(await page.locator('.experiment-matrix-table tbody tr').count() === 36, '36 fixture rows loaded');
  assert(await page.locator('.experiment-matrix-table thead th').count() >= 18, 'multiple property columns loaded');
  const scroll = page.locator('.experiment-matrix-scroll');
  const editor = page.getByRole('dialog', { name: '编辑实验标签' });
  const geometry = () => scroll.evaluate(element => {
    const rect = element.getBoundingClientRect();
    const header = element.querySelector('th.experiment-matrix-col-model-name').getBoundingClientRect();
    const name = element.querySelector('td.experiment-matrix-col-model-name').getBoundingClientRect();
    return { top: rect.top, bottom: rect.bottom, width: rect.width, height: rect.height, clientHeight: element.clientHeight, scrollHeight: element.scrollHeight, scrollTop: element.scrollTop, scrollLeft: element.scrollLeft, headerTop: header.top, nameLeft: name.left, pageY: window.scrollY };
  });
  const checkEditor = async label => {
    await editor.waitFor();
    const result = await editor.evaluate(element => {
      const rect = element.getBoundingClientRect();
      const hit = node => {
        const r = node.getBoundingClientRect();
        return node.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2));
      };
      return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom, width: innerWidth, height: innerHeight, search: hit(element.querySelector('input[type="search"]')), save: hit(element.querySelector('.primary')), cancel: hit(element.querySelector('.experiment-matrix-popover-actions button')), inTable: Boolean(element.closest('table')), focused: document.activeElement === element.querySelector('input[type="search"]') };
    });
    assert(result.left >= 11 && result.top >= 11 && result.right <= result.width - 11 && result.bottom <= result.height - 11, `${label}: editor inside viewport`);
    assert(result.search && result.save && result.cancel, `${label}: search/save/cancel hit tests pass`);
    assert(!result.inTable, `${label}: editor portaled outside table`);
    return result;
  };
  const openEdgeEditor = async index => {
    await scroll.evaluate((element, index) => {
      const button = element.querySelectorAll('button[aria-label="编辑标签"]')[index];
      const rect = element.getBoundingClientRect();
      element.scrollTop += button.getBoundingClientRect().top - (rect.top + element.clientHeight - 45);
    }, index);
    const button = page.locator('.experiment-matrix-table tbody tr').nth(index).getByRole('button', { name: '编辑标签', exact: true });
    await button.click({ timeout: 5000 });
    return button;
  };
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1024, height: 768 }, { width: 390, height: 844 }, { width: 844, height: 390 }]) {
    await page.setViewportSize(viewport);
    await scroll.evaluate(element => { element.scrollTop = 0; element.scrollLeft = 0; });
    const initial = await geometry();
    const label = `${viewport.width}x${viewport.height}`;
    assert(initial.bottom <= viewport.height && initial.height > 75 && initial.scrollHeight > initial.clientHeight, `${label}: bounded scroll area and scrollbar reachable`);
    await page.mouse.move( Math.min(viewport.width - 50, initial.width / 2 + 30), initial.top + 60 );
    await page.mouse.wheel(400, 360);
    await page.waitForTimeout(250);
    const moved = await geometry();
    assert(moved.scrollTop > 0 && moved.scrollLeft > 0 && moved.pageY === initial.pageY, `${label}: wheel scrolls both matrix axes without moving page`);
    assert(Math.abs(moved.headerTop - moved.top) < 2 && Math.abs(moved.nameLeft - initial.nameLeft) < 2, `${label}: header and experiment name stay visible after scrolling`);
    await scroll.evaluate(element => { element.scrollLeft = 0; });
    const anchor = await openEdgeEditor(12);
    const opened = await checkEditor(label);
    assert(opened.focused, `${label}: search receives focus`);
    assert(await editor.getByRole('checkbox').count() === tags.length, `${label}: all 40 tag options rendered`);
    const beforeList = await geometry();
    const options = editor.locator('.training-tag-options');
    const optionRect = await options.boundingBox();
    await page.mouse.move(optionRect.x + 30, optionRect.y + 35);
    await page.mouse.wheel(0, 600);
    await page.waitForTimeout(250);
    assert(await options.evaluate(element => element.scrollTop > 0), `${label}: tag options scroll independently`);
    const afterList = await geometry();
    assert(beforeList.scrollTop === afterList.scrollTop && beforeList.pageY === afterList.pageY, `${label}: option scroll preserves matrix/page position`);
    await editor.getByRole('searchbox', { name: '搜索标签' }).fill('验证标签 40');
    await editor.getByRole('checkbox', { name: '验证标签 40' }).check();
    await editor.getByRole('button', { name: '取消', exact: true }).click();
    assert(writes.length === 0, `${label}: cancel sends no mutation`);
    await anchor.click({ timeout: 5000 });
    await editor.getByRole('searchbox', { name: '搜索标签' }).fill('验证标签 40');
    assert(!await editor.getByRole('checkbox', { name: '验证标签 40' }).isChecked(), `${label}: cancel resets draft`);
    await editor.getByRole('searchbox', { name: '搜索标签' }).fill('');
    await page.waitForTimeout(100);
    await checkEditor(`${label} after content resize`);
    if (viewport.width === 1440 || viewport.width === 390) await page.screenshot({ path: `output/playwright/final-matrix-${viewport.width === 1440 ? 'wide' : 'narrow'}-edge.png` });
    await editor.getByRole('button', { name: '取消', exact: true }).click();
    await scroll.evaluate(element => { element.scrollLeft = 700; });
    const scrollState = await geometry();
    assert(scrollState.scrollLeft > 0 && Math.abs(scrollState.headerTop - scrollState.top) < 2, `${label}: horizontal scroll leaves sticky column headings readable`);
    if (viewport.width === 1440 || viewport.width === 390) await page.screenshot({ path: `output/playwright/final-matrix-${viewport.width === 1440 ? 'wide' : 'narrow'}-scroll.png` });
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await scroll.evaluate(element => { element.scrollLeft = 900; });
  const anchor = await openEdgeEditor(12);
  await editor.getByRole('searchbox', { name: '搜索标签' }).fill('验证标签 40');
  await editor.getByRole('checkbox', { name: '验证标签 40' }).check();
  await editor.getByRole('button', { name: '保存', exact: true }).click();
  await editor.waitFor({ state: 'hidden' });
  assert(writes.length === 1 && writes[0].id === 99023 && writes[0].ids.includes(9039), 'save uses the correct task ID and tag IDs');
  assert(await page.locator('.experiment-matrix-table tbody tr').nth(12).getByText('验证标签 40', { exact: true }).count() === 1, 'saved tags refreshed in the same matrix row');
  await anchor.click({ timeout: 5000 });
  await editor.getByRole('searchbox', { name: '搜索标签' }).fill('验证标签 40');
  assert(await editor.getByRole('checkbox', { name: '验证标签 40' }).isChecked(), 'saved tag is selected on reopening');
  rejectSave = true;
  await editor.getByRole('checkbox', { name: '验证标签 40' }).uncheck();
  await editor.getByRole('button', { name: '保存', exact: true }).click();
  await editor.getByRole('alert').waitFor();
  await checkEditor('permission error');
  assert(writes.length === 1, 'permission error keeps saved tags unchanged');
  await editor.getByRole('searchbox', { name: '搜索标签' }).press('Escape');
  assert(!await editor.isVisible(), 'Escape dismisses editor');
  assert(await anchor.evaluate(element => document.activeElement === element), 'Escape returns focus to the anchor');
  rejectSave = false;
  await anchor.click({ timeout: 5000 });
  await page.getByRole('button', { name: '实验配置', exact: true }).click();
  assert(!await editor.isVisible(), 'leaving matrix closes portaled editor');
  await page.getByRole('button', { name: '实验矩阵', exact: true }).click();
  assert(!await editor.isVisible(), 'returning to matrix does not reopen cancelled editor');
  await page.getByRole('button', { name: '显示属性', exact: true }).click();
  await page.setViewportSize({ width: 1024, height: 768 });
  await scroll.evaluate(element => { element.scrollLeft = 0; });
  await openEdgeEditor(12);
  await checkEditor('1024 with property panel');
  await editor.getByRole('button', { name: '取消', exact: true }).click();
  await page.getByRole('button', { name: '关闭显示属性' }).click();
  await scroll.evaluate(element => { element.scrollLeft = 0; });
  await openEdgeEditor(10);
  await editor.getByRole('searchbox', { name: '搜索标签' }).fill('新增验证标签');
  await editor.getByRole('button', { name: '新建“新增验证标签”', exact: true }).click();
  await editor.getByRole('checkbox', { name: '新增验证标签' }).waitFor();
  assert(await editor.getByRole('checkbox', { name: '新增验证标签' }).isChecked(), 'newly created tag joins draft selection');
  await checkEditor('after creating a tag');
  await page.mouse.move(800, 500);
  await page.mouse.wheel(0, -80);
  await page.waitForTimeout(200);
  await checkEditor('while matrix scrolls under the open editor');
  const oldAnchor = page.locator('.experiment-matrix-table tbody tr').nth(10).getByRole('button', { name: '编辑标签', exact: true });
  await editor.getByRole('searchbox', { name: '搜索标签' }).press('Escape');
  await oldAnchor.click();
  await editor.getByRole('searchbox', { name: '搜索标签' }).fill('验证标签 40');
  await page.locator('.experiment-matrix-table tbody tr').nth(8).getByRole('button', { name: '编辑标签', exact: true }).click();
  assert(await editor.count() === 1, 'opening another row leaves only one tag editor');
  await editor.getByRole('button', { name: '取消', exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: '显示属性', exact: true }).click();
  const narrowProperties = await geometry();
  assert(narrowProperties.height > 100 && narrowProperties.bottom < 844, 'narrow property panel preserves usable matrix scroll area');
  await page.getByRole('button', { name: '关闭显示属性' }).click();
  assert(errors.length === 0, 'no browser runtime errors');
  assert(consoleErrors.length === 0, 'no unexpected browser console errors');
  return { passed: checks.length, rows: tasks.length, columns: await page.locator('thead th').count(), tags: tags.length, checks, errors, consoleErrors };
}
