// Isolated matrix action checks. All API requests (including deletes/tests) are intercepted.
// Run with playwright-cli run-code --filename scripts/audit/experiment-matrix/actions-browser-check.js.
async (page) => {
  const checks = [];
  const errors = [];
  const writes = [];
  const contexts = [];
  const userId = 99101;
  const makeTask = (id, name, status = 'completed', dataset = 'openmars_mcd', available = true) => ({
    id, user_id: userId, custom_model_name: name, status, dataset_id: dataset,
    model_available: available, model_source: 'official', progress: status === 'completed' ? 100 : 0,
    current_epoch: 2, total_epochs: 2, start_time: new Date(Date.UTC(2026, 9, 12, 0, 0, 20 - id % 20)).toISOString(),
    hyperparameters: JSON.stringify({ model_architecture: dataset.startsWith('earth_') ? 'dlinear' : 'convlstm',
      training_dataset: dataset, model_source: 'official', window: 3, horizon: 3, epochs: 2, batch_size: 1, learning_rate: .001 }),
    metrics: JSON.stringify({ rmse: .2, mae: .1, r2: .9 }), tags: [],
  });
  let tasks = [makeTask(99110, '可用火星模型'), makeTask(99111, '可用地球模型', 'completed', 'earth_merra2_3hourly_v1'),
    makeTask(99112, '排队任务', 'queued'), makeTask(99113, '运行任务', 'running'),
    makeTask(99114, '失败任务', 'failed', 'openmars_mcd', false),
    makeTask(99115, '无权重任务', 'completed', 'openmars_mcd', false),
    makeTask(99116, '旧日频任务', 'completed', 'earth_merra2_daily_v2')];
  let denyDelete = true;
  let blockDelete = false;
  let releaseDelete;
  let denyEarth = true;
  page.on('pageerror', error => errors.push(error.message));
  await page.unroute('**/api/**');
  await page.route('**/api/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    let body = {};
    let status = 200;
    if (path === '/api/auth/me') body = { id: userId, username: 'matrix-actions', email: 'matrix-actions@example.invalid', role: 'user', is_admin: false };
    else if (path === '/api/training/tasks') body = tasks;
    else if (path === '/api/training/tags' || ['/api/training/scripts', '/api/user-models', '/api/training/weights'].includes(path)) body = [];
    else if (path === '/api/notifications') body = { items: [], unread_count: 0 };
    else if (path === '/api/datasets') body = { datasets: ['openmars_mcd', 'earth_merra2_3hourly_v1'].map(id => ({ dataset_id: id, id,
      planet: id.startsWith('earth_') ? 'earth' : 'mars', display_name: id, status: 'available', capabilities: { training: true } })), default_earth_dataset_id: 'earth_merra2_3hourly_v1' };
    else if (path.endsWith('/logs')) body = { lines: [] };
    else if (path.endsWith('/diagnostics/context')) {
      contexts.push(Number(url.searchParams.get('training_task_id')));
      body = { available: !denyEarth, task_id: 99111, reason: denyEarth ? { message: '隔离验证：数据身份不匹配' } : null };
    } else if (path === '/api/earth/predict/diagnostics' && request.method() === 'POST') {
      writes.push({ action: 'earth-diagnostics', id: request.postDataJSON().training_task_id });
      status = 409;
      body = { detail: { code: 'fixture_no_compute', message: '隔离验证不执行真实诊断计算' } };
    } else if (request.method() === 'DELETE' && /\/training\/tasks\/\d+$/.test(path)) {
      const id = Number(path.split('/').at(-1));
      writes.push({ action: 'delete', id });
      if (blockDelete) await new Promise(resolve => { releaseDelete = resolve; });
      if (denyDelete) { status = 403; body = { detail: '隔离验证：无权删除此实验' }; }
      else { tasks = tasks.filter(task => task.id !== id); body = { status: 'success' }; }
    } else if (request.method() === 'POST' && path.endsWith('/action')) {
      writes.push({ action: url.searchParams.get('action'), id: Number(path.split('/').at(-2)) });
      body = { status: 'success', data: { metrics: { rmse: .2, mae: .1, r2: .9, mse: .04, mape: .1, smape: .1 }, scatter: { x: [1, 2], y: [1.1, 2.1] } } };
    } else if (request.method() !== 'GET') throw new Error(`Unexpected mutation: ${request.method()} ${path}`);
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  });
  await page.routeWebSocket('**/ws/**', () => {});
  await page.addInitScript(({ userId }) => {
    localStorage.setItem('astraatmos_token', 'isolated-actions-fixture');
    localStorage.setItem(`astraatmos_experiment_matrix_columns:${userId}`, JSON.stringify({ columns: ['model_name', 'tags', 'architecture', 'dataset', 'status', 'progress', 'start_time'] }));
  }, { userId });
  const assert = (condition, message) => { if (!condition) throw new Error(message); checks.push(message); };
  const row = name => page.locator('.experiment-matrix-table tbody tr').filter({ hasText: name });
  const open = async name => {
    const button = row(name).getByRole('button', { name: '更多操作', exact: true });
    await button.click();
    await page.getByRole('menu').waitFor();
    await page.waitForFunction(() => {
      const paper = document.querySelector('.experiment-matrix-action-menu');
      return paper && getComputedStyle(paper).opacity === '1';
    });
    return button;
  };
  const menuItem = label => page.getByRole('menuitem', { name: new RegExp(`^${label}`) });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('http://127.0.0.1:5173/#/training');
  await page.reload();
  await page.getByRole('button', { name: '实验矩阵', exact: true }).click();
  await row('可用火星模型').waitFor();
  assert(contexts.length === 0, 'matrix does not preflight every Earth task before a menu opens');
  for (const name of ['排队任务', '运行任务', '失败任务', '无权重任务', '旧日频任务']) {
    await open(name);
    assert(await menuItem('用于预测').getAttribute('aria-disabled') === 'true', `${name}: prediction is disabled`);
    assert(await menuItem('模型测试').getAttribute('aria-disabled') === 'true', `${name}: test is disabled`);
    if (['排队任务', '运行任务'].includes(name)) assert(await menuItem('删除实验记录').getAttribute('aria-disabled') === 'true', `${name}: active task deletion is disabled`);
    else assert(await menuItem('删除实验记录').getAttribute('aria-disabled') !== 'true', `${name}: terminal task can be deleted`);
    await page.keyboard.press('Escape');
  }
  const anchor = await open('可用火星模型');
  assert(await menuItem('用于预测').getAttribute('aria-disabled') !== 'true', 'valid Mars prediction is enabled');
  await page.keyboard.press('Escape');
  assert(await anchor.evaluate(element => document.activeElement === element), 'Escape returns focus to the more-actions button');
  await open('可用火星模型');
  await menuItem('模型测试').click();
  await page.getByRole('dialog', { name: '模型性能评估', exact: true }).waitFor();
  assert(writes.some(write => write.action === 'test' && write.id === 99110), 'Mars model test uses the clicked task');
  await page.getByRole('button', { name: '关闭测试结果', exact: true }).click();
  await open('可用地球模型');
  await page.getByRole('menuitem', { name: /模型测试.*数据身份不匹配/ }).waitFor();
  assert(await menuItem('模型测试').getAttribute('aria-disabled') === 'true', 'failed Earth artifact check disables testing');
  denyEarth = false;
  await menuItem('重新核验').click();
  await page.waitForFunction(() => [...document.querySelectorAll('[role="menuitem"]')].some(item => item.textContent === '模型测试' && item.getAttribute('aria-disabled') !== 'true'));
  assert(contexts.length === 2 && contexts.every(id => id === 99111), 'Earth check and retry use the clicked task');
  await menuItem('模型测试').click();
  await page.getByRole('heading', { name: '地球训练后诊断', exact: true }).waitFor();
  assert(!writes.some(write => write.action === 'test' && write.id === 99111), 'Earth testing opens Earth diagnostics without calling the Mars test API');
  await page.getByRole('button', { name: '关闭测试结果', exact: true }).click();
  await open('失败任务');
  await menuItem('删除实验记录').click();
  const dialog = page.getByRole('dialog', { name: '确认删除', exact: true });
  await dialog.waitFor();
  assert((await dialog.textContent()).includes('#99114') && (await dialog.textContent()).includes('模型权重') && (await dialog.textContent()).includes('预测分析缓存'), 'delete confirmation names the task and destructive scope');
  await dialog.getByRole('button', { name: '取消', exact: true }).click();
  assert(!writes.some(write => write.action === 'delete'), 'cancelling deletion sends no request');
  await open('失败任务');
  await menuItem('删除实验记录').click();
  await dialog.getByRole('button', { name: '确认删除', exact: true }).click();
  await dialog.getByRole('alert').waitFor();
  assert((await dialog.getByRole('alert').textContent()).includes('无权删除'), 'delete errors remain visible for retry');
  assert(await row('失败任务').count() === 1, 'failed deletion retains the experiment row');
  denyDelete = false;
  blockDelete = true;
  await dialog.getByRole('button', { name: '确认删除', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('[role="dialog"][aria-busy="true"]'));
  assert(await dialog.getByRole('button', { name: '正在删除...', exact: true }).isDisabled() && await dialog.getByRole('button', { name: '取消', exact: true }).isDisabled(), 'busy deletion disables repeat submit and cancellation');
  releaseDelete();
  await dialog.waitFor({ state: 'hidden' });
  await row('失败任务').waitFor({ state: 'hidden' });
  assert(writes.filter(write => write.action === 'delete').length === 2, 'delete retry sends exactly two isolated requests');
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    await page.evaluate(() => window.scrollTo(0, 0));
    await open('可用火星模型');
    const bounds = await page.getByRole('menu').evaluate(element => {
      const rect = element.getBoundingClientRect();
      return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom, width: innerWidth, height: innerHeight, inTable: Boolean(element.closest('table')) };
    });
    assert(bounds.left >= 11 && bounds.top >= 11 && bounds.right <= bounds.width - 11 && bounds.bottom <= bounds.height - 11 && !bounds.inTable, `${viewport.width}px: action menu is portaled and inside the viewport`);
    await page.screenshot({ path: `output/playwright/matrix-actions-${viewport.width}.png` });
    await page.keyboard.press('Escape');
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  for (const [name, expectedMode] of [['可用火星模型', 'trained'], ['可用地球模型', 'earth']]) {
    await open(name);
    await menuItem('用于预测').click();
    await page.waitForURL(new RegExp(`mode=${expectedMode}`));
    assert(page.url().includes('from=training'), `${name}: prediction navigation carries training origin and planet mode`);
    await page.goto('http://127.0.0.1:5173/#/training');
    await page.reload();
    await page.getByRole('button', { name: '实验矩阵', exact: true }).click();
    await row(name).waitFor();
  }
  assert(errors.length === 0, 'no browser runtime errors');
  return { passed: checks.length, checks, writes, contexts, errors };
}
