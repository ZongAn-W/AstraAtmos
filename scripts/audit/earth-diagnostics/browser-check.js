// Isolated browser fixtures only; no real API writes, model execution or training.
// Run through playwright-cli run-code --filename scripts/audit/earth-diagnostics/browser-check.js.
async (page) => {
  const checks = [], errors = [], requests = [];
  const userId = 99031;
  const datasetId = 'earth_merra2_3hourly_v1';
  const fingerprint = 'f'.repeat(64);
  const tasks = [99031, 99032].map((id, index) => ({
    id, user_id: userId, custom_model_name: `诊断夹具 ${index + 1}`, status: 'completed',
    model_available: true, is_earth_task: true, model_source: 'official',
    dataset_id: datasetId, dataset_version: 'v1', dataset_fingerprint: fingerprint,
    dataset_identity_status: 'verified', dataset_snapshot: {},
    output_model_path: `synthetic-${id}.pth`, progress: 100, current_epoch: 1, total_epochs: 1,
    start_time: '2026-10-08T00:00:00Z', end_time: '2026-10-08T00:01:00Z',
    hyperparameters: JSON.stringify({ training_dataset: datasetId, model_architecture: 'dlinear',
      model_source: 'official', window: 2, horizon: 1, selected_channels: ['U10M'], epochs: 1 }),
    metrics: JSON.stringify({ splits: { test: { overall: { rmse: 2, mae: 1 } } } }), tags: [],
  }));
  const context = (id) => ({
    planet: 'earth', task_id: id, dataset_id: datasetId, dataset_version: 'v1',
    dataset_fingerprint: fingerprint, target: 'TO3', target_unit: 'DU', window: 2, horizon: 1,
    frequency_hours: 3, step_unit: 'hour', step: 3, time_zone: 'UTC',
    input_channel_order: ['TO3', 'U10M'], input_units: ['DU', 'm s-1'],
    grid: { shape: [240, 480], latitude: [-89.625, 89.625], longitude: [-179.625, 179.625] },
    origins: { start: '2021-07-01T04:30:00Z', end: '2021-07-01T04:30:00Z', count: 1,
      dates: ['2021-07-01T04:30:00Z'], timestamps: ['2021-07-01T04:30:00Z'] },
    model: { model_source: 'official', model_architecture: 'dlinear' },
    metrics: { splits: { test: { overall: { rmse: 2, mae: 1 } } } }, run: {}, warnings: [],
  });
  const scope = {
    split: 'test', window_count: 4, available_window_count: 20, valid_points: 460800,
    grid_shape: [240, 480],
    time_range: { input_start: '2021-07-01T01:30:00Z', target_start: '2021-07-01T07:30:00Z', target_end: '2021-07-02T07:30:00Z' },
    spatial_coverage: { coverage: 'global', all_grid_cells: true,
      latitude_range: [-89.625, 89.625], longitude_range: [-179.625, 179.625] },
    sampling: { method: 'seeded_uniform_windows_without_replacement', seed: 42 },
    windows: [{ test_window_index: 0, forecast_origin: '2021-07-01T04:30:00Z',
      input_start: '2021-07-01T01:30:00Z', input_end: '2021-07-01T04:30:00Z',
      target_start: '2021-07-01T07:30:00Z', target_end: '2021-07-01T07:30:00Z' }],
  };
  const result = (id, parameters, hit = false) => ({
    ...context(id), status: 'completed', unit: 'DU', scope,
    full_test_metrics: { scope: 'full_test', source: 'verified_checkpoint_test_metrics',
      metrics: { overall: { mse: 4, rmse: 2, mae: 1, r2: .8, mape: 1, smape: 1 }, window_count: 20 },
      test_range: { date_start: '2021-07-01T01:30:00Z', date_end: '2021-07-31T22:30:00Z', window_count: 20 },
      split_policy: 'published_manifest_splits' },
    sample_metrics: { overall: { mse: 1, rmse: 1, mae: 1, r2: .9, mape: 1, smape: 1 } },
    scatter: { reference: [240, 250, 260, 270], prediction: [241, 251, 261, 271],
      point_count: 4, unit: 'DU', scope: 'sampled_points_from_selected_test_windows' },
    histogram: { edges: [.5, 1, 1.5], counts: [0, 460800], valid_points: 460800,
      summary: { mean: 1, std: 0, min: 1, max: 1, count: 460800 }, unit: 'DU', residual_definition: 'prediction-reference' },
    pfi: { status: 'completed', baseline_rmse: 1, seed: 42, window_count: 4, scope,
      items: [
        { channel: 'TO3', importance_mean: 2, importance_std: .1, repeats: [{ rmse: 3, delta_rmse: 2 }] },
        { channel: 'U10M', importance_mean: -.1, importance_std: .01, repeats: [{ rmse: .9, delta_rmse: -.1 }] },
      ] },
    cache: { hit, key: 'synthetic', algorithm_version: 'earth_test_diagnostics_v1' },
    parameters,
  });
  let fail = false, hold = false, releaseHeld = null, responseCount = 0;
  page.on('pageerror', error => errors.push(error.message));
  await page.unroute('**/api/**');
  await page.route('**/api/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    let body = {}, status = 200;
    if (path === '/api/auth/me') body = { id: userId, email: 'diagnostics@example.invalid', username: 'diagnostic-fixture', role: 'user' };
    else if (path === '/api/training/tasks') body = tasks;
    else if (path === '/api/earth/predict/context') body = context(Number(url.searchParams.get('training_task_id')));
    else if (path === '/api/earth/predict/diagnostics/context') body = { available: true, target_unit: 'DU', task_id: Number(url.searchParams.get('training_task_id')) };
    else if (path === '/api/earth/predict/diagnostics') {
      const parameters = request.postDataJSON();
      requests.push(parameters);
      if (hold) await new Promise(resolve => { releaseHeld = resolve; });
      if (fail) { status = 409; body = { detail: { code: 'invalid_earth_training_artifact', message: '合成权重已变化，请重试' } }; }
      else body = result(parameters.training_task_id, parameters, responseCount++ > 0);
    } else if (path === '/api/datasets') body = { default_earth_dataset_id: datasetId,
      datasets: [{ id: datasetId, dataset_id: datasetId, planet: 'earth', display_name: 'Earth synthetic', status: 'available', capabilities: { training: true } }] };
    else if (path.endsWith('/logs')) body = { lines: [] };
    else if (path === '/api/notifications') body = { items: [], unread_count: 0 };
    else if (['/api/training/scripts', '/api/training/tags', '/api/user-models', '/api/training/weights'].includes(path)) body = [];
    else if (request.method() !== 'GET') throw new Error(`Unexpected API mutation: ${path}`);
    try { await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) }); }
    catch (error) { if (!request.failure()) throw error; }
  });
  await page.routeWebSocket('**/ws/**', () => {});
  await page.addInitScript(() => {
    localStorage.setItem('astraatmos_token', 'isolated-earth-diagnostics-fixture');
    localStorage.setItem('astraatmos_settings', JSON.stringify({ language: 'zh', theme: 'dark' }));
  });
  const assert = (condition, name) => { if (!condition) throw new Error(name); checks.push(name); };
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('http://127.0.0.1:5173/#/predict?mode=earth');
  await page.reload();
  const modelPicker = page.locator('button[aria-haspopup="listbox"]');
  await modelPicker.waitFor();
  await modelPicker.click();
  await page.getByRole('option', { name: /诊断夹具 1/ }).click();
  const panel = page.locator('[data-earth-diagnostics="true"]');
  await panel.getByText('任务可诊断。', { exact: false }).waitFor();
  assert(await panel.getByRole('button', { name: '计算测试集诊断', exact: true }).isEnabled(), 'ready state allows explicit diagnostics');
  assert(requests.length === 0, 'prediction page does not start expensive inference on mount');
  await panel.getByRole('button', { name: '计算测试集诊断', exact: true }).click();
  await panel.getByRole('heading', { name: '完整测试集指标', exact: true }).waitFor();
  assert(requests.length === 1 && requests[0].sample_windows === 4 && requests[0].pfi_repeats === 3
    && requests[0].seed === 42 && requests[0].scatter_points === 5000, 'both endpoints share bounded defaults');
  assert(await panel.locator('.js-plotly-plot').count() === 3, 'scatter histogram and PFI render');
  assert(await panel.getByText('2.000 ± 0.100', { exact: true }).count() === 1, 'PFI mean and std visible');
  assert(await panel.getByText('-0.100 ± 0.010', { exact: true }).count() === 1, 'negative PFI retained');
  assert(await panel.getByText('每次 RMSE 增量（DU）', { exact: true }).count() === 1, 'every-repeat increments visible');
  await page.setViewportSize({ width: 390, height: 844 });
  await panel.scrollIntoViewIfNeeded();
  await panel.getByRole('heading', { name: '完整测试集指标', exact: true }).scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const geometry = await panel.evaluate(element => ({
    width: element.getBoundingClientRect().width,
    client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth,
    columns: getComputedStyle(element.querySelector('.earth-diagnostic-chart-grid')).gridTemplateColumns,
  }));
  assert(geometry.scroll <= geometry.client + 2 && geometry.width <= 390, '390px layout has no page overflow');
  assert(!geometry.columns.includes(' '), 'narrow diagnostic charts stack in one column');
  await page.screenshot({ path: 'output/playwright/earth-diagnostics-390.png', fullPage: true });

  fail = true;
  await panel.getByRole('button', { name: '重新计算 / 复用缓存', exact: true }).click();
  await panel.getByRole('alert').waitFor();
  assert((await panel.getByRole('alert').innerText()).includes('合成权重已变化'), 'structured error message is readable');
  assert(!(await panel.innerText()).includes('[object Object]'), 'errors never render object Object');
  fail = false;
  await panel.getByRole('button', { name: '重新计算 / 复用缓存', exact: true }).click();
  await panel.getByRole('heading', { name: '完整测试集指标', exact: true }).waitFor();
  assert((await panel.innerText()).includes('缓存命中'), 'retry can reuse verified diagnostic cache');
  await panel.getByRole('spinbutton', { name: '诊断窗口数', exact: true }).fill('2');
  await panel.getByText('任务可诊断。', { exact: false }).waitFor();
  assert(await panel.getByRole('heading', { name: '完整测试集指标', exact: true }).count() === 0, 'sampling change clears previous result');
  hold = true;
  await panel.getByRole('button', { name: '计算测试集诊断', exact: true }).click();
  await panel.getByRole('button', { name: '计算诊断（含 PFI）…', exact: true }).waitFor();
  assert(await panel.getByRole('button', { name: '计算诊断（含 PFI）…', exact: true }).isDisabled(), 'loading disables duplicate compute');
  hold = false;
  releaseHeld();
  await panel.getByRole('heading', { name: '完整测试集指标', exact: true }).waitFor();
  assert(requests.at(-1).sample_windows === 2, 'new sampling parameters reach backend');

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('http://127.0.0.1:5173/#/training');
  await page.getByRole('button', { name: '训练监视', exact: true }).click();
  await page.getByRole('button', { name: /诊断夹具 1 已完成/ }).click();
  await page.getByRole('button', { name: '模型测试', exact: true }).waitFor();
  assert(await page.getByRole('button', { name: '模型测试', exact: true }).isEnabled(), 'Earth result test button verifies artifact availability');
  await page.getByRole('button', { name: '模型测试', exact: true }).click();
  const modal = page.getByRole('dialog', { name: '模型测试结果', exact: false });
  await page.getByRole('dialog').locator('[data-earth-diagnostics="true"]').getByRole('heading', { name: '完整测试集指标', exact: true }).waitFor();
  assert(requests.at(-1).sample_windows === 4, 'Earth model test automatically requests same defaults');
  assert(await page.getByRole('dialog').getByText('残差分布（DU）', { exact: true }).count() === 1, 'Earth test modal renders residual diagnostics');
  await page.getByRole('button', { name: '关闭测试结果', exact: true }).click();
  assert(errors.length === 0, 'no browser runtime errors');
  return { passed: checks.length, checks, errors, diagnosticRequests: requests.length };
}
