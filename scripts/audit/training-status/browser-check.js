// Run with playwright-cli run-code --filename scripts/audit/training-status/browser-check.js.
// All API requests and sockets are isolated; this never starts or stops training.
async (page) => {
  const base = new URL(page.url()).origin;
  const errors = [];
  let requests = 0;
  let inFlight = 0;
  let maxInFlight = 0;
  let completed = false;
  const task = {
    id: 99090, user_id: 99009, custom_model_name: 'Slow task status probe',
    status: 'queued', queue_position: 1, progress: 0, current_epoch: 0, total_epochs: 20,
    model_script: 'earth_daily.py', model_source: 'uploaded', tags: [],
    dataset_id: 'earth_merra2_3hourly_5deg_v1',
    hyperparameters: JSON.stringify({ window: 20, horizon: 20, epochs: 20, batch_size: 32 }),
  };
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') throw new Error(`Unexpected mutation ${path}`);
    let body = {};
    if (path === '/api/auth/me') body = { id: 99009, username: 'status-probe', role: 'user' };
    else if (path === '/api/training/tasks') {
      const sequence = ++requests;
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      const snapshot = sequence === 1 ? task : {
        ...task, status: completed ? 'completed' : 'running', queue_position: null,
        progress: completed ? 100 : 8, current_epoch: completed ? 20 : 2,
        current_loss: .06, eta: '30:00', loss_history: JSON.stringify({train:[.147848],val:[.061655]}),
      };
      // Exceed the five-second polling interval without blocking log requests.
      if (sequence > 1) await new Promise(resolve => setTimeout(resolve, 6200));
      inFlight -= 1;
      await route.fulfill({status:200, contentType:'application/json', body:JSON.stringify([snapshot])});
      return;
    } else if (path.endsWith('/logs')) body = {lines:[
      'Epoch 1/20 Loss=0.147848 Val Loss=0.061655', 'Epoch 2/20 Batch 20/127 Loss=0.064866',
    ]};
    else if (path === '/api/user-models' || path === '/api/training/weights') body = {items:[]};
    else if (path === '/api/training/scripts') body = ['demo3.py'];
    else if (path === '/api/training/tags') body = [];
    else if (path === '/api/datasets') body = {items:[]};
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(body)});
  });
  await page.routeWebSocket('**/ws/**', () => {});
  await page.addInitScript(() => localStorage.setItem('astraatmos_token','isolated-status-fixture'));
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(`${base}/?status-audit=${Date.now()}#/training`);
  await page.getByRole('button',{name:'status-probe',exact:true}).waitFor();
  await page.getByRole('button',{name:'训练监视',exact:true}).click();
  const row = page.locator('.experiment-directory-row').filter({hasText:task.custom_model_name});
  await row.waitFor();
  await row.click();
  const initial = await row.getAttribute('data-status');
  await page.getByRole('log').getByText(/Epoch 2\/20/).waitFor();
  let updated = false;
  try {
    await page.locator('.experiment-directory-row[data-status="running"]').waitFor({timeout:14000});
    updated = true;
  } catch { /* Return the observed failure so the old bundle can be compared. */ }
  const status = await row.getAttribute('data-status');
  const lossChart = (await page.locator('.experiment-monitor-chart').textContent()).includes('等待训练指标数据');
  const result = {initial,status,updated,requests,maxInFlight,errors,lossChartEmpty:lossChart};
  await page.screenshot({path:'output/playwright/training-status-slow-poll.png',fullPage:true});
  if (updated) {
    completed = true;
    await page.locator('.experiment-directory-row[data-status="completed"]').waitFor({timeout:20000});
    result.finalStatus = await row.getAttribute('data-status');
  }
  await page.route(`${base}/__training_status_done__`, route => route.fulfill({
    status:200, contentType:'text/html', body:'<!doctype html><title>Training status audit complete</title>',
  }));
  await page.goto(`${base}/__training_status_done__`);
  await page.unrouteAll({behavior:'wait'});
  if (!updated || result.finalStatus !== 'completed' || maxInFlight !== 1 || errors.length || lossChart) {
    throw new Error(`Training status synchronization failed: ${JSON.stringify(result)}`);
  }
  return result;
}
