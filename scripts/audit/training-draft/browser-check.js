// Run with playwright-cli run-code --filename scripts/audit/training-draft/browser-check.js.
// All server requests, writes and sockets are isolated; no real training starts.
async (page) => {
  const checks = [];
  const errors = [];
  const writes = [];
  const base = new URL(page.url()).origin;
  const earthId = 'earth_merra2_3hourly_v1';
  const schema = { width: { type: 'int', default: 32, min: 1, max: 100, label: '草稿宽度' } };
  const models = [{ id: 'draft-model', display_name: '草稿测试模型', version: 1, validation_status: 'valid',
    param_schema: schema, validation_report: {}, original_filename: 'draft.py' }];
  const sourceTask = {
    id:501,user_id:99009,status:'completed',progress:100,custom_model_name:'草稿迁移来源',
    model_source:'uploaded',uploaded_model_id:'draft-model',uploaded_model_version:1,
    dataset_id:'openmars_mcd',model_script:'demo3.py',output_model_path:'fixture.pth',model_available:true,tags:[],
    hyperparameters:JSON.stringify({model_source:'uploaded',model_architecture:'predrnnv2',
      training_dataset:'openmars_mcd',selected_channels:[],use_sphere:false,window:7,horizon:2,
      epochs:10,batch_size:32,learning_rate:.001,seed:11,early_stopping_patience:0,
      train_ratio:.7,validation_ratio:.2,test_ratio:.1,stlstm_hidden_dims:[64,64,64],
      _uploaded_model_id:'draft-model',_uploaded_model_version:1,custom_model_params:{width:41}}),
    metrics:JSON.stringify({rmse:1,mae:1,r2:0}),
  };
  const assertion = (condition, message) => { if (!condition) throw new Error(message); checks.push(message); };
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('dialog', (dialog) => dialog.dismiss());
  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    let body = {};
    if (path === '/api/auth/me') body = { id: 99009, username: 'draft-audit', role: 'user', is_admin: false };
    else if (path === '/api/user-models') body = { items: models };
    else if (path.includes('/earth-compatibility')) body = {
      package_id: 'draft-model', dataset_id: earthId, status: 'available', compatible: true,
      contract_schema: 'astraatmos_earth_3hourly_uploaded_model_v1', reasons: [],
      frequency_hours: 3, grid_shape: [240, 480], contract: { window: Array.from({length:240},(_,i)=>i+1), horizon: Array.from({length:240},(_,i)=>i+1) },
    };
    else if (path === '/api/training/tasks') body = [sourceTask];
    else if (path === '/api/training/tags') body = [];
    else if (path === '/api/training/scripts') body = ['demo3.py'];
    else if (path === '/api/training/weights') body = { items: [] };
    else if (path === '/api/notifications') body = { items: [], unread_count: 0 };
    else if (path === '/api/datasets') body = { default_earth_dataset_id: earthId, items: [
      { dataset_id: 'openmars_mcd', planet: 'mars', availability: 'available', capabilities: {training:true} },
      { dataset_id: 'mcd_overview', planet: 'mars', availability: 'available', capabilities: {training:true} },
      { dataset_id: earthId, planet: 'earth', availability: 'available', capabilities: {training:true},
        dataset_version:'v1', dataset_fingerprint:'a'.repeat(64), frequency_hours:3,
        time:{count:5848,start:'2020-01-01T01:30:00Z',end:'2021-12-31T22:30:00Z'}, grid_shape:[240,480] },
    ] };
    else if (path === '/api/training/start') {
      writes.push(request.postDataJSON());
      await route.fulfill({ status: 200, contentType:'application/json', body:JSON.stringify({
        id:99009,status:'queued',custom_model_name:'草稿恢复验证',model_source:'uploaded',
        dataset_id:'openmars_mcd',hyperparameters:JSON.stringify(writes.at(-1).hyperparameters),tags:[],
      }) });
      return;
    } else if (request.method() !== 'GET') throw new Error(`Unexpected write ${path}`);
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(body)});
  });
  await page.routeWebSocket('**/ws/**', () => {});
  await page.addInitScript(() => { localStorage.setItem('astraatmos_token','isolated-draft-fixture'); });
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(`${base}/#/training`);
  await page.reload();
  await page.locator('[data-config-canvas="true"]').waitFor();
  const dataset = (id) => page.locator(`[data-training-dataset-option="${id}"]`);
  const value = (key) => page.locator(`[data-parameter="${key}"] input`);
  const tab = (name) => page.getByRole('tab',{name,exact:true});
  const customTab = () => page.getByRole('tab', {name:/^自定义模型参数/});
  const width = () => page.getByLabel('草稿宽度', {exact:false});
  const setParams = async (n) => {
    await customTab().click();
    await width().fill(String(n));
  };
  const checkWidth = async (n, message) => {
    await customTab().click();
    assertion(await width().inputValue() === String(n), message);
  };
  await dataset('mcd_overview').click();
  await tab('训练参数').click();
  await value('windowValue').fill('5');
  await value('horizon').fill('4');
  await setParams(19);
  await dataset(earthId).click();
  await page.locator('[data-earth-model-source-option="uploaded"]').click();
  await page.getByRole('button', {name:'管理模型',exact:true}).click();
  await page.locator('[data-uploaded-model-item="draft-model"]').click();
  await tab('输入与预测').click();
  await value('windowValue').fill('12');
  await value('horizon').fill('8');
  await setParams(73);
  await dataset(earthId).click();
  await checkWidth(73,'clicking selected Earth dataset preserves parameters');
  await dataset('mcd_overview').click();
  await checkWidth(19,'mcd_overview restores its own uploaded parameters');
  await tab('训练参数').click();
  assertion(await value('windowValue').inputValue()==='5','mcd_overview restores window 5');
  assertion(await value('horizon').inputValue()==='4','mcd_overview restores horizon 4');
  await dataset('openmars_mcd').click();
  assertion(await value('windowValue').inputValue()==='5','direct Mars dataset switch preserves window');
  await dataset(earthId).click();
  await checkWidth(73,'Earth restores parameters for the same uploaded model');
  await tab('输入与预测').click();
  assertion(await value('windowValue').inputValue()==='12','Earth restores window 12');
  assertion(await value('horizon').inputValue()==='8','Earth restores horizon 8');
  await dataset('openmars_mcd').click();
  await checkWidth(19,'openmars_mcd round trip restores Mars parameters');
  await page.setViewportSize({width:390,height:844});
  await dataset(earthId).click();
  await checkWidth(73,'390px scene restore keeps Earth custom params');
  await page.screenshot({path:'output/playwright/training-draft-mobile.png',fullPage:true});
  await page.setViewportSize({width:1440,height:1000});
  await dataset('openmars_mcd').click();
  await checkWidth(19,'return from mobile Earth still keeps Mars params');
  await page.locator('#experiment-name-input').fill('草稿恢复验证');
  const taskRefresh = page.waitForResponse(response => response.url().endsWith('/api/training/tasks'));
  await page.locator('.experiment-run-bar').getByRole('button').filter({hasText:'开始'}).click();
  await taskRefresh;
  assertion(writes.length===1,'one intercepted training request');
  assertion(writes[0].hyperparameters.window===5 && writes[0].hyperparameters.horizon===4,'submitted Mars window/horizon match restored draft');
  assertion(writes[0].hyperparameters.custom_model_params.width===19,'submitted custom params match restored Mars draft');
  assertion(writes[0].hyperparameters.training_dataset==='openmars_mcd','submitted identity remains Mars');

  // Exercise migration restoration through the actual page effects and setters.
  await page.reload();
  await page.locator('[data-uploaded-model-card="true"]').waitFor();
  await tab('训练参数').click();
  await value('windowValue').fill('5');
  await value('horizon').fill('4');
  await setParams(19);
  await tab('迁移学习').click();
  await page.getByLabel('启用迁移学习',{exact:true}).check();
  await page.getByRole('combobox',{name:'来源任务',exact:true}).selectOption('501');
  await checkWidth(41,'migration applies source custom params');
  await dataset(earthId).click();
  await dataset('mcd_overview').click();
  await tab('迁移学习').click();
  assertion(await page.getByRole('combobox',{name:/^来源任务/}).inputValue()==='501','scene round trip restores migration source');
  await page.getByLabel('启用迁移学习',{exact:true}).uncheck();
  await checkWidth(19,'turning off restored migration recovers original params');
  await tab('训练参数').click();
  assertion(await value('windowValue').inputValue()==='5','migration undo recovers original window');

  // Copy replaces old scenes; new experiment must not resurrect the copy.
  await page.getByRole('button',{name:'训练监视',exact:true}).click();
  await page.locator('.experiment-directory-row').filter({hasText:'草稿迁移来源'}).click();
  await page.getByRole('button',{name:'复制配置',exact:true}).click();
  await checkWidth(41,'copy loads saved custom params');
  await dataset(earthId).click();
  await dataset('mcd_overview').click();
  await checkWidth(41,'copy survives scene round trip');
  await page.getByRole('button',{name:'训练监视',exact:true}).click();
  await page.getByRole('button',{name:'新建实验',exact:true}).click();
  await checkWidth(32,'new experiment resets custom params to defaults');
  await dataset(earthId).click();
  await dataset('mcd_overview').click();
  await checkWidth(32,'new experiment does not revive copied scene');
  assertion(errors.length===0,`no runtime errors: ${errors.join(';')}`);
  return {passed:checks.length,checks,writes,errors};
}
