// Synthetic API fixtures, isolated browser only. No training or actual prediction is run.
// npx --package @playwright/cli playwright-cli -s=shared-prediction run-code --filename scripts/audit/shared-prediction/browser-check.js
async (page) => {
  const checks = [], errors = [], requests = [];
  const check = (condition, label) => { if (!condition) throw new Error(label); checks.push(label); };
  const dataset = 'earth_merra2_3hourly_v1', fingerprint = 'f'.repeat(64);
  const origin = '2021-07-01T04:30:00Z';
  const grid = { shape: [4, 8], latitude: [-67.5,-22.5,22.5,67.5], longitude: [-157.5,-112.5,-67.5,-22.5,22.5,67.5,112.5,157.5], latitude_range: [-90,90], longitude_range: [-180,180] };
  const field = base => ({ field: grid.latitude.map((lat,row) => grid.longitude.map((lon,col) => base + row + col / 10)), minVal: base, maxVal: base + 3.7, valid_cells:32 });
  const overall = { mse: 1.5625, rmse:1.25, mae:.75, r2:.812345, mape:1.75, smape:1.85 };
  const tasks = [99041,99042,99043].map((id,index) => ({
    id, user_id:99040, custom_model_name: index < 2 ? `合成地球模型 ${index+1}` : '合成火星模型', status:'completed', model_available:true,
    dataset_id:index < 2 ? dataset : 'openmars_mcd', dataset_version:'v1', dataset_fingerprint:fingerprint, tags:[],
    hyperparameters: { training_dataset:index < 2 ? dataset : 'openmars_mcd', model_architecture:index < 2 ? 'dlinear':'convlstm', model_source:'official', window:2, horizon:3, selected_channels:index < 2 ? ['U10M']:['Temperature'] },
  }));
  const context = id => ({ planet:'earth', task_id:id, dataset_id:dataset, dataset_version:'v1', dataset_fingerprint:fingerprint,
    target:'TO3',target_unit:'DU',window:2,horizon:3,frequency_hours:3,grid,input_channel_order:['TO3','U10M'],
    origins:{start:origin,end:origin,count:1,timestamps:[origin]}, model:{model_source:'official',model_architecture:'dlinear'},
    run:{}, metrics:{splits:{test:{overall,window_count:20}},split_policy:'published_manifest_splits',
      split_ratios:{train_ratio:.7,validation_ratio:.2,test_ratio:.1},
      split_ranges:{test:{date_start:'2021-07-01T07:30:00Z',date_end:'2021-07-01T13:30:00Z',window_count:20}}}, warnings:[] });
  const earthResult = id => ({ ...context(id), forecast_origin:origin,origin_split:'test',input_timestamps:['2021-07-01T01:30:00Z',origin],
    target_timestamps:['2021-07-01T07:30:00Z','2021-07-01T10:30:00Z','2021-07-01T13:30:00Z'],
    reference:[field(300),field(301),field(302)],prediction:[field(300.123456),field(301.123456),field(302.123456)],residual:[field(-2),field(-1),field(-3)],
    metrics:{unit:'DU',target:'TO3',overall,by_lead:[1,2,3].map(lead_step=>({lead_step,...overall}))},
    export_ref:{planet:'earth',task_id:id,snapshot_id:'synthetic-earth',dataset_id:dataset,dataset_fingerprint:fingerprint} });
  const marsResult = { planet:'mars',horizon:3,ls_values:[90,95,100],selected_variables:['Temperature'],
    ground_truth:[field(10),field(11),field(12)].map(f=>({...f,lat:grid.latitude,lon:grid.longitude})),
    prediction:[field(11),field(12),field(13)].map(f=>({...f,lat:grid.latitude,lon:grid.longitude})),
    residual:[field(-1),field(-1),field(-1)].map(f=>({...f,lat:grid.latitude,lon:grid.longitude})),
    metrics:{overall:{rmse:2.5,mae:1,ssim:.9,r2:.8}},
    export_ref:{planet:'mars',task_id:99043,snapshot_id:'synthetic-mars'} };
  const diagnostic = (id,parameters) => ({...context(id),unit:'DU',parameters,
    scope:{split:'test',window_count:4,available_window_count:20,valid_points:32,grid_shape:grid.shape,
      time_range:{input_start:origin,target_start:'2021-07-01T07:30:00Z',target_end:'2021-07-10T13:30:00Z'},spatial_coverage:{latitude_range:[-90,90],longitude_range:[-180,180]},sampling:{seed:42}},
    full_test_metrics:{metrics:{overall,window_count:20}},sample_metrics:{overall},
    scatter:{reference:[300,301,302,303],prediction:[301,302,303,304],point_count:4,unit:'DU'},
    histogram:{edges:[-2,0,2],counts:[10,22],valid_points:32,unit:'DU',residual_definition:'prediction-reference',summary:{mean:.75,std:.2,count:32}},
    pfi:{status:'completed',baseline_rmse:1.25,items:[{channel:'TO3',importance_mean:2.123456,importance_std:.1,repeats:[]},{channel:'U10M',importance_mean:-.123456,importance_std:.01,repeats:[]}]},cache:{hit:false} });
  let failure = false, mismatched = false, held = false, release, pending;
  page.on('pageerror',error=>errors.push(error.message));
  await page.unroute('**/api/**');
  await page.route('**/api/**',async route=>{
    const request=route.request(), url=new URL(request.url()), path=url.pathname;
    let body={},status=200;
    if(path==='/api/auth/me') body={id:99040,username:'synthetic-browser-fixture',email:'synthetic@example.invalid',role:'user'};
    else if(path==='/api/training/tasks') body=tasks;
    else if(path==='/api/earth/predict/context') body=context(Number(url.searchParams.get('training_task_id')));
    else if(path==='/api/earth/predict/diagnostics/context') body={available:true,task_id:Number(url.searchParams.get('training_task_id'))};
    else if(path==='/api/earth/predict/run') {
      const params=request.postDataJSON();requests.push({path,params});
      if(held) await new Promise(resolve=>{release=resolve;pending=params;});
      if(failure) {status=409;body={detail:{code:'invalid_earth_training_artifact',message:'合成 API 错误，可重试'}};}
      else body=earthResult(mismatched ? 99042 : params.training_task_id);
    }
    else if(path==='/api/earth/predict/diagnostics') {const params=request.postDataJSON();requests.push({path,params});body=diagnostic(params.training_task_id,params);}
    else if(path==='/api/predict/run') {requests.push({path,params:request.postDataJSON()});body=marsResult;}
    else if(path==='/api/predict/metrics') body={overall:{rmse:2.5,mae:1,ssim:.9,r2:.8},aggregation:{overall:'pooled_test_set_pixels'}};
    else if(path==='/api/predict/error-distribution') body={hist_trues:{bin_edges:[10,20,30],counts:[8,24]},hist_preds:{bin_edges:[10,20,30],counts:[7,25]},hist_errors:{bin_edges:[-1,0,1],counts:[12,20]},rmse:2.5,mae:1};
    else if(path==='/api/predict/permutation-importance') body={baseline_value:.8,items:[{name:'Temperature',importance:-.012345}],export_ref:{planet:'mars',task_id:99043,snapshot_id:'synthetic-pfi'}};
    else if(path==='/api/notifications') body={items:[],unread_count:0};
    else if(['/api/training/tags','/api/user-models','/api/training/scripts'].includes(path)) body=[];
    else if(path==='/api/datasets') body={default_earth_dataset_id:dataset,datasets:[]};
    else if(request.method()!=='GET') throw new Error(`Unexpected mutation ${path}`);
    try {await route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});} catch(error) {if(!request.failure()) throw error;}
  });
  await page.routeWebSocket('**/ws/**',()=>{});
  await page.addInitScript(()=>{
    if(!localStorage.getItem('shared_prediction_fixture')) {
      localStorage.setItem('astraatmos_token','synthetic-isolated-browser-only');
      localStorage.setItem('astraatmos_settings',JSON.stringify({language:'zh',theme:'dark',colormap:'viridis',precision:4,units:{ozone:'DU'}}));
      localStorage.setItem('shared_prediction_fixture','1');
    }
  });
  await page.evaluate(()=>localStorage.removeItem('shared_prediction_fixture'));
  await page.setViewportSize({width:1440,height:900});
  await page.goto('http://127.0.0.1:5173/#/predict?mode=earth');
  await page.reload();
  const earth=page.locator('[data-single-model-workbench="earth"]');
  const run=page.getByRole('button',{name:/开始预测/});
  await run.waitFor(); await run.isEnabled();
  await page.locator('[data-prediction-origin="earth"]').waitFor();
  await page.waitForFunction(()=>!document.querySelector('input[data-prediction-origin="earth"]').disabled);
  check(requests.length===0,'no inference or diagnostics on Earth mount');
  check(await page.locator('.predict-sidebar input[aria-label="预测步长"]').getAttribute('readonly')!==null,'Earth horizon is read-only');
  check(await earth.locator('.prediction-display--empty').count()===1 && await earth.locator('.prediction-display__card').count()===0,'Earth without results uses one compact guide');
  await earth.getByRole('button',{name:'选择训练模型',exact:true}).click();
  check(await page.locator('[role="listbox"]').isVisible() && requests.length===0,'empty guide opens the existing model selector without inference');
  await page.keyboard.press('Escape');
  await run.click();
  await earth.locator('.prediction-earth-field').first().waitFor();
  check(requests[0].params.training_task_id===99041 && !('horizon' in requests[0].params),'Earth request runs full horizon');
  const display=earth.locator('[data-testid="prediction-display"]');
  check((await display.locator('.prediction-display__triptych .prediction-display__header').allTextContents()).map(t=>t.trim().slice(0,2)).join(',')==='参考,预测,残差','triptych order reference prediction residual');
  check(await earth.locator('[data-predict-metrics][data-evaluation-scope="current_window"] [data-predict-metric]').count()===6,'Earth six metrics');
  check(await earth.locator('[data-evaluation-scope="current_window"] [data-predict-metric="rmse"] strong').innerText()==='1.2500','Earth DU and four decimal precision');
  check(await earth.locator('.prediction-earth-field[data-colormap="viridis"]').count()===2,'Earth follows colormap preference');
  check((await earth.locator('[data-predict-metrics][data-evaluation-scope="full_test"]').innerText()).includes('发布测试分区：2021-07-01T07:30:00Z — 2021-07-01T13:30:00Z · 20 个窗口'),'Earth full-test metrics reuse the Mars split line from the checkpoint binding');
  check(await display.getByRole('button',{name:'导出科研图',exact:true}).isEnabled(),'Earth supported triptych export remains available');
  const requestCount=requests.length;
  const fab=page.locator('.settings-fab');
  await fab.locator('> button').click();await fab.getByText('语言',{exact:true}).hover();await fab.getByText('English',{exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('[data-single-model-workbench="earth"]')?.textContent.includes('Current forecast window'));
  check(await earth.locator('.prediction-earth-field').count()===3 && requests.length===requestCount,'live language change preserves Earth fields without inference');
  await fab.getByText('Language',{exact:true}).hover();await fab.getByText('中文',{exact:true}).click();await fab.locator('> button').click();
  const before=requests.length;
  await display.getByRole('button',{name:'+6h · 2021-07-01T10:30:00Z UTC',exact:true}).click();
  check(requests.length===before,'time-step switch does not request prediction');
  await display.getByRole('button',{name:'预测结果',exact:true}).click();
  check(await display.locator('.prediction-display__triptych').count()===0 && await display.locator('.prediction-earth-field').count()===1,'single chart switch');
  await display.getByRole('button',{name:'全屏查看',exact:true}).click();
  const dialog=page.getByRole('dialog');await dialog.waitFor();
  check((await dialog.innerText()).includes('+6h') && !(await dialog.innerText()).includes('Ls='),'Earth fullscreen UTC only');
  check(await dialog.locator('[data-testid="earth-map-svg"]').count()===1 && !(await dialog.innerText()).includes('μm-atm'),'Earth fullscreen map and DU');
  check((await dialog.innerText()).includes('8 × 4'),'fullscreen actual grid dimensions');
  await page.screenshot({path:'output/playwright/shared-earth-fullscreen-desktop.png'});
  await page.keyboard.press('Escape'); await dialog.waitFor({state:'detached'});
  check(await page.evaluate(()=>document.body.style.overflow)!=='hidden','Escape restores page scrolling');
  await page.setViewportSize({width:390,height:844});
  await display.getByRole('button',{name:'三联对比',exact:true}).click();
  await page.waitForTimeout(300);
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+2),'390px Earth page no horizontal overflow');
  check((await display.locator('.prediction-display__triptych').evaluate(e=>getComputedStyle(e).gridTemplateColumns)).split(' ').length===1,'390px triptych stacks');
  await display.scrollIntoViewIfNeeded();await page.screenshot({path:'output/playwright/shared-earth-390.png'});
  await display.getByRole('button',{name:'全屏查看',exact:true}).first().click();await dialog.waitFor();
  check(await dialog.locator('.prediction-fullscreen__close').isVisible(),'390px fullscreen close reachable');
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=390),'390px fullscreen no page overflow');
  await page.screenshot({path:'output/playwright/shared-earth-fullscreen-390.png'});await page.keyboard.press('Escape');
  await page.setViewportSize({width:1440,height:900});
  const diagnostics=page.locator('[data-earth-diagnostics]');
  await diagnostics.getByRole('button',{name:'计算测试集诊断',exact:true}).click();
  await diagnostics.locator('[data-pfi-axis-label]').waitFor();
  const params=requests.at(-1).params;
  check(params.sample_windows===4 && params.pfi_repeats===3 && params.seed===42 && params.scatter_points===5000 && params.histogram_bins===40,'diagnostic manual defaults retained');
  check((await diagnostics.innerText()).includes('ΔRMSE') && (await diagnostics.innerText()).includes('-0.1235'),'Earth negative DU PFI');
  check(await diagnostics.locator('[data-predict-metrics][data-evaluation-scope="full_test"]').count()===1 && await diagnostics.locator('[data-predict-metrics][data-evaluation-scope="sampled_diagnostic"]').count()===1,'full-test and sampled metrics separated');
  check(await diagnostics.locator('button').filter({hasText:'导出科研图'}).count()===0,'Earth diagnostic export remains unavailable');
  await page.setViewportSize({width:390,height:844});await diagnostics.scrollIntoViewIfNeeded();await page.waitForTimeout(250);
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=390),'390px sampled diagnostic charts no overflow');
  await page.screenshot({path:'output/playwright/shared-earth-diagnostics-390.png'});
  await page.setViewportSize({width:1440,height:900});
  await page.locator('button[aria-haspopup="listbox"]').click();
  await page.getByRole('option',{name:/合成地球模型 2/}).click();
  await page.waitForFunction(()=>!document.querySelector('input[data-prediction-origin="earth"]').disabled);
  check(await earth.locator('.prediction-earth-field').count()===0 && await diagnostics.locator('[data-pfi-axis-label]').count()===0,'task change clears fields and diagnostics');
  failure=true;await run.click();await page.getByRole('alert').waitFor();
  failure=false;await page.getByRole('button',{name:'重试预测',exact:true}).click();await earth.locator('.prediction-earth-field').first().waitFor();
  check(true,'error and retry recover without changing origin');
  await page.getByRole('button',{name:'火星',exact:true}).click();
  const mars=page.locator('[data-single-model-workbench="mars"]');await mars.waitFor();
  check(await earth.count()===0 && await mars.locator('.prediction-earth-field').count()===0,'planet switch clears Earth workbench');
  await page.getByRole('button',{name:/开始预测/}).click();await mars.locator('canvas').first().waitFor();
  await mars.locator('[data-pfi-axis-label]').waitFor();
  check((await mars.innerText()).includes('Ls=') && (await mars.innerText()).includes('ΔR²'),'Mars Ls and delta R-squared');
  check(await mars.locator('[data-predict-metric="rmse"] strong').innerText()==='0.2500','Mars DU converted once');
  await page.getByRole('button',{name:/开始预测/}).click();
  await page.waitForFunction(()=>document.querySelector('[data-single-model-workbench="mars"] [data-predict-metrics]')?.dataset.evaluationScope==='full_test'
    && document.querySelector('[data-single-model-workbench="mars"] [data-predict-metric="rmse"] strong')?.textContent==='0.2500');
  check(true,'repeat Mars run retains complete-test metrics');
  await page.setViewportSize({width:390,height:844});await mars.scrollIntoViewIfNeeded();await page.waitForTimeout(250);
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=390),'390px Mars layout no overflow');
  await page.screenshot({path:'output/playwright/shared-mars-390.png'});await page.setViewportSize({width:1440,height:900});
  await mars.locator('[data-testid="prediction-display"]').getByRole('button',{name:'全屏查看',exact:true}).first().click();await dialog.waitFor();
  check((await dialog.innerText()).includes('火星球面') && await dialog.locator('canvas').count()>0,'Mars fullscreen sphere');await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'地球',exact:true}).click();await run.waitFor();
  await page.waitForFunction(()=>!document.querySelector('input[data-prediction-origin="earth"]').disabled);
  await page.locator('button[aria-haspopup="listbox"]').click();await page.getByRole('option',{name:/合成地球模型 1/}).click();
  await page.waitForFunction(()=>!document.querySelector('input[data-prediction-origin="earth"]').disabled);
  mismatched=true;await run.click();await page.getByRole('alert').filter({hasText:'预测响应身份已变化'}).waitFor();
  check(await earth.locator('.prediction-earth-field').count()===0,'wrong task response rejected');mismatched=false;
  held=true;await run.click();await page.waitForTimeout(100);
  await page.evaluate(()=>window.location.hash='#/predict?mode=trained');await mars.waitFor();held=false;release();await page.waitForTimeout(250);
  check(await earth.count()===0,'late Earth response after route switch rejected');
  await page.evaluate(()=>{const settings=JSON.parse(localStorage.getItem('astraatmos_settings'));settings.theme='light';settings.colormap='plasma';settings.precision=6;localStorage.setItem('astraatmos_settings',JSON.stringify(settings));});
  await page.goto('http://127.0.0.1:5173/#/predict?mode=earth');await page.reload();await run.waitFor();
  await page.waitForFunction(()=>!document.querySelector('input[data-prediction-origin="earth"]').disabled);await run.click();await earth.locator('.prediction-earth-field').first().waitFor();
  check(await page.locator('html').getAttribute('data-theme')==='light','light theme rendered');
  check(await earth.locator('[data-evaluation-scope="current_window"] [data-predict-metric="rmse"] strong').innerText()==='1.250000' && await earth.locator('[data-colormap="plasma"]').count()===2,'six decimals and palette apply after settings change');
  await display.scrollIntoViewIfNeeded();await page.screenshot({path:'output/playwright/shared-earth-light-desktop.png'});
  await page.setViewportSize({width:390,height:844});await display.scrollIntoViewIfNeeded();await page.waitForTimeout(250);
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=390),'390px light theme no overflow');await page.screenshot({path:'output/playwright/shared-earth-light-390.png'});
  await page.evaluate(()=>window.dispatchEvent(new Event('astraatmos:logout')));
  await page.waitForFunction(()=>document.querySelectorAll('.prediction-earth-field').length===0);
  check(await page.locator('.prediction-earth-field').count()===0,'logout clears private prediction');
  await page.getByRole('button', { name: '地球', exact: true }).click(); await earth.waitFor();
  check(await earth.locator('.prediction-display--empty').count() === 1 && await earth.locator('.prediction-metric-value').count() === 0, 'Earth without a model uses a guide without empty metric cards');
  check(await earth.getByRole('link', { name: '去模型训练', exact: true }).getAttribute('href') === '#/training', 'Earth no-model guide leads to existing training');
  check(await earth.locator('h2').count() === 0 && await earth.locator('[data-earth-context-model]').count() === 0, 'Earth single-model page adds no panel of its own');
  check(await page.locator('[data-prediction-origin]').count() === 1, 'Earth keeps exactly one shared origin control in the sidebar');
  check(await page.getByRole('button', { name: '刷新起点范围', exact: true }).count() === 0, 'Earth no longer adds a reload-origin button Mars does not have');
  check(await page.locator('.predict-sidebar').getByRole('button', { name: '重试', exact: true }).count() === 0, 'Earth context retry only appears after a real failure');
  await page.getByRole('button', { name: '火星', exact: true }).click(); await mars.waitFor();
  check(await mars.locator('.prediction-display--empty').count() === 1 && await mars.locator('.prediction-metric-value').count() === 0, 'Mars uses the same compact no-model guide');
  check(errors.length===0,`browser runtime errors: ${errors.join('; ')}`);
  return {synthetic:true,passed:checks.length,checks,errors,requests:requests.length};
}
