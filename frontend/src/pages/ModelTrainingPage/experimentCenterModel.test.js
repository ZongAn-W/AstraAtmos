import test from 'node:test';
import assert from 'node:assert/strict';
import {
  EXPERIMENT_STAGES,
  EXPERIMENT_STATUS_FILTERS,
  buildExperimentCopyName,
  buildExperimentSummary,
  canUseTaskForPrediction,
  isRetiredEarthTask,
  cloneExperimentConfigValue,
  countExperimentStatuses,
  filterExperimentTasks,
  formatExperimentMetricValue,
  getExperimentFailureMessage,
  getExperimentLogLineTone,
  getExperimentStage,
  normalizeTaskChannels,
  parseTaskHyperparameters,
  parseTaskMetrics,
  readExperimentConfig,
  readExperimentMetrics,
  resolveActiveTaskProgress,
  readExperimentConfigDataset,
} from './experimentCenterModel.js';

test('daily tasks retain their identity but cannot be used for prediction', () => {
  for (const datasetId of ['earth_merra2_daily_v1', 'earth_merra2_daily_v2']) {
    const task = { dataset_id: datasetId, status: 'completed', model_available: true };
    assert.equal(isRetiredEarthTask(task), true);
    assert.equal(canUseTaskForPrediction(task), false);
    assert.equal(isRetiredEarthTask({ hyperparameters: JSON.stringify({ training_dataset: datasetId }) }), true);
  }
  assert.equal(canUseTaskForPrediction({ dataset_id: 'earth_merra2_3hourly_v1', status: 'completed', model_available: true }), true);
});

test('没有活动任务时进入配置阶段', () => {
  assert.equal(getExperimentStage({ activeTask: null, isCreating: false }), 'configure');
  assert.equal(getExperimentStage({ activeTask: undefined, isCreating: false }), 'configure');
});

test('运行中和排队任务进入监控阶段', () => {
  assert.equal(getExperimentStage({ activeTask: { status: 'pending' }, isCreating: false }), 'monitor');
  assert.equal(getExperimentStage({ activeTask: { status: 'running' }, isCreating: false }), 'monitor');
});

test('完成、失败和停止后的任务进入结果阶段', () => {
  for (const status of ['completed', 'failed', 'stopped', 'cancelled', 'unknown-state']) {
    assert.equal(getExperimentStage({ activeTask: { status }, isCreating: false }), 'result');
  }
});

test('新建实验优先于当前已完成任务', () => {
  assert.equal(getExperimentStage({ activeTask: { status: 'completed' }, isCreating: true }), 'configure');
  assert.equal(getExperimentStage({ activeTask: { status: 'running' }, isCreating: true }), 'configure');
});

test('阶段标识只包含配置、监控和结果三种取值', () => {
  assert.deepEqual(EXPERIMENT_STAGES, ['configure', 'monitor', 'result']);
});

test('筛选标识与目录状态分组一致', () => {
  assert.deepEqual(EXPERIMENT_STATUS_FILTERS, ['all', 'queued', 'running', 'completed', 'failed', 'cancelled']);
});

test('坏的 metrics JSON 不阻塞结果页', () => {
  assert.deepEqual(parseTaskMetrics('{bad'), {});
  assert.deepEqual(parseTaskMetrics(null), {});
  assert.deepEqual(parseTaskMetrics(undefined), {});
  assert.deepEqual(parseTaskMetrics('[]'), {});
  assert.deepEqual(parseTaskMetrics('"text"'), {});
  assert.deepEqual(parseTaskMetrics(42), {});
  assert.deepEqual(parseTaskMetrics('{"rmse": 0.5}'), { rmse: 0.5 });
  assert.deepEqual(parseTaskMetrics({ mae: 0.2 }), { mae: 0.2 });
});

test('坏的超参数 JSON 回退为空对象', () => {
  assert.deepEqual(parseTaskHyperparameters('{bad'), {});
  assert.deepEqual(parseTaskHyperparameters('[]'), {});
  assert.deepEqual(parseTaskHyperparameters(null), {});
  assert.deepEqual(parseTaskHyperparameters({ window: 7 }), { window: 7 });
  assert.deepEqual(parseTaskHyperparameters('{"window": 7}'), { window: 7 });
});

test('状态和名称筛选可叠加且不修改原数组', () => {
  const tasks = [
    { id: 1, status: 'completed', custom_model_name: 'Ozone baseline', tags: [{ id: 2 }] },
    { id: 2, status: 'running', custom_model_name: 'Dust run', tags: [] },
  ];
  const result = filterExperimentTasks(tasks, { status: 'completed', search: 'ozone', tagIds: [2] });
  assert.deepEqual(result.map(task => task.id), [1]);
  assert.equal(tasks.length, 2);
});

test('筛选结果保留传入任务对象的引用顺序', () => {
  const tasks = [
    { id: 4, status: 'failed', custom_model_name: 'B' },
    { id: 3, status: 'running', custom_model_name: 'A' },
    { id: 8, status: 'failed', custom_model_name: 'C' },
  ];
  assert.deepEqual(filterExperimentTasks(tasks, { status: 'failed' }).map(task => task.id), [4, 8]);
  assert.deepEqual(filterExperimentTasks(tasks, {}).map(task => task.id), [4, 3, 8]);
  assert.deepEqual(filterExperimentTasks(tasks, { status: 'running' }).map(task => task.id), [3]);
});

test('运行中筛选同时覆盖排队任务', () => {
  const tasks = [
    { id: 1, status: 'pending', custom_model_name: 'queued' },
    { id: 2, status: 'running', custom_model_name: 'active' },
    { id: 3, status: 'completed', custom_model_name: 'done' },
  ];
  assert.deepEqual(filterExperimentTasks(tasks, { status: 'running' }).map(task => task.id), [1, 2]);
});

test('名称搜索大小写不敏感并可作为任务 ID 搜索', () => {
  const tasks = [
    { id: 11, status: 'completed', custom_model_name: 'Ozone Baseline' },
    { id: 12, status: 'completed', custom_model_name: null },
  ];
  assert.deepEqual(filterExperimentTasks(tasks, { search: 'OZONE' }).map(task => task.id), [11]);
  assert.deepEqual(filterExperimentTasks(tasks, { search: '12' }).map(task => task.id), [12]);
  assert.deepEqual(filterExperimentTasks(tasks, { search: '   ' }).map(task => task.id), [11, 12]);
});

test('未分组筛选只保留没有标签的任务', () => {
  const tasks = [
    { id: 1, status: 'completed', custom_model_name: 'tagged', tags: [{ id: 5 }] },
    { id: 2, status: 'completed', custom_model_name: 'loose', tags: [] },
  ];
  assert.deepEqual(filterExperimentTasks(tasks, { untagged: true }).map(task => task.id), [2]);
  assert.deepEqual(filterExperimentTasks(tasks, { tagIds: [5] }).map(task => task.id), [1]);
  assert.deepEqual(filterExperimentTasks(tasks, { tagIds: [5, 6] }), []);
});

test('异常任务列表不会抛出', () => {
  assert.deepEqual(filterExperimentTasks(null, { status: 'completed' }), []);
  assert.deepEqual(filterExperimentTasks(undefined, {}), []);
  assert.deepEqual(filterExperimentTasks([null, { id: 1, status: 'completed' }], {}).length, 1);
});

test('摘要优先使用任务名称、架构、数据集和指标', () => {
  const summary = buildExperimentSummary({
    id: 7,
    custom_model_name: 'Ozone baseline',
    status: 'completed',
    model_source: 'official',
    hyperparameters: JSON.stringify({ model_architecture: 'simvp', training_dataset: 'mcd_overview' }),
    metrics: JSON.stringify({ rmse: 0.0312, mae: 0.0184 }),
  });
  assert.equal(summary.name, 'Ozone baseline');
  assert.equal(summary.architecture, 'simvp');
  assert.equal(summary.dataset, 'mcd_overview');
  assert.equal(summary.targetVariable, 'O3');
  assert.deepEqual(summary.inputVariables, ['O3']);
  assert.equal(summary.metrics.rmse, 0.0312);
});

test('summary preserves an independent environment target and input list', () => {
  const summary = buildExperimentSummary({
    id: 14,
    dataset_id: 'earth_merra2_3hourly_v1',
    hyperparameters: JSON.stringify({ target_variable: 'T2M', input_variables: ['TO3', 'SWGDN'] }),
  });
  assert.equal(summary.targetVariable, 'T2M');
  assert.deepEqual(summary.inputVariables, ['TO3', 'SWGDN']);
});

test('摘要对缺名、坏 JSON 和缺失指标给出稳定回退', () => {
  const summary = buildExperimentSummary({ id: 9, status: 'failed', hyperparameters: '{bad', metrics: null });
  assert.equal(summary.id, 9);
  assert.equal(summary.name, '');
  assert.equal(summary.status, 'failed');
  assert.equal(summary.architecture, '');
  assert.equal(summary.dataset, '');
  assert.deepEqual(summary.metrics, {});
  assert.equal(summary.modelAvailable, false);
});

test('摘要读取后端返回的数据集身份与模型可用状态', () => {
  const summary = buildExperimentSummary({
    id: 12,
    custom_model_name: 'Dust run',
    status: 'completed',
    model_available: true,
    dataset_id: 'openmars_mcd',
    progress: 100,
  });
  assert.equal(summary.dataset, 'openmars_mcd');
  assert.equal(summary.modelAvailable, true);
});

test('指标解析兼容大写键、别名与非法数值', () => {
  const metrics = readExperimentMetrics({
    RMSE: 0.4,
    mae: '0.2',
    MSE: null,
    'R2': 0.9,
    r_squared: 0.8,
    MAPE: Number.NaN,
    smape: 3,
  });
  assert.equal(metrics.rmse, 0.4);
  assert.equal(metrics.mae, '0.2');
  // 空值与 NaN 视为该指标缺失：结果页显示“指标暂不可用”，不渲染 NaN。
  assert.equal(metrics.mse, undefined);
  assert.deepEqual(metrics.r2, 0.9);
  assert.equal(metrics.mape, undefined);
  assert.equal(metrics.smape, 3);
  assert.deepEqual(readExperimentMetrics(null), {});
  assert.deepEqual(readExperimentMetrics([1, 2]), {});
  assert.deepEqual(readExperimentMetrics('{"RMSE": 0.1, "unknown": 5}'), { rmse: 0.1 });
});

test('目录计数按状态分组统计并合计全部', () => {
  const counts = countExperimentStatuses([
    { id: 1, status: 'pending' },
    { id: 2, status: 'running' },
    { id: 3, status: 'completed' },
    { id: 4, status: 'failed' },
    { id: 5, status: 'completed' },
  ]);
  assert.deepEqual(counts, { all: 5, queued: 0, running: 2, completed: 2, failed: 1, cancelled: 0 });
});

test('结果阶段使用任务自身的进度而不是上一个任务的缓存', () => {
  const task = {
    id: 21,
    status: 'completed',
    progress: 100,
    current_epoch: 10,
    total_epochs: 10,
    current_loss: 0.012,
    eta: '--:--',
    loss_history: JSON.stringify({ train: [1, 0.5], val: [1.1, 0.6] }),
  };
  const stale = { progress: 4, current_epoch: 1, total_epochs: 10, loss_history: { train: [9], val: [9] } };
  const resolved = resolveActiveTaskProgress(task, stale);
  assert.equal(resolved.progress, 100);
  assert.equal(resolved.current_epoch, 10);
  assert.equal(resolved.current_loss, 0.012);
  assert.deepEqual(resolved.loss_history, { train: [1, 0.5], val: [1.1, 0.6] });
});

test('监控阶段优先使用实时推送的进度', () => {
  const task = { id: 22, status: 'running', progress: 10, current_epoch: 1, total_epochs: 10 };
  const live = { progress: 42, current_epoch: 4, total_epochs: 10, current_loss: 0.3, loss_history: { train: [2], val: [3] } };
  const resolved = resolveActiveTaskProgress(task, live);
  assert.equal(resolved.progress, 42);
  assert.equal(resolved.current_epoch, 4);
  assert.equal(resolved.current_loss, 0.3);
});

test('没有活动任务时进度回退为空状态', () => {
  assert.deepEqual(resolveActiveTaskProgress(null, { progress: 80 }), {
    progress: 0,
    current_epoch: 0,
    total_epochs: 0,
    current_loss: null,
    eta: '--:--',
    loss_history: { train: [], val: [] },
  });
});

test('副本配置读取数据集、命名与标签', () => {
  const config = readExperimentConfigDataset({
    id: 31,
    custom_model_name: 'Copy me',
    hyperparameters: JSON.stringify({ training_dataset: 'mcd_overview' }),
    tags: [{ id: 3 }, { id: 4 }],
  });
  assert.equal(config.trainingDataset, 'mcd_overview');
  assert.equal(config.name, 'Copy me');
  assert.deepEqual(config.tagIds, [3, 4]);
  assert.equal(config.sourceTaskId, 31);
});

test('副本配置在字段缺失时安全回退', () => {
  const config = readExperimentConfigDataset({ id: 32, hyperparameters: '{bad' });
  assert.equal(config.trainingDataset, '');
  assert.equal(config.name, '');
  assert.deepEqual(config.tagIds, []);
  assert.equal(config.sourceTaskId, 32);
  assert.deepEqual(readExperimentConfigDataset(null).tagIds, []);
});

test('复制配置会读取训练轮次、批大小、学习率、随机种子和早停参数', () => {
  const config = readExperimentConfig({
    custom_model_name: 'baseline',
    hyperparameters: JSON.stringify({
      training_dataset: 'mcd_overview',
      model_architecture: 'simvp',
      selected_channels: ['U', 'V', 'T'],
      use_sphere: true,
      window: 8,
      horizon: 3,
      epochs: 40,
      batch_size: 16,
      learning_rate: 0.0003,
      seed: 42,
      early_stopping_patience: 7,
      spatial_hidden_dim: 64,
      temporal_hidden_dim: 128,
    }),
  });

  assert.equal(config.epochs, 40);
  assert.equal(config.batchSize, 16);
  assert.equal(config.learningRate, 0.0003);
  assert.equal(config.seed, 42);
  assert.equal(config.earlyStoppingPatience, 7);
});

test('复制配置覆盖数据集、来源、通道、窗口、步长、SPHERE 与结构参数', () => {
  const config = readExperimentConfig({
    id: 51,
    custom_model_name: 'simvp run',
    hyperparameters: JSON.stringify({
      training_dataset: 'mcd_overview',
      model_architecture: 'simvp',
      model_source: 'official',
      selected_channels: ['V', 'U'],
      use_sphere: true,
      window: 8,
      horizon: 4,
      epochs: 12,
      batch_size: 8,
      learning_rate: 0.002,
      spatial_hidden_dim: 32,
      temporal_hidden_dim: 96,
      temporal_depth: 3,
      dropout: 0.2,
    }),
    tags: [{ id: 9 }],
  }, {
    // 页面传入通道表顺序，读取结果与训练提交时的顺序一致。
    channelOrder: ['U', 'V', 'D', 'S', 'T'],
  });

  assert.equal(config.sourceTaskId, 51);
  assert.equal(config.trainingDataset, 'mcd_overview');
  assert.equal(config.modelSource, 'official');
  // 通道按通道表顺序归一化，而不是按任务里的书写顺序。
  assert.deepEqual(config.selectedChannels, ['U', 'V']);
  assert.equal(config.windowValue, 8);
  assert.equal(config.horizon, 4);
  assert.equal(config.useSphere, true);
  assert.equal(config.modelArchitecture, 'simvp');
  assert.deepEqual(config.architectureParamsByModel.simvp, {
    spatial_hidden_dim: 32,
    temporal_hidden_dim: 96,
    temporal_depth: 3,
    dropout: 0.2,
  });
  assert.deepEqual(config.tagIds, [9]);
  assert.deepEqual(config.warnings, []);
});

test('复制配置读取循环网络的隐藏层维度', () => {
  const config = readExperimentConfig({
    custom_model_name: 'predrnn',
    hyperparameters: JSON.stringify({
      model_architecture: 'convlstm',
      stlstm_hidden_dims: [32, 48],
      window: 5,
      horizon: 2,
      epochs: 20,
      batch_size: 8,
      learning_rate: 0.002,
    }),
  });
  assert.deepEqual(config.hiddenDims, [32, 48]);
  assert.deepEqual(config.warnings, []);
});

test('复制配置始终关闭迁移学习并清空迁移来源', () => {
  const config = readExperimentConfig({
    custom_model_name: 'transferred',
    hyperparameters: JSON.stringify({
      model_architecture: 'simvp',
      transfer_learning: true,
      transfer_source_type: 'task',
      transfer_source_task_id: 12,
      freeze_mode: 'backbone',
      finetune_learning_rate: 0.00005,
      spatial_hidden_dim: 64,
      temporal_hidden_dim: 128,
      temporal_depth: 2,
      dropout: 0.1,
    }),
  });
  assert.equal(config.transferEnabled, false);
  assert.equal(config.transferSourceType, 'task');
  assert.equal(config.transferSourceTaskId, '');
  assert.equal(config.transferFreezeMode, 'none');
});

test('复制配置在超参数不可读时给出警告并保留可用字段', () => {
  const config = readExperimentConfig({
    id: 52,
    custom_model_name: 'broken',
    dataset_id: 'openmars_mcd',
    tags: [{ id: 4 }],
    hyperparameters: '{bad',
  });
  assert.equal(config.customModelName, 'broken (Copy)');
  assert.equal(config.trainingDataset, 'openmars_mcd');
  assert.deepEqual(config.tagIds, [4]);
  assert.ok(config.warnings.includes('unreadable_hyperparameters'));
  // 回退值来自默认契约，仍然可直接提交前修改。
  assert.equal(config.epochs, 10);
  assert.equal(config.modelArchitecture, 'predrnnv2');
});

test('复制配置在结构参数不完整时给出警告但保留已有参数', () => {
  const config = readExperimentConfig({
    custom_model_name: 'partial',
    hyperparameters: JSON.stringify({
      model_architecture: 'simvp',
      spatial_hidden_dim: 64,
      epochs: 5,
    }),
  });
  assert.equal(config.epochs, 5);
  assert.equal(config.architectureParamsByModel.simvp.spatial_hidden_dim, 64);
  assert.equal(config.architectureParamsByModel.simvp.temporal_hidden_dim, 128);
  assert.ok(config.warnings.includes('incomplete_structure'));
});

test('复制配置在上传模型不可用时给出警告且不伪造可用模型', () => {
  const config = readExperimentConfig({
    id: 53,
    custom_model_name: 'uploaded run',
    model_source: 'uploaded',
    uploaded_model_id: 'model-a',
    uploaded_model_version: 2,
    hyperparameters: JSON.stringify({ model_source: 'uploaded', custom_model_params: { depth: 3 } }),
  }, {
    uploadedModels: [{ id: 'model-a', version: 1, validation_status: 'valid' }],
  });

  assert.equal(config.modelSource, 'uploaded');
  assert.equal(config.selectedUploadedModelId, '');
  assert.ok(config.warnings.includes('uploaded_model_unavailable'));
});

/** 复制上传模型配置：自定义参数必须原样带回。 */
test('复制上传模型实验时保留 custom_model_params', () => {
  const sourceParams = {
    hidden_size: 128,
    dropout: 0.2,
    nested: {
      depth: 3,
    },
  };

  const config = readExperimentConfig(
    {
      id: 12,
      custom_model_name: 'Uploaded ozone model',
      model_source: 'uploaded',
      uploaded_model_id: 'model-1',
      uploaded_model_version: 4,
      hyperparameters: JSON.stringify({
        model_source: 'uploaded',
        training_dataset: 'openmars_mcd',
        selected_channels: ['U', 'V'],
        window: 8,
        horizon: 3,
        epochs: 40,
        batch_size: 16,
        learning_rate: 0.0003,
        custom_model_params: sourceParams,
      }),
    },
    {
      uploadedModels: [
        {
          id: 'model-1',
          version: 4,
          validation_status: 'valid',
          display_name: 'Uploaded ozone model',
        },
      ],
      channelOrder: ['U', 'V', 'D', 'S', 'T'],
      existingNames: ['Uploaded ozone model'],
    }
  );

  assert.deepEqual(config.customModelParams, sourceParams);
  assert.equal(config.selectedUploadedModelId, 'model-1');
  assert.equal(config.selectedUploadedModelVersion, 4);
  assert.equal(config.warnings.includes('custom_model_params_missing'), false);

  assert.notEqual(config.customModelParams, sourceParams);
  assert.notEqual(config.customModelParams.nested, sourceParams.nested);
});

test('custom_model_params 非普通对象时返回空对象并产生警告', () => {
  const config = readExperimentConfig(
    {
      id: 13,
      model_source: 'uploaded',
      uploaded_model_id: 'model-1',
      uploaded_model_version: 4,
      hyperparameters: JSON.stringify({
        model_source: 'uploaded',
        custom_model_params: [1, 2, 3],
      }),
    },
    {
      uploadedModels: [
        {
          id: 'model-1',
          version: 4,
          validation_status: 'valid',
        },
      ],
      channelOrder: ['U', 'V', 'D', 'S', 'T'],
    }
  );

  assert.deepEqual(config.customModelParams, {});
  assert.equal(config.warnings.includes('custom_model_params_missing'), true);
});

test('官方模型复制配置不会读取 custom_model_params', () => {
  const config = readExperimentConfig(
    {
      id: 14,
      model_source: 'official',
      hyperparameters: JSON.stringify({
        model_source: 'official',
        model_architecture: 'simvp',
        custom_model_params: {
          should_not_be_used: true,
        },
      }),
    },
    {
      channelOrder: ['U', 'V', 'D', 'S', 'T'],
    }
  );

  assert.deepEqual(config.customModelParams, {});
});

test('复制配置始终返回可选链安全的 customModelParams', () => {
  assert.deepEqual(readExperimentConfig(null).customModelParams, {});
  assert.deepEqual(
    readExperimentConfig({ id: 15, model_source: 'uploaded', hyperparameters: '{bad' }).customModelParams,
    {}
  );
  assert.deepEqual(
    readExperimentConfig({
      id: 16,
      model_source: 'uploaded',
      hyperparameters: JSON.stringify({ model_source: 'uploaded' }),
    }).customModelParams,
    {}
  );
  // 空对象是合法的自定义参数：不应产生缺失警告。
  assert.deepEqual(
    readExperimentConfig({
      id: 17,
      model_source: 'uploaded',
      hyperparameters: JSON.stringify({ model_source: 'uploaded', custom_model_params: {} }),
    }).customModelParams,
    {}
  );
  assert.equal(
    readExperimentConfig({
      id: 17,
      model_source: 'uploaded',
      hyperparameters: JSON.stringify({ model_source: 'uploaded', custom_model_params: {} }),
    }).warnings.includes('custom_model_params_missing'),
    false
  );
});

test('深拷贝函数不与原对象共享嵌套引用', () => {
  const source = { a: 1, list: [1, { deep: 2 }], nested: { b: { c: 3 } } };
  const copy = cloneExperimentConfigValue(source);
  assert.deepEqual(copy, source);
  assert.notEqual(copy, source);
  assert.notEqual(copy.list, source.list);
  assert.notEqual(copy.list[1], source.list[1]);
  assert.notEqual(copy.nested.b, source.nested.b);
  copy.nested.b.c = 99;
  assert.equal(source.nested.b.c, 3);
  assert.equal(cloneExperimentConfigValue(null), null);
  assert.equal(cloneExperimentConfigValue(5), 5);
});

test('复制配置在上传模型可用时带回模型 ID 与版本', () => {  const config = readExperimentConfig({
    id: 54,
    custom_model_name: 'uploaded ok',
    model_source: 'uploaded',
    uploaded_model_id: 'model-b',
    uploaded_model_version: 3,
    hyperparameters: JSON.stringify({
      model_source: 'uploaded',
      custom_model_params: { depth: 2 },
      window: 3,
      horizon: 3,
      epochs: 10,
      batch_size: 32,
      learning_rate: 0.001,
    }),
  }, {
    uploadedModels: [{ id: 'model-b', version: 3, validation_status: 'valid', display_name: 'MyNet' }],
  });

  assert.equal(config.selectedUploadedModelId, 'model-b');
  assert.equal(config.selectedUploadedModelVersion, 3);
  assert.equal(config.uploadedModelName, 'MyNet');
  assert.deepEqual(config.warnings, []);
});

test('复制出的实验名称不会与原任务重名', () => {
  assert.equal(buildExperimentCopyName('baseline', []), 'baseline (Copy)');
  assert.equal(buildExperimentCopyName('baseline', ['baseline (Copy)']), 'baseline (Copy) 2');
  assert.equal(buildExperimentCopyName('baseline', ['baseline (Copy)', 'baseline (Copy) 2']), 'baseline (Copy) 3');
  assert.equal(readExperimentConfig({ custom_model_name: 'baseline' }, {
    existingNames: ['baseline (Copy)'],
  }).customModelName, 'baseline (Copy) 2');
});

test('失败任务从 metrics 读取 error、error_code 与恢复建议', () => {
  const cuda = getExperimentFailureMessage({
    status: 'failed',
    metrics: JSON.stringify({ error_code: 'cuda_out_of_memory', error: 'GPU memory is exhausted' }),
  });
  assert.equal(cuda.code, 'cuda_out_of_memory');
  assert.equal(cuda.messageKey, 'experimentCenter.failureCudaOom');
  assert.equal(cuda.suggestionKey, 'experimentCenter.failureCudaOomSuggestion');
  assert.equal(cuda.detail, 'GPU memory is exhausted');
  assert.equal(cuda.stopped, false);

  const artifact = getExperimentFailureMessage({
    status: 'failed',
    metrics: JSON.stringify({ error: 'Invalid model artifact: no valid weight file was produced' }),
  });
  assert.equal(artifact.code, 'invalid_model_artifact');
  assert.notEqual(artifact.suggestionKey, '');
});

test('失败任务没有 error_code 时仍能识别显存错误', () => {
  const message = getExperimentFailureMessage({
    status: 'failed',
    metrics: JSON.stringify({ error: 'RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB' }),
  });
  assert.equal(message.code, 'cuda_out_of_memory');
  assert.equal(message.suggestionKey, 'experimentCenter.failureCudaOomSuggestion');
});

test('用户停止的任务显示为已停止并读取 note', () => {
  const stopped = getExperimentFailureMessage({
    status: 'failed',
    metrics: JSON.stringify({ note: 'Stopped by user' }),
  });
  assert.equal(stopped.code, 'stopped_by_user');
  assert.equal(stopped.stopped, true);
  assert.equal(stopped.messageKey, 'experimentCenter.failureStopped');
  assert.equal(stopped.detail, 'Stopped by user');
});

test('失败原因 JSON 非法或字段异常时安全回退', () => {
  assert.deepEqual(
    Object.keys(getExperimentFailureMessage({ status: 'failed', metrics: '{bad' })).sort(),
    ['code', 'detail', 'errorCode', 'hasReason', 'messageKey', 'stopped', 'suggestionKey'].sort()
  );
  assert.equal(getExperimentFailureMessage({ status: 'failed', metrics: '{bad' }).code, 'training_failed');
  assert.equal(getExperimentFailureMessage({ status: 'completed', metrics: null }).code, '');
  const objectError = getExperimentFailureMessage({
    status: 'failed',
    metrics: JSON.stringify({ error: { message: 'disk full' } }),
  });
  assert.equal(objectError.detail, 'disk full');
  assert.equal(objectError.code, 'training_failed');
});

test('用于预测只对已完成且有有效权重的任务开放', () => {
  assert.equal(canUseTaskForPrediction({ status: 'completed', model_available: true }), true);
  assert.equal(canUseTaskForPrediction({ status: 'completed', model_available: false }), false);
  assert.equal(canUseTaskForPrediction({ status: 'completed' }), false);
  assert.equal(canUseTaskForPrediction({ status: 'failed', model_available: true }), false);
  assert.equal(canUseTaskForPrediction(null), false);
});

test('日志行着色区分错误、警告、指标与普通行', () => {
  assert.equal(getExperimentLogLineTone('Traceback (most recent call last)'), 'error');
  assert.equal(getExperimentLogLineTone('CUDA out of memory, training failed'), 'error');
  assert.equal(getExperimentLogLineTone('UserWarning: something'), 'warning');
  assert.equal(getExperimentLogLineTone('Epoch 3 val_loss=0.21'), 'metric');
  assert.equal(getExperimentLogLineTone('Training started'), 'info');
  assert.equal(getExperimentLogLineTone(''), 'default');
});

test('任务通道优先读取统一脚本的 selected_channels', () => {
  const order = ['U', 'V', 'D', 'S', 'T'];
  assert.deepEqual(
    normalizeTaskChannels({ model_script: 'demo3.py', hyperparameters: '{"selected_channels":["T","U"]}' }, order),
    ['U', 'T']
  );
  assert.deepEqual(normalizeTaskChannels({ model_script: 'demo3-UD.py', hyperparameters: '{}' }, order), ['U', 'D']);
  assert.deepEqual(normalizeTaskChannels({ model_script: 'demo3.py', hyperparameters: '{}' }, order), []);
});

test('指标数值格式化对缺失值和小数给出稳定输出', () => {
  assert.equal(formatExperimentMetricValue(undefined), '--');
  assert.equal(formatExperimentMetricValue(null), '--');
  assert.equal(formatExperimentMetricValue('--'), '--');
  assert.equal(formatExperimentMetricValue(0.03125), '0.0313');
  assert.equal(formatExperimentMetricValue(0.000012), '1.200e-5');
  assert.equal(formatExperimentMetricValue(12345.6789), '12345.68');
  assert.equal(formatExperimentMetricValue('n/a'), 'n/a');
});
