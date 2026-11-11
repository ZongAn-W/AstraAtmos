import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildCompareModelSummary,
  buildErrorHistogramTraces,
  buildCompareParameterRows,
  compareResultMatchesTasks,
  buildPfiMatrix,
  buildStepCurveTraces,
  getCompareSelectionState,
  resolveEarthCompareMetricUnit,
  sortCompareItems,
} from './compareTrainingModelsData.js';

test('comparison selection sorts and deduplicates valid task ids', () => {
  assert.deepEqual(getCompareSelectionState([23, '12', 23, -1, 'bad', 18]), {
    ids: [12, 18, 23],
    canCompare: true,
    count: 3,
  });
});

test('builds model summary text from a completed training task', () => {
  const task = {
    id: 12,
    custom_model_name: 'MY27 UVD',
    model_source: 'official',
    hyperparameters: JSON.stringify({
      model_architecture: 'convlstm',
      selected_channels: ['U', 'V', 'D'],
      window: 4,
      horizon: 3,
      _data_source: 'personal',
    }),
  };

  assert.deepEqual(buildCompareModelSummary(task), {
    taskId: 12,
    modelName: 'MY27 UVD',
    modelSource: 'official',
    architecture: 'convlstm',
    selectedChannels: ['O3', 'U', 'V', 'D'],
    inputChannelText: 'O3 / U / V / D',
    targetText: 'O3 (um-atm)',
    window: 4,
    horizon: 3,
    dataSource: 'default',
  });
});

test('model summary preserves top-level target and complete inputs when hyperparameters omit them', () => {
  const summary = buildCompareModelSummary({
    id: 44,
    target_variable: 'T',
    target_unit: 'K',
    input_variables: ['O3', 'U', 'S'],
    hyperparameters: JSON.stringify({ selected_channels: ['U', 'S'] }),
  });
  assert.equal(summary.targetText, 'T (K)');
  assert.deepEqual(summary.selectedChannels, ['O3', 'U', 'S']);
  assert.equal(summary.inputChannelText, 'O3 / U / S');
});

test('sorts compare items by rmse ascending by default', () => {
  const items = [
    { task_id: 1, metrics: { overall: { rmse: 4.2, r2: 0.5 } } },
    { task_id: 2, metrics: { overall: { rmse: 2.1, r2: 0.7 } } },
    { task_id: 3, metrics: { overall: { rmse: 3.0, r2: 0.8 } } },
  ];

  assert.deepEqual(sortCompareItems(items).map((item) => item.task_id), [2, 3, 1]);
});

test('sorts compare items by r2 descending when requested', () => {
  const items = [
    { task_id: 1, metrics: { overall: { rmse: 4.2, r2: 0.5 } } },
    { task_id: 2, metrics: { overall: { rmse: 2.1, r2: 0.7 } } },
    { task_id: 3, metrics: { overall: { rmse: 3.0, r2: 0.8 } } },
  ];

  assert.deepEqual(sortCompareItems(items, { metric: 'r2', direction: 'desc' }).map((item) => item.task_id), [3, 2, 1]);
});

test('builds parameter matrix rows from compare payload items', () => {
  const rows = buildCompareParameterRows([
    {
      task_id: 12,
      model_name: 'MY27 UVD',
      model_source: 'official',
      architecture: 'convlstm',
      selected_channels: ['U', 'V', 'D'],
      hyperparameters: {
        use_sphere: true,
        window: 4,
        horizon: 3,
        epochs: 20,
        batch_size: 16,
        learning_rate: 0.0005,
        early_stopping_patience: 5,
        seed: 11,
        _data_source: 'personal',
      },
    },
  ]);

  assert.deepEqual(rows, [
    {
      taskId: 12,
      modelName: 'MY27 UVD',
      modelSource: 'official',
      architecture: 'convlstm',
      target: 'O3 (um-atm)',
      selectedChannels: 'O3 / U / V / D',
      useSphere: 'On',
      window: '4',
      horizon: '3',
      epochs: '20',
      batchSize: '16',
      learningRate: '0.0005',
      earlyStoppingPatience: '5',
      seed: '11',
      dataSource: 'default',
    },
  ]);
});

test('parameter rows and comparison result guard preserve a non-ozone task identity', () => {
  const task = { id: 7, target_variable: 'T', target_unit: 'K', input_variables: ['O3', 'U'] };
  const peer = { id: 8, target_variable: 'T', target_unit: 'K', input_variables: ['O3', 'U'] };
  const items = [task, peer].map((item) => ({
    task_id: item.id,
    target_variable: item.target_variable,
    target_unit: item.target_unit,
    input_variables: item.input_variables,
    hyperparameters: { selected_channels: ['U'] },
  }));
  const rows = buildCompareParameterRows(items, 'mars');
  assert.deepEqual(rows.map(({ target, selectedChannels }) => [target, selectedChannels]), [
    ['T (K)', 'O3 / U'],
    ['T (K)', 'O3 / U'],
  ]);
  assert.equal(compareResultMatchesTasks({ items }, [task, peer], 'mars'), true);
  assert.equal(compareResultMatchesTasks({ items: [{ ...items[0], input_variables: ['O3', 'V'] }, items[1]] }, [task, peer], 'mars'), false);
  assert.equal(compareResultMatchesTasks({ items: items.map(({ task_id }) => ({ task_id })) }, [task, peer], 'mars'), false);
});

test('Earth comparison guard rejects missing or mismatched input identities', () => {
  const tasks = [17, 18].map((id) => ({
    id,
    target_variable: 'T2M',
    target_unit: 'K',
    input_variables: ['TO3', 'U10M'],
  }));
  const items = tasks.map((task) => ({
    task_id: task.id,
    planet: 'earth',
    target_variable: task.target_variable,
    target_unit: task.target_unit,
    input_variables: task.input_variables,
  }));
  assert.equal(compareResultMatchesTasks({ items }, tasks, 'earth'), true);
  assert.equal(compareResultMatchesTasks({ items: [{ ...items[0], input_variables: undefined }, items[1]] }, tasks, 'earth'), false);
  assert.equal(compareResultMatchesTasks({ items: [{ ...items[0], target_unit: undefined }, items[1]] }, tasks, 'earth'), false);
  assert.equal(compareResultMatchesTasks({ items: [{ ...items[0], input_variables: ['TO3', 'SWGDN'] }, items[1]] }, tasks, 'earth'), false);
});

test('Earth comparison metric units follow the forecast target when payload units are absent', () => {
  assert.equal(resolveEarthCompareMetricUnit('rmse', 'K'), 'K');
  assert.equal(resolveEarthCompareMetricUnit('mae', 'W m-2'), 'W m-2');
  assert.equal(resolveEarthCompareMetricUnit('mse', 'K'), 'K²');
  assert.equal(resolveEarthCompareMetricUnit('mse', 'W m-2'), '(W m-2)²');
  assert.equal(resolveEarthCompareMetricUnit('mse', '1', null, true), '无量纲');
  assert.equal(resolveEarthCompareMetricUnit('rmse', 'K', { rmse: 'kelvin' }), 'kelvin');
  assert.equal(resolveEarthCompareMetricUnit('r2', 'K'), 'dimensionless');
});

test('builds per-step metric traces from compare payload items', () => {
  const traces = buildStepCurveTraces([
    {
      model_name: 'A',
      task_id: 1,
      metrics: {
        per_step: [
          { step: 1, rmse: 1.2, r2: 0.7 },
          { step: 2, rmse: 1.8, r2: 0.6 },
        ],
      },
    },
    {
      model_name: 'B',
      task_id: 2,
      metrics: {
        per_step: [
          { step: 1, rmse: 0.9, r2: 0.8 },
        ],
      },
    },
  ], 'rmse');

  assert.deepEqual(traces.map((trace) => [trace.name, trace.x, trace.y]), [
    ['A', [1, 2], [1.2, 1.8]],
    ['B', [1], [0.9]],
  ]);
});

test('builds comparable error histogram traces', () => {
  const traces = buildErrorHistogramTraces([
    {
      task_id: 1,
      model_name: 'A',
      distribution: {
        hist_errors: { bin_edges: [-1, 0, 1], counts: [2, 3] },
      },
    },
  ]);

  assert.deepEqual(traces, [
    {
      taskId: 1,
      name: 'A',
      centers: [-0.5, 0.5],
      counts: [2, 3],
    },
  ]);
});

test('builds PFI matrix rows and feature columns', () => {
  const matrix = buildPfiMatrix([
    {
      task_id: 1,
      model_name: 'A',
      items: [
        { name: 'Ozone', importance: 0.5 },
        { name: 'Temperature', importance: 0.2 },
      ],
    },
    {
      task_id: 2,
      model_name: 'B',
      items: [
        { name: 'Ozone', importance: 0.4 },
        { name: 'Dust_Optical_Depth', importance: 0.3 },
      ],
    },
  ]);

  assert.deepEqual(matrix.features, ['Ozone', 'Temperature', 'Dust_Optical_Depth']);
  assert.deepEqual(matrix.rows, [
    { taskId: 1, modelName: 'A', values: { Ozone: 0.5, Temperature: 0.2, Dust_Optical_Depth: 0 } },
    { taskId: 2, modelName: 'B', values: { Ozone: 0.4, Temperature: 0, Dust_Optical_Depth: 0.3 } },
  ]);
});
