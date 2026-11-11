import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildTrainedModelParameterItems,
  buildTrainingTaskHandoff,
  getCompletedTrainingModelOptions,
  parseTrainingTaskHandoff,
} from './trainedModelSelection.js';

test('keeps only server-confirmed available training tasks as prediction model options', () => {
  const tasks = [
    { id: 1, status: 'completed', model_available: true, output_model_path: 'D:/models/a.pth', custom_model_name: 'Dust run' },
    { id: 2, status: 'running', model_available: false, output_model_path: 'D:/models/b.pth', custom_model_name: 'Running run' },
    { id: 3, status: 'completed', model_available: false, output_model_path: '', custom_model_name: 'No weights' },
    { id: 4, status: 'completed', model_available: true, output_model_path: 'D:/models/c.pth', custom_model_name: '' },
    { id: 5, status: 'completed', model_available: false, output_model_path: 'D:/models/missing.pth', custom_model_name: 'Historical broken task' },
    { id: 6, status: 'completed', output_model_path: 'D:/models/legacy.pth', custom_model_name: 'No availability field' },
  ];

  assert.deepEqual(getCompletedTrainingModelOptions(tasks), [
    { id: 1, label: 'Dust run', task: tasks[0] },
    { id: 4, label: 'Task #4', task: tasks[3] },
  ]);
});

test('builds and parses training task handoff payloads', () => {
  const payload = buildTrainingTaskHandoff({
    id: 7,
    custom_model_name: 'MY27 UVDST',
  }, 'user:101');

  assert.deepEqual(parseTrainingTaskHandoff(JSON.stringify(payload), 'user:101'), {
    taskId: 7,
    modelName: 'MY27 UVDST',
    scope: 'user:101',
  });
  assert.equal(parseTrainingTaskHandoff(JSON.stringify(payload), 'user:202'), null);
  assert.equal(parseTrainingTaskHandoff(JSON.stringify(payload), 'anonymous'), null);
});

test('carries the selected environment target and complete inputs through a training handoff', () => {
  const payload = buildTrainingTaskHandoff({
    id: 17,
    dataset_id: 'earth_merra2_3hourly_v1',
    hyperparameters: JSON.stringify({ target_variable: 'T2M', input_variables: ['TO3', 'SWGDN'] }),
  }, 'user:101');
  assert.deepEqual(parseTrainingTaskHandoff(JSON.stringify(payload), 'user:101'), {
    taskId: 17,
    modelName: 'Task #17',
    scope: 'user:101',
    targetVariable: 'T2M',
    targetUnit: 'K',
    inputVariables: ['TO3', 'SWGDN'],
  });
});

test('ignores invalid training task handoff payloads', () => {
  assert.equal(parseTrainingTaskHandoff(''), null);
  assert.equal(parseTrainingTaskHandoff('not json'), null);
  assert.equal(parseTrainingTaskHandoff(JSON.stringify({ taskId: 'abc' })), null);
  assert.equal(buildTrainingTaskHandoff({ id: 7 }, null), null);
});

test('builds display parameters from selected trained model task hyperparameters', () => {
  const items = buildTrainedModelParameterItems({
    id: 8,
    custom_model_name: 'MY27 UVDST',
    hyperparameters: JSON.stringify({
      model_source: 'official',
      model_architecture: 'convlstm',
      selected_channels: ['U', 'V', 'D'],
      use_sphere: true,
      window: 4,
      horizon: 3,
      epochs: 20,
      batch_size: 16,
      learning_rate: 0.0005,
      early_stopping_patience: 5,
      seed: 11,
      _data_source: 'personal',
    }),
  }, { isZh: false });

  assert.deepEqual(items.map((item) => [item.label, item.value]), [
    ['Model name', 'MY27 UVDST'],
    ['Task ID', '#8'],
    ['Source', 'Server-managed data'],
    ['Model type', 'Official'],
    ['Architecture', 'ConvLSTM'],
    ['SPHERE', 'On'],
    ['Prediction target', 'O3'],
    ['Input channels', 'O3 / U / V / D'],
    ['Window', '4'],
    ['Horizon', '3'],
    ['Epochs', '20'],
    ['Batch', '16'],
    ['Learning rate', '0.0005'],
    ['Early stopping', '5'],
    ['Seed', '11'],
  ]);
});
