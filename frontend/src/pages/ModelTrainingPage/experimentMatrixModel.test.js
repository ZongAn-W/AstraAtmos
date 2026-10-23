import test from 'node:test';
import assert from 'node:assert/strict';
import {
  filterMatrixTasks,
  formatMatrixValue,
  getMatrixPropertyDefinitions,
  getMatrixStorageKey,
  getMatrixValue,
  MATRIX_DEFAULT_COLUMNS,
  restoreMatrixColumnPreferences,
  sortMatrixTasks,
} from './experimentMatrixModel.js';

const tasks = [
  {
    id: 1, custom_model_name: 'Zero run', status: 'completed', progress: 0, start_time: '2026-01-02T00:00:00Z',
    tags: [{ id: 5, name: 'baseline' }], dataset_id: 'openmars_mcd',
    hyperparameters: JSON.stringify({ model_architecture: 'simvp', epochs: 0, use_sphere: false, custom_model_params: { width: 0 } }),
    metrics: JSON.stringify({ rmse: 0 }),
  },
  {
    id: 2, custom_model_name: 'Later run', status: 'running', progress: 12, start_time: '2026-01-03T00:00:00Z',
    tags: [], dataset_id: 'earth_merra2_daily_v2',
    hyperparameters: JSON.stringify({ model_architecture: 'dlinear', learning_rate: 0.01 }),
    metrics: '{}',
  },
];

test('matrix values preserve zero and false and use saved task configuration', () => {
  assert.equal(getMatrixValue(tasks[0], 'training:epochs'), 0);
  assert.equal(getMatrixValue(tasks[0], 'training:use_sphere'), false);
  assert.equal(getMatrixValue(tasks[0], 'custom:width'), 0);
  assert.equal(getMatrixValue(tasks[0], 'architecture'), 'SimVP');
  assert.equal(formatMatrixValue(false, { type: 'text' }, tasks[0]), '否');
});

test('matrix definitions add custom attributes without exposing internal keys', () => {
  const definitions = getMatrixPropertyDefinitions([{ ...tasks[0], hyperparameters: JSON.stringify({ custom_model_params: { width: 2 }, _uploaded_model_id: 'secret', source_code_hash: 'secret' }) }]);
  assert.ok(definitions.some((item) => item.key === 'custom:width'));
  assert.ok(!definitions.some((item) => item.key.includes('_uploaded_model_id') || item.key.includes('source_code_hash')));
});

test('matrix keeps same-name custom and top-level parameters separate', () => {
  const definitions = getMatrixPropertyDefinitions([{
    ...tasks[0],
    hyperparameters: JSON.stringify({
      model_architecture: 'simvp',
      width: 3,
      custom_model_params: { width: 7 },
    }),
  }]);
  assert.ok(definitions.some((item) => item.key === 'custom:width'));
  assert.ok(definitions.some((item) => item.key === 'config:width'));
  const task = { hyperparameters: JSON.stringify({ width: 3, custom_model_params: { width: 7 } }) };
  assert.equal(getMatrixValue(task, 'custom:width'), 7);
  assert.equal(getMatrixValue(task, 'config:width'), 3);
});

test('matrix preference restore filters removed columns and appends newly available defaults', () => {
  const definitions = getMatrixPropertyDefinitions(tasks);
  const restored = restoreMatrixColumnPreferences(JSON.stringify({ columns: ['model_name', 'removed', 'tags'] }), definitions);
  assert.deepEqual(restored.slice(0, 2), ['model_name', 'tags']);
  assert.ok(MATRIX_DEFAULT_COLUMNS.every((key) => restored.includes(key)));
  assert.deepEqual(MATRIX_DEFAULT_COLUMNS, ['model_name', 'tags', 'architecture', 'dataset', 'status', 'progress', 'start_time']);
});

test('matrix column preferences are scoped by account', () => {
  assert.notEqual(getMatrixStorageKey(101), getMatrixStorageKey(202));
  assert.equal(getMatrixStorageKey(), 'astraatmos_experiment_matrix_columns:guest');
});

test('matrix filtering and sorting keep missing values at the end', () => {
  assert.deepEqual(filterMatrixTasks(tasks, { search: 'zero' }).map((task) => task.id), [1]);
  assert.deepEqual(filterMatrixTasks(tasks, { status: 'running' }).map((task) => task.id), [2]);
  const definitions = getMatrixPropertyDefinitions(tasks);
  const sorted = sortMatrixTasks(tasks, 'metric:rmse', 'asc', definitions);
  assert.deepEqual(sorted.map((task) => task.id), [1, 2]);
});

test('matrix sorts saved numeric extensions by value and tags by name', () => {
  const definitions = getMatrixPropertyDefinitions(tasks);
  const numericTasks = [
    { id: 1, hyperparameters: JSON.stringify({ custom_model_params: { width: 2 } }) },
    { id: 2, hyperparameters: JSON.stringify({ custom_model_params: { width: 10 } }) },
  ];
  assert.deepEqual(sortMatrixTasks(numericTasks, 'custom:width', 'asc', definitions).map((task) => task.id), [1, 2]);
  const tagTasks = [
    { id: 1, tags: [{ id: 1, name: 'zeta' }] },
    { id: 2, tags: [{ id: 2, name: 'alpha' }] },
    { id: 3, tags: [] },
  ];
  assert.deepEqual(sortMatrixTasks(tagTasks, 'tags', 'asc', definitions).map((task) => task.id), [2, 1, 3]);
});
