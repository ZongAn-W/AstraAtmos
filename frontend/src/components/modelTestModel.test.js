import test from 'node:test';
import assert from 'node:assert/strict';
import { marsModelTestResponseMatchesTask, modelTestMetricUnit, modelTestScatterRange } from './modelTestModel.js';

test('model test scatter range includes signed physical wind values', () => {
  const range = modelTestScatterRange([-4, -1], [-3, 2]);
  assert.ok(range[0] < -4);
  assert.ok(range[1] > 2);
  assert.equal(range[0] <= range[1], true);
});

test('model test scatter keeps the zero origin for non-negative ozone or temperature values', () => {
  const range = modelTestScatterRange([200, 240], [210, 230]);
  assert.equal(range[0], 0);
  assert.ok(range[1] > 240);
});

test('large paired test samples do not overflow the argument stack', () => {
  const range = modelTestScatterRange(Array(200000).fill(-3), Array(200000).fill(-1));
  assert.ok(range[0] < -3);
  assert.ok(range[1] > -1);
});

test('model test metric labels carry the selected target unit', () => {
  assert.equal(modelTestMetricUnit('rmse', 'm/s'), 'm/s');
  assert.equal(modelTestMetricUnit('mse', 'm/s'), '(m/s)²');
  assert.equal(modelTestMetricUnit('mse', 'K'), 'K²');
  assert.equal(modelTestMetricUnit('mse', 'um-atm'), '(um-atm)²');
  assert.equal(modelTestMetricUnit('mse', '1'), '1');
  assert.equal(modelTestMetricUnit('r2', 'K'), '');
});

test('Mars model test responses must identify non-ozone targets and complete inputs', () => {
  const task = { target_variable: 'T', target_unit: 'K', input_variables: ['O3', 'U'] };
  assert.equal(marsModelTestResponseMatchesTask({ target_variable: 'T', target_unit: 'K', input_variables: ['O3', 'U'] }, task), true);
  assert.equal(marsModelTestResponseMatchesTask({ target_variable: 'O3', target_unit: 'um-atm', input_variables: ['O3', 'U'] }, task), false);
  assert.equal(marsModelTestResponseMatchesTask({ target_variable: 'T', target_unit: 'K', input_variables: ['O3', 'V'] }, task), false);
  assert.equal(marsModelTestResponseMatchesTask({ y_true: [1], y_pred: [1] }, task), false);
  assert.equal(marsModelTestResponseMatchesTask({ y_true: [1], y_pred: [1] }, {}), true);
  assert.equal(marsModelTestResponseMatchesTask({ target_variable: 'O3', target_unit: 'um-atm' }, {}), false);
});
