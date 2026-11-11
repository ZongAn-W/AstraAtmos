import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEFAULT_TRAINING_DEFAULTS, getTrainingDefaultUpdates, normalizeTrainingDefaults } from '../../utils/trainingDefaults.js';

const pageSource = readFileSync(new URL('../ModelTrainingPage.jsx', import.meta.url), 'utf8');

test('training defaults include editable strategy settings with the supported baseline split', () => {
  assert.deepEqual(DEFAULT_TRAINING_DEFAULTS, {
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    window: 3,
    horizon: 3,
    trainRatio: 0.7,
    validationRatio: 0.2,
    testRatio: 0.1,
    earlyStoppingPatience: 0,
    seed: 11,
    transferEnabled: false,
    transferFreezeMode: 'none',
    finetuneLearningRate: 0.0001,
  });
});

test('normalizes valid saved training parameter and strategy defaults', () => {
  assert.deepEqual(normalizeTrainingDefaults({
    epochs: '24',
    batchSize: 8,
    learningRate: '0.0005',
    window: 6,
    horizon: 2,
    trainRatio: 0.6,
    validationRatio: 0.25,
    testRatio: 0.15,
    earlyStoppingPatience: 5,
    seed: 0,
    transferEnabled: true,
    transferFreezeMode: 'backbone',
    finetuneLearningRate: 0.00002,
  }), {
    epochs: 24,
    batchSize: 8,
    learningRate: 0.0005,
    window: 6,
    horizon: 2,
    trainRatio: 0.6,
    validationRatio: 0.25,
    testRatio: 0.15,
    earlyStoppingPatience: 5,
    seed: 0,
    transferEnabled: true,
    transferFreezeMode: 'backbone',
    finetuneLearningRate: 0.00002,
  });
});

test('preserves an explicitly disabled validation split', () => {
  assert.deepEqual(normalizeTrainingDefaults({
    trainRatio: 0.8,
    validationRatio: 0,
    testRatio: 0.2,
  }).validationRatio, 0);
});

test('falls back to supported parameter defaults when saved values exceed form/API bounds', () => {
  assert.deepEqual(normalizeTrainingDefaults({
    epochs: 1001,
    batchSize: 65,
    learningRate: 1.01,
    window: 241,
    horizon: 0,
    earlyStoppingPatience: 201,
    seed: 2147483648,
    finetuneLearningRate: 1.1,
  }), DEFAULT_TRAINING_DEFAULTS);

  assert.deepEqual(normalizeTrainingDefaults({
    epochs: 1000,
    batchSize: 64,
    learningRate: 1,
    window: 240,
    horizon: 240,
    earlyStoppingPatience: 200,
    seed: 2147483647,
    finetuneLearningRate: 1,
  }), {
    ...DEFAULT_TRAINING_DEFAULTS,
    epochs: 1000,
    batchSize: 64,
    learningRate: 1,
    window: 240,
    horizon: 240,
    earlyStoppingPatience: 200,
    seed: 2147483647,
    finetuneLearningRate: 1,
  });
});

test('falls back to the complete baseline split when saved ratios do not sum to one', () => {
  const normalized = normalizeTrainingDefaults({
    trainRatio: 0.6,
    validationRatio: 0.2,
    testRatio: 0.3,
    transferEnabled: true,
  });

  assert.equal(normalized.trainRatio, 0.7);
  assert.equal(normalized.validationRatio, 0.2);
  assert.equal(normalized.testRatio, 0.1);
  assert.equal(normalized.transferEnabled, true);
});

test('changed settings update untouched draft fields while retaining explicit edits', () => {
  const defaults = { ...DEFAULT_TRAINING_DEFAULTS, epochs: '24', batchSize: '8', learningRate: '0.0005' };
  const updates = getTrainingDefaultUpdates(defaults, { editedFields: new Set(['epochs']) });
  assert.equal('epochs' in updates, false);
  assert.equal(updates.batchSize, 8);
  assert.equal(updates.learningRate, 0.0005);
  assert.deepEqual(defaults.epochs, '24');
});

test('copied configurations keep all loaded values until a new experiment resets edits', () => {
  const defaults = { ...DEFAULT_TRAINING_DEFAULTS, epochs: 24, batchSize: 8 };
  assert.deepEqual(getTrainingDefaultUpdates(defaults, { editedFields: Object.keys(DEFAULT_TRAINING_DEFAULTS) }), {});
  assert.deepEqual(getTrainingDefaultUpdates(defaults), defaults);
});

test('clearing a setting or typing an invalid intermediate number does not reset the draft', () => {
  const updates = getTrainingDefaultUpdates({
    ...DEFAULT_TRAINING_DEFAULTS,
    epochs: '', batchSize: '65', learningRate: 0, seed: null,
    transferFreezeMode: 'unknown', transferEnabled: 'true',
  });
  for (const key of ['epochs', 'batchSize', 'learningRate', 'seed', 'transferFreezeMode', 'transferEnabled']) {
    assert.equal(key in updates, false, key);
  }
  assert.equal(updates.window, 3);
  assert.equal(getTrainingDefaultUpdates({ ...DEFAULT_TRAINING_DEFAULTS, seed: '0' }).seed, 0);
});

test('split updates wait for a complete valid total and preserve a customized split as a whole', () => {
  const valid = { ...DEFAULT_TRAINING_DEFAULTS, trainRatio: '0.6', validationRatio: '0.25', testRatio: '0.15' };
  for (const [defaults, options] of [
    [{ ...valid, testRatio: 0.1 }, {}],
    [{ ...valid, validationRatio: '' }, {}],
    [valid, { editedFields: ['testRatio'] }],
  ]) {
    const updates = getTrainingDefaultUpdates(defaults, options);
    assert.equal('trainRatio' in updates, false);
    assert.equal('validationRatio' in updates, false);
    assert.equal('testRatio' in updates, false);
  }
  const updates = getTrainingDefaultUpdates(valid);
  assert.equal(updates.trainRatio, 0.6);
  assert.equal(updates.validationRatio, 0.25);
  assert.equal(updates.testRatio, 0.15);
});

test('dataset and transfer constraints exclude fixed fields while common defaults still update', () => {
  const updates = getTrainingDefaultUpdates({ ...DEFAULT_TRAINING_DEFAULTS, epochs: 24, window: 6, transferEnabled: true }, {
    excludedFields: ['window', 'horizon', 'trainRatio', 'validationRatio', 'testRatio', 'transferEnabled'],
  });
  assert.equal(updates.epochs, 24);
  for (const key of ['window', 'horizon', 'trainRatio', 'validationRatio', 'testRatio', 'transferEnabled']) {
    assert.equal(key in updates, false, key);
  }
});

test('new experiments apply current defaults while copied configurations keep task values', () => {
  const createStart = pageSource.indexOf('const handleCreateExperiment = () => {');
  const createEnd = pageSource.indexOf('const handleSelectTask =', createStart);
  const createHandler = pageSource.slice(createStart, createEnd);
  assert.match(createHandler, /normalizeTrainingDefaults\(settings\.trainingDefaults\)/);
  [
    'setEpochs(defaults.epochs)',
    'setBatchSize(modelSource === EARTH_MODEL_SOURCE_UPLOADED && selectedUploadedModelIsFullGrid ? 1 : defaults.batchSize)',
    'setLearningRate(defaults.learningRate)',
    'setTrainRatio(splitDefaults.trainRatio)',
    'setValidationRatio(splitDefaults.validationRatio)',
    'setTestRatio(splitDefaults.testRatio)',
    'setEarlyStoppingPatience(defaults.earlyStoppingPatience)',
    'setSeed(defaults.seed)',
    'setTransferEnabled(earthMode ? false : defaults.transferEnabled)',
    'setTransferFreezeMode(defaults.transferFreezeMode)',
    'setFinetuneLearningRate(defaults.finetuneLearningRate)',
    'setWindow(defaults.window)',
    'setHorizon(defaults.horizon)',
  ].forEach((statement) => assert.ok(createHandler.includes(statement), `missing ${statement}`));

  const copyStart = pageSource.indexOf('const handleCopyConfig = (task) => {');
  const copyEnd = pageSource.indexOf('const presetStatusText =', copyStart);
  const copyHandler = pageSource.slice(copyStart, copyEnd);
  assert.doesNotMatch(copyHandler, /normalizeTrainingDefaults/);
  ['setEpochs(config.epochs)', 'setBatchSize(config.batchSize)', 'setLearningRate(config.learningRate)']
    .forEach((statement) => assert.ok(copyHandler.includes(statement), `missing ${statement}`));
});

test('training validates split totals before either Earth or Mars submits a task', () => {
  const start = pageSource.indexOf('const handleStartTraining = async () => {');
  const end = pageSource.indexOf('const handleStopTask =', start);
  const handler = pageSource.slice(start, end);
  const splitValidationIndex = handler.indexOf('if (!splitRatios.valid)');
  const earthPathIndex = handler.indexOf('if (earthMode) {');
  const marsSubmitIndex = handler.indexOf('buildTrainingHyperparameters({');
  assert.ok(splitValidationIndex >= 0);
  assert.ok(earthPathIndex > splitValidationIndex);
  assert.ok(marsSubmitIndex > splitValidationIndex);
});
