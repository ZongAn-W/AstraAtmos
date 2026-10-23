import test from 'node:test';
import assert from 'node:assert/strict';

import {
  EARTH_CHANNEL_ORDER,
  EARTH_DATASET_ID,
  EARTH_HORIZON,
  EARTH_3HOURLY_DATASET_ID,
  EARTH_MODEL_ARCHITECTURE,
  EARTH_MODEL_SOURCE_OFFICIAL,
  EARTH_OPTIONAL_CHANNELS,
  EARTH_WINDOW,
  applyEarthTrainingDefaults,
  buildEarthTrainingHyperparameters,
  captureTrainingDraft,
  describeEarthInputUnits,
  describeEarthSplitSamples,
  getEarthTrainingReadiness,
  getEarthTrainingProfile,
  classifyEarthTrainingError,
  getEarthUploadedSelectionBlocker,
  isEarthTrainingDataset,
  normalizeEarthSelectedChannels,
  readEarthDatasetAvailability,
  readEarthUploadedModelCompatibility,
  resolveEarthTrainingRestore,
  resolveMarsTrainingRestore,
} from './earthTrainingConfig.js';
import { sanitizeTrainingDataset } from './trainingParamSanitizers.js';

const AVAILABLE_DESCRIPTOR = {
  dataset_id: EARTH_DATASET_ID,
  display_name: 'MERRA-2 three-hourly global 0.75 degree v1',
  planet: 'earth',
  dataset_version: 'v1',
  dataset_fingerprint: 'f'.repeat(64),
  availability: 'available',
  availability_reason: null,
  capabilities: { metadata: true, training: true, web_overview: true, trained_prediction: true },
  splits: {
    train: { start: '2020-01-01', end: '2020-12-31', days: 366 },
    validation: { start: '2021-01-01', end: '2021-06-30', days: 181 },
    test: { start: '2021-07-01', end: '2021-12-31', days: 184 },
  },
  training_profile: { window: EARTH_WINDOW, horizon: EARTH_HORIZON, grid_shape: [240, 480] },
};

test('Earth payload fixes the dataset, window/horizon and channel order', () => {
  const payload = buildEarthTrainingHyperparameters({ selectedChannels: ['SWGDN', 'U10M'] });
  assert.equal(payload.training_dataset, 'earth_merra2_3hourly_v1');
  assert.equal(payload.model_architecture, 'dlinear');
  assert.equal(payload.model_source, 'official');
  assert.equal(payload.window, 56);
  assert.equal(payload.horizon, 24);
  assert.equal(payload.use_sphere, false);
  assert.equal(payload.transfer_learning, false);
  // 请求顺序不影响模型通道顺序：规范顺序固定为 U10M、V10M、T2M、SWGDN。
  assert.deepEqual(payload.selected_channels, ['U10M', 'SWGDN']);
});

test('three-hourly Earth profile uses the 240x480 56-to-24 contract', () => {
  const profile = getEarthTrainingProfile(EARTH_3HOURLY_DATASET_ID);
  assert.deepEqual(profile.gridShape, [240, 480]);
  assert.equal(profile.frequencyHours, 3);
  assert.equal(profile.stepUnit, 'hour');
  assert.equal(profile.step, 3);
  assert.equal(profile.window, 56);
  assert.equal(profile.horizon, 24);
  assert.deepEqual(profile.modelSources, ['official', 'uploaded']);
  const payload = buildEarthTrainingHyperparameters({ datasetId: EARTH_3HOURLY_DATASET_ID });
  assert.equal(payload.training_dataset, EARTH_3HOURLY_DATASET_ID);
  assert.equal(payload.window, 56);
  assert.equal(payload.horizon, 24);
  assert.equal(payload.model_source, 'official');
  const unsupportedUpload = buildEarthTrainingHyperparameters({
    datasetId: EARTH_3HOURLY_DATASET_ID,
    modelSource: 'uploaded',
  });
  assert.equal(unsupportedUpload.model_source, 'uploaded');
});

test('Earth training errors keep dataset, configuration and checkpoint categories distinct', () => {
  assert.equal(classifyEarthTrainingError('dataset_unavailable'), 'dataset_unavailable');
  assert.equal(classifyEarthTrainingError('dataset_training_configuration_not_supported'), 'configuration_unsupported');
  assert.equal(classifyEarthTrainingError('checkpoint_grid_incompatible'), 'checkpoint_incompatible');
});

test('uploaded compatibility reads contract axes when the server nests them', () => {
  const verdict = readEarthUploadedModelCompatibility({
    compatible: true,
    datasets: {
      earth_merra2_3hourly_v1: {
        dataset: 'earth_merra2_3hourly_v1', frequency_hours: 3,
        grid_shape: [240, 480], window: 56, horizon: 24,
      },
    },
  });
  assert.equal(verdict.dataset, EARTH_3HOURLY_DATASET_ID);
  assert.equal(verdict.frequencyHours, 3);
  assert.deepEqual(verdict.gridShape, [240, 480]);
  assert.equal(verdict.window, 56);
  assert.equal(verdict.horizon, 24);
});

test('Earth payload preserves the ozone-only selection instead of restoring defaults', () => {
  const payload = buildEarthTrainingHyperparameters({ selectedChannels: [] });
  assert.deepEqual(payload.selected_channels, []);
});

test('Earth payload accepts settings-provided custom windows', () => {
  const payload = buildEarthTrainingHyperparameters({ datasetId: EARTH_3HOURLY_DATASET_ID, windowValue: 16, horizon: 8 });
  assert.equal(payload.window, 16);
  assert.equal(payload.horizon, 8);
});

test('Earth payload sends requested split ratios with the configured windows', () => {
  const payload = buildEarthTrainingHyperparameters({
    trainRatio: 0.6,
    validationRatio: 0.25,
    testRatio: 0.15,
  });

  assert.equal(payload.train_ratio, 0.6);
  assert.equal(payload.validation_ratio, 0.25);
  assert.equal(payload.test_ratio, 0.15);
  assert.equal(payload.window, 56);
  assert.equal(payload.horizon, 24);
});

test('Earth payload never carries server identity or Mars-only fields', () => {
  const payload = buildEarthTrainingHyperparameters({ selectedChannels: EARTH_OPTIONAL_CHANNELS });
  [
    'dataset_version',
    'dataset_fingerprint',
    'dataset_identity_status',
    'dataset_snapshot',
    'transfer_source_task_id',
    'transfer_weight_id',
    'freeze_mode',
    '_data_source',
    '_uploaded_model_path',
    'custom_model_params',
  ].forEach((key) => {
    assert.equal(Object.hasOwn(payload, key), false, `${key} must not be sent`);
  });
});

test('Earth payload clamps only to the documented Earth ranges', () => {
  const payload = buildEarthTrainingHyperparameters({
    epochs: 5000,
    batchSize: 500,
    learningRate: 12,
    seed: -3,
    earlyStoppingPatience: 9999,
    linearHiddenLayers: 9,
  });
  assert.equal(payload.epochs, 1000);
  assert.equal(payload.batch_size, 64);
  assert.equal(payload.learning_rate, 1);
  assert.equal(payload.seed, 0);
  assert.equal(payload.early_stopping_patience, 200);
  assert.equal(payload.linear_hidden_layers, 4);
});

test('Earth channel normalisation drops unknown and duplicate channels', () => {
  assert.deepEqual(normalizeEarthSelectedChannels(['U10M', 'U10M', 'NOPE', 't2m']), ['U10M', 'T2M']);
  assert.deepEqual(normalizeEarthSelectedChannels('U10M'), []);
  assert.deepEqual(normalizeEarthSelectedChannels(undefined), []);
});

test('Earth channel order keeps TO3 first', () => {
  assert.deepEqual(EARTH_CHANNEL_ORDER, ['TO3', 'U10M', 'V10M', 'T2M', 'SWGDN']);
});

test('dataset availability separates wiring from data state', () => {
  const available = readEarthDatasetAvailability(AVAILABLE_DESCRIPTOR);
  assert.equal(available.selectable, true);
  assert.equal(available.datasetVersion, 'v1');
  assert.equal(available.fingerprint, 'f'.repeat(64));

  const missing = readEarthDatasetAvailability({ ...AVAILABLE_DESCRIPTOR, availability: 'missing', availability_reason: 'package_missing' });
  assert.equal(missing.selectable, false);
  assert.equal(missing.reason, 'package_missing');

  const notWired = readEarthDatasetAvailability({
    ...AVAILABLE_DESCRIPTOR,
    capabilities: { metadata: true, training: false, web_overview: true, trained_prediction: false },
  });
  assert.equal(notWired.selectable, false);

  assert.equal(readEarthDatasetAvailability(null).selectable, false);
});

test('split sample counts follow the published days instead of hard-coded numbers', () => {
  const rows = describeEarthSplitSamples(AVAILABLE_DESCRIPTOR.splits);
  assert.deepEqual(rows.map((row) => row.name), ['train', 'validation', 'test']);
  assert.deepEqual(rows.map((row) => row.windows), [2849, 1369, 1393]);
  assert.deepEqual(describeEarthSplitSamples(null), []);
});

test('three-hour split sample counts use published step counts', () => {
  const rows = describeEarthSplitSamples({
    train: { start: '2020-01-01T01:30:00Z', end: '2020-12-31T22:30:00Z', steps: 2928 },
  }, 56, 24);
  assert.equal(rows[0].windows, 2849);
  assert.equal(rows[0].days, 366);
});

test('input units list the target first and follow the current selection', () => {
  assert.deepEqual(describeEarthInputUnits(['SWGDN', 'U10M']), [
    { channel: 'TO3', unit: 'DU' },
    { channel: 'U10M', unit: 'm s-1' },
    { channel: 'SWGDN', unit: 'W m-2' },
  ]);
});

test('readiness blocks unavailable data, missing name and out-of-range values', () => {
  const ready = getEarthTrainingReadiness({
    datasetAvailability: readEarthDatasetAvailability(AVAILABLE_DESCRIPTOR),
    epochs: 10,
    batchSize: 8,
    learningRate: 0.001,
    modelName: 'Earth DLinear',
  });
  assert.equal(ready.ready, true);
  assert.deepEqual(ready.blockers, []);

  const blocked = getEarthTrainingReadiness({
    datasetAvailability: readEarthDatasetAvailability({ ...AVAILABLE_DESCRIPTOR, availability: 'invalid' }),
    epochs: 0,
    batchSize: 999,
    learningRate: 0,
    modelName: '   ',
  });
  assert.equal(blocked.ready, false);
  ['dataset_unavailable', 'model_name_required', 'epochs_out_of_range', 'batch_size_out_of_range', 'learning_rate_out_of_range']
    .forEach((code) => assert.ok(blocked.blockers.includes(code), `${code} should block`));
});

test('switching to Earth keeps each scene draft and restores it on the way back', () => {
  const marsValues = {
    trainingDataset: 'mcd_overview',
    modelSource: 'uploaded',
    selectedUploadedModelId: 'model-1',
    modelArchitecture: 'simvp',
    useSphere: true,
    windowValue: 5,
    horizon: 4,
    transferEnabled: true,
    selectedChannels: ['U', 'T'],
    customModelParams: { hidden_dim: 32 },
  };
  const marsDraft = captureTrainingDraft(marsValues);
  const earth = applyEarthTrainingDefaults({ storedMarsSnapshot: marsDraft });

  // Earth 生效值必须覆盖火星字段（草稿只保存、不应用）。
  assert.equal(earth.trainingDataset, EARTH_DATASET_ID);
  assert.equal(earth.modelSource, EARTH_MODEL_SOURCE_OFFICIAL);
  assert.equal(earth.modelArchitecture, EARTH_MODEL_ARCHITECTURE);
  assert.equal(earth.useSphere, false);
  assert.equal(earth.windowValue, EARTH_WINDOW);
  assert.equal(earth.horizon, EARTH_HORIZON);
  assert.equal(earth.transferEnabled, false);
  assert.equal(earth.selectedUploadedModelId, '');
  assert.deepEqual(earth.selectedChannels, EARTH_OPTIONAL_CHANNELS);
  assert.deepEqual(earth.customModelParams, {});
  assert.deepEqual(earth.storedMarsSnapshot, marsDraft);

  // 切回火星恢复原值，包括上传模型与自定义参数，避免 Earth 通道流入火星请求。
  const restored = resolveMarsTrainingRestore(marsDraft);
  assert.equal(restored.trainingDataset, 'mcd_overview');
  assert.equal(restored.modelSource, 'uploaded');
  assert.equal(restored.selectedUploadedModelId, 'model-1');
  assert.deepEqual(restored.customModelParams, { hidden_dim: 32 });
  assert.deepEqual(restored.selectedChannels, ['U', 'T']);
  assert.equal(resolveMarsTrainingRestore(null), null);
});

test('Earth keeps its own draft when returning from Mars', () => {
  const earthDraft = captureTrainingDraft({
    trainingDataset: EARTH_DATASET_ID,
    modelSource: 'uploaded',
    selectedUploadedModelId: 'earth-model-9',
    modelArchitecture: 'uploaded',
    selectedChannels: ['T2M'],
    customModelParams: { hidden_dim: 24 },
  });
  const restored = resolveEarthTrainingRestore(earthDraft);
  // Earth 的固定契约始终生效，即使草稿声明了别的值。
  assert.equal(restored.trainingDataset, EARTH_DATASET_ID);
  assert.equal(restored.windowValue, EARTH_WINDOW);
  assert.equal(restored.horizon, EARTH_HORIZON);
  assert.equal(restored.useSphere, false);
  assert.equal(restored.transferEnabled, false);
  // 草稿里的上传模型与参数被保留，不退回官方默认。
  assert.equal(restored.modelSource, 'uploaded');
  assert.equal(restored.modelArchitecture, 'uploaded');
  assert.equal(restored.selectedUploadedModelId, 'earth-model-9');
  assert.deepEqual(restored.customModelParams, { hidden_dim: 24 });
  assert.deepEqual(restored.selectedChannels, ['T2M']);
});

test('an unknown model source in a draft falls back to official, never to uploaded', () => {
  assert.equal(resolveEarthTrainingRestore(null), null);
  const restored = resolveEarthTrainingRestore({
    modelSource: 'carrier-pigeon',
    selectedUploadedModelId: 'should-not-stay-uploaded',
  });
  assert.equal(restored.modelSource, 'official');
  assert.equal(restored.modelArchitecture, EARTH_MODEL_ARCHITECTURE);

  const defaults = applyEarthTrainingDefaults({});
  assert.equal(defaults.modelSource, EARTH_MODEL_SOURCE_OFFICIAL);
  assert.equal(defaults.modelArchitecture, EARTH_MODEL_ARCHITECTURE);
});

test('Earth uploaded payload pins the source and sends only custom parameter values', () => {
  const uploaded = buildEarthTrainingHyperparameters({
    modelSource: 'uploaded',
    customModelParams: { hidden_dim: 16, dropout: 0 },
    selectedChannels: ['T2M'],
  });
  assert.equal(uploaded.model_source, 'uploaded');
  assert.equal(uploaded.model_architecture, 'uploaded');
  assert.deepEqual(uploaded.custom_model_params, { hidden_dim: 16, dropout: 0 });
  // 上传模型的结构由源码决定：不发送官方 DLinear 的线性层数开关。
  assert.equal(Object.hasOwn(uploaded, 'linear_hidden_layers'), false);
  // 身份字段永远由服务端生成。
  ['dataset_version', 'dataset_fingerprint', 'dataset_snapshot', '_uploaded_model_path', '_uploaded_model_id']
    .forEach((key) => assert.equal(Object.hasOwn(uploaded, key), false, `${key} must not be sent`));

  const official = buildEarthTrainingHyperparameters({ modelSource: 'official', linearHiddenLayers: 3 });
  assert.equal(official.model_source, 'official');
  assert.equal(official.model_architecture, 'dlinear');
  assert.equal(official.linear_hidden_layers, 3);
  assert.equal(Object.hasOwn(official, 'custom_model_params'), false);
});

test('uploaded Earth compatibility is read from the server verdict, not inferred', () => {
  assert.deepEqual(readEarthUploadedModelCompatibility(null), {
    known: false,
    compatible: false,
    reason: 'earth_compatibility_unknown',
  });

  const compatible = readEarthUploadedModelCompatibility({
    compatible: true,
    reasons: [],
    warnings: [],
    output_shape: [1, 3, 1, 36, 72],
    declares_earth_feed: true,
  });
  assert.equal(compatible.known, true);
  assert.equal(compatible.compatible, true);
  assert.deepEqual(compatible.outputShape, [1, 3, 1, 36, 72]);

  const incompatible = readEarthUploadedModelCompatibility({
    compatible: false,
    reasons: ['MODEL_SPEC does not declare the earth_merra2 dataset feed'],
  });
  assert.equal(incompatible.compatible, false);
  assert.match(incompatible.reason, /does not declare/);
});

test('the Earth uploaded gate blocks missing, unverified and incompatible models', () => {
  assert.equal(getEarthUploadedSelectionBlocker({ modelSource: 'official' }), null);
  assert.equal(
    getEarthUploadedSelectionBlocker({ modelSource: 'uploaded', uploadedModelId: '' }),
    'uploaded_model_required',
  );
  assert.equal(
    getEarthUploadedSelectionBlocker({ modelSource: 'uploaded', uploadedModelId: 'm1' }),
    'earth_compatibility_unknown',
  );
  assert.equal(
    getEarthUploadedSelectionBlocker({
      modelSource: 'uploaded', uploadedModelId: 'm1',
      compatibility: { dataset_id: EARTH_DATASET_ID, compatible: false, reasons: ['grid mismatch'] },
    }),
    'grid mismatch',
  );
  assert.equal(
    getEarthUploadedSelectionBlocker({
      modelSource: 'uploaded', uploadedModelId: 'm1',
      compatibility: { dataset_id: EARTH_DATASET_ID, status: 'available', compatible: true, reasons: [],
        datasets: { earth_merra2_3hourly_v1: { schema: 'astraatmos_earth_3hourly_uploaded_model_v1' } } },
    }),
    null,
  );
});

test('the shared sanitizer keeps Earth out of Mars requests unless Earth is allowed', () => {
  assert.equal(sanitizeTrainingDataset(EARTH_DATASET_ID), 'openmars_mcd');
  assert.equal(sanitizeTrainingDataset(EARTH_DATASET_ID, { allowEarth: true }), EARTH_DATASET_ID);
  assert.equal(isEarthTrainingDataset(EARTH_DATASET_ID), true);
  assert.equal(isEarthTrainingDataset('openmars_mcd'), false);
});

test('retired daily descriptors, payloads and saved drafts cannot start training', () => {
  for (const datasetId of ['earth_merra2_daily_v1', 'earth_merra2_daily_v2']) {
    assert.equal(isEarthTrainingDataset(datasetId), false);
    assert.throws(() => buildEarthTrainingHyperparameters({ datasetId }), /dataset_retired/);
    assert.equal(readEarthDatasetAvailability({ ...AVAILABLE_DESCRIPTOR, dataset_id: datasetId }).selectable, false);
    assert.equal(resolveEarthTrainingRestore({ trainingDataset: datasetId, modelSource: 'uploaded' }), null);
  }
});
