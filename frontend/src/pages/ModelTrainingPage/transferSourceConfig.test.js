import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  applyTransferStructureConfig,
  captureTransferStructureSnapshot,
  getAvailableTransferSourceTasks,
  hasTransferSourceTask,
  readTransferSourceTaskConfig,
} from './transferSourceConfig.js';

const CHANNEL_ORDER = ['U', 'V', 'D', 'S', 'T'];

function taskWith(hyperparameters, overrides = {}) {
  return {
    id: 41,
    model_source: 'official',
    uploaded_model_id: null,
    uploaded_model_version: null,
    hyperparameters: JSON.stringify(hyperparameters),
    ...overrides,
  };
}

function currentStructure(overrides = {}) {
  return {
    modelSource: 'official',
    selectedUploadedModelId: '',
    customModelParams: {},
    selectedChannels: ['T'],
    modelArchitecture: 'simvp',
    useSphere: false,
    hiddenDims: [32, 32],
    stlstmLayers: 2,
    architectureParamsByModel: {
      simvp: {
        spatial_hidden_dim: 32,
        temporal_hidden_dim: 64,
        temporal_depth: 2,
        dropout: 0.1,
      },
    },
    windowValue: 2,
    horizon: 1,
    ...overrides,
  };
}

test('reads every strict-loading structure field from an official recurrent task', () => {
  const task = taskWith({
    model_source: 'official',
    model_architecture: 'predrnnv2',
    selected_channels: ['D', 'U'],
    window: 4,
    horizon: 2,
    use_sphere: true,
    stlstm_hidden_dims: [48, 64, 96],
    epochs: 90,
    batch_size: 8,
    learning_rate: 0.002,
    early_stopping_patience: 7,
    seed: 99,
  });

  const result = readTransferSourceTaskConfig(task, {
    channelOrder: CHANNEL_ORDER,
    uploadedModels: [],
  });

  assert.deepEqual(result, {
    modelSource: 'official',
    selectedUploadedModelId: '',
    customModelParams: {},
    selectedChannels: ['U', 'D'],
    targetVariable: 'O3',
    inputVariables: ['O3', 'U', 'D'],
    modelArchitecture: 'predrnnv2',
    useSphere: true,
    hiddenDims: [48, 64, 96],
    stlstmLayers: 3,
    architectureParams: {},
    windowValue: 4,
    horizon: 2,
  });
  assert.equal('epochs' in result, false);
  assert.equal('batchSize' in result, false);
  assert.equal('learningRate' in result, false);
  assert.equal('seed' in result, false);
});

test('reads only the selected official architecture parameters', () => {
  const task = taskWith({
    model_source: 'official',
    model_architecture: 'patchtst',
    selected_channels: ['V'],
    window: 6,
    horizon: 3,
    use_sphere: false,
    patch_len: 3,
    stride: 2,
    d_model: 96,
    n_heads: 4,
    e_layers: 2,
    d_ff: 192,
    dropout: 0.2,
    spatial_hidden_dim: 999,
  });

  const result = readTransferSourceTaskConfig(task, {
    channelOrder: CHANNEL_ORDER,
    uploadedModels: [],
  });

  assert.deepEqual(result.architectureParams, {
    patch_len: 3,
    stride: 2,
    d_model: 96,
    n_heads: 4,
    e_layers: 2,
    d_ff: 192,
    dropout: 0.2,
  });
  assert.equal('spatial_hidden_dim' in result.architectureParams, false);
});

test('reads an available uploaded model and its structure parameters', () => {
  const task = taskWith(
    {
      model_source: 'uploaded',
      selected_channels: ['S'],
      window: 3,
      horizon: 2,
      use_sphere: false,
      custom_model_params: { hidden_dim: 48, use_bias: true },
    },
    {
      model_source: 'uploaded',
      uploaded_model_id: 'uploaded-1',
      uploaded_model_version: 3,
    }
  );

  const result = readTransferSourceTaskConfig(task, {
    channelOrder: CHANNEL_ORDER,
    uploadedModels: [{ id: 'uploaded-1', version: 3, validation_status: 'valid' }],
  });

  assert.equal(result.modelSource, 'uploaded');
  assert.equal(result.selectedUploadedModelId, 'uploaded-1');
  assert.deepEqual(result.customModelParams, { hidden_dim: 48, use_bias: true });
  assert.deepEqual(result.selectedChannels, ['S']);
  assert.equal(result.windowValue, 3);
  assert.equal(result.horizon, 2);
});

test('rejects malformed, incomplete, and unavailable source tasks without partial output', () => {
  assert.throws(
    () => readTransferSourceTaskConfig(
      { model_source: 'official', hyperparameters: '{bad-json' },
      { channelOrder: CHANNEL_ORDER, uploadedModels: [] }
    ),
    /unreadable/i
  );

  assert.throws(
    () => readTransferSourceTaskConfig(
      taskWith({
        model_source: 'official',
        model_architecture: 'predrnnv2',
        selected_channels: ['U'],
        window: 3,
        use_sphere: false,
        stlstm_hidden_dims: [64],
      }),
      { channelOrder: CHANNEL_ORDER, uploadedModels: [] }
    ),
    /incomplete/i
  );

  assert.throws(
    () => readTransferSourceTaskConfig(
      taskWith(
        {
          model_source: 'uploaded',
          selected_channels: [],
          window: 3,
          horizon: 3,
          use_sphere: false,
          custom_model_params: {},
        },
        {
          model_source: 'uploaded',
          uploaded_model_id: 'missing-model',
          uploaded_model_version: 1,
        }
      ),
      { channelOrder: CHANNEL_ORDER, uploadedModels: [] }
    ),
    /unavailable/i
  );
});

test('preserves the original snapshot while applying multiple source tasks', () => {
  const original = currentStructure();
  const snapshot = captureTransferStructureSnapshot(original);
  const first = applyTransferStructureConfig(original, {
    modelSource: 'official',
    selectedUploadedModelId: '',
    customModelParams: {},
    selectedChannels: ['U'],
    modelArchitecture: 'predrnnv2',
    useSphere: false,
    hiddenDims: [64, 64],
    stlstmLayers: 2,
    architectureParams: {},
    windowValue: 3,
    horizon: 3,
  });
  const second = applyTransferStructureConfig(first, {
    modelSource: 'official',
    selectedUploadedModelId: '',
    customModelParams: {},
    selectedChannels: ['D'],
    modelArchitecture: 'simvp',
    useSphere: true,
    hiddenDims: [],
    stlstmLayers: 0,
    architectureParams: {
      spatial_hidden_dim: 96,
      temporal_hidden_dim: 128,
      temporal_depth: 3,
      dropout: 0.2,
    },
    windowValue: 4,
    horizon: 2,
  });

  second.selectedChannels.push('S');
  second.architectureParamsByModel.simvp.spatial_hidden_dim = 200;

  assert.deepEqual(snapshot, original);
  assert.notStrictEqual(snapshot.selectedChannels, original.selectedChannels);
  assert.notStrictEqual(snapshot.architectureParamsByModel, original.architectureParamsByModel);
  assert.deepEqual(captureTransferStructureSnapshot(snapshot), original);
});

test('detects when a selected completed source task disappears', () => {
  assert.equal(hasTransferSourceTask([{ id: 12 }, { id: 41 }], '41'), true);
  assert.equal(hasTransferSourceTask([{ id: 12 }], '41'), false);
  assert.equal(hasTransferSourceTask([{ id: 41 }], ''), false);
});

test('keeps only server-confirmed available tasks as transfer-learning sources', () => {
  const tasks = [
    { id: 12, status: 'completed', model_available: true, output_model_path: 'D:/models/valid.pth' },
    { id: 13, status: 'completed', model_available: false, output_model_path: 'D:/models/missing.pth' },
    { id: 14, status: 'running', model_available: false, output_model_path: 'D:/models/running.pth' },
    { id: 15, status: 'completed', output_model_path: 'D:/models/legacy.pth' },
  ];

  assert.deepEqual(getAvailableTransferSourceTasks(tasks), [tasks[0]]);
});
