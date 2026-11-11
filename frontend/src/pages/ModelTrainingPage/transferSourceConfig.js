import {
  MODEL_STRUCTURE_PARAM_CONFIG,
  RECURRENT_MODEL_ARCHITECTURES,
} from './trainingParamSanitizers.js';

const SUPPORTED_OFFICIAL_ARCHITECTURES = new Set([
  ...RECURRENT_MODEL_ARCHITECTURES,
  ...Object.keys(MODEL_STRUCTURE_PARAM_CONFIG),
]);
const THREE_VALUE_LIST_FIELDS = new Set(['patch_size', 'cuboid_size']);
const OPEN_INTERVAL_FIELDS = new Set(['initial_history_weight', 'initial_translation_weight']);

export class TransferSourceConfigError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'TransferSourceConfigError';
    this.code = code;
  }
}

function cloneValue(value) {
  if (Array.isArray(value)) return value.map(cloneValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, cloneValue(item)]));
  }
  return value;
}

function incomplete(message) {
  throw new TransferSourceConfigError('incomplete', `Transfer source configuration is incomplete: ${message}`);
}

function parseHyperparameters(raw) {
  try {
    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('not an object');
    return parsed;
  } catch {
    throw new TransferSourceConfigError('unreadable', 'Transfer source configuration is unreadable');
  }
}

function requirePositiveInteger(value, key) {
  if (!Number.isInteger(value) || value <= 0) incomplete(key);
  return value;
}

function readChannels(value, channelOrder) {
  if (!Array.isArray(value)) incomplete('selected_channels');
  const supported = new Set(channelOrder);
  const requested = value.map((channel) => String(channel).toUpperCase());
  if (new Set(requested).size !== requested.length || requested.some((channel) => !supported.has(channel))) {
    incomplete('selected_channels');
  }
  const requestedSet = new Set(requested);
  return channelOrder.filter((channel) => requestedSet.has(channel));
}

function readBoolean(value, key) {
  if (typeof value !== 'boolean') incomplete(key);
  return value;
}

function readNumber(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value)) incomplete(field.key);

  if (field.type === 'integer') {
    if (!Number.isInteger(value) || value <= 0) incomplete(field.key);
  } else if (field.type === 'dropout') {
    if (value < 0 || value > 0.9) incomplete(field.key);
  } else if (field.type === 'boundedFloat') {
    const minimum = OPEN_INTERVAL_FIELDS.has(field.key) ? 0 : 0;
    if (value < minimum || value > 0.9 || (OPEN_INTERVAL_FIELDS.has(field.key) && value === 0)) {
      incomplete(field.key);
    }
  } else if (field.type === 'nonNegativeNumber' && value < 0) {
    incomplete(field.key);
  }
  return value;
}

function readArchitectureParams(hyperparameters, architecture) {
  const fields = MODEL_STRUCTURE_PARAM_CONFIG[architecture] || [];
  return Object.fromEntries(fields.map((field) => {
    const value = hyperparameters[field.key];
    if (field.type !== 'integerList') return [field.key, readNumber(value, field)];
    if (!Array.isArray(value) || value.length === 0) incomplete(field.key);
    if (THREE_VALUE_LIST_FIELDS.has(field.key) && value.length !== 3) incomplete(field.key);
    if (value.some((item) => !Number.isInteger(item) || item <= 0)) incomplete(field.key);
    return [field.key, [...value]];
  }));
}

function readModelSource(task, hyperparameters) {
  const modelSource = String(task?.model_source || hyperparameters.model_source || '').toLowerCase();
  if (modelSource !== 'official' && modelSource !== 'uploaded') incomplete('model_source');
  return modelSource;
}

function readOfficialConfig(base, hyperparameters) {
  const rawArchitecture = String(hyperparameters.model_architecture || '').toLowerCase();
  const architecture = rawArchitecture === 'predrnnv2_sphere' ? 'predrnnv2' : rawArchitecture;
  if (!SUPPORTED_OFFICIAL_ARCHITECTURES.has(architecture)) incomplete('model_architecture');

  if (RECURRENT_MODEL_ARCHITECTURES.includes(architecture)) {
    const hiddenDims = hyperparameters.stlstm_hidden_dims;
    if (!Array.isArray(hiddenDims) || hiddenDims.length === 0) incomplete('stlstm_hidden_dims');
    if (hiddenDims.some((value) => !Number.isInteger(value) || value <= 0)) incomplete('stlstm_hidden_dims');
    return {
      ...base,
      selectedUploadedModelId: '',
      customModelParams: {},
      modelArchitecture: architecture,
      hiddenDims: [...hiddenDims],
      stlstmLayers: hiddenDims.length,
      architectureParams: {},
    };
  }

  return {
    ...base,
    selectedUploadedModelId: '',
    customModelParams: {},
    modelArchitecture: architecture,
    hiddenDims: [],
    stlstmLayers: 0,
    architectureParams: readArchitectureParams(hyperparameters, architecture),
  };
}

function readUploadedConfig(base, task, hyperparameters, uploadedModels) {
  const uploadedModelId = String(task?.uploaded_model_id || hyperparameters._uploaded_model_id || '').trim();
  const uploadedModelVersion = task?.uploaded_model_version ?? hyperparameters._uploaded_model_version;
  const model = uploadedModels.find((item) => item.id === uploadedModelId);
  if (!model || model.validation_status !== 'valid' || model.version !== uploadedModelVersion) {
    throw new TransferSourceConfigError('unavailable', 'Transfer source uploaded model is unavailable');
  }
  const customModelParams = hyperparameters.custom_model_params;
  if (!customModelParams || typeof customModelParams !== 'object' || Array.isArray(customModelParams)) {
    incomplete('custom_model_params');
  }

  return {
    ...base,
    selectedUploadedModelId: uploadedModelId,
    customModelParams: cloneValue(customModelParams),
    modelArchitecture: null,
    hiddenDims: [],
    stlstmLayers: 0,
    architectureParams: {},
  };
}

export function readTransferSourceTaskConfig(task, { channelOrder = [], uploadedModels = [] } = {}) {
  const hyperparameters = parseHyperparameters(task?.hyperparameters);
  const modelSource = readModelSource(task, hyperparameters);
  const base = {
    modelSource,
    selectedChannels: readChannels(hyperparameters.selected_channels, channelOrder),
    targetVariable: hyperparameters.target_variable || 'O3',
    inputVariables: Array.isArray(hyperparameters.input_variables) ? [...hyperparameters.input_variables] : ['O3', ...readChannels(hyperparameters.selected_channels, channelOrder)],
    useSphere: readBoolean(hyperparameters.use_sphere, 'use_sphere'),
    windowValue: requirePositiveInteger(hyperparameters.window, 'window'),
    horizon: requirePositiveInteger(hyperparameters.horizon, 'horizon'),
  };

  return modelSource === 'official'
    ? readOfficialConfig(base, hyperparameters)
    : readUploadedConfig(base, task, hyperparameters, uploadedModels);
}

export function captureTransferStructureSnapshot(structureState) {
  return cloneValue(structureState);
}

export function hasTransferSourceTask(tasks, taskId) {
  if (!taskId) return false;
  return tasks.some((task) => String(task.id) === String(taskId));
}

export function getAvailableTransferSourceTasks(tasks = []) {
  return (Array.isArray(tasks) ? tasks : []).filter(
    (task) => task?.status === 'completed' && task?.model_available === true
  );
}

export function applyTransferStructureConfig(structureState, sourceConfig) {
  const next = captureTransferStructureSnapshot(structureState);
  next.modelSource = sourceConfig.modelSource;
  next.selectedUploadedModelId = sourceConfig.selectedUploadedModelId;
  next.customModelParams = cloneValue(sourceConfig.customModelParams);
  next.selectedChannels = [...sourceConfig.selectedChannels];
  next.targetVariable = sourceConfig.targetVariable || 'O3';
  next.inputVariables = sourceConfig.inputVariables || ['O3', ...sourceConfig.selectedChannels];
  next.useSphere = sourceConfig.useSphere;
  next.windowValue = sourceConfig.windowValue;
  next.horizon = sourceConfig.horizon;

  if (sourceConfig.modelSource === 'official') {
    next.modelArchitecture = sourceConfig.modelArchitecture;
    next.hiddenDims = [...sourceConfig.hiddenDims];
    next.stlstmLayers = sourceConfig.stlstmLayers;
    next.architectureParamsByModel = {
      ...next.architectureParamsByModel,
      [sourceConfig.modelArchitecture]: cloneValue(sourceConfig.architectureParams),
    };
  }
  return next;
}
