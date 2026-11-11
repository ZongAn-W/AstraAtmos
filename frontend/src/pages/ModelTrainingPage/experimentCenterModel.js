/**
 * 实验中心纯函数契约。
 *
 * 这些函数只做展示层推导：不请求接口、不启动训练、不修改传入的任务对象。
 * 阶段、目录筛选、指标解析和摘要都从这里取值，组件只负责渲染。
 */

import { isEarthMetrics } from '../../utils/earthMetricMeta.js';
import { resolveEnvironmentTask } from '../../utils/environmentVariables.js';
import {
  MODEL_STRUCTURE_PARAM_CONFIG,
  RECURRENT_MODEL_ARCHITECTURES,
  TRAINING_DATASET_IDS,
  getModelStructureConfig,
  isRecurrentArchitecture,
  isRetiredEarthDataset,
  sanitizeNonNegativeInteger,
  sanitizePositiveInteger,
  sanitizePositiveNumber,
  sanitizeTrainingDataset,
} from './trainingParamSanitizers.js';
import { EARTH_DEFAULT_SPLIT_RATIOS } from './earthTrainingConfig.js';

export const EXPERIMENT_STAGES = ['configure', 'monitor', 'result'];
export const EXPERIMENT_STATUS_FILTERS = ['all', 'queued', 'running', 'completed', 'failed', 'cancelled'];

/** 「复制配置」警告：界面按 code 取本地化文案。 */
export const EXPERIMENT_CONFIG_WARNING_CODES = [
  'unreadable_hyperparameters',
  'missing_training_fields',
  'incomplete_structure',
  'uploaded_model_unavailable',
  'custom_model_params_missing',
];

export const ARCHITECTURE_DEFAULT_PARAMS = Object.freeze(
  Object.fromEntries(
    Object.entries(MODEL_STRUCTURE_PARAM_CONFIG).map(([modelId, fields]) => [
      modelId,
      Object.freeze(Object.fromEntries(fields.map((field) => [field.key, field.defaultValue]))),
    ])
  )
);

const DEFAULT_HIDDEN_DIMS = Object.freeze([64, 64, 64]);
const MAX_SEED = 2147483647;
const MAX_EARLY_STOPPING = 200;
const SUPPORTED_ARCHITECTURES = new Set([
  ...RECURRENT_MODEL_ARCHITECTURES,
  ...Object.keys(MODEL_STRUCTURE_PARAM_CONFIG),
]);
const THREE_VALUE_LIST_FIELDS = new Set(['patch_size', 'cuboid_size']);
const OPEN_INTERVAL_FIELDS = new Set(['initial_history_weight', 'initial_translation_weight']);

const MONITOR_STATUSES = new Set(['pending', 'queued', 'running']);
const KNOWN_STATUSES = new Set(['pending', 'queued', 'running', 'completed', 'failed', 'cancelled']);

/** 目录里“运行中”分组同时覆盖排队任务。 */
const STATUS_FILTER_MEMBERS = {
  queued: new Set(['queued']),
  running: new Set(['pending', 'running']),
  completed: new Set(['completed']),
  failed: new Set(['failed']),
  cancelled: new Set(['cancelled']),
};

const ARCHITECTURE_LABELS = {
  predrnnv2: 'PredRNNv2',
  predrnnpp: 'PredRNN++',
  convlstm: 'ConvLSTM',
  simvp: 'SimVP',
  dlinear: 'DLinear',
  informer: 'Informer',
  autoformer: 'Autoformer',
  patchtst: 'PatchTST',
  timemixer: 'TimeMixer',
  timexer: 'TimeXer',
  tsmixer: 'TSMixer',
  crossformer: 'Crossformer',
  earthformer: 'Earthformer',
  etsformer: 'ETSformer',
  fedformer: 'FEDformer',
  itransformer: 'iTransformer',
  mau: 'MAU',
  nbeats: 'N-BEATS',
  nhits: 'N-HiTS',
  pyraformer: 'Pyraformer',
  rnn_cnn_rnn: 'RNN-CNN-RNN',
  cnn_rnn_cnn_rnn_cnn: 'CNN-RNN-CNN-RNN-CNN',
  simvp_3dconv: 'SimVP-3DConv',
  simvp_hybrid3d: 'SimVP-Hybrid3D',
  convlstm_mst: 'ConvLSTM-MST',
  dlinear_mst: 'DLinear-MST',
  convlstm_phase_gated_mst: 'ConvLSTM-PhaseGated-MST',
  convlstm_mst_feature_refiner: 'ConvLSTM-MST-Feature',
  convlstm_climatology_anomaly: 'ConvLSTM-Climatology-Anomaly',
};

/**
 * 官方模型架构注册表（单一来源）。
 *
 * 这里只描述“有哪些官方骨干、属于哪个家族、家族怎么解释”，
 * 不描述任何训练参数——结构参数仍由 trainingParamSanitizers 的
 * `MODEL_STRUCTURE_PARAM_CONFIG` 提供，两侧必须同时列出同一批 id。
 * 页面、官方模型选择器、配置检查器与运行条都从这里取值。
 */
export const MODEL_ARCHITECTURE_FAMILIES = Object.freeze([
  Object.freeze({ id: 'recurrent', label: '循环网络', labelEn: 'Recurrent' }),
  Object.freeze({ id: 'convolutional', label: '卷积与时空', labelEn: 'Convolutional' }),
  Object.freeze({ id: 'transformer', label: 'Transformer', labelEn: 'Transformer' }),
  Object.freeze({ id: 'linear', label: '线性与 MLP', labelEn: 'Linear / MLP' }),
  Object.freeze({ id: 'hybrid', label: '实验变体', labelEn: 'Experimental variants' }),
]);

export const MODEL_ARCHITECTURE_FAMILY_HINTS = Object.freeze({
  recurrent: Object.freeze({
    label: '循环时空网络',
    labelEn: 'Recurrent spatiotemporal',
    hint: '逐帧递推的时空记忆网络，适合连续 Ls 演变的臭氧场。',
    hintEn: 'Frame-by-frame recurrent memory networks for continuous Ls evolution.',
  }),
  convolutional: Object.freeze({
    label: '卷积与时空',
    labelEn: 'Convolutional',
    hint: '以卷积或 3D 卷积一次性生成整段预测，训练更稳定。',
    hintEn: 'Convolutional and 3D-convolutional heads that emit the whole horizon at once.',
  }),
  transformer: Object.freeze({
    label: 'Transformer',
    labelEn: 'Transformer',
    hint: '注意力长程建模，对通道多、序列长的配置更敏感。',
    hintEn: 'Attention-based long-range modelling; sensitive to channel and sequence length.',
  }),
  linear: Object.freeze({
    label: '线性与 MLP',
    labelEn: 'Linear / MLP',
    hint: '轻量线性与 MLP 家族，显存占用小、收敛快，适合作为基线。',
    hintEn: 'Lightweight linear and MLP families with low memory cost — good baselines.',
  }),
  hybrid: Object.freeze({
    label: '实验变体',
    labelEn: 'Experimental variants',
    hint: '平台内置的实验结构变体（MST、相位门控、气候态距平等）。',
    hintEn: 'In-house experimental variants (MST, phase gating, climatology anomaly and more).',
  }),
});

const ARCHITECTURE_FAMILY_OVERRIDES = {
  rnn_cnn_rnn: 'hybrid',
  cnn_rnn_cnn_rnn_cnn: 'hybrid',
  simvp_3dconv: 'hybrid',
  simvp_hybrid3d: 'hybrid',
  convlstm_mst: 'hybrid',
  dlinear_mst: 'hybrid',
  convlstm_phase_gated_mst: 'hybrid',
  convlstm_mst_feature_refiner: 'hybrid',
  convlstm_climatology_anomaly: 'hybrid',
  mau: 'transformer',
};

const ARCHITECTURE_FAMILY_DEFAULTS = Object.freeze({
  recurrent: Object.freeze(['predrnnv2', 'predrnnpp', 'convlstm']),
  convolutional: Object.freeze(['simvp', 'earthformer']),
  linear: Object.freeze(['dlinear', 'nbeats', 'nhits', 'tsmixer']),
  transformer: Object.freeze([
    'informer',
    'autoformer',
    'patchtst',
    'timemixer',
    'timexer',
    'crossformer',
    'etsformer',
    'fedformer',
    'itransformer',
    'pyraformer',
  ]),
});

/** 官方模型架构清单：id + 展示名，顺序即界面顺序。 */
export const MODEL_ARCHITECTURES = Object.freeze(
  Object.entries(ARCHITECTURE_LABELS).map(([id, label]) => Object.freeze({ id, label, family: getModelArchitectureFamily(id) }))
);

/** 架构所属家族；未登记的 id 回退到“实验变体”，不会抛错。 */
export function getModelArchitectureFamily(architecture) {
  const id = String(architecture || '').trim().toLowerCase();
  if (!id) return 'hybrid';
  if (ARCHITECTURE_FAMILY_OVERRIDES[id]) return ARCHITECTURE_FAMILY_OVERRIDES[id];
  const match = Object.entries(ARCHITECTURE_FAMILY_DEFAULTS)
    .find(([, ids]) => ids.includes(id));
  return match ? match[0] : 'hybrid';
}

/** 家族的中英文说明；界面按当前语言取其中一份。 */
export function getModelArchitectureFamilyHint(family) {
  const key = String(family || '').trim().toLowerCase();
  return MODEL_ARCHITECTURE_FAMILY_HINTS[key] || MODEL_ARCHITECTURE_FAMILY_HINTS.hybrid;
}

/**
 * 官方模型选择器的纯筛选函数：名称搜索 + 家族筛选。
 * `family` 为 null / 'all' 时不按家族过滤；搜索匹配展示名与 id，忽略大小写。
 */
export function filterModelArchitectures(architectures = [], { search = '', family = 'all' } = {}) {
  const query = String(search ?? '').trim().toLocaleLowerCase();
  const familyKey = String(family ?? 'all').trim().toLowerCase();
  return (Array.isArray(architectures) ? architectures : []).filter((item) => {
    if (!item) return false;
    if (familyKey && familyKey !== 'all' && getModelArchitectureFamily(item.id) !== familyKey) return false;
    if (!query) return true;
    return String(item.label || '').toLocaleLowerCase().includes(query)
      || String(item.id || '').toLocaleLowerCase().includes(query);
  });
}

/** 结果面板展示的核心指标，顺序即展示顺序。 */
export const EXPERIMENT_METRIC_KEYS = ['rmse', 'mae', 'mse', 'r2', 'mape', 'smape'];

const METRIC_ALIASES = {
  rmse: ['rmse', 'RMSE', 'root_mean_squared_error'],
  mae: ['mae', 'MAE', 'mean_absolute_error'],
  mse: ['mse', 'MSE', 'mean_squared_error'],
  r2: ['r2', 'R2', 'r_squared', 'r2_score', 'R²'],
  mape: ['mape', 'MAPE'],
  smape: ['smape', 'SMAPE'],
};

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

/**
 * 「复制配置」使用的深拷贝：嵌套对象与数组都不与原任务共享引用，
 * 复制后修改参数不会改到任务记录或上传模型的原始对象。
 */
export function cloneExperimentConfigValue(value) {
  if (Array.isArray(value)) return value.map(cloneExperimentConfigValue);

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, cloneExperimentConfigValue(item)])
    );
  }

  return value;
}

function parseObjectPayload(raw) {
  if (raw == null || raw === '') return {};
  if (isPlainObject(raw)) return raw;
  if (typeof raw !== 'string') return {};
  try {
    const parsed = JSON.parse(raw);
    return isPlainObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** 解析任务超参数；坏 JSON、数组与空值统一回退为空对象。 */
export function parseTaskHyperparameters(raw) {
  return parseObjectPayload(raw);
}

/** 解析任务指标；坏 JSON、数组与空值统一回退为空对象。 */
export function parseTaskMetrics(raw) {
  return parseObjectPayload(raw);
}

/**
 * 推导实验中心当前阶段。
 *
 * 用户显式新建实验时优先进入配置；否则按当前任务状态推导：
 * pending/running → monitor，其余已知与未知状态 → result，方便旧任务仍可查看。
 */
export function getExperimentStage({ activeTask = null, isCreating = false } = {}) {
  if (isCreating) return 'configure';
  const status = String(activeTask?.status || '').toLowerCase();
  if (!status) return 'configure';
  if (MONITOR_STATUSES.has(status)) return 'monitor';
  return 'result';
}

export function getExperimentStatusFilter(status) {
  return EXPERIMENT_STATUS_FILTERS.includes(status) ? status : 'all';
}

/** 目录状态分组：未知状态不进入任何受限分组，只在“全部”中出现。 */
export function matchesExperimentStatusFilter(taskStatus, statusFilter) {
  const filter = getExperimentStatusFilter(statusFilter);
  if (filter === 'all') return true;
  return STATUS_FILTER_MEMBERS[filter].has(String(taskStatus || '').toLowerCase());
}

export function countExperimentStatuses(tasks = []) {
  const counts = Object.fromEntries(EXPERIMENT_STATUS_FILTERS.map((filter) => [filter, 0]));
  (Array.isArray(tasks) ? tasks : []).forEach((task) => {
    if (!task) return;
    counts.all += 1;
    EXPERIMENT_STATUS_FILTERS
      .filter((filter) => filter !== 'all')
      .forEach((filter) => {
        if (matchesExperimentStatusFilter(task.status, filter)) counts[filter] += 1;
      });
  });
  return counts;
}

function normalizeTagIds(tagIds) {
  if (!Array.isArray(tagIds)) return [];
  return tagIds.map((id) => Number(id)).filter((id) => Number.isFinite(id));
}

function taskMatchesSearch(task, query) {
  if (!query) return true;
  const name = String(task.custom_model_name || '').toLocaleLowerCase();
  if (name.includes(query)) return true;
  return String(task.id ?? '').toLocaleLowerCase() === query;
}

/**
 * 组合目录筛选：状态、名称搜索、标签交集与未分组。
 * 返回新数组，保留传入顺序，不修改原数组和任务对象。
 */
export function filterExperimentTasks(tasks = [], {
  status = 'all',
  search = '',
  tagIds = [],
  untagged = false,
} = {}) {
  const query = String(search ?? '').trim().toLocaleLowerCase();
  const selected = normalizeTagIds(tagIds);
  const seen = new Set();

  return (Array.isArray(tasks) ? tasks : []).filter((task) => {
    if (!task) return false;
    const key = task.id ?? Symbol('task');
    if (seen.has(key)) return false;
    seen.add(key);

    if (!matchesExperimentStatusFilter(task.status, status)) return false;

    const tags = (task.tags || []).map((tag) => Number(tag?.id));
    if (untagged ? tags.length > 0 : !selected.every((id) => tags.includes(id))) return false;

    return taskMatchesSearch(task, query);
  });
}

/** 归一化指标对象：只保留可识别的指标键，未提供的键不出现在结果里。 */
export function readExperimentMetrics(raw, split = 'test') {
  const parsed = parseTaskMetrics(raw);
  const earth = isEarthMetrics(parsed);
  const metrics = earth ? (parsed.splits?.[split]?.overall || {}) : parsed;
  const normalized = {};
  EXPERIMENT_METRIC_KEYS.forEach((key) => {
    const alias = METRIC_ALIASES[key].find((candidate) => {
      const value = metrics[candidate];
      if (earth) return typeof value === 'number' && Number.isFinite(value);
      return value !== undefined && value !== null && !(typeof value === 'number' && Number.isNaN(value));
    });
    if (alias) normalized[key] = metrics[alias];
  });
  return normalized;
}

export function getExperimentArchitectureLabel(architecture) {
  const raw = String(architecture || '').trim().toLowerCase();
  if (!raw) return '';
  const normalized = raw === 'predrnnv2_sphere' ? 'predrnnv2' : raw;
  return ARCHITECTURE_LABELS[normalized] || architecture;
}

/**
 * 任务摘要：名称、架构、数据集、指标、状态和模型可用性。
 * 只读取已有字段，缺失时给出空值而不是抛错。
 */
export function buildExperimentSummary(task) {
  if (!task) {
    return {
      id: null,
      name: '',
      status: '',
      architecture: '',
      architectureLabel: '',
      dataset: '',
      modelSource: '',
      modelAvailable: false,
      targetVariable: '',
      inputVariables: [],
      progress: 0,
      startTime: null,
      metrics: {},
      task: null,
    };
  }

  const hyperparameters = parseTaskHyperparameters(task.hyperparameters);
  const architecture = String(hyperparameters.model_architecture || '').trim().toLowerCase();
  const dataset = String(task.dataset_id || hyperparameters.training_dataset || '').trim();
  const environmentTask = resolveEnvironmentTask(task);

  return {
    id: task.id ?? null,
    name: String(task.custom_model_name || '').trim(),
    status: String(task.status || '').toLowerCase(),
    architecture: architecture === 'predrnnv2_sphere' ? 'predrnnv2' : architecture,
    architectureLabel: getExperimentArchitectureLabel(architecture),
    dataset,
    modelSource: String(task.model_source || hyperparameters.model_source || '').toLowerCase(),
    modelAvailable: task.model_available === true,
    targetVariable: environmentTask.target,
    inputVariables: [...environmentTask.inputs],
    progress: Number(task.progress) || 0,
    startTime: task.start_time || null,
    metrics: readExperimentMetrics(task.metrics),
    task,
  };
}

function normalizeLossHistory(raw) {
  const parsed = parseObjectPayload(raw);
  return {
    train: Array.isArray(parsed.train) ? parsed.train : [],
    val: Array.isArray(parsed.val) ? parsed.val : [],
  };
}

/**
 * 解析结果阶段真正可以展示的进度。
 *
 * 监控阶段优先使用实时推送的数据；结果阶段改用任务行自身的字段，
 * 这样切换任务后不会短暂显示上一个任务的进度和 Loss 曲线。
 */
export function resolveActiveTaskProgress(activeTask, liveProgress) {
  const fallback = {
    progress: 0,
    current_epoch: 0,
    total_epochs: 0,
    current_loss: null,
    eta: '--:--',
    loss_history: { train: [], val: [] },
  };
  if (!activeTask) return fallback;

  const stage = getExperimentStage({ activeTask, isCreating: false });
  if (stage === 'monitor' && liveProgress) {
    return {
      progress: Number(liveProgress.progress) || 0,
      current_epoch: Number(liveProgress.current_epoch) || 0,
      total_epochs: Number(liveProgress.total_epochs) || 0,
      current_loss: liveProgress.current_loss ?? null,
      eta: liveProgress.eta || '--:--',
      loss_history: normalizeLossHistory(liveProgress.loss_history),
    };
  }

  return {
    progress: Number(activeTask.progress) || 0,
    current_epoch: Number(activeTask.current_epoch) || 0,
    total_epochs: Number(activeTask.total_epochs) || 0,
    current_loss: activeTask.current_loss ?? null,
    eta: activeTask.eta || '--:--',
    loss_history: normalizeLossHistory(activeTask.loss_history),
  };
}

/**
 * 「复制配置」读取到的父级表单字段。
 * 只读取命名、数据集和标签；模型结构由 transferSourceConfig 的规范化逻辑负责。
 */
export function readExperimentConfigDataset(task) {
  if (!task) return { sourceTaskId: null, name: '', trainingDataset: '', tagIds: [] };
  const hyperparameters = parseTaskHyperparameters(task.hyperparameters);
  return {
    sourceTaskId: task.id ?? null,
    name: String(task.custom_model_name || '').trim(),
    trainingDataset: String(task.dataset_id || hyperparameters.training_dataset || '').trim(),
    tagIds: (task.tags || []).map((tag) => Number(tag?.id)).filter((id) => Number.isFinite(id)),
  };
}

function readPositiveInteger(hyperparameters, key) {
  const value = hyperparameters[key];
  return Number.isInteger(value) && value > 0 ? value : null;
}

function readPositiveNumber(hyperparameters, key) {
  const value = hyperparameters[key];
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}

function readArchitectureParams(hyperparameters, architecture) {
  const fields = getModelStructureConfig(architecture);
  if (fields.length === 0) return { complete: false, params: {} };

  const params = {};
  let complete = true;
  fields.forEach((field) => {
    const value = hyperparameters[field.key];
    if (field.type === 'integerList') {
      const list = Array.isArray(value) && value.every((item) => Number.isInteger(item) && item > 0)
        ? [...value]
        : null;
      const lengthOk = THREE_VALUE_LIST_FIELDS.has(field.key) ? list?.length === 3 : Boolean(list?.length);
      if (!list || !lengthOk) {
        complete = false;
        return;
      }
      params[field.key] = list;
      return;
    }
    if (field.type === 'dropout') {
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 0.9) {
        complete = false;
        return;
      }
      params[field.key] = value;
      return;
    }
    if (field.type === 'boundedFloat' || field.type === 'nonNegativeNumber') {
      const minimum = field.type === 'boundedFloat' && OPEN_INTERVAL_FIELDS.has(field.key) ? 0.000001 : 0;
      if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || (field.type === 'boundedFloat' && value > 0.9)) {
        complete = false;
        return;
      }
      params[field.key] = value;
      return;
    }
    if (!Number.isInteger(value) || value <= 0) {
      complete = false;
      return;
    }
    params[field.key] = value;
  });

  return { complete, params };
}

/**
 * 「复制配置」读取任务的完整训练配置。
 *
 * 覆盖命名、数据集、模型来源与结构、通道、窗口/步长、训练超参数、隐藏层、
 * 模型专属结构参数与标签；迁移学习一律关闭（原任务不会被当作迁移来源）。
 * 读取失败时给出可用字段并附带 `warnings`（按 code 本地化），不抛错。
 */
export function readExperimentConfig(task, {
  uploadedModels = [],
  channelOrder = [],
  nameSuffix = ' (Copy)',
  existingNames = [],
  defaults = {},
} = {}) {
  const base = {
    sourceTaskId: task?.id ?? null,
    customModelName: '',
    trainingDataset: String(defaults.trainingDataset || '') || 'openmars_mcd',
    modelSource: 'official',
    selectedUploadedModelId: '',
    selectedUploadedModelVersion: null,
    uploadedModelName: '',
    // 默认空对象而不是 undefined：页面可以安全地直接 setState。
    customModelParams: {},
    selectedChannels: [],
    modelArchitecture: 'predrnnv2',
    useSphere: false,
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    trainRatio: EARTH_DEFAULT_SPLIT_RATIOS.train_ratio,
    validationRatio: EARTH_DEFAULT_SPLIT_RATIOS.validation_ratio,
    testRatio: EARTH_DEFAULT_SPLIT_RATIOS.test_ratio,
    windowValue: 3,
    horizon: 3,
    earlyStoppingPatience: 0,
    seed: 11,
    hiddenDims: [...DEFAULT_HIDDEN_DIMS],
    architectureParamsByModel: Object.fromEntries(
      Object.entries(ARCHITECTURE_DEFAULT_PARAMS).map(([modelId, params]) => [modelId, { ...params }])
    ),
    transferEnabled: false,
    transferSourceType: 'task',
    transferSourceTaskId: '',
    transferFreezeMode: 'none',
    finetuneLearningRate: 0.0001,
    tagIds: [],
    warnings: [],
  };
  if (!task) return base;

  const warnings = [];
  const addWarning = (code) => {
    if (!warnings.includes(code)) warnings.push(code);
  };

  const rawHyperparameters = task.hyperparameters;
  let hyperparameters = parseTaskHyperparameters(rawHyperparameters);
  const hyperparametersReadable = typeof rawHyperparameters === 'string'
    ? rawHyperparameters.trim() !== '' && Object.keys(hyperparameters).length > 0
    : isPlainObject(rawHyperparameters);
  if (!hyperparametersReadable) {
    hyperparameters = {};
    addWarning('unreadable_hyperparameters');
  }
  const readFailureWarning = hyperparametersReadable ? 'missing_training_fields' : 'unreadable_hyperparameters';

  const name = String(task.custom_model_name || '').trim();
  const trainingDatasetRaw = String(task.dataset_id || hyperparameters.training_dataset || '').trim();

  const modelSource = String(task.model_source || hyperparameters.model_source || 'official').toLowerCase() === 'uploaded'
    ? 'uploaded'
    : 'official';

  let selectedUploadedModelId = '';
  let selectedUploadedModelVersion = null;
  let uploadedModelName = '';
  let customModelParams = {};
  if (modelSource === 'uploaded') {
    const uploadedId = String(task.uploaded_model_id || hyperparameters._uploaded_model_id || '').trim();
    const uploadedVersion = task.uploaded_model_version ?? hyperparameters._uploaded_model_version ?? null;
    const model = uploadedModels.find((item) => item.id === uploadedId);
    const available = Boolean(model) && model.validation_status === 'valid' && model.version === uploadedVersion;
    if (available) {
      selectedUploadedModelId = uploadedId;
      selectedUploadedModelVersion = uploadedVersion ?? null;
      uploadedModelName = model.display_name || model.original_filename || '';
    } else {
      addWarning('uploaded_model_unavailable');
    }
    // 上传模型的自定义参数必须原样带回，否则复制出的实验会退回参数默认值。
    const rawCustomModelParams = hyperparameters.custom_model_params;
    customModelParams = isPlainObject(rawCustomModelParams)
      ? cloneExperimentConfigValue(rawCustomModelParams)
      : {};
    if (rawCustomModelParams === undefined || rawCustomModelParams === null) {
      // 旧任务完全没有该字段：单独提示，避免误以为已复制到参数。
      addWarning('custom_model_params_missing');
    } else if (!isPlainObject(rawCustomModelParams)) {
      // 存在但不是普通对象（数组、字符串、非法结构）：同样回退为空对象。
      addWarning('custom_model_params_missing');
    }
  }

  let architecture = String(hyperparameters.model_architecture || '').trim().toLowerCase();
  if (architecture === 'predrnnv2_sphere') architecture = 'predrnnv2';
  if (modelSource === 'official' && !SUPPORTED_ARCHITECTURES.has(architecture)) {
    addWarning(readFailureWarning);
    architecture = base.modelArchitecture;
  } else if (modelSource !== 'official') {
    architecture = base.modelArchitecture;
  }

  const isRecurrent = isRecurrentArchitecture(architecture);
  let hiddenDims = [...DEFAULT_HIDDEN_DIMS];
  if (isRecurrent) {
    const rawDims = hyperparameters.stlstm_hidden_dims;
    if (Array.isArray(rawDims) && rawDims.length > 0 && rawDims.every((dim) => Number.isInteger(dim) && dim > 0)) {
      hiddenDims = [...rawDims];
    } else if (rawDims !== undefined && rawDims !== null) {
      // 字段存在但格式不对才算异常；旧任务不带该字段时使用默认层宽，不打扰用户。
      addWarning(readFailureWarning);
    }
  }

  const architectureParamsByModel = Object.fromEntries(
    Object.entries(ARCHITECTURE_DEFAULT_PARAMS).map(([modelId, params]) => [modelId, { ...params }])
  );
  if (modelSource === 'official' && !isRecurrent) {
    const structure = readArchitectureParams(hyperparameters, architecture);
    architectureParamsByModel[architecture] = { ...architectureParamsByModel[architecture], ...structure.params };
    if (!structure.complete) addWarning('incomplete_structure');
  }

  const selectedChannels = channelOrder.length > 0
    ? normalizeTaskChannels(task, channelOrder)
    : (Array.isArray(hyperparameters.selected_channels)
        ? hyperparameters.selected_channels.map((channel) => String(channel).toUpperCase())
        : []);

  const epochs = readPositiveInteger(hyperparameters, 'epochs');
  const batchSize = readPositiveInteger(hyperparameters, 'batch_size');
  const windowValue = readPositiveInteger(hyperparameters, 'window');
  const horizon = readPositiveInteger(hyperparameters, 'horizon');
  const learningRate = readPositiveNumber(hyperparameters, 'learning_rate');
  if (!epochs || !batchSize || !windowValue || !horizon || !learningRate) addWarning(readFailureWarning);

  const tagIds = (task.tags || [])
    .map((tag) => Number(tag?.id))
    .filter((id) => Number.isFinite(id));

  return {
    ...base,
    sourceTaskId: task.id ?? null,
    customModelName: buildExperimentCopyName(name, existingNames, nameSuffix),
    // Earth daily and three-hourly tasks keep their registered dataset identity when
    // copied; the page controller still isolates them from the Mars request builder.
    trainingDataset: sanitizeTrainingDataset(trainingDatasetRaw || base.trainingDataset, { allowEarth: true }),
    modelSource,
    selectedUploadedModelId,
    selectedUploadedModelVersion,
    uploadedModelName,
    customModelParams,
    selectedChannels,
    targetVariable: hyperparameters.target_variable || (String(trainingDatasetRaw).startsWith('earth_') ? 'TO3' : 'O3'),
    inputVariables: Array.isArray(hyperparameters.input_variables) ? [...hyperparameters.input_variables]
      : [String(trainingDatasetRaw).startsWith('earth_') ? 'TO3' : 'O3', ...(String(trainingDatasetRaw).startsWith('earth_') ? (hyperparameters.selected_channels || []).filter(c => c !== 'TO3') : selectedChannels)],
    modelArchitecture: architecture,
    useSphere: hyperparameters.use_sphere === true,
    epochs: epochs ?? base.epochs,
    batchSize: batchSize ?? base.batchSize,
    learningRate: learningRate ?? base.learningRate,
    trainRatio: Number.isFinite(Number(hyperparameters.train_ratio)) ? Number(hyperparameters.train_ratio) : base.trainRatio,
    validationRatio: Number.isFinite(Number(hyperparameters.validation_ratio)) ? Number(hyperparameters.validation_ratio) : base.validationRatio,
    testRatio: Number.isFinite(Number(hyperparameters.test_ratio)) ? Number(hyperparameters.test_ratio) : base.testRatio,
    windowValue: windowValue ?? base.windowValue,
    horizon: horizon ?? base.horizon,
    earlyStoppingPatience: sanitizeNonNegativeInteger(
      hyperparameters.early_stopping_patience,
      base.earlyStoppingPatience,
      MAX_EARLY_STOPPING
    ),
    seed: sanitizeNonNegativeInteger(hyperparameters.seed, base.seed, MAX_SEED),
    hiddenDims,
    architectureParamsByModel,
    transferEnabled: false,
    transferSourceType: 'task',
    transferSourceTaskId: '',
    transferFreezeMode: 'none',
    finetuneLearningRate: 0.0001,
    tagIds,
    warnings,
  };
}

/** 复制出的实验不能与原任务同名：依次尝试后缀、后缀 2、后缀 3… */
export function buildExperimentCopyName(name, existingNames = [], suffix = ' (Copy)') {
  const baseName = String(name || '').trim();
  const taken = new Set((Array.isArray(existingNames) ? existingNames : [])
    .map((item) => String(item || '').trim())
    .filter(Boolean));
  if (!baseName) return suffix.trim();
  if (!taken.has(baseName) && !taken.has(`${baseName}${suffix}`)) return `${baseName}${suffix}`;

  for (let index = 2; index < 100; index += 1) {
    const candidate = `${baseName}${suffix} ${index}`;
    if (!taken.has(candidate)) return candidate;
  }
  return `${baseName}${suffix} ${Date.now()}`;
}

function stringifyErrorValue(value) {
  if (value === undefined || value === null) return '';
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) return value.map(stringifyErrorValue).filter(Boolean).join('; ');
  if (isPlainObject(value)) {
    const nested = value.message || value.detail || value.error || value.reason;
    if (nested) return stringifyErrorValue(nested);
    try {
      return JSON.stringify(value);
    } catch {
      return '';
    }
  }
  return String(value);
}

const FAILURE_MESSAGE_KEYS = {
  cuda_out_of_memory: {
    message: 'experimentCenter.failureCudaOom',
    suggestion: 'experimentCenter.failureCudaOomSuggestion',
  },
  invalid_model_artifact: {
    message: 'experimentCenter.failureInvalidArtifact',
    suggestion: 'experimentCenter.failureInvalidArtifactSuggestion',
  },
  stopped_by_user: {
    message: 'experimentCenter.failureStopped',
    suggestion: '',
  },
  training_failed: {
    message: 'experimentCenter.failureGeneric',
    suggestion: 'experimentCenter.failureGenericSuggestion',
  },
};

/**
 * 从任务 metrics 推导失败/停止原因。
 *
 * 后端把失败原因写在 `metrics.error` / `metrics.error_code`，用户停止写 `metrics.note`。
 * 非法 JSON 安全回退；`messageKey` 交给组件用当前语言取文案。
 */
export function getExperimentFailureMessage(task, { stoppedStatuses = ['stopped', 'cancelled'] } = {}) {
  const metrics = parseTaskMetrics(task?.metrics);
  const status = String(task?.status || '').toLowerCase();
  const rawErrorCode = stringifyErrorValue(metrics.error_code).toLowerCase();
  const rawError = stringifyErrorValue(metrics.error);
  const rawNote = stringifyErrorValue(metrics.note);
  const combined = `${rawError} ${rawNote}`.toLowerCase();

  let code = rawErrorCode;
  if (!code) {
    if (/stopped by user|cancelled by user|stopped|cancelled/.test(combined)) code = 'stopped_by_user';
    else if (/cuda out of memory|outofmemoryerror|memory is exhausted/.test(combined)) code = 'cuda_out_of_memory';
    else if (/invalid model artifact|no valid weight|weight file/.test(combined)) code = 'invalid_model_artifact';
    else if (rawError || rawNote) code = 'training_failed';
  }
  if (!code && stoppedStatuses.includes(status)) code = 'stopped_by_user';
  if (!code && status === 'failed') code = 'training_failed';

  const keys = FAILURE_MESSAGE_KEYS[code] || FAILURE_MESSAGE_KEYS.training_failed;
  const hasReadableReason = Boolean(rawError || rawNote || rawErrorCode);

  return {
    code: code || '',
    stopped: code === 'stopped_by_user',
    messageKey: keys.message,
    suggestionKey: keys.suggestion,
    detail: rawError || rawNote,
    errorCode: rawErrorCode,
    hasReason: hasReadableReason || Boolean(code),
  };
}

/** 「用于预测」只对已完成且有有效权重的任务开放。 */
export function isRetiredEarthTask(task) {
  return isRetiredEarthDataset(task?.dataset_id || parseTaskHyperparameters(task?.hyperparameters).training_dataset);
}

export function canUseTaskForPrediction(task) {
  return Boolean(task) && !isRetiredEarthTask(task)
    && String(task.status || '').toLowerCase() === 'completed' && task.model_available === true;
}

export function isKnownExperimentTaskStatus(status) {
  return KNOWN_STATUSES.has(String(status || '').toLowerCase());
}

/** 日志行着色：错误、警告和指标行在目录与监控区共用同一判定。 */
export function getExperimentLogLineTone(line) {
  const value = String(line || '').toLowerCase();
  if (value.includes('traceback') || value.includes('error') || value.includes('failed') || value.includes('exception')) {
    return 'error';
  }
  if (value.includes('warning') || value.includes('warn')) {
    return 'warning';
  }
  if (
    value.includes('epoch') ||
    value.includes('loss') ||
    value.includes('mse') ||
    value.includes('rmse') ||
    value.includes('mae') ||
    value.includes('r-squared')
  ) {
    return 'metric';
  }
  if (value.includes('started') || value.includes('training') || value.includes('[step')) {
    return 'info';
  }
  return 'default';
}

/**
 * 任务实际使用的输入通道。统一训练脚本读取 selected_channels；
 * 旧脚本回退到文件名后缀，保证历史记录仍可显示通道摘要。
 */
export function normalizeTaskChannels(task, channelOrder = []) {
  const hyperparameters = parseTaskHyperparameters(task?.hyperparameters);
  const selected = Array.isArray(hyperparameters.selected_channels) ? hyperparameters.selected_channels : [];
  const selectedSet = new Set(selected.map((channel) => String(channel).toUpperCase()));
  const normalized = channelOrder.filter((channel) => selectedSet.has(channel));
  if (normalized.length > 0 || task?.model_script === 'demo3.py') return normalized;

  const suffix = String(task?.model_script || '').replace('demo3-', '').replace('.py', '');
  const suffixSet = new Set(suffix.split('').map((channel) => channel.toUpperCase()));
  return channelOrder.filter((channel) => suffixSet.has(channel));
}

/** 指标数值格式：缺失显示 --，非法值不渲染成 NaN。 */
export function formatExperimentMetricValue(value) {
  if (value === undefined || value === null || value === '') return '--';
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  if (numeric !== 0 && Math.abs(numeric) < 0.0001) return numeric.toExponential(3);
  if (Math.abs(numeric) >= 1000) return numeric.toFixed(2);
  return numeric.toFixed(4);
}
