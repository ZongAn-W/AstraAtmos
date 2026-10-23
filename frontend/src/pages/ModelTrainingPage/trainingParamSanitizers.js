const DEFAULT_MIN_POSITIVE_NUMBER = 0.000001;
const MAX_SEED = 2147483647;
const OPEN_INTERVAL_FLOAT_FIELDS = new Set(['initial_history_weight', 'initial_translation_weight']);
const THREE_VALUE_INTEGER_LIST_FIELDS = new Set(['patch_size', 'cuboid_size']);
export const RECURRENT_MODEL_ARCHITECTURES = ['predrnnv2', 'predrnnpp', 'convlstm'];
export const TRANSFER_FREEZE_MODES = ['none', 'backbone', 'head'];
export const UPLOADED_MODEL_TRANSFER_FREEZE_MODES = ['none'];
export const TRAINING_DATASET_OPENMARS_MCD = 'openmars_mcd';
export const TRAINING_DATASET_MCD_OVERVIEW = 'mcd_overview';
export const TRAINING_DATASET_EARTH_MERRA2_V2 = 'earth_merra2_daily_v2';
export const TRAINING_DATASET_EARTH_MERRA2_3HOURLY_V1 = 'earth_merra2_3hourly_v1';
export const TRAINING_DATASET_EARTH_MERRA2_3HOURLY_5DEG_V1 = 'earth_merra2_3hourly_5deg_v1';
export const RETIRED_EARTH_DATASET_IDS = ['earth_merra2_daily_v1', TRAINING_DATASET_EARTH_MERRA2_V2];
export function isRetiredEarthDataset(datasetId) {
  return RETIRED_EARTH_DATASET_IDS.includes(String(datasetId || '').trim().toLowerCase());
}
/** Mars 训练身份；Earth 走独立的数据准备与训练路径，不进入旧 runner。 */
export const MARS_TRAINING_DATASET_IDS = [TRAINING_DATASET_OPENMARS_MCD, TRAINING_DATASET_MCD_OVERVIEW];
export const EARTH_TRAINING_DATASET_IDS = [
  TRAINING_DATASET_EARTH_MERRA2_3HOURLY_V1,
  TRAINING_DATASET_EARTH_MERRA2_3HOURLY_5DEG_V1,
];
/** 注册表已登记、训练页可选的全部数据集（顺序即选项顺序）。 */
export const TRAINING_DATASET_IDS = [
  TRAINING_DATASET_OPENMARS_MCD,
  TRAINING_DATASET_MCD_OVERVIEW,
  TRAINING_DATASET_EARTH_MERRA2_3HOURLY_V1,
  TRAINING_DATASET_EARTH_MERRA2_3HOURLY_5DEG_V1,
];

export const MODEL_STRUCTURE_PARAM_CONFIG = {
  simvp: [
    { key: 'spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'temporal_depth', defaultValue: 2, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  dlinear: [
    { key: 'linear_hidden_layers', defaultValue: 2, type: 'integer' },
  ],
  informer: [
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'n_heads', defaultValue: 2, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  autoformer: [
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'n_heads', defaultValue: 2, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'moving_avg', defaultValue: 3, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  patchtst: [
    { key: 'patch_len', defaultValue: 2, type: 'integer' },
    { key: 'stride', defaultValue: 1, type: 'integer' },
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'n_heads', defaultValue: 2, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  timemixer: [
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'moving_avg', defaultValue: 3, type: 'integer' },
    { key: 'down_sampling_window', defaultValue: 2, type: 'integer' },
    { key: 'down_sampling_layers', defaultValue: 1, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  timexer: [
    { key: 'patch_len', defaultValue: 2, type: 'integer' },
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'n_heads', defaultValue: 2, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  tsmixer: [
    { key: 'hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  crossformer: [
    { key: 'seg_len', defaultValue: 1, type: 'integer' },
    { key: 'win_size', defaultValue: 2, type: 'integer' },
    { key: 'factor', defaultValue: 2, type: 'integer' },
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'n_heads', defaultValue: 2, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  earthformer: [
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'n_heads', defaultValue: 2, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'patch_size', defaultValue: [1, 2, 2], type: 'integerList' },
    { key: 'cuboid_size', defaultValue: [1, 2, 2], type: 'integerList' },
    { key: 'mlp_ratio', defaultValue: 2, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  etsformer: [
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'top_k_freq', defaultValue: 2, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  fedformer: [
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'moving_avg', defaultValue: 3, type: 'integer' },
    { key: 'modes', defaultValue: 4, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  itransformer: [
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'n_heads', defaultValue: 2, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  mau: [
    { key: 'hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'tau', defaultValue: 2, type: 'integer' },
    { key: 'kernel_size', defaultValue: 3, type: 'integer' },
    { key: 'gamma', defaultValue: 1.0, type: 'nonNegativeNumber' },
  ],
  nbeats: [
    { key: 'stack_count', defaultValue: 1, type: 'integer' },
    { key: 'blocks_per_stack', defaultValue: 1, type: 'integer' },
    { key: 'hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  nhits: [
    { key: 'stack_count', defaultValue: 1, type: 'integer' },
    { key: 'blocks_per_stack', defaultValue: 1, type: 'integer' },
    { key: 'hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'pooling_sizes', defaultValue: [1], type: 'integerList' },
    { key: 'downsample_factors', defaultValue: [1], type: 'integerList' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  pyraformer: [
    { key: 'd_model', defaultValue: 64, type: 'integer' },
    { key: 'n_heads', defaultValue: 2, type: 'integer' },
    { key: 'e_layers', defaultValue: 1, type: 'integer' },
    { key: 'd_ff', defaultValue: 128, type: 'integer' },
    { key: 'pooling_sizes', defaultValue: [1, 2], type: 'integerList' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  rnn_cnn_rnn: [
    { key: 'spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'cnn_depth', defaultValue: 1, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  cnn_rnn_cnn_rnn_cnn: [
    { key: 'spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'cnn_depth', defaultValue: 1, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  simvp_3dconv: [
    { key: 'spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'num_downsample', defaultValue: 1, type: 'integer' },
    { key: 'temporal_depth', defaultValue: 1, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  simvp_hybrid3d: [
    { key: 'spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'num_downsample', defaultValue: 1, type: 'integer' },
    { key: 'temporal_depth', defaultValue: 1, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  convlstm_mst: [
    { key: 'hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'mst_spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'mst_temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'mst_num_downsample', defaultValue: 1, type: 'integer' },
    { key: 'mst_temporal_depth', defaultValue: 1, type: 'integer' },
    { key: 'kernel_size', defaultValue: 3, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  dlinear_mst: [
    { key: 'linear_hidden_layers', defaultValue: 2, type: 'integer' },
    { key: 'mst_spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'mst_temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'mst_num_downsample', defaultValue: 1, type: 'integer' },
    { key: 'mst_temporal_depth', defaultValue: 1, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  convlstm_phase_gated_mst: [
    { key: 'hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'phase_context_dim', defaultValue: 8, type: 'integer' },
    { key: 'mst_spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'mst_temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'mst_num_downsample', defaultValue: 1, type: 'integer' },
    { key: 'mst_temporal_depth', defaultValue: 1, type: 'integer' },
    { key: 'kernel_size', defaultValue: 3, type: 'integer' },
    { key: 'initial_history_weight', defaultValue: 0.7, type: 'boundedFloat' },
    { key: 'initial_translation_weight', defaultValue: 0.7, type: 'boundedFloat' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  convlstm_mst_feature_refiner: [
    { key: 'hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'mst_spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'mst_temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'mst_num_downsample', defaultValue: 1, type: 'integer' },
    { key: 'mst_temporal_depth', defaultValue: 1, type: 'integer' },
    { key: 'kernel_size', defaultValue: 3, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
  convlstm_climatology_anomaly: [
    { key: 'hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'climatology_hidden_dim', defaultValue: 8, type: 'integer' },
    { key: 'mst_spatial_hidden_dim', defaultValue: 64, type: 'integer' },
    { key: 'mst_temporal_hidden_dim', defaultValue: 128, type: 'integer' },
    { key: 'mst_num_downsample', defaultValue: 1, type: 'integer' },
    { key: 'mst_temporal_depth', defaultValue: 1, type: 'integer' },
    { key: 'kernel_size', defaultValue: 3, type: 'integer' },
    { key: 'dropout', defaultValue: 0.1, type: 'dropout' },
  ],
};

export const MODEL_STRUCTURE_PARAM_LABELS = {
  en: {
    spatial_hidden_dim: 'Spatial Hidden Dim',
    temporal_hidden_dim: 'Temporal Hidden Dim',
    temporal_depth: 'Temporal Depth',
    linear_hidden_layers: 'Linear Hidden Layers',
    d_model: 'Model Dim',
    n_heads: 'Attention Heads',
    e_layers: 'Encoder Layers',
    d_ff: 'FFN Dim',
    patch_len: 'Patch Length',
    stride: 'Stride',
    moving_avg: 'Moving Avg',
    down_sampling_window: 'Down Sampling Window',
    down_sampling_layers: 'Down Sampling Layers',
    hidden_dim: 'Hidden Dim',
    seg_len: 'Segment Length',
    win_size: 'Window Size',
    factor: 'Router Factor',
    num_downsample: 'Downsample Blocks',
    cnn_depth: 'CNN Depth',
    mst_spatial_hidden_dim: 'MST Spatial Hidden Dim',
    mst_temporal_hidden_dim: 'MST Temporal Hidden Dim',
    mst_num_downsample: 'MST Downsample Blocks',
    mst_temporal_depth: 'MST Temporal Depth',
    phase_context_dim: 'Phase Context Dim',
    climatology_hidden_dim: 'Climatology Hidden Dim',
    kernel_size: 'Kernel Size',
    tau: 'Tau',
    stack_count: 'Stack Count',
    blocks_per_stack: 'Blocks Per Stack',
    top_k_freq: 'Top-K Frequency',
    modes: 'Frequency Modes',
    mlp_ratio: 'MLP Ratio',
    gamma: 'Gamma',
    initial_history_weight: 'Initial History Weight',
    initial_translation_weight: 'Initial Translation Weight',
    patch_size: 'Patch Size',
    cuboid_size: 'Cuboid Size',
    pooling_sizes: 'Pooling Sizes',
    downsample_factors: 'Downsample Factors',
    dropout: 'Dropout',
  },
  zh: {
    spatial_hidden_dim: '空间隐藏维度',
    temporal_hidden_dim: '时间隐藏维度',
    temporal_depth: '时间深度',
    linear_hidden_layers: '线性隐藏层数',
    d_model: '模型维度',
    n_heads: '注意力头数',
    e_layers: '编码器层数',
    d_ff: '前馈维度',
    patch_len: 'Patch 长度',
    stride: '步长',
    moving_avg: '移动平均窗口',
    down_sampling_window: '下采样窗口',
    down_sampling_layers: '下采样层数',
    hidden_dim: '隐藏维度',
    seg_len: '分段长度',
    win_size: '窗口大小',
    factor: '路由因子',
    num_downsample: '下采样块数',
    cnn_depth: 'CNN 深度',
    mst_spatial_hidden_dim: 'MST 空间隐藏维度',
    mst_temporal_hidden_dim: 'MST 时间隐藏维度',
    mst_num_downsample: 'MST 下采样块数',
    mst_temporal_depth: 'MST 时间深度',
    phase_context_dim: '相位上下文维度',
    climatology_hidden_dim: '气候态隐藏维度',
    kernel_size: '卷积核大小',
    tau: 'Tau',
    stack_count: '堆叠数',
    blocks_per_stack: '每堆块数',
    top_k_freq: 'Top-K 频率',
    modes: '频率模式数',
    mlp_ratio: 'MLP 比率',
    gamma: 'Gamma',
    initial_history_weight: '初始历史权重',
    initial_translation_weight: '初始翻译权重',
    patch_size: 'Patch 尺寸',
    cuboid_size: 'Cuboid 尺寸',
    pooling_sizes: '池化尺寸',
    downsample_factors: '下采样因子',
    dropout: 'Dropout',
  },
};

function isBlank(value) {
  return value === '' || value === null || value === undefined;
}

export function sanitizePositiveInteger(value, fallback, min = 1, max = Number.POSITIVE_INFINITY) {
  if (isBlank(value)) return fallback;

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;

  const rounded = Math.round(parsed);
  if (rounded < min) return min;
  if (rounded > max) return max;
  return rounded;
}

export function sanitizeNonNegativeInteger(value, fallback, max = Number.POSITIVE_INFINITY) {
  if (isBlank(value)) return fallback;

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;

  const rounded = Math.round(parsed);
  if (rounded < 0) return 0;
  if (rounded > max) return max;
  return rounded;
}

export function sanitizePositiveNumber(
  value,
  fallback,
  min = DEFAULT_MIN_POSITIVE_NUMBER,
  max = Number.POSITIVE_INFINITY
) {
  if (isBlank(value)) return fallback;

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  if (parsed < min) return min;
  if (parsed > max) return max;
  return parsed;
}

export function sanitizeDropout(value, fallback = 0.1) {
  if (isBlank(value)) return fallback;

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  if (parsed < 0) return 0;
  if (parsed > 0.9) return 0.9;
  return parsed;
}

export function sanitizePositiveIntegerList(value, fallback = []) {
  if (isBlank(value)) return Array.isArray(fallback) ? [...fallback] : [];

  const items = Array.isArray(value)
    ? value
    : String(value).trim().replace(/^\[/, '').replace(/\]$/, '').replace(/;/g, ',').split(',');
  const parsed = items
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item))
    .map((item) => Math.round(item))
    .filter((item) => item >= 1);

  return parsed.length > 0 ? parsed : Array.isArray(fallback) ? [...fallback] : [];
}

export function sanitizeFixedPositiveIntegerList(value, fallback = [], length = 3) {
  const parsed = sanitizePositiveIntegerList(value, fallback).slice(0, length);
  const fallbackItems = Array.isArray(fallback) && fallback.length > 0 ? fallback : Array(length).fill(1);

  while (parsed.length < length) {
    parsed.push(fallbackItems[parsed.length] ?? fallbackItems[fallbackItems.length - 1] ?? 1);
  }

  return parsed;
}

export function sanitizeEarthformerSizeList(value, fallback = [1, 2, 2]) {
  const normalized = sanitizeFixedPositiveIntegerList(value, fallback, 3);
  normalized[0] = 1;
  return normalized;
}

export function sanitizeTransferFreezeMode(value) {
  const normalized = String(value || '').toLowerCase();
  return TRANSFER_FREEZE_MODES.includes(normalized) ? normalized : 'none';
}

export function getTransferFreezeModes(modelSource) {
  return String(modelSource || '').toLowerCase() === 'uploaded'
    ? UPLOADED_MODEL_TRANSFER_FREEZE_MODES
    : TRANSFER_FREEZE_MODES;
}

export function sanitizeTransferFreezeModeForModelSource(value, modelSource) {
  const normalized = sanitizeTransferFreezeMode(value);
  return getTransferFreezeModes(modelSource).includes(normalized) ? normalized : 'none';
}

/**
 * 规范化训练数据集。
 *
 * 默认只接受火星数据集：通用构建器服务于旧 runner，不能因为 Earth 出现在选项
 * 列表里就把 Earth 身份静默带进火星请求。Earth 路径显式传入 allowEarth。
 */
export function sanitizeTrainingDataset(value, { allowEarth = false } = {}) {
  const normalized = String(value || TRAINING_DATASET_OPENMARS_MCD).toLowerCase();
  const allowed = allowEarth ? TRAINING_DATASET_IDS : MARS_TRAINING_DATASET_IDS;
  return allowed.includes(normalized) ? normalized : TRAINING_DATASET_OPENMARS_MCD;
}

export function isRecurrentArchitecture(modelArchitecture) {
  return RECURRENT_MODEL_ARCHITECTURES.includes(String(modelArchitecture || '').toLowerCase());
}

export function getModelStructureConfig(modelArchitecture) {
  return MODEL_STRUCTURE_PARAM_CONFIG[String(modelArchitecture || '').toLowerCase()] || [];
}

export function getModelStructureParamLabel(key, language = 'en') {
  const locale = String(language || '').toLowerCase().startsWith('en') ? 'en' : 'zh';
  return MODEL_STRUCTURE_PARAM_LABELS[locale]?.[key] || MODEL_STRUCTURE_PARAM_LABELS.en[key] || key;
}

export function createDefaultArchitectureParamsByModel() {
  return Object.fromEntries(
    Object.entries(MODEL_STRUCTURE_PARAM_CONFIG).map(([modelId, config]) => [
      modelId,
      Object.fromEntries(config.map((field) => [field.key, field.defaultValue])),
    ])
  );
}

function sanitizeArchitectureParam(value, field) {
  if (field.type === 'dropout') return sanitizeDropout(value, field.defaultValue);
  if (field.type === 'integerList') {
    return THREE_VALUE_INTEGER_LIST_FIELDS.has(field.key)
      ? sanitizeEarthformerSizeList(value, field.defaultValue)
      : sanitizePositiveIntegerList(value, field.defaultValue);
  }
  if (field.type === 'boundedFloat') {
    const minimum = OPEN_INTERVAL_FLOAT_FIELDS.has(field.key) ? DEFAULT_MIN_POSITIVE_NUMBER : 0;
    return sanitizePositiveNumber(value, field.defaultValue, minimum, 0.9);
  }
  if (field.type === 'nonNegativeNumber') return sanitizePositiveNumber(value, field.defaultValue, 0);
  return sanitizePositiveInteger(value, field.defaultValue);
}

export function buildTrainingHyperparameters({
  epochs,
  batchSize,
  learningRate,
  hiddenDims,
  windowValue,
  horizon,
  earlyStoppingPatience,
  seed,
  selectedChannels,
  channelOrder,
  modelArchitecture,
  modelSource = 'official',
  useSphere,
  architectureParamsByModel = {},
  transferLearning = null,
  trainingDataset = TRAINING_DATASET_OPENMARS_MCD,
  trainRatio = 0.7,
  validationRatio = 0.2,
  testRatio = 0.1,
}) {
  const normalizedArchitecture = String(modelArchitecture || '').toLowerCase();
  const hyperparameters = {
    epochs: sanitizePositiveInteger(epochs, 10),
    batch_size: sanitizePositiveInteger(batchSize, 32),
    learning_rate: sanitizePositiveNumber(learningRate, 0.001),
    training_dataset: sanitizeTrainingDataset(trainingDataset),
    window: sanitizePositiveInteger(windowValue, 3),
    horizon: sanitizePositiveInteger(horizon, 3),
    early_stopping_patience: sanitizeNonNegativeInteger(earlyStoppingPatience, 0, 200),
    seed: sanitizeNonNegativeInteger(seed, 11, MAX_SEED),
    selected_channels: channelOrder.filter((channel) => selectedChannels.includes(channel)),
    model_architecture: normalizedArchitecture,
    use_sphere: Boolean(useSphere),
    train_ratio: Number(trainRatio),
    validation_ratio: Number(validationRatio),
    test_ratio: Number(testRatio),
  };

  const transferEnabled = Boolean(transferLearning?.enabled);
  if (transferEnabled) {
    const sourceType = String(transferLearning?.sourceType || 'task').toLowerCase() === 'upload'
      ? 'upload'
      : 'task';
    hyperparameters.transfer_learning = true;
    hyperparameters.transfer_source_type = sourceType;
    hyperparameters.transfer_source_task_id = sanitizeNonNegativeInteger(
      transferLearning?.sourceTaskId,
      0,
      MAX_SEED
    );
    hyperparameters.transfer_weight_id = String(transferLearning?.weightId || '').trim();
    hyperparameters.transfer_load_mode = 'strict';
    hyperparameters.freeze_mode = sanitizeTransferFreezeModeForModelSource(
      transferLearning?.freezeMode,
      modelSource
    );
    hyperparameters.finetune_learning_rate = sanitizePositiveNumber(
      transferLearning?.finetuneLearningRate,
      sanitizePositiveNumber(learningRate, 0.001) * 0.1
    );
  }

  if (isRecurrentArchitecture(normalizedArchitecture)) {
    const recurrentHiddenDims = Array.isArray(hiddenDims) ? hiddenDims : [64, 64, 64];
    hyperparameters.stlstm_hidden_dims = recurrentHiddenDims.map((dim) => sanitizePositiveInteger(dim, 64));
    return hyperparameters;
  }

  const selectedParams = architectureParamsByModel?.[normalizedArchitecture] || {};
  for (const field of getModelStructureConfig(normalizedArchitecture)) {
    hyperparameters[field.key] = sanitizeArchitectureParam(selectedParams[field.key], field);
  }

  return hyperparameters;
}
