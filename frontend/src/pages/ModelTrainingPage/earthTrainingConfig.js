/**
 * Earth MERRA-2 训练配置的纯函数。
 *
 * 这里集中 Earth 的固定契约（数据集身份、通道顺序、频率对应的输入/输出窗口、单位），
 * 让训练页与测试共用同一份定义，而不是在 JSX 里散落字面量。所有函数都不依赖
 * React，也不发请求：请求构造与校验可以单独测试。
 *
 * Earth 训练首期只支持官方 DLinear，TO3 必选、四个辅助变量可选；数据集版本、
 * 发布指纹与网格由服务端 registry 提供，这里不复制一份。
 */

import {
  TRAINING_DATASET_EARTH_MERRA2_3HOURLY_V1,
  TRAINING_DATASET_EARTH_MERRA2_3HOURLY_5DEG_V1,
  isRetiredEarthDataset,
  sanitizeNonNegativeInteger,
  sanitizePositiveInteger,
  sanitizePositiveNumber,
} from './trainingParamSanitizers.js';
import { isUploadedModelValidationTimeout, UPLOADED_MODEL_VALIDATION_TIMEOUT } from './uploadedModelValidation.js';

export const EARTH_3HOURLY_DATASET_ID = TRAINING_DATASET_EARTH_MERRA2_3HOURLY_V1;
export const EARTH_3HOURLY_5DEG_DATASET_ID = TRAINING_DATASET_EARTH_MERRA2_3HOURLY_5DEG_V1;
export const EARTH_DATASET_ID = EARTH_3HOURLY_DATASET_ID;
export const EARTH_3HOURLY_UPLOAD_SCHEMA = 'astraatmos_earth_3hourly_uploaded_model_v1';
export const EARTH_3HOURLY_5DEG_SCHEMA = 'astraatmos_earth_3hourly_5deg_model_v1';
export const EARTH_3HOURLY_FULL_GRID_SCHEMA = 'astraatmos_earth_3hourly_fullgrid_model_v2';
export const EARTH_MODEL_ARCHITECTURE = 'dlinear';
/** 上传模型的架构标记；真正的代码由服务端固定的模型引用决定。 */
export const EARTH_UPLOADED_ARCHITECTURE = 'uploaded';
/** Earth 支持两种模型来源：官方 DLinear 与用户上传模型。 */
export const EARTH_MODEL_SOURCE_OFFICIAL = 'official';
export const EARTH_MODEL_SOURCE_UPLOADED = 'uploaded';
export const EARTH_MODEL_SOURCES = [EARTH_MODEL_SOURCE_OFFICIAL, EARTH_MODEL_SOURCE_UPLOADED];
export const EARTH_MODEL_SOURCE = EARTH_MODEL_SOURCE_OFFICIAL;
export const EARTH_WINDOW = 56;
export const EARTH_HORIZON = 24;
export const EARTH_3HOURLY_WINDOW = 56;
export const EARTH_3HOURLY_HORIZON = 24;
export const EARTH_3HOURLY_FREQUENCY_HOURS = 3;
export const EARTH_3HOURLY_GRID_SHAPE = [240, 480];
export const EARTH_DAILY_GRID_SHAPE = [36, 72];
export const EARTH_DEFAULT_SPLIT_RATIOS = {
  train_ratio: 0.7,
  validation_ratio: 0.2,
  test_ratio: 0.1,
};
export const EARTH_TARGET_CHANNEL = 'TO3';
export const EARTH_TARGET_UNIT = 'DU';

export const EARTH_DATASET_PROFILES = Object.freeze({
  [EARTH_3HOURLY_5DEG_DATASET_ID]: Object.freeze({
    datasetId: EARTH_3HOURLY_5DEG_DATASET_ID, frequencyHours: 3, stepUnit: 'hour', step: 3,
    window: 56, horizon: 24, gridShape: [36, 72], modelSources: EARTH_MODEL_SOURCES,
  }),
  [EARTH_3HOURLY_DATASET_ID]: Object.freeze({
    datasetId: EARTH_3HOURLY_DATASET_ID,
    frequencyHours: EARTH_3HOURLY_FREQUENCY_HOURS,
    stepUnit: 'hour',
    step: 3,
    window: EARTH_3HOURLY_WINDOW,
    horizon: EARTH_3HOURLY_HORIZON,
    gridShape: EARTH_3HOURLY_GRID_SHAPE,
    modelSources: EARTH_MODEL_SOURCES,
  }),
});

export function getEarthTrainingProfile(datasetId = EARTH_DATASET_ID) {
  if (isRetiredEarthDataset(datasetId)) throw new Error('dataset_retired');
  const profile = EARTH_DATASET_PROFILES[String(datasetId || '').toLowerCase()]
    || EARTH_DATASET_PROFILES[EARTH_DATASET_ID];
  return {
    ...profile,
    gridShape: [...profile.gridShape],
    modelSources: [...profile.modelSources],
    // Keep descriptor-style aliases available to upload/checkpoint UI adapters.
    frequency_hours: profile.frequencyHours,
    step_unit: profile.stepUnit,
    grid_shape: [...profile.gridShape],
  };
}

/** 规范通道顺序：TO3 固定第一，辅助变量按此顺序进入模型。 */
export const EARTH_CHANNEL_ORDER = ['TO3', 'U10M', 'V10M', 'T2M', 'SWGDN'];
export const EARTH_OPTIONAL_CHANNELS = ['U10M', 'V10M', 'T2M', 'SWGDN'];

export const EARTH_CHANNEL_META = {
  TO3: { name: 'Total column ozone', short: 'TO3', unit: 'DU', locked: true },
  U10M: { name: '10 m eastward wind', short: 'U10M', unit: 'm s-1' },
  V10M: { name: '10 m northward wind', short: 'V10M', unit: 'm s-1' },
  T2M: { name: '2 m air temperature', short: 'T2M', unit: 'K' },
  SWGDN: { name: 'Surface incoming shortwave flux', short: 'SWGDN', unit: 'W m-2' },
};

export const EARTH_PARAM_BOUNDS = {
  window: { min: 1, max: 240, fallback: 56 },
  horizon: { min: 1, max: 240, fallback: 24 },
  epochs: { min: 1, max: 1000, fallback: 10 },
  batch_size: { min: 1, max: 64, fallback: 8 },
  seed: { min: 0, max: 2147483647, fallback: 11 },
  early_stopping_patience: { min: 0, max: 200, fallback: 0 },
  linear_hidden_layers: { min: 1, max: 4, fallback: 2 },
};

export function normalizeEarthSplitRatios(trainRatio, validationRatio, testRatio, { allowEmptyValidation = false } = {}) {
  const values = [trainRatio, validationRatio, testRatio].map((value) =>
    ((typeof value === 'number' || (typeof value === 'string' && value.trim() !== ''))
      ? Number(value) : NaN));
  const total = values.reduce((sum, value) => sum + value, 0);
  return {
    train_ratio: values[0],
    validation_ratio: values[1],
    test_ratio: values[2],
    valid: values[0] > 0 && (allowEmptyValidation ? values[1] >= 0 : values[1] > 0) && values[2] > 0 && values.every((value) => value < 1) && Math.abs(total - 1) <= 1e-6,
    total,
  };
}

export function isEarthTrainingDataset(datasetId) {
  return Object.hasOwn(EARTH_DATASET_PROFILES, String(datasetId || '').toLowerCase());
}

/** 按规范顺序返回选中的辅助通道，去重并丢弃非法项。 */
export function normalizeEarthSelectedChannels(selectedChannels) {
  const requested = new Set(
    (Array.isArray(selectedChannels) ? selectedChannels : [])
      .map((value) => String(value || '').trim().toUpperCase())
      .filter((value) => EARTH_OPTIONAL_CHANNELS.includes(value)),
  );
  return EARTH_OPTIONAL_CHANNELS.filter((channel) => requested.has(channel));
}

export function getEarthChannelOptions() {
  return EARTH_OPTIONAL_CHANNELS.map((channel) => ({
    channel,
    ...EARTH_CHANNEL_META[channel],
  }));
}

/**
 * 构造 Earth 训练超参数。
 *
 * 只发送服务端白名单字段：不发送 dataset_version / dataset_fingerprint /
 * dataset_snapshot（由服务端生成），也不发送迁移字段。选择上传模型时只发送
 * `model_source` 与自定义参数值；模型 ID/版本/内容哈希与快照由服务端根据
 * 上传记录生成并固定。
 */
export function buildEarthTrainingHyperparameters({
  datasetId = EARTH_DATASET_ID,
  selectedChannels = EARTH_OPTIONAL_CHANNELS,
  windowValue,
  horizon,
  epochs,
  batchSize,
  learningRate,
  seed,
  earlyStoppingPatience,
  linearHiddenLayers,
  modelSource = EARTH_MODEL_SOURCE_OFFICIAL,
  customModelParams = null,
  trainRatio = EARTH_DEFAULT_SPLIT_RATIOS.train_ratio,
  validationRatio = EARTH_DEFAULT_SPLIT_RATIOS.validation_ratio,
  testRatio = EARTH_DEFAULT_SPLIT_RATIOS.test_ratio,
} = {}) {
  const profile = getEarthTrainingProfile(datasetId);
  // Preserve an explicitly requested source so an unsupported three-hourly upload
  // fails with the server's structured configuration error instead of silently
  // becoming an official run.
  const requestedSource = modelSource === EARTH_MODEL_SOURCE_UPLOADED
    ? EARTH_MODEL_SOURCE_UPLOADED
    : EARTH_MODEL_SOURCE_OFFICIAL;
  const uploaded = requestedSource === EARTH_MODEL_SOURCE_UPLOADED;
  const ratios = normalizeEarthSplitRatios(trainRatio, validationRatio, testRatio);
  if (!ratios.valid) throw new Error('Earth ratios must all be positive and total 100%');
  const hyperparameters = {
    train_ratio: ratios.train_ratio,
    validation_ratio: ratios.validation_ratio,
    test_ratio: ratios.test_ratio,
    training_dataset: profile.datasetId,
    model_architecture: uploaded ? EARTH_UPLOADED_ARCHITECTURE : EARTH_MODEL_ARCHITECTURE,
    model_source: requestedSource,
    window: sanitizePositiveInteger(windowValue, profile.window, EARTH_PARAM_BOUNDS.window.min, EARTH_PARAM_BOUNDS.window.max),
    horizon: sanitizePositiveInteger(horizon, profile.horizon, EARTH_PARAM_BOUNDS.horizon.min, EARTH_PARAM_BOUNDS.horizon.max),
    use_sphere: false,
    transfer_learning: false,
    selected_channels: normalizeEarthSelectedChannels(selectedChannels),
    // 上限必须与服务端契约一致：超界值在此夹到文档范围，而不是留给服务端 422。
    epochs: sanitizePositiveInteger(epochs, EARTH_PARAM_BOUNDS.epochs.fallback, EARTH_PARAM_BOUNDS.epochs.min, EARTH_PARAM_BOUNDS.epochs.max),
    batch_size: sanitizePositiveInteger(batchSize, EARTH_PARAM_BOUNDS.batch_size.fallback, EARTH_PARAM_BOUNDS.batch_size.min, EARTH_PARAM_BOUNDS.batch_size.max),
    learning_rate: sanitizePositiveNumber(learningRate, 0.001, 0.000001, 1),
    seed: sanitizeNonNegativeInteger(seed, EARTH_PARAM_BOUNDS.seed.fallback, EARTH_PARAM_BOUNDS.seed.max),
    early_stopping_patience: sanitizeNonNegativeInteger(earlyStoppingPatience, 0, EARTH_PARAM_BOUNDS.early_stopping_patience.max),
  };
  if (uploaded) {
    // 官方模型没有 linear_hidden_layers 这个开关；上传模型由源码自己决定结构。
    hyperparameters.custom_model_params = customModelParams && typeof customModelParams === 'object'
      ? { ...customModelParams }
      : {};
  } else {
    hyperparameters.linear_hidden_layers = sanitizePositiveInteger(
      linearHiddenLayers,
      EARTH_PARAM_BOUNDS.linear_hidden_layers.fallback,
      EARTH_PARAM_BOUNDS.linear_hidden_layers.min,
      EARTH_PARAM_BOUNDS.linear_hidden_layers.max,
    );
  }
  return hyperparameters;
}

/**
 * 读取某个上传模型的 Earth 兼容性结论。
 *
 * 结论来自服务端（上传校验时的 Earth dry-run），前端只负责展示：Mars 可用不等于
 * Earth 可用，缺少结论一律按不可用处理并说明原因。
 */
export function readEarthUploadedModelCompatibility(compatibility) {
  if (!compatibility || typeof compatibility !== 'object') {
    return { known: false, compatible: false, reason: 'earth_compatibility_unknown' };
  }
  const reasons = Array.isArray(compatibility.reasons) ? compatibility.reasons.filter(Boolean) : [];
  const timedOut = isUploadedModelValidationTimeout(compatibility);
  const contracts = compatibility.contract || compatibility.earth_contract
    || ([EARTH_3HOURLY_DATASET_ID, EARTH_3HOURLY_5DEG_DATASET_ID].includes(compatibility.dataset_id)
      || (!compatibility.dataset_id && compatibility.datasets?.earth_merra2_3hourly_v1)
      ? (compatibility.datasets?.[compatibility.dataset_id] || compatibility.datasets?.earth_merra2_3hourly_v1) : compatibility.datasets?.earth_merra2) || {};
  return {
    known: !timedOut && compatibility.status !== 'unknown',
    compatible: !timedOut && compatibility.compatible === true && compatibility.status !== 'unknown',
    reason: timedOut ? UPLOADED_MODEL_VALIDATION_TIMEOUT : reasons[0] || null,
    reasons,
    warnings: Array.isArray(compatibility.warnings) ? compatibility.warnings : [],
    outputShape: Array.isArray(compatibility.output_shape) ? compatibility.output_shape : null,
    declaresEarthFeed: compatibility.declares_earth_feed === true,
    contractSchema: compatibility.contract_schema || contracts.schema || null,
    dataset: compatibility.dataset_id || compatibility.dataset || compatibility.training_dataset || contracts.dataset || contracts.training_dataset || null,
    frequencyHours: Number.isFinite(Number(compatibility.frequency_hours ?? contracts.frequency_hours)) ? Number(compatibility.frequency_hours ?? contracts.frequency_hours) : null,
    gridShape: Array.isArray(compatibility.grid_shape) ? compatibility.grid_shape : (Array.isArray(contracts.grid_shape) ? contracts.grid_shape : null),
    window: Number.isFinite(Number(compatibility.window ?? contracts.window)) ? Number(compatibility.window ?? contracts.window) : null,
    horizon: Number.isFinite(Number(compatibility.horizon ?? contracts.horizon)) ? Number(compatibility.horizon ?? contracts.horizon) : null,
    windows: Array.isArray(contracts.window) ? contracts.window : null,
    horizons: Array.isArray(contracts.horizon) ? contracts.horizon : null,
  };
}

/** 当前选中的上传模型是否可用于 Earth 训练；返回阻塞原因。 */
export function getEarthUploadedSelectionBlocker({ modelSource, uploadedModelId, compatibility, datasetId = EARTH_DATASET_ID,
  windowValue, horizon, batchSize } = {}) {
  if (modelSource !== EARTH_MODEL_SOURCE_UPLOADED) return null;
  const profile = getEarthTrainingProfile(datasetId);
  if (!profile.modelSources.includes(EARTH_MODEL_SOURCE_UPLOADED)) return 'dataset_training_configuration_not_supported';
  if (!uploadedModelId) return 'uploaded_model_required';
  const verdict = readEarthUploadedModelCompatibility(compatibility);
  if (compatibility?.package_id && compatibility.package_id !== uploadedModelId) return 'earth_compatibility_unknown';
  if (verdict.reason === UPLOADED_MODEL_VALIDATION_TIMEOUT) return UPLOADED_MODEL_VALIDATION_TIMEOUT;
  if (!verdict.known) return 'earth_compatibility_unknown';
  if (isEarthTrainingDataset(profile.datasetId) && verdict.dataset !== profile.datasetId) return 'earth_compatibility_unknown';
  if (!verdict.compatible) return verdict.reason || 'uploaded_model_not_earth_compatible';
  if (isEarthTrainingDataset(profile.datasetId)
    && (compatibility.status !== 'available' || !(profile.datasetId === EARTH_3HOURLY_5DEG_DATASET_ID ? [EARTH_3HOURLY_5DEG_SCHEMA] : [EARTH_3HOURLY_UPLOAD_SCHEMA, EARTH_3HOURLY_FULL_GRID_SCHEMA]).includes(verdict.contractSchema))) return 'earth_compatibility_unknown';
  if (verdict.contractSchema === EARTH_3HOURLY_FULL_GRID_SCHEMA
      && batchSize !== undefined && (!Number.isInteger(Number(batchSize))
        || Number(batchSize) < EARTH_PARAM_BOUNDS.batch_size.min
        || Number(batchSize) > EARTH_PARAM_BOUNDS.batch_size.max)) return 'earth_full_grid_batch_invalid';
  const mismatches = [];
  if (verdict.dataset && verdict.dataset !== profile.datasetId) mismatches.push('checkpoint_dataset_incompatible');
  if (verdict.frequencyHours !== null && verdict.frequencyHours !== profile.frequencyHours) mismatches.push('checkpoint_frequency_incompatible');
  if (verdict.gridShape && JSON.stringify(verdict.gridShape) !== JSON.stringify(profile.gridShape)) mismatches.push('checkpoint_grid_incompatible');
  const requestedWindow = windowValue === undefined ? profile.window : Number(windowValue);
  const requestedHorizon = horizon === undefined ? profile.horizon : Number(horizon);
  if (verdict.windows ? !verdict.windows.includes(requestedWindow)
    : verdict.window !== null && verdict.window !== requestedWindow) mismatches.push('checkpoint_window_incompatible');
  if (verdict.horizons ? !verdict.horizons.includes(requestedHorizon)
    : verdict.horizon !== null && verdict.horizon !== requestedHorizon) mismatches.push('checkpoint_horizon_incompatible');
  if (mismatches.length) return mismatches[0];
  return null;
}

/**
 * 从 registry 描述符读取 Earth 训练可用性。
 *
 * 入口已接通（capabilities.training）与数据是否齐备（availability）是两件事：
 * 缺包时仍展示选项与原因，只是不允许提交。
 */
export function readEarthDatasetAvailability(descriptor) {
  if (!descriptor) {
    return { present: false, selectable: false, reason: 'catalog_missing' };
  }
  const available = descriptor.availability === 'available';
  const wired = !isRetiredEarthDataset(descriptor.dataset_id) && descriptor.capabilities?.training !== false;
  return {
    present: true,
    datasetId: descriptor.dataset_id || null,
    selectable: available && wired,
    availability: descriptor.availability,
    reason: isRetiredEarthDataset(descriptor.dataset_id) ? 'dataset_retired'
      : descriptor.availability_reason || (wired ? null : 'training_not_wired'),
    datasetVersion: descriptor.dataset_version || null,
    fingerprint: descriptor.dataset_fingerprint || null,
    splits: descriptor.splits || null,
    time: descriptor.time || null,
    trainingProfile: descriptor.training_profile || null,
    frequencyHours: Number(descriptor.frequency_hours ?? descriptor.training_profile?.frequency_hours ?? 24),
    stepUnit: descriptor.step_unit || descriptor.training_profile?.step_unit || 'day',
    step: Number(descriptor.step ?? descriptor.training_profile?.step ?? 1),
    gridShape: descriptor.grid_shape || descriptor.training_profile?.grid_shape || null,
    target: descriptor.target || descriptor.training_profile?.target || EARTH_TARGET_CHANNEL,
    targetUnit: descriptor.target_unit || descriptor.training_profile?.target_unit || EARTH_TARGET_UNIT,
    displayName: descriptor.display_name || descriptor.dataset_id,
  };
}

/** Published three-hour split counts, including descriptors without explicit steps. */
export function describeEarthSplitSamples(splits, window = EARTH_WINDOW, horizon = EARTH_HORIZON) {
  if (!splits || typeof splits !== 'object') return [];
  return ['train', 'validation', 'test']
    .filter((name) => splits[name])
    .map((name) => {
      const declaredDays = Number(splits[name].days) || 0;
      const hasExplicitSteps = Number.isFinite(Number(splits[name].steps));
      const steps = hasExplicitSteps ? Number(splits[name].steps) : (declaredDays * 24 / 3);
      const days = hasExplicitSteps ? steps / 8 : declaredDays;
      const sampleUnits = steps;
      return {
        name,
        start: splits[name].start,
        end: splits[name].end,
        days,
        windows: Math.max(0, sampleUnits - window - horizon + 1),
        steps,
      };
    });
}

/** 展示用单位行：辅助输入各自单位，目标固定 DU。 */
export function describeEarthInputUnits(selectedChannels) {
  const channels = [EARTH_TARGET_CHANNEL, ...normalizeEarthSelectedChannels(selectedChannels)];
  return channels.map((channel) => ({
    channel,
    unit: EARTH_CHANNEL_META[channel]?.unit || '',
  }));
}

/** 提交前的本地就绪检查；服务端仍会独立复核。 */
export function getEarthTrainingReadiness({
  datasetAvailability,
  epochs,
  batchSize,
  learningRate,
  modelName,
} = {}) {
  const blockers = [];
  if (!datasetAvailability?.present) blockers.push('catalog_missing');
  else if (!datasetAvailability.selectable) blockers.push('dataset_unavailable');
  if (!String(modelName || '').trim()) blockers.push('model_name_required');
  const parsedEpochs = Number(epochs);
  if (!Number.isFinite(parsedEpochs) || parsedEpochs < 1 || parsedEpochs > 1000) blockers.push('epochs_out_of_range');
  const parsedBatch = Number(batchSize);
  if (!Number.isFinite(parsedBatch) || parsedBatch < 1 || parsedBatch > 64) blockers.push('batch_size_out_of_range');
  const parsedLr = Number(learningRate);
  if (!Number.isFinite(parsedLr) || parsedLr <= 0 || parsedLr > 1) blockers.push('learning_rate_out_of_range');
  return { ready: blockers.length === 0, blockers };
}

/** Map server/API failures to the three user-visible Earth training categories. */
export function classifyEarthTrainingError(errorOrCode) {
  const code = String(errorOrCode?.code || errorOrCode?.error_code || errorOrCode || '').toLowerCase();
  if (/(dataset|package|data).*?(unavailable|missing|invalid|changed)/.test(code)
    || code === 'catalog_missing') return 'dataset_unavailable';
  if (code === 'dataset_retired' || /(configuration|config|not_supported|unsupported)/.test(code)) return 'configuration_unsupported';
  if (/(checkpoint|artifact|model).*?(incompat|invalid|mismatch|damaged)/.test(code)
    || code.includes('earth_compatibility')) return 'checkpoint_incompatible';
  return 'training_failed';
}

/**
 * 场景草稿快照：切走时保存、切回时恢复。
 *
 * Earth 与火星各自保留完整草稿（含上传模型选择与自定义参数），因此两个场景的
 * 配置不会互相覆盖。
 */
export function captureTrainingDraft(values) {
  return {
    trainingDataset: values.trainingDataset,
    modelSource: values.modelSource,
    selectedUploadedModelId: values.selectedUploadedModelId || '',
    modelArchitecture: values.modelArchitecture,
    useSphere: Boolean(values.useSphere),
    windowValue: values.windowValue,
    horizon: values.horizon,
    trainRatio: values.trainRatio,
    validationRatio: values.validationRatio,
    testRatio: values.testRatio,
    transferEnabled: Boolean(values.transferEnabled),
    selectedChannels: Array.isArray(values.selectedChannels) ? [...values.selectedChannels] : [],
    customModelParams: values.customModelParams && typeof values.customModelParams === 'object'
      ? { ...values.customModelParams }
      : {},
  };
}

/** 兼容旧命名：火星草稿就是通用草稿。 */
export const captureMarsTrainingSnapshot = captureTrainingDraft;

/**
 * Earth 生效时的表单覆盖值。
 *
 * ``storedMarsSnapshot`` 只被保存，不会被应用：这里的返回值必须让 Earth 的固定
 * 契约生效，因此固定值写在展开之后。
 */
export function applyEarthTrainingDefaults({ datasetId = EARTH_DATASET_ID, storedMarsSnapshot = null } = {}) {
  const profile = getEarthTrainingProfile(datasetId);
  return {
    trainingDataset: profile.datasetId,
    modelSource: EARTH_MODEL_SOURCE_OFFICIAL,
    selectedUploadedModelId: '',
    modelArchitecture: EARTH_MODEL_ARCHITECTURE,
    useSphere: false,
    windowValue: profile.window,
    horizon: profile.horizon,
    transferEnabled: false,
    selectedChannels: [...EARTH_OPTIONAL_CHANNELS],
    customModelParams: {},
    storedMarsSnapshot,
  };
}

/** 从火星切到 Earth 时恢复的 Earth 草稿；没有草稿则回到 Earth 默认值。 */
export function resolveEarthTrainingRestore(snapshot) {
  if (!snapshot || isRetiredEarthDataset(snapshot.trainingDataset)) return null;
  const modelSource = EARTH_MODEL_SOURCES.includes(snapshot.modelSource)
    ? snapshot.modelSource
    : EARTH_MODEL_SOURCE_OFFICIAL;
  return {
    trainingDataset: EARTH_DATASET_ID,
    modelSource,
    selectedUploadedModelId: snapshot.selectedUploadedModelId || '',
    modelArchitecture: modelSource === EARTH_MODEL_SOURCE_UPLOADED
      ? EARTH_UPLOADED_ARCHITECTURE
      : EARTH_MODEL_ARCHITECTURE,
    useSphere: false,
    windowValue: sanitizePositiveInteger(snapshot.windowValue, EARTH_WINDOW, 1, 240),
    horizon: sanitizePositiveInteger(snapshot.horizon, EARTH_HORIZON, 1, 240),
    trainRatio: snapshot.trainRatio ?? 0.7,
    validationRatio: snapshot.validationRatio ?? 0.2,
    testRatio: snapshot.testRatio ?? 0.1,
    transferEnabled: false,
    selectedChannels: normalizeEarthSelectedChannels(snapshot.selectedChannels),
    customModelParams: snapshot.customModelParams && typeof snapshot.customModelParams === 'object'
      ? { ...snapshot.customModelParams }
      : {},
  };
}

/** 从 Earth 切回火星时恢复的字段；没有快照则回到页面默认值。 */
export function resolveMarsTrainingRestore(snapshot) {
  if (!snapshot) return null;
  return {
    trainingDataset: snapshot.trainingDataset,
    modelSource: snapshot.modelSource,
    selectedUploadedModelId: snapshot.selectedUploadedModelId || '',
    modelArchitecture: snapshot.modelArchitecture,
    useSphere: Boolean(snapshot.useSphere),
    windowValue: snapshot.windowValue,
    horizon: snapshot.horizon,
    trainRatio: snapshot.trainRatio ?? 0.7,
    validationRatio: snapshot.validationRatio ?? 0.2,
    testRatio: snapshot.testRatio ?? 0.1,
    transferEnabled: Boolean(snapshot.transferEnabled),
    selectedChannels: Array.isArray(snapshot.selectedChannels) ? [...snapshot.selectedChannels] : [],
    customModelParams: snapshot.customModelParams && typeof snapshot.customModelParams === 'object'
      ? { ...snapshot.customModelParams }
      : {},
  };
}

/** Preview the same raw UTC allocation as earth_raw_utc_timeline_v1. */
export function describeEarthTaskSplits(time, ratios, window, horizon) {
  const checked = normalizeEarthSplitRatios(ratios?.trainRatio, ratios?.validationRatio, ratios?.testRatio);
  if (!checked.valid) return { valid: false, error: 'ratios', ranges: [] };
  const count = time?.count;
  const first = Date.parse(time?.start);
  const last = Date.parse(time?.end);
  const cadence = 3 * 60 * 60 * 1000;
  if (!Number.isInteger(count) || count < 3 || !Number.isFinite(first)
      || first + (count - 1) * cadence !== last) return { valid: false, error: 'timeline', ranges: [] };
  if (!Number.isInteger(Number(window)) || !Number.isInteger(Number(horizon))
      || Number(window) < 1 || Number(horizon) < 1) return { valid: false, error: 'windows', ranges: [] };
  const values = [checked.train_ratio, checked.validation_ratio, checked.test_ratio];
  // Exact decimal weights keep remainder ties identical to the server Decimal allocation.
  const decimals = values.map(value => {
    const [mantissa, exponent = '0'] = String(value).split('e');
    const [integer, fraction = ''] = mantissa.split('.');
    return { digits: BigInt(integer + fraction), places: fraction.length - Number(exponent) };
  });
  const places = Math.max(...decimals.map(value => value.places));
  const weights = decimals.map(value => value.digits * 10n ** BigInt(places - value.places));
  const total = weights.reduce((sum, value) => sum + value, 0n);
  const numerators = weights.map(value => value * BigInt(count));
  const sizes = numerators.map(value => Number(value / total));
  const order = [0, 1, 2].sort((a, b) => {
    const left = numerators[a] % total, right = numerators[b] % total;
    return left === right ? a - b : left > right ? -1 : 1;
  });
  order.slice(0, count - sizes.reduce((sum, value) => sum + value, 0)).forEach(index => { sizes[index] += 1; });
  let cursor = 0;
  const ranges = ['train', 'validation', 'test'].map((name, index) => {
    const start = cursor;
    cursor += sizes[index];
    return { name, raw_start: start, raw_end: cursor, step_count: sizes[index],
      date_start: new Date(first + start * cadence).toISOString().replace('.000Z', 'Z'),
      date_end: new Date(first + (cursor - 1) * cadence).toISOString().replace('.000Z', 'Z'),
      window_count: Math.max(0, sizes[index] - Number(window) - Number(horizon) + 1) };
  });
  return { valid: ranges.every(item => item.window_count > 0),
    error: ranges.some(item => item.window_count === 0) ? 'short_partition' : null, ranges };
}

export function getEarthSplitDefaults(defaults) {
  const split = normalizeEarthSplitRatios(defaults?.trainRatio, defaults?.validationRatio, defaults?.testRatio);
  return split.valid ? { trainRatio: split.train_ratio, validationRatio: split.validation_ratio, testRatio: split.test_ratio }
    : { trainRatio: 0.7, validationRatio: 0.2, testRatio: 0.1 };
}
