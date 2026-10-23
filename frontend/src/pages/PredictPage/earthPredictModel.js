/**
 * Earth 历史预测页的纯函数：模式、缓存身份、日期起点与场组装。
 *
 * 与火星预测共用页面外壳，但身份必须分开：Earth 的结果按「星球 + 数据集身份 +
 * 任务 + 预测起点」区分，不能与火星的 MY/Ls 缓存互相命中。这里不引入 React，
 * 便于直接单元测试。
 */

export const PREDICT_MODEL_MODE_EARTH = 'earth';
export const EARTH_PREDICT_QUERY_VALUE = 'earth';
export const EARTH_PLANET = 'earth';
export const EARTH_TARGET_UNIT = 'DU';

export const EARTH_METRIC_KEYS = ['mse', 'rmse', 'mae', 'r2', 'mape', 'smape'];
export const EARTH_FIELD_KINDS = ['reference', 'prediction', 'residual'];
export const EARTH_3HOURLY_DATASET_ID = 'earth_merra2_3hourly_v1';
export const EARTH_THREE_HOURLY_DATASET_IDS = [EARTH_3HOURLY_DATASET_ID, 'earth_merra2_3hourly_5deg_v1'];

/** Return the cadence advertised by the server, retaining daily defaults. */
export function getEarthCadence(value) {
  const frequencyHours = Number(value?.frequency_hours ?? value?.temporal?.frequency_hours);
  const step = Number(value?.step ?? value?.temporal?.step);
  const threeHourly = EARTH_THREE_HOURLY_DATASET_IDS.includes(String(value?.dataset_id || ''))
    || frequencyHours === 3;
  return {
    threeHourly,
    frequencyHours: threeHourly ? 3 : (Number.isFinite(frequencyHours) ? frequencyHours : 24),
    step: threeHourly ? 3 : (Number.isFinite(step) ? step : 1),
    unit: threeHourly ? 'hour' : 'day',
  };
}

export function earthTimestampList(value, fallback = []) {
  const list = (Array.isArray(value?.target_timestamps) && value.target_timestamps.length
    ? value.target_timestamps
    : (Array.isArray(value?.timestamps) && value.timestamps.length
      ? value.timestamps : (value?.target_dates || fallback)));
  return Array.isArray(list) ? list : [];
}

export function earthLeadLabel(index, value, labels = {}) {
  const cadence = getEarthCadence(value);
  const n = Number(index) + 1;
  return cadence.threeHourly ? `+${n * 3}h` : `+${n}`;
}

/**
 * 从 hash 读取 Earth 预测请求。
 *
 * 训练页 handoff 用 `?from=training&mode=earth` 进入；未知 mode 返回 null，
 * 让页面沿用原有模式解析，不劫持火星入口。
 */
export function readEarthPredictRequestFromHash(hash) {
  const text = String(hash || '');
  const queryIndex = text.indexOf('?');
  if (queryIndex === -1) return null;
  const params = new URLSearchParams(text.slice(queryIndex + 1));
  if (params.get('mode') !== EARTH_PREDICT_QUERY_VALUE) return null;
  return {
    fromTraining: params.get('from') === 'training',
    taskId: normalizePositiveInteger(params.get('task_id')),
  };
}

export function normalizePositiveInteger(value) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

/**
 * Earth 预测缓存身份。
 *
 * 必须包含训练任务、数据集版本与指纹、预测起点与三天目标日期：任何一项变化都
 * 代表不同结果，不能在缓存里互相覆盖。同时带 planet 前缀，避免与火星键混淆。
 */
export function buildEarthPredictKey({
  taskId,
  datasetId,
  datasetVersion,
  datasetFingerprint,
  forecastOrigin,
  targetDates = [],
  targetTimestamps = null,
} = {}) {
  const normalizedTask = normalizePositiveInteger(taskId);
  if (!normalizedTask || !forecastOrigin) return null;
  return [
    `planet:${EARTH_PLANET}`,
    `task:${normalizedTask}`,
    `ds:${datasetId || 'none'}`,
    `v:${datasetVersion || 'none'}`,
    `fp:${datasetFingerprint || 'none'}`,
    `origin:${forecastOrigin}`,
    `targets:${(Array.isArray(targetTimestamps || targetDates) ? (targetTimestamps || targetDates) : []).join(',')}`,
  ].join('|');
}

/** 同一任务下切换预测起点时必须丢弃旧结果。 */
export function shouldClearEarthResult(previousKey, nextKey) {
  if (!previousKey) return false;
  return previousKey !== nextKey;
}

/**
 * 把 API 返回的场转成 EarthMap2D 需要的 payload。
 *
 * 单位固定 DU，不做任何换算：地球 TO3 本身就是 DU，不能套用火星的 μm-atm 换算。
 */
export function buildEarthFieldPayload({ response, dayIndex, kind, colormap, colorRange = null }) {
  const days = Array.isArray(response?.[kind]) ? response[kind] : [];
  const day = days[dayIndex];
  if (!day || !Array.isArray(day.field) || day.field.length === 0) return null;
  const lat = response?.grid?.latitude;
  const lon = response?.grid?.longitude;
  if (!Array.isArray(lat) || !Array.isArray(lon) || lat.length === 0 || lon.length === 0) return null;
  if (day.field.length !== lat.length || day.field[0]?.length !== lon.length) return null;
  const validAxis = axis => axis.every(Number.isFinite) && (axis.length < 2
    || axis.slice(1).every((value, index) => axis[1] > axis[0] ? value > axis[index] : value < axis[index]));
  if (!validAxis(lat) || !validAxis(lon)
    || day.field.some(row => !Array.isArray(row) || row.length !== lon.length || !row.every(Number.isFinite))) return null;
  const range = colorRange || {};
  const min = Number.isFinite(range.min) ? range.min : day.minVal;
  const max = Number.isFinite(range.max) ? range.max : day.maxVal;
  return {
    field: day.field,
    lat,
    lon,
    colormap,
    color_range: { min, max },
    unit: EARTH_TARGET_UNIT,
    coverage: {
      latitude_range: response?.grid?.latitude_range || null,
      longitude_range: response?.grid?.longitude_range || null,
    },
    valid_cells: day.valid_cells,
  };
}

/**
 * 三天的固定色阶：预测与参考共用同一区间，残差以 0 为中心对称。
 *
 * 预测/参考共用色阶才能直接比高低；残差用对称区间才能看出偏差方向。
 */
export function resolveEarthColorRanges(response, dayIndex = null) {
  const collect = (kind) => {
    const allDays = Array.isArray(response?.[kind]) ? response[kind] : [];
    const days = Number.isInteger(dayIndex) ? allDays.slice(dayIndex, dayIndex + 1) : allDays;
    const mins = days.map((day) => Number(day?.minVal)).filter(Number.isFinite);
    const maxs = days.map((day) => Number(day?.maxVal)).filter(Number.isFinite);
    if (mins.length === 0 || maxs.length === 0) return null;
    return { min: Math.min(...mins), max: Math.max(...maxs) };
  };
  const physical = collect('prediction') && collect('reference')
    ? {
        min: Math.min(collect('prediction').min, collect('reference').min),
        max: Math.max(collect('prediction').max, collect('reference').max),
      }
    : (collect('prediction') || collect('reference'));
  const residual = collect('residual');
  const residualMagnitude = residual
    ? Math.max(Math.abs(residual.min), Math.abs(residual.max))
    : 0;
  if (physical && physical.min === physical.max) {
    const pad = Math.max(Math.abs(physical.min) * .01, 1e-6);
    physical.min -= pad; physical.max += pad;
  }
  return {
    physical,
    residual: residualMagnitude > 0
      ? { min: -residualMagnitude, max: residualMagnitude }
      : residual ? { min: -1, max: 1 } : null,
  };
}

export function earthFieldLabel(kind, labels = {}) {
  if (kind === 'prediction') return labels.prediction || 'Prediction';
  if (kind === 'reference') return labels.reference || 'Reference';
  if (kind === 'residual') return labels.residual || 'Residual';
  return kind;
}

/** 六项指标只读取有限值，缺失项保留为空。 */
export function readEarthMetric(metrics, key) {
  const value = metrics?.overall?.[key];
  return Number.isFinite(value) ? value : null;
}

export function readEarthLeadMetric(metrics, leadDay, key) {
  const row = (metrics?.by_lead || []).find((item) => (
    Number(item?.lead_day) === Number(leadDay)
    || Number(item?.lead_step) === Number(leadDay)
  ));
  const value = row?.[key];
  return Number.isFinite(value) ? value : null;
}

export function readEarthHorizonMetric(metrics, horizonHours, key) {
  const row = (metrics?.by_horizon || []).find((item) => Number(item?.horizon_hours) === Number(horizonHours));
  const value = row?.[key];
  return Number.isFinite(value) ? value : null;
}

/** 起点是否落在服务端给出的可选范围内。 */
export function isOriginSelectable(origins, origin) {
  if (!origin) return false;
  const dates = Array.isArray(origins?.timestamps) && origins.timestamps.length
    ? origins.timestamps : origins?.dates;
  if (Array.isArray(dates) && dates.length > 0) return dates.includes(origin);
  if (origins?.start && origins?.end) return origin >= origins.start && origin <= origins.end;
  return false;
}

/** 默认起点：优先今天之前的最近可用日期，否则用可选范围末端的最后一个。 */
export function pickDefaultOrigin(origins, today = null) {
  const candidateDates = Array.isArray(origins?.timestamps) && origins.timestamps.length
    ? origins.timestamps : origins?.dates;
  const dates = Array.isArray(candidateDates) ? candidateDates : [];
  if (dates.length === 0) return origins?.end || '';
  if (!today) return dates[dates.length - 1];
  for (let index = dates.length - 1; index >= 0; index -= 1) {
    if (dates[index] <= today) return dates[index];
  }
  return dates[0];
}

const EARTH_ERROR_MESSAGES = {
  dataset_retired: 'earthDatasetRetired',
  earth_comparison_incompatible: 'earthComparisonIncompatible',
  earth_prediction_origin_out_of_range: 'earthOriginOutOfRange',
  invalid_earth_prediction_origin: 'earthOriginInvalid',
  dataset_version_changed: 'earthDatasetChanged',
  invalid_earth_training_artifact: 'earthArtifactInvalid',
  earth_prediction_task_not_completed: 'earthTaskNotCompleted',
  earth_prediction_data_unavailable: 'earthPredictionDataUnavailable',
  earth_training_profile_unsupported: 'earthTrainingProfileUnsupported',
  dataset_prediction_not_supported: 'earthNotAnEarthTask',
  dataset_unavailable: 'earthDatasetUnavailable',
};

/** 服务端错误码 → 翻译键；未知错误保留原始 message。 */
export function resolveEarthPredictErrorMessage(error) {
  const code = error?.code;
  if (code && EARTH_ERROR_MESSAGES[code]) {
    return { key: EARTH_ERROR_MESSAGES[code], fallback: error.message || code };
  }
  return { key: null, fallback: error?.message || String(error || '') };
}

export function isEarthTask(task) {
  if (!task) return false;
  if (typeof task.is_earth_task === 'boolean') return task.is_earth_task;
  return String(task.dataset_id || '') === 'earth_merra2_daily_v2'
    || String(task.dataset_id || '') === 'earth_merra2_daily_v1'
    || EARTH_THREE_HOURLY_DATASET_IDS.includes(String(task.dataset_id || ''));
}

/**
 * Earth 预测可用的任务：已完成、权重可用，且身份是 Earth。
 *
 * 与火星选择器分开实现，避免 Earth 任务漏进火星下拉（或反向污染）。选项标签带上
 * 模型来源，让「官方 DLinear」与「上传模型」在历史里可区分。
 */
export function getEarthTrainingModelOptions(tasks = []) {
  return (Array.isArray(tasks) ? tasks : [])
    .filter((task) => task?.status === 'completed' && task?.model_available === true
      && EARTH_THREE_HOURLY_DATASET_IDS.includes(task.dataset_id))
    .map((task) => {
      const base = task.custom_model_name || `Task #${task.id}`;
      const identity = readEarthTaskModelIdentity(task);
      return {
        id: Number(task.id),
        label: identity.uploadedModelName ? `${base} · ${identity.uploadedModelName}` : base,
        task,
        modelSource: identity.modelSource,
      };
    });
}

/**
 * 从训练任务读取模型来源与上传模型身份。
 *
 * 训练记录由服务端固定写入（`_uploaded_model_*`），这里只读展示；缺失时按官方
 * DLinear 处理，保持旧任务（无上传字段）仍可被识别与预测。
 */
export function readEarthTaskModelIdentity(task) {
  const hyperparameters = task?.hyperparameters || {};
  const modelSource = String(
    hyperparameters.model_source || task?.model_source || 'official',
  ).toLowerCase() === 'uploaded'
    ? 'uploaded'
    : 'official';
  const uploadedModelId = hyperparameters._uploaded_model_id || task?.uploaded_model_id || null;
  return {
    modelSource,
    uploaded: modelSource === 'uploaded',
    uploadedModelId: uploadedModelId ? String(uploadedModelId) : null,
    uploadedModelName: hyperparameters._uploaded_model_name || null,
    uploadedModelVersion: hyperparameters._uploaded_model_version ?? task?.uploaded_model_version ?? null,
    uploadedModelContentHash: hyperparameters._uploaded_model_content_hash || null,
    customModelParams: hyperparameters.custom_model_params || {},
  };
}

/** 预测结果里的模型身份（服务端在 /predict 与 /context 都会返回）。 */
export function readEarthResponseModelIdentity(response) {
  const identity = response?.model && typeof response.model === 'object' ? response.model : null;
  if (!identity) {
    const architecture = response?.model_architecture || null;
    return {
      known: Boolean(architecture),
      modelSource: architecture === 'uploaded' ? 'uploaded' : 'official',
      uploaded: architecture === 'uploaded',
      uploadedModelId: null,
      uploadedModelName: null,
      uploadedModelVersion: null,
      uploadedModelContentHash: null,
    };
  }
  const modelSource = String(identity.model_source || 'official').toLowerCase() === 'uploaded'
    ? 'uploaded'
    : 'official';
  return {
    known: true,
    modelSource,
    uploaded: modelSource === 'uploaded',
    uploadedModelId: identity.uploaded_model_id || null,
    uploadedModelName: identity.uploaded_model_name || null,
    uploadedModelVersion: identity.uploaded_model_version ?? null,
    uploadedModelContentHash: identity.uploaded_model_content_hash || null,
    sourceEmbedded: identity.uploaded_model_source_embedded === true,
  };
}

/** 一句话描述预测所用模型；上传模型带名称、版本与指纹前缀。 */
export function describeEarthModelIdentity(identity, labels = {}) {
  if (!identity) return '';
  if (!identity.uploaded) {
    return labels.official || 'DLinear (official)';
  }
  const parts = [identity.uploadedModelName || labels.uploadedFallback || 'Uploaded model'];
  if (identity.uploadedModelVersion !== null && identity.uploadedModelVersion !== undefined) {
    parts.push(`v${identity.uploadedModelVersion}`);
  }
  const description = parts.join(' ');
  const hash = identity.uploadedModelContentHash
    ? String(identity.uploadedModelContentHash).slice(0, 12)
    : '';
  return hash ? `${description} · ${hash}` : description;
}

const EARTH_TASK_SPLIT_POLICY = 'earth_task_split_v1';
const EARTH_SPLIT_NAMES = ['train', 'validation', 'test'];

/** 兼容 start/end 与 date_start/date_end 两种发布字段，缺失项不猜。 */
export function earthSplitRanges(source) {
  const raw = source?.split_ranges || source?.task_split?.ranges || {};
  const ranges = {};
  for (const name of EARTH_SPLIT_NAMES) {
    const range = raw?.[name];
    if (!range) continue;
    const dateStart = range.date_start ?? range.start;
    const dateEnd = range.date_end ?? range.end;
    if (!dateStart || !dateEnd) continue;
    const windowCount = Number(range.window_count);
    ranges[name] = {
      date_start: String(dateStart),
      date_end: String(dateEnd),
      ...(Number.isInteger(windowCount) && windowCount > 0 ? { window_count: windowCount } : {}),
    };
  }
  return ranges;
}

/**
 * 把 Earth 检查点保存的划分绑定整理成与火星相同的 `split_meta` 形状，
 * 供共用的 `getSplitLabel` 直接渲染；没有可读边界时返回 null，不伪造划分行。
 */
export function buildEarthSplitLabelSource(source) {
  const ranges = earthSplitRanges(source);
  const testRange = ranges.test;
  if (!testRange) return null;
  const ratios = source?.split_ratios || source?.task_split?.ratios || null;
  const policy = source?.split_policy;
  if (policy === EARTH_TASK_SPLIT_POLICY) {
    return {
      split_meta: {
        source: EARTH_TASK_SPLIT_POLICY,
        test_range: testRange,
        ...(ratios ? { ratios } : {}),
      },
    };
  }
  if (policy === 'published_manifest_splits') {
    return { split_meta: { source: policy, test_range: testRange } };
  }
  return null;
}
