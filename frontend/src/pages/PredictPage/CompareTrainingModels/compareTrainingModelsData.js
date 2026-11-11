import { environmentIdentityMetadata, resolveEnvironmentTask } from '../../../utils/environmentVariables.js';
import { earthMetricUnit } from '../../../utils/earthMetricMeta.js';

const FEATURE_ORDER = ['Ozone', 'Temperature', 'Dust_Optical_Depth', 'Solar_Flux_DN', 'U_Wind', 'V_Wind'];

function parseHyperparameters(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function formatValue(value) {
  if (Array.isArray(value)) return value.length ? value.join(' / ') : '--';
  if (typeof value === 'boolean') return value ? 'On' : 'Off';
  if (value === 0) return '0';
  return value == null || value === '' ? '--' : String(value);
}

export function metricValue(item, metric) {
  const raw = item?.metrics?.overall?.[metric];
  if (raw == null || raw === '' || typeof raw === 'boolean') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export function normalizeCompareDataSource() {
  return 'default';
}

export function buildCompareModelSummary(task) {
  const hypers = parseHyperparameters(task?.hyperparameters);
  const earth = task?.is_earth_task === true || String(task?.dataset_id || hypers.training_dataset || '').startsWith('earth_');
  const target = resolveEnvironmentTask({
    ...task,
    hyperparameters: { ...hypers, selected_channels: hypers.selected_channels ?? task?.selected_channels },
  }, earth ? 'earth' : 'mars');
  const selectedChannels = target.inputs;
  const modelSource = String(hypers.model_source || task?.model_source || 'official').toLowerCase();
  const architecture = String(hypers.model_architecture || (modelSource === 'uploaded' ? 'uploaded' : 'predrnnv2')).toLowerCase();
  const dataSource = normalizeCompareDataSource(hypers._effective_data_source || hypers._data_source);
  const taskId = Number(task?.id);

  return {
    taskId,
    modelName: task?.custom_model_name || `Task #${taskId}`,
    modelSource,
    architecture,
    selectedChannels,
    inputChannelText: selectedChannels.join(' / '),
    targetText: `${target.target} (${target.unit})`,
    window: hypers.window,
    horizon: hypers.horizon,
    dataSource,
  };
}

export function sortCompareItems(items = [], options = {}) {
  const metric = options.metric || 'rmse';
  const direction = options.direction || (['r2', 'ssim'].includes(metric) ? 'desc' : 'asc');
  const fallback = direction === 'asc' ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;

  return [...(Array.isArray(items) ? items : [])].sort((a, b) => {
    const av = metricValue(a, metric) ?? fallback;
    const bv = metricValue(b, metric) ?? fallback;
    return direction === 'asc' ? av - bv : bv - av;
  });
}

export function buildCompareParameterRows(items = [], planet = 'mars') {
  return (Array.isArray(items) ? items : []).map((item) => {
    const hypers = parseHyperparameters(item?.hyperparameters);
    const target = resolveEnvironmentTask({
      ...item,
      hyperparameters: { ...hypers, selected_channels: hypers.selected_channels ?? item?.selected_channels },
    }, planet);
    return {
      taskId: Number(item?.task_id),
      modelName: item?.model_name || `Task #${item?.task_id}`,
      modelSource: formatValue(item?.model_source || hypers.model_source),
      architecture: formatValue(item?.architecture || hypers.model_architecture),
      target: `${target.target} (${target.unit})`,
      selectedChannels: formatValue(target.inputs),
      useSphere: formatValue(Boolean(hypers.use_sphere)),
      window: formatValue(hypers.window),
      horizon: formatValue(hypers.horizon),
      epochs: formatValue(hypers.epochs),
      batchSize: formatValue(hypers.batch_size),
      learningRate: formatValue(hypers.learning_rate),
      earlyStoppingPatience: formatValue(hypers.early_stopping_patience),
      seed: formatValue(hypers.seed),
      dataSource: formatValue(normalizeCompareDataSource(hypers._effective_data_source || hypers._data_source)),
    };
  });
}

export function resolveEarthCompareMetricUnit(metricKey, targetUnit, metricUnits = null, isZh = false) {
  return earthMetricUnit(metricKey, isZh, targetUnit, metricUnits);
}

export function compareResultMatchesTasks(result, tasks = [], planet = 'mars') {
  const items = Array.isArray(result?.items) ? result.items : [];
  if (!Array.isArray(tasks) || tasks.length < 2 || items.length !== tasks.length) return false;
  const expectedById = new Map(tasks.map((task) => [Number(task?.id), task]));
  if (expectedById.size !== tasks.length) return false;

  return items.every((item) => {
    const expectedTask = expectedById.get(Number(item?.task_id));
    if (!expectedTask) return false;
    const expected = resolveEnvironmentTask(expectedTask, planet);
    const actual = resolveEnvironmentTask(item, planet);
    const expectedIdentity = environmentIdentityMetadata(expectedTask);
    const actualIdentity = environmentIdentityMetadata(item);
    if ((expectedIdentity.declared || actualIdentity.declared) && !actualIdentity.complete) return false;
    if (!expectedIdentity.declared && !actualIdentity.declared
      && expected.target !== (planet === 'earth' ? 'TO3' : 'O3')) return false;
    return actual.target === expected.target
      && actual.unit === expected.unit
      && JSON.stringify(actual.inputs) === JSON.stringify(expected.inputs);
  });
}

export function getCompareSelectionState(selectedIds = []) {
  const ids = [...new Set((Array.isArray(selectedIds) ? selectedIds : [])
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item) && item > 0))]
    .sort((a, b) => a - b);
  return {
    ids,
    canCompare: ids.length >= 2,
    count: ids.length,
  };
}

export function buildStepCurveTraces(items = [], metric = 'rmse') {
  return (Array.isArray(items) ? items : []).map((item) => {
    const steps = Array.isArray(item?.metrics?.per_step) ? item.metrics.per_step : [];
    const validSteps = steps.filter((step) => Number.isFinite(Number(step.step)) && step[metric] != null && Number.isFinite(Number(step[metric])));
    return {
      taskId: Number(item?.task_id),
      name: item?.model_name || `Task #${item?.task_id}`,
      x: validSteps.map((step) => Number(step.step)),
      y: validSteps.map((step) => Number(step[metric])),
    };
  }).filter((trace) => trace.x.length > 0 && trace.y.length > 0);
}

function histogramCenters(edges = []) {
  return edges.slice(0, -1).map((edge, index) => (Number(edge) + Number(edges[index + 1])) / 2);
}

export function buildErrorHistogramTraces(items = []) {
  return (Array.isArray(items) ? items : []).map((item) => {
    const hist = item?.distribution?.hist_errors || {};
    return {
      taskId: Number(item?.task_id),
      name: item?.model_name || `Task #${item?.task_id}`,
      centers: histogramCenters(hist.bin_edges || []).filter((value) => Number.isFinite(value)),
      counts: (Array.isArray(hist.counts) ? hist.counts : []).map((value) => Number(value)).filter((value) => Number.isFinite(value)),
    };
  }).filter((trace) => trace.centers.length > 0 && trace.counts.length > 0);
}

export function buildPfiMatrix(items = []) {
  const observed = [];
  (Array.isArray(items) ? items : []).forEach((item) => {
    (Array.isArray(item?.items) ? item.items : []).forEach((entry) => {
      const name = String(entry?.name || '').trim();
      if (name && !observed.includes(name)) observed.push(name);
    });
  });
  const features = [
    ...FEATURE_ORDER.filter((name) => observed.includes(name)),
    ...observed.filter((name) => !FEATURE_ORDER.includes(name)),
  ];
  const rows = (Array.isArray(items) ? items : []).map((item) => {
    const values = Object.fromEntries(features.map((feature) => [feature, 0]));
    (Array.isArray(item?.items) ? item.items : []).forEach((entry) => {
      const name = String(entry?.name || '').trim();
      const importance = Number(entry?.importance);
      if (name && Number.isFinite(importance)) {
        values[name] = importance;
      }
    });
    return {
      taskId: Number(item?.task_id),
      modelName: item?.model_name || `Task #${item?.task_id}`,
      values,
    };
  });
  return { features, rows };
}
