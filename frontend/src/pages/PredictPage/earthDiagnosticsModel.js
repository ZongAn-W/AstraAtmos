import { createPredictRequestCoordinator } from './predictRequestCoordinator.js';
import { environmentIdentityMetadata, resolveEnvironmentTask } from '../../utils/environmentVariables.js';

export const EARTH_DIAGNOSTIC_DEFAULTS = Object.freeze({
  sample_windows: 4, scatter_points: 5000, histogram_bins: 40,
  seed: 42, pfi_repeats: 3, include_pfi: true,
});

export function buildEarthDiagnosticParameters(taskId, options = {}) {
  const id = Number(taskId);
  if (!Number.isInteger(id) || id <= 0) throw new Error('A valid training task is required');
  const values = { ...EARTH_DIAGNOSTIC_DEFAULTS, ...options };
  const bounded = (key, min, max) => {
    const value = Number(values[key]);
    if (!Number.isInteger(value) || value < min || value > max) {
      throw new Error(`${key} must be an integer between ${min} and ${max}`);
    }
    return value;
  };
  return { training_task_id: id, sample_windows: bounded('sample_windows', 1, 8),
    scatter_points: bounded('scatter_points', 1, 20000), histogram_bins: bounded('histogram_bins', 5, 100),
    seed: bounded('seed', 0, 4294967295), pfi_repeats: bounded('pfi_repeats', 1, 5),
    include_pfi: values.include_pfi !== false };
}

export function earthDiagnosticTaskAvailability(task, isZh = true) {
  let hypers = task?.hyperparameters;
  if (typeof hypers === 'string') {
    try { hypers = JSON.parse(hypers); } catch { hypers = {}; }
  }
  const dataset = String(task?.dataset_id || hypers?.training_dataset || '');
  if (dataset === 'earth_merra2_daily_v1' || dataset === 'earth_merra2_daily_v2') {
    return { available: false, reason: isZh ? '日频地球数据集已停用，旧任务不能运行测试。' : 'Daily Earth datasets are retired; this task cannot be tested.' };
  }
  if (task?.status !== 'completed') return { available: false,
    reason: isZh ? '只有已完成的任务可以测试。' : 'Only completed tasks can be tested.' };
  if (task?.model_available !== true) return { available: false,
    reason: isZh ? '模型产物不可用，请检查权重与任务状态。' : 'The model artifact is unavailable; check its weights and task status.' };
  return { available: true, reason: '' };
}

/** Guard both cancellation and late responses, including fetch implementations that ignore abort. */
export function createEarthDiagnosticRequestGuard() {
  const coordinator = createPredictRequestCoordinator();
  return {
    start: (key) => coordinator.start('earth-diagnostics', key),
    isCurrent: (token) => coordinator.isCurrent(token),
    accepts: (token, payload) => coordinator.isCurrent(token)
      && Number(payload?.task_id) === Number(token.taskId)
      && earthDiagnosticMatchesIdentity(payload, token.identity),
    invalidate: () => coordinator.invalidateAll(),
  };
}

export function earthDiagnosticErrorText(error, fallback = 'Diagnostics failed') {
  const detail = error?.response?.data?.detail || error?.detail;
  if (typeof detail === 'string') return detail;
  if (detail?.message) return String(detail.message);
  if (typeof error?.message === 'string') return error.message;
  return typeof error?.reason === 'string' ? error.reason
    : error?.reason?.message || error?.reason?.code || error?.code || fallback;
}

/** Pairing is checked before plotting; invalid values are never quietly filtered. */
export function earthDiagnosticScatter(scatter, expectedUnit = 'DU') {
  const reference = scatter?.reference;
  const prediction = scatter?.prediction;
  if (!Array.isArray(reference) || !Array.isArray(prediction) || reference.length !== prediction.length) {
    throw new Error('Invalid paired scatter data');
  }
  if (!reference.length) throw new Error('No valid diagnostic scatter samples');
  if (scatter.unit != null && scatter.unit !== expectedUnit) throw new Error(`Diagnostic scatter must use ${expectedUnit}`);
  if (scatter.point_count != null && scatter.point_count !== reference.length) {
    throw new Error('Diagnostic scatter count does not match its paired samples');
  }
  let min = Infinity;
  let max = -Infinity;
  for (let index = 0; index < reference.length; index += 1) {
    if (!Number.isFinite(reference[index]) || !Number.isFinite(prediction[index])) {
      throw new Error('Non-finite diagnostic scatter samples');
    }
    min = Math.min(min, reference[index], prediction[index]);
    max = Math.max(max, reference[index], prediction[index]);
  }
  const padding = Math.max((max - min) * .04, .01);
  return { reference, prediction, range: [min - padding, max + padding] };
}

export function earthDiagnosticHistogram(histogram, expectedCount = null, expectedUnit = 'DU') {
  const { edges, counts } = histogram || {};
  if (!Array.isArray(edges) || !Array.isArray(counts) || edges.length !== counts.length + 1 || !counts.length) {
    throw new Error('Invalid diagnostic histogram bins');
  }
  if (!edges.every(Number.isFinite) || !counts.every((n) => Number.isInteger(n) && n >= 0)
    || edges.some((edge, index) => index > 0 && edge <= edges[index - 1])) {
    throw new Error('Invalid diagnostic histogram values');
  }
  if (!counts.some((n) => n > 0)) throw new Error('No valid diagnostic residual samples');
  if (histogram.unit != null && histogram.unit !== expectedUnit) throw new Error(`Diagnostic residuals must use ${expectedUnit}`);
  if (histogram.residual_definition != null && histogram.residual_definition !== 'prediction-reference') {
    throw new Error('Diagnostic residual definition must be prediction-reference');
  }
  const sampleCount = counts.reduce((sum, count) => sum + count, 0);
  const declaredCounts = [expectedCount, histogram.valid_points, histogram.summary?.count].filter((n) => n != null);
  if (declaredCounts.some((n) => !Number.isSafeInteger(n) || n !== sampleCount)) {
    throw new Error('Diagnostic histogram counts do not cover the declared evaluation points');
  }
  return { centers: counts.map((_, index) => (edges[index] + edges[index + 1]) / 2),
    widths: counts.map((_, index) => edges[index + 1] - edges[index]), counts,
    sampleCount };
}

export function earthDiagnosticIdentityKey(taskId, identity = {}) {
  identity = identity || {};
  const environment = resolveEnvironmentTask(identity, 'earth');
  return JSON.stringify([Number(taskId), identity.dataset_id, identity.dataset_fingerprint, identity.dataset_version,
    environment.target, environment.unit,
    identity.window, identity.horizon, environment.inputs, identity.task_split, identity.model,
    identity.output_model_path, identity.checkpoint_sha256, identity.hyperparameters,
    identity.metrics?.split_ranges, identity.metrics?.split_policy, identity.metrics?.task_split]);
}

export function earthDiagnosticRequestKey(taskId, identity, parameters, userId) {
  return JSON.stringify([userId ?? null, earthDiagnosticIdentityKey(taskId, identity), parameters]);
}

/** Known request identity must match the returned physical Earth result. */
export function earthDiagnosticMatchesIdentity(payload, identity = {}) {
  identity = identity || {};
  if (payload?.planet != null && payload.planet !== 'earth') return false;
  const target = resolveEnvironmentTask(identity, 'earth');
  const payloadTarget = resolveEnvironmentTask(payload, 'earth');
  const identityMetadata = environmentIdentityMetadata(identity);
  const payloadMetadata = environmentIdentityMetadata(payload);
  if (identityMetadata.complete && !payloadMetadata.complete) return false;
  if (payload?.target_unit != null && payload.target_unit !== target.unit) return false;
  if (payloadTarget.target !== target.target) return false;
  for (const key of ['dataset_id', 'dataset_version', 'dataset_fingerprint', 'window', 'horizon', 'checkpoint_sha256']) {
    if (identity[key] != null && payload?.[key] !== identity[key]) return false;
  }
  const identityDeclaresInputs = [identity.input_variables, identity.input_channel_order]
    .some((value) => Array.isArray(value) && value.length > 0);
  const payloadDeclaresInputs = [payload?.input_variables, payload?.input_channel_order]
    .some((value) => Array.isArray(value) && value.length > 0);
  if (identityDeclaresInputs
    && (!payloadDeclaresInputs || JSON.stringify(payloadTarget.inputs) !== JSON.stringify(target.inputs))) return false;
  return true;
}

export function earthDiagnosticPfiItems(pfi) {
  return (pfi?.items || []).map((item) => ({ ...item,
    baseline_rmse: item.baseline_rmse ?? pfi.baseline_rmse ?? pfi.baseline_value,
    importance_mean: item.importance_mean ?? item.importance ?? item.mean,
    importance_std: item.importance_std ?? item.std,
    repeats: Array.isArray(item.repeats) ? item.repeats : [],
  }));
}
