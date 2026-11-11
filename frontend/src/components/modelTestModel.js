import { marsResponseMatchesTask } from '../pages/PredictPage/singleModelAdapters.js';

/** Return a padded axis that contains both positive and negative physical values. */
export function modelTestScatterRange(reference = [], prediction = []) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const values of [reference, prediction]) for (const value of Array.isArray(values) ? values : []) {
    if (!Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  if (!Number.isFinite(minimum)) return [0, 1];
  const span = maximum - minimum;
  const padding = Math.max(span * 0.1, Math.max(Math.abs(minimum), Math.abs(maximum), 1) * 0.02);
  // Preserve the historical zero origin for non-negative variables while keeping
  // signed wind fields fully visible.
  return [minimum >= 0 ? 0 : minimum - padding, maximum + padding];
}

export function modelTestMetricUnit(metric, targetUnit) {
  if (metric === 'rmse' || metric === 'mae') return targetUnit || '';
  if (metric !== 'mse') return '';
  if (!targetUnit || targetUnit === '1') return targetUnit || '';
  return /^[A-Za-zµμ]+$/.test(targetUnit) ? `${targetUnit}²` : `(${targetUnit})²`;
}

export function marsModelTestResponseMatchesTask(response, task) {
  return marsResponseMatchesTask(response, { task });
}
