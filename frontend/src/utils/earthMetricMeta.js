export const EARTH_METRIC_META = Object.freeze([
  { key: 'mse', name: 'MSE', zh: 'MSE 均方误差', en: 'MSE Mean squared error', unit: 'DU²', color: '#f0ad66' },
  { key: 'rmse', name: 'RMSE', zh: 'RMSE 均方根误差', en: 'RMSE Root mean squared error', unit: 'DU', color: '#e68383' },
  { key: 'mae', name: 'MAE', zh: 'MAE 平均绝对误差', en: 'MAE Mean absolute error', unit: 'DU', color: '#e6bc54' },
  { key: 'r2', name: 'R²', zh: 'R² 决定系数', en: 'R² Coefficient of determination', unit: '1', color: '#4acfac' },
  { key: 'mape', name: 'MAPE', zh: 'MAPE 平均绝对百分比误差', en: 'MAPE Mean absolute percentage error', unit: '%', color: '#79bbff' },
  { key: 'smape', name: 'SMAPE', zh: 'SMAPE 对称平均绝对百分比误差', en: 'SMAPE Symmetric mean absolute percentage error', unit: '%', color: '#d49bb7' },
]);

export function earthMetricUnit(key, isZh = false, targetUnit = 'DU', metricUnits = null) {
  if (['mse', 'rmse', 'mae'].includes(key) && metricUnits?.[key]) return metricUnits[key];
  if (key === 'mse') {
    if (targetUnit === '1') return isZh ? '无量纲' : 'dimensionless';
    return /^[A-Za-zµμ]+$/.test(targetUnit) ? `${targetUnit}²` : `(${targetUnit})²`;
  }
  if (key === 'rmse' || key === 'mae') return targetUnit;
  const unit = EARTH_METRIC_META.find(metric => metric.key === key)?.unit || '';
  return unit === '1' ? (isZh ? '无量纲' : 'dimensionless') : unit;
}

export function getEarthMetricPolicyLabel(metrics, isZh = true) {
  if (typeof metrics === 'string') {
    try { metrics = JSON.parse(metrics); } catch { metrics = null; }
  }
  const aggregation = metrics?.aggregation;
  const spatialWeighting = metrics?.metric_policy?.spatial_weighting;
  if (aggregation === 'forecast_origin_lead_cell_area_weighted'
    || aggregation === 'user_forecast_origin_lead_cell_area_weighted'
    || spatialWeighting === 'spherical_cell_area') {
    return isZh
      ? '球面 cell-area 加权（按发布 lat_bounds）'
      : 'Spherical cell-area weighted (published lat_bounds)';
  }
  if (aggregation === 'forecast_origin_lead_grid_uniform'
    || aggregation === 'user_forecast_origin_lead_grid_uniform') {
    return isZh
      ? '格点等权（历史指标口径）'
      : 'Grid-uniform (historical metric policy)';
  }
  return isZh ? '指标口径未记录' : 'Metric policy not recorded';
}

export function isEarthMetrics(metrics) {
  return String(metrics?.schema || '').startsWith('earth_training_metrics')
    || (metrics?.target === 'TO3' && metrics?.unit === 'DU');
}
