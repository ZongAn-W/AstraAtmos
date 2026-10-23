import { getEarthMetricPolicyLabel } from '../../utils/earthMetricMeta.js';

const AGGREGATION_LABELS = {
  mean_over_forecast_steps: {
    zh: '整体（各预测步平均）',
    en: 'Overall (mean across forecast steps)',
  },
  pooled_test_set_pixels: {
    zh: '整体（完整测试集汇总；RMSE/MAE/R² 按像素合并）',
    en: 'Overall (full test-set summary; RMSE/MAE/R² pool pixels)',
  },
};

function formatPercent(value) {
  const numeric = Number(value) * 100;
  if (!Number.isFinite(numeric)) return '--';
  return `${Number.isInteger(numeric) ? numeric : numeric.toFixed(1)}%`;
}

export function getMetricAggregationLabel(metrics, isZh = true) {
  if (typeof metrics?.aggregation === 'string'
    && (metrics.aggregation.startsWith('forecast_origin_lead_')
      || metrics.aggregation.startsWith('user_forecast_origin_lead_'))) {
    return getEarthMetricPolicyLabel(metrics, isZh);
  }
  const label = AGGREGATION_LABELS[metrics?.aggregation?.overall];
  if (!label) return isZh ? '整体指标' : 'Overall metrics';
  return isZh ? label.zh : label.en;
}

export function getSplitLabel(metrics, isZh = true) {
  const splitMeta = metrics?.split_meta;
  if (splitMeta?.source === 'published_manifest_splits' && splitMeta.test_range) return isZh
    ? `发布测试分区：${splitMeta.test_range.date_start} — ${splitMeta.test_range.date_end} · ${splitMeta.test_range.window_count} 个窗口`
    : `Published test split: ${splitMeta.test_range.date_start} — ${splitMeta.test_range.date_end} · ${splitMeta.test_range.window_count} windows`;
  // Earth 任务级自定义划分与发布分区同样带真实边界，直接给边界而不退回比例行。
  if (splitMeta?.source === 'earth_task_split_v1' && splitMeta.test_range) return isZh
    ? `任务自定义测试分区：${splitMeta.test_range.date_start} — ${splitMeta.test_range.date_end} · ${splitMeta.test_range.window_count} 个窗口`
    : `Task test split: ${splitMeta.test_range.date_start} — ${splitMeta.test_range.date_end} · ${splitMeta.test_range.window_count} windows`;
  if (!splitMeta?.ratios) return '';

  if (splitMeta.legacy_compatibility) {
    return isZh
      ? '测试集划分：历史兼容 80%/20%（无验证集）'
      : 'Test split: legacy 80%/20% compatibility (no validation set)';
  }

  const ratios = splitMeta.ratios;
  const values = [ratios.train_ratio, ratios.validation_ratio, ratios.test_ratio].map(formatPercent);
  return isZh
    ? `测试集划分：训练/验证/测试 ${values.join('/')}`
    : `Test split: train/validation/test ${values.join('/')}`;
}
