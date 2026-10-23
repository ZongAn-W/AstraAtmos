import { EARTH_METRIC_META, earthMetricUnit } from '../../utils/earthMetricMeta.js';
import { convertOzone, ozoneLabel } from '../../utils/units.js';
import { getMetricAggregationLabel, getSplitLabel } from './predictionMetricMeta.js';
import { currentStepScatter } from './researchExportModel.js';
import { earthDiagnosticHistogram, earthDiagnosticScatter, earthDiagnosticPfiItems } from './earthDiagnosticsModel.js';
import { getEarthMetricPolicyLabel } from '../../utils/earthMetricMeta.js';

const selectLanguage = (options) => options.isZh ?? ((options.settings || options).language !== 'en');
const text = (isZh, zh, en) => isZh ? zh : en;
const scopeNames = {
  current_window: ['当前预测窗口', 'Current forecast window'],
  full_test: ['完整测试集', 'Full test set'],
  sampled_diagnostic: ['抽样诊断窗口', 'Sampled diagnostic windows'],
};

export function predictionScopeLabel(scope, isZh = true) {
  const labels = scopeNames[scope];
  return labels ? labels[isZh ? 0 : 1] : '';
}

export function predictionChartTheme(settings = {}, overrides = {}) {
  const light = settings.theme === 'light';
  return {
    text: overrides.plotTextColor || (light ? '#334155' : '#cbd5e1'),
    muted: overrides.plotText60 || (light ? '#64748b' : '#94a3b8'),
    grid: overrides.plotGridColor || (light ? '#e2e8f0' : '#263647'),
  };
}

/** Only display values are converted; the original result and export identity stay untouched. */
export function predictionMetricCards(metrics, presentation) {
  const values = metrics?.overall || metrics || {};
  return presentation.items.map((item) => ({ ...item,
    value: Number.isFinite(values[item.key]) ? presentation.convertValue(values[item.key], item.key) : null,
  }));
}

function histogram(histogram, convert) {
  if (!histogram?.bin_edges || !histogram?.counts) return null;
  return {
    centers: histogram.bin_edges.slice(0, -1).map((edge, index) => convert((edge + histogram.bin_edges[index + 1]) / 2)),
    widths: histogram.bin_edges.slice(0, -1).map((edge, index) => convert(histogram.bin_edges[index + 1] - edge)),
    counts: histogram.counts,
  };
}

function scatterRange(reference, prediction) {
  let min = Infinity;
  let max = -Infinity;
  for (const values of [reference, prediction]) for (const value of values) {
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return Number.isFinite(min) ? (min === max ? [min - .01, max + .01] : [min, max]) : undefined;
}

export function createMarsPredictionPresentation(options = {}) {
  const settings = options.settings || options;
  const isZh = selectLanguage(options);
  const unit = options.ozoneUnit || settings.units?.ozone || 'um-atm';
  const unitLabel = ozoneLabel(unit);
  const convert = (value) => convertOzone(value, unit);
  return {
    metrics: {
      items: [
        { key: 'rmse', name: 'RMSE', unit: unitLabel, better: '↓', color: '#ff8f68' },
        { key: 'mae', name: 'MAE', unit: unitLabel, better: '↓', color: '#ff8f68' },
        { key: 'ssim', name: 'SSIM', unit: '', better: '↑', color: '#4acfac' },
        { key: 'r2', name: 'R²', unit: '', better: '↑', color: '#4acfac' },
      ],
      title: text(isZh, '模型评价指标', 'Model evaluation metrics'), missingLabel: '--',
      convertValue: (value, key) => ['rmse', 'mae'].includes(key) ? convert(value) : value,
      describe: (metrics) => {
        const scope = metrics?.aggregation?.overall === 'pooled_test_set_pixels' ? 'full_test' : 'current_window';
        return { scope, subtitle: `${predictionScopeLabel(scope, isZh)} · ${getMetricAggregationLabel(metrics, isZh)}`,
          splitLabel: getSplitLabel(metrics, isZh) };
      },
    },
    distribution: {
      unitLabel,
      readData: (data, { predictionResult, step = 0 } = {}) => {
        const pairs = currentStepScatter(predictionResult, step, unit) || { trues: [], preds: [] };
        return {
          unitLabel,
          scatter: { reference: pairs.trues, prediction: pairs.preds, range: scatterRange(pairs.trues, pairs.preds),
            scope: 'current_window', scopeText: text(isZh, `当前预测步 ${step + 1} · 普通散点`, `Forecast step ${step + 1} · ordinary scatter`),
            exportEnabled: true, exportRef: predictionResult?.export_ref },
          distributions: { reference: histogram(data?.hist_trues, convert), prediction: histogram(data?.hist_preds, convert),
            scope: 'full_test', scopeText: predictionScopeLabel('full_test', isZh) },
          residual: { ...histogram(data?.hist_errors, convert), scope: 'full_test',
            scopeText: text(isZh, '完整测试集；残差 = 预测 − 参考', 'Full test set; residual = prediction − reference'),
            rmse: Number.isFinite(data?.rmse) ? convert(data.rmse) : null,
            mae: Number.isFinite(data?.mae) ? convert(data.mae) : null },
        };
      },
    },
    pfi: {
      readData: (data) => ({
        title: text(isZh, '置换特征重要性（ΔR²）', 'Permutation feature importance (ΔR²)'),
        items: (data?.items || []).map((item) => ({ name: item.name, translateKey: `predict.variables.${item.name}`, value: item.importance })),
        baseline: data?.baseline_value, baselineLabel: text(isZh, '基线 R²', 'Baseline R²'), unitLabel: '', axisLabel: 'ΔR²',
        scope: 'sampled_diagnostic',
        scopeText: text(isZh,
          `抽样测试集（最多 40 个窗口${data?.sampling?.sample_size != null ? `；n=${data.sampling.sample_size}` : '；历史缓存未记录样本数'}）；独立于当前预测窗口，ΔR² 可为负。`,
          `Sampled test set (up to 40 windows${data?.sampling?.sample_size != null ? `; n=${data.sampling.sample_size}` : '; historical sample count unavailable'}); independent of the current prediction window. ΔR² may be negative.`),
        helpText: text(isZh, 'PFI 衡量置换输入后的 R² 下降程度，表示模型在此抽样范围下的输入敏感性，不能解释为因果贡献。',
          'PFI measures the decrease in R² after permuting an input. It describes model sensitivity within this sample, not causal contribution.'),
        exportEnabled: true, exportRef: data?.export_ref,
        emptyLabel: text(isZh, '暂无 PFI 分析数据。运行预测即可生成。', 'No PFI analysis data. Run prediction to generate it.'),
      }),
    },
  };
}

export function createEarthPredictionPresentation(options = {}) {
  const isZh = selectLanguage(options);
  const scope = options.scope || 'current_window';
  const splitLabelSource = options.splitLabelSource || null;
  // getSplitLabel reads metrics.split_meta, so keep the shared wrapper shape.
  const splitLabelMetrics = splitLabelSource ? { split_meta: splitLabelSource } : null;
  return {
    metrics: {
      items: EARTH_METRIC_META.map((item) => ({ ...item, label: item[isZh ? 'zh' : 'en'],
        unit: earthMetricUnit(item.key, isZh), better: item.key === 'r2' ? '↑' : '↓' })),
      title: text(isZh, '模型评价指标', 'Model evaluation metrics'), missingLabel: text(isZh, '未提供', 'Not provided'),
      convertValue: (value) => value,
      describe: (metrics) => ({ scope, subtitle: `${predictionScopeLabel(scope, isZh)} · ${text(isZh,
        '反归一化 TO3（DU）；', 'Physical TO3 (DU); ')}${getEarthMetricPolicyLabel(metrics, isZh)}`,
        // Same split line Mars shows; the source is the bound checkpoint test split.
        splitLabel: getSplitLabel(splitLabelMetrics, isZh) }),
    },
    distribution: {
      unitLabel: 'DU',
      readData: (data) => {
        const pairs = earthDiagnosticScatter(data.scatter);
        const bins = earthDiagnosticHistogram(data.histogram, data.scope?.valid_points);
        return {
          unitLabel: 'DU',
          scatter: { reference: pairs.reference, prediction: pairs.prediction, range: pairs.range,
            scope: 'sampled_diagnostic', scopeText: text(isZh, '抽样诊断窗口 · 流式抽样散点', 'Diagnostic windows · streaming-sampled scatter'),
            note: text(isZh, `仅显示 ${pairs.reference.length.toLocaleString()} 对抽样点；诊断指标覆盖相同窗口全部提前步与格点。`,
              `Only ${pairs.reference.length.toLocaleString()} sampled pairs; diagnostic metrics cover every lead and grid cell in these windows.`), exportEnabled: false },
          residual: { ...bins, scope: 'sampled_diagnostic',
            scopeText: text(isZh, '抽样诊断窗口；残差 = 预测 − 参考', 'Diagnostic windows; residual = prediction − reference'),
            note: text(isZh, `直方图按样本计数（${bins.sampleCount.toLocaleString()} 个残差），不做面积加权；加权误差指标见上方。`,
              `Histogram bars are sample counts (${bins.sampleCount.toLocaleString()} residuals), not area-weighted density; area-weighted error metrics are above.`),
            summary: data.histogram.summary, rmse: data.sample_metrics?.overall?.rmse, mae: data.sample_metrics?.overall?.mae },
        };
      },
    },
    pfi: {
      readData: (data) => ({
        title: text(isZh, '置换特征重要性（ΔRMSE）', 'Permutation feature importance (ΔRMSE)'),
        items: earthDiagnosticPfiItems(data).map((item) => ({ name: item.channel, value: item.importance_mean, std: item.importance_std })),
        baseline: data?.baseline_rmse ?? data?.baseline_value, baselineLabel: text(isZh, '基线 RMSE', 'Baseline RMSE'),
        unitLabel: 'DU', axisLabel: 'ΔRMSE (DU)', scope: 'sampled_diagnostic',
        scopeText: `${text(isZh, '相同抽样 test 窗口、全部提前步与全球格点；', 'Identical sampled test windows, all leads and global cells; ')}${getEarthMetricPolicyLabel(data?.sample_metrics, isZh)} · ${text(isZh, '置换后 RMSE − 基线 RMSE（DU），负值保留。', 'permuted RMSE − baseline RMSE (DU), retaining negative values.')}`,
        helpText: text(isZh, '误差条为重复置换标准差。每个输入通道整段历史窗口置换，含历史 TO3；未来目标和参考真值保持固定。PFI 表示模型输入敏感性，不能解释为因果贡献。',
          'Error bars show the standard deviation across repeats. Each input channel, including historical TO3, is permuted as a whole window; targets and references stay fixed. PFI describes sensitivity, not causal contribution.'), exportEnabled: false,
        emptyLabel: text(isZh, '此抽样范围没有可计算的 PFI 通道，请查看诊断原因或调整窗口。',
          'No PFI channels are computable in this sample. Inspect the diagnostic reason or adjust the windows.'),
      }),
    },
  };
}
