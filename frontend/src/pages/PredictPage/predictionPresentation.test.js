import test from 'node:test';
import assert from 'node:assert/strict';
import { createMarsPredictionPresentation, createEarthPredictionPresentation,
  predictionMetricCards, predictionScopeLabel, predictionChartTheme } from './predictionPresentation.js';
import { fmtNum } from '../../utils/fmt.js';

const earthData = {
  scatter: { reference: [300, 320], prediction: [302, 319], unit: 'DU', point_count: 2 },
  histogram: { edges: [-2, 0, 2, 4], counts: [1, 0, 1], valid_points: 2,
    residual_definition: 'prediction-reference', unit: 'DU', summary: { count: 2, mean: .5 } },
  scope: { valid_points: 2 }, sample_metrics: { overall: { rmse: 1.5, mae: 1 } },
};

test('Earth cards expose six physical metrics without applying Mars unit conversion or filling missing values', () => {
  const presentation = createEarthPredictionPresentation({ settings: { language: 'en', units: { ozone: 'DU' } } }).metrics;
  const source = { overall: { mse: 25, rmse: 5, mae: 4, r2: -.2, mape: 1.3, smape: null } };
  const cards = predictionMetricCards(source, presentation);
  assert.deepEqual(cards.map((card) => card.key), ['mse', 'rmse', 'mae', 'r2', 'mape', 'smape']);
  assert.deepEqual(cards.map((card) => card.unit), ['DU²', 'DU', 'DU', 'dimensionless', '%', '%']);
  assert.deepEqual(cards.map((card) => card.value), [25, 5, 4, -.2, 1.3, null]);
  assert.equal(presentation.missingLabel, 'Not provided');
  assert.equal(source.overall.rmse, 5);
});

test('Mars cards keep the existing four metrics and convert only ozone errors in the display', () => {
  const presentation = createMarsPredictionPresentation({ settings: { units: { ozone: 'DU' } } }).metrics;
  const source = { overall: { rmse: 20, mae: 10, ssim: .6, r2: -.3 } };
  assert.deepEqual(predictionMetricCards(source, presentation).map((card) => card.value), [2, 1, .6, -.3]);
  assert.equal(source.overall.rmse, 20);
  assert.equal(presentation.describe({ aggregation: { overall: 'pooled_test_set_pixels' } }).scope, 'full_test');
  assert.equal(presentation.describe(source).scope, 'current_window');
});

test('evaluation configuration keeps forecast, complete test, and sampled diagnostic scopes distinct', () => {
  for (const scope of ['current_window', 'full_test', 'sampled_diagnostic']) {
    const presentation = createEarthPredictionPresentation({ language: 'en', scope }).metrics;
    assert.equal(presentation.describe({ aggregation: 'forecast_origin_lead_grid_uniform' }).scope, scope);
    assert.ok(presentation.describe().subtitle.startsWith(predictionScopeLabel(scope, false)));
  }
  assert.equal(new Set(['current_window', 'full_test', 'sampled_diagnostic'].map(predictionScopeLabel)).size, 3);
});

test('Earth metric subtitles use the response aggregation and metric policy', () => {
  const presentation = createEarthPredictionPresentation({ language: 'en' }).metrics;
  assert.match(presentation.describe({ aggregation: 'forecast_origin_lead_cell_area_weighted',
    metric_policy: { spatial_weighting: 'spherical_cell_area' } }).subtitle, /Spherical cell-area weighted/);
  assert.match(presentation.describe({ aggregation: 'forecast_origin_lead_grid_uniform' }).subtitle,
    /Grid-uniform \(historical metric policy\)/);
  // Without a bound checkpoint split there is no split line, exactly like Mars.
  assert.equal(createEarthPredictionPresentation({ language: 'en' }).metrics.describe({}).splitLabel, '');
});

test('Earth complete-test metrics reuse the Mars split line from the bound checkpoint test partition', () => {
  // earthPredictModel.buildEarthSplitLabelSource returns { split_meta }, and
  // EarthPredictPanel passes that inner object here; the presentation re-wraps it
  // because the shared getSplitLabel reads metrics.split_meta.
  const splitMeta = { source: 'published_manifest_splits',
    test_range: { date_start: '2021-11-01', date_end: '2021-12-28', window_count: 128 } };
  const presentation = createEarthPredictionPresentation(
    { language: 'en', scope: 'full_test', splitLabelSource: splitMeta }).metrics;
  assert.equal(presentation.describe({}).splitLabel,
    'Published test split: 2021-11-01 — 2021-12-28 · 128 windows');
  const task = createEarthPredictionPresentation({ language: 'en', scope: 'full_test',
    splitLabelSource: { source: 'earth_task_split_v1', test_range: splitMeta.test_range } }).metrics;
  assert.equal(task.describe({}).splitLabel,
    'Task test split: 2021-11-01 — 2021-12-28 · 128 windows');
  const ratios = createEarthPredictionPresentation({ language: 'en', scope: 'full_test',
    splitLabelSource: { source: 'earth_task_split_v1',
      ratios: { train_ratio: .7, validation_ratio: .2, test_ratio: .1 } } }).metrics;
  assert.equal(ratios.describe({}).splitLabel, 'Test split: train/validation/test 70%/20%/10%');
  assert.equal(createEarthPredictionPresentation({ language: 'en', scope: 'full_test' }).metrics
    .describe({}).splitLabel, '');
});

test('Earth diagnostics retain paired DU values and true server histogram counts, with no scatter or PFI export', () => {
  const config = createEarthPredictionPresentation({ language: 'en' });
  const plots = config.distribution.readData(earthData);
  assert.deepEqual(plots.scatter.reference, [300, 320]);
  assert.deepEqual(plots.scatter.prediction, [302, 319]);
  assert.equal(plots.unitLabel, 'DU');
  assert.equal(plots.scatter.exportEnabled, false);
  assert.equal(plots.scatter.scope, 'sampled_diagnostic');
  assert.equal(plots.distributions, undefined);
  assert.deepEqual(plots.residual.centers, [-1, 1, 3]);
  assert.deepEqual(plots.residual.counts, [1, 0, 1]);
  assert.equal(plots.residual.rmse, 1.5);
  assert.equal(config.pfi.readData(null).exportEnabled, false);
  assert.throws(() => config.distribution.readData({ ...earthData, scope: { valid_points: 3 } }), /declared evaluation points/);
  assert.throws(() => config.distribution.readData({ ...earthData, scatter: { ...earthData.scatter, unit: 'um-atm' } }), /DU/);
});

test('Mars current-step scatter and full-test histograms have separate scopes and supported export identities', () => {
  const exportRef = { token: 'private-existing-snapshot' };
  const result = { ground_truth: [{ field: [[20]] }, { field: [[50]] }],
    prediction: [{ field: [[30]] }, { field: [[40]] }], export_ref: exportRef };
  const dist = { hist_trues: { bin_edges: [10, 30], counts: [2] }, hist_preds: { bin_edges: [20, 40], counts: [2] },
    hist_errors: { bin_edges: [-10, 10], counts: [2] }, rmse: 10, mae: 10 };
  const config = createMarsPredictionPresentation({ language: 'en', ozoneUnit: 'DU' }).distribution;
  const plots = config.readData(dist, { predictionResult: result, step: 1 });
  assert.deepEqual(plots.scatter.reference, [5]);
  assert.deepEqual(plots.scatter.prediction, [4]);
  assert.equal(plots.scatter.exportRef, exportRef);
  assert.equal(plots.scatter.scope, 'current_window');
  assert.equal(plots.distributions.scope, 'full_test');
  assert.equal(plots.residual.scope, 'full_test');
  assert.deepEqual(plots.distributions.reference.centers, [2]);
  assert.equal(plots.residual.rmse, 1);
});

test('PFI adapts delta RMSE DU versus delta R2 and preserves negative and zero values and Earth repeat std', () => {
  const earth = createEarthPredictionPresentation({ language: 'en' }).pfi.readData({ baseline_rmse: 3,
    items: [{ channel: 'TO3', importance: -1, importance_std: .2 }, { channel: 'U', importance: 0, importance_std: 0 }] });
  assert.equal(earth.axisLabel, 'ΔRMSE (DU)');
  assert.equal(earth.baseline, 3);
  assert.deepEqual(earth.items.map((item) => item.value), [-1, 0]);
  assert.deepEqual(earth.items.map((item) => item.std), [.2, 0]);
  assert.match(earth.scopeText, /permuted RMSE/);
  const mars = createMarsPredictionPresentation({ language: 'en' }).pfi.readData({ baseline_value: .9,
    items: [{ name: 'U_Wind', importance: -.1 }, { name: 'V_Wind', importance: 0 }], sampling: { sample_size: 4 } });
  assert.equal(mars.axisLabel, 'ΔR²');
  assert.deepEqual(mars.items.map((item) => item.value), [-.1, 0]);
  assert.match(mars.scopeText, /n=4/);
  assert.equal(mars.exportEnabled, true);
});

test('chart themes use the same settings and metrics remain precision aware in either theme', () => {
  assert.equal(predictionChartTheme({ theme: 'light' }).text, '#334155');
  assert.equal(predictionChartTheme({ theme: 'dark' }).text, '#cbd5e1');
  assert.equal(predictionChartTheme({ theme: 'dark' }, { plotGridColor: '#123456' }).grid, '#123456');
  const metric = predictionMetricCards({ overall: { rmse: 1.234567 } }, createEarthPredictionPresentation().metrics)[1];
  assert.equal(fmtNum(metric.value, 2), '1.23');
  assert.equal(fmtNum(metric.value, 4), '1.2346');
  assert.equal(fmtNum(metric.value, 'full'), '1.234567');
});
