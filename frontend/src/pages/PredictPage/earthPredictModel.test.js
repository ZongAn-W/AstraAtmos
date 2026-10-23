import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildEarthFieldPayload,
  buildEarthPredictKey,
  buildEarthSplitLabelSource,
  earthLeadLabel,
  earthSplitRanges,
  getEarthCadence,
  getEarthTrainingModelOptions,
  isEarthTask,
  isOriginSelectable,
  pickDefaultOrigin,
  readEarthLeadMetric,
  readEarthMetric,
  readEarthHorizonMetric,
  readEarthPredictRequestFromHash,
  resolveEarthColorRanges,
  resolveEarthPredictErrorMessage,
  shouldClearEarthResult,
} from './earthPredictModel.js';

function buildEarthResponse(overrides = {}) {
  const day = (offset) => ({
    field: [[280 + offset, 300 + offset], [260 + offset, 320 + offset]],
    minVal: 260 + offset,
    maxVal: 320 + offset,
    valid_cells: 4,
  });
  return {
    planet: 'earth',
    task_id: 12,
    dataset_id: 'earth_merra2_daily_v2',
    dataset_version: 'v2',
    dataset_fingerprint: 'a'.repeat(64),
    target: 'TO3',
    target_unit: 'DU',
    forecast_origin: '2021-07-08',
    origin_split: 'test',
    input_dates: ['2021-07-02', '2021-07-03', '2021-07-04', '2021-07-05', '2021-07-06', '2021-07-07', '2021-07-08'],
    target_dates: ['2021-07-09', '2021-07-10', '2021-07-11'],
    window: 7,
    horizon: 3,
    grid: {
      shape: [2, 2],
      latitude: [-45, 45],
      longitude: [-135, 135],
      latitude_range: [-90, 90],
      longitude_range: [-180, 180],
    },
    prediction: [day(1), day(2), day(3)],
    reference: [day(0), day(0), day(0)],
    residual: [
      { field: [[1, 1], [1, 1]], minVal: 1, maxVal: 1, valid_cells: 4 },
      { field: [[2, 2], [2, 2]], minVal: 2, maxVal: 2, valid_cells: 4 },
      { field: [[-3, -3], [-3, -3]], minVal: -3, maxVal: -3, valid_cells: 4 },
    ],
    metrics: {
      unit: 'DU',
      target: 'TO3',
      aggregation: 'user_forecast_origin_lead_grid_uniform',
      reference_available: true,
      overall: { rmse: 2.5, mae: 1.5 },
      by_lead: [
        { lead_day: 1, rmse: 1, mae: 0.5 },
        { lead_day: 2, rmse: 2, mae: 1.5 },
        { lead_day: 3, rmse: 3, mae: 2.5 },
      ],
    },
    ...overrides,
  };
}

test('hash query opens Earth mode only for the explicit earth mode', () => {
  assert.deepEqual(readEarthPredictRequestFromHash('#/predict?from=training&mode=earth'), {
    fromTraining: true,
    taskId: null,
  });
  assert.deepEqual(readEarthPredictRequestFromHash('#/predict?mode=earth&task_id=42'), {
    fromTraining: false,
    taskId: 42,
  });
  assert.equal(readEarthPredictRequestFromHash('#/predict?mode=trained'), null);
  assert.equal(readEarthPredictRequestFromHash('#/predict'), null);
  assert.equal(readEarthPredictRequestFromHash(''), null);
});

test('Earth cache key separates planet, dataset identity, task and origin', () => {
  const base = buildEarthPredictKey({
    taskId: 12,
    datasetId: 'earth_merra2_daily_v2',
    datasetVersion: 'v2',
    datasetFingerprint: 'a'.repeat(64),
    forecastOrigin: '2021-07-08',
    targetDates: ['2021-07-09', '2021-07-10', '2021-07-11'],
  });
  assert.match(base, /^planet:earth\|/);
  assert.match(base, /task:12/);
  assert.match(base, /origin:2021-07-08/);
  assert.match(base, /targets:2021-07-09,2021-07-10,2021-07-11/);

  const otherOrigin = buildEarthPredictKey({
    taskId: 12,
    datasetId: 'earth_merra2_daily_v2',
    datasetVersion: 'v2',
    datasetFingerprint: 'a'.repeat(64),
    forecastOrigin: '2021-07-09',
  });
  assert.notEqual(base, otherOrigin);

  assert.equal(buildEarthPredictKey({ taskId: 0, forecastOrigin: '2021-07-08' }), null);
  assert.equal(buildEarthPredictKey({ taskId: 12 }), null);

  const threeHourly = buildEarthPredictKey({
    taskId: 12,
    datasetId: 'earth_merra2_3hourly_v1',
    datasetVersion: 'v1',
    datasetFingerprint: 'b'.repeat(64),
    forecastOrigin: '2021-07-08T01:30:00Z',
    targetTimestamps: ['2021-07-08T04:30:00Z', '2021-07-08T07:30:00Z'],
  });
  assert.match(threeHourly, /origin:2021-07-08T01:30:00Z/);
  assert.match(threeHourly, /targets:2021-07-08T04:30:00Z,2021-07-08T07:30:00Z/);
});

test('Earth cadence and leads distinguish 3-hour tasks from daily tasks', () => {
  const cadence = getEarthCadence({ dataset_id: 'earth_merra2_3hourly_v1', frequency_hours: 3, step: 3 });
  assert.equal(cadence.threeHourly, true);
  assert.equal(earthLeadLabel(0, { dataset_id: 'earth_merra2_3hourly_v1' }), '+3h');
  assert.equal(earthLeadLabel(23, { dataset_id: 'earth_merra2_3hourly_v1' }), '+72h');
  assert.equal(earthLeadLabel(2, { dataset_id: 'earth_merra2_daily_v2' }), '+3');
  assert.equal(getEarthCadence({ dataset_id: 'earth_merra2_daily_v2' }).threeHourly, false);
});

test('changing the Earth result identity clears the previous result', () => {
  assert.equal(shouldClearEarthResult(null, 'planet:earth|task:1'), false);
  assert.equal(shouldClearEarthResult('planet:earth|task:1', 'planet:earth|task:1'), false);
  assert.equal(shouldClearEarthResult('planet:earth|task:1', 'planet:earth|task:2'), true);
});

test('field payload keeps DU values and uses the published grid', () => {
  const response = buildEarthResponse();
  const payload = buildEarthFieldPayload({
    response,
    dayIndex: 0,
    kind: 'prediction',
    colormap: 'inferno',
    colorRange: { min: 200, max: 400 },
  });
  assert.deepEqual(payload.lat, [-45, 45]);
  assert.deepEqual(payload.lon, [-135, 135]);
  assert.deepEqual(payload.color_range, { min: 200, max: 400 });
  assert.equal(payload.unit, 'DU');
  // 值必须原样保留：地球 TO3 本身就是 DU，不能套用火星的 μm-atm 换算。
  assert.deepEqual(payload.field[0], [281, 301]);
  assert.deepEqual(payload.coverage.latitude_range, [-90, 90]);

  // 网格与场形状不一致时返回 null 而不是画出错位的图。
  const mismatched = { ...response, grid: { ...response.grid, latitude: [-45] } };
  assert.equal(buildEarthFieldPayload({ response: mismatched, dayIndex: 0, kind: 'prediction', colormap: 'inferno' }), null);
  assert.equal(buildEarthFieldPayload({ response, dayIndex: 9, kind: 'prediction', colormap: 'inferno' }), null);
});

test('physical color range is shared by prediction and reference, residual is centred', () => {
  const ranges = resolveEarthColorRanges(buildEarthResponse());
  assert.deepEqual(ranges.physical, { min: 260, max: 323 });
  assert.deepEqual(ranges.residual, { min: -3, max: 3 });
});

test('metrics read only finite values', () => {
  const response = buildEarthResponse();
  assert.equal(readEarthMetric(response.metrics, 'rmse'), 2.5);
  assert.equal(readEarthMetric(response.metrics, 'mae'), 1.5);
  assert.equal(readEarthMetric({ overall: { rmse: Number.NaN } }, 'rmse'), null);
  assert.equal(readEarthLeadMetric(response.metrics, 3, 'rmse'), 3);
  assert.equal(readEarthLeadMetric(response.metrics, 9, 'rmse'), null);
  assert.equal(readEarthLeadMetric({ by_lead: [{ lead_step: 8, rmse: 4 }] }, 8, 'rmse'), 4);
  assert.equal(readEarthHorizonMetric({ by_horizon: [{ horizon_hours: 24, rmse: 5 }] }, 24, 'rmse'), 5);
});

test('origin selection follows the server provided range', () => {
  const origins = {
    start: '2020-01-07',
    end: '2021-12-28',
    count: 722,
    dates: ['2020-01-07', '2020-01-08', '2020-01-09', '2021-12-28'],
  };
  assert.equal(isOriginSelectable(origins, '2020-01-09'), true);
  assert.equal(isOriginSelectable(origins, '2020-01-06'), false);
  assert.equal(isOriginSelectable(origins, ''), false);
  // 没有显式日期列表时退化为闭区间判断。
  assert.equal(isOriginSelectable({ start: '2020-01-07', end: '2021-12-28' }, '2020-06-01'), true);
  assert.equal(isOriginSelectable({ start: '2020-01-07', end: '2021-12-28' }, '2022-01-01'), false);

  assert.equal(pickDefaultOrigin(origins, '2020-06-01'), '2020-01-09');
  assert.equal(pickDefaultOrigin(origins, '2019-01-01'), '2020-01-07');
  assert.equal(pickDefaultOrigin(origins), '2021-12-28');
  assert.equal(pickDefaultOrigin({ end: '2021-12-28' }), '2021-12-28');
});

test('server error codes map to translation keys without losing the message', () => {
  assert.equal(
    resolveEarthPredictErrorMessage({ code: 'earth_prediction_origin_out_of_range', message: 'x' }).key,
    'earthOriginOutOfRange'
  );
  assert.equal(resolveEarthPredictErrorMessage({ code: 'dataset_version_changed' }).key, 'earthDatasetChanged');
  assert.equal(resolveEarthPredictErrorMessage({ code: 'invalid_earth_training_artifact' }).key, 'earthArtifactInvalid');
  const unknown = resolveEarthPredictErrorMessage({ code: 'weird', message: 'boom' });
  assert.equal(unknown.key, null);
  assert.equal(unknown.fallback, 'boom');
});

test('Earth task selection only offers completed Earth tasks with weights', () => {
  const tasks = [
    { id: 1, status: 'completed', model_available: true, dataset_id: 'earth_merra2_daily_v2', custom_model_name: 'Earth A' },
    { id: 2, status: 'completed', model_available: true, dataset_id: 'openmars_mcd', custom_model_name: 'Mars A' },
    { id: 3, status: 'running', model_available: false, dataset_id: 'earth_merra2_daily_v2' },
    { id: 4, status: 'completed', model_available: true, is_earth_task: true, dataset_id: null, custom_model_name: 'Legacy flag' },
    { id: 5, status: 'completed', model_available: false, dataset_id: 'earth_merra2_daily_v2' },
  ];
  const options = getEarthTrainingModelOptions(tasks);
  assert.deepEqual(options.map((option) => option.id), []);
  const active = getEarthTrainingModelOptions([...tasks,
    { ...tasks[0], id: 6, dataset_id: 'earth_merra2_3hourly_v1' },
    { ...tasks[0], id: 7, dataset_id: 'earth_merra2_3hourly_v1', model_available: false },
  ]);
  assert.deepEqual(active.map((option) => option.id), [6]);
  assert.equal(active[0].label, 'Earth A');
  assert.equal(getEarthTrainingModelOptions(null).length, 0);

  assert.equal(isEarthTask(tasks[0]), true);
  assert.equal(isEarthTask(tasks[1]), false);
  assert.equal(isEarthTask(tasks[3]), true);
  assert.equal(isEarthTask(null), false);
});

test('Earth split label source normalizes published and task-split boundaries without inventing counts', () => {
  const published = {
    split_policy: 'published_manifest_splits',
    split_ranges: {
      train: { date_start: '2020-01-01T01:30:00Z', date_end: '2021-06-30T22:30:00Z', window_count: 4200 },
      test: { date_start: '2021-11-01T01:30:00Z', date_end: '2021-12-28T22:30:00Z', window_count: 128 },
    },
  };
  assert.deepEqual(buildEarthSplitLabelSource(published), { split_meta: {
    source: 'published_manifest_splits',
    test_range: { date_start: '2021-11-01T01:30:00Z', date_end: '2021-12-28T22:30:00Z', window_count: 128 },
  } });
  // Daily-era manifests used start/end instead of date_start/date_end.
  const legacyKeys = { split_policy: 'published_manifest_splits',
    split_ranges: { test: { start: '2021-01-01', end: '2021-01-31', window_count: 24 } } };
  assert.equal(buildEarthSplitLabelSource(legacyKeys).split_meta.test_range.date_start, '2021-01-01');

  const taskSplit = {
    split_policy: 'earth_task_split_v1',
    split_ratios: { train_ratio: .7, validation_ratio: .2, test_ratio: .1 },
    task_split: { policy: 'earth_task_split_v1', ranges: {
      test: { date_start: '2021-11-05T00:00:00Z', date_end: '2021-12-28T21:00:00Z', step_count: 440, window_count: 361 },
    } },
  };
  const built = buildEarthSplitLabelSource(taskSplit);
  assert.equal(built.split_meta.source, 'earth_task_split_v1');
  assert.equal(built.split_meta.test_range.window_count, 361);
  assert.deepEqual(built.split_meta.ratios, taskSplit.split_ratios);

  // Unsupported policy, missing test range and missing boundaries must not produce a split line.
  assert.deepEqual(earthSplitRanges({ split_ranges: { test: { date_start: '2021-01-01' } } }), {});
  assert.equal(buildEarthSplitLabelSource({ split_policy: 'legacy_compatibility',
    split_ranges: published.split_ranges }), null);
  assert.equal(buildEarthSplitLabelSource({}), null);
  assert.equal(buildEarthSplitLabelSource(null), null);
});
