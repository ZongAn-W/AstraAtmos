import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildEarthDiagnosticParameters, createEarthDiagnosticRequestGuard, earthDiagnosticTaskAvailability,
  earthDiagnosticScatter, earthDiagnosticHistogram, earthDiagnosticErrorText, earthDiagnosticIdentityKey,
  earthDiagnosticPfiItems,
  earthDiagnosticMatchesIdentity, earthDiagnosticRequestKey,
} from './earthDiagnosticsModel.js';

test('Earth diagnostics use identical bounded defaults in test modal and prediction analysis', () => {
  assert.deepEqual(buildEarthDiagnosticParameters('31'), { training_task_id: 31, sample_windows: 4,
    scatter_points: 5000, histogram_bins: 40, seed: 42, pfi_repeats: 3, include_pfi: true });
  assert.equal(buildEarthDiagnosticParameters(31, { sample_windows: 1, pfi_repeats: 5 }).pfi_repeats, 5);
  for (const options of [{ sample_windows: 9 }, { pfi_repeats: 6 }, { scatter_points: 20001 },
    { histogram_bins: 4 }, { seed: -1 }]) {
    assert.throws(() => buildEarthDiagnosticParameters(31, options));
  }
  assert.throws(() => buildEarthDiagnosticParameters('bad'));
  assert.doesNotThrow(() => earthDiagnosticIdentityKey(null, null));
  assert.equal(buildEarthDiagnosticParameters(31, { seed: 4294967295 }).seed, 4294967295);
  assert.throws(() => buildEarthDiagnosticParameters(31, { seed: 4294967296 }));
  assert.equal(buildEarthDiagnosticParameters(31, { scatter_points: 20000, histogram_bins: 5 }).scatter_points, 20000);
});

test('tests require completed effective weights; retired Earth tasks remain disabled', () => {
  assert.equal(earthDiagnosticTaskAvailability({ status: 'running', model_available: true }).available, false);
  assert.equal(earthDiagnosticTaskAvailability({ status: 'completed', model_available: false }).available, false);
  assert.equal(earthDiagnosticTaskAvailability({ status: 'completed', model_available: true,
    dataset_id: 'earth_merra2_daily_v2' }).available, false);
  assert.equal(earthDiagnosticTaskAvailability({ status: 'completed', model_available: true,
    dataset_id: 'earth_merra2_3hourly_v1' }).available, true);
});

test('scatter preserves pairs and physical values including negative values; y=x uses both ranges', () => {
  const samples = { reference: [-3, 300, 340], prediction: [-2, 302, 335] };
  const output = earthDiagnosticScatter(samples);
  assert.deepEqual(output.reference, samples.reference);
  assert.deepEqual(output.prediction, samples.prediction);
  assert.ok(output.range[0] < -3 && output.range[1] > 340);
  assert.throws(() => earthDiagnosticScatter({ reference: [1], prediction: [] }), /paired/);
  assert.throws(() => earthDiagnosticScatter({ reference: [], prediction: [] }), /No valid/);
  assert.throws(() => earthDiagnosticScatter({ reference: [1, NaN], prediction: [2, 4] }), /Non-finite/);
});

test('histogram uses server counts and bin edges without resampling or dropping invalid values', () => {
  assert.deepEqual(earthDiagnosticHistogram({ edges: [-3, -1, 1, 3], counts: [2, 0, 4] }),
    { centers: [-2, 0, 2], widths: [2, 2, 2], counts: [2, 0, 4], sampleCount: 6 });
  for (const histogram of [{ edges: [0, 1], counts: [0] }, { edges: [0, 0], counts: [1] },
    { edges: [0, Infinity], counts: [1] }, { edges: [0, 1], counts: [-1] }]) {
    assert.throws(() => earthDiagnosticHistogram(histogram));
  }
  const histogram = { edges: [-1, 0, 1], counts: [2, 4], valid_points: 6,
    residual_definition: 'prediction-reference', unit: 'DU', summary: { count: 6 } };
  assert.equal(earthDiagnosticHistogram(histogram, 6).sampleCount, 6);
  assert.throws(() => earthDiagnosticHistogram(histogram, 7), /declared evaluation points/);
  assert.throws(() => earthDiagnosticHistogram({ ...histogram, summary: { count: 5 } }), /declared evaluation points/);
  assert.throws(() => earthDiagnosticHistogram({ ...histogram, residual_definition: 'reference-prediction' }), /prediction-reference/);
});

test('switching task or sampling cancels the old request and late responses cannot be accepted', async () => {
  const guard = createEarthDiagnosticRequestGuard();
  const old = guard.start('task:31|windows:4'); old.taskId = 31;
  let resolveOld;
  const late = new Promise((resolve) => { resolveOld = resolve; });
  const current = guard.start('task:32|windows:4'); current.taskId = 32;
  assert.equal(old.signal.aborted, true);
  resolveOld({ task_id: 31 });
  assert.equal(guard.accepts(old, await late), false);
  assert.equal(guard.accepts(current, { task_id: 31 }), false);
  assert.equal(guard.accepts(current, { task_id: 32 }), true);
  const changed = guard.start('task:32|windows:8'); changed.taskId = 32;
  assert.equal(guard.accepts(current, { task_id: 32 }), false);
  assert.equal(guard.accepts(changed, { task_id: 32 }), true);
  guard.invalidate();
  assert.equal(changed.signal.aborted, true);
  assert.equal(guard.accepts(changed, { task_id: 32 }), false);
});

test('artifact/data/window/channel changes have separate request identities', () => {
  const base = { dataset_id: 'earth_merra2_3hourly_v1', dataset_fingerprint: 'release-1', window: 56, horizon: 24,
    input_variables: ['TO3', 'U'], model: { source_hash: 'a' } };
  for (const changed of [{ dataset_fingerprint: 'release-2' }, { window: 71 }, { horizon: 9 },
    { input_variables: ['TO3', 'T'] }, { model: { source_hash: 'b' } }, { checkpoint_sha256: 'weights-b' },
    { metrics: { split_ranges: { test: [800, 1000] } } }, { dataset_id: 'different_release' }]) {
    assert.notEqual(earthDiagnosticIdentityKey(31, base), earthDiagnosticIdentityKey(31, { ...base, ...changed }));
  }
});

test('account and parameter changes clear request identity independently of a shared task', () => {
  const parameters = buildEarthDiagnosticParameters(31);
  const key = earthDiagnosticRequestKey(31, null, parameters, 7);
  assert.notEqual(key, earthDiagnosticRequestKey(31, null, parameters, 8));
  assert.notEqual(key, earthDiagnosticRequestKey(31, null, parameters, null));
  assert.notEqual(key, earthDiagnosticRequestKey(31, null, { ...parameters, seed: 43 }, 7));
  assert.notEqual(key, earthDiagnosticRequestKey(31, null, { ...parameters, pfi_repeats: 5 }, 7));
});

test('a current response with changed data, artifact, channels or physical unit is rejected', () => {
  const identity = { dataset_id: 'earth_merra2_3hourly_v1', dataset_version: 'v1', dataset_fingerprint: 'fp',
    checkpoint_sha256: 'weights', window: 71, horizon: 9, input_channel_order: ['TO3', 'U'] };
  const response = { ...identity, task_id: 31, planet: 'earth', target_unit: 'DU' };
  assert.equal(earthDiagnosticMatchesIdentity(response, identity), true);
  for (const change of [{ planet: 'mars' }, { target_unit: 'normalized' }, { dataset_fingerprint: 'other' },
    { checkpoint_sha256: 'other' }, { window: 56 }, { horizon: 24 }, { input_channel_order: ['U', 'TO3'] }]) {
    assert.equal(earthDiagnosticMatchesIdentity({ ...response, ...change }, identity), false);
  }
  const guard = createEarthDiagnosticRequestGuard();
  const token = guard.start('current'); token.taskId = 31; token.identity = identity;
  assert.equal(guard.accepts(token, { ...response, dataset_fingerprint: 'changed' }), false);
  assert.equal(guard.accepts(token, response), true);
});

test('Earth diagnostic identity reads computed task inputs and compares returned input variables', () => {
  const identity = { target_variable: 'T2M', target_unit: 'K', input_variables: ['TO3', 'U10M'],
    dataset_id: 'earth_merra2_3hourly_v1' };
  const response = { task_id: 41, planet: 'earth', target_variable: 'T2M', target_unit: 'K',
    input_variables: ['TO3', 'U10M'], dataset_id: identity.dataset_id };
  assert.equal(earthDiagnosticMatchesIdentity(response, identity), true);
  assert.equal(earthDiagnosticMatchesIdentity({ ...response, input_variables: ['TO3', 'SWGDN'] }, identity), false);
  assert.equal(earthDiagnosticMatchesIdentity({ ...response, input_variables: undefined }, identity), false);
  assert.equal(earthDiagnosticMatchesIdentity({ ...response, target_unit: undefined }, identity), false);
  assert.notEqual(earthDiagnosticIdentityKey(41, identity), earthDiagnosticIdentityKey(41,
    { ...identity, input_variables: ['TO3', 'SWGDN'] }));
  assert.equal(earthDiagnosticMatchesIdentity({ task_id: 41, dataset_id: identity.dataset_id,
    target_variable: 'T2M', target_unit: 'K' }, identity), false);
  const legacyIdentity = { dataset_id: identity.dataset_id, target_variable: 'TO3', target_unit: 'DU' };
  assert.equal(earthDiagnosticMatchesIdentity({ task_id: 41, dataset_id: identity.dataset_id,
    target_variable: 'TO3', target_unit: 'DU' }, legacyIdentity), true);
});

test('structured failures render a readable message including availability context', () => {
  assert.equal(earthDiagnosticErrorText({ reason: { code: 'invalid_artifact', message: 'Weight file changed' } }), 'Weight file changed');
  assert.equal(earthDiagnosticErrorText({ response: { data: { detail: { code: 'not_allowed', message: 'No access' } } } }), 'No access');
  assert.equal(earthDiagnosticErrorText(new Error('Synthetic model failed')), 'Synthetic model failed');
});

test('PFI retains negative and zero importances and unavailable channels, using the shared baseline', () => {
  const items = earthDiagnosticPfiItems({ baseline_rmse: 2, items: [
    { channel: 'TO3', importance: -1, importance_std: .1, repeats: [{ rmse: 1, delta_rmse: -1 }] },
    { channel: 'U', importance: 0, importance_std: 0, repeats: [{ rmse: 2, delta_rmse: 0 }] },
    { channel: 'T', importance: null, status: 'unavailable', reason: 'Identical windows', repeats: [] },
  ] });
  assert.equal(items[0].importance_mean, -1);
  assert.equal(items[1].importance_mean, 0);
  assert.equal(items[1].importance_std, 0);
  assert.equal(items[0].baseline_rmse, 2);
  assert.equal(items[2].status, 'unavailable');
});
