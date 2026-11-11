import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('./EarthCompareWorkspace.jsx', import.meta.url), 'utf8');
const pageSource = readFileSync(new URL('../PredictPage.jsx', import.meta.url), 'utf8');

test('Earth comparison clears stale results and coordinates the current selection', () => {
  assert.match(source, /createPredictRequestCoordinator/);
  assert.match(source, /PREDICT_REQUEST_CHANNELS\.compareMetrics/);
  assert.match(source, /requestCoordinator\.invalidateAll\(\)/);
  assert.match(source, /\}, \[key, options, requestCoordinator\]\)/);
  assert.match(source, /setData\(null\);\s*setLoading\(true\);\s*setError\(null\)/);
  assert.match(source, /compareEarthModels\(selection\.ids, \{ signal: requestToken\.signal \}\)/);
  assert.match(source, /requestCoordinator\.isCurrent\(requestToken, requestKey\)/);
  assert.match(source, /compareResultMatchesTasks\(result, selectedOptions\.map\(option => option\.task\), 'earth'\)/);
  assert.ok(source.indexOf('if (!compareResultMatchesTasks(result') < source.indexOf('setData(result)'));
  assert.match(source, /catch \(e\) \{[\s\S]*?setData\(null\);/);
  assert.match(source, /<CompareTrainingModelsPanel[^>]*data=\{data\}/);
});

test('Earth result rendering remains keyed to the active selection and Mars comparison keeps guarded requests', () => {
  assert.match(source, /const key = `\$\{scope\}:\$\{selection\.ids\.join\(','\)\}`/);
  assert.match(source, /<CompareTrainingModelsPanel planet="earth" key=\{key\}/);
  assert.match(pageSource, /PREDICT_REQUEST_CHANNELS\.compareMetrics/);
  assert.match(pageSource, /isRequestCurrent\(requestToken, requestContextKey\)/);
  assert.match(pageSource, /setCompareTrainingMetricsData\(compareResult\)/);
});
