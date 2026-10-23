import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const pageSource = fs.readFileSync(path.join(currentDir, '..', 'PredictPage.jsx'), 'utf8');
const aiPageSource = fs.readFileSync(path.join(currentDir, '..', 'AIPage.jsx'), 'utf8');

test('prediction page resolves identity before reading scoped cache', () => {
  assert.match(pageSource, /resolvePredictCacheScope/);
  assert.match(pageSource, /const predictScope = resolvePredictCacheScope\(\{ user, isLoading \}\)/);
  assert.match(pageSource, /getEmptyPredictCache\(predictScope\)/);
  assert.doesNotMatch(pageSource, /const _c = getPredictCache\(\)/);
  assert.match(pageSource, /if \(!predictScope\) return/);
});

test('identity scope changes reset every sensitive result before restoring the new scope', () => {
  assert.match(pageSource, /clearPredictCache/);
  assert.match(pageSource, /previousScope\?\.startsWith\('user:'\)/);
  assert.match(pageSource, /useLayoutEffect\(\(\) => \{[\s\S]*?predictScope/);
  assert.match(pageSource, /requestCoordinator\.invalidateAll\(\)/);
  assert.match(pageSource, /setResults\(null\)/);
  assert.match(pageSource, /setMetrics\(null\)/);
  assert.match(pageSource, /setErrorDistData\(null\)/);
  assert.match(pageSource, /setPfiData\(null\)/);
  assert.match(pageSource, /setPerformanceData\(null\)/);
  assert.match(pageSource, /setSelectedTrainingTaskId\(null\)/);
  assert.match(pageSource, /setSelectedCompareTrainingTaskIds\(\[\]\)/);
  assert.match(pageSource, /setCompareTrainingMetricsData\(null\)/);
  assert.match(pageSource, /setCompareTrainingErrorData\(null\)/);
  assert.match(pageSource, /setCompareTrainingPfiData\(null\)/);
  assert.match(pageSource, /setError\(null\)/);
  assert.match(pageSource, /setLoading\(false\)/);
  assert.match(pageSource, /setFullscreen3D\(null\)/);
  assert.match(pageSource, /getPredictCache\(predictScope\)/);
});

test('all prediction cache reads and writes carry the resolved identity scope', () => {
  assert.doesNotMatch(pageSource, /getPredictCache\(\s*\)/);
  assert.doesNotMatch(pageSource, /setPredictCache\(\s*\{/);
  assert.match(pageSource, /const scope = predictScopeRef\.current/);
  assert.match(pageSource, /setPredictCache\(scope, updates\)/);
  assert.match(pageSource, /setPredictUiPreferences\(\{ viewMode \}\)/);
});

test('async prediction completions require the request-start identity scope', () => {
  assert.match(pageSource, /scope:\s*predictScopeRef\.current/);
  assert.match(pageSource, /requestToken\.scope === predictScopeRef\.current/);
  assert.match(pageSource, /predictScopeRef\.current === predictScope/);
  assert.match(pageSource, /PREDICT_REQUEST_CHANNELS\.performance/);
});

test('cached trained tasks are restored only after current-user task validation', () => {
  assert.match(pageSource, /validatePredictCacheTrainingTasks/);
  assert.match(pageSource, /trainingTasksLoaded/);
  assert.match(pageSource, /trainingModelOptions\.map\(\(option\) => option\.id\)/);
  assert.match(pageSource, /setPredictCache\(predictScope, restoredCache\)/);
});

test('training task loading keeps failure distinct from an empty successful result', () => {
  assert.match(pageSource, /const \[trainingTasksError, setTrainingTasksError\] = useState\(null\)/);
  assert.match(pageSource, /const loadTrainingTasks = useCallback\(\(\) =>/);
  assert.match(pageSource, /fetchTasks\(\{ signal: controller\.signal \}\)/);
  assert.match(pageSource, /setTrainingTasksLoaded\(true\)/);
  assert.match(pageSource, /setTrainingTasksError\(requestError/);
  assert.match(pageSource, /\.then\(\(items\) => \{[\s\S]*?setTrainingTasks\(Array\.isArray\(items\) \? items : \[\]\)/);
});

test('training task responses are guarded by request id, user, and scope', () => {
  assert.match(pageSource, /trainingTasksRequestRef\.current\.controller\?\.abort\(\)/);
  assert.match(pageSource, /trainingTasksRequestRef\.current\.id === requestId/);
  assert.match(pageSource, /predictScopeRef\.current === scope/);
  assert.match(pageSource, /user\?\.id === userId/);
  assert.match(pageSource, /onRetryTrainingTasks=\{loadTrainingTasks\}/);
  assert.match(pageSource, /earthContextRequestRef\.current !== controller \|\| scope !== predictScopeRef\.current/);
  assert.match(pageSource, /Number\(payload\.task_id\) !== normalized \|\| payload\.planet !== 'earth'/);
});

test('single and comparison analyses restore only for their exact request keys', () => {
  assert.match(pageSource, /metricsKey:\s*buildPredictMetricsKey\(restoredContext\)/);
  assert.match(pageSource, /errorDistKey:\s*buildErrorDistributionKey\(restoredContext\)/);
  assert.match(pageSource, /pfiKey:\s*buildPermutationImportanceKey\(restoredContext\)/);
  assert.match(pageSource, /validatedCache\.compareTrainingMetricsKey === restoredCompareMetricsKey/);
  assert.match(pageSource, /validatedCache\.compareTrainingErrorKey === restoredCompareErrorKey/);
  assert.match(pageSource, /validatedCache\.compareTrainingPfiKey === restoredComparePfiKey/);
});

test('AI page reads prediction context only from its resolved identity scope', () => {
  assert.match(aiPageSource, /useAuth/);
  assert.match(aiPageSource, /resolvePredictCacheScope/);
  assert.match(aiPageSource, /getPredictCache\(predictScope\)/);
  assert.doesNotMatch(aiPageSource, /getPredictCache\(\s*\)/);
});
