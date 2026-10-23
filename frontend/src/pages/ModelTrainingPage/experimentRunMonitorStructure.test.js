import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const monitorSource = readFileSync(new URL('./ExperimentRunMonitor.jsx', import.meta.url), 'utf8');
const logPanelSource = readFileSync(new URL('./ExperimentLogPanel.jsx', import.meta.url), 'utf8');
const progressSource = readFileSync(new URL('../../components/TrainingProgressMonitor.jsx', import.meta.url), 'utf8');
const chartSource = readFileSync(new URL('../../components/LossEvolutionChart.jsx', import.meta.url), 'utf8');
const contextSource = readFileSync(new URL('../../contexts/TrainingContext.jsx', import.meta.url), 'utf8');
const pageSource = readFileSync(new URL('../ModelTrainingPage.jsx', import.meta.url), 'utf8');

test('run monitor reuses the existing progress monitor, loss chart and shared log panel', () => {
  assert.match(monitorSource, /<TrainingProgressMonitor/);
  assert.match(monitorSource, /<LossEvolutionChart/);
  assert.match(monitorSource, /<ExperimentLogPanel/);
  assert.match(monitorSource, /compact/);
});

test('run monitor surfaces status, epoch, loss and eta through the shared progress monitor', () => {
  assert.match(monitorSource, /copy\.currentStatus/);
  // 百分比、Epoch、Loss、ETA 与状态都由既有进度组件承担，页面不再重复渲染一套指标卡。
  assert.match(monitorSource, /<TrainingProgressMonitor/);
  assert.match(monitorSource, /currentEpoch=\{resolvedProgress\.current_epoch/);
  assert.match(monitorSource, /loss=\{resolvedProgress\.current_loss\}/);
  assert.match(monitorSource, /eta=\{resolvedProgress\.eta/);
  assert.doesNotMatch(monitorSource, /modelTraining\.statsProgress/);
  assert.doesNotMatch(monitorSource, /experiment-monitor-metric-/);
});

test('run monitor explains a failed run that never reached the result stage', () => {
  assert.match(monitorSource, /getExperimentFailureMessage\(activeTask\)/);
  assert.match(monitorSource, /failure\?\.hasReason/);
  assert.match(monitorSource, /t\(failure\.messageKey\)/);
  assert.match(monitorSource, /failure\.suggestionKey/);
});

test('shared log panel keeps line numbers, tones, live region and auto-follow state', () => {
  assert.match(logPanelSource, /experiment-monitor-log-row/);
  assert.match(logPanelSource, /getExperimentLogLineTone/);
  assert.match(logPanelSource, /role="log"/);
  assert.match(logPanelSource, /aria-live="polite"/);
  assert.match(logPanelSource, /copy\.autoScrollOn/);
  assert.match(logPanelSource, /copy\.autoScrollPaused/);
  assert.match(logPanelSource, /copy\.logLines\(lines\.length\)/);
  assert.match(logPanelSource, /String\(index \+ 1\)\.padStart\(3, '0'\)/);
});

test('shared log panel never requests logs or creates its own polling', () => {
  for (const source of [monitorSource, logPanelSource]) {
    assert.doesNotMatch(source, /setInterval/);
    assert.doesNotMatch(source, /clearInterval/);
    assert.doesNotMatch(source, /new WebSocket/);
    assert.doesNotMatch(source, /fetchLogs/);
    assert.doesNotMatch(source, /startTrainingTask/);
    assert.doesNotMatch(source, /window\.addEventListener/);
  }
});

test('run monitor shows the active task id and empty states', () => {
  assert.match(logPanelSource, /task #\$\{task\.id\}/);
  assert.match(logPanelSource, /copy\.selectedTask/);
  assert.match(logPanelSource, /copy\.noTaskSelectedHint/);
  assert.match(logPanelSource, /emptyLabel \|\| copy\.waitingLogs/);
});

test('stop training only appears for pending or running tasks and is labelled', () => {
  assert.match(monitorSource, /isActiveTrainingStatus\(status\)/);
  assert.match(monitorSource, /isActive && status !== 'queued' && activeTask \?/);
  assert.match(monitorSource, /aria-label=\{`\$\{copy\.stopTraining\} #\$\{activeTask\.id\}`\}/);
  assert.match(monitorSource, /onClick=\{\(\) => onStop\(activeTask\.id\)\}/);
  // 目录精简后，排队任务仍能在所选实验的监控区取消。
  assert.match(monitorSource, /status === 'queued' && activeTask \?/);
  assert.match(monitorSource, /aria-label=\{`\$\{copy\.cancelQueued\} #\$\{activeTask\.id\}`\}/);
  assert.match(monitorSource, /onClick=\{\(\) => onCancel\(activeTask\.id\)\}/);
  assert.match(pageSource, /onCancel=\{handleCancelTask\}/);
});

test('training context keeps task polling, log polling and websocket ownership', () => {
  assert.match(contextSource, /setInterval\(\(\) => loadTasks\(\{ reusePending: true \}\)\.catch\(\(\) => \{\}\), 5000\)/);
  assert.match(contextSource, /setInterval\(pollLogs, 3000\)/);
  assert.match(contextSource, /new WebSocket\(wsUrl\)/);
  assert.match(contextSource, /training_update/);
});

test('page controller derives the monitor stage from the active task status', () => {
  assert.match(pageSource, /getExperimentStage\(\{ activeTask, isCreating \}\)/);
  assert.match(pageSource, /resolveActiveTaskProgress\(activeTask, progressData\)/);
  assert.match(pageSource, /onScroll=\{handleScroll\}/);
  assert.match(pageSource, /logContainerRef/);
});

test('loss chart keeps its default behaviour and gains a compact mode plus an empty label', () => {
  assert.match(chartSource, /compact = false/);
  assert.match(chartSource, /const chartHeight =/);
  assert.match(chartSource, /height: chartHeight/);
  assert.match(chartSource, /emptyLabel \|\| t\('modelTraining\.charts\.noMetrics'\)/);
  assert.match(chartSource, /trainLoss\.length === 0 \?/);
});

test('progress monitor keeps a single shared status definition', () => {
  assert.match(progressSource, /getTrainingStatusMeta/);
  assert.doesNotMatch(progressSource, /function getStatusMeta/);
});
