import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const inspectorSource = readFileSync(new URL('./ExperimentConfigInspector.jsx', import.meta.url), 'utf8');
const runBarSource = readFileSync(new URL('./ExperimentRunBar.jsx', import.meta.url), 'utf8');
const pageSource = readFileSync(new URL('../ModelTrainingPage.jsx', import.meta.url), 'utf8');
const cssSource = readFileSync(new URL('./experimentCenter.css', import.meta.url), 'utf8');

/** 去掉块注释与整行注释：文档注释会提到接口名称，断言“没有调用”时必须只看代码。 */
function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split('\n')
    .filter((line) => !line.trim().startsWith('//') && !line.trim().startsWith('*'))
    .join('\n');
}

test('配置检查器只保留当前模型与需要处理的问题', () => {
  assert.match(inspectorSource, /data-config-inspector="true"/);
  assert.match(inspectorSource, /copy\.inspectorCurrentModel/);
  assert.match(inspectorSource, /data-inspector-field="current-model"/);
  assert.match(inspectorSource, /data-inspector-field="model-source"/);
  assert.match(inspectorSource, /copy\.inspectorReadiness/);
  assert.match(inspectorSource, /data-inspector-issues="true"/);
  // 信息减法：常态输入变量 / 时序 / 数据集摘要与折叠详情已删除，
  // 这些值在主表单的真实控件里都能读到。
  assert.doesNotMatch(inspectorSource, /data-inspector-field="payload"/);
  assert.doesNotMatch(inspectorSource, /data-inspector-field="channel-count"/);
  assert.doesNotMatch(inspectorSource, /data-inspector-field="window-horizon"/);
  assert.doesNotMatch(inspectorSource, /data-inspector-field="epochs"/);
  assert.doesNotMatch(inspectorSource, /data-inspector-field="dataset"/);
  assert.doesNotMatch(inspectorSource, /experiment-inspector-telemetry/);
  assert.doesNotMatch(inspectorSource, /experiment-inspector-details/);
  // 正常时只有一行结论，不逐条重复「已通过」。
  assert.match(inspectorSource, /copy\.inspectorAllClear/);
  // 不再重复模型家族、官方架构数量、完整训练参数与多个大按钮。
  assert.doesNotMatch(inspectorSource, /MODEL_ARCHITECTURE_FAMILIES/);
  assert.doesNotMatch(inspectorSource, /inspectorFamily/);
  assert.doesNotMatch(inspectorSource, /inspectorCandidates/);
  assert.doesNotMatch(inspectorSource, /learningRate/);
  assert.doesNotMatch(inspectorSource, /batchSize/);
  assert.doesNotMatch(inspectorSource, /experiment-inspector-action/);
});

test('检查器不编造资源估算', () => {
  // 曾用 window - horizon + 1 推导“训练样本”，那不是数据集长度。
  assert.doesNotMatch(inspectorSource, /sampleCount/);
  assert.doesNotMatch(inspectorSource, /stepsPerEpoch/);
  assert.doesNotMatch(inspectorSource, /inspectorEstimate/);
  assert.doesNotMatch(inspectorSource, /formatNumber/);
  assert.match(pageSource, /const readiness = useMemo\(\(\) => \{/);
});

test('检查器不复制整张表单，也不再有折叠详情', () => {
  assert.doesNotMatch(inspectorSource, /<input/);
  assert.doesNotMatch(inspectorSource, /<select/);
  assert.doesNotMatch(inspectorSource, /<textarea/);
  assert.doesNotMatch(inspectorSource, /<DynamicModelParamsForm/);
  assert.doesNotMatch(inspectorSource, /<TagPicker/);
  // 信息减法：折叠详情整块删除，不再需要 open 状态。
  assert.doesNotMatch(inspectorSource, /useState/);
  assert.doesNotMatch(inspectorSource, /experiment-inspector-details/);
});

test('就绪状态与主按钮可点性分开，避免“有错误却显示可以开始训练”', () => {
  // 页面控制器统一算就绪状态：canTrain 才是能开始真实训练。
  assert.match(pageSource, /const readiness = useMemo\(\(\) => \{/);
  assert.match(pageSource, /return \{ canTrain: blockers\.length === 0, blockers \};/);
  assert.match(pageSource, /if \(!user\) blockers\.push\(\{ code: 'login'/);
  assert.match(pageSource, /code: 'name-missing'/);
  assert.match(pageSource, /code: 'model-missing'/);
  // startDisabled 仍然负责按钮可点性（访客可点，用于弹登录）。
  assert.match(pageSource, /const startDisabled = user/);
  // 检查器 / 画布 / 运行条都读 readiness。
  assert.match(inspectorSource, /readiness\?\.canTrain/);
  assert.match(inspectorSource, /data-inspector-ready=\{canTrain \? 'true' : 'false'\}/);
  assert.match(inspectorSource, /data-inspector-state=\{canTrain \? 'ready' : \(user \? 'blocked' : 'guest'\)\}/);
  assert.match(runBarSource, /data-run-ready=\{canTrain \? 'true' : 'false'\}/);
  assert.match(runBarSource, /data-run-interactive=\{startDisabled \? 'false' : 'true'\}/);
  assert.match(runBarSource, /copy\.runBarReadyToStart/);
  assert.match(runBarSource, /copy\.runBarNeedsAttention/);
  assert.match(runBarSource, /copy\.runBarGuest/);
  // 检查器只列出「需要处理」的原因，正常时一行结论，不会全绿却同时列出问题。
  assert.match(inspectorSource, /data-inspector-clear="true"/);
  assert.match(inspectorSource, /data-inspector-issues="true"/);
});

test('检查器与运行条都不请求训练接口、不创建轮询', () => {
  for (const source of [stripComments(inspectorSource), stripComments(runBarSource)]) {
    assert.doesNotMatch(source, /startTrainingTask/);
    assert.doesNotMatch(source, /fetchLogs/);
    assert.doesNotMatch(source, /setInterval/);
    assert.doesNotMatch(source, /new WebSocket/);
    assert.doesNotMatch(source, /useTraining\(/);
  }
});

test('运行条是横跨浏览器的细条：左侧状态，右侧唯一主按钮', () => {
  assert.match(runBarSource, /data-run-bar="true"/);
  assert.match(runBarSource, /experiment-run-bar-info/);
  assert.match(runBarSource, /experiment-run-bar-state/);
  assert.match(runBarSource, /experiment-run-bar-actions/);
  assert.match(runBarSource, /data-run-bar-start="true"/);
  assert.match(runBarSource, /onClick=\{onStart\}/);
  assert.match(runBarSource, /disabled=\{startDisabled\}/);
  assert.match(runBarSource, /pendingSubmission/);
  assert.match(pageSource, /onStart=\{handleStartTraining\}/);
  // 信息减法：数据集 / 模型 / 输入 / 轮数的长摘要已删除，这些值在画布控件里可读。
  assert.doesNotMatch(runBarSource, /data-run-bar-summary/);
  assert.doesNotMatch(runBarSource, /datasetLabel|channelLabel/);
  assert.doesNotMatch(runBarSource, /data-run-bar-field/);
});

test('运行条样式：fixed 细条、细边线、主按钮尺寸', () => {
  assert.match(cssSource, /\.experiment-run-bar\s*\{[^}]*position:\s*fixed/s);
  assert.match(cssSource, /\.experiment-run-bar\s*\{[^}]*border-top:\s*1px solid/s);
  assert.match(cssSource, /\.experiment-run-bar-start\s*\{[^}]*min-height:\s*44px/s);
  // 放大文字时允许换行而不是裁掉内容。
  assert.match(cssSource, /\.experiment-run-bar-info\s*\{[^}]*min-width:\s*0/s);
});

test('页面控制器仍然独占训练提交、日志轮询与阶段推导', () => {
  assert.match(pageSource, /const handleStartTraining = async \(\) => \{/);
  assert.match(pageSource, /await startTrainingTask\(/);
  assert.match(pageSource, /await stopTrainingTask\(taskId\)/);
  assert.match(pageSource, /useTraining\(\)/);
  assert.match(pageSource, /getExperimentStage\(\{ activeTask, isCreating \}\)/);
  assert.match(pageSource, /<ExperimentRunMonitor/);
  assert.match(pageSource, /<ExperimentResultPanel/);
  assert.match(pageSource, /<ExperimentDirectory/);
  assert.match(pageSource, /const uploadedModelStartBlocked =/);
});

test('访客 / 登出不再被悄悄切回官方模型', () => {
  // 历史上登出会把 modelSource 改回 official，掩盖了“上传模型优先”的默认值。
  const pageCode = stripComments(pageSource);
  const logoutEffect = pageCode.slice(
    pageCode.indexOf('if (!user) {'),
    pageCode.indexOf('fetchUserModels()')
  );
  assert.doesNotMatch(logoutEffect, /setModelSource/);
  assert.doesNotMatch(pageCode, /setModelSource\('official'\)/);
  assert.match(pageSource, /const \[modelSource, setModelSource\] = useState\('uploaded'\)/);
  // 用户主动切换与复制配置仍然按原业务规则保留来源。
  assert.match(pageSource, /onModelSourceChange: handleModelSourceChange/);
  assert.match(pageSource, /setModelSource\(config\.modelSource\)/);
});

test('检查器与运行条的响应式规则不会造成横向溢出', () => {
  const narrow = cssSource.slice(cssSource.indexOf('@media (max-width: 900px)'));
  assert.match(narrow, /\.experiment-inspector-panel\s*\{[\s\S]{0,120}?position:\s*static/);
  assert.match(narrow, /\.experiment-run-bar\s*\{\s*\n\s*position:\s*static/);
  assert.match(cssSource, /@media \(max-width: 640px\)[\s\S]*?\.experiment-run-bar-start\s*\{[^}]*width:\s*100%/s);
  assert.match(cssSource, /\.experiment-run-bar-info\s*\{[^}]*min-width:\s*0/s);
});
