import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const configSource = readFileSync(new URL('./ExperimentConfigWorkspace.jsx', import.meta.url), 'utf8');
const inspectorSource = readFileSync(new URL('./ExperimentConfigInspector.jsx', import.meta.url), 'utf8');
const runBarSource = readFileSync(new URL('./ExperimentRunBar.jsx', import.meta.url), 'utf8');
const pageSource = readFileSync(new URL('../ModelTrainingPage.jsx', import.meta.url), 'utf8');
const lockTestSource = readFileSync(new URL('./transferSourceLockStructure.test.js', import.meta.url), 'utf8');

/**
 * 去掉块注释与整行注释：文档注释会提到训练接口名称，
 * 断言“没有调用训练接口”时必须只看真实代码。
 */
function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split('\n')
    .filter((line) => !line.trim().startsWith('//') && !line.trim().startsWith('*'))
    .join('\n');
}

const configCode = stripComments(configSource);

/** 断言 token 在源码中的出现顺序，用于锁定配置画布的信息层级。 */
function assertOrder(source, tokens) {
  let cursor = -1;
  tokens.forEach((token) => {
    const index = source.indexOf(token, cursor + 1);
    assert.notEqual(index, -1, `missing token: ${token}`);
    assert.ok(index > cursor, `token out of order: ${token}`);
    cursor = index;
  });
}

test('canvas head carries the editable name and no duplicated configuration summary', () => {
  assertOrder(configSource, [
    'experiment-canvas-head',
    'data-config-group="name"',
    'experiment-canvas-name',
    'data-config-group="dataset"',
  ]);
  // 旧的大段「当前配置」摘要已删除，摘要集中在检查器。
  assert.doesNotMatch(configSource, /experiment-config-summary/);
  assert.doesNotMatch(configSource, /copy\.summaryLabel/);
  assert.doesNotMatch(configSource, /copy\.summaryHint/);
  // 信息减法：画布头部不再重复任务说明与就绪胶囊（就绪由运行条与检查器播报）。
  assert.doesNotMatch(configSource, /experiment-canvas-meta/);
  assert.doesNotMatch(configSource, /experiment-canvas-state/);
  // 字段级错误仍然保留在名称输入旁。
  assert.match(configSource, /experiment-canvas-name-error/);
  assert.match(configSource, /aria-errormessage=\{modelNameError/);
  // 四个分区同一套卡片与轨道：01 不再有旧的蓝色渐变。
  assert.match(
    readFileSync(new URL('./experimentCenter.css', import.meta.url), 'utf8'),
    /\.experiment-canvas-head,\s*\.experiment-canvas-section\s*\{[^}]*border-radius:\s*13px[^}]*background:\s*var\(--experiment-surface-1\)/s
  );
});

test('configuration canvas keeps the agreed section order', () => {
  // 画布四个部分：01 模型名称 → 02 数据集 → 03 模型 → 04 超参数（超参数内含页签）。
  assertOrder(configSource, [
    'data-config-group="name"',
    'copy.sectionName',
    'data-config-group="dataset"',
    'copy.sectionDataset',
    'data-config-group="model"',
    'copy.sectionModel',
    'data-config-group="expert"',
    'copy.sectionExpert',
    'role="tablist"',
  ]);
  // 四个分区都有序号，边界一眼可辨。
  assertOrder(configSource, [
    '<span className="experiment-section-index">01</span>',
    '<span className="experiment-section-index">02</span>',
    '<span className="experiment-section-index">03</span>',
    '<span className="experiment-section-index">04</span>',
  ]);
  assertOrder(configSource, [
    "const CONFIG_EXPERT_TABS = ['payload', 'training']",
    '...CONFIG_EXPERT_TABS',
  ]);
  assert.match(configSource, /experiment-choice-block/);
});

test('dataset and model each own one section', () => {
  const datasetBlock = configSource.slice(
    configSource.indexOf('data-config-group="dataset"'),
    configSource.indexOf('data-config-group="model"')
  );
  assert.match(datasetBlock, /copy\.trainingDataset/);
  assert.match(datasetBlock, /experiment-dataset-list/);
  assert.match(datasetBlock, /data-training-dataset-option=\{option\.value\}/);
  // 数据集不再用原生下拉，两个选项直接列出来。
  assert.doesNotMatch(datasetBlock, /<select/);
  assert.doesNotMatch(datasetBlock, /model-source-toggle/);

  const modelBlock = configSource.slice(
    configSource.indexOf('data-config-group="model"'),
    configSource.indexOf('data-config-group="expert"')
  );
  assert.match(modelBlock, /experiment-source-toggle/);
  assert.match(modelBlock, /data-model-source-option=\{option\.value\}/);
  assert.doesNotMatch(modelBlock, /experiment-dataset-list/);
  // 并排的两列任务定义网格已拆掉，两个部分各自独占分区。
  assert.doesNotMatch(configSource, /experiment-task-grid/);
  assert.match(
    readFileSync(new URL('./experimentCenter.css', import.meta.url), 'utf8'),
    /\.experiment-section-index\s*\{[^}]*font-family:\s*var\(--experiment-mono\)/s
  );
});

test('every hyperparameter tab renders the same field-block skeleton', () => {
  const payloadBlock = configSource.slice(
    configSource.indexOf('data-config-group="payload"'),
    configSource.indexOf('data-config-group="training"')
  );
  // 六个页签的正文都是「标题 + 同一套字段块」，不再各自一种形态。
  assert.match(payloadBlock, /experiment-expert-heading/);
  assert.match(payloadBlock, /experiment-payload-bar/);
  assert.match(payloadBlock, /experiment-channel-chip/);
  const trainingBlock = configSource.slice(
    configSource.indexOf('data-config-group="training"'),
    configSource.indexOf("activeTab === 'customParams'")
  );
  assert.match(trainingBlock, /experiment-expert-heading/);
  assert.match(trainingBlock, /experiment-param-grid/);
  // 标签选择器收进同一个块里，去掉自己的 fieldset 描边。
  assert.match(configSource, /experiment-tag-block[\s\S]{0,400}?<TagPicker/);

  const css = readFileSync(new URL('./experimentCenter.css', import.meta.url), 'utf8');
  assert.match(
    css,
    /\.experiment-payload-bar,\s*\.experiment-param-grid,\s*\.experiment-expert-fields,\s*\.experiment-expert-stack\s*\{[^}]*grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)/s
  );
  assert.match(
    css,
    /\.experiment-payload-lock,\s*\.experiment-channel-chip,\s*\.experiment-param-cell,\s*\.experiment-expert-field\s*\{[^}]*padding:\s*9px 10px[^}]*border-radius:\s*9px/s
  );
  assert.match(css, /\.experiment-tag-block \.training-tag-picker\s*\{[^}]*border:\s*0/s);
  // 标签选择器是独立控件：它的后代规则不能挂在字段块下，否则搜索框与标签项的
  // input / span 会被字段块的规则改掉（标签名曾被压成一列一个字）。
  assert.doesNotMatch(css, /\.experiment-expert-field \.training-tag-/);
  assert.match(css, /\.experiment-tag-block \.training-tag-options\s*\{[^}]*minmax\(150px,\s*1fr\)/s);

  // 自定义参数表单也用同一套块，并且不再自带重复标题。
  const formSource = readFileSync(new URL('./DynamicModelParamsForm.jsx', import.meta.url), 'utf8');
  assert.match(formSource, /className="experiment-expert-fields"/);
  assert.match(formSource, /className="experiment-expert-field"/);
  assert.match(formSource, /hideTitle/);
  assert.doesNotMatch(formSource, /model-training-field-grid/);
  assert.match(configSource, /hideTitle/);
});

test('uploaded model is the primary entry with a compact card and secondary management', () => {
  const sourceGroup = configSource.slice(
    configSource.indexOf('data-config-group="model"'),
    configSource.indexOf('data-config-group="expert"')
  );
  assert.match(sourceGroup, /<UploadedModelPanel/);
  assert.match(sourceGroup, /onUpload=\{onUploadModel\}/);
  assert.match(sourceGroup, /onRevalidate=\{onRevalidateModel\}/);
  assert.match(sourceGroup, /onDelete=\{onDeleteUploadedModel\}/);
  assert.match(sourceGroup, /inlineError=\{validation\.uploadedModelInlineError\}/);
  // 自定义参数编辑入口只保留在右侧检查器，画布卡片不再重复提供。
  assert.doesNotMatch(sourceGroup, /onEditParams/);
  assert.match(sourceGroup, /templateDownloadUrl=\{resources\.templateDownloadUrl\}/);
  assert.match(sourceGroup, /guideDownloadUrl=\{resources\.guideDownloadUrl\}/);
  // 上传模型面板只在模型来源为 uploaded 时渲染；官方模型时渲染架构选择器。
  assert.match(sourceGroup, /\{isUploaded \? \(\s*<UploadedModelPanel/);
  // 未选择有效上传模型时开始按钮保持不可用。
  assert.match(pageSource, /uploadedModelStartBlocked/);
  assert.match(pageSource, /modelSource === 'uploaded' &&[\s\S]{0,160}?validation_status !== 'valid'/);
  assert.match(pageSource, /const uploadedModelInlineError = modelSource !== 'uploaded'/);
});

test('uploaded model panel keeps management behind a disclosure instead of a permanent list', () => {
  const panelSource = readFileSync(new URL('./UploadedModelPanel.jsx', import.meta.url), 'utf8');
  assert.match(panelSource, /experiment-uploaded-card/);
  assert.match(panelSource, /data-uploaded-model-upload="true"/);
  assert.doesNotMatch(panelSource, /data-uploaded-model-params/);
  assert.match(panelSource, /useState\(false\)/);
  assert.match(panelSource, /experiment-uploaded-manage/);
  assert.match(panelSource, /aria-expanded=\{manageOpen\}/);
  assert.match(panelSource, /data-uploaded-model-manage="true"/);
  // 展开区只剩「重新校验 / 上传模型 / 删除」与校验详情；文件格式要求与校验状态行已删除。
  assert.match(panelSource, /onRevalidate\(selected\.id\)/);
  assert.match(panelSource, /onDelete\(selected\.id\)/);
  assert.match(panelSource, /labels\.replace/);
  assert.doesNotMatch(panelSource, /formatItems|experiment-uploaded-format|experiment-uploaded-foot|validationLabel/);
});

test('official model architecture selector only renders for the official source', () => {
  const sourceGroup = configSource.slice(
    configSource.indexOf('data-config-group="model"'),
    configSource.indexOf('data-config-group="expert"')
  );
  assert.match(sourceGroup, /<ModelArchitectureSelector/);
  assert.match(sourceGroup, /onSelect=\{onArchitectureSelect\}/);
  assert.match(sourceGroup, /expanded=\{architecturePickerOpen\}/);
  assert.match(sourceGroup, /onToggleSphere/);
  assert.match(sourceGroup, /\{isUploaded \? \([\s\S]*?<UploadedModelPanel[\s\S]*?\) : \([\s\S]*?<ModelArchitectureSelector/);
  assert.match(pageSource, /getModelTrainingControlVisibility\(modelSource\)/);
});

test('input and prediction group is a compact payload bar', () => {
  const payloadBlock = configSource.slice(
    configSource.indexOf('data-config-group="payload"'),
    configSource.indexOf('data-config-group="training"')
  );
  assert.match(payloadBlock, /experiment-payload-bar/);
  assert.match(payloadBlock, /data-parameter="target_variable"/);
  assert.match(payloadBlock, /onTargetVariableChange\(event\.target\.value\)/);
  assert.doesNotMatch(payloadBlock, /data-payload-base="true"/);
  assert.match(payloadBlock, /channelOrder\.map\(\(channel\) =>/);
  assert.match(payloadBlock, /className="experiment-channel-chip"/);
  assert.match(payloadBlock, /onChannelToggle\(channel\)/);
  assert.match(payloadBlock, /copy\.payloadCountLabel\(\(inputVariables \|\| selectedChannels\)\.length, channelOrder\.length\)/);
  // 不再把 O₃ / 驱动变量 / 计数分别扩成大块卡片。
  assert.doesNotMatch(payloadBlock, /experiment-payload-base-title/);
  assert.doesNotMatch(payloadBlock, /experiment-payload-drivers/);
  // 信息减法：整块「输入 → 模型 → 输出」序列示意图已删除；
  // 窗口、步长与模型名仍在真实控件里呈现。
  assert.doesNotMatch(configSource, /experiment-flow|data-sequence/);
  assert.match(configSource, /copy\.windowLabel/);
  assert.match(configSource, /copy\.horizonLabel/);
  assert.match(configSource, /<ModelArchitectureSelector/);
});

test('training parameters render as a readable matrix', () => {
  const trainingBlock = configSource.slice(
    configSource.indexOf('data-config-group="training"'),
    configSource.indexOf("activeTab === 'customParams'")
  );
  ['windowValue', 'horizon', 'epochs', 'batchSize', 'learningRate'].forEach((key) => {
    assert.match(configSource, new RegExp(`key: '${key}'`), `${key} should be a matrix field`);
  });
  assert.match(trainingBlock, /parameterFields\.filter[\s\S]*\.map/);
  assert.match(configSource, /experiment-param-grid/);
  assert.match(trainingBlock, /experiment-param-cell/);
  assert.match(trainingBlock, /onFoldChange\(field\.key/);
  const matrix = configSource.slice(
    configSource.indexOf('const parameterFields = ['),
    configSource.indexOf('const expertTabLabels')
  );
  // 矩阵字段定义里仍带 code，但标签旁不再渲染 WINDOW / BATCH 这类英文小码。
  assert.match(configSource, /code: copy\.codeWindow/);
  assert.match(configSource, /code: copy\.codeBatch/);
  assert.match(configSource, /code: copy\.codeLr/);
  const paramLabelStart = trainingBlock.indexOf('experiment-param-label');
  const paramLabelBlock = trainingBlock.slice(paramLabelStart, trainingBlock.indexOf('</span>', paramLabelStart));
  assert.doesNotMatch(paramLabelBlock, /<code>/);
  assert.doesNotMatch(paramLabelBlock, /field\.key/);
  // 参数格只留「标签 + 数值」：下面的量纲小字（时间步 / 预测步 / epochs / 样本 / 学习率）已按用户要求删除。
  assert.doesNotMatch(trainingBlock, /field\.unit/);
  assert.doesNotMatch(trainingBlock, /<small>/);
  assert.doesNotMatch(configSource, /unitSteps|unitPredictSteps|unitEpochs|unitSamples|unitLearningRate/);
});

test('expert parameters use side tabs whose panels render the matching real fields', () => {
  // 超参数页签：先输入与预测 / 训练参数，再按模型来源追加专家字段页签。
  assert.match(configSource, /const CONFIG_EXPERT_TABS = \['payload', 'training'\]/);
  assert.match(configSource, /const UPLOADED_EXPERT_TABS = \[\.\.\.CONFIG_EXPERT_TABS, 'customParams', 'strategy', 'transfer', 'tags'\]/);
  assert.match(configSource, /const OFFICIAL_EXPERT_TABS = \[\.\.\.CONFIG_EXPERT_TABS, 'structure', 'strategy', 'transfer', 'tags'\]/);
  assert.match(configSource, /role="tablist"/);
  assert.match(configSource, /role="tabpanel"/);
  assert.match(configSource, /aria-selected=\{activeTab === tab\}/);
  assert.match(configSource, /data-expert-panel=\{activeTab\}/);
  assert.match(configSource, /data-expert-tab=\{tab\}/);
  // 每个页签面板渲染真实字段而不是只改标题。
  assert.match(configSource, /activeTab === 'payload' \? \(/);
  assert.match(configSource, /activeTab === 'training' \? \(/);
  assert.match(configSource, /activeTab === 'customParams' \? \(/);
  assert.match(configSource, /activeTab === 'structure' \? \(/);
  assert.match(configSource, /activeTab === 'strategy' \? \(/);
  assert.match(configSource, /activeTab === 'transfer' \? \(/);
  assert.match(configSource, /activeTab === 'tags' \? \(/);
  assert.match(configSource, /onFoldChange\('seed'/);
  assert.match(configSource, /onFoldChange\('earlyStoppingPatience'/);
  assert.match(configSource, /onLayersChange/);
  assert.match(configSource, /onDimChange/);
  assert.match(configSource, /onStructureParamChange\(normalizedArchitecture, field/);
  assert.match(configSource, /onTransferSourceTaskChange/);
  assert.match(configSource, /onSelectTrainingWeight/);
  assert.match(configSource, /onUploadWeight/);
  assert.match(configSource, /onDeleteWeight/);
  assert.match(configSource, /copy\.freezeMode/);
  assert.match(configSource, /copy\.finetuneLearningRate/);
  assert.match(configSource, /<TagPicker/);
  // 不适用的参数不做成空页签。
  assert.doesNotMatch(configSource, /expertNeedsOfficial/);
  assert.doesNotMatch(configSource, /expertNeedsUploaded/);
});

test('custom model params stay dynamic and editing them is a first-class action', () => {
  assert.match(configSource, /<DynamicModelParamsForm/);
  assert.match(configSource, /schema=\{selectedUploadedParamSchema\}/);
  assert.match(configSource, /onChange=\{onCustomModelParamChange\}/);
  assert.match(inspectorSource, /copy\.editCustomParams/);
  assert.doesNotMatch(configSource, /onEditParams/);
  assert.match(inspectorSource, /onEditCustomParams/);
  assert.match(pageSource, /onEditCustomParams=\{\(\) => setExpertTab\('customParams'\)\}/);
});

test('the run bar owns the only primary call to action for starting a run', () => {
  const primaryButtons = [...configSource.matchAll(/experiment-center-button-primary/g)];
  assert.equal(primaryButtons.length, 0, '配置画布不应出现主按钮样式');
  assert.equal((configSource.match(/onClick=\{onStart\}/g) || []).length, 0);
  assert.equal((runBarSource.match(/onClick=\{onStart\}/g) || []).length, 1);
  assert.match(runBarSource, /data-run-bar-start="true"/);
  assert.match(runBarSource, /disabled=\{startDisabled\}/);
  assert.match(pageSource, /onStart=\{handleStartTraining\}/);
});

test('configuration canvas reuses the existing training controls instead of reimplementing them', () => {
  assert.match(configSource, /<UploadedModelPanel/);
  assert.match(configSource, /<DynamicModelParamsForm/);
  assert.match(configSource, /<ModelArchitectureSelector/);
  assert.match(configSource, /<TagPicker/);
  assert.match(configSource, /from '\.\/trainingParamSanitizers'/);
  // 模型来源胶囊留在「模型」分区里，ModelSourceSelector 组件本身不再单独渲染。
  assert.doesNotMatch(configSource, /<ModelSourceSelector/);
  assert.match(configSource, /experiment-source-option/);
});

test('configuration canvas submits through the parent callback only', () => {
  assert.doesNotMatch(configCode, /startTrainingTask/);
  assert.doesNotMatch(configCode, /fetchLogs/);
  assert.doesNotMatch(configCode, /setInterval/);
  assert.doesNotMatch(configCode, /new WebSocket/);
  assert.match(pageSource, /onStart: handleStartTraining/);
});

test('configuration canvas keeps dataset, model source, channels, window/horizon and training folds', () => {
  ['trainingDataset', 'modelSource', 'selectedChannels', 'windowValue', 'horizon', 'epochs', 'batchSize', 'learningRate']
    .forEach((field) => assert.match(configSource, new RegExp(field), `${field} should be part of the form values`));
  assert.match(configSource, /TRAINING_DATASET_OPENMARS_MCD/);
  assert.match(configSource, /TRAINING_DATASET_MCD_OVERVIEW/);
  assert.match(configSource, /onTrainingDatasetChange/);
  assert.match(configSource, /onChannelToggle/);
  assert.match(configSource, /onFoldChange\(field\.key/);
});

test('structure-defining controls stay locked for a synchronized transfer source', () => {
  assert.match(configSource, /const transferStructureLocked\s*=|transferStructureLocked,/);
  assert.match(configSource, /data-model-source-option=\{option\.value\}[\s\S]{0,400}?disabled=\{!resources\.user \|\| transferStructureDisabled\}/);
  assert.match(configSource, /<UploadedModelPanel[\s\S]*?selectionDisabled=\{transferStructureLocked\}/);
  assert.match(configSource, /<UploadedModelPanel[\s\S]*?busy=\{isProcessing \|\| transferStructureLocked\}/);
  assert.match(configSource, /<DynamicModelParamsForm[\s\S]*?disabled=\{transferStructureLocked\}/);
  assert.match(configSource, /data-channel=\{channel\}[\s\S]{0,200}?disabled=\{transferStructureLocked\}/);
  assert.match(configSource, /<ModelArchitectureSelector[\s\S]{0,900}?disabled=\{transferStructureLocked\}/);
  assert.match(configSource, /checked=\{useSphere\}[\s\S]{0,160}?disabled=\{transferStructureLocked\}/);
  // 结构参数与窗口/步长输入的锁定量必须落在真正的输入控件上。
  const trainingBlock = configSource.slice(
    configSource.indexOf('data-config-group="training"'),
    configSource.indexOf("activeTab === 'customParams'")
  );
  assert.match(trainingBlock, /data-parameter=\{field\.key\}[\s\S]{0,900}?disabled=\{field\.locked \? transferStructureLocked : undefined\}/);
  assert.match(configSource, /value=\{values\.stlstmLayers\}[\s\S]{0,200}?disabled=\{transferStructureLocked\}/);
  assert.match(configSource, /value=\{dim\}[\s\S]{0,300}?disabled=\{transferStructureLocked\}/);
  assert.match(configSource, /type=\{field\.type === 'integerList' \? 'text' : 'number'\}[\s\S]{0,700}?disabled=\{transferStructureLocked\}/);
});

test('fine-tuning controls stay editable while the source structure is locked', () => {
  const matrix = configSource.slice(
    configSource.indexOf('const parameterFields = ['),
    configSource.indexOf('const expertTabLabels')
  );
  for (const key of ['epochs', 'batchSize', 'learningRate']) {
    assert.match(matrix, new RegExp(`key: '${key}'`), `${key} should be a matrix field`);
  }
  // 参数矩阵本身不含锁定量，避免把轮次/批大小/学习率一起锁住。
  assert.doesNotMatch(matrix, /transferStructureLocked/);
  assert.match(matrix, /locked: true/);
  assert.doesNotMatch(configSource, /copy\.freezeMode[\s\S]{0,400}?transferStructureLocked/);
  assert.doesNotMatch(configSource, /copy\.finetuneLearningRate[\s\S]{0,500}?transferStructureLocked/);
  assert.doesNotMatch(configSource, /copy\.randomSeed[\s\S]{0,400}?transferStructureLocked/);
});

test('copy configuration warnings render inside the config canvas', () => {
  assert.match(configSource, /copyConfigWarnings/);
  assert.match(configSource, /experiment-config-warning/);
  assert.match(configSource, /copy\.copyConfigWarningLabels\[code\]/);
});

test('transfer lock coverage moved with the form into the config canvas', () => {
  assert.match(pageSource, /const transferStructureLocked\s*=/);
  assert.match(lockTestSource, /pageSource/);
});
