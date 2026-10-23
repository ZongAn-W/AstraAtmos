import { useEffect, useState } from 'react';
import C from '../../constants/colors';
import UploadedModelPanel from './UploadedModelPanel';
import DynamicModelParamsForm from './DynamicModelParamsForm';
import ModelArchitectureSelector from './ModelArchitectureSelector';
import { TagPicker } from '../../components/TrainingTags/TagControls';
import {
  TRAINING_DATASET_EARTH_MERRA2_3HOURLY_V1,
  TRAINING_DATASET_EARTH_MERRA2_3HOURLY_5DEG_V1,
  TRAINING_DATASET_MCD_OVERVIEW,
  TRAINING_DATASET_OPENMARS_MCD,
  getModelStructureParamLabel,
  isRecurrentArchitecture,
} from './trainingParamSanitizers';
import { EARTH_MODEL_ARCHITECTURE, EARTH_PARAM_BOUNDS, getEarthTrainingProfile, isEarthTrainingDataset, describeEarthTaskSplits } from './earthTrainingConfig';
import { getExperimentArchitectureLabel } from './experimentCenterModel';
import { getUploadedModelValidationStatus } from './uploadedModelValidation';
import './experimentCenter.css';

const OPEN_INTERVAL_FLOAT_FIELDS = new Set(['initial_history_weight', 'initial_translation_weight']);
const BASE_INPUT_CHANNEL = 'O3';

/** Earth 的模型来源选项；顺序与火星一致（官方/上传由页面控制器决定标签）。 */
const EARTH_MODEL_SOURCE_OPTIONS = [
  { value: 'official', label: (copy) => copy.modelSourceOfficial },
  { value: 'uploaded', label: (copy) => copy.modelSourceUploaded },
];

/**
 * 专家参数页签。
 *
 * 页签集合按模型来源决定，避免出现“去另一个地方”的空页签：
 * 上传模型 → 自定义参数 / 训练策略 / 迁移学习 / 实验标签；
 * 官方模型 → 模型结构 / 训练策略 / 迁移学习 / 实验标签。
 * 权重上传与冻结策略都在「迁移学习」里，不再单独占一个页签。
 */
const CONFIG_EXPERT_TABS = ['payload', 'training'];
const UPLOADED_EXPERT_TABS = [...CONFIG_EXPERT_TABS, 'customParams', 'strategy', 'transfer', 'tags'];
const OFFICIAL_EXPERT_TABS = [...CONFIG_EXPERT_TABS, 'structure', 'strategy', 'transfer', 'tags'];

/**
 * 配置实验工作区（中间配置画布）。
 *
 * 只接收数据和回调：不调用 startTrainingTask，不持有轮询；
 * 「开始实验」由底部运行条通过父组件的 onStart 提交，避免出现两套提交状态。
 * 表单状态全部来自页面控制器，因此切换页签、收起目录或筛选目录都不会丢值。
 */
export default function ExperimentConfigWorkspace({
  values,
  resources,
  validation,
  actions,
  ui,
  copy,
  t,
  isZh,
  channelOrder,
  channelMap,
  modelNameLabel,
  structureLabelLanguage,
  availableTransferFreezeModes,
  transferFreezeModeLabels,
  presetStatusText,
  controlVisibility,
  pendingSubmission,
  copyConfigWarnings = [],
  expertTab,
  onExpertTabChange,
  readiness,
}) {
  const {
    customModelName,
    newTaskTagIds,
    trainingDataset,
    modelSource,
    selectedUploadedModelId,
    selectedChannels,
    modelArchitecture,
    useSphere,
    epochs,
    batchSize,
    learningRate,
    trainRatio,
    validationRatio,
    testRatio,
    windowValue,
    horizon,
    earlyStoppingPatience,
    seed,
    transferEnabled,
    transferSourceType,
    transferSourceTaskId,
    transferFreezeMode,
    finetuneLearningRate,
  } = values;
  const {
    uploadedModels,
    completedTransferTasks,
    trainingWeights,
    selectedUploadedModel,
    selectedUploadedModelLabel,
    selectedUploadedParamSchema,
    selectedTrainingWeight,
    selectedTrainingWeightId,
    tagState,
    isProcessing,
    earthDatasetAvailability,
  } = resources;
  const {
    modelNameError,
    visibleCustomModelParamErrors,
    transferStartBlocked,
    transferStructureLocked,
    transferStructureDisabled,
    customParamCount,
  } = validation;
  const {
    onModelNameChange,
    onTrainingDatasetChange,
    onModelSourceChange,
    onSelectUploadedModel,
    onUploadModel,
    onRevalidateModel,
    onDeleteUploadedModel,
    onRenameUploadedModel,
    onDownloadUploadedModel,
    onCustomModelParamChange,
    onChannelToggle,
    onArchitectureSelect,
    onToggleSphere,
    onToggleArchitecturePicker,
    onStructureParamChange,
    onLayersChange,
    onDimChange,
    onTransferEnabledChange,
    onTransferSourceTypeChange,
    onTransferSourceTaskChange,
    onSelectTrainingWeight,
    onUploadWeight,
    onDeleteWeight,
    onFoldChange,
    onSplitRatioChange,
    onTagIdsChange,
    onCreateTag,
  } = actions;
  const { sectionTitleStyle, fieldLabelStyle, fieldHintStyle, inputStyle } = ui;

  const normalizedArchitecture = values.normalizedModelArchitecture;
  const activeStructureConfig = values.activeStructureConfig || [];
  const activeStructureParams = values.activeStructureParams || {};
  const structureSummary = values.structureSummary;
  const officialModelControls = controlVisibility.officialModelControls;
  const isUploaded = modelSource === 'uploaded';
  const uploadedModelLabels = {
    title: copy.groupUploadedModel,
    upload: copy.uploadModel,
    uploadAgain: copy.uploadedModelReplace,
    downloadGuide: copy.downloadModelGuide,
    downloadTemplate: copy.downloadModelTemplate,
    downloadSharedGateTemplate: copy.downloadSharedGateTemplate,
    uploading: copy.uploadingModel,
    revalidate: copy.revalidateModel,
    replace: copy.uploadedModelReplace,
    manage: copy.uploadedModelManage,
    delete: copy.deleteUploadedModel,
    download: copy.downloadUploadedModel,
    rename: copy.renameUploadedModel,
    name: copy.uploadedModelName,
    saveName: copy.saveUploadedModelName,
    cancelRename: copy.cancelUploadedModelRename,
    nameRequired: copy.uploadedModelNameRequired,
    valid: copy.uploadedModelValid,
    invalid: copy.uploadedModelInvalid,
    pending: copy.uploadedModelPending,
    timeout: copy.uploadedModelTimeout,
    ready: copy.uploadedModelReady,
    unnamed: copy.uploadedModelUnnamed,
    noFilename: copy.uploadedModelNoFilename,
    missing: copy.uploadedModelEmptyTitle,
    hint: copy.uploadedModelEmptyHint,
    summaryLabel: copy.uploadedModelSummaryLabel,
    summaryParamCount: copy.uploadedModelParamCount,
    versionLabel: copy.inspectorModelVersion,
    officialHint: copy.modelSourceOfficialHint,
  };
  // Earth uses dataset-specific upload verdicts and keeps transfer learning disabled.
  const isEarth = isEarthTrainingDataset(trainingDataset);
  const isEarthThreeHourly = isEarth;
  const earthProfile = getEarthTrainingProfile(trainingDataset);
  const architecturePickerOpen = resources.architecturePickerOpen;
  const isRecurrentModel = isRecurrentArchitecture(normalizedArchitecture);

  const expertTabs = isEarth
    ? ['payload', 'training', ...(isUploaded ? ['customParams'] : []), 'strategy', 'tags']
    : (isUploaded ? UPLOADED_EXPERT_TABS : OFFICIAL_EXPERT_TABS);
  // 载荷与训练参数是每次配置都要过一遍的，作为超参数模块的头两个页签并默认打开第一页。
  const defaultExpertTab = 'payload';
  const [localTab, setLocalTab] = useState(defaultExpertTab);
  const activeTab = expertTabs.includes(expertTab) ? expertTab : (expertTabs.includes(localTab) ? localTab : defaultExpertTab);
  const setActiveTab = (tab) => {
    if (!expertTabs.includes(tab)) return;
    setLocalTab(tab);
    onExpertTabChange?.(tab);
  };

  // 切换模型来源时把页签落到该来源最常用的那一页。
  useEffect(() => {
    setLocalTab(defaultExpertTab);
  }, [defaultExpertTab]);

  // Earth 的参数范围与火星不同：用 Earth 自己的上下限，不悄悄夹到火星范围。
  const earthBounds = isEarth
    ? {
        epochs: { min: EARTH_PARAM_BOUNDS.epochs.min, max: EARTH_PARAM_BOUNDS.epochs.max },
        batchSize: { min: EARTH_PARAM_BOUNDS.batch_size.min, max: EARTH_PARAM_BOUNDS.batch_size.max },
        learningRate: { min: '0.000001', max: '1' },
      }
    : {};
  const boundFor = (key, fallback) => earthBounds[key] || fallback;

  const taskSplitPreview = describeEarthTaskSplits(earthDatasetAvailability?.time,
    { trainRatio, validationRatio, testRatio }, windowValue, horizon);
  const uploadedModel = selectedUploadedModel;
  const uploadedValidationStatus = getUploadedModelValidationStatus(uploadedModel);
  const uploadedStatusLabel = uploadedValidationStatus === 'valid'
    ? copy.uploadedModelValid
    : uploadedValidationStatus === 'pending'
      ? copy.uploadedModelPending
      : uploadedValidationStatus === 'timeout'
        ? copy.uploadedModelTimeout : copy.uploadedModelInvalid;
  const modelBlockState = isUploaded
    ? (uploadedModel
        ? (uploadedValidationStatus === 'valid' ? 'valid' : 'invalid')
        : 'empty')
    : 'official';

  const parameterFields = [
    {
      key: 'windowValue',
      label: isEarth ? `${copy.windowLabel} (${isZh ? '三小时步' : '3-hour steps'})` : copy.windowLabel,
      code: copy.codeWindow,
      step: '1',
      min: '1',
      max: isEarth ? '240' : '30',
      locked: true,
    },
    {
      key: 'horizon',
      label: isEarth ? `${copy.horizonLabel} (${isZh ? '三小时步' : '3-hour steps'})` : copy.horizonLabel,
      code: copy.codeHorizon,
      step: '1',
      min: '1',
      max: isEarth ? '240' : '30',
      locked: true,
    },
    {
      key: 'epochs',
      label: copy.epochsLabel,
      code: copy.codeEpochs,
      step: '1',
      min: boundFor('epochs', { min: '1' }).min,
      max: boundFor('epochs', {}).max,
    },
    {
      key: 'batchSize',
      label: copy.batchSizeLabel,
      code: copy.codeBatch,
      step: '1',
      min: boundFor('batchSize', { min: '1' }).min,
      max: boundFor('batchSize', {}).max,
    },
    {
      key: 'learningRate',
      label: copy.learningRateLabel,
      code: copy.codeLr,
      step: '0.0001',
      min: boundFor('learningRate', { min: '0.000001' }).min,
      max: boundFor('learningRate', {}).max,
    },
  ];
  const expertTabLabels = {
    payload: copy.sectionPayload,
    training: copy.sectionTraining,
    customParams: copy.expertTabCustomParams,
    structure: copy.expertStructureTab,
    strategy: copy.expertTabStrategy,
    transfer: copy.expertTabTransfer,
    tags: copy.expertTabTags,
  };
  // 训练数据集直接列成可选项，不用下拉：展开菜单反而多一次点击。
  // Earth 选项即使当前不可用也保留可见，原因显示在下方 Earth 面板里。
  const datasetOptions = [
    { value: TRAINING_DATASET_OPENMARS_MCD, label: copy.datasetOpenMarsMcd },
    { value: TRAINING_DATASET_MCD_OVERVIEW, label: copy.datasetMcdOverview },
    { value: TRAINING_DATASET_EARTH_MERRA2_3HOURLY_V1, label: copy.datasetEarthMerra23HourlyV1 },
    { value: TRAINING_DATASET_EARTH_MERRA2_3HOURLY_5DEG_V1, label: copy.datasetEarthMerra25Degree },
  ];
  return (
    <div className="experiment-center-stack" data-config-canvas="true">
      {copyConfigWarnings.length > 0 ? (
        <div className="experiment-config-warning" role="status">
          <strong>{copy.copyConfigWarningTitle}</strong>
          {copyConfigWarnings.map((code) => (
            <span key={code}>{copy.copyConfigWarningLabels[code] || copy.copyConfigWarningFallback}</span>
          ))}
        </div>
      ) : null}

      {/* 01 模型名称：可编辑的实验名称与字段级错误；
          任务说明与就绪胶囊已在信息减法中删除（就绪状态由底部运行条与右侧检查器播报）。 */}
      <header className="experiment-canvas-head" data-config-group="name">
        <div className="experiment-section-kicker">
          <span className="experiment-section-index">01</span>
          <span className="experiment-section-title" style={sectionTitleStyle}>{copy.sectionName}</span>
        </div>
        <div className="experiment-canvas-head-main">
          <label className="sr-only" htmlFor="experiment-name-input">{copy.canvasNameLabel}</label>
          <input
            id="experiment-name-input"
            type="text"
            className="experiment-canvas-name"
            placeholder={t('modelTraining.modelNamingPlaceholder')}
            value={customModelName}
            onChange={onModelNameChange}
            aria-invalid={Boolean(modelNameError)}
            aria-errormessage={modelNameError ? 'experiment-name-error' : undefined}
          />
          {modelNameError ? (
            <p className="experiment-canvas-name-error" id="experiment-name-error" role="alert">{modelNameError}</p>
          ) : null}
        </div>
      </header>

      {/* 02 数据集 */}
      <section className="experiment-canvas-section" data-config-group="dataset">
        <div className="experiment-section-kicker">
          <span className="experiment-section-index">02</span>
          <span className="experiment-section-title" style={sectionTitleStyle}>{copy.sectionDataset}</span>
        </div>

        <div className="experiment-choice-block" data-active="true">
          <div className="experiment-dataset-list" role="radiogroup" aria-label={copy.trainingDataset} data-training-dataset-list="true">
            {datasetOptions.map((option) => {
              const active = trainingDataset === option.value;
              return (
                <button
                  key={option.value}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  className="experiment-dataset-option"
                  data-training-dataset-option={option.value}
                  onClick={() => onTrainingDatasetChange(option.value)}
                >
                  <strong>{option.label}</strong>
                </button>
              );
            })}
          </div>
          {isEarth ? (
            <div className="experiment-earth-contract" data-earth-training-contract="true">
              <span>{earthProfile.frequencyHours === 3 ? 'UTC · 3-hourly' : 'UTC · daily'}</span>
              <span>{`${earthProfile.gridShape[0]}×${earthProfile.gridShape[1]}`}</span>
              <span>{`${windowValue} → ${horizon} steps`}</span>
              <span>TO3 · DU</span>
              {earthDatasetAvailability?.selectable === false ? (
                <strong data-earth-unavailable="true">{earthDatasetAvailability.reason || copy.earthUnavailableFallback}</strong>
              ) : null}
            </div>
          ) : null}
        </div>
      </section>

      {/* 03 模型 */}
      <section className="experiment-canvas-section" data-config-group="model">
        <div className="experiment-section-kicker">
          <span className="experiment-section-index">03</span>
          <span className="experiment-section-title" style={sectionTitleStyle}>{copy.sectionModel}</span>
        </div>

        <div className="experiment-choice-block" data-active="true" data-model-block={isEarth ? 'earth' : modelBlockState}>
          {isEarth ? (
            <div className="experiment-earth-model" data-earth-model-block="true">
              <div className="experiment-source-toggle" role="group" aria-label={copy.modelSource} data-model-source-toggle="true">
                {EARTH_MODEL_SOURCE_OPTIONS.filter((option) => earthProfile.modelSources.includes(option.value)).map((option) => {
                  const active = modelSource === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      className="experiment-source-option"
                      aria-pressed={active}
                      data-model-source-option={option.value}
                      data-earth-model-source-option={option.value}
                      disabled={isProcessing}
                      onClick={() => onModelSourceChange(option.value)}
                    >
                      <strong>{option.label(copy)}</strong>
                    </button>
                  );
                })}
              </div>
              {isUploaded ? (
                <UploadedModelPanel
                  models={uploadedModels}
                  selectedId={selectedUploadedModelId}
                  onSelect={onSelectUploadedModel}
                  onUpload={onUploadModel}
                  onRevalidate={onRevalidateModel}
                  onDelete={onDeleteUploadedModel}
                  onRename={onRenameUploadedModel}
                  onDownload={onDownloadUploadedModel}
                  uploading={resources.uploadingModel}
                  busy={isProcessing}
                  selectionDisabled={false}
                  guideDownloadUrl={resources.guideDownloadUrl}
                  templateDownloadUrl={resources.templateDownloadUrl}
                  sharedGateTemplateDownloadUrl={resources.sharedGateTemplateDownloadUrl}
                  inlineError={resources.earthUploadedInlineError || ''}
                  statusLabel={resources.earthUploadedStatusLabel || ''}
                  statusTone={resources.earthUploadedStatusTone || 'ok'}
                  labels={uploadedModelLabels}
                />
              ) : (
                <>
                  <span className="experiment-payload-lock" data-earth-official-lock="true">
                    <span>{copy.modelSourceOfficial}</span>
                    <b>{getExperimentArchitectureLabel(EARTH_MODEL_ARCHITECTURE)}</b>
                  </span>
                  <p className="experiment-expert-note">{isEarthThreeHourly ? copy.earthThreeHourlyModelFixedNote(earthProfile.gridShape) : copy.earthModelFixedNote}</p>
                </>
              )}
              {isUploaded && resources.earthUploadedNotice ? (
                <p
                  className="experiment-expert-note"
                  data-earth-uploaded-notice="true"
                  data-tone={resources.earthUploadedNoticeTone || 'ok'}
                >
                  {resources.earthUploadedNotice}
                </p>
              ) : null}
            </div>
          ) : (
          <>
          <div className="experiment-source-toggle" role="group" aria-label={copy.modelSource} data-model-source-toggle="true">
              {[
                { value: 'uploaded', label: copy.modelSourceUploaded },
                { value: 'official', label: copy.modelSourceOfficial },
              ].map((option) => {
                const active = modelSource === option.value;
                return (
                  <button
                    key={option.value}
                    type="button"
                    className="experiment-source-option"
                    aria-pressed={active}
                    data-model-source-option={option.value}
                    disabled={!resources.user || transferStructureDisabled}
                    onClick={() => onModelSourceChange(option.value)}
                  >
                    <strong>{option.label}</strong>
                  </button>
                );
              })}
            </div>

            {isUploaded ? (
              <UploadedModelPanel
                models={uploadedModels}
                selectedId={selectedUploadedModelId}
                onSelect={onSelectUploadedModel}
                onUpload={onUploadModel}
                onRevalidate={onRevalidateModel}
                onDelete={onDeleteUploadedModel}
                onRename={onRenameUploadedModel}
                onDownload={onDownloadUploadedModel}
                uploading={resources.uploadingModel}
                busy={isProcessing || transferStructureLocked}
                selectionDisabled={transferStructureLocked}
                guideDownloadUrl={resources.guideDownloadUrl}
                templateDownloadUrl={resources.templateDownloadUrl}
                sharedGateTemplateDownloadUrl={resources.sharedGateTemplateDownloadUrl}
                inlineError={validation.uploadedModelInlineError}
                statusLabel={uploadedStatusLabel}
                statusTone={uploadedValidationStatus === 'valid' ? 'ok' : 'warn'}
                labels={uploadedModelLabels}
                sectionTitleStyle={sectionTitleStyle}
                fieldHintStyle={fieldHintStyle}
              />
            ) : (
              <div className="experiment-official-block">
                <ModelArchitectureSelector
                  value={modelArchitecture}
                  onSelect={onArchitectureSelect}
                  copy={copy}
                  isZh={isZh}
                  disabled={transferStructureLocked}
                  sectionTitleStyle={sectionTitleStyle}
                  fieldHintStyle={fieldHintStyle}
                  inputStyle={inputStyle}
                  expanded={architecturePickerOpen}
                  onToggleExpanded={onToggleArchitecturePicker}
                />
                <label className="experiment-toggle-line">
                  <input type="checkbox" checked={useSphere} onChange={onToggleSphere} disabled={transferStructureLocked} />
                  {copy.sphereToggle}
                </label>
              </div>
            )}
          </>
          )}
        </div>
      </section>

      {/* 04 超参数：输入与预测、训练参数、专家字段都在这里，用侧边页签切换 */}
      <section className="experiment-canvas-section" data-config-group="expert">
        <div className="experiment-section-kicker">
          <span className="experiment-section-index">04</span>
          <span className="experiment-section-title" style={sectionTitleStyle}>{copy.sectionExpert}</span>
        </div>

        <div className="experiment-expert" data-expert-open="true">
          <div className="experiment-expert-tabs" role="tablist" aria-label={copy.sectionExpert}>
            {expertTabs.map((tab) => (
              <button
                key={tab}
                type="button"
                role="tab"
                id={`experiment-expert-tab-${tab}`}
                aria-selected={activeTab === tab}
                aria-controls={`experiment-expert-panel-${tab}`}
                className="experiment-expert-tab"
                data-expert-tab={tab}
                onClick={() => setActiveTab(tab)}
              >
                {expertTabLabels[tab]}
                {tab === 'customParams' && customParamCount > 0 ? (
                  <span className="experiment-expert-tab-count">{customParamCount}</span>
                ) : null}
                {tab === 'transfer' && transferEnabled ? (
                  <span className="experiment-expert-tab-flag" aria-hidden="true">●</span>
                ) : null}
              </button>
            ))}
          </div>

          <div
            className="experiment-expert-content"
            role="tabpanel"
            id={`experiment-expert-panel-${activeTab}`}
            aria-labelledby={`experiment-expert-tab-${activeTab}`}
            data-expert-panel={activeTab}
          >
            {/* 输入与预测：载荷块与其它页签同一套字段块 */}
            {activeTab === 'payload' ? (
              <div data-config-group="payload">
                <h4 className="experiment-expert-heading">{copy.sectionPayload}</h4>
                {isEarth ? <>
                  <div className="experiment-param-grid">
                    {parameterFields.filter((field) => ['windowValue', 'horizon'].includes(field.key)).map((field) => (
                      <label className="experiment-param-cell" key={field.key} data-parameter={field.key}>
                        <span className="experiment-param-label">{field.label}</span>
                        <input type="number" value={values[field.key]} min={field.min} max={field.max} step="1"
                          aria-label={field.label} onChange={(event) => onFoldChange(field.key, event.target.value)} />
                      </label>
                    ))}
                  </div>
                  <p className="experiment-expert-note" data-earth-window-duration="true">
                    {isZh
                      ? `输入 ${Number(windowValue) * 3 / 24} 天 → 输出 ${Number(horizon) * 3 / 24} 天；每步 3 小时，改变窗口后需重新训练。`
                      : `Input ${Number(windowValue) * 3 / 24} days → output ${Number(horizon) * 3 / 24} days; each step is 3 hours. Changing windows requires training a new model.`}
                  </p>
                </> : null}
                <div className="experiment-payload-bar" role="group" aria-label={t('modelTraining.inputChannels')}>
                  <span className="experiment-payload-lock" data-payload-base="true">
                    <span>{isEarth ? copy.earthBaseInput : copy.inspectorBaseInput}</span>
                    <b>{isEarth ? 'TO3' : 'O₃'}</b>
                  </span>
                  {channelOrder.map((channel) => {
                    const active = selectedChannels.includes(channel);
                    return (
                      <button
                        key={channel}
                        type="button"
                        className="experiment-channel-chip"
                        aria-pressed={active}
                        data-channel={channel}
                        disabled={transferStructureLocked}
                        onClick={() => onChannelToggle(channel)}
                      >
                        <span>{channelMap[channel]?.name || channel}</span>
                        <b>{channelMap[channel]?.short || channel}</b>
                      </button>
                    );
                  })}
                  <span className="experiment-payload-count" data-payload-count={selectedChannels.length}>
                    {copy.payloadCountLabel(selectedChannels.length, channelOrder.length)}
                  </span>
                </div>
              </div>
            ) : null}

            {/* 训练参数矩阵 */}
            {activeTab === 'training' ? (
              <div data-config-group="training">
                <h4 className="experiment-expert-heading">{copy.sectionTraining}</h4>
                <div className="experiment-param-grid">
                  {parameterFields.filter((field) => !isEarth || !['windowValue', 'horizon'].includes(field.key)).map((field) => (
                    <label className="experiment-param-cell" key={field.key} data-parameter={field.key}>
                      <span className="experiment-param-label">
                        {field.label}
                      </span>
                      <input
                        type="number"
                        value={values[field.key]}
                        step={field.step}
                        min={field.min}
                        max={field.max}
                        disabled={field.locked ? transferStructureLocked : undefined}
                        aria-label={field.label}
                        onChange={(event) => onFoldChange(field.key, event.target.value)}
                      />
                    </label>
                  ))}
                </div>
              </div>
            ) : null}

            {activeTab === 'customParams' ? (
              isUploaded ? (
                uploadedModel ? (
                  <>
                    <h4 className="experiment-expert-heading">
                      {`${uploadedModel.original_filename || copy.uploadedModelUnnamed} / ${copy.expertTabCustomParams}`}
                    </h4>
                    <DynamicModelParamsForm
                      schema={selectedUploadedParamSchema}
                      values={values.customModelParams}
                      errors={visibleCustomModelParamErrors}
                      onChange={onCustomModelParamChange}
                      disabled={transferStructureLocked}
                      labels={{
                        title: copy.customModelParams,
                        empty: copy.customModelParamsEmpty,
                        rangeHint: copy.paramRangeHint,
                        minHint: copy.paramMinHint,
                        maxHint: copy.paramMaxHint,
                      }}
                      hideTitle
                      sectionTitleStyle={sectionTitleStyle}
                      fieldLabelStyle={fieldLabelStyle}
                      fieldHintStyle={fieldHintStyle}
                      inputStyle={inputStyle}
                    />
                  </>
                ) : (
                  <div className="experiment-expert-empty" role="status">{copy.expertNoUploadedModel}</div>
                )
              ) : null
            ) : null}

            {activeTab === 'structure' ? (
              <>
                <h4 className="experiment-expert-heading">
                  {`${getExperimentArchitectureLabel(modelArchitecture)} / ${copy.expertStructureTab}`}
                </h4>
                <p className="experiment-expert-note">{copy.expertStructureHint}</p>
                <div className="experiment-expert-fields">
                  {isRecurrentModel ? (
                    <>
                      <label className="experiment-expert-field">
                        <span>{t('modelTraining.stlstmLayers')}</span>
                        <input
                          type="number"
                          value={values.stlstmLayers}
                          min="1"
                          max="10"
                          disabled={transferStructureLocked}
                          onChange={onLayersChange}
                        />
                      </label>
                      {values.hiddenDims.map((dim, index) => (
                        <label className="experiment-expert-field" key={`dim-${index}`}>
                          <span>{`${t('modelTraining.layer')} ${index + 1} ${t('modelTraining.layerDim')}`}</span>
                          <input
                            type="number"
                            value={dim}
                            min="1"
                            disabled={transferStructureLocked}
                            onChange={(event) => onDimChange(index, event.target.value)}
                          />
                        </label>
                      ))}
                    </>
                  ) : (
                    activeStructureConfig.map((field) => (
                      <label className="experiment-expert-field" key={field.key}>
                        <span>{getModelStructureParamLabel(field.key, structureLabelLanguage)}</span>
                        <input
                          type={field.type === 'integerList' ? 'text' : 'number'}
                          value={
                            Array.isArray(activeStructureParams[field.key] ?? field.defaultValue)
                              ? (activeStructureParams[field.key] ?? field.defaultValue).join(',')
                              : activeStructureParams[field.key] ?? field.defaultValue
                          }
                          disabled={transferStructureLocked}
                          min={
                            field.type === 'boundedFloat' && OPEN_INTERVAL_FLOAT_FIELDS.has(field.key)
                              ? '0.000001'
                              : ['dropout', 'boundedFloat', 'nonNegativeNumber'].includes(field.type)
                                ? '0'
                                : '1'
                          }
                          max={['dropout', 'boundedFloat'].includes(field.type) ? '0.9' : undefined}
                          step={['dropout', 'boundedFloat', 'nonNegativeNumber'].includes(field.type) ? '0.05' : '1'}
                          onChange={(event) => onStructureParamChange(normalizedArchitecture, field, event.target.value)}
                        />
                      </label>
                    ))
                  )}
                </div>
                <div className="experiment-expert-summary">
                  <span className="experiment-center-hint" data-expert-structure-summary="true">{structureSummary}</span>
                  {!officialModelControls ? null : (
                    <button type="button" className="experiment-expert-action" onClick={onToggleArchitecturePicker}>
                      {architecturePickerOpen ? copy.collapse : copy.expand}
                    </button>
                  )}
                </div>
              </>
            ) : null}

            {activeTab === 'strategy' ? (
              <>
                <h4 className="experiment-expert-heading">{copy.expertTabStrategy}</h4>
                <p className="experiment-expert-note">{copy.expertStrategyHint}</p>
                {isEarth ? (
                  <div className="experiment-earth-splits" data-earth-task-split-preview="true">
                    <p className="experiment-expert-note">{isZh
                      ? '按完整 UTC 三小时时间轴连续划分；比例均须大于 0、合计 100%。窗口完整留在各分区内，归一化只拟合训练区间。'
                      : 'Split the full UTC 3-hour timeline chronologically. All ratios must be positive and total 100%. Windows stay within partitions; normalization fits training only.'}</p>
                    {taskSplitPreview.ranges.map(item => (
                      <div className="experiment-earth-split" key={item.name} data-earth-split={item.name}>
                        <span>{item.name}</span><b>{item.date_start} → {item.date_end}</b>
                        <small>{item.step_count} {isZh ? '时间步' : 'time steps'} · {item.window_count} {isZh ? '窗口' : 'windows'}</small>
                      </div>
                    ))}
                    {!taskSplitPreview.valid ? <p role="alert">{isZh
                      ? '划分无效：请检查比例总和、非空验证集和各分区长度（至少输入窗口 + 输出窗口）；数据目录须提供完整时间轴。'
                      : 'Invalid split: check ratio sum, positive validation and partition length (at least window + horizon); a complete catalog timeline is required.'}</p> : null}
                  </div>
                ) : null}
                <div className="experiment-expert-fields">
                  {[
                    ['trainRatio', trainRatio, isZh ? '训练集比例 (%)' : 'Train ratio (%)'],
                    ['validationRatio', validationRatio, isZh ? '验证集比例 (%)' : 'Validation ratio (%)'],
                    ['testRatio', testRatio, isZh ? '测试集比例 (%)' : 'Test ratio (%)'],
                  ].map(([key, value, label]) => (
                    <label className="experiment-expert-field" key={key} data-earth-task-split={isEarth ? key : undefined}>
                      <span>{label}</span>
                      <input type="number" min={isEarth ? '0.01' : key === 'validationRatio' ? '0' : '1'} max={isEarth ? '99.99' : '98'} step={isEarth ? '0.01' : '1'}
                        value={value === '' ? '' : Number(value) * 100}
                        onChange={(event) => onSplitRatioChange?.(key, event.target.value === '' ? '' : Number(event.target.value) / 100)} />
                    </label>
                  ))}
                  <label className="experiment-expert-field">
                    <span>{copy.randomSeed}</span>
                    <input
                      type="number"
                      value={seed}
                      min="0"
                      max="2147483647"
                      onChange={(event) => onFoldChange('seed', event.target.value)}
                    />
                  </label>
                  <label className="experiment-expert-field">
                    <span>{t('modelTraining.earlyStopPatience')}</span>
                    <input
                      type="number"
                      value={earlyStoppingPatience}
                      min="0"
                      max="200"
                      onChange={(event) => onFoldChange('earlyStoppingPatience', event.target.value)}
                    />
                  </label>
                </div>
                <p className="experiment-expert-note">{t('modelTraining.earlyStopNote')}</p>
              </>
            ) : null}

            {activeTab === 'transfer' ? (
              <>
                <h4 className="experiment-expert-heading">{copy.expertTabTransfer}</h4>
                <p className="experiment-expert-note">{copy.expertTransferHintLabel}</p>
                <label className="experiment-toggle-line">
                  <input
                    type="checkbox"
                    checked={transferEnabled}
                    onChange={(event) => onTransferEnabledChange(event.target.checked)}
                  />
                  {copy.transferEnable}
                </label>

                {transferEnabled ? (
                  <div className="experiment-expert-stack">
                    <div className="experiment-config-choice">
                      {[
                        ['task', copy.transferTaskSource],
                        ['upload', copy.transferUploadSource],
                      ].map(([value, label]) => (
                        <button
                          key={value}
                          type="button"
                          className="experiment-config-chip"
                          aria-pressed={transferSourceType === value}
                          onClick={() => onTransferSourceTypeChange(value)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>

                    {transferSourceType === 'task' ? (
                      <label className="experiment-expert-field">
                        <span>{copy.transferSourceTask}</span>
                        <select value={transferSourceTaskId} onChange={(event) => onTransferSourceTaskChange(event.target.value)}>
                          <option value="">{copy.transferNoTasks}</option>
                          {completedTransferTasks.map((task) => (
                            <option key={task.id} value={task.id}>
                              #{task.id} {task.custom_model_name || task.model_script}
                            </option>
                          ))}
                        </select>
                        {transferStructureLocked ? (
                          <small className="experiment-expert-ok">{copy.transferConfigSynced}</small>
                        ) : null}
                      </label>
                    ) : (
                      <>
                        <label className="experiment-expert-field">
                          <span>{copy.transferWeightFile}</span>
                          <select value={selectedTrainingWeightId} onChange={(event) => onSelectTrainingWeight(event.target.value)}>
                            <option value="">{copy.transferNoWeights}</option>
                            {trainingWeights.map((weight) => (
                              <option key={weight.id} value={weight.id}>
                                {weight.original_filename} / {weight.status}
                              </option>
                            ))}
                          </select>
                        </label>
                        <div className="experiment-expert-actions">
                          <label className="experiment-expert-action" data-disabled={resources.uploadingWeight ? 'true' : 'false'}>
                            {resources.uploadingWeight ? copy.uploadingWeight : copy.uploadWeight}
                            <input
                              type="file"
                              accept=".pth,.pt"
                              disabled={resources.uploadingWeight}
                              style={{ display: 'none' }}
                              onChange={(event) => {
                                const file = event.target.files?.[0];
                                event.target.value = '';
                                if (file) onUploadWeight(file);
                              }}
                            />
                          </label>
                          {selectedTrainingWeight ? (
                            <button
                              type="button"
                              className="experiment-expert-action is-danger"
                              onClick={() => onDeleteWeight(selectedTrainingWeight.id)}
                              disabled={isProcessing}
                            >
                              {copy.deleteWeight}
                            </button>
                          ) : null}
                        </div>
                      </>
                    )}

                    <div className="experiment-expert-fields">
                      <label className="experiment-expert-field">
                        <span>{copy.transferStrict}</span>
                        <input value="strict" disabled readOnly />
                      </label>
                      <label className="experiment-expert-field">
                        <span>{copy.freezeMode}</span>
                        <select value={transferFreezeMode} onChange={(event) => onFoldChange('transferFreezeMode', event.target.value)}>
                          {availableTransferFreezeModes.map((mode) => (
                            <option key={mode} value={mode}>
                              {transferFreezeModeLabels[mode] || mode}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="experiment-expert-field">
                        <span>{copy.finetuneLearningRate}</span>
                        <input
                          type="number"
                          step="0.00001"
                          min="0.000001"
                          value={finetuneLearningRate}
                          onChange={(event) => onFoldChange('finetuneLearningRate', event.target.value)}
                        />
                      </label>
                    </div>
                    {transferStartBlocked ? (
                      <div className="experiment-result-note" data-tone="warning" role="status">
                        <strong>{copy.transferSelectSource}</strong>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </>
            ) : null}

            {activeTab === 'tags' ? (
              <>
                <h4 className="experiment-expert-heading">{copy.expertTabTags}</h4>
                <p className="experiment-expert-note">{copy.expertTagsHintLabel}</p>
                <div className="experiment-tag-block">
                  <TagPicker
                    tags={tagState.tags}
                    value={newTaskTagIds}
                    onChange={onTagIdsChange}
                    onCreate={onCreateTag}
                    disabled={!resources.user || tagState.loading || tagState.busy || isProcessing || Boolean(tagState.error)}
                    isZh={isZh}
                    label={copy.expertTagsLabel}
                  />
                </div>
                {tagState.error && <div role="alert" className="training-tag-hint">{tagState.error}</div>}
              </>
            ) : null}
          </div>
        </div>
      </section>

      <div className="experiment-center-hint experiment-canvas-footnote" role="status">
        {resources.user ? presetStatusText : copy.loginRequiredToUse}
        {pendingSubmission ? ` · ${pendingSubmission}` : ''}
      </div>
    </div>
  );
}
