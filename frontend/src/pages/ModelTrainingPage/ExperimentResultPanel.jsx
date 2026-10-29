import { useEffect, useMemo, useState } from 'react';
import { useT } from '../../i18n';
import TrainingTaskParameters from './TrainingTaskParameters';
import LossEvolutionChart from '../../components/LossEvolutionChart';
import ExperimentLogPanel from './ExperimentLogPanel';
import {
  EXPERIMENT_METRIC_KEYS,
  buildExperimentSummary,
  canUseTaskForPrediction,
  isRetiredEarthTask,
  formatExperimentMetricValue,
  getExperimentArchitectureLabel,
  getExperimentFailureMessage,
  normalizeTaskChannels,
  parseTaskHyperparameters,
  readExperimentMetrics,
} from './experimentCenterModel';
import { EARTH_METRIC_META, earthMetricUnit, getEarthMetricPolicyLabel } from '../../utils/earthMetricMeta.js';
import AnimatedNumber from '../../components/ui/AnimatedNumber';
import { useSettings } from '../../contexts/SettingsContext';
import { useAuth } from '../../contexts/AuthContext';
import { getTrainingStatusMeta } from './trainingStatusMeta';
import { fetchEarthDiagnosticContext } from '../../services/earthPredict.js';
import { earthDiagnosticTaskAvailability, earthDiagnosticErrorText } from '../PredictPage/earthDiagnosticsModel.js';
import './experimentCenter.css';

const METRIC_LABEL_KEYS = {
  rmse: 'experimentCenter.metricRmse',
  mae: 'experimentCenter.metricMae',
  mse: 'experimentCenter.metricMse',
  r2: 'experimentCenter.metricR2',
  mape: 'experimentCenter.metricMape',
  smape: 'experimentCenter.metricSmape',
};

function DetailField({ label, value }) {
  return (
    <div className="experiment-result-detail">
      <dt>{label}</dt>
      <dd>
        {value || '--'}
      </dd>
    </div>
  );
}

function getUploadedModelIdentity(task, hyperparameters) {
  const modelSource = String(task?.model_source || hyperparameters?.model_source || '').trim().toLowerCase();
  if (modelSource !== 'uploaded') return null;

  const modelName = String(
    task?.uploaded_model_name
      || hyperparameters?._uploaded_model_name
      || task?.uploaded_model_id
      || hyperparameters?._uploaded_model_id
      || '',
  ).trim();
  const versionValue = task?.uploaded_model_version ?? hyperparameters?._uploaded_model_version;
  const modelVersion = versionValue == null ? '' : String(versionValue).trim();
  const storedFilename = String(hyperparameters?._uploaded_model_filename || '').trim();
  const storedPath = String(hyperparameters?._uploaded_model_path || '').trim();
  const pathFilename = storedPath.split(/[\\/]/).pop() || '';

  return {
    name: modelName,
    version: modelVersion,
    filename: storedFilename || pathFilename,
  };
}

/**
 * 实验结果工作区：状态摘要、失败原因、指标、Loss、完整参数、数据集身份、
 * 运行日志与后续动作。
 *
 * 「用于预测」由父组件写入现有 TRAINING_TASK_HANDOFF_KEY 后跳转，只对
 * `completed && model_available` 开放；「去模型比较」跳转预测页并显式进入
 * 比较模式，不伪造预选结果。模型可用性只看后端返回的 model_available。
 */
export default function ExperimentResultPanel({
  activeTask,
  progress,
  logs,
  isLight,
  isProcessing,
  copy,
  channelOrder,
  channelMap,
  baselineLabel,
  onPredict,
  onCompare,
  onCopyConfig,
  onRename,
  onDelete,
  onTest,
  onFormatValue,
}) {
  const t = useT();
  const { settings } = useSettings();
  const { user } = useAuth();
  const isZh = settings.language !== 'en';
  const [logOpen, setLogOpen] = useState(false);
  const [testAvailability, setTestAvailability] = useState(null);
  const [testCheckRevision, setTestCheckRevision] = useState(0);
  const summary = useMemo(() => buildExperimentSummary(activeTask), [activeTask]);
  const hyperparameters = useMemo(
    () => parseTaskHyperparameters(activeTask?.hyperparameters),
    [activeTask?.hyperparameters]
  );
  const uploadedModel = useMemo(
    () => getUploadedModelIdentity(activeTask, hyperparameters),
    [activeTask, hyperparameters]
  );
  const failure = useMemo(() => getExperimentFailureMessage(activeTask), [activeTask]);
  const localTestAvailability = earthDiagnosticTaskAvailability(activeTask, isZh);
  const earthTask = summary.dataset.startsWith('earth_');
  const diagnosticIdentity = JSON.stringify([user?.id, activeTask?.id, activeTask?.status, activeTask?.model_available,
    activeTask?.dataset_id, activeTask?.dataset_version, activeTask?.dataset_fingerprint,
    activeTask?.checkpoint_sha256, activeTask?.output_model_path, activeTask?.hyperparameters]);
  useEffect(() => {
    setTestAvailability(null);
    if (!earthTask || !localTestAvailability.available) return;
    const controller = new AbortController();
    fetchEarthDiagnosticContext(activeTask.id, { signal: controller.signal })
      .then((context) => { if (!controller.signal.aborted) setTestAvailability({
        identity: diagnosticIdentity, available: context.available,
        reason: context.available ? '' : earthDiagnosticErrorText(context),
      }); })
      .catch((error) => { if (!controller.signal.aborted) setTestAvailability({
        identity: diagnosticIdentity, available: false, reason: earthDiagnosticErrorText(error),
      }); });
    return () => controller.abort();
  }, [diagnosticIdentity, earthTask, localTestAvailability.available, testCheckRevision]);

  if (!activeTask) {
    return (
      <div className="experiment-center-empty">
        <div>{t('experimentCenter.noActiveTask')}</div>
      </div>
    );
  }

  const statusMeta = getTrainingStatusMeta(activeTask.status, t);
  const channels = normalizeTaskChannels(activeTask, channelOrder);
  const channelLabel = channels.length > 0
    ? channels.map((channel) => channelMap[channel]?.name || channel).join(', ')
    : baselineLabel;
  const lossHistory = progress?.loss_history || { train: [], val: [] };
  const hasLossHistory = Array.isArray(lossHistory.train) && lossHistory.train.length > 0;
  const modelAvailable = activeTask.status === 'completed' && activeTask.model_available === true;
  const predictionAvailable = canUseTaskForPrediction(activeTask);
  const retiredDataset = isRetiredEarthTask(activeTask);
  const earth = summary.dataset.startsWith('earth_');
  const testVerified = testAvailability?.identity === diagnosticIdentity ? testAvailability : null;
  const testAvailable = localTestAvailability.available && (!earth || testVerified?.available === true);
  const testReason = !localTestAvailability.available ? localTestAvailability.reason
    : earth && !testVerified ? (isZh ? '正在核验任务、数据与模型产物…' : 'Checking task, data and model artifact…')
      : testVerified?.reason || '';

  const metrics = EXPERIMENT_METRIC_KEYS
    .filter((key) => earth || summary.metrics[key] !== undefined)
    .map((key) => ({ key, label: earth ? EARTH_METRIC_META.find(metric => metric.key === key)[isZh ? 'zh' : 'en'] : t(METRIC_LABEL_KEYS[key]),
      raw: summary.metrics[key],
      value: summary.metrics[key] === undefined ? t('experimentCenter.metricNotProvided') : formatExperimentMetricValue(summary.metrics[key]) }));
  const validationMetrics = earth ? readExperimentMetrics(activeTask.metrics, 'validation') : {};
  const metricPolicyLabel = earth ? getEarthMetricPolicyLabel(activeTask.metrics, isZh) : '';

  const datasetFingerprint = activeTask.dataset_fingerprint || '';
  const datasetVersion = activeTask.dataset_version || '';
  const datasetStatus = activeTask.dataset_identity_status || '';

  return (
    <div className="experiment-center-stack experiment-result">
      <div className="experiment-result-head">
        <div className="experiment-result-identity">
          <h3 className="experiment-result-name">
            {summary.name || t('experimentCenter.unnamedExperiment')}
          </h3>
          <p className="experiment-result-meta experiment-center-hint">
            {`${getExperimentArchitectureLabel(summary.architecture) || copy.uploadedModelsLabel} · ${summary.dataset || '--'}`}
          </p>
        </div>
        <div className="experiment-center-actions">
          <span className="experiment-directory-badge experiment-result-status" data-status={activeTask.status}>
            <span className="experiment-directory-status-dot" />
            {statusMeta.label}
          </span>
          <span
            className="experiment-directory-badge experiment-result-status"
            data-status={modelAvailable ? 'completed' : 'failed'}
          >
            {modelAvailable ? copy.weightsAvailable : copy.weightsUnavailable}
          </span>
        </div>
      </div>

      {/* 失败、停止与缺少权重的任务：显示 metrics 里的真实原因与恢复建议。 */}
      {retiredDataset ? (
        <div className="experiment-result-note" data-tone="warning" role="status">
          {t('predict.earthDatasetRetired')}
        </div>
      ) : null}
      {!modelAvailable ? (
        <div
          className="experiment-result-note"
          data-tone={failure.stopped ? 'warning' : 'error'}
          role="status"
        >
          <strong>{failure.stopped ? t(failure.messageKey) : copy.modelUnavailableTitle}</strong>
          {failure.hasReason ? (
            <>
              {!failure.stopped ? <span>{t(failure.messageKey)}</span> : null}
              {failure.suggestionKey ? <span>{t(failure.suggestionKey)}</span> : null}
              {failure.detail ? <span className="experiment-center-hint">{failure.detail}</span> : null}
              {failure.errorCode ? <span className="experiment-center-hint">{failure.errorCode}</span> : null}
            </>
          ) : (
            <span>{copy.notCompletedReason}</span>
          )}
        </div>
      ) : null}

      <section className="experiment-result-section" data-earth-training-metrics={earth ? 'true' : undefined} aria-labelledby="experiment-result-metrics-title">
        <h4 className="experiment-result-section-title" id="experiment-result-metrics-title">{earth ? t('experimentCenter.fullTestMetrics') : copy.metricsTitle}</h4>
        {earth ? <p className="experiment-center-hint" data-earth-metric-policy={activeTask.metrics?.aggregation || 'unknown'}>
          {metricPolicyLabel}
        </p> : null}
        {metrics.length > 0 ? (
          <div className="experiment-result-metrics">
            {metrics.map((metric) => (
              <div className="experiment-result-metric" key={metric.key}>
                <div className="experiment-result-metric-label">{metric.label}</div>
                <div className="experiment-result-metric-value" data-result-metric={metric.key}>
                  {typeof metric.raw === 'number' && Number.isFinite(metric.raw)
                    ? <AnimatedNumber value={metric.raw} format={formatExperimentMetricValue} />
                    : metric.value}
                </div>
                {earth ? <small>{earthMetricUnit(metric.key, isZh)}</small> : null}
              </div>
            ))}
          </div>
        ) : (
          <div className="experiment-result-empty experiment-center-hint">{copy.metricsUnavailable}</div>
        )}
        {earth ? (
          <>
            <h5 className="experiment-result-group-title">{t('experimentCenter.validationMetrics')}</h5>
            <div className="experiment-result-metrics" data-earth-validation-metrics="true">
              {EXPERIMENT_METRIC_KEYS.map((key) => (
                <div className="experiment-result-metric" key={key}>
                  <div className="experiment-result-metric-label">{EARTH_METRIC_META.find(metric => metric.key === key)[isZh ? 'zh' : 'en']}</div>
                  <div className="experiment-result-metric-value">{validationMetrics[key] === undefined
                    ? t('experimentCenter.metricNotProvided') : formatExperimentMetricValue(validationMetrics[key])}</div>
                  <small>{earthMetricUnit(key, isZh)}</small>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </section>

      <LossEvolutionChart
        lossHistory={lossHistory}
        isLight={isLight}
        compact
        height={280}
        title={copy.lossTitle}
        className="experiment-result-chart"
        subtitle={hasLossHistory ? copy.lossEpochCount(lossHistory.train.length) : null}
        emptyLabel={copy.noLossHistory}
      />

      {/* 默认折叠的运行日志：复用 TrainingContext 的 logs，不新增请求或轮询。 */}
      <ExperimentLogPanel
        task={activeTask}
        logs={logs}
        isLight={isLight}
        autoFollow={false}
        collapsible
        open={logOpen}
        onToggle={() => setLogOpen((value) => !value)}
        title={copy.viewRunLogs}
        hint={logOpen ? copy.logLines((logs || []).length) : copy.viewRunLogsHint}
        emptyLabel={copy.noLogsForTask}
        copy={copy}
      />

      <section className="experiment-result-section" aria-labelledby="experiment-result-details-title">
        <h4 className="experiment-result-section-title" id="experiment-result-details-title">{copy.detailsTitle}</h4>
        <h5 className="experiment-result-group-title">{copy.datasetIdentityTitle}</h5>
        <dl className="experiment-result-detail-grid">
          <DetailField label={copy.datasetLabel} value={summary.dataset} />
          <DetailField label={copy.datasetVersionLabel} value={datasetVersion} />
          <DetailField label={copy.datasetStatusLabel} value={datasetStatus} />
          <DetailField label={copy.datasetFingerprintLabel} value={datasetFingerprint ? `${datasetFingerprint.slice(0, 12)}…` : ''} />
        </dl>
        <h5 className="experiment-result-group-title">{copy.trainingSettingsTitle}</h5>
        <dl className="experiment-result-detail-grid">
          <DetailField label={copy.modelSourceLabel} value={summary.modelSource === 'uploaded' ? copy.uploadedModelsLabel : copy.officialLabel} />
          {uploadedModel ? <DetailField label={copy.customModelLabel} value={uploadedModel.name} /> : null}
          {uploadedModel?.version ? <DetailField label={copy.customModelVersionLabel} value={uploadedModel.version} /> : null}
          {uploadedModel?.filename ? <DetailField label={copy.customModelFileLabel} value={uploadedModel.filename} /> : null}
          <DetailField label={copy.inputChannelsLabel} value={channelLabel} />
          <DetailField label={copy.windowLabel} value={hyperparameters.window != null ? String(hyperparameters.window) : ''} />
          <DetailField label={copy.horizonLabel} value={hyperparameters.horizon != null ? String(hyperparameters.horizon) : ''} />
          <DetailField label={copy.epochsLabel} value={hyperparameters.epochs != null ? String(hyperparameters.epochs) : ''} />
          <DetailField label={copy.batchSizeLabel} value={hyperparameters.batch_size != null ? String(hyperparameters.batch_size) : ''} />
          <DetailField label={copy.learningRateLabel} value={hyperparameters.learning_rate != null ? String(hyperparameters.learning_rate) : ''} />
        </dl>
        <div className="experiment-result-parameters">
          <TrainingTaskParameters
            showSummary={false}
            hyperparameters={hyperparameters}
            modelName={summary.name || activeTask.id}
            t={t}
            formatValue={onFormatValue}
          />
        </div>
      </section>

      <div className="experiment-center-action-row">
        <div className="experiment-center-actions">
          {predictionAvailable ? (
            <button
              type="button"
              className="experiment-center-button experiment-center-button-primary"
              onClick={() => onPredict(activeTask)}
            >
              {copy.useForPrediction}
            </button>
          ) : null}
          <button
            type="button"
            className="experiment-center-button"
            disabled={!predictionAvailable}
            title={predictionAvailable ? undefined : copy.compareUnavailableToast}
            onClick={() => onCompare(activeTask)}
          >
            {copy.goCompare}
          </button>
          <button type="button" className="experiment-center-button" disabled={retiredDataset} onClick={() => onCopyConfig(activeTask)}>
            {copy.copyConfig}
          </button>
        </div>
        <div className="experiment-center-actions">
          <button
            type="button"
            className="experiment-center-button"
            disabled={isProcessing}
            onClick={() => onRename(activeTask)}
          >
            {copy.renameModel}
          </button>
          <button
            type="button"
            className="experiment-center-button"
            disabled={!testAvailable || isProcessing}
            title={testReason || undefined}
            onClick={() => onTest(activeTask.id)}
          >
            {copy.testModel}
          </button>
          <button
            type="button"
            className="experiment-center-button experiment-center-button-quiet"
            disabled={isProcessing}
            onClick={() => onDelete(activeTask.id)}
          >
            {copy.deleteRecord}
          </button>
        </div>
      </div>
      {testReason ? <p className="experiment-center-hint" role="status" data-model-test-unavailable="true">{testReason}</p> : null}
      {earth && localTestAvailability.available && testVerified?.available === false ? <button
        type="button" className="experiment-center-button" onClick={() => setTestCheckRevision((n) => n + 1)}>
        {isZh ? '重新核验测试可用性' : 'Recheck test availability'}
      </button> : null}
    </div>
  );
}
