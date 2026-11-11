import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { compareEarthModels } from '../../services/earthPredict';
import { useT } from '../../i18n';
import PredictSidebar from './PredictSidebar';
import CompareTrainingModelsPanel from './CompareTrainingModels/CompareTrainingModelsPanel';
import { compareResultMatchesTasks, getCompareSelectionState, buildCompareModelSummary } from './CompareTrainingModels/compareTrainingModelsData';
import { getEarthTrainingModelOptions, resolveEarthPredictErrorMessage } from './earthPredictModel';
import { PREDICT_MODEL_MODE_COMPARE } from './predictModelModes';
import { earthPredictionAdapter } from './PredictionPlanetAdapter';
import { sameEnvironmentTarget } from '../../utils/environmentVariables.js';
import {
  PREDICT_REQUEST_CHANNELS,
  createPredictRequestCoordinator,
  isAbortError,
} from './predictRequestCoordinator';

export default function EarthCompareWorkspace({ tasks, scope, tasksLoading, tasksError, onRetryTasks, isLight, precision, isZh, plotTextColor, plotGridColor }) {
  const t = useT();
  const options = useMemo(() => getEarthTrainingModelOptions(tasks), [tasks]);
  const [ids, setIds] = useState([]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const requestCoordinatorRef = useRef(null);
  if (!requestCoordinatorRef.current) {
    requestCoordinatorRef.current = createPredictRequestCoordinator();
  }
  const requestCoordinator = requestCoordinatorRef.current;
  const selection = getCompareSelectionState(ids);
  const selectedOptions = options.filter(option => selection.ids.includes(option.id));
  const horizons = selectedOptions.map(option => Number(buildCompareModelSummary(option.task).horizon));
  const horizon = horizons.length && horizons.every(value => Number.isInteger(value) && value >= 1 && value <= 240 && value === horizons[0]) ? horizons[0] : null;
  const key = `${scope}:${selection.ids.join(',')}`;
  useLayoutEffect(() => {
    requestCoordinator.invalidateAll();
    setData(null); setError(null); setLoading(false);
    return () => requestCoordinator.invalidateAll();
  }, [key, options, requestCoordinator]);
  useEffect(() => {
    const allowed = new Set(options.map(option => option.id));
    setIds(current => current.filter(id => allowed.has(id)));
  }, [options]);
  const run = async () => {
    if (!selection.canCompare || loading) return;
    if (!sameEnvironmentTarget(selectedOptions.map(option => option.task), 'earth')) {
      setError(isZh ? '请选择相同预测目标的模型进行比较。' : 'Choose models with the same prediction target.');
      return;
    }
    const requestToken = requestCoordinator.start(PREDICT_REQUEST_CHANNELS.compareMetrics, key);
    const requestKey = key;
    // A repeated comparison for the same selection must replace the previous
    // result before the new response is known.
    setData(null);
    setLoading(true); setError(null);
    try {
      const result = await compareEarthModels(selection.ids, { signal: requestToken.signal });
      if (!requestCoordinator.isCurrent(requestToken, requestKey)) return;
      if (!compareResultMatchesTasks(result, selectedOptions.map(option => option.task), 'earth')) {
        throw new Error(isZh
          ? '地球模型对比响应与所选任务的目标变量或输入变量不一致，请重新选择模型后重试。'
          : 'The Earth comparison response does not match the selected task targets or inputs. Reselect the models and retry.');
      }
      setData(result);
    } catch (e) {
      if (!requestCoordinator.isCurrent(requestToken, requestKey) || isAbortError(e)) return;
      setData(null);
      const message = resolveEarthPredictErrorMessage(e);
      setError(message.key ? t(`predict.${message.key}`) : message.fallback);
    } finally {
      if (requestCoordinator.isCurrent(requestToken, requestKey)
        && requestCoordinator.finish(requestToken, requestKey)) {
        setLoading(false);
      }
    }
  };
  return <div className="predict-workspace" data-earth-compare-workspace>
    <PredictSidebar adapter={earthPredictionAdapter({ context: horizon ? { horizon, dataset_id: 'earth_merra2_3hourly_v1' } : null, isZh })} isLight={isLight} modelMode={PREDICT_MODEL_MODE_COMPARE}
      trainingModelOptions={options} selectedCompareTrainingTaskIds={ids} setSelectedCompareTrainingTaskIds={setIds}
      trainingTasksLoading={tasksLoading} trainingTasksError={tasksError} onRetryTrainingTasks={onRetryTasks}
      loading={loading} requestContextLocked={loading} error={error}
      predStep={horizon ?? ''} predictionHorizonLimit={horizon} handlePredict={run}
      analysisVisibility={{ inputVariables: false, systemHyperparams: false }} precision={precision} />
    <div style={{ minWidth: 0 }}>
      <p className="earth-predict-note">{t('predict.earthCompareNote')}</p>
      <CompareTrainingModelsPanel planet="earth" key={key} data={data} loading={loading} selectedCount={selection.count}
        precision={precision} isZh={isZh} plotTextColor={plotTextColor} plotGridColor={plotGridColor} />
    </div>
  </div>;
}
