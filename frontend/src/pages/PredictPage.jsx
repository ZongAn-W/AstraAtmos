import { useState, useCallback, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import {
  clearPredictCache,
  getPredictCache,
  getEmptyPredictCache,
  getPredictionResultCacheForContext,
  resolvePredictCacheScope,
  setPredictCache,
  setPredictUiPreferences,
} from '../stores/predictCache';
import C from '../constants/colors';
import { useT } from '../i18n';
import { useSettings } from '../contexts/SettingsContext';
import { useAuth } from '../contexts/AuthContext';
import SectionTitle from '../components/SectionTitle';

import {
  runPrediction,
  fetchPredictMetrics,
  fetchErrorDistribution,
  fetchPermutationImportance,
  fetchTasks,
  compareTrainingModelErrorDistributions,
  compareTrainingModelPfi,
  compareTrainingModels,
} from '../services/api';

import { VARIABLE_DEFS } from './PredictPage/PredictComponents';
import PredictSidebar from './PredictPage/PredictSidebar';
import ErrorDistributionChart from './PredictPage/ErrorDistributionChart';
import PermutationImportanceChart from './PredictPage/PermutationImportanceChart';
import { getPredictAnalysisVisibility } from './PredictPage/predictAnalysisVisibility';
import {
  TRAINING_TASK_HANDOFF_KEY,
  getCompletedTrainingModelOptions,
  parseTrainingTaskHandoff,
} from './PredictPage/trainedModelSelection';
import {
  buildPerformanceMetricsFromEval,
} from './PredictPage/trainedModelAnalysisData';
import {
  buildErrorDistributionKey,
  buildPermutationImportanceKey,
  buildPredictionContextKey,
  buildPredictMetricsKey,
  buildTrainingModelCompareKey,
} from './PredictPage/predictAnalysisCacheKeys';
import {
  PREDICT_REQUEST_CHANNELS,
  createPredictRequestCoordinator,
  isAbortError,
} from './PredictPage/predictRequestCoordinator';
import {
  PREDICT_MODEL_MODE_COMPARE,
  PREDICT_MODEL_MODE_EARTH,
  PREDICT_MODEL_MODE_EARTH_COMPARE,
  PREDICT_MODEL_MODE_TRAINED,
  normalizePredictModelMode,
  readPredictModeFromHash,
} from './PredictPage/predictModelModes';
import EarthPredictPanel from './PredictPage/EarthPredictPanel';
import EarthCompareWorkspace from './PredictPage/EarthCompareWorkspace';
import PredictModeSelector from './PredictPage/PredictModeSelector';
import {
  buildEarthPredictKey,
  getEarthTrainingModelOptions,
  isEarthTask,
  pickDefaultOrigin,
  resolveEarthPredictErrorMessage,
  shouldClearEarthResult,
} from './PredictPage/earthPredictModel';
import { fetchEarthPredictContext, runEarthPrediction } from '../services/earthPredict';
import CompareTrainingModelsPanel from './PredictPage/CompareTrainingModels/CompareTrainingModelsPanel';
import { getCompareSelectionState } from './PredictPage/CompareTrainingModels/compareTrainingModelsData';
import {
  clampPredictionHorizon,
  resolvePredictionHorizonLimit,
} from './PredictPage/predictionHorizon';
import { validatePredictCacheTrainingTasks } from './PredictPage/predictCacheTaskValidation';
import SingleModelWorkbench from './PredictPage/SingleModelWorkbench';
import { earthPredictionAdapter, marsPredictionAdapter } from './PredictPage/PredictionPlanetAdapter';
import { earthResponseMatchesRequest } from './PredictPage/singleModelAdapters';
import './PredictPage/singleModelWorkbench.css';
import { selectPredictionMetrics } from './PredictPage/predictionMetricSelection';

export default function PredictPage() {
  const t = useT();
  const translateRef = useRef(t);
  translateRef.current = t;
  const { settings } = useSettings();
  const { user, isLoading } = useAuth();
  const precision = settings.precision;
  const ozoneUnit = settings.units.ozone;
  const isLight = settings.theme === 'light';

  const VARIABLES = VARIABLE_DEFS.map((v) => ({ ...v, label: t(`predict.variables.${v.id}`) }));

  const plotTextColor = isLight ? 'rgba(23,33,47,0.96)' : 'rgba(236,244,255,0.96)';
  const plotText60 = isLight ? 'rgba(23,33,47,0.76)' : 'rgba(214,228,244,0.78)';
  const plotGridColor = isLight ? 'rgba(23,33,47,0.12)' : 'rgba(160,196,240,0.16)';

  const predictScope = resolvePredictCacheScope({ user, isLoading });
  const emptyCache = getEmptyPredictCache(predictScope);
  const requestCoordinatorRef = useRef(null);
  if (!requestCoordinatorRef.current) {
    requestCoordinatorRef.current = createPredictRequestCoordinator();
  }
  const requestCoordinator = requestCoordinatorRef.current;
  const predictScopeRef = useRef(null);
  const cacheReadyScopeRef = useRef(null);
  const restoredScopeRef = useRef(null);
  const [cacheReadyScope, setCacheReadyScope] = useState(null);
  const pendingTrainingTaskHandoffRef = useRef(null);
  const writePredictCache = useCallback((updates) => {
    const scope = predictScopeRef.current;
    if (!scope || cacheReadyScopeRef.current !== scope) return false;
    return setPredictCache(scope, updates);
  }, []);

  const [selectedVars, setSelectedVars] = useState(() => VARIABLE_DEFS.map((v) => v.id));
  const [predStep, setPredStep] = useState(3);
  const [lsStart, setLsStart] = useState(90);
  const dataSourceMode = 'default';
  // 训练页可以通过 hash query 显式请求模式（如 mode=trained_compare）。
  const [hashRequestedMode, setHashRequestedMode] = useState(() => readPredictModeFromHash(window.location.hash));
  const [modelMode, setModelMode] = useState(
    () => readPredictModeFromHash(window.location.hash) || normalizePredictModelMode()
  );
  const [trainingTasks, setTrainingTasks] = useState([]);
  const [trainingTasksScope, setTrainingTasksScope] = useState(null);
  const [trainingTasksLoading, setTrainingTasksLoading] = useState(false);
  const [trainingTasksLoaded, setTrainingTasksLoaded] = useState(false);
  const [trainingTasksError, setTrainingTasksError] = useState(null);
  const trainingTasksRequestRef = useRef({ id: 0, controller: null });
  const previousModelModeRef = useRef(modelMode);
  const [selectedTrainingTaskId, setSelectedTrainingTaskId] = useState(null);
  const [selectedCompareTrainingTaskIds, setSelectedCompareTrainingTaskIds] = useState([]);
  const [activeHorizon, setActiveHorizon] = useState(0);
  const [viewMode, setViewMode] = useState(emptyCache.viewMode);

  const [loading, setLoading] = useState(false);
  const [resultContextKey, setResultContextKey] = useState(null);
  const [results, setResults] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [errorDistData, setErrorDistData] = useState(null);
  const [pfiData, setPfiData] = useState(null);
  const [metricsKey, setMetricsKey] = useState(null);
  const [errorDistKey, setErrorDistKey] = useState(null);
  const [pfiKey, setPfiKey] = useState(null);
  const [compareTrainingMetricsData, setCompareTrainingMetricsData] = useState(null);
  const [compareTrainingMetricsKey, setCompareTrainingMetricsKey] = useState(null);
  const [compareTrainingErrorData, setCompareTrainingErrorData] = useState(null);
  const [compareTrainingErrorKey, setCompareTrainingErrorKey] = useState(null);
  const [compareTrainingPfiData, setCompareTrainingPfiData] = useState(null);
  const [compareTrainingPfiKey, setCompareTrainingPfiKey] = useState(null);
  const [compareTrainingLoading, setCompareTrainingLoading] = useState(false);
  const [compareTrainingErrorLoading, setCompareTrainingErrorLoading] = useState(false);
  const [compareTrainingPfiLoading, setCompareTrainingPfiLoading] = useState(false);
  const [error, setError] = useState(null);

  const [fullscreen3D, setFullscreen3D] = useState(null);

  const [performanceData, setPerformanceData] = useState(null);
  const [performanceKey, setPerformanceKey] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [errorDistLoading, setErrorDistLoading] = useState(false);
  const [pfiLoading, setPfiLoading] = useState(false);

  const [compareConfigs, setCompareConfigs] = useState([]);
  const [selectedCompareIds, setSelectedCompareIds] = useState([]);

  // ── 地球历史预测（与火星模式完全分离的状态）─────────────────────────
  const [earthTaskId, setEarthTaskId] = useState('');
  const [earthOrigin, setEarthOrigin] = useState('');
  const [earthDay, setEarthDay] = useState(0);
  const [earthContext, setEarthContext] = useState(null);
  const [earthContextLoading, setEarthContextLoading] = useState(false);
  const [earthContextError, setEarthContextError] = useState(null);
  const [earthResult, setEarthResult] = useState(null);
  const [earthResultKey, setEarthResultKey] = useState(null);
  const [earthResultError, setEarthResultError] = useState(null);
  const [earthLoading, setEarthLoading] = useState(false);
  const earthRequestRef = useRef(null);
  const earthContextRequestRef = useRef(null);
  const earthRunRequestRef = useRef(null);

  const analysisVisibility = useMemo(
    () => getPredictAnalysisVisibility(modelMode),
    [modelMode]
  );
  const trainingModelOptions = useMemo(
    () => getCompletedTrainingModelOptions(
      trainingTasksScope === predictScope ? trainingTasks : []
    ),
    [predictScope, trainingTasks, trainingTasksScope]
  );
  const selectedTrainingOption = useMemo(
    () => trainingModelOptions.find((option) => option.id === Number(selectedTrainingTaskId)) || null,
    [selectedTrainingTaskId, trainingModelOptions]
  );
  const compareSelection = useMemo(
    () => getCompareSelectionState(selectedCompareTrainingTaskIds),
    [selectedCompareTrainingTaskIds]
  );
  const compareSelectionIdKey = compareSelection.ids.join(',');
  const selectedCompareTrainingTasks = useMemo(() => {
    const selectedIds = new Set(compareSelection.ids);
    return trainingModelOptions
      .filter((option) => selectedIds.has(option.id))
      .map((option) => option.task);
  }, [compareSelectionIdKey, trainingModelOptions]);
  const predictionHorizonLimit = useMemo(
    () => resolvePredictionHorizonLimit({
      modelMode,
      selectedTask: selectedTrainingOption?.task,
      selectedTasks: selectedCompareTrainingTasks,
    }),
    [modelMode, selectedCompareTrainingTasks, selectedTrainingOption]
  );
  const currentPredictionContext = useMemo(() => ({
    modelMode,
    trainingTaskId: modelMode === PREDICT_MODEL_MODE_TRAINED
      ? Number(selectedTrainingTaskId) || null
      : null,
    horizon: predStep,
    selectedVars,
    lsStart,
  }), [lsStart, modelMode, predStep, selectedTrainingTaskId, selectedVars]);
  const currentPredictionContextKey = useMemo(
    () => buildPredictionContextKey(currentPredictionContext),
    [currentPredictionContext]
  );
  const currentPageRequestContextKey = `${currentPredictionContextKey}|compare:${compareSelectionIdKey}`;
  const previousRequestContextKeyRef = useRef(currentPageRequestContextKey);
  const currentCompareTrainingMetricsKey = useMemo(
    () => buildTrainingModelCompareKey({
      taskIds: compareSelection.ids,
      horizon: predStep,
      compareType: 'metrics',
    }),
    [compareSelectionIdKey, predStep]
  );
  const activeCompareTrainingData = currentCompareTrainingMetricsKey === compareTrainingMetricsKey
    ? compareTrainingMetricsData
    : null;
  const currentCompareTrainingErrorKey = useMemo(
    () => buildTrainingModelCompareKey({
      taskIds: compareSelection.ids,
      horizon: predStep,
      compareType: 'error-distribution',
    }),
    [compareSelectionIdKey, predStep]
  );
  const activeCompareTrainingErrorData = currentCompareTrainingErrorKey === compareTrainingErrorKey
    ? compareTrainingErrorData
    : null;
  const currentCompareTrainingPfiKey = useMemo(
    () => buildTrainingModelCompareKey({
      taskIds: compareSelection.ids,
      horizon: predStep,
      compareType: 'pfi',
    }),
    [compareSelectionIdKey, predStep]
  );
  const currentPerformanceContextKey = useMemo(() => {
    const selectedConfigIds = [...selectedCompareIds].map(String).sort().join(',');
    return `${currentPredictionContextKey}|performance:${selectedConfigIds}`;
  }, [currentPredictionContextKey, selectedCompareIds]);
  const currentRequestContextKeysRef = useRef({});
  currentRequestContextKeysRef.current = {
    [PREDICT_REQUEST_CHANNELS.single]: currentPredictionContextKey,
    [PREDICT_REQUEST_CHANNELS.compareMetrics]: currentCompareTrainingMetricsKey,
    [PREDICT_REQUEST_CHANNELS.compareErrorDistribution]: currentCompareTrainingErrorKey,
    [PREDICT_REQUEST_CHANNELS.comparePfi]: currentCompareTrainingPfiKey,
    [PREDICT_REQUEST_CHANNELS.performance]: currentPerformanceContextKey,
  };
  const isRequestCurrent = (requestToken, requestContextKey) => (
    requestToken.scope === predictScopeRef.current
    &&
    currentRequestContextKeysRef.current[requestToken.channel] === requestContextKey
    && requestCoordinator.isCurrent(requestToken, requestContextKey)
  );
  const isRequestLatest = (requestToken, requestContextKey) => (
    requestToken.scope === predictScopeRef.current
    &&
    currentRequestContextKeysRef.current[requestToken.channel] === requestContextKey
    && requestCoordinator.isLatest(requestToken, requestContextKey)
  );
  const activeCompareTrainingPfiData = currentCompareTrainingPfiKey === compareTrainingPfiKey
    ? compareTrainingPfiData
    : null;
  const hasCurrentSingleResult = modelMode !== PREDICT_MODEL_MODE_COMPARE
    && resultContextKey === currentPredictionContextKey;
  const activeResults = hasCurrentSingleResult ? results : null;
  const activeMetrics = hasCurrentSingleResult ? metrics : null;
  const activeErrorDistData = hasCurrentSingleResult ? errorDistData : null;
  const activePfiData = hasCurrentSingleResult ? pfiData : null;
  const activeError = previousRequestContextKeyRef.current === currentPageRequestContextKey
    ? error
    : null;
  const requestContextLocked = loading
    || metricsLoading
    || errorDistLoading
    || pfiLoading
    || compareTrainingLoading
    || compareTrainingErrorLoading
    || compareTrainingPfiLoading
    || earthLoading || earthContextLoading;

  const toggleVar = (id) => {
    setSelectedVars((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  useEffect(() => {
    if (!predictScope) return;
    const handoff = parseTrainingTaskHandoff(
      sessionStorage.getItem(TRAINING_TASK_HANDOFF_KEY),
      predictScope
    );
    sessionStorage.removeItem(TRAINING_TASK_HANDOFF_KEY);
    pendingTrainingTaskHandoffRef.current = handoff;
  }, [predictScope]);

  useLayoutEffect(() => {
    if (predictScopeRef.current === predictScope) return;

    const previousScope = predictScopeRef.current;
    requestCoordinator.invalidateAll();
    if (previousScope?.startsWith('user:')) clearPredictCache(previousScope);
    predictScopeRef.current = predictScope;
    cacheReadyScopeRef.current = null;
    restoredScopeRef.current = null;
    setCacheReadyScope(null);
    previousRequestContextKeyRef.current = null;

    setSelectedVars(VARIABLE_DEFS.map((variable) => variable.id));
    setPredStep(3);
    setLsStart(90);
    setModelMode(normalizePredictModelMode());
    setSelectedTrainingTaskId(null);
    setSelectedCompareTrainingTaskIds([]);
    setTrainingTasks([]);
    setTrainingTasksScope(null);
    setTrainingTasksLoading(false);
    setTrainingTasksLoaded(false);
    setTrainingTasksError(null);
    setCompareConfigs([]);
    setSelectedCompareIds([]);

    setLoading(false);
    setMetricsLoading(false);
    setErrorDistLoading(false);
    setPfiLoading(false);
    setCompareTrainingLoading(false);
    setCompareTrainingErrorLoading(false);
    setCompareTrainingPfiLoading(false);
    setError(null);

    setResultContextKey(null);
    setResults(null);
    setMetrics(null);
    setErrorDistData(null);
    setPfiData(null);
    setPerformanceData(null);
    setPerformanceKey(null);
    setMetricsKey(null);
    setErrorDistKey(null);
    setPfiKey(null);
    setActiveHorizon(0);
    setCompareTrainingMetricsData(null);
    setCompareTrainingMetricsKey(null);
    setCompareTrainingErrorData(null);
    setCompareTrainingErrorKey(null);
    setCompareTrainingPfiData(null);
    setCompareTrainingPfiKey(null);
    setFullscreen3D(null);
    earthContextRequestRef.current?.abort();
    earthRunRequestRef.current?.abort();
    trainingTasksRequestRef.current.controller?.abort();
    setEarthTaskId(''); setEarthOrigin(''); setEarthContext(null);
    setEarthResult(null); setEarthResultError(null); setEarthContextError(null);
    setEarthLoading(false); setEarthContextLoading(false);

    if (!predictScope) return;
  }, [predictScope, requestCoordinator]);

  const loadTrainingTasks = useCallback(() => {
    const scope = predictScope;
    const userId = user?.id;
    if (!scope || !userId) return Promise.resolve([]);

    trainingTasksRequestRef.current.controller?.abort();
    const requestId = trainingTasksRequestRef.current.id + 1;
    const controller = new AbortController();
    trainingTasksRequestRef.current = { id: requestId, controller };
    const isCurrent = () => (
      trainingTasksRequestRef.current.id === requestId
      && trainingTasksRequestRef.current.controller === controller
      && predictScopeRef.current === scope
      && user?.id === userId
    );

    setTrainingTasksLoading(true);
    setTrainingTasksError(null);

    return fetchTasks({ signal: controller.signal })
      .then((items) => {
        if (!isCurrent()) return items;
        setTrainingTasks(Array.isArray(items) ? items : []);
        setTrainingTasksScope(scope);
        setTrainingTasksLoaded(true);
        return items;
      })
      .catch((requestError) => {
        if (!isCurrent() || requestError?.name === 'AbortError') return undefined;
        setTrainingTasksError(requestError instanceof Error ? requestError : new Error(String(requestError || '')));
        return undefined;
      })
      .finally(() => {
        if (!isCurrent()) return;
        setTrainingTasksLoading(false);
        trainingTasksRequestRef.current = { id: requestId, controller: null };
      });
  }, [predictScope, user?.id]);

  useEffect(() => {
    if (isLoading) return undefined;

    if (!user) {
      trainingTasksRequestRef.current.controller?.abort();
      trainingTasksRequestRef.current = {
        id: trainingTasksRequestRef.current.id + 1,
        controller: null,
      };
      setTrainingTasks([]);
      setTrainingTasksScope(null);
      setTrainingTasksLoading(false);
      setTrainingTasksLoaded(false);
      setTrainingTasksError(null);
      setSelectedTrainingTaskId(null);
      setSelectedCompareTrainingTaskIds([]);
      return undefined;
    }

    loadTrainingTasks();
    return () => {
      trainingTasksRequestRef.current.controller?.abort();
    };
  }, [isLoading, loadTrainingTasks, user?.id]);

  useEffect(() => {
    if (!predictScope || predictScopeRef.current !== predictScope) return;
    const authenticatedScope = predictScope.startsWith('user:');
    if (authenticatedScope && (
      trainingTasksScope !== predictScope
      || trainingTasksLoading
      || !trainingTasksLoaded
    )) return;
    if (restoredScopeRef.current === predictScope) return;

    const accessibleTaskIds = trainingModelOptions.map((option) => option.id);
    const validatedCache = validatePredictCacheTrainingTasks(
      getPredictCache(predictScope),
      accessibleTaskIds
    );

    const cachedParams = validatedCache.params || {};
    const restoredSelectedVars = Array.isArray(cachedParams.selectedVars)
      ? cachedParams.selectedVars
      : VARIABLE_DEFS.map((variable) => variable.id);
    const restoredPredStep = cachedParams.predStep ?? 3;
    const restoredLsStart = cachedParams.lsStart ?? 90;
    const restoredModelMode = normalizePredictModelMode(cachedParams.modelMode);
    let restoredTrainingTaskId = cachedParams.trainingTaskId ?? null;

    const handoff = pendingTrainingTaskHandoffRef.current;
    const handoffTask = trainingTasks.find(task => Number(task.id) === Number(handoff?.taskId));
    if (handoffTask && isEarthTask(handoffTask) && getEarthTrainingModelOptions([handoffTask]).length) {
      setEarthTaskId(String(handoffTask.id));
    }
    if (authenticatedScope && handoff && accessibleTaskIds.includes(Number(handoff.taskId))) {
      restoredTrainingTaskId = Number(handoff.taskId);
      pendingTrainingTaskHandoffRef.current = null;
    } else if (!authenticatedScope || handoff) {
      pendingTrainingTaskHandoffRef.current = null;
    }

    // hash 显式请求的模式（训练页「去模型比较」/ 单模型 handoff）优先于缓存里的模式：
    // 否则上次浏览留下的 trained 模式会把比较模式顶掉，URL 与实际界面不一致。
    const urlRequestedMode = readPredictModeFromHash(window.location.hash);
    const effectiveRestoredModelMode = handoff
      ? (isEarthTask(handoffTask) ? PREDICT_MODEL_MODE_EARTH : PREDICT_MODEL_MODE_TRAINED)
      : (urlRequestedMode || restoredModelMode);
    const restoredContext = {
      modelMode: effectiveRestoredModelMode,
      trainingTaskId: effectiveRestoredModelMode === PREDICT_MODEL_MODE_TRAINED
        ? restoredTrainingTaskId
        : null,
      horizon: restoredPredStep,
      selectedVars: restoredSelectedVars,
      lsStart: restoredLsStart,
    };
    const restoredContextKey = buildPredictionContextKey(restoredContext);
    const restoredPerformanceKey = `${restoredContextKey}|performance:${[
      ...(validatedCache.selectedCompareIds || []),
    ].map(String).sort().join(',')}`;
    const restoredResult = getPredictionResultCacheForContext(
      validatedCache,
      predictScope,
      restoredContextKey,
      {
        metricsKey: buildPredictMetricsKey(restoredContext),
        errorDistKey: buildErrorDistributionKey(restoredContext),
        pfiKey: buildPermutationImportanceKey(restoredContext),
        performanceKey: restoredPerformanceKey,
      }
    );

    const restoredCompareIds = getCompareSelectionState(
      validatedCache.selectedCompareTrainingTaskIds
    ).ids;
    const restoredCompareMetricsKey = buildTrainingModelCompareKey({
      taskIds: restoredCompareIds,
      horizon: restoredPredStep,
      compareType: 'metrics',
    });
    const restoredCompareErrorKey = buildTrainingModelCompareKey({
      taskIds: restoredCompareIds,
      horizon: restoredPredStep,
      compareType: 'error-distribution',
    });
    const restoredComparePfiKey = buildTrainingModelCompareKey({
      taskIds: restoredCompareIds,
      horizon: restoredPredStep,
      compareType: 'pfi',
    });
    const compareMetricsMatch = Boolean(restoredCompareMetricsKey)
      && validatedCache.compareTrainingMetricsKey === restoredCompareMetricsKey;
    const compareErrorMatch = Boolean(restoredCompareErrorKey)
      && validatedCache.compareTrainingErrorKey === restoredCompareErrorKey;
    const comparePfiMatch = Boolean(restoredComparePfiKey)
      && validatedCache.compareTrainingPfiKey === restoredComparePfiKey;
    const restoredCache = {
      ...validatedCache,
      ...restoredResult,
      compareTrainingMetricsData: compareMetricsMatch
        ? validatedCache.compareTrainingMetricsData
        : null,
      compareTrainingMetricsKey: compareMetricsMatch ? restoredCompareMetricsKey : null,
      compareTrainingErrorData: compareErrorMatch
        ? validatedCache.compareTrainingErrorData
        : null,
      compareTrainingErrorKey: compareErrorMatch ? restoredCompareErrorKey : null,
      compareTrainingPfiData: comparePfiMatch ? validatedCache.compareTrainingPfiData : null,
      compareTrainingPfiKey: comparePfiMatch ? restoredComparePfiKey : null,
    };
    setPredictCache(predictScope, restoredCache);

    setSelectedVars(restoredSelectedVars);
    setPredStep(restoredPredStep);
    setLsStart(restoredLsStart);
    setModelMode(effectiveRestoredModelMode);
    setSelectedTrainingTaskId(restoredTrainingTaskId);
    setSelectedCompareTrainingTaskIds(restoredCache.selectedCompareTrainingTaskIds || []);
    setCompareConfigs(restoredCache.compareConfigs || []);
    setSelectedCompareIds(restoredCache.selectedCompareIds || []);

    setResultContextKey(restoredResult.resultContextKey);
    setResults(restoredResult.results);
    setMetrics(restoredResult.metrics);
    setErrorDistData(restoredResult.errorDistData);
    setPfiData(restoredResult.pfiData);
    setPerformanceData(restoredResult.performanceData);
    setPerformanceKey(restoredResult.performanceKey);
    setMetricsKey(restoredResult.metricsKey);
    setErrorDistKey(restoredResult.errorDistKey);
    setPfiKey(restoredResult.pfiKey);
    setActiveHorizon(restoredResult.activeHorizon);
    setCompareTrainingMetricsData(restoredCache.compareTrainingMetricsData);
    setCompareTrainingMetricsKey(restoredCache.compareTrainingMetricsKey);
    setCompareTrainingErrorData(restoredCache.compareTrainingErrorData);
    setCompareTrainingErrorKey(restoredCache.compareTrainingErrorKey);
    setCompareTrainingPfiData(restoredCache.compareTrainingPfiData);
    setCompareTrainingPfiKey(restoredCache.compareTrainingPfiKey);

    const restoredCompareIdKey = restoredCompareIds.join(',');
    previousRequestContextKeyRef.current = `${restoredContextKey}|compare:${restoredCompareIdKey}`;
    restoredScopeRef.current = predictScope;
    setCacheReadyScope(predictScope);
  }, [
    predictScope,
    trainingModelOptions,
    trainingTasksLoaded,
    trainingTasksLoading,
    trainingTasksScope,
  ]);

  useLayoutEffect(() => {
    cacheReadyScopeRef.current = cacheReadyScope === predictScope ? predictScope : null;
  }, [cacheReadyScope, predictScope]);

  // 监听 hash：训练页每次跳转都带着希望进入的模式。
  useEffect(() => {
    const syncRequestedMode = () => setHashRequestedMode(readPredictModeFromHash(window.location.hash));
    window.addEventListener('hashchange', syncRequestedMode);
    syncRequestedMode();
    return () => window.removeEventListener('hashchange', syncRequestedMode);
  }, []);

  /**
   * hash 显式请求的模式优先于预测缓存里的模式：进入 #/predict?mode=trained_compare
   * 与单模型 handoff 都从训练页发起，必须由 URL 决定，而不是上次浏览留下的缓存。
   */
  useEffect(() => {
    if (!hashRequestedMode) return;
    setModelMode(hashRequestedMode);
  }, [hashRequestedMode]);

  useEffect(() => {
    if (modelMode !== PREDICT_MODEL_MODE_TRAINED || trainingTasksLoading || !trainingTasksLoaded) return;
    if (trainingModelOptions.length === 0) {
      setSelectedTrainingTaskId(null);
      return;
    }
    if (!selectedTrainingOption) {
      setSelectedTrainingTaskId(trainingModelOptions[0].id);
    }
  }, [modelMode, selectedTrainingOption, trainingModelOptions, trainingTasksLoaded, trainingTasksLoading]);

  useEffect(() => {
    if (predictionHorizonLimit == null) return;
    setPredStep((current) => clampPredictionHorizon(current, predictionHorizonLimit));
  }, [predictionHorizonLimit]);

  useLayoutEffect(() => {
    if (cacheReadyScopeRef.current !== predictScope) return;
    if (previousRequestContextKeyRef.current === currentPageRequestContextKey) return;
    previousRequestContextKeyRef.current = currentPageRequestContextKey;
    requestCoordinator.invalidateAll();

    setLoading(false);
    setMetricsLoading(false);
    setErrorDistLoading(false);
    setPfiLoading(false);
    setCompareTrainingLoading(false);
    setCompareTrainingErrorLoading(false);
    setCompareTrainingPfiLoading(false);
    setError(null);

    setResultContextKey(null);
    setResults(null);
    setMetrics(null);
    setErrorDistData(null);
    setPfiData(null);
    setPerformanceData(null);
    setPerformanceKey(null);
    setMetricsKey(null);
    setErrorDistKey(null);
    setPfiKey(null);
    setActiveHorizon(0);
    setCompareTrainingMetricsData(null);
    setCompareTrainingMetricsKey(null);
    setCompareTrainingErrorData(null);
    setCompareTrainingErrorKey(null);
    setCompareTrainingPfiData(null);
    setCompareTrainingPfiKey(null);

    writePredictCache({
      resultContextKey: null,
      results: null,
      metrics: null,
      errorDistData: null,
      pfiData: null,
      performanceData: null,
      performanceKey: null,
      metricsKey: null,
      errorDistKey: null,
      pfiKey: null,
      activeHorizon: 0,
      compareTrainingMetricsData: null,
      compareTrainingMetricsKey: null,
      compareTrainingErrorData: null,
      compareTrainingErrorKey: null,
      compareTrainingPfiData: null,
      compareTrainingPfiKey: null,
      selectedCompareTrainingTaskIds: compareSelection.ids,
      params: {
        selectedVars,
        predStep,
        lsStart,
        dataSource: dataSourceMode,
        modelMode,
        trainingTaskId: currentPredictionContext.trainingTaskId,
        compareTrainingTaskIds: compareSelection.ids,
      },
    });
  }, [
    compareSelection.ids,
    currentPageRequestContextKey,
    currentPredictionContext.trainingTaskId,
    dataSourceMode,
    lsStart,
    modelMode,
    predStep,
    requestCoordinator,
    selectedVars,
    predictScope,
    writePredictCache,
  ]);

  useLayoutEffect(() => () => {
    requestCoordinator.invalidateAll();
  }, [requestCoordinator]);

  const handlePredict = useCallback(async () => {
    if (predictionHorizonLimit == null || predStep < 1 || predStep > predictionHorizonLimit) {
      setError(settings?.language !== 'en'
        ? '当前模型没有有效的输出窗口配置。'
        : 'The selected model does not have a valid output horizon.');
      return;
    }
    if (modelMode === PREDICT_MODEL_MODE_COMPARE) {
      const compareTaskIds = compareSelection.ids;
      if (!compareSelection.canCompare) {
        setError(settings?.language !== 'en' ? '至少选择 2 个已完成训练模型。' : 'Select at least 2 completed trained models.');
        return;
      }
      const nextCompareKey = currentCompareTrainingMetricsKey;
      if (!nextCompareKey) {
        setError(settings?.language !== 'en' ? '至少选择 2 个已完成训练模型。' : 'Select at least 2 completed trained models.');
        return;
      }
      setError(null);
      if (nextCompareKey === compareTrainingMetricsKey
        && compareTrainingMetricsData?.items?.length >= 2
        && compareTrainingMetricsData.items.every((item) => item.metrics?.export_ref)) {
        return;
      }
      const requestContextKey = nextCompareKey;
      const requestToken = {
        ...requestCoordinator.start(
        PREDICT_REQUEST_CHANNELS.compareMetrics,
        requestContextKey
        ),
        scope: predictScopeRef.current,
      };
      setCompareTrainingLoading(true);
      try {
        const compareResult = await compareTrainingModels(compareTaskIds, {
          horizon: predStep,
          signal: requestToken.signal,
        });
        if (!isRequestCurrent(requestToken, requestContextKey)) return;
        setCompareTrainingMetricsData(compareResult);
        setCompareTrainingMetricsKey(nextCompareKey);
        writePredictCache({
          compareTrainingMetricsData: compareResult,
          compareTrainingMetricsKey: nextCompareKey,
          selectedCompareTrainingTaskIds: compareTaskIds,
          params: {
            selectedVars,
            predStep,
            lsStart,
            dataSource: dataSourceMode,
            modelMode,
            compareTrainingTaskIds: compareTaskIds,
          },
        });
      } catch (e) {
        if (!isRequestLatest(requestToken, requestContextKey) || isAbortError(e)) return;
        setError(e.message || (settings?.language !== 'en' ? '多模型对比失败。' : 'Training model comparison failed.'));
      } finally {
        if (isRequestCurrent(requestToken, requestContextKey)
          && requestCoordinator.finish(requestToken, requestContextKey)) {
          setCompareTrainingLoading(false);
        }
      }
      return;
    }

    const trainingTaskId = modelMode === PREDICT_MODEL_MODE_TRAINED ? Number(selectedTrainingTaskId) : null;
    if (modelMode === PREDICT_MODEL_MODE_TRAINED && (!Number.isFinite(trainingTaskId) || trainingTaskId <= 0)) {
      setError(settings?.language !== 'en' ? '请先选择一个已完成的训练模型。' : 'Select a completed trained model first.');
      return;
    }

    const body = marsPredictionAdapter({ ozoneUnit, isZh: settings.language !== 'en', horizonLimit: predictionHorizonLimit }).request({
      taskId: trainingTaskId, variables: selectedVars, horizon: predStep, origin: lsStart,
    });
    const analysisContext = {
      modelMode,
      trainingTaskId,
      horizon: predStep,
      selectedVars,
      dataSourceMode,
      lsStart,
    };
    const requestContextKey = buildPredictionContextKey(analysisContext);
    const requestToken = {
      ...requestCoordinator.start(
      PREDICT_REQUEST_CHANNELS.single,
      requestContextKey
      ),
      scope: predictScopeRef.current,
    };
    setError(null);
    setLoading(true);
    const nextMetricsKey = buildPredictMetricsKey(analysisContext);
    const nextErrorDistKey = analysisVisibility.errorDistribution
      ? buildErrorDistributionKey(analysisContext)
      : null;
    const nextPfiKey = analysisVisibility.permutationImportance
      ? buildPermutationImportanceKey(analysisContext)
      : null;
    const shouldFetchMetrics = modelMode === PREDICT_MODEL_MODE_TRAINED
      && Boolean(nextMetricsKey)
      && (nextMetricsKey !== metricsKey || !metrics);
    const shouldFetchErrorDist = Boolean(nextErrorDistKey) && (nextErrorDistKey !== errorDistKey || !errorDistData);
    const shouldFetchPfi = Boolean(nextPfiKey) && (nextPfiKey !== pfiKey || !pfiData?.export_ref);
    const metricsPromise = shouldFetchMetrics
      ? fetchPredictMetrics(body, {
          dataSource: dataSourceMode,
          signal: requestToken.signal,
        })
      : Promise.resolve(null);

    if (shouldFetchMetrics) {
      setMetricsLoading(true);
    }
    if (shouldFetchErrorDist) {
      setErrorDistLoading(true);
    }
    if (shouldFetchPfi) {
      setPfiLoading(true);
    }

    try {
      const [predResult, metricsResult] = await Promise.all([
        runPrediction(body, {
          dataSource: dataSourceMode,
          signal: requestToken.signal,
        }),
        metricsPromise,
      ]);
      if (!isRequestCurrent(requestToken, requestContextKey)) return;
      const resolvedMetrics = selectPredictionMetrics({
        trained: modelMode === PREDICT_MODEL_MODE_TRAINED, fetched: metricsResult, cached: metrics,
        prediction: predResult.metrics, cacheMatches: nextMetricsKey === metricsKey,
      });

      const errorDistPromise = analysisVisibility.errorDistribution
        ? !nextErrorDistKey
          ? Promise.resolve(null)
          : !shouldFetchErrorDist
          ? Promise.resolve(errorDistData)
          : modelMode === PREDICT_MODEL_MODE_TRAINED
          ? fetchErrorDistribution(predResult.selected_variables || [], {
              trainingTaskId,
              horizon: predStep,
              signal: requestToken.signal,
            })
          : fetchErrorDistribution(selectedVars, { signal: requestToken.signal })
        : Promise.resolve(null);
      const pfiVariables = modelMode === PREDICT_MODEL_MODE_TRAINED
        ? (predResult.selected_variables || [])
        : selectedVars;
      const pfiPromise = analysisVisibility.permutationImportance
        ? !shouldFetchPfi
          ? Promise.resolve(pfiData)
          : fetchPermutationImportance(pfiVariables, {
              trainingTaskId,
              lsStart,
              horizon: predStep,
              signal: requestToken.signal,
            })
        : Promise.resolve(null);
      const [errorDistResult, pfiResult] = await Promise.all([errorDistPromise, pfiPromise]);
      if (!isRequestCurrent(requestToken, requestContextKey)) return;
      const nextPerformanceData = modelMode === PREDICT_MODEL_MODE_TRAINED
        ? { results: { current: buildPerformanceMetricsFromEval(resolvedMetrics) } }
        : performanceData;
      const nextPerformanceKey = modelMode === PREDICT_MODEL_MODE_TRAINED
        ? currentPerformanceContextKey
        : performanceKey;

      setResults(predResult);
      setMetrics(resolvedMetrics);
      setErrorDistData(errorDistResult);
      setPfiData(pfiResult);
      setResultContextKey(requestContextKey);
      if (nextMetricsKey) setMetricsKey(nextMetricsKey);
      setErrorDistKey(nextErrorDistKey);
      if (nextPfiKey) setPfiKey(nextPfiKey);
      if (modelMode === PREDICT_MODEL_MODE_TRAINED) {
        setPerformanceData(nextPerformanceData);
        setPerformanceKey(nextPerformanceKey);
      }
      setActiveHorizon(0);
      writePredictCache({
        resultContextKey: requestContextKey,
        results: predResult,
        metrics: resolvedMetrics,
        errorDistData: errorDistResult,
        pfiData: pfiResult,
        metricsKey: nextMetricsKey,
        errorDistKey: nextErrorDistKey,
        pfiKey: nextPfiKey,
        performanceData: nextPerformanceData,
        performanceKey: nextPerformanceKey,
        activeHorizon: 0,
        params: {
          selectedVars,
          predStep,
          lsStart,
          dataSource: dataSourceMode,
          modelMode,
          trainingTaskId,
        },
      });
    } catch (e) {
      if (!isRequestLatest(requestToken, requestContextKey) || isAbortError(e)) return;
      setError(e.message || t('predict.errorPrefix'));
    } finally {
      if (isRequestCurrent(requestToken, requestContextKey)
        && requestCoordinator.finish(requestToken, requestContextKey)) {
        setLoading(false);
        setMetricsLoading(false);
        setErrorDistLoading(false);
        setPfiLoading(false);
      }
    }
  }, [
    analysisVisibility.errorDistribution,
    analysisVisibility.permutationImportance,
    compareSelection,
    compareTrainingMetricsData,
    compareTrainingMetricsKey,
    currentCompareTrainingMetricsKey,
    currentPerformanceContextKey,
    dataSourceMode,
    errorDistData,
    errorDistKey,
    lsStart,
    metrics,
    metricsKey,
    modelMode,
    performanceData,
    performanceKey,
    predictionHorizonLimit,
    predStep,
    pfiData,
    pfiKey,
    requestCoordinator,
    selectedTrainingTaskId,
    selectedVars,
    settings?.language,
    t,
    writePredictCache,
  ]);

  const handleLoadCompareErrorDistribution = useCallback(async () => {
    if (!compareSelection.canCompare || !currentCompareTrainingErrorKey) {
      setError(settings?.language !== 'en' ? '至少选择 2 个已完成训练模型。' : 'Select at least 2 completed trained models.');
      return;
    }
    if (activeCompareTrainingErrorData) return;
    const requestContextKey = currentCompareTrainingErrorKey;
    const requestToken = {
      ...requestCoordinator.start(
      PREDICT_REQUEST_CHANNELS.compareErrorDistribution,
      requestContextKey
      ),
      scope: predictScopeRef.current,
    };
    setError(null);
    setCompareTrainingErrorLoading(true);
    try {
      const result = await compareTrainingModelErrorDistributions(compareSelection.ids, {
        horizon: predStep,
        signal: requestToken.signal,
      });
      if (!isRequestCurrent(requestToken, requestContextKey)) return;
      setCompareTrainingErrorData(result);
      setCompareTrainingErrorKey(requestContextKey);
      writePredictCache({
        compareTrainingErrorData: result,
        compareTrainingErrorKey: requestContextKey,
      });
    } catch (e) {
      if (!isRequestLatest(requestToken, requestContextKey) || isAbortError(e)) return;
      setError(e.message || (settings?.language !== 'en' ? '误差分布对比失败。' : 'Error distribution comparison failed.'));
    } finally {
      if (isRequestCurrent(requestToken, requestContextKey)
        && requestCoordinator.finish(requestToken, requestContextKey)) {
        setCompareTrainingErrorLoading(false);
      }
    }
  }, [
    activeCompareTrainingErrorData,
    compareSelection,
    currentCompareTrainingErrorKey,
    predStep,
    requestCoordinator,
    settings?.language,
    writePredictCache,
  ]);

  const handleLoadComparePfi = useCallback(async () => {
    if (!compareSelection.canCompare || !currentCompareTrainingPfiKey) {
      setError(settings?.language !== 'en' ? '至少选择 2 个已完成训练模型。' : 'Select at least 2 completed trained models.');
      return;
    }
    if (activeCompareTrainingPfiData) return;
    const requestContextKey = currentCompareTrainingPfiKey;
    const requestToken = {
      ...requestCoordinator.start(
      PREDICT_REQUEST_CHANNELS.comparePfi,
      requestContextKey
      ),
      scope: predictScopeRef.current,
    };
    setError(null);
    setCompareTrainingPfiLoading(true);
    try {
      const result = await compareTrainingModelPfi(compareSelection.ids, {
        horizon: predStep,
        signal: requestToken.signal,
      });
      if (!isRequestCurrent(requestToken, requestContextKey)) return;
      setCompareTrainingPfiData(result);
      setCompareTrainingPfiKey(requestContextKey);
      writePredictCache({
        compareTrainingPfiData: result,
        compareTrainingPfiKey: requestContextKey,
      });
    } catch (e) {
      if (!isRequestLatest(requestToken, requestContextKey) || isAbortError(e)) return;
      setError(e.message || (settings?.language !== 'en' ? 'PFI 对比失败。' : 'PFI comparison failed.'));
    } finally {
      if (isRequestCurrent(requestToken, requestContextKey)
        && requestCoordinator.finish(requestToken, requestContextKey)) {
        setCompareTrainingPfiLoading(false);
      }
    }
  }, [
    activeCompareTrainingPfiData,
    compareSelection,
    currentCompareTrainingPfiKey,
    predStep,
    requestCoordinator,
    settings?.language,
    writePredictCache,
  ]);

  useEffect(() => { setPredictUiPreferences({ viewMode }); }, [viewMode]);
  useEffect(() => { writePredictCache({ activeHorizon }); }, [activeHorizon, writePredictCache]);
  useEffect(() => { writePredictCache({ compareConfigs }); }, [compareConfigs, writePredictCache]);
  useEffect(() => { writePredictCache({ selectedCompareIds }); }, [selectedCompareIds, writePredictCache]);
  useEffect(() => {
    writePredictCache({ selectedCompareTrainingTaskIds: compareSelection.ids });
  }, [compareSelectionIdKey, writePredictCache]);

  useEffect(() => {
    if (modelMode !== PREDICT_MODEL_MODE_COMPARE) return;
    setResults(null);
    setMetrics(null);
    setPerformanceData(null);
    setPerformanceKey(null);
    setErrorDistData(null);
    setPfiData(null);
  }, [modelMode]);

  // A mode switch is a useful recovery action after a transient task-directory failure.
  // Only retry when the mode actually changed, so a persistent outage does not loop.
  useEffect(() => {
    const changed = previousModelModeRef.current !== modelMode;
    previousModelModeRef.current = modelMode;
    if (!changed || !trainingTasksError || trainingTasksLoading) return;
    loadTrainingTasks();
  }, [loadTrainingTasks, modelMode, trainingTasksError, trainingTasksLoading]);

  // ── 地球历史预测 ───────────────────────────────────────────────────
  const earthTasks = useMemo(() => trainingTasksScope === predictScope ? trainingTasks : [], [trainingTasksScope, predictScope, trainingTasks]);
  const earthTaskOptions = useMemo(() => getEarthTrainingModelOptions(earthTasks), [earthTasks]);

  const selectedEarthTask = useMemo(
    () => earthTaskOptions.find((option) => String(option.id) === String(earthTaskId)) || null,
    [earthTaskOptions, earthTaskId],
  );

  /**
   * 加载地球预测上下文（可选日期范围、网格、单位、已训练指标）。
   *
   * 没有可选起点时立即提示，并且不发送预测请求；服务器仍会独立复核起点。
   */
  const loadEarthContext = useCallback(async (taskId) => {
    const normalized = Number(taskId);
    if (!Number.isFinite(normalized) || normalized <= 0) {
      setEarthContext(null);
      setEarthContextError(null);
      return null;
    }
    const controller = new AbortController();
    const scope = predictScopeRef.current;
    earthContextRequestRef.current?.abort();
    earthContextRequestRef.current = controller;
    setEarthContextLoading(true);
    earthRunRequestRef.current?.abort();
    setEarthLoading(false);
    setEarthContext(null);
    setEarthResult(null);
    setEarthResultKey(null);
    setEarthResultError(null);
    setEarthDay(0);
    setEarthContextError(null);
    try {
      const payload = await fetchEarthPredictContext(normalized, { signal: controller.signal });
      if (controller.signal.aborted || earthContextRequestRef.current !== controller || scope !== predictScopeRef.current) return;
      if (Number(payload.task_id) !== normalized || payload.planet !== 'earth') throw new Error('Prediction context identity does not match');
      setEarthContext(payload);
      setEarthOrigin((current) => {
        const availableOrigins = Array.isArray(payload?.origins?.timestamps) && payload.origins.timestamps.length
          ? payload.origins.timestamps : (payload?.origins?.dates || []);
        return (
        current && availableOrigins.includes(current)
          ? current
          : pickDefaultOrigin(payload?.origins)
        );
      });
      return payload;
    } catch (requestError) {
      if (controller.signal.aborted) return null;
      setEarthContext(null);
      const resolved = resolveEarthPredictErrorMessage(requestError);
      setEarthContextError(resolved.key ? translateRef.current(`predict.${resolved.key}`) : resolved.fallback);
      return null;
    } finally {
      if (!controller.signal.aborted) setEarthContextLoading(false);
    }
  }, []);

  useEffect(() => {
    if (modelMode !== PREDICT_MODEL_MODE_EARTH) return;
    if (!selectedEarthTask && earthTaskOptions.length > 0) {
      setEarthTaskId(String(earthTaskOptions[0].id));
      return;
    }
    if (!selectedEarthTask) { setEarthTaskId(''); setEarthContext(null); }
    return () => earthContextRequestRef.current?.abort();
  }, [modelMode, earthTaskId, selectedEarthTask, earthTaskOptions]);

  /** 切换模式或任务时丢弃地球结果，避免显示上一个任务/起点的场。 */
  useLayoutEffect(() => {
    if (modelMode === PREDICT_MODEL_MODE_EARTH) return;
    earthContextRequestRef.current?.abort();
    earthRunRequestRef.current?.abort();
    setEarthContext(null);
    setEarthLoading(false);
    setEarthContextLoading(false);
    setEarthResult(null);
    setEarthResultKey(null);
    setEarthResultError(null);
    setEarthDay(0);
  }, [modelMode]);

  const handleEarthTaskChange = useCallback((value) => {
    earthContextRequestRef.current?.abort();
    earthRunRequestRef.current?.abort();
    setEarthLoading(false);
    setEarthTaskId(value);
    setEarthResult(null);
    setEarthResultKey(null);
    setEarthResultError(null);
    setEarthDay(0);
    setEarthContext(null);
    setEarthOrigin('');
  }, []);

  const handleEarthOriginChange = useCallback((value) => {
    earthRunRequestRef.current?.abort();
    setEarthLoading(false);
    setEarthOrigin(value);
    setEarthResult(null);
    setEarthResultKey(null);
    setEarthResultError(null);
    setEarthDay(0);
  }, []);

  /**
   * 运行地球历史预测。
   *
   * 用「星球 + 数据集身份 + 任务 + 起点」作为结果身份：服务端返回的指纹与本地
   * 计算不一致时丢弃迟到响应，避免旧结果覆盖新起点。
   */
  const handleRunEarthPredict = useCallback(async () => {
    const taskId = Number(earthTaskId);
    if (!Number.isFinite(taskId) || taskId <= 0 || earthLoading || earthContextLoading) return;
    const context = earthContext && Number(earthContext.task_id) === taskId
      ? earthContext
      : await loadEarthContext(taskId);
    if (!context) return;
    const adapter = earthPredictionAdapter({ context, isZh: settings.language !== 'en', colormap: settings.colormap });
    const origin = adapter.origin.selectable(earthOrigin) ? earthOrigin : pickDefaultOrigin(context.origins);
    if (!origin) return;
    if (origin !== earthOrigin) setEarthOrigin(origin);
    const requestIdentity = { taskId, origin, context, scope: predictScopeRef.current };
    const controller = new AbortController();
    earthRunRequestRef.current?.abort();
    earthRunRequestRef.current = controller;
    setEarthLoading(true);
    setEarthResultError(null);
    setEarthResult(null);
    setEarthResultKey(null);
    try {
      const payload = await runEarthPrediction(adapter.request({ taskId, origin }), { signal: controller.signal });
      if (controller.signal.aborted || earthRunRequestRef.current !== controller) return;
      if (!earthResponseMatchesRequest(payload, requestIdentity, predictScopeRef.current)) {
        throw new Error(settings.language !== 'en' ? '预测响应身份已变化，请刷新起点范围后重试。' : 'Prediction identity changed. Reload the origin range and retry.');
      }
      const nextKey = buildEarthPredictKey({
        taskId,
        datasetId: payload?.dataset_id,
        datasetVersion: payload?.dataset_version,
        datasetFingerprint: payload?.dataset_fingerprint,
        forecastOrigin: payload?.forecast_origin,
        targetDates: payload?.target_timestamps || payload?.target_dates,
        targetTimestamps: payload?.target_timestamps,
      });
      if (earthRequestRef.current
        && shouldClearEarthResult(earthRequestRef.current, nextKey)) {
        setEarthResult(null);
      }
      earthRequestRef.current = nextKey;
      setEarthResultKey(nextKey);
      setEarthResult(payload);
      setEarthDay(0);
    } catch (requestError) {
      if (controller.signal.aborted) return;
      const resolved = resolveEarthPredictErrorMessage(requestError);
      setEarthResultError(resolved.key ? t(`predict.${resolved.key}`) : resolved.fallback);
    } finally {
      if (!controller.signal.aborted) setEarthLoading(false);
    }
  }, [earthContext, earthContextLoading, earthLoading, earthOrigin, earthTaskId, loadEarthContext, settings.language, settings.colormap, t]);

  /**
   * Earth 只保留侧栏行星契约需要的文案。单模型页面其余部分的措辞与火星一致，
   * 由共用侧栏和共用工作台直接提供，不再单独维护一套介绍性词表。
   */
  const earthCopy = useMemo(() => ({
    taskLabel: t('predict.earthTaskLabel'),
    taskPlaceholder: t('predict.earthTaskPlaceholder'),
    originLabel: t('predict.earthOriginLabel'),
    originOutOfRange: t('predict.earthOriginOutOfRange'),
    emptyNoTask: t('predict.earthNoCompletedTask'),
  }), [t]);

  const isEarthMode = modelMode === PREDICT_MODEL_MODE_EARTH;
  const isEarthCompareMode = modelMode === PREDICT_MODEL_MODE_EARTH_COMPARE;
  const earthAdapter = useMemo(() => earthPredictionAdapter({ context: earthContext, isZh: settings.language !== 'en', colormap: settings.colormap }),
    [earthContext, settings.language, settings.colormap]);
  const marsAdapter = useMemo(() => marsPredictionAdapter({ ozoneUnit, isZh: settings.language !== 'en', horizonLimit: predictionHorizonLimit }),
    [ozoneUnit, settings.language, predictionHorizonLimit]);
  useEffect(() => () => {
    earthContextRequestRef.current?.abort();
    earthRunRequestRef.current?.abort();
  }, [modelMode, predictScope]);

  const step = activeResults ? Math.min(activeHorizon, (activeResults.horizon || 1) - 1) : 0;

  return (
    <div className="page-enter prediction-page">
      <SectionTitle title={t('predict.title')} subtitle={t('predict.subtitle')} />
      <PredictModeSelector mode={modelMode} onChange={setModelMode} disabled={requestContextLocked} />

      {hashRequestedMode === PREDICT_MODEL_MODE_COMPARE && modelMode === PREDICT_MODEL_MODE_COMPARE ? (
        <div className="predict-mode-hint" role="status">
          {t('experimentCenter.compareModeHint')}
        </div>
      ) : null}

      {isEarthCompareMode ? <EarthCompareWorkspace key={predictScope} tasks={earthTasks} scope={predictScope}
        tasksLoading={trainingTasksLoading} tasksError={trainingTasksError} onRetryTasks={loadTrainingTasks}
        isLight={isLight} precision={precision} isZh={settings?.language !== 'en'}
        plotTextColor={plotTextColor} plotGridColor={plotGridColor} /> : null}
      <div className={isEarthCompareMode ? undefined : 'predict-workspace'}>
        {!isEarthCompareMode ? (
        <PredictSidebar
          adapter={isEarthMode ? earthAdapter : marsAdapter}
          originValue={isEarthMode ? earthOrigin : lsStart}
          onOriginChange={isEarthMode ? handleEarthOriginChange : setLsStart}
          originDisabled={isEarthMode && !earthContext}
          originHint={isEarthMode && earthOrigin && earthContext && !earthAdapter.origin.selectable(earthOrigin)
            ? earthCopy.originOutOfRange : null}
          contextLoading={isEarthMode && earthContextLoading}
          contextError={isEarthMode ? earthContextError : null}
          deferContext={isEarthMode}
          runDisabled={isEarthMode && (!earthTaskId || (earthContext && earthOrigin && !earthAdapter.origin.selectable(earthOrigin)))}
          isLight={isLight}
          loading={isEarthMode ? (earthLoading || earthContextLoading) : modelMode === PREDICT_MODEL_MODE_COMPARE ? compareTrainingLoading : loading}
          requestContextLocked={requestContextLocked}
          error={isEarthMode ? earthResultError : activeError}
          modelMode={isEarthMode ? PREDICT_MODEL_MODE_TRAINED : modelMode}
          trainingModelOptions={isEarthMode ? earthTaskOptions : trainingModelOptions}
          selectedTrainingTaskId={isEarthMode ? earthTaskId : selectedTrainingTaskId}
          setSelectedTrainingTaskId={isEarthMode ? handleEarthTaskChange : setSelectedTrainingTaskId}
          selectedCompareTrainingTaskIds={selectedCompareTrainingTaskIds}
          setSelectedCompareTrainingTaskIds={setSelectedCompareTrainingTaskIds}
          trainingTasksLoading={trainingTasksLoading}
           trainingTasksError={trainingTasksError}
           onRetryTrainingTasks={loadTrainingTasks}
          selectedTrainingOption={isEarthMode ? selectedEarthTask : selectedTrainingOption}
          analysisVisibility={analysisVisibility}
          lsStart={lsStart}
          setLsStart={setLsStart}
          predStep={isEarthMode ? earthContext?.horizon ?? '' : predStep}
          setPredStep={setPredStep}
          predictionHorizonLimit={isEarthMode ? earthContext?.horizon ?? null : predictionHorizonLimit}
          selectedVars={selectedVars}
          toggleVar={toggleVar}
          VARIABLES={VARIABLES}
          handlePredict={isEarthMode ? handleRunEarthPredict : handlePredict}
          precision={precision}
        />
        ) : null}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Earth only assembles context and adaptation; single-model display is shared. */}
          {isEarthMode ? (
            <>
              {earthTaskOptions.length === 0 ? (
                <div className="predict-mode-hint" role="status" data-earth-no-task-hint="true">
                  {trainingTasksError
                    ? (settings?.language === 'en' ? 'Training models could not be loaded. Retry to try again.' : '训练任务加载失败，请重试。')
                    : t('predict.earthNoCompletedTask')}
                </div>
              ) : null}
              <EarthPredictPanel
                key={`${predictScope}:earth:${earthTaskId}`}
                adapter={earthAdapter}
                identityKey={`${predictScope}:${earthTaskId}:${earthOrigin}:${earthResultKey}`}
                identityScope={predictScope}
                context={earthContext}                result={earthResult}
                loading={earthLoading || earthContextLoading}
                selectedDay={earthDay}
                onSelectDay={setEarthDay}
                taskOptions={earthTaskOptions}
                selectedTaskId={earthContext ? earthTaskId : null}
                modelsLoading={trainingTasksLoading}
                modelsError={trainingTasksError}
                onRetryModels={loadTrainingTasks}
              />
            </>
          ) : null}

          {analysisVisibility.predictionFields ? (
          <SingleModelWorkbench key={`${predictScope}:mars:${selectedTrainingTaskId}`}
            adapter={marsAdapter} result={activeResults} metrics={activeMetrics} metricsLoading={metricsLoading}
            hasAvailableModels={trainingModelOptions.length > 0} modelsLoading={trainingTasksLoading}
            modelsError={trainingTasksError} onRetryModels={loadTrainingTasks}
            viewMode={viewMode} setViewMode={setViewMode}
            fullscreen={fullscreen3D} onFullscreenChange={setFullscreen3D}
            loading={loading} activeStep={activeHorizon} onStepChange={setActiveHorizon}
            identityKey={currentPageRequestContextKey}>

          {analysisVisibility.errorDistribution ? (
            <ErrorDistributionChart
              predictionResult={activeResults}
              step={step}
              data={activeErrorDistData}
              loading={errorDistLoading}
              isLight={isLight}
              plotTextColor={plotTextColor}
              plotText60={plotText60}
              plotGridColor={plotGridColor}
              presentation={marsAdapter.presentation.distribution}
            />
          ) : null}

          {analysisVisibility.permutationImportance ? (
            <PermutationImportanceChart
              data={activePfiData}
              loading={pfiLoading}
              plotTextColor={plotTextColor}
              plotText60={plotText60}
              plotGridColor={plotGridColor}
              presentation={marsAdapter.presentation.pfi}
            />
          ) : null}
          </SingleModelWorkbench>
          ) : null}

          {analysisVisibility.compareSummary ? (
            <CompareTrainingModelsPanel
              data={activeCompareTrainingData}
              loading={compareTrainingLoading}
              errorDistributionData={activeCompareTrainingErrorData}
              errorDistributionLoading={compareTrainingErrorLoading}
              onLoadErrorDistribution={handleLoadCompareErrorDistribution}
              pfiData={activeCompareTrainingPfiData}
              pfiLoading={compareTrainingPfiLoading}
              onLoadPfi={handleLoadComparePfi}
              selectedCount={compareSelection.count}
              precision={precision}
              isZh={settings?.language !== 'en'}
              plotTextColor={plotTextColor}
              plotGridColor={plotGridColor}
            />
          ) : null}
        </div>
      </div>

    </div>
  );
}
