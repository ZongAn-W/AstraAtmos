import { predictionFieldGrid } from './predictionFieldGrid.js';
import { buildEarthFieldPayload, earthLeadLabel, earthTimestampList, getEarthCadence, isOriginSelectable, resolveEarthColorRanges } from './earthPredictModel.js';
import { clampPredictionHorizon } from './predictionHorizon.js';
import { environmentDisplay, environmentIdentityMetadata, resolveEnvironmentTask } from '../../utils/environmentVariables.js';

const identity = value => value;
const fieldTitle = (kind, isZh) => ({
  truth: isZh ? '参考' : 'Reference', prediction: isZh ? '预测' : 'Prediction', residual: isZh ? '残差' : 'Residual',
}[kind]);

export function utcInputValue(value) { return String(value || '').replace(/Z$/, '').slice(0, 16); }
export function utcOriginValue(value) { return value ? `${value.length === 16 ? `${value}:00` : value}Z` : ''; }

/** Reject a response against the request's frozen identity, before displaying or exporting it. */
export function earthResponseMatchesRequest(response, { taskId, origin, context, scope }, currentScope) {
  const sameAxis = axis => JSON.stringify(response?.grid?.[axis]) === JSON.stringify(context?.grid?.[axis]);
  const fieldsValid = ['reference', 'prediction', 'residual'].every(kind => Array.isArray(response?.[kind])
    && response[kind].length === context?.horizon && response[kind].every((_, dayIndex) => Boolean(
      buildEarthFieldPayload({ response, dayIndex, kind, colormap: 'inferno' }))));
  const sameModel = ['model_source', 'model_architecture', 'uploaded_model_id', 'uploaded_model_version', 'uploaded_model_content_hash']
    .every(key => (response?.model?.[key] ?? null) === (context?.model?.[key] ?? null));
  const responseEnvironment = resolveEnvironmentTask(response, 'earth');
  const contextEnvironment = resolveEnvironmentTask(context, 'earth');
  const contextIdentity = environmentIdentityMetadata(context);
  const responseIdentity = environmentIdentityMetadata(response);
  const sameInputs = JSON.stringify(responseEnvironment.inputs) === JSON.stringify(contextEnvironment.inputs);
  return Boolean(scope && scope === currentScope && context
    && Number(response?.task_id) === Number(taskId)
    && response?.forecast_origin === origin
    && response?.planet === 'earth'
    && responseEnvironment.target === contextEnvironment.target
    && responseEnvironment.unit === contextEnvironment.unit
    && (!contextIdentity.complete || responseIdentity.complete)
    && sameInputs
    && sameAxis('latitude') && sameAxis('longitude') && fieldsValid && sameModel
    && response?.dataset_id === context.dataset_id
    && response?.dataset_version === context.dataset_version
    && response?.dataset_fingerprint === context.dataset_fingerprint
    && response?.horizon === context.horizon
    && response?.window === context.window);
}

/** Validate Mars result metadata against the selected task. Only fully metadata-free legacy O3
 * responses remain readable; partial identities cannot establish a trustworthy contract. */
export function marsResponseMatchesTask(response, { task, selectedVariables = [] } = {}) {
  if (!task || !response) return false;
  const expected = resolveEnvironmentTask(task, 'mars');
  const expectedIdentity = environmentIdentityMetadata(task);
  const actualIdentity = environmentIdentityMetadata(response);
  if ((expectedIdentity.declared || actualIdentity.declared) && !actualIdentity.complete) return false;
  if (!expectedIdentity.declared && !actualIdentity.declared) return expected.target === 'O3';
  const actual = resolveEnvironmentTask(response, 'mars');
  const expectedInputs = JSON.stringify(expected.inputs);
  const actualInputs = JSON.stringify(actual.inputs);
  const requestedVariables = Array.isArray(selectedVariables) ? selectedVariables : [];
  return actual.target === expected.target
    && actual.unit === expected.unit
    && actualInputs === expectedInputs
    && (!requestedVariables.length || requestedVariables.every((value) => expected.inputs.includes(value)));
}

export function normalizeEarthPrediction(response, colormap = 'inferno') {
  if (!response) return null;
  const normalize = kind => (response[kind] || []).map((day, dayIndex) => {
    const ranges = resolveEarthColorRanges(response, dayIndex);
    const payload = buildEarthFieldPayload({ response, dayIndex, kind,
      colormap: kind === 'residual' ? 'rdbu' : colormap,
      colorRange: kind === 'residual' ? ranges.residual : ranges.physical });
    return payload ? { ...day, lat: payload.lat, lon: payload.lon, payload } : null;
  });
  return { ...response, ground_truth: normalize('reference'), prediction: normalize('prediction'), residual: normalize('residual') };
}

export function createEarthPredictionAdapter({ context, isZh = true, colormap = 'inferno' } = {}) {
  const cadence = getEarthCadence(context);
  const target = environmentDisplay(context, 'earth');
  return {
    id: 'earth', ...target, grid: predictionFieldGrid, fieldTitle: (kind, zh) => `${fieldTitle(kind, zh)} · ${target[zh ? 'zh' : 'en']}`,
    capabilities: { triptychExport: true, scatterExport: false, diagnosticExport: false, manualDiagnostics: true, fullscreen: 'earth-map' },
    origin: { kind: cadence.threeHourly ? 'datetime-local' : 'date', label: isZh ? '预测起点（UTC）' : 'Forecast origin (UTC)',
      toInput: cadence.threeHourly ? utcInputValue : identity, fromInput: cadence.threeHourly ? utcOriginValue : identity,
      min: context?.origins?.start, max: context?.origins?.end, step: cadence.threeHourly ? 10800 : undefined,
      selectable: value => isOriginSelectable(context?.origins, value) },
    horizon: { editable: false, limit: context?.horizon ?? null,
      label: context ? `${context.horizon} ${isZh ? '步' : 'steps'} · ${context.horizon * cadence.frequencyHours}h` : '—',
      hint: isZh ? '每次运行完整模型输出窗口；展示步只切换查看内容。' : 'Run the full model horizon; the display step only changes the viewed field.' },
    request: ({ taskId, origin }) => ({ trainingTaskId: Number(taskId), forecastOrigin: origin }),
    normalizeResult: response => normalizeEarthPrediction(response, colormap),
    stepLabel: (response, index) => `${earthLeadLabel(index, response)} · ${earthTimestampList(response)[index] || ''} UTC`,
    fullscreenLabel: isZh ? '地球地图 · 放大查看' : 'Earth map · expanded view',
    fullscreenButtonLabel: isZh ? '全屏查看' : 'View fullscreen',
    emptyTitle: isZh ? '单模型预测工作台' : 'Single-model prediction workbench',
    emptyDescription: isZh ? '选择地球训练模型与 UTC 起点，运行完整输出窗口。' : 'Choose an Earth model and UTC origin to run the full horizon.',
  };
}

export function createMarsPredictionAdapter({ ozoneUnit = 'DU', isZh = true, horizonLimit = null, task = null } = {}) {
  const target = environmentDisplay(task, 'mars', ozoneUnit);
  return {
    id: 'mars', ...target, grid: predictionFieldGrid, fieldTitle: (kind, zh) => `${fieldTitle(kind, zh)} · ${target[zh ? 'zh' : 'en']}`,
    capabilities: { triptychExport: true, scatterExport: true, diagnosticExport: true, manualDiagnostics: false, fullscreen: 'mars-sphere' },
    origin: { kind: 'range', label: isZh ? '起始太阳黄经 Ls' : 'Starting solar longitude Ls', min: 0, max: 355, step: 1,
      toInput: identity, fromInput: Number, selectable: value => Number.isFinite(value) && value >= 0 && value <= 355 },
    horizon: { editable: true, limit: horizonLimit, clamp: value => clampPredictionHorizon(value, horizonLimit),
      hint: isZh ? `当前最大输出窗口：${horizonLimit ?? '—'}` : `Current maximum: ${horizonLimit ?? '—'}` },
    request: ({ taskId, origin, horizon, variables }) => ({ selected_variables: variables, horizon,
      ls_start: origin, ...(taskId ? { training_task_id: Number(taskId) } : {}) }),
    normalizeResult: identity,
    stepLabel: (response, index) => `${isZh ? '第' : 'Step '}${index + 1}${isZh ? '步' : ''}${response?.ls_values?.[index] != null ? ` · Ls=${response.ls_values[index].toFixed(3)}°` : ''}`,
    fullscreenLabel: isZh ? '火星球面 · 放大查看' : 'Mars sphere · expanded view',
    fullscreenButtonLabel: isZh ? '全屏查看' : 'View fullscreen',
  };
}
