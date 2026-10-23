import { useMemo } from 'react';
import { useSettings } from '../../contexts/SettingsContext';
import SingleModelWorkbench from './SingleModelWorkbench';
import ForecastMetricDetails from './ForecastMetricDetails';
import EarthDiagnosticPanel from './EarthDiagnosticPanel';
import PredictStatus from './PredictStatus';
import { buildEarthSplitLabelSource } from './earthPredictModel';
import { createEarthPredictionPresentation } from './predictionPresentation';

/**
 * Earth keeps only its own context identity, result adaptation and manual
 * diagnostics. Every block that Mars shows — panels, steps, metrics, charts and
 * the fullscreen shell — comes from the shared workbench, so this component
 * adds no introduction panel of its own.
 */
export default function EarthPredictPanel({ adapter, context, result, loading, selectedDay, onSelectDay,
  selectedTaskId, taskOptions, identityKey, identityScope, modelsLoading = false,
  modelsError = null, onRetryModels }) {
  const { settings } = useSettings();
  const isZh = settings.language !== 'en';
  const colormap = settings.colormap;
  const normalized = useMemo(() => adapter.normalizeResult(result), [adapter, result]);
  const warnings = result?.warnings?.length ? result.warnings : context?.warnings;
  /**
   * 完整测试集分区沿用火星的 `split_meta` 形状：Earth 契约把边界放在 split_ranges /
   * task_split.ranges，这里整理成共用 `getSplitLabel` 能读的结构，读不到就不显示该行。
   */
  const workbenchAdapter = useMemo(() => {
    const splitLabelSource = buildEarthSplitLabelSource({
      split_policy: context?.metrics?.split_policy,
      split_ranges: context?.metrics?.split_ranges,
      split_ratios: context?.metrics?.split_ratios,
      task_split: context?.metrics?.task_split,
    })?.split_meta || null;
    return { ...adapter,
      presentation: { ...adapter.presentation,
        metrics: createEarthPredictionPresentation({ isZh, colormap, splitLabelSource }).metrics },
      secondaryMetrics: context ? {
        data: context.metrics?.splits?.test,
        presentation: createEarthPredictionPresentation({ isZh, colormap, scope: 'full_test', splitLabelSource }).metrics,
      } : null };
  }, [adapter, colormap, context, isZh]);
  return <div className="earth-predict-panel" data-earth-predict-panel="true">
    {Array.isArray(warnings) && warnings.length ? <PredictStatus message={warnings.join(' · ')} /> : null}
    <SingleModelWorkbench adapter={workbenchAdapter} result={normalized} metrics={result?.metrics} loading={loading}
      activeStep={selectedDay} onStepChange={onSelectDay} identityKey={identityKey}
      hasAvailableModels={taskOptions.length > 0} modelsLoading={modelsLoading}
      modelsError={modelsError} onRetryModels={onRetryModels}>
      {result ? <ForecastMetricDetails result={result} adapter={adapter} precision={settings.precision} /> : null}
      <EarthDiagnosticPanel key={`${identityScope}:${selectedTaskId || 'no-task'}`} taskId={selectedTaskId}
        identity={{ ...(taskOptions.find(option => Number(option.id) === Number(selectedTaskId))?.task || {}), ...(context || {}) }} />
    </SingleModelWorkbench>
  </div>;
}
