import { useMemo, useState } from 'react';
import { Panel } from '../../components/ui/Controls';
import { useT } from '../../i18n';
import PredictMetrics from './PredictMetrics.jsx';
import ErrorDistributionChart from './ErrorDistributionChart.jsx';
import PermutationImportanceChart from './PermutationImportanceChart.jsx';
import { createEarthPredictionPresentation } from './predictionPresentation.js';
import { fmtNum } from '../../utils/fmt.js';
import { useSettings } from '../../contexts/SettingsContext.jsx';
import useEarthDiagnostics from './useEarthDiagnostics.js';
import { earthDiagnosticPfiItems } from './earthDiagnosticsModel.js';
import './earthDiagnostics.css';

const jsonLabel = (item) => typeof item === 'string' ? item : item == null ? '—' : JSON.stringify(item);
const timeLabel = (range) => range ? `${range.target_start || range.date_start || range.start || '—'} → ${range.target_end || range.date_end || range.end || '—'}` : '—';

/**
 * 地球训练后诊断：手动触发的 test 分区诊断。
 * 文案走共用的 `predict.earthDiagnostics.*`，标题层级使用共用图表卡样式，
 * 区块顺序与火星的诊断卡片一致；触发方式仍是独立手动，不随预测自动运行。
 */
export default function EarthDiagnosticPanel({ taskId, identity, autoLoad = false }) {
  const t = useT();
  const { settings } = useSettings();
  const d = t('predict.earthDiagnostics');
  const value = (number) => Number.isFinite(number) ? fmtNum(number, settings.precision) : '—';
  const presentation = useMemo(() => createEarthPredictionPresentation({ settings }), [settings]);
  const fullPresentation = useMemo(() => createEarthPredictionPresentation({ settings, scope: 'full_test' }).metrics, [settings]);
  const samplePresentation = useMemo(() => createEarthPredictionPresentation({ settings, scope: 'sampled_diagnostic' }).metrics, [settings]);
  const [windows, setWindows] = useState(4);
  const [repeats, setRepeats] = useState(3);
  const options = useMemo(() => ({ sample_windows: windows, pfi_repeats: repeats }), [windows, repeats]);
  const { status, data, error, run } = useEarthDiagnostics(taskId, { identity, options, autoLoad });
  const busy = status === 'checking' || status === 'computing';
  const scope = data?.scope;
  const full = data?.full_test_metrics;
  const fullMetrics = full?.metrics?.overall;
  const pfi = data?.pfi;
  const timeRange = scope?.time_range;
  const rangeText = timeRange ? [timeRange.start || timeRange.input_start || timeRange.target_start,
    timeRange.end || timeRange.target_end].filter(Boolean).join(' → ') || jsonLabel(timeRange) : '—';
  const pfiItems = earthDiagnosticPfiItems(pfi);
  const sampleMetrics = data?.sample_metrics?.overall;

  return <Panel className="prediction-panel">
    <section className="earth-diagnostics" data-earth-diagnostics="true" aria-label={d.title}>
      <header className="prediction-chart-header">
        <div><h3 className="prediction-chart-title">{d.title}</h3>
          <p className="prediction-chart-subtitle">{d.subtitle}</p></div>
        <strong className="earth-diagnostic-unit">DU</strong>
      </header>
      <div className="earth-diagnostic-controls">
        <label>{d.windowCount}
          <input type="number" min="1" max="8" value={windows} disabled={busy}
            onChange={(event) => { const n = Number(event.target.value); if (Number.isInteger(n) && n >= 1 && n <= 8) setWindows(n); }} /></label>
        <label>{d.pfiRepeats}
          <input type="number" min="1" max="5" value={repeats} disabled={busy}
            onChange={(event) => { const n = Number(event.target.value); if (Number.isInteger(n) && n >= 1 && n <= 5) setRepeats(n); }} /></label>
        <button type="button" disabled={!taskId || busy || status === 'unavailable'} onClick={run}>
          {busy ? (status === 'checking' ? d.checking : d.computing)
            : (status === 'error' || data ? d.recompute : d.compute)}
        </button>
        {status === 'unavailable' ? <button type="button" onClick={run}>{d.recheck}</button> : null}
      </div>
      <p className="earth-diagnostic-note">{d.defaults}</p>
      {busy ? <p role="status" className="earth-diagnostic-status">{d.busy}</p> : null}
      {error ? <p role="alert" className="earth-diagnostic-status" data-tone="error">{error}</p> : null}
      {!taskId ? <p className="earth-diagnostic-status">{d.selectTask}</p> : null}
      {status === 'ready' ? <p className="earth-diagnostic-status">{d.ready}</p> : null}
      {data ? <>
        <h3 className="prediction-chart-title prediction-chart-title--sub">{d.fullTestTitle}</h3>
        <p className="earth-diagnostic-note">{full?.available === false || !fullMetrics
          ? d.fullTestMissing : d.fullTestReused}</p>
        <p className="earth-diagnostic-note">{d.fullTestRange}{timeLabel(full?.test_range)} · {full?.metrics?.window_count ?? full?.test_range?.window_count ?? '—'} {d.windowsSuffix} · {d.partitionFixed}</p>
        <PredictMetrics metrics={full?.metrics ? { ...full.metrics, overall: fullMetrics,
          aggregation: full.aggregation, metric_policy: full.metric_policy, schema: full.schema } : null}
          precision={settings.precision} presentation={fullPresentation} flattened />
        <h3 className="prediction-chart-title prediction-chart-title--sub">{d.samplingTitle}</h3>
        <dl className="earth-diagnostic-facts">
          <div><dt>{d.partitionWindows}</dt><dd>{`${scope?.split || 'test'} · ${scope?.window_count ?? '—'} / ${scope?.available_window_count ?? '—'}`}</dd></div>
          <div><dt>{d.utcRange}</dt><dd>{rangeText}</dd></div>
          <div><dt>{d.spatialCoverage}</dt><dd>{`${d.globalGrid} · ${scope?.grid_shape?.join(' × ') || '—'} · ${d.latitude} ${scope?.spatial_coverage?.latitude_range?.join(' → ') || '—'}° · ${d.longitude} ${scope?.spatial_coverage?.longitude_range?.join(' → ') || '—'}°`}</dd></div>
          <div><dt>{d.validPairs}</dt><dd>{scope?.valid_points?.toLocaleString() ?? '—'}</dd></div>
          <div><dt>{d.samplingStrategy}</dt><dd>{d.samplingStrategyValue} · seed {scope?.sampling?.seed ?? data.parameters?.seed ?? 42}</dd></div>
          <div><dt>{d.cacheAndAlgorithm}</dt><dd>{`${data.cache?.hit ? d.cacheHit : d.computed} · ${data.cache?.algorithm_version || '—'}`}</dd></div>
        </dl>
        {scope?.windows?.length ? <details className="earth-diagnostic-note"><summary>{d.inspectWindows}</summary>
          <ul>{scope.windows.map((window) => <li key={window.test_window_index}>{`#${window.test_window_index} · ${d.windowOrigin} ${window.forecast_origin} · ${d.windowInput} ${window.input_start} → ${window.input_end} · ${d.windowTarget} ${window.target_start} → ${window.target_end}`}</li>)}</ul>
        </details> : null}
        {data.warnings?.length ? <div className="earth-diagnostic-status" role="status">{data.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div> : null}
        <h3 className="prediction-chart-title prediction-chart-title--sub">{d.sampleMetricsTitle}</h3>
        <p className="earth-diagnostic-note">{d.sampleMetricsNote}</p>
        <PredictMetrics metrics={data.sample_metrics ? { ...data.sample_metrics, overall: sampleMetrics } : null}
          precision={settings.precision} presentation={samplePresentation} flattened />
        <ErrorDistributionChart data={data} presentation={presentation.distribution} precision={settings.precision} />
        {/* The shared PFI card renders its own title, scope and help text; per-channel
            unavailability is reported in the table below rather than repeated here. */}
        {pfiItems.length ? <>
          <PermutationImportanceChart data={pfi} presentation={presentation.pfi} precision={settings.precision} />
          <p className="earth-diagnostic-note">{pfi.window_count ?? pfi.scope?.window_count ?? '—'} {d.pfiScopeSuffix} {timeLabel(pfi.scope?.time_range)} · seed {pfi.seed ?? 42}</p>
          <div className="earth-diagnostic-table-scroll"><table>
            <thead><tr>{[d.pfiChannel, d.pfiBaseline, d.pfiMean, d.pfiEachRmse, d.pfiEachDelta].map((text) => <th key={text}>{text}</th>)}</tr></thead>
            <tbody>{pfiItems.map((item) => <tr key={item.channel}><th>{item.channel}</th><td>{value(item.baseline_rmse)}</td><td>{Number.isFinite(item.importance_mean) ? `${value(item.importance_mean)} ± ${value(item.importance_std)}` : jsonLabel(item.reason || d.pfiNotComputable)}</td><td>{item.repeats.map((repeat) => typeof repeat === 'number' ? value(repeat) : value(repeat.permuted_rmse ?? repeat.rmse)).join(', ') || '—'}</td><td>{item.repeats.map((repeat) => value(typeof repeat === 'number' ? repeat - item.baseline_rmse : repeat.delta_rmse ?? repeat.importance)).join(', ') || '—'}</td></tr>)}</tbody>
          </table></div>
        </> : <p className="earth-diagnostic-status">{jsonLabel(pfi?.reason?.message || pfi?.reason || pfi?.message || d.pfiUnavailable)}</p>}
        <p className="earth-diagnostic-note">{d.exportNote}</p>
      </> : null}
    </section>
  </Panel>;
}
