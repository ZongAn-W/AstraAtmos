import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import C from '../../../constants/colors';
import { Button, Panel, SegmentedControl } from '../../../components/ui/Controls';
import { fmtNum } from '../../../utils/fmt';
import { METRIC_META } from '../PredictComponents';
import {
  buildErrorHistogramTraces,
  buildCompareParameterRows,
  buildPfiMatrix,
  buildStepCurveTraces,
  normalizeCompareDataSource,
  sortCompareItems,
  metricValue,
  resolveEarthCompareMetricUnit,
} from './compareTrainingModelsData';
import { EARTH_METRIC_META } from '../../../utils/earthMetricMeta.js';
import { getMetricAggregationLabel, getSplitLabel } from '../predictionMetricMeta';
import ResearchExportButton from '../ResearchExportButton';
import { useSettings } from '../../../contexts/SettingsContext';
import { convertCompareMetrics } from '../researchExportModel';
import { environmentDisplay, sameEnvironmentTarget, resolveEnvironmentTask } from '../../../utils/environmentVariables.js';

const GOOD_METRICS = new Set(['r2', 'ssim']);

function metricLabel(metric, unit) {
  const valueUnit = metric.key === 'mse' ? (metric.unit || `${unit}²`) : ['rmse', 'mae'].includes(metric.key) ? unit : (metric.unit === '1' ? '' : metric.unit);
  return metric.name + (valueUnit ? ` (${valueUnit})` : '');
}

function withAlpha(color, alpha) {
  if (!color) return `rgba(255,255,255,${alpha})`;
  if (color.startsWith('#')) {
    let hex = color.slice(1);
    if (hex.length === 3) hex = hex.split('').map((char) => char + char).join('');
    const value = Number.parseInt(hex, 16);
    return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
  }
  return color;
}

function MetricTabs({ activeMetric, setActiveMetric, metrics = METRIC_META }) {
  return (
    <SegmentedControl value={activeMetric} onChange={setActiveMetric} className="prediction-compare__metric-tabs"
      options={metrics.map(metric => ({ value: metric.key, label: metric.name,
        description: metric.zh ? `${metric.zh} / ${metric.en}` : metric.name }))} />
  );
}

function LoadingState({ isZh }) {
  return (
    <Panel className="prediction-panel">
      <div style={{ minHeight: 180, display: 'grid', placeItems: 'center', gap: 12 }}>
        <div style={{ width: 28, height: 28, border: `3px solid ${C.border}`, borderTop: `3px solid ${C.green}`, borderRadius: '50%', animation: 'spin-slow 0.9s linear infinite' }} />
        <div style={{ color: C.ice60, fontSize: 'calc(12px * var(--font-scale, 1))' }}>
          {isZh ? '正在计算多模型测试集指标…' : 'Computing multi-model test-set metrics...'}
        </div>
      </div>
    </Panel>
  );
}

function EmptyState({ isZh, selectedCount }) {
  return (
    <Panel className="prediction-panel">
      <div style={{ padding: '16px 4px', textAlign: 'center' }}>
        <div style={{ color: 'var(--text-primary)', fontSize: 'calc(var(--type-control) * var(--font-scale, 1))', fontWeight: 600 }}>
          {selectedCount >= 2
            ? (isZh ? '点击左侧“开始对比”生成结果' : 'Start comparison from the sidebar')
            : (isZh ? '至少选择 2 个训练模型' : 'Select at least 2 trained models')}
        </div>
        <div style={{ color: 'var(--text-secondary)', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', lineHeight: 1.5, marginTop: 8 }}>
          {isZh
            ? '对比结果会基于完整测试集指标，而不是当前预测窗口。'
            : 'Comparison results use full test-set metrics rather than the current prediction window.'}
        </div>
      </div>
    </Panel>
  );
}

function SummaryTable({ items, precision, isZh, unit, metrics, planet }) {
  const [sortMetric, setSortMetric] = useState('rmse');
  const [direction, setDirection] = useState('asc');
  const sorted = useMemo(() => sortCompareItems(items, { metric: sortMetric, direction }), [direction, items, sortMetric]);
  const columns = [
    { key: 'model', label: isZh ? '模型名' : 'Model' },
    ...metrics.map(metric => ({ key: metric.key, label: metricLabel(metric, unit) })),
    { key: 'architecture', label: isZh ? '架构' : 'Architecture' },
    { key: 'channels', label: isZh ? '输入通道' : 'Channels' },
    { key: 'window', label: 'Window' },
    { key: 'horizon', label: 'Horizon' },
    { key: 'source', label: isZh ? '数据源' : 'Data source' },
  ];

  const setSort = (metric) => {
    if (sortMetric === metric) {
      setDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
      return;
    }
    setSortMetric(metric);
    setDirection(GOOD_METRICS.has(metric) ? 'desc' : 'asc');
  };

  return (
    <Panel className="prediction-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 14, alignItems: 'flex-start', marginBottom: 14, flexWrap: 'wrap' }}>
        <div>
          <div className="prediction-chart-title">
            {isZh ? '多模型综合排名' : 'Multi-model ranking'}
          </div>
          <div style={{ color: C.ice50, fontSize: 'calc(11px * var(--font-scale, 1))', lineHeight: 1.55, marginTop: 5 }}>
            {isZh
              ? '整体指标使用完整测试集；默认按 RMSE 升序，支持按已提供的指标排序。'
              : 'Overall metrics summarize the full test set. Sort by RMSE ascending or another available metric.'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {metrics.map((metric) => (
            <Button size="compact"
              key={metric.key}
              type="button"
              onClick={() => setSort(metric.key)}
              aria-pressed={sortMetric === metric.key}
              style={{ background: sortMetric === metric.key ? 'var(--surface-3)' : 'transparent',
                color: sortMetric === metric.key ? 'var(--brand-ice)' : 'var(--text-secondary)',
                boxShadow: sortMetric === metric.key ? 'inset 0 -2px var(--line-active)' : 'none' }}
            >
              {metric.name}{sortMetric === metric.key ? (direction === 'asc' ? ' ↑' : ' ↓') : ''}
            </Button>
          ))}
        </div>
      </div>

      <div style={{ overflowX: 'auto', borderTop: '1px solid var(--line-subtle)' }}>
        <table style={{ width: '100%', minWidth: 920, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: C.bgMuted }}>
              {columns.map((column) => (
                <th key={column.key} style={{ padding: '11px 12px', textAlign: 'left', color: 'var(--text-secondary)', fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', fontWeight: 600 }}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((item, index) => {
              const hypers = item.hyperparameters || {};
              const overall = item.metrics?.overall || {};
              const aggregationLabel = getMetricAggregationLabel(item.metrics, isZh);
              const splitLabel = getSplitLabel(item.metrics, isZh);
              return (
                <tr key={item.task_id} style={{ borderTop: `1px solid ${C.border}` }}>
                  <td style={{ padding: '12px', color: C.ice, fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', fontWeight: 600 }}>
                    <span style={{ color: index === 0 ? C.green : C.ice40, marginRight: 8 }}>{metricValue(item, sortMetric) == null ? '--' : `#${index + 1}`}</span>
                    {item.model_name || `Task #${item.task_id}`}
                    <div style={{ color: 'var(--text-secondary)', fontSize: 'calc(var(--type-micro) * var(--font-scale, 1))', marginTop: 3 }}>Task #{item.task_id}</div>
                    <div style={{ color: 'var(--brand-ice)', fontSize: 'calc(var(--type-micro) * var(--font-scale, 1))', lineHeight: 1.5, marginTop: 5, fontWeight: 600 }}>
                      {aggregationLabel}
                    </div>
                    {splitLabel && (
                      <div style={{ color: 'var(--text-secondary)', fontSize: 'calc(var(--type-micro) * var(--font-scale, 1))', lineHeight: 1.5, marginTop: 2 }}>
                        {splitLabel}
                      </div>
                    )}
                  </td>
                  {metrics.map(({ key: metric }) => (
                    <td key={metric} style={{ padding: '12px', color: GOOD_METRICS.has(metric) ? C.success : C.action, fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', fontWeight: 700, fontFamily: 'var(--font-display)', fontVariantNumeric: 'tabular-nums' }}>
                      {metricValue(item, metric) == null ? (isZh ? '未提供' : 'Not provided') : fmtNum(overall[metric], precision)}
                    </td>
                  ))}
                  <td style={{ padding: '12px', color: C.ice70, fontSize: 'calc(11px * var(--font-scale, 1))', fontWeight: 700 }}>{item.architecture || '--'}</td>
                  <td style={{ padding: '12px', color: C.ice70, fontSize: 'calc(11px * var(--font-scale, 1))', fontWeight: 700 }}>{resolveEnvironmentTask(item, planet).inputs.join(' / ')}</td>
                  <td style={{ padding: '12px', color: C.ice70, fontSize: 'calc(11px * var(--font-scale, 1))' }}>{hypers.window ?? '--'}</td>
                  <td style={{ padding: '12px', color: C.ice70, fontSize: 'calc(11px * var(--font-scale, 1))' }}>{hypers.horizon ?? '--'}</td>
                  <td style={{ padding: '12px', color: C.ice70, fontSize: 'calc(11px * var(--font-scale, 1))' }}>{normalizeCompareDataSource(hypers._effective_data_source || hypers._data_source)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function MetricBars({ items, precision, isZh, plotTextColor, plotGridColor, unit, metrics }) {
  const [activeMetric, setActiveMetric] = useState('rmse');
  const metricMeta = metrics.find((metric) => metric.key === activeMetric) || METRIC_META[0];
  const sorted = useMemo(() => sortCompareItems(items, { metric: activeMetric }), [activeMetric, items]);
  const color = GOOD_METRICS.has(activeMetric) ? C.green : C.mars;

  return (
    <Panel className="prediction-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 14, alignItems: 'flex-start', marginBottom: 14, flexWrap: 'wrap' }}>
        <div>
          <div className="prediction-chart-title">
            {isZh ? '多模型指标柱状图' : 'Metric comparison'}
          </div>
          <div style={{ color: C.ice50, fontSize: 'calc(11px * var(--font-scale, 1))', lineHeight: 1.55, marginTop: 5 }}>
            {unit === 'DU' && metrics.some(metric => metric.key === 'mse')
              ? (isZh ? '完整测试集：MSE/RMSE/MAE/MAPE/SMAPE 越低越好，R² 越高越好。' : 'Full test set: lower MSE/RMSE/MAE/MAPE/SMAPE and higher R² are better.')
              : (isZh ? '整体指标使用完整测试集；RMSE/MAE 越低越好，SSIM/R² 越高越好。' : 'Overall metrics summarize the full test set; lower RMSE/MAE and higher SSIM/R² are better.')}
          </div>
        </div>
        <MetricTabs activeMetric={activeMetric} setActiveMetric={setActiveMetric} metrics={metrics} />
      </div>

      <Plot
        data={[
          {
            type: 'bar',
            orientation: 'h',
            x: sorted.map((item) => metricValue(item, activeMetric)),
            y: sorted.map((item) => item.model_name || `Task #${item.task_id}`),
            marker: {
              color: sorted.map((_, index) => withAlpha(color, Math.max(0.42, 0.96 - index * 0.06))),
              line: { color: withAlpha(color, 0.9), width: 1 },
            },
            customdata: sorted.map((item) => [item.task_id, metricValue(item, activeMetric) == null ? '--' : fmtNum(metricValue(item, activeMetric), precision)]),
            hovertemplate: `<b>%{y}</b><br>Task #%{customdata[0]}<br>${metricMeta.name}: <b>%{customdata[1]}</b><extra></extra>`,
          },
        ]}
        layout={{
          autosize: true,
          height: Math.max(300, sorted.length * 46 + 100),
          margin: { l: 150, r: 24, t: 10, b: 42 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: plotTextColor, family: 'var(--font-body)' },
          xaxis: { title: metricLabel(metricMeta, unit), gridcolor: plotGridColor, zerolinecolor: plotGridColor },
          yaxis: { type: 'category', automargin: true, autorange: 'reversed' },
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: Math.max(300, sorted.length * 46 + 100), flexShrink: 0 }}
      />
    </Panel>
  );
}

function ParameterMatrix({ items, isZh, planet }) {
  const rows = useMemo(() => buildCompareParameterRows(items, planet), [items, planet]);
  const columns = [
    { key: 'modelName', label: isZh ? '模型名' : 'Model' },
    { key: 'taskId', label: 'Task ID' },
    { key: 'modelSource', label: isZh ? '模型来源' : 'Source' },
    { key: 'architecture', label: isZh ? '架构' : 'Architecture' },
    { key: 'target', label: isZh ? '预测目标' : 'Target' },
    { key: 'selectedChannels', label: isZh ? '输入通道' : 'Channels' },
    ...(planet === 'earth' ? [] : [{ key: 'useSphere', label: 'SPHERE' }]),
    { key: 'window', label: 'Window' },
    { key: 'horizon', label: 'Horizon' },
    { key: 'epochs', label: isZh ? '轮次' : 'Epochs' },
    { key: 'batchSize', label: 'Batch' },
    { key: 'learningRate', label: 'LR' },
    { key: 'earlyStoppingPatience', label: isZh ? '早停' : 'Early stop' },
    { key: 'seed', label: 'Seed' },
    { key: 'dataSource', label: isZh ? '数据源' : 'Data source' },
  ];

  return (
    <Panel className="prediction-panel">
      <div className="prediction-chart-title">
        {isZh ? '模型参数矩阵' : 'Parameter matrix'}
      </div>
      <div style={{ color: C.ice50, fontSize: 'calc(11px * var(--font-scale, 1))', lineHeight: 1.55, marginTop: 5, marginBottom: 14 }}>
        {isZh ? '横向对比训练配置，帮助解释指标差异。' : 'Compare training configurations side by side to explain metric differences.'}
      </div>
      <div style={{ overflowX: 'auto', borderTop: '1px solid var(--line-subtle)' }}>
        <table style={{ width: '100%', minWidth: 1120, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: C.bgMuted }}>
              {columns.map((column) => (
                <th key={column.key} style={{ padding: '11px 12px', textAlign: 'left', color: 'var(--text-secondary)', fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', fontWeight: 600 }}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.taskId} style={{ borderTop: `1px solid ${C.border}` }}>
                {columns.map((column) => (
                  <td key={column.key} title={String(row[column.key] ?? '--')} style={{ padding: '12px', color: column.key === 'modelName' ? C.ice : C.ice70, fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', fontWeight: 600, maxWidth: column.key === 'modelName' ? 190 : 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {row[column.key] ?? '--'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function StepCurves({ items, isZh, plotTextColor, plotGridColor, unit, metrics, planet }) {
  const [activeMetric, setActiveMetric] = useState('rmse');
  const metricMeta = metrics.find((metric) => metric.key === activeMetric) || METRIC_META[0];
  const traces = useMemo(() => buildStepCurveTraces(items, activeMetric), [activeMetric, items]);

  return (
    <Panel className="prediction-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 14, alignItems: 'flex-start', marginBottom: 14, flexWrap: 'wrap' }}>
        <div>
          <div className="prediction-chart-title">
            {isZh ? '逐步性能' : 'Step-by-step performance'}
          </div>
          <div style={{ color: C.ice50, fontSize: 'calc(11px * var(--font-scale, 1))', lineHeight: 1.55, marginTop: 5 }}>
            {isZh ? '逐步指标使用完整测试集，查看提前量增加时各模型如何变化。' : 'Per-step metrics summarize the full test set; inspect how each model changes as the lead increases.'}
          </div>
        </div>
        <MetricTabs activeMetric={activeMetric} setActiveMetric={setActiveMetric} metrics={metrics} />
      </div>

      {['rmse', 'mae', 'r2', 'ssim'].includes(activeMetric) ? (
        <ResearchExportButton sources={items.filter(item => item.metrics?.per_step?.every(row => Number.isFinite(row[activeMetric]))).map((item) => item.metrics?.export_ref)}
          sourceLabels={items.filter(item => item.metrics?.per_step?.every(row => Number.isFinite(row[activeMetric]))).map((item) => item.model_name || `Task #${item.task_id}`)} kind="step_curves" metric={activeMetric}
          targetVariable={resolveEnvironmentTask(items[0], planet).target} targetUnit={unit} />
      ) : null}

      <Plot
        data={traces.map((trace, index) => ({
          type: 'scatter',
          mode: 'lines+markers',
          x: trace.x.map((step) => planet === 'earth' ? step * 3 : step),
          y: trace.y,
          name: trace.name,
          line: { width: 2, dash: ['solid', 'dash', 'dashdot', 'dot'][index % 4] },
          marker: { size: 7 },
          hovertemplate: `<b>${trace.name}</b><br>%{x}<br>${metricMeta.name}: <b>%{y:.4f}</b><extra></extra>`,
        }))}
        layout={{
          autosize: true,
          height: 340,
          margin: { l: 54, r: 24, t: 10, b: 46 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: plotTextColor, family: 'var(--font-body)' },
          xaxis: { title: planet === 'earth' ? (isZh ? '提前量（小时）' : 'Lead (hours)') : (isZh ? '预测步' : 'Forecast step'), gridcolor: plotGridColor },
          yaxis: { title: metricLabel(metricMeta, unit), gridcolor: plotGridColor, zerolinecolor: plotGridColor },
          legend: { orientation: 'h', y: -0.24, x: 0, font: { size: 11 } },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: 340, flexShrink: 0 }}
      />
    </Panel>
  );
}

function LazyDiagnosticCard({ title, description, buttonText, loading, data, onLoad, children, isZh }) {
  return (
    <Panel className="prediction-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 14, alignItems: 'flex-start', marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ maxWidth: 680 }}>
          <div className="prediction-chart-title">
            {title}
          </div>
          <div style={{ color: C.ice50, fontSize: 'calc(11px * var(--font-scale, 1))', lineHeight: 1.55, marginTop: 5 }}>
            {description}
          </div>
        </div>
        <Button
          type="button"
          onClick={onLoad}
          disabled={loading}
        >
          {loading ? (isZh ? '计算中…' : 'Computing...') : data ? (isZh ? '刷新' : 'Refresh') : buttonText}
        </Button>
      </div>
      {loading ? (
        <div style={{ height: 220, display: 'grid', placeItems: 'center', background: 'var(--surface-2)', borderRadius: 'var(--radius-control)' }}>
          <div style={{ display: 'grid', gap: 10, justifyItems: 'center' }}>
            <div style={{ width: 24, height: 24, border: `2px solid ${C.border}`, borderTop: `2px solid ${C.green}`, borderRadius: '50%', animation: 'spin-slow 0.8s linear infinite' }} />
            <div style={{ color: C.ice50, fontSize: 'calc(11px * var(--font-scale, 1))' }}>{isZh ? '正在读取完整测试集结果' : 'Reading full test-set results'}</div>
          </div>
        </div>
      ) : data ? children : (
        <div style={{ padding: '20px 8px', borderTop: '1px solid var(--line-subtle)', color: 'var(--text-secondary)', textAlign: 'center', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', lineHeight: 1.5 }}>
          {isZh ? '该模块计算较重，点击按钮后按需加载。' : 'This module is computed on demand because it can be heavier.'}
        </div>
      )}
    </Panel>
  );
}

function ErrorDistributionCompare({ data, loading, onLoad, isZh, plotTextColor, plotGridColor, unit }) {
  const traces = useMemo(() => buildErrorHistogramTraces(data?.items || []), [data]);

  return (
    <LazyDiagnosticCard
      title={isZh ? '多模型误差分布对比' : 'Error distribution comparison'}
      description={isZh ? '叠加各模型完整测试集误差直方图，观察偏差集中区间和长尾。' : 'Overlay full test-set error histograms to inspect bias regions and long tails.'}
      buttonText={isZh ? '加载误差分布' : 'Load distributions'}
      loading={loading}
      data={data}
      onLoad={onLoad}
      isZh={isZh}
    >
      <Plot
        data={traces.map((trace) => ({
          type: 'bar',
          x: trace.centers,
          y: trace.counts,
          name: trace.name,
          opacity: 0.58,
          hovertemplate: `<b>${trace.name}</b><br>${isZh ? '误差' : 'Error'}: %{x:.4f}<br>${isZh ? '数量' : 'Count'}: %{y}<extra></extra>`,
        }))}
        layout={{
          autosize: true,
          height: 340,
          barmode: 'overlay',
          margin: { l: 54, r: 24, t: 10, b: 46 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: plotTextColor, family: 'var(--font-body)' },
          xaxis: { title: `${isZh ? '预测减参考' : 'Prediction - reference'} (${unit})`, gridcolor: plotGridColor, zerolinecolor: plotGridColor },
          yaxis: { title: isZh ? '数量' : 'Count', gridcolor: plotGridColor },
          legend: { orientation: 'h', y: -0.24, x: 0, font: { size: 11 } },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
      />
    </LazyDiagnosticCard>
  );
}

function PfiCompare({ data, loading, onLoad, isZh, plotTextColor, plotGridColor }) {
  const matrix = useMemo(() => buildPfiMatrix(data?.items || []), [data]);

  return (
    <LazyDiagnosticCard
      title={isZh ? 'PFI 特征重要性矩阵' : 'PFI importance matrix'}
      description={isZh ? '行是模型，列是特征；ΔR² 来自各模型最多 40 个抽样测试窗口，保留负值。' : 'Rows are models and columns are features; ΔR² uses up to 40 sampled test windows per model, retaining negative values.'}
      buttonText={isZh ? '加载 PFI 对比' : 'Load PFI'}
      loading={loading}
      data={data}
      onLoad={onLoad}
      isZh={isZh}
    >
      {matrix.features.length > 0 ? (
        <Plot
          data={[
            {
              type: 'heatmap',
              x: matrix.features,
              y: matrix.rows.map((row) => row.modelName),
              z: matrix.rows.map((row) => matrix.features.map((feature) => row.values[feature] || 0)),
              colorscale: [
                [0, 'rgba(121,187,255,0.12)'],
                [0.5, 'rgba(121,187,255,0.72)'],
                [1, 'rgba(99,232,191,0.95)'],
              ],
              hovertemplate: `<b>%{y}</b><br>%{x}<br>${isZh ? '重要性' : 'Importance'}: <b>%{z:.5f}</b><extra></extra>`,
              colorbar: { thickness: 12, title: isZh ? '重要性' : 'Importance' },
            },
          ]}
          layout={{
            autosize: true,
            height: Math.max(280, matrix.rows.length * 42 + 150),
            margin: { l: 150, r: 42, t: 10, b: 92 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: plotTextColor, family: 'var(--font-body)' },
            xaxis: { tickangle: -28, gridcolor: plotGridColor },
            yaxis: { gridcolor: plotGridColor },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
        />
      ) : (
        <div style={{ padding: 24, borderTop: '1px solid var(--line-subtle)', color: 'var(--text-secondary)', textAlign: 'center', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))' }}>
          {isZh ? '暂无 PFI 数据。' : 'No PFI data available.'}
        </div>
      )}
    </LazyDiagnosticCard>
  );
}

export default function CompareTrainingModelsPanel({
  planet = 'mars',
  data,
  loading,
  errorDistributionData,
  errorDistributionLoading,
  onLoadErrorDistribution,
  pfiData,
  pfiLoading,
  onLoadPfi,
  selectedCount,
  precision,
  isZh,
  plotTextColor,
  plotGridColor,
}) {
  const { settings } = useSettings();
  const earth = planet === 'earth';
  const rawItems = Array.isArray(data?.items) ? data.items : [];
  const target = environmentDisplay(rawItems[0] || data, planet, settings.units.ozone);
  const unit = target.unit;
  const items = earth || target.target !== 'O3' ? rawItems : convertCompareMetrics(rawItems, settings.units.ozone);
  const metrics = earth ? EARTH_METRIC_META.map(metric => ({ ...metric, unit: resolveEarthCompareMetricUnit(metric.key, unit, items[0]?.metrics?.metric_units, isZh) })) : METRIC_META.filter(metric => items.some(item => Number.isFinite(item.metrics?.overall?.[metric.key])));
  const convertedErrors = errorDistributionData ? { ...errorDistributionData, items: errorDistributionData.items?.map((item) => ({
    ...item, distribution: { ...item.distribution, hist_errors: { ...item.distribution?.hist_errors,
      bin_edges: item.distribution?.hist_errors?.bin_edges?.map(target.convertValue),
    } },
  })) } : null;

  if (loading) return <LoadingState isZh={isZh} />;
  if (!sameEnvironmentTarget(rawItems, planet)) return <div role="alert">{isZh ? '请选择相同预测目标的模型进行比较。' : 'Choose models with the same prediction target.'}</div>;
  if (!items.length) return <EmptyState isZh={isZh} selectedCount={selectedCount} />;

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <SummaryTable items={items} precision={precision} isZh={isZh} unit={unit} metrics={metrics} planet={planet} />
      <MetricBars metrics={metrics}
        unit={unit}
        items={items}
        precision={precision}
        isZh={isZh}
        plotTextColor={plotTextColor}
        plotGridColor={plotGridColor}
      />
      <StepCurves metrics={metrics} planet={planet}
        unit={unit}
        items={items}
        isZh={isZh}
        plotTextColor={plotTextColor}
        plotGridColor={plotGridColor}
      />
      {!earth ? <ErrorDistributionCompare
        unit={unit}
        data={convertedErrors}
        loading={errorDistributionLoading}
        onLoad={onLoadErrorDistribution}
        isZh={isZh}
        plotTextColor={plotTextColor}
        plotGridColor={plotGridColor}
      /> : null}
      {!earth ? <PfiCompare
        data={pfiData}
        loading={pfiLoading}
        onLoad={onLoadPfi}
        isZh={isZh}
        plotTextColor={plotTextColor}
        plotGridColor={plotGridColor}
      /> : null}
      <ParameterMatrix items={items} isZh={isZh} planet={planet} />
    </div>
  );
}
