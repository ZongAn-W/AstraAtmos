import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useSettings } from '../../contexts/SettingsContext';
import { fmtNum } from '../../utils/fmt';
import ResearchExportButton from './ResearchExportButton';
import PredictionChartCard from './PredictionChartCard.jsx';
import { createMarsPredictionPresentation, predictionChartTheme } from './predictionPresentation.js';

export default function ErrorDistributionChart({ data, predictionResult, step = 0, loading,
  plotTextColor, plotText60, plotGridColor, presentation, precision }) {
  const { settings } = useSettings();
  const zh = settings.language !== 'en';
  const config = presentation || createMarsPredictionPresentation({ settings }).distribution;
  const state = useMemo(() => {
    if (!data) return {};
    try { return { plots: config.readData(data, { predictionResult, step }) }; }
    catch (error) { return { error: error.message }; }
  }, [data, predictionResult, step, config]);
  const theme = predictionChartTheme(settings, { plotTextColor, plotText60, plotGridColor });
  const decimals = precision ?? settings.precision;
  const format = (value) => Number.isFinite(value) ? fmtNum(value, decimals) : '—';
  const unit = state.plots?.unitLabel || config.unitLabel;
  const baseLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', autosize: true,
    font: { color: theme.text, family: 'Inter, system-ui, sans-serif' },
    margin: { t: 28, r: 20, l: 56, b: 58 },
    xaxis: { gridcolor: theme.grid, zerolinecolor: theme.grid, tickfont: { size: 11, color: theme.muted }, automargin: true },
    yaxis: { gridcolor: theme.grid, zerolinecolor: theme.grid, tickfont: { size: 11, color: theme.muted }, automargin: true },
  };
  const plotConfig = { displayModeBar: false, responsive: true };
  const plot = (traces, layout) => <div className="prediction-chart-plot"><Plot data={traces}
    layout={{ ...baseLayout, ...layout }} useResizeHandler config={plotConfig} style={{ width: '100%', height: '100%' }} /></div>;
  if (!data && !loading) return null;
  if (loading || !data) return <div className="prediction-chart-grid" role="status"
    aria-label={zh ? '诊断图加载中' : 'Loading diagnostic charts'}>
    {[0, 1, 2].map((index) => <div className="prediction-chart-skeleton" key={index} />)}</div>;
  if (state.error) return <div role="alert" className="prediction-chart-status prediction-chart-error">{state.error}</div>;
  const { scatter, distributions, residual } = state.plots;
  const hoverDigits = decimals === 'full' ? '' : `:.${decimals}f`;
  const roleColors = settings.theme === 'light'
    ? { reference: '#236387', prediction: '#b94e30', residual: '#7254b5' }
    : { reference: '#9ad9ef', prediction: '#f19a78', residual: '#b39bff' };
  const barTrace = (bins, name, color) => ({ x: bins?.centers || [], y: bins?.counts || [], width: bins?.widths,
    type: 'bar', name, marker: { color }, opacity: .8 });
  return <div className="prediction-chart-grid" data-predict-distribution="true" data-field-unit={unit}>
    {scatter ? <PredictionChartCard title={zh ? '参考与预测散点' : 'Reference vs prediction'}
      subtitle={scatter.scopeText} scope={scatter.scope} note={scatter.note}
      action={scatter.exportEnabled ? <ResearchExportButton sources={[scatter.exportRef]} kind="scatter" step={step} targetVariable={config.targetVariable} targetUnit={unit} /> : null}>
      {plot([
        { x: scatter.reference, y: scatter.prediction, type: 'scattergl', mode: 'markers',
          marker: { color: roleColors.prediction, size: 3, opacity: .65 },
          hovertemplate: `${zh ? '参考' : 'Reference'}: %{x${hoverDigits}} ${unit}<br>${zh ? '预测' : 'Prediction'}: %{y${hoverDigits}} ${unit}<extra></extra>` },
        { x: scatter.range || [], y: scatter.range || [], type: 'scatter', mode: 'lines', name: 'y = x',
          line: { color: roleColors.reference, dash: 'dash', width: 2 }, hoverinfo: 'name' },
      ], { showlegend: false,
        xaxis: { ...baseLayout.xaxis, range: scatter.range, title: { text: `${zh ? '参考' : 'Reference'} (${unit})` } },
        yaxis: { ...baseLayout.yaxis, range: scatter.range, scaleanchor: 'x', scaleratio: 1,
          title: { text: `${zh ? '预测' : 'Prediction'} (${unit})` } },
      })}
    </PredictionChartCard> : null}
    {distributions ? <PredictionChartCard title={zh ? '参考与预测分布' : 'Reference and prediction distributions'}
      subtitle={`${distributions.scopeText} · ${unit}`} scope={distributions.scope}>
      {plot([barTrace(distributions.reference, zh ? '参考' : 'Reference', roleColors.reference),
        barTrace(distributions.prediction, zh ? '预测' : 'Prediction', roleColors.prediction)], {
        barmode: 'overlay', xaxis: { ...baseLayout.xaxis, title: { text: unit } },
        legend: { orientation: 'h', y: 1.15, x: .5, xanchor: 'center', font: { size: 11 } },
      })}
    </PredictionChartCard> : null}
    {residual ? <PredictionChartCard title={zh ? '残差分布' : 'Residual distribution'}
      subtitle={residual.scopeText} scope={residual.scope} note={residual.note}>
      <div className="prediction-chart-scores"><span>RMSE: {format(residual.rmse)} {unit}</span><span>MAE: {format(residual.mae)} {unit}</span></div>
      {plot([barTrace(residual, zh ? '残差' : 'Residual', roleColors.residual)], { showlegend: false, bargap: 0,
        xaxis: { ...baseLayout.xaxis, title: { text: `${zh ? '预测 − 参考' : 'Prediction − reference'} (${unit})` } },
        yaxis: { ...baseLayout.yaxis, title: { text: zh ? '点位数' : 'Count' } },
        shapes: [{ type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: '#f0ad66', width: 2, dash: 'dash' } }],
      })}
      {residual.summary ? <p className="prediction-chart-subtitle">{zh ? '均值 / 标准差 / 最小 / 最大' : 'Mean / std / min / max'} ({unit}): {['mean', 'std', 'min', 'max'].map((key) => format(residual.summary[key])).join(' / ')}</p> : null}
    </PredictionChartCard> : null}
  </div>;
}
