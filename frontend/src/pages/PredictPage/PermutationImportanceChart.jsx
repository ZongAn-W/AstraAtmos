import Plot from 'react-plotly.js';
import C from '../../constants/colors';
import { useT } from '../../i18n';
import ResearchExportButton from './ResearchExportButton';
import { useSettings } from '../../contexts/SettingsContext';
import { fmtNum } from '../../utils/fmt';
import PredictionChartCard from './PredictionChartCard.jsx';
import { createMarsPredictionPresentation, predictionChartTheme } from './predictionPresentation.js';

export default function PermutationImportanceChart({ data, loading, plotTextColor, plotGridColor,
  plotText60, presentation, precision }) {
  const t = useT();
  const { settings } = useSettings();
  const config = presentation || createMarsPredictionPresentation({ settings }).pfi;
  const model = config.readData(data);
  const items = model.items.filter((item) => Number.isFinite(item.value));
  const theme = predictionChartTheme(settings, { plotTextColor, plotText60, plotGridColor });
  const decimals = precision ?? settings.precision;
  const names = items.map((item) => item.translateKey ? t(item.translateKey) || item.name : item.name);
  const hoverDigits = decimals === 'full' ? '' : `:.${decimals}f`;
  const trace = {
    x: items.map((item) => item.value), y: names, type: 'bar', orientation: 'h',
    marker: { color: items.map((item) => item.value < 0 ? '#f0ad66' : C.blue), opacity: .8 },
    error_x: { type: 'data', array: items.map((item) => Number.isFinite(item.std) ? item.std : 0),
      visible: items.some((item) => Number.isFinite(item.std)) },
    hovertemplate: `%{y}: %{x${hoverDigits}} ${model.unitLabel}<extra></extra>`,
  };
  return <PredictionChartCard title={model.title} subtitle={model.scopeText} scope={model.scope} note={model.helpText}
    action={model.exportEnabled ? <ResearchExportButton sources={[model.exportRef]} kind="pfi" disabled={loading} targetVariable={config.targetVariable} targetUnit={config.targetUnit} /> : null}>
    <div className="prediction-chart-scores" data-pfi-axis-label={model.axisLabel}>{model.baselineLabel}: {Number.isFinite(model.baseline)
      ? fmtNum(model.baseline, decimals) : '—'} {model.unitLabel}</div>
    {loading ? <div className="prediction-chart-skeleton" role="status" aria-label={t('predict.generatingHint')} />
      : items.length ? <div className="prediction-chart-plot" style={{ height: Math.max(250, items.length * 44) }}>
        <Plot data={[trace]} layout={{ autosize: true, margin: { l: 104, r: 24, t: 10, b: 60 },
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: theme.text, family: 'Inter, system-ui, sans-serif' },
          xaxis: { title: { text: model.axisLabel, font: { size: 11, color: theme.text } },
            tickfont: { size: 11, color: theme.muted }, gridcolor: theme.grid,
            zeroline: true, zerolinecolor: theme.grid, zerolinewidth: 1, automargin: true },
          yaxis: { autorange: 'reversed', tickfont: { size: 11, color: theme.text }, gridcolor: 'transparent', automargin: true },
          hovermode: 'closest',
        }} config={{ displayModeBar: false, responsive: true }} useResizeHandler style={{ width: '100%', height: '100%' }} />
      </div> : <div className="prediction-chart-status">{model.emptyLabel}</div>}
  </PredictionChartCard>;
}
