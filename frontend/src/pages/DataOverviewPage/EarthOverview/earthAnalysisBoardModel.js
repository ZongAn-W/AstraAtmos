/** Plotly data for Earth theme boards; scientific summaries stay backend-owned. */
import { EARTH_VARIABLE_UNITS } from './earthOverviewModel.js';
import {
  bandLabel,
  buildBandDiagnostics,
  buildEnvironmentSeries,
  buildExtremesTable,
  buildRegionalTrend,
  buildRelationship,
  buildSeasonalHeatmap,
  buildSpatialAnomaly,
  variableLabel,
} from './earthResearchModel.js';

const CHOSEN_COLOR = '#6aa9ff';
const OZONE_COLOR = '#ff9b70';
const finiteOrNull = (value) => (Number.isFinite(value) ? value : null);
const hasFinite = (values) => values.some(Number.isFinite);
const axis = (text) => ({ title: { text } });
const dateAxis = (isZh) => ({ ...axis(isZh ? '日期（UTC 日平均）' : 'Date (UTC daily mean)'), type: 'date' });
const valuesAtDates = (values, dates) => dates.map((_, index) => finiteOrNull(values?.[index]));
const seriesLabel = (variable, units, isZh) => `${variableLabel(variable, isZh)}${units ? ` (${units})` : ''}`;

function dateValues(values) {
  return Array.isArray(values) && values.every((value) => (
    typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value))
  )) ? [...values] : [];
}

function numericAxis(values) {
  return Array.isArray(values) && values.every(Number.isFinite) ? [...values] : [];
}

function matrixValues(matrix, x, y) {
  if (!x.length || !y.length || !Array.isArray(matrix) || matrix.length !== y.length
    || matrix.some((row) => !Array.isArray(row) || row.length !== x.length)) return [];
  return matrix.map((row) => row.map(finiteOrNull));
}

function lineTrace({ x, y, name, units = '', color = CHOSEN_COLOR, dash = 'solid' }) {
  return {
    type: 'scatter', mode: 'lines', x: [...x], y, name,
    connectgaps: false,
    line: { color, width: 2, dash },
    hovertemplate: `%{x}<br>%{y:.3f}${units ? ` ${units}` : ''}<extra>%{fullData.name}</extra>`,
  };
}

function bandPanel({ id, title, rows, metric, units, isZh }) {
  const x = (rows || []).map((row) => finiteOrNull(row[metric]));
  return {
    id, title,
    data: [{
      type: 'bar', orientation: 'h',
      x, y: (rows || []).map((row) => bandLabel(row.bandId, isZh)),
      marker: { color: CHOSEN_COLOR },
      hovertemplate: `%{y}<br>%{x:.3f} ${units}<extra></extra>`,
    }],
    layout: {
      xaxis: axis(`${title}${units ? ` (${units})` : ''}`),
      yaxis: { ...axis(isZh ? '纬带' : 'Latitude band'), type: 'category', autorange: 'reversed' },
      showlegend: false,
    },
    empty: !hasFinite(x),
  };
}

function temporalBoard({ suite, variable, isZh }) {
  const seasonal = buildSeasonalHeatmap(suite, variable, { isZh });
  const regional = buildRegionalTrend(suite, { isZh });
  const selected = regional?.series.find((entry) => entry.id === variable);
  const extremes = buildExtremesTable(suite, variable);
  const dates = dateValues(suite?.dates);
  const latitude = numericAxis(seasonal?.y);
  const z = matrixValues(seasonal?.z, dates, latitude);
  const units = selected?.units || seasonal?.units || EARTH_VARIABLE_UNITS[variable] || '';
  const label = seriesLabel(variable, units, isZh);
  const raw = valuesAtDates(selected?.values, dates);
  return {
    panels: [
      {
        id: 'earth-seasonal',
        title: isZh ? `${variableLabel(variable, true)}季节结构` : `${variableLabel(variable, false)} seasonal structure`,
        data: [{
          type: 'heatmap', x: dates, y: latitude, z,
          colorscale: 'Viridis', colorbar: { title: { text: label } },
          hoverongaps: false,
          hovertemplate: `%{x}<br>${isZh ? '纬度' : 'Latitude'} %{y}°<br>%{z:.3f} ${units}<extra></extra>`,
        }],
        layout: { xaxis: dateAxis(isZh), yaxis: axis(isZh ? '纬度（°）' : 'Latitude (°)'), showlegend: false },
        empty: !z.some(hasFinite),
      },
      {
        id: 'earth-global-trend',
        title: isZh ? '全球面积加权均值' : 'Global area-weighted mean',
        data: [lineTrace({ x: dates, y: raw, name: label, units })],
        layout: { xaxis: dateAxis(isZh), yaxis: axis(label), showlegend: false },
        empty: !hasFinite(raw),
      },
      bandPanel({
        id: 'earth-band-amplitude', title: isZh ? '纬带全年振幅' : 'Annual latitude-band amplitude',
        rows: extremes, metric: 'peakToPeak', units, isZh,
      }),
    ],
    note: isZh
      ? '使用 UTC 日平均数据与原始物理单位。季节矩阵为同纬度经度均值；全球曲线和纬带日均曲线由后端按球面单元面积加权，振幅为各纬带日均曲线全年最大值减最小值。'
      : 'UTC daily means retain their physical units. The seasonal matrix uses same-latitude longitude means. Global and latitude-band daily series use backend cell-area weighting; amplitude is the annual maximum minus minimum of each band series.',
  };
}

/** Population Z-score of finite samples; missing dates remain gaps. */
function standardized(values, dates) {
  const raw = valuesAtDates(values, dates);
  const finite = raw.filter(Number.isFinite);
  const unavailable = { values: raw.map(() => null), unavailable: true };
  // Check equality before accumulating: repeated decimals can acquire a tiny
  // variance from rounding alone, producing a misleading nonzero Z-score.
  if (finite.length < 2 || finite.every((value) => value === finite[0])) return unavailable;
  const mean = finite.reduce((sum, value) => sum + value / finite.length, 0);
  const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2 / finite.length, 0);
  const deviation = Math.sqrt(variance);
  if (!Number.isFinite(deviation) || deviation === 0) return unavailable;
  return {
    values: raw.map((value) => (value === null ? null : finiteOrNull((value - mean) / deviation))),
    unavailable: false,
  };
}

function driversBoard({ suite, driver, scope, isZh }) {
  const kind = driver === 'SWGDN' ? 'solar_ozone' : driver === 'T2M' ? 'temperature_ozone' : null;
  const dates = dateValues(suite?.dates);
  const relationship = kind ? buildRelationship({ ...suite, dates }, scope, kind) : null;
  const environment = kind ? buildEnvironmentSeries(suite, { bandId: scope, isZh }) : null;
  const driverUnits = relationship?.driverUnits || EARTH_VARIABLE_UNITS[driver] || '';
  const driverName = seriesLabel(driver, driverUnits, isZh);
  const ozoneName = seriesLabel('TO3', EARTH_VARIABLE_UNITS.TO3, isZh);
  const scopeName = bandLabel(scope, isZh);
  const pairs = dates.flatMap((date, index) => (
    Number.isFinite(relationship?.x[index]) && Number.isFinite(relationship?.y[index])
      ? [{ date, x: relationship.x[index], y: relationship.y[index] }] : []
  ));
  const scatterTraces = [{
    type: 'scatter', mode: 'markers',
    x: pairs.map((pair) => pair.x), y: pairs.map((pair) => pair.y), text: pairs.map((pair) => pair.date),
    name: isZh ? '同期日均值' : 'Concurrent daily means',
    marker: { color: CHOSEN_COLOR, size: 5, opacity: 0.65 },
    hovertemplate: `${driverName}: %{x:.3f}<br>${ozoneName}: %{y:.3f}<br>%{text}<extra></extra>`,
  }];
  const regression = relationship?.regression;
  if (pairs.length > 1 && Number.isFinite(regression?.slope) && Number.isFinite(regression?.intercept)) {
    const xs = pairs.map((pair) => pair.x);
    const endpoints = [Math.min(...xs), Math.max(...xs)];
    const fitted = endpoints.map((value) => value * regression.slope + regression.intercept);
    if (endpoints[0] !== endpoints[1] && fitted.every(Number.isFinite)) {
      scatterTraces.push({
        type: 'scatter', mode: 'lines', x: endpoints, y: fitted,
        line: { color: OZONE_COLOR, width: 2, dash: 'dash' },
        name: isZh ? '线性拟合' : 'Linear fit', hoverinfo: 'skip',
      });
    }
  }
  const driverTrend = standardized(environment?.series.find((entry) => entry.id === driver)?.values, dates);
  const ozoneTrend = standardized(environment?.series.find((entry) => entry.id === 'TO3')?.values, dates);
  const lag = (relationship?.lag || []).filter((row) => Number.isFinite(row?.lag_days));
  const lagValues = lag.map((row) => finiteOrNull(row.r));
  let note = isZh
    ? '同期曲线使用所选范围的单元面积加权日均值；Z-score 仅对各曲线的有限值标准化并保留缺测。散点拟合与相关系数来自后端；相关不代表因果。正滞后表示驱动领先臭氧，使用重叠样本，不跨年补点；共同季节性与自相关意味着峰值滞后不能解释为物理响应时间。'
    : 'Concurrent curves use cell-area-weighted daily means for the selected scope; each Z-score uses finite samples and preserves gaps. Fits and correlations come from the backend; correlation does not establish causality. Positive lag means the driver leads ozone, using overlapping samples without cross-year padding. Shared seasonality and autocorrelation mean a peak lag is not a physical response time.';
  if (driverTrend.unavailable || ozoneTrend.unavailable) {
    note += isZh
      ? ' 常量序列或有限样本不足的序列无法标准化，显示为空缺。'
      : ' Constant series or series with too few finite samples have no Z-score and remain empty.';
  }
  return {
    panels: [
      {
        id: 'earth-relationship',
        title: isZh ? `${scopeName}：${variableLabel(driver, true)}与臭氧` : `${scopeName}: ${variableLabel(driver, false)} and ozone`,
        data: scatterTraces,
        layout: { xaxis: axis(driverName), yaxis: axis(ozoneName), showlegend: true },
        empty: !pairs.length,
      },
      {
        id: 'earth-concurrent-trend',
        title: isZh ? `${scopeName}同期变化（标准化）` : `${scopeName} concurrent variation (standardized)`,
        data: [
          lineTrace({ x: dates, y: driverTrend.values, name: variableLabel(driver, isZh) }),
          lineTrace({ x: dates, y: ozoneTrend.values, name: variableLabel('TO3', isZh), color: OZONE_COLOR, dash: 'dash' }),
        ],
        layout: { xaxis: dateAxis(isZh), yaxis: axis(isZh ? 'Z-score（无量纲）' : 'Z-score (dimensionless)'), showlegend: true },
        empty: !hasFinite(driverTrend.values) && !hasFinite(ozoneTrend.values),
      },
      {
        id: 'earth-lag-correlation',
        title: isZh ? '滞后相关' : 'Lagged correlation',
        data: [{
          ...lineTrace({ x: lag.map((row) => row.lag_days), y: lagValues, name: 'Pearson r' }),
          mode: 'lines+markers', marker: { color: CHOSEN_COLOR, size: 4 },
          customdata: lag.map((row) => [finiteOrNull(row.n), row.reason ?? null]),
          hovertemplate: `${isZh ? '滞后' : 'Lag'} %{x} ${isZh ? '天' : 'days'}<br>r=%{y:.3f}<br>n=%{customdata[0]}<extra></extra>`,
        }],
        layout: {
          xaxis: axis(isZh ? '滞后（天，正值表示驱动领先）' : 'Lag (days, positive = driver leads)'),
          yaxis: { ...axis('Pearson r'), range: [-1, 1] }, showlegend: false,
        },
        empty: !hasFinite(lagValues),
      },
    ],
    note,
  };
}

function dynamicsBoard({ spatial, variable, isZh }) {
  const anomaly = buildSpatialAnomaly(spatial);
  const rows = buildBandDiagnostics(spatial);
  const longitude = numericAxis(anomaly?.x);
  const latitude = numericAxis(anomaly?.y);
  const z = matrixValues(anomaly?.z, longitude, latitude);
  const units = anomaly?.units || EARTH_VARIABLE_UNITS[variable] || '';
  const label = `${variableLabel(variable, isZh)}${isZh ? '距平' : ' anomaly'}${units ? ` (${units})` : ''}`;
  const range = [anomaly?.colorRange?.min, anomaly?.colorRange?.max].filter(Number.isFinite);
  const bound = (range.length ? range : z.flat().filter(Number.isFinite))
    .reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 1e-9);
  return {
    panels: [
      {
        id: 'earth-spatial-anomaly',
        title: isZh ? `${variableLabel(variable, true)}空间距平` : `${variableLabel(variable, false)} spatial anomaly`,
        data: [{
          type: 'heatmap', x: longitude, y: latitude, z,
          colorscale: 'RdBu', reversescale: true, zmin: -bound, zmax: bound, zmid: 0,
          colorbar: { title: { text: label } }, hoverongaps: false,
          hovertemplate: `${isZh ? '经度' : 'Longitude'} %{x}°<br>${isZh ? '纬度' : 'Latitude'} %{y}°<br>${isZh ? '距平' : 'Anomaly'} %{z:.3f} ${units}<extra></extra>`,
        }],
        layout: {
          xaxis: axis(isZh ? '经度（°）' : 'Longitude (°)'),
          yaxis: axis(isZh ? '纬度（°）' : 'Latitude (°)'), showlegend: false,
        },
        empty: !z.some(hasFinite),
      },
      bandPanel({ id: 'earth-band-rms', title: isZh ? '纬带距平 RMS' : 'Latitude-band anomaly RMS', rows, metric: 'rms', units, isZh }),
      bandPanel({ id: 'earth-band-span', title: isZh ? '纬带距平峰谷跨度' : 'Latitude-band anomaly span', rows, metric: 'peakToPeak', units, isZh }),
    ],
    note: isZh
      ? '空间距平为年平均场减去同纬度经度均值，不是多年气候距平。RMS 由后端按单元面积加权，包含经度点数；峰谷跨度为纬带内距平最大值减最小值。'
      : 'Spatial anomaly is the annual mean field minus the same-latitude longitude mean, not a multi-year climatological anomaly. Backend RMS uses cell-area weighting, including longitude points; span is the maximum minus minimum anomaly within each band.',
  };
}

export function buildEarthAnalysisBoard({
  mode = 'temporal', suite, spatial, variable = 'TO3', driver = 'T2M', scope = 'global', isZh = true,
} = {}) {
  if (mode === 'drivers') return driversBoard({ suite, driver, scope, isZh });
  if (mode === 'dynamics') return dynamicsBoard({ spatial, variable, isZh });
  return temporalBoard({ suite, variable, isZh });
}
