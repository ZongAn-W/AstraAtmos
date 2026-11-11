import React from 'react';
import ExpandMoreRounded from '@mui/icons-material/ExpandMoreRounded';
import { variableLabel } from './earthResearchModel.js';
import './spatialDiagnosticsGuide.css';

/** Shared reading guide for the spatial board and the individual analysis. */
export default function SpatialDiagnosticsGuide({ variableId, units, isZh = true }) {
  const label = variableLabel(variableId, isZh);
  const copy = isZh ? {
    title: '空间诊断读图指南',
    intro: `以${label}的年平均场为基础，比较同一纬度上不同地区的高低差异。`,
    legend: ['偏低 · 负距平', '接近平均', '偏高 · 正距平'],
    detailsTitle: '计算口径与解读边界',
    expand: '展开说明',
    collapse: '收起说明',
    exampleTitle: '用一组数值理解距平',
    exampleHint: '示例数据，仅用于解释计算',
    annualLabel: '某地年平均值',
    baselineLabel: '减去同纬度均值',
    resultLabel: '得到距平',
    spanLabel: '同一纬带的峰谷跨度',
    exampleUnit: 'DU 为多布森单位，表示臭氧柱总量。',
    unitsTitle: '如何理解单位',
    scopeTitle: '适用范围',
    samplingTitle: '为什么地图与柱状图的极值可能不同',
    bandsTitle: '纬带范围',
    reference: `距平 = 某地的年平均${label} − 同一年、同一纬度所有经度的平均值。它回答的是“这个位置比同纬度的其他地方偏高还是偏低”。`,
    items: [
      {
        id: 'map', label: '空间距平', question: '差异在哪里',
        summary: '在地图上定位偏高与偏低的地区。颜色越深，偏离同纬度平均水平越明显。',
        title: '空间距平：看差异在哪里',
        body: '横轴是经度，纵轴是纬度。蓝色为正距平，表示偏高；红色为负距平，表示偏低；接近灰色表示接近同纬度平均水平。颜色越深，偏离越大。各纬度使用各自的基准，不能用这张图比较不同纬度的绝对含量。',
      },
      {
        id: 'rms', label: '纬带 RMS', question: '整体偏离有多大',
        summary: '用面积加权的均方根概括整个纬带的空间差异。数值越大，整体偏离越明显。',
        title: '纬带 RMS：看整体偏离程度',
        body: 'RMS 是均方根：将纬带内各格点的距平平方，按格点面积求加权平均，再开平方。它用一个非负数概括整个纬带的空间差异，数值越大，整体偏离同纬度平均水平的程度越大；较大的偏离对 RMS 的影响也更大。',
      },
      {
        id: 'span', label: '峰谷跨度', question: '最大的高低差是多少',
        summary: '纬带内最大距平减去最小距平，对少量极端格点更敏感；结合 RMS 判断差异是否广泛。',
        title: '峰谷跨度：看最大的高低差',
        body: '峰谷跨度 = 纬带内最大距平 − 最小距平。它表示偏高与偏低地区之间最明显的差距，对少量极端格点较敏感。RMS 较小但跨度较大时，差异可能集中在少数位置，需要结合地图判断。',
      },
    ],
    example: variableId === 'TO3'
      ? '独立的 RMS 读数示例：RMS 为 8 DU 表示整体空间偏离约为这一量级，不代表臭氧总量只有 8 DU。RMS 需要纬带内全部格点参与计算，不能仅由最大、最小距平推导。'
      : `三项结果均沿用所选变量的单位${units ? `（${units}）` : ''}，表示差值或差值的幅度。温度的距平、RMS 和跨度均表示温差。`,
    bands: '纬带范围：热带约为 30°S–30°N，南北半球中纬度为各自的 30°–60°，极区为各自的 60°–90°。',
    sampling: 'RMS 和峰谷跨度基于原始网格统计；地图为便于显示做了空间平均，局部极值可能被减弱，因此不能直接用图上颜色的极值反推柱状图读数。',
    scope: '这些图描述所选年份年平均场的空间差异。它们不表示年内随时间的波动、多年气候异常或模型预测误差，也不能仅凭数值较大就判断数据质量或预测精度。',
  } : {
    title: 'A guide to spatial diagnostics',
    intro: `Use the annual mean ${label} field to compare locations along the same latitude.`,
    legend: ['Lower · negative', 'Near the mean', 'Higher · positive'],
    detailsTitle: 'Methods and interpretation limits',
    expand: 'Show details', collapse: 'Hide details',
    exampleTitle: 'Read an anomaly with an example',
    exampleHint: 'Illustrative values, not the current results',
    annualLabel: 'Local annual mean', baselineLabel: 'Subtract the same-latitude mean',
    resultLabel: 'Resulting anomaly', spanLabel: 'Peak-to-peak span within a band',
    exampleUnit: 'DU means Dobson unit, a measure of the ozone column.',
    unitsTitle: 'Understanding the units', scopeTitle: 'Interpretation limits',
    samplingTitle: 'Why map and bar-chart extremes can differ', bandsTitle: 'Latitude bands',
    reference: `Anomaly = the annual mean ${label} at a location minus the mean across all longitudes at the same latitude in the same year. It shows whether that location is higher or lower than other locations along its latitude.`,
    items: [
      {
        id: 'map', label: 'Spatial anomaly', question: 'Where are the differences?',
        summary: 'Locate higher and lower regions on the map. Deeper colours indicate larger departures from the same-latitude mean.',
        title: 'Spatial anomaly: locate the differences',
        body: 'Longitude is horizontal and latitude is vertical. Blue indicates a positive anomaly (higher), red a negative anomaly (lower), and near-grey values are close to the same-latitude mean. Deeper colours indicate larger departures. Each latitude has its own reference, so this map does not compare absolute levels across latitudes.',
      },
      {
        id: 'rms', label: 'Latitude-band RMS', question: 'How strong is the overall departure?',
        summary: 'The area-weighted root mean square summarises spatial differences across the whole band. Larger values mean stronger overall departures.',
        title: 'Latitude-band RMS: assess the overall departure',
        body: 'RMS is the root mean square: square the anomalies, average them across the band using cell-area weights, then take the square root. This non-negative value summarises spatial differences across the whole band. Larger values mean stronger departures from the same-latitude means, with larger departures contributing more strongly.',
      },
      {
        id: 'span', label: 'Peak-to-peak span', question: 'What is the largest contrast?',
        summary: 'The largest anomaly minus the smallest within a band. It is more sensitive to extreme cells; compare it with RMS to assess how widespread differences are.',
        title: 'Peak-to-peak span: assess the largest contrast',
        body: 'Span = the largest anomaly minus the smallest anomaly within the band. It describes the strongest contrast between higher and lower locations and is sensitive to extreme cells. A small RMS with a large span can indicate that differences are concentrated in a few locations; check the map for their distribution.',
      },
    ],
    example: variableId === 'TO3'
      ? 'A separate RMS reading example: an RMS of 8 DU describes the scale of spatial departures; it does not mean the ozone column itself is 8 DU. RMS uses all cells in the band and cannot be derived from the largest and smallest anomalies alone.'
      : `All three results retain the selected variable’s units${units ? ` (${units})` : ''} and describe differences or their magnitude. Temperature anomalies, RMS and spans represent temperature differences.`,
    bands: 'Latitude bands: tropics cover approximately 30°S–30°N, midlatitudes cover 30°–60° in each hemisphere, and polar regions cover 60°–90°.',
    sampling: 'RMS and span use the original grid. The map is spatially averaged for display, which can soften local extremes, so its displayed colour extremes may not reproduce the bar-chart statistics.',
    scope: 'These plots describe spatial differences in the selected year’s annual mean field. They do not measure variation over time within the year, anomalies against a multi-year climate baseline, or model prediction errors. Larger values alone do not establish data quality or prediction accuracy.',
  };

  return (
    <section className="spatial-diagnostics-guide" aria-label={copy.title}>
      <header className="spatial-guide-heading">
        <h4>{copy.title}</h4>
        <p>{copy.intro}</p>
      </header>
      <div className="spatial-guide-overview">
        <dl className="spatial-guide-metrics">
          {copy.items.map(item => (
            <div key={item.id}>
              <dt><span>{item.label}</span><span className="spatial-guide-question">{item.question}</span></dt>
              <dd>
                <p>{item.summary}</p>
                {item.id === 'map' ? (
                  <div className="spatial-guide-legend">
                    {copy.legend.map((text, index) => <span key={text}><i className={`spatial-guide-swatch spatial-guide-swatch--${index}`} aria-hidden="true" />{text}</span>)}
                  </div>
                ) : null}
              </dd>
            </div>
          ))}
        </dl>
        <figure className="spatial-guide-example">
          <figcaption>
            <h5>{variableId === 'TO3' ? copy.exampleTitle : copy.unitsTitle}</h5>
            {variableId === 'TO3' ? <p>{copy.exampleHint}</p> : null}
          </figcaption>
          {variableId === 'TO3' ? (
            <>
              <dl className="spatial-guide-calculation">
                <div><dt>{copy.annualLabel}</dt><dd>320 <span>DU</span></dd></div>
                <div><dt>{copy.baselineLabel}</dt><dd>300 <span>DU</span></dd></div>
                <div className="spatial-guide-result"><dt>{copy.resultLabel}</dt><dd>+20 <span>DU</span></dd></div>
              </dl>
              <div className="spatial-guide-span-example"><span>{copy.spanLabel}</span><strong>+20 − (−10) = 30 DU</strong></div>
              <p className="spatial-guide-unit">{copy.exampleUnit}</p>
            </>
          ) : <p>{copy.example}</p>}
        </figure>
      </div>
      <details className="spatial-guide-details">
        <summary>
          <span>{copy.detailsTitle}</span>
          <span className="spatial-guide-disclosure" aria-hidden="true">
            <span className="spatial-guide-show">{copy.expand}</span><span className="spatial-guide-hide">{copy.collapse}</span>
            <ExpandMoreRounded fontSize="small" />
          </span>
        </summary>
        <div className="spatial-guide-details-content">
          <p className="spatial-guide-reference">{copy.reference}</p>
          <dl>
            {copy.items.map(item => <div key={item.id}><dt>{item.title}</dt><dd>{item.body}</dd></div>)}
            <div><dt>{copy.unitsTitle}</dt><dd>{copy.example}</dd></div>
            <div><dt>{copy.bandsTitle}</dt><dd>{copy.bands}</dd></div>
            <div><dt>{copy.samplingTitle}</dt><dd>{copy.sampling}</dd></div>
            <div><dt>{copy.scopeTitle}</dt><dd>{copy.scope}</dd></div>
          </dl>
        </div>
      </details>
    </section>
  );
}
