import React, { useMemo } from 'react';
import AnalysisBoard from '../workbench/AnalysisBoard.jsx';
import ChartRequestError from '../OverviewCharts/ChartRequestError.jsx';
import { reasonText } from '../workbench/OverviewCard.jsx';
import { buildEarthAnalysisBoard } from './earthAnalysisBoardModel.js';
import SpatialDiagnosticsGuide from './SpatialDiagnosticsGuide.jsx';

export default function EarthAnalysisBoard({ state, mode, variable, driver, scope, isZh, onRetry }) {
  const model = useMemo(() => state?.status === 'ready'
    ? buildEarthAnalysisBoard({ mode, suite: state.data, spatial: state.data, variable, driver, scope, isZh })
    : null, [state, mode, variable, driver, scope, isZh]);
  if (state?.status === 'error') return <ChartRequestError isZh={isZh} onRetry={onRetry} />;
  if (state?.status === 'unsupported') return <div className="analysis-state" role="status">{reasonText(state.reason, isZh)}</div>;
  if (state?.status !== 'ready') return <div className="analysis-state" role="status">{isZh ? '正在加载主题图表…' : 'Loading theme charts…'}</div>;
  const note = mode === 'dynamics'
    ? <SpatialDiagnosticsGuide variableId={variable} units={state.data?.units} isZh={isZh} />
    : model?.note;
  return <AnalysisBoard key={mode} panels={model?.panels} note={note} isZh={isZh} />;
}
