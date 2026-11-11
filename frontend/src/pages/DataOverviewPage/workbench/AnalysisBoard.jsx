import React, { useEffect, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import { useSettings } from '../../../contexts/SettingsContext';
import { useOverviewLayout } from './OverviewShell.jsx';
import './analysisBoard.css';

function BoardPlot({ panel, focused, isZh }) {
  const host = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const { settings } = useSettings();
  const light = settings?.theme === 'light';
  const text = light ? '#233548' : '#dce8f4';
  const grid = light ? 'rgba(35,53,72,.12)' : 'rgba(170,197,224,.14)';
  useEffect(() => {
    const node = host.current;
    if (!node) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: Math.floor(entry.contentRect.width), height: Math.floor(entry.contentRect.height) });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  const semantic = panel.layout || {};
  const horizontalBars = panel.data?.find(trace => trace.type === 'bar' && trace.orientation === 'h');
  const heatmap = panel.data?.some(trace => trace.type === 'heatmap');
  const plotData = panel.data?.map(trace => trace.type === 'heatmap' ? {
    ...trace,
    colorbar: {
      ...trace.colorbar, orientation: 'h', x: 0, xanchor: 'left', y: -0.23, yanchor: 'top',
      len: 1, thickness: 10, outlinewidth: 0, tickfont: { size: 10, color: text },
      title: { text: typeof trace.colorbar?.title === 'string' ? trace.colorbar.title : trace.colorbar?.title?.text, side: 'bottom', font: { size: 11, color: text } },
    },
  } : trace);
  const axis = { automargin: true, gridcolor: grid, zerolinecolor: grid, tickfont: { color: text, size: 11 }, titlefont: { color: text, size: 12 }, nticks: focused ? 7 : 4 };
  return (
    <div className="analysis-board-plot" ref={host} aria-label={panel.title}>
      {panel.empty || !panel.data?.length ? (
        <div className="analysis-board-empty" role="status">{isZh ? '当前条件下暂无可用数据' : 'No data for these settings'}</div>
      ) : size.width > 0 && size.height > 0 ? (
        <Plot data={plotData} layout={{
          autosize: false, width: size.width, height: size.height,
          paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
          font: { family: 'Arial, sans-serif', color: text, size: 11 },
          margin: { l: 58, r: 16, t: 12, b: heatmap ? 110 : 44 },
          showlegend: panel.data.filter(trace => trace.showlegend !== false).length > 1,
          legend: { orientation: 'h', x: 0, y: 1.15, font: { size: 11, color: text } },
          ...semantic,
          xaxis: { ...axis, ...semantic.xaxis },
          yaxis: { ...axis, ...semantic.yaxis, ...(horizontalBars ? { title: '', type: 'category', tickmode: 'array', tickvals: horizontalBars.y, ticktext: horizontalBars.y.map(label => String(label).replace(/ (\d.*)$/, '<br>$1')) } : {}) },
        }} config={{ responsive: false, displayModeBar: focused, displaylogo: false, scrollZoom: false }}
          style={{ width: '100%', height: '100%' }} />
      ) : null}
    </div>
  );
}

/** A theme shares conditions; focusing a panel changes only its presentation. */
export default function AnalysisBoard({ panels = [], note = null, isZh = true }) {
  const [focusedId, setFocusedId] = useState(null);
  const { openPanel } = useOverviewLayout();
  const buttonRefs = useRef(new Map());
  const panelIds = panels.map(panel => panel.id).join(':');
  useEffect(() => { setFocusedId(null); }, [panelIds]);
  const focused = panels.some(panel => panel.id === focusedId) ? focusedId : null;
  useEffect(() => {
    if (!focused || openPanel) return undefined;
    const returnToBoard = (event) => {
      if (event.key !== 'Escape' || event.defaultPrevented || event.target.closest?.('[role="dialog"]')) return;
      setFocusedId(null);
      buttonRefs.current.get(focused)?.focus({ preventScroll: true });
    };
    document.addEventListener('keydown', returnToBoard);
    return () => document.removeEventListener('keydown', returnToBoard);
  }, [focused, openPanel]);
  const toggleFocus = (id) => {
    const next = focused === id ? null : id;
    setFocusedId(next);
    // The same button stays mounted, retaining keyboard focus across resizing.
    requestAnimationFrame(() => buttonRefs.current.get(id)?.focus({ preventScroll: true }));
  };
  return (
    <div className="analysis-board" data-focused-panel={focused || ''}>
      <div className="analysis-board-grid">
        {panels.map((panel, index) => (
          <section key={panel.id} className="analysis-board-panel" data-board-panel={panel.id}
            data-primary={index === 0} hidden={Boolean(focused && focused !== panel.id)}>
            <header className="analysis-board-panel-head">
              <h3>{panel.title}</h3>
              <button type="button" ref={node => {
                if (node) buttonRefs.current.set(panel.id, node); else buttonRefs.current.delete(panel.id);
              }} onClick={() => toggleFocus(panel.id)} aria-expanded={focused === panel.id}
                aria-label={focused === panel.id
                  ? (isZh ? `返回组合看板：${panel.title}` : `Back to board: ${panel.title}`)
                  : (isZh ? `放大：${panel.title}` : `Expand: ${panel.title}`)}>
                {focused === panel.id ? (isZh ? '返回组合 ↙' : 'Back ↙') : (isZh ? '放大 ↗' : 'Expand ↗')}
              </button>
            </header>
            {(!focused || focused === panel.id) ? <BoardPlot panel={panel} focused={focused === panel.id} isZh={isZh} /> : null}
          </section>
        ))}
      </div>
      {note ? <div className="analysis-board-note">{note}</div> : null}
    </div>
  );
}
