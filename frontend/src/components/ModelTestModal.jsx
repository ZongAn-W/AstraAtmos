import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import Plot from 'react-plotly.js';
import C from '../constants/colors';
import { useT } from '../i18n';
import { useSettings } from '../contexts/SettingsContext';
import { performTaskAction } from '../services/api';
import { getMetricAggregationLabel, getSplitLabel } from '../pages/PredictPage/predictionMetricMeta';
import EarthDiagnosticPanel from '../pages/PredictPage/EarthDiagnosticPanel.jsx';
import { isEarthTask } from '../pages/PredictPage/earthPredictModel.js';
import { earthDiagnosticErrorText } from '../pages/PredictPage/earthDiagnosticsModel.js';
import { resolveEnvironmentTask, environmentVariableLabel } from '../utils/environmentVariables.js';
import { marsModelTestResponseMatchesTask, modelTestMetricUnit, modelTestScatterRange } from './modelTestModel.js';

/**
 * 模型测试结果弹窗
 * 展示核心指标卡片 & 真实值 vs 预测值的密度散点图
 */
export default function ModelTestModal({ taskId, task, onClose }) {
  const t = useT();
  const { settings } = useSettings();
  const isLight = settings.theme === 'light';
  const isZh = settings?.language !== 'en';
  const earth = isEarthTask(task);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [retry, setRetry] = useState(0);
  const target = resolveEnvironmentTask(data || task, earth ? 'earth' : 'mars');
  const referenceAxis = `${isZh ? '参考' : 'Reference'} ${environmentVariableLabel(target.target, target.planet, isZh)} (${target.unit})`;
  const predictionAxis = `${isZh ? '预测' : 'Prediction'} ${environmentVariableLabel(target.target, target.planet, isZh)} (${target.unit})`;
  const requestEpoch = useRef(0);

  // 锁定背景滚动
  useEffect(() => {
    if (taskId) {
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = 'auto';
      };
    }
  }, [taskId]);

  useEffect(() => {
    const epoch = ++requestEpoch.current;
    const controller = new AbortController();
    setData(null);
    setError(null);
    if (!taskId || earth) { setLoading(false); return () => controller.abort(); }
    setLoading(true);
    performTaskAction(taskId, 'test', { signal: controller.signal })
      .then(res => {
        if (controller.signal.aborted || epoch !== requestEpoch.current) return;
        if (res.status === 'success') {
          if (!marsModelTestResponseMatchesTask(res.data, task)) {
            setError(isZh
              ? '测试响应的预测目标与所选训练任务不一致，请重新加载任务。'
              : 'The test response target does not match the selected training task. Reload the task and retry.');
            return;
          }
          setData(res.data);
        } else {
          setError(res.message || 'Unknown error');
        }
      })
      .catch(err => {
        if (controller.signal.aborted || epoch !== requestEpoch.current) return;
        setError(earthDiagnosticErrorText(err));
      })
      .finally(() => { if (!controller.signal.aborted && epoch === requestEpoch.current) setLoading(false); });
    return () => controller.abort();
  }, [taskId, task, earth, retry, isZh]);

  if (!taskId) return null;

  const overlayStyle = {
    position: 'fixed',
    top: 0, left: 0, right: 0, bottom: 0,
    background: 'rgba(0,0,0,0.7)',
    backdropFilter: 'blur(8px)',
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20
  };

  const modalStyle = {
    width: '100%',
    maxWidth: 900,
    maxHeight: '90vh',
    background: isLight ? '#ffffff' : '#0f1423',
    borderRadius: 16,
    border: `1px solid ${isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)'}`,
    boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    position: 'relative',
    color: isLight ? '#333' : '#eee'
  };

  const headerStyle = {
    padding: '20px 24px',
    borderBottom: `1px solid ${isLight ? '#eee' : '#222'}`,
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  };

  const contentStyle = {
    padding: 24,
    overflowY: 'auto',
    flex: 1
  };

  const closeBtnStyle = {
    background: 'transparent',
    border: 'none',
    color: isLight ? '#999' : '#666',
    fontSize: 'calc(24px * var(--font-scale, 1))',
    cursor: 'pointer',
    padding: 4,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  };

  // 渲染指标卡片
  const renderMetrics = (metrics) => {
    const items = [
      { label: 'R²', value: metrics.r2?.toFixed(4), color: C.blue },
      { label: 'MSE', unit: modelTestMetricUnit('mse', target.unit), value: metrics.mse?.toFixed(6), color: C.mars },
      { label: 'RMSE', unit: modelTestMetricUnit('rmse', target.unit), value: metrics.rmse?.toFixed(4), color: C.ice },
      { label: 'MAPE', value: metrics.mape?.toFixed(2) + '%', color: '#ff9800' },
      { label: 'SMAPE', value: metrics.smape?.toFixed(2) + '%', color: C.green }
    ];

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {items.map(it => (
          <div key={it.label} style={{
            padding: '14px 18px',
            borderRadius: 12,
            background: isLight ? '#f5f7fa' : 'rgba(255,255,255,0.03)',
            border: `1px solid ${isLight ? '#e1e4e8' : 'rgba(255,255,255,0.05)'}`,
            display: 'flex',
            flexDirection: 'column',
            gap: 2
          }}>
            <span style={{ fontSize: 'calc(10px * var(--font-scale, 1))', fontWeight: 700, opacity: 0.5, letterSpacing: 1 }}>{it.label}{it.unit ? ` (${it.unit})` : ''}</span>
            <span style={{ fontSize: 'calc(18px * var(--font-scale, 1))', fontWeight: 800, color: it.color }}>{it.value ?? 'N/A'}</span>
          </div>
        ))}
      </div>
    );
  };

  // 渲染 Plotly 散点图
  const renderScatter = (yTrue, yPred) => {
    if (!yTrue || yTrue.length === 0) return null;

    // Physical winds may be negative; range calculation also avoids spreading large arrays.
    const range = modelTestScatterRange(yTrue, yPred);
    
    return (
      <div style={{ width: '100%', height: 450, background: 'transparent' }}>
        <Plot
          data={[
            {
              x: yTrue,
              y: yPred,
              mode: 'markers',
              type: 'scattergl', // 使用 WebGL 提升海量点位渲染性能
              name: 'Samples',
              marker: {
                size: 4,
                opacity: 0.5,
                color: yPred,
                colorscale: isLight ? 'Viridis' : 'Inferno',
                showscale: false
              },
              hovertemplate: `${referenceAxis}: %{x:.2f}<br>${predictionAxis}: %{y:.2f}<extra></extra>`
            },
            {
              x: range,
              y: range,
              mode: 'lines',
              type: 'scatter',
              name: t('modelTest.scatter.perfectFit'),
              line: {
                color: isLight ? '#ff6b35' : '#ff4d4d',
                width: 2,
                dash: 'dash'
              },
              hoverinfo: 'none'
            }
          ]}
          layout={{
            autosize: true,
            plot_bgcolor: 'transparent',
            paper_bgcolor: 'transparent',
            margin: { l: 60, r: 20, t: 40, b: 60 },
            xaxis: {
              title: { text: referenceAxis, font: { size: 12, color: isLight ? '#666' : '#999' } },
              gridcolor: isLight ? '#eee' : '#1a1a1a',
              zeroline: false,
              color: isLight ? '#333' : '#eee',
              range,
            },
            yaxis: {
              title: { text: predictionAxis, font: { size: 12, color: isLight ? '#666' : '#999' } },
              gridcolor: isLight ? '#eee' : '#1a1a1a',
              zeroline: false,
              color: isLight ? '#333' : '#eee',
              range,
            },
            showlegend: false,
            font: { family: 'inherit' }
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    );
  };

  const modalContent = (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} role="dialog" aria-modal="true" aria-label={t('modelTest.title')} onClick={e => e.stopPropagation()}>
        <div style={headerStyle}>
          <div>
            <h2 style={{ margin: 0, fontSize: 'calc(18px * var(--font-scale, 1))', fontWeight: 800, color: C.mars }}>{t('modelTest.title')}</h2>
            <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', fontWeight: 700, opacity: 0.4, letterSpacing: 1, marginTop: 2 }}>{t('modelTest.subtitle')} — TASK #{taskId}</div>
          </div>
          <button style={closeBtnStyle} aria-label={isZh ? '关闭测试结果' : 'Close test results'} onClick={onClose}>&times;</button>
        </div>

        <div style={contentStyle}>
          {earth ? <EarthDiagnosticPanel key={taskId} taskId={taskId} identity={task} autoLoad /> : null}
          {!earth && loading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 400, gap: 16 }}>
              <div className="loading-spinner" style={{ width: 40, height: 40, border: `3px solid ${C.mars}22`, borderTop: `3px solid ${C.mars}`, borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
              <div style={{ fontSize: 'calc(14px * var(--font-scale, 1))', fontWeight: 600, opacity: 0.6 }}>{t('modelTest.loading')}</div>
            </div>
          )}

          {!earth && error && (
            <div style={{ padding: 40, textAlign: 'center' }}>
              <div style={{ fontSize: 'calc(48px * var(--font-scale, 1))', marginBottom: 16 }}>⚠️</div>
              <h3 style={{ margin: '0 0 8px 0', color: C.mars }}>{t('modelTest.error')}</h3>
              <p style={{ margin: 0, opacity: 0.6, fontSize: 'calc(14px * var(--font-scale, 1))' }}>{error}</p>
              <button type="button" onClick={() => setRetry((n) => n + 1)} style={{ marginTop: 16, cursor: 'pointer' }}>{isZh ? '重试' : 'Retry'}</button>
              <button onClick={onClose} style={{ marginTop: 24, padding: '8px 24px', borderRadius: 8, background: C.mars, color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 700 }}>{t('explore.upload.closeResult')}</button>
            </div>
          )}

          {!earth && data && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 240px), 1fr))', gap: 24, alignItems: 'start' }}>
              {/* Left Column: Metrics */}
              <div>
                <div style={{ fontSize: 'calc(13px * var(--font-scale, 1))', fontWeight: 800, marginBottom: 16, opacity: 0.8, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ width: 4, height: 14, background: C.mars, borderRadius: 2 }} />
                  {t('modelTest.metricsTitle')}
                </div>

                {data.metric_meta && (
                  <div style={{
                    margin: '-6px 0 16px',
                    padding: '8px 10px',
                    borderRadius: 8,
                    background: isLight ? '#f5f7fa' : 'rgba(255,255,255,0.03)',
                    color: isLight ? '#667085' : '#aab2c0',
                    fontSize: 'calc(10px * var(--font-scale, 1))',
                    lineHeight: 1.5
                  }}>
                    <div>{getMetricAggregationLabel(data.metric_meta, isZh)}</div>
                    {getSplitLabel(data.metric_meta, isZh) && (
                      <div>{getSplitLabel(data.metric_meta, isZh)}</div>
                    )}
                  </div>
                )}
                
                {data.metrics ? renderMetrics(data.metrics) : (
                  <div style={{ padding: '20px', textAlign: 'center', opacity: 0.5, fontStyle: 'italic' }}>
                    {t('modelTest.noMetrics')}
                  </div>
                )}
              </div>

              {/* Right Column: Visualization */}
              <div>
                <div style={{ fontSize: 'calc(13px * var(--font-scale, 1))', fontWeight: 800, marginBottom: 16, opacity: 0.8, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ width: 4, height: 14, background: C.blue, borderRadius: 2 }} />
                  {t('modelTest.visualTitle')}
                </div>

                <div style={{ 
                  borderRadius: 12, 
                  background: isLight ? '#fcfcfc' : 'rgba(0,0,0,0.2)', 
                  border: `1px solid ${isLight ? '#eee' : 'rgba(255,255,255,0.05)'}`,
                  padding: '10px'
                }}>
                  {renderScatter(data.y_true, data.y_pred)}
                </div>
              </div>
            </div>
          )}
        </div>
        
        <style>{`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
}
