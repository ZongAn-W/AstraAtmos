import { useEffect, useRef, useState, useId } from 'react';
import { useSettings } from '../../contexts/SettingsContext';
import { requestResearchExport, researchExportCapabilities } from '../../services/api';
import { researchExportDefaults, researchFigureSize, MAX_RESEARCH_CURVE_MODELS,
  initialResearchModelSelection, selectedResearchSources, researchSelectionError } from './researchExportModel';
import './researchExport.css';

function ExportPanel({ sources, sourceLabels, kind, step, metric, onClose, targetVariable, targetUnit }) {
  const { settings } = useSettings();
  const zh = settings.language !== 'en';
  const copy = (cn, en) => zh ? cn : en;
  const dialogRef = useRef(null);
  const titleId = useId();
  const nativeTarget = targetVariable && !['O3', 'TO3'].includes(targetVariable);
  const [options, setOptions] = useState(() => ({ ...researchExportDefaults(settings, sources[0].planet), ...(nativeTarget ? { unit: 'native' } : {}), ...researchFigureSize('double', kind) }));
  const [preset, setPreset] = useState('double');
  const [bundle, setBundle] = useState(false);
  const [capabilities, setCapabilities] = useState(null);
  const [preview, setPreview] = useState(null);
  const [readyKey, setReadyKey] = useState(null);
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');
  const [selectedIds, setSelectedIds] = useState(() => initialResearchModelSelection(sources, kind));
  const downloadController = useRef(null);
  const figureSources = selectedResearchSources(sources, selectedIds);
  const selectionError = researchSelectionError(kind, figureSources.length, settings.language);
  const payload = { sources: figureSources, kind, step, metric, options, bundle };
  const previewKey = JSON.stringify({ sources: figureSources, kind, step, metric, options });
  const change = (key, value) => setOptions((old) => ({ ...old, [key]: value }));

  function keepKeyboardFocus(event) {
    if (event.key !== 'Tab') return;
    const controls = [...dialogRef.current.querySelectorAll('button, input, select, [tabindex]')]
      .filter((element) => element.tabIndex >= 0 && !element.matches(':disabled') && element.getClientRects().length > 0);
    const first = controls[0];
    const last = controls.at(-1);
    if (!first) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault(); last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault(); first.focus();
    }
  }

  useEffect(() => {
    const dialog = dialogRef.current;
    const restoreFocus = document.activeElement;
    dialog.showModal();
    const controller = new AbortController();
    researchExportCapabilities({ signal: controller.signal }).then(setCapabilities).catch((e) => {
      if (e.name !== 'AbortError') setError(e.message);
    });
    return () => {
      controller.abort(); downloadController.current?.abort();
      if (dialog.open) dialog.close();
      if (restoreFocus?.isConnected) restoreFocus.focus();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let alive = true;
    let url;
    setReadyKey(null); setPreview(null); setError('');
    if (selectionError) { setBusy(false); return () => controller.abort(); }
    setBusy(true);
    const timer = setTimeout(async () => {
      try {
        const blob = await requestResearchExport(JSON.parse(previewKey), { preview: true, signal: controller.signal, language: settings.language });
        if (!alive) return;
        url = URL.createObjectURL(blob);
        setPreview(url); setReadyKey(previewKey);
      } catch (e) {
        if (alive && e.name !== 'AbortError') setError(e.message);
      } finally { if (alive) setBusy(false); }
    }, 350);
    return () => { alive = false; clearTimeout(timer); controller.abort(); if (url) URL.revokeObjectURL(url); };
  }, [previewKey, selectionError, settings.language]);

  async function download() {
    setDownloading(true); setError('');
    const controller = new AbortController(); downloadController.current = controller;
    try {
      const blob = await requestResearchExport(payload, { signal: controller.signal, language: settings.language });
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a'); link.href = url;
      link.download = `astraatmos-${kind}.${bundle ? 'zip' : options.format}`;
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { if (e.name !== 'AbortError') setError(e.message); }
    finally { if (!controller.signal.aborted) setDownloading(false); }
  }

  const field = (label, control) => <label className="research-export-field"><span>{label}</span>{control}</label>;
  return (
    <dialog ref={dialogRef} className="research-export-dialog" aria-labelledby={titleId} onCancel={onClose} onKeyDown={keepKeyboardFocus}>
      <header><div><h2 id={titleId}>{copy('导出科研图', 'Export scientific figure')}</h2>
        <p>{copy('仅导出已有结果；本次设置不修改全局偏好。', 'Export existing results. Overrides apply to this export only.')}</p></div>
        <button type="button" onClick={onClose} aria-label={copy('关闭', 'Close')}>×</button></header>
      <div className="research-export-body">
        <fieldset className="research-export-controls" disabled={downloading} aria-busy={downloading}>
          {kind === 'step_curves' && <fieldset className="research-export-models">
            <legend>{copy('本图模型', 'Models in this figure')} · {figureSources.length}/{MAX_RESEARCH_CURVE_MODELS}</legend>
            <p>{copy(`当前对比 ${sources.length} 个模型，请选择本图使用的 2–8 个模型。`, `Comparing ${sources.length} models. Select 2–8 for this figure.`)}</p>
            <div className="research-export-model-list">
              {sources.map((source, index) => {
                const checked = selectedIds.includes(source.id);
                return <label className="research-export-check" key={source.id}>
                  <input type="checkbox" checked={checked} disabled={!checked && figureSources.length >= MAX_RESEARCH_CURVE_MODELS}
                    onChange={(event) => setSelectedIds((ids) => event.target.checked ? [...ids, source.id] : ids.filter((id) => id !== source.id))} />
                  <span>{sourceLabels[index] || `${copy('模型', 'Model')} #${source.task_id}`}</span>
                </label>;
              })}
            </div>
          </fieldset>}
          {field(copy('版面尺寸', 'Figure size'), <select value={preset} onChange={(e) => {
            setPreset(e.target.value); if (e.target.value !== 'custom') setOptions((old) => ({ ...old, ...researchFigureSize(e.target.value, kind) }));
          }}><option value="single">{copy('单栏 · 85 mm', 'Single column · 85 mm')}</option><option value="double">{copy('双栏 · 180 mm', 'Double column · 180 mm')}</option><option value="custom">{copy('自定义', 'Custom')}</option></select>)}
          <div className="research-export-pair">
            {field(copy('宽 (mm)', 'Width (mm)'), <input type="number" min="70" max="240" step="1" value={options.width_mm} disabled={preset !== 'custom'} onChange={(e) => change('width_mm', Number(e.target.value))} />)}
            {field(copy('高 (mm)', 'Height (mm)'), <input type="number" min="55" max="260" step="1" value={options.height_mm} disabled={preset !== 'custom'} onChange={(e) => change('height_mm', Number(e.target.value))} />)}
          </div>
          {field(copy('图中文字', 'Figure language'), <select value={options.language} onChange={(e) => change('language', e.target.value)}><option value="zh">中文</option><option value="en">English</option></select>)}
          {field(copy('字体', 'Font'), <select value={options.font} onChange={(e) => change('font', e.target.value)}>{['auto', 'sans', 'serif'].map((font, i) => <option key={font} value={font}>{[copy('自动', 'Automatic'), copy('无衬线', 'Sans serif'), copy('衬线', 'Serif')][i]}{capabilities ? ` · ${capabilities.fonts[font].join(' / ')}` : ''}</option>)}</select>)}
          <div className="research-export-pair">
            {field(copy('字号 (pt)', 'Font size (pt)'), <input type="number" min="6" max="16" step="1" value={options.font_size} onChange={(e) => change('font_size', Number(e.target.value))} />)}
            {field(copy('线宽 (pt)', 'Line width (pt)'), <input type="number" min="0.3" max="3" step="0.1" value={options.line_width} onChange={(e) => change('line_width', Number(e.target.value))} />)}
          </div>
          {field(copy('色带', 'Colormap'), <select value={options.colormap} onChange={(e) => change('colormap', e.target.value)}>{['viridis', 'inferno', 'cividis', 'plasma', 'magma', 'jet', 'rdbu'].map((value) => <option key={value}>{value}</option>)}</select>)}
          {field(copy('单位', 'Unit'), <select value={options.unit} disabled={nativeTarget || sources[0].planet === 'earth'} onChange={(e) => change('unit', e.target.value)}>{nativeTarget ? <option value="native">{targetUnit || copy('原生单位', 'Native unit')}</option> : <><option value="um-atm">μm-atm</option><option value="DU">DU</option></>}</select>)}
          <div className="research-export-pair">
            {field(copy('文件格式', 'Format'), <select value={options.format} onChange={(e) => change('format', e.target.value)}>{['pdf', 'svg', 'png'].map((v) => <option key={v} value={v}>{v.toUpperCase()}</option>)}</select>)}
            {field('DPI', <select value={options.dpi} onChange={(e) => change('dpi', Number(e.target.value))}><option value="300">300</option><option value="600">600</option></select>)}
          </div>
          <label className="research-export-check"><input type="checkbox" checked={options.include_title} onChange={(e) => change('include_title', e.target.checked)} />{copy('添加标题／目标时间', 'Include title / target time')}</label>
          {kind === 'triptych' && <label className="research-export-check"><input type="checkbox" checked={options.panel_labels} onChange={(e) => change('panel_labels', e.target.checked)} />{copy('添加 (a)、(b)、(c)', 'Include (a), (b), (c)')}</label>}
          <label className="research-export-check"><input type="checkbox" checked={bundle} onChange={(e) => setBundle(e.target.checked)} />{copy('下载数据与 MATLAB 脚本 (.zip)', 'Include data and MATLAB script (.zip)')}</label>
        </fieldset>
        <div className="research-export-preview" aria-busy={busy}>
          <div role="status" aria-live="polite">{selectionError || (busy ? copy('正在生成预览…', 'Rendering preview…') : preview ? `${options.width_mm} × ${options.height_mm} mm · ${options.dpi} DPI` : copy('预览不可用', 'Preview unavailable'))}</div>
          {preview && <img src={preview} alt={copy('最终布局预览', 'Final figure layout preview')} />}
          <p>{kind === 'triptych' ? copy('参考与预测共用色阶；残差 = 预测 − 参考，色阶关于零对称。', 'Reference and prediction share limits. Residual = prediction - reference, with symmetric limits.') : kind === 'scatter' ? copy(`当前预测步 ${step + 1} 的全部格点；普通散点，等比例坐标。`, `All grid cells at forecast step ${step + 1}; ordinary scatter, equal axes.`) : kind === 'pfi' ? copy('PFI 来自最多 40 个抽样测试窗口，ΔR² 可为负；不代表当前预测窗口。', 'PFI uses up to 40 sampled test windows; ΔR² may be negative. It is independent of the current prediction window.') : copy('逐步指标来自各模型绑定的完整测试集；不同数据集或划分需分别解读。', 'Per-step metrics use each model’s full test set; interpret different datasets/splits separately.')}</p>
          {bundle && <p>{copy('数据包含真实坐标、单位和统计范围。MATLAB 重建脚本未执行验证。', 'Data includes real coordinates, units and statistical scope. MATLAB script execution is unverified.')}</p>}
        </div>
      </div>
      {error && <p className="research-export-error" role="alert">{copy('导出失败：', 'Export failed: ')}{error}</p>}
      <footer><button type="button" onClick={onClose}>{copy('关闭', 'Close')}</button><button type="button" disabled={Boolean(selectionError) || busy || downloading || readyKey !== previewKey} onClick={download}>{downloading ? copy('正在导出…', 'Exporting…') : copy('下载', 'Download')}</button></footer>
    </dialog>
  );
}

export default function ResearchExportButton({ sources = [], sourceLabels = [], kind, step = 0, metric = 'rmse', disabled = false, targetVariable, targetUnit }) {
  const { settings } = useSettings();
  const [open, setOpen] = useState(false);
  const valid = sources.length > 0 && sources.every(Boolean);
  const identity = JSON.stringify({ sources, kind, step, metric, disabled, targetVariable, targetUnit });
  useEffect(() => { setOpen(false); }, [identity]);
  return <>
    <button className="research-export-button" type="button" disabled={disabled || !valid}
      title={!valid ? (settings.language === 'en' ? 'Refresh the result to enable export' : '重新获取结果后可导出') : undefined}
      onClick={() => setOpen(true)}>{settings.language === 'en' ? 'Export scientific figure' : '导出科研图'}</button>
    {open && valid && <ExportPanel sources={sources} sourceLabels={sourceLabels} kind={kind} step={step} metric={metric} targetVariable={targetVariable} targetUnit={targetUnit} onClose={() => setOpen(false)} />}
  </>;
}
