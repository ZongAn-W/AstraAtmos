import { useT } from '../../i18n';
import { Button, Panel, SegmentedControl } from '../../components/ui/Controls';
import { FieldCanvas, LoadingBox, EmptyBox } from './PredictComponents';
import ResearchExportButton from './ResearchExportButton';
import { useSettings } from '../../contexts/SettingsContext';
import { DISPLAY_FIELD_KINDS, predictionPhysicalRange, predictionStepItems } from './predictionDisplayModel';
import './predictionDisplay.css';

function SegmentedTabs({ items, activeId, onChange }) {
  return (
    <SegmentedControl className="prediction-display__tabs" value={activeId}
      options={items.map(item => ({ value: item.id, label: item.label }))} onChange={onChange} />
  );
}

export default function PredictDisplay({
  viewMode,
  setViewMode,
  VIEW_MODES,
  results,
  activeHorizon,
  setActiveHorizon,
  loading,
  truthField,
  predField,
  residField,
  stepLs,
  setFullscreen3D,
  TRIPTYCH_PANELS,
  adapter,
  hasAvailableModels = false,
  modelsLoading = false,
  modelsError = null,
  onRetryModels,
}) {
  const t = useT();
  const { settings } = useSettings();
  const isZh = settings.language !== 'en';
  const physicalRange = predictionPhysicalRange(truthField, predField);
  const currentStep = (results && adapter?.stepLabel?.(results, activeHorizon))
    || (stepLs != null ? `Ls=${stepLs.toFixed(3)}°` : '');
  const defaultPanels = [
    { key: 'truth', title: t('predict.panels.truth'), mode: 'inferno' },
    { key: 'prediction', title: t('predict.panels.prediction'), mode: 'inferno' },
    { key: 'residual', title: t('predict.panels.residual'), mode: 'rdbu' },
  ];
  const panels = DISPLAY_FIELD_KINDS.map((key, index) => ({
    ...(TRIPTYCH_PANELS?.find((panel) => panel.key === key) || defaultPanels[index]), key,
    color: key === 'truth' ? 'var(--brand-ice)' : key === 'prediction' ? 'var(--brand-orange)' : 'var(--status-diagnostic)',
  }));
  const renderField = (fieldData, colorMode, height, colorRange) => adapter?.renderField
    ? adapter.renderField({ fieldData, colorMode, height, colorRange, precision: settings.precision, fullscreen: false })
    : <FieldCanvas fieldData={fieldData} colorMode={colorMode} h={height} colorRange={colorRange} />;
  const openFullscreen = (fieldData, colorMode, kind, colorRange) => setFullscreen3D({
    fieldData, colorMode, kind, colorRange,
  });
  const fieldTitle = (kind) => adapter?.fieldTitle?.(kind, isZh) || t(`predict.panels.${kind}`);
  const isEmpty = !loading && !truthField && !predField && !residField;

  // Reuse the sidebar's existing selector without changing model selection or request state.
  const openModelSelection = () => {
    const section = document.getElementById('prediction-model-selection');
    const selector = section?.querySelector('[aria-haspopup="listbox"]');
    section?.scrollIntoView({ block: 'nearest' });
    if (selector && !selector.disabled) {
      selector.focus({ preventScroll: true });
      if (selector.getAttribute('aria-expanded') !== 'true') selector.click();
    } else {
      section?.focus({ preventScroll: true });
    }
  };

  if (isEmpty) {
    return (
      <div className="prediction-display prediction-display--empty" data-testid="prediction-display" data-empty="true">
        <section className="prediction-display__empty" aria-label={isZh ? '预测准备' : 'Prepare prediction'}>
          <div className="prediction-display__empty-copy">
            <h3 className="prediction-display__empty-title">
              {isZh ? '选择已完成模型，开始预测' : 'Select a completed model to start predicting'}
            </h3>
            <p className="prediction-display__empty-description">
              {modelsError
                ? (isZh ? '训练任务加载失败，当前无法确认可用模型。' : 'Training models could not be loaded, so available models are unknown.')
                : (isZh ? '需要先选择具有有效输出窗口的训练模型' : 'Choose a trained model with a valid output horizon.')}
            </p>
          </div>
          {modelsLoading ? (
            <Button size="compact" disabled>{isZh ? '正在加载模型…' : 'Loading models…'}</Button>
          ) : modelsError ? (
            <Button size="compact" onClick={onRetryModels} disabled={!onRetryModels} className="prediction-display__empty-action">
              {isZh ? '重试加载模型' : 'Retry loading models'}
            </Button>
          ) : hasAvailableModels ? (
            <Button size="compact" onClick={openModelSelection} className="prediction-display__empty-action">
              {isZh ? '选择训练模型' : 'Select a model'}
            </Button>
          ) : (
            <a href="#/training" className="ui-button ui-button--secondary ui-button--compact prediction-display__empty-action">
              {isZh ? '去模型训练' : 'Go to Model Training'}
            </a>
          )}
        </section>
      </div>
    );
  }

  return (
    <div className="prediction-display" data-testid="prediction-display" data-empty="false">
      <SegmentedTabs
        items={VIEW_MODES}
        activeId={viewMode}
        onChange={setViewMode}
      />
      {adapter?.capabilities?.triptychExport !== false && (
        <ResearchExportButton sources={[results?.export_ref]} kind="triptych" step={activeHorizon} disabled={loading} targetVariable={adapter?.target} targetUnit={adapter?.unit} />
      )}

      {results && results.horizon > 1 && (
        <div className="prediction-display__steps">
          <span style={{ fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', color: 'var(--text-secondary)' }}>
            {t('predict.showStep')}
          </span>
          <SegmentedTabs
            items={predictionStepItems(results, adapter, (i) => `${t('predict.display.stepLabelFunc', { step: i + 1 })}${results.ls_values?.[i] != null ? ` · Ls=${results.ls_values[i].toFixed(3)}°` : ''}`)}
            activeId={String(activeHorizon)}
            onChange={(nextId) => setActiveHorizon(Number(nextId))}
          />
        </div>
      )}

      {viewMode === 'triptych' && (
        <div className="prediction-display__triptych">
          {panels.map((panel, i) => {
            const fieldData = i === 0 ? truthField : i === 1 ? predField : residField;
            return (
              <Panel key={panel.key} className="prediction-display__card" style={{ padding: 16 }}>
                <div className="prediction-display__header">
                  <div style={{ color: panel.color, fontSize: 'calc(14px * var(--font-scale, 1))', fontWeight: 700, fontFamily: 'var(--font-display)' }}>
                    {fieldTitle(panel.key)}
                  </div>
                  {currentStep && (
                    <div className="prediction-display__time" style={{ color: 'var(--text-secondary)', fontSize: 'calc(var(--type-micro) * var(--font-scale, 1))' }}>
                      {currentStep}
                    </div>
                  )}
                </div>

                {loading ? (
                  <LoadingBox h={220} />
                ) : fieldData ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {renderField(fieldData, panel.mode, 220, i < 2 ? physicalRange : undefined)}
                    <Button
                      type="button"
                      onClick={() => openFullscreen(fieldData, panel.mode, panel.key, i < 2 ? physicalRange : undefined)}
                      className="prediction-display__fullscreen-action"
                    >
                      {adapter?.fullscreenButtonLabel || t('predict.display.viewFullscreen')}
                    </Button>
                  </div>
                ) : (
                  <EmptyBox h={96} />
                )}
              </Panel>
            );
          })}
        </div>
      )}

      {viewMode !== 'triptych' && (() => {
        const isResid = viewMode === 'diff';
        const fd = viewMode === 'original' ? truthField : viewMode === 'prediction' ? predField : residField;
        const kind = viewMode === 'original' ? 'truth' : viewMode === 'prediction' ? 'prediction' : 'residual';
        const panelTitle = `${fieldTitle(kind)}${currentStep ? ` · ${currentStep}` : ''}`;
        const panelColor = viewMode === 'original' ? 'var(--brand-ice)' : viewMode === 'prediction' ? 'var(--brand-orange)' : 'var(--status-diagnostic)';

        return (
          <Panel className="prediction-display__card" style={{ padding: 20 }}>
            <div style={{ color: panelColor, fontSize: 'calc(15px * var(--font-scale, 1))', fontWeight: 700, fontFamily: 'var(--font-display)', marginBottom: 14 }}>
              {panelTitle}
            </div>

            {loading ? (
              <LoadingBox h={400} />
            ) : fd ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {renderField(fd, isResid ? 'rdbu' : 'inferno', 400, isResid ? undefined : physicalRange)}
                <Button
                  type="button"
                  onClick={() => openFullscreen(fd, isResid ? 'rdbu' : 'inferno', kind, isResid ? undefined : physicalRange)}
                  className="prediction-display__fullscreen-action"
                >
                  {adapter?.fullscreenButtonLabel || t('predict.display.viewFullscreen')}
                </Button>
              </div>
            ) : (
              <EmptyBox h={112} />
            )}
          </Panel>
        );
      })()}
    </div>
  );
}
