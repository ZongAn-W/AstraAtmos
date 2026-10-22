import { useEffect, useMemo, useRef, useState } from 'react';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import KeyboardArrowDownRoundedIcon from '@mui/icons-material/KeyboardArrowDownRounded';
import C from '../../constants/colors';
import { useT } from '../../i18n';
import { Button, Panel } from '../../components/ui/Controls';
import MorphSubmitButton from '../../components/ui/MorphSubmitButton';
import { fmtNum } from '../../utils/fmt';
import { useSettings } from '../../contexts/SettingsContext';
import { buildTrainedModelParameterItems } from './trainedModelSelection';
import { buildCompareModelSummary, getCompareSelectionState } from './CompareTrainingModels/compareTrainingModelsData';
import { PREDICT_MODEL_MODE_COMPARE, PREDICT_MODEL_MODE_TRAINED } from './predictModelModes';
import { clampPredictionHorizon } from './predictionHorizon';
import { useTrainingTags } from '../../components/TrainingTags/useTrainingTags';
import { TagChips, TagFilter } from '../../components/TrainingTags/TagControls';
import { addVisibleSelection, filterTaggedTasks } from '../../components/TrainingTags/trainingTagFilters';
import { createMarsPredictionAdapter } from './singleModelAdapters';
import { PredictionOriginControl } from './PredictionPlanetAdapter';
import PredictStatus from './PredictStatus';
import './predictSidebar.css';

function SectionTitle({ title, subtitle }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div className="predict-sidebar__title">
        {title}
      </div>
      {subtitle ? (
        <div className="predict-sidebar__description">
          {subtitle}
        </div>
      ) : null}
    </div>
  );
}

function TrainedModelDropdown({
  options,
  currentOption,
  value,
  onChange,
  disabled,
  loading,
  isLight,
  isZh,
}) {
  const rootRef = useRef(null);
  const [open, setOpen] = useState(false);
  const selectedOption = options.find((option) => option.id === Number(value)) || currentOption || null;
  const hasOptions = options.length > 0;
  const isDisabled = disabled || loading || !hasOptions;
  const displayText = loading
    ? (isZh ? '正在加载训练模型...' : 'Loading trained models...')
    : selectedOption
      ? `${selectedOption.label} · #${selectedOption.id}`
      : (isZh ? '选择已完成训练模型' : 'Select a completed model');
  const panelBg = 'var(--surface-1)';
  const itemBg = 'var(--surface-2)';
  const activeBg = 'var(--surface-3)';

  useEffect(() => {
    if (isDisabled) setOpen(false);
  }, [isDisabled]);

  useEffect(() => {
    if (!open) return undefined;
    const handleOutsideClick = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [open]);

  return (
    <div
      ref={rootRef}
      onKeyDown={(event) => {
        if (event.key === 'Escape') setOpen(false);
      }}
      style={{ position: 'relative', width: '100%', minWidth: 0, zIndex: open ? 70 : 'auto' }}
    >
      <button
        type="button"
        disabled={isDisabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={displayText}
        onClick={() => setOpen((valueNow) => !valueNow)}
        style={{
          width: '100%',
          minWidth: 0,
          minHeight: 'var(--control-height)',
          boxSizing: 'border-box',
          padding: '11px 12px',
          borderRadius: 'var(--radius-control)',
          border: `1px solid ${open ? 'var(--line-active)' : 'var(--line-default)'}`,
          background: 'var(--surface-2)',
          color: isDisabled ? 'var(--text-disabled)' : 'var(--text-primary)',
          fontSize: 'calc(var(--type-control) * var(--font-scale, 1))',
          fontWeight: 600,
          fontFamily: 'var(--font-body)',
          outline: 'none',
          cursor: isDisabled ? 'not-allowed' : 'pointer',
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) auto',
          alignItems: 'center',
          gap: 10,
          textAlign: 'left',
          boxShadow: open ? '0 0 0 3px color-mix(in srgb, var(--line-active) 18%, transparent)' : 'none',
          transition: 'border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease',
        }}
      >
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {displayText}
        </span>
        <KeyboardArrowDownRoundedIcon
          aria-hidden="true"
          sx={{
            color: open ? 'var(--brand-ice)' : 'var(--text-secondary)',
            fontSize: 18,
            transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.18s ease, color 0.18s ease',
            flexShrink: 0,
          }}
        />
      </button>

      {open ? (
        <div
          role="listbox"
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: 'calc(100% + 8px)',
            zIndex: 80,
            padding: 8,
            boxSizing: 'border-box',
            borderRadius: 'var(--radius-panel)',
            border: '1px solid var(--line-default)',
            background: panelBg,
            boxShadow: 'var(--shadow-panel)',
            maxHeight: 260,
            overflowY: 'auto',
            overflowX: 'hidden',
          }}
        >
          {options.map((option) => {
            const active = option.id === Number(value);
            const summary = buildCompareModelSummary(option.task);
            return (
              <button
                key={option.id}
                type="button"
                role="option"
                aria-selected={active}
                title={option.label}
                onClick={() => {
                  onChange(option.id);
                  setOpen(false);
                }}
                style={{
                  width: '100%',
                  minWidth: 0,
                  boxSizing: 'border-box',
                  padding: '10px 11px',
                  borderRadius: 'var(--radius-control)',
                  border: '1px solid transparent',
                  boxShadow: active ? 'inset 2px 0 var(--line-active)' : 'none',
                  background: active ? activeBg : 'transparent',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                  display: 'grid',
                  gridTemplateColumns: 'minmax(0, 1fr) auto',
                  gap: 10,
                  alignItems: 'center',
                  textAlign: 'left',
                }}
                onMouseEnter={(event) => {
                  if (!active) event.currentTarget.style.background = itemBg;
                }}
                onMouseLeave={(event) => {
                  if (!active) event.currentTarget.style.background = 'transparent';
                }}
              >
                <span style={{ minWidth: 0 }}>
                  <span style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 'calc(var(--type-control) * var(--font-scale, 1))', fontWeight: 600 }}>
                    {option.label}
                  </span>
                  <span style={{ display: 'block', marginTop: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-secondary)', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', lineHeight: 1.5 }}>
                    #{summary.taskId} · {summary.architecture} · {summary.inputChannelText}
                  </span>
                  <span style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-secondary)', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', lineHeight: 1.5 }}>
                    {summary.modelSource} · W{summary.window || '--'} · H{summary.horizon || '--'} · {summary.dataSource}
                  </span>
                  <TagChips tags={option.task.tags} />
                </span>
                {active ? (
                  <CheckRoundedIcon aria-hidden="true" sx={{ color: 'var(--brand-ice)', fontSize: 16, flexShrink: 0 }} />
                ) : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function ModelSourceControl({
  modelMode,
  trainingModelOptions,
  selectedTrainingTaskId,
  setSelectedTrainingTaskId,
  selectedCompareTrainingTaskIds,
  setSelectedCompareTrainingTaskIds,
  trainingTasksLoading,
  trainingTasksError,
  onRetryTrainingTasks,
  selectedTrainingOption,
  requestContextLocked,
  isLight,
  isZh,
}) {
  const hasTrainingModels = trainingModelOptions.length > 0;
  const tagState = useTrainingTags();
  const [tagFilter, setTagFilter] = useState({ tagIds: [], untagged: false });
  useEffect(() => {
    if (tagState.loading || tagState.error) return;
    const known = new Set(tagState.tags.map(tag => tag.id));
    setTagFilter(prev => prev.tagIds.some(id => !known.has(id)) ? { ...prev, tagIds: prev.tagIds.filter(id => known.has(id)) } : prev);
  }, [tagState.tags, tagState.loading, tagState.error]);
  const taggedOptions = useMemo(() => {
    const visible = new Set(filterTaggedTasks(trainingModelOptions.map(option => option.task), tagFilter).map(task => task.id));
    return trainingModelOptions.filter(option => visible.has(option.id));
  }, [trainingModelOptions, tagFilter]);
  const parameterItems = buildTrainedModelParameterItems(selectedTrainingOption?.task, { isZh });
  const [searchTerm, setSearchTerm] = useState('');
  const normalizedSearch = searchTerm.trim().toLowerCase();
  const filteredCompareOptions = useMemo(() => (
    taggedOptions.filter((option) => {
      if (!normalizedSearch) return true;
      const summary = buildCompareModelSummary(option.task);
      return [
        option.label,
        String(option.id),
        summary.architecture,
        summary.inputChannelText,
        summary.dataSource,
      ].join(' ').toLowerCase().includes(normalizedSearch);
    })
  ), [normalizedSearch, taggedOptions]);
  const selectedCompareOptions = trainingModelOptions.filter(option => selectedCompareTrainingTaskIds.includes(option.id));
  const hiddenSelectedCount = selectedCompareOptions.filter(option => !filteredCompareOptions.some(visible => visible.id === option.id)).length;
  const compareSelection = getCompareSelectionState(selectedCompareTrainingTaskIds);


  return (
    <Panel id="prediction-model-selection" tabIndex={-1} className="prediction-panel prediction-model-selection"
      aria-label={isZh ? '模型选择' : 'Model selection'}>
      <SectionTitle
        title={isZh ? '模型选择' : 'Models'}
        subtitle={
          modelMode === PREDICT_MODEL_MODE_COMPARE
            ? (isZh ? '选择多个已完成训练任务，比较完整测试集表现。' : 'Compare completed training tasks across the full test set.')
            : modelMode === PREDICT_MODEL_MODE_TRAINED
            ? (isZh ? '使用训练页面已完成任务的模型权重进行预测。' : 'Use weights produced by a completed training task.')
            : ''
        }
        accent={modelMode === PREDICT_MODEL_MODE_COMPARE ? C.green : C.blue}
      />

      <TagFilter tags={tagState.tags} {...tagFilter} onChange={setTagFilter} isZh={isZh} disabled={tagState.loading || requestContextLocked} />
      {trainingTasksError ? (
        <div role="alert" className="training-tag-toolbar" style={{ color: C.mars, marginTop: 10 }}>
          <span>{isZh ? '训练任务加载失败，无法确认可用模型。' : 'Training models could not be loaded, so available models are unknown.'}</span>
          {onRetryTrainingTasks ? (
            <button className="training-tag-button" type="button" onClick={onRetryTrainingTasks} disabled={trainingTasksLoading}>
              {isZh ? '重试加载模型' : 'Retry loading models'}
            </button>
          ) : null}
        </div>
      ) : null}
      {tagState.error && <div role="alert" className="training-tag-toolbar" style={{ color: C.mars }}>
        {tagState.error}<button className="training-tag-button" onClick={() => tagState.refresh().catch(() => {})}>{isZh ? '重试标签' : 'Retry tags'}</button>
      </div>}

      {modelMode === PREDICT_MODEL_MODE_TRAINED ? (
        <div style={{ marginTop: 14, display: 'grid', gap: 8 }}>
          <TrainedModelDropdown
            options={taggedOptions}
            currentOption={selectedTrainingOption}
            value={selectedTrainingTaskId || ''}
            disabled={requestContextLocked || trainingTasksLoading || !hasTrainingModels}
            loading={trainingTasksLoading}
            isLight={isLight}
            isZh={isZh}
            onChange={(nextId) => setSelectedTrainingTaskId(Number(nextId) || null)}
          />

          <div className="predict-sidebar__description">
            {trainingTasksError
              ? (isZh ? '无法确认可用模型，请重试加载。' : 'Available models are unknown; retry loading models.')
              : hasTrainingModels
              ? `${isZh ? '当前模型' : 'Current model'}: ${selectedTrainingOption?.label || '--'}`
              : (isZh ? '暂无可用模型。请先在模型训练中完成一个实验，再返回选择模型。' : 'No model available. Complete an experiment in Model Training, then select its model here.')}
          </div>
          <TagChips tags={selectedTrainingOption?.task.tags} />
          {selectedTrainingOption && !taggedOptions.some(option => option.id === selectedTrainingOption.id) && (
            <div className="training-tag-hint" role="status">{isZh ? '当前模型不在筛选结果内，仍保留选中。' : 'The current model is outside these filters and remains selected.'}</div>
          )}
          {hasTrainingModels && taggedOptions.length === 0 && <div className="training-tag-hint">{isZh ? '没有匹配标签的可用模型。' : 'No available models match these tags.'}</div>}

          {parameterItems.length > 0 ? (
            <div
              style={{
                marginTop: 6,
                padding: 12,
                borderTop: '1px solid var(--line-subtle)',
              }}
            >
              <div style={{ color: C.ice, fontSize: 'calc(12px * var(--font-scale, 1))', fontWeight: 700, marginBottom: 10 }}>
                {isZh ? '所选模型参数' : 'Selected model parameters'}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                {parameterItems.map((item) => (
                  <div
                    key={item.label}
                    style={{
                      minWidth: 0,
                      padding: '9px 10px',
                      borderRadius: 'var(--radius-control)',
                      background: 'var(--surface-2)',
                    }}
                  >
                    <div style={{ color: 'var(--text-secondary)', fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', fontWeight: 600, letterSpacing: 0 }}>
                      {item.label}
                    </div>
                    <div
                      title={item.value}
                      style={{
                        color: C.ice,
                        fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))',
                        fontWeight: 700,
                        lineHeight: 1.45,
                        marginTop: 5,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {item.value}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {modelMode === PREDICT_MODEL_MODE_COMPARE ? (
        <div style={{ marginTop: 14, display: 'grid', gap: 10 }}>
          <input
            type="search"
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder={isZh ? '搜索模型名 / 架构 / 通道' : 'Search name, architecture, channels'}
            disabled={trainingTasksLoading || !hasTrainingModels}
            style={{
              width: '100%',
              minWidth: 0,
              padding: '11px 12px',
              borderRadius: 'var(--radius-control)',
              border: '1px solid var(--line-default)',
              background: 'var(--surface-2)',
              color: C.ice,
              fontSize: 'calc(var(--type-control) * var(--font-scale, 1))',
              fontWeight: 600,
              fontFamily: 'var(--font-body)',
              outline: 'none',
              boxSizing: 'border-box',
            }}
          />

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <Button size="compact"
              type="button"
              disabled={requestContextLocked || !filteredCompareOptions.length}
              onClick={() => setSelectedCompareTrainingTaskIds(previous => addVisibleSelection(previous, filteredCompareOptions.map((option) => option.id)))}
            >
              {isZh ? '全选当前结果' : 'Select visible'}
            </Button>
            <Button size="compact"
              type="button"
              disabled={requestContextLocked || compareSelection.count === 0}
              onClick={() => setSelectedCompareTrainingTaskIds([])}
            >
              {isZh ? '清空选择' : 'Clear selection'}
            </Button>
            <span style={{ marginLeft: 'auto', color: compareSelection.canCompare ? 'var(--status-success)' : 'var(--text-secondary)', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', fontWeight: 600 }}>
              {isZh ? `已选 ${compareSelection.count}，筛选外 ${hiddenSelectedCount}` : `${compareSelection.count} selected, ${hiddenSelectedCount} hidden`}
            </span>
          </div>

          {selectedCompareOptions.length > 0 && <div className="training-tag-chips" aria-label={isZh ? '已选对比模型' : 'Selected comparison models'}>
            {selectedCompareOptions.map(option => <button type="button" key={option.id} className="training-tag-button" disabled={requestContextLocked}
              aria-label={isZh ? `取消选择 ${option.label}` : `Deselect ${option.label}`}
              onClick={() => setSelectedCompareTrainingTaskIds(ids => ids.filter(id => id !== option.id))}>{option.label} ×</button>)}
          </div>}

          <div style={{ maxHeight: 280, overflowY: 'auto', overflowX: 'hidden', paddingRight: 4, display: 'grid', gap: 8 }}>
            {trainingTasksLoading ? (
              <div className="predict-sidebar__description" role="status" style={{ padding: 16, background: 'var(--surface-2)', textAlign: 'center' }}>
                {isZh ? '正在加载训练模型…' : 'Loading trained models…'}
              </div>
            ) : null}
            {filteredCompareOptions.map((option) => {
              const active = selectedCompareTrainingTaskIds.includes(option.id);
              const summary = buildCompareModelSummary(option.task);
              return (
                <label
                  key={option.id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'auto 1fr',
                    gap: 10,
                    padding: '10px 11px',
                    borderRadius: 'var(--radius-control)',
                    border: '1px solid var(--line-subtle)',
                    boxShadow: active ? 'inset 2px 0 var(--line-active)' : 'none',
                    background: active ? 'var(--surface-3)' : 'var(--surface-2)',
                    cursor: 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={active}
                    disabled={requestContextLocked}
                    onChange={() => {
                      setSelectedCompareTrainingTaskIds((prev) => (
                        prev.includes(option.id)
                          ? prev.filter((id) => id !== option.id)
                          : [...prev, option.id]
                      ));
                    }}
                    style={{ accentColor: 'var(--brand-ice)', marginTop: 2 }}
                  />
                  <span style={{ minWidth: 0 }}>
                    <span style={{ display: 'block', color: C.ice, fontSize: 'calc(var(--type-control) * var(--font-scale, 1))', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {summary.modelName}
                    </span>
                    <span style={{ display: 'block', color: 'var(--text-secondary)', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', lineHeight: 1.5, marginTop: 3 }}>
                      #{summary.taskId} · {summary.architecture} · {summary.inputChannelText}
                    </span>
                    <span style={{ display: 'block', color: 'var(--text-secondary)', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', lineHeight: 1.5 }}>
                      {summary.modelSource} · W{summary.window || '--'} · H{summary.horizon || '--'} · {summary.dataSource}
                    </span>
                    <TagChips tags={option.task.tags} />
                  </span>
                </label>
              );
            })}

            {!trainingTasksLoading && !trainingTasksError && filteredCompareOptions.length === 0 ? (
              <div className="predict-sidebar__description" style={{ padding: 16, background: 'var(--surface-2)', textAlign: 'center' }}>
                {hasTrainingModels
                  ? (isZh ? '没有匹配的训练模型。' : 'No matching trained models.')
                  : (isZh ? '暂无可对比的已完成训练模型。' : 'No completed trained models are available.')}
              </div>
            ) : null}
          </div>

          <div style={{ color: compareSelection.canCompare ? 'var(--text-secondary)' : 'var(--status-warning)', fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', lineHeight: 1.5 }}>
            {compareSelection.canCompare
              ? (isZh ? '点击“开始对比”查看完整测试集指标。' : 'Start comparison to view full test-set metrics.')
              : (isZh ? '至少选择 2 个模型才允许开始对比。' : 'Select at least 2 models to start comparison.')}
          </div>
        </div>
      ) : null}
    </Panel>
  );
}

function ModelHyperparams({ t, isZh }) {
  const params = [
    { label: 'Epochs', val: '30', color: C.mars },
    { label: 'Layers', val: '3 (ST-LSTM)', color: C.blue },
    { label: 'Hidden', val: '[64, 64, 64]', color: C.blue },
    { label: 'Filter', val: '3 x 3', color: C.green },
    { label: 'Window', val: '3', color: C.purple },
    { label: 'Horizon', val: '3', color: C.purple },
    { label: 'LR', val: '0.001', color: '#d9a441' },
    { label: 'Batch', val: '32', color: C.ice60 },
  ];

  return (
    <Panel className="prediction-panel">
      <SectionTitle
        title={t('predict.hyperTitle')}
        subtitle={isZh ? '当前预测流程使用的模型配置。' : 'Model configuration used for the current prediction flow.'}
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        {params.map((p) => (
          <div key={p.label} style={{ padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 'var(--radius-control)' }}>
            <div style={{ fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: 0 }}>
              {p.label}
            </div>
            <div style={{ fontSize: 'calc(12px * var(--font-scale, 1))', color: p.color, fontWeight: 700, marginTop: 5 }}>
              {p.val}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

export default function PredictSidebar({
  adapter,
  originValue,
  onOriginChange,
  originDisabled = false,
  originHint,
  contextLoading = false,
  contextError,
  deferContext = false,
  runDisabled = false,
  isLight,
  loading,
  requestContextLocked = false,
  error,
  modelMode,
  trainingModelOptions = [],
  selectedTrainingTaskId,
  setSelectedTrainingTaskId,
  selectedCompareTrainingTaskIds = [],
  setSelectedCompareTrainingTaskIds,
  trainingTasksLoading = false,
  trainingTasksError = null,
  onRetryTrainingTasks,
  selectedTrainingOption,
  analysisVisibility = {},
  lsStart,
  setLsStart,
  predStep,
  setPredStep,
  predictionHorizonLimit,
  selectedVars,
  toggleVar,
  VARIABLES,
  handlePredict,
  precision,
}) {
  const t = useT();
  const { settings } = useSettings();
  const isZh = settings?.language !== 'en';
  const canShowInputVariables = analysisVisibility.inputVariables !== false;
  const canShowSystemHyperparams = analysisVisibility.systemHyperparams !== false;
  const isCompareMode = modelMode === PREDICT_MODEL_MODE_COMPARE;
  const compareSelection = getCompareSelectionState(selectedCompareTrainingTaskIds);
  const planetAdapter = adapter || createMarsPredictionAdapter({ isZh, horizonLimit: predictionHorizonLimit });
  const usesSavedTestMetrics = isCompareMode && planetAdapter.id === 'earth';
  const invalidPredictionHorizon = !usesSavedTestMetrics && !deferContext
    && (predictionHorizonLimit == null || predStep > predictionHorizonLimit);
  const disabledReason = loading ? null
    : contextLoading ? (isZh ? '正在读取预测起点，请稍候' : 'Loading forecast origins; please wait')
    : contextError ? (isZh ? '预测起点读取失败，请重试' : 'Forecast origins could not be loaded; please retry')
    : isCompareMode && !compareSelection.canCompare ? (isZh ? '请至少选择两个已完成的训练模型' : 'Select at least two completed training models')
    : !usesSavedTestMetrics && !deferContext && predictionHorizonLimit == null ? (isZh ? '请选择已完成的训练模型' : 'Select a completed training model')
    : !usesSavedTestMetrics && !deferContext && predStep > predictionHorizonLimit ? (isZh ? '预测步长不能超过模型输出窗口' : 'The prediction horizon exceeds the model output window')
    : runDisabled ? (isZh ? '请选择有效范围内的预测起点' : 'Select a forecast origin within the available range')
    : null;

  return (
    <div className="predict-sidebar">
      <ModelSourceControl
        modelMode={modelMode}
        trainingModelOptions={trainingModelOptions}
        selectedTrainingTaskId={selectedTrainingTaskId}
        setSelectedTrainingTaskId={setSelectedTrainingTaskId}
        selectedCompareTrainingTaskIds={selectedCompareTrainingTaskIds}
        setSelectedCompareTrainingTaskIds={setSelectedCompareTrainingTaskIds}
        trainingTasksLoading={trainingTasksLoading}
        trainingTasksError={trainingTasksError}
        onRetryTrainingTasks={onRetryTrainingTasks}
        selectedTrainingOption={selectedTrainingOption}
        requestContextLocked={requestContextLocked}
        isLight={isLight}
        isZh={isZh}
      />

      <Panel className="prediction-panel">
        <SectionTitle
          title={t('predict.sidebar.predictionControl')}
          subtitle={usesSavedTestMetrics
            ? (isZh ? '完整测试集指标' : 'Full test-set metrics')
            : isCompareMode
            ? (isZh ? '选择对比使用的测试集预测步长。' : 'Choose the test-set horizon used for comparison.')
            : planetAdapter.horizon.hint}
          accent={isCompareMode ? C.green : C.mars}
        />

        {!isCompareMode ? <div style={{ marginBottom: 16 }}>
          <PredictionOriginControl adapter={planetAdapter} value={originValue ?? lsStart}
            onChange={onOriginChange || setLsStart} disabled={originDisabled || requestContextLocked} />
          {originHint ? <p className="prediction-control-hint">{originHint}</p> : null}
        </div> : null}
        {usesSavedTestMetrics ? <div style={{ marginBottom: 12 }}>
          <div className="predict-sidebar__description">
            {isZh ? '测试集输出窗口' : 'Test-set output window'}
          </div>
          <p className="predict-sidebar__saved-horizon" data-saved-comparison-horizon>
            {predictionHorizonLimit != null ? planetAdapter.horizon.label
              : compareSelection.count === 0 ? (isZh ? '尚未选择模型' : 'No models selected')
              : (isZh ? '所选模型的输出窗口不同或缺失' : 'Selected output windows differ or are missing')}
          </p>
        </div> : <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', color: 'var(--text-secondary)', fontWeight: 600, marginBottom: 8 }}>
            {t('predict.horizon')}
          </div>
          <input
            type="number"
            min="1"
            max={predictionHorizonLimit}
            step="1"
            value={predStep}
            readOnly={!planetAdapter.horizon.editable}
            disabled={requestContextLocked || predictionHorizonLimit == null}
            aria-label={t('predict.horizon')}
            onChange={(event) => {
              const nextHorizon = clampPredictionHorizon(event.target.value, predictionHorizonLimit);
              if (nextHorizon != null) setPredStep(nextHorizon);
            }}
            style={{
              width: '100%',
              boxSizing: 'border-box',
              padding: '10px 12px',
              borderRadius: 8,
              border: `1px solid ${C.border}`,
              background: C.bgMuted,
              color: C.ice,
              minHeight: 'var(--control-height)',
              fontSize: 'calc(var(--type-control) * var(--font-scale, 1))',
              fontWeight: 700,
              fontFamily: 'var(--font-display)',
              cursor: predictionHorizonLimit == null ? 'not-allowed' : 'text',
              opacity: predictionHorizonLimit == null ? 0.65 : 1,
            }}
          />
          <div className="predict-sidebar__description">
            {!planetAdapter.horizon.editable ? planetAdapter.horizon.label : predictionHorizonLimit == null
              ? (isZh ? '请选择具有有效输出窗口的训练模型' : 'Select a trained model with a valid output horizon')
              : planetAdapter.horizon.hint}
          </div>
        </div>}

        <div className="predict-sidebar__actions">
          <MorphSubmitButton
            label={isCompareMode
              ? (isZh ? '开始对比' : 'Start comparison')
              : t('predict.runBtn')}
            loadingLabel={t('predict.runningBtn')}
            successLabel={isZh ? '已提交' : 'Submitted'}
            onAction={handlePredict}
            disabled={loading || runDisabled
              || invalidPredictionHorizon
              || (isCompareMode && !compareSelection.canCompare)}
            className="predict-sidebar__action"
            aria-describedby={disabledReason ? 'prediction-run-disabled-reason' : undefined}
          />
          {disabledReason ? <p id="prediction-run-disabled-reason" className="predict-sidebar__disabled-reason" role="status">
            {disabledReason}
          </p> : null}
        </div>

        <PredictStatus loading={contextLoading} message={contextLoading ? (isZh ? '正在读取预测起点…' : 'Loading forecast origins…') : null}
          error={contextError} onRetry={handlePredict} retryLabel={isZh ? '重试' : 'Retry'} />
        <PredictStatus error={error} onRetry={handlePredict} retryLabel={isCompareMode
          ? (isZh ? '重试对比' : 'Retry comparison') : (isZh ? '重试预测' : 'Retry prediction')} />
      </Panel>

      {canShowInputVariables ? (
        <Panel className="prediction-panel">
        <SectionTitle
          title={t('predict.sidebar.inputVariables')}
          subtitle={isZh ? '选择参与预测的输入变量。' : 'Choose the input variables used in the model.'}
          accent={C.blue}
        />

        <div style={{ display: 'grid', gap: 8 }}>
          {VARIABLES.map((v) => {
            const active = selectedVars.includes(v.id);
            return (
              <label
                key={v.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-control)',
                  background: active ? 'var(--surface-3)' : 'var(--surface-2)',
                  border: '1px solid var(--line-subtle)',
                  boxShadow: active ? 'inset 2px 0 var(--line-active)' : 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
              >
                <input
                  type="checkbox"
                  checked={active}
                  disabled={requestContextLocked}
                  onChange={() => toggleVar(v.id)}
                  style={{ accentColor: 'var(--brand-ice)' }}
                />
                <span style={{ flex: 1, fontSize: 'calc(var(--type-control) * var(--font-scale, 1))', color: active ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                  {v.label}
                </span>
              </label>
            );
          })}
        </div>
        </Panel>
      ) : null}

      {canShowSystemHyperparams ? <ModelHyperparams t={t} isZh={isZh} /> : null}

    </div>
  );
}
