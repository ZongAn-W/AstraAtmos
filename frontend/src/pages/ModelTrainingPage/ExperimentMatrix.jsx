import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import Popper from '@mui/material/Popper';
import ClickAwayListener from '@mui/material/ClickAwayListener';
import EditRoundedIcon from '@mui/icons-material/EditRounded';
import SettingsRoundedIcon from '@mui/icons-material/SettingsRounded';
import SaveRoundedIcon from '@mui/icons-material/SaveRounded';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import { createTrainingTag, replaceTrainingTaskTags, updateTrainingTaskTags } from '../../services/api';
import { TagChips, TagPicker } from '../../components/TrainingTags/TagControls';
import { TagManager } from '../../components/TrainingTags/TagDialogs';
import { validateTrainedModelName, normalizeTrainedModelName } from './trainedModelRename';
import ExperimentMatrixActions from './ExperimentMatrixActions';
import {
  filterMatrixTasks,
  formatMatrixValue,
  getMatrixPropertyDefinitions,
  getMatrixStorageKey,
  getMatrixValue,
  MATRIX_DEFAULT_COLUMNS,
  MATRIX_GROUPS,
  MATRIX_STATUS_FILTERS,
  restoreMatrixColumnPreferences,
  serializeMatrixColumnPreferences,
  sortMatrixTasks,
} from './experimentMatrixModel';
import './experimentMatrix.css';

const STATUS_LABELS = {
  all: ['全部', 'All'], queued: ['排队中', 'Queued'], running: ['运行中', 'Running'],
  completed: ['已完成', 'Completed'], failed: ['失败', 'Failed'], cancelled: ['已取消', 'Cancelled'],
};

function readStoredPreferences(scope) {
  if (typeof window === 'undefined') return null;
  try { return window.localStorage.getItem(getMatrixStorageKey(scope)); } catch { return null; }
}

function writeStoredPreferences(scope, value) {
  if (typeof window === 'undefined') return;
  try { window.localStorage.setItem(getMatrixStorageKey(scope), value); } catch { /* storage can be unavailable in private browsing */ }
}

function stopRowEvent(event) {
  event.stopPropagation();
}

function NameCell({ task, tasks, isZh, onSave }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(task.custom_model_name || '');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const inputRef = useRef(null);
  useEffect(() => { if (editing) inputRef.current?.focus(); }, [editing]);
  const begin = () => { setDraft(task.custom_model_name || ''); setError(''); setEditing(true); };
  const cancel = () => { if (!saving) { setDraft(task.custom_model_name || ''); setError(''); setEditing(false); } };
  const save = async () => {
    const validation = validateTrainedModelName(draft, tasks, task.id, isZh ? 'zh' : 'en');
    if (validation) { setError(validation); return; }
    setSaving(true); setError('');
    try { await onSave(task, normalizeTrainedModelName(draft)); setEditing(false); }
    catch (saveError) { setError(saveError.message || (isZh ? '保存名称失败' : 'Could not save name')); }
    finally { setSaving(false); }
  };
  if (!editing) return <div className="experiment-matrix-name-cell">
    <span className="experiment-matrix-name" title={task.custom_model_name || ''}>{task.custom_model_name || (isZh ? '未命名实验' : 'Unnamed experiment')}</span>
    <button type="button" className="experiment-matrix-icon-button" title={isZh ? '编辑名称' : 'Edit name'} aria-label={isZh ? '编辑名称' : 'Edit name'} onClick={(event) => { stopRowEvent(event); begin(); }}><EditRoundedIcon fontSize="small" /></button>
  </div>;
  return <div className="experiment-matrix-name-editor" onClick={stopRowEvent}>
    <input ref={inputRef} value={draft} maxLength={255} aria-label={isZh ? '实验名称' : 'Experiment name'} aria-invalid={Boolean(error)} onChange={(event) => { setDraft(event.target.value); setError(''); }} onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); save(); } if (event.key === 'Escape') { event.preventDefault(); cancel(); } }} disabled={saving} />
    <span className="experiment-matrix-editor-actions">
      <button type="button" className="experiment-matrix-icon-button" title={isZh ? '保存名称' : 'Save name'} aria-label={isZh ? '保存名称' : 'Save name'} disabled={saving} onClick={save}><SaveRoundedIcon fontSize="small" /></button>
      <button type="button" className="experiment-matrix-icon-button" title={isZh ? '取消编辑' : 'Cancel editing'} aria-label={isZh ? '取消编辑' : 'Cancel editing'} disabled={saving} onClick={cancel}><CloseRoundedIcon fontSize="small" /></button>
    </span>
    {error ? <span role="alert" className="experiment-matrix-cell-error">{error}</span> : null}
  </div>;
}

function TagEditorPopover({ anchorRef, tagState, ids, setIds, error, isZh, onCancel, onSave }) {
  const [popover, setPopover] = useState(null);
  const popperRef = useRef(null);

  useLayoutEffect(() => {
    if (!popover) return undefined;
    popover.querySelector('input[type="search"]')?.focus({ preventScroll: true });
    const observer = new ResizeObserver(() => popperRef.current?.update());
    observer.observe(popover);
    return () => observer.disconnect();
  }, [popover]);

  // The body portal escapes sticky-cell stacking and scroll clipping; Popper handles viewport edges.
  return <Popper open anchorEl={anchorRef.current} placement="bottom-start" popperRef={popperRef}
    popperOptions={{ strategy: 'fixed' }} sx={{ zIndex: 9800 }}
    modifiers={[
      { name: 'offset', options: { offset: [0, 8] } },
      { name: 'flip', options: { rootBoundary: 'viewport', padding: 12 } },
      { name: 'preventOverflow', options: { rootBoundary: 'viewport', altAxis: true, tether: false, padding: 12 } },
    ]}>
    <ClickAwayListener mouseEvent="onMouseDown" onClickAway={(event) => { if (!anchorRef.current?.contains(event.target)) onCancel(); }}>
      <div ref={setPopover} className="experiment-matrix-tag-popover" role="dialog"
        aria-label={isZh ? '编辑实验标签' : 'Edit experiment tags'} onClick={stopRowEvent}
        onKeyDown={(event) => { event.stopPropagation(); if (event.key === 'Escape') { event.preventDefault(); onCancel(); anchorRef.current?.focus({ preventScroll: true }); } }}>
      <TagPicker
        tags={tagState.tags}
        value={ids.filter((id) => tagState.tags.some((tag) => tag.id === id))}
        onChange={setIds}
        onCreate={(name) => tagState.mutate(() => createTrainingTag(name))}
        disabled={tagState.busy || tagState.loading}
        isZh={isZh}
        label={isZh ? '选择标签' : 'Choose tags'}
      />
      <div className="experiment-matrix-popover-actions">
        <button type="button" className="experiment-matrix-text-button" onClick={onCancel}>{isZh ? '取消' : 'Cancel'}</button>
        <button type="button" className="experiment-matrix-text-button primary" disabled={tagState.busy} onClick={onSave}>{isZh ? '保存' : 'Save'}</button>
      </div>
      {error ? <div role="alert" className="experiment-matrix-cell-error">{error}</div> : null}
      </div>
    </ClickAwayListener>
  </Popper>;
}

function TagsCell({ task, tagState, isZh, active }) {
  const [open, setOpen] = useState(false);
  const [ids, setIds] = useState((task.tags || []).map((tag) => tag.id));
  const [error, setError] = useState('');
  const anchorRef = useRef(null);
  const cancel = useCallback(() => { setOpen(false); setError(''); }, []);
  useEffect(() => { if (!active) cancel(); }, [active, cancel]);
  useEffect(() => { if (!open) setIds((task.tags || []).map((tag) => tag.id)); }, [task.tags, open]);
  const save = async () => {
    setError('');
    try { await tagState.mutate(() => replaceTrainingTaskTags(task.id, ids)); setOpen(false); }
    catch (saveError) { setError(saveError.message); }
  };
  return <div className="experiment-matrix-tags-cell" onClick={stopRowEvent}>
    <div className="experiment-matrix-tags-line"><TagChips tags={task.tags || []} /><button ref={anchorRef} type="button" className="experiment-matrix-icon-button" title={isZh ? '编辑标签' : 'Edit tags'} aria-label={isZh ? '编辑标签' : 'Edit tags'} aria-expanded={open} onClick={() => setOpen((value) => !value)}><EditRoundedIcon fontSize="small" /></button></div>
    {open && active ? <TagEditorPopover anchorRef={anchorRef} tagState={tagState} ids={ids} setIds={setIds} error={error} isZh={isZh} onCancel={cancel} onSave={save} /> : null}
  </div>;
}

function PropertyPanel({ definitions, columns, setColumns, isZh, onClose }) {
  const [search, setSearch] = useState('');
  const [dragKey, setDragKey] = useState(null);
  const selected = new Set(columns);
  const query = search.trim().toLocaleLowerCase();
  const visible = definitions.filter((definition) => !query || `${definition.label} ${definition.labelEn}`.toLocaleLowerCase().includes(query));
  const toggle = (key) => {
    const definition = definitions.find((item) => item.key === key);
    if (definition?.fixed) return;
    setColumns((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key]);
  };
  const move = (from, to) => {
    if (!from || from === to) return;
    setColumns((current) => {
      const next = [...current]; const fromIndex = next.indexOf(from); const toIndex = next.indexOf(to);
      const fromDefinition = definitions.find((item) => item.key === from);
      const toDefinition = definitions.find((item) => item.key === to);
      if (fromIndex < 0 || toIndex < 0 || fromDefinition?.fixed || toDefinition?.fixed) return current;
      next.splice(fromIndex, 1); next.splice(toIndex, 0, from); return next;
    });
  };
  return <aside className="experiment-matrix-property-panel" aria-label={isZh ? '显示属性' : 'Visible properties'}>
    <div className="experiment-matrix-panel-head"><strong>{isZh ? '显示属性' : 'Visible properties'}</strong><button type="button" className="experiment-matrix-icon-button" aria-label={isZh ? '关闭显示属性' : 'Close properties'} onClick={onClose}><CloseRoundedIcon fontSize="small" /></button></div>
    <input className="experiment-matrix-property-search" type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder={isZh ? '搜索属性' : 'Search properties'} aria-label={isZh ? '搜索属性' : 'Search properties'} />
    <div className="experiment-matrix-panel-actions"><button type="button" className="experiment-matrix-text-button" onClick={() => setColumns(definitions.map((definition) => definition.key))}>{isZh ? '全部显示' : 'Show all'}</button><button type="button" className="experiment-matrix-text-button" onClick={() => setColumns([...MATRIX_DEFAULT_COLUMNS])}>{isZh ? '恢复默认' : 'Restore default'}</button></div>
    {MATRIX_GROUPS.map((group) => {
      const groupItems = visible.filter((definition) => definition.group === group.id);
      if (!groupItems.length) return null;
      return <section key={group.id} className="experiment-matrix-property-group"><h3>{isZh ? group.label : group.labelEn}</h3><ul>{groupItems.map((definition) => { const index = columns.indexOf(definition.key); const previous = index > 0 ? columns[index - 1] : null; const next = index >= 0 && index < columns.length - 1 ? columns[index + 1] : null; return <li key={definition.key} draggable={selected.has(definition.key) && !definition.fixed} onDragStart={() => setDragKey(definition.key)} onDragOver={(event) => event.preventDefault()} onDrop={() => { move(dragKey, definition.key); setDragKey(null); }}><label><input type="checkbox" checked={selected.has(definition.key)} disabled={definition.fixed} onChange={() => toggle(definition.key)} /><span>{isZh ? definition.label : definition.labelEn}</span></label>{selected.has(definition.key) && !definition.fixed ? <span className="experiment-matrix-column-arrows"><button type="button" className="experiment-matrix-icon-button" aria-label={isZh ? `上移${definition.label}` : `Move ${definition.labelEn} left`} title={isZh ? '上移列' : 'Move column left'} disabled={!previous || definitions.find((item) => item.key === previous)?.fixed} onClick={() => move(definition.key, previous)}>←</button><button type="button" className="experiment-matrix-icon-button" aria-label={isZh ? `下移${definition.label}` : `Move ${definition.labelEn} right`} title={isZh ? '下移列' : 'Move column right'} disabled={!next || definitions.find((item) => item.key === next)?.fixed} onClick={() => move(definition.key, next)}>→</button></span> : null}</li>; })}</ul></section>;
    })}
  </aside>;
}

export default function ExperimentMatrix({
  tasks = [], tagState, tasksLoading = false, tasksError = false, user, active = true,
  isZh = true, locale = 'zh-CN', isProcessing = false, onSelectTask, onCopyConfig, onRenameTask, onRetryTasks,
  onPredict, onTest, onDelete,
}) {
  const definitions = useMemo(() => getMatrixPropertyDefinitions(tasks), [tasks]);
  const definitionMap = useMemo(() => new Map(definitions.map((definition) => [definition.key, definition])), [definitions]);
  const [columns, setColumns] = useState(() => {
    if (typeof window === 'undefined') return [...MATRIX_DEFAULT_COLUMNS];
    return restoreMatrixColumnPreferences(readStoredPreferences(user?.id), definitions);
  });
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('all');
  const [tagIds, setTagIds] = useState([]);
  const [bulkTagIds, setBulkTagIds] = useState([]);
  const [sort, setSort] = useState({ key: 'start_time', direction: 'desc' });
  const [showProperties, setShowProperties] = useState(false);
  const [selected, setSelected] = useState([]);
  const [tagManagerOpen, setTagManagerOpen] = useState(false);
  const scrollRef = useRef(null);
  const preferenceRawRef = useRef(readStoredPreferences(user?.id));
  const preferenceScopeRef = useRef(user?.id || 'guest');
  const preferenceDefinitionsRef = useRef('');
  const preferenceColumnsRef = useRef(columns);
  const skipNextPreferenceWriteRef = useRef(false);
  preferenceColumnsRef.current = columns;

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const scope = user?.id || 'guest';
    const signature = definitions.map((definition) => definition.key).join('|');
    const scopeChanged = preferenceScopeRef.current !== scope;
    const definitionsChanged = preferenceDefinitionsRef.current !== signature;
    if (!scopeChanged && !definitionsChanged) return;
    if (scopeChanged) {
      preferenceScopeRef.current = scope;
      preferenceRawRef.current = readStoredPreferences(user?.id);
      // The write effect also depends on user.id; do not persist the previous
      // account's in-memory columns into the newly selected account.
      skipNextPreferenceWriteRef.current = true;
    }
    preferenceDefinitionsRef.current = signature;
    const raw = preferenceRawRef.current || readStoredPreferences(user?.id);
    const restored = restoreMatrixColumnPreferences(raw || serializeMatrixColumnPreferences(preferenceColumnsRef.current), definitions);
    const current = preferenceColumnsRef.current;
    if (scopeChanged || restored.length !== current.length || restored.some((key, index) => key !== current[index])) {
      skipNextPreferenceWriteRef.current = true;
      setColumns(restored);
    }
  }, [definitions, user?.id]);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (skipNextPreferenceWriteRef.current) {
      skipNextPreferenceWriteRef.current = false;
      return;
    }
    // Keep unknown custom columns in storage until the task payload that defines
    // them arrives; this makes polling and account switches backward compatible.
    const raw = preferenceRawRef.current || '';
    const hasUnresolvedColumns = raw && /custom:|architecture:|training:|metric:/.test(raw)
      && !tasks.length;
    if (hasUnresolvedColumns) return;
    const serialized = serializeMatrixColumnPreferences(columns);
    preferenceRawRef.current = serialized;
    writeStoredPreferences(user?.id, serialized);
  }, [columns, tasks.length, user?.id]);

  const filtered = useMemo(() => filterMatrixTasks(tasks, { search, status, tagIds }), [tasks, search, status, tagIds]);
  const sorted = useMemo(() => sortMatrixTasks(filtered, sort.key, sort.direction, definitions), [filtered, sort, definitions]);
  const tableMinWidth = useMemo(() => columns.reduce((width, key) => {
    if (key === 'model_name') return width + 230;
    if (key === 'tags') return width + 190;
    return width + 150;
  }, 42 + 140), [columns]);
  const selectedVisible = selected.filter((id) => sorted.some((task) => task.id === id));
  const changeSort = (key) => setSort((current) => current.key === key ? { key, direction: current.direction === 'asc' ? 'desc' : 'asc' } : { key, direction: 'asc' });
  const selectAll = () => setSelected(selectedVisible.length === sorted.length ? [] : sorted.map((task) => task.id));
  const bulkTags = async (operation) => {
    if (!selectedVisible.length || !bulkTagIds.length) return;
    await tagState.mutate(() => updateTrainingTaskTags(selectedVisible, bulkTagIds, operation));
    setSelected([]);
    setBulkTagIds([]);
  };
  const state = tasksError ? 'error' : tasksLoading && !tasks.length ? 'loading' : !tasks.length ? 'empty' : !sorted.length ? 'no-match' : 'ready';
  const statusText = (value) => STATUS_LABELS[value]?.[isZh ? 0 : 1] || value;

  return <div className="experiment-matrix" data-matrix-state={state}>
    <div className="experiment-matrix-toolbar">
      <div className="experiment-matrix-toolbar-main"><input type="search" className="experiment-matrix-search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder={isZh ? '搜索实验名称' : 'Search experiment names'} aria-label={isZh ? '搜索实验名称' : 'Search experiment names'} />
        <select value={status} onChange={(event) => setStatus(event.target.value)} aria-label={isZh ? '按状态筛选' : 'Filter by status'}>{MATRIX_STATUS_FILTERS.map((value) => <option key={value} value={value}>{statusText(value)}</option>)}</select>
        <details className="experiment-matrix-tag-filter"><summary>{isZh ? '标签筛选' : 'Tag filter'}{tagIds.length ? ` · ${tagIds.length}` : ''}</summary><div className="experiment-matrix-tag-filter-body"><TagPicker tags={tagState?.tags || []} value={tagIds} onChange={setTagIds} isZh={isZh} disabled={tagState?.loading || tagState?.busy} label={isZh ? '按标签筛选' : 'Filter by tags'} /></div></details>
      </div>
      <button type="button" className="experiment-matrix-properties-button" aria-expanded={showProperties} onClick={() => setShowProperties((value) => !value)}><SettingsRoundedIcon fontSize="small" />{isZh ? '显示属性' : 'Visible properties'}</button>
    </div>
    {selectedVisible.length ? <div className="experiment-matrix-bulk" role="group" aria-label={isZh ? '批量标签操作' : 'Bulk tag actions'}><span>{isZh ? `已选 ${selectedVisible.length} 条` : `${selectedVisible.length} selected`}</span><button type="button" className="experiment-matrix-text-button" onClick={selectAll}>{isZh ? '全选当前筛选结果' : 'Select filtered'}</button><TagPicker tags={tagState?.tags || []} value={bulkTagIds} onChange={setBulkTagIds} isZh={isZh} disabled={tagState?.loading || tagState?.busy} label={isZh ? '选择批量标签' : 'Choose bulk tags'} /><button type="button" className="experiment-matrix-text-button" disabled={tagState?.busy || !bulkTagIds.length} onClick={() => bulkTags('add')}>{isZh ? '添加标签' : 'Add tags'}</button><button type="button" className="experiment-matrix-text-button" disabled={tagState?.busy || !bulkTagIds.length} onClick={() => bulkTags('remove')}>{isZh ? '移除标签' : 'Remove tags'}</button></div> : null}
    <div className="experiment-matrix-layout">
      <div ref={scrollRef} className="experiment-matrix-scroll" tabIndex={0} aria-label={isZh ? '实验矩阵表格' : 'Experiment matrix table'}>
        {state === 'loading' ? <div className="experiment-matrix-state" role="status" data-matrix-skeleton="true">
          <span className="sr-only">{isZh ? '正在加载实验…' : 'Loading experiments…'}</span>
          <div className="experiment-matrix-skeleton">
            {[0, 1, 2].map((index) => (
              <span key={index} className="ui-skeleton" style={{ height: 52, borderRadius: 10, animationDelay: `${index * 90}ms` }} />
            ))}
          </div>
        </div> : null}
        {state === 'error' ? <div className="experiment-matrix-state" role="alert"><span>{isZh ? '实验矩阵加载失败。' : 'Could not load the experiment matrix.'}</span><button type="button" className="experiment-matrix-text-button" onClick={onRetryTasks}>{isZh ? '重试' : 'Retry'}</button></div> : null}
        {state === 'empty' ? <div className="experiment-matrix-state" role="status">{isZh ? '暂无训练实验。' : 'No training experiments yet.'}</div> : null}
        {state === 'no-match' ? <div className="experiment-matrix-state" role="status">{isZh ? '没有匹配的实验，请调整筛选条件。' : 'No matching experiments. Adjust the filters.'}</div> : null}
        {state === 'ready' ? <table className="experiment-matrix-table" style={{ width: `calc(${tableMinWidth - 230}px + var(--matrix-name-width))`, minWidth: `calc(${tableMinWidth - 230}px + var(--matrix-name-width))` }}><colgroup><col style={{ width: 42 }} />{columns.map((key) => <col key={key} style={{ width: key === 'model_name' ? 'var(--matrix-name-width)' : key === 'tags' ? 190 : 150 }} />)}<col style={{ width: 140 }} /></colgroup><thead><tr><th className="experiment-matrix-select-col"><input type="checkbox" checked={Boolean(sorted.length && selectedVisible.length === sorted.length)} onChange={selectAll} aria-label={isZh ? '全选当前筛选结果' : 'Select all filtered experiments'} /></th>{columns.map((key) => { const definition = definitionMap.get(key); if (!definition) return null; return <th key={key} className={`experiment-matrix-th experiment-matrix-col-${definition.key.replace(/[:_]/g, '-')}`} aria-sort={sort.key === key ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}><button type="button" onClick={() => changeSort(key)}>{isZh ? definition.label : definition.labelEn}<span aria-hidden="true">{sort.key === key ? (sort.direction === 'asc' ? ' ↑' : ' ↓') : ''}</span></button></th>; })}<th className="experiment-matrix-actions-col">{isZh ? '操作' : 'Actions'}</th></tr></thead><tbody>{sorted.map((task) => <tr key={task.id} tabIndex={0} onClick={() => onSelectTask?.(task.id)} onKeyDown={(event) => { if (event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); onSelectTask?.(task.id); } }}>
          <td className="experiment-matrix-select-col"><input type="checkbox" checked={selectedVisible.includes(task.id)} onClick={stopRowEvent} onChange={(event) => setSelected((current) => event.target.checked ? [...new Set([...current, task.id])] : current.filter((id) => id !== task.id))} aria-label={isZh ? `选择实验 ${task.custom_model_name || task.id}` : `Select experiment ${task.custom_model_name || task.id}`} /></td>
          {columns.map((key) => { const definition = definitionMap.get(key); if (!definition) return null; const value = getMatrixValue(task, key); return <td key={key} className={`experiment-matrix-td experiment-matrix-col-${definition.key.replace(/[:_]/g, '-')}`}>{key === 'model_name' ? <NameCell task={task} tasks={tasks} isZh={isZh} onSave={onRenameTask} /> : key === 'tags' ? <TagsCell task={task} tagState={tagState} isZh={isZh} active={active} /> : <span title={formatMatrixValue(value, definition, task, locale)}>{formatMatrixValue(value, definition, task, locale)}</span>}</td>; })}
          <td className="experiment-matrix-actions-col"><ExperimentMatrixActions task={task} active={active} userId={user?.id} isZh={isZh} isProcessing={isProcessing} onSelectTask={onSelectTask} onCopyConfig={onCopyConfig} onPredict={onPredict} onTest={onTest} onDelete={onDelete} /></td>
        </tr>)}</tbody></table> : null}
      </div>
      {showProperties ? <PropertyPanel definitions={definitions} columns={columns} setColumns={setColumns} isZh={isZh} onClose={() => setShowProperties(false)} /> : null}
    </div>
    <div className="experiment-matrix-footer"><span>{isZh ? `显示 ${sorted.length} / ${tasks.length} 条实验` : `Showing ${sorted.length} / ${tasks.length} experiments`}</span><div className="experiment-matrix-footer-actions">{sorted.length ? <button type="button" className="experiment-matrix-text-button" onClick={selectAll}>{selectedVisible.length === sorted.length ? (isZh ? '清空选择' : 'Clear selection') : (isZh ? '全选当前筛选结果' : 'Select filtered')}</button> : null}<button type="button" className="experiment-matrix-text-button" disabled={tagState?.scope == null || tagState?.busy} onClick={() => setTagManagerOpen(true)}>{isZh ? '管理标签' : 'Manage tags'}</button></div></div>
    {tagManagerOpen ? <TagManager state={tagState} isZh={isZh} onClose={() => setTagManagerOpen(false)} /> : null}
  </div>;
}
