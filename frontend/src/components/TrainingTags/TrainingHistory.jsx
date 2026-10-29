import { useEffect, useId, useMemo, useState } from 'react';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import C from '../../constants/colors';
import { replaceTrainingTaskTags, updateTrainingTaskTags } from '../../services/api';
import { filterTaggedTasks } from './trainingTagFilters';
import { TagChips, TagFilter, TagPicker } from './TagControls';
import { EditTaskTagsDialog, TagManager } from './TagDialogs';

export default function TrainingHistory({ tasks, tagState, isZh, renderTask, renderMode = 'list', statusMatcher = null, headerExtra = null, emptyState = null, groupTasks = null }) {
  const isDirectory = renderMode === 'directory';
  const [filter, setFilter] = useState({ tagIds: [], untagged: false });
  const [search, setSearch] = useState('');
  const [tagFilterOpen, setTagFilterOpen] = useState(false);
  const tagFilterId = useId();
  const [selected, setSelected] = useState([]);
  const [batchTagIds, setBatchTagIds] = useState([]);
  const [editing, setEditing] = useState(null);
  const [managing, setManaging] = useState(false);
  const [error, setError] = useState('');
  const [collapsedGroups, setCollapsedGroups] = useState([]);
  const filtered = useMemo(
    () => filterTaggedTasks(tasks, { ...filter, search, statusMatcher }),
    [tasks, filter, search, statusMatcher]
  );
  const available = new Set(filtered.map(task => task.id));
  const visibleSelection = selected.filter(id => available.has(id));

  useEffect(() => {
    if (tagState.loading || tagState.error) return;
    const known = new Set(tagState.tags.map(tag => tag.id));
    setFilter(prev => prev.tagIds.some(id => !known.has(id)) ? { ...prev, tagIds: prev.tagIds.filter(id => known.has(id)) } : prev);
    setBatchTagIds(prev => prev.filter(id => known.has(id)));
  }, [tagState.tags, tagState.loading, tagState.error]);

  useEffect(() => { setSelected([]); }, [filter, search, statusMatcher]);

  async function batch(operation) {
    setError('');
    try {
      await tagState.mutate(() => updateTrainingTaskTags(visibleSelection, batchTagIds, operation));
      setSelected([]);
      setBatchTagIds([]);
    } catch (err) { setError(err.message); }
  }

  if (isDirectory) return <div className={`training-tag-directory${emptyState ? ` is-${emptyState.reason}` : ''}`}>
    <div className="experiment-directory-search">
      <input className="training-tag-input" type="search" aria-label={isZh ? '搜索实验名称' : 'Search experiments'} placeholder={isZh ? '搜索实验名称' : 'Search experiments'} value={search} onChange={event => { setSelected([]); setSearch(event.target.value); }} />
    </div>
    <div className="experiment-directory-tools">
      {headerExtra ? headerExtra(selected, filtered) : null}
      <button type="button" className="experiment-directory-tag-toggle" aria-expanded={tagFilterOpen} aria-controls={tagFilterId} onClick={() => setTagFilterOpen(open => !open)}>
        <span>{isZh ? '标签筛选' : 'Tag filter'}</span>
        {filter.untagged || filter.tagIds.length > 0 ? <span className="experiment-directory-tag-count">{filter.untagged ? 1 : filter.tagIds.length}</span> : null}
        <KeyboardArrowDownIcon aria-hidden="true" />
      </button>
    </div>
    <div id={tagFilterId} className="experiment-directory-tag-panel" hidden={!tagFilterOpen}>
      <TagFilter tags={tagState.tags} {...filter} isZh={isZh} disabled={tagState.loading || tagState.busy} onChange={next => { setSelected([]); setFilter(next); }} />
      <div className="experiment-directory-tag-panel-footer">
        <button type="button" className="experiment-directory-manage-tags" disabled={tagState.scope === null || tagState.busy} onClick={() => setManaging(true)}>{isZh ? '管理标签' : 'Manage tags'}</button>
      </div>
    </div>
    {tagState.loading && <div role="status" className="training-tag-hint">{isZh ? '正在加载标签…' : 'Loading tags…'}</div>}
    {tagState.error && <div role="alert" className="training-tag-toolbar" style={{ color: C.mars }}>
      {tagState.error}<button className="training-tag-button" onClick={() => tagState.refresh().catch(() => {})}>{isZh ? '重试' : 'Retry'}</button>
    </div>}
    {selected.length > 0 && <div className="experiment-directory-selection">
      <span className="training-tag-hint" role="status">{isZh ? `已选 ${visibleSelection.length}` : `${visibleSelection.length} selected`}</span>
      <button className="training-tag-button" onClick={() => setSelected(filtered.map(task => task.id))}>{isZh ? '全选当前结果' : 'Select visible'}</button>
      <button className="training-tag-button" onClick={() => setSelected([])}>{isZh ? '清空选择' : 'Clear'}</button>
      <TagPicker tags={tagState.tags} value={batchTagIds} onChange={setBatchTagIds} isZh={isZh} label={isZh ? '批量调整标签' : 'Edit tags in bulk'} disabled={tagState.busy || tagState.loading} />
      <button className="training-tag-button" disabled={tagState.busy || !batchTagIds.length} onClick={() => batch('add')}>{isZh ? '添加所选标签' : 'Add tags'}</button>
      <button className="training-tag-button" disabled={tagState.busy || !batchTagIds.length} onClick={() => batch('remove')}>{isZh ? '移除所选标签' : 'Remove tags'}</button>
    </div>}
    {error && <div role="alert" style={{ color: C.mars }}>{error}</div>}
    <div className="experiment-directory-list">
      {(groupTasks ? groupTasks(filtered) : [{ id: 'all', label: '', tasks: filtered }]).map(group => (
        <section className="experiment-directory-group" key={group.id}>
          {group.label ? <button type="button" className="experiment-directory-group-toggle" aria-expanded={!collapsedGroups.includes(group.id)} onClick={() => setCollapsedGroups(previous => previous.includes(group.id) ? previous.filter(id => id !== group.id) : [...previous, group.id])}>
            <span>{group.label}</span><span>{group.tasks.length}</span><span aria-hidden="true">{collapsedGroups.includes(group.id) ? '+' : '−'}</span>
          </button> : null}
          {!collapsedGroups.includes(group.id) ? <div className="experiment-directory-group-items">{group.tasks.map(task => renderTask(
            task,
            <div className="experiment-directory-tag-list">
              <TagChips tags={task.tags} />
              {!task.tags?.length && <span className="training-tag-hint">{isZh ? '未分组' : 'Untagged'}</span>}
            </div>,
            {
              selection: (
                <label className="experiment-directory-select" title={isZh ? '选择实验' : 'Select experiment'} onClick={event => event.stopPropagation()}>
                  <input type="checkbox" checked={visibleSelection.includes(task.id)} disabled={tagState.busy} aria-label={isZh ? `选择实验 ${task.custom_model_name || task.id}` : `Select experiment ${task.custom_model_name || task.id}`} onChange={event => setSelected(prev => event.target.checked ? [...new Set([...prev, task.id])] : prev.filter(id => id !== task.id))} />
                </label>
              ),
              edit: (
                <button type="button" className="experiment-directory-edit-tags" title={isZh ? '编辑标签' : 'Edit tags'} aria-label={isZh ? `编辑标签 · ${task.custom_model_name || task.id}` : `Edit tags · ${task.custom_model_name || task.id}`} disabled={tagState.busy || tagState.loading || Boolean(tagState.error)} onClick={event => { event.stopPropagation(); setEditing(task); }}>
                  <EditOutlinedIcon aria-hidden="true" />
                </button>
              ),
            },
          ))}</div> : null}
        </section>
      ))}
    </div>
    {!filtered.length && (() => {
      const copy = emptyState || {};
      const strings = isZh ? copy.zh : copy.en;
      const reason = copy.reason || (tasks.length ? 'no-match' : 'no-experiments');
      const message = strings
        ? (reason === 'loading' ? strings.loading
          : reason === 'error' ? strings.error
            : reason === 'no-match' ? strings.noMatch
              : strings.noExperiments)
        : (tasks.length ? (isZh ? '没有匹配的实验，请调整筛选条件。' : 'No matching experiments. Adjust your filters.') : (isZh ? '暂无实验。新建实验后，实验会出现在这里。' : 'No experiments yet. Create one and it will appear here.'));
      return (
        <div className="experiment-directory-empty" data-empty-reason={reason} role={reason === 'error' ? 'alert' : 'status'}>
          {reason === 'loading' ? (
            // 加载态用骨架行代替纯文字，避免目录先闪一行字再变成列表
            <div className="experiment-directory-skeleton" data-directory-skeleton="true">
              {[0, 1, 2].map((index) => (
                <span key={index} className="ui-skeleton" style={{ height: 64, borderRadius: 12, animationDelay: `${index * 90}ms` }} />
              ))}
            </div>
          ) : null}
          <p className="experiment-directory-empty-text">{message}</p>
          {copy.showRetry && copy.onRetry ? (
            <div className="experiment-directory-empty-actions">
              {/* 只有取任务列表失败时才出现：重试走页面控制器传进来的加载函数，不新增请求逻辑。 */}
              <button type="button" className="training-tag-button" data-empty-action="retry" onClick={copy.onRetry}>{copy.retryLabel || (isZh ? '重试' : 'Retry')}</button>
            </div>
          ) : null}
          {copy.showAction && copy.onAction ? (
            <div className="experiment-directory-empty-actions">
              <button type="button" className="experiment-center-button experiment-center-button-primary" data-empty-action="create" onClick={copy.onAction}>{copy.actionLabel}</button>
            </div>
          ) : null}
        </div>
      );
    })()}
    {managing && <TagManager state={tagState} isZh={isZh} onClose={() => setManaging(false)} />}
    {editing && <EditTaskTagsDialog title={isZh ? `编辑标签 · ${editing.custom_model_name || editing.id}` : `Edit tags · ${editing.custom_model_name || editing.id}`} initialIds={(editing.tags || []).map(tag => tag.id)} state={tagState} isZh={isZh} onClose={() => setEditing(null)} onSave={ids => tagState.mutate(() => replaceTrainingTaskTags(editing.id, ids))} />}
  </div>;

  return <div>
    <div className="training-tag-toolbar">
      <input className="training-tag-input" style={{ flex: '1 1 220px' }} type="search" aria-label={isZh ? '搜索模型名称' : 'Search model names'} placeholder={isZh ? '搜索模型名称' : 'Search model names'} value={search} onChange={event => { setSelected([]); setSearch(event.target.value); }} />
      <button className="training-tag-button" disabled={tagState.scope === null || tagState.busy} onClick={() => setManaging(true)}>{isZh ? '管理标签' : 'Manage tags'}</button>
    </div>
    <TagFilter tags={tagState.tags} {...filter} isZh={isZh} disabled={tagState.loading || tagState.busy} onChange={next => { setSelected([]); setFilter(next); }} />
    {tagState.loading && <div role="status" className="training-tag-hint">{isZh ? '正在加载标签…' : 'Loading tags…'}</div>}
    {tagState.error && <div role="alert" className="training-tag-toolbar" style={{ color: C.mars }}>
      {tagState.error}<button className="training-tag-button" onClick={() => tagState.refresh().catch(() => {})}>{isZh ? '重试' : 'Retry'}</button>
    </div>}
    <div className="training-tag-toolbar" style={{ margin: '14px 0' }}>
      <button className="training-tag-button" disabled={!filtered.length || tagState.busy} onClick={() => setSelected(filtered.map(task => task.id))}>{isZh ? '全选当前结果' : 'Select visible'}</button>
      <button className="training-tag-button" disabled={!visibleSelection.length || tagState.busy} onClick={() => setSelected([])}>{isZh ? '清空选择' : 'Clear selection'}</button>
      <span className="training-tag-hint" role="status">{isZh ? `显示 ${filtered.length} / ${tasks.length}，已选 ${visibleSelection.length}` : `Showing ${filtered.length} / ${tasks.length}, ${visibleSelection.length} selected`}</span>
    </div>
    {visibleSelection.length > 0 && <div style={{ border: `1px solid ${C.border}`, borderRadius: 12, padding: 14, marginBottom: 14 }}>
      <TagPicker tags={tagState.tags} value={batchTagIds} onChange={setBatchTagIds} isZh={isZh} label={isZh ? '批量调整标签' : 'Edit tags in bulk'} disabled={tagState.busy || tagState.loading} />
      <div className="training-tag-toolbar">
        <button className="training-tag-button" disabled={tagState.busy || !batchTagIds.length} onClick={() => batch('add')}>{isZh ? '添加所选标签' : 'Add selected tags'}</button>
        <button className="training-tag-button" disabled={tagState.busy || !batchTagIds.length} onClick={() => batch('remove')}>{isZh ? '移除所选标签' : 'Remove selected tags'}</button>
      </div>
    </div>}
    {error && <div role="alert" style={{ color: C.mars }}>{error}</div>}
    <div style={{ display: 'grid', gap: 14 }}>
      {filtered.map(task => renderTask(task, <div className="training-tag-toolbar" onClick={event => event.stopPropagation()} style={{ marginBottom: 12 }}>
        <label className="training-tag-option">
          <input type="checkbox" checked={visibleSelection.includes(task.id)} disabled={tagState.busy} aria-label={isZh ? `选择模型 ${task.custom_model_name || task.id}` : `Select model ${task.custom_model_name || task.id}`} onChange={event => setSelected(prev => event.target.checked ? [...new Set([...prev, task.id])] : prev.filter(id => id !== task.id))} />
          {isZh ? '选择' : 'Select'}
        </label>
        <TagChips tags={task.tags} />
        {!task.tags?.length && <span className="training-tag-hint">{isZh ? '未分组' : 'Untagged'}</span>}
        <button className="training-tag-button" style={{ marginLeft: 'auto' }} disabled={tagState.busy || tagState.loading || Boolean(tagState.error)} onClick={() => setEditing(task)}>{isZh ? '编辑标签' : 'Edit tags'}</button>
      </div>))}
    </div>
    {!filtered.length && <div style={{ padding: 32, textAlign: 'center', color: C.ice60 }}>
      {tasks.length ? (isZh ? '没有匹配的模型，请调整筛选条件。' : 'No matching models. Adjust your filters.') : (isZh ? '暂无训练记录。启动训练后，可在此整理模型。' : 'No training records yet. Start a training run to organize models here.')}
    </div>}
    {managing && <TagManager state={tagState} isZh={isZh} onClose={() => setManaging(false)} />}
    {editing && <EditTaskTagsDialog title={isZh ? `编辑标签 · ${editing.custom_model_name || editing.id}` : `Edit tags · ${editing.custom_model_name || editing.id}`} initialIds={(editing.tags || []).map(tag => tag.id)} state={tagState} isZh={isZh} onClose={() => setEditing(null)} onSave={ids => tagState.mutate(() => replaceTrainingTaskTags(editing.id, ids))} />}
  </div>;
}
