import { useMemo, useState } from 'react';
import { useT } from '../../i18n';
import { staggerStyle } from '../../components/ui/motion';
import TrainingHistory from '../../components/TrainingTags/TrainingHistory';
import { createTrainingStatusMatcher } from '../../components/TrainingTags/trainingTagFilters';
import {
  EXPERIMENT_STATUS_FILTERS,
  countExperimentStatuses,
} from './experimentCenterModel';
import './experimentCenter.css';

function groupDirectoryTasks(items, isZh, showRecent) {
  const groups = [
    { id: 'recent', label: isZh ? '最近实验' : 'Recent experiments', tasks: [] },
    { id: 'queued', label: isZh ? '排队中' : 'Queued', tasks: [] },
    { id: 'running', label: isZh ? '运行中' : 'Running', tasks: [] },
    { id: 'completed', label: isZh ? '已完成' : 'Completed', tasks: [] },
    { id: 'failed', label: isZh ? '失败或需修复' : 'Failed or needs attention', tasks: [] },
    { id: 'cancelled', label: isZh ? '已取消' : 'Cancelled', tasks: [] },
  ];
  const recentIds = new Set(showRecent ? [...items]
    .sort((a, b) => Date.parse(b.end_time || b.start_time || 0) - Date.parse(a.end_time || a.start_time || 0))
    .slice(0, 3).map((task) => task.id) : []);
  items.forEach((task) => {
    const status = String(task.status || '').toLowerCase();
    const group = recentIds.has(task.id) && status !== 'queued' ? groups[0]
      : status === 'queued' ? groups[1]
        : status === 'running' || status === 'pending' ? groups[2]
          : status === 'completed' ? groups[3]
            : status === 'cancelled' ? groups[5] : groups[4];
    group.tasks.push(task);
  });
  return groups.filter((group) => group.tasks.length);
}

/**
 * 目录行只展示实验名称与实验标签，标签操作直接可用。
 */
function ExperimentDirectoryRow({
  task,
  tagControls,
  selectionControl,
  editControl,
  t,
  isActive,
  onSelect,
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      className="experiment-directory-row"
      data-active={isActive ? 'true' : 'false'}
      data-status={task.status || 'unknown'}
      aria-pressed={isActive}
      onClick={() => onSelect(task.id)}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) return;
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onSelect(task.id);
        }
      }}
    >
      <div className="experiment-directory-row-top">
        {selectionControl}
        <span className="experiment-directory-name" title={task.custom_model_name || t('modelTraining.unnamedModel')}>
          {task.custom_model_name || t('modelTraining.unnamedModel')}
        </span>
        {editControl}
      </div>
      <div className="experiment-directory-row-tags" onClick={(event) => event.stopPropagation()}>
        {tagControls}
      </div>
    </div>
  );
}

/**
 * 左侧实验目录：状态分组、名称搜索、标签筛选、未分组筛选、批量标签和紧凑实验行。
 *
 * 搜索、标签和批量标签逻辑复用 TrainingHistory；目录内不请求任何训练接口。
 */
export default function ExperimentDirectory({
  tasks,
  tagState,
  activeTaskId,
  onSelectTask,
  tasksLoading = false,
  tasksError = false,
  onCreateTask,
  onRetryTasks,
  locale,
  isZh,
}) {
  const t = useT();
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState('newest');
  const counts = useMemo(() => countExperimentStatuses(tasks), [tasks]);
  const statusMatcher = useMemo(() => createTrainingStatusMatcher(statusFilter), [statusFilter]);
  const sortedTasks = useMemo(() => [...tasks].sort((a, b) => {
    if (sortOrder === 'name') return String(a.custom_model_name || '').localeCompare(String(b.custom_model_name || ''), locale);
    const difference = Date.parse(b.end_time || b.start_time || 0) - Date.parse(a.end_time || a.start_time || 0);
    return (sortOrder === 'oldest' ? -difference : difference) || Number(b.id) - Number(a.id);
  }), [tasks, sortOrder, locale]);

  const filterLabels = {
    all: t('experimentCenter.filterAll'),
    running: t('experimentCenter.filterRunning'),
    completed: t('experimentCenter.filterCompleted'),
    failed: t('experimentCenter.filterFailed'),
    queued: t('experimentCenter.filterQueued'),
    cancelled: t('experimentCenter.filterCancelled'),
  };

  // 目录只有三种需要区别对待的状态：还在取任务列表、取失败、或确实没有实验。
  // 只看真实加载标记（tasksLoading 由首次任务请求复位），不要用 scope 之类的
  // 登录信息推断，否则登录用户会永远停在「加载中」。这三个标记只影响展示与密度，
  // 不参与任何筛选或选中判断。
  const isEmptyDirectory = tasks.length === 0;
  const isPending = isEmptyDirectory && !tasksError && tasksLoading;
  const directoryState = tasksError ? 'error' : (isPending ? 'loading' : (isEmptyDirectory ? 'empty' : 'ready'));

  return (
    <div className="experiment-directory" data-directory-state={directoryState}>
      <div className="experiment-directory-head">
        <div className="experiment-directory-title">{t('experimentCenter.directoryTitle')}</div>
        <span className="training-tag-hint" role="status">
          {t('experimentCenter.directoryCount', { count: counts.all })}
        </span>
      </div>

      <div className="experiment-directory-filters" role="group" aria-label={t('experimentCenter.statusFilterLabel')}>
        {EXPERIMENT_STATUS_FILTERS.map((filter) => (
          <button
            key={filter}
            type="button"
            className="experiment-directory-filter"
            aria-pressed={statusFilter === filter}
            onClick={() => setStatusFilter(filter)}
          >
            {filterLabels[filter]}
            <span className="experiment-directory-filter-count">{counts[filter]}</span>
          </button>
        ))}
      </div>

      <TrainingHistory
        key={tagState.scope ?? 'guest'}
        tasks={sortedTasks}
        tagState={tagState}
        isZh={isZh}
        renderMode="directory"
        statusMatcher={statusMatcher}
        groupTasks={(items) => groupDirectoryTasks(items, isZh, statusFilter === 'all' && sortOrder === 'newest')}
        headerExtra={() => (
          <label className="experiment-directory-sort">
            <span className="sr-only">{isZh ? '排序' : 'Sort'}</span>
            <select value={sortOrder} onChange={(event) => setSortOrder(event.target.value)}>
              <option value="newest">{isZh ? '最近更新' : 'Newest'}</option>
              <option value="oldest">{isZh ? '最早更新' : 'Oldest'}</option>
              <option value="name">{isZh ? '名称' : 'Name'}</option>
            </select>
          </label>
        )}
        emptyState={{
          reason: isEmptyDirectory ? (tasksError ? 'error' : (isPending ? 'loading' : 'no-experiments')) : 'no-match',
          zh: {
            loading: '正在加载实验…',
            error: '实验目录加载失败。',
            noExperiments: '暂无实验。新建实验后，实验会出现在这里。',
            noMatch: '没有匹配的实验，请调整筛选条件。',
          },
          en: {
            loading: 'Loading experiments…',
            error: 'Could not load the experiment directory.',
            noExperiments: 'No experiments yet. Create one and it will appear here.',
            noMatch: 'No matching experiments. Adjust your filters.',
          },
          actionLabel: isZh ? '新建实验' : 'New experiment',
          retryLabel: isZh ? '重试' : 'Retry',
          showAction: isEmptyDirectory && !tasksError && !isPending,
          showRetry: tasksError,
          onAction: onCreateTask,
          onRetry: onRetryTasks,
        }}
        renderTask={(task, tagControls, rowControls) => (
          <div
            key={task.id}
            className="motion-stagger-item"
            style={staggerStyle(Math.min(12, sortedTasks.findIndex((item) => item.id === task.id)))}
          >
            <ExperimentDirectoryRow
              task={task}
              tagControls={tagControls}
              selectionControl={rowControls.selection}
              editControl={rowControls.edit}
              t={t}
              isActive={task.id === activeTaskId}
              onSelect={onSelectTask}
            />
          </div>
        )}
      />
    </div>
  );
}
