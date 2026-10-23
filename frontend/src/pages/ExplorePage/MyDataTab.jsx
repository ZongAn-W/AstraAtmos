import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import C from '../../constants/colors';
import { useT } from '../../i18n';
import { useAuth } from '../../contexts/AuthContext';
import { useSettings } from '../../contexts/SettingsContext';
import { useToast } from '../../contexts/ToastContext';
import GlowCard from '../../components/GlowCard';
import HoldToDeleteButton from '../../components/ui/HoldToDeleteButton';
import { staggerStyle } from '../../components/ui/motion';
import ContributeModal from '../../components/ContributeModal';
import ContributeHistoryPanel from '../../components/ContributeHistoryPanel';
import {
  deleteUpload,
  fetchUserBands,
  fetchUserDataSummary,
  fetchUserGlobeData,
  fetchUserHeatmap,
  getMyUploads,
} from '../../services/api';
import HeatmapCanvas from './HeatmapCanvas';
import LineChart from './LineChart';
import GlobePlot from './GlobePlot';
import { LoadingBox } from './ExploreComponents';
import { getRawDatasetUsage } from './rawDatasetUsage';

function createCopy(isZh) {
  return {
    myHubTitle: isZh ? '数据总览原始数据集工作台' : 'Data Overview Raw Dataset Workbench',
    myHubDesc: isZh
      ? '这里管理用户上传的 MCD / OpenMARS / NOMAD 原始数据，并统一校验为数据总览可消费格式。MCD 可作为数据总览整页数据源；OpenMARS 与 NOMAD 仅用于三维臭氧多源展示。训练与预测数据由服务器后台管理。'
      : 'Manage uploaded MCD / OpenMARS / NOMAD raw datasets and validate them into Data Overview-ready formats. MCD can drive Data Overview; OpenMARS and NOMAD are only 3D ozone sources. Training and prediction data are server-managed.',
    contributionTitle: isZh ? '平台贡献流程' : 'Platform Contribution Flow',
    contributionDesc: isZh
      ? '只有通过基础校验并保持可管理状态的数据，才适合进入公共贡献。提交后会先进入管理员审核，而不会直接并入官方资产。'
      : 'Only datasets that passed base validation and remain manageable should enter public contribution. After submission they go to admin review instead of directly becoming official assets.',
    contributionOpen: isZh ? '打开贡献流程' : 'Open Contribution Flow',
    contributionHistory: isZh ? '查看贡献记录' : 'View Contribution History',
    contributionNone: isZh ? '当前没有可提交贡献的数据集。' : 'No datasets are currently eligible for contribution.',
    canUseTitle: isZh ? '数据总览使用条件' : 'Data Overview Usage',
    canUseDesc: isZh
      ? '判断该原始数据是否通过总览字段、Ls 与 5° 网格校验。普通用户上传数据不再参与训练或预测。'
      : 'Whether this raw dataset passes the overview field, Ls, and 5° grid checks. User uploads no longer feed training or prediction.',
    canContributeTitle: isZh ? '公共贡献条件' : 'Contribution Conditions',
    canContributeDesc: isZh
      ? '是否适合进入平台审核流程，争取成为平台公共数据资产。'
      : 'Whether it is suitable to enter the platform review flow and potentially become a shared platform asset.',
    lifeCycleTitle: isZh ? '生命周期状态' : 'Lifecycle Status',
    sourceModeTitle: isZh ? '数据总览用途' : 'Overview Role',
    sourceModeDesc: isZh
      ? 'MCD 用于整页数据总览；OpenMARS / NOMAD 仅用于 3D 臭氧多源图层。三类数据都会先标准化到总览使用的字段命名。'
      : 'MCD drives the full overview page; OpenMARS / NOMAD are only 3D ozone layers. All three are normalized to the field names consumed by Overview.',
    ruleValid: isZh ? '上传校验通过' : 'Upload validation passed',
    ruleBuildReady: isZh ? '文件状态可用于数据总览' : 'Upload status is usable for Data Overview',
    ruleAdopted: isZh ? '数据类型与字段契约属于 MCD / OpenMARS / NOMAD' : 'Type and field contract are MCD / OpenMARS / NOMAD',
    ruleRejected: isZh ? '用途限定在数据总览，不进入训练或预测' : 'Scoped to Data Overview, not training or prediction',
    ruleContributeBase: isZh ? '状态仍为 valid，可提交审核' : 'Status is still valid and can be submitted',
    ruleContributePending: isZh ? '已进入管理员审核队列' : 'Already in admin review queue',
    ruleContributeApproved: isZh ? '已并入平台官方数据资产' : 'Already merged into official platform assets',
    ruleContributeRejected: isZh ? '被驳回后需重新整理再上传' : 'Rejected datasets need a new upload and fix cycle',
    lifecycleUploading: isZh ? '上传后等待系统处理' : 'Uploaded and waiting for system processing',
    lifecycleBuilding: isZh ? '等待系统完成基础处理' : 'Waiting for base processing',
    lifecycleReady: isZh ? '可用于数据总览可视化' : 'Usable in Data Overview visualization',
    lifecyclePending: isZh ? '已提交平台审核' : 'Submitted for admin review',
    lifecycleApproved: isZh ? '已并入平台官方资产' : 'Merged into official platform assets',
    lifecycleRejected: isZh ? '审核未通过' : 'Rejected in review',
    lifecycleInvalid: isZh ? '基础校验未通过' : 'Failed base validation',
    lifecycleFallback: isZh ? '已上传但暂不可用于数据总览' : 'Uploaded but not currently usable in Data Overview',
    overviewTitle: isZh ? '当前工作重点' : 'Operational Focus',
    overviewDesc: isZh
      ? '先确认哪些数据已接入可用，哪些还在处理中，哪些值得送审，再进入单条数据的处理。'
      : 'Confirm what is already usable, what is still processing, and what is ready for review before working on a specific file.',
    filterAll: isZh ? '全部数据' : 'All datasets',
    filterUsable: isZh ? '可用于总览' : 'Overview usable',
    filterAttention: isZh ? '需要关注' : 'Needs attention',
    filterContribution: isZh ? '贡献流程' : 'Contribution flow',
    searchPlaceholder: isZh ? '按文件名、类型或火星年搜索' : 'Search by filename, type, or Mars year',
    queueTitle: isZh ? '数据队列' : 'Dataset Queue',
    queueDesc: isZh
      ? '先在队列里筛选和定位数据，再到右侧集中查看状态、动作和高级详情。'
      : 'Filter and locate datasets in the queue, then inspect status, actions, and advanced details on the right.',
    noMatchTitle: isZh ? '没有匹配的数据' : 'No matching datasets',
    noMatchDesc: isZh ? '试试切换筛选条件或修改搜索关键词。' : 'Try another filter or change the search term.',
    detailTitle: isZh ? '数据详情' : 'Dataset Detail',
    detailEmptyTitle: isZh ? '选择一条数据开始查看' : 'Select a dataset to inspect',
    detailEmptyDesc: isZh
      ? '从左侧数据队列中选中一条记录后，这里会集中展示它的接入状态、可执行动作和高级详情入口。'
      : 'Choose a record from the dataset queue to see its ingestion status, available actions, and advanced detail entry point.',
    advancedTitle: isZh ? '高级详情' : 'Advanced Details',
    advancedDesc: isZh
      ? '展开后查看校验规则、可用页面、元数据和图表预览。'
      : 'Expand to inspect validation rules, available pages, metadata, and preview charts.',
    advancedShow: isZh ? '展开高级详情' : 'Show advanced details',
    advancedHide: isZh ? '收起高级详情' : 'Hide advanced details',
    personalRoleTitle: isZh ? '三类个人数据' : 'Three Personal Source Types',
    uploadStatusTitle: isZh ? '上传状态' : 'Upload Status',
    rawMcdSource: isZh ? '原始 MCD 3h' : 'raw MCD 3h',
    cacheBuilding: isZh ? '缓存生成中' : 'Cache building',
    cacheReady: isZh ? '缓存完成' : 'Cache ready',
    cacheFailed: isZh ? '缓存失败' : 'Cache failed',
    cacheOverview: isZh ? '数据总览缓存' : 'Data Overview cache',
    cache3h: isZh ? '3h 数据缓存' : '3h data cache',
    accessTitle: isZh ? '可用页面' : 'Available In',
    accessEmpty: isZh ? '当前不能在数据总览中选择使用。' : 'This dataset cannot currently be selected in Data Overview.',
    contributionReadyStat: isZh ? '可送审' : 'Ready to submit',
    contributionReadyDesc: isZh ? '满足基础校验并可进入平台贡献的数据。' : 'Datasets that passed base validation and can enter platform contribution.',
    reviewQueueStat: isZh ? '审核中' : 'In review',
    reviewQueueDesc: isZh ? '已经提交给平台管理员审核的数据。' : 'Datasets already submitted to platform review.',
    approvedStat: isZh ? '已并入' : 'Approved',
    approvedDesc: isZh ? '已经进入平台官方数据资产层的数据。' : 'Datasets already merged into official platform assets.',
    buildPendingStat: isZh ? '处理中' : 'Building',
    buildPendingDesc: isZh ? '后台仍在校验或等待系统处理的数据。' : 'Datasets still validating or awaiting processing.',
    readySignal: isZh ? '可在数据总览中选择使用' : 'Ready for Data Overview selection',
    attentionSignal: isZh ? '建议优先关注这条数据的状态' : 'This item likely needs attention first',
    contributionSignal: isZh ? '满足公共贡献基础条件' : 'Eligible for public contribution',
    reviewSignalLabel: isZh ? '已进入平台审核流程' : 'Already in platform review',
    approvedSignalLabel: isZh ? '已并入官方数据资产' : 'Already merged into official assets',
    officialCenterLink: isZh ? '官方数据中心会显示已并入的平台数据状态。' : 'Approved platform datasets are reflected in the official dataset center above.',
    yes: isZh ? '是' : 'YES',
    no: isZh ? '否' : 'NO',
    unknownError: isZh ? '未知错误' : 'Unknown error',
    autoClose: isZh ? '后自动关闭' : 'auto close',
    uploadNetworkError: isZh ? '网络异常，请检查连接。' : 'Network error, please check your connection.',
  };
}

function CloudUploadIcon({ size = 40, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="16 16 12 12 8 16" />
      <line x1="12" y1="12" x2="12" y2="21" />
      <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
    </svg>
  );
}

function FolderOpenIcon({ size = 32, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function FileIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
      <polyline points="13 2 13 9 20 9" />
    </svg>
  );
}

function LockIcon({ size = 48, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

function SearchIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function formatFileSize(bytes) {
  if (!bytes && bytes !== 0) return '--';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatDate(iso) {
  if (!iso) return '--';
  return iso.replace('T', ' ').slice(0, 16);
}

function formatLsLabel(start, end, digits = 1) {
  if (start == null || end == null) return '--';
  return `${Number(start).toFixed(digits)}° - ${Number(end).toFixed(digits)}°`;
}

function MetaRow({ label, value }) {
  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
      <span style={{ fontSize: 'calc(10px * var(--font-scale, 1))', color: C.ice30, whiteSpace: 'nowrap', minWidth: 58 }}>{label}</span>
      <span style={{ fontSize: 'calc(12px * var(--font-scale, 1))', color: C.ice60, fontWeight: 500 }}>{value ?? '--'}</span>
    </div>
  );
}

function Badge({ label, color, bg }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '4px 10px',
        borderRadius: 999,
        background: bg,
        color,
        fontSize: 'calc(10px * var(--font-scale, 1))',
        fontWeight: 700,
        whiteSpace: 'nowrap',
      }}
    >
      {label}
    </span>
  );
}

function PersonalSummaryStrip({ summaryStats, t, copy }) {
  const items = [
    { label: t('explore.myData.statUploads'), value: summaryStats.total, accent: C.blue },
    { label: t('explore.myData.statUsable'), value: summaryStats.usable, accent: summaryStats.usable > 0 ? C.green : C.ice60 },
    { label: copy.reviewQueueStat, value: summaryStats.inReview, accent: summaryStats.inReview > 0 ? '#f59e0b' : C.ice60 },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
      {items.map((item) => (
        <div
          key={item.label}
          style={{
            minHeight: 68,
            padding: '11px 13px',
            borderRadius: 10,
            border: `1px solid ${C.border}`,
            background: 'rgba(255,255,255,0.025)',
          }}
        >
          <div style={{ fontSize: 'calc(10px * var(--font-scale, 1))', color: C.ice30, marginBottom: 5 }}>{item.label}</div>
          <div style={{ fontSize: 'calc(18px * var(--font-scale, 1))', color: item.accent, fontWeight: 700, fontFamily: "'Orbitron', sans-serif" }}>
            {item.value}
          </div>
        </div>
      ))}
    </div>
  );
}

function ActionBtn({ label, borderColor, textColor, activeBg, active, onClick, disabled, loading }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: '8px 13px',
        background: active ? activeBg : 'rgba(255,255,255,0.04)',
        border: `1px solid ${borderColor ?? C.border}`,
        borderRadius: 10,
        color: loading ? C.ice30 : textColor ?? C.ice60,
        fontSize: 'calc(11px * var(--font-scale, 1))',
        fontWeight: 700,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.6 : 1,
        transition: 'all 0.15s',
        whiteSpace: 'nowrap',
        fontFamily: 'inherit',
      }}
    >
      {loading ? '...' : label}
    </button>
  );
}

function FilterPill({ active, label, accent = C.blue, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '7px 12px',
        borderRadius: 999,
        border: `1px solid ${active ? accent : C.border}`,
        background: active ? `${accent}1a` : 'rgba(255,255,255,0.03)',
        color: active ? accent : C.ice60,
        fontSize: 'calc(11px * var(--font-scale, 1))',
        fontWeight: 700,
        cursor: 'pointer',
        fontFamily: 'inherit',
        transition: 'all 0.15s',
      }}
    >
      {label}
    </button>
  );
}

function EmptyState({ icon, title, desc }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        minHeight: 260,
        textAlign: 'center',
        padding: '32px 20px',
        border: `1px dashed ${C.border}`,
        borderRadius: 16,
        background: 'rgba(255,255,255,0.02)',
      }}
    >
      <div style={{ color: C.ice30, opacity: 0.55 }}>{icon}</div>
      <div style={{ fontSize: 'calc(15px * var(--font-scale, 1))', color: C.ice, fontWeight: 700 }}>{title}</div>
      <div style={{ maxWidth: 420, fontSize: 'calc(12px * var(--font-scale, 1))', color: C.ice30, lineHeight: 1.8 }}>{desc}</div>
    </div>
  );
}

function getUploadStatusMeta(status, t) {
  const palette = {
    valid: { color: C.blue, bg: 'rgba(74,158,255,0.1)' },
    cache_building: { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
    cache_failed: { color: C.mars, bg: 'rgba(199,91,57,0.1)' },
    invalid: { color: C.mars, bg: 'rgba(199,91,57,0.1)' },
    pending_review: { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
    approved: { color: C.green, bg: 'rgba(74,207,172,0.1)' },
    rejected: { color: C.mars, bg: 'rgba(199,91,57,0.1)' },
  };
  const base = palette[status] || { color: C.ice30, bg: 'rgba(255,255,255,0.05)' };
  return {
    ...base,
    label: status ? t(`explore.myData.status.${status}`) : '--',
  };
}

function normalizeType(type) {
  return String(type || '').trim().toLowerCase();
}

function getCacheItem(upload, cacheType) {
  const artifacts = Array.isArray(upload?.cache_status?.artifacts) ? upload.cache_status.artifacts : [];
  const jobs = Array.isArray(upload?.cache_status?.jobs) ? upload.cache_status.jobs : [];
  const artifact = artifacts.find((item) => item.cache_type === cacheType && item.status === 'ready');
  const job = jobs.find((item) => item.job_type === cacheType);
  return { artifact, job };
}

function getCacheLabel(upload, cacheType, copy) {
  const { artifact, job } = getCacheItem(upload, cacheType);
  if (artifact) return copy.cacheReady;
  if (job?.status === 'failed') return copy.cacheFailed;
  if (job) return `${copy.cacheBuilding}${job.progress != null ? ` ${Math.round(job.progress)}%` : ''}`;
  return normalizeType(upload?.data_type) === 'mcd' ? copy.cacheBuilding : '--';
}

function getOverviewRoleMeta(datasetUsage, isZh) {
  if (datasetUsage.key === 'overview_mcd') {
    return { label: isZh ? '整页 MCD' : 'Full-page MCD', color: C.green, bg: 'rgba(74,207,172,0.1)' };
  }
  if (datasetUsage.key === 'ozone_openmars') {
    return { label: '3D OpenMARS', color: C.blue, bg: 'rgba(74,158,255,0.1)' };
  }
  if (datasetUsage.key === 'ozone_nomad') {
    return { label: '3D NOMAD', color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' };
  }
  return { label: isZh ? '未进入总览' : 'Not in overview', color: C.ice30, bg: 'rgba(255,255,255,0.05)' };
}

function deriveDatasetState(upload, isZh) {
  const datasetUsage = getRawDatasetUsage(upload, isZh);
  const type = normalizeType(upload?.data_type);
  const color = datasetUsage.usable ? C.green : (upload?.status === 'invalid' || upload?.status === 'rejected' ? C.mars : C.ice30);

  return {
    key: datasetUsage.key,
    usable: datasetUsage.usable,
    label: datasetUsage.label,
    desc: upload?.validation_message && !datasetUsage.usable ? upload.validation_message : datasetUsage.desc,
    color,
    bg: datasetUsage.usable ? 'rgba(74,207,172,0.1)' : (color === C.mars ? 'rgba(199,91,57,0.1)' : 'rgba(255,255,255,0.05)'),
    modeMeta: getOverviewRoleMeta(datasetUsage, isZh),
    usagePages: datasetUsage.pages,
    typeSupported: ['mcd', 'openmars', 'nomad'].includes(type),
  };
}

function deriveAnalysisCondition(datasetState, copy) {
  return {
    ok: datasetState.usable,
    rules: [
      { label: copy.ruleValid, ok: datasetState.key !== 'unusable' },
      { label: copy.ruleBuildReady, ok: datasetState.usable },
      { label: copy.ruleAdopted, ok: datasetState.typeSupported },
      { label: copy.ruleRejected, ok: true },
    ],
  };
}

function deriveContributionCondition(upload, copy) {
  const status = upload?.status;
  const isValid = status === 'valid';
  const isPending = status === 'pending_review';
  const isApproved = status === 'approved';
  const isRejected = status === 'rejected';

  return {
    ok: isValid,
    rules: [
      { label: copy.ruleContributeBase, ok: isValid },
      { label: copy.ruleContributePending, ok: isPending },
      { label: copy.ruleContributeApproved, ok: isApproved },
      { label: copy.ruleContributeRejected, ok: !isRejected },
    ],
  };
}

function deriveLifecycleLabel(upload, datasetState, copy) {
  if (upload?.status === 'cache_building') return copy.cacheBuilding;
  if (upload?.status === 'cache_failed') return copy.cacheFailed;
  if (upload?.status === 'invalid') return copy.lifecycleInvalid;
  if (upload?.status === 'rejected') return copy.lifecycleRejected;
  if (upload?.status === 'approved') return copy.lifecycleApproved;
  if (upload?.status === 'pending_review') return copy.lifecyclePending;
  if (datasetState.usable) return copy.lifecycleReady;
  if (datasetState.key === 'unusable' || datasetState.key === 'unknown') return copy.lifecycleFallback;
  return copy.lifecycleUploading;
}

function isAttentionState(ctx) {
  return !ctx?.datasetState?.usable && !isContributionFlowItem(ctx);
}

function isContributionFlowItem(ctx) {
  if (!ctx?.upload) return false;
  return ctx.contributionCondition.ok || ctx.upload.status === 'pending_review' || ctx.upload.status === 'approved';
}

function deriveSignalMeta(ctx, copy) {
  if (ctx.upload.status === 'approved') {
    return {
      label: copy.approvedSignalLabel,
      color: C.green,
      bg: 'rgba(74,207,172,0.12)',
    };
  }
  if (ctx.upload.status === 'pending_review') {
    return {
      label: copy.reviewSignalLabel,
      color: '#f59e0b',
      bg: 'rgba(245,158,11,0.12)',
    };
  }
  if (ctx.datasetState.usable) {
    return {
      label: copy.readySignal,
      color: C.green,
      bg: 'rgba(74,207,172,0.12)',
    };
  }
  if (ctx.contributionCondition.ok) {
    return {
      label: copy.contributionSignal,
      color: C.blue,
      bg: 'rgba(74,158,255,0.12)',
    };
  }
  return {
    label: copy.attentionSignal,
    color: C.mars,
    bg: 'rgba(199,91,57,0.12)',
  };
}

function UploadZone({
  t,
  copy,
  uploadState,
  uploadProgress,
  uploadPhase,
  uploadResult,
  isDragging,
  fileInputRef,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileChange,
  onReset,
}) {
  const isIdle = uploadState === 'idle';
  const isUploading = uploadState === 'uploading';
  const isResult = uploadState === 'result';

  const resultOk = uploadResult?.ok;
  const hasWarnings = resultOk && uploadResult?.data?.warnings?.length > 0;
  const resultType = !uploadResult ? null : resultOk ? (hasWarnings ? 'warning' : 'success') : 'error';

  const borderColor = isDragging
    ? C.blue
    : isResult
      ? resultType === 'success'
        ? '#22c55e'
        : resultType === 'warning'
          ? '#f59e0b'
          : C.mars
      : C.border;

  const bgColor = isDragging
    ? 'rgba(74,158,255,0.05)'
    : isResult
      ? resultType === 'success'
        ? 'rgba(34,197,94,0.03)'
        : resultType === 'warning'
          ? 'rgba(245,158,11,0.03)'
          : 'rgba(199,91,57,0.03)'
      : 'transparent';

  const resultIconColor = resultType === 'success' ? '#22c55e' : resultType === 'warning' ? '#f59e0b' : C.mars;

  return (
    <div
      style={{
        border: `2px ${isDragging || isResult ? 'solid' : 'dashed'} ${borderColor}`,
        borderRadius: 18,
        background: bgColor,
        padding: '32px 28px',
        minHeight: 250,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: isIdle ? 'pointer' : 'default',
        transition: 'border-color 0.2s, background 0.2s',
        userSelect: 'none',
      }}
      onClick={() => isIdle && fileInputRef.current?.click()}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <input ref={fileInputRef} type="file" accept=".nc,.nc4,.netcdf" style={{ display: 'none' }} onChange={onFileChange} />

      {isIdle && (
        <>
          <div style={{ color: isDragging ? C.blue : C.ice30, marginBottom: 12, opacity: isDragging ? 1 : 0.72 }}>
            <CloudUploadIcon size={42} color="currentColor" />
          </div>
          <div
            style={{
              fontSize: 'calc(16px * var(--font-scale, 1))',
              fontWeight: 700,
              color: isDragging ? C.blue : C.ice,
              fontFamily: "'Orbitron', sans-serif",
              marginBottom: 8,
              textAlign: 'center',
            }}
          >
            {isDragging ? t('explore.upload.dragActive') : t('explore.upload.title')}
          </div>
          {!isDragging && (
            <>
              <div style={{ fontSize: 'calc(12px * var(--font-scale, 1))', color: C.ice60, maxWidth: 640, textAlign: 'center', lineHeight: 1.8, marginBottom: 18 }}>
                {t('explore.upload.subtitle')}
              </div>
              <div
                style={{
                  fontSize: 'calc(11px * var(--font-scale, 1))',
                  color: C.ice30,
                  border: `1px solid ${C.border}`,
                  borderRadius: 10,
                  padding: '6px 14px',
                  letterSpacing: 0.3,
                }}
              >
                {t('explore.upload.dragHint')} | {t('explore.upload.clickHint')}
              </div>
            </>
          )}
        </>
      )}

      {isUploading && (
        <div style={{ width: '100%', maxWidth: 520, display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ color: C.ice60, flexShrink: 0 }}>
              <FileIcon size={18} color="currentColor" />
            </span>
            <span style={{ fontSize: 'calc(12px * var(--font-scale, 1))', color: C.ice60, flex: 1 }}>
              {uploadPhase === 'validating' ? t('explore.upload.validating') : t('explore.upload.uploading')}
            </span>
            <span style={{ fontSize: 'calc(12px * var(--font-scale, 1))', color: C.ice60, fontWeight: 600, minWidth: 38, textAlign: 'right' }}>
              {uploadProgress}%
            </span>
          </div>
          <div style={{ height: 6, background: C.border, borderRadius: 999, overflow: 'hidden' }}>
            <div
              style={{
                height: '100%',
                width: `${uploadProgress}%`,
                background: uploadPhase === 'validating' ? '#f59e0b' : C.blue,
                borderRadius: 999,
                transition: 'width 0.25s ease, background 0.3s',
              }}
            />
          </div>
        </div>
      )}

      {isResult && uploadResult && (
        <div style={{ width: '100%', maxWidth: 620, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
            <div
              style={{
                fontSize: 'calc(14px * var(--font-scale, 1))',
                fontWeight: 700,
                color: resultIconColor,
                fontFamily: "'Orbitron', sans-serif",
                letterSpacing: 1,
              }}
            >
              {resultType === 'success' ? t('explore.upload.success') : resultType === 'warning' ? t('explore.upload.warning') : t('explore.upload.failed')}
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onReset();
              }}
              style={{
                background: 'none',
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                color: C.ice60,
                fontSize: 'calc(11px * var(--font-scale, 1))',
                cursor: 'pointer',
                padding: '4px 11px',
                fontFamily: 'inherit',
              }}
            >
              {t('explore.upload.closeResult')}
            </button>
          </div>

          {resultOk && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
                gap: '8px 16px',
                padding: '14px 16px',
                background: 'rgba(255,255,255,0.03)',
                border: `1px solid ${C.border}`,
                borderRadius: 12,
              }}
            >
              {uploadResult.data.data_type && <MetaRow label={t('explore.upload.dataType')} value={uploadResult.data.data_type} />}
              {uploadResult.data.mars_year != null && <MetaRow label={t('explore.upload.marsYear')} value={`MY ${uploadResult.data.mars_year}`} />}
              {uploadResult.data.ls_range?.[0] != null && uploadResult.data.ls_range?.[1] != null && (
                <MetaRow label={t('explore.upload.lsRange')} value={formatLsLabel(uploadResult.data.ls_range[0], uploadResult.data.ls_range[1])} />
              )}
              {uploadResult.data.grid_size && (
                <MetaRow label={t('explore.upload.gridSize')} value={`${uploadResult.data.grid_size[0]} x ${uploadResult.data.grid_size[1]}`} />
              )}
              {uploadResult.data.variables?.length > 0 && <MetaRow label={t('explore.upload.variables')} value={uploadResult.data.variables.join(', ')} />}
            </div>
          )}

          {hasWarnings && (
            <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: '#f59e0b', lineHeight: 1.6 }}>
              <span style={{ fontWeight: 700 }}>{t('explore.upload.warnings')}: </span>
              {uploadResult.data.warnings.join(' | ')}
            </div>
          )}

          {!resultOk && (
            <div
              style={{
                fontSize: 'calc(12px * var(--font-scale, 1))',
                color: C.mars,
                lineHeight: 1.6,
                padding: '10px 14px',
                background: 'rgba(199,91,57,0.06)',
                border: '1px solid rgba(199,91,57,0.25)',
                borderRadius: 10,
              }}
            >
              <span style={{ fontWeight: 700 }}>{t('explore.upload.errorDetail')}: </span>
              {uploadResult.data?.error || uploadResult.data?.detail || copy.unknownError}
            </div>
          )}

          <div style={{ fontSize: 'calc(10px * var(--font-scale, 1))', color: C.ice30, textAlign: 'right' }}>{`${resultOk ? '5s' : '8s'} ${copy.autoClose}`}</div>
        </div>
      )}
    </div>
  );
}

function RuleList({ title, ok, desc, rules, copy }) {
  return (
    <div
      style={{
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: '14px 15px',
        background: 'rgba(255,255,255,0.02)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
        <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice, fontWeight: 700 }}>{title}</div>
        <Badge
          label={ok ? copy.yes : copy.no}
          color={ok ? C.green : C.mars}
          bg={ok ? 'rgba(74,207,172,0.1)' : 'rgba(199,91,57,0.1)'}
        />
      </div>
      <div style={{ marginTop: 6, fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice30, lineHeight: 1.7 }}>{desc}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginTop: 12 }}>
        {rules.map((rule) => (
          <div key={rule.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: rule.ok ? C.green : C.mars,
                flexShrink: 0,
              }}
            />
            <span style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: rule.ok ? C.ice60 : C.ice30 }}>{rule.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CompactQueueItem({ ctx, selected, onSelect, copy }) {
  const { upload, datasetState, contributionCondition } = ctx;
  const signalMeta = deriveSignalMeta(ctx, copy);

  return (
    <button
      onClick={onSelect}
      style={{
        width: '100%',
        textAlign: 'left',
        border: `1px solid ${selected ? C.blue : C.border}`,
        borderRadius: 12,
        padding: '11px 12px',
        background: selected ? 'rgba(74,158,255,0.08)' : 'rgba(255,255,255,0.02)',
        cursor: 'pointer',
        fontFamily: 'inherit',
        transition: 'border-color 0.15s, background 0.15s',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: C.ice30, flexShrink: 0 }}>
              <FileIcon size={14} color="currentColor" />
            </span>
            <span
              style={{
                fontSize: 'calc(13px * var(--font-scale, 1))',
                fontWeight: 700,
                color: C.ice,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={upload.filename}
            >
              {upload.filename}
            </span>
          </div>
          <div style={{ marginTop: 7, display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice30 }}>
            <span>{upload.data_type || '--'}</span>
            <span>{upload.mars_year != null ? `MY ${upload.mars_year}` : 'MY --'}</span>
            <span>{formatLsLabel(upload.ls_start, upload.ls_end, 0)}</span>
          </div>
          {normalizeType(upload.data_type) === 'mcd' && (
            <div
              style={{
                display: 'grid',
                gap: 4,
                marginTop: 8,
                fontSize: 'calc(10px * var(--font-scale, 1))',
                color: C.ice30,
                lineHeight: 1.5,
              }}
            >
              <span style={{ color: C.ice60 }}>{copy.rawMcdSource}</span>
              <span>{copy.cacheOverview}: {getCacheLabel(upload, 'mcd_overview', copy)}</span>
              <span>{copy.cache3h}: {getCacheLabel(upload, 'mcd_3h', copy)}</span>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6, flexShrink: 0 }}>
          <Badge label={signalMeta.label} color={signalMeta.color} bg={signalMeta.bg} />
          {contributionCondition.ok && <Badge label={copy.contributionReadyStat} color={C.blue} bg="rgba(74,158,255,0.1)" />}
        </div>
      </div>
    </button>
  );
}

function SnapshotTile({ label, value, desc, accent = C.ice }) {
  return (
    <div
      style={{
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: '14px 15px',
        background: 'rgba(255,255,255,0.02)',
      }}
    >
      <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice30 }}>{label}</div>
      <div style={{ marginTop: 8, fontSize: 'calc(14px * var(--font-scale, 1))', color: accent, fontWeight: 700 }}>{value}</div>
      {desc ? <div style={{ marginTop: 7, fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice30, lineHeight: 1.7 }}>{desc}</div> : null}
    </div>
  );
}

function PersonalCoreDetail({
  copy,
  t,
  viewingCtx,
  viewingUpload,
  viewingSignal,
  viewActionLoading,
  onContribute,
  onDelete,
  advancedOpen,
  onToggleAdvanced,
  viewData,
  viewLs,
  onLsChange,
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 15 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 'calc(18px * var(--font-scale, 1))', color: C.ice, fontWeight: 700, lineHeight: 1.45 }} title={viewingUpload.filename}>
            {viewingUpload.filename}
          </div>
          <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <Badge label={viewingCtx.datasetState.label} color={viewingCtx.datasetState.color} bg={viewingCtx.datasetState.bg} />
            <Badge label={viewingCtx.uploadStatusMeta.label} color={viewingCtx.uploadStatusMeta.color} bg={viewingCtx.uploadStatusMeta.bg} />
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <ActionBtn
            label={viewingUpload.is_public ? t('explore.contribute.contributed') : t('explore.myData.contributeBtn')}
            borderColor={viewingCtx.contributionCondition.ok ? 'rgba(74,158,255,0.5)' : C.border}
            textColor={viewingCtx.contributionCondition.ok ? C.blue : C.ice30}
            activeBg="rgba(74,158,255,0.10)"
            onClick={onContribute}
            disabled={viewActionLoading || viewingUpload.is_public || !viewingCtx.contributionCondition.ok}
          />
          <HoldToDeleteButton
            size="compact"
            label={t('explore.myData.deleteBtn')}
            hint={t('explore.myData.deleteHoldHint')}
            doneLabel={t('explore.myData.deleteSuccess')}
            onConfirm={onDelete}
            persist={viewActionLoading}
          />
        </div>
      </div>

      {viewingSignal && (
        <div
          style={{
            border: `1px solid ${viewingSignal.color}33`,
            borderRadius: 12,
            padding: '10px 12px',
            background: viewingSignal.bg,
            color: viewingSignal.color,
            fontSize: 'calc(12px * var(--font-scale, 1))',
            fontWeight: 700,
          }}
        >
          {viewingSignal.label}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
        <SnapshotTile label={copy.lifeCycleTitle} value={viewingCtx.lifecycleLabel} accent={C.ice} />
        <SnapshotTile label={copy.sourceModeTitle} value={viewingCtx.datasetState.modeMeta.label} accent={viewingCtx.datasetState.modeMeta.color} />
        <SnapshotTile label={copy.uploadStatusTitle} value={viewingCtx.uploadStatusMeta.label} accent={viewingCtx.uploadStatusMeta.color} />
      </div>

      <button
        onClick={onToggleAdvanced}
        aria-expanded={advancedOpen}
        style={{
          alignSelf: 'flex-start',
          padding: '9px 14px',
          borderRadius: 10,
          border: `1px solid ${advancedOpen ? 'rgba(74,158,255,0.45)' : C.border}`,
          background: advancedOpen ? 'rgba(74,158,255,0.10)' : 'rgba(255,255,255,0.04)',
          color: advancedOpen ? C.blue : C.ice60,
          fontSize: 'calc(12px * var(--font-scale, 1))',
          fontWeight: 700,
          cursor: 'pointer',
          fontFamily: 'inherit',
        }}
      >
        {advancedOpen ? copy.advancedHide : copy.advancedShow}
      </button>

      <AdvancedDetailSection
        open={advancedOpen}
        copy={copy}
        t={t}
        viewingCtx={viewingCtx}
        viewingUpload={viewingUpload}
        viewData={viewData}
        viewLs={viewLs}
        onLsChange={onLsChange}
      />
    </div>
  );
}

function UsagePanel({ pages, copy }) {
  if (!pages.length) {
    return (
      <div
        style={{
          border: `1px solid ${C.border}`,
          borderRadius: 14,
          padding: '14px 15px',
          background: 'rgba(255,255,255,0.02)',
          fontSize: 'calc(11px * var(--font-scale, 1))',
          color: C.ice30,
          lineHeight: 1.7,
        }}
      >
        <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice, fontWeight: 700, marginBottom: 8 }}>{copy.accessTitle}</div>
        {copy.accessEmpty}
      </div>
    );
  }

  return (
    <div
      style={{
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: '14px 15px',
        background: 'rgba(255,255,255,0.02)',
      }}
    >
      <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice, fontWeight: 700, marginBottom: 10 }}>{copy.accessTitle}</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {pages.map((page) => (
          <Badge key={page} label={page} color={C.green} bg="rgba(74,207,172,0.1)" />
        ))}
      </div>
    </div>
  );
}

function PreviewContent({ viewData, viewLs, onLsChange, t }) {
  if (viewData.loading) return <LoadingBox h={320} />;

  if (viewData.error) {
    return (
      <div style={{ padding: '24px', textAlign: 'center', color: C.mars, fontSize: 'calc(13px * var(--font-scale, 1))', border: `1px dashed ${C.border}`, borderRadius: 14 }}>
        {viewData.error}
      </div>
    );
  }

  if (!viewData.summary) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {viewData.summary.has_ozone && (
        <>
          {viewData.summary.ls_range && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice60, fontFamily: "'Orbitron', sans-serif", letterSpacing: 1, whiteSpace: 'nowrap' }}>Ls</span>
              <input
                type="range"
                min={Math.round(viewData.summary.ls_range[0])}
                max={Math.round(viewData.summary.ls_range[1])}
                step={5}
                value={viewLs}
                onChange={(e) => onLsChange(Number(e.target.value))}
                style={{ flex: 1, accentColor: C.mars }}
              />
              <span style={{ fontSize: 'calc(14px * var(--font-scale, 1))', fontWeight: 700, color: C.mars, fontFamily: "'Orbitron', sans-serif", minWidth: 56, textAlign: 'right' }}>
                {viewLs}°
              </span>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            <div>
              <div style={{ fontSize: 'calc(10px * var(--font-scale, 1))', fontWeight: 700, color: C.mars, fontFamily: "'Orbitron', sans-serif", letterSpacing: 2, marginBottom: 8 }}>
                {t('explore.viewer.globeTitle')}
              </div>
              {viewData.globe ? <GlobePlot data={viewData.globe} h={260} /> : <LoadingBox h={260} />}
            </div>
            <div>
              <div style={{ fontSize: 'calc(10px * var(--font-scale, 1))', fontWeight: 700, color: C.blue, fontFamily: "'Orbitron', sans-serif", letterSpacing: 2, marginBottom: 8 }}>
                {t('explore.viewer.heatmapTitle')}
              </div>
              {viewData.heatmap ? <HeatmapCanvas data={viewData.heatmap} h={260} /> : <LoadingBox h={260} />}
            </div>
          </div>

          {viewData.bands && (
            <div>
              <div style={{ fontSize: 'calc(10px * var(--font-scale, 1))', fontWeight: 700, color: C.blue, fontFamily: "'Orbitron', sans-serif", letterSpacing: 2, marginBottom: 8 }}>
                {t('explore.viewer.bandsTitle')}
              </div>
              <LineChart data={viewData.bands} h={210} />
            </div>
          )}
        </>
      )}

      {!viewData.summary.has_ozone && viewData.summary.has_mcd_vars?.length > 0 && (
        <div style={{ padding: '32px 20px', textAlign: 'center', border: `1px dashed ${C.border}`, borderRadius: 14 }}>
          <div style={{ fontSize: 'calc(13px * var(--font-scale, 1))', color: C.ice60, marginBottom: 8 }}>{t('explore.viewer.mcdOnly')}</div>
          <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice30 }}>
            {t('explore.viewer.mcdVars')}: {viewData.summary.has_mcd_vars.join(', ')}
          </div>
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: '8px 16px',
          padding: '14px 16px',
          background: 'rgba(255,255,255,0.03)',
          border: `1px solid ${C.border}`,
          borderRadius: 14,
          fontSize: 'calc(11px * var(--font-scale, 1))',
        }}
      >
        <div>
          <span style={{ color: C.ice30 }}>{t('explore.viewer.dataType')}: </span>
          <span style={{ color: C.ice60 }}>{viewData.summary.data_type}</span>
        </div>
        <div>
          <span style={{ color: C.ice30 }}>{t('explore.viewer.gridSize')}: </span>
          <span style={{ color: C.ice60 }}>{viewData.summary.lat_points}x{viewData.summary.lon_points}</span>
        </div>
        <div>
          <span style={{ color: C.ice30 }}>{t('explore.viewer.lsPoints')}: </span>
          <span style={{ color: C.ice60 }}>{viewData.summary.ls_points}</span>
        </div>
        {viewData.summary.ls_range && (
          <div>
            <span style={{ color: C.ice30 }}>{t('explore.viewer.lsRange')}: </span>
            <span style={{ color: C.ice60 }}>
              {formatLsLabel(viewData.summary.ls_range[0], viewData.summary.ls_range[1])}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function AdvancedDetailSection({ open, copy, t, viewingCtx, viewingUpload, viewData, viewLs, onLsChange }) {
  if (!open) return null;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        borderTop: `1px solid ${C.border}`,
        paddingTop: 16,
      }}
    >
      <div>
        <div
          style={{
            fontSize: 'calc(11px * var(--font-scale, 1))',
            fontWeight: 700,
            color: C.blue,
            fontFamily: "'Orbitron', sans-serif",
            letterSpacing: 2,
            marginBottom: 8,
          }}
        >
          {copy.advancedTitle}
        </div>
        <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice30, lineHeight: 1.75 }}>
          {copy.advancedDesc}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
        <RuleList
          title={copy.canUseTitle}
          ok={viewingCtx.analysisCondition.ok}
          desc={copy.canUseDesc}
          rules={viewingCtx.analysisCondition.rules}
          copy={copy}
        />
        <RuleList
          title={copy.canContributeTitle}
          ok={viewingCtx.contributionCondition.ok}
          desc={copy.canContributeDesc}
          rules={viewingCtx.contributionCondition.rules}
          copy={copy}
        />
      </div>

      <UsagePanel pages={viewingCtx.datasetState.usagePages} copy={copy} />

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: '8px 14px',
          padding: '14px 16px',
          background: 'rgba(255,255,255,0.03)',
          border: `1px solid ${C.border}`,
          borderRadius: 14,
        }}
      >
        <MetaRow label={t('explore.myData.cardType')} value={viewingUpload.data_type || '--'} />
        <MetaRow label={t('explore.myData.cardMarsYear')} value={viewingUpload.mars_year != null ? `MY ${viewingUpload.mars_year}` : '--'} />
        <MetaRow label={t('explore.myData.cardLs')} value={formatLsLabel(viewingUpload.ls_start, viewingUpload.ls_end)} />
        <MetaRow label={t('explore.myData.cardSize')} value={formatFileSize(viewingUpload.file_size)} />
        <MetaRow label={t('explore.myData.cardTime')} value={formatDate(viewingUpload.created_at)} />
      </div>

      <div>
        <div
          style={{
            fontSize: 'calc(11px * var(--font-scale, 1))',
            fontWeight: 700,
            color: C.blue,
            fontFamily: "'Orbitron', sans-serif",
            letterSpacing: 2,
            marginBottom: 10,
          }}
        >
          {t('explore.myData.viewerTitle')}
        </div>
        <PreviewContent viewData={viewData} viewLs={viewLs} onLsChange={onLsChange} t={t} />
      </div>
    </div>
  );
}

function LoginPrompt({ t, openAuthModal }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 40px', gap: 20, textAlign: 'center' }}>
      <div style={{ color: C.ice30, opacity: 0.5 }}>
        <LockIcon size={52} color="currentColor" />
      </div>
      <div style={{ fontSize: 'calc(16px * var(--font-scale, 1))', color: C.ice, fontWeight: 600 }}>{t('explore.upload.loginPrompt')}</div>
      <button
        onClick={() => openAuthModal('login')}
        style={{
          padding: '10px 28px',
          borderRadius: 10,
          background: C.mars,
          border: 'none',
          color: '#fff',
          fontSize: 'calc(13px * var(--font-scale, 1))',
          fontWeight: 700,
          cursor: 'pointer',
          fontFamily: "'Orbitron', sans-serif",
          letterSpacing: 1,
        }}
      >
        {t('explore.upload.loginBtn')}
      </button>
    </div>
  );
}

export default function MyDataTab({ reviewSignal = 0 }) {
  const t = useT();
  const { user, openAuthModal } = useAuth();
  const { settings } = useSettings();
  const { showToast } = useToast();
  const isZh = settings.language !== 'en';
  const copy = useMemo(() => createCopy(isZh), [isZh]);

  const [uploadState, setUploadState] = useState('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadPhase, setUploadPhase] = useState('uploading');
  const [uploadResult, setUploadResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const [uploads, setUploads] = useState([]);
  const [uploadsLoading, setUploadsLoading] = useState(false);
  const [viewingId, setViewingId] = useState(null);
  const [activeFilter, setActiveFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  const [viewData, setViewData] = useState({
    summary: null,
    globe: null,
    heatmap: null,
    bands: null,
    loading: false,
    error: null,
  });
  const [viewLs, setViewLs] = useState(90);
  const [actionLoading, setActionLoading] = useState({});
  const [collapsingId, setCollapsingId] = useState(null);
  const [contributeOpen, setContributeOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const previewRequestRef = useRef(0);

  const loadWorkbench = useCallback(async () => {
    if (!user) return;
    setUploadsLoading(true);
    try {
      const uploadsRes = await getMyUploads();
      setUploads(Array.isArray(uploadsRes) ? uploadsRes : []);
    } catch (error) {
      console.error('Load uploads error:', error);
    } finally {
      setUploadsLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (user) loadWorkbench();
  }, [user, loadWorkbench, reviewSignal]);

  const loadViewData = useCallback(async (uploadId, requestId) => {
    setViewData({ summary: null, globe: null, heatmap: null, bands: null, loading: true, error: null });
    try {
      const summary = await fetchUserDataSummary(uploadId);
      if (requestId !== previewRequestRef.current) return;

      let defaultLs = 90;
      if (summary.ls_range) {
        defaultLs = Math.round((summary.ls_range[0] + summary.ls_range[1]) / 2);
      }
      setViewLs(defaultLs);

      let globe = null;
      let heatmap = null;
      let bands = null;
      if (summary.has_ozone) {
        [globe, heatmap, bands] = await Promise.all([
          fetchUserGlobeData(uploadId, defaultLs).catch(() => null),
          fetchUserHeatmap(uploadId, 'o3col').catch(() => null),
          fetchUserBands(uploadId).catch(() => null),
        ]);
      }

      if (requestId !== previewRequestRef.current) return;
      setViewData({ summary, globe, heatmap, bands, loading: false, error: null });
    } catch (e) {
      if (requestId !== previewRequestRef.current) return;
      setViewData((prev) => ({ ...prev, loading: false, error: e.message || t('common.error') }));
    }
  }, [t]);

  useEffect(() => {
    if (viewingId != null && advancedOpen) {
      const requestId = previewRequestRef.current + 1;
      previewRequestRef.current = requestId;
      loadViewData(viewingId, requestId);
    } else {
      previewRequestRef.current += 1;
      setViewData({ summary: null, globe: null, heatmap: null, bands: null, loading: false, error: null });
    }
  }, [viewingId, advancedOpen, loadViewData]);

  const onSelectDataset = useCallback((uploadId) => {
    previewRequestRef.current += 1;
    setAdvancedOpen(false);
    setViewingId(uploadId);
    setViewData({ summary: null, globe: null, heatmap: null, bands: null, loading: false, error: null });
  }, []);

  const handleViewLsChange = useCallback(async (newLs) => {
    setViewLs(newLs);
    if (viewingId == null) return;
    const requestId = previewRequestRef.current + 1;
    previewRequestRef.current = requestId;
    try {
      const globe = await fetchUserGlobeData(viewingId, newLs);
      if (requestId !== previewRequestRef.current) return;
      setViewData((prev) => ({ ...prev, globe }));
    } catch {
      // keep previous preview
    }
  }, [viewingId]);

  const uploadContexts = useMemo(() => {
    return uploads.map((upload) => {
      const datasetState = deriveDatasetState(upload, isZh);
      return {
        upload,
        uploadStatusMeta: getUploadStatusMeta(upload.status, t),
        datasetState,
        analysisCondition: deriveAnalysisCondition(datasetState, copy),
        contributionCondition: deriveContributionCondition(upload, copy),
        lifecycleLabel: deriveLifecycleLabel(upload, datasetState, copy),
      };
    });
  }, [uploads, isZh, t, copy]);

  const uploadContextMap = useMemo(() => new Map(uploadContexts.map((ctx) => [ctx.upload.id, ctx])), [uploadContexts]);
  const sortedUploadContexts = useMemo(() => {
    return [...uploadContexts].sort((a, b) => {
      const aCreated = a.upload.created_at || '';
      const bCreated = b.upload.created_at || '';
      if (aCreated === bCreated) return Number(b.upload.id || 0) - Number(a.upload.id || 0);
      return bCreated.localeCompare(aCreated);
    });
  }, [uploadContexts]);

  const validContributionUploads = useMemo(
    () => uploadContexts.filter((ctx) => ctx.contributionCondition.ok).map((ctx) => ctx.upload),
    [uploadContexts]
  );

  const summaryStats = useMemo(() => {
    const usable = uploadContexts.filter((ctx) => ctx.datasetState.usable).length;
    const building = uploadContexts.filter((ctx) => ctx.lifecycleLabel === copy.lifecycleUploading).length;
    const contributionReady = uploadContexts.filter((ctx) => ctx.contributionCondition.ok).length;
    const inReview = uploadContexts.filter((ctx) => ctx.upload.status === 'pending_review').length;
    const approved = uploadContexts.filter((ctx) => ctx.upload.status === 'approved').length;

    return {
      total: uploads.length,
      usable,
      building,
      contributionReady,
      inReview,
      approved,
    };
  }, [uploads.length, uploadContexts, copy.lifecycleUploading]);

  const filterDefs = useMemo(() => ([
    { key: 'all', label: copy.filterAll, accent: C.blue },
    { key: 'usable', label: copy.filterUsable, accent: C.green },
    { key: 'attention', label: copy.filterAttention, accent: C.mars },
    { key: 'contribution', label: copy.filterContribution, accent: '#f59e0b' },
  ]), [copy]);

  const filteredUploadContexts = useMemo(() => {
    const keyword = searchTerm.trim().toLowerCase();
    return sortedUploadContexts.filter((ctx) => {
      const matchesFilter = (() => {
        if (activeFilter === 'usable') return ctx.datasetState.usable;
        if (activeFilter === 'attention') return isAttentionState(ctx);
        if (activeFilter === 'contribution') return isContributionFlowItem(ctx);
        return true;
      })();

      if (!matchesFilter) return false;
      if (!keyword) return true;

      const haystack = [
        ctx.upload.filename,
        ctx.upload.data_type,
        ctx.upload.mars_year != null ? `my ${ctx.upload.mars_year}` : '',
        ctx.datasetState.label,
        ctx.lifecycleLabel,
      ]
        .join(' ')
        .toLowerCase();

      return haystack.includes(keyword);
    });
  }, [sortedUploadContexts, activeFilter, searchTerm]);

  useEffect(() => {
    if (!sortedUploadContexts.length) {
      setViewingId(null);
      return;
    }
    const validIds = new Set(sortedUploadContexts.map((ctx) => ctx.upload.id));
    if (!viewingId || !validIds.has(viewingId)) {
      setViewingId(sortedUploadContexts[0].upload.id);
    }
  }, [sortedUploadContexts, viewingId]);

  useEffect(() => {
    if (!filteredUploadContexts.length) return;
    const visibleIds = new Set(filteredUploadContexts.map((ctx) => ctx.upload.id));
    if (!visibleIds.has(viewingId)) {
      setViewingId(filteredUploadContexts[0].upload.id);
    }
  }, [filteredUploadContexts, viewingId]);

  const handleFile = useCallback((file) => {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['nc', 'nc4', 'netcdf'].includes(ext)) {
      showToast(t('explore.upload.wrongFormat'), 'error');
      return;
    }

    setUploadState('uploading');
    setUploadProgress(0);
    setUploadPhase('uploading');
    setUploadResult(null);

    const token = localStorage.getItem('aresvision_token');
    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const pct = Math.min(88, Math.round((e.loaded / e.total) * 88));
        setUploadProgress(pct);
      }
    });

    xhr.addEventListener('load', () => {
      setUploadPhase('validating');
      setUploadProgress(100);

      setTimeout(async () => {
        let payload;
        try {
          const data = JSON.parse(xhr.responseText);
          const ok = xhr.status >= 200 && xhr.status < 400 && data.status !== 'invalid';
          payload = { ok, data, status: xhr.status };
        } catch {
          payload = { ok: false, data: { detail: 'Response parse failed' }, status: xhr.status };
        }

        setUploadResult(payload);
        setUploadState('result');
        await loadWorkbench();
      }, 700);
    });

    xhr.addEventListener('error', () => {
      setUploadResult({ ok: false, data: { detail: copy.uploadNetworkError }, status: 0 });
      setUploadState('result');
    });

    xhr.open('POST', '/api/upload/nc');
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.send(formData);
  }, [loadWorkbench, showToast, t]);

  useEffect(() => {
    if (uploadState !== 'result') return;
    const delay = uploadResult?.ok ? 5000 : 8000;
    const timer = setTimeout(() => {
      setUploadState('idle');
      setUploadResult(null);
      setUploadProgress(0);
    }, delay);
    return () => clearTimeout(timer);
  }, [uploadState, uploadResult]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    if (uploadState === 'idle') setIsDragging(true);
  }, [uploadState]);

  const onDragLeave = useCallback(() => setIsDragging(false), []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onFileChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
    e.target.value = '';
  }, [handleFile]);

  const handleDeleteConfirm = async (id) => {
    if (id == null) return;
    setActionLoading((prev) => ({ ...prev, [id]: true }));
    try {
      await deleteUpload(id);
      // 先播放收起动画，再从列表移除，避免条目瞬间消失
      setCollapsingId(id);
      showToast(t('explore.myData.deleteSuccess'), 'success');
      window.setTimeout(() => {
        setUploads((prev) => prev.filter((u) => u.id !== id));
        setCollapsingId(null);
        if (viewingId === id) setViewingId(null);
        loadWorkbench();
      }, 240);
    } catch (e) {
      showToast(e.message || 'Delete failed', 'error');
    } finally {
      setActionLoading((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }
  };

  if (!user) {
    return <LoginPrompt t={t} openAuthModal={openAuthModal} />;
  }

  const viewingUpload = viewingId != null ? uploads.find((u) => u.id === viewingId) : null;
  const viewingCtx = viewingId != null ? uploadContextMap.get(viewingId) : null;
  const viewingSignal = viewingCtx ? deriveSignalMeta(viewingCtx, copy) : null;
  const viewActionLoading = viewingUpload ? actionLoading[viewingUpload.id] : false;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
      <GlowCard style={{ padding: '22px 24px', background: 'linear-gradient(135deg, rgba(74,158,255,0.08), rgba(255,255,255,0.02))' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div>
            <div
              style={{
                fontSize: 'calc(11px * var(--font-scale, 1))',
                fontWeight: 700,
                color: C.blue,
                fontFamily: "'Orbitron', sans-serif",
                letterSpacing: 2,
                marginBottom: 10,
              }}
            >
              {t('explore.myData.ingestTitle')}
            </div>
            <div style={{ fontSize: 'calc(20px * var(--font-scale, 1))', color: C.ice, fontWeight: 700, lineHeight: 1.4 }}>{copy.myHubTitle}</div>
            <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              <Badge label={t('explore.defaultDataset.heroTags.mcd')} color={C.mars} bg="rgba(199,91,57,0.1)" />
              <Badge label={t('explore.defaultDataset.heroTags.openmars')} color={C.blue} bg="rgba(74,158,255,0.1)" />
              <Badge label="NOMAD" color={C.green} bg="rgba(74,207,172,0.1)" />
            </div>
          </div>

          <PersonalSummaryStrip summaryStats={summaryStats} t={t} copy={copy} />

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 18, alignItems: 'start' }}>
            <UploadZone
              t={t}
              copy={copy}
              uploadState={uploadState}
              uploadProgress={uploadProgress}
              uploadPhase={uploadPhase}
              uploadResult={uploadResult}
              isDragging={isDragging}
              fileInputRef={fileInputRef}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onFileChange={onFileChange}
              onReset={() => {
                setUploadState('idle');
                setUploadResult(null);
                setUploadProgress(0);
              }}
            />

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div
                style={{
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: '15px 16px',
                  background: 'rgba(255,255,255,0.025)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 13,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                  <div style={{ fontSize: 'calc(12px * var(--font-scale, 1))', fontWeight: 700, color: C.ice }}>
                    {copy.contributionTitle}
                  </div>
                  <Badge label={`${copy.contributionReadyStat}: ${summaryStats.contributionReady}`} color={summaryStats.contributionReady > 0 ? C.blue : C.ice30} bg="rgba(255,255,255,0.05)" />
                </div>
                <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice30, lineHeight: 1.7 }}>
                  {validContributionUploads.length > 0
                    ? `${copy.contributionReadyStat}: ${summaryStats.contributionReady} | ${copy.reviewQueueStat}: ${summaryStats.inReview}`
                    : copy.contributionNone}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                  <button
                    onClick={() => setContributeOpen(true)}
                    disabled={validContributionUploads.length === 0}
                    style={{
                      padding: '9px 16px',
                      borderRadius: 10,
                      border: '1px solid rgba(74,158,255,0.35)',
                      background: validContributionUploads.length > 0 ? 'rgba(74,158,255,0.12)' : 'rgba(255,255,255,0.04)',
                      color: validContributionUploads.length > 0 ? C.blue : C.ice30,
                      fontSize: 'calc(12px * var(--font-scale, 1))',
                      fontWeight: 700,
                      cursor: validContributionUploads.length > 0 ? 'pointer' : 'not-allowed',
                      fontFamily: 'inherit',
                    }}
                  >
                    {copy.contributionOpen}
                  </button>
                  <button
                    onClick={() => setHistoryOpen(true)}
                    style={{
                      padding: '9px 16px',
                      borderRadius: 10,
                      border: `1px solid ${C.border}`,
                      background: 'rgba(255,255,255,0.04)',
                      color: C.ice60,
                      fontSize: 'calc(12px * var(--font-scale, 1))',
                      fontWeight: 600,
                      cursor: 'pointer',
                      fontFamily: 'inherit',
                    }}
                  >
                    {copy.contributionHistory}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </GlowCard>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 18, alignItems: 'start' }}>
        <GlowCard style={{ padding: '18px 18px 16px' }}>
          <div
            style={{
              fontSize: 'calc(11px * var(--font-scale, 1))',
              fontWeight: 700,
              color: C.blue,
              fontFamily: "'Orbitron', sans-serif",
              letterSpacing: 2,
              marginBottom: 8,
            }}
          >
            {copy.queueTitle}
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
            {filterDefs.map((filter) => (
              <FilterPill
                key={filter.key}
                active={activeFilter === filter.key}
                label={filter.label}
                accent={filter.accent}
                onClick={() => setActiveFilter(filter.key)}
              />
            ))}
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              border: `1px solid ${C.border}`,
              borderRadius: 12,
              padding: '10px 12px',
              background: 'rgba(255,255,255,0.02)',
              marginBottom: 14,
            }}
          >
            <span style={{ color: C.ice30, flexShrink: 0 }}>
              <SearchIcon size={16} color="currentColor" />
            </span>
            <input
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder={copy.searchPlaceholder}
              style={{
                flex: 1,
                border: 'none',
                outline: 'none',
                background: 'transparent',
                color: C.ice,
                fontSize: 'calc(12px * var(--font-scale, 1))',
                fontFamily: 'inherit',
              }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, gap: 10 }}>
            <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: C.ice30 }}>{uploadsLoading ? t('common.loading') : t('explore.myData.count', { n: filteredUploadContexts.length })}</div>
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: C.ice30,
                  fontSize: 'calc(11px * var(--font-scale, 1))',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                {t('explore.myData.cancelBtn')}
              </button>
            )}
          </div>

          {uploadsLoading && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }} data-queue-skeleton="true">
              {[0, 1, 2].map((index) => (
                <span
                  key={index}
                  className="ui-skeleton"
                  style={{ height: 62, borderRadius: 12, animationDelay: `${index * 90}ms` }}
                />
              ))}
            </div>
          )}

          {!uploadsLoading && !sortedUploadContexts.length && (
            <EmptyState
              icon={<FolderOpenIcon size={34} color="currentColor" />}
              title={t('explore.myData.emptyHint')}
              desc={t('explore.myData.ingestDesc')}
            />
          )}

          {!uploadsLoading && !!sortedUploadContexts.length && !filteredUploadContexts.length && (
            <EmptyState
              icon={<SearchIcon size={28} color="currentColor" />}
              title={copy.noMatchTitle}
              desc={copy.noMatchDesc}
            />
          )}

          {!uploadsLoading && filteredUploadContexts.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9, maxHeight: 820, overflowY: 'auto', paddingRight: 2 }}>
              {filteredUploadContexts.map((ctx, index) => (
                <div
                  key={ctx.upload.id}
                  className={`motion-stagger-item${collapsingId === ctx.upload.id ? ' motion-collapsing' : ''}`}
                  style={staggerStyle(index)}
                >
                  <CompactQueueItem
                    ctx={ctx}
                    selected={viewingId === ctx.upload.id}
                    onSelect={() => onSelectDataset(ctx.upload.id)}
                    copy={copy}
                  />
                </div>
              ))}
            </div>
          )}
        </GlowCard>

        <GlowCard breathe style={{ padding: 22 }}>
          <div
            style={{
              fontSize: 'calc(11px * var(--font-scale, 1))',
              fontWeight: 700,
              color: C.blue,
              fontFamily: "'Orbitron', sans-serif",
              letterSpacing: 2,
              marginBottom: 10,
            }}
          >
            {copy.detailTitle}
          </div>

          {!viewingUpload || !viewingCtx ? (
            <EmptyState
              icon={<FileIcon size={34} color="currentColor" />}
              title={copy.detailEmptyTitle}
              desc={copy.detailEmptyDesc}
            />
          ) : (
            <PersonalCoreDetail
              copy={copy}
              t={t}
              viewingCtx={viewingCtx}
              viewingUpload={viewingUpload}
              viewingSignal={viewingSignal}
              viewActionLoading={viewActionLoading}
              onContribute={() => setContributeOpen(true)}
              onDelete={() => handleDeleteConfirm(viewingUpload.id)}
              advancedOpen={advancedOpen}
              onToggleAdvanced={() => setAdvancedOpen((value) => !value)}
              viewData={viewData}
              viewLs={viewLs}
              onLsChange={handleViewLsChange}
            />
          )}
        </GlowCard>
      </div>

      <>
        <ContributeModal
          open={contributeOpen}
          onClose={() => setContributeOpen(false)}
          validUploads={validContributionUploads}
          onDone={loadWorkbench}
        />

        <ContributeHistoryPanel open={historyOpen} onClose={() => setHistoryOpen(false)} />
      </>
    </div>
  );
}
