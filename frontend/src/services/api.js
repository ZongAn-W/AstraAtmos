/**
 * AstraAtmos API 请求封装
 * Vite proxy: 前端 /api/* → localhost:8000/api/*
 */

import { handleUnauthorizedResponse } from '../stores/authPredictionSession.js';
import { researchExportErrorMessage } from './researchExportErrors.js';

const BASE = '/api';

// ─── 认证工具 ───

/** 从 localStorage 读取 token，自动附加到请求头 */
async function authedFetch(url, options = {}) {
  const token = localStorage.getItem('astraatmos_token');
  const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData;
  const headers = {
    ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
    ...options.headers,
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(url, { ...options, headers });
  // Token 过期或无效时清理私有预测会话并触发 AuthContext 全局登出。
  handleUnauthorizedResponse(res);
  return res;
}

function resolveDataSource(optionsOrSource) {
  if (typeof optionsOrSource === 'string') return optionsOrSource;
  return optionsOrSource?.dataSource || 'default';
}

function appendDataSource(url, optionsOrSource) {
  const dataSource = resolveDataSource(optionsOrSource);
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}data_source=${encodeURIComponent(dataSource)}`;
}

function appendQueryParams(url, params) {
  const query = params instanceof URLSearchParams ? params.toString() : String(params || '');
  if (!query) return url;
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}${query}`;
}

function buildOverviewSourceQuery(options = {}) {
  const params = new URLSearchParams();
  const mcdUploadId = options?.mcdUploadId ?? options?.mcd_upload_id;
  const openmarsUploadId = options?.openmarsUploadId ?? options?.openmars_upload_id;
  const nomadUploadId = options?.nomadUploadId ?? options?.nomad_upload_id;
  if (mcdUploadId) params.set('mcd_upload_id', String(mcdUploadId));
  if (openmarsUploadId) params.set('openmars_upload_id', String(openmarsUploadId));
  if (nomadUploadId) params.set('nomad_upload_id', String(nomadUploadId));
  return params;
}

function appendOverviewSource(url, options = {}) {
  return appendQueryParams(url, buildOverviewSourceQuery(options));
}

async function throwResponseError(res) {
  const payload = await res.json().catch(() => null);
  const detail = payload?.detail;
  const message = typeof detail === 'string'
    ? detail
    : detail != null
      ? JSON.stringify(detail)
      : `${res.status} ${res.statusText}`.trim();
  throw new Error(message);
}

// ─── 认证接口 ───

export async function apiLogin(email, password) {
  const res = await fetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function apiRegister(email, username, password, verificationCode) {
  const res = await fetch(`${BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, username, password, verification_code: verificationCode }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function apiSendCode(email, purpose = 'register') {
  const res = await fetch(`${BASE}/auth/send-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, purpose }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function apiResetPassword(email, verificationCode, newPassword) {
  const res = await fetch(`${BASE}/auth/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, verification_code: verificationCode, new_password: newPassword }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function apiGetMe() {
  const res = await authedFetch(`${BASE}/auth/me`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

export async function apiChangePassword(oldPassword, newPassword) {
  const res = await authedFetch(`${BASE}/auth/change-password`, {
    method: 'PUT',
    body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function fetchGlobeData(marsYear = 27, ls = 10, variableOrSignal = 'o3col', maybeSignal = null, options = {}) {
  const variable = typeof variableOrSignal === 'string' ? variableOrSignal : 'o3col';
  const signal = typeof variableOrSignal === 'string' ? maybeSignal : variableOrSignal;
  const opts = signal ? { signal } : {};
  const url = appendDataSource(`${BASE}/explore/globe?my=${marsYear}&ls=${ls}&variable=${encodeURIComponent(variable)}`, options);
  const res = await authedFetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchSeasonalHeatmap(marsYear = 27, options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/seasonal-heatmap?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchSeasonalBands(marsYear = 27, options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/seasonal-bands?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchEnvHeatmap(marsYear = 27, variable, options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/env-heatmap?my=${marsYear}&variable=${variable}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchCorrelation(marsYear = 27, options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/correlation?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchDataInfo(options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/info`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewInfo(options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/info`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewGlobeData(marsYear = 27, ls = 10, variable = 'o3col', signal = null, options = {}) {
  const opts = signal ? { signal } : {};
  const url = appendOverviewSource(`${BASE}/explore/overview/globe?my=${marsYear}&ls=${ls}&variable=${encodeURIComponent(variable)}`, options);
  const res = await authedFetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewOzoneSources(marsYear = 27, ls = 10, options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/ozone-sources?my=${marsYear}&ls=${ls}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewPointProbe(marsYear = 27, lat = 0, lng = 0, ls = 0, variable = 'o3col', options = {}) {
  const { signal, ...sourceOptions } = options || {};
  const url = appendOverviewSource(
    `${BASE}/explore/overview/point-probe?my=${marsYear}&lat=${lat}&lng=${lng}&ls=${ls}&variable=${encodeURIComponent(variable)}`,
    sourceOptions,
  );
  const res = await authedFetch(url, signal ? { signal } : {});
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewSeasonalHeatmap(marsYear = 27, options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/seasonal-heatmap?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewEnvHeatmap(marsYear = 27, variable, options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/env-heatmap?my=${marsYear}&variable=${variable}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewCorrelation(marsYear = 27, options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/correlation?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewCouplingData(marsYear = 27, var1 = 'o3col', var2 = 'Temperature', options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/coupling?my=${marsYear}&var1=${var1}&var2=${var2}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewZonalAnomaly(marsYear = 27, variable = 'o3col', options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/zonal-anomaly?my=${marsYear}&variable=${variable}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewSolarPhotochemical(marsYear = 27, latBand = 'Equatorial (30S-30N)', options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/solar-photochemical?my=${marsYear}&lat_band=${encodeURIComponent(latBand)}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewPolarDynamics(marsYear = 27, options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/polar-dynamics?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewResearchSuite(marsYear = 27, options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/research-suite?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewPhaseSpace(marsYear = 27, driver = 'Temperature', options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/phase-space?my=${marsYear}&driver=${encodeURIComponent(driver)}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchOverviewDiurnal(marsYear = 27, ls = 90, latBand = 'Equatorial (30S-30N)', options = {}) {
  const res = await authedFetch(appendOverviewSource(`${BASE}/explore/overview/diurnal?my=${marsYear}&ls=${ls}&lat_band=${encodeURIComponent(latBand)}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchCouplingData(marsYear = 27, var1 = 'o3col', var2 = 'Dust_Optical_Depth', options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/coupling?my=${marsYear}&var1=${var1}&var2=${var2}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchZonalAnomaly(marsYear = 27, variable = 'o3col', options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/zonal-anomaly?my=${marsYear}&variable=${variable}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchSolarPhotochemical(marsYear = 27, latBand = 'Equatorial (30S-30N)', options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/solar-photochemical?my=${marsYear}&lat_band=${encodeURIComponent(latBand)}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchPolarDynamics(marsYear = 27, options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/polar-dynamics?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchResearchSuite(marsYear = 27, options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/research-suite?my=${marsYear}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchPhaseSpace(marsYear = 27, driver = 'Dust_Optical_Depth', options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/explore/phase-space?my=${marsYear}&driver=${encodeURIComponent(driver)}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function runPrediction(body, options = {}) {
  const url = appendDataSource(`${BASE}/predict/run`, options);
  const res = await authedFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: options.signal,
  });
  if (!res.ok) await throwResponseError(res);
  return res.json();
}

export async function fetchPredictMetrics(body, options = {}) {
  const url = appendDataSource(`${BASE}/predict/metrics`, options);
  const res = await authedFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: options.signal,
  });
  if (!res.ok) await throwResponseError(res);
  return res.json();
}

export async function fetchDiurnal(marsYear = 27, ls = 90, latBand = 'Equatorial (30S-30N)', options = {}) {
  const res = await authedFetch(appendDataSource(`${BASE}/predict/diurnal?my=${marsYear}&ls=${ls}&lat_band=${encodeURIComponent(latBand)}`, options));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchErrorDistribution(vars = [], options = {}) {
  const varsStr = vars.length > 0 ? vars.join(',') : 'Temperature,Dust_Optical_Depth,Solar_Flux_DN,U_Wind,V_Wind';
  const params = new URLSearchParams({ vars: varsStr });
  if (options.trainingTaskId) params.set('training_task_id', String(options.trainingTaskId));
  if (options.horizon != null) params.set('horizon', String(options.horizon));
  const res = await authedFetch(`${BASE}/predict/error-distribution?${params.toString()}`, {
    signal: options.signal,
  });
  if (!res.ok) await throwResponseError(res);
  return res.json();
}

export async function compareTrainingModels(taskIds, options = {}) {
  const res = await authedFetch(`${BASE}/predict/training-models/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      task_ids: taskIds,
      horizon: options.horizon ?? 3,
    }),
    signal: options.signal,
  });
  if (!res.ok) await throwResponseError(res);
  return res.json();
}

export async function compareTrainingModelErrorDistributions(taskIds, options = {}) {
  const res = await authedFetch(`${BASE}/predict/training-models/compare-error-distribution`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      task_ids: taskIds,
      horizon: options.horizon ?? 3,
    }),
    signal: options.signal,
  });
  if (!res.ok) await throwResponseError(res);
  return res.json();
}

export async function compareTrainingModelPfi(taskIds, options = {}) {
  const res = await authedFetch(`${BASE}/predict/training-models/compare-pfi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      task_ids: taskIds,
      horizon: options.horizon ?? 3,
    }),
    signal: options.signal,
  });
  if (!res.ok) await throwResponseError(res);
  return res.json();
}

export async function fetchPermutationImportance(vars = [], options = {}) {
  const varsStr = vars.length > 0 ? vars.join(',') : 'Temperature,Dust_Optical_Depth,Solar_Flux_DN,U_Wind,V_Wind';
  const params = new URLSearchParams({ vars: varsStr });
  if (options.trainingTaskId) params.set('training_task_id', String(options.trainingTaskId));
  if (options.lsStart != null) params.set('ls_start', String(options.lsStart));
  if (options.horizon != null) params.set('horizon', String(options.horizon));
  const res = await authedFetch(`${BASE}/predict/permutation-importance?${params.toString()}`, {
    signal: options.signal,
  });
  if (!res.ok) await throwResponseError(res);
  return res.json();
}
// ─── 上传接口 ───

export async function getMyUploads() {
  const res = await authedFetch(`${BASE}/upload/my-uploads`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function deleteUpload(uploadId) {
  const res = await authedFetch(`${BASE}/upload/${uploadId}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function contributeUpload(uploadId, description = '') {
  const res = await authedFetch(`${BASE}/upload/${uploadId}/contribute`, {
    method: 'POST',
    body: JSON.stringify({ description }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getPendingReviews() {
  const res = await authedFetch(`${BASE}/upload/pending-reviews`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function reviewUpload(uploadId, action, reason = '') {
  const res = await authedFetch(`${BASE}/upload/${uploadId}/review`, {
    method: 'POST',
    body: JSON.stringify({ action, reason }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getApprovedDatasets() {
  const res = await authedFetch(`${BASE}/upload/approved-datasets`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function revokeDataset(uploadId) {
  const res = await authedFetch(`${BASE}/upload/${uploadId}/revoke`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getDataGovernanceOverview(scope = 'mine') {
  const res = await authedFetch(`${BASE}/upload/governance/overview?scope=${encodeURIComponent(scope)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getDataGovernanceQuality(uploadId) {
  const res = await authedFetch(`${BASE}/upload/governance/quality/${uploadId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getDataGovernanceLineage(uploadId) {
  const res = await authedFetch(`${BASE}/upload/governance/lineage/${uploadId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getNotifications() {
  const res = await authedFetch(`${BASE}/notification/list`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getUnreadCount() {
  const res = await authedFetch(`${BASE}/notification/unread-count`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

export async function markNotificationRead(id) {
  const res = await authedFetch(`${BASE}/notification/mark-read/${id}`, { method: 'POST' });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

export async function markAllNotificationsRead() {
  const res = await authedFetch(`${BASE}/notification/mark-all-read`, { method: 'POST' });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

// ─── 反馈接口 ───

export async function submitFeedback(formData) {
  const token = localStorage.getItem('astraatmos_token');
  const headers = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${BASE}/feedback/submit`, {
    method: 'POST',
    headers,
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getFeedbackList(status = '') {
  const url = status ? `${BASE}/feedback/list?status=${encodeURIComponent(status)}` : `${BASE}/feedback/list`;
  const res = await authedFetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function getFeedbackCount() {
  const res = await authedFetch(`${BASE}/feedback/count`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function resolveFeedback(feedbackId) {
  const res = await authedFetch(`${BASE}/feedback/${feedbackId}/resolve`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

// ─── 用户数据分析接口 ───

export async function fetchUserDataSummary(uploadId) {
  const res = await authedFetch(`${BASE}/user-data/summary/${uploadId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function fetchUserGlobeData(uploadId, ls = 10) {
  const res = await authedFetch(`${BASE}/user-data/globe/${uploadId}?ls=${ls}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function fetchUserHeatmap(uploadId, variable = 'o3col') {
  const res = await authedFetch(`${BASE}/user-data/seasonal-heatmap/${uploadId}?variable=${encodeURIComponent(variable)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function fetchUserBands(uploadId) {
  const res = await authedFetch(`${BASE}/user-data/seasonal-bands/${uploadId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function aiChat(question, context = null, history = null) {
  const payload = { question, context };
  if (Array.isArray(history) && history.length > 0) {
    payload.history = history;
  }
  const res = await fetch(`${BASE}/ai/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ─── 模型训练接口 ───

export async function copilotChat(question, context = null, history = null) {
  const payload = { question, context };
  if (Array.isArray(history) && history.length > 0) {
    payload.history = history;
  }
  const res = await fetch(`${BASE}/copilot/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchScripts() {
  const res = await authedFetch(`${BASE}/training/scripts`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

/**
 * 把训练启动失败的响应体翻译成可展示的错误。
 *
 * 后端在数据集与训练契约拒绝时返回 `{detail: {code, message}}`，Pydantic 校验
 * 失败时返回字符串或数组 detail。这里保留 code 与 HTTP 状态，避免前端只显示
 * 裸状态码，也避免把结构化错误当成普通字符串丢掉。
 */
async function buildTrainingStartError(res) {
  const payload = await res.json().catch(() => null);
  const detail = payload?.detail;
  const structured = detail && typeof detail === 'object' && !Array.isArray(detail) ? detail : null;
  let message = '';
  if (structured?.message) message = String(structured.message);
  else if (typeof detail === 'string') message = detail;
  else if (Array.isArray(detail)) {
    message = detail
      .map((item) => {
        const loc = Array.isArray(item?.loc) ? item.loc.filter((part) => part !== 'body').join('.') : '';
        return loc ? `${loc}: ${item?.msg || ''}` : (item?.msg || '');
      })
      .filter(Boolean)
      .join('; ');
  }
  const error = new Error(message || `${res.status}`);
  error.status = res.status;
  error.code = structured?.code || 'training_start_failed';
  error.availabilityReason = structured?.availability_reason ?? null;
  return error;
}

export async function startTrainingTask(
  model_script,
  hyperparameters,
  model_name = null,
  data_source = 'default',
  options = {}
) {
  const body = {
    model_script,
    hyperparameters,
    model_name,
    data_source,
    model_source: options.modelSource || 'official',
    uploaded_model_id: options.uploadedModelId || null,
    tag_ids: options.tagIds || [],
  };
  // 顶层 dataset_id 是服务端身份入口；未提供时保持旧请求形态，由后端按
  // legacy hyperparameters.training_dataset 解析。
  if (options.datasetId) body.dataset_id = options.datasetId;
  const res = await authedFetch(`${BASE}/training/start`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await buildTrainingStartError(res);
  return res.json();
}

export async function uploadTrainingWeight(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await authedFetch(`${BASE}/training/weights`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function fetchTrainingWeights() {
  const res = await authedFetch(`${BASE}/training/weights`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function deleteTrainingWeight(weightId) {
  const res = await authedFetch(`${BASE}/training/weights/${encodeURIComponent(weightId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function uploadUserModel(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await authedFetch(`${BASE}/user-models`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function fetchUserModels() {
  const res = await authedFetch(`${BASE}/user-models`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export function getUserModelDownloadUrl(kind) {
  return `${BASE}/user-models/downloads/${encodeURIComponent(kind)}`;
}

export async function researchExportCapabilities({ signal } = {}) {
  const res = await authedFetch(`${BASE}/predict/research-export/capabilities`, { signal });
  if (!res.ok) await throwResponseError(res);
  return res.json();
}

export async function requestResearchExport(payload, { preview = false, signal, language = 'zh' } = {}) {
  const res = await authedFetch(`${BASE}/predict/research-export/${preview ? 'preview' : 'download'}`, {
    method: 'POST', body: JSON.stringify(payload), signal,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(researchExportErrorMessage(body?.detail, res.status, language));
  }
  return res.blob();
}

export async function renameUserModel(modelId, displayName) {
  const res = await authedFetch(`${BASE}/user-models/${encodeURIComponent(modelId)}`, {
    method: 'PATCH',
    body: JSON.stringify({ display_name: displayName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function downloadUserModel(modelId) {
  const res = await authedFetch(`${BASE}/user-models/${encodeURIComponent(modelId)}/download`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.blob();
}

/**
 * 上传模型的 Earth 兼容性结论。
 *
 * “Mars 可用”不等于“Earth 可用”，所以训练页在选择 Earth + 上传模型时必须问服务端，
 * 不能在前端推断。结论来自上传校验时的 Earth dry-run。
 */
export async function fetchUploadedModelEarthCompatibility(modelId, { signal, datasetId = 'earth_merra2' } = {}) {
  const res = await authedFetch(
    `${BASE}/user-models/${encodeURIComponent(modelId)}/earth-compatibility?dataset_id=${encodeURIComponent(datasetId)}`,
    { signal },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function revalidateUserModel(modelId) {
  const res = await authedFetch(`${BASE}/user-models/${modelId}/validate`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function deleteUserModel(modelId) {
  const res = await authedFetch(`${BASE}/user-models/${modelId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function fetchTasks(options = {}) {
  const res = await authedFetch(`${BASE}/training/tasks`, options);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

export async function fetchLogs(taskId) {
  const res = await authedFetch(`${BASE}/training/tasks/${taskId}/logs`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

async function trainingTagRequest(path, method = 'GET', body) {
  const res = await authedFetch(`${BASE}/training${path}`, {
    method,
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!res.ok) await throwResponseError(res);
  return res.status === 204 ? null : res.json();
}

export const fetchTrainingTags = () => trainingTagRequest('/tags');
export const createTrainingTag = (name) => trainingTagRequest('/tags', 'POST', { name });
export const renameTrainingTag = (id, name) => trainingTagRequest(`/tags/${id}`, 'PATCH', { name });
export const deleteTrainingTag = (id) => trainingTagRequest(`/tags/${id}`, 'DELETE');
export const replaceTrainingTaskTags = (taskId, tagIds) => (
  trainingTagRequest(`/tasks/${taskId}/tags`, 'PUT', { tag_ids: tagIds })
);
export const updateTrainingTaskTags = (taskIds, tagIds, operation) => (
  trainingTagRequest('/task-tags', 'PATCH', { task_ids: taskIds, tag_ids: tagIds, operation })
);

export async function renameTrainingModel(taskId, modelName) {
  const res = await authedFetch(`${BASE}/training/tasks/${taskId}/name`, {
    method: 'PATCH',
    body: JSON.stringify({ model_name: modelName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function stopTrainingTask(taskId) {
  const res = await authedFetch(`${BASE}/training/tasks/${taskId}/stop`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function cancelTrainingTask(taskId) {
  const res = await authedFetch(`${BASE}/training/tasks/${taskId}/cancel`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function deleteTrainingTask(taskId) {
  const res = await authedFetch(`${BASE}/training/tasks/${taskId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status}`);
  }
  return res.json();
}

export async function performTaskAction(taskId, action, { signal } = {}) {
  const res = await authedFetch(`${BASE}/training/tasks/${taskId}/action?action=${encodeURIComponent(action)}`, {
    method: 'POST',
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = err.detail;
    const message = typeof detail === 'string' ? detail
      : detail?.message || (detail != null ? JSON.stringify(detail) : `${res.status}`);
    const error = new Error(message);
    error.code = detail?.code || 'task_action_failed';
    error.status = res.status;
    throw error;
  }
  return res.json();
}

