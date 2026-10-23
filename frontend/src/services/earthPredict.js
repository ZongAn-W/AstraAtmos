/**
 * Earth 历史预测 API 客户端
 *
 * 与火星预测完全分开：本模块只访问 `/api/earth/predict/*`，因此 Earth 任务不会
 * 经过火星推理、火星数据准备或火星预测缓存。错误保留结构化 code/status，便于
 * 预测页区分「日期越界」「数据集指纹变化」「权重损坏」「无权限」。
 */

import { handleUnauthorizedResponse } from '../stores/authPredictionSession.js';

const BASE = '/api';
const EARTH_PREDICT_PATH = `${BASE}/earth/predict`;

export class EarthPredictApiError extends Error {
  constructor(message, { code = 'invalid_request', status = 0, availabilityReason = null } = {}) {
    super(message);
    this.name = 'EarthPredictApiError';
    this.code = code;
    this.status = status;
    this.availabilityReason = availabilityReason;
  }
}

function describeDetail(detail) {
  if (typeof detail === 'string' && detail.trim()) return detail.trim();
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        const loc = Array.isArray(item?.loc) ? item.loc.filter((part) => part !== 'body').join('.') : '';
        const msg = item?.msg || '';
        return loc ? `${loc}: ${msg}` : msg;
      })
      .filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return '';
}

async function readError(response) {
  const payload = await response.json().catch(() => null);
  const detail = payload?.detail;
  const structured = detail && typeof detail === 'object' && !Array.isArray(detail) ? detail : null;
  const message = describeDetail(structured ? structured.message : detail)
    || `${response.status} ${response.statusText}`.trim();
  return new EarthPredictApiError(message, {
    code: structured?.code || 'invalid_request',
    status: response.status,
    availabilityReason: structured?.availability_reason ?? null,
  });
}

function requireAuthHeaders() {
  const token = typeof localStorage !== 'undefined' ? localStorage.getItem('astraatmos_token') : null;
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function request(path, { method = 'GET', body, signal } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: requireAuthHeaders(),
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    throw new EarthPredictApiError('Network request failed', { code: 'network_error' });
  }
  if (!response.ok) {
    handleUnauthorizedResponse(response);
    throw await readError(response);
  }
  try {
    return await response.json();
  } catch {
    throw new EarthPredictApiError('Invalid JSON response', {
      code: 'invalid_response',
      status: response.status,
    });
  }
}

export function fetchEarthPredictContext(trainingTaskId, { signal } = {}) {
  const params = new URLSearchParams({ training_task_id: String(trainingTaskId) });
  return request(`${EARTH_PREDICT_PATH}/context?${params.toString()}`, { signal });
}

export function runEarthPrediction({ trainingTaskId, forecastOrigin }, { signal } = {}) {
  return request(`${EARTH_PREDICT_PATH}/run`, {
    method: 'POST',
    body: { training_task_id: trainingTaskId, forecast_origin: forecastOrigin },
    signal,
  });
}

export function compareEarthModels(taskIds, { signal } = {}) {
  return request(`${EARTH_PREDICT_PATH}/training-models/compare`, {
    method: 'POST', body: { task_ids: taskIds }, signal,
  });
}

/** Every request reaches the server so cached diagnostics still recheck artifact/data identity. */
export function fetchEarthDiagnosticContext(trainingTaskId, { signal } = {}) {
  const params = new URLSearchParams({ training_task_id: String(trainingTaskId) });
  return request(`${EARTH_PREDICT_PATH}/diagnostics/context?${params.toString()}`, { signal });
}

export function runEarthDiagnostics(parameters, { signal } = {}) {
  return request(`${EARTH_PREDICT_PATH}/diagnostics`, {
    method: 'POST', body: parameters, signal,
  });
}
