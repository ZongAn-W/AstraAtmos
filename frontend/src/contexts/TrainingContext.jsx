import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { fetchTasks, fetchLogs } from '../services/api';
import { useAuth } from './AuthContext';
import { reconcileActiveTrainingTaskId } from './trainingTaskSelection';
import { requestNotificationRefresh } from '../notifications/notificationEvents';
import { createScopedRequestGate } from '../components/TrainingTags/trainingTagFilters';

const TrainingContext = createContext();

export const useTraining = () => useContext(TrainingContext);

export const TrainingProvider = ({ children, enabled = true }) => {
  const { user } = useAuth();
  return <UserTrainingProvider key={user?.id ?? 'guest'} user={user} enabled={enabled}>{children}</UserTrainingProvider>;
};

const UserTrainingProvider = ({ children, enabled, user }) => {
  const [tasks, setTasks] = useState([]);
  // 任务列表的首次加载状态：只用于区分「还在加载」与「确实没有实验」，
  // 不参与任何查询、筛选或选中逻辑；5 秒轮询期间不会重新置为 true。
  const [tasksLoading, setTasksLoading] = useState(false);
  const [tasksError, setTasksError] = useState(false);
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [progressData, setProgressData] = useState(null);
  const [logs, setLogs] = useState([]);
  // 用户主动进入「新建实验」后，不再让任务轮询自动选中运行中的任务
  // （否则 5 秒轮询或目录重渲染都会把用户从配置画布拽回监控阶段）。
  const suppressAutoSelectRef = useRef(false);
  const setSuppressAutoSelect = useCallback((value) => {
    suppressAutoSelectRef.current = Boolean(value);
  }, []);

  const wsRef = useRef(null);
  const pollingRef = useRef(null);
  const logPollingRef = useRef(null);
  const loadTasksRef = useRef(null);
  const pendingTasksRef = useRef(null);
  const tasksGate = useRef(createScopedRequestGate());
  tasksGate.current.setScope(enabled ? user?.id ?? null : null);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; tasksGate.current.invalidate(); };
  }, []);

  const clearTaskPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const clearLogPolling = useCallback(() => {
    if (logPollingRef.current) {
      clearInterval(logPollingRef.current);
      logPollingRef.current = null;
    }
  }, []);

  const closeWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const loadTasks = useCallback(async ({ reusePending = false } = {}) => {
    if (!enabled || !user) return;
    const pending = pendingTasksRef.current;
    if (reusePending && pending && tasksGate.current.isCurrent(pending.request)) {
      return pending.promise;
    }
    const request = tasksGate.current.start();
    const operation = { request, promise: null };
    pendingTasksRef.current = operation;
    setTasksLoading(true);
    operation.promise = (async () => {
      try {
        const data = await fetchTasks();
        if (!mountedRef.current || !tasksGate.current.isCurrent(request)) return;
        setTasks(data);
        setTasksError(false);
        setActiveTaskId((currentTaskId) => (
          reconcileActiveTrainingTaskId(data, currentTaskId, {
            suppressAutoSelect: suppressAutoSelectRef.current,
          })
        ));
      } catch (err) {
        console.error('Failed to load tasks', err);
        if (mountedRef.current && tasksGate.current.isCurrent(request)) setTasksError(true);
        throw err;
      } finally {
        if (pendingTasksRef.current === operation) pendingTasksRef.current = null;
        if (mountedRef.current && tasksGate.current.isCurrent(request)) setTasksLoading(false);
      }
    })();
    return operation.promise;
  }, [enabled, user]);

  useEffect(() => {
    loadTasksRef.current = loadTasks;
  }, [loadTasks]);

  // Poll training task list only when provider is enabled.
  useEffect(() => {
    clearTaskPolling();

    if (enabled && user) {
      loadTasks().catch(() => {});
      // A slow response must finish before another poll supersedes its token.
      // Explicit refreshes still start a fresh request after a mutation.
      pollingRef.current = setInterval(() => loadTasks({ reusePending: true }).catch(() => {}), 5000);
    } else {
      // 未登录 / 未启用时没有任务请求，加载标记必须一起复位，
      // 否则目录会一直停在「正在加载实验…」而看不到空状态。
      setTasksLoading(false);
      setTasksError(false);
      if (!user) {
        setTasks([]);
        setActiveTaskId(null);
        setProgressData(null);
        setLogs([]);
      }
    }

    return () => clearTaskPolling();
  }, [enabled, user, loadTasks, clearTaskPolling]);

  // Poll logs only when enabled and task is selected.
  useEffect(() => {
    clearLogPolling();
    let active = true;

    if (!enabled || !activeTaskId || !user) {
      setLogs([]);
      return;
    }

    const pollLogs = async () => {
      try {
        const data = await fetchLogs(activeTaskId);
        if (active) setLogs(data.lines || []);
      } catch (err) {
        console.error('Error polling logs', err);
      }
    };

    pollLogs();
    logPollingRef.current = setInterval(pollLogs, 3000);

    return () => { active = false; clearLogPolling(); };
  }, [enabled, activeTaskId, user, clearLogPolling]);

  // Sync progressData from active task row.
  useEffect(() => {
    const task = tasks.find((t) => t.id === activeTaskId);
    if (!task) return;

    let historyBuffer = { train: [], val: [] };
    if (task.loss_history) {
      try {
        historyBuffer = JSON.parse(task.loss_history);
      } catch (e) {
        console.error('History parse error', e);
      }
    }

    setProgressData({
      progress: task.progress || 0,
      current_epoch: task.current_epoch || 0,
      total_epochs: task.total_epochs || 0,
      current_loss: task.current_loss,
      eta: task.eta || '--:--',
      loss_history: historyBuffer,
    });
  }, [activeTaskId, tasks]);

  const currentUserId = user?.id || null;

  // WebSocket live updates only when enabled and task is running.
  useEffect(() => {
    if (!enabled || !activeTaskId || !currentUserId) {
      closeWs();
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/api/ws/training/${activeTaskId}`;

    if (wsRef.current?.url === wsUrl && wsRef.current.readyState <= WebSocket.OPEN) {
      return;
    }

    closeWs();
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    let heartbeatTimer = null;

    ws.onopen = () => {
      heartbeatTimer = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 4000);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'training_update') {
          if (msg.data && msg.data.loss_history) {
            setProgressData(msg.data);
          } else {
            setProgressData((prev) => ({ ...prev, ...msg.data }));
          }
        } else if (msg.type === 'status_update') {
          loadTasksRef.current?.().catch(() => {});
          if (msg.status === 'failed') {
            requestNotificationRefresh();
          }
        }
      } catch (e) {
        console.error('WS parse error', e);
      }
    };

    ws.onclose = () => {
      if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
      if (wsRef.current === ws) wsRef.current = null;
    };

    return () => {
      if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
      if (wsRef.current === ws) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, activeTaskId, currentUserId, closeWs]);

  const value = {
    tasks,
    setTasks,
    tasksLoading,
    tasksError,
    setTasksError,
    activeTaskId,
    setActiveTaskId,
    setSuppressAutoSelect,
    progressData,
    setProgressData,
    logs,
    setLogs,
    loadTasks,
  };

  return <TrainingContext.Provider value={value}>{children}</TrainingContext.Provider>;
};
