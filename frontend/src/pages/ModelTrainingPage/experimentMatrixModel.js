import {
  EXPERIMENT_STATUS_FILTERS,
  getExperimentArchitectureLabel,
  parseTaskHyperparameters,
  readExperimentMetrics,
} from './experimentCenterModel.js';
import { getModelStructureConfig } from './trainingParamSanitizers.js';
import { earthMetricUnit } from '../../utils/earthMetricMeta.js';

export const MATRIX_COLUMN_STORAGE_PREFIX = 'astraatmos_experiment_matrix_columns';

const RESERVED_HYPERPARAMETERS = new Set([
  'model_architecture', 'model_source', 'training_dataset', 'selected_channels',
  'window', 'horizon', 'epochs', 'batch_size', 'learning_rate', 'train_ratio',
  'validation_ratio', 'test_ratio', 'seed', 'early_stopping_patience',
  'use_sphere', 'transfer_enabled', 'transfer_source_type', 'transfer_source_task_id',
  'transfer_freeze_mode', 'finetune_learning_rate', 'stlstm_layers', 'stlstm_hidden_dims',
  'custom_model_params',
]);
const INTERNAL_KEY = /(^_|path|filepath|file_path|hash|fingerprint|snapshot|identity|source_code|script)/i;

const BASIC_COLUMNS = [
  { key: 'model_name', group: 'basic', label: '模型名称', labelEn: 'Model name', fixed: true, type: 'text' },
  { key: 'tags', group: 'basic', label: '标签', labelEn: 'Tags', fixed: true, type: 'tags' },
  { key: 'architecture', group: 'basic', label: '架构', labelEn: 'Architecture', type: 'text' },
  { key: 'dataset', group: 'basic', label: '数据集', labelEn: 'Dataset', type: 'text' },
  { key: 'status', group: 'basic', label: '状态', labelEn: 'Status', type: 'text' },
  { key: 'progress', group: 'basic', label: '进度', labelEn: 'Progress', type: 'number', unit: '%' },
  { key: 'start_time', group: 'basic', label: '开始时间', labelEn: 'Start time', type: 'date' },
  { key: 'model_package', group: 'basic', label: '上传模型包', labelEn: 'Uploaded package', type: 'text' },
  { key: 'model_source', group: 'basic', label: '模型来源', labelEn: 'Model source', type: 'text' },
];

const TRAINING_FIELDS = [
  ['window', '输入窗口', 'Input window'], ['horizon', '预测步长', 'Forecast horizon'],
  ['epochs', '训练轮次', 'Epochs'], ['batch_size', '批大小', 'Batch size'],
  ['learning_rate', '学习率', 'Learning rate'], ['train_ratio', '训练集比例', 'Train ratio'],
  ['validation_ratio', '验证集比例', 'Validation ratio'], ['test_ratio', '测试集比例', 'Test ratio'],
  ['seed', '随机种子', 'Random seed'], ['early_stopping_patience', '早停轮数', 'Early stopping'],
  ['selected_channels', '辅助通道', 'Driver channels'], ['use_sphere', '球面特征', 'Spherical features'],
  ['transfer_enabled', '迁移学习', 'Transfer learning'],
  ['transfer_source_type', '迁移来源类型', 'Transfer source type'],
  ['transfer_freeze_mode', '冻结策略', 'Freeze strategy'],
  ['finetune_learning_rate', '微调学习率', 'Fine-tune learning rate'],
];

const ARCHITECTURE_FIELDS = [
  ['stlstm_layers', 'ST-LSTM 层数', 'ST-LSTM layers'],
  ['stlstm_hidden_dims', '隐藏层宽度', 'Hidden dimensions'],
];

const METRIC_FIELDS = [
  ['rmse', 'RMSE', 'RMSE'], ['mae', 'MAE', 'MAE'], ['mse', 'MSE', 'MSE'],
  ['r2', 'R²', 'R²'], ['mape', 'MAPE', 'MAPE'], ['smape', 'SMAPE', 'SMAPE'],
];

export const MATRIX_GROUPS = Object.freeze([
  { id: 'basic', label: '基本信息', labelEn: 'Basic information' },
  { id: 'training', label: '训练参数', labelEn: 'Training parameters' },
  { id: 'architecture', label: '架构参数', labelEn: 'Architecture parameters' },
  { id: 'custom', label: '自定义参数', labelEn: 'Custom parameters' },
  { id: 'metrics', label: '评价指标', labelEn: 'Evaluation metrics' },
]);

export const MATRIX_DEFAULT_COLUMNS = Object.freeze([
  'model_name', 'tags', 'architecture', 'dataset', 'status', 'progress', 'start_time',
]);
export const MATRIX_STATUS_FILTERS = Object.freeze(EXPERIMENT_STATUS_FILTERS);

function labelFor(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function dynamicColumn(key, group, label, labelEn, type = 'text') {
  return { key, group, label, labelEn, type };
}

export function getMatrixPropertyDefinitions(tasks = []) {
  const definitions = [...BASIC_COLUMNS];
  TRAINING_FIELDS.forEach(([key, label, labelEn]) => definitions.push(dynamicColumn(`training:${key}`, 'training', label, labelEn, key.includes('ratio') || key === 'learning_rate' ? 'number' : 'text')));
  ARCHITECTURE_FIELDS.forEach(([key, label, labelEn]) => definitions.push(dynamicColumn(`architecture:${key}`, 'architecture', label, labelEn, 'text')));
  if (tasks.some(task => parseTaskHyperparameters(task?.hyperparameters)._earth_task_split)) {
    definitions.push(dynamicColumn('split:policy', 'training', '任务划分策略', 'Task split policy'));
    for (const [name, label, labelEn] of [['train', '训练', 'Train'], ['validation', '验证', 'Validation'], ['test', '测试', 'Test']]) {
      for (const [key, suffix, suffixEn, type] of [['date_start', '开始 UTC', 'start UTC', 'text'], ['date_end', '结束 UTC', 'end UTC', 'text'], ['step_count', '时间步数', 'time steps', 'number'], ['window_count', '窗口数', 'windows', 'number']]) {
        definitions.push(dynamicColumn('split:' + name + ':' + key, 'training', label + suffix, labelEn + ' ' + suffixEn, type));
      }
    }
  }
  const customKeys = new Set();
  const unknownKeys = new Set();
  const architectureKeys = new Set(ARCHITECTURE_FIELDS.map(([key]) => key));
  (Array.isArray(tasks) ? tasks : []).forEach((task) => {
    const hyperparameters = parseTaskHyperparameters(task?.hyperparameters);
    getModelStructureConfig(hyperparameters.model_architecture).forEach((field) => architectureKeys.add(field.key));
    Object.keys(hyperparameters.custom_model_params || {}).forEach((key) => customKeys.add(key));
    Object.keys(hyperparameters).forEach((key) => {
      if (!RESERVED_HYPERPARAMETERS.has(key) && !INTERNAL_KEY.test(key)) {
        if (architectureKeys.has(key)) return;
        unknownKeys.add(key);
      }
    });
  });
  [...architectureKeys].filter((key) => !ARCHITECTURE_FIELDS.some(([fieldKey]) => fieldKey === key)).sort((a, b) => a.localeCompare(b)).forEach((key) => definitions.push(dynamicColumn(`architecture:${key}`, 'architecture', `架构 · ${key}`, `Architecture · ${key}`, 'text')));
  [...customKeys].sort((a, b) => a.localeCompare(b)).forEach((key) => definitions.push(dynamicColumn(`custom:${key}`, 'custom', `自定义 · ${key}`, `Custom · ${key}`, 'text')));
  // Top-level extension fields and uploaded-model parameters can share a name
  // while describing different things. Keep their namespaces distinct so a
  // historical task never loses one of the values during column discovery.
  [...unknownKeys].sort((a, b) => a.localeCompare(b)).forEach((key) => definitions.push(dynamicColumn(`config:${key}`, 'custom', `配置 · ${key}`, `Config · ${key}`, 'text')));
  const hasEarth = tasks.some(task => String(task?.dataset_id || parseTaskHyperparameters(task?.hyperparameters).training_dataset || '').startsWith('earth_'));
  METRIC_FIELDS.forEach(([key, label, labelEn]) => definitions.push(dynamicColumn(`metric:${key}`, 'metrics',
    hasEarth ? `${label}（Earth 完整测试集）` : `${label}（任务单位）`,
    hasEarth ? `${labelEn} (Earth full test set)` : `${labelEn} (task unit)`, 'number')));
  return definitions.filter((column, index, all) => all.findIndex((item) => item.key === column.key) === index);
}

export function getMatrixStorageKey(scope) {
  return `${MATRIX_COLUMN_STORAGE_PREFIX}:${scope || 'guest'}`;
}

export function restoreMatrixColumnPreferences(raw, definitions = getMatrixPropertyDefinitions(), defaults = MATRIX_DEFAULT_COLUMNS) {
  const available = new Set(definitions.map((definition) => definition.key));
  const fallback = defaults.filter((key) => available.has(key));
  const fixed = definitions.filter((definition) => definition.fixed).map((definition) => definition.key);
  try {
    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
    const stored = Array.isArray(parsed) ? parsed : parsed?.columns;
    if (!Array.isArray(stored)) return [...fallback];
    const result = [...new Set(stored.filter((key) => available.has(key) && !fixed.includes(key)))];
    fixed.reverse().forEach((key) => result.unshift(key));
    fallback.forEach((key) => { if (!result.includes(key)) result.push(key); });
    return result.length ? result : [...fallback];
  } catch {
    return [...fallback];
  }
}

export function serializeMatrixColumnPreferences(columns) {
  return JSON.stringify({ version: 1, columns: [...new Set(Array.isArray(columns) ? columns : [])] });
}

export function getMatrixValue(task, columnKey) {
  const hyperparameters = parseTaskHyperparameters(task?.hyperparameters);
  const metrics = readExperimentMetrics(task?.metrics);
  if (columnKey === 'model_name') return task?.custom_model_name ?? '';
  if (columnKey === 'tags') return Array.isArray(task?.tags) ? task.tags : [];
  if (columnKey === 'architecture') return getExperimentArchitectureLabel(hyperparameters.model_architecture) || '';
  if (columnKey === 'dataset') return task?.dataset_id || hyperparameters.training_dataset || '';
  if (columnKey === 'status') return String(task?.status || '').toLowerCase();
  if (columnKey === 'progress') return task?.progress === 0 ? 0 : (task?.progress ?? null);
  if (columnKey === 'start_time') return task?.start_time || null;
  if (columnKey === 'model_package') return task?.uploaded_model_name || hyperparameters._uploaded_model_name || '';
  if (columnKey === 'model_source') return String(task?.model_source || hyperparameters.model_source || '');
  if (columnKey.startsWith('split:')) {
    const [, name, field] = columnKey.split(':');
    const split = hyperparameters._earth_task_split;
    return name === 'policy' ? split?.policy : split?.ranges?.[name]?.[field];
  }
  if (columnKey.startsWith('training:')) return hyperparameters[columnKey.slice(9)];
  if (columnKey.startsWith('architecture:')) return hyperparameters[columnKey.slice(13)];
  if (columnKey.startsWith('custom:')) return hyperparameters.custom_model_params?.[columnKey.slice(7)];
  if (columnKey.startsWith('config:')) return hyperparameters[columnKey.slice(7)];
  if (columnKey.startsWith('metric:')) return metrics[columnKey.slice(7)];
  return undefined;
}

export function formatMatrixValue(value, column, task, locale = 'zh-CN') {
  if (value === undefined || value === null || value === '') {
    return column?.key?.startsWith('metric:') && String(task?.dataset_id || parseTaskHyperparameters(task?.hyperparameters).training_dataset || '').startsWith('earth_')
      ? (locale.startsWith('zh') ? '未提供' : 'Not provided') : '—';
  }
  if (column?.key === 'tags') return value;
  if (column?.type === 'date') {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? '—' : parsed.toLocaleString(locale, { dateStyle: 'medium', timeStyle: 'short' });
  }
  if (column?.key === 'progress') return `${Number(value).toFixed(0)}%`;
  if (column?.key?.startsWith('metric:')) {
    const dataset = String(task?.dataset_id || parseTaskHyperparameters(task?.hyperparameters).training_dataset || '');
    const unit = dataset.startsWith('earth_') ? earthMetricUnit(column.key.slice(7), locale.startsWith('zh')) : 'μm-atm';
    return `${Number.isFinite(Number(value)) ? Number(value).toFixed(4) : String(value)} ${unit}`;
  }
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (typeof value === 'number') return String(value);
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function filterMatrixTasks(tasks = [], { search = '', status = 'all', tagIds = [] } = {}) {
  const query = String(search || '').trim().toLocaleLowerCase();
  const wanted = (Array.isArray(tagIds) ? tagIds : []).map(Number);
  return (Array.isArray(tasks) ? tasks : []).filter((task) => {
    const name = String(task?.custom_model_name || '').toLocaleLowerCase();
    if (query && !name.includes(query) && !String(task?.id ?? '').includes(query)) return false;
    if (status !== 'all') {
      const taskStatus = String(task?.status || '').toLowerCase();
      if (status === 'running' && !['running', 'pending'].includes(taskStatus)) return false;
      if (status !== 'running' && taskStatus !== status) return false;
    }
    const tags = (task?.tags || []).map((tag) => Number(tag?.id));
    return wanted.every((id) => tags.includes(id));
  });
}

export function sortMatrixTasks(tasks = [], column, direction = 'asc', definitions = []) {
  const definition = definitions.find((item) => item.key === column) || { type: 'text' };
  const sign = direction === 'desc' ? -1 : 1;
  const getComparable = (task) => {
    const value = getMatrixValue(task, column);
    if (value === undefined || value === null || value === '') return { missing: true, value: null };
    if (definition.type === 'date') {
      const time = Date.parse(value);
      return { missing: Number.isNaN(time), value: time };
    }
    if (Array.isArray(value)) {
      const names = value.map((item) => typeof item === 'object' ? item?.name : item).filter(Boolean);
      return { missing: names.length === 0, value: names.join(', ').toLocaleLowerCase() };
    }
    if (definition.type === 'number' || typeof value === 'number') {
      const numeric = Number(value);
      return { missing: !Number.isFinite(numeric), value: numeric };
    }
    return { missing: false, value: String(value).toLocaleLowerCase() };
  };
  return [...(Array.isArray(tasks) ? tasks : [])].sort((a, b) => {
    const left = getComparable(a); const right = getComparable(b);
    if (left.missing !== right.missing) return left.missing ? 1 : -1;
    if (left.missing) return Number(a?.id || 0) - Number(b?.id || 0);
    if (left.value < right.value) return -1 * sign;
    if (left.value > right.value) return 1 * sign;
    return Number(a?.id || 0) - Number(b?.id || 0);
  });
}
