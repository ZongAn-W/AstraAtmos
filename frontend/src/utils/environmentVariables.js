import { convertOzone, ozoneLabel } from './units.js';

export const ENVIRONMENT_VARIABLES = Object.freeze({
  mars: Object.freeze({
    O3: { zh: '臭氧', en: 'Ozone', unit: 'um-atm' },
    U: { zh: '东西向风', en: 'Eastward wind', unit: 'm/s' },
    V: { zh: '南北向风', en: 'Northward wind', unit: 'm/s' },
    D: { zh: '沙尘光学厚度', en: 'Dust optical depth', unit: '1' },
    S: { zh: '下行太阳辐射', en: 'Incoming solar flux', unit: 'W/m²' },
    T: { zh: '温度', en: 'Temperature', unit: 'K' },
  }),
  earth: Object.freeze({
    TO3: { zh: '总柱臭氧', en: 'Total column ozone', unit: 'DU' },
    U10M: { zh: '10米东西向风', en: '10 m eastward wind', unit: 'm s-1' },
    V10M: { zh: '10米南北向风', en: '10 m northward wind', unit: 'm s-1' },
    T2M: { zh: '2米气温', en: '2 m air temperature', unit: 'K' },
    SWGDN: { zh: '地表下行短波辐射', en: 'Surface incoming shortwave flux', unit: 'W m-2' },
  }),
});

function parameters(source) {
  const value = source?.hyperparameters || source || {};
  if (typeof value !== 'string') return value;
  try { return JSON.parse(value); } catch { return {}; }
}

function legacyChannels(value) {
  if (Array.isArray(value)) return value;
  if (typeof value !== 'string') return [];
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) return parsed;
  } catch { /* Historical selections also used comma or plus separators. */ }
  return value.replaceAll('+', ',').split(',').map(channel => channel.trim()).filter(Boolean);
}

export function environmentPlanet(source, fallback = 'mars') {
  const h = parameters(source);
  return String(source?.dataset_id || h.training_dataset || '').startsWith('earth_') ? 'earth' : fallback;
}

/** Missing fields retain the historical ozone target and ozone-first inputs. */
export function resolveEnvironmentTask(source, planet = environmentPlanet(source)) {
  const h = parameters(source);
  const variables = ENVIRONMENT_VARIABLES[planet] || ENVIRONMENT_VARIABLES.mars;
  const baseline = planet === 'earth' ? 'TO3' : 'O3';
  const target = h.target_variable || h.target || source?.target_variable || source?.target || baseline;
  const explicit = [h.input_variables, h.input_channel_order, source?.input_variables, source?.input_channel_order]
    .find(value => Array.isArray(value) && value.length);
  const inputs = explicit ? [...explicit]
    : [...new Set([baseline, ...legacyChannels(h.selected_channels).filter(c => c !== baseline)])];
  const meta = variables[target] || { zh: target, en: target, unit: h.target_unit || source?.target_unit || '' };
  return { planet, target, inputs, ...meta, unit: h.target_unit || source?.target_unit || meta.unit };
}

export function environmentIdentityMetadata(source) {
  const h = parameters(source);
  const target = h.target_variable || h.target || source?.target_variable || source?.target;
  const unit = h.target_unit || source?.target_unit;
  const inputs = [h.input_variables, h.input_channel_order, source?.input_variables, source?.input_channel_order]
    .find(value => Array.isArray(value) && value.length > 0);
  const hasTarget = typeof target === 'string' && target.length > 0;
  const hasUnit = typeof unit === 'string' && unit.length > 0;
  const hasInputs = Array.isArray(inputs);
  return { declared: hasTarget || hasUnit || hasInputs, complete: hasTarget && hasUnit && hasInputs };
}

export function environmentVariableLabel(variable, planet = 'mars', isZh = true) {
  const meta = ENVIRONMENT_VARIABLES[planet]?.[variable];
  return meta ? `${meta[isZh ? 'zh' : 'en']} (${variable})` : variable;
}

export function environmentDisplay(source, planet = 'mars', ozoneUnit = 'um-atm') {
  const task = resolveEnvironmentTask(source, planet);
  const ozone = planet === 'mars' && task.target === 'O3';
  const unit = ozone ? ozoneLabel(ozoneUnit) : task.unit;
  return { ...task, unit, deltaUnit: `Δ${unit}`, convertValue: ozone ? value => convertOzone(value, ozoneUnit) : value => value };
}

export function sameEnvironmentTarget(tasks, planet = 'mars') {
  return new Set(tasks.map(task => resolveEnvironmentTask(task, planet).target)).size <= 1;
}
