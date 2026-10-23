/**
 * AstraAtmos 单位换算工具
 * 所有换算仅在显示层进行，不修改后端原始数据
 * 原始数据单位：臭氧 μm-atm，温度 K，风速 m/s，气压 Pa
 */

// ── 臭氧柱浓度 ──────────────────────────────────────────────
// 1 μm-atm = 0.1 DU（火星臭氧柱量级：1-50 μm-atm）
export function convertOzone(val, unit) {
  return unit === 'DU' ? val * 0.1 : val;
}
export function ozoneLabel(unit) {
  return unit === 'DU' ? 'DU' : 'μm-atm';
}
export function ozoneDeltaLabel(unit) {
  return unit === 'DU' ? 'Δ DU' : 'Δ μm-atm';
}

// ── 温度 ─────────────────────────────────────────────────────
export function convertTemp(val, unit) {
  return unit === 'C' ? val - 273.15 : val;
}
export function tempLabel(unit) {
  return unit === 'C' ? '°C' : 'K';
}

// ── 风速 ─────────────────────────────────────────────────────
export function convertWind(val, unit) {
  return unit === 'km/h' ? val * 3.6 : val;
}
export function windLabel(unit) {
  return unit === 'km/h' ? 'km/h' : 'm/s';
}

// ── 气压 ─────────────────────────────────────────────────────
export function convertPressure(val, unit) {
  return unit === 'hPa' ? val / 100 : val;
}
export function pressureLabel(unit) {
  return unit === 'hPa' ? 'hPa' : 'Pa';
}
