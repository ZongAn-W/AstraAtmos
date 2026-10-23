/**
 * AstraAtmos 数值格式化工具
 * 根据用户设置的精度档位格式化数值
 */

/**
 * 精度感知格式化
 * @param {number}        v         - 原始数值
 * @param {number|string} precision - 2 | 4 | 6 | 'full'
 * @returns {string}
 */
export function fmtNum(v, precision = 2) {
  if (v == null || isNaN(v)) return '—';
  if (precision === 'full') return String(v);
  const n = Number(precision);
  if (v === 0) return (0).toFixed(n);
  if (Math.abs(v) > 0 && Math.abs(v) < Math.pow(10, -(n + 1))) {
    return v.toExponential(Math.min(n, 3));
  }
  return v.toFixed(n);
}
