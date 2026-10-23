/**
 * 微交互纯逻辑：时长、曲线与几何计算。
 * 与 DOM 无关，便于 node --test 直接验证；组件只负责把结果映射到样式。
 */

/** 提交按钮收缩为圆形加载态的时长（规格 300ms）。 */
export const MORPH_SHRINK_MS = 300;
/** 成功态保持绿色对勾的时长，之后恢复原状。 */
export const MORPH_SUCCESS_MS = 900;
/** 果冻开关滑动、回弹与底色过渡时长（规格 350ms）。 */
export const JELLY_MS = 350;
/** 弹簧回弹曲线：前段平缓、末段轻微越过目标再回落，位移可见且收尾干净。 */
export const JELLY_SPRING = 'cubic-bezier(0.45, 0.05, 0.38, 1.28)';
/** 长按确认所需时长（规格 1.5s）。 */
export const HOLD_CONFIRM_MS = 1500;
/** 中途松手后进度归零的动画时长。 */
export const HOLD_RESET_MS = 180;

export function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

/**
 * 长按进度：elapsed 毫秒占 holdMs 的比例，钳制在 [0, 1]。
 */
export function holdProgress(elapsedMs, holdMs = HOLD_CONFIRM_MS) {
  if (!Number.isFinite(holdMs) || holdMs <= 0) return 1;
  return clamp01(elapsedMs / holdMs);
}

/**
 * 拖动行程比例：指针相对起点的位移占可用行程的比例。
 * 行程为 0（轨道与滑块同宽）时视为已到终点。
 */
export function jellyTravel(dragX, startX, travelDistance) {
  if (!Number.isFinite(travelDistance) || travelDistance <= 0) return 1;
  return clamp01(Math.abs(dragX - startX) / travelDistance);
}

/**
 * 果冻拉伸量：起点与终点都为 0，行程中段最大，形成“拉长—回弹”。
 * 用 sin(pi * travel) 保证两端收敛到 0，不必额外判断方向。
 */
/**
 * 果冻拉伸量：由指针速度驱动——快速滑动时拉长，停下即收缩回弹。
 * reference 为达到最大拉伸所需的参考速度（px/ms）。
 */
export function jellyStretchFromVelocity(velocity, maxStretch, reference = 1.6) {
  const peak = Number.isFinite(maxStretch) ? Math.max(0, maxStretch) : 0;
  const ref = Number.isFinite(reference) && reference > 0 ? reference : 1.6;
  return peak * clamp01(Math.abs(velocity) / ref);
}

/** 拉长时的压扁系数：横向拉得越长，纵向压得越扁。 */
export function jellySquash(stretch, maxStretch, depth = 0.3) {
  const peak = Number.isFinite(maxStretch) && maxStretch > 0 ? maxStretch : 1;
  return 1 - depth * clamp01(stretch / peak);
}

/**
 * 指针 x 在轨道上的归一化位置，扣除内边距与滑块宽度后映射到 [0, 1]。
 */
export function pointerRatio({ clientX, trackLeft, trackWidth, padding, thumbWidth }) {
  const usable = trackWidth - padding * 2 - thumbWidth;
  if (!Number.isFinite(usable) || usable <= 0) return 0;
  return clamp01((clientX - trackLeft - padding) / usable);
}

/**
 * 滑块几何：拉伸后的宽度与左偏移，保持始终位于轨道内。
 */
export function thumbGeometry({ trackWidth, padding, thumbSize, ratio, stretch = 0 }) {
  const width = thumbSize + Math.max(0, stretch);
  const usable = Math.max(0, trackWidth - padding * 2 - width);
  return { width, left: padding + usable * clamp01(ratio) };
}

/**
 * 松手时的目标状态：位移不足视为点击（取反），否则按落点是否过半判定。
 */
export function nextToggleValue({
  startChecked,
  endRatio,
  travel,
  tapThreshold = 0.12,
  commitThreshold = 0.5,
}) {
  if (travel < tapThreshold) return !startChecked;
  return clamp01(endRatio) >= commitThreshold;
}

/** 圆形进度环：周长与 dash 偏移，progress=1 时恰好填满。 */
export function ringDash({ radius, progress }) {
  const circumference = 2 * Math.PI * radius;
  return { circumference, dashOffset: circumference * (1 - clamp01(progress)) };
}
