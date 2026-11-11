import { useAnimatedNumber } from './motion';

/**
 * 数值滚动展示：值变化时在 duration 内缓动，再交给 format 输出。
 * 非有限数值（缺失、占位）直接回退到原值，避免格式化出 NaN。
 */
export default function AnimatedNumber({ value, format = (input) => String(input), duration = 420, enabled = true }) {
  const animated = useAnimatedNumber(value, { duration, enabled });
  const shown = typeof animated === 'number' && Number.isFinite(animated) ? animated : value;
  return <>{format(shown)}</>;
}
