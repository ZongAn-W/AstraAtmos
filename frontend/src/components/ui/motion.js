import { useEffect, useRef, useState } from 'react';

const REDUCED_QUERY = '(prefers-reduced-motion: reduce)';

export function prefersReducedMotion() {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
  return window.matchMedia(REDUCED_QUERY).matches;
}

/**
 * 数字滚动：数值变化时在 duration 内缓动到目标值。
 * 传 null/undefined 或关闭 enabled 时直接返回原值，便于占位符（如 '--'）原样显示。
 */
export function useAnimatedNumber(target, { duration = 420, enabled = true } = {}) {
  const isNumber = typeof target === 'number' && Number.isFinite(target);
  const [display, setDisplay] = useState(isNumber ? target : target);
  const fromRef = useRef(isNumber ? target : 0);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!isNumber) {
      cancelAnimationFrame(rafRef.current);
      setDisplay(target);
      return undefined;
    }
    if (!enabled || prefersReducedMotion()) {
      cancelAnimationFrame(rafRef.current);
      fromRef.current = target;
      setDisplay(target);
      return undefined;
    }
    const from = Number.isFinite(fromRef.current) ? fromRef.current : target;
    if (from === target) {
      setDisplay(target);
      return undefined;
    }
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - (1 - t) ** 3;
      const value = from + (target - from) * eased;
      setDisplay(value);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = target;
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // target 变化才重新起动画；duration/enabled 变化时同样需要重算
  }, [isNumber, target, duration, enabled]);

  return display;
}

/** 错落入场：按索引给出递增延迟，配合 .motion-stagger-item 使用 */
export function staggerStyle(index, { step = 30, max = 300 } = {}) {
  return { '--stagger-delay': `${Math.min(index * step, max)}ms` };
}

/**
 * 主题切换的圆形扩散：从指针位置展开；不支持 View Transition 或用户要求减少动效时直接切换。
 */
export function withThemeReveal(update, point) {
  if (typeof document === 'undefined') {
    update();
    return;
  }
  const root = document.documentElement;
  if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
    root.style.setProperty('--theme-reveal-x', `${point.x}px`);
    root.style.setProperty('--theme-reveal-y', `${point.y}px`);
  } else {
    root.style.removeProperty('--theme-reveal-x');
    root.style.removeProperty('--theme-reveal-y');
  }
  if (typeof document.startViewTransition !== 'function' || prefersReducedMotion()) {
    update();
    return;
  }
  document.startViewTransition(update);
}
