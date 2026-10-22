import { useCallback, useEffect, useRef, useState } from 'react';
import './microInteractions.css';
import { MORPH_SHRINK_MS, MORPH_SUCCESS_MS } from './microInteractionModel';

const PHASES = { IDLE: 'idle', LOADING: 'loading', SUCCESS: 'success' };

/**
 * 提交三连变：点击 → 收缩成圆形加载 → 绿色打勾 → 恢复原状。
 * 收缩时长由 --morph-shrink-ms 控制（默认 300ms）。
 *
 * 宽度处理：静止态不写死宽度（保持内容自然宽度），点击瞬间读取当前宽度并
 * 锁一帧作为过渡起点，随后交给 CSS 的圆形宽度，从而实现可动画的收缩；
 * 恢复时同样先锁回起始宽度，动画结束后再解除锁定。
 */
export default function MorphSubmitButton({
  label,
  loadingLabel = '',
  successLabel = '',
  onAction,
  disabled = false,
  className = '',
  ...rest
}) {
  const [phase, setPhase] = useState(PHASES.IDLE);
  const [lockedWidth, setLockedWidth] = useState(null);
  const rootRef = useRef(null);
  const timers = useRef([]);
  const alive = useRef(true);

  const later = useCallback((fn, ms) => {
    timers.current.push(setTimeout(fn, ms));
  }, []);

  useEffect(() => () => {
    alive.current = false;
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }, []);

  const handleClick = useCallback(async () => {
    if (disabled || phase !== PHASES.IDLE) return;
    const node = rootRef.current;
    const width = node ? node.getBoundingClientRect().width : 0;

    // 先锁定当前宽度，下一帧再解除，让 CSS 圆形宽度参与过渡
    setLockedWidth(width > 0 ? width : null);
    setPhase(PHASES.LOADING);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (alive.current) setLockedWidth(null);
    }));

    const shrinkDone = new Promise(resolve => later(resolve, MORPH_SHRINK_MS));
    try {
      await Promise.all([Promise.resolve(onAction?.()), shrinkDone]);
      if (!alive.current) return;
      setPhase(PHASES.SUCCESS);
      later(() => {
        if (!alive.current) return;
        setLockedWidth(width > 0 ? width : null);
        setPhase(PHASES.IDLE);
        later(() => {
          if (alive.current) setLockedWidth(null);
        }, MORPH_SHRINK_MS + 40);
      }, MORPH_SUCCESS_MS);
    } catch {
      if (alive.current) {
        setLockedWidth(null);
        setPhase(PHASES.IDLE);
      }
    }
  }, [disabled, later, onAction, phase]);

  return (
    <button
      ref={rootRef}
      type="button"
      className={`morph-submit morph-submit--${phase} ${className}`.trim()}
      style={{
        '--morph-shrink-ms': `${MORPH_SHRINK_MS}ms`,
        ...(lockedWidth ? { width: `${lockedWidth}px` } : {}),
      }}
      onClick={handleClick}
      disabled={disabled || phase !== PHASES.IDLE}
      aria-busy={phase === PHASES.LOADING}
      data-phase={phase}
      data-morph-submit="true"
      {...rest}
    >
      <span className="morph-submit__label" aria-hidden={phase !== PHASES.IDLE}>{label}</span>
      <span className="morph-submit__loader" aria-hidden="true">
        <svg className="morph-submit__spinner" viewBox="0 0 24 24">
          <circle className="morph-submit__spinner-track" cx="12" cy="12" r="9" />
          <circle className="morph-submit__spinner-arc" cx="12" cy="12" r="9" />
        </svg>
      </span>
      <svg className="morph-submit__check" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5.5 12.6l4.1 4.1L18.6 7.6" />
      </svg>
      <span className="morph-submit__sr" role="status">
        {phase === PHASES.LOADING ? loadingLabel : phase === PHASES.SUCCESS ? successLabel : ''}
      </span>
    </button>
  );
}
