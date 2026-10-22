import { useCallback, useEffect, useRef, useState } from 'react';
import './microInteractions.css';
import { HOLD_CONFIRM_MS, HOLD_RESET_MS, holdProgress, ringDash } from './microInteractionModel';

const SIZES = { default: 88, compact: 44 };
const RING_INSET = 6;
const RESET_AFTER_MS = 1600;

/**
 * 长按确认删除：按住外圈 1.5 秒填满才触发，中途松手进度归零，
 * 完成后显示对勾。键盘 Space/Enter 按住同样有效。
 * size='compact' 用于列表行内的小尺寸圆形删除键。
 */
export default function HoldToDeleteButton({
  label,
  hint,
  doneLabel = '',
  holdMs = HOLD_CONFIRM_MS,
  onConfirm,
  persist = false,
  size = 'default',
  className = '',
}) {
  const sizePx = SIZES[size] ?? SIZES.default;
  const radius = sizePx / 2 - RING_INSET;
  const strokeWidth = size === 'compact' ? 3 : 4;
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState('idle'); // idle | holding | done
  const phaseRef = useRef('idle');
  const raf = useRef(0);
  const startedAt = useRef(0);
  const alive = useRef(true);
  const timers = useRef([]);

  const updatePhase = useCallback((next) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  useEffect(() => () => {
    alive.current = false;
    cancelAnimationFrame(raf.current);
    timers.current.forEach(clearTimeout);
  }, []);

  const stop = useCallback(() => {
    cancelAnimationFrame(raf.current);
    raf.current = 0;
    // 只有“按住中”松手才归零；完成后松手不能清掉对勾状态
    if (phaseRef.current !== 'holding') return;
    updatePhase('idle');
    setProgress(0);
  }, [updatePhase]);

  const finish = useCallback(() => {
    cancelAnimationFrame(raf.current);
    raf.current = 0;
    setProgress(1);
    updatePhase('done');
    onConfirm?.();
    if (!persist) {
      timers.current.push(setTimeout(() => {
        if (!alive.current) return;
        updatePhase('idle');
        setProgress(0);
      }, RESET_AFTER_MS));
    }
  }, [onConfirm, persist, updatePhase]);

  const tick = useCallback(() => {
    const value = holdProgress(performance.now() - startedAt.current, holdMs);
    setProgress(value);
    if (value >= 1) {
      finish();
      return;
    }
    raf.current = requestAnimationFrame(tick);
  }, [finish, holdMs]);

  const begin = useCallback(() => {
    if (phaseRef.current === 'done') return;
    cancelAnimationFrame(raf.current);
    startedAt.current = performance.now();
    updatePhase('holding');
    setProgress(0);
    raf.current = requestAnimationFrame(tick);
  }, [tick, updatePhase]);

  const handleKeyDown = useCallback((event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    if (event.repeat) return;
    begin();
  }, [begin]);

  const handleKeyUp = useCallback((event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    stop();
  }, [stop]);

  const { circumference, dashOffset } = ringDash({ radius, progress });
  const holding = phase === 'holding';
  const done = phase === 'done';
  const compact = size === 'compact';

  return (
    <button
      type="button"
      className={`hold-delete hold-delete--${phase}${compact ? ' hold-delete--compact' : ''} ${className}`.trim()}
      style={{
        '--hold-size': `${sizePx}px`,
        '--hold-stroke': strokeWidth,
        '--hold-confirm-ms': `${holdMs}ms`,
        '--hold-reset-ms': `${HOLD_RESET_MS}ms`,
      }}
      data-hold-delete="true"
      data-size={size}
      data-phase={phase}
      data-hold-progress={progress.toFixed(3)}
      aria-label={hint ? `${label}，${hint}` : label}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture?.(event.pointerId);
        begin();
      }}
      onPointerUp={stop}
      onPointerCancel={stop}
      onPointerLeave={stop}
      onKeyDown={handleKeyDown}
      onKeyUp={handleKeyUp}
      onContextMenu={(event) => event.preventDefault()}
    >
      <svg className="hold-delete__ring" viewBox={`0 0 ${sizePx} ${sizePx}`} aria-hidden="true">
        <circle className="hold-delete__track" cx={sizePx / 2} cy={sizePx / 2} r={radius} />
        <circle
          className="hold-delete__fill"
          cx={sizePx / 2}
          cy={sizePx / 2}
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          style={{ transition: holding ? 'none' : `stroke-dashoffset ${HOLD_RESET_MS}ms ease-out` }}
        />
      </svg>
      <span className="hold-delete__ripple" aria-hidden="true" />
      <span className="hold-delete__content" aria-hidden="true">
        <span className="hold-delete__label">{label}</span>
        {hint ? <span className="hold-delete__hint">{hint}</span> : null}
      </span>
      <svg className="hold-delete__check" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5.5 12.6l4.1 4.1L18.6 7.6" />
      </svg>
      <span className="hold-delete__sr" role="status">
        {done ? doneLabel : holding ? `${Math.round(progress * 100)}%` : ''}
      </span>
    </button>
  );
}
