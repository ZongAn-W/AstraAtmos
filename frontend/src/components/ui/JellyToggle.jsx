import { useCallback, useEffect, useRef, useState } from 'react';
import './microInteractions.css';
import {
  JELLY_MS,
  JELLY_SPRING,
  jellySquash,
  jellyStretchFromVelocity,
  nextToggleValue,
  pointerRatio,
  thumbGeometry,
} from './microInteractionModel';

const TRACK_HEIGHT = 26;
const TRACK_WIDTH = 52;
const THUMB_SIZE = 20;
const PADDING = 3;
const MAX_STRETCH = 11;
/** 拉伸收敛速度：每帧向目标靠拢的比例，越小越“黏” */
const STRETCH_EASE = 0.38;
/** 超过该位移才进入拖动，保证单击时不先把滑块瞬移到指针下 */
const DRAG_SLOP = 3;

/**
 * 果冻开关：拖动时滑块按指针速度横向拉长并压扁，停下或到终点回弹，
 * 底色灰→绿，350ms 弹簧曲线。位移不足视为点击并取反；键盘 Enter/Space 直接切换。
 */
export default function JellyToggle({
  checked = false,
  onChange,
  label,
  disabled = false,
  className = '',
}) {
  const trackRef = useRef(null);
  const metrics = useRef({ width: TRACK_WIDTH, left: 0 });
  const [drag, setDrag] = useState(null);
  const [stretch, setStretch] = useState(0);

  const dragRef = useRef(null);
  const stretchRef = useRef(0);
  const targetRef = useRef(0);
  const rafRef = useRef(0);
  const lastPoint = useRef({ x: 0, t: 0 });

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  const tick = useCallback(() => {
    // 指针停下后目标值自然衰减，滑块随之收回，形成“停下即回弹”
    targetRef.current *= 0.88;
    const next = stretchRef.current + (targetRef.current - stretchRef.current) * STRETCH_EASE;
    stretchRef.current = Math.abs(targetRef.current - next) < 0.05 ? targetRef.current : next;
    setStretch(stretchRef.current);
    if (dragRef.current) rafRef.current = requestAnimationFrame(tick);
  }, []);

  const readMetrics = () => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return metrics.current;
    metrics.current = { width: rect.width, left: rect.left };
    return metrics.current;
  };

  const ratioFromEvent = (clientX) => {
    const m = readMetrics();
    return pointerRatio({
      clientX,
      trackLeft: m.left,
      trackWidth: m.width,
      padding: PADDING,
      thumbWidth: THUMB_SIZE,
    });
  };

  /** 用移动距离与时间差估算速度（px/ms），dt 下限避免首帧尖峰 */
  const trackVelocity = (clientX) => {
    const now = performance.now();
    const prev = lastPoint.current;
    const dt = Math.max(8, now - prev.t);
    const velocity = Math.abs(clientX - prev.x) / dt;
    lastPoint.current = { x: clientX, t: now };
    return velocity;
  };

  const handlePointerDown = useCallback((event) => {
    if (disabled) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    readMetrics();
    const ratio = ratioFromEvent(event.clientX);
    const next = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startRatio: ratio,
      ratio,
      travel: 0,
      moved: false,
    };
    dragRef.current = next;
    lastPoint.current = { x: event.clientX, t: performance.now() };
    targetRef.current = 0;
    setDrag(next);
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
  }, [disabled, tick]);

  const handlePointerMove = useCallback((event) => {
    const current = dragRef.current;
    if (!current || event.pointerId !== current.pointerId) return;
    // 未超过阈值前不跟随指针：单击应当保持原位，交给松手后的过渡
    if (!current.moved && Math.abs(event.clientX - current.startX) < DRAG_SLOP) return;
    const ratio = ratioFromEvent(event.clientX);
    const next = {
      ...current,
      moved: true,
      ratio,
      travel: Math.abs(ratio - current.startRatio),
    };
    dragRef.current = next;
    targetRef.current = jellyStretchFromVelocity(trackVelocity(event.clientX), MAX_STRETCH);
    setDrag(next);
  }, []);

  const finishDrag = useCallback((event, commit) => {
    const current = dragRef.current;
    if (!current || event.pointerId !== current.pointerId) return;
    const ratio = ratioFromEvent(event.clientX);
    const travel = current.moved ? Math.abs(ratio - current.startRatio) : 0;
    dragRef.current = null;
    cancelAnimationFrame(rafRef.current);
    targetRef.current = 0;
    stretchRef.current = 0;
    setStretch(0);
    setDrag(null);
    if (!commit) return;
    const next = nextToggleValue({ startChecked: checked, endRatio: ratio, travel });
    if (next !== checked) onChange?.(next);
  }, [checked, onChange]);

  const handleKeyDown = useCallback((event) => {
    if (disabled) return;
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    onChange?.(!checked);
  }, [checked, disabled, onChange]);

  const ratio = drag?.moved ? drag.ratio : (checked ? 1 : 0);
  const squash = jellySquash(stretch, MAX_STRETCH);
  const geo = thumbGeometry({
    trackWidth: metrics.current.width || TRACK_WIDTH,
    padding: PADDING,
    thumbSize: THUMB_SIZE,
    ratio,
    stretch,
  });

  return (
    <span
      ref={trackRef}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      aria-disabled={disabled || undefined}
      tabIndex={disabled ? -1 : 0}
      className={`jelly-toggle ${className}`.trim()}
      data-checked={checked ? 'true' : 'false'}
      data-dragging={drag?.moved ? 'true' : 'false'}
      data-jelly-toggle="true"
      style={{
        '--jelly-height': `${TRACK_HEIGHT}px`,
        '--jelly-width': `${TRACK_WIDTH}px`,
        '--jelly-thumb': `${THUMB_SIZE}px`,
        '--jelly-ms': `${JELLY_MS}ms`,
        '--jelly-spring': JELLY_SPRING,
      }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={(event) => finishDrag(event, true)}
      onPointerCancel={(event) => finishDrag(event, false)}
      onKeyDown={handleKeyDown}
    >
      <span
        className="jelly-toggle__thumb"
        aria-hidden="true"
        style={{
          width: `${geo.width}px`,
          transform: `translateX(${geo.left - PADDING}px) scaleY(${squash.toFixed(3)})`,
        }}
      />
    </span>
  );
}
