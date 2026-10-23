import test from 'node:test';
import assert from 'node:assert/strict';

import {
  HOLD_CONFIRM_MS,
  JELLY_MS,
  JELLY_SPRING,
  MORPH_SHRINK_MS,
  clamp01,
  holdProgress,
  jellySquash,
  jellyStretchFromVelocity,
  jellyTravel,
  nextToggleValue,
  pointerRatio,
  ringDash,
  thumbGeometry,
} from './microInteractionModel.js';

test('spec durations match the interaction requirements', () => {
  assert.equal(MORPH_SHRINK_MS, 300);
  assert.equal(JELLY_MS, 350);
  assert.equal(HOLD_CONFIRM_MS, 1500);
  assert.match(JELLY_SPRING, /^cubic-bezier\(/);
});

test('clamp01 bounds non-finite and out-of-range input', () => {
  assert.equal(clamp01(-1), 0);
  assert.equal(clamp01(0.42), 0.42);
  assert.equal(clamp01(3), 1);
  assert.equal(clamp01(Number.NaN), 0);
});

test('holdProgress fills linearly over 1500ms and clamps at both ends', () => {
  assert.equal(holdProgress(0), 0);
  assert.equal(holdProgress(750), 0.5);
  assert.equal(holdProgress(1500), 1);
  assert.equal(holdProgress(3000), 1);
  assert.equal(holdProgress(-50), 0);
});

test('holdProgress treats a non-positive duration as instantly complete', () => {
  assert.equal(holdProgress(10, 0), 1);
});

test('jellyStretchFromVelocity grows with speed and saturates', () => {
  assert.equal(jellyStretchFromVelocity(0, 10), 0);
  assert.equal(jellyStretchFromVelocity(0.8, 10), 5);
  assert.equal(jellyStretchFromVelocity(1.6, 10), 10);
  assert.equal(jellyStretchFromVelocity(9, 10), 10);
  assert.equal(jellyStretchFromVelocity(-1.6, 10), 10);
});

test('jellyStretchFromVelocity ignores bad peaks and references', () => {
  assert.equal(jellyStretchFromVelocity(1, -4), 0);
  assert.equal(jellyStretchFromVelocity(1, Number.NaN), 0);
  assert.equal(jellyStretchFromVelocity(1.6, 8, 0), 8);
});

test('jellySquash compresses the thumb as it stretches', () => {
  assert.equal(jellySquash(0, 10), 1);
  assert.ok(Math.abs(jellySquash(10, 10) - 0.7) < 1e-9);
  assert.ok(jellySquash(5, 10) < 1);
  assert.ok(jellySquash(5, 10) > jellySquash(10, 10));
});

test('jellyTravel maps displacement to a 0..1 ratio', () => {
  assert.equal(jellyTravel(100, 100, 40), 0);
  assert.equal(jellyTravel(120, 100, 40), 0.5);
  assert.equal(jellyTravel(200, 100, 40), 1);
  assert.equal(jellyTravel(60, 100, 40), 1);
});

test('jellyTravel treats a zero-length run as fully travelled', () => {
  assert.equal(jellyTravel(100, 100, 0), 1);
});

test('pointerRatio maps the usable track span onto 0..1', () => {
  // 轨道 52，内边距 3，滑块 20：可用行程 26，从 x=3 到 x=29
  const base = { trackLeft: 0, trackWidth: 52, padding: 3, thumbWidth: 20 };
  assert.equal(pointerRatio({ ...base, clientX: 3 }), 0);
  assert.equal(pointerRatio({ ...base, clientX: 16 }), 0.5);
  assert.equal(pointerRatio({ ...base, clientX: 29 }), 1);
  assert.equal(pointerRatio({ ...base, clientX: 80 }), 1);
});

test('thumbGeometry keeps the stretched thumb inside the track', () => {
  const atStart = thumbGeometry({ trackWidth: 52, padding: 3, thumbSize: 20, ratio: 0, stretch: 0 });
  assert.deepEqual(atStart, { width: 20, left: 3 });

  const atEnd = thumbGeometry({ trackWidth: 52, padding: 3, thumbSize: 20, ratio: 1, stretch: 0 });
  assert.equal(atEnd.left + atEnd.width, 49);

  const stretched = thumbGeometry({ trackWidth: 52, padding: 3, thumbSize: 20, ratio: 0.5, stretch: 10 });
  assert.equal(stretched.width, 30);
  assert.ok(stretched.left >= 3);
  assert.ok(stretched.left + stretched.width <= 49);
});

test('nextToggleValue treats a short move as a tap and toggles', () => {
  assert.equal(nextToggleValue({ startChecked: false, endRatio: 0.05, travel: 0.03 }), true);
  assert.equal(nextToggleValue({ startChecked: true, endRatio: 0.95, travel: 0.03 }), false);
});

test('nextToggleValue commits by drop position after a real drag', () => {
  assert.equal(nextToggleValue({ startChecked: false, endRatio: 0.2, travel: 0.5 }), false);
  assert.equal(nextToggleValue({ startChecked: false, endRatio: 0.8, travel: 0.5 }), true);
  assert.equal(nextToggleValue({ startChecked: true, endRatio: 0.4, travel: 0.5 }), false);
});

test('ringDash fills the ring exactly when progress reaches one', () => {
  const empty = ringDash({ radius: 38, progress: 0 });
  assert.ok(Math.abs(empty.circumference - 2 * Math.PI * 38) < 1e-9);
  assert.equal(empty.dashOffset, empty.circumference);

  const half = ringDash({ radius: 38, progress: 0.5 });
  assert.ok(Math.abs(half.dashOffset - half.circumference / 2) < 1e-9);

  const full = ringDash({ radius: 38, progress: 1 });
  assert.equal(full.dashOffset, 0);
});
