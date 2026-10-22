// 微交互演示页浏览器验收：提交三连变、果冻开关、长按确认删除。
// npx --yes --package @playwright/cli playwright-cli -s=uikit run-code --filename scripts/audit/uikit/browser-check.js
async (page) => {
  const checks = [], errors = [];
  const check = (condition, label) => { if (!condition) throw new Error(label); checks.push(label); };
  const near = (a, b, tolerance = 1.5) => Math.abs(a - b) <= tolerance;
  page.on('pageerror', error => errors.push(error.message));

  const base = 'http://127.0.0.1:5173';
  // 带查询参数强制重新加载文档，避免复用会话里已加载的旧 bundle
  const runId = Date.now();
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${base}/?uikit-audit=${runId}#/uikit`);
  await page.getByRole('heading', { name: '微交互演示', exact: true }).waitFor();

  const navBottom = await page.locator('.nav-glass').evaluate(el => el.getBoundingClientRect().bottom);
  const titleTop = await page.locator('.uikit-page__title').evaluate(el => el.getBoundingClientRect().top);
  check(titleTop >= navBottom, `page title clears the fixed navbar (${titleTop} >= ${navBottom})`);

  // ── 1. 提交三连变 ─────────────────────────────
  const morph = page.locator('.uikit-page [data-morph-submit="true"]');
  check(await morph.count() === 1, 'morph button rendered');
  const idleBox = await morph.boundingBox();
  const shrinkDuration = await morph.evaluate(el => getComputedStyle(el).transitionDuration);
  check(shrinkDuration.startsWith('0.3s'), `shrink transition is 300ms (got ${shrinkDuration})`);

  await morph.click();
  await page.waitForTimeout(150);
  check(await morph.getAttribute('data-phase') === 'loading', 'click enters loading phase');
  await page.waitForTimeout(300);
  const loadingBox = await morph.boundingBox();
  check(near(loadingBox.width, loadingBox.height), 'button shrinks into a circle while loading');
  check(loadingBox.width < idleBox.width - 10, `loading width ${loadingBox.width} is smaller than idle width ${idleBox.width}`);
  await morph.screenshot({ path: 'output/playwright/uikit-morph-loading.png' });

  await page.locator('[data-morph-submit][data-phase="success"]').waitFor({ timeout: 4000 });
  await page.waitForTimeout(320); // 等待 240ms 背景色过渡完成
  const successBg = await morph.evaluate(el => getComputedStyle(el).backgroundColor);
  check(successBg === 'rgb(99, 232, 191)', `success state turns green (got ${successBg})`);
  check(await morph.locator('.morph-submit__check').isVisible(), 'success state shows the check mark');
  const successBox = await morph.boundingBox();
  check(near(successBox.width, successBox.height), 'success state stays circular');
  await morph.screenshot({ path: 'output/playwright/uikit-morph-success.png' });

  await page.locator('[data-morph-submit][data-phase="idle"]').waitFor({ timeout: 4000 });
  await page.waitForTimeout(400); // 等恢复原状的 300ms 过渡走完
  const restoredBox = await morph.boundingBox();
  check(near(restoredBox.width, idleBox.width, 2), 'button restores its original width');
  check((await page.locator('[data-demo="morph-state"]').innerText()).includes('1'), 'submission counter increments once');

  // ── 2. 果冻开关 ─────────────────────────────
  const jelly = page.locator('.uikit-page [data-jelly-toggle="true"]');
  const thumb = jelly.locator('.jelly-toggle__thumb');
  check(await jelly.getAttribute('role') === 'switch', 'jelly toggle exposes switch role');
  check(await jelly.getAttribute('data-checked') === 'false', 'jelly toggle starts off');

  const jellyDuration = await thumb.evaluate(el => getComputedStyle(el).transitionDuration);
  check(jellyDuration.startsWith('0.35s'), `jelly transition is 350ms (got ${jellyDuration})`);
  const jellyCurve = await thumb.evaluate(el => getComputedStyle(el).transitionTimingFunction);
  const curveNumbers = (jellyCurve.match(/cubic-bezier\(([^)]+)\)/)?.[1] || '').split(',').map(Number);
  check(
    curveNumbers.length === 4 && curveNumbers.every(Number.isFinite) && curveNumbers[3] > 1,
    `jelly uses a spring curve with overshoot (got ${jellyCurve})`,
  );

  const trackBox = await jelly.boundingBox();
  const thumbBox = await thumb.boundingBox();
  const startX = thumbBox.x + thumbBox.width / 2;
  const midY = trackBox.y + trackBox.height / 2;

  await page.mouse.move(startX, midY);
  await page.mouse.down();
  await page.mouse.move(startX + 14, midY, { steps: 2 });
  await page.waitForTimeout(60);
  const stretchedBox = await thumb.boundingBox();
  check(stretchedBox.width > thumbBox.width + 1, 'thumb stretches horizontally while dragging');
  await jelly.screenshot({ path: 'output/playwright/uikit-jelly-stretch.png' });
  await page.mouse.move(trackBox.x + trackBox.width - 6, midY, { steps: 4 });
  await page.mouse.up();
  await page.waitForTimeout(450);

  check(await jelly.getAttribute('data-checked') === 'true', 'drag to the end commits the on state');
  const trackBg = await jelly.evaluate(el => getComputedStyle(el).backgroundColor);
  check(trackBg === 'rgb(99, 232, 191)', `track turns green (got ${trackBg})`);
  const settledThumb = await thumb.boundingBox();
  check(near(settledThumb.width, thumbBox.width, 1.5), 'thumb springs back to its normal width');

  // 点击（位移不足）应取反；同时确认单击有可见的位移动画，而不是瞬移
  const thumbX = () => thumb.evaluate(el => new DOMMatrixReadOnly(getComputedStyle(el).transform).m41);
  const beforeX = await thumbX();
  const onThumb = await thumb.boundingBox();
  await page.mouse.move(onThumb.x + onThumb.width / 2, midY);
  await page.mouse.down();
  await page.waitForTimeout(70);
  const pressX = await thumbX();
  check(Math.abs(pressX - beforeX) < 1, 'pointerdown keeps the thumb in place (no teleport)');
  await page.mouse.up();
  await page.waitForTimeout(70);
  const midX = await thumbX();
  check(Math.abs(midX - beforeX) > 0.5, 'thumb starts moving right after a tap');
  check(Math.abs(midX) > 2, 'tap motion is still in progress shortly after release, not instant');
  await page.waitForTimeout(450);
  const endX = await thumbX();
  check(Math.abs(endX) < 1.5, 'thumb settles at the other side after the transition');
  check(await jelly.getAttribute('data-checked') === 'false', 'tap toggles the switch back off');

  // 键盘
  await jelly.focus();
  await page.keyboard.press('Enter');
  await page.waitForTimeout(400);
  check(await jelly.getAttribute('data-checked') === 'true', 'Enter toggles the switch');
  await page.keyboard.press(' ');
  await page.waitForTimeout(400);
  check(await jelly.getAttribute('data-checked') === 'false', 'Space toggles the switch');

  // ── 3. 长按确认删除 ─────────────────────────
  const hold = page.locator('.uikit-page [data-hold-delete="true"]').first();
  check(await hold.count() === 1, 'hold-to-delete control rendered');
  const holdBox = await hold.boundingBox();
  const hx = holdBox.x + holdBox.width / 2;
  const hy = holdBox.y + holdBox.height / 2;

  // 中途松手
  await page.mouse.move(hx, hy);
  await page.mouse.down();
  await page.waitForTimeout(600);
  const midway = Number(await hold.getAttribute('data-hold-progress'));
  check(midway > 0.2 && midway < 0.9, `ring fills gradually while held (got ${midway})`);
  await page.locator('.uikit-card').nth(2).screenshot({ path: 'output/playwright/uikit-hold-progress.png' });
  await page.mouse.up();
  await page.waitForTimeout(350);
  const resetValue = Number(await hold.getAttribute('data-hold-progress'));
  check(resetValue < 0.05, `releasing mid-way resets progress to zero (got ${resetValue})`);
  check(await page.locator('.uikit-row--removed').count() === 0, 'partial hold does not delete anything');
  check(await hold.getAttribute('data-phase') === 'idle', 'partial hold returns to idle');

  // 完整按住 1.5s
  await page.mouse.move(hx, hy);
  await page.mouse.down();
  await page.waitForTimeout(1650);
  check(await hold.getAttribute('data-phase') === 'done', 'completing 1.5s hold switches to done');
  check(await hold.locator('.hold-delete__check').isVisible(), 'completion shows the check mark');
  await page.mouse.up();
  await page.waitForTimeout(700);
  check(await page.locator('.uikit-row--removed').count() === 1, 'row is deleted after a completed hold');

  // 列表行内的紧凑尺寸（业务页删除按钮使用）
  const compact = page.locator('.uikit-page [data-hold-delete="true"][data-size="compact"]').first();
  check(await compact.count() === 1, 'compact hold-to-delete variant rendered');
  const compactBox = await compact.boundingBox();
  check(
    Math.round(compactBox.width) === 44 && Math.round(compactBox.height) === 44,
    `compact variant is 44px (${compactBox.width}x${compactBox.height})`,
  );
  const compactCx = compactBox.x + compactBox.width / 2;
  const compactCy = compactBox.y + compactBox.height / 2;
  await page.mouse.move(compactCx, compactCy);
  await page.mouse.down();
  await page.waitForTimeout(1650);
  await page.mouse.up();
  check(await compact.getAttribute('data-phase') === 'done', 'compact variant completes its hold');

  await page.screenshot({ path: 'output/playwright/uikit-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}/?uikit-audit=${runId}-mobile#/uikit`);
  await page.getByRole('heading', { name: '微交互演示', exact: true }).waitFor();
  check(await page.evaluate(() => document.documentElement.scrollWidth <= 390), '390px layout has no horizontal overflow');
  await page.screenshot({ path: 'output/playwright/uikit-390.png', fullPage: true });

  check(errors.length === 0, `no browser runtime errors (${errors.join(' | ')})`);
  return { passed: checks.length, checks, errors };
}
