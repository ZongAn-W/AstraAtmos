// 第二批动效验收：数字滚动、骨架屏、错落、Toast 退场、导航指示条、弹窗入场、主题切换
// npx --yes --package @playwright/cli playwright-cli -s=motion run-code --filename scripts/audit/uikit/motion-check.js
async (page) => {
  const checks = [], errors = [];
  const check = (condition, label) => { if (!condition) throw new Error(label); checks.push(label); };
  page.on('pageerror', error => errors.push(error.message));
  const base = 'http://127.0.0.1:5173';
  const runId = Date.now();
  await page.setViewportSize({ width: 1280, height: 900 });

  // ── 演示页：数字滚动 / 骨架屏 / 错落 / Toast ──
  await page.goto(`${base}/?motion=${runId}#/uikit`);
  await page.getByRole('heading', { name: '微交互演示', exact: true }).waitFor();

  const number = page.locator('[data-demo-number="true"]');
  const initial = Number(await number.innerText());
  await page.getByRole('button', { name: '0.9461', exact: true }).click();
  await page.waitForTimeout(90);
  const midway = Number(await number.innerText());
  await page.waitForTimeout(650);
  const settled = Number(await number.innerText());
  check(Math.abs(settled - 0.9461) < 0.0002, `number settles on the target (${settled})`);
  check(midway > initial && midway < settled - 0.0005, `number is still rolling shortly after the change (${initial} -> ${midway} -> ${settled})`);

  const skeleton = page.locator('[data-skeleton-demo="true"] .ui-skeleton').first();
  const skeletonAnim = await skeleton.evaluate(el => {
    const style = getComputedStyle(el);
    return { name: style.animationName, duration: style.animationDuration, size: style.backgroundSize };
  });
  check(skeletonAnim.name === 'motion-shimmer', `skeleton uses the shimmer animation (${skeletonAnim.name})`);
  check(skeletonAnim.duration.startsWith('1.4s'), `skeleton shimmer duration (${skeletonAnim.duration})`);

  const staggerItems = page.locator('[data-stagger-demo="true"] .motion-stagger-item');
  check(await staggerItems.count() === 4, 'stagger demo renders four rows');
  const delays = await staggerItems.evaluateAll(nodes => nodes.map(node => getComputedStyle(node).animationDelay));
  check(delays.join('|') === '0s|0.03s|0.06s|0.09s', `stagger delays cascade (${delays.join(',')})`);
  const staggerName = await staggerItems.first().evaluate(el => getComputedStyle(el).animationName);
  check(staggerName === 'motion-fade-up', `stagger uses the fade-up keyframes (${staggerName})`);

  await page.getByRole('button', { name: '显示一条提示', exact: true }).click();
  const toast = page.locator('[data-toast="true"]');
  await toast.waitFor({ timeout: 3000 });
  check(await toast.getAttribute('data-toast-exiting') === 'false', 'toast enters without an exit marker');
  await toast.click();
  await page.waitForTimeout(40);
  check(await page.locator('[data-toast="true"][data-toast-exiting="true"]').count() === 1, 'toast plays the exit animation before unmounting');
  await page.waitForTimeout(320);
  check(await page.locator('[data-toast="true"]').count() === 0, 'toast unmounts after the exit animation');

  // ── 应用页：导航指示条滑动 ──
  await page.goto(`${base}/?motion=${runId}b#/`);
  const indicator = page.locator('.nav-indicator');
  await indicator.waitFor({ timeout: 10000 });
  const readX = () => indicator.evaluate(el => new DOMMatrixReadOnly(getComputedStyle(el).transform).m41);
  await page.waitForTimeout(300);
  const homeX = await readX();
  await page.getByRole('button', { name: '数据总览', exact: true }).click();
  // 页面切换有 200ms 过渡，指示条在 current 变更后才开始滑动
  await page.waitForTimeout(250);
  const midX = await readX();
  await page.waitForTimeout(400);
  const overviewX = await readX();
  check(Math.abs(overviewX - homeX) > 20, `indicator moves to the new page (${homeX} -> ${overviewX})`);
  check(Math.abs(midX - homeX) > 1 && Math.abs(midX - overviewX) > 1, `indicator slides instead of jumping (mid ${midX})`);

  // ── 弹窗入场 ──
  await page.goto(`${base}/?motion=${runId}c#/`);
  await page.getByRole('button', { name: /登录|注册/ }).first().click();
  const modal = page.locator('.av-modal').first();
  await modal.waitFor({ timeout: 10000 });
  const modalAnim = await modal.evaluate(el => getComputedStyle(el).animationName);
  check(modalAnim === 'motion-modal-in', `modal plays its entrance animation (${modalAnim})`);
  await page.keyboard.press('Escape');

  // ── 主题切换 ──
  await page.evaluate(() => {
    window.__vtCalls = 0;
    const original = document.startViewTransition?.bind(document);
    if (original) {
      document.startViewTransition = (callback) => {
        window.__vtCalls += 1;
        return original(callback);
      };
    }
  });
  await page.locator('.av-quick-trigger').first().click();
  await page.getByRole('menuitem', { name: /主题|Theme/ }).first().click();
  const themeBefore = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  const targetThemeLabel = themeBefore === 'light' ? /深色|Dark/ : /浅色|Light/;
  await page.getByRole('menuitemradio', { name: targetThemeLabel }).first().click();
  await page.waitForTimeout(600);
  const themeAfter = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  check(themeAfter !== themeBefore, `theme switches (${themeBefore} -> ${themeAfter})`);
  const viewTransitionCalls = await page.evaluate(() => window.__vtCalls);
  const revealOrigin = await page.evaluate(() => document.documentElement.style.getPropertyValue('--theme-reveal-x'));
  if (await page.evaluate(() => typeof document.startViewTransition === 'function')) {
    check(viewTransitionCalls >= 1, `theme switch starts a view transition (${viewTransitionCalls})`);
    check(revealOrigin.trim().endsWith('px'), `reveal origin is anchored at the clicked control (${revealOrigin})`);
  }
  await page.screenshot({ path: 'output/playwright/motion-light-theme.png' });

  check(errors.length === 0, `no runtime errors (${errors.join(' | ')})`);
  return { passed: checks.length, checks, errors };
}
