// 业务页接入验收：设置面板果冻开关、预测页三连变按钮。
// npx --yes --package @playwright/cli playwright-cli -s=apply run-code --filename scripts/audit/uikit/app-integration-check.js
async (page) => {
  const checks = [], errors = [];
  const check = (condition, label) => { if (!condition) throw new Error(label); checks.push(label); };
  page.on('pageerror', error => errors.push(error.message));
  const base = 'http://127.0.0.1:5173';
  const runId = Date.now();
  await page.setViewportSize({ width: 1440, height: 900 });

  // ── 设置面板：开关应为果冻滑块，且切换有过渡 ──
  await page.goto(`${base}/?apply=${runId}#/`);
  await page.locator('.av-quick-trigger').first().click();
  await page.getByRole('menuitem', { name: /设置|Settings/ }).first().click();
  const panel = page.getByRole('dialog').first();
  await panel.waitFor();

  const switches = panel.locator('[data-jelly-toggle="true"]');
  const switchCount = await switches.count();
  check(switchCount >= 1, `settings panel renders jelly switches (${switchCount})`);
  check(await panel.locator('input[type="checkbox"]').count() === 0, 'settings panel no longer uses native checkboxes');

  const target = switches.first();
  const firstState = await target.getAttribute('data-checked');
  const thumb = target.locator('.jelly-toggle__thumb');
  const duration = await thumb.evaluate(el => getComputedStyle(el).transitionDuration);
  check(duration.startsWith('0.35s'), `settings switch keeps the 350ms spring (got ${duration})`);
  await target.click();
  await page.waitForTimeout(120);
  const movingX = await thumb.evaluate(el => new DOMMatrixReadOnly(getComputedStyle(el).transform).m41);
  await page.waitForTimeout(420);
  const settledState = await target.getAttribute('data-checked');
  const settledX = await thumb.evaluate(el => new DOMMatrixReadOnly(getComputedStyle(el).transform).m41);
  check(settledState !== firstState, 'settings switch flips state after click');
  check(Math.abs(settledX - movingX) > 1, 'settings switch animates instead of jumping');
  await panel.screenshot({ path: 'output/playwright/app-settings-switch.png' });
  await page.keyboard.press('Escape');
  await page.waitForTimeout(250);

  // ── 预测页：主按钮应为三连变组件 ──
  await page.goto(`${base}/?apply=${runId}b#/predict`);
  const runButton = page.locator('.predict-sidebar__actions [data-morph-submit="true"]');
  await runButton.waitFor({ timeout: 15000 });
  check(await runButton.count() === 1, 'predict sidebar uses the morph submit button');
  const runDuration = await runButton.evaluate(el => getComputedStyle(el).transitionDuration);
  check(runDuration.startsWith('0.3s'), `predict button keeps the 300ms shrink (got ${runDuration})`);
  const label = (await runButton.innerText()).trim();
  check(label.length > 0, `predict button keeps its label (${label})`);
  await page.screenshot({ path: 'output/playwright/app-predict-button.png' });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}/?apply=${runId}c#/predict`);
  await page.locator('.predict-sidebar__actions [data-morph-submit="true"]').waitFor({ timeout: 15000 });
  check(await page.evaluate(() => document.documentElement.scrollWidth <= 390), '390px predict page has no horizontal overflow');

  check(errors.length === 0, `no runtime errors (${errors.join(' | ')})`);
  return { passed: checks.length, checks, errors };
}
