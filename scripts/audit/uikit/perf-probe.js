// 动效流畅度探针：在真实交互过程中采样 rAF 帧间隔，量化是否存在掉帧。
// npx --yes --package @playwright/cli playwright-cli -s=uikit run-code --filename scripts/audit/uikit/perf-probe.js
async (page) => {
  const base = 'http://127.0.0.1:5173';
  await page.setViewportSize({ width: 1200, height: 900 });
  await page.goto(`${base}/?perf=${Date.now()}#/uikit`);
  await page.getByRole('heading', { name: '微交互演示', exact: true }).waitFor();

  await page.evaluate(() => {
    window.__frames = [];
    let last = performance.now();
    window.__sampling = true;
    const loop = (t) => {
      window.__frames.push(t - last);
      last = t;
      if (window.__sampling) requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  });

  // 1) 提交三连变
  const morph = page.locator('[data-morph-submit="true"]');
  await morph.click();
  await page.waitForTimeout(1500);
  await page.locator('[data-morph-submit][data-phase="idle"]').waitFor({ timeout: 5000 });
  await page.waitForTimeout(400);

  // 2) 果冻开关：快速往返拖动
  const jelly = page.locator('[data-jelly-toggle="true"]');
  const track = await jelly.boundingBox();
  const y = track.y + track.height / 2;
  await page.mouse.move(track.x + 14, y);
  await page.mouse.down();
  for (let i = 0; i < 12; i += 1) {
    await page.mouse.move(track.x + 14 + (i % 2 === 0 ? 26 : 0), y);
    await page.waitForTimeout(16);
  }
  await page.mouse.up();
  await page.waitForTimeout(500);

  // 3) 长按确认删除
  const hold = page.locator('[data-hold-delete="true"]').first();
  const hb = await hold.boundingBox();
  await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(1700);
  await page.mouse.up();
  await page.waitForTimeout(400);

  const stats = await page.evaluate(() => {
    window.__sampling = false;
    const d = window.__frames.slice(1).filter((value) => value > 0);
    const sorted = [...d].sort((a, b) => a - b);
    const sum = d.reduce((acc, value) => acc + value, 0);
    const pick = (p) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
    return {
      frames: d.length,
      avg: Number((sum / d.length).toFixed(2)),
      p50: Number(pick(0.5).toFixed(2)),
      p95: Number(pick(0.95).toFixed(2)),
      max: Number(Math.max(...d).toFixed(2)),
      over32ms: d.filter((value) => value > 32).length,
      over50ms: d.filter((value) => value > 50).length,
    };
  });

  return stats;
}
