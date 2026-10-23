// 动画帧采集：用于目视评估动效的连贯性（非常规验收脚本）
// npx --yes --package @playwright/cli playwright-cli -s=uikit run-code --filename scripts/audit/uikit/capture-frames.js
async (page) => {
  const base = 'http://127.0.0.1:5173';
  const dir = 'output/playwright/frames';
  await page.setViewportSize({ width: 1200, height: 900 });
  await page.goto(`${base}/?frames=${Date.now()}#/uikit`);
  await page.getByRole('heading', { name: '微交互演示', exact: true }).waitFor();

  const cards = page.locator('.uikit-card');
  const clipOf = async (locator) => {
    const box = await locator.boundingBox();
    return { x: Math.max(0, box.x - 12), y: Math.max(0, box.y - 12), width: Math.min(900, box.width + 24), height: box.height + 24 };
  };

  // 1) 提交三连变：点击后每 ~55ms 一帧
  const morphCard = cards.nth(0);
  const morphClip = await clipOf(morphCard);
  const morph = page.locator('[data-morph-submit="true"]');
  await morph.click();
  for (let i = 1; i <= 12; i += 1) {
    await page.screenshot({ path: `${dir}/morph-${String(i).padStart(2, '0')}.png`, clip: morphClip });
    await page.waitForTimeout(50);
  }
  await page.locator('[data-morph-submit][data-phase="idle"]').waitFor({ timeout: 5000 });

  // 2) 果冻开关：拖动过程逐帧
  const jellyCard = cards.nth(1);
  const jellyClip = await clipOf(jellyCard);
  const jelly = page.locator('[data-jelly-toggle="true"]');
  const track = await jelly.boundingBox();
  const startX = track.x + 16;
  const y = track.y + track.height / 2;
  await page.mouse.move(startX, y);
  await page.mouse.down();
  for (let i = 1; i <= 6; i += 1) {
    // 快速横移制造速度，再立刻截图，捕捉“拉长—回弹”的瞬间
    await page.mouse.move(startX + i * 9, y, { steps: 1 });
    await page.screenshot({ path: `${dir}/jelly-${String(i).padStart(2, '0')}.png`, clip: jellyClip });
  }
  await page.mouse.up();
  await page.waitForTimeout(420);

  // 2b) 果冻开关：单击“关闭 → 开启”的过渡逐帧
  if (await jelly.getAttribute('data-checked') === 'true') {
    await jelly.click();
    await page.waitForTimeout(520);
  }
  await jelly.click();
  for (let i = 1; i <= 8; i += 1) {
    await page.screenshot({ path: `${dir}/jellytap-${String(i).padStart(2, '0')}.png`, clip: jellyClip });
  }
  await page.waitForTimeout(420);

  // 3) 长按确认：进度推进逐帧
  const holdCard = cards.nth(2);
  const holdClip = await clipOf(holdCard);
  const hold = page.locator('[data-hold-delete="true"]').first();
  const hb = await hold.boundingBox();
  await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2);
  await page.mouse.down();
  for (let i = 1; i <= 6; i += 1) {
    await page.waitForTimeout(230);
    await page.screenshot({ path: `${dir}/hold-${String(i).padStart(2, '0')}.png`, clip: holdClip });
  }
  await page.mouse.up();
  await page.waitForTimeout(200);

  return { captured: 26 };
}
