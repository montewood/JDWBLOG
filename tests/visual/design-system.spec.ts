import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const routes = [
  { name: 'home', path: '/' },
  { name: 'posts', path: '/post/' },
  { name: 'post-regex', path: '/post/regex/' },
  { name: 'projects', path: '/project/' },
  { name: 'study-notes', path: '/courses/' },
] as const;

const viewports = [
  { name: 'mobile', width: 390, height: 844 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1440, height: 1200 },
] as const;

const modes = ['light', 'dark'] as const;

for (const route of routes) {
  for (const viewport of viewports) {
    for (const mode of modes) {
      test(`${route.name} ${viewport.name} ${mode}`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.goto(route.path, { waitUntil: 'networkidle' });
        await page.evaluate(async (selectedMode) => {
          document.documentElement.classList.toggle('dark', selectedMode === 'dark');
          await document.fonts.ready;
        }, mode);

        await expect(page).toHaveScreenshot(
          `${route.name}-${viewport.name}-${mode}.png`,
          {
            animations: 'disabled',
            caret: 'hide',
            mask: [page.locator('iframe')],
            fullPage: false,
          },
        );
      });
    }
  }
}

for (const route of routes) {
  test(`${route.name} has no serious accessibility violations`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1200 });
    await page.goto(route.path, { waitUntil: 'networkidle' });
    const results = await new AxeBuilder({ page })
      .exclude('iframe')
      .options({ runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa'] })
      .analyze();

    const serious = results.violations.filter(
      ({ impact }) => impact === 'serious' || impact === 'critical',
    );
    expect(serious).toEqual([]);
  });
}

test('mobile layouts do not overflow horizontally', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });

  for (const route of routes) {
    await page.goto(route.path, { waitUntil: 'networkidle' });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, `${route.path} horizontal overflow`).toBeLessThanOrEqual(1);
  }
});

test('Shinylive remains an isolated runnable app', async ({ page }) => {
  await page.goto('/apps/profile-widget/', { waitUntil: 'domcontentloaded' });
  await expect(page.locator('#root')).toBeVisible();

  const manifest = await page.request.get('/apps/profile-widget/app.json');
  expect(manifest.ok()).toBeTruthy();
  const files = (await manifest.json()) as Array<{ name: string; content: string }>;
  expect(files.some(({ name, content }) => name === 'app.R' && content.includes('shinyApp(ui, server)'))).toBeTruthy();
});
