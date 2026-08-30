import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const analyticsPattern =
  'https://raw.githubusercontent.com/montewood/gh-action/refs/heads/main/output/GA-*.json';
const dailyAnalyticsPayload = JSON.stringify({
  schemaVersion: 2,
  date: '2026-07-18',
  activeUsers: 2,
  totalUsers: 3,
});

test.beforeEach(async ({ page }) => {
  await page.route(analyticsPattern, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: dailyAnalyticsPayload,
    });
  });
  await page.route('**/api/dashboard-launch**', async (route) => {
    const isLaunch = route.request().method() === 'POST';
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(isLaunch
        ? {
            allowed: true,
            incremented: true,
            count: 1,
            limit: 20,
            remaining: 19,
            expiresAt: Date.now() + 30 * 60 * 1000,
            appUrl: '/apps/profile-widget/?launch_token=test-token',
          }
        : {
            count: 0,
            limit: 20,
            remaining: 20,
            available: true,
          }),
    });
  });
});

const routes = [
  { name: 'home', path: '/' },
  { name: 'posts', path: '/post/' },
  { name: 'post-regex', path: '/post/regex/' },
  { name: 'post-rstudio', path: '/post/rstudio-1-rstudio-server/' },
  { name: 'projects', path: '/project/' },
  { name: 'privacy', path: '/privacy/' },
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
  test.setTimeout(90_000);
  await page.goto('/apps/profile-widget/', { waitUntil: 'domcontentloaded' });
  await expect(page.locator('#root')).toBeVisible();

  const manifest = await page.request.get('/apps/profile-widget/app.json');
  expect(manifest.ok()).toBeTruthy();
  const files = (await manifest.json()) as Array<{ name: string; content: string }>;
  expect(files.some(({ name, content }) => name === 'app.R' && content.includes('shinyApp(ui, server)'))).toBeTruthy();

  const appFrame = page.frameLocator('iframe');
  await expect(
    appFrame.getByRole('heading', { name: 'JDW Blog R Dashboard' }),
  ).toBeVisible({ timeout: 60_000 });
  await expect(appFrame.locator('#load_status')).toContainText(
    '90일 수집 완료',
    { timeout: 30_000 },
  );
  await expect(appFrame.locator('#observation_total')).toHaveText('2');
});

test('homepage defers Shinylive until the dashboard button is clicked', async ({ page }) => {
  const shinyliveRequests: string[] = [];
  const analyticsRequests: string[] = [];
  let launchRequests = 0;

  page.on('request', (request) => {
    if (request.url().includes('/apps/profile-widget/')) {
      shinyliveRequests.push(request.url());
    }
    if (request.url().includes('/output/GA-')) {
      analyticsRequests.push(request.url());
    }
    if (
      request.url().includes('/api/dashboard-launch') &&
      request.method() === 'POST'
    ) {
      launchRequests += 1;
    }
  });

  await page.goto('/', { waitUntil: 'networkidle' });
  await expect(page.locator('[data-dashboard-latest]')).toHaveText('2');
  await expect(page.locator('[data-dashboard-quota]')).toHaveText('20/20');
  await expect(page.locator('[data-dashboard-open]')).toBeEnabled();
  await expect(page.locator('[data-dashboard-table-body]')).toHaveCount(0);
  await expect(page.locator('[data-dashboard-frame-host] iframe')).toHaveCount(0);
  expect(analyticsRequests).toHaveLength(90);
  expect(shinyliveRequests).toEqual([]);

  await page.route('**/apps/profile-widget/**', async (route) => {
    await route.abort();
  });
  await page.locator('[data-dashboard-open]').click();

  const dialog = page.locator('[data-dashboard-dialog]');
  await expect(dialog).toBeVisible();
  await expect(dialog.locator('iframe')).toHaveAttribute(
    'src',
    '/apps/profile-widget/?launch_token=test-token',
  );
  expect(shinyliveRequests.length).toBeGreaterThan(0);
  expect(launchRequests).toBe(1);

  await page.locator('[data-dashboard-close]').click();
  await expect(dialog).not.toBeVisible();
  await expect(page.locator('[data-dashboard-frame-host] iframe')).toHaveCount(0);
  await expect(page.locator('[data-dashboard-open]')).toBeFocused();

  await page.locator('[data-dashboard-open]').click();
  expect(launchRequests).toBe(1);
});

test('post metadata is generated from Hugo and linked taxonomies', async ({ page }) => {
  await page.goto('/', { waitUntil: 'networkidle' });
  const post = page.locator('.jdw-post-list__item', {
    has: page.locator('a[href="/post/rstudio-1-rstudio-server/"]'),
  });

  // Production sets HUGO_ENABLEGITINFO, so its Lastmod comes from the commit
  // date. Here enableGitInfo stays off and the front matter date is rendered.
  await expect(post.locator('time')).toHaveAttribute('datetime', '2023-10-24');
  await expect(post.locator('.jdw-post-list__meta')).toContainText('4분 읽기');
  await expect(post.getByRole('link', { name: 'Colab', exact: true })).toHaveAttribute(
    'href',
    '/categories/colab/',
  );
  await expect(post.getByRole('link', { name: '#Rstudio' })).toHaveAttribute(
    'href',
    '/tags/rstudio/',
  );
});

test('post pages use the article-focused layout', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1200 });
  await page.goto('/post/regex/', { waitUntil: 'networkidle' });

  await expect(page.locator('.jdw-post-shell')).toBeVisible();
  await expect(page.locator('.hb-sidebar-container')).toHaveCount(0);
  await expect(page.locator('.hb-toc')).toHaveCount(0);
  await expect(page.locator('.jdw-post__toc')).toBeVisible();
  await expect(page.locator('.jdw-post__author img')).toBeVisible();
  await expect(page.locator('.jdw-post__related')).toBeVisible();
  await expect(page.getByRole('link', { name: '#정규표현식' })).toHaveAttribute(
    'href',
    /\/tags\//,
  );

  await expect(
    page.locator('script[src$="index_files/htmlwidgets/htmlwidgets.js"]'),
  ).toHaveCount(1);
  await expect(
    page.locator('link[href$="index_files/str_view/str_view.css"]'),
  ).toHaveCount(1);
});

test('post table of contents switches to a disclosure on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/post/regex/', { waitUntil: 'networkidle' });

  await expect(page.locator('.jdw-post__toc')).not.toBeVisible();
  await expect(page.locator('.jdw-post__toc-mobile')).toBeVisible();
  await expect(page.locator('#TableOfContentsMobile')).toHaveCount(1);
  await expect(page.locator('#TableOfContents')).toHaveCount(1);
});

test('post lead images are centered within the article column', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1200 });
  await page.goto('/post/rstudio-1-rstudio-server/', { waitUntil: 'networkidle' });

  const content = page.locator('.jdw-post__content');
  const leadImage = content.locator('> p > img').first();
  await expect(leadImage).toBeVisible();

  const contentBox = await content.boundingBox();
  const imageBox = await leadImage.boundingBox();
  expect(contentBox).not.toBeNull();
  expect(imageBox).not.toBeNull();

  const contentCenter = contentBox!.x + contentBox!.width / 2;
  const imageCenter = imageBox!.x + imageBox!.width / 2;
  expect(Math.abs(contentCenter - imageCenter)).toBeLessThanOrEqual(1);
});

test('desktop header aligns with the homepage content shell', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1200 });
  await page.goto('/', { waitUntil: 'networkidle' });

  const navbarBox = await page.locator('.navbar').boundingBox();
  const aboutBox = await page.locator('.jdw-about').boundingBox();
  expect(navbarBox).not.toBeNull();
  expect(aboutBox).not.toBeNull();

  expect(Math.abs(navbarBox!.x - aboutBox!.x)).toBeLessThanOrEqual(1);
  expect(
    Math.abs(
      navbarBox!.x + navbarBox!.width - (aboutBox!.x + aboutBox!.width),
    ),
  ).toBeLessThanOrEqual(1);
});
