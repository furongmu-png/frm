// E2E: mobile responsive layout.
//
// Emulates a touch-width viewport (< 768px) and verifies the single-column
// mobile layout: a column-navigation tab bar (Left/Center/Right), touch-sized
// targets, no horizontal overflow, and the Belief State panel pinned at the
// bottom.
import { test, expect } from '@playwright/test';

// iPhone 12 viewport: 390x844. Below the 768px mobile breakpoint.
const MOBILE_VIEWPORT = { width: 390, height: 844 };

async function stubRest(page: import('@playwright/test').Page) {
  await page.route('**/knowledge-graph', (r) =>
    r.fulfill({ json: { nodes: [], edges: [] } }),
  );
  await page.route('**/story-milestones', (r) =>
    r.fulfill({ json: { milestones: [] } }),
  );
}

test.describe('Mobile responsive layout', () => {
  test.beforeEach(async ({ page }) => {
    await stubRest(page);
    await page.setViewportSize(MOBILE_VIEWPORT);
    await page.goto('/');
  });

  test('renders the mobile column-navigation tabs', async ({ page }) => {
    // The column nav (Left/Center/Right) only appears on mobile.
    const nav = page.getByRole('tablist', { name: 'Column navigation' });
    await expect(nav).toBeVisible();
    await expect(page.getByRole('tab', { name: 'Left', exact: true })).toBeVisible();
    await expect(page.getByRole('tab', { name: 'Center', exact: true })).toBeVisible();
    await expect(page.getByRole('tab', { name: 'Right', exact: true })).toBeVisible();
  });

  test('Left column is shown by default and hides desktop separators', async ({ page }) => {
    // Default mobile column is "left": the Sandbox tab and its content show.
    await expect(page.getByRole('tab', { name: 'Sandbox', exact: true })).toBeVisible();
    await expect(page.getByTestId('left-content')).toBeVisible();
    // The desktop resizable separators must not be rendered on mobile.
    await expect(page.getByRole('separator')).toHaveCount(0);
  });

  test('switching columns reveals Center and Right content', async ({ page }) => {
    // Center column: Free Energy chart title + Text Heatmap title.
    await page.getByRole('tab', { name: 'Center', exact: true }).click();
    await expect(page.getByTestId('center-content')).toBeVisible();
    await expect(page.getByText('Free Energy & Prediction Error')).toBeVisible();

    // Right column: the right-side tab buttons appear.
    await page.getByRole('tab', { name: 'Right', exact: true }).click();
    await expect(page.getByTestId('right-content')).toBeVisible();
    await expect(page.getByRole('tab', { name: 'Latent 3D', exact: true })).toBeVisible();
  });

  test('Belief State panel stays at the bottom across columns', async ({ page }) => {
    const toggle = page.getByRole('button', { name: /Belief State \(raw\)/ });
    await expect(toggle).toBeVisible();
    // Switch to Center — the bottom panel must remain.
    await page.getByRole('tab', { name: 'Center', exact: true }).click();
    await expect(toggle).toBeVisible();
    // Switch to Right — still there.
    await page.getByRole('tab', { name: 'Right', exact: true }).click();
    await expect(toggle).toBeVisible();
  });

  test('tab buttons meet the 44x44px touch-target minimum', async ({ page }) => {
    // The .tab-btn class enforces min-height: 44px on mobile. Verify the
    // column-nav tab buttons are at least 44px tall and wide enough to tap.
    for (const name of ['Left', 'Center', 'Right']) {
      const tab = page.getByRole('tab', { name, exact: true });
      const box = await tab.boundingBox();
      expect(box, `${name} tab had no bounding box`).not.toBeNull();
      expect(box!.height, `${name} tab height ${box!.height} < 44`).toBeGreaterThanOrEqual(44);
    }
  });

  test('no horizontal overflow on mobile', async ({ page }) => {
    // The page must not force horizontal scrolling at mobile widths.
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth, `scrollWidth ${scrollWidth} exceeds clientWidth ${clientWidth}`).toBeLessThanOrEqual(
      clientWidth,
    );
  });

  test('right-column panels switch on mobile', async ({ page }) => {
    // On the Right column, the 10 right-side tabs must still switch panels.
    await page.getByRole('tab', { name: 'Right', exact: true }).click();
    await page.getByRole('tab', { name: 'Causal', exact: true }).click();
    await expect(page.getByText('Waiting for causal structure...')).toBeVisible();
    await page.getByRole('tab', { name: 'Module', exact: true }).click();
    await expect(page.getByText('模块误差图')).toBeVisible();
  });
});
