// E2E: automated accessibility (a11y) scans with @axe-core/playwright.
//
// Runs axe-core against the desktop layout, then again after switching to
// each lazy-loaded right-column panel and the left "Text" tab, so the
// scans cover every panel's rendered DOM (including Suspense-loaded chunks).
//
// The dashboard's visualizations (recharts, cytoscape, three.js) are wrapped
// in role="figure"/role="img" containers with descriptive aria-labels, and
// decorative legends are marked aria-hidden, so the charting libraries'
// internal <svg>/<canvas> subtrees do not produce axe violations.
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// Right-column tabs that lazy-load a panel worth scanning. Each entry pairs
// the tab's accessible button name with a distinctive selector that proves
// the panel actually mounted before we scan it.
const RIGHT_TABS: Array<[string, string]> = [
  ['Latent 3D', 'Waiting for latent data...'],
  ['Causal', 'Waiting for causal structure...'],
  ['KG', 'Waiting for knowledge graph...'],
  ['Authoring', 'Model will self-author content as it learns...'],
  ['Story', 'No milestones detected yet.'],
  ['Module', '模块误差图'],
  ['Confidence', '置信度仪表盘'],
  ['Comm', '通信日志'],
  ['Exp', '实验日志'],
  ['Logic', '逻辑约束'],
];

async function stubRest(page: import('@playwright/test').Page) {
  // No backend in E2E — stub the REST endpoints the app fetches on load.
  await page.route('**/knowledge-graph', (r) =>
    r.fulfill({ json: { nodes: [], edges: [] } }),
  );
  await page.route('**/story-milestones', (r) =>
    r.fulfill({ json: { milestones: [] } }),
  );
}

async function scan(page: import('@playwright/test').Page) {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  return results;
}

test.describe('Accessibility (axe-core)', () => {
  test.beforeEach(async ({ page }) => {
    await stubRest(page);
    // Force a desktop viewport so the three-column layout renders.
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/');
  });

  test('initial desktop layout has no axe violations', async ({ page }) => {
    // Wait for the header and connection indicator to settle.
    await expect(page.locator('h1')).toContainText('ZeroDataModel');
    const results = await scan(page);
    expect(results.violations, JSON.stringify(results.violations, null, 2)).toEqual([]);
  });

  test('connection status is an accessible live region', async ({ page }) => {
    const status = page.locator('header [role="status"]');
    await expect(status).toBeVisible();
    await expect(status).toHaveAttribute('aria-live', 'polite');
    await expect(status).toHaveAttribute('aria-label', /Connection status:/);
  });

  test('tab widgets follow the WAI-ARIA tabs pattern', async ({ page }) => {
    // The left-column tablist exposes tab/tabpanel roles with proper
    // labelling relationships.
    const leftTablist = page.getByRole('tablist', { name: 'Left column panels' });
    await expect(leftTablist).toBeVisible();
    const sandboxTab = page.getByRole('tab', { name: 'Sandbox', exact: true });
    await expect(sandboxTab).toHaveAttribute('aria-selected', 'true');
    // The active tab controls the visible tabpanel.
    await expect(sandboxTab).toHaveAttribute('aria-controls', 'left-panel-sandbox');
    const panel = page.locator('#left-panel-sandbox');
    await expect(panel).toHaveAttribute('role', 'tabpanel');
    await expect(panel).toHaveAttribute('aria-labelledby', 'left-tab-sandbox');
  });

  test('arrow keys move focus across left-column tabs', async ({ page }) => {
    const sandbox = page.getByRole('tab', { name: 'Sandbox', exact: true });
    const text = page.getByRole('tab', { name: 'Text', exact: true });
    await sandbox.focus();
    // ArrowRight should activate + focus the Text tab (roving tabindex).
    await page.keyboard.press('ArrowRight');
    await expect(text).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByText('Waiting for text stream...')).toBeVisible();
    // ArrowLeft returns to Sandbox.
    await page.keyboard.press('ArrowLeft');
    await expect(sandbox).toHaveAttribute('aria-selected', 'true');
  });

  for (const [tabName, waitFor] of RIGHT_TABS) {
    test(`${tabName} panel has no axe violations`, async ({ page }) => {
      await page.getByRole('tab', { name: tabName, exact: true }).click();
      await expect(page.getByText(waitFor)).toBeVisible();
      const results = await scan(page);
      expect(
        results.violations,
        `${tabName}: ${JSON.stringify(results.violations, null, 2)}`,
      ).toEqual([]);
    });
  }

  test('Text panel has no axe violations', async ({ page }) => {
    await page.getByRole('tab', { name: 'Text', exact: true }).click();
    await expect(page.getByText('Waiting for text stream...')).toBeVisible();
    const results = await scan(page);
    expect(results.violations, JSON.stringify(results.violations, null, 2)).toEqual([]);
  });

  test('Belief State panel is keyboard-operable and has no axe violations', async ({ page }) => {
    // The collapsible header is a real <button> with aria-expanded/aria-controls.
    const toggle = page.getByRole('button', { name: /Belief State \(raw\)/ });
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await toggle.click();
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    const results = await scan(page);
    expect(results.violations, JSON.stringify(results.violations, null, 2)).toEqual([]);
  });
});
