// E2E: tab navigation.
// Verifies left-column tab switching (Sandbox ↔ Text) and that all ten
// right-column tabs switch and reveal their panel content.
import { test, expect } from '@playwright/test';

// [tab button label, distinctive content shown after switching]
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

test.beforeEach(async ({ page }) => {
  // Stub REST endpoints — no backend available in E2E.
  await page.route('**/knowledge-graph', (r) =>
    r.fulfill({ json: { nodes: [], edges: [] } }),
  );
  await page.route('**/story-milestones', (r) =>
    r.fulfill({ json: { milestones: [] } }),
  );
  await page.goto('/');
});

test.describe('Tab navigation', () => {
  test('switches left column between Sandbox and Text', async ({ page }) => {
    // Default tab is Sandbox.
    await expect(page.getByRole('tab', { name: 'Sandbox', exact: true })).toBeVisible();
    // Switch to Text — TextExplorer shows its empty-state placeholder.
    await page.getByRole('tab', { name: 'Text', exact: true }).click();
    await expect(page.getByText('Waiting for text stream...')).toBeVisible();
    // Switch back to Sandbox — the Text placeholder is removed.
    await page.getByRole('tab', { name: 'Sandbox', exact: true }).click();
    await expect(page.getByText('Waiting for text stream...')).toHaveCount(0);
  });

  test('switches through all 10 right-column tabs', async ({ page }) => {
    for (const [tabName, expectedContent] of RIGHT_TABS) {
      // click() auto-scrolls the horizontally-scrollable tab bar into view.
      await page.getByRole('tab', { name: tabName, exact: true }).click();
      await expect(page.getByText(expectedContent)).toBeVisible();
    }
  });
});
