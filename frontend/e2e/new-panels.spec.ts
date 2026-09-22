// E2E: the five new panels (ModuleGraphPanel, ConfidenceDashboard,
// CommunicationLog, ExperimentLog, LogicPanel).
// Without a backend they must render their empty state and never throw.
import { test, expect } from '@playwright/test';

// [tab button label, panel title shown on render]
const NEW_PANELS: Array<[string, string]> = [
  ['Module', '模块误差图'],
  ['Confidence', '置信度仪表盘'],
  ['Comm', '通信日志'],
  ['Exp', '实验日志'],
  ['Logic', '逻辑约束'],
];

// Collected per-test; reset in beforeEach.
let pageErrors: string[] = [];

test.beforeEach(async ({ page }) => {
  pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));
  // Stub REST endpoints — no backend available in E2E.
  await page.route('**/knowledge-graph', (r) =>
    r.fulfill({ json: { nodes: [], edges: [] } }),
  );
  await page.route('**/story-milestones', (r) =>
    r.fulfill({ json: { milestones: [] } }),
  );
  await page.goto('/');
});

test.describe('New panels render without backend', () => {
  for (const [tabName, title] of NEW_PANELS) {
    test(`${tabName} panel renders with empty state and no JS error`, async ({ page }) => {
      await page.getByRole('tab', { name: tabName, exact: true }).click();
      // Panel title (header) is visible.
      await expect(page.getByText(title)).toBeVisible();
      // No snapshot → empty-state placeholder shown.
      await expect(page.getByText('等待数据...')).toBeVisible();
      // No uncaught exceptions were thrown while rendering.
      expect(pageErrors, `uncaught page errors: ${pageErrors.join('; ')}`).toEqual([]);
    });
  }
});
