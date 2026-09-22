// E2E: basic app launch flow.
// Verifies the page renders its title, connection indicator, three-column
// layout, and the bottom Belief State panel — all without a backend.
import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  // The app talks to a backend on :8000 (REST) and :8765 (WebSocket).
  // There is no backend in the E2E environment, so we stub the REST
  // endpoints with empty payloads. The WebSocket is intentionally left
  // to fail — the page must degrade gracefully to "Disconnected".
  await page.route('**/knowledge-graph', (r) =>
    r.fulfill({ json: { nodes: [], edges: [] } }),
  );
  await page.route('**/story-milestones', (r) =>
    r.fulfill({ json: { milestones: [] } }),
  );
});

test.describe('App launch', () => {
  test('renders the main title and connection status indicator', async ({ page }) => {
    await page.goto('/');
    // The visible heading carries the app name.
    await expect(page.locator('h1')).toContainText('ZeroDataModel');
    await expect(page.locator('h1')).toContainText('Window of Consciousness');
    // Connection indicator exists (Connected or Disconnected).
    const status = page.locator('header span').first();
    await expect(status).toBeVisible();
    await expect(status).toHaveText(/Connected|Disconnected/);
  });

  test('shows the three-column layout', async ({ page }) => {
    await page.goto('/');
    // Left column: Sandbox / Text tab buttons.
    await expect(page.getByRole('tab', { name: 'Sandbox', exact: true })).toBeVisible();
    await expect(page.getByRole('tab', { name: 'Text', exact: true })).toBeVisible();
    // Center column: Free Energy chart panel title.
    await expect(page.getByText('Free Energy & Prediction Error')).toBeVisible();
    // Right column: right-side tab buttons (first one always in view).
    await expect(page.getByRole('tab', { name: 'Latent 3D', exact: true })).toBeVisible();
  });

  test('renders the bottom Belief State raw panel', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('Belief State (raw)')).toBeVisible();
  });
});
