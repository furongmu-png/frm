// E2E: robustness without a backend.
// Verifies the page loads and tabs switch without uncaught JS errors,
// and that the connection indicator settles on "Disconnected".
import { test, expect } from '@playwright/test';

// Console errors that are expected when no backend is reachable:
//   - "[ws] error"        : logged by useModelStore on WebSocket failure
//   - "WebSocket connection to 'ws://localhost:8765' failed …"
//   - "Failed to fetch"   : any un-mocked REST attempt (caught by the store)
// Any console.error not matching this pattern is a real app error.
const EXPECTED_CONSOLE_ERR = /\[ws\]|\[api\]|WebSocket|8765|Failed to fetch|connection/i;

let pageErrors: string[] = [];
let consoleErrors: string[] = [];

test.beforeEach(async ({ page }) => {
  pageErrors = [];
  consoleErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  // Stub REST endpoints — no backend available in E2E.
  await page.route('**/knowledge-graph', (r) =>
    r.fulfill({ json: { nodes: [], edges: [] } }),
  );
  await page.route('**/story-milestones', (r) =>
    r.fulfill({ json: { milestones: [] } }),
  );
});

function assertNoErrors() {
  expect(pageErrors, `uncaught page errors: ${pageErrors.join('; ')}`).toEqual([]);
  const unexpected = consoleErrors.filter((e) => !EXPECTED_CONSOLE_ERR.test(e));
  expect(unexpected, `unexpected console errors: ${unexpected.join(' | ')}`).toEqual([]);
}

test.describe('No crash without backend', () => {
  test('page loads without JS errors', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('ZeroDataModel');
    // Let the WebSocket connection attempt settle (async failure).
    await page.waitForTimeout(500);
    assertNoErrors();
  });

  test('tab switching does not throw JS errors', async ({ page }) => {
    await page.goto('/');
    // Switch left tabs back and forth.
    await page.getByRole('tab', { name: 'Text', exact: true }).click();
    await page.getByRole('tab', { name: 'Sandbox', exact: true }).click();
    // Switch through every right-column tab.
    for (const name of [
      'Latent 3D', 'Causal', 'KG', 'Authoring', 'Story',
      'Module', 'Confidence', 'Comm', 'Exp', 'Logic',
    ]) {
      await page.getByRole('tab', { name, exact: true }).click();
    }
    // App header still present — no white screen from a render crash.
    await expect(page.locator('h1')).toContainText('ZeroDataModel');
    assertNoErrors();
  });

  test('shows Disconnected status after connection failure', async ({ page }) => {
    await page.goto('/');
    // Wait for the WebSocket attempt to fail and the store to settle.
    await page.waitForTimeout(500);
    const status = page.locator('header span').first();
    await expect(status).toContainText('Disconnected');
    assertNoErrors();
  });
});
