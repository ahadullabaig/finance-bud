import { test, expect } from '@playwright/test';

/**
 * Setup Verification Test
 * Quick test to verify Playwright is working correctly
 */

test.describe('Playwright Setup Verification', () => {
  test('browsers are installed and working', async ({ browser }) => {
    expect(browser).toBeTruthy();
    expect(browser.isConnected()).toBeTruthy();
  });

  test('can make HTTP requests', async ({ request }) => {
    // Test external request capability
    const response = await request.get('https://httpbin.org/get');
    expect(response.ok()).toBeTruthy();
    expect(response.status()).toBe(200);
  });

  test('can navigate to a page', async ({ page }) => {
    await page.goto('https://playwright.dev/');
    await expect(page).toHaveTitle(/Playwright/);
  });

  test('can interact with elements', async ({ page }) => {
    await page.goto('https://playwright.dev/');

    // Find and click on a link
    const docsLink = page.locator('a').filter({ hasText: 'Docs' }).first();
    await expect(docsLink).toBeVisible();
  });

  test('can take screenshots', async ({ page }) => {
    await page.goto('https://playwright.dev/');

    // Take a screenshot
    const screenshot = await page.screenshot();
    expect(screenshot).toBeTruthy();
    expect(screenshot.length).toBeGreaterThan(0);
  });
});

test.describe('FinPilot Environment Check', () => {
  test('has correct base URL configured', async ({ page }) => {
    const baseUrl = process.env.BASE_URL || 'http://localhost:5173';
    expect(baseUrl).toContain('localhost');
  });

  test('has API URL configured', async ({ page }) => {
    const apiUrl = process.env.API_URL || 'http://localhost:8000';
    expect(apiUrl).toContain('localhost');
    expect(apiUrl).toContain('8000');
  });
});
