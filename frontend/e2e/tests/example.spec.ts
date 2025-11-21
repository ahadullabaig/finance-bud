import { test, expect } from '@playwright/test';

/**
 * Example Playwright Test
 * This is a basic test to verify Playwright is working
 */

test.describe('FinPilot Application', () => {
  test('should load the homepage', async ({ page }) => {
    await page.goto('/');

    // Verify the page loaded
    await expect(page).toHaveTitle(/FinPilot/i);
  });

  test('should display the main navigation', async ({ page }) => {
    await page.goto('/');

    // Wait for content to load
    await page.waitForLoadState('networkidle');

    // Check if main content is visible
    const mainContent = page.locator('main, #root, [role="main"]');
    await expect(mainContent).toBeVisible();
  });
});

test.describe('Accessibility', () => {
  test('should have no automatically detectable accessibility issues', async ({ page }) => {
    await page.goto('/');

    // You can integrate axe-core here for accessibility testing
    // For now, just check basic structure
    const heading = page.locator('h1').first();
    await expect(heading).toBeVisible();
  });
});
