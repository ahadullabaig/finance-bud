import { Page, expect } from '@playwright/test';

/**
 * Helper utilities for Playwright tests
 */

/**
 * Wait for API call to complete
 */
export async function waitForAPICall(page: Page, urlPattern: string | RegExp) {
  return page.waitForResponse(
    response => {
      const url = response.url();
      const matches = typeof urlPattern === 'string'
        ? url.includes(urlPattern)
        : urlPattern.test(url);
      return matches && response.status() === 200;
    },
    { timeout: 10000 }
  );
}

/**
 * Login helper (when authentication is implemented)
 */
export async function login(page: Page, email: string, password: string) {
  await page.goto('/login');
  await page.fill('[name="email"]', email);
  await page.fill('[name="password"]', password);
  await page.click('button[type="submit"]');

  // Wait for redirect or success indicator
  await page.waitForURL(/\/(dashboard|home)/);
}

/**
 * Create a financial plan through the UI
 */
export async function createFinancialPlan(page: Page, planData: any) {
  await page.goto('/plans/new');

  await page.fill('[name="goal"]', planData.user_goal);
  await page.fill('[name="current_savings"]', planData.current_state.total_assets.toString());
  await page.fill('[name="monthly_income"]', planData.current_state.monthly_income.toString());
  await page.fill('[name="monthly_expenses"]', planData.current_state.monthly_expenses.toString());

  if (planData.risk_profile?.tolerance) {
    await page.selectOption('[name="risk_tolerance"]', planData.risk_profile.tolerance);
  }

  await page.click('button[type="submit"]');

  // Wait for plan creation
  await page.waitForSelector('[data-testid="financial-plan"]', { timeout: 15000 });
}

/**
 * Check if element is visible with custom timeout
 */
export async function expectVisible(page: Page, selector: string, timeout = 5000) {
  await page.waitForSelector(selector, { state: 'visible', timeout });
  const element = page.locator(selector);
  await expect(element).toBeVisible();
}

/**
 * Take screenshot with timestamp
 */
export async function takeTimestampedScreenshot(page: Page, name: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  await page.screenshot({
    path: `screenshots/${name}-${timestamp}.png`,
    fullPage: true
  });
}

/**
 * Mock API response
 */
export async function mockAPIResponse(page: Page, url: string | RegExp, response: any) {
  await page.route(url, route => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(response)
    });
  });
}

/**
 * Check console for errors
 */
export async function checkConsoleErrors(page: Page) {
  const errors: string[] = [];

  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });

  return errors;
}

/**
 * Wait for network idle
 */
export async function waitForNetworkIdle(page: Page, timeout = 5000) {
  await page.waitForLoadState('networkidle', { timeout });
}

/**
 * Check accessibility violations
 * Requires @axe-core/playwright
 */
export async function checkA11y(page: Page) {
  // This is a placeholder - you'd need to install @axe-core/playwright
  // and import { injectAxe, checkA11y } from 'axe-playwright';

  // await injectAxe(page);
  // await checkA11y(page, null, {
  //   detailedReport: true,
  //   detailedReportOptions: {
  //     html: true
  //   }
  // });
}

/**
 * Fill form with retry
 */
export async function fillFormField(page: Page, selector: string, value: string, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      await page.fill(selector, value);
      const filledValue = await page.inputValue(selector);
      if (filledValue === value) {
        return;
      }
    } catch (error) {
      if (i === retries - 1) throw error;
      await page.waitForTimeout(1000);
    }
  }
}

/**
 * Generate test ID
 */
export function generateTestId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Format currency for testing
 */
export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(amount);
}

/**
 * Parse currency string to number
 */
export function parseCurrency(currencyString: string): number {
  return parseFloat(currencyString.replace(/[^0-9.-]+/g, ''));
}

/**
 * Generate random financial data for testing
 */
export function generateRandomFinancialData() {
  return {
    assets: Math.floor(Math.random() * 500000) + 10000,
    liabilities: Math.floor(Math.random() * 100000),
    income: Math.floor(Math.random() * 10000) + 3000,
    expenses: Math.floor(Math.random() * 5000) + 1500
  };
}
