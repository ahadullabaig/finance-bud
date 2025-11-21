import { test, expect } from '@playwright/test';

/**
 * End-to-End User Journey Tests
 * Tests complete user workflows through the FinPilot application
 */

test.describe('User Journey: Financial Planning', () => {
  test.skip('should complete full planning workflow', async ({ page }) => {
    // 1. Navigate to homepage
    await page.goto('/');
    await expect(page).toHaveTitle(/FinPilot/i);

    // 2. Click on "Create Plan" or similar button
    const createPlanButton = page.getByRole('button', { name: /create plan/i });
    await createPlanButton.click();

    // 3. Fill out planning form
    await page.fill('[name="goal"]', 'Retire in 20 years with $1,000,000');
    await page.fill('[name="current_savings"]', '50000');
    await page.fill('[name="monthly_income"]', '8000');
    await page.fill('[name="monthly_expenses"]', '5000');

    // 4. Select risk tolerance
    await page.selectOption('[name="risk_tolerance"]', 'moderate');

    // 5. Submit the form
    const submitButton = page.getByRole('button', { name: /generate plan/i });
    await submitButton.click();

    // 6. Wait for plan generation
    await page.waitForSelector('[data-testid="financial-plan"]', { timeout: 10000 });

    // 7. Verify plan is displayed
    const plan = page.locator('[data-testid="financial-plan"]');
    await expect(plan).toBeVisible();

    // 8. Verify plan contains steps
    const planSteps = page.locator('[data-testid="plan-step"]');
    await expect(planSteps).toHaveCount(expect.any(Number));

    // 9. Check if plan can be saved
    const saveButton = page.getByRole('button', { name: /save plan/i });
    await expect(saveButton).toBeEnabled();
  });
});

test.describe('User Journey: Market Data Exploration', () => {
  test.skip('should view market data for stocks', async ({ page }) => {
    await page.goto('/');

    // Navigate to market data section
    const marketLink = page.getByRole('link', { name: /market/i });
    await marketLink.click();

    // Search for a stock symbol
    const searchInput = page.getByPlaceholder(/search symbol/i);
    await searchInput.fill('AAPL');
    await searchInput.press('Enter');

    // Wait for market data to load
    await page.waitForSelector('[data-testid="market-data"]', { timeout: 5000 });

    // Verify market data is displayed
    const stockPrice = page.locator('[data-testid="stock-price"]');
    await expect(stockPrice).toBeVisible();

    // Verify historical chart is visible
    const chart = page.locator('[data-testid="price-chart"]');
    await expect(chart).toBeVisible();
  });
});

test.describe('User Journey: Risk Assessment', () => {
  test.skip('should complete risk assessment questionnaire', async ({ page }) => {
    await page.goto('/risk-assessment');

    // Question 1: Investment horizon
    await page.check('[name="investment_horizon"][value="long_term"]');
    await page.click('button:has-text("Next")');

    // Question 2: Risk comfort
    await page.check('[name="risk_comfort"][value="moderate"]');
    await page.click('button:has-text("Next")');

    // Question 3: Experience
    await page.check('[name="experience"][value="intermediate"]');
    await page.click('button:has-text("Complete Assessment")');

    // Wait for results
    await page.waitForSelector('[data-testid="risk-profile"]');

    // Verify risk profile is shown
    const riskProfile = page.locator('[data-testid="risk-profile"]');
    await expect(riskProfile).toContainText(/moderate/i);
  });
});

test.describe('Performance Tests', () => {
  test('should load homepage within 3 seconds', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');
    const loadTime = Date.now() - startTime;

    expect(loadTime).toBeLessThan(3000);
  });

  test.skip('should handle navigation smoothly', async ({ page }) => {
    await page.goto('/');

    // Measure navigation performance
    const navigationLinks = ['/', '/plans', '/market', '/about'];

    for (const link of navigationLinks) {
      const startTime = Date.now();
      await page.goto(link);
      const loadTime = Date.now() - startTime;

      expect(loadTime).toBeLessThan(2000);
    }
  });
});
