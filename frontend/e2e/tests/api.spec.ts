import { test, expect } from '@playwright/test';

/**
 * API Testing with Playwright
 * Tests the backend API endpoints
 */

const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

test.describe('API Health Checks', () => {
  test('should return healthy status from health endpoint', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/health`);

    expect(response.ok()).toBeTruthy();
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty('status');
    expect(data.status).toBe('healthy');
  });
});

test.describe('API Core Endpoints', () => {
  test.skip('should create a financial plan', async ({ request }) => {
    // Skip until backend is fully set up
    const planRequest = {
      user_id: 'test_user_001',
      user_goal: 'Save $50,000 for house down payment',
      current_state: {
        balance: 10000,
        monthly_income: 5000,
        monthly_expenses: 3000
      },
      risk_profile: {
        tolerance: 'moderate'
      },
      tax_considerations: {
        rate: 0.25
      },
      time_horizon: 36, // 3 years
      correlation_id: 'test_corr_001',
      session_id: 'test_session_001'
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/plan`, {
      data: planRequest,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data).toHaveProperty('plan_id');
  });

  test.skip('should retrieve market data', async ({ request }) => {
    const symbol = 'AAPL';
    const response = await request.get(`${API_BASE_URL}/api/v1/market/${symbol}`);

    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data).toHaveProperty('symbol');
    expect(data.symbol).toBe(symbol);
  });
});

test.describe('API Error Handling', () => {
  test('should return 404 for non-existent endpoint', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/v1/nonexistent`);
    expect(response.status()).toBe(404);
  });

  test.skip('should return 400 for invalid plan request', async ({ request }) => {
    const invalidRequest = {
      // Missing required fields
      user_id: 'test_user'
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/plan`, {
      data: invalidRequest
    });

    expect(response.status()).toBe(400);
  });
});
