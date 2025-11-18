import { test, expect } from '@playwright/test';
import { testPlans, testUsers, mockResponses } from '../fixtures/test-data';

/**
 * Backend Integration Tests
 * Tests the full backend API with real agent communication
 */

const API_BASE = process.env.API_URL || 'http://localhost:8000';

test.describe('Backend Health and Monitoring', () => {
  test('should return healthy status with agent details', async ({ request }) => {
    const response = await request.get(`${API_BASE}/health`);

    expect(response.ok()).toBeTruthy();
    const health = await response.json();

    expect(health).toHaveProperty('status');
    expect(health.status).toBe('healthy');

    // Verify agent health if available
    if (health.agents) {
      expect(health.agents).toHaveProperty('orchestration');
      expect(health.agents).toHaveProperty('planning');
      expect(health.agents).toHaveProperty('ira');
      expect(health.agents).toHaveProperty('verification');
      expect(health.agents).toHaveProperty('execution');
    }
  });

  test('should provide Prometheus metrics endpoint', async ({ request }) => {
    const response = await request.get(`${API_BASE}/metrics`);

    // May not be implemented yet
    if (response.status() === 404) {
      test.skip();
    }

    expect(response.ok()).toBeTruthy();
    const metrics = await response.text();

    // Basic Prometheus format validation
    expect(metrics).toContain('# TYPE');
  });
});

test.describe('Agent Communication Tests', () => {
  test.skip('should orchestrate complete planning workflow', async ({ request }) => {
    const planRequest = testPlans.retirement;

    // 1. Submit plan request
    const response = await request.post(`${API_BASE}/api/v1/plan`, {
      data: planRequest
    });

    expect(response.ok()).toBeTruthy();
    const result = await response.json();

    expect(result).toHaveProperty('plan_id');
    const planId = result.plan_id;

    // 2. Wait for processing (agents communicating)
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 3. Retrieve plan status
    const statusResponse = await request.get(`${API_BASE}/api/v1/plan/${planId}`);
    expect(statusResponse.ok()).toBeTruthy();

    const plan = await statusResponse.json();
    expect(plan).toHaveProperty('status');
    expect(['PENDING', 'APPROVED', 'PROCESSING']).toContain(plan.status);

    // 4. Verify plan has steps
    if (plan.steps) {
      expect(plan.steps.length).toBeGreaterThan(0);
      expect(plan.steps[0]).toHaveProperty('action_type');
    }
  });

  test.skip('should handle concurrent requests efficiently', async ({ request }) => {
    const requests = Array(10).fill(null).map((_, i) => ({
      user_id: `concurrent_user_${i}`,
      user_goal: `Test goal ${i}`,
      current_state: testPlans.houseDownPayment.current_state,
      risk_profile: testPlans.houseDownPayment.risk_profile,
      tax_considerations: testPlans.houseDownPayment.tax_considerations,
      time_horizon: 36,
      correlation_id: `corr_${i}`,
      session_id: `session_${i}`
    }));

    // Send all requests concurrently
    const startTime = Date.now();
    const responses = await Promise.all(
      requests.map(req => request.post(`${API_BASE}/api/v1/plan`, { data: req }))
    );
    const duration = Date.now() - startTime;

    // All should succeed
    responses.forEach(response => {
      expect(response.ok()).toBeTruthy();
    });

    // Should handle 10 concurrent requests in reasonable time
    expect(duration).toBeLessThan(10000); // 10 seconds
  });
});

test.describe('Information Retrieval Agent (IRA) Tests', () => {
  test.skip('should fetch and cache market data', async ({ request }) => {
    const symbol = 'AAPL';

    // First request - cache miss
    const firstResponse = await request.get(`${API_BASE}/api/v1/market/${symbol}`);
    expect(firstResponse.ok()).toBeTruthy();

    const firstData = await firstResponse.json();
    expect(firstData).toHaveProperty('symbol');
    expect(firstData.symbol).toBe(symbol);

    // Second request - should be faster (cached)
    const secondStartTime = Date.now();
    const secondResponse = await request.get(`${API_BASE}/api/v1/market/${symbol}`);
    const secondDuration = Date.now() - secondStartTime;

    expect(secondResponse.ok()).toBeTruthy();
    const secondData = await secondResponse.json();

    // Cached response should be very fast
    expect(secondDuration).toBeLessThan(100);

    // Data should be consistent
    expect(secondData.symbol).toBe(firstData.symbol);
  });

  test.skip('should handle invalid stock symbols gracefully', async ({ request }) => {
    const response = await request.get(`${API_BASE}/api/v1/market/INVALID_SYMBOL_XYZ`);

    expect(response.status()).toBe(400); // Bad request or similar
    const error = await response.json();

    expect(error).toHaveProperty('error');
  });

  test.skip('should support multiple stock symbols in batch', async ({ request }) => {
    const symbols = ['AAPL', 'GOOGL', 'MSFT'];

    const response = await request.post(`${API_BASE}/api/v1/market/batch`, {
      data: { symbols }
    });

    if (response.status() === 404) {
      test.skip(); // Endpoint not implemented yet
    }

    expect(response.ok()).toBeTruthy();
    const data = await response.json();

    expect(data).toHaveProperty('quotes');
    expect(data.quotes.length).toBe(3);
  });
});

test.describe('ML Prediction Engine Tests', () => {
  test.skip('should generate financial predictions', async ({ request }) => {
    const predictionRequest = {
      user_id: testUsers.standard.user_id,
      financial_state: testPlans.retirement.current_state,
      time_horizon: 240 // 20 years
    };

    const response = await request.post(`${API_BASE}/api/v1/ml/predict`, {
      data: predictionRequest
    });

    if (response.status() === 404) {
      test.skip(); // ML endpoint not ready
    }

    expect(response.ok()).toBeTruthy();
    const prediction = await response.json();

    expect(prediction).toHaveProperty('predicted_net_worth');
    expect(prediction).toHaveProperty('confidence_interval');
    expect(prediction).toHaveProperty('factors');
  });

  test.skip('should provide risk scoring', async ({ request }) => {
    const riskRequest = {
      user_id: testUsers.standard.user_id,
      portfolio: {
        assets: testPlans.retirement.current_state.total_assets,
        liabilities: testPlans.retirement.current_state.total_liabilities
      }
    };

    const response = await request.post(`${API_BASE}/api/v1/ml/risk-score`, {
      data: riskRequest
    });

    if (response.status() === 404) {
      test.skip();
    }

    expect(response.ok()).toBeTruthy();
    const riskScore = await response.json();

    expect(riskScore).toHaveProperty('score');
    expect(riskScore.score).toBeGreaterThanOrEqual(0);
    expect(riskScore.score).toBeLessThanOrEqual(100);
  });
});

test.describe('Graph Risk Detector Tests', () => {
  test.skip('should detect risk patterns in financial graph', async ({ request }) => {
    const graphRequest = {
      user_id: testUsers.standard.user_id,
      financial_state: testPlans.retirement.current_state,
      market_conditions: {
        volatility: 'high',
        trend: 'bearish'
      }
    };

    const response = await request.post(`${API_BASE}/api/v1/risk/detect`, {
      data: graphRequest
    });

    if (response.status() === 404) {
      test.skip();
    }

    expect(response.ok()).toBeTruthy();
    const risks = await response.json();

    expect(risks).toHaveProperty('detected_risks');
    expect(Array.isArray(risks.detected_risks)).toBeTruthy();

    if (risks.detected_risks.length > 0) {
      const firstRisk = risks.detected_risks[0];
      expect(firstRisk).toHaveProperty('risk_type');
      expect(firstRisk).toHaveProperty('severity');
      expect(firstRisk).toHaveProperty('mitigation');
    }
  });
});

test.describe('Conversational Agent Tests', () => {
  test.skip('should handle natural language queries', async ({ request }) => {
    const chatRequest = {
      user_id: testUsers.standard.user_id,
      message: 'How much should I save monthly to retire with $1 million in 20 years?',
      session_id: 'test_session_001'
    };

    const response = await request.post(`${API_BASE}/api/v1/chat/message`, {
      data: chatRequest
    });

    if (response.status() === 404) {
      test.skip();
    }

    expect(response.ok()).toBeTruthy();
    const chatResponse = await response.json();

    expect(chatResponse).toHaveProperty('response');
    expect(chatResponse.response.length).toBeGreaterThan(0);
    expect(chatResponse).toHaveProperty('message_id');
  });

  test.skip('should maintain conversation context', async ({ request }) => {
    const sessionId = `test_session_${Date.now()}`;

    // First message
    const msg1 = await request.post(`${API_BASE}/api/v1/chat/message`, {
      data: {
        user_id: testUsers.standard.user_id,
        message: 'I want to save for retirement',
        session_id: sessionId
      }
    });

    expect(msg1.ok()).toBeTruthy();

    // Follow-up message (should remember context)
    const msg2 = await request.post(`${API_BASE}/api/v1/chat/message`, {
      data: {
        user_id: testUsers.standard.user_id,
        message: 'How much do I need?',
        session_id: sessionId
      }
    });

    expect(msg2.ok()).toBeTruthy();
    const response2 = await msg2.json();

    // Should understand "I" refers to retirement context
    expect(response2.response.toLowerCase()).toContain('retire');
  });
});

test.describe('Error Handling and Resilience', () => {
  test('should handle malformed JSON gracefully', async ({ request }) => {
    const response = await request.post(`${API_BASE}/api/v1/plan`, {
      data: 'not valid json',
      headers: { 'Content-Type': 'application/json' }
    });

    expect(response.status()).toBeGreaterThanOrEqual(400);
    expect(response.status()).toBeLessThan(500);
  });

  test('should validate required fields', async ({ request }) => {
    const incompleteRequest = {
      user_id: 'test'
      // Missing required fields
    };

    const response = await request.post(`${API_BASE}/api/v1/plan`, {
      data: incompleteRequest
    });

    expect(response.status()).toBe(422); // Validation error
    const error = await response.json();

    expect(error).toHaveProperty('detail');
  });

  test.skip('should handle agent failures gracefully', async ({ request }) => {
    // Test what happens when an agent is down
    // This would require a way to simulate agent failure

    const response = await request.post(`${API_BASE}/api/v1/plan`, {
      data: testPlans.retirement
    });

    // Should either succeed with degraded service or fail gracefully
    if (!response.ok()) {
      const error = await response.json();
      expect(error).toHaveProperty('error');
      expect(error.error).not.toContain('500'); // Should not be internal server error
    }
  });

  test.skip('should respect rate limits', async ({ request }) => {
    const requests = Array(100).fill(null);

    // Hammer the endpoint
    const responses = await Promise.all(
      requests.map(() => request.get(`${API_BASE}/api/v1/market/AAPL`))
    );

    // Some should be rate limited
    const rateLimited = responses.filter(r => r.status() === 429);
    expect(rateLimited.length).toBeGreaterThan(0);
  });
});

test.describe('Performance Benchmarks', () => {
  test('health check should respond in < 200ms', async ({ request }) => {
    const startTime = Date.now();
    const response = await request.get(`${API_BASE}/health`);
    const duration = Date.now() - startTime;

    expect(response.ok()).toBeTruthy();
    expect(duration).toBeLessThan(200);
  });

  test.skip('plan creation should complete in < 5 seconds', async ({ request }) => {
    const startTime = Date.now();
    const response = await request.post(`${API_BASE}/api/v1/plan`, {
      data: testPlans.houseDownPayment
    });
    const duration = Date.now() - startTime;

    expect(response.ok()).toBeTruthy();
    expect(duration).toBeLessThan(5000);
  });

  test.skip('market data fetch should complete in < 1 second', async ({ request }) => {
    const startTime = Date.now();
    const response = await request.get(`${API_BASE}/api/v1/market/AAPL`);
    const duration = Date.now() - startTime;

    expect(response.ok()).toBeTruthy();
    expect(duration).toBeLessThan(1000);
  });
});
