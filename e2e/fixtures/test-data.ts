/**
 * Test Data Fixtures for Playwright Tests
 * Reusable test data for E2E tests
 */

export const testUsers = {
  standard: {
    user_id: 'test_user_001',
    email: 'test@finpilot.com',
    name: 'Test User'
  },
  premium: {
    user_id: 'premium_user_001',
    email: 'premium@finpilot.com',
    name: 'Premium User'
  }
};

export const testPlans = {
  retirement: {
    user_id: 'test_user_001',
    user_goal: 'Retire comfortably in 20 years with $1,000,000',
    current_state: {
      total_assets: 100000,
      total_liabilities: 20000,
      monthly_income: 8000,
      monthly_expenses: 5000
    },
    risk_profile: {
      tolerance: 'moderate',
      level: 'MODERATE'
    },
    tax_considerations: {
      filing_status: 'single',
      rate: 0.25
    },
    time_horizon: 240 // 20 years in months
  },

  houseDownPayment: {
    user_id: 'test_user_001',
    user_goal: 'Save $50,000 for house down payment in 3 years',
    current_state: {
      total_assets: 10000,
      total_liabilities: 5000,
      monthly_income: 5000,
      monthly_expenses: 3000
    },
    risk_profile: {
      tolerance: 'conservative',
      level: 'LOW'
    },
    tax_considerations: {
      filing_status: 'single',
      rate: 0.22
    },
    time_horizon: 36 // 3 years
  },

  debtPayoff: {
    user_id: 'test_user_001',
    user_goal: 'Pay off $30,000 in student loans in 5 years',
    current_state: {
      total_assets: 5000,
      total_liabilities: 30000,
      monthly_income: 4500,
      monthly_expenses: 3200
    },
    risk_profile: {
      tolerance: 'conservative',
      level: 'LOW'
    },
    tax_considerations: {
      filing_status: 'single',
      rate: 0.22
    },
    time_horizon: 60 // 5 years
  }
};

export const testMarketData = {
  symbols: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],

  mockQuote: {
    symbol: 'AAPL',
    price: 175.50,
    change: 2.35,
    change_percent: 1.36,
    volume: 52340000,
    market_cap: 2750000000000,
    timestamp: new Date().toISOString()
  }
};

export const apiEndpoints = {
  health: '/health',
  plan: '/api/v1/plan',
  market: '/api/v1/market',
  risk: '/api/v1/risk',
  ml: '/api/v1/ml',
  chat: '/api/v1/chat'
};

export const testConstraints = {
  budget: {
    name: 'Monthly Budget Constraint',
    constraint_type: 'BUDGET',
    priority: 'HIGH',
    description: 'Do not exceed monthly spending budget',
    validation_rule: 'monthly_expenses <= budget_limit'
  },

  riskLimit: {
    name: 'Risk Exposure Limit',
    constraint_type: 'RISK',
    priority: 'MEDIUM',
    description: 'Maintain portfolio risk within acceptable range',
    validation_rule: 'portfolio_risk <= max_risk_threshold'
  }
};

export const mockResponses = {
  healthCheck: {
    status: 'healthy',
    agents: {
      orchestration: { status: 'healthy', uptime: 3600 },
      planning: { status: 'healthy', uptime: 3600 },
      ira: { status: 'healthy', uptime: 3600 },
      verification: { status: 'healthy', uptime: 3600 },
      execution: { status: 'healthy', uptime: 3600 }
    }
  },

  financialPlan: {
    plan_id: 'plan_001',
    user_id: 'test_user_001',
    status: 'APPROVED',
    steps: [
      {
        step_id: 'step_001',
        sequence_number: 1,
        action_type: 'INVESTMENT',
        description: 'Invest in diversified index fund',
        estimated_impact: 15000
      },
      {
        step_id: 'step_002',
        sequence_number: 2,
        action_type: 'SAVINGS',
        description: 'Increase monthly savings by $500',
        estimated_impact: 6000
      }
    ]
  }
};
