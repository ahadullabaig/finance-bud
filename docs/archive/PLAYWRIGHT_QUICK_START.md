# 🎭 Playwright Quick Start Guide

## ✅ Installation Complete!

Playwright has been successfully installed and configured for your FinPilot project.

---

## 📦 What's Installed

- ✅ **Playwright v1.56.1**
- ✅ **Chromium** 141.0.7390.37
- ✅ **Firefox** 142.0.1
- ✅ **FFMPEG** (for video recording)
- ✅ **7 Test Files** with comprehensive examples
- ✅ **Helper utilities** and test fixtures
- ✅ **CI/CD workflow** for GitHub Actions

---

## 🚀 Quick Commands

### Run Tests

```bash
# Run all tests (headless)
npm run test:e2e

# Run with interactive UI (recommended for development)
npm run test:e2e:ui

# Run and see browser
npm run test:e2e:headed

# Debug mode
npm run test:e2e:debug

# View last test report
npm run test:e2e:report

# Generate tests by recording actions
npm run test:e2e:codegen
```

### Run Specific Tests

```bash
# Run one test file
npx playwright test e2e/tests/example.spec.ts

# Run only tests matching pattern
npx playwright test --grep "health check"

# Run on specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox
```

---

## 📁 Test Files Created

### 1. **verify-setup.spec.ts** - Setup Verification
Tests that Playwright is working correctly:
- Browser connectivity
- HTTP requests
- Element interactions
- Screenshot capability

### 2. **example.spec.ts** - Basic UI Tests
- Homepage loading
- Navigation
- Accessibility checks

### 3. **api.spec.ts** - API Testing
- Health check endpoint
- Plan creation
- Market data
- Error handling

### 4. **user-journey.spec.ts** - User Workflows
- Complete financial planning flow
- Market data exploration
- Risk assessment
- Performance tests

### 5. **backend-integration.spec.ts** - Agent System Tests
Comprehensive backend testing:
- Agent health monitoring
- Multi-agent communication
- IRA (market data caching)
- ML prediction engine
- Graph risk detector
- Conversational agent
- Error resilience
- Performance benchmarks

---

## 🎯 Try It Now!

### Option 1: Run Verification Test (No servers needed)
```bash
npx playwright test e2e/tests/verify-setup.spec.ts --project=chromium
```
This tests Playwright itself against public websites.

### Option 2: Interactive UI Mode
```bash
npm run test:e2e:ui
```
This opens a visual interface where you can:
- See all tests
- Run tests individually
- See what's happening step-by-step
- Debug easily

### Option 3: Generate Tests by Recording
```bash
# Make sure your frontend is running first
npm run dev
# In another terminal:
npm run test:e2e:codegen
```
- A browser opens
- You interact with your app
- Playwright generates test code automatically
- Copy the generated code to your test files

---

## 📝 Test Data Available

Pre-configured test data in `e2e/fixtures/test-data.ts`:

```typescript
import { testPlans, testUsers, testMarketData } from '../fixtures/test-data';

// Use in tests
test('create plan', async ({ request }) => {
  const response = await request.post('/api/v1/plan', {
    data: testPlans.retirement  // Pre-built test plan
  });
});
```

**Available Data:**
- `testUsers` - Standard and premium users
- `testPlans` - Retirement, house down payment, debt payoff
- `testMarketData` - Stock symbols and mock quotes
- `testConstraints` - Budget and risk constraints
- `mockResponses` - Expected API responses

---

## 🛠️ Helper Functions Available

In `e2e/utils/helpers.ts`:

```typescript
import {
  waitForAPICall,
  expectVisible,
  mockAPIResponse,
  takeTimestampedScreenshot
} from '../utils/helpers';

// Wait for specific API call
await waitForAPICall(page, '/api/v1/market');

// Check element visible with timeout
await expectVisible(page, '[data-testid="plan"]', 5000);

// Mock API response
await mockAPIResponse(page, /\/api\/market/, mockData);

// Take screenshot with timestamp
await takeTimestampedScreenshot(page, 'test-result');
```

---

## 🎨 Writing Your First Test

Create `e2e/tests/my-test.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';

test.describe('My Feature', () => {
  test('should work', async ({ page }) => {
    // Navigate to page
    await page.goto('/');

    // Interact with elements
    await page.click('[data-testid="button"]');
    await page.fill('input[name="name"]', 'Test User');

    // Make assertions
    await expect(page.locator('.result')).toBeVisible();
    await expect(page.locator('.result')).toContainText('Success');
  });
});
```

---

## 🔍 Finding Elements

### Recommended Selectors (in order of preference):

```typescript
// 1. Test ID (best - most stable)
page.locator('[data-testid="create-plan"]')

// 2. Role (good for accessibility)
page.getByRole('button', { name: 'Create Plan' })

// 3. Text content
page.getByText('Create Plan')

// 4. Label (for form inputs)
page.getByLabel('Email address')

// 5. Placeholder
page.getByPlaceholder('Enter your email')

// 6. CSS selector (least stable)
page.locator('.btn-primary')
```

---

## 🐛 Debugging

### When Test Fails:

1. **Check Screenshots**
   ```
   test-results/<test-name>/test-failed-1.png
   ```

2. **View Trace**
   ```bash
   npx playwright show-trace test-results/<test-name>/trace.zip
   ```

3. **Run in Debug Mode**
   ```bash
   npm run test:e2e:debug
   ```

4. **Use UI Mode**
   ```bash
   npm run test:e2e:ui
   ```

---

## 📊 CI/CD Integration

Workflow: `.github/workflows/playwright.yml`

**Runs on:**
- Push to `main` or `develop`
- Pull requests

**What it does:**
- Starts both frontend and backend servers
- Runs all E2E tests
- Uploads reports as artifacts
- Fails build if tests fail

**View results:**
- Go to GitHub Actions tab
- Click on workflow run
- Download artifacts (playwright-report, test-results)

---

## 🎯 What Tests Are Ready

### ✅ Ready to Run Now
- Setup verification tests (no server needed)
- Example tests (basic structure)

### ⏳ Waiting for Backend
Most tests are marked with `.skip()` because they require:
- Backend API running
- Agent system initialized
- Database configured
- External APIs connected

**To enable tests:**
1. Implement the feature
2. Remove `.skip()` from the test
3. Run the test

Example:
```typescript
// Currently:
test.skip('should create plan', async ({ request }) => {
  // Test code
});

// After implementation:
test('should create plan', async ({ request }) => {
  // Test code runs!
});
```

---

## 💡 Tips & Best Practices

### 1. Use Test IDs in Your Components
```tsx
<button data-testid="create-plan-button">Create Plan</button>
```

### 2. Wait for Elements Properly
```typescript
// ❌ Bad
await page.click('button');

// ✅ Good
await page.waitForSelector('button');
await page.click('button');

// ✅ Better
await expect(page.locator('button')).toBeVisible();
await page.click('button');
```

### 3. Use Page Object Model
```typescript
class PlanPage {
  constructor(private page: Page) {}

  async createPlan(goal: string) {
    await this.page.fill('[name="goal"]', goal);
    await this.page.click('[data-testid="submit"]');
  }

  async getPlanId() {
    return this.page.locator('[data-testid="plan-id"]').textContent();
  }
}

// Use in test
const planPage = new PlanPage(page);
await planPage.createPlan('Retire with $1M');
```

### 4. Mock External Dependencies
```typescript
await page.route('**/api/external/**', route => {
  route.fulfill({
    status: 200,
    body: JSON.stringify({ data: 'mocked' })
  });
});
```

---

## 📚 Documentation

- **Local Docs:** See [e2e/README.md](./e2e/README.md)
- **Full Setup:** See [PLAYWRIGHT_SETUP.md](./PLAYWRIGHT_SETUP.md)
- **Integration Plan:** See [INTEGRATION_ROADMAP.md](./INTEGRATION_ROADMAP.md)
- **Official Docs:** [playwright.dev](https://playwright.dev/)

---

## 🎉 You're All Set!

Your Playwright testing environment is ready to use. Start by:

1. Running the verification test
2. Exploring UI mode
3. Recording some tests with codegen
4. Writing your own tests as you build features

**Happy Testing! 🚀**
