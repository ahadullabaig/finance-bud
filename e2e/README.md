# FinPilot E2E Testing with Playwright

Comprehensive end-to-end testing suite for the FinPilot application using Playwright.

## 📋 Overview

This directory contains all E2E tests for FinPilot, including:
- UI/UX tests for the React frontend
- API endpoint testing
- User journey tests
- Performance tests
- Accessibility tests

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Playwright browsers installed

### Installation

```bash
# Install dependencies (from root directory)
npm install

# Install Playwright browsers
npx playwright install chromium firefox
```

## 🧪 Running Tests

### All Tests

```bash
# Run all tests headless
npm run test:e2e

# Run with UI mode (interactive)
npm run test:e2e:ui

# Run in headed mode (see browser)
npm run test:e2e:headed

# Run in debug mode
npm run test:e2e:debug
```

### Specific Tests

```bash
# Run a specific test file
npx playwright test e2e/tests/example.spec.ts

# Run tests matching a pattern
npx playwright test --grep "user journey"

# Run only API tests
npx playwright test e2e/tests/api.spec.ts
```

### By Browser

```bash
# Run on specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit

# Run on mobile
npx playwright test --project="Mobile Chrome"
```

## 📁 Directory Structure

```
e2e/
├── tests/              # Test files
│   ├── example.spec.ts     # Basic tests
│   ├── api.spec.ts         # API endpoint tests
│   └── user-journey.spec.ts # Complete user workflows
├── fixtures/           # Test data and fixtures
│   └── test-data.ts        # Reusable test data
├── utils/             # Helper utilities
│   └── helpers.ts          # Test helper functions
└── README.md          # This file
```

## ✍️ Writing Tests

### Basic Test Structure

```typescript
import { test, expect } from '@playwright/test';

test.describe('Feature Name', () => {
  test('should do something', async ({ page }) => {
    await page.goto('/');

    // Your test logic
    const element = page.locator('[data-testid="my-element"]');
    await expect(element).toBeVisible();
  });
});
```

### Using Test Data

```typescript
import { testPlans } from '../fixtures/test-data';

test('should create retirement plan', async ({ page }) => {
  const plan = testPlans.retirement;
  // Use plan data in your test
});
```

### Using Helpers

```typescript
import { waitForAPICall, expectVisible } from '../utils/helpers';

test('should fetch data', async ({ page }) => {
  await page.goto('/');

  const apiResponse = await waitForAPICall(page, '/api/v1/market');
  await expectVisible(page, '[data-testid="market-data"]');
});
```

## 🎯 Test Categories

### UI Tests (`example.spec.ts`)
- Component rendering
- User interactions
- Visual regression
- Responsive design

### API Tests (`api.spec.ts`)
- Endpoint availability
- Request/response validation
- Error handling
- Rate limiting

### User Journey Tests (`user-journey.spec.ts`)
- Complete workflows
- Multi-step processes
- Integration scenarios
- Performance metrics

## 📊 Test Reports

### View Reports

```bash
# Show HTML report
npm run test:e2e:report

# View JSON results
cat test-results/results.json

# View JUnit XML (for CI)
cat test-results/junit.xml
```

### Report Locations

- HTML Report: `playwright-report/index.html`
- JSON Results: `test-results/results.json`
- JUnit XML: `test-results/junit.xml`
- Screenshots: `test-results/<test-name>/screenshots/`
- Videos: `test-results/<test-name>/videos/`

## 🔧 Configuration

Playwright configuration is in `playwright.config.ts` at the root.

### Key Settings

- **Base URL**: `http://localhost:5173` (Vite dev server)
- **API URL**: `http://localhost:8000` (FastAPI backend)
- **Timeout**: 30 seconds per test
- **Retries**: 2 retries on CI, 0 locally
- **Browsers**: Chromium, Firefox, WebKit, Mobile Chrome, Mobile Safari

### Environment Variables

```bash
# Set base URL
BASE_URL=http://localhost:5173

# Set API URL
API_URL=http://localhost:8000

# Run in CI mode
CI=true
```

## 🐛 Debugging

### Debug Mode

```bash
# Run test in debug mode
npm run test:e2e:debug

# Debug specific test
npx playwright test --debug e2e/tests/example.spec.ts
```

### Playwright Inspector

The debug mode opens Playwright Inspector where you can:
- Step through tests
- Inspect page state
- View console logs
- See network requests

### Screenshots and Videos

Tests automatically capture:
- Screenshots on failure
- Videos on failure (retained)
- Traces on first retry

### View Traces

```bash
# View trace for failed test
npx playwright show-trace test-results/<test-name>/trace.zip
```

## 🎨 Code Generation

Generate tests by recording your actions:

```bash
# Start codegen
npm run test:e2e:codegen

# Or specify starting URL
npx playwright codegen http://localhost:5173
```

This opens a browser where your actions are recorded and converted to test code.

## 🏗️ Best Practices

### 1. Use Data Test IDs

```html
<button data-testid="create-plan-button">Create Plan</button>
```

```typescript
await page.click('[data-testid="create-plan-button"]');
```

### 2. Wait for Elements

```typescript
// Good
await page.waitForSelector('[data-testid="plan"]');

// Better
await expect(page.locator('[data-testid="plan"]')).toBeVisible();
```

### 3. Use Page Object Model

```typescript
class PlanPage {
  constructor(private page: Page) {}

  async createPlan(goal: string) {
    await this.page.fill('[name="goal"]', goal);
    await this.page.click('button[type="submit"]');
  }
}
```

### 4. Isolate Tests

Each test should be independent:
- Set up its own data
- Clean up after itself
- Not rely on other tests

### 5. Mock External APIs

```typescript
await page.route('**/api/external/**', route => {
  route.fulfill({
    status: 200,
    body: JSON.stringify({ data: 'mocked' })
  });
});
```

## 🔄 CI/CD Integration

Tests run automatically on:
- Push to `main` or `develop`
- Pull requests

### GitHub Actions

Workflow: `.github/workflows/playwright.yml`

### View Results

- Artifacts uploaded on every run
- Reports available for 30 days
- Test results in PR comments

## 📝 Adding New Tests

1. Create test file in `e2e/tests/`
2. Import necessary fixtures and helpers
3. Write tests following existing patterns
4. Run locally to verify
5. Commit and push

### Template

```typescript
import { test, expect } from '@playwright/test';
import { testData } from '../fixtures/test-data';
import { helpers } from '../utils/helpers';

test.describe('My Feature', () => {
  test.beforeEach(async ({ page }) => {
    // Setup
    await page.goto('/');
  });

  test('should work correctly', async ({ page }) => {
    // Test logic
  });

  test.afterEach(async ({ page }) => {
    // Cleanup
  });
});
```

## 🎓 Resources

- [Playwright Documentation](https://playwright.dev/)
- [Best Practices](https://playwright.dev/docs/best-practices)
- [API Reference](https://playwright.dev/docs/api/class-playwright)
- [Selectors Guide](https://playwright.dev/docs/selectors)

## 🆘 Troubleshooting

### Tests Timing Out

- Increase timeout in test: `test.setTimeout(60000)`
- Check network calls with DevTools
- Ensure services are running

### Flaky Tests

- Add proper waits
- Check for race conditions
- Use `waitForLoadState('networkidle')`

### Browser Not Installed

```bash
npx playwright install
```

### Tests Pass Locally But Fail on CI

- Check environment variables
- Verify service availability
- Review CI logs for errors

## 📞 Support

For issues or questions:
- Check existing tests for examples
- Review Playwright docs
- Ask the team in Slack #testing channel

---

**Happy Testing! 🎭**
