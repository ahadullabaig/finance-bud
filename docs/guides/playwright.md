# 🎭 Playwright E2E Testing Setup - Complete Guide

**Status:** ✅ Installed and Configured
**Date:** 2025-11-18
**Version:** Playwright 1.56.1

---

## 📦 What's Been Installed

### 1. **Playwright Test Framework**
- `@playwright/test` v1.56.1
- `playwright` v1.56.1
- Installed as dev dependencies

### 2. **Browsers**
- ✅ **Chromium** 141.0.7390.37 (build 1194)
- ✅ **Firefox** (downloading)
- ⏳ **WebKit** (optional, can install with `npx playwright install webkit`)

### 3. **Supporting Tools**
- FFMPEG for video recording
- Chromium Headless Shell for CI/CD
- Trace viewer for debugging

---

## 📁 Project Structure Created

```
finance-bud/
├── playwright.config.ts           # Main Playwright configuration
├── e2e/                           # All E2E tests
│   ├── tests/
│   │   ├── example.spec.ts        # Basic UI tests
│   │   ├── api.spec.ts            # API endpoint tests
│   │   ├── user-journey.spec.ts   # Complete user workflows
│   │   └── backend-integration.spec.ts  # Backend agent tests
│   ├── fixtures/
│   │   └── test-data.ts           # Reusable test data
│   ├── utils/
│   │   └── helpers.ts             # Test helper functions
│   └── README.md                  # E2E testing documentation
├── .github/workflows/
│   └── playwright.yml             # CI/CD workflow for Playwright
└── package.json                   # Updated with test scripts
```

---

## 🚀 Available Test Scripts

Run these commands from the project root:

### Basic Testing
```bash
# Run all tests headless
npm run test:e2e

# Run with UI mode (interactive, visual)
npm run test:e2e:ui

# Run in headed mode (see browser)
npm run test:e2e:headed
```

### Debugging
```bash
# Debug mode with Playwright Inspector
npm run test:e2e:debug

# View HTML report
npm run test:e2e:report
```

### Code Generation
```bash
# Generate tests by recording actions
npm run test:e2e:codegen
```

### Advanced Usage
```bash
# Run specific test file
npx playwright test e2e/tests/api.spec.ts

# Run only tests matching a pattern
npx playwright test --grep "health check"

# Run on specific browser
npx playwright test --project=chromium

# Run with specific workers (parallelization)
npx playwright test --workers=4
```

---

## 📝 Test Files Created

### 1. **example.spec.ts** - Basic UI Tests
```typescript
- Homepage loading
- Navigation tests
- Accessibility checks
```

### 2. **api.spec.ts** - API Endpoint Tests
```typescript
- Health check endpoint
- Plan creation API
- Market data API
- Error handling
```

### 3. **user-journey.spec.ts** - User Workflows
```typescript
- Complete financial planning workflow
- Market data exploration
- Risk assessment questionnaire
- Performance benchmarks
```

### 4. **backend-integration.spec.ts** - Agent System Tests
```typescript
- Agent health monitoring
- Multi-agent communication
- IRA (Information Retrieval Agent) tests
- ML prediction engine tests
- Graph risk detector tests
- Conversational agent tests
- Error handling & resilience
- Performance benchmarks
```

---

## 🎯 Configuration Highlights

From `playwright.config.ts`:

### Browsers Configured
- ✅ Desktop Chrome (Chromium)
- ✅ Desktop Firefox
- ✅ Desktop Safari (WebKit)
- ✅ Mobile Chrome (Pixel 5)
- ✅ Mobile Safari (iPhone 12)

### Test Settings
- **Base URL:** `http://localhost:5173` (Vite dev server)
- **API URL:** `http://localhost:8000` (FastAPI backend)
- **Timeout:** 30 seconds per test
- **Retries:** 2 on CI, 0 locally
- **Parallel:** Yes (can be disabled on CI)

### Automatic Server Startup
Playwright will automatically start:
1. **Frontend:** `npm run dev` on port 5173
2. **Backend:** `uvicorn main:app` on port 8000

No need to start servers manually!

### Reports Generated
- **HTML Report:** `playwright-report/index.html`
- **JSON Results:** `test-results/results.json`
- **JUnit XML:** `test-results/junit.xml` (for CI)

### Screenshots & Videos
- **Screenshots:** Only on failure
- **Videos:** Retained on failure
- **Traces:** Captured on first retry

---

## ✅ How to Run Your First Test

### Step 1: Ensure Servers Can Start
```bash
# Make sure dependencies are installed
npm install

# Verify frontend can run
npm run dev
# (Stop with Ctrl+C after verifying)

# Verify backend can run
# python3 -m uvicorn main:app --reload
# (Stop with Ctrl+C after verifying)
```

### Step 2: Run Tests
```bash
# Run all tests (servers start automatically!)
npm run test:e2e
```

### Step 3: View Results
```bash
# Open HTML report
npm run test:e2e:report
```

---

## 🧪 Test Examples

### Simple UI Test
```typescript
test('should load homepage', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/FinPilot/i);
});
```

### API Test
```typescript
test('health check', async ({ request }) => {
  const response = await request.get('http://localhost:8000/health');
  expect(response.ok()).toBeTruthy();

  const data = await response.json();
  expect(data.status).toBe('healthy');
});
```

### User Journey Test
```typescript
test('create financial plan', async ({ page }) => {
  await page.goto('/');
  await page.click('[data-testid="create-plan"]');
  await page.fill('[name="goal"]', 'Retire with $1M');
  await page.click('button[type="submit"]');

  await expect(page.locator('[data-testid="plan"]')).toBeVisible();
});
```

---

## 🎨 Using Test Data & Helpers

### Test Data (Fixtures)
```typescript
import { testPlans, testUsers } from '../fixtures/test-data';

test('use test data', async ({ request }) => {
  const response = await request.post('/api/v1/plan', {
    data: testPlans.retirement  // Pre-configured test data
  });
});
```

### Helper Functions
```typescript
import { waitForAPICall, expectVisible } from '../utils/helpers';

test('use helpers', async ({ page }) => {
  await page.goto('/');

  const apiResponse = await waitForAPICall(page, '/api/v1/market');
  await expectVisible(page, '[data-testid="data"]');
});
```

---

## 🐛 Debugging Tips

### 1. Use UI Mode (Recommended)
```bash
npm run test:e2e:ui
```
This opens an interactive UI where you can:
- See each test
- Step through actions
- Inspect DOM
- View console logs
- See network requests

### 2. Use Debug Mode
```bash
npm run test:e2e:debug
```
Opens Playwright Inspector for step-by-step debugging.

### 3. Use Headed Mode
```bash
npm run test:e2e:headed
```
See the actual browser as tests run.

### 4. View Traces
When a test fails on first retry, a trace is captured:
```bash
npx playwright show-trace test-results/<test-name>/trace.zip
```

### 5. Check Screenshots
Failed tests automatically capture screenshots in:
```
test-results/<test-name>/test-failed-1.png
```

---

## 📊 CI/CD Integration

### GitHub Actions Workflow
File: `.github/workflows/playwright.yml`

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

**What It Does:**
1. Sets up Node.js 18
2. Sets up Python 3.11
3. Installs dependencies
4. Installs Playwright browsers
5. Runs all E2E tests
6. Uploads test reports as artifacts
7. Uploads test results

**View Reports:**
- Go to GitHub Actions
- Click on workflow run
- Download artifacts (playwright-report, test-results)

---

## 🎯 Test Coverage

### Currently Implemented

#### ✅ Basic Tests (example.spec.ts)
- Homepage loading ✓
- Navigation ✓
- Accessibility basics ✓

#### ✅ API Tests (api.spec.ts)
- Health check ✓
- Error handling ✓

#### ⏳ User Journey Tests (user-journey.spec.ts)
- Tests are written but marked as `.skip()`
- Will be enabled as features are implemented

#### ⏳ Backend Integration Tests (backend-integration.spec.ts)
- Comprehensive agent tests written
- Marked as `.skip()` until backend is ready
- Covers:
  - Agent communication
  - IRA (market data)
  - ML predictions
  - Risk detection
  - Conversational AI
  - Performance benchmarks

### Why `.skip()`?
Tests are pre-written for features being developed. As each feature is completed, simply remove `.skip()` to activate the test.

---

## 🔧 Customization

### Add New Test
1. Create file: `e2e/tests/my-feature.spec.ts`
2. Import Playwright: `import { test, expect } from '@playwright/test';`
3. Write tests:
```typescript
test.describe('My Feature', () => {
  test('should work', async ({ page }) => {
    // Your test
  });
});
```

### Add Test Data
Edit `e2e/fixtures/test-data.ts`:
```typescript
export const myTestData = {
  // Your test data
};
```

### Add Helper Function
Edit `e2e/utils/helpers.ts`:
```typescript
export async function myHelper(page: Page) {
  // Your helper logic
}
```

### Modify Configuration
Edit `playwright.config.ts`:
```typescript
export default defineConfig({
  // Your custom settings
});
```

---

## 📚 Resources

### Official Documentation
- [Playwright Docs](https://playwright.dev/)
- [Best Practices](https://playwright.dev/docs/best-practices)
- [API Reference](https://playwright.dev/docs/api/class-playwright)

### Project Documentation
- [E2E Testing Guide](./e2e/README.md)
- [Integration Roadmap](./INTEGRATION_ROADMAP.md)

### Quick Reference
```bash
# Common commands cheat sheet

# Run tests
npm run test:e2e              # All tests headless
npm run test:e2e:ui           # Interactive UI mode
npm run test:e2e:headed       # See browser
npm run test:e2e:debug        # Debug mode

# Specific tests
npx playwright test <file>    # Run one file
npx playwright test --grep <pattern>  # Pattern match
npx playwright test --project chromium  # One browser

# Reports
npm run test:e2e:report       # View HTML report
npx playwright show-trace <file>  # View trace

# Generate tests
npm run test:e2e:codegen      # Record actions

# Install
npx playwright install        # Install all browsers
npx playwright install chromium  # Install one browser
```

---

## 🎉 What You Can Do Now

### 1. Run Basic Tests
```bash
npm run test:e2e
```
Currently, only basic tests will run (health check, homepage).

### 2. Explore UI Mode
```bash
npm run test:e2e:ui
```
See the interactive testing interface.

### 3. Generate Tests by Recording
```bash
npm run test:e2e:codegen
```
Record your actions and generate test code.

### 4. Enable Tests as Features Complete
Remove `.skip()` from tests in:
- `user-journey.spec.ts`
- `backend-integration.spec.ts`

### 5. Write Custom Tests
Create new test files for your specific features.

---

## ⚠️ Important Notes

### Current Status
- ✅ Playwright installed
- ✅ Configuration complete
- ✅ Test structure created
- ✅ CI/CD workflow added
- ⏳ Most tests are `.skip()` until backend features are ready

### Before Running Full Test Suite
1. Ensure backend API is running (`python3 -m uvicorn main:app`)
2. Ensure frontend is running (`npm run dev`)
3. Or let Playwright start them automatically!

### Known Limitations
- Some system dependencies may not be available (OS not officially supported)
- Browsers downloaded as fallback builds (ubuntu20.04-x64)
- WebKit not installed yet (optional)

### Next Steps
1. Implement backend features
2. Remove `.skip()` from corresponding tests
3. Add more custom tests as needed
4. Configure MCP for advanced testing (optional)

---

## 🆘 Troubleshooting

### Tests Fail to Start
```bash
# Reinstall browsers
npx playwright install

# Check if servers can start
npm run dev  # Frontend
python3 -m uvicorn main:app  # Backend
```

### Timeout Errors
```typescript
// Increase timeout for specific test
test.setTimeout(60000);  // 60 seconds
```

### Flaky Tests
- Add proper waits: `await page.waitForLoadState('networkidle')`
- Use `expect` with timeout: `await expect(element).toBeVisible({ timeout: 10000 })`

### Browser Not Found
```bash
# Clear cache and reinstall
rm -rf ~/.cache/ms-playwright
npx playwright install
```

---

## 📈 Metrics & Reporting

### What Gets Measured
- Test pass/fail rate
- Test duration
- Screenshots on failure
- Videos on failure
- Traces on retry
- Network requests
- Console logs
- Performance metrics

### Where to Find Reports
- **Local:** `playwright-report/index.html`
- **CI/CD:** GitHub Actions artifacts
- **JSON:** `test-results/results.json`
- **JUnit:** `test-results/junit.xml`

---

## ✨ Summary

You now have a **professional-grade E2E testing setup** with:

✅ Playwright Test Framework installed
✅ Multiple browsers configured
✅ Comprehensive test suite structure
✅ Test data fixtures
✅ Helper utilities
✅ CI/CD integration
✅ Automatic server startup
✅ Multiple reporting formats
✅ Debug tools
✅ Code generation

**Ready to test!** 🚀

---

**Need Help?**
- Check [e2e/README.md](./e2e/README.md) for detailed testing guide
- Review [INTEGRATION_ROADMAP.md](./INTEGRATION_ROADMAP.md) for backend setup
- Visit [Playwright Docs](https://playwright.dev/) for framework documentation

**Happy Testing! 🎭**
