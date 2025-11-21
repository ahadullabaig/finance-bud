# FinPilot Test Suite

Comprehensive test suite for the FinPilot VP-MAS system, organized by test type.

## Directory Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_agents.py              # Core agent functionality tests
│   ├── test_planning_agent.py      # Planning agent unit tests
│   ├── test_retriever.py           # Information retrieval agent tests
│   ├── test_verification.py        # Verification agent tests
│   ├── test_verification_accuracy.py  # Verification accuracy tests
│   ├── test_ml_prediction_engine.py   # ML prediction engine tests
│   └── domain/                     # Domain-specific unit tests
│
├── integration/             # Integration tests
│   ├── test_integration.py         # General integration tests
│   ├── test_communication_framework.py  # Agent communication tests
│   └── test_cmvl_advanced.py       # CMVL workflow integration tests
│
├── performance/             # Performance and benchmark tests
│   ├── test_performance.py         # Performance benchmarks
│   └── test_performance_realtime.py   # Real-time performance tests
│
├── api/                     # API endpoint tests
│   ├── test_ml_api_endpoints.py    # ML API endpoint tests
│   └── test_external_apis.py       # External API integration tests
│
├── scenarios/               # End-to-end scenario tests
│   ├── test_demo_scenarios.py      # Demo scenario tests
│   ├── test_trigger_simulation.py  # Trigger simulation tests
│   └── test_constraint_violations.py  # Constraint violation tests
│
├── ui/                      # UI/Frontend tests
│   └── test_reason_graph_visual.py # Reason graph visualization tests
│
├── verifier/                # Verifier-specific tests
│   ├── test_canonical_vectors.py   # Canonical vector tests
│   └── canonical_vectors.json      # Test data for canonical vectors
│
├── contract/                # Contract/schema validation tests
│
├── mock_data.py            # Shared mock data for tests
└── __init__.py             # Test package initialization
```

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories

**Unit Tests:**
```bash
pytest tests/unit/ -v
```

**Integration Tests:**
```bash
pytest tests/integration/ -v
```

**Performance Tests:**
```bash
pytest tests/performance/ -v
```

**API Tests:**
```bash
pytest tests/api/ -v
```

**Scenario Tests:**
```bash
pytest tests/scenarios/ -v
```

### Run Tests by Marker

**Async Tests:**
```bash
pytest -m asyncio
```

**Integration Tests:**
```bash
pytest -m integration
```

**Performance Tests:**
```bash
pytest -m benchmark
```

### Test Coverage

Generate coverage report:
```bash
pytest tests/ --cov=. --cov-report=html
```

View coverage:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Guidelines

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Use mocks for external dependencies
- Fast execution (< 1 second per test)
- High coverage of edge cases

### Integration Tests (`tests/integration/`)
- Test multiple components together
- Test inter-agent communication
- Test workflow orchestration
- May use real dependencies (not external APIs)

### Performance Tests (`tests/performance/`)
- Benchmark critical operations
- Test scalability and throughput
- Use `pytest-benchmark` for consistent measurement
- Document performance baselines

### API Tests (`tests/api/`)
- Test API endpoint contracts
- Test request/response validation
- Test error handling
- May mock external service calls

### Scenario Tests (`tests/scenarios/`)
- End-to-end user scenarios
- Test complete workflows
- Test trigger detection and response
- Test constraint violations

## Continuous Integration

Tests are run automatically on:
- Pull request creation
- Commits to main branch
- Nightly builds (full test suite)

### CI Configuration

See `.github/workflows/` for GitHub Actions configuration.

## Writing New Tests

1. **Choose the right directory** based on test type
2. **Use descriptive test names** - `test_<feature>_<scenario>_<expected_result>`
3. **Use fixtures** from `conftest.py` for common setup
4. **Use mock data** from `mock_data.py` for consistency
5. **Add markers** (`@pytest.mark.asyncio`, `@pytest.mark.integration`, etc.)
6. **Document complex tests** with docstrings

### Example Test Structure

```python
import pytest
from tests.mock_data import create_mock_financial_goal

@pytest.mark.asyncio
async def test_planning_agent_generates_valid_plan():
    """Test that planning agent generates a valid financial plan."""
    # Arrange
    goal = create_mock_financial_goal()

    # Act
    result = await planning_agent.generate_plan(goal)

    # Assert
    assert result.success
    assert len(result.plan_steps) > 0
```

## Test Data

Shared test data and fixtures are located in:
- `mock_data.py` - Mock financial data, goals, plans
- `conftest.py` - Pytest fixtures and configuration
- `verifier/canonical_vectors.json` - Canonical test vectors

## Debugging Tests

**Run single test:**
```bash
pytest tests/unit/test_agents.py::test_specific_function -v
```

**Run with print output:**
```bash
pytest tests/unit/test_agents.py -v -s
```

**Run with pdb debugger:**
```bash
pytest tests/unit/test_agents.py --pdb
```

**Show test durations:**
```bash
pytest tests/ --durations=10
```

## Maintenance

- Review and update tests when adding new features
- Remove obsolete tests when features are removed
- Keep mock data synchronized with production schemas
- Update this README when adding new test categories
