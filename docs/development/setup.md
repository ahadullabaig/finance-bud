# FinPilot VP-MAS Development Environment Setup

## Overview

This document provides comprehensive setup instructions for the FinPilot Verifiable Planning Multi-Agent System (VP-MAS) development environment, including mock interfaces, testing frameworks, and CI/CD pipeline configuration.

## Requirements Covered

- **9.4**: Mock agent interfaces for independent development
- **9.5**: Testing frameworks and CI/CD pipeline structure  
- **11.2**: API endpoint contracts and documentation standards
- **11.2**: Logging and monitoring standards across all agents

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd finpilot-vp-mas

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .
```

### 2. Run Basic Tests

```bash
# Run data model validation tests
python data_models/test_schemas.py

# Run agent tests with pytest
pytest tests/test_agents.py -v

# Run integration tests
pytest tests/test_integration.py -v

# Run all tests with coverage
pytest --cov=agents --cov=data_models --cov-report=html
```

### 3. Start Mock Agents

```python
# Example: Start mock orchestration agent
from agents.mock_interfaces import MockOrchestrationAgent
from agents.communication import AgentCommunicationFramework

# Create communication framework
framework = AgentCommunicationFramework()

# Create and register mock agent
agent = MockOrchestrationAgent()
framework.register_agent(agent, ['workflow_management'])

# Start agent
await agent.start()
```

## Development Architecture

### 4-Person Parallel Development Structure

The system is designed for 4 developers working independently:

- **Person A**: Orchestration Agent + Communication Framework
- **Person B**: Information Retrieval Agent + Market Data
- **Person C**: Planning Agent + Guided Search Module  
- **Person D**: Verification Agent + ReasonGraph Visualization

### Mock Interface System

Each developer can work independently using comprehensive mock interfaces:

```python
# Mock agents provide realistic behavior
mock_oa = MockOrchestrationAgent()
mock_pa = MockPlanningAgent() 
mock_ira = MockInformationRetrievalAgent()
mock_va = MockVerificationAgent()
mock_ea = MockExecutionAgent()

# All agents support the same communication protocols
message = AgentMessage(
    agent_id="test_sender",
    target_agent_id=mock_pa.agent_id,
    message_type=MessageType.REQUEST,
    payload={"planning_request": {...}},
    correlation_id="test_correlation",
    session_id="test_session",
    trace_id="test_trace"
)

response = await mock_pa.process_message(message)
```

## Testing Framework

### Test Categories

1. **Unit Tests** (`tests/test_agents.py`)
   - Individual agent functionality
   - Data model validation
   - Mock interface behavior

2. **Integration Tests** (`tests/test_integration.py`)
   - Agent communication protocols
   - End-to-end workflows
   - CMVL trigger scenarios

3. **Performance Tests** (`tests/test_performance.py`)
   - Load testing
   - Benchmark measurements
   - Resource utilization

### Running Tests

```bash
# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m performance           # Performance tests only

# Run tests with specific coverage
pytest --cov=agents tests/test_agents.py

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto

# Generate HTML coverage report
pytest --cov=agents --cov-report=html
```

### Mock Data Generation

```python
from tests.mock_data import MockDataGenerator

# Generate realistic test data
generator = MockDataGenerator(seed=42)

# Generate planning request
request = generator.generate_enhanced_plan_request("balanced")

# Generate market data for different scenarios
market_data = generator.generate_market_data("volatile")

# Generate trigger events
trigger = generator.generate_trigger_event("high")
```

## API Contracts and Documentation

### Standardized API Contracts

All agent interactions follow standardized contracts defined in `api/contracts.py`:

```python
from api.contracts import APIContracts

# Get contract for specific endpoint
contract = APIContracts.PLANNING_CONTRACTS["generate_plan"]

# Validate request against contract
is_valid = APIContracts.validate_contract(
    "planning", 
    "generate_plan", 
    request_data
)
```

### API Response Format

All APIs use standardized response format:

```json
{
  "success": true,
  "data": {...},
  "error": null,
  "error_code": null,
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345",
  "execution_time": 0.25,
  "api_version": "v1"
}
```

### Error Handling

Standardized error codes and messages:

```python
from api.contracts import COMMON_ERROR_CODES, HTTP_STATUS_CODES

# Common error patterns
{
  "success": false,
  "error": "Request format is invalid",
  "error_code": "INVALID_REQUEST",
  "suggested_action": "Check request schema"
}
```

## Logging and Monitoring Standards

### Structured Logging

All agents use structured logging with correlation tracking:

```python
from utils.logger import StructuredLogger, PerformanceMonitor

# Create agent logger
logger = StructuredLogger("planning_agent", "pa_001")

# Log with structured data
logger.info(
    "Plan generation started",
    user_id="user_123",
    correlation_id="corr_456",
    session_id="sess_789",
    goal_type="retirement"
)

# Log execution with audit trail
logger.log_execution(
    operation_name="generate_plan",
    status=ExecutionStatus.COMPLETED,
    input_data={"user_goal": "retirement"},
    output_data={"plan_steps": 5},
    execution_time=2.3,
    correlation_id="corr_456"
)
```

### Performance Monitoring

```python
# Decorator for automatic performance monitoring
@PerformanceMonitor(logger).monitor_execution("plan_generation")
async def generate_plan(request):
    # Function automatically logged with performance metrics
    return plan

# Context manager for code blocks
with PerformanceMonitor(logger).monitor_context("data_processing"):
    # Code block automatically monitored
    process_data()
```

### Audit Trail and Compliance

```python
# Export audit trail for compliance
audit_trail = logger.export_audit_trail(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

# Log financial transactions for compliance
logger.log_financial_transaction(
    transaction_type="investment",
    amount=50000.0,
    asset="SPY",
    user_id="user_123",
    session_id="sess_789",
    compliance_status="approved"
)
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline includes:

1. **Python Testing** (multiple versions)
2. **Frontend Testing** (TypeScript/React)
3. **Code Quality** (Black, Flake8, MyPy, Bandit)
4. **Performance Testing** (benchmarks)
5. **Docker Build and Test**
6. **Integration Testing**
7. **Deployment** (staging → production)

### Running Locally

```bash
# Run code quality checks
black --check .
flake8 .
mypy data_models/ agents/

# Run security analysis
bandit -r . -f json
safety check

# Build and test Docker containers
docker build -t finpilot-backend -f docker/Dockerfile.backend .
docker-compose -f docker-compose.test.yml up --build
```

## Development Workflow

### 1. Independent Development

Each developer works on their assigned agent:

```bash
# Create feature branch
git checkout -b feature/person-a-orchestration

# Develop using mock interfaces
python -c "
from agents.mock_interfaces import MockPlanningAgent
agent = MockPlanningAgent()
# Test your orchestration logic with mock planning agent
"

# Run tests frequently
pytest tests/test_agents.py::TestMockAgents::test_orchestration_agent_planning_request -v
```

### 2. Integration Testing

```bash
# Test agent communication
pytest tests/test_integration.py::TestIntegrationScenarios::test_complete_planning_workflow -v

# Test CMVL scenarios
pytest tests/test_integration.py::TestIntegrationScenarios::test_cmvl_trigger_scenario -v
```

### 3. Performance Validation

```bash
# Run performance benchmarks
pytest tests/test_performance.py --benchmark-only

# Monitor resource usage
pytest tests/test_integration.py::TestIntegrationScenarios::test_performance_under_load -v
```

## Debugging and Troubleshooting

### Log Analysis

```bash
# View structured logs
tail -f logs/finpilot.agents.planning.pa_001.log | jq .

# Filter by correlation ID
grep "corr_123" logs/*.log | jq .
```

### Health Monitoring

```python
# Check agent health
health = agent.get_health_status()
print(f"Agent {agent.agent_id} status: {health['status']}")
print(f"Success rate: {health['success_rate']:.2%}")

# Check system health
system_health = framework.get_system_health()
print(f"Total messages: {system_health['total_messages']}")
print(f"Success rate: {system_health['success_rate']:.2%}")
```

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Async Issues**: Use `pytest-asyncio` for async tests
3. **Mock Data**: Use `MockDataGenerator` with fixed seed for reproducible tests
4. **Correlation Tracking**: Always pass correlation_id through message chains

## Next Steps

1. **Start Development**: Choose your assigned agent and begin implementation
2. **Run Tests**: Ensure all mock interfaces work correctly
3. **Integration**: Test communication between your agent and mocks
4. **Documentation**: Update API contracts as you develop
5. **Performance**: Monitor and optimize your agent's performance

## Support

- Check `tests/` directory for comprehensive examples
- Review `api/contracts.py` for API specifications
- Use `utils/logger.py` for consistent logging
- Refer to `data_models/schemas.py` for data contracts

The development environment is designed to enable parallel development while maintaining integration compatibility. Each developer can work independently using realistic mock interfaces, then integrate seamlessly when ready.