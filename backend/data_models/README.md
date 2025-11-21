# FinPilot Multi-Agent System - Data Models Documentation

## Overview

This directory contains comprehensive Pydantic data contracts for the FinPilot VP-MAS (Verifiable Planning Multi-Agent System). These models define the structure and validation rules for all inter-agent communication, financial data, and system operations.

## Requirements Coverage

This implementation covers the following requirements:
- **6.1, 6.2**: Agent communication protocols with structured data contracts
- **9.1, 9.3**: Parallel development support with clear interfaces
- **28.1, 28.4, 28.5**: Enhanced data contracts with correlation tracking and performance metrics

## Model Categories

### 1. Core Communication Models

#### AgentMessage
- **Purpose**: Inter-agent communication with correlation tracking
- **Key Features**: Performance metrics, retry logic, expiration handling
- **Used By**: All agents for structured communication

#### EnhancedPlanRequest
- **Purpose**: Comprehensive planning requests with multi-constraint support
- **Key Features**: Risk profiles, tax context, regulatory requirements
- **Used By**: Orchestration Agent → Planning Agent

#### PlanStep
- **Purpose**: Individual financial plan steps with detailed metadata
- **Key Features**: Rationale, confidence scores, tax implications
- **Used By**: Planning Agent → Verification Agent

#### VerificationReport
- **Purpose**: Comprehensive verification results with compliance analysis
- **Key Features**: Risk assessment, regulatory compliance, remediation suggestions
- **Used By**: Verification Agent → Orchestration Agent

### 2. Market Data and Trigger Models

#### MarketData
- **Purpose**: Real-time market data with predictive indicators
- **Key Features**: Multi-source integration, correlation analysis, quality scoring
- **Used By**: Information Retrieval Agent

#### TriggerEvent
- **Purpose**: Market and life event triggers for CMVL activation
- **Key Features**: Severity assessment, impact scoring, cascade risk analysis
- **Used By**: Information Retrieval Agent → Orchestration Agent

### 3. Financial State and Constraint Models

#### FinancialState
- **Purpose**: Comprehensive user financial situation
- **Key Features**: Portfolio breakdown, tax context, regulatory status
- **Used By**: All agents for financial context

#### Constraint
- **Purpose**: Financial constraints with regulatory and tax context
- **Key Features**: Dynamic thresholds, compliance requirements, violation tracking
- **Used By**: Planning Agent, Verification Agent

#### ExecutionLog
- **Purpose**: Comprehensive audit trail for all operations
- **Key Features**: Performance metrics, compliance tracking, error handling
- **Used By**: All agents for audit and debugging

### 4. Reasoning and Search Models

#### SearchPath
- **Purpose**: Detailed search paths with visualization metadata
- **Key Features**: Heuristic scores, constraint analysis, ReasonGraph support
- **Used By**: Planning Agent for transparency

#### ReasoningTrace
- **Purpose**: Complete reasoning audit trail
- **Key Features**: Decision points, performance metrics, quality assessment
- **Used By**: All agents for decision transparency

### 5. Risk and Compliance Models

#### RiskProfile
- **Purpose**: Comprehensive user risk assessment
- **Key Features**: Behavioral analysis, stress testing, goal prioritization
- **Used By**: Planning Agent, Verification Agent

#### TaxContext
- **Purpose**: Tax optimization and compliance context
- **Key Features**: Contribution limits, loss harvesting, optimization opportunities
- **Used By**: Planning Agent, Execution Agent

#### RegulatoryRequirement
- **Purpose**: Regulatory compliance requirements
- **Key Features**: Applicability rules, penalties, monitoring requirements
- **Used By**: Verification Agent

#### ComplianceStatus
- **Purpose**: Compliance tracking and remediation
- **Key Features**: Risk scoring, remediation plans, audit trails
- **Used By**: Verification Agent

## Usage Examples

### Basic Agent Communication
```python
from data_models import AgentMessage, MessageType, Priority

message = AgentMessage(
    agent_id="planning_agent_001",
    target_agent_id="verification_agent_001",
    message_type=MessageType.REQUEST,
    payload={"plan_request": "retirement_planning"},
    correlation_id="corr_123",
    session_id="sess_456",
    priority=Priority.HIGH
)
```

### Financial Planning Request
```python
from data_models import EnhancedPlanRequest

request = EnhancedPlanRequest(
    user_id="user_001",
    user_goal="Save ₹20L for retirement in 10 years",
    current_state={"portfolio_value": 500000},
    time_horizon=120,
    correlation_id="corr_123",
    session_id="sess_456"
)
```

### Market Trigger Event
```python
from data_models import TriggerEvent, MarketEventType, SeverityLevel

trigger = TriggerEvent(
    trigger_type="market_event",
    event_type=MarketEventType.VOLATILITY_SPIKE,
    severity=SeverityLevel.HIGH,
    description="Market volatility increased by 40%",
    impact_score=0.75,
    detector_agent_id="ira_001"
)
```

## Validation and Quality Assurance

### Built-in Validation
- All models include comprehensive Pydantic validation
- Financial amounts validated for positivity and reasonableness
- Percentages and scores validated within appropriate ranges
- Dates validated for logical consistency

### Custom Validators
- `validators.py` provides additional validation utilities
- Financial calculations with error checking
- Data quality validation for IDs and formats
- Compliance validation for regulatory requirements

## Integration Guidelines

### For Agent Development
1. Import required models from `data_models`
2. Use type hints with Pydantic models for IDE support
3. Validate all input data using model constructors
4. Include correlation IDs for all inter-agent messages
5. Log performance metrics for monitoring

### For Testing
1. Use model `.Config.schema_extra["example"]` for test data
2. Validate serialization/deserialization with `.json()` and `.parse_raw()`
3. Test constraint validation with invalid data
4. Verify correlation ID propagation across agents

### For Documentation
1. All models include comprehensive docstrings
2. Field descriptions explain purpose and usage
3. Examples provided in schema_extra
4. Validation rules documented in field definitions

## Performance Considerations

### Memory Usage
- Models use appropriate data types (Decimal for financial amounts)
- Optional fields reduce memory footprint when not needed
- Lazy loading supported for large nested structures

### Serialization
- All models support JSON serialization via Pydantic
- Custom serializers for Decimal and datetime types
- Efficient serialization for large data structures

### Validation Performance
- Validators optimized for common use cases
- Caching of validation results where appropriate
- Minimal overhead for basic field validation

## Compliance and Audit

### Regulatory Compliance
- Models support regulatory requirement tracking
- Compliance status monitoring built-in
- Audit trail generation for all operations

### Data Privacy
- No PII stored in model examples
- Configurable data retention policies
- Secure handling of sensitive financial data

### Audit Trail
- Complete operation logging with ExecutionLog
- Correlation tracking across all operations
- Performance metrics for system monitoring

## Future Enhancements

### Planned Features
- Machine learning model integration for predictive analytics
- Enhanced visualization metadata for ReasonGraph
- Advanced compliance automation
- Real-time data validation and quality monitoring

### Extensibility
- Models designed for easy extension
- Plugin architecture for custom validators
- Configurable validation rules
- Dynamic schema evolution support