# FinPilot Multi-Agent System - API Documentation

## Overview

This document provides comprehensive documentation for all agent API routes, message formats, and integration patterns in the FinPilot VP-MAS (Verifiable Planning Multi-Agent System).

## Table of Contents

1. [Communication Framework](#communication-framework)
2. [Agent Message Formats](#agent-message-formats)
3. [Agent-Specific APIs](#agent-specific-apis)
4. [Workflow Engine APIs](#workflow-engine-apis)
5. [Trigger Simulation APIs](#trigger-simulation-apis)
6. [Error Handling](#error-handling)
7. [Performance Monitoring](#performance-monitoring)

## Communication Framework

### Base Message Structure

All inter-agent communication uses the standardized `AgentMessage` format:

```json
{
  "message_id": "uuid4-string",
  "agent_id": "sender-agent-id",
  "target_agent_id": "target-agent-id-or-null-for-broadcast",
  "message_type": "request|response|notification|error|heartbeat",
  "payload": {
    "action": "specific-action-name",
    "parameters": {},
    "data": {}
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "correlation_id": "uuid4-string",
  "session_id": "uuid4-string",
  "priority": "critical|high|medium|low",
  "trace_id": "uuid4-string",
  "performance_metrics": {
    "execution_time": 0.0,
    "memory_usage": 0.0,
    "api_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "error_count": 0,
    "success_rate": 1.0,
    "throughput": 0.0
  },
  "retry_count": 0,
  "expires_at": "2024-01-01T00:05:00Z"
}
```

### Message Types

- **REQUEST**: Request for action or data from another agent
- **RESPONSE**: Response to a previous request
- **NOTIFICATION**: Broadcast information (no response expected)
- **ERROR**: Error notification
- **HEARTBEAT**: Health check/keep-alive message

### Priority Levels

- **CRITICAL**: Emergency situations, market crashes, system failures
- **HIGH**: Important planning requests, CMVL triggers
- **MEDIUM**: Standard operations, routine planning
- **LOW**: Background tasks, maintenance operations

## Agent Message Formats

### Orchestration Agent (OA)

#### Parse User Goal
**Request:**
```json
{
  "message_type": "request",
  "payload": {
    "action": "parse_goal",
    "parameters": {
      "goal_text": "Save $100,000 for retirement in 10 years",
      "user_context": {
        "age": 35,
        "income": 75000,
        "current_savings": 25000
      }
    }
  }
}
```

**Response:**
```json
{
  "message_type": "response",
  "payload": {
    "status": "parsed",
    "structured_goal": {
      "goal_type": "retirement_savings",
      "target_amount": 100000,
      "time_horizon_months": 120,
      "priority": "high",
      "constraints": [
        {
          "type": "time",
          "value": 120,
          "description": "Must achieve goal within 10 years"
        }
      ]
    },
    "workflow_plan": {
      "workflow_id": "uuid4-string",
      "estimated_completion": "2024-01-01T00:30:00Z",
      "steps": [
        {"agent": "information_retrieval", "action": "fetch_market_data"},
        {"agent": "planning", "action": "generate_plan"},
        {"agent": "verification", "action": "verify_plan"},
        {"agent": "execution", "action": "execute_plan"}
      ]
    }
  }
}
```

#### Handle Trigger Event
**Request:**
```json
{
  "message_type": "request",
  "payload": {
    "action": "handle_trigger",
    "parameters": {
      "trigger_event": {
        "trigger_id": "uuid4-string",
        "trigger_type": "market_event",
        "event_type": "volatility_spike",
        "severity": "high",
        "description": "Market volatility increased to 35%",
        "source_data": {
          "volatility_index": 0.35,
          "affected_sectors": ["technology", "energy"]
        },
        "impact_score": 0.7,
        "confidence_score": 0.9
      }
    }
  }
}
```

**Response:**
```json
{
  "message_type": "response",
  "payload": {
    "status": "cmvl_activated",
    "cmvl_response": {
      "trigger_id": "uuid4-string",
      "response_time": 0.05,
      "actions_initiated": [
        "market_data_refresh",
        "plan_re_evaluation",
        "risk_assessment_update"
      ],
      "priority_escalation": false,
      "estimated_completion": "2024-01-01T00:02:00Z"
    }
  }
}
```

### Planning Agent (PA)

#### Generate Financial Plan
**Request:**
```json
{
  "message_type": "request",
  "payload": {
    "action": "generate_plan",
    "parameters": {
      "planning_request": {
        "user_goal": "retirement_planning",
        "current_state": {
          "total_assets": 50000,
          "monthly_income": 6250,
          "monthly_expenses": 4500,
          "risk_tolerance": "moderate"
        },
        "constraints": [
          {
            "type": "budget",
            "max_monthly_investment": 1500
          },
          {
            "type": "risk",
            "max_portfolio_volatility": 0.15
          }
        ],
        "time_horizon": 120,
        "optimization_preferences": {
          "tax_efficiency": "high",
          "liquidity_preference": "medium"
        }
      }
    }
  }
}
```

**Response:**
```json
{
  "message_type": "response",
  "payload": {
    "status": "plan_generated",
    "plan_data": {
      "plan_id": "uuid4-string",
      "selected_strategy": "balanced",
      "confidence_score": 0.85,
      "plan_steps": [
        {
          "step_id": "uuid4-string",
          "sequence_number": 1,
          "action_type": "emergency_fund",
          "description": "Build emergency fund to 6 months expenses",
          "amount": 27000,
          "target_date": "2024-04-01T00:00:00Z",
          "rationale": "Conservative approach prioritizes safety",
          "confidence_score": 0.9,
          "risk_level": "low"
        },
        {
          "step_id": "uuid4-string",
          "sequence_number": 2,
          "action_type": "investment_allocation",
          "description": "Invest in diversified portfolio (60/40 stocks/bonds)",
          "amount": 60000,
          "target_date": "2024-06-01T00:00:00Z",
          "rationale": "Balanced approach for moderate risk tolerance",
          "confidence_score": 0.8,
          "risk_level": "medium"
        }
      ],
      "search_paths": [
        {
          "path_id": "uuid4-string",
          "strategy": "conservative",
          "combined_score": 0.75,
          "risk_score": 0.2,
          "expected_return": 0.06,
          "status": "explored"
        },
        {
          "path_id": "uuid4-string",
          "strategy": "balanced",
          "combined_score": 0.85,
          "risk_score": 0.4,
          "expected_return": 0.08,
          "status": "selected"
        }
      ],
      "reasoning_trace": {
        "trace_id": "uuid4-string",
        "final_decision": "Implement balanced strategy",
        "decision_rationale": "Strategy selected based on combined heuristic score of 0.850",
        "confidence_score": 0.85,
        "decision_points": [
          {
            "decision_type": "strategy_selection",
            "options_considered": [
              {"strategy": "conservative", "score": 0.75},
              {"strategy": "balanced", "score": 0.85}
            ],
            "chosen_option": {"strategy": "balanced", "score": 0.85},
            "rationale": "Balanced strategy provides optimal risk-return profile"
          }
        ]
      }
    }
  }
}
```

### Information Retrieval Agent (IRA)

#### Fetch Market Data
**Request:**
```json
{
  "message_type": "request",
  "payload": {
    "action": "fetch_market_data",
    "parameters": {
      "data_types": ["volatility", "interest_rates", "sector_trends"],
      "refresh_cache": false,
      "include_predictions": true
    }
  }
}
```

**Response:**
```json
{
  "message_type": "response",
  "payload": {
    "status": "data_retrieved",
    "market_data": {
      "data_id": "uuid4-string",
      "timestamp": "2024-01-01T00:00:00Z",
      "source": "alpha_vantage",
      "market_volatility": 0.15,
      "interest_rates": {
        "federal_funds": 0.0525,
        "10_year_treasury": 0.045,
        "30_year_mortgage": 0.072
      },
      "sector_trends": {
        "technology": 0.08,
        "healthcare": 0.05,
        "financial": 0.03,
        "energy": -0.02
      },
      "economic_sentiment": 0.1,
      "collection_method": "api_aggregation",
      "refresh_frequency": 300
    },
    "data_quality_score": 0.95,
    "collection_time": 0.2
  }
}
```

#### Detect Market Triggers
**Request:**
```json
{
  "message_type": "request",
  "payload": {
    "action": "detect_triggers",
    "parameters": {
      "monitoring_thresholds": {
        "volatility_spike": 0.3,
        "sentiment_crash": -0.5,
        "rate_change": 0.005
      },
      "lookback_period": "1h"
    }
  }
}
```

**Response:**
```json
{
  "message_type": "response",
  "payload": {
    "status": "triggers_detected",
    "triggers": [
      {
        "trigger_id": "uuid4-string",
        "trigger_type": "market_event",
        "event_type": "volatility_spike",
        "severity": "high",
        "description": "Market volatility increased to 35%",
        "source_data": {
          "volatility": 0.35,
          "previous_volatility": 0.15,
          "change_rate": 1.33
        },
        "impact_score": 0.7,
        "confidence_score": 0.9,
        "detected_at": "2024-01-01T00:00:00Z"
      }
    ],
    "monitoring_active": true,
    "next_check": "2024-01-01T00:05:00Z"
  }
}
```

### Verification Agent (VA)

#### Verify Financial Plan
**Request:**
```json
{
  "message_type": "request",
  "payload": {
    "action": "verify_plan",
    "parameters": {
      "verification_request": {
        "plan_id": "uuid4-string",
        "plan_steps": [
          {
            "step_id": "uuid4-string",
            "action_type": "investment_allocation",
            "amount": 60000,
            "risk_level": "medium"
          }
        ],
        "user_constraints": [
          {
            "constraint_type": "budget",
            "max_investment": 100000
          },
          {
            "constraint_type": "risk",
            "max_volatility": 0.2
          }
        ]
      }
    }
  }
}
```

**Response:**
```json
{
  "message_type": "response",
  "payload": {
    "status": "verification_completed",
    "verification_report": {
      "report_id": "uuid4-string",
      "plan_id": "uuid4-string",
      "verification_status": "approved",
      "constraints_checked": 10,
      "constraints_passed": 9,
      "constraint_violations": [
        {
          "constraint": "liquidity_requirement",
          "severity": "medium",
          "description": "Investment may impact liquidity requirements",
          "recommendation": "Consider maintaining higher cash reserves"
        }
      ],
      "overall_risk_score": 0.15,
      "approval_rationale": "Plan meets most constraints with minor liquidity concern",
      "confidence_score": 0.85,
      "verification_time": 0.15
    },
    "recommendations": [
      "Plan meets all constraints and can proceed",
      "Monitor liquidity requirements closely",
      "Review plan quarterly for optimization opportunities"
    ]
  }
}
```

#### Handle CMVL Trigger
**Request:**
```json
{
  "message_type": "request",
  "payload": {
    "action": "handle_cmvl_trigger",
    "parameters": {
      "cmvl_trigger": {
        "trigger_id": "uuid4-string",
        "severity": "high",
        "trigger_type": "market_event",
        "affected_plans": ["plan-uuid-1", "plan-uuid-2"]
      }
    }
  }
}
```

**Response:**
```json
{
  "message_type": "response",
  "payload": {
    "status": "cmvl_activated",
    "cmvl_response": {
      "trigger_id": "uuid4-string",
      "monitoring_frequency": "real_time",
      "verification_actions": [
        "constraint_re_evaluation",
        "risk_assessment_update",
        "compliance_check",
        "plan_validation"
      ],
      "estimated_completion": "2024-01-01T00:02:00Z",
      "auto_remediation": true
    }
  }
}
```

### Execution Agent (EA)

#### Execute Financial Plan
**Request:**
```json
{
  "message_type": "request",
  "payload": {
    "action": "execute_plan",
    "parameters": {
      "execution_request": {
        "plan_id": "uuid4-string",
        "plan_steps": [
          {
            "step_id": "uuid4-string",
            "action_type": "investment_allocation",
            "amount": 60000,
            "target_allocation": {
              "stocks": 0.6,
              "bonds": 0.4
            }
          }
        ],
        "execution_mode": "standard",
        "dry_run": false
      }
    }
  }
}
```

**Response:**
```json
{
  "message_type": "response",
  "payload": {
    "status": "execution_completed",
    "execution_results": [
      {
        "step_id": "uuid4-string",
        "status": "completed",
        "transaction_id": "uuid4-string",
        "amount_executed": 60000,
        "fees": 60,
        "execution_time": 0.05,
        "portfolio_impact": {
          "old_allocation": {"stocks": 0.5, "bonds": 0.5},
          "new_allocation": {"stocks": 0.6, "bonds": 0.4}
        }
      }
    ],
    "portfolio_updated": true,
    "total_execution_time": 0.2
  }
}
```

## Workflow Engine APIs

### Create Workflow
**Endpoint:** `POST /api/workflows`

**Request:**
```json
{
  "workflow_type": "financial_planning",
  "user_id": "user-uuid",
  "parameters": {
    "goal_text": "Save for retirement",
    "planning_request": {
      "time_horizon": 120,
      "risk_tolerance": "moderate"
    }
  },
  "priority": "high"
}
```

**Response:**
```json
{
  "workflow_id": "uuid4-string",
  "status": "created",
  "estimated_completion": "2024-01-01T00:30:00Z"
}
```

### Start Workflow
**Endpoint:** `POST /api/workflows/{workflow_id}/start`

**Response:**
```json
{
  "workflow_id": "uuid4-string",
  "status": "running",
  "started_at": "2024-01-01T00:00:00Z"
}
```

### Get Workflow Status
**Endpoint:** `GET /api/workflows/{workflow_id}`

**Response:**
```json
{
  "workflow_id": "uuid4-string",
  "workflow_name": "Financial Planning",
  "workflow_type": "financial_planning",
  "state": "running",
  "progress_percentage": 65.0,
  "tasks": {
    "task-1": {
      "task_name": "Parse User Goal",
      "state": "completed",
      "result": {"status": "parsed"}
    },
    "task-2": {
      "task_name": "Generate Plan",
      "state": "running",
      "started_at": "2024-01-01T00:05:00Z"
    }
  },
  "performance_metrics": {
    "execution_time_seconds": 300,
    "successful_tasks": 2,
    "failed_tasks": 0
  }
}
```

## Trigger Simulation APIs

### Generate Market Trigger
**Endpoint:** `POST /api/triggers/market`

**Request:**
```json
{
  "scenario": "volatility_spike",
  "parameters": {
    "volatility_level": 0.35,
    "affected_sectors": ["technology", "energy"]
  }
}
```

**Response:**
```json
{
  "trigger": {
    "trigger_id": "uuid4-string",
    "trigger_type": "market_event",
    "event_type": "volatility_spike",
    "severity": "high",
    "description": "Market volatility increased to 35%",
    "impact_score": 0.7,
    "confidence_score": 0.9
  }
}
```

### Generate Life Event Trigger
**Endpoint:** `POST /api/triggers/life-event`

**Request:**
```json
{
  "scenario": "job_loss",
  "parameters": {
    "severance_months": 3,
    "benefits_continuation": true
  }
}
```

**Response:**
```json
{
  "trigger": {
    "trigger_id": "uuid4-string",
    "trigger_type": "life_event",
    "severity": "high",
    "description": "User reported job loss with 3-month severance",
    "impact_score": 0.8,
    "confidence_score": 1.0
  }
}
```

### Generate Compound Triggers
**Endpoint:** `POST /api/triggers/compound`

**Request:**
```json
{
  "scenarios": ["job_loss", "market_crash"],
  "correlation_factor": 1.5
}
```

**Response:**
```json
{
  "triggers": [
    {
      "trigger_id": "uuid4-string-1",
      "trigger_type": "life_event",
      "severity": "high"
    },
    {
      "trigger_id": "uuid4-string-2",
      "trigger_type": "market_event",
      "severity": "critical"
    }
  ],
  "compound_impact_score": 0.95,
  "correlation_id": "uuid4-string"
}
```

## Error Handling

### Standard Error Response Format
```json
{
  "message_type": "error",
  "payload": {
    "error_code": "VALIDATION_ERROR",
    "error_message": "Invalid parameter: time_horizon must be positive",
    "error_details": {
      "field": "time_horizon",
      "provided_value": -5,
      "expected": "positive integer"
    },
    "correlation_id": "uuid4-string",
    "timestamp": "2024-01-01T00:00:00Z",
    "retry_recommended": true,
    "suggested_action": "Provide a positive value for time_horizon"
  }
}
```

### Common Error Codes

- **VALIDATION_ERROR**: Invalid input parameters
- **AGENT_NOT_FOUND**: Target agent not available
- **TIMEOUT_ERROR**: Operation timed out
- **CIRCUIT_BREAKER_OPEN**: Circuit breaker preventing operation
- **CONSTRAINT_VIOLATION**: Financial constraint violated
- **INSUFFICIENT_DATA**: Required data not available
- **SYSTEM_OVERLOAD**: System under high load

## Performance Monitoring

### System Health Endpoint
**Endpoint:** `GET /api/system/health`

**Response:**
```json
{
  "framework_uptime": 3600.0,
  "total_messages": 1500,
  "successful_messages": 1425,
  "failed_messages": 75,
  "success_rate": 0.95,
  "registered_agents": 5,
  "agent_health": {
    "orchestration_agent": {
      "status": "running",
      "uptime_seconds": 3600,
      "success_rate": 0.98,
      "queue_size": 2
    }
  },
  "circuit_breakers": {
    "planning_agent": {
      "state": "closed",
      "failure_count": 0
    }
  },
  "active_correlations": 25
}
```

### Performance Metrics Endpoint
**Endpoint:** `GET /api/system/metrics`

**Response:**
```json
{
  "throughput_metrics": {
    "messages_per_second": 125.5,
    "peak_throughput": 200.0,
    "average_latency_ms": 45.2
  },
  "resource_usage": {
    "memory_usage_mb": 512.0,
    "cpu_usage_percent": 25.5,
    "active_connections": 15
  },
  "workflow_metrics": {
    "active_workflows": 8,
    "completed_workflows": 142,
    "average_completion_time": 180.5
  }
}
```

## Integration Examples

### Complete Financial Planning Flow
```python
# 1. Create workflow
workflow_response = await create_workflow({
    "workflow_type": "financial_planning",
    "user_id": "user-123",
    "parameters": {
        "goal_text": "Save $100K for retirement",
        "time_horizon": 120
    }
})

# 2. Start workflow
await start_workflow(workflow_response["workflow_id"])

# 3. Monitor progress
while True:
    status = await get_workflow_status(workflow_response["workflow_id"])
    if status["state"] in ["completed", "failed"]:
        break
    await asyncio.sleep(5)

# 4. Handle results
if status["state"] == "completed":
    plan_results = status["results"]
    print(f"Plan generated with {len(plan_results['plan_steps'])} steps")
```

### CMVL Trigger Handling
```python
# 1. Generate market trigger
trigger = await generate_market_trigger({
    "scenario": "market_crash",
    "parameters": {"decline_percentage": 0.20}
})

# 2. Send to orchestration agent
message = create_agent_message(
    sender_id="market_monitor",
    target_id="orchestration_agent",
    message_type="request",
    payload={"trigger_event": trigger}
)

# 3. Process CMVL response
response = await send_message(message)
if response["payload"]["cmvl_activated"]:
    print("CMVL activated, monitoring plan adjustments")
```

This documentation provides comprehensive coverage of all agent APIs, message formats, and integration patterns for the FinPilot Multi-Agent System. Use this as a reference for implementing agent communication and building integrations with the system.