"""
Test suite for Advanced CMVL Implementation (Task 20)

Tests all components of the Continuous Monitoring and Verification Loop:
- Sophisticated trigger response system
- Dynamic replanning engine
- Intelligent constraint re-evaluator
- Performance monitoring
- Predictive real-time verification
- Concurrent trigger handling
- Predictive monitoring system

Requirements: 2.1, 2.2, 2.3, 2.4, 38.1, 38.2, 38.3, 38.4, 38.5
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from agents.verifier import VerificationAgent
from agents.cmvl_advanced import (
    SophisticatedTriggerResponseSystem,
    DynamicReplanningEngine,
    IntelligentConstraintReevaluator,
    CMVLPerformanceMonitor,
    PredictiveRealTimeVerifier,
    ConcurrentTriggerHandler,
    PredictiveMonitoringSystem,
    AdvancedCMVLMonitor
)
from data_models.schemas import (
    AgentMessage, MessageType, Priority,
    TriggerEvent, MarketEventType, SeverityLevel
)


@pytest.fixture
def verification_agent():
    """Create verification agent for testing"""
    return VerificationAgent("test_verifier_001")


@pytest.fixture
def sample_triggers():
    """Create sample trigger events"""
    return [
        TriggerEvent(
            trigger_id=str(uuid4()),
            trigger_type="market_event",
            event_type=MarketEventType.VOLATILITY_SPIKE,
            severity=SeverityLevel.HIGH,
            description="Market volatility spike detected",
            source_data={"volatility": 0.45, "change_percent": -5.2},
            detected_at=datetime.utcnow(),
            impact_score=0.7,
            confidence_score=0.85,
            detector_agent_id="ira_001",
            correlation_id=str(uuid4())
        ),
        TriggerEvent(
            trigger_id=str(uuid4()),
            trigger_type="market_event",
            event_type=MarketEventType.INTEREST_RATE_CHANGE,
            severity=SeverityLevel.MEDIUM,
            description="Interest rate change detected",
            source_data={"rate_change": 0.25},
            detected_at=datetime.utcnow(),
            impact_score=0.5,
            confidence_score=0.9,
            detector_agent_id="ira_001",
            correlation_id=str(uuid4())
        )
    ]


@pytest.fixture
def sample_plan():
    """Create sample financial plan"""
    return {
        "plan_id": str(uuid4()),
        "user_id": "user_001",
        "steps": [
            {"step_id": "1", "action_type": "invest", "amount": 10000},
            {"step_id": "2", "action_type": "save", "amount": 5000}
        ],
        "created_at": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_constraints():
    """Create sample constraints"""
    return {
        "emergency_fund": {"threshold": 10000, "current": 15000},
        "debt_ratio": {"max": 0.36, "current": 0.25},
        "risk_tolerance": {"level": "moderate"}
    }


# Test 20.1: Sophisticated Trigger Response System
@pytest.mark.asyncio
async def test_trigger_prioritization(sample_triggers):
    """Test intelligent trigger prioritization (Req 38.1)"""
    system = SophisticatedTriggerResponseSystem()
    
    priorities = system.prioritize_triggers(sample_triggers)
    
    assert len(priorities) == len(sample_triggers)
    assert priorities[0].priority_score >= priorities[1].priority_score
    assert all(0 <= p.priority_score <= 1 for p in priorities)
    assert priorities[0].severity == SeverityLevel.HIGH  # Higher severity first


@pytest.mark.asyncio
async def test_coordinated_response_generation(sample_triggers, sample_plan):
    """Test coordinated response strategy generation (Req 38.1)"""
    system = SophisticatedTriggerResponseSystem()
    
    response = system.generate_coordinated_response(sample_triggers, sample_plan)
    
    assert "strategy_id" in response
    assert "coordinated_actions" in response
    assert "resource_allocation" in response
    assert "rollback_plan" in response
    assert len(response["coordinated_actions"]) > 0
    assert response["rollback_plan"]["rollback_available"] is True


# Test 20.2: Dynamic Replanning Engine
@pytest.mark.asyncio
async def test_replanning_evaluation(sample_plan, sample_triggers, sample_constraints):
    """Test replanning need evaluation with predictive analysis (Req 38.2)"""
    engine = DynamicReplanningEngine()
    
    decision = await engine.evaluate_replanning_need(
        sample_plan, sample_triggers, sample_constraints
    )
    
    assert hasattr(decision, 'should_replan')
    assert hasattr(decision, 'confidence')
    assert hasattr(decision, 'predicted_improvement')
    assert hasattr(decision, 'rollback_available')
    assert 0 <= decision.confidence <= 1
    assert 0 <= decision.predicted_improvement <= 1
    assert decision.rollback_available is True


@pytest.mark.asyncio
async def test_replanning_with_rollback(sample_plan, sample_triggers):
    """Test replanning execution with rollback mechanism (Req 38.2)"""
    engine = DynamicReplanningEngine()
    session_id = str(uuid4())
    
    result = await engine.execute_replanning_with_rollback(
        sample_plan, sample_triggers, session_id
    )
    
    assert "success" in result
    assert "snapshot_id" in result
    if result["success"] and not result.get("rolled_back"):
        assert "new_plan" in result
        assert "improvement_score" in result


# Test 20.3: Intelligent Constraint Re-evaluator
@pytest.mark.asyncio
async def test_constraint_reevaluation(sample_constraints, sample_triggers):
    """Test constraint re-evaluation with forward-looking analysis (Req 38.3)"""
    reevaluator = IntelligentConstraintReevaluator()
    market_conditions = {"volatility": 0.3, "trend": "bearish"}
    
    result = await reevaluator.reevaluate_constraints(
        sample_constraints, sample_triggers, market_conditions
    )
    
    assert "constraints_evaluated" in result
    assert "constraints_modified" in result
    assert "new_constraints" in result
    assert "scenario_analysis" in result
    assert result["constraints_evaluated"] > 0


# Test 20.4: CMVL Performance Monitor
def test_cycle_monitoring():
    """Test CMVL cycle performance monitoring (Req 38.4)"""
    monitor = CMVLPerformanceMonitor()
    cycle_id = str(uuid4())
    
    # Start monitoring
    metrics = monitor.start_cycle_monitoring(cycle_id)
    assert metrics.cycle_id == cycle_id
    
    # End monitoring
    result = monitor.end_cycle_monitoring(
        cycle_id,
        triggers_processed=2,
        verifications_completed=2,
        replanning_triggered=True,
        constraints_reevaluated=5
    )
    
    assert "cycle_id" in result
    assert "duration" in result
    assert "success_rate" in result
    assert "recommendations" in result
    assert result["success_rate"] == 1.0


def test_performance_analytics():
    """Test performance analytics generation (Req 38.4)"""
    monitor = CMVLPerformanceMonitor()
    
    # Simulate multiple cycles
    for i in range(3):
        cycle_id = str(uuid4())
        monitor.start_cycle_monitoring(cycle_id)
        monitor.end_cycle_monitoring(
            cycle_id, triggers_processed=2, verifications_completed=2,
            replanning_triggered=False, constraints_reevaluated=3
        )
    
    analytics = monitor.get_performance_analytics()
    
    assert "total_cycles" in analytics
    assert "average_cycle_duration" in analytics
    assert "average_success_rate" in analytics
    assert analytics["total_cycles"] == 3


# Test 20.5: Predictive Real-Time Verifier
@pytest.mark.asyncio
async def test_verification_with_confidence(sample_plan, sample_constraints):
    """Test predictive verification with confidence intervals (Req 38.5)"""
    verifier = PredictiveRealTimeVerifier()
    market_conditions = {"volatility": 0.2}
    
    result = await verifier.verify_with_confidence(
        sample_plan, sample_constraints, market_conditions
    )
    
    assert "verification_id" in result
    assert "is_valid" in result
    assert "confidence_score" in result
    assert "confidence_intervals" in result
    assert "future_validity_prediction" in result
    assert 0 <= result["confidence_score"] <= 1
    
    # Check confidence intervals
    intervals = result["confidence_intervals"]
    assert "lower_bound" in intervals
    assert "upper_bound" in intervals
    assert intervals["lower_bound"] <= result["confidence_score"] <= intervals["upper_bound"]


# Test 20.6: Concurrent Trigger Handler
@pytest.mark.asyncio
async def test_concurrent_trigger_handling(sample_triggers):
    """Test concurrent trigger handling with resource allocation (Req 38.1)"""
    handler = ConcurrentTriggerHandler(max_concurrent=5)
    session_id = str(uuid4())
    
    result = await handler.handle_concurrent_triggers(sample_triggers, session_id)
    
    assert "triggers_processed" in result
    assert "successful" in result
    assert "failed" in result
    assert "resource_allocation" in result
    assert "coordinated_response" in result
    assert result["triggers_processed"] == len(sample_triggers)
    assert result["successful"] + result["failed"] == len(sample_triggers)


# Test 20.7: Predictive Monitoring System
@pytest.mark.asyncio
async def test_predictive_monitoring_start():
    """Test predictive monitoring system startup (Req 38.4)"""
    system = PredictiveMonitoringSystem()
    plan_id = str(uuid4())
    session_id = str(uuid4())
    config = {"check_interval_seconds": 30}
    
    result = await system.start_predictive_monitoring(plan_id, session_id, config)
    
    assert "monitoring_id" in result
    assert "status" in result
    assert result["status"] == "active"
    
    # Stop monitoring
    stop_result = system.stop_monitoring(result["monitoring_id"])
    assert stop_result["status"] == "stopped"


# Test 20.8: End-to-End CMVL Integration
@pytest.mark.asyncio
async def test_cmvl_end_to_end(verification_agent, sample_triggers, sample_plan, sample_constraints):
    """Test complete CMVL cycle end-to-end (All Req 38.x)"""
    # Create CMVL trigger message
    message = AgentMessage(
        agent_id="orchestrator_001",
        target_agent_id=verification_agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "cmvl_trigger": sample_triggers[0].dict(),
            "triggers": [t.dict() for t in sample_triggers],
            "current_plan": sample_plan,
            "constraints": sample_constraints,
            "market_conditions": {"volatility": 0.3}
        },
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    # Handle CMVL trigger
    response = await verification_agent._handle_cmvl_trigger(message)
    
    # Verify response structure
    assert response.message_type == MessageType.RESPONSE
    payload = response.payload
    
    assert "cmvl_id" in payload
    assert "cmvl_activated" in payload
    assert payload["cmvl_activated"] is True
    
    # Check all CMVL components
    assert "concurrent_handling" in payload
    assert "replanning" in payload
    assert "constraint_reevaluation" in payload
    assert "verification" in payload
    assert "predictive_monitoring" in payload
    assert "performance_metrics" in payload
    
    # Verify concurrent handling
    assert payload["concurrent_handling"]["triggers_processed"] > 0
    
    # Verify replanning decision
    assert "should_replan" in payload["replanning"]
    assert "confidence" in payload["replanning"]
    
    # Verify constraint re-evaluation
    assert payload["constraint_reevaluation"]["constraints_evaluated"] > 0
    
    # Verify verification with confidence
    assert "confidence_score" in payload["verification"]
    assert "confidence_intervals" in payload["verification"]
    
    # Verify performance metrics
    assert "cycle_duration" in payload["performance_metrics"]
    assert "success_rate" in payload["performance_metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
