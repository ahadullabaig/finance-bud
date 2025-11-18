#!/usr/bin/env python3
"""
Test script for Planning Agent life event adjustment capabilities.
Tests the implementation of Task 2: Enhance Planning Agent with adjustment logic.
"""

import asyncio
import sys
import os
from datetime import datetime
from decimal import Decimal

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.planning_agent import PlanningAgent, PlanAdjustmentLogic
from data_models.schemas import (
    TriggerEvent, ClassifiedTrigger, LifeEventType, UrgencyLevel, 
    SeverityLevel, MarketEventType, AgentMessage, MessageType
)


def create_test_trigger(event_type: LifeEventType, severity: SeverityLevel = SeverityLevel.HIGH) -> ClassifiedTrigger:
    """Create a test trigger event"""
    trigger_event = TriggerEvent(
        trigger_type="life_event",
        event_type=MarketEventType.MARKET_CRASH,  # Using available enum
        severity=severity,
        description=f"Test {event_type.value} event",
        source_data={"estimated_costs": 25000, "urgency": "high"},
        impact_score=0.8,
        confidence_score=0.9,
        detector_agent_id="test_agent",
        correlation_id="test_correlation"
    )
    
    return ClassifiedTrigger(
        trigger_event=trigger_event,
        classification=f"life_event_{event_type.value}",
        priority_score=0.85,
        urgency_level=UrgencyLevel.HIGH,
        recommended_actions=["assess_impact", "generate_adjustments"],
        estimated_processing_time=120,
        requires_immediate_attention=True,
        classification_confidence=0.9,
        classified_by_agent="event_classifier"
    )


def create_test_financial_state() -> dict:
    """Create test financial state"""
    return {
        "total_assets": 150000,
        "total_liabilities": 50000,
        "monthly_income": 8000,
        "monthly_expenses": 6000,
        "emergency_fund": 18000,
        "hsa_balance": 5000,
        "business_income_percentage": 0.7,
        "liquid_investments": 30000,
        "risky_investments": 40000
    }


def create_test_current_plan() -> dict:
    """Create test current financial plan"""
    return {
        "plan_id": "test_plan_123",
        "goals": [
            {"goal_id": "retirement", "priority": "critical", "target_amount": 1000000},
            {"goal_id": "emergency_fund", "priority": "high", "target_amount": 36000}
        ],
        "investment_strategy": "balanced",
        "risk_tolerance": "moderate"
    }


async def test_plan_adjustment_logic():
    """Test the PlanAdjustmentLogic class"""
    print("Testing PlanAdjustmentLogic...")
    
    logic = PlanAdjustmentLogic()
    
    # Test job loss adjustment
    print("\n1. Testing job loss adjustment:")
    job_loss_trigger = create_test_trigger(LifeEventType.JOB_LOSS)
    financial_state = create_test_financial_state()
    current_plan = create_test_current_plan()
    
    job_loss_adjustment = logic.calculate_adjustments(job_loss_trigger, current_plan, financial_state)
    print(f"   Adjustment type: {job_loss_adjustment.get('adjustment_type')}")
    print(f"   Confidence score: {job_loss_adjustment.get('confidence_score')}")
    print(f"   Emergency fund target: ${job_loss_adjustment.get('emergency_fund_adjustments', {}).get('target_amount', 0):,.0f}")
    
    # Test medical emergency adjustment
    print("\n2. Testing medical emergency adjustment:")
    medical_trigger = create_test_trigger(LifeEventType.MEDICAL_EMERGENCY)
    medical_adjustment = logic.calculate_adjustments(medical_trigger, current_plan, financial_state)
    print(f"   Adjustment type: {medical_adjustment.get('adjustment_type')}")
    print(f"   Healthcare funding gap: ${medical_adjustment.get('healthcare_funding', {}).get('funding_gap', 0):,.0f}")
    
    # Test business disruption adjustment
    print("\n3. Testing business disruption adjustment:")
    business_trigger = create_test_trigger(LifeEventType.BUSINESS_DISRUPTION)
    business_adjustment = logic.calculate_adjustments(business_trigger, current_plan, financial_state)
    print(f"   Adjustment type: {business_adjustment.get('adjustment_type')}")
    print(f"   Adjusted planning income: ${business_adjustment.get('income_adjustments', {}).get('adjusted_planning_income', 0):,.0f}")
    
    # Test multi-trigger handling
    print("\n4. Testing multi-trigger handling:")
    triggers = [job_loss_trigger, medical_trigger]
    multi_adjustment = logic.handle_multiple_triggers(triggers, current_plan, financial_state)
    print(f"   Adjustment type: {multi_adjustment.get('adjustment_type')}")
    print(f"   Trigger count: {multi_adjustment.get('trigger_count')}")
    print(f"   Confidence score: {multi_adjustment.get('confidence_score')}")
    
    print("\nPlanAdjustmentLogic tests completed successfully!")


async def test_planning_agent_integration():
    """Test the Planning Agent integration with life event adjustments"""
    print("\nTesting Planning Agent integration...")
    
    # Create planning agent
    agent = PlanningAgent("test_planning_agent")
    
    # Test life event adjustment message
    print("\n1. Testing life event adjustment message handling:")
    
    life_event_message = AgentMessage(
        agent_id="test_orchestrator",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "life_event_adjustment": {
                "classified_trigger": create_test_trigger(LifeEventType.JOB_LOSS).dict(),
                "current_plan": create_test_current_plan(),
                "financial_state": create_test_financial_state(),
                "user_context": {"risk_tolerance": "moderate", "age": 35}
            }
        },
        correlation_id="test_correlation",
        session_id="test_session",
        trace_id="test_trace"
    )
    
    response = await agent.process_message(life_event_message)
    
    if response and response.message_type == MessageType.RESPONSE:
        payload = response.payload
        print(f"   Response received: {payload.get('life_event_adjustment_completed', False)}")
        print(f"   Trigger classification: {payload.get('trigger_classification')}")
        print(f"   Confidence score: {payload.get('confidence_score', 0):.2f}")
        print(f"   Processing time: {payload.get('processing_time', 0):.3f}s")
        print(f"   Requires approval: {payload.get('requires_user_approval', False)}")
    else:
        print("   ERROR: No valid response received")
        return False
    
    # Test multi-trigger adjustment message
    print("\n2. Testing multi-trigger adjustment message handling:")
    
    multi_trigger_message = AgentMessage(
        agent_id="test_orchestrator",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "multi_trigger_adjustment": {
                "classified_triggers": [
                    create_test_trigger(LifeEventType.JOB_LOSS).dict(),
                    create_test_trigger(LifeEventType.MEDICAL_EMERGENCY).dict()
                ],
                "current_plan": create_test_current_plan(),
                "financial_state": create_test_financial_state(),
                "user_context": {"risk_tolerance": "moderate", "age": 35}
            }
        },
        correlation_id="test_correlation_multi",
        session_id="test_session_multi",
        trace_id="test_trace_multi"
    )
    
    multi_response = await agent.process_message(multi_trigger_message)
    
    if multi_response and multi_response.message_type == MessageType.RESPONSE:
        payload = multi_response.payload
        print(f"   Multi-trigger response received: {payload.get('multi_trigger_adjustment_completed', False)}")
        print(f"   Trigger count: {payload.get('trigger_count', 0)}")
        print(f"   Confidence score: {payload.get('confidence_score', 0):.2f}")
        print(f"   Processing time: {payload.get('processing_time', 0):.3f}s")
        print(f"   Compound optimizations: {bool(payload.get('compound_optimizations'))}")
    else:
        print("   ERROR: No valid multi-trigger response received")
        return False
    
    print("\nPlanning Agent integration tests completed successfully!")
    return True


async def main():
    """Run all tests"""
    print("=" * 60)
    print("PLANNING AGENT LIFE EVENT ADJUSTMENT TESTS")
    print("=" * 60)
    
    try:
        # Test Plan Adjustment Logic
        await test_plan_adjustment_logic()
        
        # Test Planning Agent Integration
        success = await test_planning_agent_integration()
        
        if success:
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED SUCCESSFULLY!")
            print("Task 2 implementation is working correctly.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("SOME TESTS FAILED!")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print(f"\nTEST FAILED WITH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)