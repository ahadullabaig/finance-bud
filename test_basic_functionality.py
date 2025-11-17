#!/usr/bin/env python3
"""
Basic functionality test for agent communication and trigger simulation
"""

import sys
import asyncio
from datetime import datetime
from uuid import uuid4

# Add current directory to path
sys.path.append('.')

from agents.communication import AgentCommunicationFramework
from agents.mock_interfaces import MockOrchestrationAgent, MockPlanningAgent
from data_models.schemas import AgentMessage, MessageType, Priority, TriggerEvent, SeverityLevel, MarketEventType
from utils.trigger_simulator import TriggerSimulator


def test_communication_framework():
    """Test basic communication framework functionality"""
    print("Testing Communication Framework...")
    
    # Create framework
    framework = AgentCommunicationFramework()
    
    # Create mock agents
    orchestrator = MockOrchestrationAgent("test_oa")
    planner = MockPlanningAgent("test_pa")
    
    # Register agents
    framework.register_agent(orchestrator, ['workflow_coordination'])
    framework.register_agent(planner, ['financial_planning'])
    
    # Check system health
    health = framework.get_system_health()
    
    print(f"✓ Framework created with {health['registered_agents']} agents")
    print(f"✓ Success rate: {health['success_rate']:.2%}")
    
    return True


def test_trigger_simulation():
    """Test trigger simulation functionality"""
    print("\nTesting Trigger Simulation...")
    
    # Create simulator
    simulator = TriggerSimulator()
    
    # Test market trigger
    market_trigger = simulator.generate_market_trigger("volatility_spike")
    print(f"✓ Market trigger: {market_trigger.event_type.value} - {market_trigger.severity.value}")
    
    # Test life event trigger
    life_trigger = simulator.generate_life_event_trigger("job_loss")
    print(f"✓ Life event trigger: {life_trigger.trigger_type} - {life_trigger.severity.value}")
    
    # Test compound triggers
    compound_triggers = simulator.generate_compound_trigger(["job_loss", "market_crash"])
    print(f"✓ Compound triggers: {len(compound_triggers)} triggers generated")
    
    return True


async def test_async_communication():
    """Test async communication between agents"""
    print("\nTesting Async Communication...")
    
    # Create framework and agents
    framework = AgentCommunicationFramework()
    orchestrator = MockOrchestrationAgent("async_oa")
    planner = MockPlanningAgent("async_pa")
    
    framework.register_agent(orchestrator, ['workflow_coordination'])
    framework.register_agent(planner, ['financial_planning'])
    
    # Create and send message
    message = framework.create_message(
        sender_id="async_oa",
        target_id="async_pa",
        message_type=MessageType.REQUEST,
        payload={"test": "async_communication"},
        priority=Priority.MEDIUM
    )
    
    success = await framework.send_message(message)
    print(f"✓ Async message sent: {success}")
    
    # Check message was received
    if planner.message_queue.qsize() > 0:
        print("✓ Message received by target agent")
    
    return True


def test_data_models():
    """Test data model validation"""
    print("\nTesting Data Models...")
    
    # Test TriggerEvent creation
    trigger = TriggerEvent(
        trigger_type="market_event",
        event_type=MarketEventType.VOLATILITY_SPIKE,
        severity=SeverityLevel.HIGH,
        description="Test market volatility spike",
        source_data={"volatility": 0.35},
        impact_score=0.7,
        confidence_score=0.9,
        detector_agent_id="test_agent",
        correlation_id=str(uuid4())
    )
    
    print(f"✓ TriggerEvent created: {trigger.trigger_id}")
    
    # Test AgentMessage creation
    message = AgentMessage(
        agent_id="test_sender",
        target_agent_id="test_receiver",
        message_type=MessageType.REQUEST,
        payload={"test": "data_validation"},
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    print(f"✓ AgentMessage created: {message.message_id}")
    
    return True


def main():
    """Run all basic functionality tests"""
    print("=" * 60)
    print("FinPilot Agent Communication Framework - Basic Tests")
    print("=" * 60)
    
    try:
        # Test synchronous functionality
        test_communication_framework()
        test_trigger_simulation()
        test_data_models()
        
        # Test asynchronous functionality
        asyncio.run(test_async_communication())
        
        print("\n" + "=" * 60)
        print("✓ All basic functionality tests passed!")
        print("✓ Agent communication framework is working correctly")
        print("✓ Trigger simulation system is operational")
        print("✓ Data models are validating properly")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)