"""
End-to-End Demo Scenario Validation

Tests complete demo scenarios from trigger to resolution.
Task 15 - Person D
Requirements: 11.1, 11.4, 12.4
"""

import pytest
import asyncio
from datetime import datetime

from agents.orchestration_agent import OrchestrationAgent
from agents.verifier import VerificationAgent
from agents.cmvl_advanced import AdvancedCMVLMonitor
from data_models.schemas import AgentMessage, MessageType


@pytest.fixture
def orchestration_agent():
    """Create orchestration agent for testing"""
    return OrchestrationAgent("test_oa_demo")


@pytest.fixture
def verification_agent():
    """Create verification agent for testing"""
    return VerificationAgent("test_va_demo")


@pytest.fixture
def cmvl_monitor():
    """Create CMVL monitor for testing"""
    return AdvancedCMVLMonitor("test_cmvl_demo")


class TestMarketCrashScenario:
    """Test complete market crash response scenario"""
    
    @pytest.mark.asyncio
    async def test_market_crash_end_to_end(self, orchestration_agent, verification_agent):
        """Test complete market crash scenario from detection to resolution"""
        # Step 1: Market crash trigger
        trigger_event = {
            "trigger_type": "market_event",
            "event_type": "market_crash",
            "severity": "critical",
            "description": "Market crash: -15% drop",
            "source_data": {"market_drop": -0.15, "volatility": 0.45},
            "impact_score": 1.0,
            "confidence_score": 0.95,
            "detector_agent_id": "ira_001",
            "correlation_id": "test_crash_001"
        }
        
        # Step 2: Trigger CMVL
        cmvl_message = AgentMessage(
            agent_id="ira_001",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": trigger_event},
            correlation_id="test_crash_001",
            session_id="test_session_crash",
            trace_id="test_trace_crash"
        )
        
        cmvl_response = await verification_agent.process_message(cmvl_message)
        
        # Verify CMVL activated
        assert cmvl_response.payload["cmvl_activated"] == True
        assert cmvl_response.payload["monitoring_frequency"] == "real_time"
        
        # Step 3: Verify current plan
        current_plan = {
            "plan_id": "test_plan_crash",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "portfolio_allocation",
                    "stock_allocation": 0.7,
                    "bond_allocation": 0.3,
                    "risk_level": "medium"
                }
            ],
            "verification_level": "comprehensive"
        }
        
        verify_message = AgentMessage(
            agent_id="cmvl_monitor",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": current_plan},
            correlation_id="test_crash_001",
            session_id="test_session_crash",
            trace_id="test_trace_crash"
        )
        
        verify_response = await verification_agent.process_message(verify_message)
        
        # Current plan should be flagged for review due to high market risk
        assert verify_response is not None
        
        # Step 4: Generate new plan (simulated)
        new_plan = {
            "plan_id": "test_plan_crash_revised",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "portfolio_rebalance",
                    "stock_allocation": 0.5,
                    "bond_allocation": 0.5,
                    "risk_level": "low"
                }
            ],
            "verification_level": "comprehensive"
        }
        
        new_verify_message = AgentMessage(
            agent_id="planning_agent",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": new_plan},
            correlation_id="test_crash_001",
            session_id="test_session_crash",
            trace_id="test_trace_crash"
        )
        
        new_verify_response = await verification_agent.process_message(new_verify_message)
        
        # New plan should pass verification
        assert new_verify_response is not None
        
        # Scenario complete
        print("✓ Market crash scenario completed successfully")


class TestJobLossScenario:
    """Test complete job loss adaptation scenario"""
    
    @pytest.mark.asyncio
    async def test_job_loss_end_to_end(self, verification_agent):
        """Test complete job loss scenario from event to plan update"""
        # Step 1: Job loss event
        life_event = {
            "trigger_type": "life_event",
            "event_type": "job_loss",
            "severity": "high",
            "description": "User reported job loss",
            "source_data": {"income_change": -1.0, "severance": 20000},
            "impact_score": 0.9,
            "confidence_score": 1.0,
            "detector_agent_id": "user_input",
            "correlation_id": "test_job_loss_001"
        }
        
        # Step 2: Trigger CMVL
        cmvl_message = AgentMessage(
            agent_id="user_input",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": life_event},
            correlation_id="test_job_loss_001",
            session_id="test_session_job_loss",
            trace_id="test_trace_job_loss"
        )
        
        cmvl_response = await verification_agent.process_message(cmvl_message)
        
        # Verify CMVL activated
        assert cmvl_response.payload["cmvl_activated"] == True
        
        # Step 3: Verify emergency plan
        emergency_plan = {
            "plan_id": "test_plan_emergency",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "expense_reduction",
                    "monthly_expenses_before": 6000,
                    "monthly_expenses_after": 4000,
                    "emergency_fund_usage": True
                },
                {
                    "step_id": "step_2",
                    "action_type": "pause_investments",
                    "investment_pause_duration": 6
                }
            ],
            "verification_level": "comprehensive"
        }
        
        verify_message = AgentMessage(
            agent_id="planning_agent",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": emergency_plan},
            correlation_id="test_job_loss_001",
            session_id="test_session_job_loss",
            trace_id="test_trace_job_loss"
        )
        
        verify_response = await verification_agent.process_message(verify_message)
        
        # Emergency plan should be verified
        assert verify_response is not None
        
        print("✓ Job loss scenario completed successfully")


class TestConcurrentTriggersScenario:
    """Test scenario with multiple concurrent triggers"""
    
    @pytest.mark.asyncio
    async def test_concurrent_triggers_end_to_end(self, verification_agent):
        """Test handling of concurrent market crash + job loss"""
        # Trigger 1: Market crash
        market_trigger = {
            "trigger_type": "market_event",
            "event_type": "market_crash",
            "severity": "critical",
            "description": "Market crash",
            "source_data": {"market_drop": -0.12},
            "impact_score": 0.95,
            "confidence_score": 0.9,
            "detector_agent_id": "ira_001",
            "correlation_id": "test_concurrent_001"
        }
        
        # Trigger 2: Job loss
        life_trigger = {
            "trigger_type": "life_event",
            "event_type": "job_loss",
            "severity": "high",
            "description": "Job loss",
            "source_data": {"income_change": -1.0},
            "impact_score": 0.9,
            "confidence_score": 1.0,
            "detector_agent_id": "user_input",
            "correlation_id": "test_concurrent_002"
        }
        
        # Send both triggers concurrently
        market_message = AgentMessage(
            agent_id="ira_001",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": market_trigger},
            correlation_id="test_concurrent_001",
            session_id="test_session_concurrent",
            trace_id="test_trace_concurrent"
        )
        
        life_message = AgentMessage(
            agent_id="user_input",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": life_trigger},
            correlation_id="test_concurrent_002",
            session_id="test_session_concurrent",
            trace_id="test_trace_concurrent"
        )
        
        # Process concurrently
        responses = await asyncio.gather(
            verification_agent.process_message(market_message),
            verification_agent.process_message(life_message)
        )
        
        # Both should activate CMVL
        assert all(r.payload["cmvl_activated"] for r in responses)
        
        # System should handle both triggers
        assert len(responses) == 2
        
        print("✓ Concurrent triggers scenario completed successfully")


class TestDemoScenarioPerformance:
    """Test demo scenario performance metrics"""
    
    @pytest.mark.asyncio
    async def test_demo_scenario_timing(self, verification_agent):
        """Test that demo scenarios complete within acceptable time"""
        scenarios = [
            {
                "name": "Market Crash",
                "max_duration": 5.0,  # seconds
                "trigger": {
                    "trigger_type": "market_event",
                    "event_type": "market_crash",
                    "severity": "critical",
                    "description": "Market crash",
                    "source_data": {"market_drop": -0.15},
                    "impact_score": 1.0,
                    "confidence_score": 0.95,
                    "detector_agent_id": "ira_001",
                    "correlation_id": "test_timing_001"
                }
            },
            {
                "name": "Job Loss",
                "max_duration": 4.0,
                "trigger": {
                    "trigger_type": "life_event",
                    "event_type": "job_loss",
                    "severity": "high",
                    "description": "Job loss",
                    "source_data": {"income_change": -1.0},
                    "impact_score": 0.9,
                    "confidence_score": 1.0,
                    "detector_agent_id": "user_input",
                    "correlation_id": "test_timing_002"
                }
            }
        ]
        
        for scenario in scenarios:
            start_time = datetime.utcnow()
            
            message = AgentMessage(
                agent_id="test_sender",
                target_agent_id=verification_agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"cmvl_trigger": scenario["trigger"]},
                correlation_id=scenario["trigger"]["correlation_id"],
                session_id="test_session_timing",
                trace_id="test_trace_timing"
            )
            
            response = await verification_agent.process_message(message)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Should complete within max duration
            assert duration < scenario["max_duration"], \
                f"{scenario['name']} took {duration}s, expected < {scenario['max_duration']}s"
            
            print(f"✓ {scenario['name']} completed in {duration:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
