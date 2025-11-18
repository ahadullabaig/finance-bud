"""
Verification Agent Test Suite

Comprehensive tests for verification agent functionality and CMVL.
Person D - Task 23
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal

from agents.verifier import VerificationAgent
from data_models.schemas import (
    AgentMessage, MessageType, Priority,
    VerificationStatus, TriggerEvent, MarketEventType, SeverityLevel,
    PlanStep
)


@pytest.fixture
def verification_agent():
    """Create verification agent instance for testing"""
    return VerificationAgent("test_verification_001")


@pytest.fixture
def sample_plan_steps():
    """Create sample plan steps for testing"""
    return [
        {
            "step_id": "step_1",
            "sequence_number": 1,
            "action_type": "emergency_fund",
            "description": "Build emergency fund",
            "amount": 30000,
            "risk_level": "low",
            "confidence_score": 0.9
        },
        {
            "step_id": "step_2",
            "sequence_number": 2,
            "action_type": "equity_investment",
            "description": "Invest in diversified portfolio",
            "amount": 50000,
            "risk_level": "medium",
            "confidence_score": 0.8
        }
    ]


@pytest.fixture
def sample_verification_request(sample_plan_steps):
    """Create sample verification request"""
    return {
        "plan_id": "test_plan_001",
        "plan_steps": sample_plan_steps,
        "verification_level": "comprehensive"
    }


class TestVerificationAgent:
    """Test suite for Verification Agent core functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, verification_agent):
        """Test agent initializes correctly"""
        assert verification_agent.agent_id == "test_verification_001"
        assert verification_agent.agent_type == "verification"
        assert len(verification_agent.constraint_rules) > 0
        assert len(verification_agent.regulatory_rules) > 0
        assert verification_agent.cmvl_active == False
    
    @pytest.mark.asyncio
    async def test_verify_valid_plan(self, verification_agent, sample_verification_request):
        """Test verification of a valid plan"""
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": sample_verification_request},
            correlation_id="test_corr_001",
            session_id="test_session_001",
            trace_id="test_trace_001"
        )
        
        response = await verification_agent.process_message(message)
        
        assert response is not None
        assert response.message_type == MessageType.RESPONSE
        assert "verification_report" in response.payload
        
        report = response.payload["verification_report"]
        assert report["plan_id"] == "test_plan_001"
        assert "verification_status" in report
        assert "overall_risk_score" in report
        assert "confidence_score" in report
    
    @pytest.mark.asyncio
    async def test_verify_plan_with_violations(self, verification_agent):
        """Test verification of plan with constraint violations"""
        # Create plan with violations
        plan_with_violations = {
            "plan_id": "test_plan_002",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "high_risk_investment",
                    "amount": 150000,  # Exceeds limit
                    "risk_level": "high"
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan_with_violations},
            correlation_id="test_corr_002",
            session_id="test_session_002",
            trace_id="test_trace_002"
        )
        
        response = await verification_agent.process_message(message)
        
        assert response is not None
        report = response.payload["verification_report"]
        step_results = response.payload["step_results"]
        
        # Should have violations
        assert len(step_results) > 0
        assert len(step_results[0]["violations"]) > 0
        assert report["overall_risk_score"] > 0
    
    @pytest.mark.asyncio
    async def test_constraint_checking(self, verification_agent):
        """Test specific constraint checking"""
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "constraint_check": {
                    "constraints": ["emergency_fund", "debt_to_income", "risk_tolerance"]
                }
            },
            correlation_id="test_corr_003",
            session_id="test_session_003",
            trace_id="test_trace_003"
        )
        
        response = await verification_agent.process_message(message)
        
        assert response is not None
        assert "constraint_results" in response.payload
        assert len(response.payload["constraint_results"]) == 3
        assert "overall_compliance" in response.payload



class TestCMVL:
    """Test suite for CMVL (Continuous Monitoring and Verification Loop)"""
    
    @pytest.mark.asyncio
    async def test_cmvl_trigger_activation(self, verification_agent):
        """Test CMVL activation on trigger event"""
        trigger_event = {
            "trigger_type": "market_event",
            "event_type": "volatility_spike",
            "severity": "high",
            "description": "Market volatility increased significantly",
            "source_data": {"volatility": 0.45},
            "impact_score": 0.8,
            "confidence_score": 0.9,
            "detector_agent_id": "ira_001",
            "correlation_id": "test_corr_004"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": trigger_event},
            correlation_id="test_corr_004",
            session_id="test_session_004",
            trace_id="test_trace_004"
        )
        
        response = await verification_agent.process_message(message)
        
        assert response is not None
        assert response.payload["cmvl_activated"] == True
        assert "cmvl_id" in response.payload
        assert "monitoring_frequency" in response.payload
        assert "verification_actions" in response.payload
        assert len(response.payload["verification_actions"]) > 0
        
        # Check CMVL session was created
        assert len(verification_agent.cmvl_sessions) > 0
    
    @pytest.mark.asyncio
    async def test_cmvl_critical_severity(self, verification_agent):
        """Test CMVL with critical severity trigger"""
        trigger_event = {
            "trigger_type": "market_event",
            "event_type": "market_crash",
            "severity": "critical",
            "description": "Market crash detected",
            "source_data": {"market_drop": -0.15},
            "impact_score": 1.0,
            "confidence_score": 0.95,
            "detector_agent_id": "ira_001",
            "correlation_id": "test_corr_005"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": trigger_event},
            correlation_id="test_corr_005",
            session_id="test_session_005",
            trace_id="test_trace_005"
        )
        
        response = await verification_agent.process_message(message)
        
        # Critical severity should have real-time monitoring
        assert response.payload["monitoring_frequency"] == "real_time"
        # Critical severity should NOT have auto-remediation
        assert response.payload["auto_remediation"] == False
    
    @pytest.mark.asyncio
    async def test_cmvl_medium_severity(self, verification_agent):
        """Test CMVL with medium severity trigger"""
        trigger_event = {
            "trigger_type": "market_event",
            "event_type": "interest_rate_change",
            "severity": "medium",
            "description": "Interest rate changed",
            "source_data": {"rate_change": 0.25},
            "impact_score": 0.5,
            "confidence_score": 0.9,
            "detector_agent_id": "ira_001",
            "correlation_id": "test_corr_006"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": trigger_event},
            correlation_id="test_corr_006",
            session_id="test_session_006",
            trace_id="test_trace_006"
        )
        
        response = await verification_agent.process_message(message)
        
        # Medium severity should have hourly monitoring
        assert response.payload["monitoring_frequency"] == "hourly"
        # Medium severity should have auto-remediation
        assert response.payload["auto_remediation"] == True


class TestRegulatoryCompliance:
    """Test suite for regulatory compliance checking"""
    
    @pytest.mark.asyncio
    async def test_regulatory_check(self, verification_agent):
        """Test regulatory compliance checking"""
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "regulatory_check": {
                    "rules": ["accredited_investor", "pattern_day_trader"]
                }
            },
            correlation_id="test_corr_007",
            session_id="test_session_007",
            trace_id="test_trace_007"
        )
        
        response = await verification_agent.process_message(message)
        
        assert response is not None
        assert "regulatory_results" in response.payload
        assert len(response.payload["regulatory_results"]) == 2
        assert "overall_compliance" in response.payload


class TestVerificationPerformance:
    """Test suite for verification performance"""
    
    @pytest.mark.asyncio
    async def test_verification_speed(self, verification_agent, sample_verification_request):
        """Test verification completes within acceptable time"""
        start_time = datetime.utcnow()
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": sample_verification_request},
            correlation_id="test_corr_008",
            session_id="test_session_008",
            trace_id="test_trace_008"
        )
        
        response = await verification_agent.process_message(message)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Verification should complete in under 1 second
        assert duration < 1.0
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_bulk_verification(self, verification_agent):
        """Test verification of multiple plans"""
        plans = []
        for i in range(10):
            plans.append({
                "plan_id": f"test_plan_{i}",
                "plan_steps": [
                    {
                        "step_id": f"step_{i}_1",
                        "action_type": "investment",
                        "amount": 50000,
                        "risk_level": "medium"
                    }
                ],
                "verification_level": "basic"
            })
        
        start_time = datetime.utcnow()
        
        tasks = []
        for plan in plans:
            message = AgentMessage(
                agent_id="test_sender",
                target_agent_id=verification_agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"verification_request": plan},
                correlation_id=f"test_corr_{plan['plan_id']}",
                session_id="test_session_bulk",
                trace_id="test_trace_bulk"
            )
            tasks.append(verification_agent.process_message(message))
        
        responses = await asyncio.gather(*tasks)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # All verifications should complete
        assert len(responses) == 10
        assert all(r is not None for r in responses)
        
        # Should complete in reasonable time (under 5 seconds for 10 plans)
        assert duration < 5.0


class TestVerificationHistory:
    """Test suite for verification history tracking"""
    
    @pytest.mark.asyncio
    async def test_history_tracking(self, verification_agent, sample_verification_request):
        """Test verification history is tracked"""
        initial_history_count = len(verification_agent.verification_history)
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": sample_verification_request},
            correlation_id="test_corr_009",
            session_id="test_session_009",
            trace_id="test_trace_009"
        )
        
        await verification_agent.process_message(message)
        
        # History should have one more entry
        assert len(verification_agent.verification_history) == initial_history_count + 1
        
        # Check history entry structure
        latest_entry = verification_agent.verification_history[-1]
        assert "timestamp" in latest_entry
        assert "plan_id" in latest_entry
        assert "status" in latest_entry
        assert "violations" in latest_entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
