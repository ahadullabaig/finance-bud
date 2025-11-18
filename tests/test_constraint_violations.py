"""
Constraint Violation Test Scenarios

Comprehensive tests for constraint violation detection and handling.
Task 15 - Person D
Requirements: 11.1, 11.4, 12.4
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime

from agents.verifier import VerificationAgent
from data_models.schemas import (
    AgentMessage, MessageType, ConstraintType, ConstraintPriority
)


@pytest.fixture
def verification_agent():
    """Create verification agent for testing"""
    return VerificationAgent("test_va_constraints")


class TestEmergencyFundConstraints:
    """Test emergency fund constraint violations"""
    
    @pytest.mark.asyncio
    async def test_insufficient_emergency_fund(self, verification_agent):
        """Test detection of insufficient emergency fund"""
        plan = {
            "plan_id": "test_emergency_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "high_risk_investment",
                    "amount": 100000,
                    "risk_level": "high",
                    "emergency_fund_remaining": 5000,  # Below 3 months expenses
                    "monthly_expenses": 4000
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan},
            correlation_id="test_corr_001",
            session_id="test_session_001",
            trace_id="test_trace_001"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should detect emergency fund violation
        assert len(step_results[0]["violations"]) > 0
        violations = [v for v in step_results[0]["violations"] 
                     if "emergency" in v["constraint_name"].lower()]
        assert len(violations) > 0
        assert violations[0]["severity"] in ["high", "critical"]


class TestDebtConstraints:
    """Test debt-related constraint violations"""
    
    @pytest.mark.asyncio
    async def test_excessive_debt_to_income(self, verification_agent):
        """Test detection of excessive debt-to-income ratio"""
        plan = {
            "plan_id": "test_debt_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "new_loan",
                    "amount": 50000,
                    "monthly_payment": 2000,
                    "monthly_income": 5000,  # 40% DTI - exceeds 36% limit
                    "existing_debt_payments": 0
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan},
            correlation_id="test_corr_002",
            session_id="test_session_002",
            trace_id="test_trace_002"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should detect DTI violation
        violations = [v for v in step_results[0]["violations"] 
                     if "debt" in v["constraint_name"].lower()]
        assert len(violations) > 0


class TestRiskConstraints:
    """Test risk-related constraint violations"""
    
    @pytest.mark.asyncio
    async def test_excessive_portfolio_risk(self, verification_agent):
        """Test detection of excessive portfolio risk"""
        plan = {
            "plan_id": "test_risk_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "high_risk_investment",
                    "amount": 200000,
                    "risk_level": "high",
                    "portfolio_risk_score": 0.85,  # Very high risk
                    "user_risk_tolerance": "conservative"
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan},
            correlation_id="test_corr_003",
            session_id="test_session_003",
            trace_id="test_trace_003"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should detect risk tolerance violation
        violations = [v for v in step_results[0]["violations"] 
                     if "risk" in v["constraint_name"].lower()]
        assert len(violations) > 0


class TestLiquidityConstraints:
    """Test liquidity constraint violations"""
    
    @pytest.mark.asyncio
    async def test_insufficient_liquidity(self, verification_agent):
        """Test detection of insufficient liquidity"""
        plan = {
            "plan_id": "test_liquidity_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "illiquid_investment",
                    "amount": 150000,
                    "liquidity_score": 0.1,  # Very illiquid
                    "total_liquid_assets": 50000,
                    "required_liquidity": 100000
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan},
            correlation_id="test_corr_004",
            session_id="test_session_004",
            trace_id="test_trace_004"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should detect liquidity violation
        violations = [v for v in step_results[0]["violations"] 
                     if "liquidity" in v["constraint_name"].lower()]
        assert len(violations) > 0


class TestMultipleConstraintViolations:
    """Test scenarios with multiple constraint violations"""
    
    @pytest.mark.asyncio
    async def test_multiple_violations_single_step(self, verification_agent):
        """Test detection of multiple violations in single step"""
        plan = {
            "plan_id": "test_multi_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "risky_leveraged_investment",
                    "amount": 300000,
                    "risk_level": "high",
                    "emergency_fund_remaining": 2000,
                    "monthly_expenses": 5000,
                    "portfolio_risk_score": 0.9,
                    "user_risk_tolerance": "conservative",
                    "debt_to_income_ratio": 0.45
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan},
            correlation_id="test_corr_005",
            session_id="test_session_005",
            trace_id="test_trace_005"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should detect multiple violations
        assert len(step_results[0]["violations"]) >= 3
        
        # Check for different violation types
        violation_types = set(v["constraint_type"] for v in step_results[0]["violations"])
        assert len(violation_types) >= 2


class TestConstraintPriority:
    """Test constraint priority handling"""
    
    @pytest.mark.asyncio
    async def test_critical_constraint_rejection(self, verification_agent):
        """Test that critical constraints cause immediate rejection"""
        plan = {
            "plan_id": "test_priority_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "investment",
                    "amount": 50000,
                    "violates_critical_constraint": True,
                    "emergency_fund_remaining": 0  # Critical violation
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan},
            correlation_id="test_corr_006",
            session_id="test_session_006",
            trace_id="test_trace_006"
        )
        
        response = await verification_agent.process_message(message)
        report = response.payload["verification_report"]
        
        # Plan should be rejected due to critical violation
        assert report["verification_status"] == "rejected"


class TestConstraintRecovery:
    """Test constraint violation recovery suggestions"""
    
    @pytest.mark.asyncio
    async def test_recovery_suggestions(self, verification_agent):
        """Test that violations include recovery suggestions"""
        plan = {
            "plan_id": "test_recovery_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "investment",
                    "amount": 100000,
                    "emergency_fund_remaining": 8000,
                    "monthly_expenses": 4000
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan},
            correlation_id="test_corr_007",
            session_id="test_session_007",
            trace_id="test_trace_007"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        if len(step_results[0]["violations"]) > 0:
            # Violations should include suggestions
            for violation in step_results[0]["violations"]:
                assert "suggestion" in violation or "remediation" in violation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
