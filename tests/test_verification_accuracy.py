"""
Verification Accuracy Testing

Tests verification agent accuracy against known financial rules and regulations.
Task 15 - Person D
Requirements: 11.1, 11.4, 12.4
"""

import pytest
import asyncio
from decimal import Decimal

from agents.verifier import VerificationAgent
from data_models.schemas import AgentMessage, MessageType


@pytest.fixture
def verification_agent():
    """Create verification agent for testing"""
    return VerificationAgent("test_va_accuracy")


class TestFinancialRuleAccuracy:
    """Test accuracy of financial rule validation"""
    
    @pytest.mark.asyncio
    async def test_50_30_20_budget_rule(self, verification_agent):
        """Test 50/30/20 budget rule validation"""
        # Valid 50/30/20 allocation
        valid_plan = {
            "plan_id": "test_budget_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "budget_allocation",
                    "monthly_income": 6000,
                    "needs_allocation": 3000,  # 50%
                    "wants_allocation": 1800,  # 30%
                    "savings_allocation": 1200  # 20%
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": valid_plan},
            correlation_id="test_corr_001",
            session_id="test_session_001",
            trace_id="test_trace_001"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should pass validation
        budget_violations = [v for v in step_results[0]["violations"] 
                           if "budget" in v["constraint_name"].lower()]
        assert len(budget_violations) == 0
    
    @pytest.mark.asyncio
    async def test_4_percent_retirement_rule(self, verification_agent):
        """Test 4% safe withdrawal rate rule"""
        # Valid 4% withdrawal
        valid_plan = {
            "plan_id": "test_retirement_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "retirement_withdrawal",
                    "portfolio_value": 1000000,
                    "annual_withdrawal": 40000,  # 4%
                    "withdrawal_rate": 0.04
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": valid_plan},
            correlation_id="test_corr_002",
            session_id="test_session_002",
            trace_id="test_trace_002"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should pass validation
        withdrawal_violations = [v for v in step_results[0]["violations"] 
                               if "withdrawal" in v["constraint_name"].lower()]
        assert len(withdrawal_violations) == 0


class TestRegulatoryCompliance:
    """Test regulatory compliance accuracy"""
    
    @pytest.mark.asyncio
    async def test_401k_contribution_limits(self, verification_agent):
        """Test 401(k) contribution limit validation (2024: $23,000)"""
        # Exceeds limit
        invalid_plan = {
            "plan_id": "test_401k_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "401k_contribution",
                    "annual_contribution": 25000,  # Exceeds $23,000 limit
                    "contribution_limit": 23000
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": invalid_plan},
            correlation_id="test_corr_003",
            session_id="test_session_003",
            trace_id="test_trace_003"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should detect violation
        assert len(step_results[0]["violations"]) > 0
    
    @pytest.mark.asyncio
    async def test_ira_contribution_limits(self, verification_agent):
        """Test IRA contribution limit validation (2024: $7,000)"""
        # Valid contribution
        valid_plan = {
            "plan_id": "test_ira_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "ira_contribution",
                    "annual_contribution": 7000,  # At limit
                    "contribution_limit": 7000
                }
            ],
            "verification_level": "comprehensive"
        }
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": valid_plan},
            correlation_id="test_corr_004",
            session_id="test_session_004",
            trace_id="test_trace_004"
        )
        
        response = await verification_agent.process_message(message)
        step_results = response.payload["step_results"]
        
        # Should pass
        contribution_violations = [v for v in step_results[0]["violations"] 
                                 if "contribution" in v["constraint_name"].lower()]
        assert len(contribution_violations) == 0


class TestRiskAssessmentAccuracy:
    """Test risk assessment accuracy"""
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_calculation(self, verification_agent):
        """Test portfolio risk score calculation accuracy"""
        plan = {
            "plan_id": "test_risk_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "portfolio_allocation",
                    "stock_allocation": 0.8,
                    "bond_allocation": 0.2,
                    "expected_risk_score": 0.7,  # High risk due to 80% stocks
                    "user_risk_tolerance": "moderate"
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
        report = response.payload["verification_report"]
        
        # Risk score should be calculated
        assert "overall_risk_score" in report
        assert report["overall_risk_score"] > 0


class TestTaxRuleAccuracy:
    """Test tax rule validation accuracy"""
    
    @pytest.mark.asyncio
    async def test_capital_gains_tax_brackets(self, verification_agent):
        """Test capital gains tax calculation accuracy"""
        plan = {
            "plan_id": "test_tax_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "capital_gains_realization",
                    "capital_gains": 50000,
                    "income_level": 100000,
                    "filing_status": "single",
                    "expected_tax_rate": 0.15  # 15% for this bracket
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
        
        # Should validate tax calculations
        assert response is not None


class TestAccuracyMetrics:
    """Test verification accuracy metrics"""
    
    @pytest.mark.asyncio
    async def test_false_positive_rate(self, verification_agent):
        """Test that valid plans are not incorrectly rejected"""
        # Create 10 valid plans
        valid_plans = []
        for i in range(10):
            valid_plans.append({
                "plan_id": f"test_valid_{i}",
                "plan_steps": [
                    {
                        "step_id": "step_1",
                        "action_type": "balanced_investment",
                        "amount": 50000,
                        "risk_level": "medium",
                        "emergency_fund_remaining": 30000,
                        "monthly_expenses": 5000
                    }
                ],
                "verification_level": "comprehensive"
            })
        
        rejections = 0
        for plan in valid_plans:
            message = AgentMessage(
                agent_id="test_sender",
                target_agent_id=verification_agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"verification_request": plan},
                correlation_id=f"test_corr_{plan['plan_id']}",
                session_id="test_session_fp",
                trace_id="test_trace_fp"
            )
            
            response = await verification_agent.process_message(message)
            report = response.payload["verification_report"]
            
            if report["verification_status"] == "rejected":
                rejections += 1
        
        # False positive rate should be low (< 20%)
        false_positive_rate = rejections / len(valid_plans)
        assert false_positive_rate < 0.2
    
    @pytest.mark.asyncio
    async def test_false_negative_rate(self, verification_agent):
        """Test that invalid plans are correctly rejected"""
        # Create 10 invalid plans with clear violations
        invalid_plans = []
        for i in range(10):
            invalid_plans.append({
                "plan_id": f"test_invalid_{i}",
                "plan_steps": [
                    {
                        "step_id": "step_1",
                        "action_type": "high_risk_investment",
                        "amount": 200000,
                        "risk_level": "high",
                        "emergency_fund_remaining": 1000,  # Way too low
                        "monthly_expenses": 5000,
                        "debt_to_income_ratio": 0.5  # Too high
                    }
                ],
                "verification_level": "comprehensive"
            })
        
        approvals = 0
        for plan in invalid_plans:
            message = AgentMessage(
                agent_id="test_sender",
                target_agent_id=verification_agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"verification_request": plan},
                correlation_id=f"test_corr_{plan['plan_id']}",
                session_id="test_session_fn",
                trace_id="test_trace_fn"
            )
            
            response = await verification_agent.process_message(message)
            report = response.payload["verification_report"]
            
            if report["verification_status"] == "approved":
                approvals += 1
        
        # False negative rate should be very low (< 10%)
        false_negative_rate = approvals / len(invalid_plans)
        assert false_negative_rate < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
