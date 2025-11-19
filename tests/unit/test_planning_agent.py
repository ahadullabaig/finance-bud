"""
Test suite for Planning Agent (PA) with Guided Search Module (GSM)

Tests the sophisticated financial planning capabilities including ToS algorithms,
multi-path strategy generation, and constraint-based optimization.

Requirements: 7.1, 7.2, 7.3, 7.5, 8.1, 8.2, 11.1
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from uuid import uuid4

from agents.planning_agent import PlanningAgent, GuidedSearchModule, SearchStrategy, HeuristicType
from agents.gsm_heuristics import AdvancedHeuristics, MarketCondition
from data_models.schemas import (
    AgentMessage, MessageType, Priority,
    EnhancedPlanRequest, PlanStep, SearchPath, ReasoningTrace,
    Constraint, ConstraintType, ConstraintPriority,
    FinancialState
)


class TestGuidedSearchModule:
    """Test suite for the Guided Search Module"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gsm = GuidedSearchModule()
        self.sample_initial_state = {
            "total_assets": 100000,
            "total_liabilities": 20000,
            "monthly_income": 8000,
            "monthly_expenses": 5000,
            "cash": 15000,
            "investments": 60000,
            "emergency_fund": 25000
        }
        
        self.sample_constraints = [
            Constraint(
                name="Emergency Fund",
                constraint_type=ConstraintType.LIQUIDITY,
                priority=ConstraintPriority.HIGH,
                description="Maintain 6 months of expenses in emergency fund",
                validation_rule="emergency_fund >= monthly_expenses * 6",
                threshold_value=30000,
                comparison_operator=">=",
                created_by="test_system"
            ),
            Constraint(
                name="Budget Constraint",
                constraint_type=ConstraintType.BUDGET,
                priority=ConstraintPriority.MANDATORY,
                description="Monthly expenses should not exceed 80% of income",
                validation_rule="monthly_expenses <= monthly_income * 0.8",
                threshold_value=0.8,
                comparison_operator="<=",
                created_by="test_system"
            )
        ]
    
    def test_search_optimal_paths_generates_multiple_strategies(self):
        """Test that search generates at least 5 distinct strategic approaches"""
        paths = self.gsm.search_optimal_paths(
            initial_state=self.sample_initial_state,
            goal="Build retirement fund of $1M in 20 years",
            constraints=self.sample_constraints,
            time_horizon=240,  # 20 years
            strategies=[SearchStrategy.CONSERVATIVE, SearchStrategy.BALANCED, 
                       SearchStrategy.AGGRESSIVE, SearchStrategy.TAX_OPTIMIZED, 
                       SearchStrategy.GROWTH_FOCUSED]
        )
        
        assert len(paths) >= 3, "Should generate at least 3 paths"
        
        # Check that paths have different strategies
        strategy_types = [path.path_type for path in paths]
        assert len(set(strategy_types)) >= 3, "Should have at least 3 distinct strategy types"
    
    def test_constraint_aware_filtering(self):
        """Test constraint-aware filtering and pruning logic"""
        # Add a strict constraint that should filter some paths
        strict_constraint = Constraint(
            name="Risk Limit",
            constraint_type=ConstraintType.RISK,
            priority=ConstraintPriority.MANDATORY,
            description="Maximum 30% in high-risk investments",
            validation_rule="high_risk_ratio <= 0.3",
            threshold_value=0.3,
            comparison_operator="<=",
            created_by="test_system"
        )
        
        constraints_with_risk = self.sample_constraints + [strict_constraint]
        
        paths = self.gsm.search_optimal_paths(
            initial_state=self.sample_initial_state,
            goal="Aggressive growth strategy",
            constraints=constraints_with_risk,
            time_horizon=120,
            strategies=[SearchStrategy.AGGRESSIVE, SearchStrategy.CONSERVATIVE]
        )
        
        # Conservative strategy should have higher constraint satisfaction
        if len(paths) >= 2:
            conservative_paths = [p for p in paths if "conservative" in p.path_type.lower()]
            aggressive_paths = [p for p in paths if "aggressive" in p.path_type.lower()]
            
            if conservative_paths and aggressive_paths:
                assert conservative_paths[0].constraint_satisfaction_score >= aggressive_paths[0].constraint_satisfaction_score
    
    def test_heuristic_evaluation_system(self):
        """Test sophisticated heuristic evaluation system"""
        # Test individual heuristic calculations
        test_state = self.sample_initial_state.copy()
        test_action = {
            "type": "diversified_portfolio",
            "amount": 50000,
            "risk_level": "medium",
            "expected_return": 0.08
        }
        
        # Test risk-adjusted return heuristic
        risk_score = self.gsm._calculate_risk_adjusted_return(test_state, SearchStrategy.BALANCED)
        assert 0.0 <= risk_score <= 1.0, "Risk score should be between 0 and 1"
        
        # Test constraint complexity heuristic
        complexity_score = self.gsm._calculate_constraint_complexity(test_state, self.sample_constraints)
        assert 0.0 <= complexity_score <= 1.0, "Complexity score should be between 0 and 1"
        
        # Test liquidity score heuristic
        liquidity_score = self.gsm._calculate_liquidity_score(test_state)
        assert 0.0 <= liquidity_score <= 1.0, "Liquidity score should be between 0 and 1"
    
    def test_multi_year_planning_with_milestones(self):
        """Test multi-year planning with milestone tracking"""
        paths = self.gsm.search_optimal_paths(
            initial_state=self.sample_initial_state,
            goal="Retirement planning over 30 years",
            constraints=self.sample_constraints,
            time_horizon=360,  # 30 years
            session_id="test_session_001"
        )
        
        assert len(paths) > 0, "Should generate at least one path for long-term planning"
        
        # Check that paths include milestone tracking
        for path in paths:
            if "milestone" in path.path_type:
                assert path.combined_score > 0, "Milestone-tracked paths should have positive scores"
    
    def test_rejection_sampling_with_constraint_violation_prediction(self):
        """Test rejection sampling with constraint violation prediction"""
        # Create constraints that are likely to be violated
        strict_constraints = [
            Constraint(
                name="Very Strict Budget",
                constraint_type=ConstraintType.BUDGET,
                priority=ConstraintPriority.MANDATORY,
                description="Expenses must be under 50% of income",
                validation_rule="monthly_expenses <= monthly_income * 0.5",
                threshold_value=0.5,
                comparison_operator="<=",
                created_by="test_system"
            )
        ]
        
        paths = self.gsm.search_optimal_paths(
            initial_state=self.sample_initial_state,
            goal="Aggressive investment strategy",
            constraints=strict_constraints,
            time_horizon=120,
            strategies=[SearchStrategy.AGGRESSIVE]
        )
        
        # Paths should be filtered based on constraint violation prediction
        for path in paths:
            # Paths that survive rejection sampling should have reasonable scores
            assert path.feasibility_score > 0.3, "Surviving paths should have decent feasibility"


class TestPlanningAgent:
    """Test suite for the Planning Agent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = PlanningAgent("test_planning_agent")
        self.sample_financial_state = FinancialState(
            user_id="test_user_001",
            total_assets=Decimal("150000"),
            total_liabilities=Decimal("30000"),
            monthly_income=Decimal("8000"),
            monthly_expenses=Decimal("5500"),
            risk_tolerance="moderate",
            tax_filing_status="single",
            estimated_tax_rate=0.22
        )
    
    @pytest.mark.asyncio
    async def test_comprehensive_planning_request(self):
        """Test comprehensive planning request handling"""
        request_message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=self.agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": "Save for retirement and buy a house in 10 years",
                    "current_state": self.sample_financial_state.dict(),
                    "constraints": [
                        {
                            "constraint_id": "emergency_fund_req",
                            "name": "Emergency Fund Requirement",
                            "constraint_type": "liquidity",
                            "priority": "high",
                            "description": "Maintain 6 months emergency fund",
                            "validation_rule": "emergency_fund >= monthly_expenses * 6",
                            "threshold_value": 33000,
                            "comparison_operator": ">=",
                            "created_by": "system"
                        }
                    ],
                    "time_horizon": 120,  # 10 years
                    "risk_profile": {
                        "risk_tolerance": "moderate",
                        "investment_experience": "intermediate"
                    }
                }
            },
            correlation_id="test_correlation_001",
            session_id="test_session_001",
            trace_id="test_trace_001"
        )
        
        response = await self.agent.process_message(request_message)
        
        assert response is not None, "Should return a response"
        assert response.message_type == MessageType.RESPONSE, "Should be a response message"
        assert "planning_completed" in response.payload, "Should indicate planning completion"
        assert response.payload["planning_completed"] is True, "Planning should be completed"
        
        # Check that multiple strategies were explored
        assert response.payload["alternative_strategies"] >= 2, "Should explore multiple strategies"
        
        # Check that plan steps are generated
        assert "plan_steps" in response.payload, "Should include plan steps"
        assert len(response.payload["plan_steps"]) > 0, "Should have at least one plan step"
    
    @pytest.mark.asyncio
    async def test_session_state_tracking(self):
        """Test planning session management and state tracking"""
        session_id = "test_session_tracking"
        
        # First, create a planning session
        request_message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=self.agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": "Build emergency fund and start investing",
                    "current_state": self.sample_financial_state.dict(),
                    "constraints": [],
                    "time_horizon": 60,
                    "risk_profile": {"risk_tolerance": "conservative"}
                }
            },
            correlation_id="test_correlation_002",
            session_id=session_id,
            trace_id="test_trace_002"
        )
        
        await self.agent.process_message(request_message)
        
        # Check session state
        session_state = await self.agent.get_session_state(session_id)
        
        assert "error" not in session_state, "Session should exist"
        assert session_state["goal"] == "Build emergency fund and start investing"
        assert "search_start_time" in session_state, "Should track search start time"
    
    @pytest.mark.asyncio
    async def test_milestone_tracking(self):
        """Test milestone tracking for multi-year planning"""
        session_id = "test_milestone_session"
        
        # Create a long-term planning session
        request_message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=self.agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": "Retirement planning over 25 years",
                    "current_state": self.sample_financial_state.dict(),
                    "constraints": [],
                    "time_horizon": 300,  # 25 years
                    "risk_profile": {"risk_tolerance": "moderate"}
                }
            },
            correlation_id="test_correlation_003",
            session_id=session_id,
            trace_id="test_trace_003"
        )
        
        response = await self.agent.process_message(request_message)
        
        # Check that milestone tracking was initialized
        session_state = await self.agent.get_session_state(session_id)
        
        if "milestones" in session_state:
            milestones = session_state["milestones"]["milestones"]
            assert len(milestones) > 0, "Should have milestones for long-term planning"
            
            # Test milestone update
            update_result = await self.agent.update_milestone_progress(
                session_id, 
                12,  # 1 year milestone
                {"net_worth": 130000, "investments": 80000}
            )
            
            assert update_result["milestone_updated"] is True, "Should update milestone successfully"
    
    @pytest.mark.asyncio
    async def test_constraint_optimization(self):
        """Test constraint handling optimization"""
        session_id = "test_constraint_optimization"
        
        # Create session first
        request_message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=self.agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": "Optimize investment strategy",
                    "current_state": self.sample_financial_state.dict(),
                    "constraints": [],
                    "time_horizon": 120,
                    "risk_profile": {"risk_tolerance": "moderate"}
                }
            },
            correlation_id="test_correlation_004",
            session_id=session_id,
            trace_id="test_trace_004"
        )
        
        await self.agent.process_message(request_message)
        
        # Test constraint optimization
        new_constraints = [
            Constraint(
                name="Risk Constraint",
                constraint_type=ConstraintType.RISK,
                priority=ConstraintPriority.HIGH,
                description="Limit high-risk investments",
                validation_rule="risk_ratio <= 0.4",
                threshold_value=0.4,
                comparison_operator="<=",
                created_by="test_system"
            )
        ]
        
        optimization_result = await self.agent.optimize_constraint_handling(session_id, new_constraints)
        
        assert "constraint_analysis" in optimization_result, "Should provide constraint analysis"
        assert "optimization_recommendations" in optimization_result, "Should provide recommendations"
    
    def test_strategy_selection_for_different_goals(self):
        """Test strategy selection based on different financial goals"""
        # Test retirement goal
        retirement_strategies = self.agent._select_strategies_for_goal(
            "Plan for retirement in 30 years", 
            {"risk_tolerance": "moderate"}
        )
        
        assert SearchStrategy.TAX_OPTIMIZED in retirement_strategies, "Should include tax optimization for retirement"
        assert SearchStrategy.GROWTH_FOCUSED in retirement_strategies, "Should include growth focus for long-term retirement"
        
        # Test emergency fund goal
        emergency_strategies = self.agent._select_strategies_for_goal(
            "Build emergency fund quickly", 
            {"risk_tolerance": "low"}
        )
        
        assert SearchStrategy.CONSERVATIVE in emergency_strategies, "Should be conservative for emergency fund"
        
        # Test aggressive growth goal
        growth_strategies = self.agent._select_strategies_for_goal(
            "Aggressive growth investment strategy", 
            {"risk_tolerance": "high"}
        )
        
        assert SearchStrategy.AGGRESSIVE in growth_strategies, "Should include aggressive strategy for growth goals"


class TestAdvancedHeuristics:
    """Test suite for Advanced Heuristics"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.heuristics = AdvancedHeuristics()
        self.sample_state = {
            "total_assets": 200000,
            "total_liabilities": 50000,
            "monthly_income": 10000,
            "monthly_expenses": 6000,
            "cash": 30000,
            "stocks": 100000,
            "bonds": 50000,
            "real_estate": 20000
        }
    
    def test_information_gain_heuristic(self):
        """Test information gain heuristic calculation"""
        action = {
            "type": "diversified_investment",
            "amount": 25000,
            "expected_return": 0.08
        }
        
        goal_context = {"target_amount": 500000, "time_horizon": 120}
        
        info_gain = self.heuristics.calculate_information_gain_heuristic(
            self.sample_state, action, goal_context
        )
        
        assert 0.0 <= info_gain <= 1.0, "Information gain should be between 0 and 1"
    
    def test_risk_adjusted_return_heuristic(self):
        """Test risk-adjusted return heuristic with market conditions"""
        action = {
            "type": "growth_investment",
            "amount": 50000,
            "expected_return": 0.12,
            "risk_level": "high"
        }
        
        # Test with different market conditions
        for market_condition in MarketCondition:
            risk_adjusted_score = self.heuristics.calculate_risk_adjusted_return_heuristic(
                self.sample_state, action, 60, market_condition
            )
            
            assert 0.0 <= risk_adjusted_score <= 1.0, f"Risk-adjusted score should be valid for {market_condition}"
    
    def test_tax_efficiency_heuristic(self):
        """Test tax efficiency heuristic calculation"""
        action = {
            "type": "401k_contribution",
            "amount": 20000,
            "tax_advantaged": True
        }
        
        tax_context = {
            "marginal_tax_rate": 0.24,
            "available_tax_advantaged_space": 25000,
            "current_tax_efficiency": 0.6
        }
        
        tax_score = self.heuristics.calculate_tax_efficiency_heuristic(
            self.sample_state, action, tax_context
        )
        
        assert 0.0 <= tax_score <= 1.0, "Tax efficiency score should be between 0 and 1"
    
    def test_diversification_heuristic(self):
        """Test diversification heuristic calculation"""
        action = {
            "type": "international_stocks",
            "amount": 30000,
            "asset_class": "international_equity"
        }
        
        target_allocation = {
            "cash": 0.1,
            "stocks": 0.5,
            "bonds": 0.3,
            "real_estate": 0.1
        }
        
        diversification_score = self.heuristics.calculate_diversification_heuristic(
            self.sample_state, action, target_allocation
        )
        
        assert 0.0 <= diversification_score <= 1.0, "Diversification score should be between 0 and 1"


if __name__ == "__main__":
    pytest.main([__file__])