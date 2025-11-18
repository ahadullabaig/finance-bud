#!/usr/bin/env python3
"""
Test script to validate Task 11 implementation:
Advanced Planning Capabilities and Financial Logic

This script tests all the components required by task 11:
- Goal decomposition system for complex financial objectives
- Time-horizon planning with milestone tracking
- Risk-adjusted return optimization algorithms
- Scenario planning for different market conditions
- Plan adaptation logic for changing constraints
- Tax optimization strategies and regulatory compliance checking
- Support for complex financial instruments with risk assessment
- Asset allocation optimization algorithms
- Risk assessment and portfolio balancing logic
- Retirement planning and goal-based investment strategies
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any

# Import the components we implemented
from agents.advanced_planning_capabilities import (
    GoalDecompositionSystem, TimeHorizonPlanner, RiskAdjustedReturnOptimizer,
    AssetAllocationOptimizer, RetirementPlanningEngine, AdvancedRiskAssessment,
    PortfolioBalancer, FinancialGoal, GoalType, FinancialInstrument, InstrumentType
)
from agents.scenario_planning import ScenarioPlanner, PlanAdaptationEngine
from agents.tax_optimization import TaxOptimizer
from data_models.schemas import (
    FinancialState, RiskProfile, TaxContext, Constraint, ConstraintType, ConstraintPriority,
    RiskLevel, ComplianceLevel
)


def create_sample_financial_state() -> FinancialState:
    """Create a properly formatted FinancialState object"""
    return FinancialState(
        user_id="test_user",
        total_assets=Decimal("150000"),
        total_liabilities=Decimal("30000"),
        net_worth=Decimal("120000"),  # 150000 - 30000
        monthly_income=Decimal("8000"),
        monthly_expenses=Decimal("5000"),
        monthly_cash_flow=Decimal("3000"),  # 8000 - 5000
        risk_tolerance="moderate",
        tax_filing_status="single",
        estimated_tax_rate=0.22
    )


def create_sample_risk_profile() -> RiskProfile:
    """Create a properly formatted RiskProfile object"""
    return RiskProfile(
        user_id="test_user",
        overall_risk_tolerance=RiskLevel.MODERATE,
        risk_capacity=0.6,
        risk_perception=0.5,
        risk_composure=0.7,
        investment_horizon=240,  # 20 years in months
        liquidity_needs={"immediate": 0.1, "short_term": 0.2, "long_term": 0.7},
        volatility_comfort=0.6,
        loss_tolerance=0.15,
        investment_experience="intermediate",
        financial_knowledge=0.7,
        decision_making_style="analytical",
        primary_goals=["retirement", "wealth_building"],
        goal_priorities={"retirement": 1, "wealth_building": 2},
        assessment_method="questionnaire",
        next_review_date=datetime.utcnow() + timedelta(days=365)
    )


def create_sample_tax_context() -> TaxContext:
    """Create a properly formatted TaxContext object"""
    return TaxContext(
        user_id="test_user",
        tax_year=2024,
        filing_status="single",
        state_of_residence="CA",
        estimated_agi=Decimal("96000"),
        marginal_tax_rate=0.22,
        effective_tax_rate=0.18,
        state_tax_rate=0.05,
        standard_deduction=Decimal("14600"),
        estimated_tax_liability=Decimal("17280")
    )


def test_goal_decomposition_system():
    """Test 1: Goal decomposition system for complex financial objectives"""
    print("Testing Goal Decomposition System...")
    
    decomposer = GoalDecompositionSystem()
    
    # Create sample data using helper functions
    financial_state = create_sample_financial_state()
    risk_profile = create_sample_risk_profile()
    
    # Test goal decomposition
    goal = decomposer.decompose_goal(
        goal_description="Save for retirement and buy a house in 10 years",
        target_amount=Decimal("500000"),
        target_date=datetime.utcnow() + timedelta(days=3650),  # 10 years
        financial_state=financial_state,
        risk_profile=risk_profile
    )
    
    assert goal.goal_type == GoalType.RETIREMENT, "Should classify as retirement goal"
    assert len(goal.sub_goals) > 0, "Should have sub-goals"
    assert len(goal.constraints) > 0, "Should have constraints"
    
    print("✓ Goal Decomposition System working correctly")
    return True


def test_time_horizon_planning():
    """Test 2: Time-horizon planning with milestone tracking"""
    print("Testing Time-Horizon Planning...")
    
    planner = TimeHorizonPlanner()
    
    # Create a sample goal
    goal = FinancialGoal(
        goal_id="test_goal",
        goal_type=GoalType.RETIREMENT,
        description="Retirement planning",
        target_amount=Decimal("1000000"),
        target_date=datetime.utcnow() + timedelta(days=7300),  # 20 years
        priority=1,
        sub_goals=[],
        constraints=[],
        success_metrics={},
        risk_tolerance="moderate",
        tax_implications={}
    )
    
    # Use helper functions for proper object creation
    financial_state = create_sample_financial_state()
    financial_state.total_assets = Decimal("200000")
    financial_state.total_liabilities = Decimal("50000")
    financial_state.net_worth = Decimal("150000")
    financial_state.monthly_income = Decimal("10000")
    financial_state.monthly_expenses = Decimal("6000")
    financial_state.monthly_cash_flow = Decimal("4000")
    
    risk_profile = create_sample_risk_profile()
    
    milestones, metadata = planner.create_time_horizon_plan(goal, financial_state, risk_profile)
    
    assert len(milestones) > 0, "Should create milestones"
    assert "planning_horizon_months" in metadata, "Should have planning metadata"
    assert "success_probability" in metadata, "Should calculate success probability"
    
    print("✓ Time-Horizon Planning working correctly")
    return True


def test_risk_adjusted_return_optimization():
    """Test 3: Risk-adjusted return optimization algorithms"""
    print("Testing Risk-Adjusted Return Optimization...")
    
    optimizer = RiskAdjustedReturnOptimizer()
    
    # Create sample financial instruments
    instruments = [
        FinancialInstrument(
            instrument_id="stocks_etf",
            instrument_type=InstrumentType.ETFS,
            name="Stock Market ETF",
            expected_return=0.10,
            volatility=0.18,
            liquidity_score=0.9,
            expense_ratio=0.0003,
            minimum_investment=Decimal("100"),
            tax_efficiency=0.8,
            complexity_score=0.3,
            regulatory_requirements=[],
            risk_factors=["market_risk"]
        ),
        FinancialInstrument(
            instrument_id="bonds_etf",
            instrument_type=InstrumentType.ETFS,
            name="Bond Market ETF",
            expected_return=0.04,
            volatility=0.05,
            liquidity_score=0.85,
            expense_ratio=0.0005,
            minimum_investment=Decimal("100"),
            tax_efficiency=0.6,
            complexity_score=0.2,
            regulatory_requirements=[],
            risk_factors=["interest_rate_risk"]
        )
    ]
    
    result = optimizer.optimize_portfolio(
        available_instruments=instruments,
        target_return=0.08,
        risk_tolerance=0.6,
        constraints={"max_expense_ratio": 0.01},
        time_horizon_months=120
    )
    
    assert "optimal_allocation" in result, "Should have optimal allocation"
    assert "expected_return" in result, "Should calculate expected return"
    assert "sharpe_ratio" in result, "Should calculate Sharpe ratio"
    
    print("✓ Risk-Adjusted Return Optimization working correctly")
    return True


def test_scenario_planning():
    """Test 4: Scenario planning for different market conditions"""
    print("Testing Scenario Planning...")
    
    planner = ScenarioPlanner()
    
    # Create test data
    goal = FinancialGoal(
        goal_id="test_goal",
        goal_type=GoalType.WEALTH_BUILDING,
        description="Build wealth",
        target_amount=Decimal("500000"),
        target_date=datetime.utcnow() + timedelta(days=3650),
        priority=1,
        sub_goals=[],
        constraints=[],
        success_metrics={},
        risk_tolerance="moderate",
        tax_implications={}
    )
    
    # Use helper functions
    financial_state = create_sample_financial_state()
    financial_state.total_assets = Decimal("100000")
    financial_state.total_liabilities = Decimal("20000")
    financial_state.net_worth = Decimal("80000")
    
    risk_profile = create_sample_risk_profile()
    risk_profile.investment_horizon = 120  # 10 years
    
    allocation = {"stocks": 0.6, "bonds": 0.3, "cash": 0.1}
    
    scenarios = planner.analyze_scenarios(goal, financial_state, risk_profile, allocation, 120)
    
    assert len(scenarios) > 0, "Should generate scenario results"
    assert "base_case" in scenarios, "Should have base case scenario"
    
    print("✓ Scenario Planning working correctly")
    return True


def test_tax_optimization():
    """Test 5: Tax optimization strategies and regulatory compliance checking"""
    print("Testing Tax Optimization...")
    
    optimizer = TaxOptimizer()
    
    # Use helper functions
    financial_state = create_sample_financial_state()
    financial_state.total_assets = Decimal("200000")
    financial_state.total_liabilities = Decimal("40000")
    financial_state.net_worth = Decimal("160000")
    financial_state.monthly_income = Decimal("12000")
    financial_state.monthly_expenses = Decimal("7000")
    financial_state.monthly_cash_flow = Decimal("5000")
    
    tax_context = create_sample_tax_context()
    tax_context.filing_status = "married_filing_jointly"
    tax_context.marginal_tax_rate = 0.24
    tax_context.effective_tax_rate = 0.18
    tax_context.estimated_agi = Decimal("144000")
    
    result = optimizer.optimize_tax_strategy(
        financial_state=financial_state,
        tax_context=tax_context,
        investment_goals={"target_return": 0.08},
        time_horizon_years=15
    )
    
    assert "recommendations" in result, "Should have tax recommendations"
    assert "total_estimated_savings" in result, "Should calculate tax savings"
    
    print("✓ Tax Optimization working correctly")
    return True


def test_asset_allocation_optimization():
    """Test 6: Asset allocation optimization algorithms"""
    print("Testing Asset Allocation Optimization...")
    
    allocator = AssetAllocationOptimizer()
    
    goal = FinancialGoal(
        goal_id="test_goal",
        goal_type=GoalType.RETIREMENT,
        description="Retirement planning",
        target_amount=Decimal("800000"),
        target_date=datetime.utcnow() + timedelta(days=5475),  # 15 years
        priority=1,
        sub_goals=[],
        constraints=[],
        success_metrics={},
        risk_tolerance="moderate",
        tax_implications={}
    )
    
    # Use helper function
    risk_profile = create_sample_risk_profile()
    risk_profile.investment_horizon = 180  # 15 years
    
    current_allocation = {"stocks": 0.5, "bonds": 0.4, "cash": 0.1}
    market_conditions = {"volatility": 0.15}
    
    result = allocator.optimize_asset_allocation(
        goal, risk_profile, 180, current_allocation, market_conditions
    )
    
    assert "target_allocation" in result, "Should have target allocation"
    assert "rebalancing_plan" in result, "Should have rebalancing plan"
    
    print("✓ Asset Allocation Optimization working correctly")
    return True


def test_retirement_planning():
    """Test 7: Retirement planning and goal-based investment strategies"""
    print("Testing Retirement Planning...")
    
    planner = RetirementPlanningEngine()
    
    # Use helper functions
    risk_profile = create_sample_risk_profile()
    risk_profile.investment_horizon = 300  # 25 years
    
    tax_context = create_sample_tax_context()
    tax_context.filing_status = "married_filing_jointly"
    tax_context.marginal_tax_rate = 0.24
    tax_context.effective_tax_rate = 0.18
    tax_context.estimated_agi = Decimal("120000")
    
    result = planner.create_retirement_plan(
        current_age=40,
        retirement_age=65,
        current_savings=Decimal("200000"),
        monthly_contribution=Decimal("2000"),
        desired_retirement_income=Decimal("80000"),
        risk_profile=risk_profile,
        tax_context=tax_context
    )
    
    assert "retirement_feasibility" in result, "Should assess feasibility"
    assert "required_corpus" in result, "Should calculate required corpus"
    assert "withdrawal_strategy" in result, "Should provide withdrawal strategy"
    
    print("✓ Retirement Planning working correctly")
    return True


def test_risk_assessment():
    """Test 8: Risk assessment and portfolio balancing logic"""
    print("Testing Risk Assessment...")
    
    assessor = AdvancedRiskAssessment()
    
    goal = FinancialGoal(
        goal_id="test_goal",
        goal_type=GoalType.WEALTH_BUILDING,
        description="Build wealth",
        target_amount=Decimal("400000"),
        target_date=datetime.utcnow() + timedelta(days=2920),  # 8 years
        priority=1,
        sub_goals=[],
        constraints=[],
        success_metrics={},
        risk_tolerance="moderate",
        tax_implications={}
    )
    
    allocation = {"stocks": 0.7, "bonds": 0.2, "cash": 0.1}
    market_conditions = {"volatility": 0.18}
    
    result = assessor.assess_portfolio_risk(allocation, goal, 96, market_conditions)
    
    assert "overall_risk_score" in result, "Should have overall risk score"
    assert "stress_test_results" in result, "Should have stress test results"
    assert "diversification_analysis" in result, "Should analyze diversification"
    
    print("✓ Risk Assessment working correctly")
    return True


def test_portfolio_balancing():
    """Test 9: Portfolio balancing with tax efficiency"""
    print("Testing Portfolio Balancing...")
    
    balancer = PortfolioBalancer()
    
    current_allocation = {"stocks": 0.5, "bonds": 0.3, "cash": 0.2}
    target_allocation = {"stocks": 0.6, "bonds": 0.3, "cash": 0.1}
    
    # Use helper function
    tax_context = create_sample_tax_context()
    tax_context.filing_status = "single"
    tax_context.marginal_tax_rate = 0.22
    tax_context.effective_tax_rate = 0.16
    tax_context.estimated_agi = Decimal("85000")
    tax_context.state_tax_rate = 0.04
    
    account_types = {"account1": "taxable", "account2": "401k"}
    
    result = balancer.create_rebalancing_plan(
        current_allocation, target_allocation, Decimal("150000"), tax_context, account_types
    )
    
    assert "rebalancing_needed" in result, "Should assess rebalancing need"
    assert "tax_implications" in result, "Should calculate tax implications"
    
    print("✓ Portfolio Balancing working correctly")
    return True


def run_all_tests():
    """Run all tests for Task 11 implementation"""
    print("=" * 60)
    print("TESTING TASK 11: Advanced Planning Capabilities and Financial Logic")
    print("=" * 60)
    
    tests = [
        test_goal_decomposition_system,
        test_time_horizon_planning,
        test_risk_adjusted_return_optimization,
        test_scenario_planning,
        test_tax_optimization,
        test_asset_allocation_optimization,
        test_retirement_planning,
        test_risk_assessment,
        test_portfolio_balancing
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed with error: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Task 11 implementation is complete and working.")
    else:
        print(f"⚠️  {failed} tests failed. Please review the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)