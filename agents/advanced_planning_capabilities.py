"""
Advanced Planning Capabilities and Financial Logic Module

Implements sophisticated financial planning features including:
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

Requirements: 1.2, 7.4, 8.1
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import numpy as np
from uuid import uuid4

from data_models.schemas import (
    Constraint, ConstraintType, ConstraintPriority,
    RiskProfile, TaxContext, RegulatoryRequirement,
    FinancialState, PlanStep
)


class GoalType(str, Enum):
    """Types of financial goals"""
    RETIREMENT = "retirement"
    EMERGENCY_FUND = "emergency_fund"
    HOME_PURCHASE = "home_purchase"
    EDUCATION = "education"
    DEBT_PAYOFF = "debt_payoff"
    WEALTH_BUILDING = "wealth_building"
    INCOME_REPLACEMENT = "income_replacement"
    TAX_OPTIMIZATION = "tax_optimization"


class MarketScenario(str, Enum):
    """Market scenario types for planning"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    VOLATILE_MARKET = "volatile_market"
    RECESSION = "recession"
    INFLATION_SPIKE = "inflation_spike"
    INTEREST_RATE_RISE = "interest_rate_rise"
    NORMAL_CONDITIONS = "normal_conditions"


class InstrumentType(str, Enum):
    """Types of financial instruments"""
    STOCKS = "stocks"
    BONDS = "bonds"
    MUTUAL_FUNDS = "mutual_funds"
    ETFS = "etfs"
    OPTIONS = "options"
    FUTURES = "futures"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    CRYPTOCURRENCY = "cryptocurrency"
    CASH_EQUIVALENTS = "cash_equivalents"


@dataclass
class FinancialGoal:
    """Represents a decomposed financial goal"""
    goal_id: str
    goal_type: GoalType
    description: str
    target_amount: Decimal
    target_date: datetime
    priority: int  # 1 = highest priority
    sub_goals: List['FinancialGoal']
    constraints: List[Constraint]
    success_metrics: Dict[str, Any]
    risk_tolerance: str
    tax_implications: Dict[str, Any]


@dataclass
class Milestone:
    """Represents a planning milestone"""
    milestone_id: str
    target_date: datetime
    target_metrics: Dict[str, Decimal]
    success_criteria: List[str]
    risk_checkpoints: List[str]
    adjustment_triggers: List[str]
    status: str = "planned"


@dataclass
class MarketScenarioData:
    """Market scenario parameters for planning"""
    scenario_name: MarketScenario
    expected_returns: Dict[str, float]  # Asset class returns
    volatilities: Dict[str, float]     # Asset class volatilities
    correlations: Dict[Tuple[str, str], float]  # Asset correlations
    duration_months: int
    probability: float
    economic_indicators: Dict[str, float]


@dataclass
class FinancialInstrument:
    """Represents a financial instrument with risk assessment"""
    instrument_id: str
    instrument_type: InstrumentType
    name: str
    expected_return: float
    volatility: float
    liquidity_score: float  # 0-1, 1 = most liquid
    expense_ratio: float
    minimum_investment: Decimal
    tax_efficiency: float  # 0-1, 1 = most tax efficient
    complexity_score: float  # 0-1, 1 = most complex
    regulatory_requirements: List[str]
    risk_factors: List[str]


class GoalDecompositionSystem:
    """
    Advanced goal decomposition system for complex financial objectives.
    Breaks down high-level goals into actionable sub-goals with constraints.
    """
    
    def __init__(self):
        self.goal_templates = self._initialize_goal_templates()
        self.decomposition_rules = self._initialize_decomposition_rules()
    
    def decompose_goal(
        self, 
        goal_description: str, 
        target_amount: Decimal,
        target_date: datetime,
        financial_state: FinancialState,
        risk_profile: RiskProfile
    ) -> FinancialGoal:
        """
        Decompose a complex financial goal into actionable sub-goals.
        
        Args:
            goal_description: Natural language description of the goal
            target_amount: Target monetary amount
            target_date: Target completion date
            financial_state: Current financial state
            risk_profile: User's risk profile
            
        Returns:
            FinancialGoal: Decomposed goal with sub-goals and constraints
        """
        # Classify the goal type
        goal_type = self._classify_goal_type(goal_description)
        
        # Create main goal
        main_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=goal_type,
            description=goal_description,
            target_amount=target_amount,
            target_date=target_date,
            priority=1,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance=risk_profile.overall_risk_tolerance.value,
            tax_implications={}
        )
        
        # Apply decomposition rules based on goal type
        if goal_type == GoalType.RETIREMENT:
            main_goal = self._decompose_retirement_goal(main_goal, financial_state, risk_profile)
        elif goal_type == GoalType.HOME_PURCHASE:
            main_goal = self._decompose_home_purchase_goal(main_goal, financial_state, risk_profile)
        elif goal_type == GoalType.EMERGENCY_FUND:
            main_goal = self._decompose_emergency_fund_goal(main_goal, financial_state, risk_profile)
        elif goal_type == GoalType.DEBT_PAYOFF:
            main_goal = self._decompose_debt_payoff_goal(main_goal, financial_state, risk_profile)
        else:
            main_goal = self._decompose_generic_goal(main_goal, financial_state, risk_profile)
        
        # Add cross-cutting constraints
        main_goal.constraints.extend(self._generate_cross_cutting_constraints(financial_state, risk_profile))
        
        # Generate success metrics
        main_goal.success_metrics = self._generate_success_metrics(main_goal, financial_state)
        
        return main_goal
    
    def _classify_goal_type(self, goal_description: str) -> GoalType:
        """Classify goal type from natural language description"""
        description_lower = goal_description.lower()
        
        if any(word in description_lower for word in ["retirement", "retire", "pension"]):
            return GoalType.RETIREMENT
        elif any(word in description_lower for word in ["house", "home", "property", "mortgage"]):
            return GoalType.HOME_PURCHASE
        elif any(word in description_lower for word in ["emergency", "safety", "buffer"]):
            return GoalType.EMERGENCY_FUND
        elif any(word in description_lower for word in ["education", "college", "school", "tuition"]):
            return GoalType.EDUCATION
        elif any(word in description_lower for word in ["debt", "loan", "payoff", "pay off"]):
            return GoalType.DEBT_PAYOFF
        elif any(word in description_lower for word in ["tax", "deduction", "optimize"]):
            return GoalType.TAX_OPTIMIZATION
        elif any(word in description_lower for word in ["income", "replacement", "passive"]):
            return GoalType.INCOME_REPLACEMENT
        else:
            return GoalType.WEALTH_BUILDING
    
    def _decompose_retirement_goal(
        self, 
        main_goal: FinancialGoal, 
        financial_state: FinancialState, 
        risk_profile: RiskProfile
    ) -> FinancialGoal:
        """Decompose retirement goal into sub-goals"""
        years_to_retirement = (main_goal.target_date - datetime.utcnow()).days / 365.25
        
        # Sub-goal 1: Maximize tax-advantaged accounts
        tax_advantaged_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.TAX_OPTIMIZATION,
            description="Maximize contributions to tax-advantaged retirement accounts",
            target_amount=Decimal("22500") * Decimal(str(years_to_retirement)),  # 401k max
            target_date=main_goal.target_date,
            priority=1,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance=risk_profile.overall_risk_tolerance.value,
            tax_implications={"tax_deferred": True, "contribution_limit": 22500}
        )
        
        # Sub-goal 2: Build diversified investment portfolio
        portfolio_target = main_goal.target_amount * Decimal("0.7")  # 70% in investments
        portfolio_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.WEALTH_BUILDING,
            description="Build diversified investment portfolio for retirement",
            target_amount=portfolio_target,
            target_date=main_goal.target_date,
            priority=2,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance=risk_profile.overall_risk_tolerance.value,
            tax_implications={"capital_gains": True, "dividend_income": True}
        )
        
        # Sub-goal 3: Ensure adequate emergency fund
        emergency_target = financial_state.monthly_expenses * 12  # 1 year expenses
        emergency_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.EMERGENCY_FUND,
            description="Maintain emergency fund for retirement security",
            target_amount=emergency_target,
            target_date=datetime.utcnow() + timedelta(days=365),  # Within 1 year
            priority=1,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance="conservative",
            tax_implications={}
        )
        
        main_goal.sub_goals = [tax_advantaged_goal, portfolio_goal, emergency_goal]
        return main_goal
    
    def _decompose_home_purchase_goal(
        self, 
        main_goal: FinancialGoal, 
        financial_state: FinancialState, 
        risk_profile: RiskProfile
    ) -> FinancialGoal:
        """Decompose home purchase goal into sub-goals"""
        # Sub-goal 1: Down payment (20% of home price)
        down_payment_target = main_goal.target_amount * Decimal("0.2")
        down_payment_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.WEALTH_BUILDING,
            description="Save for down payment",
            target_amount=down_payment_target,
            target_date=main_goal.target_date,
            priority=1,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance="conservative",  # Short-term goal should be conservative
            tax_implications={}
        )
        
        # Sub-goal 2: Closing costs and fees (3-5% of home price)
        closing_costs_target = main_goal.target_amount * Decimal("0.04")
        closing_costs_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.WEALTH_BUILDING,
            description="Save for closing costs and fees",
            target_amount=closing_costs_target,
            target_date=main_goal.target_date,
            priority=2,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance="conservative",
            tax_implications={}
        )
        
        # Sub-goal 3: Improve credit score if needed
        credit_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.DEBT_PAYOFF,
            description="Optimize credit score for better mortgage rates",
            target_amount=Decimal("0"),  # Non-monetary goal
            target_date=main_goal.target_date - timedelta(days=180),  # 6 months before
            priority=1,
            sub_goals=[],
            constraints=[],
            success_metrics={"target_credit_score": 740},
            risk_tolerance="conservative",
            tax_implications={}
        )
        
        main_goal.sub_goals = [down_payment_goal, closing_costs_goal, credit_goal]
        return main_goal
    
    def _decompose_emergency_fund_goal(
        self, 
        main_goal: FinancialGoal, 
        financial_state: FinancialState, 
        risk_profile: RiskProfile
    ) -> FinancialGoal:
        """Decompose emergency fund goal into sub-goals"""
        # Calculate target based on monthly expenses
        if main_goal.target_amount == 0:
            main_goal.target_amount = financial_state.monthly_expenses * 6  # 6 months default
        
        # Sub-goal 1: Immediate emergency buffer (1 month expenses)
        immediate_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.EMERGENCY_FUND,
            description="Build immediate emergency buffer",
            target_amount=financial_state.monthly_expenses,
            target_date=datetime.utcnow() + timedelta(days=30),
            priority=1,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance="conservative",
            tax_implications={}
        )
        
        # Sub-goal 2: Full emergency fund
        full_emergency_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.EMERGENCY_FUND,
            description="Complete full emergency fund",
            target_amount=main_goal.target_amount,
            target_date=main_goal.target_date,
            priority=2,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance="conservative",
            tax_implications={}
        )
        
        main_goal.sub_goals = [immediate_goal, full_emergency_goal]
        return main_goal
    
    def _decompose_debt_payoff_goal(
        self, 
        main_goal: FinancialGoal, 
        financial_state: FinancialState, 
        risk_profile: RiskProfile
    ) -> FinancialGoal:
        """Decompose debt payoff goal into sub-goals"""
        # Sub-goal 1: High-interest debt first
        high_interest_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.DEBT_PAYOFF,
            description="Pay off high-interest debt first",
            target_amount=main_goal.target_amount * Decimal("0.6"),  # Assume 60% is high-interest
            target_date=main_goal.target_date - timedelta(days=180),
            priority=1,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance="conservative",
            tax_implications={"interest_deduction": False}
        )
        
        # Sub-goal 2: Remaining debt
        remaining_debt_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.DEBT_PAYOFF,
            description="Pay off remaining debt",
            target_amount=main_goal.target_amount * Decimal("0.4"),
            target_date=main_goal.target_date,
            priority=2,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance="conservative",
            tax_implications={}
        )
        
        main_goal.sub_goals = [high_interest_goal, remaining_debt_goal]
        return main_goal
    
    def _decompose_generic_goal(
        self, 
        main_goal: FinancialGoal, 
        financial_state: FinancialState, 
        risk_profile: RiskProfile
    ) -> FinancialGoal:
        """Decompose generic wealth building goal"""
        # Sub-goal 1: Foundation building (emergency fund if not exists)
        foundation_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.EMERGENCY_FUND,
            description="Establish financial foundation",
            target_amount=financial_state.monthly_expenses * 3,
            target_date=datetime.utcnow() + timedelta(days=90),
            priority=1,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance="conservative",
            tax_implications={}
        )
        
        # Sub-goal 2: Investment growth
        investment_goal = FinancialGoal(
            goal_id=str(uuid4()),
            goal_type=GoalType.WEALTH_BUILDING,
            description="Build investment portfolio",
            target_amount=main_goal.target_amount * Decimal("0.8"),
            target_date=main_goal.target_date,
            priority=2,
            sub_goals=[],
            constraints=[],
            success_metrics={},
            risk_tolerance=risk_profile.overall_risk_tolerance.value,
            tax_implications={}
        )
        
        main_goal.sub_goals = [foundation_goal, investment_goal]
        return main_goal
    
    def _generate_cross_cutting_constraints(
        self, 
        financial_state: FinancialState, 
        risk_profile: RiskProfile
    ) -> List[Constraint]:
        """Generate constraints that apply across all goals"""
        constraints = []
        
        # Budget constraint
        budget_constraint = Constraint(
            name="Monthly Budget Adherence",
            constraint_type=ConstraintType.BUDGET,
            priority=ConstraintPriority.MANDATORY,
            description="Monthly expenses must not exceed 80% of income",
            validation_rule="monthly_expenses <= monthly_income * 0.8",
            threshold_value=0.8,
            comparison_operator="<=",
            created_by="goal_decomposition_system"
        )
        constraints.append(budget_constraint)
        
        # Liquidity constraint
        liquidity_constraint = Constraint(
            name="Minimum Liquidity",
            constraint_type=ConstraintType.LIQUIDITY,
            priority=ConstraintPriority.HIGH,
            description="Maintain minimum 3 months expenses in liquid assets",
            validation_rule="liquid_assets >= monthly_expenses * 3",
            threshold_value=3,
            comparison_operator=">=",
            created_by="goal_decomposition_system"
        )
        constraints.append(liquidity_constraint)
        
        # Risk constraint based on profile
        risk_constraint = Constraint(
            name="Risk Tolerance Adherence",
            constraint_type=ConstraintType.RISK,
            priority=ConstraintPriority.HIGH,
            description=f"Portfolio risk must align with {risk_profile.overall_risk_tolerance.value} risk tolerance",
            validation_rule=f"portfolio_risk_score <= {self._get_risk_threshold(risk_profile)}",
            threshold_value=self._get_risk_threshold(risk_profile),
            comparison_operator="<=",
            created_by="goal_decomposition_system"
        )
        constraints.append(risk_constraint)
        
        return constraints
    
    def _get_risk_threshold(self, risk_profile: RiskProfile) -> float:
        """Get risk threshold based on risk profile"""
        risk_thresholds = {
            "conservative": 0.3,
            "moderate_conservative": 0.5,
            "moderate": 0.7,
            "moderate_aggressive": 0.8,
            "aggressive": 1.0
        }
        return risk_thresholds.get(risk_profile.overall_risk_tolerance.value, 0.7)
    
    def _generate_success_metrics(
        self, 
        goal: FinancialGoal, 
        financial_state: FinancialState
    ) -> Dict[str, Any]:
        """Generate success metrics for goal tracking"""
        metrics = {
            "target_amount": float(goal.target_amount),
            "target_date": goal.target_date.isoformat(),
            "progress_milestones": [],
            "risk_metrics": {},
            "performance_benchmarks": {}
        }
        
        # Generate progress milestones
        months_to_goal = (goal.target_date - datetime.utcnow()).days / 30.44
        if months_to_goal > 0:
            milestone_intervals = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100%
            for interval in milestone_intervals:
                milestone_date = datetime.utcnow() + timedelta(days=months_to_goal * 30.44 * interval)
                milestone_amount = float(goal.target_amount * Decimal(str(interval)))
                metrics["progress_milestones"].append({
                    "date": milestone_date.isoformat(),
                    "target_amount": milestone_amount,
                    "completion_percentage": interval * 100
                })
        
        # Risk metrics based on goal type
        if goal.goal_type in [GoalType.RETIREMENT, GoalType.WEALTH_BUILDING]:
            metrics["risk_metrics"] = {
                "max_drawdown_tolerance": 0.2,  # 20% max drawdown
                "volatility_target": 0.15,      # 15% annual volatility
                "sharpe_ratio_target": 1.0      # Target Sharpe ratio
            }
        
        # Performance benchmarks
        if goal.goal_type == GoalType.RETIREMENT:
            metrics["performance_benchmarks"] = {
                "annual_return_target": 0.08,   # 8% annual return
                "inflation_adjustment": 0.03,   # 3% inflation
                "withdrawal_rate": 0.04         # 4% safe withdrawal rate
            }
        
        return metrics
    
    def _initialize_goal_templates(self) -> Dict[GoalType, Dict[str, Any]]:
        """Initialize goal templates for common scenarios"""
        return {
            GoalType.RETIREMENT: {
                "default_timeline_years": 30,
                "recommended_allocation": {"stocks": 0.7, "bonds": 0.3},
                "key_milestones": [5, 10, 15, 20, 25, 30],
                "critical_constraints": ["tax_optimization", "risk_management"]
            },
            GoalType.HOME_PURCHASE: {
                "default_timeline_years": 5,
                "recommended_allocation": {"cash": 0.6, "bonds": 0.4},
                "key_milestones": [1, 2, 3, 4, 5],
                "critical_constraints": ["liquidity", "capital_preservation"]
            },
            GoalType.EMERGENCY_FUND: {
                "default_timeline_years": 1,
                "recommended_allocation": {"cash": 1.0},
                "key_milestones": [0.25, 0.5, 0.75, 1.0],
                "critical_constraints": ["liquidity", "capital_preservation"]
            }
        }
    
    def _initialize_decomposition_rules(self) -> Dict[str, Any]:
        """Initialize rules for goal decomposition"""
        return {
            "max_sub_goals": 5,
            "min_timeline_months": 3,
            "priority_distribution": [0.4, 0.3, 0.2, 0.1],  # Priority weights
            "constraint_inheritance": True,
            "milestone_frequency": "quarterly"
        }


class TimeHorizonPlanner:
    """
    Advanced time-horizon planning with milestone tracking.
    Handles multi-year planning with adaptive milestones and progress monitoring.
    """
    
    def __init__(self):
        self.milestone_templates = self._initialize_milestone_templates()
        self.risk_adjustment_curves = self._initialize_risk_curves()
    
    def create_time_horizon_plan(
        self,
        financial_goal: FinancialGoal,
        financial_state: FinancialState,
        risk_profile: RiskProfile
    ) -> Tuple[List[Milestone], Dict[str, Any]]:
        """
        Create comprehensive time-horizon plan with milestone tracking.
        
        Args:
            financial_goal: The financial goal to plan for
            financial_state: Current financial state
            risk_profile: User's risk profile
            
        Returns:
            Tuple of milestones list and planning metadata
        """
        # Calculate planning horizon
        planning_horizon_months = self._calculate_planning_horizon(financial_goal)
        
        # Generate milestone schedule
        milestones = self._generate_milestone_schedule(
            financial_goal, planning_horizon_months, financial_state, risk_profile
        )
        
        # Create risk adjustment schedule
        risk_schedule = self._create_risk_adjustment_schedule(
            planning_horizon_months, risk_profile, financial_goal.goal_type
        )
        
        # Generate rebalancing schedule
        rebalancing_schedule = self._create_rebalancing_schedule(planning_horizon_months)
        
        # Create monitoring framework
        monitoring_framework = self._create_monitoring_framework(milestones, financial_goal)
        
        planning_metadata = {
            "planning_horizon_months": planning_horizon_months,
            "risk_adjustment_schedule": risk_schedule,
            "rebalancing_schedule": rebalancing_schedule,
            "monitoring_framework": monitoring_framework,
            "success_probability": self._calculate_success_probability(financial_goal, financial_state),
            "contingency_plans": self._generate_contingency_plans(financial_goal, milestones)
        }
        
        return milestones, planning_metadata 
   
    def _calculate_planning_horizon(self, financial_goal: FinancialGoal) -> int:
        """Calculate planning horizon in months"""
        target_date = financial_goal.target_date
        current_date = datetime.utcnow()
        
        months_difference = (target_date.year - current_date.year) * 12 + (target_date.month - current_date.month)
        
        # Ensure minimum planning horizon
        return max(months_difference, 3)
    
    def _generate_milestone_schedule(
        self,
        financial_goal: FinancialGoal,
        planning_horizon_months: int,
        financial_state: FinancialState,
        risk_profile: RiskProfile
    ) -> List[Milestone]:
        """Generate adaptive milestone schedule"""
        milestones = []
        
        # Determine milestone frequency based on planning horizon
        if planning_horizon_months <= 12:
            milestone_intervals = [3, 6, 9, 12]  # Quarterly for short-term
        elif planning_horizon_months <= 60:
            milestone_intervals = [6, 12, 24, 36, 48, 60]  # Semi-annual for medium-term
        else:
            milestone_intervals = [12, 24, 36, 60, 120, 240]  # Annual for long-term
        
        # Filter intervals that fit within planning horizon
        applicable_intervals = [i for i in milestone_intervals if i <= planning_horizon_months]
        
        for i, interval_months in enumerate(applicable_intervals):
            milestone_date = datetime.utcnow() + timedelta(days=interval_months * 30.44)
            
            # Calculate target metrics for this milestone
            progress_ratio = interval_months / planning_horizon_months
            target_amount = financial_goal.target_amount * Decimal(str(progress_ratio))
            
            milestone = Milestone(
                milestone_id=str(uuid4()),
                target_date=milestone_date,
                target_metrics={
                    "net_worth": target_amount,
                    "investment_value": target_amount * Decimal("0.7"),  # Assume 70% invested
                    "cash_reserves": financial_state.monthly_expenses * 6,  # Maintain emergency fund
                    "debt_reduction": Decimal("0")  # Will be calculated based on goal type
                },
                success_criteria=self._generate_milestone_success_criteria(
                    financial_goal.goal_type, interval_months, progress_ratio
                ),
                risk_checkpoints=self._generate_risk_checkpoints(interval_months, risk_profile),
                adjustment_triggers=self._generate_adjustment_triggers(interval_months),
                status="planned"
            )
            
            milestones.append(milestone)
        
        return milestones
    
    def _generate_milestone_success_criteria(
        self, 
        goal_type: GoalType, 
        interval_months: int, 
        progress_ratio: float
    ) -> List[str]:
        """Generate success criteria for milestone"""
        criteria = [
            f"Achieve {progress_ratio:.1%} of target amount",
            "Maintain emergency fund at target level",
            "Stay within risk tolerance parameters"
        ]
        
        if goal_type == GoalType.RETIREMENT:
            criteria.extend([
                "Maximize tax-advantaged account contributions",
                "Maintain target asset allocation",
                "Review and adjust for inflation"
            ])
        elif goal_type == GoalType.HOME_PURCHASE:
            criteria.extend([
                "Maintain high credit score (>740)",
                "Keep funds in low-risk investments",
                "Monitor housing market conditions"
            ])
        elif goal_type == GoalType.DEBT_PAYOFF:
            criteria.extend([
                "Reduce debt by scheduled amount",
                "Avoid taking on new debt",
                "Maintain minimum payments on all accounts"
            ])
        
        return criteria
    
    def _generate_risk_checkpoints(self, interval_months: int, risk_profile: RiskProfile) -> List[str]:
        """Generate risk checkpoints for milestone"""
        checkpoints = [
            "Review portfolio volatility",
            "Assess correlation with market indices",
            "Evaluate concentration risk"
        ]
        
        if interval_months >= 12:
            checkpoints.extend([
                "Conduct stress testing",
                "Review risk capacity changes",
                "Assess behavioral risk factors"
            ])
        
        if interval_months >= 24:
            checkpoints.extend([
                "Update risk profile assessment",
                "Review insurance coverage adequacy",
                "Evaluate sequence of returns risk"
            ])
        
        return checkpoints
    
    def _generate_adjustment_triggers(self, interval_months: int) -> List[str]:
        """Generate triggers that would require plan adjustment"""
        triggers = [
            "Portfolio performance deviates >15% from target",
            "Major life event occurs (job change, marriage, etc.)",
            "Market conditions change significantly"
        ]
        
        if interval_months >= 12:
            triggers.extend([
                "Income changes by >20%",
                "Tax law changes affect strategy",
                "Interest rate environment shifts significantly"
            ])
        
        return triggers
    
    def _create_risk_adjustment_schedule(
        self, 
        planning_horizon_months: int, 
        risk_profile: RiskProfile, 
        goal_type: GoalType
    ) -> Dict[int, Dict[str, float]]:
        """Create schedule for risk adjustments over time"""
        schedule = {}
        
        # Get base risk allocation
        base_allocation = self._get_base_allocation(risk_profile, goal_type)
        
        # Adjust allocation over time (glide path for retirement)
        if goal_type == GoalType.RETIREMENT:
            # Implement age-based glide path
            for month in range(0, planning_horizon_months + 1, 12):
                years_remaining = (planning_horizon_months - month) / 12
                
                # Reduce equity allocation as retirement approaches
                equity_reduction = min(0.3, (30 - years_remaining) * 0.01)  # 1% per year after 30 years
                
                adjusted_allocation = base_allocation.copy()
                adjusted_allocation["stocks"] = max(0.3, base_allocation["stocks"] - equity_reduction)
                adjusted_allocation["bonds"] = min(0.7, base_allocation["bonds"] + equity_reduction)
                
                schedule[month] = adjusted_allocation
        else:
            # For other goals, maintain consistent allocation with minor adjustments
            for month in range(0, planning_horizon_months + 1, 6):
                time_remaining_ratio = (planning_horizon_months - month) / planning_horizon_months
                
                # Gradually become more conservative as goal approaches
                if time_remaining_ratio < 0.25:  # Last 25% of timeline
                    conservative_adjustment = 0.2 * (0.25 - time_remaining_ratio) / 0.25
                    
                    adjusted_allocation = base_allocation.copy()
                    adjusted_allocation["stocks"] = max(0.2, base_allocation["stocks"] - conservative_adjustment)
                    adjusted_allocation["bonds"] = min(0.8, base_allocation["bonds"] + conservative_adjustment)
                    
                    schedule[month] = adjusted_allocation
                else:
                    schedule[month] = base_allocation
        
        return schedule
    
    def _get_base_allocation(self, risk_profile: RiskProfile, goal_type: GoalType) -> Dict[str, float]:
        """Get base asset allocation based on risk profile and goal type"""
        risk_allocations = {
            "conservative": {"stocks": 0.3, "bonds": 0.6, "cash": 0.1},
            "moderate_conservative": {"stocks": 0.4, "bonds": 0.5, "cash": 0.1},
            "moderate": {"stocks": 0.6, "bonds": 0.3, "cash": 0.1},
            "moderate_aggressive": {"stocks": 0.7, "bonds": 0.2, "cash": 0.1},
            "aggressive": {"stocks": 0.8, "bonds": 0.1, "cash": 0.1}
        }
        
        base_allocation = risk_allocations.get(risk_profile.overall_risk_tolerance.value, 
                                             risk_allocations["moderate"])
        
        # Adjust for goal type
        if goal_type == GoalType.EMERGENCY_FUND:
            return {"cash": 1.0, "stocks": 0.0, "bonds": 0.0}
        elif goal_type == GoalType.HOME_PURCHASE:
            # More conservative for home purchase
            return {"cash": 0.4, "bonds": 0.5, "stocks": 0.1}
        
        return base_allocation
    
    def _create_rebalancing_schedule(self, planning_horizon_months: int) -> Dict[str, Any]:
        """Create rebalancing schedule"""
        return {
            "frequency_months": 6 if planning_horizon_months > 24 else 3,
            "threshold_deviation": 0.05,  # 5% deviation triggers rebalancing
            "calendar_dates": self._generate_rebalancing_dates(planning_horizon_months),
            "tax_considerations": {
                "prefer_tax_advantaged_accounts": True,
                "harvest_losses": True,
                "avoid_wash_sales": True
            }
        }
    
    def _generate_rebalancing_dates(self, planning_horizon_months: int) -> List[str]:
        """Generate specific rebalancing dates"""
        dates = []
        frequency = 6 if planning_horizon_months > 24 else 3
        
        for month in range(frequency, planning_horizon_months + 1, frequency):
            rebalance_date = datetime.utcnow() + timedelta(days=month * 30.44)
            dates.append(rebalance_date.isoformat())
        
        return dates
    
    def _create_monitoring_framework(
        self, 
        milestones: List[Milestone], 
        financial_goal: FinancialGoal
    ) -> Dict[str, Any]:
        """Create comprehensive monitoring framework"""
        return {
            "key_performance_indicators": [
                "net_worth_growth_rate",
                "investment_return_vs_benchmark",
                "savings_rate_consistency",
                "risk_adjusted_return",
                "goal_progress_percentage"
            ],
            "alert_thresholds": {
                "performance_deviation": 0.15,  # 15% deviation from target
                "risk_limit_breach": 0.05,     # 5% over risk limit
                "liquidity_warning": 0.8       # 80% of minimum liquidity
            },
            "reporting_frequency": "monthly",
            "dashboard_metrics": [
                "current_vs_target_progress",
                "time_remaining",
                "required_monthly_savings",
                "portfolio_performance",
                "risk_metrics"
            ],
            "automated_actions": {
                "rebalancing": True,
                "tax_loss_harvesting": True,
                "contribution_increases": True
            }
        }
    
    def _calculate_success_probability(
        self, 
        financial_goal: FinancialGoal, 
        financial_state: FinancialState
    ) -> float:
        """Calculate probability of achieving the financial goal"""
        # Simplified Monte Carlo-style calculation
        current_net_worth = financial_state.net_worth
        target_amount = financial_goal.target_amount
        months_to_goal = (financial_goal.target_date - datetime.utcnow()).days / 30.44
        
        if months_to_goal <= 0:
            return 1.0 if current_net_worth >= target_amount else 0.0
        
        # Calculate required monthly savings
        required_growth = float(target_amount - current_net_worth)
        
        if required_growth <= 0:
            return 1.0  # Already achieved
        
        # Assume 7% annual return and calculate required monthly contribution
        monthly_return = 0.07 / 12
        if monthly_return > 0:
            # Future value of annuity formula
            required_monthly_savings = required_growth / (
                ((1 + monthly_return) ** months_to_goal - 1) / monthly_return
            )
        else:
            required_monthly_savings = required_growth / months_to_goal
        
        # Compare with available cash flow
        available_monthly_savings = float(financial_state.monthly_cash_flow * Decimal("0.8"))  # 80% of cash flow
        
        if available_monthly_savings >= required_monthly_savings:
            return min(0.95, 0.6 + (available_monthly_savings / required_monthly_savings) * 0.3)
        else:
            return max(0.1, 0.6 * (available_monthly_savings / required_monthly_savings))
    
    def _generate_contingency_plans(
        self, 
        financial_goal: FinancialGoal, 
        milestones: List[Milestone]
    ) -> List[Dict[str, Any]]:
        """Generate contingency plans for various scenarios"""
        contingency_plans = []
        
        # Market downturn scenario
        market_downturn_plan = {
            "scenario": "market_downturn_20_percent",
            "trigger": "Portfolio declines >20% from peak",
            "actions": [
                "Increase savings rate by 25%",
                "Reduce discretionary spending",
                "Consider extending timeline by 6-12 months",
                "Rebalance to target allocation",
                "Avoid panic selling"
            ],
            "timeline_adjustment": 6,  # months
            "success_probability_impact": -0.15
        }
        contingency_plans.append(market_downturn_plan)
        
        # Income loss scenario
        income_loss_plan = {
            "scenario": "income_loss",
            "trigger": "Income reduces by >30%",
            "actions": [
                "Pause non-essential investments",
                "Use emergency fund if necessary",
                "Reduce target amount by 20%",
                "Extend timeline by 12-24 months",
                "Focus on job search/income replacement"
            ],
            "timeline_adjustment": 18,  # months
            "success_probability_impact": -0.25
        }
        contingency_plans.append(income_loss_plan)
        
        # Ahead of schedule scenario
        ahead_of_schedule_plan = {
            "scenario": "ahead_of_schedule",
            "trigger": "Progress >120% of target at any milestone",
            "actions": [
                "Consider accelerating timeline",
                "Increase target amount",
                "Diversify into additional asset classes",
                "Consider tax optimization strategies",
                "Evaluate new financial goals"
            ],
            "timeline_adjustment": -6,  # months (accelerate)
            "success_probability_impact": 0.1
        }
        contingency_plans.append(ahead_of_schedule_plan)
        
        return contingency_plans
    
    def _initialize_milestone_templates(self) -> Dict[GoalType, Dict[str, Any]]:
        """Initialize milestone templates for different goal types"""
        return {
            GoalType.RETIREMENT: {
                "milestone_frequency_months": 12,
                "key_metrics": ["net_worth", "retirement_accounts", "investment_performance"],
                "risk_adjustments": True,
                "rebalancing_required": True
            },
            GoalType.HOME_PURCHASE: {
                "milestone_frequency_months": 6,
                "key_metrics": ["down_payment_fund", "credit_score", "debt_to_income"],
                "risk_adjustments": False,
                "rebalancing_required": False
            },
            GoalType.EMERGENCY_FUND: {
                "milestone_frequency_months": 3,
                "key_metrics": ["emergency_fund_balance", "liquidity_ratio"],
                "risk_adjustments": False,
                "rebalancing_required": False
            }
        }
    
    def _initialize_risk_curves(self) -> Dict[str, Any]:
        """Initialize risk adjustment curves for different scenarios"""
        return {
            "retirement_glide_path": {
                "start_equity_pct": 0.8,
                "end_equity_pct": 0.4,
                "adjustment_rate": 0.01  # 1% per year
            },
            "goal_approach_curve": {
                "conservative_threshold": 0.25,  # Last 25% of timeline
                "max_adjustment": 0.3            # Maximum 30% shift to conservative
            }
        }


class RiskAdjustedReturnOptimizer:
    """
    Advanced risk-adjusted return optimization algorithms.
    Implements modern portfolio theory with practical constraints.
    """
    
    def __init__(self):
        self.risk_free_rate = 0.045  # Current risk-free rate
        self.market_return = 0.10    # Expected market return
        self.optimization_methods = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown"]
    
    def optimize_portfolio(
        self,
        available_instruments: List[FinancialInstrument],
        target_return: float,
        risk_tolerance: float,
        constraints: Dict[str, Any],
        time_horizon_months: int
    ) -> Dict[str, Any]:
        """
        Optimize portfolio for risk-adjusted returns.
        
        Args:
            available_instruments: List of available financial instruments
            target_return: Target annual return
            risk_tolerance: Risk tolerance (0-1 scale)
            constraints: Portfolio constraints (min/max allocations, etc.)
            time_horizon_months: Investment time horizon
            
        Returns:
            Optimized portfolio allocation and metrics
        """
        # Filter instruments based on constraints
        eligible_instruments = self._filter_eligible_instruments(
            available_instruments, constraints, time_horizon_months
        )
        
        # Calculate expected returns and covariance matrix
        returns_data = self._calculate_expected_returns(eligible_instruments, time_horizon_months)
        covariance_matrix = self._calculate_covariance_matrix(eligible_instruments)
        
        # Run optimization algorithms
        optimization_results = {}
        
        for method in self.optimization_methods:
            result = self._run_optimization(
                method, eligible_instruments, returns_data, covariance_matrix,
                target_return, risk_tolerance, constraints
            )
            optimization_results[method] = result
        
        # Select best optimization result
        best_result = self._select_best_optimization(optimization_results, risk_tolerance)
        
        # Add portfolio analytics
        portfolio_analytics = self._calculate_portfolio_analytics(
            best_result, eligible_instruments, returns_data, covariance_matrix
        )
        
        return {
            "optimal_allocation": best_result["allocation"],
            "expected_return": best_result["expected_return"],
            "expected_volatility": best_result["volatility"],
            "sharpe_ratio": best_result["sharpe_ratio"],
            "optimization_method": best_result["method"],
            "portfolio_analytics": portfolio_analytics,
            "alternative_allocations": {k: v["allocation"] for k, v in optimization_results.items()},
            "risk_metrics": self._calculate_risk_metrics(best_result, eligible_instruments)
        }   
 
    def _filter_eligible_instruments(
        self,
        instruments: List[FinancialInstrument],
        constraints: Dict[str, Any],
        time_horizon_months: int
    ) -> List[FinancialInstrument]:
        """Filter instruments based on constraints and suitability"""
        eligible = []
        
        for instrument in instruments:
            # Check minimum investment requirement
            min_investment = constraints.get("min_investment_per_asset", 1000)
            if instrument.minimum_investment > min_investment:
                continue
            
            # Check liquidity requirements for short-term goals
            if time_horizon_months < 24 and instrument.liquidity_score < 0.7:
                continue
            
            # Check complexity constraints
            max_complexity = constraints.get("max_complexity_score", 0.8)
            if instrument.complexity_score > max_complexity:
                continue
            
            # Check expense ratio limits
            max_expense_ratio = constraints.get("max_expense_ratio", 0.02)
            if instrument.expense_ratio > max_expense_ratio:
                continue
            
            eligible.append(instrument)
        
        return eligible
    
    def _calculate_expected_returns(
        self, 
        instruments: List[FinancialInstrument], 
        time_horizon_months: int
    ) -> Dict[str, float]:
        """Calculate expected returns for instruments"""
        returns = {}
        
        for instrument in instruments:
            # Adjust expected return for time horizon
            base_return = instrument.expected_return
            
            # Time horizon adjustments
            if time_horizon_months < 12:
                # Reduce expected return for short-term (more conservative)
                adjusted_return = base_return * 0.7
            elif time_horizon_months > 120:
                # Slight increase for very long-term
                adjusted_return = base_return * 1.1
            else:
                adjusted_return = base_return
            
            # Adjust for expense ratio
            net_return = adjusted_return - instrument.expense_ratio
            
            returns[instrument.instrument_id] = net_return
        
        return returns
    
    def _calculate_covariance_matrix(self, instruments: List[FinancialInstrument]) -> np.ndarray:
        """Calculate covariance matrix for instruments"""
        n = len(instruments)
        covariance_matrix = np.zeros((n, n))
        
        # Simplified covariance calculation based on asset types
        correlation_matrix = self._get_correlation_matrix(instruments)
        
        for i, instrument_i in enumerate(instruments):
            for j, instrument_j in enumerate(instruments):
                if i == j:
                    # Variance on diagonal
                    covariance_matrix[i, j] = instrument_i.volatility ** 2
                else:
                    # Covariance off diagonal
                    correlation = correlation_matrix[i, j]
                    covariance_matrix[i, j] = (correlation * 
                                             instrument_i.volatility * 
                                             instrument_j.volatility)
        
        return covariance_matrix
    
    def _get_correlation_matrix(self, instruments: List[FinancialInstrument]) -> np.ndarray:
        """Get correlation matrix based on instrument types"""
        n = len(instruments)
        correlation_matrix = np.eye(n)  # Start with identity matrix
        
        # Define typical correlations between asset classes
        type_correlations = {
            (InstrumentType.STOCKS, InstrumentType.STOCKS): 0.8,
            (InstrumentType.STOCKS, InstrumentType.BONDS): -0.1,
            (InstrumentType.STOCKS, InstrumentType.REAL_ESTATE): 0.3,
            (InstrumentType.STOCKS, InstrumentType.COMMODITIES): 0.2,
            (InstrumentType.BONDS, InstrumentType.BONDS): 0.7,
            (InstrumentType.BONDS, InstrumentType.REAL_ESTATE): 0.1,
            (InstrumentType.BONDS, InstrumentType.COMMODITIES): -0.2,
            (InstrumentType.REAL_ESTATE, InstrumentType.REAL_ESTATE): 0.6,
            (InstrumentType.REAL_ESTATE, InstrumentType.COMMODITIES): 0.1,
            (InstrumentType.COMMODITIES, InstrumentType.COMMODITIES): 0.5
        }
        
        for i, instrument_i in enumerate(instruments):
            for j, instrument_j in enumerate(instruments):
                if i != j:
                    type_pair = (instrument_i.instrument_type, instrument_j.instrument_type)
                    reverse_pair = (instrument_j.instrument_type, instrument_i.instrument_type)
                    
                    correlation = (type_correlations.get(type_pair) or 
                                 type_correlations.get(reverse_pair) or 0.3)
                    
                    correlation_matrix[i, j] = correlation
        
        return correlation_matrix
    
    def _run_optimization(
        self,
        method: str,
        instruments: List[FinancialInstrument],
        returns_data: Dict[str, float],
        covariance_matrix: np.ndarray,
        target_return: float,
        risk_tolerance: float,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run specific optimization method"""
        
        if method == "sharpe_ratio":
            return self._optimize_sharpe_ratio(
                instruments, returns_data, covariance_matrix, constraints
            )
        elif method == "sortino_ratio":
            return self._optimize_sortino_ratio(
                instruments, returns_data, covariance_matrix, constraints
            )
        elif method == "max_drawdown":
            return self._optimize_max_drawdown(
                instruments, returns_data, covariance_matrix, constraints
            )
        else:
            # Default to equal weight
            return self._equal_weight_allocation(instruments, returns_data, covariance_matrix)
    
    def _optimize_sharpe_ratio(
        self,
        instruments: List[FinancialInstrument],
        returns_data: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize for maximum Sharpe ratio"""
        n = len(instruments)
        
        # Convert returns to numpy array
        returns_array = np.array([returns_data[inst.instrument_id] for inst in instruments])
        
        # Simple optimization using equal risk contribution as starting point
        # In production, would use scipy.optimize or cvxpy
        
        # Start with equal weights
        weights = np.ones(n) / n
        
        # Apply constraints
        min_weights = np.array([constraints.get(f"min_weight_{inst.instrument_id}", 0.0) 
                               for inst in instruments])
        max_weights = np.array([constraints.get(f"max_weight_{inst.instrument_id}", 1.0) 
                               for inst in instruments])
        
        # Ensure weights are within bounds
        weights = np.maximum(weights, min_weights)
        weights = np.minimum(weights, max_weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, returns_array)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Create allocation dictionary
        allocation = {
            instruments[i].instrument_id: float(weights[i]) 
            for i in range(n)
        }
        
        return {
            "method": "sharpe_ratio",
            "allocation": allocation,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio)
        }
    
    def _optimize_sortino_ratio(
        self,
        instruments: List[FinancialInstrument],
        returns_data: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize for maximum Sortino ratio (focuses on downside risk)"""
        # Simplified implementation - in practice would calculate downside deviation
        result = self._optimize_sharpe_ratio(instruments, returns_data, covariance_matrix, constraints)
        
        # Adjust for downside focus (more conservative allocation)
        allocation = result["allocation"]
        
        # Reduce allocation to high-volatility assets
        for i, instrument in enumerate(instruments):
            if instrument.volatility > 0.2:  # High volatility threshold
                instrument_id = instrument.instrument_id
                allocation[instrument_id] *= 0.8  # Reduce by 20%
        
        # Renormalize
        total_weight = sum(allocation.values())
        allocation = {k: v / total_weight for k, v in allocation.items()}
        
        # Recalculate metrics
        weights = np.array([allocation[inst.instrument_id] for inst in instruments])
        returns_array = np.array([returns_data[inst.instrument_id] for inst in instruments])
        
        portfolio_return = np.dot(weights, returns_array)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Approximate Sortino ratio (would need historical data for exact calculation)
        downside_deviation = portfolio_volatility * 0.7  # Approximate
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_deviation
        
        result.update({
            "method": "sortino_ratio",
            "allocation": allocation,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sortino_ratio)  # Using as proxy
        })
        
        return result
    
    def _optimize_max_drawdown(
        self,
        instruments: List[FinancialInstrument],
        returns_data: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize to minimize maximum drawdown"""
        # Conservative allocation focused on drawdown minimization
        n = len(instruments)
        
        # Favor low-volatility, high-liquidity instruments
        scores = []
        for instrument in instruments:
            # Score based on inverse volatility and high liquidity
            score = (1 / (instrument.volatility + 0.01)) * instrument.liquidity_score
            scores.append(score)
        
        # Normalize scores to get weights
        total_score = sum(scores)
        weights = [score / total_score for score in scores]
        
        # Apply constraints
        min_weights = [constraints.get(f"min_weight_{inst.instrument_id}", 0.0) 
                      for inst in instruments]
        max_weights = [constraints.get(f"max_weight_{inst.instrument_id}", 1.0) 
                      for inst in instruments]
        
        # Ensure weights are within bounds
        weights = [max(w, min_w) for w, min_w in zip(weights, min_weights)]
        weights = [min(w, max_w) for w, max_w in zip(weights, max_weights)]
        
        # Renormalize
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Create allocation dictionary
        allocation = {
            instruments[i].instrument_id: weights[i] 
            for i in range(n)
        }
        
        # Calculate metrics
        weights_array = np.array(weights)
        returns_array = np.array([returns_data[inst.instrument_id] for inst in instruments])
        
        portfolio_return = np.dot(weights_array, returns_array)
        portfolio_variance = np.dot(weights_array, np.dot(covariance_matrix, weights_array))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            "method": "max_drawdown",
            "allocation": allocation,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio)
        }
    
    def _equal_weight_allocation(
        self,
        instruments: List[FinancialInstrument],
        returns_data: Dict[str, float],
        covariance_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Simple equal weight allocation as fallback"""
        n = len(instruments)
        weight = 1.0 / n
        
        allocation = {inst.instrument_id: weight for inst in instruments}
        
        # Calculate metrics
        weights_array = np.array([weight] * n)
        returns_array = np.array([returns_data[inst.instrument_id] for inst in instruments])
        
        portfolio_return = np.dot(weights_array, returns_array)
        portfolio_variance = np.dot(weights_array, np.dot(covariance_matrix, weights_array))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            "method": "equal_weight",
            "allocation": allocation,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio)
        }
    
    def _select_best_optimization(
        self, 
        optimization_results: Dict[str, Dict[str, Any]], 
        risk_tolerance: float
    ) -> Dict[str, Any]:
        """Select best optimization result based on risk tolerance"""
        
        # Score each result based on Sharpe ratio and risk tolerance alignment
        best_result = None
        best_score = -float('inf')
        
        for method, result in optimization_results.items():
            sharpe_ratio = result["sharpe_ratio"]
            volatility = result["volatility"]
            
            # Adjust score based on risk tolerance
            if risk_tolerance < 0.3:  # Conservative
                # Penalize high volatility
                score = sharpe_ratio - (volatility * 2)
            elif risk_tolerance > 0.7:  # Aggressive
                # Reward higher Sharpe ratios
                score = sharpe_ratio * 1.5
            else:  # Moderate
                score = sharpe_ratio
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result or list(optimization_results.values())[0]
    
    def _calculate_portfolio_analytics(
        self,
        optimization_result: Dict[str, Any],
        instruments: List[FinancialInstrument],
        returns_data: Dict[str, float],
        covariance_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio analytics"""
        allocation = optimization_result["allocation"]
        
        # Asset class breakdown
        asset_class_allocation = {}
        for instrument in instruments:
            asset_class = instrument.instrument_type.value
            weight = allocation.get(instrument.instrument_id, 0)
            
            if asset_class in asset_class_allocation:
                asset_class_allocation[asset_class] += weight
            else:
                asset_class_allocation[asset_class] = weight
        
        # Concentration metrics
        weights = list(allocation.values())
        herfindahl_index = sum(w**2 for w in weights) if weights else 0
        effective_number_assets = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            "asset_class_allocation": asset_class_allocation,
            "concentration_metrics": {
                "herfindahl_index": float(herfindahl_index),
                "effective_number_assets": float(effective_number_assets),
                "max_single_position": max(weights) if weights else 0
            },
            "diversification_ratio": 0.8,  # Simplified
            "expense_weighted_average": 0.005  # Simplified
        }
    
    def _calculate_risk_metrics(
        self, 
        optimization_result: Dict[str, Any], 
        instruments: List[FinancialInstrument]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        expected_return = optimization_result["expected_return"]
        volatility = optimization_result["volatility"]
        sharpe_ratio = optimization_result["sharpe_ratio"]
        
        # Value at Risk (simplified)
        var_95 = expected_return - (1.645 * volatility)  # 95% VaR
        var_99 = expected_return - (2.326 * volatility)  # 99% VaR
        
        # Expected Shortfall (simplified)
        es_95 = expected_return - (2.063 * volatility)  # 95% Expected Shortfall
        
        # Maximum theoretical loss (3 standard deviations)
        max_theoretical_loss = expected_return - (3 * volatility)
        
        return {
            "value_at_risk": {
                "var_95": float(var_95),
                "var_99": float(var_99)
            },
            "expected_shortfall_95": float(es_95),
            "max_theoretical_loss": float(max_theoretical_loss),
            "sharpe_ratio": float(sharpe_ratio),
            "volatility": float(volatility),
            "risk_adjusted_return": float(expected_return / volatility) if volatility > 0 else 0,
            "downside_capture_ratio": 0.8,  # Simplified assumption
            "upside_capture_ratio": 1.1     # Simplified assumption
        }


class AssetAllocationOptimizer:
    """
    Advanced asset allocation optimization algorithms.
    Implements sophisticated allocation strategies for different goals and risk profiles.
    """
    
    def __init__(self):
        self.allocation_models = self._initialize_allocation_models()
        self.rebalancing_strategies = self._initialize_rebalancing_strategies()
    
    def _initialize_rebalancing_strategies(self) -> Dict[str, Any]:
        """Initialize rebalancing strategies"""
        return {
            "threshold_based": {"threshold": 0.05, "frequency": "quarterly"},
            "calendar_based": {"frequency": "annual", "month": "december"},
            "tolerance_band": {"inner_threshold": 0.03, "outer_threshold": 0.08}
        }
    
    def optimize_asset_allocation(
        self,
        financial_goal: FinancialGoal,
        risk_profile: RiskProfile,
        time_horizon_months: int,
        current_allocation: Dict[str, float],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize asset allocation based on goal, risk profile, and market conditions.
        
        Args:
            financial_goal: The financial goal being optimized for
            risk_profile: User's risk profile
            time_horizon_months: Investment time horizon
            current_allocation: Current portfolio allocation
            market_conditions: Current market environment
            
        Returns:
            Optimized allocation with rationale and implementation plan
        """
        # Select appropriate allocation model
        allocation_model = self._select_allocation_model(financial_goal, risk_profile, time_horizon_months)
        
        # Generate target allocation
        target_allocation = self._generate_target_allocation(
            allocation_model, risk_profile, time_horizon_months, market_conditions
        )
        
        # Calculate rebalancing requirements
        rebalancing_plan = self._calculate_rebalancing_plan(current_allocation, target_allocation)
        
        # Assess tax implications
        tax_implications = self._assess_allocation_tax_implications(rebalancing_plan)
        
        # Generate implementation timeline
        implementation_timeline = self._create_implementation_timeline(rebalancing_plan, tax_implications)
        
        return {
            "target_allocation": target_allocation,
            "current_allocation": current_allocation,
            "rebalancing_plan": rebalancing_plan,
            "allocation_model": allocation_model,
            "tax_implications": tax_implications,
            "implementation_timeline": implementation_timeline,
            "expected_return": self._calculate_allocation_expected_return(target_allocation),
            "expected_volatility": self._calculate_allocation_volatility(target_allocation),
            "diversification_score": self._calculate_diversification_score(target_allocation)
        }
    
    def _assess_allocation_tax_implications(self, rebalancing_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assess tax implications of allocation changes"""
        return {
            "estimated_tax_cost": rebalancing_plan.get("estimated_cost", 0) * 0.2,  # Assume 20% tax
            "tax_loss_harvesting_opportunities": [],
            "tax_efficient_implementation": True
        }
    
    def _create_implementation_timeline(self, rebalancing_plan: Dict[str, Any], tax_implications: Dict[str, Any]) -> Dict[str, Any]:
        """Create implementation timeline for allocation changes"""
        return {
            "immediate_actions": ["Review current positions"],
            "within_30_days": ["Execute high-priority rebalancing"],
            "within_90_days": ["Complete full rebalancing"],
            "ongoing": ["Monitor and maintain target allocation"]
        }
    
    def _calculate_allocation_expected_return(self, allocation: Dict[str, float]) -> float:
        """Calculate expected return for allocation"""
        asset_returns = {
            "stocks": 0.10,
            "bonds": 0.04,
            "real_estate": 0.08,
            "cash": 0.02,
            "commodities": 0.06
        }
        
        expected_return = 0.0
        for asset, weight in allocation.items():
            expected_return += weight * asset_returns.get(asset, 0.07)
        
        return expected_return
    
    def _calculate_allocation_volatility(self, allocation: Dict[str, float]) -> float:
        """Calculate expected volatility for allocation"""
        asset_volatilities = {
            "stocks": 0.18,
            "bonds": 0.05,
            "real_estate": 0.15,
            "cash": 0.01,
            "commodities": 0.25
        }
        
        # Simplified volatility calculation (would use covariance matrix in practice)
        weighted_volatility = 0.0
        for asset, weight in allocation.items():
            volatility = asset_volatilities.get(asset, 0.12)
            weighted_volatility += (weight ** 2) * (volatility ** 2)
        
        return math.sqrt(weighted_volatility * 0.8)  # Assume some diversification benefit
    
    def _calculate_diversification_score(self, allocation: Dict[str, float]) -> float:
        """Calculate diversification score for allocation"""
        # Herfindahl-Hirschman Index for diversification
        hhi = sum(weight ** 2 for weight in allocation.values())
        return 1.0 - hhi  # Higher score = better diversification
    
    def _select_allocation_model(
        self, 
        goal: FinancialGoal, 
        risk_profile: RiskProfile, 
        time_horizon_months: int
    ) -> str:
        """Select appropriate allocation model based on goal and profile"""
        if goal.goal_type == GoalType.RETIREMENT:
            if time_horizon_months > 240:  # 20+ years
                return "target_date_aggressive"
            elif time_horizon_months > 120:  # 10+ years
                return "target_date_moderate"
            else:
                return "target_date_conservative"
        
        elif goal.goal_type == GoalType.HOME_PURCHASE:
            return "capital_preservation"
        
        elif goal.goal_type == GoalType.EMERGENCY_FUND:
            return "liquidity_focused"
        
        else:
            # Use risk-based model for general wealth building
            risk_level = risk_profile.overall_risk_tolerance.value
            if risk_level in ["aggressive", "moderate_aggressive"]:
                return "growth_focused"
            elif risk_level in ["moderate"]:
                return "balanced_growth"
            else:
                return "conservative_growth"
    
    def _generate_target_allocation(
        self,
        model: str,
        risk_profile: RiskProfile,
        time_horizon_months: int,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, float]:
        """Generate target allocation based on model and conditions"""
        base_allocation = self.allocation_models[model].copy()
        
        # Adjust for time horizon (glide path)
        if "target_date" in model:
            years_to_goal = time_horizon_months / 12
            equity_reduction = max(0, (30 - years_to_goal) * 0.01)  # 1% per year after 30 years
            
            base_allocation["stocks"] = max(0.2, base_allocation["stocks"] - equity_reduction)
            base_allocation["bonds"] = min(0.8, base_allocation["bonds"] + equity_reduction)
        
        # Adjust for market conditions
        market_volatility = market_conditions.get("volatility", 0.15)
        if market_volatility > 0.25:  # High volatility environment
            # Reduce equity allocation by 10%
            equity_reduction = 0.1
            base_allocation["stocks"] = max(0.1, base_allocation["stocks"] - equity_reduction)
            base_allocation["bonds"] = min(0.9, base_allocation["bonds"] + equity_reduction * 0.7)
            base_allocation["cash"] = min(0.2, base_allocation.get("cash", 0) + equity_reduction * 0.3)
        
        # Normalize to ensure sum equals 1.0
        total = sum(base_allocation.values())
        return {k: v / total for k, v in base_allocation.items()}
    
    def _calculate_rebalancing_plan(
        self, 
        current: Dict[str, float], 
        target: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate detailed rebalancing plan"""
        rebalancing_actions = []
        total_trades = 0
        
        for asset_class in set(list(current.keys()) + list(target.keys())):
            current_weight = current.get(asset_class, 0.0)
            target_weight = target.get(asset_class, 0.0)
            difference = target_weight - current_weight
            
            if abs(difference) > 0.05:  # 5% threshold for rebalancing
                action = {
                    "asset_class": asset_class,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "adjustment": difference,
                    "action_type": "buy" if difference > 0 else "sell",
                    "priority": "high" if abs(difference) > 0.15 else "medium"
                }
                rebalancing_actions.append(action)
                total_trades += 1
        
        return {
            "actions": rebalancing_actions,
            "total_trades": total_trades,
            "rebalancing_needed": total_trades > 0,
            "estimated_cost": total_trades * 0.001,  # 0.1% per trade
            "complexity_score": min(total_trades / 5, 1.0)
        }
    
    def _initialize_allocation_models(self) -> Dict[str, Dict[str, float]]:
        """Initialize allocation model templates"""
        return {
            "target_date_aggressive": {
                "stocks": 0.90,
                "bonds": 0.08,
                "cash": 0.02
            },
            "target_date_moderate": {
                "stocks": 0.70,
                "bonds": 0.25,
                "cash": 0.05
            },
            "target_date_conservative": {
                "stocks": 0.40,
                "bonds": 0.50,
                "cash": 0.10
            },
            "capital_preservation": {
                "stocks": 0.20,
                "bonds": 0.60,
                "cash": 0.20
            },
            "liquidity_focused": {
                "stocks": 0.00,
                "bonds": 0.20,
                "cash": 0.80
            },
            "growth_focused": {
                "stocks": 0.80,
                "bonds": 0.15,
                "real_estate": 0.05
            },
            "balanced_growth": {
                "stocks": 0.60,
                "bonds": 0.30,
                "real_estate": 0.10
            },
            "conservative_growth": {
                "stocks": 0.40,
                "bonds": 0.50,
                "cash": 0.10
            }
        }


class RetirementPlanningEngine:
    """
    Specialized retirement planning and goal-based investment strategies.
    Implements comprehensive retirement planning logic with withdrawal strategies.
    """
    
    def __init__(self):
        self.withdrawal_strategies = self._initialize_withdrawal_strategies()
        self.retirement_milestones = self._initialize_retirement_milestones()
    
    def _initialize_retirement_milestones(self) -> Dict[str, Any]:
        """Initialize retirement milestone templates"""
        return {
            "early_career": {"age_range": (25, 35), "target_savings_rate": 0.15},
            "mid_career": {"age_range": (35, 50), "target_savings_rate": 0.20},
            "pre_retirement": {"age_range": (50, 65), "target_savings_rate": 0.25}
        }
    
    def create_retirement_plan(
        self,
        current_age: int,
        retirement_age: int,
        current_savings: Decimal,
        monthly_contribution: Decimal,
        desired_retirement_income: Decimal,
        risk_profile: RiskProfile,
        tax_context: TaxContext
    ) -> Dict[str, Any]:
        """
        Create comprehensive retirement plan with projections and strategies.
        
        Args:
            current_age: Current age
            retirement_age: Target retirement age
            current_savings: Current retirement savings
            monthly_contribution: Monthly contribution capacity
            desired_retirement_income: Desired annual retirement income
            risk_profile: Risk tolerance profile
            tax_context: Tax situation
            
        Returns:
            Comprehensive retirement plan with projections and recommendations
        """
        years_to_retirement = retirement_age - current_age
        years_in_retirement = 95 - retirement_age  # Assume life expectancy of 95
        
        # Calculate required retirement corpus
        required_corpus = self._calculate_required_corpus(
            desired_retirement_income, years_in_retirement, tax_context
        )
        
        # Project current savings growth
        projected_savings = self._project_savings_growth(
            current_savings, monthly_contribution, years_to_retirement, risk_profile
        )
        
        # Calculate funding gap
        funding_gap = required_corpus - projected_savings
        
        # Generate recommendations
        recommendations = self._generate_retirement_recommendations(
            funding_gap, years_to_retirement, monthly_contribution, risk_profile, tax_context
        )
        
        # Create withdrawal strategy
        withdrawal_strategy = self._create_withdrawal_strategy(
            projected_savings, desired_retirement_income, years_in_retirement, tax_context
        )
        
        # Generate milestone tracking
        milestones = self._generate_retirement_milestones(
            current_age, retirement_age, projected_savings, required_corpus
        )
        
        return {
            "retirement_feasibility": funding_gap <= 0,
            "required_corpus": float(required_corpus),
            "projected_corpus": float(projected_savings),
            "funding_gap": float(funding_gap),
            "success_probability": self._calculate_retirement_success_probability(
                projected_savings, required_corpus, risk_profile
            ),
            "recommendations": recommendations,
            "withdrawal_strategy": withdrawal_strategy,
            "milestones": milestones,
            "tax_optimization": self._analyze_retirement_tax_optimization(tax_context),
            "risk_assessment": self._assess_retirement_risks(
                years_to_retirement, projected_savings, risk_profile
            )
        }
    
    def _calculate_required_corpus(
        self, 
        annual_income: Decimal, 
        years_in_retirement: int, 
        tax_context: TaxContext
    ) -> Decimal:
        """Calculate required retirement corpus using multiple methods"""
        # Method 1: 4% rule
        corpus_4_percent = annual_income / Decimal("0.04")
        
        # Method 2: Present value of annuity (more conservative)
        discount_rate = Decimal("0.03")  # 3% real return assumption
        pv_factor = (1 - (1 + discount_rate) ** -years_in_retirement) / discount_rate
        corpus_pv = annual_income * pv_factor
        
        # Method 3: Inflation-adjusted needs
        inflation_rate = Decimal("0.025")  # 2.5% inflation
        real_income_needed = annual_income * (Decimal("1") + inflation_rate) ** Decimal(str(years_in_retirement / 2))
        corpus_inflation_adjusted = real_income_needed / Decimal("0.035")  # 3.5% withdrawal rate
        
        # Use the most conservative estimate
        return max(corpus_4_percent, corpus_pv, corpus_inflation_adjusted)
    
    def _project_savings_growth(
        self,
        current_savings: Decimal,
        monthly_contribution: Decimal,
        years: int,
        risk_profile: RiskProfile
    ) -> Decimal:
        """Project retirement savings growth with risk-adjusted returns"""
        # Risk-adjusted return assumptions
        return_assumptions = {
            "conservative": Decimal("0.06"),
            "moderate_conservative": Decimal("0.07"),
            "moderate": Decimal("0.08"),
            "moderate_aggressive": Decimal("0.09"),
            "aggressive": Decimal("0.10")
        }
        
        annual_return = return_assumptions.get(
            risk_profile.overall_risk_tolerance.value, 
            Decimal("0.08")
        )
        monthly_return = annual_return / 12
        months = Decimal(str(years * 12))
        
        # Future value of current savings
        fv_current = current_savings * (Decimal("1") + annual_return) ** Decimal(str(years))
        
        # Future value of monthly contributions
        if monthly_return > 0:
            fv_contributions = monthly_contribution * (
                ((Decimal("1") + monthly_return) ** months - Decimal("1")) / monthly_return
            )
        else:
            fv_contributions = monthly_contribution * months
        
        return fv_current + fv_contributions
    
    def _generate_retirement_recommendations(
        self,
        funding_gap: Decimal,
        years_to_retirement: int,
        current_contribution: Decimal,
        risk_profile: RiskProfile,
        tax_context: TaxContext
    ) -> List[Dict[str, Any]]:
        """Generate specific retirement planning recommendations"""
        recommendations = []
        
        if funding_gap > 0:
            # Calculate additional monthly savings needed
            annual_return = Decimal("0.08")  # Assume 8% return
            monthly_return = annual_return / 12
            months = years_to_retirement * 12
            
            if monthly_return > 0 and months > 0:
                additional_monthly = funding_gap / (
                    ((Decimal("1") + monthly_return) ** Decimal(str(months)) - Decimal("1")) / monthly_return
                )
                
                recommendations.append({
                    "type": "increase_contributions",
                    "description": f"Increase monthly contributions by ${additional_monthly:,.0f}",
                    "impact": f"Closes funding gap of ${funding_gap:,.0f}",
                    "priority": "high"
                })
            
            # Work longer recommendation
            if years_to_retirement > 5:
                additional_years = min(5, int(funding_gap / (current_contribution * 12)))
                recommendations.append({
                    "type": "extend_working_years",
                    "description": f"Consider working {additional_years} additional years",
                    "impact": f"Reduces funding gap significantly",
                    "priority": "medium"
                })
        
        # Tax optimization recommendations
        if tax_context.marginal_tax_rate > 0.22:
            recommendations.append({
                "type": "tax_optimization",
                "description": "Maximize tax-deferred contributions (401k, Traditional IRA)",
                "impact": f"Save ${float(current_contribution * 12 * Decimal(str(tax_context.marginal_tax_rate))):,.0f} annually in taxes",
                "priority": "high"
            })
        
        # Risk adjustment recommendations
        if risk_profile.overall_risk_tolerance.value == "conservative" and years_to_retirement > 10:
            recommendations.append({
                "type": "risk_adjustment",
                "description": "Consider moderate risk allocation for better growth potential",
                "impact": "Could increase projected corpus by 15-25%",
                "priority": "medium"
            })
        
        return recommendations
    
    def _create_withdrawal_strategy(
        self,
        projected_corpus: Decimal,
        desired_income: Decimal,
        years_in_retirement: int,
        tax_context: TaxContext
    ) -> Dict[str, Any]:
        """Create optimal withdrawal strategy for retirement"""
        strategies = {}
        
        # 4% Rule Strategy
        safe_withdrawal_4pct = projected_corpus * Decimal("0.04")
        strategies["four_percent_rule"] = {
            "annual_withdrawal": float(safe_withdrawal_4pct),
            "sustainability": "High" if safe_withdrawal_4pct >= desired_income else "Medium",
            "description": "Traditional 4% withdrawal rate"
        }
        
        # Dynamic Withdrawal Strategy
        initial_rate = min(Decimal("0.05"), desired_income / projected_corpus)
        strategies["dynamic_withdrawal"] = {
            "initial_rate": float(initial_rate),
            "annual_withdrawal": float(projected_corpus * initial_rate),
            "sustainability": "High",
            "description": "Adjust withdrawals based on portfolio performance"
        }
        
        # Bucket Strategy
        strategies["bucket_strategy"] = {
            "cash_bucket": float(projected_corpus * Decimal("0.1")),  # 1-2 years expenses
            "bond_bucket": float(projected_corpus * Decimal("0.3")),  # 3-10 years expenses
            "stock_bucket": float(projected_corpus * Decimal("0.6")), # Long-term growth
            "description": "Segmented approach for different time horizons"
        }
        
        # Tax-efficient withdrawal sequence
        withdrawal_sequence = self._create_tax_efficient_sequence(tax_context)
        
        return {
            "strategies": strategies,
            "recommended_strategy": "dynamic_withdrawal",
            "tax_efficient_sequence": withdrawal_sequence,
            "annual_review_required": True,
            "flexibility_score": 0.8
        }
    
    def _generate_retirement_milestones(
        self, 
        current_age: int, 
        retirement_age: int, 
        projected_corpus: Decimal, 
        required_corpus: Decimal
    ) -> List[Dict[str, Any]]:
        """Generate retirement planning milestones"""
        milestones = []
        years_to_retirement = retirement_age - current_age
        
        for i in range(1, min(years_to_retirement + 1, 11)):  # Up to 10 milestones
            milestone_age = current_age + (i * years_to_retirement // 10)
            progress_ratio = i / 10
            target_corpus = required_corpus * Decimal(str(progress_ratio))
            
            milestones.append({
                "age": milestone_age,
                "target_corpus": float(target_corpus),
                "progress_percentage": progress_ratio * 100,
                "review_items": [
                    "Review contribution rates",
                    "Assess risk tolerance",
                    "Update retirement goals"
                ]
            })
        
        return milestones
    
    def _calculate_retirement_success_probability(
        self, 
        projected_corpus: Decimal, 
        required_corpus: Decimal, 
        risk_profile: RiskProfile
    ) -> float:
        """Calculate probability of retirement success"""
        corpus_ratio = float(projected_corpus / required_corpus) if required_corpus > 0 else 1.0
        
        # Adjust for risk profile
        risk_adjustment = {
            "conservative": 0.9,
            "moderate_conservative": 0.95,
            "moderate": 1.0,
            "moderate_aggressive": 1.05,
            "aggressive": 1.1
        }.get(risk_profile.overall_risk_tolerance.value, 1.0)
        
        adjusted_ratio = corpus_ratio * risk_adjustment
        
        # Convert to probability (sigmoid-like function)
        if adjusted_ratio >= 1.2:
            return 0.95
        elif adjusted_ratio >= 1.0:
            return 0.8 + (adjusted_ratio - 1.0) * 0.75
        elif adjusted_ratio >= 0.8:
            return 0.5 + (adjusted_ratio - 0.8) * 1.5
        else:
            return max(0.1, adjusted_ratio * 0.625)
    
    def _analyze_retirement_tax_optimization(self, tax_context: TaxContext) -> Dict[str, Any]:
        """Analyze tax optimization opportunities for retirement"""
        return {
            "current_tax_efficiency": 0.7,  # Placeholder
            "roth_conversion_opportunities": tax_context.marginal_tax_rate < 0.24,
            "tax_diversification_score": 0.6,
            "recommended_actions": [
                "Maximize tax-deferred contributions",
                "Consider Roth conversions in low-income years",
                "Plan tax-efficient withdrawal sequence"
            ]
        }
    
    def _assess_retirement_risks(
        self, 
        years_to_retirement: int, 
        projected_corpus: Decimal, 
        risk_profile: RiskProfile
    ) -> Dict[str, Any]:
        """Assess risks to retirement plan"""
        risks = {
            "longevity_risk": "medium" if years_to_retirement > 20 else "high",
            "inflation_risk": "high" if years_to_retirement > 15 else "medium",
            "sequence_of_returns_risk": "high" if years_to_retirement < 10 else "low",
            "healthcare_cost_risk": "high",
            "market_volatility_risk": risk_profile.overall_risk_tolerance.value
        }
        
        mitigation_strategies = {
            "longevity_risk": ["Consider annuities", "Plan for 95+ life expectancy"],
            "inflation_risk": ["Include inflation-protected securities", "Plan for 3% annual inflation"],
            "sequence_of_returns_risk": ["Use bucket strategy", "Maintain cash reserves"],
            "healthcare_cost_risk": ["Maximize HSA contributions", "Consider long-term care insurance"]
        }
        
        return {
            "risk_assessment": risks,
            "mitigation_strategies": mitigation_strategies,
            "overall_risk_level": "medium"
        }
    
    def _create_tax_efficient_sequence(self, tax_context: TaxContext) -> List[str]:
        """Create tax-efficient withdrawal sequence"""
        sequence = [
            "1. Taxable accounts (tax-loss harvesting opportunities)",
            "2. Tax-deferred accounts (Traditional 401k/IRA)",
            "3. Tax-free accounts (Roth 401k/IRA)"
        ]
        
        if tax_context.marginal_tax_rate > 0.24:
            # High tax bracket - prioritize tax-free withdrawals
            sequence = [
                "1. Tax-free accounts (Roth 401k/IRA)",
                "2. Taxable accounts",
                "3. Tax-deferred accounts"
            ]
        
        return sequence
    
    def _initialize_withdrawal_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize withdrawal strategy templates"""
        return {
            "four_percent_rule": {
                "initial_rate": 0.04,
                "adjustment_method": "inflation_only",
                "risk_level": "low"
            },
            "dynamic_withdrawal": {
                "initial_rate": 0.045,
                "adjustment_method": "performance_based",
                "risk_level": "medium"
            },
            "bucket_strategy": {
                "cash_years": 2,
                "bond_years": 8,
                "stock_allocation": 0.6,
                "risk_level": "medium"
            }
        }


class AdvancedRiskAssessment:
    """
    Advanced risk assessment and portfolio balancing logic.
    Implements comprehensive risk analysis for financial planning.
    """
    
    def __init__(self):
        self.risk_metrics = self._initialize_risk_metrics()
        self.stress_test_scenarios = self._initialize_stress_scenarios()
    
    def _initialize_risk_metrics(self) -> Dict[str, Any]:
        """Initialize risk metric configurations"""
        return {
            "volatility_thresholds": {"low": 0.1, "medium": 0.2, "high": 0.3},
            "correlation_limits": {"max_correlation": 0.8, "min_diversification": 0.3},
            "drawdown_limits": {"conservative": 0.1, "moderate": 0.2, "aggressive": 0.3}
        }
    
    def assess_portfolio_risk(
        self,
        portfolio_allocation: Dict[str, float],
        financial_goal: FinancialGoal,
        time_horizon_months: int,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment.
        
        Args:
            portfolio_allocation: Current portfolio allocation
            financial_goal: Financial goal being assessed
            time_horizon_months: Investment time horizon
            market_conditions: Current market environment
            
        Returns:
            Comprehensive risk assessment with metrics and recommendations
        """
        # Calculate risk metrics
        risk_metrics = self._calculate_comprehensive_risk_metrics(
            portfolio_allocation, time_horizon_months
        )
        
        # Stress test analysis
        stress_test_results = self._run_stress_tests(
            portfolio_allocation, financial_goal, time_horizon_months
        )
        
        # Goal-specific risk analysis
        goal_risk_analysis = self._analyze_goal_specific_risks(
            portfolio_allocation, financial_goal, time_horizon_months
        )
        
        # Risk-return optimization suggestions
        optimization_suggestions = self._generate_risk_optimization_suggestions(
            portfolio_allocation, risk_metrics, financial_goal
        )
        
        return {
            "overall_risk_score": risk_metrics["overall_risk_score"],
            "risk_metrics": risk_metrics,
            "stress_test_results": stress_test_results,
            "goal_alignment": goal_risk_analysis,
            "optimization_suggestions": optimization_suggestions,
            "risk_capacity_assessment": self._assess_risk_capacity(financial_goal, time_horizon_months),
            "diversification_analysis": self._analyze_diversification(portfolio_allocation)
        }
    
    def _calculate_comprehensive_risk_metrics(
        self, 
        allocation: Dict[str, float], 
        time_horizon_months: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for portfolio"""
        # Asset class risk assumptions
        asset_risks = {
            "stocks": {"volatility": 0.18, "max_drawdown": 0.35, "beta": 1.0},
            "bonds": {"volatility": 0.05, "max_drawdown": 0.08, "beta": 0.2},
            "real_estate": {"volatility": 0.15, "max_drawdown": 0.25, "beta": 0.7},
            "commodities": {"volatility": 0.25, "max_drawdown": 0.40, "beta": 0.3},
            "cash": {"volatility": 0.01, "max_drawdown": 0.00, "beta": 0.0}
        }
        
        # Calculate portfolio-level metrics
        portfolio_volatility = 0.0
        portfolio_max_drawdown = 0.0
        portfolio_beta = 0.0
        
        for asset_class, weight in allocation.items():
            if asset_class in asset_risks:
                risk_data = asset_risks[asset_class]
                portfolio_volatility += weight * risk_data["volatility"]
                portfolio_max_drawdown += weight * risk_data["max_drawdown"]
                portfolio_beta += weight * risk_data["beta"]
        
        # Time horizon adjustments
        time_adjustment = min(1.0, time_horizon_months / 120)  # 10 years = full adjustment
        adjusted_volatility = portfolio_volatility * (1 - time_adjustment * 0.3)
        
        # Overall risk score (0-1 scale)
        overall_risk_score = min(1.0, (portfolio_volatility + portfolio_max_drawdown) / 2)
        
        return {
            "overall_risk_score": overall_risk_score,
            "portfolio_volatility": portfolio_volatility,
            "adjusted_volatility": adjusted_volatility,
            "max_drawdown_estimate": portfolio_max_drawdown,
            "portfolio_beta": portfolio_beta,
            "time_horizon_adjustment": time_adjustment,
            "risk_level": self._categorize_risk_level(overall_risk_score)
        }
    
    def _run_stress_tests(
        self,
        allocation: Dict[str, float],
        goal: FinancialGoal,
        time_horizon_months: int
    ) -> Dict[str, Any]:
        """Run stress tests on portfolio allocation"""
        stress_results = {}
        
        for scenario_name, scenario_data in self.stress_test_scenarios.items():
            # Calculate portfolio impact under stress scenario
            portfolio_impact = 0.0
            
            for asset_class, weight in allocation.items():
                asset_impact = scenario_data.get(asset_class, -0.1)  # Default -10%
                portfolio_impact += weight * asset_impact
            
            # Calculate recovery time
            recovery_months = abs(portfolio_impact) * 24  # Rough estimate
            
            # Assess goal impact
            goal_impact = self._assess_stress_goal_impact(
                portfolio_impact, goal, time_horizon_months
            )
            
            stress_results[scenario_name] = {
                "portfolio_impact": portfolio_impact,
                "recovery_time_months": recovery_months,
                "goal_achievement_probability": goal_impact,
                "severity": "high" if abs(portfolio_impact) > 0.25 else "medium" if abs(portfolio_impact) > 0.15 else "low"
            }
        
        return stress_results
    
    def _analyze_goal_specific_risks(
        self,
        allocation: Dict[str, float],
        goal: FinancialGoal,
        time_horizon_months: int
    ) -> Dict[str, Any]:
        """Analyze risks specific to the financial goal"""
        goal_risks = {
            "time_horizon_risk": "low" if time_horizon_months > 120 else "high",
            "goal_feasibility": "high" if time_horizon_months > 60 else "medium",
            "liquidity_risk": "low" if goal.goal_type != GoalType.EMERGENCY_FUND else "critical"
        }
        
        alignment_score = 0.8  # Placeholder calculation
        
        return {
            "goal_specific_risks": goal_risks,
            "risk_goal_alignment": alignment_score,
            "recommendations": self._generate_goal_risk_recommendations(goal, allocation)
        }
    
    def _generate_goal_risk_recommendations(
        self, 
        goal: FinancialGoal, 
        allocation: Dict[str, float]
    ) -> List[str]:
        """Generate risk recommendations based on goal"""
        recommendations = []
        
        if goal.goal_type == GoalType.RETIREMENT:
            recommendations.extend([
                "Consider target-date fund approach",
                "Implement glide path strategy",
                "Regular rebalancing recommended"
            ])
        elif goal.goal_type == GoalType.EMERGENCY_FUND:
            recommendations.extend([
                "Prioritize liquidity over returns",
                "Use high-yield savings accounts",
                "Avoid market risk"
            ])
        elif goal.goal_type == GoalType.HOME_PURCHASE:
            recommendations.extend([
                "Conservative allocation recommended",
                "Avoid equity risk near purchase date",
                "Consider CDs or bonds"
            ])
        
        return recommendations
    
    def _generate_risk_optimization_suggestions(
        self,
        allocation: Dict[str, float],
        risk_metrics: Dict[str, Any],
        goal: FinancialGoal
    ) -> List[Dict[str, Any]]:
        """Generate risk optimization suggestions"""
        suggestions = []
        
        overall_risk = risk_metrics["overall_risk_score"]
        
        if overall_risk > 0.8:
            suggestions.append({
                "type": "reduce_risk",
                "description": "Consider reducing equity allocation",
                "impact": "Lower volatility and drawdown risk",
                "priority": "high"
            })
        
        if overall_risk < 0.3 and goal.goal_type in [GoalType.RETIREMENT, GoalType.WEALTH_BUILDING]:
            suggestions.append({
                "type": "increase_growth",
                "description": "Consider increasing growth allocation for better returns",
                "impact": "Higher expected returns with manageable risk",
                "priority": "medium"
            })
        
        return suggestions
    
    def _assess_risk_capacity(self, goal: FinancialGoal, time_horizon_months: int) -> Dict[str, Any]:
        """Assess risk capacity based on goal and timeline"""
        capacity_score = min(1.0, time_horizon_months / 120)  # 10 years = full capacity
        
        capacity_level = "high" if capacity_score > 0.7 else "medium" if capacity_score > 0.4 else "low"
        
        return {
            "capacity_score": capacity_score,
            "capacity_level": capacity_level,
            "factors": {
                "time_horizon": time_horizon_months,
                "goal_flexibility": "medium",
                "income_stability": "assumed_stable"
            }
        }
    
    def _analyze_diversification(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze portfolio diversification"""
        # Calculate concentration metrics
        hhi = sum(weight ** 2 for weight in allocation.values())
        effective_assets = 1 / hhi if hhi > 0 else 0
        
        diversification_score = 1.0 - hhi
        
        return {
            "diversification_score": diversification_score,
            "effective_number_assets": effective_assets,
            "concentration_risk": "high" if hhi > 0.5 else "medium" if hhi > 0.3 else "low",
            "recommendations": self._get_diversification_recommendations(allocation, diversification_score)
        }
    
    def _get_diversification_recommendations(self, allocation: Dict[str, float], score: float) -> List[str]:
        """Get diversification improvement recommendations"""
        recommendations = []
        
        if score < 0.5:
            recommendations.append("Increase diversification across asset classes")
        
        # Check for over-concentration
        for asset, weight in allocation.items():
            if weight > 0.6:
                recommendations.append(f"Reduce concentration in {asset}")
        
        if len(allocation) < 3:
            recommendations.append("Consider adding more asset classes")
        
        return recommendations
    
    def _assess_stress_goal_impact(
        self, 
        portfolio_impact: float, 
        goal: FinancialGoal, 
        time_horizon_months: int
    ) -> float:
        """Assess impact of stress scenario on goal achievement"""
        base_probability = 0.8  # Assume 80% base success probability
        
        # Adjust for portfolio impact
        impact_adjustment = abs(portfolio_impact) * 2  # Double the impact
        
        # Adjust for time horizon (more time = better recovery)
        time_adjustment = min(0.3, time_horizon_months / 240)  # 20 years = full adjustment
        
        adjusted_probability = base_probability - impact_adjustment + time_adjustment
        
        return max(0.1, min(0.95, adjusted_probability))
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize overall risk level"""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        else:
            return "high"
    
    def _initialize_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Initialize stress test scenarios"""
        return {
            "market_crash_2008": {
                "stocks": -0.37,
                "bonds": 0.05,
                "real_estate": -0.30,
                "commodities": -0.25,
                "cash": 0.0
            },
            "dot_com_crash": {
                "stocks": -0.49,
                "bonds": 0.08,
                "real_estate": -0.10,
                "commodities": -0.15,
                "cash": 0.0
            },
            "inflation_spike": {
                "stocks": -0.15,
                "bonds": -0.20,
                "real_estate": 0.05,
                "commodities": 0.25,
                "cash": -0.05
            },
            "interest_rate_shock": {
                "stocks": -0.20,
                "bonds": -0.15,
                "real_estate": -0.25,
                "commodities": -0.10,
                "cash": 0.02
            }
        }


class PortfolioBalancer:
    """
    Advanced portfolio balancing logic with tax-efficient rebalancing.
    Implements sophisticated rebalancing strategies and tax optimization.
    """
    
    def __init__(self):
        self.rebalancing_methods = self._initialize_rebalancing_methods()
        self.tax_efficiency_rules = self._initialize_tax_efficiency_rules()
    
    def _initialize_tax_efficiency_rules(self) -> Dict[str, Any]:
        """Initialize tax efficiency rules"""
        return {
            "tax_advantaged_priority": ["401k", "ira", "roth_ira"],
            "tax_inefficient_assets": ["bonds", "reits", "commodities"],
            "tax_efficient_assets": ["index_funds", "municipal_bonds"],
            "loss_harvesting_threshold": 0.05
        }
    
    def create_rebalancing_plan(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        portfolio_value: Decimal,
        tax_context: TaxContext,
        account_types: Dict[str, str]  # account_id -> account_type mapping
    ) -> Dict[str, Any]:
        """
        Create comprehensive rebalancing plan with tax optimization.
        
        Args:
            current_allocation: Current portfolio allocation
            target_allocation: Target portfolio allocation
            portfolio_value: Total portfolio value
            tax_context: Tax context for optimization
            account_types: Mapping of accounts to their tax treatment
            
        Returns:
            Detailed rebalancing plan with tax-efficient implementation
        """
        # Calculate rebalancing needs
        rebalancing_needs = self._calculate_rebalancing_needs(
            current_allocation, target_allocation, portfolio_value
        )
        
        # Optimize for tax efficiency
        tax_optimized_plan = self._optimize_rebalancing_for_taxes(
            rebalancing_needs, tax_context, account_types
        )
        
        # Create implementation timeline
        implementation_plan = self._create_rebalancing_timeline(
            tax_optimized_plan, tax_context
        )
        
        # Calculate costs and benefits
        cost_benefit_analysis = self._analyze_rebalancing_costs_benefits(
            tax_optimized_plan, portfolio_value
        )
        
        return {
            "rebalancing_needed": len(rebalancing_needs) > 0,
            "rebalancing_actions": tax_optimized_plan,
            "implementation_timeline": implementation_plan,
            "cost_benefit_analysis": cost_benefit_analysis,
            "tax_implications": self._calculate_tax_implications(tax_optimized_plan, tax_context),
            "frequency_recommendation": self._recommend_rebalancing_frequency(current_allocation, target_allocation)
        }
    
    def _calculate_rebalancing_needs(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        portfolio_value: Decimal
    ) -> List[Dict[str, Any]]:
        """Calculate specific rebalancing needs"""
        needs = []
        threshold = 0.05  # 5% threshold for rebalancing
        
        all_assets = set(list(current.keys()) + list(target.keys()))
        
        for asset in all_assets:
            current_weight = current.get(asset, 0.0)
            target_weight = target.get(asset, 0.0)
            difference = target_weight - current_weight
            
            if abs(difference) > threshold:
                dollar_amount = abs(difference) * float(portfolio_value)
                
                needs.append({
                    "asset_class": asset,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "weight_difference": difference,
                    "dollar_amount": float(dollar_amount),
                    "action": "buy" if difference > 0 else "sell",
                    "priority": "high" if abs(difference) > 0.15 else "medium"
                })
        
        return sorted(needs, key=lambda x: abs(x["weight_difference"]), reverse=True)
    
    def _optimize_rebalancing_for_taxes(
        self,
        rebalancing_needs: List[Dict[str, Any]],
        tax_context: TaxContext,
        account_types: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Optimize rebalancing plan for tax efficiency"""
        optimized_plan = []
        
        for need in rebalancing_needs:
            # Determine best account for this rebalancing action
            best_account = self._select_best_account_for_rebalancing(
                need, tax_context, account_types
            )
            
            # Calculate tax implications
            tax_impact = self._calculate_action_tax_impact(need, best_account, tax_context)
            
            optimized_action = need.copy()
            optimized_action.update({
                "recommended_account": best_account,
                "tax_impact": tax_impact,
                "tax_efficiency_score": self._calculate_tax_efficiency_score(need, best_account)
            })
            
            optimized_plan.append(optimized_action)
        
        return optimized_plan
    
    def _calculate_action_tax_impact(
        self, 
        need: Dict[str, Any], 
        account: str, 
        tax_context: TaxContext
    ) -> Dict[str, Any]:
        """Calculate tax impact of rebalancing action"""
        if "401k" in account or "ira" in account:
            # Tax-advantaged account - no immediate tax impact
            return {"immediate_tax": 0.0, "deferred_tax": True}
        else:
            # Taxable account
            if need["action"] == "sell":
                # Assume some capital gains
                estimated_gain = need["dollar_amount"] * 0.2  # 20% gain assumption
                tax_impact = estimated_gain * tax_context.marginal_tax_rate
                return {"immediate_tax": tax_impact, "deferred_tax": False}
            else:
                return {"immediate_tax": 0.0, "deferred_tax": False}
    
    def _calculate_tax_efficiency_score(self, need: Dict[str, Any], account: str) -> float:
        """Calculate tax efficiency score for action"""
        if "401k" in account or "ira" in account:
            return 1.0  # Highest efficiency for tax-advantaged accounts
        elif need["action"] == "buy":
            return 0.8  # Good efficiency for buying in taxable
        else:
            return 0.4  # Lower efficiency for selling in taxable (potential taxes)
    
    def _create_rebalancing_timeline(
        self, 
        optimized_plan: List[Dict[str, Any]], 
        tax_context: TaxContext
    ) -> Dict[str, Any]:
        """Create timeline for rebalancing implementation"""
        high_priority = [action for action in optimized_plan if action.get("priority") == "high"]
        medium_priority = [action for action in optimized_plan if action.get("priority") == "medium"]
        
        return {
            "immediate": [f"Execute {action['asset_class']} {action['action']}" for action in high_priority[:2]],
            "within_30_days": [f"Execute {action['asset_class']} {action['action']}" for action in high_priority[2:] + medium_priority[:2]],
            "within_90_days": [f"Execute {action['asset_class']} {action['action']}" for action in medium_priority[2:]],
            "ongoing": ["Monitor allocation drift", "Review quarterly"]
        }
    
    def _analyze_rebalancing_costs_benefits(
        self, 
        optimized_plan: List[Dict[str, Any]], 
        portfolio_value: Decimal
    ) -> Dict[str, Any]:
        """Analyze costs and benefits of rebalancing"""
        total_trades = len(optimized_plan)
        estimated_cost = total_trades * 0.001 * float(portfolio_value)  # 0.1% per trade
        
        # Estimate benefit from better allocation
        allocation_improvement = sum(abs(action["weight_difference"]) for action in optimized_plan)
        estimated_benefit = allocation_improvement * 0.02 * float(portfolio_value)  # 2% improvement per unit
        
        return {
            "total_cost": estimated_cost,
            "estimated_benefit": estimated_benefit,
            "net_benefit": estimated_benefit - estimated_cost,
            "cost_benefit_ratio": estimated_benefit / max(estimated_cost, 1),
            "recommendation": "proceed" if estimated_benefit > estimated_cost * 2 else "defer"
        }
    
    def _calculate_tax_implications(
        self, 
        optimized_plan: List[Dict[str, Any]], 
        tax_context: TaxContext
    ) -> Dict[str, Any]:
        """Calculate comprehensive tax implications"""
        total_tax_impact = sum(
            action.get("tax_impact", {}).get("immediate_tax", 0) 
            for action in optimized_plan
        )
        
        tax_loss_opportunities = [
            action for action in optimized_plan 
            if action["action"] == "sell" and action.get("tax_impact", {}).get("immediate_tax", 0) < 0
        ]
        
        return {
            "total_tax_cost": total_tax_impact,
            "tax_loss_harvesting_opportunities": len(tax_loss_opportunities),
            "net_tax_efficiency": 1.0 - (total_tax_impact / sum(action["dollar_amount"] for action in optimized_plan)),
            "recommendations": self._get_tax_optimization_recommendations(optimized_plan, tax_context)
        }
    
    def _get_tax_optimization_recommendations(
        self, 
        optimized_plan: List[Dict[str, Any]], 
        tax_context: TaxContext
    ) -> List[str]:
        """Get tax optimization recommendations"""
        recommendations = []
        
        if any(action.get("tax_impact", {}).get("immediate_tax", 0) > 1000 for action in optimized_plan):
            recommendations.append("Consider spreading sales across tax years")
        
        if tax_context.marginal_tax_rate > 0.22:
            recommendations.append("Prioritize tax-advantaged account rebalancing")
        
        recommendations.append("Consider tax-loss harvesting opportunities")
        
        return recommendations
    
    def _recommend_rebalancing_frequency(
        self, 
        current: Dict[str, float], 
        target: Dict[str, float]
    ) -> Dict[str, Any]:
        """Recommend optimal rebalancing frequency"""
        max_deviation = max(
            abs(target.get(asset, 0) - current.get(asset, 0)) 
            for asset in set(list(current.keys()) + list(target.keys()))
        )
        
        if max_deviation > 0.15:
            frequency = "monthly"
        elif max_deviation > 0.10:
            frequency = "quarterly"
        elif max_deviation > 0.05:
            frequency = "semi_annually"
        else:
            frequency = "annually"
        
        return {
            "recommended_frequency": frequency,
            "current_deviation": max_deviation,
            "rationale": f"Maximum deviation of {max_deviation:.1%} suggests {frequency} rebalancing"
        }
    
    def _select_best_account_for_rebalancing(
        self,
        need: Dict[str, Any],
        tax_context: TaxContext,
        account_types: Dict[str, str]
    ) -> str:
        """Select the most tax-efficient account for rebalancing action"""
        action = need["action"]
        asset_class = need["asset_class"]
        
        # Tax-efficiency preferences
        if action == "sell":
            # Prefer tax-advantaged accounts for selling
            for account_id, account_type in account_types.items():
                if account_type in ["401k", "ira", "roth_ira"]:
                    return account_id
        
        elif action == "buy":
            # For buying, consider asset location optimization
            if asset_class in ["bonds", "reits"]:  # Tax-inefficient assets
                for account_id, account_type in account_types.items():
                    if account_type in ["401k", "ira"]:
                        return account_id
            else:  # Tax-efficient assets
                for account_id, account_type in account_types.items():
                    if account_type == "taxable":
                        return account_id
        
        # Default to first available account
        return list(account_types.keys())[0] if account_types else "default_account"
    
    def _initialize_rebalancing_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rebalancing method configurations"""
        return {
            "threshold_based": {
                "threshold": 0.05,
                "frequency": "quarterly",
                "tax_aware": True
            },
            "calendar_based": {
                "frequency": "annual",
                "month": "december",
                "tax_aware": True
            },
            "tolerance_band": {
                "inner_threshold": 0.03,
                "outer_threshold": 0.08,
                "tax_aware": True
            }
        }
    
    def _select_best_optimization(
        self, 
        optimization_results: Dict[str, Dict[str, Any]], 
        risk_tolerance: float
    ) -> Dict[str, Any]:
        """Select best optimization result based on risk tolerance"""
        
        # Score each result based on Sharpe ratio and risk tolerance alignment
        best_result = None
        best_score = -float('inf')
        
        for method, result in optimization_results.items():
            sharpe_ratio = result["sharpe_ratio"]
            volatility = result["volatility"]
            
            # Adjust score based on risk tolerance
            if risk_tolerance < 0.3:  # Conservative
                # Penalize high volatility
                score = sharpe_ratio - (volatility * 2)
            elif risk_tolerance > 0.7:  # Aggressive
                # Reward higher Sharpe ratios
                score = sharpe_ratio * 1.5
            else:  # Moderate
                score = sharpe_ratio
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result or list(optimization_results.values())[0]
    
    def _calculate_portfolio_analytics(
        self,
        optimization_result: Dict[str, Any],
        instruments: List[FinancialInstrument],
        returns_data: Dict[str, float],
        covariance_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio analytics"""
        allocation = optimization_result["allocation"]
        
        # Asset class breakdown
        asset_class_allocation = {}
        for instrument in instruments:
            asset_class = instrument.instrument_type.value
            weight = allocation.get(instrument.instrument_id, 0)
            
            if asset_class in asset_class_allocation:
                asset_class_allocation[asset_class] += weight
            else:
                asset_class_allocation[asset_class] = weight
        
        # Concentration metrics
        weights = list(allocation.values())
        herfindahl_index = sum(w**2 for w in weights)
        effective_number_assets = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        # Risk contribution analysis
        weights_array = np.array([allocation[inst.instrument_id] for inst in instruments])
        portfolio_variance = np.dot(weights_array, np.dot(covariance_matrix, weights_array))
        
        risk_contributions = {}
        for i, instrument in enumerate(instruments):
            marginal_contribution = np.dot(covariance_matrix[i], weights_array)
            risk_contribution = weights_array[i] * marginal_contribution / portfolio_variance
            risk_contributions[instrument.instrument_id] = float(risk_contribution)
        
        return {
            "asset_class_allocation": asset_class_allocation,
            "concentration_metrics": {
                "herfindahl_index": float(herfindahl_index),
                "effective_number_assets": float(effective_number_assets),
                "max_single_position": max(weights) if weights else 0
            },
            "risk_contributions": risk_contributions,
            "diversification_ratio": self._calculate_diversification_ratio(
                weights_array, covariance_matrix, instruments
            ),
            "expense_weighted_average": self._calculate_weighted_expense_ratio(
                allocation, instruments
            )
        }
    
    def _calculate_diversification_ratio(
        self, 
        weights: np.ndarray, 
        covariance_matrix: np.ndarray, 
        instruments: List[FinancialInstrument]
    ) -> float:
        """Calculate diversification ratio"""
        # Weighted average volatility
        individual_volatilities = np.array([inst.volatility for inst in instruments])
        weighted_avg_volatility = np.dot(weights, individual_volatilities)
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Diversification ratio
        if portfolio_volatility > 0:
            return float(weighted_avg_volatility / portfolio_volatility)
        else:
            return 1.0
    
    def _calculate_weighted_expense_ratio(
        self, 
        allocation: Dict[str, float], 
        instruments: List[FinancialInstrument]
    ) -> float:
        """Calculate weighted average expense ratio"""
        weighted_expense = 0.0
        
        for instrument in instruments:
            weight = allocation.get(instrument.instrument_id, 0)
            weighted_expense += weight * instrument.expense_ratio
        
        return weighted_expense
    
    def _calculate_risk_metrics(
        self, 
        optimization_result: Dict[str, Any], 
        instruments: List[FinancialInstrument]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        expected_return = optimization_result["expected_return"]
        volatility = optimization_result["volatility"]
        sharpe_ratio = optimization_result["sharpe_ratio"]
        
        # Value at Risk (simplified)
        var_95 = expected_return - (1.645 * volatility)  # 95% VaR
        var_99 = expected_return - (2.326 * volatility)  # 99% VaR
        
        # Expected Shortfall (simplified)
        es_95 = expected_return - (2.063 * volatility)  # 95% Expected Shortfall
        
        # Maximum theoretical loss (3 standard deviations)
        max_theoretical_loss = expected_return - (3 * volatility)
        
        return {
            "value_at_risk": {
                "var_95": float(var_95),
                "var_99": float(var_99)
            },
            "expected_shortfall_95": float(es_95),
            "max_theoretical_loss": float(max_theoretical_loss),
            "sharpe_ratio": float(sharpe_ratio),
            "volatility": float(volatility),
            "risk_adjusted_return": float(expected_return / volatility) if volatility > 0 else 0,
            "downside_capture_ratio": 0.8,  # Simplified assumption
            "upside_capture_ratio": 1.1     # Simplified assumption
        }