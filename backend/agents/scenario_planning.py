"""
Scenario Planning Module for Different Market Conditions

Implements comprehensive scenario planning capabilities including:
- Market scenario modeling and stress testing
- Plan adaptation logic for changing constraints
- Tax optimization strategies and regulatory compliance
- Asset allocation optimization under different conditions
- Risk assessment and portfolio balancing logic

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

from .advanced_planning_capabilities import (
    MarketScenario, MarketScenarioData, FinancialGoal, 
    FinancialInstrument, InstrumentType
)
from data_models.schemas import (
    FinancialState, RiskProfile, TaxContext, 
    Constraint, ConstraintType, ConstraintPriority
)


class ScenarioType(str, Enum):
    """Types of planning scenarios"""
    BASE_CASE = "base_case"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    STRESS_TEST = "stress_test"
    BLACK_SWAN = "black_swan"


@dataclass
class ScenarioResult:
    """Results from scenario analysis"""
    scenario_name: str
    scenario_type: ScenarioType
    probability: float
    expected_outcome: Dict[str, Any]
    risk_metrics: Dict[str, float]
    required_adjustments: List[str]
    success_probability: float
    timeline_impact: int  # months adjustment needed


class ScenarioPlanner:
    """
    Advanced scenario planning system for different market conditions.
    Analyzes plan performance under various economic scenarios.
    """
    
    def __init__(self):
        self.market_scenarios = self._initialize_market_scenarios()
        self.stress_test_scenarios = self._initialize_stress_test_scenarios()
        self.economic_indicators = self._initialize_economic_indicators()
    
    def analyze_scenarios(
        self,
        financial_goal: FinancialGoal,
        financial_state: FinancialState,
        risk_profile: RiskProfile,
        current_allocation: Dict[str, float],
        time_horizon_months: int
    ) -> Dict[str, ScenarioResult]:
        """
        Analyze financial plan under multiple scenarios.
        
        Args:
            financial_goal: The financial goal being planned for
            financial_state: Current financial state
            risk_profile: User's risk profile
            current_allocation: Current portfolio allocation
            time_horizon_months: Planning time horizon
            
        Returns:
            Dictionary of scenario results
        """
        scenario_results = {}
        
        # Analyze base case scenario
        base_case = self._analyze_base_case_scenario(
            financial_goal, financial_state, risk_profile, 
            current_allocation, time_horizon_months
        )
        scenario_results["base_case"] = base_case
        
        # Analyze market scenarios
        for scenario_name, scenario_data in self.market_scenarios.items():
            if scenario_data.duration_months <= time_horizon_months:
                result = self._analyze_market_scenario(
                    scenario_name, scenario_data, financial_goal, 
                    financial_state, current_allocation, time_horizon_months
                )
                scenario_results[scenario_name] = result
        
        # Analyze stress test scenarios
        for scenario_name, scenario_data in self.stress_test_scenarios.items():
            result = self._analyze_stress_test_scenario(
                scenario_name, scenario_data, financial_goal,
                financial_state, current_allocation, time_horizon_months
            )
            scenario_results[scenario_name] = result
        
        return scenario_results
    
    def _analyze_base_case_scenario(
        self,
        financial_goal: FinancialGoal,
        financial_state: FinancialState,
        risk_profile: RiskProfile,
        current_allocation: Dict[str, float],
        time_horizon_months: int
    ) -> ScenarioResult:
        """Analyze base case scenario with normal market conditions"""
        
        # Base case assumptions
        annual_return = 0.08  # 8% expected return
        annual_volatility = 0.12  # 12% volatility
        inflation_rate = 0.03  # 3% inflation
        
        # Calculate expected outcome
        current_value = float(financial_state.net_worth)
        monthly_savings = float(financial_state.monthly_cash_flow * Decimal("0.8"))
        
        # Future value calculation with monthly contributions
        monthly_return = annual_return / 12
        months = time_horizon_months
        
        if monthly_return > 0:
            # Future value of current assets
            fv_current = current_value * ((1 + monthly_return) ** months)
            
            # Future value of monthly contributions
            if monthly_savings > 0:
                fv_contributions = monthly_savings * (
                    ((1 + monthly_return) ** months - 1) / monthly_return
                )
            else:
                fv_contributions = 0
            
            total_expected_value = fv_current + fv_contributions
        else:
            total_expected_value = current_value + (monthly_savings * months)
        
        # Adjust for inflation
        real_value = total_expected_value / ((1 + inflation_rate) ** (months / 12))
        
        # Calculate success probability
        target_amount = float(financial_goal.target_amount)
        success_probability = min(0.95, max(0.1, real_value / target_amount))
        
        return ScenarioResult(
            scenario_name="Base Case",
            scenario_type=ScenarioType.BASE_CASE,
            probability=0.6,  # 60% probability of base case
            expected_outcome={
                "final_value": real_value,
                "target_amount": target_amount,
                "shortfall": max(0, target_amount - real_value),
                "surplus": max(0, real_value - target_amount)
            },
            risk_metrics={
                "volatility": annual_volatility,
                "max_drawdown": 0.15,
                "var_95": -0.12,
                "expected_shortfall": -0.18
            },
            required_adjustments=[],
            success_probability=success_probability,
            timeline_impact=0
        )
    
    def _analyze_market_scenario(
        self,
        scenario_name: str,
        scenario_data: MarketScenarioData,
        financial_goal: FinancialGoal,
        financial_state: FinancialState,
        current_allocation: Dict[str, float],
        time_horizon_months: int
    ) -> ScenarioResult:
        """Analyze specific market scenario"""
        
        # Calculate portfolio return under this scenario
        portfolio_return = self._calculate_scenario_portfolio_return(
            current_allocation, scenario_data.expected_returns
        )
        
        # Calculate portfolio volatility
        portfolio_volatility = self._calculate_scenario_portfolio_volatility(
            current_allocation, scenario_data.volatilities, scenario_data.correlations
        )
        
        # Project outcomes
        current_value = float(financial_state.net_worth)
        monthly_savings = float(financial_state.monthly_cash_flow * Decimal("0.8"))
        
        # Adjust for scenario duration and severity
        scenario_months = min(scenario_data.duration_months, time_horizon_months)
        remaining_months = time_horizon_months - scenario_months
        
        # Value during scenario period
        scenario_monthly_return = portfolio_return / 12
        if scenario_monthly_return != 0:
            scenario_end_value = current_value * ((1 + scenario_monthly_return) ** scenario_months)
            scenario_contributions = monthly_savings * (
                ((1 + scenario_monthly_return) ** scenario_months - 1) / scenario_monthly_return
            ) if scenario_monthly_return > 0 else monthly_savings * scenario_months
        else:
            scenario_end_value = current_value
            scenario_contributions = monthly_savings * scenario_months
        
        scenario_total = scenario_end_value + scenario_contributions
        
        # Recovery period (remaining months at normal returns)
        if remaining_months > 0:
            normal_return = 0.08 / 12  # 8% annual return
            if normal_return > 0:
                final_value = scenario_total * ((1 + normal_return) ** remaining_months)
                final_contributions = monthly_savings * (
                    ((1 + normal_return) ** remaining_months - 1) / normal_return
                )
            else:
                final_value = scenario_total
                final_contributions = monthly_savings * remaining_months
            
            total_expected_value = final_value + final_contributions
        else:
            total_expected_value = scenario_total
        
        # Calculate success probability and required adjustments
        target_amount = float(financial_goal.target_amount)
        success_probability = min(0.95, max(0.05, total_expected_value / target_amount))
        
        required_adjustments = []
        timeline_impact = 0
        
        if success_probability < 0.7:
            required_adjustments.extend([
                "Increase monthly savings by 20-30%",
                "Consider extending timeline",
                "Reduce target amount by 10-15%"
            ])
            timeline_impact = 6  # 6 months extension
        
        if portfolio_volatility > 0.25:
            required_adjustments.append("Reduce portfolio risk/volatility")
        
        return ScenarioResult(
            scenario_name=scenario_name,
            scenario_type=ScenarioType.PESSIMISTIC if portfolio_return < 0 else ScenarioType.OPTIMISTIC,
            probability=scenario_data.probability,
            expected_outcome={
                "final_value": total_expected_value,
                "target_amount": target_amount,
                "shortfall": max(0, target_amount - total_expected_value),
                "surplus": max(0, total_expected_value - target_amount),
                "scenario_impact": scenario_total - current_value
            },
            risk_metrics={
                "volatility": portfolio_volatility,
                "scenario_return": portfolio_return,
                "max_drawdown": abs(min(0, portfolio_return)) * 1.5,
                "recovery_time_months": scenario_months
            },
            required_adjustments=required_adjustments,
            success_probability=success_probability,
            timeline_impact=timeline_impact
        )
    
    def _analyze_stress_test_scenario(
        self,
        scenario_name: str,
        scenario_data: Dict[str, Any],
        financial_goal: FinancialGoal,
        financial_state: FinancialState,
        current_allocation: Dict[str, float],
        time_horizon_months: int
    ) -> ScenarioResult:
        """Analyze stress test scenario"""
        
        # Extract stress test parameters
        market_decline = scenario_data.get("market_decline", 0.3)  # 30% decline
        duration_months = scenario_data.get("duration_months", 18)  # 18 months
        recovery_months = scenario_data.get("recovery_months", 24)  # 24 months recovery
        
        current_value = float(financial_state.net_worth)
        monthly_savings = float(financial_state.monthly_cash_flow * Decimal("0.8"))
        
        # Stress period: immediate decline + slow recovery
        stress_end_value = current_value * (1 - market_decline)
        
        # During stress period, assume minimal returns
        stress_contributions = monthly_savings * duration_months * 0.5  # Reduced savings ability
        stress_total = stress_end_value + stress_contributions
        
        # Recovery period
        remaining_months = max(0, time_horizon_months - duration_months)
        if remaining_months > 0:
            # Gradual recovery to normal returns
            recovery_return = 0.12 / 12  # Higher returns during recovery
            recovery_months_actual = min(recovery_months, remaining_months)
            
            recovery_value = stress_total * ((1 + recovery_return) ** recovery_months_actual)
            recovery_contributions = monthly_savings * (
                ((1 + recovery_return) ** recovery_months_actual - 1) / recovery_return
            )
            
            # Normal period after recovery
            post_recovery_months = remaining_months - recovery_months_actual
            if post_recovery_months > 0:
                normal_return = 0.08 / 12
                final_value = (recovery_value + recovery_contributions) * (
                    (1 + normal_return) ** post_recovery_months
                )
                final_contributions = monthly_savings * (
                    ((1 + normal_return) ** post_recovery_months - 1) / normal_return
                )
                total_expected_value = final_value + final_contributions
            else:
                total_expected_value = recovery_value + recovery_contributions
        else:
            total_expected_value = stress_total
        
        # Calculate impact and adjustments
        target_amount = float(financial_goal.target_amount)
        success_probability = min(0.8, max(0.1, total_expected_value / target_amount))
        
        required_adjustments = [
            "Build larger emergency fund",
            "Increase monthly savings by 40-50%",
            "Consider more conservative allocation",
            "Extend timeline by 12-24 months",
            "Reduce target amount by 20-25%"
        ]
        
        return ScenarioResult(
            scenario_name=scenario_name,
            scenario_type=ScenarioType.STRESS_TEST,
            probability=scenario_data.get("probability", 0.1),
            expected_outcome={
                "final_value": total_expected_value,
                "target_amount": target_amount,
                "shortfall": max(0, target_amount - total_expected_value),
                "surplus": max(0, total_expected_value - target_amount),
                "max_decline": market_decline,
                "recovery_duration": recovery_months
            },
            risk_metrics={
                "max_drawdown": market_decline,
                "volatility": 0.35,  # High volatility during stress
                "time_to_recovery": recovery_months,
                "stress_duration": duration_months
            },
            required_adjustments=required_adjustments,
            success_probability=success_probability,
            timeline_impact=18  # 18 months extension recommended
        )
    
    def _calculate_scenario_portfolio_return(
        self, 
        allocation: Dict[str, float], 
        scenario_returns: Dict[str, float]
    ) -> float:
        """Calculate portfolio return under specific scenario"""
        portfolio_return = 0.0
        
        # Map allocation to asset classes
        asset_class_mapping = {
            "stocks": ["stocks", "equity", "growth"],
            "bonds": ["bonds", "fixed_income", "treasury"],
            "real_estate": ["real_estate", "reit"],
            "commodities": ["commodities", "gold", "oil"],
            "cash": ["cash", "money_market"]
        }
        
        for asset_id, weight in allocation.items():
            # Determine asset class
            asset_class = "stocks"  # Default
            for ac, keywords in asset_class_mapping.items():
                if any(keyword in asset_id.lower() for keyword in keywords):
                    asset_class = ac
                    break
            
            # Get scenario return for this asset class
            scenario_return = scenario_returns.get(asset_class, 0.08)  # Default 8%
            portfolio_return += weight * scenario_return
        
        return portfolio_return
    
    def _calculate_scenario_portfolio_volatility(
        self,
        allocation: Dict[str, float],
        scenario_volatilities: Dict[str, float],
        correlations: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate portfolio volatility under scenario"""
        # Simplified calculation - in practice would use full covariance matrix
        weighted_volatility = 0.0
        
        for asset_id, weight in allocation.items():
            # Map to asset class
            asset_class = "stocks"  # Default
            if "bond" in asset_id.lower():
                asset_class = "bonds"
            elif "real_estate" in asset_id.lower() or "reit" in asset_id.lower():
                asset_class = "real_estate"
            elif "cash" in asset_id.lower():
                asset_class = "cash"
            
            volatility = scenario_volatilities.get(asset_class, 0.15)
            weighted_volatility += (weight ** 2) * (volatility ** 2)
        
        # Add correlation effects (simplified)
        correlation_adjustment = 0.8  # Assume moderate positive correlation
        portfolio_volatility = math.sqrt(weighted_volatility * correlation_adjustment)
        
        return portfolio_volatility
    
    def _initialize_market_scenarios(self) -> Dict[str, MarketScenarioData]:
        """Initialize predefined market scenarios"""
        scenarios = {}
        
        # Bull Market Scenario
        scenarios["bull_market"] = MarketScenarioData(
            scenario_name=MarketScenario.BULL_MARKET,
            expected_returns={
                "stocks": 0.15,
                "bonds": 0.06,
                "real_estate": 0.12,
                "commodities": 0.10,
                "cash": 0.02
            },
            volatilities={
                "stocks": 0.18,
                "bonds": 0.04,
                "real_estate": 0.15,
                "commodities": 0.25,
                "cash": 0.01
            },
            correlations={
                ("stocks", "bonds"): 0.1,
                ("stocks", "real_estate"): 0.6,
                ("stocks", "commodities"): 0.3,
                ("bonds", "real_estate"): 0.2
            },
            duration_months=36,
            probability=0.25,
            economic_indicators={
                "gdp_growth": 0.04,
                "unemployment": 0.04,
                "inflation": 0.025
            }
        )
        
        # Bear Market Scenario
        scenarios["bear_market"] = MarketScenarioData(
            scenario_name=MarketScenario.BEAR_MARKET,
            expected_returns={
                "stocks": -0.15,
                "bonds": 0.08,
                "real_estate": -0.05,
                "commodities": -0.10,
                "cash": 0.03
            },
            volatilities={
                "stocks": 0.35,
                "bonds": 0.06,
                "real_estate": 0.25,
                "commodities": 0.40,
                "cash": 0.01
            },
            correlations={
                ("stocks", "bonds"): -0.3,
                ("stocks", "real_estate"): 0.8,
                ("stocks", "commodities"): 0.5,
                ("bonds", "real_estate"): -0.1
            },
            duration_months=18,
            probability=0.15,
            economic_indicators={
                "gdp_growth": -0.02,
                "unemployment": 0.08,
                "inflation": 0.01
            }
        )
        
        # High Inflation Scenario
        scenarios["inflation_spike"] = MarketScenarioData(
            scenario_name=MarketScenario.INFLATION_SPIKE,
            expected_returns={
                "stocks": 0.05,
                "bonds": -0.02,
                "real_estate": 0.08,
                "commodities": 0.20,
                "cash": 0.05
            },
            volatilities={
                "stocks": 0.25,
                "bonds": 0.12,
                "real_estate": 0.18,
                "commodities": 0.30,
                "cash": 0.01
            },
            correlations={
                ("stocks", "bonds"): -0.2,
                ("stocks", "commodities"): 0.4,
                ("bonds", "commodities"): -0.3,
                ("real_estate", "commodities"): 0.3
            },
            duration_months=24,
            probability=0.20,
            economic_indicators={
                "gdp_growth": 0.02,
                "unemployment": 0.05,
                "inflation": 0.06
            }
        )
        
        return scenarios
    
    def _initialize_stress_test_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Initialize stress test scenarios"""
        return {
            "market_crash_2008": {
                "market_decline": 0.37,  # 37% decline
                "duration_months": 18,
                "recovery_months": 30,
                "probability": 0.05,
                "description": "2008-style financial crisis"
            },
            "dot_com_crash": {
                "market_decline": 0.49,  # 49% decline
                "duration_months": 30,
                "recovery_months": 60,
                "probability": 0.03,
                "description": "Technology bubble burst"
            },
            "covid_crash": {
                "market_decline": 0.34,  # 34% decline
                "duration_months": 2,
                "recovery_months": 12,
                "probability": 0.08,
                "description": "Pandemic-style shock"
            },
            "stagflation": {
                "market_decline": 0.25,  # 25% decline
                "duration_months": 48,
                "recovery_months": 36,
                "probability": 0.10,
                "description": "Prolonged stagflation period"
            }
        }
    
    def _initialize_economic_indicators(self) -> Dict[str, Dict[str, float]]:
        """Initialize economic indicator ranges for scenarios"""
        return {
            "gdp_growth": {"min": -0.05, "max": 0.06, "normal": 0.025},
            "unemployment": {"min": 0.03, "max": 0.12, "normal": 0.05},
            "inflation": {"min": -0.01, "max": 0.08, "normal": 0.03},
            "interest_rates": {"min": 0.00, "max": 0.08, "normal": 0.04},
            "corporate_earnings_growth": {"min": -0.20, "max": 0.25, "normal": 0.08}
        }


class PlanAdaptationEngine:
    """
    Plan adaptation logic for changing constraints and market conditions.
    Automatically adjusts financial plans based on new information.
    """
    
    def __init__(self):
        self.adaptation_rules = self._initialize_adaptation_rules()
    
    def _initialize_adaptation_rules(self):
        """Initialize adaptation rules for plan modification"""
        return {
            "market_volatility": {
                "threshold": 0.3,
                "action": "reduce_risk_exposure"
            },
            "income_change": {
                "threshold": 0.2,
                "action": "adjust_savings_rate"
            },
            "constraint_violation": {
                "threshold": 0.1,
                "action": "relax_constraints"
            }
        }
        self.constraint_handlers = self._initialize_constraint_handlers()
    
    def adapt_plan(
        self,
        original_goal: FinancialGoal,
        current_state: FinancialState,
        new_constraints: List[Constraint],
        scenario_results: Dict[str, ScenarioResult],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt financial plan based on changing conditions.
        
        Args:
            original_goal: Original financial goal
            current_state: Current financial state
            new_constraints: New or updated constraints
            scenario_results: Results from scenario analysis
            market_conditions: Current market conditions
            
        Returns:
            Adapted plan with recommendations
        """
        adaptations = {
            "timeline_adjustments": [],
            "allocation_changes": {},
            "savings_rate_changes": {},
            "risk_adjustments": [],
            "constraint_modifications": [],
            "contingency_activations": []
        }
        
        # Analyze constraint changes
        constraint_impacts = self._analyze_constraint_changes(new_constraints, original_goal)
        adaptations["constraint_modifications"] = constraint_impacts
        
        # Analyze scenario performance
        scenario_adaptations = self._analyze_scenario_performance(scenario_results)
        adaptations.update(scenario_adaptations)
        
        # Market condition adaptations
        market_adaptations = self._adapt_to_market_conditions(market_conditions, original_goal)
        adaptations["allocation_changes"].update(market_adaptations.get("allocation_changes", {}))
        
        # Timeline and target adjustments
        timeline_adaptations = self._calculate_timeline_adjustments(scenario_results, original_goal)
        adaptations["timeline_adjustments"] = timeline_adaptations
        
        # Generate implementation plan
        implementation_plan = self._create_implementation_plan(adaptations, original_goal)
        
        return {
            "adaptations": adaptations,
            "implementation_plan": implementation_plan,
            "success_probability_change": self._calculate_success_probability_change(
                scenario_results, adaptations
            ),
            "risk_impact_assessment": self._assess_risk_impact(adaptations),
            "recommended_actions": self._generate_recommended_actions(adaptations)
        }