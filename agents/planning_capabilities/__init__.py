"""
Advanced Planning Capabilities Module

Sophisticated financial planning features organized by domain.

Current structure: Main implementation in parent advanced_planning_capabilities.py
Future organization:
- models.py - Data models, enums, and dataclasses
- goal_decomposition.py - GoalDecompositionSystem (~469 lines)
- time_horizon.py - TimeHorizonPlanner (~450 lines)
- portfolio_optimization.py - Optimization algorithms (~504 lines)
- asset_allocation.py - AssetAllocationOptimizer (~267 lines)
- retirement_planning.py - RetirementPlanningEngine (~396 lines)
- risk_assessment.py - AdvancedRiskAssessment (~334 lines)
- portfolio_balancer.py - PortfolioBalancer (~464 lines)
"""

# Import from parent directory for backward compatibility
from ..advanced_planning_capabilities import (
    # Enums
    GoalType,
    MarketScenario,
    InstrumentType,
    # Data classes
    FinancialGoal,
    Milestone,
    MarketScenarioData,
    FinancialInstrument,
    # Systems
    GoalDecompositionSystem,
    TimeHorizonPlanner,
    RiskAdjustedReturnOptimizer,
    AssetAllocationOptimizer,
    RetirementPlanningEngine,
    AdvancedRiskAssessment,
    PortfolioBalancer,
)

__all__ = [
    # Enums
    "GoalType",
    "MarketScenario",
    "InstrumentType",
    # Data classes
    "FinancialGoal",
    "Milestone",
    "MarketScenarioData",
    "FinancialInstrument",
    # Systems
    "GoalDecompositionSystem",
    "TimeHorizonPlanner",
    "RiskAdjustedReturnOptimizer",
    "AssetAllocationOptimizer",
    "RetirementPlanningEngine",
    "AdvancedRiskAssessment",
    "PortfolioBalancer",
]
