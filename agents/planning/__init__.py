"""
Planning Agent Module

Modular planning agent implementation with separated concerns:
- plan_adjustment: Life event adjustment logic
- Future: search_algorithms (GSM/ToS), core (main PlanningAgent)
"""

from .plan_adjustment import PlanAdjustmentLogic

__all__ = [
    "PlanAdjustmentLogic",
]
