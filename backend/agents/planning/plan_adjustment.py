"""
Plan Adjustment Logic for handling life events and generating plan modifications.

Implements specific adjustment strategies for job loss, medical emergencies,
business disruptions, and multi-trigger handling.

Requirements: 1.3, 2.3, 3.2, 5.2, 7.2
"""

import logging
from typing import Any, Dict, List, Optional

from data_models.schemas import (
    ClassifiedTrigger, LifeEventType, UrgencyLevel
)


class PlanAdjustmentLogic:
    """
    Core Plan Adjustment Logic for handling life events and generating plan modifications.

    Implements specific adjustment strategies for:
    - Job loss scenarios
    - Medical emergencies
    - Business disruptions
    - Multi-trigger handling for simultaneous life events

    Requirements: 1.3, 2.3, 3.2, 5.2, 7.2
    """

    def __init__(self):
        self.logger = logging.getLogger("finpilot.planning.adjustment_logic")

        # Adjustment strategy configurations
        self.job_loss_config = {
            "emergency_fund_multiplier": 1.5,  # Increase emergency fund by 50%
            "expense_reduction_target": 0.25,  # Reduce expenses by 25%
            "investment_pause_threshold": 0.8,  # Pause investments if emergency fund < 80% target
            "income_replacement_months": 6,    # Plan for 6 months without income
            "priority_expenses": ["housing", "utilities", "food", "insurance", "debt_payments"]
        }

        self.medical_emergency_config = {
            "healthcare_fund_target": 10000,   # Target healthcare emergency fund
            "hsa_utilization_priority": True,  # Prioritize HSA usage
            "insurance_coverage_check": True,  # Verify insurance coverage
            "emergency_fund_preservation": 0.7, # Preserve 70% of emergency fund
            "payment_plan_consideration": True  # Consider payment plans
        }

        self.business_disruption_config = {
            "income_volatility_buffer": 0.3,   # 30% income volatility buffer
            "business_expense_adjustment": 0.2, # Adjust for 20% higher business expenses
            "cash_flow_smoothing": True,       # Implement cash flow smoothing
            "scenario_planning_months": 12,    # Plan for 12-month scenarios
            "contingency_fund_target": 0.15   # 15% of assets in contingency fund
        }

        # Multi-trigger handling weights
        self.trigger_priority_weights = {
            LifeEventType.MEDICAL_EMERGENCY: 1.0,
            LifeEventType.JOB_LOSS: 0.9,
            LifeEventType.BUSINESS_DISRUPTION: 0.8,
            LifeEventType.INCOME_CHANGE: 0.6,
            LifeEventType.MAJOR_EXPENSE: 0.5,
            LifeEventType.FAMILY_CHANGE: 0.4
        }

    def calculate_adjustments(
        self,
        trigger: ClassifiedTrigger,
        current_plan: Dict[str, Any],
        financial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate plan adjustments for a single trigger event.

        Args:
            trigger: Classified trigger event
            current_plan: Current financial plan
            financial_state: Current financial state

        Returns:
            Dictionary containing adjustment recommendations
        """
        try:
            self.logger.info(f"Calculating adjustments for trigger: {trigger.classification}")

            # Determine trigger type from classification
            trigger_type = self._extract_trigger_type(trigger.classification)

            if trigger_type == LifeEventType.JOB_LOSS:
                return self._calculate_job_loss_adjustments(trigger, current_plan, financial_state)
            elif trigger_type == LifeEventType.MEDICAL_EMERGENCY:
                return self._calculate_medical_emergency_adjustments(trigger, current_plan, financial_state)
            elif trigger_type == LifeEventType.BUSINESS_DISRUPTION:
                return self._calculate_business_disruption_adjustments(trigger, current_plan, financial_state)
            else:
                return self._calculate_general_adjustments(trigger, current_plan, financial_state)

        except Exception as e:
            self.logger.error(f"Failed to calculate adjustments: {str(e)}")
            raise

    def handle_multiple_triggers(
        self,
        triggers: List[ClassifiedTrigger],
        current_plan: Dict[str, Any],
        financial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle multiple simultaneous life events with integrated solutions.

        Args:
            triggers: List of classified trigger events
            current_plan: Current financial plan
            financial_state: Current financial state

        Returns:
            Dictionary containing integrated adjustment recommendations
        """
        try:
            self.logger.info(f"Handling {len(triggers)} simultaneous triggers")

            # Sort triggers by priority
            sorted_triggers = self._prioritize_triggers(triggers)

            # Calculate individual adjustments
            individual_adjustments = []
            for trigger in sorted_triggers:
                adjustment = self.calculate_adjustments(trigger, current_plan, financial_state)
                individual_adjustments.append({
                    "trigger": trigger,
                    "adjustment": adjustment
                })

            # Integrate adjustments with conflict resolution
            integrated_adjustment = self._integrate_adjustments(
                individual_adjustments, current_plan, financial_state
            )

            # Apply compound scenario optimizations
            optimized_adjustment = self._optimize_compound_scenario(
                integrated_adjustment, sorted_triggers, financial_state
            )

            return optimized_adjustment

        except Exception as e:
            self.logger.error(f"Failed to handle multiple triggers: {str(e)}")
            raise

    def preserve_core_goals(self, adjustment: Dict[str, Any], original_goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ensure core financial goals are preserved in adjustments.

        Args:
            adjustment: Proposed adjustment
            original_goals: List of original financial goals

        Returns:
            Adjustment with core goal preservation
        """
        preserved_adjustment = adjustment.copy()

        # Identify critical goals that must be preserved
        critical_goals = [goal for goal in original_goals if goal.get("priority", "medium") == "critical"]

        # Adjust timeline rather than abandoning goals
        for goal in critical_goals:
            goal_id = goal.get("goal_id")
            if goal_id in preserved_adjustment.get("affected_goals", {}):
                affected_goal = preserved_adjustment["affected_goals"][goal_id]

                # Extend timeline instead of reducing target
                if affected_goal.get("target_reduction", 0) > 0.1:  # More than 10% reduction
                    timeline_extension = affected_goal["target_reduction"] * 24  # Convert to months
                    affected_goal["timeline_extension_months"] = timeline_extension
                    affected_goal["target_reduction"] = 0.05  # Minimal reduction

                    self.logger.info(f"Preserved goal {goal_id} by extending timeline by {timeline_extension} months")

        return preserved_adjustment

    def _extract_trigger_type(self, classification: str) -> Optional[LifeEventType]:
        """Extract LifeEventType from classification string"""
        for event_type in LifeEventType:
            if event_type.value in classification:
                return event_type
        return None

    def _calculate_job_loss_adjustments(
        self,
        trigger: ClassifiedTrigger,
        current_plan: Dict[str, Any],
        financial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate specific adjustments for job loss scenarios"""
        config = self.job_loss_config

        # Current financial metrics
        monthly_expenses = financial_state.get("monthly_expenses", 5000)
        emergency_fund = financial_state.get("emergency_fund", 15000)
        monthly_income = financial_state.get("monthly_income", 0)  # Will be 0 after job loss

        # Calculate required emergency fund
        target_emergency_fund = monthly_expenses * config["income_replacement_months"] * config["emergency_fund_multiplier"]
        emergency_fund_gap = max(0, target_emergency_fund - emergency_fund)

        # Calculate expense reductions
        target_expense_reduction = monthly_expenses * config["expense_reduction_target"]
        new_monthly_expenses = monthly_expenses - target_expense_reduction

        # Determine investment pauses
        pause_investments = emergency_fund < (target_emergency_fund * config["investment_pause_threshold"])

        adjustments = {
            "adjustment_type": "job_loss_response",
            "urgency_level": trigger.urgency_level.value,
            "confidence_score": 0.85,

            "emergency_fund_adjustments": {
                "current_amount": emergency_fund,
                "target_amount": target_emergency_fund,
                "funding_gap": emergency_fund_gap,
                "funding_sources": ["reduce_investments", "liquidate_non_essential_assets"],
                "timeline_months": 1  # Immediate adjustment
            },

            "expense_adjustments": {
                "current_monthly_expenses": monthly_expenses,
                "target_monthly_expenses": new_monthly_expenses,
                "reduction_amount": target_expense_reduction,
                "priority_expenses": config["priority_expenses"],
                "reduction_categories": ["entertainment", "dining_out", "subscriptions", "travel"]
            },

            "investment_adjustments": {
                "pause_new_investments": pause_investments,
                "reduce_401k_contributions": True,
                "maintain_employer_match": True,  # Always maintain employer match
                "liquidate_for_emergency": emergency_fund_gap > 0
            },

            "income_replacement": {
                "unemployment_benefits_estimate": monthly_income * 0.4,  # Typical 40% replacement
                "job_search_timeline_months": 6,
                "retraining_budget": 2000,
                "networking_budget": 500
            },

            "timeline_adjustments": {
                "retirement_delay_months": 12,
                "major_purchase_delays": ["home_purchase", "car_purchase"],
                "goal_timeline_extensions": 18
            },

            "rationale": f"Job loss requires immediate cash flow preservation and emergency fund strengthening. "
                        f"Reducing expenses by ${target_expense_reduction:.0f}/month and building emergency fund to "
                        f"${target_emergency_fund:.0f} for {config['income_replacement_months']}-month coverage."
        }

        return adjustments

    def _calculate_medical_emergency_adjustments(
        self,
        trigger: ClassifiedTrigger,
        current_plan: Dict[str, Any],
        financial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate specific adjustments for medical emergency scenarios"""
        config = self.medical_emergency_config

        # Current financial metrics
        emergency_fund = financial_state.get("emergency_fund", 15000)
        hsa_balance = financial_state.get("hsa_balance", 0)
        monthly_income = financial_state.get("monthly_income", 8000)

        # Estimate medical costs from trigger data
        estimated_medical_costs = trigger.trigger_event.source_data.get("estimated_costs", 25000)

        # Calculate healthcare funding strategy
        hsa_available = hsa_balance if config["hsa_utilization_priority"] else 0
        emergency_fund_allocation = emergency_fund * (1 - config["emergency_fund_preservation"])
        funding_gap = max(0, estimated_medical_costs - hsa_available - emergency_fund_allocation)

        adjustments = {
            "adjustment_type": "medical_emergency_response",
            "urgency_level": trigger.urgency_level.value,
            "confidence_score": 0.80,

            "healthcare_funding": {
                "estimated_medical_costs": estimated_medical_costs,
                "hsa_utilization": hsa_available,
                "emergency_fund_allocation": emergency_fund_allocation,
                "funding_gap": funding_gap,
                "funding_sources": ["liquidate_investments", "medical_loan", "payment_plan"]
            },

            "emergency_fund_preservation": {
                "preserve_amount": emergency_fund * config["emergency_fund_preservation"],
                "rationale": "Maintain emergency fund for ongoing expenses during recovery"
            },

            "investment_adjustments": {
                "liquidate_amount": min(funding_gap, financial_state.get("liquid_investments", 0)),
                "pause_contributions": True,
                "maintain_employer_match": True
            },

            "insurance_optimization": {
                "verify_coverage": config["insurance_coverage_check"],
                "maximize_benefits": True,
                "appeal_denials": True,
                "negotiate_rates": True
            },

            "cash_flow_management": {
                "payment_plan_setup": config["payment_plan_consideration"],
                "expense_prioritization": ["medical_bills", "essential_living", "insurance_premiums"],
                "income_protection": "consider_disability_insurance"
            },

            "timeline_adjustments": {
                "goal_delays": ["vacation", "major_purchases"],
                "recovery_period_months": 6,
                "financial_recovery_months": 12
            },

            "rationale": f"Medical emergency requires immediate healthcare funding of ${estimated_medical_costs:.0f}. "
                        f"Utilizing HSA (${hsa_available:.0f}) and emergency fund allocation (${emergency_fund_allocation:.0f}) "
                        f"while preserving core emergency reserves for ongoing expenses."
        }

        return adjustments

    def _calculate_business_disruption_adjustments(
        self,
        trigger: ClassifiedTrigger,
        current_plan: Dict[str, Any],
        financial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate specific adjustments for business disruption scenarios"""
        config = self.business_disruption_config

        # Current financial metrics
        monthly_income = financial_state.get("monthly_income", 8000)
        business_income_percentage = financial_state.get("business_income_percentage", 0.7)
        total_assets = financial_state.get("total_assets", 200000)

        # Calculate income volatility impact
        business_income = monthly_income * business_income_percentage
        income_volatility_buffer = business_income * config["income_volatility_buffer"]
        adjusted_planning_income = business_income - income_volatility_buffer

        # Calculate contingency fund requirements
        target_contingency_fund = total_assets * config["contingency_fund_target"]
        current_contingency = financial_state.get("contingency_fund", 0)
        contingency_gap = max(0, target_contingency_fund - current_contingency)

        adjustments = {
            "adjustment_type": "business_disruption_response",
            "urgency_level": trigger.urgency_level.value,
            "confidence_score": 0.75,

            "income_adjustments": {
                "original_business_income": business_income,
                "volatility_buffer": income_volatility_buffer,
                "adjusted_planning_income": adjusted_planning_income,
                "income_smoothing_enabled": config["cash_flow_smoothing"]
            },

            "contingency_planning": {
                "target_contingency_fund": target_contingency_fund,
                "current_contingency": current_contingency,
                "funding_gap": contingency_gap,
                "scenario_planning_months": config["scenario_planning_months"]
            },

            "business_expense_adjustments": {
                "expense_increase_buffer": config["business_expense_adjustment"],
                "cash_flow_management": True,
                "expense_categorization": ["essential", "growth", "discretionary"]
            },

            "investment_strategy": {
                "increase_liquidity": True,
                "reduce_illiquid_investments": True,
                "maintain_diversification": True,
                "business_investment_separation": True
            },

            "scenario_planning": {
                "best_case": {"income_recovery_months": 3, "growth_rate": 0.1},
                "base_case": {"income_recovery_months": 6, "growth_rate": 0.05},
                "worst_case": {"income_recovery_months": 12, "growth_rate": -0.1}
            },

            "timeline_adjustments": {
                "goal_timeline_buffers": 6,  # Add 6-month buffers to all goals
                "major_decisions_delay": 3,   # Delay major financial decisions by 3 months
                "review_frequency_increase": "monthly"  # Increase review frequency
            },

            "rationale": f"Business disruption requires income volatility planning and increased liquidity. "
                        f"Adjusting planning income to ${adjusted_planning_income:.0f}/month and building "
                        f"contingency fund to ${target_contingency_fund:.0f} for business stability."
        }

        return adjustments

    def _calculate_general_adjustments(
        self,
        trigger: ClassifiedTrigger,
        current_plan: Dict[str, Any],
        financial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate general adjustments for other trigger types"""
        return {
            "adjustment_type": "general_response",
            "urgency_level": trigger.urgency_level.value,
            "confidence_score": 0.60,
            "recommended_actions": trigger.recommended_actions,
            "rationale": f"General adjustment for {trigger.classification} trigger"
        }

    def _prioritize_triggers(self, triggers: List[ClassifiedTrigger]) -> List[ClassifiedTrigger]:
        """Prioritize triggers based on type and severity"""
        def priority_key(trigger):
            trigger_type = self._extract_trigger_type(trigger.classification)
            type_weight = self.trigger_priority_weights.get(trigger_type, 0.5)
            return (trigger.priority_score * type_weight, trigger.urgency_level == UrgencyLevel.IMMEDIATE)

        return sorted(triggers, key=priority_key, reverse=True)

    def _integrate_adjustments(
        self,
        individual_adjustments: List[Dict[str, Any]],
        current_plan: Dict[str, Any],
        financial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate multiple adjustments with conflict resolution"""
        integrated = {
            "adjustment_type": "multi_trigger_response",
            "trigger_count": len(individual_adjustments),
            "confidence_score": 0.70,
            "emergency_fund_adjustments": {},
            "expense_adjustments": {},
            "investment_adjustments": {},
            "timeline_adjustments": {},
            "rationale_components": []
        }

        # Aggregate emergency fund requirements
        max_emergency_target = 0
        emergency_sources = set()

        for adj_data in individual_adjustments:
            adjustment = adj_data["adjustment"]
            if "emergency_fund_adjustments" in adjustment:
                ef_adj = adjustment["emergency_fund_adjustments"]
                max_emergency_target = max(max_emergency_target, ef_adj.get("target_amount", 0))
                emergency_sources.update(ef_adj.get("funding_sources", []))

            integrated["rationale_components"].append(adjustment.get("rationale", ""))

        integrated["emergency_fund_adjustments"] = {
            "target_amount": max_emergency_target,
            "funding_sources": list(emergency_sources)
        }

        # Aggregate expense reductions (take maximum)
        max_expense_reduction = 0
        all_reduction_categories = set()

        for adj_data in individual_adjustments:
            adjustment = adj_data["adjustment"]
            if "expense_adjustments" in adjustment:
                exp_adj = adjustment["expense_adjustments"]
                max_expense_reduction = max(max_expense_reduction, exp_adj.get("reduction_amount", 0))
                all_reduction_categories.update(exp_adj.get("reduction_categories", []))

        integrated["expense_adjustments"] = {
            "total_reduction_amount": max_expense_reduction,
            "reduction_categories": list(all_reduction_categories)
        }

        # Combine investment adjustments (most conservative approach)
        integrated["investment_adjustments"] = {
            "pause_new_investments": any(
                adj_data["adjustment"].get("investment_adjustments", {}).get("pause_new_investments", False)
                for adj_data in individual_adjustments
            ),
            "maintain_employer_match": True  # Always maintain employer match
        }

        # Combine timeline adjustments (maximum delays)
        max_timeline_extension = 0
        for adj_data in individual_adjustments:
            adjustment = adj_data["adjustment"]
            if "timeline_adjustments" in adjustment:
                timeline_adj = adjustment["timeline_adjustments"]
                max_timeline_extension = max(
                    max_timeline_extension,
                    timeline_adj.get("goal_timeline_extensions", 0)
                )

        integrated["timeline_adjustments"] = {
            "goal_timeline_extensions": max_timeline_extension
        }

        # Create integrated rationale
        integrated["rationale"] = "Multi-trigger response addressing: " + "; ".join(integrated["rationale_components"])

        return integrated

    def _optimize_compound_scenario(
        self,
        integrated_adjustment: Dict[str, Any],
        triggers: List[ClassifiedTrigger],
        financial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply optimizations specific to compound scenarios"""
        optimized = integrated_adjustment.copy()

        # Increase confidence penalty for compound scenarios
        complexity_penalty = min(0.2, len(triggers) * 0.05)
        optimized["confidence_score"] = max(0.5, optimized["confidence_score"] - complexity_penalty)

        # Add compound scenario specific recommendations
        optimized["compound_scenario_optimizations"] = {
            "prioritize_liquidity": True,
            "increase_review_frequency": "weekly",
            "consider_professional_advice": len(triggers) >= 3,
            "stress_test_adjustments": True
        }

        # Adjust emergency fund for compound scenarios
        if "emergency_fund_adjustments" in optimized:
            current_target = optimized["emergency_fund_adjustments"].get("target_amount", 0)
            compound_multiplier = 1 + (len(triggers) - 1) * 0.1  # 10% increase per additional trigger
            optimized["emergency_fund_adjustments"]["target_amount"] = current_target * compound_multiplier

        return optimized
