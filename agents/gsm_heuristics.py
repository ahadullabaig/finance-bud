"""
Advanced Heuristics Module for Guided Search Module (GSM)

Implements sophisticated heuristic evaluation functions for financial planning
path optimization, including risk-adjusted returns, tax efficiency, and
constraint complexity analysis.

Requirements: 7.1, 7.2, 7.3, 8.1
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
from enum import Enum

from data_models.schemas import Constraint, ConstraintType, ConstraintPriority


class MarketCondition(str, Enum):
    """Market condition types for heuristic adjustments"""
    BULL = "bull"
    BEAR = "bear"
    VOLATILE = "volatile"
    STABLE = "stable"
    RECESSION = "recession"


class AdvancedHeuristics:
    """
    Advanced heuristic evaluation system for financial planning optimization.
    
    Provides sophisticated scoring algorithms for path evaluation including
    information gain, state similarity, constraint complexity, and market-aware
    risk-adjusted return calculations.
    """
    
    def __init__(self):
        self.market_condition = MarketCondition.STABLE
        self.risk_free_rate = 0.045  # Current risk-free rate
        self.market_return = 0.10    # Expected market return
        self.inflation_rate = 0.03   # Expected inflation
        
        # Heuristic calibration parameters
        self.calibration_params = {
            "risk_aversion_factor": 2.0,
            "tax_efficiency_weight": 0.22,  # Marginal tax rate
            "liquidity_premium": 0.02,
            "diversification_threshold": 0.7,
            "constraint_penalty_factor": 0.5
        }
    
    def calculate_information_gain_heuristic(
        self, 
        current_state: Dict[str, Any], 
        action: Dict[str, Any],
        goal_context: Dict[str, Any]
    ) -> float:
        """
        Calculate information gain heuristic based on how much an action
        reduces uncertainty about achieving the financial goal.
        """
        # Calculate current uncertainty (entropy)
        current_entropy = self._calculate_state_entropy(current_state, goal_context)
        
        # Simulate state after action
        projected_state = self._project_state_after_action(current_state, action)
        projected_entropy = self._calculate_state_entropy(projected_state, goal_context)
        
        # Information gain is reduction in entropy
        information_gain = current_entropy - projected_entropy
        
        # Normalize to [0, 1] range
        max_possible_gain = current_entropy
        normalized_gain = information_gain / max(max_possible_gain, 0.001)
        
        return max(0.0, min(1.0, normalized_gain))
    
    def calculate_state_similarity_heuristic(
        self,
        current_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        distance_metric: str = "euclidean"
    ) -> float:
        """
        Calculate similarity between current state and goal state using
        various distance metrics adapted for financial planning.
        """
        # Extract key financial metrics
        current_metrics = self._extract_financial_metrics(current_state)
        goal_metrics = self._extract_financial_metrics(goal_state)
        
        if distance_metric == "euclidean":
            distance = self._euclidean_distance(current_metrics, goal_metrics)
        elif distance_metric == "manhattan":
            distance = self._manhattan_distance(current_metrics, goal_metrics)
        elif distance_metric == "cosine":
            distance = self._cosine_distance(current_metrics, goal_metrics)
        else:
            distance = self._weighted_financial_distance(current_metrics, goal_metrics)
        
        # Convert distance to similarity score
        max_distance = self._calculate_max_possible_distance(goal_metrics)
        similarity = 1.0 - (distance / max(max_distance, 0.001))
        
        return max(0.0, min(1.0, similarity))
    
    def calculate_constraint_complexity_heuristic(
        self,
        financial_state: Dict[str, Any],
        constraints: List[Constraint],
        action: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate constraint complexity heuristic based on how well the state
        satisfies constraints and the difficulty of maintaining satisfaction.
        """
        if not constraints:
            return 1.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for constraint in constraints:
            # Calculate satisfaction score for this constraint
            satisfaction_score = self._evaluate_constraint_satisfaction(
                financial_state, constraint, action
            )
            
            # Weight by constraint priority
            priority_weight = self._get_constraint_priority_weight(constraint.priority)
            
            # Adjust for constraint complexity
            complexity_factor = self._calculate_constraint_complexity(constraint)
            adjusted_score = satisfaction_score * complexity_factor
            
            total_score += adjusted_score * priority_weight
            total_weight += priority_weight
        
        return total_score / max(total_weight, 0.001)
    
    def calculate_risk_adjusted_return_heuristic(
        self,
        financial_state: Dict[str, Any],
        action: Dict[str, Any],
        time_horizon: int,
        market_conditions: Optional[MarketCondition] = None
    ) -> float:
        """
        Calculate sophisticated risk-adjusted return heuristic using
        Sharpe ratio, Sortino ratio, and market condition adjustments.
        """
        if market_conditions:
            self.market_condition = market_conditions
        
        # Calculate expected return for the action
        expected_return = self._calculate_expected_return(action, time_horizon)
        
        # Calculate risk metrics
        volatility = self._calculate_action_volatility(action)
        downside_risk = self._calculate_downside_risk(action)
        
        # Calculate Sharpe ratio
        excess_return = expected_return - self.risk_free_rate
        sharpe_ratio = excess_return / max(volatility, 0.001)
        
        # Calculate Sortino ratio (focuses on downside risk)
        sortino_ratio = excess_return / max(downside_risk, 0.001)
        
        # Combine ratios with market condition adjustments
        market_adjustment = self._get_market_condition_adjustment()
        
        # Weight Sharpe and Sortino ratios
        combined_ratio = (sharpe_ratio * 0.6 + sortino_ratio * 0.4) * market_adjustment
        
        # Normalize to [0, 1] range
        # Typical good Sharpe ratios are 1.0-2.0, excellent > 2.0
        normalized_score = self._sigmoid_normalize(combined_ratio, midpoint=1.5, steepness=2.0)
        
        return max(0.0, min(1.0, normalized_score))
    
    def calculate_tax_efficiency_heuristic(
        self,
        financial_state: Dict[str, Any],
        action: Dict[str, Any],
        tax_context: Dict[str, Any]
    ) -> float:
        """
        Calculate tax efficiency heuristic considering tax-advantaged accounts,
        tax-loss harvesting opportunities, and timing optimization.
        """
        # Extract tax-relevant information
        marginal_tax_rate = tax_context.get("marginal_tax_rate", 0.22)
        tax_advantaged_space = tax_context.get("available_tax_advantaged_space", 0)
        current_tax_efficiency = self._calculate_current_tax_efficiency(financial_state, tax_context)
        
        # Calculate tax impact of action
        action_tax_impact = self._calculate_action_tax_impact(action, tax_context)
        
        # Tax-advantaged account utilization score
        tax_account_score = self._calculate_tax_account_utilization_score(
            action, tax_advantaged_space, marginal_tax_rate
        )
        
        # Tax-loss harvesting opportunity score
        tax_loss_score = self._calculate_tax_loss_harvesting_score(
            financial_state, action, tax_context
        )
        
        # Asset location optimization score
        asset_location_score = self._calculate_asset_location_score(
            financial_state, action, tax_context
        )
        
        # Combine scores
        combined_score = (
            tax_account_score * 0.4 +
            tax_loss_score * 0.3 +
            asset_location_score * 0.2 +
            (1.0 - abs(action_tax_impact)) * 0.1
        )
        
        return max(0.0, min(1.0, combined_score))
    
    def calculate_liquidity_score_heuristic(
        self,
        financial_state: Dict[str, Any],
        action: Dict[str, Any],
        liquidity_requirements: Dict[str, Any]
    ) -> float:
        """
        Calculate liquidity score based on maintaining adequate liquid assets
        for emergency needs and planned expenses.
        """
        # Current liquidity position
        current_liquid_assets = self._calculate_liquid_assets(financial_state)
        monthly_expenses = financial_state.get("monthly_expenses", 3000)
        
        # Liquidity impact of action
        action_liquidity_impact = self._calculate_action_liquidity_impact(action)
        projected_liquid_assets = current_liquid_assets + action_liquidity_impact
        
        # Required liquidity levels
        emergency_requirement = monthly_expenses * liquidity_requirements.get("emergency_months", 6)
        planned_expenses = liquidity_requirements.get("planned_expenses_12m", 0)
        total_requirement = emergency_requirement + planned_expenses
        
        # Calculate liquidity ratio
        liquidity_ratio = projected_liquid_assets / max(total_requirement, 0.001)
        
        # Optimal liquidity is 1.0-1.5x requirements
        if liquidity_ratio < 0.5:
            score = liquidity_ratio * 0.4  # Severe penalty for low liquidity
        elif liquidity_ratio < 1.0:
            score = 0.2 + (liquidity_ratio - 0.5) * 1.6  # Linear improvement
        elif liquidity_ratio <= 1.5:
            score = 1.0  # Optimal range
        else:
            # Penalty for excess liquidity (opportunity cost)
            excess_penalty = min(0.3, (liquidity_ratio - 1.5) * 0.1)
            score = 1.0 - excess_penalty
        
        return max(0.0, min(1.0, score))
    
    def calculate_diversification_heuristic(
        self,
        financial_state: Dict[str, Any],
        action: Dict[str, Any],
        target_allocation: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate diversification score using correlation analysis and
        concentration risk metrics.
        """
        # Get current portfolio allocation
        current_allocation = self._extract_portfolio_allocation(financial_state)
        
        # Project allocation after action
        projected_allocation = self._project_allocation_after_action(current_allocation, action)
        
        # Calculate concentration risk (Herfindahl-Hirschman Index)
        hhi = sum(weight ** 2 for weight in projected_allocation.values())
        concentration_score = 1.0 - hhi  # Lower HHI = better diversification
        
        # Calculate deviation from target allocation if provided
        target_deviation_score = 1.0
        if target_allocation:
            target_deviation_score = self._calculate_target_deviation_score(
                projected_allocation, target_allocation
            )
        
        # Calculate correlation-based diversification
        correlation_score = self._calculate_correlation_diversification_score(
            projected_allocation, action
        )
        
        # Combine scores
        diversification_score = (
            concentration_score * 0.4 +
            target_deviation_score * 0.3 +
            correlation_score * 0.3
        )
        
        return max(0.0, min(1.0, diversification_score))
    
    # Helper methods for heuristic calculations
    
    def _calculate_state_entropy(self, state: Dict[str, Any], goal_context: Dict[str, Any]) -> float:
        """Calculate entropy (uncertainty) of a financial state"""
        # Simplified entropy calculation based on goal achievement uncertainty
        net_worth = state.get("total_assets", 0) - state.get("total_liabilities", 0)
        goal_amount = goal_context.get("target_amount", 100000)
        
        # Calculate progress toward goal
        progress = min(net_worth / max(goal_amount, 0.001), 1.0)
        
        # Entropy is highest at 50% progress, lowest at 0% and 100%
        if progress <= 0 or progress >= 1:
            return 0.1  # Low entropy at extremes
        else:
            # Maximum entropy at 0.5 progress
            return -progress * math.log2(progress) - (1 - progress) * math.log2(1 - progress)
    
    def _project_state_after_action(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Project financial state after applying an action"""
        projected_state = state.copy()
        
        action_type = action.get("type", "")
        amount = action.get("amount", 0)
        
        if action_type == "investment":
            projected_state["investments"] = projected_state.get("investments", 0) + amount
            projected_state["cash"] = projected_state.get("cash", 0) - amount
        elif action_type == "debt_paydown":
            projected_state["total_liabilities"] = max(0, projected_state.get("total_liabilities", 0) - amount)
            projected_state["cash"] = projected_state.get("cash", 0) - amount
        
        # Recalculate derived metrics
        projected_state["total_assets"] = (
            projected_state.get("cash", 0) +
            projected_state.get("investments", 0) +
            projected_state.get("real_estate", 0)
        )
        
        return projected_state
    
    def _extract_financial_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Extract key financial metrics for distance calculations"""
        return {
            "net_worth": state.get("total_assets", 0) - state.get("total_liabilities", 0),
            "liquidity_ratio": state.get("cash", 0) / max(state.get("monthly_expenses", 1), 1),
            "debt_to_income": state.get("total_liabilities", 0) / max(state.get("monthly_income", 1), 1),
            "investment_ratio": state.get("investments", 0) / max(state.get("total_assets", 1), 1),
            "savings_rate": (state.get("monthly_income", 0) - state.get("monthly_expenses", 0)) / max(state.get("monthly_income", 1), 1)
        }
    
    def _euclidean_distance(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between financial metric vectors"""
        distance_squared = 0.0
        
        for key in metrics1:
            if key in metrics2:
                diff = metrics1[key] - metrics2[key]
                distance_squared += diff ** 2
        
        return math.sqrt(distance_squared)
    
    def _manhattan_distance(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> float:
        """Calculate Manhattan distance between financial metric vectors"""
        distance = 0.0
        
        for key in metrics1:
            if key in metrics2:
                distance += abs(metrics1[key] - metrics2[key])
        
        return distance
    
    def _cosine_distance(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> float:
        """Calculate cosine distance between financial metric vectors"""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for key in metrics1:
            if key in metrics2:
                dot_product += metrics1[key] * metrics2[key]
                norm1 += metrics1[key] ** 2
                norm2 += metrics2[key] ** 2
        
        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        
        cosine_similarity = dot_product / (norm1 * norm2)
        return 1.0 - cosine_similarity
    
    def _weighted_financial_distance(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> float:
        """Calculate weighted distance with financial importance weights"""
        weights = {
            "net_worth": 0.3,
            "liquidity_ratio": 0.2,
            "debt_to_income": 0.2,
            "investment_ratio": 0.2,
            "savings_rate": 0.1
        }
        
        weighted_distance = 0.0
        
        for key, weight in weights.items():
            if key in metrics1 and key in metrics2:
                diff = abs(metrics1[key] - metrics2[key])
                weighted_distance += weight * diff
        
        return weighted_distance
    
    def _calculate_max_possible_distance(self, goal_metrics: Dict[str, float]) -> float:
        """Calculate maximum possible distance for normalization"""
        # Simplified: assume maximum distance is when all metrics are at opposite extremes
        return sum(abs(value) * 2 for value in goal_metrics.values())
    
    def _evaluate_constraint_satisfaction(
        self,
        financial_state: Dict[str, Any],
        constraint: Constraint,
        action: Optional[Dict[str, Any]] = None
    ) -> float:
        """Evaluate how well a constraint is satisfied (0.0 to 1.0)"""
        
        if constraint.constraint_type == ConstraintType.BUDGET:
            monthly_income = financial_state.get("monthly_income", 0)
            monthly_expenses = financial_state.get("monthly_expenses", 0)
            
            if monthly_income > 0:
                expense_ratio = monthly_expenses / monthly_income
                # Good if expenses < 80% of income
                if expense_ratio <= 0.8:
                    return 1.0
                elif expense_ratio <= 1.0:
                    return (1.0 - expense_ratio) / 0.2
                else:
                    return 0.0
            return 0.5
        
        elif constraint.constraint_type == ConstraintType.LIQUIDITY:
            emergency_fund = financial_state.get("emergency_fund", 0)
            monthly_expenses = financial_state.get("monthly_expenses", 0)
            
            if monthly_expenses > 0:
                months_covered = emergency_fund / monthly_expenses
                # Target is 6 months
                if months_covered >= 6:
                    return 1.0
                elif months_covered >= 3:
                    return months_covered / 6
                else:
                    return months_covered / 6 * 0.5
            return 0.5
        
        elif constraint.constraint_type == ConstraintType.RISK:
            # Simplified risk constraint evaluation
            return 0.8  # Assume most actions meet risk constraints
        
        return 0.7  # Default satisfaction for unknown constraints
    
    def _get_constraint_priority_weight(self, priority: ConstraintPriority) -> float:
        """Get weight for constraint priority"""
        weights = {
            ConstraintPriority.MANDATORY: 1.0,
            ConstraintPriority.HIGH: 0.8,
            ConstraintPriority.MEDIUM: 0.6,
            ConstraintPriority.LOW: 0.4,
            ConstraintPriority.OPTIONAL: 0.2
        }
        return weights.get(priority, 0.6)
    
    def _calculate_constraint_complexity(self, constraint: Constraint) -> float:
        """Calculate complexity factor for constraint (easier constraints get higher scores)"""
        # Simplified complexity based on constraint type
        complexity_factors = {
            ConstraintType.BUDGET: 0.9,      # Relatively easy to satisfy
            ConstraintType.LIQUIDITY: 0.8,   # Moderate complexity
            ConstraintType.RISK: 0.7,        # More complex to optimize
            ConstraintType.TAX: 0.6,         # Complex tax implications
            ConstraintType.REGULATORY: 0.5,  # Most complex compliance
        }
        return complexity_factors.get(constraint.constraint_type, 0.7)
    
    def _calculate_expected_return(self, action: Dict[str, Any], time_horizon: int) -> float:
        """Calculate expected return for an action"""
        base_return = action.get("expected_return", 0.08)
        risk_level = action.get("risk_level", "medium")
        
        # Adjust for risk level
        risk_adjustments = {
            "very_low": -0.02,
            "low": -0.01,
            "medium": 0.0,
            "high": 0.02,
            "very_high": 0.04
        }
        
        adjusted_return = base_return + risk_adjustments.get(risk_level, 0.0)
        
        # Adjust for time horizon (longer horizon allows for more risk)
        if time_horizon > 120:  # > 10 years
            adjusted_return += 0.01
        elif time_horizon < 24:  # < 2 years
            adjusted_return -= 0.01
        
        return adjusted_return
    
    def _calculate_action_volatility(self, action: Dict[str, Any]) -> float:
        """Calculate volatility for an action"""
        risk_level = action.get("risk_level", "medium")
        
        volatilities = {
            "very_low": 0.02,
            "low": 0.05,
            "medium": 0.12,
            "high": 0.20,
            "very_high": 0.35
        }
        
        return volatilities.get(risk_level, 0.12)
    
    def _calculate_downside_risk(self, action: Dict[str, Any]) -> float:
        """Calculate downside risk (semi-deviation) for an action"""
        volatility = self._calculate_action_volatility(action)
        # Downside risk is typically 60-80% of total volatility
        return volatility * 0.7
    
    def _get_market_condition_adjustment(self) -> float:
        """Get market condition adjustment factor"""
        adjustments = {
            MarketCondition.BULL: 1.1,
            MarketCondition.STABLE: 1.0,
            MarketCondition.VOLATILE: 0.9,
            MarketCondition.BEAR: 0.8,
            MarketCondition.RECESSION: 0.7
        }
        return adjustments.get(self.market_condition, 1.0)
    
    def _sigmoid_normalize(self, value: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
        """Normalize value using sigmoid function"""
        return 1.0 / (1.0 + math.exp(-steepness * (value - midpoint)))
    
    def _calculate_current_tax_efficiency(self, financial_state: Dict[str, Any], tax_context: Dict[str, Any]) -> float:
        """Calculate current tax efficiency of the portfolio"""
        tax_advantaged = financial_state.get("tax_advantaged_accounts", 0)
        total_investments = financial_state.get("investments", 0) + tax_advantaged
        
        if total_investments == 0:
            return 0.5
        
        return tax_advantaged / total_investments
    
    def _calculate_action_tax_impact(self, action: Dict[str, Any], tax_context: Dict[str, Any]) -> float:
        """Calculate tax impact of an action (-1 to 1, negative is bad)"""
        action_type = action.get("type", "")
        
        if "401k" in action_type or "ira" in action_type:
            return 0.5  # Tax-advantaged, positive impact
        elif "taxable" in action_type:
            return -0.2  # Taxable investment, slight negative
        elif "tax_loss" in action_type:
            return 0.3  # Tax loss harvesting, positive
        
        return 0.0  # Neutral impact
    
    def _calculate_tax_account_utilization_score(
        self, 
        action: Dict[str, Any], 
        available_space: float, 
        marginal_rate: float
    ) -> float:
        """Calculate score for tax-advantaged account utilization"""
        if "401k" in action.get("type", "") or "ira" in action.get("type", ""):
            amount = action.get("amount", 0)
            utilization = min(amount / max(available_space, 0.001), 1.0)
            return utilization * marginal_rate  # Higher score for higher tax brackets
        
        return 0.5  # Neutral for non-tax-advantaged actions
    
    def _calculate_tax_loss_harvesting_score(
        self, 
        financial_state: Dict[str, Any], 
        action: Dict[str, Any], 
        tax_context: Dict[str, Any]
    ) -> float:
        """Calculate tax loss harvesting opportunity score"""
        # Simplified: assume some positions have losses that can be harvested
        unrealized_losses = financial_state.get("unrealized_losses", 0)
        
        if unrealized_losses > 0 and "sell" in action.get("type", ""):
            return min(unrealized_losses / 10000, 1.0)  # Normalize by $10k
        
        return 0.5  # Neutral score
    
    def _calculate_asset_location_score(
        self, 
        financial_state: Dict[str, Any], 
        action: Dict[str, Any], 
        tax_context: Dict[str, Any]
    ) -> float:
        """Calculate asset location optimization score"""
        # Simplified: bonds in tax-advantaged, stocks in taxable is optimal
        action_type = action.get("type", "")
        
        if "bond" in action_type and "401k" in action_type:
            return 1.0  # Optimal: bonds in tax-advantaged
        elif "stock" in action_type and "taxable" in action_type:
            return 0.8  # Good: stocks in taxable for tax efficiency
        
        return 0.6  # Neutral score
    
    def _calculate_liquid_assets(self, financial_state: Dict[str, Any]) -> float:
        """Calculate total liquid assets"""
        return (
            financial_state.get("cash", 0) +
            financial_state.get("savings", 0) +
            financial_state.get("money_market", 0) +
            financial_state.get("short_term_bonds", 0) * 0.9  # Slightly less liquid
        )
    
    def _calculate_action_liquidity_impact(self, action: Dict[str, Any]) -> float:
        """Calculate impact of action on liquidity"""
        action_type = action.get("type", "")
        amount = action.get("amount", 0)
        
        if action_type in ["cash_reserve", "savings", "emergency_fund"]:
            return amount  # Increases liquidity
        elif action_type in ["investment", "real_estate", "illiquid"]:
            return -amount  # Decreases liquidity
        
        return 0.0  # Neutral impact
    
    def _extract_portfolio_allocation(self, financial_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract current portfolio allocation"""
        total_assets = financial_state.get("total_assets", 1)
        
        if total_assets == 0:
            return {}
        
        return {
            "cash": financial_state.get("cash", 0) / total_assets,
            "stocks": financial_state.get("stocks", 0) / total_assets,
            "bonds": financial_state.get("bonds", 0) / total_assets,
            "real_estate": financial_state.get("real_estate", 0) / total_assets,
            "commodities": financial_state.get("commodities", 0) / total_assets
        }
    
    def _project_allocation_after_action(
        self, 
        current_allocation: Dict[str, float], 
        action: Dict[str, Any]
    ) -> Dict[str, float]:
        """Project portfolio allocation after action"""
        projected = current_allocation.copy()
        action_type = action.get("type", "")
        
        # Simplified projection based on action type
        if "stock" in action_type:
            projected["stocks"] = projected.get("stocks", 0) + 0.1
            projected["cash"] = max(0, projected.get("cash", 0) - 0.1)
        elif "bond" in action_type:
            projected["bonds"] = projected.get("bonds", 0) + 0.1
            projected["cash"] = max(0, projected.get("cash", 0) - 0.1)
        
        # Normalize to ensure sum = 1.0
        total = sum(projected.values())
        if total > 0:
            projected = {k: v / total for k, v in projected.items()}
        
        return projected
    
    def _calculate_target_deviation_score(
        self, 
        current_allocation: Dict[str, float], 
        target_allocation: Dict[str, float]
    ) -> float:
        """Calculate score based on deviation from target allocation"""
        total_deviation = 0.0
        
        for asset_class in target_allocation:
            current_weight = current_allocation.get(asset_class, 0)
            target_weight = target_allocation[asset_class]
            deviation = abs(current_weight - target_weight)
            total_deviation += deviation
        
        # Convert deviation to score (lower deviation = higher score)
        max_possible_deviation = 2.0  # Maximum possible sum of absolute deviations
        score = 1.0 - (total_deviation / max_possible_deviation)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_correlation_diversification_score(
        self, 
        allocation: Dict[str, float], 
        action: Dict[str, Any]
    ) -> float:
        """Calculate diversification score based on asset correlations"""
        # Simplified correlation matrix (in practice, would use historical data)
        correlations = {
            ("stocks", "bonds"): -0.1,
            ("stocks", "real_estate"): 0.3,
            ("stocks", "commodities"): 0.2,
            ("bonds", "real_estate"): 0.1,
            ("bonds", "commodities"): -0.2,
            ("real_estate", "commodities"): 0.1
        }
        
        # Calculate weighted average correlation
        total_correlation = 0.0
        total_weight = 0.0
        
        asset_classes = list(allocation.keys())
        for i, asset1 in enumerate(asset_classes):
            for asset2 in asset_classes[i+1:]:
                weight1 = allocation[asset1]
                weight2 = allocation[asset2]
                correlation = correlations.get((asset1, asset2), correlations.get((asset2, asset1), 0.0))
                
                pair_weight = weight1 * weight2
                total_correlation += correlation * pair_weight
                total_weight += pair_weight
        
        avg_correlation = total_correlation / max(total_weight, 0.001)
        
        # Lower correlation = better diversification
        diversification_score = 1.0 - abs(avg_correlation)
        
        return max(0.0, min(1.0, diversification_score))