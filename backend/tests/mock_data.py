"""
Mock Data Generator for Testing

Provides realistic test data for financial planning scenarios,
market conditions, and user profiles for comprehensive testing.

Requirements: 11.2, 11.3
"""

import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from uuid import uuid4

from data_models.schemas import (
    EnhancedPlanRequest, FinancialState, Constraint, ConstraintType, ConstraintPriority,
    MarketData, TriggerEvent, MarketEventType, SeverityLevel,
    RiskProfile, RiskLevel, TaxContext, RegulatoryRequirement,
    PlanStep, SearchPath, DecisionPoint
)


class MockDataGenerator:
    """
    Generates realistic mock data for testing VP-MAS components.
    
    Provides various financial scenarios, market conditions, and user profiles
    to enable comprehensive testing of all system components.
    """
    
    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
        
        self.user_profiles = self._create_user_profiles()
        self.market_scenarios = self._create_market_scenarios()
        self.financial_goals = self._create_financial_goals()
    
    def generate_enhanced_plan_request(self, scenario: str = "balanced") -> EnhancedPlanRequest:
        """Generate realistic planning request"""
        profile = random.choice(self.user_profiles)
        goal = random.choice(self.financial_goals)
        
        return EnhancedPlanRequest(
            user_id=f"test_user_{uuid4().hex[:8]}",
            user_goal=goal["description"],
            current_state=self._generate_financial_state(profile).dict(),
            constraints=self._generate_constraints(profile),
            risk_profile=self._generate_risk_profile(profile).dict(),
            tax_considerations=self._generate_tax_context(profile).dict(),
            time_horizon=goal["time_horizon"],
            optimization_preferences={
                "tax_efficiency": profile["tax_priority"],
                "risk_tolerance": profile["risk_tolerance"],
                "liquidity_preference": profile["liquidity_needs"]
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4())
        )
    
    def generate_financial_state(self, profile_type: str = "middle_class") -> FinancialState:
        """Generate realistic financial state"""
        profiles = {
            "young_professional": {
                "assets_range": (50000, 150000),
                "income_range": (60000, 120000),
                "debt_range": (20000, 80000)
            },
            "middle_class": {
                "assets_range": (200000, 800000),
                "income_range": (80000, 200000),
                "debt_range": (50000, 300000)
            },
            "high_net_worth": {
                "assets_range": (1000000, 5000000),
                "income_range": (300000, 1000000),
                "debt_range": (100000, 500000)
            },
            "retiree": {
                "assets_range": (500000, 2000000),
                "income_range": (40000, 100000),
                "debt_range": (0, 100000)
            }
        }
        
        profile = profiles.get(profile_type, profiles["middle_class"])
        
        total_assets = Decimal(random.randint(*profile["assets_range"]))
        total_liabilities = Decimal(random.randint(*profile["debt_range"]))
        monthly_income = Decimal(random.randint(*profile["income_range"]) / 12)
        monthly_expenses = monthly_income * Decimal(random.uniform(0.6, 0.9))
        
        return FinancialState(
            user_id=f"test_user_{uuid4().hex[:8]}",
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            risk_tolerance=random.choice(["conservative", "moderate", "aggressive"]),
            tax_filing_status=random.choice(["single", "married_joint", "married_separate"]),
            estimated_tax_rate=random.uniform(0.15, 0.35)
        )
    
    def generate_market_data(self, scenario: str = "normal") -> MarketData:
        """Generate market data for different scenarios"""
        scenario_data = self.market_scenarios.get(scenario, self.market_scenarios["normal"])
        
        return MarketData(
            source=random.choice(["barchart", "alpha_vantage", "massive_api"]),
            market_volatility=scenario_data["volatility"] + random.uniform(-0.05, 0.05),
            interest_rates={
                "federal_funds": scenario_data["fed_rate"] + random.uniform(-0.25, 0.25),
                "10_year_treasury": scenario_data["treasury_10y"] + random.uniform(-0.3, 0.3),
                "30_year_mortgage": scenario_data["mortgage_30y"] + random.uniform(-0.5, 0.5)
            },
            sector_trends={
                "technology": scenario_data["tech_trend"] + random.uniform(-0.1, 0.1),
                "healthcare": scenario_data["health_trend"] + random.uniform(-0.05, 0.05),
                "financial": scenario_data["finance_trend"] + random.uniform(-0.08, 0.08),
                "energy": scenario_data["energy_trend"] + random.uniform(-0.15, 0.15),
                "real_estate": random.uniform(-0.1, 0.15),
                "consumer_goods": random.uniform(-0.05, 0.08)
            },
            economic_sentiment=scenario_data["sentiment"] + random.uniform(-0.2, 0.2),
            collection_method="api_aggregation",
            refresh_frequency=random.randint(60, 600)
        )
    
    def generate_trigger_event(self, severity: str = None) -> TriggerEvent:
        """Generate realistic trigger events"""
        if not severity:
            severity = random.choice(["low", "medium", "high", "critical"])
        
        trigger_types = {
            "low": {
                "events": [MarketEventType.SECTOR_ROTATION, MarketEventType.INTEREST_RATE_CHANGE],
                "descriptions": ["Minor sector rotation detected", "Small interest rate adjustment"]
            },
            "medium": {
                "events": [MarketEventType.VOLATILITY_SPIKE],
                "descriptions": ["Market volatility increased", "Economic uncertainty rising"]
            },
            "high": {
                "events": [MarketEventType.MARKET_CRASH, MarketEventType.VOLATILITY_SPIKE],
                "descriptions": ["Significant market decline", "High volatility spike detected"]
            },
            "critical": {
                "events": [MarketEventType.MARKET_CRASH, MarketEventType.REGULATORY_CHANGE],
                "descriptions": ["Major market crash", "Critical regulatory change"]
            }
        }
        
        trigger_data = trigger_types[severity]
        event_type = random.choice(trigger_data["events"])
        description = random.choice(trigger_data["descriptions"])
        
        return TriggerEvent(
            trigger_type="market_event",
            event_type=event_type,
            severity=SeverityLevel(severity.upper()),
            description=description,
            source_data={
                "volatility": random.uniform(0.1, 0.8),
                "price_change": random.uniform(-0.3, 0.3),
                "volume_spike": random.uniform(1.0, 5.0)
            },
            impact_score=random.uniform(0.3, 1.0),
            confidence_score=random.uniform(0.7, 0.95),
            detector_agent_id=f"ira_{uuid4().hex[:8]}",
            correlation_id=str(uuid4())
        )
    
    def generate_plan_steps(self, goal_type: str = "retirement", num_steps: int = None) -> List[PlanStep]:
        """Generate realistic plan steps for different goals"""
        if not num_steps:
            num_steps = random.randint(3, 8)
        
        step_templates = {
            "retirement": [
                ("emergency_fund", "Build emergency fund", 30000, "low"),
                ("401k_contribution", "Maximize 401(k) contributions", 22500, "low"),
                ("ira_contribution", "Contribute to IRA", 6500, "low"),
                ("index_fund_investment", "Invest in index funds", 50000, "medium"),
                ("bond_allocation", "Allocate to bonds", 25000, "low"),
                ("rebalancing", "Quarterly rebalancing", 0, "low")
            ],
            "house_purchase": [
                ("down_payment_savings", "Save for down payment", 80000, "low"),
                ("credit_improvement", "Improve credit score", 0, "low"),
                ("mortgage_preapproval", "Get mortgage pre-approval", 0, "low"),
                ("closing_costs", "Save for closing costs", 15000, "low"),
                ("house_hunting", "Begin house hunting", 0, "low")
            ],
            "debt_payoff": [
                ("debt_consolidation", "Consolidate high-interest debt", 25000, "medium"),
                ("payment_plan", "Create aggressive payment plan", 0, "low"),
                ("budget_optimization", "Optimize monthly budget", 0, "low"),
                ("emergency_fund", "Build small emergency fund", 5000, "low"),
                ("credit_monitoring", "Monitor credit improvement", 0, "low")
            ],
            "investment": [
                ("portfolio_analysis", "Analyze current portfolio", 0, "low"),
                ("diversification", "Diversify investments", 75000, "medium"),
                ("tax_optimization", "Optimize for taxes", 0, "medium"),
                ("risk_assessment", "Assess risk tolerance", 0, "low"),
                ("performance_monitoring", "Monitor performance", 0, "low")
            ]
        }
        
        templates = step_templates.get(goal_type, step_templates["retirement"])
        selected_templates = random.sample(templates, min(num_steps, len(templates)))
        
        steps = []
        for i, (action_type, description, base_amount, risk_level) in enumerate(selected_templates):
            amount_variation = random.uniform(0.8, 1.2)
            amount = Decimal(int(base_amount * amount_variation))
            
            step = PlanStep(
                sequence_number=i + 1,
                action_type=action_type,
                description=description,
                amount=amount,
                target_date=datetime.utcnow() + timedelta(days=random.randint(30, 365)),
                rationale=f"Step {i+1} in {goal_type} strategy",
                confidence_score=random.uniform(0.7, 0.95),
                risk_level=risk_level
            )
            steps.append(step)
        
        return steps
    
    def generate_search_paths(self, num_paths: int = 5) -> List[SearchPath]:
        """Generate realistic search paths for planning algorithms"""
        strategies = ["conservative", "balanced", "aggressive", "tax_optimized", "growth_focused"]
        paths = []
        
        for i in range(num_paths):
            strategy = strategies[i % len(strategies)]
            
            # Generate realistic scores based on strategy
            risk_scores = {
                "conservative": random.uniform(0.1, 0.3),
                "balanced": random.uniform(0.3, 0.5),
                "aggressive": random.uniform(0.6, 0.8),
                "tax_optimized": random.uniform(0.2, 0.4),
                "growth_focused": random.uniform(0.5, 0.7)
            }
            
            path = SearchPath(
                search_session_id=str(uuid4()),
                path_type="explored",
                sequence_steps=[
                    {"step": j, "action": f"action_{j}", "strategy": strategy}
                    for j in range(random.randint(3, 6))
                ],
                total_cost=random.uniform(1000, 100000),
                expected_value=random.uniform(50000, 500000),
                risk_score=risk_scores[strategy],
                feasibility_score=random.uniform(0.6, 0.95),
                combined_score=random.uniform(0.5, 0.9),
                constraint_satisfaction_score=random.uniform(0.7, 1.0),
                path_status="explored" if random.random() > 0.3 else "pruned",
                exploration_time=random.uniform(0.1, 2.0),
                created_by_agent=f"planning_agent_{uuid4().hex[:8]}"
            )
            paths.append(path)
        
        return paths
    
    def generate_constraints(self, profile: Dict) -> List[Dict]:
        """Generate realistic financial constraints"""
        constraints = []
        
        # Emergency fund constraint
        constraints.append({
            "constraint_id": str(uuid4()),
            "name": "Emergency Fund",
            "constraint_type": ConstraintType.LIQUIDITY.value,
            "priority": ConstraintPriority.MANDATORY.value,
            "description": "Maintain emergency fund of 3-6 months expenses",
            "validation_rule": "emergency_fund >= monthly_expenses * 3",
            "threshold_value": 3,
            "comparison_operator": ">=",
            "created_by": "system"
        })
        
        # Debt-to-income constraint
        constraints.append({
            "constraint_id": str(uuid4()),
            "name": "Debt-to-Income Ratio",
            "constraint_type": ConstraintType.BUDGET.value,
            "priority": ConstraintPriority.HIGH.value,
            "description": "Total debt payments should not exceed 36% of income",
            "validation_rule": "debt_payments / gross_income <= 0.36",
            "threshold_value": 0.36,
            "comparison_operator": "<=",
            "created_by": "system"
        })
        
        # Risk tolerance constraint
        if profile.get("risk_tolerance") == "conservative":
            constraints.append({
                "constraint_id": str(uuid4()),
                "name": "Conservative Risk Limit",
                "constraint_type": ConstraintType.RISK.value,
                "priority": ConstraintPriority.HIGH.value,
                "description": "Portfolio risk should remain conservative",
                "validation_rule": "portfolio_risk <= 0.3",
                "threshold_value": 0.3,
                "comparison_operator": "<=",
                "created_by": "user_preference"
            })
        
        # Tax optimization constraint
        if profile.get("tax_priority", 0.5) > 0.7:
            constraints.append({
                "constraint_id": str(uuid4()),
                "name": "Tax Efficiency",
                "constraint_type": ConstraintType.TAX.value,
                "priority": ConstraintPriority.MEDIUM.value,
                "description": "Prioritize tax-advantaged investments",
                "validation_rule": "tax_advantaged_ratio >= 0.6",
                "threshold_value": 0.6,
                "comparison_operator": ">=",
                "created_by": "user_preference"
            })
        
        return constraints
    
    def _generate_financial_state(self, profile: Dict) -> FinancialState:
        """Generate financial state based on user profile"""
        return FinancialState(
            user_id=f"test_user_{uuid4().hex[:8]}",
            total_assets=Decimal(profile["assets"]),
            total_liabilities=Decimal(profile["liabilities"]),
            monthly_income=Decimal(profile["monthly_income"]),
            monthly_expenses=Decimal(profile["monthly_expenses"]),
            risk_tolerance=profile["risk_tolerance"],
            tax_filing_status=profile["tax_status"],
            estimated_tax_rate=profile["tax_rate"]
        )
    
    def _generate_risk_profile(self, profile: Dict) -> RiskProfile:
        """Generate risk profile based on user profile"""
        return RiskProfile(
            user_id=f"test_user_{uuid4().hex[:8]}",
            overall_risk_tolerance=RiskLevel(profile["risk_tolerance"].upper()),
            risk_capacity=profile.get("risk_capacity", 0.6),
            risk_perception=profile.get("risk_perception", 0.5),
            risk_composure=profile.get("risk_composure", 0.7),
            investment_horizon=profile.get("investment_horizon", 120),
            liquidity_needs={"emergency": profile.get("liquidity_needs", 0.1)},
            volatility_comfort=profile.get("volatility_comfort", 0.5),
            loss_tolerance=profile.get("loss_tolerance", 0.2),
            investment_experience=profile.get("experience", "intermediate"),
            financial_knowledge=profile.get("knowledge", 0.6),
            decision_making_style=profile.get("decision_style", "analytical"),
            primary_goals=profile.get("goals", ["retirement"]),
            goal_priorities={"retirement": 1},
            assessment_method="questionnaire",
            next_review_date=datetime.utcnow() + timedelta(days=365)
        )
    
    def _generate_tax_context(self, profile: Dict) -> TaxContext:
        """Generate tax context based on user profile"""
        return TaxContext(
            user_id=f"test_user_{uuid4().hex[:8]}",
            tax_year=datetime.utcnow().year,
            filing_status=profile["tax_status"],
            number_of_dependents=profile.get("dependents", 0),
            state_of_residence=profile.get("state", "CA"),
            estimated_agi=Decimal(profile["monthly_income"] * 12),
            marginal_tax_rate=profile["tax_rate"],
            effective_tax_rate=profile["tax_rate"] * 0.8,
            state_tax_rate=profile.get("state_tax_rate", 0.08),
            standard_deduction=Decimal(13850 if profile["tax_status"] == "single" else 27700),
            estimated_tax_liability=Decimal(profile["monthly_income"] * 12 * profile["tax_rate"])
        )
    
    def _create_user_profiles(self) -> List[Dict]:
        """Create diverse user profiles for testing"""
        return [
            {
                "name": "Young Professional",
                "assets": 75000,
                "liabilities": 45000,
                "monthly_income": 8000,
                "monthly_expenses": 6000,
                "risk_tolerance": "moderate",
                "tax_status": "single",
                "tax_rate": 0.22,
                "tax_priority": 0.6,
                "liquidity_needs": 0.15,
                "investment_horizon": 360,
                "goals": ["retirement", "house_purchase"]
            },
            {
                "name": "Middle-aged Family",
                "assets": 450000,
                "liabilities": 180000,
                "monthly_income": 15000,
                "monthly_expenses": 12000,
                "risk_tolerance": "conservative",
                "tax_status": "married_joint",
                "tax_rate": 0.24,
                "tax_priority": 0.8,
                "liquidity_needs": 0.2,
                "investment_horizon": 180,
                "dependents": 2,
                "goals": ["retirement", "education_funding"]
            },
            {
                "name": "High Net Worth",
                "assets": 2500000,
                "liabilities": 300000,
                "monthly_income": 50000,
                "monthly_expenses": 25000,
                "risk_tolerance": "aggressive",
                "tax_status": "married_joint",
                "tax_rate": 0.35,
                "tax_priority": 0.9,
                "liquidity_needs": 0.1,
                "investment_horizon": 240,
                "goals": ["wealth_preservation", "tax_optimization"]
            },
            {
                "name": "Pre-retiree",
                "assets": 800000,
                "liabilities": 50000,
                "monthly_income": 12000,
                "monthly_expenses": 8000,
                "risk_tolerance": "conservative",
                "tax_status": "married_joint",
                "tax_rate": 0.22,
                "tax_priority": 0.7,
                "liquidity_needs": 0.25,
                "investment_horizon": 60,
                "goals": ["retirement", "healthcare_planning"]
            }
        ]
    
    def _create_market_scenarios(self) -> Dict[str, Dict]:
        """Create different market scenarios for testing"""
        return {
            "normal": {
                "volatility": 0.15,
                "fed_rate": 5.25,
                "treasury_10y": 4.5,
                "mortgage_30y": 7.2,
                "tech_trend": 0.08,
                "health_trend": 0.05,
                "finance_trend": 0.03,
                "energy_trend": -0.02,
                "sentiment": 0.1
            },
            "volatile": {
                "volatility": 0.35,
                "fed_rate": 5.25,
                "treasury_10y": 4.8,
                "mortgage_30y": 7.5,
                "tech_trend": -0.15,
                "health_trend": 0.02,
                "finance_trend": -0.08,
                "energy_trend": 0.12,
                "sentiment": -0.2
            },
            "bull_market": {
                "volatility": 0.12,
                "fed_rate": 4.0,
                "treasury_10y": 3.8,
                "mortgage_30y": 6.2,
                "tech_trend": 0.35,
                "health_trend": 0.22,
                "finance_trend": 0.28,
                "energy_trend": 0.15,
                "sentiment": 0.7
            },
            "bear_market": {
                "volatility": 0.45,
                "fed_rate": 5.5,
                "treasury_10y": 5.2,
                "mortgage_30y": 8.0,
                "tech_trend": -0.25,
                "health_trend": -0.10,
                "finance_trend": -0.30,
                "energy_trend": -0.15,
                "sentiment": -0.6
            }
        }
    
    def _create_financial_goals(self) -> List[Dict]:
        """Create realistic financial goals for testing"""
        return [
            {
                "description": "Save for retirement with $2M target",
                "time_horizon": 360,
                "goal_type": "retirement"
            },
            {
                "description": "Save $100K for house down payment",
                "time_horizon": 60,
                "goal_type": "house_purchase"
            },
            {
                "description": "Pay off $50K in credit card debt",
                "time_horizon": 36,
                "goal_type": "debt_payoff"
            },
            {
                "description": "Build emergency fund of 6 months expenses",
                "time_horizon": 12,
                "goal_type": "emergency_fund"
            },
            {
                "description": "Save $200K for children's college education",
                "time_horizon": 180,
                "goal_type": "education_funding"
            },
            {
                "description": "Create diversified investment portfolio",
                "time_horizon": 120,
                "goal_type": "investment"
            }
        ]


# Convenience functions for quick test data generation
def quick_plan_request(scenario: str = "balanced") -> EnhancedPlanRequest:
    """Quick generation of plan request for testing"""
    generator = MockDataGenerator()
    return generator.generate_enhanced_plan_request(scenario)


def quick_market_data(scenario: str = "normal") -> MarketData:
    """Quick generation of market data for testing"""
    generator = MockDataGenerator()
    return generator.generate_market_data(scenario)


def quick_trigger_event(severity: str = "medium") -> TriggerEvent:
    """Quick generation of trigger event for testing"""
    generator = MockDataGenerator()
    return generator.generate_trigger_event(severity)


def quick_financial_state(profile: str = "middle_class") -> FinancialState:
    """Quick generation of financial state for testing"""
    generator = MockDataGenerator()
    return generator.generate_financial_state(profile)