"""
Tax Optimization and Regulatory Compliance Module

Implements comprehensive tax optimization strategies and regulatory compliance checking:
- Tax-efficient investment strategies
- Tax-loss harvesting optimization
- Asset location optimization
- Retirement account optimization
- Regulatory compliance monitoring
- Tax law change adaptation

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
from uuid import uuid4

from data_models.schemas import (
    TaxContext, RegulatoryRequirement, ComplianceStatus, 
    FinancialState, Constraint, ConstraintType
)


class TaxAccountType(str, Enum):
    """Types of tax-advantaged accounts"""
    TRADITIONAL_401K = "traditional_401k"
    ROTH_401K = "roth_401k"
    TRADITIONAL_IRA = "traditional_ira"
    ROTH_IRA = "roth_ira"
    HSA = "hsa"
    TAXABLE = "taxable"
    MUNICIPAL_BONDS = "municipal_bonds"
    I_BONDS = "i_bonds"


class TaxStrategy(str, Enum):
    """Tax optimization strategies"""
    TAX_LOSS_HARVESTING = "tax_loss_harvesting"
    ASSET_LOCATION = "asset_location"
    ROTH_CONVERSION = "roth_conversion"
    TAX_DEFERRAL = "tax_deferral"
    MUNICIPAL_BOND_LADDER = "municipal_bond_ladder"
    CHARITABLE_GIVING = "charitable_giving"
    BACKDOOR_ROTH = "backdoor_roth"
    MEGA_BACKDOOR_ROTH = "mega_backdoor_roth"


@dataclass
class TaxOptimizationRecommendation:
    """Tax optimization recommendation"""
    strategy: TaxStrategy
    description: str
    estimated_tax_savings: Decimal
    implementation_steps: List[str]
    timeline: str
    risk_level: str
    prerequisites: List[str]
    annual_benefit: Decimal
    complexity_score: float


@dataclass
class AssetLocationRecommendation:
    """Asset location optimization recommendation"""
    asset_type: str
    recommended_account: TaxAccountType
    current_account: TaxAccountType
    tax_efficiency_gain: float
    rationale: str


class TaxOptimizer:
    """
    Advanced tax optimization engine for financial planning.
    Provides comprehensive tax-efficient strategies and compliance monitoring.
    """
    
    def __init__(self):
        self.tax_brackets_2024 = self._initialize_tax_brackets()
        self.contribution_limits_2024 = self._initialize_contribution_limits()
        self.tax_strategies = self._initialize_tax_strategies()
        self.asset_tax_efficiency = self._initialize_asset_tax_efficiency()
    
    def optimize_tax_strategy(
        self,
        financial_state: FinancialState,
        tax_context: TaxContext,
        investment_goals: Dict[str, Any],
        time_horizon_years: int
    ) -> Dict[str, Any]:
        """
        Generate comprehensive tax optimization strategy.
        
        Args:
            financial_state: Current financial state
            tax_context: Tax context and situation
            investment_goals: Investment goals and preferences
            time_horizon_years: Investment time horizon
            
        Returns:
            Comprehensive tax optimization recommendations
        """
        recommendations = []
        
        # Analyze current tax efficiency
        current_efficiency = self._analyze_current_tax_efficiency(financial_state, tax_context)
        
        # Generate tax-advantaged account recommendations
        account_recommendations = self._optimize_tax_advantaged_accounts(
            financial_state, tax_context, time_horizon_years
        )
        recommendations.extend(account_recommendations)
        
        # Asset location optimization
        asset_location_recs = self._optimize_asset_location(
            financial_state, tax_context, investment_goals
        )
        recommendations.extend(asset_location_recs)
        
        # Tax-loss harvesting opportunities
        tax_loss_recs = self._identify_tax_loss_harvesting_opportunities(
            financial_state, tax_context
        )
        recommendations.extend(tax_loss_recs)
        
        # Roth conversion analysis
        roth_conversion_recs = self._analyze_roth_conversion_opportunities(
            financial_state, tax_context, time_horizon_years
        )
        recommendations.extend(roth_conversion_recs)
        
        # Advanced strategies for high earners
        if tax_context.estimated_agi > Decimal("200000"):
            advanced_recs = self._generate_advanced_tax_strategies(
                financial_state, tax_context, time_horizon_years
            )
            recommendations.extend(advanced_recs)
        
        # Calculate total potential tax savings
        total_savings = sum(rec.estimated_tax_savings for rec in recommendations)
        annual_savings = sum(rec.annual_benefit for rec in recommendations)
        
        # Prioritize recommendations
        prioritized_recs = self._prioritize_recommendations(recommendations, tax_context)
        
        return {
            "current_tax_efficiency_score": current_efficiency,
            "recommendations": [rec.__dict__ for rec in prioritized_recs],
            "total_estimated_savings": float(total_savings),
            "annual_estimated_savings": float(annual_savings),
            "implementation_timeline": self._create_implementation_timeline(prioritized_recs),
            "tax_projection": self._project_tax_impact(prioritized_recs, tax_context, time_horizon_years),
            "compliance_requirements": self._identify_compliance_requirements(prioritized_recs)
        }
    
    def _analyze_current_tax_efficiency(
        self, 
        financial_state: FinancialState, 
        tax_context: TaxContext
    ) -> float:
        """Analyze current tax efficiency (0-1 scale)"""
        efficiency_factors = []
        
        # Tax-advantaged account utilization
        total_retirement_assets = (
            financial_state.total_assets * Decimal("0.6")  # Assume 60% in retirement accounts
        )
        max_annual_contributions = (
            Decimal(str(self.contribution_limits_2024["401k"])) +
            Decimal(str(self.contribution_limits_2024["ira"]))
        )
        
        if financial_state.monthly_income > 0:
            annual_income = financial_state.monthly_income * 12
            contribution_rate = min(max_annual_contributions / annual_income, 1.0)
            efficiency_factors.append(float(contribution_rate))
        
        # Asset location efficiency
        # Simplified: assume some efficiency based on total assets
        if financial_state.total_assets > 0:
            asset_location_efficiency = min(0.8, float(total_retirement_assets / financial_state.total_assets))
            efficiency_factors.append(asset_location_efficiency)
        
        # Tax bracket optimization
        marginal_rate = tax_context.marginal_tax_rate
        effective_rate = tax_context.effective_tax_rate
        
        if marginal_rate > 0:
            tax_efficiency = 1.0 - (effective_rate / marginal_rate)
            efficiency_factors.append(tax_efficiency)
        
        return sum(efficiency_factors) / len(efficiency_factors) if efficiency_factors else 0.5
    
    def _optimize_tax_advantaged_accounts(
        self,
        financial_state: FinancialState,
        tax_context: TaxContext,
        time_horizon_years: int
    ) -> List[TaxOptimizationRecommendation]:
        """Generate tax-advantaged account optimization recommendations"""
        recommendations = []
        
        annual_income = financial_state.monthly_income * 12
        
        # 401(k) optimization
        if annual_income > 0:
            max_401k = self.contribution_limits_2024["401k"]
            current_401k = float(annual_income * Decimal("0.1"))  # Assume 10% current contribution
            
            if current_401k < max_401k:
                additional_contribution = max_401k - current_401k
                tax_savings = additional_contribution * tax_context.marginal_tax_rate
                
                recommendations.append(TaxOptimizationRecommendation(
                    strategy=TaxStrategy.TAX_DEFERRAL,
                    description=f"Maximize 401(k) contribution to ${max_401k:,.0f}",
                    estimated_tax_savings=Decimal(str(tax_savings)),
                    implementation_steps=[
                        "Contact HR to increase 401(k) contribution",
                        f"Increase contribution by ${additional_contribution:,.0f} annually",
                        "Adjust budget for reduced take-home pay"
                    ],
                    timeline="Immediate - next payroll period",
                    risk_level="Low",
                    prerequisites=["Employer 401(k) plan available"],
                    annual_benefit=Decimal(str(tax_savings)),
                    complexity_score=0.2
                ))
        
        # IRA optimization
        ira_limit = self.contribution_limits_2024["ira"]
        
        # Traditional vs Roth IRA analysis
        if tax_context.marginal_tax_rate > 0.22:  # High tax bracket
            # Recommend Traditional IRA
            recommendations.append(TaxOptimizationRecommendation(
                strategy=TaxStrategy.TAX_DEFERRAL,
                description=f"Contribute ${ira_limit:,.0f} to Traditional IRA",
                estimated_tax_savings=Decimal(str(ira_limit * tax_context.marginal_tax_rate)),
                implementation_steps=[
                    "Open Traditional IRA account",
                    f"Set up automatic monthly contribution of ${ira_limit/12:,.0f}",
                    "Ensure income limits are met"
                ],
                timeline="Within 30 days",
                risk_level="Low",
                prerequisites=["Income below deduction limits"],
                annual_benefit=Decimal(str(ira_limit * tax_context.marginal_tax_rate)),
                complexity_score=0.3
            ))
        else:
            # Recommend Roth IRA
            future_tax_savings = ira_limit * 0.08 * time_horizon_years  # Assume 8% growth
            
            recommendations.append(TaxOptimizationRecommendation(
                strategy=TaxStrategy.ROTH_CONVERSION,
                description=f"Contribute ${ira_limit:,.0f} to Roth IRA",
                estimated_tax_savings=Decimal(str(future_tax_savings)),
                implementation_steps=[
                    "Open Roth IRA account",
                    f"Set up automatic monthly contribution of ${ira_limit/12:,.0f}",
                    "Plan for tax-free withdrawals in retirement"
                ],
                timeline="Within 30 days",
                risk_level="Low",
                prerequisites=["Income below contribution limits"],
                annual_benefit=Decimal(str(future_tax_savings / time_horizon_years)),
                complexity_score=0.3
            ))
        
        # HSA optimization (if applicable)
        if tax_context.filing_status in ["single", "married_filing_jointly"]:
            hsa_limit = self.contribution_limits_2024["hsa_individual"]
            triple_tax_advantage = hsa_limit * (tax_context.marginal_tax_rate + 0.0765)  # Include FICA
            
            recommendations.append(TaxOptimizationRecommendation(
                strategy=TaxStrategy.TAX_DEFERRAL,
                description=f"Maximize HSA contribution to ${hsa_limit:,.0f}",
                estimated_tax_savings=Decimal(str(triple_tax_advantage)),
                implementation_steps=[
                    "Enroll in High Deductible Health Plan (HDHP)",
                    "Open HSA account",
                    f"Contribute maximum ${hsa_limit:,.0f} annually",
                    "Invest HSA funds for long-term growth"
                ],
                timeline="Next open enrollment period",
                risk_level="Low",
                prerequisites=["HDHP enrollment", "HSA eligibility"],
                annual_benefit=Decimal(str(triple_tax_advantage)),
                complexity_score=0.4
            ))
        
        return recommendations
    
    def _optimize_asset_location(
        self,
        financial_state: FinancialState,
        tax_context: TaxContext,
        investment_goals: Dict[str, Any]
    ) -> List[TaxOptimizationRecommendation]:
        """Optimize asset location across account types"""
        recommendations = []
        
        # Asset location hierarchy (most tax-inefficient first)
        location_hierarchy = [
            ("REITs", TaxAccountType.TRADITIONAL_401K, "High dividend yield, tax-inefficient"),
            ("Bonds", TaxAccountType.TRADITIONAL_401K, "Interest taxed as ordinary income"),
            ("International stocks", TaxAccountType.TAXABLE, "Foreign tax credit benefits"),
            ("Growth stocks", TaxAccountType.ROTH_IRA, "Tax-free growth potential"),
            ("Index funds", TaxAccountType.TAXABLE, "Tax-efficient, low turnover"),
            ("Municipal bonds", TaxAccountType.TAXABLE, "Tax-free interest")
        ]
        
        total_tax_savings = 0
        
        for asset_type, recommended_account, rationale in location_hierarchy:
            # Estimate tax savings from optimal location
            annual_savings = float(financial_state.total_assets) * 0.1 * 0.02  # 2% of 10% allocation
            total_tax_savings += annual_savings
        
        if total_tax_savings > 1000:  # Only recommend if meaningful savings
            recommendations.append(TaxOptimizationRecommendation(
                strategy=TaxStrategy.ASSET_LOCATION,
                description="Optimize asset location across account types",
                estimated_tax_savings=Decimal(str(total_tax_savings * 10)),  # 10-year benefit
                implementation_steps=[
                    "Review current asset allocation across accounts",
                    "Move tax-inefficient assets to tax-advantaged accounts",
                    "Place tax-efficient assets in taxable accounts",
                    "Rebalance gradually to avoid tax consequences"
                ],
                timeline="6-12 months (gradual implementation)",
                risk_level="Medium",
                prerequisites=["Multiple account types available"],
                annual_benefit=Decimal(str(total_tax_savings)),
                complexity_score=0.7
            ))
        
        return recommendations
    
    def _identify_tax_loss_harvesting_opportunities(
        self,
        financial_state: FinancialState,
        tax_context: TaxContext
    ) -> List[TaxOptimizationRecommendation]:
        """Identify tax-loss harvesting opportunities"""
        recommendations = []
        
        # Estimate potential losses available for harvesting
        # In practice, would analyze actual portfolio positions
        estimated_losses = float(financial_state.total_assets) * 0.05  # Assume 5% in losses
        
        if estimated_losses > 3000:  # Minimum threshold for meaningful harvesting
            annual_tax_benefit = min(estimated_losses * tax_context.marginal_tax_rate, 3000)
            
            recommendations.append(TaxOptimizationRecommendation(
                strategy=TaxStrategy.TAX_LOSS_HARVESTING,
                description="Implement systematic tax-loss harvesting",
                estimated_tax_savings=Decimal(str(annual_tax_benefit)),
                implementation_steps=[
                    "Review portfolio for positions with unrealized losses",
                    "Sell losing positions to realize losses",
                    "Reinvest in similar (but not identical) assets",
                    "Avoid wash sale rules (30-day period)",
                    "Carry forward excess losses to future years"
                ],
                timeline="Ongoing throughout the year",
                risk_level="Low",
                prerequisites=["Taxable investment accounts with losses"],
                annual_benefit=Decimal(str(annual_tax_benefit)),
                complexity_score=0.6
            ))
        
        return recommendations
    
    def _analyze_roth_conversion_opportunities(
        self,
        financial_state: FinancialState,
        tax_context: TaxContext,
        time_horizon_years: int
    ) -> List[TaxOptimizationRecommendation]:
        """Analyze Roth conversion opportunities"""
        recommendations = []
        
        # Only recommend if in lower tax bracket or expecting higher future rates
        if tax_context.marginal_tax_rate < 0.24 and time_horizon_years > 10:
            
            # Calculate optimal conversion amount
            # Stay within current tax bracket
            current_bracket_top = self._get_tax_bracket_top(tax_context)
            available_bracket_space = current_bracket_top - float(tax_context.estimated_agi)
            
            if available_bracket_space > 10000:  # Minimum conversion threshold
                conversion_amount = min(available_bracket_space, 50000)  # Cap at $50k
                conversion_tax = conversion_amount * tax_context.marginal_tax_rate
                
                # Estimate future tax savings
                future_value = conversion_amount * ((1.08) ** time_horizon_years)  # 8% growth
                future_tax_savings = future_value * 0.24  # Assume 24% future rate
                net_benefit = future_tax_savings - conversion_tax
                
                if net_benefit > 0:
                    recommendations.append(TaxOptimizationRecommendation(
                        strategy=TaxStrategy.ROTH_CONVERSION,
                        description=f"Convert ${conversion_amount:,.0f} from Traditional to Roth IRA",
                        estimated_tax_savings=Decimal(str(net_benefit)),
                        implementation_steps=[
                            f"Convert ${conversion_amount:,.0f} from Traditional IRA to Roth IRA",
                            f"Pay conversion tax of ${conversion_tax:,.0f}",
                            "Ensure funds available to pay tax from non-retirement accounts",
                            "Consider spreading conversion over multiple years"
                        ],
                        timeline="Before year-end",
                        risk_level="Medium",
                        prerequisites=["Traditional IRA balance available", "Funds to pay conversion tax"],
                        annual_benefit=Decimal(str(net_benefit / time_horizon_years)),
                        complexity_score=0.8
                    ))
        
        return recommendations
    
    def _generate_advanced_tax_strategies(
        self,
        financial_state: FinancialState,
        tax_context: TaxContext,
        time_horizon_years: int
    ) -> List[TaxOptimizationRecommendation]:
        """Generate advanced tax strategies for high earners"""
        recommendations = []
        
        annual_income = financial_state.monthly_income * 12
        
        # Backdoor Roth IRA
        if annual_income > Decimal("140000"):  # Above Roth IRA income limits
            recommendations.append(TaxOptimizationRecommendation(
                strategy=TaxStrategy.BACKDOOR_ROTH,
                description="Implement Backdoor Roth IRA strategy",
                estimated_tax_savings=Decimal("12000"),  # Estimated 20-year benefit
                implementation_steps=[
                    "Contribute $6,000 to non-deductible Traditional IRA",
                    "Immediately convert to Roth IRA",
                    "Ensure no other Traditional IRA balances (pro-rata rule)",
                    "File Form 8606 with tax return"
                ],
                timeline="Annual process",
                risk_level="Medium",
                prerequisites=["Income above Roth IRA limits", "No existing Traditional IRA balances"],
                annual_benefit=Decimal("600"),
                complexity_score=0.9
            ))
        
        # Mega Backdoor Roth (if 401k allows)
        if annual_income > Decimal("200000"):
            mega_backdoor_amount = 66000 - 22500  # After-tax 401k contribution limit
            estimated_benefit = mega_backdoor_amount * 0.08 * time_horizon_years * 0.24
            
            recommendations.append(TaxOptimizationRecommendation(
                strategy=TaxStrategy.MEGA_BACKDOOR_ROTH,
                description=f"Implement Mega Backdoor Roth strategy (${mega_backdoor_amount:,.0f})",
                estimated_tax_savings=Decimal(str(estimated_benefit)),
                implementation_steps=[
                    "Verify 401(k) plan allows after-tax contributions",
                    "Verify plan allows in-service withdrawals or conversions",
                    f"Contribute ${mega_backdoor_amount:,.0f} after-tax to 401(k)",
                    "Convert after-tax contributions to Roth 401(k) or Roth IRA"
                ],
                timeline="Ongoing throughout year",
                risk_level="High",
                prerequisites=["401(k) plan supports strategy", "High income and savings capacity"],
                annual_benefit=Decimal(str(estimated_benefit / time_horizon_years)),
                complexity_score=1.0
            ))
        
        # Charitable giving strategies
        if financial_state.total_assets > Decimal("500000"):
            charitable_benefit = 10000 * tax_context.marginal_tax_rate  # $10k donation
            
            recommendations.append(TaxOptimizationRecommendation(
                strategy=TaxStrategy.CHARITABLE_GIVING,
                description="Implement tax-efficient charitable giving strategy",
                estimated_tax_savings=Decimal(str(charitable_benefit)),
                implementation_steps=[
                    "Consider donating appreciated securities instead of cash",
                    "Explore donor-advised funds for flexible giving",
                    "Bundle charitable deductions using donor-advised funds",
                    "Consider qualified charitable distributions from IRA (if over 70.5)"
                ],
                timeline="Annual planning",
                risk_level="Low",
                prerequisites=["Charitable giving goals", "Appreciated securities"],
                annual_benefit=Decimal(str(charitable_benefit)),
                complexity_score=0.7
            ))
        
        return recommendations
    
    def _prioritize_recommendations(
        self,
        recommendations: List[TaxOptimizationRecommendation],
        tax_context: TaxContext
    ) -> List[TaxOptimizationRecommendation]:
        """Prioritize recommendations by impact and feasibility"""
        
        def priority_score(rec: TaxOptimizationRecommendation) -> float:
            # Score based on annual benefit, complexity, and risk
            benefit_score = float(rec.annual_benefit) / 1000  # Normalize to thousands
            complexity_penalty = rec.complexity_score * 0.5
            risk_penalty = {"Low": 0, "Medium": 0.2, "High": 0.5}[rec.risk_level]
            
            return benefit_score - complexity_penalty - risk_penalty
        
        return sorted(recommendations, key=priority_score, reverse=True)
    
    def _create_implementation_timeline(
        self, 
        recommendations: List[TaxOptimizationRecommendation]
    ) -> Dict[str, List[str]]:
        """Create implementation timeline for recommendations"""
        timeline = {
            "immediate": [],
            "within_30_days": [],
            "within_6_months": [],
            "annual_planning": [],
            "ongoing": []
        }
        
        for rec in recommendations:
            if "immediate" in rec.timeline.lower():
                timeline["immediate"].append(rec.description)
            elif "30 days" in rec.timeline.lower():
                timeline["within_30_days"].append(rec.description)
            elif "6" in rec.timeline.lower() or "month" in rec.timeline.lower():
                timeline["within_6_months"].append(rec.description)
            elif "annual" in rec.timeline.lower():
                timeline["annual_planning"].append(rec.description)
            else:
                timeline["ongoing"].append(rec.description)
        
        return timeline
    
    def _project_tax_impact(
        self,
        recommendations: List[TaxOptimizationRecommendation],
        tax_context: TaxContext,
        time_horizon_years: int
    ) -> Dict[str, Any]:
        """Project tax impact over time horizon"""
        annual_savings = sum(rec.annual_benefit for rec in recommendations)
        total_savings = sum(rec.estimated_tax_savings for rec in recommendations)
        
        # Project cumulative savings
        cumulative_savings = []
        for year in range(1, min(time_horizon_years + 1, 21)):  # Cap at 20 years
            year_savings = float(annual_savings) * year
            cumulative_savings.append({
                "year": year,
                "cumulative_savings": year_savings,
                "effective_tax_rate_reduction": year_savings / (float(tax_context.estimated_agi) * year)
            })
        
        return {
            "annual_tax_savings": float(annual_savings),
            "total_estimated_savings": float(total_savings),
            "cumulative_projections": cumulative_savings,
            "break_even_analysis": self._calculate_break_even_periods(recommendations),
            "tax_rate_sensitivity": self._analyze_tax_rate_sensitivity(recommendations, tax_context)
        }
    
    def _identify_compliance_requirements(
        self, 
        recommendations: List[TaxOptimizationRecommendation]
    ) -> List[Dict[str, Any]]:
        """Identify compliance requirements for recommendations"""
        compliance_requirements = []
        
        strategy_compliance = {
            TaxStrategy.BACKDOOR_ROTH: {
                "forms": ["Form 8606"],
                "deadlines": ["Tax filing deadline"],
                "documentation": ["IRA contribution records", "Conversion records"],
                "ongoing_monitoring": ["Pro-rata rule compliance"]
            },
            TaxStrategy.TAX_LOSS_HARVESTING: {
                "forms": ["Schedule D"],
                "deadlines": ["December 31 for current year losses"],
                "documentation": ["Trade confirmations", "Wash sale tracking"],
                "ongoing_monitoring": ["30-day wash sale periods"]
            },
            TaxStrategy.ROTH_CONVERSION: {
                "forms": ["Form 8606"],
                "deadlines": ["December 31 for conversions"],
                "documentation": ["Conversion records", "Tax payment records"],
                "ongoing_monitoring": ["5-year rule for withdrawals"]
            }
        }
        
        for rec in recommendations:
            if rec.strategy in strategy_compliance:
                compliance_requirements.append({
                    "strategy": rec.strategy.value,
                    "requirements": strategy_compliance[rec.strategy]
                })
        
        return compliance_requirements
    
    def _calculate_break_even_periods(
        self, 
        recommendations: List[TaxOptimizationRecommendation]
    ) -> Dict[str, int]:
        """Calculate break-even periods for strategies with upfront costs"""
        break_even_periods = {}
        
        for rec in recommendations:
            if rec.strategy == TaxStrategy.ROTH_CONVERSION:
                # Estimate break-even for Roth conversions
                upfront_cost = float(rec.estimated_tax_savings) * 0.3  # Assume 30% is upfront tax
                annual_benefit = float(rec.annual_benefit)
                
                if annual_benefit > 0:
                    break_even_years = int(upfront_cost / annual_benefit)
                    break_even_periods[rec.strategy.value] = break_even_years
        
        return break_even_periods
    
    def _analyze_tax_rate_sensitivity(
        self,
        recommendations: List[TaxOptimizationRecommendation],
        tax_context: TaxContext
    ) -> Dict[str, Any]:
        """Analyze sensitivity to tax rate changes"""
        current_rate = tax_context.marginal_tax_rate
        
        # Test different tax rate scenarios
        rate_scenarios = [current_rate * 0.8, current_rate, current_rate * 1.2]
        scenario_results = {}
        
        for rate in rate_scenarios:
            scenario_savings = 0
            for rec in recommendations:
                if rec.strategy in [TaxStrategy.TAX_DEFERRAL, TaxStrategy.TAX_LOSS_HARVESTING]:
                    # These strategies benefit from higher tax rates
                    adjusted_benefit = float(rec.annual_benefit) * (rate / current_rate)
                    scenario_savings += adjusted_benefit
                else:
                    scenario_savings += float(rec.annual_benefit)
            
            scenario_results[f"{rate:.1%}"] = scenario_savings
        
        return {
            "rate_scenarios": scenario_results,
            "sensitivity_analysis": "Higher tax rates increase value of deferral strategies",
            "optimal_rate_range": f"{current_rate * 0.9:.1%} - {current_rate * 1.1:.1%}"
        }
    
    def _get_tax_bracket_top(self, tax_context: TaxContext) -> float:
        """Get the top of current tax bracket"""
        brackets = self.tax_brackets_2024[tax_context.filing_status]
        current_agi = float(tax_context.estimated_agi)
        
        for bracket_top, rate in brackets:
            if tax_context.marginal_tax_rate <= rate:
                return bracket_top
        
        return float('inf')  # Highest bracket
    
    def _initialize_tax_brackets(self) -> Dict[str, List[Tuple[float, float]]]:
        """Initialize 2024 tax brackets"""
        return {
            "single": [
                (11000, 0.10),
                (44725, 0.12),
                (95375, 0.22),
                (182050, 0.24),
                (231250, 0.32),
                (578125, 0.35),
                (float('inf'), 0.37)
            ],
            "married_filing_jointly": [
                (22000, 0.10),
                (89450, 0.12),
                (190750, 0.22),
                (364200, 0.24),
                (462500, 0.32),
                (693750, 0.35),
                (float('inf'), 0.37)
            ]
        }
    
    def _initialize_contribution_limits(self) -> Dict[str, float]:
        """Initialize 2024 contribution limits"""
        return {
            "401k": 22500,
            "401k_catchup": 7500,  # Age 50+
            "ira": 6000,
            "ira_catchup": 1000,   # Age 50+
            "hsa_individual": 3650,
            "hsa_family": 7300,
            "hsa_catchup": 1000    # Age 55+
        }
    
    def _initialize_tax_strategies(self) -> Dict[TaxStrategy, Dict[str, Any]]:
        """Initialize tax strategy metadata"""
        return {
            TaxStrategy.TAX_LOSS_HARVESTING: {
                "complexity": 0.6,
                "annual_limit": 3000,
                "carryforward": True
            },
            TaxStrategy.ASSET_LOCATION: {
                "complexity": 0.7,
                "ongoing_management": True
            },
            TaxStrategy.ROTH_CONVERSION: {
                "complexity": 0.8,
                "timing_sensitive": True
            }
        }
    
    def _initialize_asset_tax_efficiency(self) -> Dict[str, float]:
        """Initialize asset tax efficiency scores (0-1, higher is more tax-efficient)"""
        return {
            "index_funds": 0.9,
            "individual_stocks": 0.7,
            "bonds": 0.3,
            "reits": 0.2,
            "commodities": 0.4,
            "municipal_bonds": 1.0,  # Tax-free
            "international_funds": 0.6
        }