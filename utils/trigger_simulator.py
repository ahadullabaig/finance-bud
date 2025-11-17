"""
Standalone Trigger Simulator for CMVL Testing

Generates realistic market events, life events, and compound scenarios
with proper severity assessment and impact scoring.

Requirements: 2.1, 2.2, 10.2, 11.1, 11.2
"""

from datetime import datetime
from typing import Dict, List, Any
from uuid import uuid4

from data_models.schemas import TriggerEvent, SeverityLevel, MarketEventType


class TriggerSimulator:
    """
    Comprehensive trigger simulation system for testing CMVL workflows.
    
    Generates realistic market events, life events, and compound scenarios
    with proper severity assessment and impact scoring.
    """
    
    def __init__(self):
        self.market_scenarios = self._load_market_scenarios()
        self.life_event_scenarios = self._load_life_event_scenarios()
        self.compound_scenarios = self._load_compound_scenarios()
    
    def generate_market_trigger(self, scenario: str = "volatility_spike") -> TriggerEvent:
        """Generate realistic market trigger event"""
        scenario_data = self.market_scenarios.get(scenario, self.market_scenarios["volatility_spike"])
        
        return TriggerEvent(
            trigger_type="market_event",
            event_type=MarketEventType(scenario_data["event_type"]),
            severity=SeverityLevel(scenario_data["severity"]),
            description=scenario_data["description"],
            source_data=scenario_data["source_data"],
            impact_score=scenario_data["impact_score"],
            confidence_score=scenario_data["confidence_score"],
            detector_agent_id="trigger_simulator",
            correlation_id=str(uuid4())
        )
    
    def generate_life_event_trigger(self, scenario: str = "job_loss") -> TriggerEvent:
        """Generate realistic life event trigger"""
        scenario_data = self.life_event_scenarios.get(scenario, self.life_event_scenarios["job_loss"])
        
        return TriggerEvent(
            trigger_type="life_event",
            event_type=MarketEventType.VOLATILITY_SPIKE,  # Using as placeholder for life events
            severity=SeverityLevel(scenario_data["severity"]),
            description=scenario_data["description"],
            source_data=scenario_data["source_data"],
            impact_score=scenario_data["impact_score"],
            confidence_score=scenario_data["confidence_score"],
            detector_agent_id="trigger_simulator",
            correlation_id=str(uuid4())
        )
    
    def generate_compound_trigger(self, scenarios: List[str]) -> List[TriggerEvent]:
        """Generate multiple concurrent triggers for complex scenarios"""
        triggers = []
        
        for scenario in scenarios:
            if scenario in self.market_scenarios:
                triggers.append(self.generate_market_trigger(scenario))
            elif scenario in self.life_event_scenarios:
                triggers.append(self.generate_life_event_trigger(scenario))
            elif scenario in self.compound_scenarios:
                compound_data = self.compound_scenarios[scenario]
                for sub_scenario in compound_data["components"]:
                    if sub_scenario["type"] == "market":
                        triggers.append(self.generate_market_trigger(sub_scenario["scenario"]))
                    else:
                        triggers.append(self.generate_life_event_trigger(sub_scenario["scenario"]))
        
        return triggers
    
    def _load_market_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined market scenario data"""
        return {
            "volatility_spike": {
                "event_type": "volatility_spike",
                "severity": "high",
                "description": "Market volatility increased to 35% due to geopolitical tensions",
                "source_data": {
                    "volatility_index": 0.35,
                    "trigger_source": "geopolitical_event",
                    "affected_sectors": ["technology", "energy", "financial"]
                },
                "impact_score": 0.7,
                "confidence_score": 0.9
            },
            "market_crash": {
                "event_type": "market_crash",
                "severity": "critical",
                "description": "Major market crash with 20% decline in major indices",
                "source_data": {
                    "sp500_decline": -0.20,
                    "nasdaq_decline": -0.25,
                    "dow_decline": -0.18,
                    "trigger_source": "economic_data"
                },
                "impact_score": 0.95,
                "confidence_score": 0.95
            },
            "interest_rate_hike": {
                "event_type": "interest_rate_change",
                "severity": "medium",
                "description": "Federal Reserve raised interest rates by 0.75%",
                "source_data": {
                    "rate_change": 0.0075,
                    "new_rate": 0.0525,
                    "announcement_date": datetime.utcnow().isoformat()
                },
                "impact_score": 0.6,
                "confidence_score": 1.0
            },
            "sector_rotation": {
                "event_type": "sector_rotation",
                "severity": "medium",
                "description": "Major rotation from growth to value stocks",
                "source_data": {
                    "growth_performance": -0.15,
                    "value_performance": 0.12,
                    "rotation_magnitude": 0.27
                },
                "impact_score": 0.5,
                "confidence_score": 0.8
            },
            "recovery": {
                "event_type": "market_recovery",
                "severity": "low",
                "description": "Market showing strong recovery signals",
                "source_data": {
                    "recovery_strength": 0.15,
                    "breadth_indicator": 0.8,
                    "momentum_score": 0.7
                },
                "impact_score": 0.3,
                "confidence_score": 0.75
            }
        }
    
    def _load_life_event_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined life event scenario data"""
        return {
            "job_loss": {
                "severity": "high",
                "description": "User reported job loss with 3-month severance",
                "source_data": {
                    "event_type": "employment_change",
                    "income_impact": -1.0,
                    "severance_months": 3,
                    "benefits_continuation": True,
                    "reported_date": datetime.utcnow().isoformat()
                },
                "impact_score": 0.8,
                "confidence_score": 1.0
            },
            "medical_emergency": {
                "severity": "critical",
                "description": "Major medical emergency requiring immediate funds",
                "source_data": {
                    "event_type": "medical_emergency",
                    "estimated_cost": 50000,
                    "insurance_coverage": 0.8,
                    "urgency": "immediate"
                },
                "impact_score": 0.9,
                "confidence_score": 1.0
            },
            "family_addition": {
                "severity": "medium",
                "description": "New family member increases monthly expenses",
                "source_data": {
                    "event_type": "family_change",
                    "expense_increase": 1500,
                    "duration": "permanent",
                    "tax_implications": True
                },
                "impact_score": 0.6,
                "confidence_score": 1.0
            },
            "inheritance": {
                "severity": "low",
                "description": "Received inheritance requiring investment planning",
                "source_data": {
                    "event_type": "windfall",
                    "amount": 100000,
                    "tax_implications": True,
                    "investment_timeline": "long_term"
                },
                "impact_score": 0.4,
                "confidence_score": 1.0
            },
            "divorce": {
                "severity": "high",
                "description": "Divorce proceedings requiring asset division",
                "source_data": {
                    "event_type": "marital_change",
                    "asset_division": 0.5,
                    "support_obligations": True,
                    "legal_costs": 25000
                },
                "impact_score": 0.85,
                "confidence_score": 1.0
            }
        }
    
    def _load_compound_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Load compound scenario definitions"""
        return {
            "perfect_storm": {
                "description": "Job loss during market crash",
                "components": [
                    {"type": "life", "scenario": "job_loss"},
                    {"type": "market", "scenario": "market_crash"}
                ],
                "severity_multiplier": 1.5,
                "complexity_score": 0.95
            },
            "economic_pressure": {
                "description": "Medical emergency during interest rate hikes",
                "components": [
                    {"type": "life", "scenario": "medical_emergency"},
                    {"type": "market", "scenario": "interest_rate_hike"}
                ],
                "severity_multiplier": 1.3,
                "complexity_score": 0.8
            },
            "opportunity_crisis": {
                "description": "Inheritance during market volatility",
                "components": [
                    {"type": "life", "scenario": "inheritance"},
                    {"type": "market", "scenario": "volatility_spike"}
                ],
                "severity_multiplier": 1.1,
                "complexity_score": 0.7
            }
        }
    
    def get_available_scenarios(self) -> Dict[str, List[str]]:
        """Get list of available scenarios by type"""
        return {
            "market_scenarios": list(self.market_scenarios.keys()),
            "life_event_scenarios": list(self.life_event_scenarios.keys()),
            "compound_scenarios": list(self.compound_scenarios.keys())
        }
    
    def validate_trigger(self, trigger: TriggerEvent) -> bool:
        """Validate a trigger event for completeness"""
        required_fields = [
            'trigger_id', 'trigger_type', 'event_type', 'severity',
            'description', 'source_data', 'impact_score', 'confidence_score'
        ]
        
        for field in required_fields:
            if not hasattr(trigger, field) or getattr(trigger, field) is None:
                return False
        
        # Validate score ranges
        if not (0.0 <= trigger.impact_score <= 1.0):
            return False
        
        if not (0.0 <= trigger.confidence_score <= 1.0):
            return False
        
        return True