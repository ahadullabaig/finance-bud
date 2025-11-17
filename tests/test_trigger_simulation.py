"""
Trigger Simulation and CMVL Testing Framework

Tests for trigger detection, simulation, and CMVL (Continuous Monitoring 
and Verification Loop) workflows with realistic market and life event scenarios.

Requirements: 2.1, 2.2, 10.2, 11.1, 11.2
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, List, Any

from agents.mock_interfaces import (
    MockOrchestrationAgent, MockInformationRetrievalAgent, MockVerificationAgent
)
from agents.communication import AgentCommunicationFramework
from data_models.schemas import (
    AgentMessage, MessageType, Priority, TriggerEvent, SeverityLevel,
    MarketEventType, MarketData
)


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
 

class 
TestTriggerSimulation:
    """Test suite for trigger simulation and CMVL workflows"""
    
    @pytest.fixture
    def trigger_simulator(self):
        """Create trigger simulator for testing"""
        return TriggerSimulator()
    
    @pytest.fixture
    def communication_framework(self):
        """Create communication framework for CMVL testing"""
        return AgentCommunicationFramework()
    
    @pytest.fixture
    def mock_agents(self, communication_framework):
        """Create mock agents for CMVL testing"""
        agents = {
            'orchestrator': MockOrchestrationAgent("cmvl_oa_001"),
            'ira': MockInformationRetrievalAgent("cmvl_ira_001"),
            'verifier': MockVerificationAgent("cmvl_va_001")
        }
        
        for agent in agents.values():
            communication_framework.register_agent(agent)
        
        return agents
    
    def test_market_trigger_generation(self, trigger_simulator):
        """Test generation of realistic market triggers"""
        # Test volatility spike trigger
        volatility_trigger = trigger_simulator.generate_market_trigger("volatility_spike")
        
        assert volatility_trigger.trigger_type == "market_event"
        assert volatility_trigger.event_type == MarketEventType.VOLATILITY_SPIKE
        assert volatility_trigger.severity == SeverityLevel.HIGH
        assert "volatility" in volatility_trigger.description.lower()
        assert volatility_trigger.impact_score > 0.5
        assert volatility_trigger.confidence_score > 0.8
        
        # Test market crash trigger
        crash_trigger = trigger_simulator.generate_market_trigger("market_crash")
        
        assert crash_trigger.event_type == MarketEventType.MARKET_CRASH
        assert crash_trigger.severity == SeverityLevel.CRITICAL
        assert crash_trigger.impact_score > 0.9
        assert "crash" in crash_trigger.description.lower()
    
    def test_life_event_trigger_generation(self, trigger_simulator):
        """Test generation of realistic life event triggers"""
        # Test job loss trigger
        job_loss_trigger = trigger_simulator.generate_life_event_trigger("job_loss")
        
        assert job_loss_trigger.trigger_type == "life_event"
        assert job_loss_trigger.severity == SeverityLevel.HIGH
        assert "job loss" in job_loss_trigger.description.lower()
        assert job_loss_trigger.impact_score > 0.7
        assert job_loss_trigger.confidence_score == 1.0
        
        # Test medical emergency trigger
        medical_trigger = trigger_simulator.generate_life_event_trigger("medical_emergency")
        
        assert medical_trigger.severity == SeverityLevel.CRITICAL
        assert medical_trigger.impact_score > 0.8
        assert "medical" in medical_trigger.description.lower()
    
    def test_compound_trigger_generation(self, trigger_simulator):
        """Test generation of compound triggers for complex scenarios"""
        # Test perfect storm scenario
        perfect_storm_triggers = trigger_simulator.generate_compound_trigger(["perfect_storm"])
        
        assert len(perfect_storm_triggers) >= 2
        
        # Should contain both job loss and market crash
        trigger_types = [t.trigger_type for t in perfect_storm_triggers]
        assert "life_event" in trigger_types
        assert "market_event" in trigger_types
        
        # Test individual scenarios
        individual_triggers = trigger_simulator.generate_compound_trigger(["job_loss", "market_crash"])
        
        assert len(individual_triggers) == 2
        assert any(t.trigger_type == "life_event" for t in individual_triggers)
        assert any(t.trigger_type == "market_event" for t in individual_triggers)
    
    def test_trigger_severity_assessment(self, trigger_simulator):
        """Test trigger severity assessment accuracy"""
        # Critical triggers
        critical_triggers = [
            trigger_simulator.generate_market_trigger("market_crash"),
            trigger_simulator.generate_life_event_trigger("medical_emergency")
        ]
        
        for trigger in critical_triggers:
            assert trigger.severity == SeverityLevel.CRITICAL
            assert trigger.impact_score > 0.8
        
        # Medium severity triggers
        medium_triggers = [
            trigger_simulator.generate_market_trigger("interest_rate_hike"),
            trigger_simulator.generate_life_event_trigger("family_addition")
        ]
        
        for trigger in medium_triggers:
            assert trigger.severity == SeverityLevel.MEDIUM
            assert 0.4 <= trigger.impact_score <= 0.8
        
        # Low severity triggers
        low_triggers = [
            trigger_simulator.generate_market_trigger("recovery"),
            trigger_simulator.generate_life_event_trigger("inheritance")
        ]
        
        for trigger in low_triggers:
            assert trigger.severity in [SeverityLevel.LOW, SeverityLevel.MEDIUM]
            assert trigger.impact_score <= 0.6
    
    @pytest.mark.asyncio
    async def test_cmvl_trigger_workflow(self, communication_framework, mock_agents, trigger_simulator):
        """Test complete CMVL trigger and response workflow"""
        # Generate market volatility trigger
        volatility_trigger = trigger_simulator.generate_market_trigger("volatility_spike")
        
        # Send trigger to orchestration agent
        trigger_message = communication_framework.create_message(
            sender_id="test_system",
            target_id="cmvl_oa_001",
            message_type=MessageType.REQUEST,
            payload={"trigger_event": volatility_trigger.dict()},
            priority=Priority.HIGH
        )
        
        success = await communication_framework.send_message(trigger_message)
        assert success is True
        
        # Verify orchestration agent received trigger
        orchestrator = mock_agents['orchestrator']
        assert orchestrator.message_queue.qsize() > 0
        
        # Process the trigger message
        received_message = await orchestrator.message_queue.get()
        response = await orchestrator.process_message(received_message)
        
        assert response is not None
        assert response.message_type == MessageType.RESPONSE
        assert "cmvl_activated" in response.payload
        assert response.payload["cmvl_activated"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_trigger_handling(self, communication_framework, mock_agents, trigger_simulator):
        """Test handling of multiple concurrent triggers"""
        # Generate multiple triggers
        triggers = [
            trigger_simulator.generate_market_trigger("volatility_spike"),
            trigger_simulator.generate_life_event_trigger("job_loss"),
            trigger_simulator.generate_market_trigger("interest_rate_hike")
        ]
        
        # Send all triggers concurrently
        tasks = []
        for i, trigger in enumerate(triggers):
            message = communication_framework.create_message(
                sender_id="test_system",
                target_id="cmvl_oa_001",
                message_type=MessageType.REQUEST,
                payload={"trigger_event": trigger.dict(), "trigger_sequence": i},
                priority=Priority.HIGH
            )
            tasks.append(communication_framework.send_message(message))
        
        # Wait for all messages to be sent
        results = await asyncio.gather(*tasks)
        assert all(results)
        
        # Verify orchestration agent received all triggers
        orchestrator = mock_agents['orchestrator']
        assert orchestrator.message_queue.qsize() >= len(triggers)
    
    @pytest.mark.asyncio
    async def test_trigger_priority_handling(self, communication_framework, mock_agents, trigger_simulator):
        """Test trigger priority handling and escalation"""
        # Generate triggers with different severities
        triggers = [
            ("low_priority", trigger_simulator.generate_market_trigger("recovery")),
            ("critical_priority", trigger_simulator.generate_life_event_trigger("medical_emergency")),
            ("medium_priority", trigger_simulator.generate_market_trigger("interest_rate_hike"))
        ]
        
        # Send triggers with appropriate priorities
        for priority_name, trigger in triggers:
            priority = Priority.CRITICAL if trigger.severity == SeverityLevel.CRITICAL else Priority.MEDIUM
            
            message = communication_framework.create_message(
                sender_id="test_system",
                target_id="cmvl_oa_001",
                message_type=MessageType.REQUEST,
                payload={"trigger_event": trigger.dict()},
                priority=priority
            )
            
            success = await communication_framework.send_message(message)
            assert success is True
    
    @pytest.mark.asyncio
    async def test_cmvl_performance_metrics(self, communication_framework, mock_agents, trigger_simulator):
        """Test CMVL performance monitoring and metrics collection"""
        # Generate and process multiple triggers
        num_triggers = 10
        start_time = datetime.utcnow()
        
        for i in range(num_triggers):
            trigger = trigger_simulator.generate_market_trigger("volatility_spike")
            message = communication_framework.create_message(
                sender_id="test_system",
                target_id="cmvl_oa_001",
                message_type=MessageType.REQUEST,
                payload={"trigger_event": trigger.dict(), "batch_id": i},
                priority=Priority.MEDIUM
            )
            
            await communication_framework.send_message(message)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance metrics
        health = communication_framework.get_system_health()
        assert health["total_messages"] >= num_triggers
        assert health["success_rate"] > 0.9
        
        # Calculate throughput
        throughput = num_triggers / processing_time
        assert throughput > 0  # Should have measurable throughput
    
    def test_trigger_data_validation(self, trigger_simulator):
        """Test trigger data validation and schema compliance"""
        # Test all market trigger types
        market_scenarios = ["volatility_spike", "market_crash", "interest_rate_hike", "sector_rotation", "recovery"]
        
        for scenario in market_scenarios:
            trigger = trigger_simulator.generate_market_trigger(scenario)
            
            # Validate required fields
            assert trigger.trigger_id is not None
            assert trigger.trigger_type == "market_event"
            assert trigger.event_type in MarketEventType
            assert trigger.severity in SeverityLevel
            assert trigger.description is not None
            assert isinstance(trigger.source_data, dict)
            assert 0.0 <= trigger.impact_score <= 1.0
            assert 0.0 <= trigger.confidence_score <= 1.0
            assert trigger.detector_agent_id == "trigger_simulator"
            assert trigger.correlation_id is not None
        
        # Test all life event trigger types
        life_scenarios = ["job_loss", "medical_emergency", "family_addition", "inheritance", "divorce"]
        
        for scenario in life_scenarios:
            trigger = trigger_simulator.generate_life_event_trigger(scenario)
            
            # Validate required fields
            assert trigger.trigger_id is not None
            assert trigger.trigger_type == "life_event"
            assert trigger.severity in SeverityLevel
            assert trigger.description is not None
            assert isinstance(trigger.source_data, dict)
            assert 0.0 <= trigger.impact_score <= 1.0
            assert 0.0 <= trigger.confidence_score <= 1.0
    
    def test_trigger_correlation_tracking(self, trigger_simulator):
        """Test trigger correlation ID tracking for related events"""
        # Generate multiple related triggers
        correlation_id = str(uuid4())
        
        triggers = []
        for scenario in ["volatility_spike", "market_crash"]:
            trigger = trigger_simulator.generate_market_trigger(scenario)
            trigger.correlation_id = correlation_id  # Set same correlation ID
            triggers.append(trigger)
        
        # Verify all triggers have same correlation ID
        for trigger in triggers:
            assert trigger.correlation_id == correlation_id
        
        # Verify triggers are different events
        assert triggers[0].event_type != triggers[1].event_type
        assert triggers[0].trigger_id != triggers[1].trigger_id


class TestCMVLIntegration:
    """Integration tests for CMVL (Continuous Monitoring and Verification Loop)"""
    
    @pytest.fixture
    def cmvl_system(self):
        """Create complete CMVL system for integration testing"""
        communication_framework = AgentCommunicationFramework()
        
        # Create and register all agents
        agents = {
            'orchestrator': MockOrchestrationAgent("cmvl_integration_oa"),
            'ira': MockInformationRetrievalAgent("cmvl_integration_ira"),
            'verifier': MockVerificationAgent("cmvl_integration_va")
        }
        
        for agent in agents.values():
            communication_framework.register_agent(agent)
        
        return {
            'framework': communication_framework,
            'agents': agents,
            'simulator': TriggerSimulator()
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_cmvl_workflow(self, cmvl_system):
        """Test complete end-to-end CMVL workflow"""
        framework = cmvl_system['framework']
        agents = cmvl_system['agents']
        simulator = cmvl_system['simulator']
        
        # Step 1: Generate market trigger
        market_trigger = simulator.generate_market_trigger("market_crash")
        
        # Step 2: Send trigger to orchestration agent
        trigger_message = framework.create_message(
            sender_id="market_monitor",
            target_id="cmvl_integration_oa",
            message_type=MessageType.REQUEST,
            payload={"trigger_event": market_trigger.dict()},
            priority=Priority.CRITICAL
        )
        
        success = await framework.send_message(trigger_message)
        assert success is True
        
        # Step 3: Process orchestration response
        orchestrator = agents['orchestrator']
        trigger_msg = await orchestrator.message_queue.get()
        oa_response = await orchestrator.process_message(trigger_msg)
        
        assert oa_response is not None
        assert oa_response.payload["cmvl_activated"] is True
        
        # Step 4: Simulate verification agent CMVL activation
        cmvl_message = framework.create_message(
            sender_id="cmvl_integration_oa",
            target_id="cmvl_integration_va",
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": market_trigger.dict()},
            priority=Priority.CRITICAL
        )
        
        await framework.send_message(cmvl_message)
        
        # Step 5: Process verification response
        verifier = agents['verifier']
        cmvl_msg = await verifier.message_queue.get()
        va_response = await verifier.process_message(cmvl_msg)
        
        assert va_response is not None
        assert va_response.payload["cmvl_activated"] is True
        
        # Step 6: Verify system health after CMVL
        health = framework.get_system_health()
        assert health["success_rate"] > 0.9
        assert health["total_messages"] >= 2
    
    @pytest.mark.asyncio
    async def test_cmvl_rollback_scenario(self, cmvl_system):
        """Test CMVL rollback capabilities during failures"""
        framework = cmvl_system['framework']
        agents = cmvl_system['agents']
        simulator = cmvl_system['simulator']
        
        # Generate critical trigger that might require rollback
        critical_trigger = simulator.generate_life_event_trigger("medical_emergency")
        
        # Send trigger and simulate failure scenario
        message = framework.create_message(
            sender_id="emergency_system",
            target_id="cmvl_integration_oa",
            message_type=MessageType.REQUEST,
            payload={
                "trigger_event": critical_trigger.dict(),
                "simulate_failure": True  # Flag to simulate failure
            },
            priority=Priority.CRITICAL
        )
        
        success = await framework.send_message(message)
        assert success is True
        
        # Verify system can handle failure gracefully
        orchestrator = agents['orchestrator']
        assert orchestrator.message_queue.qsize() > 0
    
    @pytest.mark.asyncio
    async def test_cmvl_performance_under_load(self, cmvl_system):
        """Test CMVL performance under high trigger load"""
        framework = cmvl_system['framework']
        simulator = cmvl_system['simulator']
        
        # Generate high volume of triggers
        num_triggers = 50
        start_time = datetime.utcnow()
        
        tasks = []
        for i in range(num_triggers):
            # Alternate between different trigger types
            if i % 3 == 0:
                trigger = simulator.generate_market_trigger("volatility_spike")
            elif i % 3 == 1:
                trigger = simulator.generate_life_event_trigger("job_loss")
            else:
                trigger = simulator.generate_market_trigger("interest_rate_hike")
            
            message = framework.create_message(
                sender_id="load_test_system",
                target_id="cmvl_integration_oa",
                message_type=MessageType.REQUEST,
                payload={"trigger_event": trigger.dict(), "load_test_id": i},
                priority=Priority.MEDIUM
            )
            
            tasks.append(framework.send_message(message))
        
        # Execute all triggers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance metrics
        successful_sends = sum(1 for result in results if result is True)
        success_rate = successful_sends / num_triggers
        throughput = num_triggers / processing_time
        
        assert success_rate >= 0.9  # At least 90% success rate
        assert throughput > 10  # At least 10 triggers per second
        
        # Verify system health
        health = framework.get_system_health()
        assert health["total_messages"] >= num_triggers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])