"""
Agent Testing Framework

Tests for individual agent functionality, mock interfaces,
and agent communication protocols.

Requirements: 11.1, 11.2
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from agents.mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent,
    MockVerificationAgent,
    MockExecutionAgent
)
from agents.communication import AgentCommunicationFramework
from agents.orchestration_agent import OrchestrationAgent
from data_models.schemas import (
    AgentMessage, MessageType, Priority, TriggerEvent, SeverityLevel, 
    MarketEventType, EnhancedPlanRequest, FinancialState
)


class TestMockAgents:
    """Test suite for mock agent implementations"""
    
    @pytest.fixture
    def communication_framework(self):
        """Create communication framework for testing"""
        return AgentCommunicationFramework()
    
    @pytest.fixture
    def mock_orchestration_agent(self):
        """Create mock orchestration agent"""
        return MockOrchestrationAgent()
    
    @pytest.fixture
    def mock_planning_agent(self):
        """Create mock planning agent"""
        return MockPlanningAgent()
    
    @pytest.fixture
    def mock_ira_agent(self):
        """Create mock information retrieval agent"""
        return MockInformationRetrievalAgent()
    
    @pytest.fixture
    def mock_verification_agent(self):
        """Create mock verification agent"""
        return MockVerificationAgent()
    
    @pytest.fixture
    def mock_execution_agent(self):
        """Create mock execution agent"""
        return MockExecutionAgent()
    
    async def test_orchestration_agent_planning_request(self, mock_orchestration_agent):
        """Test orchestration agent handles planning requests"""
        message = AgentMessage(
            agent_id="test_client",
            target_agent_id=mock_orchestration_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Save for retirement"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_orchestration_agent.process_message(message)
        
        assert response is not None
        assert response.message_type == MessageType.RESPONSE
        assert "workflow_plan" in response.payload
        assert response.payload["status"] == "workflow_initiated"
    
    async def test_orchestration_agent_trigger_handling(self, mock_orchestration_agent):
        """Test orchestration agent handles trigger events"""
        message = AgentMessage(
            agent_id="test_ira",
            target_agent_id=mock_orchestration_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "trigger_event": {
                    "trigger_id": str(uuid4()),
                    "severity": "high",
                    "description": "Market volatility spike"
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_orchestration_agent.process_message(message)
        
        assert response is not None
        assert response.payload["cmvl_activated"] is True
        assert "actions_initiated" in response.payload
    
    async def test_planning_agent_plan_generation(self, mock_planning_agent):
        """Test planning agent generates comprehensive plans"""
        message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=mock_planning_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": "Build emergency fund",
                    "time_horizon": 12
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_planning_agent.process_message(message)
        
        assert response is not None
        assert response.payload["plan_generated"] is True
        assert "plan_steps" in response.payload
        assert "search_paths" in response.payload
        assert "reasoning_trace" in response.payload
        assert len(response.payload["search_paths"]) >= 3  # Multiple strategies
    
    async def test_ira_market_data_fetching(self, mock_ira_agent):
        """Test IRA fetches realistic market data"""
        message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=mock_ira_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"market_data_request": {"symbols": ["SPY", "BND"]}},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_ira_agent.process_message(message)
        
        assert response is not None
        assert "market_data" in response.payload
        market_data = response.payload["market_data"]
        assert "market_volatility" in market_data
        assert "interest_rates" in market_data
        assert "sector_trends" in market_data
    
    async def test_ira_trigger_detection(self, mock_ira_agent):
        """Test IRA detects market triggers"""
        # Set volatile scenario
        await mock_ira_agent._simulate_scenario(AgentMessage(
            agent_id="test",
            target_agent_id=mock_ira_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"scenario": "volatile"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        ))
        
        message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=mock_ira_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"trigger_detection": {}},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_ira_agent.process_message(message)
        
        assert response is not None
        assert "triggers_detected" in response.payload
        assert response.payload["monitoring_active"] is True
    
    async def test_verification_agent_plan_verification(self, mock_verification_agent):
        """Test verification agent verifies plans"""
        message = AgentMessage(
            agent_id="test_planning",
            target_agent_id=mock_verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "verification_request": {
                    "plan_id": str(uuid4()),
                    "plan_steps": [
                        {
                            "step_id": str(uuid4()),
                            "action_type": "invest",
                            "amount": 50000,
                            "risk_level": "medium"
                        }
                    ]
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_verification_agent.process_message(message)
        
        assert response is not None
        assert "verification_report" in response.payload
        report = response.payload["verification_report"]
        assert "verification_status" in report
        assert "constraints_checked" in report
        assert "overall_risk_score" in report
    
    async def test_execution_agent_plan_execution(self, mock_execution_agent):
        """Test execution agent executes plans"""
        message = AgentMessage(
            agent_id="test_verification",
            target_agent_id=mock_execution_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "execution_request": {
                    "plan_steps": [
                        {
                            "step_id": str(uuid4()),
                            "action_type": "buy_stock",
                            "amount": 10000
                        }
                    ]
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_execution_agent.process_message(message)
        
        assert response is not None
        assert response.payload["execution_completed"] is True
        assert "execution_results" in response.payload
        assert response.payload["portfolio_updated"] is True
    
    async def test_agent_health_monitoring(self, mock_orchestration_agent):
        """Test agent health monitoring"""
        await mock_orchestration_agent.start()
        
        health_status = mock_orchestration_agent.get_health_status()
        
        assert health_status["agent_id"] == mock_orchestration_agent.agent_id
        assert health_status["status"] == "running"
        assert "uptime_seconds" in health_status
        assert "success_rate" in health_status
        
        await mock_orchestration_agent.stop()
    
    async def test_agent_performance_metrics(self, mock_planning_agent):
        """Test agent performance metrics collection"""
        metrics = mock_planning_agent.get_performance_metrics()
        
        assert hasattr(metrics, 'execution_time')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'success_rate')
        assert 0.0 <= metrics.success_rate <= 1.0


class TestAgentCommunication:
    """Test suite for agent communication protocols"""
    
    @pytest.fixture
    def communication_framework(self):
        """Create communication framework"""
        return AgentCommunicationFramework()
    
    @pytest.fixture
    def test_agents(self, communication_framework):
        """Create and register test agents"""
        agents = {
            'orchestrator': MockOrchestrationAgent("test_oa_001"),
            'planner': MockPlanningAgent("test_pa_001"),
            'ira': MockInformationRetrievalAgent("test_ira_001"),
            'verifier': MockVerificationAgent("test_va_001"),
            'executor': MockExecutionAgent("test_ea_001")
        }
        
        # Register agents with capabilities
        communication_framework.register_agent(agents['orchestrator'], ['workflow_management', 'trigger_handling'])
        communication_framework.register_agent(agents['planner'], ['financial_planning', 'strategy_generation'])
        communication_framework.register_agent(agents['ira'], ['market_data', 'trigger_detection'])
        communication_framework.register_agent(agents['verifier'], ['constraint_checking', 'compliance'])
        communication_framework.register_agent(agents['executor'], ['transaction_execution', 'portfolio_management'])
        
        return agents
    
    async def test_message_routing(self, communication_framework, test_agents):
        """Test message routing between agents"""
        message = communication_framework.create_message(
            sender_id="test_oa_001",
            target_id="test_pa_001",
            message_type=MessageType.REQUEST,
            payload={"test": "routing"}
        )
        
        success = await communication_framework.send_message(message)
        assert success is True
        
        # Check message was received
        planner = test_agents['planner']
        assert planner.message_queue.qsize() > 0
    
    async def test_broadcast_messaging(self, communication_framework, test_agents):
        """Test broadcast messaging to all agents"""
        message = communication_framework.create_message(
            sender_id="test_oa_001",
            target_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            payload={"broadcast": "system_update"}
        )
        
        success = await communication_framework.send_message(message)
        assert success is True
        
        # Check all agents except sender received message
        for agent_name, agent in test_agents.items():
            if agent.agent_id != "test_oa_001":
                assert agent.message_queue.qsize() > 0
    
    async def test_correlation_tracking(self, communication_framework, test_agents):
        """Test correlation ID tracking across messages"""
        correlation_id = str(uuid4())
        
        # Send multiple related messages
        for i in range(3):
            message = communication_framework.create_message(
                sender_id="test_oa_001",
                target_id="test_pa_001",
                message_type=MessageType.REQUEST,
                payload={"sequence": i},
                correlation_id=correlation_id
            )
            await communication_framework.send_message(message)
        
        # Check correlation tracking
        trace = communication_framework.get_correlation_trace(correlation_id)
        assert len(trace) == 3
        assert all(msg.correlation_id == correlation_id for msg in trace)
    
    async def test_circuit_breaker_functionality(self, communication_framework, test_agents):
        """Test circuit breaker prevents cascade failures"""
        # This would require simulating failures in a real implementation
        # For now, test that circuit breakers are created
        assert "test_oa_001" in communication_framework.circuit_breakers
        assert "test_pa_001" in communication_framework.circuit_breakers
    
    async def test_system_health_monitoring(self, communication_framework, test_agents):
        """Test system health monitoring"""
        health = communication_framework.get_system_health()
        
        assert "framework_uptime" in health
        assert "total_messages" in health
        assert "success_rate" in health
        assert "registered_agents" in health
        assert health["registered_agents"] == len(test_agents)
        assert "agent_health" in health
        assert "circuit_breakers" in health


class TestOrchestrationAgent:
    """Test suite for the enhanced OrchestrationAgent"""
    
    @pytest.fixture
    async def orchestration_agent(self):
        """Create orchestration agent for testing"""
        agent = OrchestrationAgent("test_orchestration_agent")
        await agent.start()
        yield agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_orchestration_agent_initialization(self, orchestration_agent):
        """Test orchestration agent initializes correctly"""
        assert orchestration_agent.agent_id == "test_orchestration_agent"
        assert orchestration_agent.agent_type == "orchestration"
        assert orchestration_agent.status == "running"
        assert orchestration_agent.session_manager is not None
        assert orchestration_agent.goal_parser is not None
        assert orchestration_agent.trigger_monitor is not None
        assert orchestration_agent.task_delegator is not None
    
    @pytest.mark.asyncio
    async def test_user_goal_processing(self, orchestration_agent):
        """Test processing of user financial goals"""
        user_context = {
            "financial_state": {
                "total_assets": 50000,
                "monthly_income": 5000,
                "monthly_expenses": 3000
            },
            "risk_profile": {
                "risk_tolerance": "moderate",
                "investment_horizon": 120
            },
            "tax_context": {
                "filing_status": "single",
                "marginal_tax_rate": 0.22
            }
        }
        
        workflow_id = await orchestration_agent.process_user_goal(
            user_id="test_user_001",
            goal_text="I want to save $100,000 for retirement in 10 years",
            user_context=user_context
        )
        
        assert workflow_id is not None
        assert workflow_id in orchestration_agent.current_workflows
        
        workflow = orchestration_agent.current_workflows[workflow_id]
        assert workflow["user_id"] == "test_user_001"
        assert "retirement" in workflow["plan_request"]["optimization_preferences"]["goal_type"]
    
    @pytest.mark.asyncio
    async def test_goal_parsing_complex_scenarios(self, orchestration_agent):
        """Test goal parsing for complex multi-constraint scenarios"""
        user_context = {
            "user_id": "test_user_002",
            "financial_state": {"monthly_income": 8000},
            "risk_profile": {"risk_tolerance": "conservative"}
        }
        
        # Test complex goal with multiple constraints
        complex_goal = "I need to save for my child's college education in 8 years while also building an emergency fund of $20,000 within 2 years, but I'm conservative with investments"
        
        plan_request = await orchestration_agent.goal_parser.parse_goal(complex_goal, user_context)
        
        assert plan_request.user_goal == complex_goal
        assert plan_request.time_horizon == 96  # 8 years in months
        assert len(plan_request.constraints) > 0
        
        # Check for extracted constraints
        constraint_types = [c.get("type") for c in plan_request.constraints]
        assert "time" in [c.get("type") for c in plan_request.constraints]
        assert "risk" in [c.get("type") for c in plan_request.constraints]
    
    @pytest.mark.asyncio
    async def test_trigger_event_handling(self, orchestration_agent):
        """Test handling of trigger events for CMVL initiation"""
        trigger = TriggerEvent(
            trigger_type="market_volatility",
            event_type=MarketEventType.VOLATILITY_SPIKE,
            severity=SeverityLevel.HIGH,
            description="Market volatility spike detected",
            source_data={"volatility_index": 0.35},
            impact_score=0.8,
            confidence_score=0.9,
            detector_agent_id="test_ira",
            correlation_id=str(uuid4())
        )
        
        await orchestration_agent.handle_trigger_event(trigger)
        
        # Check trigger was registered
        assert trigger.trigger_id in orchestration_agent.trigger_monitor.active_triggers
        assert orchestration_agent.workflow_metrics["trigger_events_handled"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_trigger_handling(self, orchestration_agent):
        """Test handling of concurrent trigger events"""
        triggers = [
            TriggerEvent(
                trigger_type="market_volatility",
                event_type=MarketEventType.MARKET_CRASH,
                severity=SeverityLevel.CRITICAL,
                description="Market crash detected",
                source_data={"market_drop": -0.15},
                impact_score=0.95,
                confidence_score=0.85,
                detector_agent_id="test_ira",
                correlation_id=str(uuid4())
            ),
            TriggerEvent(
                trigger_type="life_event",
                event_type=MarketEventType.VOLATILITY_SPIKE,  # Using available enum
                severity=SeverityLevel.HIGH,
                description="Job loss reported",
                source_data={"event_type": "job_loss"},
                impact_score=0.7,
                confidence_score=1.0,
                detector_agent_id="user_input",
                correlation_id=str(uuid4())
            ),
            TriggerEvent(
                trigger_type="emergency",
                event_type=MarketEventType.VOLATILITY_SPIKE,  # Using available enum
                severity=SeverityLevel.HIGH,
                description="Family emergency",
                source_data={"emergency_type": "medical"},
                impact_score=0.8,
                confidence_score=1.0,
                detector_agent_id="user_input",
                correlation_id=str(uuid4())
            )
        ]
        
        await orchestration_agent.handle_concurrent_triggers(triggers)
        
        # Check concurrent trigger handling
        assert orchestration_agent.workflow_metrics["concurrent_triggers_handled"] > 0
    
    @pytest.mark.asyncio
    async def test_session_management(self, orchestration_agent):
        """Test session management and correlation tracking"""
        # Create session
        session_id = orchestration_agent.session_manager.create_session(
            "test_user_003", 
            "Build emergency fund"
        )
        
        assert session_id is not None
        session = orchestration_agent.session_manager.get_session(session_id)
        assert session is not None
        assert session["user_id"] == "test_user_003"
        assert session["initial_goal"] == "Build emergency fund"
        
        # Test correlation registration
        correlation_id = str(uuid4())
        orchestration_agent.session_manager.register_correlation(correlation_id, session_id)
        
        retrieved_session = orchestration_agent.session_manager.get_session_by_correlation(correlation_id)
        assert retrieved_session["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_task_delegation(self, orchestration_agent):
        """Test intelligent task delegation system"""
        plan_request = EnhancedPlanRequest(
            user_id="test_user_004",
            user_goal="Save for house down payment",
            current_state={"monthly_income": 6000},
            constraints=[],
            risk_profile={"risk_tolerance": "moderate"},
            regulatory_requirements=[],
            tax_considerations={"filing_status": "married"},
            time_horizon=60,
            correlation_id=str(uuid4()),
            session_id=str(uuid4())
        )
        
        task_id = await orchestration_agent.task_delegator.delegate_planning_task(plan_request)
        
        assert task_id is not None
        task_status = orchestration_agent.task_delegator.get_task_status(task_id)
        assert task_status is not None
        assert task_status["task_type"] == "planning"
        assert task_status["assigned_agent"] == "planning_agent"
        assert task_status["status"] == "delegated"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, orchestration_agent):
        """Test circuit breaker functionality for agent failures"""
        # Check circuit breakers are initialized
        assert "planning_agent" in orchestration_agent.circuit_breakers
        assert "verification_agent" in orchestration_agent.circuit_breakers
        assert "information_retrieval_agent" in orchestration_agent.circuit_breakers
        assert "execution_agent" in orchestration_agent.circuit_breakers
        
        # Test circuit breaker state
        planning_cb = orchestration_agent.circuit_breakers["planning_agent"]
        assert planning_cb.state == "closed"
        assert planning_cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_workflow_coordination(self, orchestration_agent):
        """Test workflow coordination and state management"""
        # Create a workflow
        plan_request = EnhancedPlanRequest(
            user_id="test_user_005",
            user_goal="Retirement planning",
            current_state={},
            constraints=[],
            risk_profile={},
            regulatory_requirements=[],
            tax_considerations={},
            time_horizon=360,
            correlation_id=str(uuid4()),
            session_id=str(uuid4())
        )
        
        workflow_id = await orchestration_agent._create_workflow(plan_request)
        
        assert workflow_id in orchestration_agent.current_workflows
        workflow = orchestration_agent.current_workflows[workflow_id]
        assert workflow["user_id"] == "test_user_005"
        assert workflow["correlation_id"] == plan_request.correlation_id
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, orchestration_agent):
        """Test comprehensive health monitoring"""
        health_status = orchestration_agent.get_comprehensive_health_status()
        
        # Check base health metrics
        assert "agent_id" in health_status
        assert "status" in health_status
        assert "uptime_seconds" in health_status
        
        # Check orchestration-specific metrics
        assert "workflow_metrics" in health_status
        assert "active_workflows" in health_status
        assert "active_sessions" in health_status
        assert "circuit_breaker_status" in health_status
        assert "communication_framework_health" in health_status
        assert "trigger_monitor_status" in health_status
        
        # Verify workflow metrics structure
        workflow_metrics = health_status["workflow_metrics"]
        assert "total_workflows" in workflow_metrics
        assert "successful_workflows" in workflow_metrics
        assert "failed_workflows" in workflow_metrics
        assert "trigger_events_handled" in workflow_metrics
    
    @pytest.mark.asyncio
    async def test_message_processing(self, orchestration_agent):
        """Test message processing from other agents"""
        # Test plan generated message
        plan_message = AgentMessage(
            agent_id="planning_agent",
            target_agent_id=orchestration_agent.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "action": "plan_generated",
                "plan": {"plan_id": str(uuid4()), "steps": []},
                "task_id": str(uuid4())
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await orchestration_agent.process_message(plan_message)
        # Response may be None as it delegates to verification
        
        # Test health check message
        health_message = AgentMessage(
            agent_id="test_monitor",
            target_agent_id=orchestration_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"action": "health_check"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        health_response = await orchestration_agent.process_message(health_message)
        assert health_response is not None
        assert health_response.message_type == MessageType.RESPONSE
        assert "health_status" in health_response.payload
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestration_agent):
        """Test error handling and recovery mechanisms"""
        # Test invalid message processing
        invalid_message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=orchestration_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"action": "invalid_action"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await orchestration_agent.process_message(invalid_message)
        # Should handle gracefully without crashing
        assert orchestration_agent.status == "running"


# Async test runner for pytest
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])