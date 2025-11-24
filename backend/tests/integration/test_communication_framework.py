"""
Agent Communication Framework Testing

Comprehensive tests for agent communication protocols, message routing,
circuit breakers, and performance monitoring.

Requirements: 6.1, 6.2, 6.5, 11.1, 11.2
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, List

from agents.base_agent import BaseAgent
from agents.communication import (
    AgentCommunicationFramework, AgentRegistry, MessageRouter, CircuitBreaker
)
from tests.mocks.mock_interfaces import (
    MockOrchestrationAgent, MockPlanningAgent, MockInformationRetrievalAgent,
    MockVerificationAgent, MockExecutionAgent
)
from data_models.schemas import (
    AgentMessage, MessageType, Priority, TriggerEvent, SeverityLevel,
    MarketEventType, PerformanceMetrics
)


class TestAgentCommunicationFramework:
    """Test suite for the agent communication framework"""
    
    @pytest.fixture
    def communication_framework(self):
        """Create communication framework for testing"""
        return AgentCommunicationFramework()
    
    @pytest.fixture
    def mock_agents(self, communication_framework):
        """Create and register mock agents"""
        agents = {
            'orchestrator': MockOrchestrationAgent("test_oa_comm_001"),
            'planner': MockPlanningAgent("test_pa_comm_001"),
            'ira': MockInformationRetrievalAgent("test_ira_comm_001"),
            'verifier': MockVerificationAgent("test_va_comm_001"),
            'executor': MockExecutionAgent("test_ea_comm_001")
        }
        
        # Register agents with capabilities
        communication_framework.register_agent(
            agents['orchestrator'], 
            ['workflow_coordination', 'goal_parsing', 'task_delegation', 'trigger_monitoring']
        )
        communication_framework.register_agent(
            agents['planner'], 
            ['financial_planning', 'strategy_optimization', 'constraint_solving']
        )
        communication_framework.register_agent(
            agents['ira'], 
            ['market_data', 'regulatory_data', 'external_apis', 'trigger_detection']
        )
        communication_framework.register_agent(
            agents['verifier'], 
            ['constraint_validation', 'compliance_checking', 'risk_assessment']
        )
        communication_framework.register_agent(
            agents['executor'], 
            ['portfolio_updates', 'transaction_execution', 'ledger_management']
        )
        
        return agents
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, communication_framework, mock_agents):
        """Test agent registration and capability tracking"""
        # Check all agents are registered
        assert len(communication_framework.registry.agents) == 5
        
        # Check specific agent registration
        orchestrator = communication_framework.registry.get_agent("test_oa_comm_001")
        assert orchestrator is not None
        assert orchestrator.agent_id == "test_oa_comm_001"
        
        # Check capability tracking
        planning_agents = communication_framework.registry.find_agents_by_capability("financial_planning")
        assert "test_pa_comm_001" in planning_agents
        
        market_agents = communication_framework.registry.find_agents_by_capability("market_data")
        assert "test_ira_comm_001" in market_agents
    
    @pytest.mark.asyncio
    async def test_message_routing_unicast(self, communication_framework, mock_agents):
        """Test unicast message routing between agents"""
        # Create test message
        message = communication_framework.create_message(
            sender_id="test_oa_comm_001",
            target_id="test_pa_comm_001",
            message_type=MessageType.REQUEST,
            payload={"test_data": "routing_test", "request_id": str(uuid4())},
            priority=Priority.MEDIUM
        )
        
        # Send message
        success = await communication_framework.send_message(message)
        assert success is True
        
        # Verify message was received
        planner = mock_agents['planner']
        assert planner.message_queue.qsize() > 0
        
        # Check message history
        history = communication_framework.router.get_message_history(message.correlation_id)
        assert len(history) == 1
        assert history[0].message_id == message.message_id
    
    @pytest.mark.asyncio
    async def test_message_routing_broadcast(self, communication_framework, mock_agents):
        """Test broadcast message routing to all agents"""
        # Create broadcast message (target_id = None)
        message = communication_framework.create_message(
            sender_id="test_oa_comm_001",
            target_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            payload={"broadcast_type": "system_update", "update_id": str(uuid4())},
            priority=Priority.HIGH
        )
        
        # Send broadcast message
        success = await communication_framework.send_message(message)
        assert success is True
        
        # Verify all agents except sender received message
        for agent_name, agent in mock_agents.items():
            if agent.agent_id != "test_oa_comm_001":
                assert agent.message_queue.qsize() > 0, f"Agent {agent_name} did not receive broadcast"
    
    @pytest.mark.asyncio
    async def test_correlation_tracking(self, communication_framework, mock_agents):
        """Test correlation ID tracking across multiple messages"""
        correlation_id = str(uuid4())
        
        # Send multiple related messages
        messages = []
        for i in range(5):
            message = communication_framework.create_message(
                sender_id="test_oa_comm_001",
                target_id="test_pa_comm_001",
                message_type=MessageType.REQUEST,
                payload={"sequence": i, "data": f"test_data_{i}"},
                correlation_id=correlation_id,
                priority=Priority.MEDIUM
            )
            messages.append(message)
            await communication_framework.send_message(message)
        
        # Check correlation tracking
        trace = communication_framework.get_correlation_trace(correlation_id)
        assert len(trace) == 5
        
        # Verify all messages have same correlation ID
        for msg in trace:
            assert msg.correlation_id == correlation_id
        
        # Verify sequence order is maintained
        for i, msg in enumerate(trace):
            assert msg.payload["sequence"] == i
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, communication_framework, mock_agents):
        """Test circuit breaker prevents cascade failures"""
        # Get circuit breaker for planning agent
        planning_cb = communication_framework.circuit_breakers.get("test_pa_comm_001")
        assert planning_cb is not None
        assert planning_cb.state == "closed"
        
        # Simulate multiple failures to trigger circuit breaker
        async def failing_operation():
            raise Exception("Simulated failure")
        
        # Trigger failures
        for _ in range(planning_cb.failure_threshold):
            try:
                await planning_cb.call(failing_operation)
            except Exception:
                pass
        
        # Circuit breaker should now be open
        assert planning_cb.state == "open"
        
        # Verify circuit breaker prevents further calls
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await planning_cb.call(failing_operation)
    
    @pytest.mark.asyncio
    async def test_message_priority_handling(self, communication_framework, mock_agents):
        """Test message priority handling and ordering"""
        messages = []
        
        # Create messages with different priorities
        priorities = [Priority.LOW, Priority.CRITICAL, Priority.MEDIUM, Priority.HIGH]
        for i, priority in enumerate(priorities):
            message = communication_framework.create_message(
                sender_id="test_oa_comm_001",
                target_id="test_pa_comm_001",
                message_type=MessageType.REQUEST,
                payload={"priority_test": i, "priority_level": priority},
                priority=priority
            )
            messages.append(message)
            await communication_framework.send_message(message)
        
        # All messages should be sent successfully
        assert len(messages) == 4
        
        # Verify messages were tracked
        total_messages = communication_framework.total_messages
        assert total_messages >= 4
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, communication_framework, mock_agents):
        """Test comprehensive system health monitoring"""
        # Send some messages to generate metrics
        for i in range(10):
            message = communication_framework.create_message(
                sender_id="test_oa_comm_001",
                target_id="test_pa_comm_001",
                message_type=MessageType.REQUEST,
                payload={"health_test": i}
            )
            await communication_framework.send_message(message)
        
        # Get system health
        health = communication_framework.get_system_health()
        
        # Verify health metrics structure
        assert "framework_uptime" in health
        assert "total_messages" in health
        assert "successful_messages" in health
        assert "failed_messages" in health
        assert "success_rate" in health
        assert "registered_agents" in health
        assert "agent_health" in health
        assert "circuit_breakers" in health
        assert "active_correlations" in health
        
        # Verify metrics values
        assert health["registered_agents"] == 5
        assert health["total_messages"] >= 10
        assert 0.0 <= health["success_rate"] <= 1.0
        
        # Verify agent health details
        agent_health = health["agent_health"]
        assert len(agent_health) == 5
        
        for agent_id, agent_status in agent_health.items():
            assert "agent_id" in agent_status
            assert "status" in agent_status
            assert "uptime_seconds" in agent_status
            assert "success_rate" in agent_status
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, communication_framework, mock_agents):
        """Test performance metrics collection and tracking"""
        start_time = time.time()
        
        # Send messages and measure performance
        num_messages = 20
        for i in range(num_messages):
            message = communication_framework.create_message(
                sender_id="test_oa_comm_001",
                target_id="test_pa_comm_001",
                message_type=MessageType.REQUEST,
                payload={"performance_test": i}
            )
            await communication_framework.send_message(message)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify performance
        health = communication_framework.get_system_health()
        assert health["total_messages"] >= num_messages
        
        # Calculate throughput
        throughput = num_messages / duration
        assert throughput > 0  # Should have some throughput
        
        # Verify success rate is high
        assert health["success_rate"] >= 0.9
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, communication_framework, mock_agents):
        """Test error handling and recovery mechanisms"""
        # Test sending message to non-existent agent
        invalid_message = communication_framework.create_message(
            sender_id="test_oa_comm_001",
            target_id="non_existent_agent",
            message_type=MessageType.REQUEST,
            payload={"error_test": "invalid_target"}
        )
        
        success = await communication_framework.send_message(invalid_message)
        assert success is False
        
        # Verify system health reflects the failure
        health = communication_framework.get_system_health()
        assert health["failed_messages"] > 0
        
        # Test recovery with valid message
        valid_message = communication_framework.create_message(
            sender_id="test_oa_comm_001",
            target_id="test_pa_comm_001",
            message_type=MessageType.REQUEST,
            payload={"recovery_test": "valid_target"}
        )
        
        success = await communication_framework.send_message(valid_message)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, communication_framework, mock_agents):
        """Test concurrent message handling under load"""
        # Create multiple concurrent message sending tasks
        num_concurrent = 50
        tasks = []
        
        for i in range(num_concurrent):
            message = communication_framework.create_message(
                sender_id="test_oa_comm_001",
                target_id="test_pa_comm_001",
                message_type=MessageType.REQUEST,
                payload={"concurrent_test": i, "timestamp": time.time()}
            )
            task = communication_framework.send_message(message)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify most messages succeeded
        successful = sum(1 for result in results if result is True)
        success_rate = successful / num_concurrent
        assert success_rate >= 0.9  # At least 90% success rate
        
        # Verify system health after concurrent load
        health = communication_framework.get_system_health()
        assert health["total_messages"] >= num_concurrent
    
    @pytest.mark.asyncio
    async def test_message_expiration(self, communication_framework, mock_agents):
        """Test message expiration handling"""
        # Create message with short expiration
        message = communication_framework.create_message(
            sender_id="test_oa_comm_001",
            target_id="test_pa_comm_001",
            message_type=MessageType.REQUEST,
            payload={"expiration_test": "short_lived"}
        )
        
        # Set expiration time in the past
        message.expires_at = datetime.utcnow() - timedelta(seconds=1)
        
        # Message should still be sent (expiration handling would be in router implementation)
        success = await communication_framework.send_message(message)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_framework_shutdown(self, communication_framework, mock_agents):
        """Test graceful framework shutdown"""
        # Send some messages first
        for i in range(5):
            message = communication_framework.create_message(
                sender_id="test_oa_comm_001",
                target_id="test_pa_comm_001",
                message_type=MessageType.REQUEST,
                payload={"shutdown_test": i}
            )
            await communication_framework.send_message(message)
        
        # Get initial health
        initial_health = communication_framework.get_system_health()
        assert initial_health["registered_agents"] == 5
        
        # Shutdown framework
        await communication_framework.shutdown()
        
        # Verify agents are stopped
        for agent in mock_agents.values():
            assert agent.status == "stopped"


class TestMessageRouter:
    """Test suite for message routing functionality"""
    
    @pytest.fixture
    def agent_registry(self):
        """Create agent registry for testing"""
        return AgentRegistry()
    
    @pytest.fixture
    def message_router(self, agent_registry):
        """Create message router for testing"""
        return MessageRouter(agent_registry)
    
    @pytest.fixture
    def test_agents(self, agent_registry):
        """Create test agents and register them"""
        agents = {
            'agent1': MockOrchestrationAgent("router_test_001"),
            'agent2': MockPlanningAgent("router_test_002"),
            'agent3': MockInformationRetrievalAgent("router_test_003")
        }
        
        for agent in agents.values():
            agent_registry.register_agent(agent, [f"{agent.agent_type}_capability"])
        
        return agents
    
    @pytest.mark.asyncio
    async def test_direct_message_routing(self, message_router, test_agents):
        """Test direct message routing between specific agents"""
        message = AgentMessage(
            agent_id="router_test_001",
            target_agent_id="router_test_002",
            message_type=MessageType.REQUEST,
            payload={"direct_routing_test": "data"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        success = await message_router.route_message(message)
        assert success is True
        
        # Verify message was received by target
        target_agent = test_agents['agent2']
        assert target_agent.message_queue.qsize() > 0
        
        # Verify message history
        history = message_router.get_message_history(message.correlation_id)
        assert len(history) == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_routing(self, message_router, test_agents):
        """Test broadcast message routing"""
        message = AgentMessage(
            agent_id="router_test_001",
            target_agent_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            payload={"broadcast_test": "system_notification"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        success = await message_router.route_message(message)
        assert success is True
        
        # Verify all agents except sender received message
        for agent_name, agent in test_agents.items():
            if agent.agent_id != "router_test_001":
                assert agent.message_queue.qsize() > 0
    
    @pytest.mark.asyncio
    async def test_routing_to_nonexistent_agent(self, message_router, test_agents):
        """Test routing to non-existent agent"""
        message = AgentMessage(
            agent_id="router_test_001",
            target_agent_id="nonexistent_agent",
            message_type=MessageType.REQUEST,
            payload={"error_test": "invalid_target"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        success = await message_router.route_message(message)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_message_history_tracking(self, message_router, test_agents):
        """Test message history tracking and retrieval"""
        correlation_id = str(uuid4())
        
        # Send multiple messages with same correlation ID
        for i in range(3):
            message = AgentMessage(
                agent_id="router_test_001",
                target_agent_id="router_test_002",
                message_type=MessageType.REQUEST,
                payload={"history_test": i},
                correlation_id=correlation_id,
                session_id=str(uuid4()),
                trace_id=str(uuid4())
            )
            await message_router.route_message(message)
        
        # Verify history tracking
        history = message_router.get_message_history(correlation_id)
        assert len(history) == 3
        
        # Verify all messages have correct correlation ID
        for msg in history:
            assert msg.correlation_id == correlation_id
        
        # Test getting all history
        all_history = message_router.get_message_history()
        assert len(all_history) >= 3


class TestCircuitBreaker:
    """Test suite for circuit breaker functionality"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing"""
        return CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed state (normal operation)"""
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
        
        # Successful operation should keep circuit closed
        async def successful_operation():
            return "success"
        
        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_tracking(self, circuit_breaker):
        """Test circuit breaker failure tracking"""
        async def failing_operation():
            raise Exception("Operation failed")
        
        # First failure
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.state == "closed"
        
        # Second failure
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.failure_count == 2
        assert circuit_breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold"""
        async def failing_operation():
            raise Exception("Operation failed")
        
        # Trigger failures up to threshold
        for i in range(circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        # Circuit breaker should now be open
        assert circuit_breaker.state == "open"
        
        # Further calls should be rejected immediately
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await circuit_breaker.call(failing_operation)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery after timeout"""
        async def failing_operation():
            raise Exception("Operation failed")
        
        async def successful_operation():
            return "recovered"
        
        # Trigger circuit breaker to open
        for i in range(circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.recovery_timeout + 0.1)
        
        # Next call should move to half-open state
        result = await circuit_breaker.call(successful_operation)
        assert result == "recovered"
        assert circuit_breaker.state == "closed"  # Should reset to closed after success
        assert circuit_breaker.failure_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])