"""
Integration and Performance Testing for Agent Communication Framework

Comprehensive tests for system integration, performance benchmarks,
and stress testing of the multi-agent communication system.

Requirements: 6.1, 6.2, 6.5, 11.1, 11.2, 2.1, 2.2, 10.2
"""

import asyncio
import pytest
import time
import statistics
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, List, Any
import json

from agents.base_agent import BaseAgent
from agents.communication import AgentCommunicationFramework
from agents.mock_interfaces import (
    MockOrchestrationAgent, MockPlanningAgent, MockInformationRetrievalAgent,
    MockVerificationAgent, MockExecutionAgent
)
from agents.workflow_engine import WorkflowEngine, WorkflowPriority
from data_models.schemas import (
    AgentMessage, MessageType, Priority, TriggerEvent, SeverityLevel,
    MarketEventType, PerformanceMetrics
)


class PerformanceTestAgent(BaseAgent):
    """Specialized agent for performance testing"""
    
    def __init__(self, agent_id: str, response_delay: float = 0.01):
        super().__init__(agent_id, "performance_test")
        self.response_delay = response_delay
        self.messages_processed = 0
        self.processing_times = []
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process message with configurable delay"""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(self.response_delay)
        
        self.messages_processed += 1
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Return response for request messages
        if message.message_type == MessageType.REQUEST:
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "status": "processed",
                    "processing_time": processing_time,
                    "message_count": self.messages_processed
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
        
        return None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            "messages_processed": self.messages_processed,
            "avg_processing_time": statistics.mean(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "median_processing_time": statistics.median(self.processing_times),
            "std_dev_processing_time": statistics.stdev(self.processing_times) if len(self.processing_times) > 1 else 0.0
        }


class TestIntegrationFramework:
    """Integration tests for the complete agent communication framework"""
    
    @pytest.fixture
    def full_agent_system(self):
        """Create complete agent system for integration testing"""
        framework = AgentCommunicationFramework()
        
        # Create all agent types
        agents = {
            'orchestrator': MockOrchestrationAgent("integration_oa_001"),
            'planner': MockPlanningAgent("integration_pa_001"),
            'ira': MockInformationRetrievalAgent("integration_ira_001"),
            'verifier': MockVerificationAgent("integration_va_001"),
            'executor': MockExecutionAgent("integration_ea_001")
        }
        
        # Register agents with capabilities
        capabilities = {
            'orchestrator': ['workflow_coordination', 'goal_parsing', 'task_delegation'],
            'planner': ['financial_planning', 'strategy_optimization', 'constraint_solving'],
            'ira': ['market_data', 'regulatory_data', 'external_apis'],
            'verifier': ['constraint_validation', 'compliance_checking', 'risk_assessment'],
            'executor': ['portfolio_updates', 'transaction_execution', 'ledger_management']
        }
        
        for agent_name, agent in agents.items():
            framework.register_agent(agent, capabilities[agent_name])
        
        return {
            'framework': framework,
            'agents': agents,
            'workflow_engine': WorkflowEngine(framework)
        }
    
    @pytest.mark.asyncio
    async def test_complete_financial_planning_workflow(self, full_agent_system):
        """Test complete financial planning workflow integration"""
        framework = full_agent_system['framework']
        agents = full_agent_system['agents']
        workflow_engine = full_agent_system['workflow_engine']
        
        # Create financial planning workflow
        workflow_id = await workflow_engine.create_workflow(
            workflow_type="financial_planning",
            user_id="test_user_001",
            parameters={
                "goal_text": "Save $100,000 for retirement in 10 years",
                "planning_request": {
                    "user_goal": "retirement_planning",
                    "time_horizon": 120,  # 10 years
                    "risk_tolerance": "moderate"
                }
            },
            priority=WorkflowPriority.HIGH
        )
        
        # Start workflow execution
        success = await workflow_engine.start_workflow(workflow_id)
        assert success is True
        
        # Wait for workflow to process
        await asyncio.sleep(1.0)
        
        # Check workflow status
        status = workflow_engine.get_workflow_status(workflow_id)
        assert status is not None
        assert status["workflow_id"] == workflow_id
        assert status["workflow_type"] == "financial_planning"
        
        # Verify system health after workflow
        health = framework.get_system_health()
        assert health["success_rate"] > 0.8
        assert health["registered_agents"] == 5
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, full_agent_system):
        """Test coordination between multiple agents"""
        framework = full_agent_system['framework']
        agents = full_agent_system['agents']
        
        # Simulate orchestrator coordinating with all agents
        orchestrator_id = "integration_oa_001"
        correlation_id = str(uuid4())
        
        # Send coordination messages to all other agents
        coordination_tasks = []
        for agent_name, agent in agents.items():
            if agent_name != 'orchestrator':
                message = framework.create_message(
                    sender_id=orchestrator_id,
                    target_id=agent.agent_id,
                    message_type=MessageType.REQUEST,
                    payload={
                        "coordination_request": True,
                        "task": f"coordinate_with_{agent_name}",
                        "priority": "high"
                    },
                    correlation_id=correlation_id,
                    priority=Priority.HIGH
                )
                coordination_tasks.append(framework.send_message(message))
        
        # Execute all coordination messages
        results = await asyncio.gather(*coordination_tasks)
        assert all(results), "All coordination messages should succeed"
        
        # Verify all agents received messages
        for agent_name, agent in agents.items():
            if agent_name != 'orchestrator':
                assert agent.message_queue.qsize() > 0, f"Agent {agent_name} should have received message"
        
        # Process messages and verify responses
        response_tasks = []
        for agent_name, agent in agents.items():
            if agent_name != 'orchestrator' and agent.message_queue.qsize() > 0:
                message = await agent.message_queue.get()
                response_tasks.append(agent.process_message(message))
        
        responses = await asyncio.gather(*response_tasks, return_exceptions=True)
        successful_responses = [r for r in responses if isinstance(r, AgentMessage)]
        
        assert len(successful_responses) >= 3, "Should receive responses from most agents"
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, full_agent_system):
        """Test error propagation and recovery mechanisms"""
        framework = full_agent_system['framework']
        agents = full_agent_system['agents']
        
        # Send message to non-existent agent
        error_message = framework.create_message(
            sender_id="integration_oa_001",
            target_id="non_existent_agent",
            message_type=MessageType.REQUEST,
            payload={"test": "error_handling"},
            priority=Priority.MEDIUM
        )
        
        success = await framework.send_message(error_message)
        assert success is False, "Message to non-existent agent should fail"
        
        # Verify system health reflects the error
        health = framework.get_system_health()
        assert health["failed_messages"] > 0
        
        # Send valid message to verify recovery
        recovery_message = framework.create_message(
            sender_id="integration_oa_001",
            target_id="integration_pa_001",
            message_type=MessageType.REQUEST,
            payload={"test": "recovery"},
            priority=Priority.MEDIUM
        )
        
        success = await framework.send_message(recovery_message)
        assert success is True, "Valid message should succeed after error"
    
    @pytest.mark.asyncio
    async def test_system_scalability(self, full_agent_system):
        """Test system scalability with additional agents"""
        framework = full_agent_system['framework']
        
        # Add additional performance test agents
        additional_agents = []
        for i in range(10):
            agent = PerformanceTestAgent(f"perf_agent_{i:03d}", response_delay=0.005)
            additional_agents.append(agent)
            framework.register_agent(agent, [f"performance_testing_{i}"])
        
        # Verify all agents are registered
        health = framework.get_system_health()
        assert health["registered_agents"] >= 15  # 5 original + 10 additional
        
        # Send messages to all additional agents
        message_tasks = []
        for agent in additional_agents:
            message = framework.create_message(
                sender_id="integration_oa_001",
                target_id=agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"scalability_test": True, "timestamp": time.time()},
                priority=Priority.MEDIUM
            )
            message_tasks.append(framework.send_message(message))
        
        # Execute all messages concurrently
        results = await asyncio.gather(*message_tasks)
        assert all(results), "All messages to additional agents should succeed"
        
        # Verify system performance under increased load
        final_health = framework.get_system_health()
        assert final_health["success_rate"] > 0.9


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for the communication framework"""
    
    @pytest.fixture
    def performance_framework(self):
        """Create framework optimized for performance testing"""
        framework = AgentCommunicationFramework()
        
        # Create performance test agents with different response times
        agents = []
        response_times = [0.001, 0.005, 0.01, 0.02, 0.05]  # Various response delays
        
        for i, delay in enumerate(response_times):
            agent = PerformanceTestAgent(f"perf_agent_{i}", response_delay=delay)
            agents.append(agent)
            framework.register_agent(agent, [f"performance_capability_{i}"])
        
        return {'framework': framework, 'agents': agents}
    
    @pytest.mark.asyncio
    async def test_message_throughput_benchmark(self, performance_framework):
        """Benchmark message throughput under various loads"""
        framework = performance_framework['framework']
        agents = performance_framework['agents']
        
        # Test different message volumes
        test_volumes = [10, 50, 100, 200]
        throughput_results = {}
        
        for volume in test_volumes:
            start_time = time.time()
            
            # Send messages to all agents
            tasks = []
            for i in range(volume):
                target_agent = agents[i % len(agents)]
                message = framework.create_message(
                    sender_id="throughput_tester",
                    target_id=target_agent.agent_id,
                    message_type=MessageType.REQUEST,
                    payload={"throughput_test": True, "message_id": i},
                    priority=Priority.MEDIUM
                )
                tasks.append(framework.send_message(message))
            
            # Execute all messages
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Calculate throughput
            duration = end_time - start_time
            throughput = volume / duration
            success_rate = sum(1 for r in results if r) / len(results)
            
            throughput_results[volume] = {
                'throughput': throughput,
                'duration': duration,
                'success_rate': success_rate
            }
            
            # Verify performance requirements
            assert success_rate >= 0.95, f"Success rate should be >= 95% for {volume} messages"
            assert throughput > 50, f"Throughput should be > 50 msg/sec for {volume} messages"
        
        # Verify throughput scales reasonably
        assert throughput_results[200]['throughput'] > throughput_results[10]['throughput'] * 0.5
    
    @pytest.mark.asyncio
    async def test_latency_benchmark(self, performance_framework):
        """Benchmark message latency and response times"""
        framework = performance_framework['framework']
        agents = performance_framework['agents']
        
        # Measure round-trip latency
        latencies = []
        num_samples = 50
        
        for i in range(num_samples):
            start_time = time.time()
            
            # Send message to fastest agent
            target_agent = agents[0]  # Fastest response time
            message = framework.create_message(
                sender_id="latency_tester",
                target_id=target_agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"latency_test": True, "sample_id": i},
                priority=Priority.HIGH
            )
            
            success = await framework.send_message(message)
            assert success is True
            
            # Wait for processing (simplified - in real system would wait for response)
            await asyncio.sleep(0.002)  # Simulate response wait
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        
        # Calculate latency statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        
        # Verify latency requirements
        assert avg_latency < 50, f"Average latency should be < 50ms, got {avg_latency:.2f}ms"
        assert p95_latency < 100, f"P95 latency should be < 100ms, got {p95_latency:.2f}ms"
        assert p99_latency < 200, f"P99 latency should be < 200ms, got {p99_latency:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_load_benchmark(self, performance_framework):
        """Benchmark system under high concurrent load"""
        framework = performance_framework['framework']
        agents = performance_framework['agents']
        
        # Test concurrent message processing
        concurrent_levels = [5, 10, 20, 50]
        load_results = {}
        
        for concurrency in concurrent_levels:
            start_time = time.time()
            
            # Create concurrent message batches
            batch_tasks = []
            messages_per_batch = 10
            
            for batch in range(concurrency):
                batch_messages = []
                for msg in range(messages_per_batch):
                    target_agent = agents[msg % len(agents)]
                    message = framework.create_message(
                        sender_id="load_tester",
                        target_id=target_agent.agent_id,
                        message_type=MessageType.REQUEST,
                        payload={
                            "load_test": True,
                            "batch_id": batch,
                            "message_id": msg
                        },
                        priority=Priority.MEDIUM
                    )
                    batch_messages.append(framework.send_message(message))
                
                # Add batch as concurrent task
                batch_task = asyncio.gather(*batch_messages)
                batch_tasks.append(batch_task)
            
            # Execute all batches concurrently
            batch_results = await asyncio.gather(*batch_tasks)
            end_time = time.time()
            
            # Calculate metrics
            total_messages = concurrency * messages_per_batch
            duration = end_time - start_time
            throughput = total_messages / duration
            
            # Count successful messages
            successful_messages = 0
            for batch_result in batch_results:
                successful_messages += sum(1 for r in batch_result if r)
            
            success_rate = successful_messages / total_messages
            
            load_results[concurrency] = {
                'throughput': throughput,
                'success_rate': success_rate,
                'duration': duration,
                'total_messages': total_messages
            }
            
            # Verify performance under load
            assert success_rate >= 0.9, f"Success rate should be >= 90% at concurrency {concurrency}"
            assert throughput > 20, f"Throughput should be > 20 msg/sec at concurrency {concurrency}"
        
        # Verify system handles increasing load gracefully
        max_concurrency_result = load_results[max(concurrent_levels)]
        assert max_concurrency_result['success_rate'] >= 0.85, "System should maintain 85%+ success rate under max load"
    
    @pytest.mark.asyncio
    async def test_memory_and_resource_usage(self, performance_framework):
        """Test memory usage and resource consumption"""
        framework = performance_framework['framework']
        agents = performance_framework['agents']
        
        # Get initial system state
        initial_health = framework.get_system_health()
        
        # Generate sustained load
        num_messages = 1000
        batch_size = 50
        
        for batch_start in range(0, num_messages, batch_size):
            batch_tasks = []
            
            for i in range(batch_size):
                if batch_start + i >= num_messages:
                    break
                
                target_agent = agents[i % len(agents)]
                message = framework.create_message(
                    sender_id="resource_tester",
                    target_id=target_agent.agent_id,
                    message_type=MessageType.REQUEST,
                    payload={
                        "resource_test": True,
                        "message_number": batch_start + i,
                        "large_payload": "x" * 1000  # 1KB payload
                    },
                    priority=Priority.MEDIUM
                )
                batch_tasks.append(framework.send_message(message))
            
            # Execute batch
            await asyncio.gather(*batch_tasks)
            
            # Small delay between batches
            await asyncio.sleep(0.01)
        
        # Get final system state
        final_health = framework.get_system_health()
        
        # Verify resource usage is reasonable
        assert final_health["total_messages"] >= num_messages
        assert final_health["success_rate"] >= 0.9
        
        # Verify no significant degradation
        message_increase = final_health["total_messages"] - initial_health["total_messages"]
        assert message_increase >= num_messages * 0.9  # At least 90% of messages processed
    
    def test_agent_performance_statistics(self, performance_framework):
        """Test individual agent performance statistics"""
        agents = performance_framework['agents']
        
        # Verify all agents have performance tracking
        for agent in agents:
            health = agent.get_health_status()
            
            assert "success_rate" in health
            assert "uptime_seconds" in health
            assert health["agent_id"] == agent.agent_id
            assert health["status"] in ["initializing", "running", "stopped"]
            
            # Test performance metrics
            metrics = agent.get_performance_metrics()
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.success_rate >= 0.0
            assert metrics.error_count >= 0


class TestStressAndReliability:
    """Stress testing and reliability validation"""
    
    @pytest.fixture
    def stress_test_system(self):
        """Create system for stress testing"""
        framework = AgentCommunicationFramework()
        
        # Create agents with various characteristics
        agents = []
        
        # Fast agents
        for i in range(5):
            agent = PerformanceTestAgent(f"fast_agent_{i}", response_delay=0.001)
            agents.append(agent)
            framework.register_agent(agent, ["fast_processing"])
        
        # Medium agents
        for i in range(5):
            agent = PerformanceTestAgent(f"medium_agent_{i}", response_delay=0.01)
            agents.append(agent)
            framework.register_agent(agent, ["medium_processing"])
        
        # Slow agents
        for i in range(3):
            agent = PerformanceTestAgent(f"slow_agent_{i}", response_delay=0.05)
            agents.append(agent)
            framework.register_agent(agent, ["slow_processing"])
        
        return {'framework': framework, 'agents': agents}
    
    @pytest.mark.asyncio
    async def test_sustained_high_load(self, stress_test_system):
        """Test system under sustained high load"""
        framework = stress_test_system['framework']
        agents = stress_test_system['agents']
        
        # Run sustained load for extended period
        duration_seconds = 10
        messages_per_second = 100
        total_messages = duration_seconds * messages_per_second
        
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Send batch of messages
            batch_size = min(20, total_messages - message_count)
            if batch_size <= 0:
                break
            
            batch_tasks = []
            for i in range(batch_size):
                target_agent = agents[message_count % len(agents)]
                message = framework.create_message(
                    sender_id="stress_tester",
                    target_id=target_agent.agent_id,
                    message_type=MessageType.REQUEST,
                    payload={
                        "stress_test": True,
                        "message_number": message_count,
                        "timestamp": time.time()
                    },
                    priority=Priority.MEDIUM
                )
                batch_tasks.append(framework.send_message(message))
                message_count += 1
            
            # Execute batch
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Brief pause to maintain target rate
            await asyncio.sleep(0.01)
        
        # Verify system health after stress test
        health = framework.get_system_health()
        assert health["success_rate"] >= 0.8, "System should maintain 80%+ success rate under stress"
        assert health["total_messages"] >= total_messages * 0.8, "Should process 80%+ of messages"
    
    @pytest.mark.asyncio
    async def test_failure_recovery(self, stress_test_system):
        """Test system recovery from various failure scenarios"""
        framework = stress_test_system['framework']
        agents = stress_test_system['agents']
        
        # Test 1: Circuit breaker activation and recovery
        target_agent = agents[0]
        circuit_breaker = framework.circuit_breakers.get(target_agent.agent_id)
        
        if circuit_breaker:
            # Force circuit breaker to open by simulating failures
            for _ in range(circuit_breaker.failure_threshold):
                try:
                    await circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Simulated failure")))
                except:
                    pass
            
            assert circuit_breaker.state == "open"
            
            # Wait for recovery timeout
            await asyncio.sleep(circuit_breaker.recovery_timeout + 0.1)
            
            # Test recovery
            message = framework.create_message(
                sender_id="recovery_tester",
                target_id=target_agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"recovery_test": True},
                priority=Priority.MEDIUM
            )
            
            success = await framework.send_message(message)
            # Circuit breaker should allow the call and potentially recover
        
        # Test 2: System recovery after errors
        initial_health = framework.get_system_health()
        
        # Send messages to non-existent agents (should fail gracefully)
        error_tasks = []
        for i in range(10):
            message = framework.create_message(
                sender_id="recovery_tester",
                target_id=f"non_existent_agent_{i}",
                message_type=MessageType.REQUEST,
                payload={"error_test": True},
                priority=Priority.MEDIUM
            )
            error_tasks.append(framework.send_message(message))
        
        error_results = await asyncio.gather(*error_tasks, return_exceptions=True)
        
        # Send valid messages to verify system recovery
        recovery_tasks = []
        for agent in agents[:5]:
            message = framework.create_message(
                sender_id="recovery_tester",
                target_id=agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"recovery_verification": True},
                priority=Priority.MEDIUM
            )
            recovery_tasks.append(framework.send_message(message))
        
        recovery_results = await asyncio.gather(*recovery_tasks)
        
        # Verify system recovered
        assert all(isinstance(r, bool) for r in recovery_results), "Recovery messages should return boolean results"
        recovery_success_rate = sum(1 for r in recovery_results if r) / len(recovery_results)
        assert recovery_success_rate >= 0.9, "System should recover with 90%+ success rate"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])