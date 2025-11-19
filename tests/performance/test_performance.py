"""
Performance Testing Framework for FinPilot VP-MAS

Comprehensive performance tests and benchmarks for all VP-MAS components
including agent communication, data processing, and system throughput.

Requirements: 11.3
"""

import asyncio
import time
import pytest
from typing import List, Dict, Any
from uuid import uuid4

from agents.mock_interfaces import (
    MockOrchestrationAgent, MockPlanningAgent, MockInformationRetrievalAgent,
    MockVerificationAgent, MockExecutionAgent
)
from agents.communication import AgentCommunicationFramework
from data_models.schemas import AgentMessage, MessageType
from tests.mock_data import MockDataGenerator


class TestPerformanceBenchmarks:
    """Performance benchmark tests for VP-MAS components"""
    
    @pytest.fixture
    def mock_data_generator(self):
        """Create mock data generator with fixed seed"""
        return MockDataGenerator(seed=42)
    
    @pytest.fixture
    async def performance_system(self):
        """Set up system for performance testing"""
        framework = AgentCommunicationFramework()
        
        agents = {
            'orchestrator': MockOrchestrationAgent("perf_oa_001"),
            'planner': MockPlanningAgent("perf_pa_001"),
            'ira': MockInformationRetrievalAgent("perf_ira_001"),
            'verifier': MockVerificationAgent("perf_va_001"),
            'executor': MockExecutionAgent("perf_ea_001")
        }
        
        # Start and register all agents
        for agent_name, agent in agents.items():
            await agent.start()
            framework.register_agent(agent, [f"{agent_name}_capabilities"])
        
        yield framework, agents
        
        # Cleanup
        for agent in agents.values():
            await agent.stop()
    
    @pytest.mark.benchmark
    def test_data_model_creation_performance(self, benchmark, mock_data_generator):
        """Benchmark data model creation performance"""
        
        def create_models():
            request = mock_data_generator.generate_enhanced_plan_request()
            market_data = mock_data_generator.generate_market_data()
            trigger = mock_data_generator.generate_trigger_event()
            return request, market_data, trigger
        
        result = benchmark(create_models)
        
        # Verify models were created successfully
        assert result[0] is not None
        assert result[1] is not None
        assert result[2] is not None
    
    @pytest.mark.benchmark
    def test_data_validation_performance(self, benchmark, mock_data_generator):
        """Benchmark data validation performance"""
        
        def validate_models():
            # Generate and validate multiple models
            models = []
            for _ in range(10):
                request = mock_data_generator.generate_enhanced_plan_request()
                models.append(request)
            return models
        
        result = benchmark(validate_models)
        assert len(result) == 10
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_agent_communication_performance(self, benchmark, performance_system):
        """Benchmark agent communication performance"""
        framework, agents = performance_system
        
        async def send_messages():
            messages_sent = 0
            for i in range(50):
                message = framework.create_message(
                    sender_id=agents['orchestrator'].agent_id,
                    target_id=agents['planner'].agent_id,
                    message_type=MessageType.REQUEST,
                    payload={"test_message": i}
                )
                
                success = await framework.send_message(message)
                if success:
                    messages_sent += 1
            
            return messages_sent
        
        result = await benchmark(send_messages)
        assert result >= 45  # Allow for some failures
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_planning_agent_performance(self, benchmark, performance_system):
        """Benchmark planning agent performance"""
        framework, agents = performance_system
        planner = agents['planner']
        
        async def generate_plan():
            message = framework.create_message(
                sender_id="perf_test",
                target_id=planner.agent_id,
                message_type=MessageType.REQUEST,
                payload={
                    "planning_request": {
                        "user_goal": "Performance test goal",
                        "time_horizon": 60
                    }
                }
            )
            
            response = await planner.process_message(message)
            return response
        
        result = await benchmark(generate_plan)
        assert result is not None
        assert result.payload["plan_generated"] is True
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_verification_agent_performance(self, benchmark, performance_system):
        """Benchmark verification agent performance"""
        framework, agents = performance_system
        verifier = agents['verifier']
        
        async def verify_plan():
            message = framework.create_message(
                sender_id="perf_test",
                target_id=verifier.agent_id,
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
                }
            )
            
            response = await verifier.process_message(message)
            return response
        
        result = await benchmark(verify_plan)
        assert result is not None
        assert "verification_report" in result.payload
    
    @pytest.mark.benchmark
    def test_mock_data_generation_throughput(self, benchmark, mock_data_generator):
        """Benchmark mock data generation throughput"""
        
        def generate_bulk_data():
            data_objects = []
            
            # Generate various types of data
            for _ in range(20):
                data_objects.append(mock_data_generator.generate_enhanced_plan_request())
                data_objects.append(mock_data_generator.generate_market_data())
                data_objects.append(mock_data_generator.generate_trigger_event())
                data_objects.extend(mock_data_generator.generate_plan_steps())
            
            return data_objects
        
        result = benchmark(generate_bulk_data)
        assert len(result) > 100  # Should generate many objects
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_system_throughput_under_load(self, benchmark, performance_system):
        """Benchmark system throughput under concurrent load"""
        framework, agents = performance_system
        
        async def concurrent_operations():
            tasks = []
            
            # Create concurrent tasks for different operations
            for i in range(20):
                # Planning requests
                planning_message = framework.create_message(
                    sender_id=f"load_test_{i}",
                    target_id=agents['planner'].agent_id,
                    message_type=MessageType.REQUEST,
                    payload={
                        "planning_request": {
                            "user_goal": f"Load test goal {i}",
                            "time_horizon": 60
                        }
                    }
                )
                tasks.append(framework.send_message(planning_message))
                
                # Market data requests
                market_message = framework.create_message(
                    sender_id=f"load_test_{i}",
                    target_id=agents['ira'].agent_id,
                    message_type=MessageType.REQUEST,
                    payload={"market_data_request": {"symbols": ["SPY"]}}
                )
                tasks.append(framework.send_message(market_message))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful operations
            successful = sum(1 for result in results if result is True)
            return successful
        
        result = await benchmark(concurrent_operations)
        assert result >= 30  # Should handle most concurrent operations
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, performance_system):
        """Test memory usage under sustained load"""
        import psutil
        import os
        
        framework, agents = performance_system
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate sustained load
        for batch in range(10):
            tasks = []
            for i in range(50):
                message = framework.create_message(
                    sender_id=f"memory_test_{batch}_{i}",
                    target_id=agents['planner'].agent_id,
                    message_type=MessageType.REQUEST,
                    payload={"planning_request": {"user_goal": f"Memory test {i}"}}
                )
                tasks.append(framework.send_message(message))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Allow some processing time
            await asyncio.sleep(0.1)
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
        
        print(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (+{memory_increase:.2f}MB)")
    
    @pytest.mark.benchmark
    def test_serialization_performance(self, benchmark, mock_data_generator):
        """Benchmark data serialization/deserialization performance"""
        
        def serialize_deserialize():
            # Generate complex data
            request = mock_data_generator.generate_enhanced_plan_request()
            market_data = mock_data_generator.generate_market_data()
            
            # Serialize to JSON
            request_json = request.json()
            market_json = market_data.json()
            
            # Deserialize from JSON
            from data_models.schemas import EnhancedPlanRequest, MarketData
            request_restored = EnhancedPlanRequest.parse_raw(request_json)
            market_restored = MarketData.parse_raw(market_json)
            
            return request_restored, market_restored
        
        result = benchmark(serialize_deserialize)
        assert result[0] is not None
        assert result[1] is not None
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_performance(self, benchmark, performance_system):
        """Benchmark complete end-to-end workflow performance"""
        framework, agents = performance_system
        
        async def complete_workflow():
            correlation_id = str(uuid4())
            session_id = str(uuid4())
            
            # Step 1: Submit goal to orchestrator
            goal_message = framework.create_message(
                sender_id="perf_test_user",
                target_id=agents['orchestrator'].agent_id,
                message_type=MessageType.REQUEST,
                payload={"user_goal": "Complete workflow performance test"},
                correlation_id=correlation_id,
                session_id=session_id
            )
            
            await framework.send_message(goal_message)
            
            # Step 2: Generate plan
            planning_message = framework.create_message(
                sender_id=agents['orchestrator'].agent_id,
                target_id=agents['planner'].agent_id,
                message_type=MessageType.REQUEST,
                payload={
                    "planning_request": {
                        "user_goal": "Complete workflow performance test",
                        "time_horizon": 60
                    }
                },
                correlation_id=correlation_id,
                session_id=session_id
            )
            
            plan_response = await agents['planner'].process_message(planning_message)
            
            # Step 3: Verify plan
            verification_message = framework.create_message(
                sender_id=agents['planner'].agent_id,
                target_id=agents['verifier'].agent_id,
                message_type=MessageType.REQUEST,
                payload={
                    "verification_request": {
                        "plan_id": str(uuid4()),
                        "plan_steps": plan_response.payload.get("plan_steps", [])
                    }
                },
                correlation_id=correlation_id,
                session_id=session_id
            )
            
            verification_response = await agents['verifier'].process_message(verification_message)
            
            # Step 4: Execute plan
            execution_message = framework.create_message(
                sender_id=agents['verifier'].agent_id,
                target_id=agents['executor'].agent_id,
                message_type=MessageType.REQUEST,
                payload={
                    "execution_request": {
                        "plan_steps": plan_response.payload.get("plan_steps", [])
                    }
                },
                correlation_id=correlation_id,
                session_id=session_id
            )
            
            execution_response = await agents['executor'].process_message(execution_message)
            
            # Verify workflow completion
            trace = framework.get_correlation_trace(correlation_id)
            
            return {
                "workflow_completed": True,
                "steps_executed": 4,
                "correlation_trace_length": len(trace),
                "final_execution_status": execution_response.payload.get("execution_completed", False)
            }
        
        result = await benchmark(complete_workflow)
        assert result["workflow_completed"] is True
        assert result["steps_executed"] == 4
        assert result["final_execution_status"] is True


# Utility functions for performance analysis
def analyze_performance_results(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results and provide performance insights"""
    
    analysis = {
        "summary": {
            "total_tests": len(benchmark_results),
            "average_execution_time": 0,
            "performance_grade": "A"
        },
        "bottlenecks": [],
        "recommendations": []
    }
    
    # Calculate average execution time
    total_time = sum(result.get("execution_time", 0) for result in benchmark_results.values())
    analysis["summary"]["average_execution_time"] = total_time / len(benchmark_results)
    
    # Identify bottlenecks
    for test_name, result in benchmark_results.items():
        execution_time = result.get("execution_time", 0)
        if execution_time > 5.0:  # More than 5 seconds
            analysis["bottlenecks"].append({
                "test": test_name,
                "execution_time": execution_time,
                "severity": "high" if execution_time > 10.0 else "medium"
            })
    
    # Generate recommendations
    if analysis["bottlenecks"]:
        analysis["recommendations"].append("Consider optimizing slow operations")
    
    if analysis["summary"]["average_execution_time"] > 2.0:
        analysis["recommendations"].append("Overall system performance could be improved")
        analysis["summary"]["performance_grade"] = "B"
    
    if analysis["summary"]["average_execution_time"] > 5.0:
        analysis["summary"]["performance_grade"] = "C"
    
    return analysis


# Performance test runner
if __name__ == "__main__":
    import pytest
    
    # Run performance tests with benchmark plugin
    pytest.main([
        __file__,
        "-v",
        "--benchmark-only",
        "--benchmark-json=benchmark_results.json",
        "--benchmark-sort=mean"
    ])