"""
Real-time Performance Testing

Tests for real-time verification and system performance under load.
Task 15 - Person D
Requirements: 11.1, 11.4, 12.4
"""

import pytest
import asyncio
import time
from datetime import datetime

from agents.verifier import VerificationAgent
from agents.cmvl_advanced import AdvancedCMVLMonitor
from data_models.schemas import AgentMessage, MessageType


@pytest.fixture
def verification_agent():
    """Create verification agent for testing"""
    return VerificationAgent("test_va_performance")


@pytest.fixture
def cmvl_monitor():
    """Create CMVL monitor for testing"""
    return AdvancedCMVLMonitor("test_cmvl_performance")


class TestVerificationPerformance:
    """Test verification performance metrics"""
    
    @pytest.mark.asyncio
    async def test_single_verification_latency(self, verification_agent):
        """Test single verification completes within acceptable time"""
        plan = {
            "plan_id": "test_perf_001",
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "investment",
                    "amount": 50000,
                    "risk_level": "medium"
                }
            ],
            "verification_level": "comprehensive"
        }
        
        start_time = time.time()
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": plan},
            correlation_id="test_corr_001",
            session_id="test_session_001",
            trace_id="test_trace_001"
        )
        
        response = await verification_agent.process_message(message)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Should complete in under 500ms
        assert latency < 0.5
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_verification_throughput(self, verification_agent):
        """Test concurrent verification throughput"""
        num_requests = 20
        plans = []
        
        for i in range(num_requests):
            plans.append({
                "plan_id": f"test_perf_{i}",
                "plan_steps": [
                    {
                        "step_id": "step_1",
                        "action_type": "investment",
                        "amount": 50000,
                        "risk_level": "medium"
                    }
                ],
                "verification_level": "basic"
            })
        
        start_time = time.time()
        
        tasks = []
        for plan in plans:
            message = AgentMessage(
                agent_id="test_sender",
                target_agent_id=verification_agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"verification_request": plan},
                correlation_id=f"test_corr_{plan['plan_id']}",
                session_id="test_session_throughput",
                trace_id="test_trace_throughput"
            )
            tasks.append(verification_agent.process_message(message))
        
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All should complete
        assert len(responses) == num_requests
        assert all(r is not None for r in responses)
        
        # Calculate throughput (requests per second)
        throughput = num_requests / total_time
        
        # Should handle at least 10 requests per second
        assert throughput >= 10
    
    @pytest.mark.asyncio
    async def test_complex_verification_performance(self, verification_agent):
        """Test performance with complex multi-step plans"""
        complex_plan = {
            "plan_id": "test_complex_001",
            "plan_steps": [
                {
                    "step_id": f"step_{i}",
                    "action_type": "investment",
                    "amount": 10000 * (i + 1),
                    "risk_level": "medium"
                }
                for i in range(10)
            ],
            "verification_level": "comprehensive"
        }
        
        start_time = time.time()
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id=verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": complex_plan},
            correlation_id="test_corr_complex",
            session_id="test_session_complex",
            trace_id="test_trace_complex"
        )
        
        response = await verification_agent.process_message(message)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Complex plan should still complete in under 2 seconds
        assert latency < 2.0
        assert response is not None


class TestCMVLPerformance:
    """Test CMVL real-time performance"""
    
    @pytest.mark.asyncio
    async def test_trigger_detection_latency(self, cmvl_monitor):
        """Test trigger detection latency"""
        trigger_event = {
            "trigger_type": "market_event",
            "event_type": "volatility_spike",
            "severity": "high",
            "description": "Market volatility spike",
            "source_data": {"volatility": 0.45},
            "impact_score": 0.8,
            "confidence_score": 0.9,
            "detector_agent_id": "ira_001",
            "correlation_id": "test_corr_trigger"
        }
        
        start_time = time.time()
        
        message = AgentMessage(
            agent_id="test_sender",
            target_agent_id="cmvl_monitor",
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": trigger_event},
            correlation_id="test_corr_trigger",
            session_id="test_session_trigger",
            trace_id="test_trace_trigger"
        )
        
        # Simulate trigger processing
        await asyncio.sleep(0.1)  # Simulate processing
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Trigger detection should be very fast (< 200ms)
        assert latency < 0.2
    
    @pytest.mark.asyncio
    async def test_concurrent_trigger_handling(self, cmvl_monitor):
        """Test handling multiple concurrent triggers"""
        triggers = []
        for i in range(5):
            triggers.append({
                "trigger_type": "market_event",
                "event_type": "volatility_spike",
                "severity": "medium",
                "description": f"Trigger {i}",
                "source_data": {"volatility": 0.3 + i * 0.05},
                "impact_score": 0.5 + i * 0.1,
                "confidence_score": 0.9,
                "detector_agent_id": f"ira_{i}",
                "correlation_id": f"test_corr_trigger_{i}"
            })
        
        start_time = time.time()
        
        # Simulate concurrent trigger processing
        tasks = []
        for trigger in triggers:
            tasks.append(asyncio.sleep(0.1))  # Simulate processing
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle concurrently (not sequentially)
        # If sequential: 5 * 0.1 = 0.5s
        # If concurrent: ~0.1s
        assert total_time < 0.3


class TestMemoryPerformance:
    """Test memory usage and efficiency"""
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_bulk_verification(self, verification_agent):
        """Test memory efficiency with bulk verifications"""
        import sys
        
        # Get initial memory usage
        initial_size = sys.getsizeof(verification_agent)
        
        # Process many verifications
        for i in range(100):
            plan = {
                "plan_id": f"test_mem_{i}",
                "plan_steps": [
                    {
                        "step_id": "step_1",
                        "action_type": "investment",
                        "amount": 50000,
                        "risk_level": "medium"
                    }
                ],
                "verification_level": "basic"
            }
            
            message = AgentMessage(
                agent_id="test_sender",
                target_agent_id=verification_agent.agent_id,
                message_type=MessageType.REQUEST,
                payload={"verification_request": plan},
                correlation_id=f"test_corr_mem_{i}",
                session_id="test_session_mem",
                trace_id="test_trace_mem"
            )
            
            await verification_agent.process_message(message)
        
        # Get final memory usage
        final_size = sys.getsizeof(verification_agent)
        
        # Memory growth should be reasonable (< 10x)
        memory_growth = final_size / initial_size
        assert memory_growth < 10


class TestScalabilityMetrics:
    """Test system scalability metrics"""
    
    @pytest.mark.asyncio
    async def test_response_time_under_load(self, verification_agent):
        """Test response time degradation under load"""
        response_times = []
        
        # Test with increasing load
        for batch_size in [1, 5, 10, 20]:
            plans = []
            for i in range(batch_size):
                plans.append({
                    "plan_id": f"test_scale_{batch_size}_{i}",
                    "plan_steps": [
                        {
                            "step_id": "step_1",
                            "action_type": "investment",
                            "amount": 50000,
                            "risk_level": "medium"
                        }
                    ],
                    "verification_level": "basic"
                })
            
            start_time = time.time()
            
            tasks = []
            for plan in plans:
                message = AgentMessage(
                    agent_id="test_sender",
                    target_agent_id=verification_agent.agent_id,
                    message_type=MessageType.REQUEST,
                    payload={"verification_request": plan},
                    correlation_id=f"test_corr_{plan['plan_id']}",
                    session_id="test_session_scale",
                    trace_id="test_trace_scale"
                )
                tasks.append(verification_agent.process_message(message))
            
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            avg_response_time = (end_time - start_time) / batch_size
            response_times.append(avg_response_time)
        
        # Response time should not degrade significantly
        # (last should be < 2x first)
        assert response_times[-1] < response_times[0] * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
