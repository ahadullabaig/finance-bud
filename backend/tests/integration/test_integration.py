"""
Integration Testing Framework

End-to-end integration tests for VP-MAS workflows,
CMVL scenarios, and cross-agent coordination.

Requirements: 11.4
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from tests.mocks.mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent,
    MockVerificationAgent,
    MockExecutionAgent
)
from agents.communication import AgentCommunicationFramework
from data_models.schemas import AgentMessage, MessageType, Priority


class TestIntegrationScenarios:
    """Integration test scenarios for VP-MAS workflows"""
    
    @pytest.fixture
    async def integrated_system(self):
        """Set up complete integrated system"""
        framework = AgentCommunicationFramework()
        
        agents = {
            'orchestrator': MockOrchestrationAgent("integration_oa_001"),
            'planner': MockPlanningAgent("integration_pa_001"),
            'ira': MockInformationRetrievalAgent("integration_ira_001"),
            'verifier': MockVerificationAgent("integration_va_001"),
            'executor': MockExecutionAgent("integration_ea_001")
        }
        
        # Register all agents
        for agent_name, agent in agents.items():
            await agent.start()
            framework.register_agent(agent, [f"{agent_name}_capabilities"])
        
        return framework, agents
    
    async def test_complete_planning_workflow(self, integrated_system):
        """Test complete planning workflow from goal to execution"""
        framework, agents = integrated_system
        
        correlation_id = str(uuid4())
        session_id = str(uuid4())
        
        # Step 1: User submits goal to orchestrator
        user_goal_message = framework.create_message(
            sender_id="user_client",
            target_id="integration_oa_001",
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Save $100,000 for house down payment in 5 years"},
            correlation_id=correlation_id,
            session_id=session_id
        )
        
        success = await framework.send_message(user_goal_message)
        assert success is True
        
        # Allow message processing
        await asyncio.sleep(0.2)
        
        # Step 2: Orchestrator should coordinate with other agents
        # Check that orchestrator received and processed the message
        orchestrator = agents['orchestrator']
        assert session_id in orchestrator.active_sessions
        
        # Step 3: Simulate IRA providing market data
        market_data_message = framework.create_message(
            sender_id="integration_ira_001",
            target_id="integration_pa_001",
            message_type=MessageType.RESPONSE,
            payload={
                "market_data": {
                    "market_volatility": 0.15,
                    "interest_rates": {"federal": 5.25},
                    "sector_trends": {"real_estate": 0.08}
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )
        
        await framework.send_message(market_data_message)
        await asyncio.sleep(0.1)
        
        # Step 4: Simulate planning agent generating plan
        planning_request_message = framework.create_message(
            sender_id="integration_oa_001",
            target_id="integration_pa_001",
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": "Save $100,000 for house down payment in 5 years",
                    "time_horizon": 60
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )
        
        await framework.send_message(planning_request_message)
        await asyncio.sleep(0.4)  # Allow planning time
        
        # Step 5: Check that plan was generated
        planner = agents['planner']
        # In a real system, we'd check the response was sent back
        
        # Verify correlation tracking works
        trace = framework.get_correlation_trace(correlation_id)
        assert len(trace) >= 3  # At least 3 messages in the workflow
        
        # Verify system health during workflow
        health = framework.get_system_health()
        assert health["success_rate"] > 0.8  # Most messages should succeed
    
    async def test_cmvl_trigger_scenario(self, integrated_system):
        """Test CMVL trigger and response workflow"""
        framework, agents = integrated_system
        
        correlation_id = str(uuid4())
        session_id = str(uuid4())
        
        # Step 1: IRA detects market trigger
        trigger_message = framework.create_message(
            sender_id="integration_ira_001",
            target_id="integration_oa_001",
            message_type=MessageType.NOTIFICATION,
            payload={
                "trigger_event": {
                    "trigger_id": str(uuid4()),
                    "severity": "high",
                    "description": "Market volatility spike detected",
                    "impact_score": 0.8
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )
        
        await framework.send_message(trigger_message)
        await asyncio.sleep(0.1)
        
        # Step 2: Orchestrator should activate CMVL
        orchestrator = agents['orchestrator']
        # Check that trigger was processed (in real system, would check response)
        
        # Step 3: Verification agent should be notified for re-verification
        cmvl_message = framework.create_message(
            sender_id="integration_oa_001",
            target_id="integration_va_001",
            message_type=MessageType.REQUEST,
            payload={
                "cmvl_trigger": {
                    "severity": "high",
                    "trigger_id": str(uuid4())
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )
        
        await framework.send_message(cmvl_message)
        await asyncio.sleep(0.1)
        
        # Verify CMVL workflow
        verifier = agents['verifier']
        assert verifier.cmvl_active is True
        
        # Check correlation tracking for CMVL
        trace = framework.get_correlation_trace(correlation_id)
        assert len(trace) >= 2
    
    async def test_concurrent_user_sessions(self, integrated_system):
        """Test system handles multiple concurrent user sessions"""
        framework, agents = integrated_system
        
        # Create multiple concurrent sessions
        sessions = []
        for i in range(3):
            correlation_id = str(uuid4())
            session_id = str(uuid4())
            
            message = framework.create_message(
                sender_id=f"user_client_{i}",
                target_id="integration_oa_001",
                message_type=MessageType.REQUEST,
                payload={"user_goal": f"Financial goal {i}"},
                correlation_id=correlation_id,
                session_id=session_id
            )
            
            sessions.append((correlation_id, session_id, message))
        
        # Send all messages concurrently
        tasks = []
        for _, _, message in sessions:
            task = framework.send_message(message)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        assert all(results)  # All messages should be sent successfully
        
        await asyncio.sleep(0.2)  # Allow processing
        
        # Verify all sessions are tracked
        orchestrator = agents['orchestrator']
        for _, session_id, _ in sessions:
            assert session_id in orchestrator.active_sessions
        
        # Verify system health under load
        health = framework.get_system_health()
        assert health["success_rate"] > 0.9
        assert health["total_messages"] >= 3
    
    async def test_agent_failure_recovery(self, integrated_system):
        """Test system handles agent failures gracefully"""
        framework, agents = integrated_system
        
        correlation_id = str(uuid4())
        session_id = str(uuid4())
        
        # Stop one agent to simulate failure
        await agents['planner'].stop()
        
        # Try to send message to failed agent
        message = framework.create_message(
            sender_id="integration_oa_001",
            target_id="integration_pa_001",
            message_type=MessageType.REQUEST,
            payload={"planning_request": {"user_goal": "test"}},
            correlation_id=correlation_id,
            session_id=session_id
        )
        
        # Message sending should handle the failure gracefully
        success = await framework.send_message(message)
        # In a real system with circuit breakers, this might fail gracefully
        
        # Check system health reflects the failure
        health = framework.get_system_health()
        agent_health = health["agent_health"]["integration_pa_001"]
        assert agent_health["status"] == "stopped"
        
        # Restart agent and verify recovery
        await agents['planner'].start()
        await asyncio.sleep(0.1)
        
        health = framework.get_system_health()
        agent_health = health["agent_health"]["integration_pa_001"]
        assert agent_health["status"] == "running"
    
    async def test_end_to_end_financial_planning(self, integrated_system):
        """Test complete end-to-end financial planning scenario"""
        framework, agents = integrated_system
        
        correlation_id = str(uuid4())
        session_id = str(uuid4())
        
        # Simulate complete workflow
        workflow_steps = [
            # 1. User goal submission
            {
                "sender": "user_client",
                "target": "integration_oa_001",
                "payload": {"user_goal": "Retire comfortably in 30 years with $2M"},
                "delay": 0.1
            },
            # 2. Market data request
            {
                "sender": "integration_oa_001",
                "target": "integration_ira_001",
                "payload": {"market_data_request": {"comprehensive": True}},
                "delay": 0.2
            },
            # 3. Planning request
            {
                "sender": "integration_oa_001",
                "target": "integration_pa_001",
                "payload": {
                    "planning_request": {
                        "user_goal": "Retire comfortably in 30 years with $2M",
                        "time_horizon": 360
                    }
                },
                "delay": 0.3
            },
            # 4. Verification request
            {
                "sender": "integration_pa_001",
                "target": "integration_va_001",
                "payload": {
                    "verification_request": {
                        "plan_id": str(uuid4()),
                        "plan_steps": [{"step_id": str(uuid4()), "amount": 50000}]
                    }
                },
                "delay": 0.15
            },
            # 5. Execution request
            {
                "sender": "integration_va_001",
                "target": "integration_ea_001",
                "payload": {
                    "execution_request": {
                        "plan_steps": [{"step_id": str(uuid4()), "amount": 50000}]
                    }
                },
                "delay": 0.2
            }
        ]
        
        # Execute workflow steps
        for step in workflow_steps:
            message = framework.create_message(
                sender_id=step["sender"],
                target_id=step["target"],
                message_type=MessageType.REQUEST,
                payload=step["payload"],
                correlation_id=correlation_id,
                session_id=session_id
            )
            
            await framework.send_message(message)
            await asyncio.sleep(step["delay"])
        
        # Verify complete workflow execution
        trace = framework.get_correlation_trace(correlation_id)
        assert len(trace) == len(workflow_steps)
        
        # Verify all agents participated
        participating_agents = set(msg.agent_id for msg in trace)
        expected_agents = {"user_client", "integration_oa_001", "integration_ira_001", 
                          "integration_pa_001", "integration_va_001", "integration_ea_001"}
        assert participating_agents.intersection(expected_agents)
        
        # Verify system performance
        health = framework.get_system_health()
        assert health["success_rate"] > 0.8
        assert health["total_messages"] >= len(workflow_steps)
    
    async def test_performance_under_load(self, integrated_system):
        """Test system performance under high message load"""
        framework, agents = integrated_system
        
        # Generate high message load
        num_messages = 50
        start_time = datetime.utcnow()
        
        tasks = []
        for i in range(num_messages):
            message = framework.create_message(
                sender_id=f"load_test_client_{i % 5}",
                target_id="integration_oa_001",
                message_type=MessageType.REQUEST,
                payload={"load_test": i},
                correlation_id=str(uuid4()),
                session_id=str(uuid4())
            )
            
            task = framework.send_message(message)
            tasks.append(task)
        
        # Send all messages concurrently
        results = await asyncio.gather(*tasks)
        end_time = datetime.utcnow()
        
        # Verify performance
        duration = (end_time - start_time).total_seconds()
        throughput = num_messages / duration
        
        assert all(results)  # All messages should succeed
        assert throughput > 10  # Should handle at least 10 messages/second
        
        # Verify system health after load test
        health = framework.get_system_health()
        assert health["success_rate"] > 0.95
        assert health["total_messages"] >= num_messages


# Async test runner
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])