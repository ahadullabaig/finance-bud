"""
Critical Flow Integration Tests - Phase 1

Comprehensive end-to-end tests for mission-critical agent flows.
These tests establish a safety net for refactoring.

Test Scenarios:
1. Full orchestration → planning → verification → execution flow
2. CMVL trigger detection and re-planning
3. Error handling and circuit breaker activation
4. Multi-agent coordination with message passing
5. Performance under load

Created: Phase 1 - Foundation & Safety Net
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from typing import Dict, Any

from agents.mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent,
    MockVerificationAgent,
    MockExecutionAgent
)
from agents.communication import AgentCommunicationFramework
from data_models.schemas import (
    AgentMessage,
    MessageType,
    Priority,
    FinancialGoal,
    Constraint,
    RiskProfile,
    TaxContext,
    FinancialState
)


class TestCriticalAgentFlows:
    """Critical end-to-end flow tests that must pass during refactoring"""

    @pytest.fixture
    async def complete_system(self):
        """Set up complete integrated system with all agents"""
        framework = AgentCommunicationFramework()

        agents = {
            'orchestrator': MockOrchestrationAgent("test_oa_001"),
            'planner': MockPlanningAgent("test_pa_001"),
            'ira': MockInformationRetrievalAgent("test_ira_001"),
            'verifier': MockVerificationAgent("test_va_001"),
            'executor': MockExecutionAgent("test_ea_001")
        }

        # Register all agents
        for agent_name, agent in agents.items():
            await agent.start()
            framework.register_agent(agent, [f"{agent_name}_capabilities"])

        yield framework, agents

        # Cleanup
        for agent in agents.values():
            await agent.stop()

    @pytest.mark.asyncio
    async def test_full_financial_planning_flow(self, complete_system):
        """
        CRITICAL: Test complete flow from goal submission to execution

        Flow: User Goal → Orchestration → IRA (market data) → Planning →
              Verification → Execution

        This test validates:
        - Agent communication framework
        - Message routing
        - Correlation tracking
        - Data model integrity
        """
        framework, agents = complete_system

        correlation_id = str(uuid4())
        session_id = str(uuid4())

        # Step 1: Submit financial goal
        goal_message = framework.create_message(
            sender_id="test_user",
            target_id="test_oa_001",
            message_type=MessageType.REQUEST,
            payload={
                "user_goal": "Retire comfortably in 20 years with $2M portfolio",
                "current_age": 45,
                "retirement_age": 65,
                "risk_tolerance": "moderate",
                "current_savings": 250000,
                "monthly_contribution": 5000
            },
            correlation_id=correlation_id,
            session_id=session_id,
            priority=Priority.HIGH
        )

        success = await framework.send_message(goal_message)
        assert success is True, "Failed to send goal message"

        await asyncio.sleep(0.2)

        # Step 2: Verify orchestrator received and processed
        orchestrator = agents['orchestrator']
        assert session_id in orchestrator.active_sessions, \
            "Orchestrator did not register session"

        # Step 3: Simulate market data retrieval
        market_data_message = framework.create_message(
            sender_id="test_ira_001",
            target_id="test_pa_001",
            message_type=MessageType.RESPONSE,
            payload={
                "market_data": {
                    "sp500_volatility": 0.18,
                    "bond_yields": {"10yr": 4.5, "30yr": 4.8},
                    "inflation_rate": 3.2,
                    "market_sentiment": "neutral"
                },
                "historical_returns": {
                    "stocks": 0.10,
                    "bonds": 0.04,
                    "real_estate": 0.08
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )

        await framework.send_message(market_data_message)
        await asyncio.sleep(0.1)

        # Step 4: Request plan generation
        planning_request = framework.create_message(
            sender_id="test_oa_001",
            target_id="test_pa_001",
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "goal": "Retire comfortably in 20 years with $2M portfolio",
                    "time_horizon_months": 240,
                    "risk_profile": "moderate",
                    "constraints": [
                        {"type": "budget", "value": 5000, "description": "Monthly contribution limit"}
                    ],
                    "current_portfolio_value": 250000
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )

        await framework.send_message(planning_request)
        await asyncio.sleep(0.5)  # Allow planning time

        # Step 5: Request plan verification
        verification_request = framework.create_message(
            sender_id="test_pa_001",
            target_id="test_va_001",
            message_type=MessageType.REQUEST,
            payload={
                "verification_request": {
                    "plan_id": str(uuid4()),
                    "constraints": [
                        {"type": "budget", "value": 5000}
                    ],
                    "steps": [
                        {"action": "invest_stocks", "allocation": 0.6},
                        {"action": "invest_bonds", "allocation": 0.3},
                        {"action": "cash_reserve", "allocation": 0.1}
                    ]
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )

        await framework.send_message(verification_request)
        await asyncio.sleep(0.2)

        # Step 6: Verify message correlation tracking
        trace = framework.get_correlation_trace(correlation_id)
        assert len(trace) >= 4, \
            f"Expected at least 4 messages in trace, got {len(trace)}"

        # Verify message types in correct order
        message_types = [msg.message_type for msg in trace]
        assert MessageType.REQUEST in message_types, "No REQUEST in trace"
        assert MessageType.RESPONSE in message_types, "No RESPONSE in trace"

        # Step 7: Verify system health
        health = framework.get_system_health()
        assert health["success_rate"] >= 0.8, \
            f"Success rate too low: {health['success_rate']}"
        assert health["active_circuits"] > 0, "No active circuit breakers"

    @pytest.mark.asyncio
    async def test_cmvl_workflow_complete(self, complete_system):
        """
        CRITICAL: Test Continuous Monitoring and Verification Loop

        Flow: Trigger Detection → Orchestration → Re-verification →
              Re-planning (if needed) → Notification

        This validates the core CMVL capability.
        """
        framework, agents = complete_system

        correlation_id = str(uuid4())
        session_id = str(uuid4())

        # Step 1: Setup existing plan in session
        initial_plan_msg = framework.create_message(
            sender_id="test_user",
            target_id="test_oa_001",
            message_type=MessageType.REQUEST,
            payload={
                "existing_plan": {
                    "plan_id": str(uuid4()),
                    "status": "active",
                    "risk_profile": "moderate"
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )

        await framework.send_message(initial_plan_msg)
        await asyncio.sleep(0.1)

        # Step 2: IRA detects critical market trigger
        trigger_message = framework.create_message(
            sender_id="test_ira_001",
            target_id="test_oa_001",
            message_type=MessageType.NOTIFICATION,
            payload={
                "trigger_event": {
                    "trigger_id": str(uuid4()),
                    "type": "market_volatility_spike",
                    "severity": "high",
                    "description": "VIX spiked above 35 - market stress detected",
                    "impact_score": 0.85,
                    "timestamp": datetime.utcnow().isoformat(),
                    "affected_assets": ["stocks", "options"],
                    "recommended_action": "re_verify_plan"
                }
            },
            correlation_id=correlation_id,
            session_id=session_id,
            priority=Priority.CRITICAL
        )

        await framework.send_message(trigger_message)
        await asyncio.sleep(0.1)

        # Step 3: Orchestrator activates CMVL
        cmvl_activation_msg = framework.create_message(
            sender_id="test_oa_001",
            target_id="test_va_001",
            message_type=MessageType.REQUEST,
            payload={
                "cmvl_activation": {
                    "trigger_severity": "high",
                    "trigger_type": "market_volatility_spike",
                    "session_id": session_id,
                    "plan_id": str(uuid4()),
                    "requested_action": "immediate_reverification"
                }
            },
            correlation_id=correlation_id,
            session_id=session_id,
            priority=Priority.CRITICAL
        )

        await framework.send_message(cmvl_activation_msg)
        await asyncio.sleep(0.2)

        # Step 4: Verification agent responds
        verifier = agents['verifier']
        assert verifier.cmvl_active is True, "CMVL not activated in verifier"

        # Step 5: If verification fails, request re-planning
        replan_request = framework.create_message(
            sender_id="test_va_001",
            target_id="test_pa_001",
            message_type=MessageType.REQUEST,
            payload={
                "replan_request": {
                    "reason": "constraints_violated_due_to_market_conditions",
                    "violation_details": {
                        "risk_exceeded": True,
                        "target_risk": 0.15,
                        "current_risk": 0.28
                    },
                    "adjustment_recommendations": [
                        "reduce_equity_exposure",
                        "increase_bond_allocation"
                    ]
                }
            },
            correlation_id=correlation_id,
            session_id=session_id,
            priority=Priority.HIGH
        )

        await framework.send_message(replan_request)
        await asyncio.sleep(0.3)

        # Verify CMVL workflow completed
        trace = framework.get_correlation_trace(correlation_id)
        assert len(trace) >= 4, "CMVL workflow incomplete"

        # Verify critical priority messages were processed
        critical_messages = [msg for msg in trace if msg.priority == Priority.CRITICAL]
        assert len(critical_messages) >= 1, "No critical messages in CMVL workflow"

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, complete_system):
        """
        CRITICAL: Test circuit breaker prevents cascading failures

        Validates system resilience under agent failures.
        """
        framework, agents = complete_system

        # Simulate rapid failures to trigger circuit breaker
        correlation_id = str(uuid4())

        for i in range(5):
            # Send messages that will fail
            fail_message = framework.create_message(
                sender_id="test_user",
                target_id="nonexistent_agent",  # Invalid target
                message_type=MessageType.REQUEST,
                payload={"test": "failure"},
                correlation_id=correlation_id,
                session_id=str(uuid4())
            )

            success = await framework.send_message(fail_message)
            # Message sending might fail, which is expected
            await asyncio.sleep(0.05)

        # Check circuit breaker status
        health = framework.get_system_health()

        # System should still be operational despite failures
        assert health is not None, "System health check failed"
        assert "circuit_breakers" in health or "active_circuits" in health, \
            "Circuit breaker info not in health check"

    @pytest.mark.asyncio
    async def test_concurrent_session_isolation(self, complete_system):
        """
        CRITICAL: Test session isolation under concurrent load

        Validates no message cross-contamination between sessions.
        """
        framework, agents = complete_system

        # Create 5 concurrent sessions
        sessions = []
        for i in range(5):
            correlation_id = str(uuid4())
            session_id = str(uuid4())

            message = framework.create_message(
                sender_id=f"test_user_{i}",
                target_id="test_oa_001",
                message_type=MessageType.REQUEST,
                payload={
                    "user_goal": f"Goal {i}",
                    "session_marker": session_id  # Track this session
                },
                correlation_id=correlation_id,
                session_id=session_id
            )

            sessions.append({
                "correlation_id": correlation_id,
                "session_id": session_id,
                "message": message
            })

        # Send all messages concurrently
        tasks = [
            framework.send_message(session["message"])
            for session in sessions
        ]

        results = await asyncio.gather(*tasks)
        assert all(results), "Some messages failed to send"

        await asyncio.sleep(0.5)  # Allow processing

        # Verify each session has isolated traces
        for session in sessions:
            trace = framework.get_correlation_trace(session["correlation_id"])
            assert len(trace) > 0, f"No trace for session {session['session_id']}"

            # Verify all messages in trace belong to this session
            for msg in trace:
                assert msg.session_id == session["session_id"], \
                    f"Message cross-contamination detected: {msg.session_id} != {session['session_id']}"

    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, complete_system):
        """
        CRITICAL: Test error handling and recovery mechanisms

        Validates errors are properly logged, propagated, and don't crash system.
        """
        framework, agents = complete_system

        correlation_id = str(uuid4())
        session_id = str(uuid4())

        # Send message with invalid payload that should trigger error handling
        error_message = framework.create_message(
            sender_id="test_user",
            target_id="test_pa_001",
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "invalid_field": "this_should_cause_validation_error",
                    # Missing required fields
                }
            },
            correlation_id=correlation_id,
            session_id=session_id
        )

        # System should handle gracefully without crashing
        try:
            await framework.send_message(error_message)
            await asyncio.sleep(0.2)
        except Exception as e:
            pytest.fail(f"System crashed on invalid message: {e}")

        # Verify system still operational
        health = framework.get_system_health()
        assert health is not None, "System not responsive after error"

        # Verify can still process valid messages
        valid_message = framework.create_message(
            sender_id="test_user",
            target_id="test_oa_001",
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Valid goal after error"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4())
        )

        success = await framework.send_message(valid_message)
        assert success, "System not recovering after error"


class TestDataModelIntegrity:
    """Test data model integrity during agent communication"""

    @pytest.mark.asyncio
    async def test_financial_goal_serialization(self):
        """Test FinancialGoal serialization/deserialization integrity"""

        goal = FinancialGoal(
            goal_id=str(uuid4()),
            description="Save $1M for retirement",
            target_amount=Decimal("1000000.00"),
            current_amount=Decimal("250000.00"),
            target_date=datetime.utcnow() + timedelta(days=7300),  # 20 years
            priority=1,
            category="retirement"
        )

        # Serialize to dict
        goal_dict = goal.model_dump()

        # Deserialize back
        goal_restored = FinancialGoal.model_validate(goal_dict)

        # Verify integrity
        assert goal_restored.goal_id == goal.goal_id
        assert goal_restored.target_amount == goal.target_amount
        assert goal_restored.description == goal.description

    @pytest.mark.asyncio
    async def test_financial_state_calculations(self):
        """Test FinancialState derived field calculations"""

        state = FinancialState(
            assets={
                "checking": Decimal("10000"),
                "savings": Decimal("50000"),
                "401k": Decimal("200000")
            },
            liabilities={
                "mortgage": Decimal("300000"),
                "car_loan": Decimal("25000")
            },
            income=Decimal("120000"),
            expenses=Decimal("80000")
        )

        # Verify calculated fields
        expected_net_worth = Decimal("260000") - Decimal("325000")
        assert state.net_worth == expected_net_worth

        expected_savings_rate = (Decimal("120000") - Decimal("80000")) / Decimal("120000")
        # Allow small floating point differences
        assert abs(float(state.savings_rate) - float(expected_savings_rate)) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
