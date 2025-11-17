#!/usr/bin/env python3
"""
Basic test for Execution Agent functionality
"""

import asyncio
from decimal import Decimal
from agents.execution_agent import ExecutionAgent, AccountType
from data_models.schemas import AgentMessage, MessageType, Priority

async def test_execution_agent_basic():
    """Test basic Execution Agent functionality"""
    
    print("🚀 Testing Execution Agent Basic Functionality")
    
    # Initialize agent
    agent = ExecutionAgent('test_execution_agent')
    print(f"✅ Agent initialized: {agent.agent_id}")
    
    # Test ledger functionality
    account_id = agent.ledger.create_account(
        account_type=AccountType.INVESTMENT,
        account_name='Test Investment Account',
        initial_balance=Decimal('10000')
    )
    print(f"✅ Created account: {account_id}")
    
    # Test portfolio value calculation
    portfolio_value = agent.ledger.get_portfolio_value(agent.market_prices)
    print(f"✅ Portfolio value: ${portfolio_value}")
    
    # Test comprehensive status
    status = agent.get_comprehensive_status()
    print(f"✅ Agent status: {status['status']}")
    print(f"✅ Portfolio value in status: ${status['portfolio_value']}")
    
    # Test real-time monitoring
    monitor_id = await agent.start_real_time_monitoring(
        user_id="test_user_001",
        monitoring_config={
            "portfolio_tracking": True,
            "risk_monitoring": True
        }
    )
    print(f"✅ Started real-time monitoring: {monitor_id}")
    
    # Test workflow registration
    workflow_success = await agent.register_workflow(
        workflow_id="test_workflow_001",
        workflow_data={
            "workflow_type": "financial_planning",
            "user_id": "test_user_001",
            "total_tasks": 3
        }
    )
    print(f"✅ Workflow registration: {workflow_success}")
    
    # Test message processing
    test_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "get_portfolio_status",
            "user_id": "test_user_001"
        },
        correlation_id="test_correlation_001",
        session_id="test_session_001",
        trace_id="test_trace_001"
    )
    
    response = await agent.process_message(test_message)
    if response and response.payload.get("portfolio_status"):
        print("✅ Message processing successful")
        print(f"✅ Portfolio status retrieved: ${response.payload['portfolio_status']['total_value']}")
    else:
        print("❌ Message processing failed")
    
    # Test tax optimization
    tax_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "optimize_taxes",
            "user_id": "test_user_001"
        },
        correlation_id="test_correlation_002",
        session_id="test_session_001",
        trace_id="test_trace_002"
    )
    
    tax_response = await agent.process_message(tax_message)
    if tax_response and tax_response.payload.get("tax_loss_opportunities") is not None:
        print("✅ Tax optimization successful")
        print(f"✅ Tax loss opportunities: {len(tax_response.payload['tax_loss_opportunities'])}")
    else:
        print("❌ Tax optimization failed")
    
    # Test compliance check
    compliance_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "check_compliance",
            "user_id": "test_user_001"
        },
        correlation_id="test_correlation_003",
        session_id="test_session_001",
        trace_id="test_trace_003"
    )
    
    compliance_response = await agent.process_message(compliance_message)
    if compliance_response and compliance_response.payload.get("compliance_report"):
        print("✅ Compliance check successful")
        report = compliance_response.payload["compliance_report"]
        print(f"✅ Compliance report generated: {report['report_id']}")
    else:
        print("❌ Compliance check failed")
    
    # Test forecast generation
    forecast_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "generate_forecast",
            "forecast_params": {
                "time_horizon_months": 12
            },
            "user_id": "test_user_001"
        },
        correlation_id="test_correlation_004",
        session_id="test_session_001",
        trace_id="test_trace_004"
    )
    
    forecast_response = await agent.process_message(forecast_message)
    if forecast_response and forecast_response.payload.get("forecast"):
        print("✅ Forecast generation successful")
        forecast = forecast_response.payload["forecast"]
        print(f"✅ Expected portfolio value in 12 months: ${forecast['forecasts']['expected_value']:.2f}")
    else:
        print("❌ Forecast generation failed")
    
    print("\n🎉 All Execution Agent tests completed successfully!")
    print(f"📊 Final execution metrics: {agent.execution_metrics}")

if __name__ == "__main__":
    asyncio.run(test_execution_agent_basic())