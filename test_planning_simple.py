#!/usr/bin/env python3
"""
Simple test for Planning Agent implementation
"""

import asyncio
from agents.planning_agent import PlanningAgent
from data_models.schemas import AgentMessage, MessageType

async def test_planning_agent():
    agent = PlanningAgent('test_agent')
    
    # Create a planning request
    request = AgentMessage(
        agent_id='test_orchestrator',
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            'planning_request': {
                'user_goal': 'Build retirement fund over 20 years',
                'current_state': {
                    'total_assets': 100000,
                    'total_liabilities': 20000,
                    'monthly_income': 8000,
                    'monthly_expenses': 5000,
                    'cash': 15000,
                    'investments': 60000,
                    'emergency_fund': 25000
                },
                'constraints': [],
                'time_horizon': 240,
                'risk_profile': {'risk_tolerance': 'moderate'}
            }
        },
        correlation_id='test_001',
        session_id='test_session_001',
        trace_id='test_trace_001'
    )
    
    print('Testing full planning agent...')
    response = await agent.process_message(request)
    
    if response and response.payload.get('planning_completed'):
        print('Planning completed successfully!')
        alt_strategies = response.payload.get('alternative_strategies', 0)
        plan_steps = len(response.payload.get('plan_steps', []))
        search_paths = len(response.payload.get('search_paths', []))
        
        print(f'Alternative strategies: {alt_strategies}')
        print(f'Plan steps: {plan_steps}')
        print(f'Search paths: {search_paths}')
        
        # Show first plan step if available
        steps = response.payload.get('plan_steps', [])
        if steps:
            first_step = steps[0]
            print(f'First step: {first_step.get("action_type", "unknown")} - ${first_step.get("amount", 0)}')
        
        return True
    else:
        print('Planning failed or incomplete')
        if response:
            print(f'Response type: {response.message_type}')
            if 'error' in response.payload:
                print(f'Error: {response.payload["error"]}')
        return False

if __name__ == '__main__':
    success = asyncio.run(test_planning_agent())
    print(f'Test result: {"PASSED" if success else "FAILED"}')