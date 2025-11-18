#!/usr/bin/env python3
"""
Debug script to test conversational agent directly
"""

import asyncio
import sys
import traceback
from agents.conversational_agent import get_conversational_agent

async def test_direct():
    """Test the conversational agent directly without API layer"""
    print("🔍 Testing Conversational Agent Directly")
    print("=" * 50)
    
    try:
        agent = get_conversational_agent()
        print(f"✅ Agent created: {agent.agent_id}")
        print(f"   Ollama available: {agent.ollama_available}")
        print(f"   Fallback enabled: {agent.use_fallback}")
        
        # Test the exact inputs that are failing
        test_inputs = [
            "hi",
            "I want to retire at 60 with $2M",
            "",
            "hello"
        ]
        
        for user_input in test_inputs:
            print(f"\n🧪 Testing input: '{user_input}'")
            try:
                result = await agent.parse_natural_language_goal(user_input)
                print(f"✅ Success: {result}")
                
                # Check required fields
                required_fields = ['goal_type', 'parsed_at', 'raw_input', 'parsing_method']
                missing = [f for f in required_fields if f not in result]
                if missing:
                    print(f"⚠️  Missing required fields: {missing}")
                else:
                    print(f"✅ All required fields present")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct())