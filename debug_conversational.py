#!/usr/bin/env python3
"""
Debug script to test conversational agent directly
"""

import asyncio
from agents.conversational_agent import get_conversational_agent

async def test_parse_goal():
    """Test the parse goal functionality directly"""
    print("🔍 Testing conversational agent parse goal directly...")
    
    try:
        agent = get_conversational_agent()
        print(f"Agent initialized: {agent.agent_id}")
        print(f"Ollama available: {agent.ollama_available}")
        
        # Test with "hi"
        print("\n--- Testing with 'hi' ---")
        result = await agent.parse_natural_language_goal("hi")
        print(f"Result: {result}")
        print(f"Keys: {list(result.keys())}")
        
        # Check required fields
        required_fields = ["goal_type", "parsed_at", "parsing_method", "raw_input"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
        else:
            print(f"✅ All required fields present")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_parse_goal())