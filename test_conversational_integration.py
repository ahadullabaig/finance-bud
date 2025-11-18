#!/usr/bin/env python3
"""
Test script for conversational endpoints integration
Tests all conversational endpoints through the main API server
"""

import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8001"

def test_endpoint(method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test an API endpoint and return the result"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        else:
            return {"success": False, "error": f"Unsupported method: {method}"}
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Run all conversational endpoint tests"""
    print("🧪 Testing Conversational Endpoints Integration")
    print("=" * 60)
    
    tests = [
        {
            "name": "Root Endpoint - Check Conversational Available",
            "method": "GET",
            "endpoint": "/",
            "expected_keys": ["conversational_available", "conversational_agent_status"]
        },
        {
            "name": "Conversational Health Check",
            "method": "GET", 
            "endpoint": "/api/conversational/health",
            "expected_keys": ["status", "agent_id", "ollama_available"]
        },
        {
            "name": "Parse Goal - Valid Input",
            "method": "POST",
            "endpoint": "/api/conversational/parse-goal",
            "data": {
                "user_input": "I want to retire at 60 with $2 million",
                "user_context": {"age": 35, "income": 100000}
            },
            "expected_keys": ["goal_type", "target_amount", "timeframe_years", "parsing_method"]
        },
        {
            "name": "Parse Goal - Empty Input (Error Handling)",
            "method": "POST",
            "endpoint": "/api/conversational/parse-goal", 
            "data": {"user_input": ""},
            "expected_keys": ["goal_type", "parsing_method"]
        },
        {
            "name": "Generate Narrative",
            "method": "POST",
            "endpoint": "/api/conversational/generate-narrative",
            "data": {
                "plan": {
                    "goal_type": "retirement",
                    "target_amount": 2000000,
                    "timeframe_years": 25,
                    "risk_tolerance": "moderate"
                }
            },
            "expected_keys": ["narrative", "generated_at"]
        },
        {
            "name": "Explain Scenario",
            "method": "POST", 
            "endpoint": "/api/conversational/explain-scenario",
            "data": {
                "scenario": {
                    "type": "market_crash",
                    "severity": "high", 
                    "description": "30% market decline"
                },
                "impact": {
                    "target_amount_change": -50000,
                    "timeframe_change": 2
                }
            },
            "expected_keys": ["explanation", "scenario_type", "generated_at"]
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n🔍 {test['name']}")
        print("-" * 40)
        
        result = test_endpoint(
            test["method"],
            test["endpoint"], 
            test.get("data")
        )
        
        if result["success"]:
            print(f"✅ Status: {result['status_code']}")
            
            # Check expected keys if provided
            if "expected_keys" in test and isinstance(result["data"], dict):
                missing_keys = []
                for key in test["expected_keys"]:
                    if key not in result["data"]:
                        missing_keys.append(key)
                
                if missing_keys:
                    print(f"⚠️  Missing expected keys: {missing_keys}")
                    result["success"] = False
                else:
                    print(f"✅ All expected keys present: {test['expected_keys']}")
            
            # Show sample data
            if isinstance(result["data"], dict):
                sample_keys = list(result["data"].keys())[:3]
                print(f"📄 Sample response keys: {sample_keys}")
            
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            if "status_code" in result:
                print(f"   Status Code: {result['status_code']}")
        
        results.append({
            "test": test["name"],
            "success": result["success"],
            "details": result
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} - {result['test']}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All conversational endpoints integration tests PASSED!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())