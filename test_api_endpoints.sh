#!/bin/bash
# Test script for Phase 6 API endpoints
# Make sure the API server is running first: python test_api_server.py

echo "=========================================="
echo "Phase 6 API Endpoint Testing"
echo "=========================================="
echo ""

BASE_URL="http://localhost:8000"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}1. Testing Health Endpoints${NC}"
echo "---"
curl -s "$BASE_URL/health" | python -m json.tool
echo ""
curl -s "$BASE_URL/api/conversational/health" | python -m json.tool
echo ""
curl -s "$BASE_URL/api/risk/health" | python -m json.tool
echo ""
curl -s "$BASE_URL/api/ml/health" | python -m json.tool
echo ""
echo ""

echo -e "${BLUE}2. Testing Conversational Agent - Parse Goal${NC}"
echo "---"
curl -s -X POST "$BASE_URL/api/conversational/parse-goal" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I want to save $50,000 for a house down payment in 3 years",
    "user_context": {"age": 30, "income": 75000}
  }' | python -m json.tool
echo ""
echo ""

echo -e "${BLUE}3. Testing Risk Detection - Analyze Transactions${NC}"
echo "---"
curl -s -X POST "$BASE_URL/api/risk/analyze-transactions" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "lookback_days": 90
  }' | python -m json.tool
echo ""
echo ""

echo -e "${BLUE}4. Testing Risk Detection - Detect Fraud${NC}"
echo "---"
curl -s -X POST "$BASE_URL/api/risk/detect-fraud" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"amount": 5000, "merchant": "Unknown", "timestamp": "2024-01-15T02:30:00"},
      {"amount": 5100, "merchant": "Unknown", "timestamp": "2024-01-15T02:35:00"},
      {"amount": 4900, "merchant": "Unknown", "timestamp": "2024-01-15T02:40:00"}
    ],
    "user_profile": {"average_transaction": 150}
  }' | python -m json.tool
echo ""
echo ""

echo -e "${BLUE}5. Testing ML Predictions - Market Trend${NC}"
echo "---"
curl -s -X POST "$BASE_URL/api/ml/predict-market" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "horizon_days": 30
  }' | python -m json.tool
echo ""
echo ""

echo -e "${BLUE}6. Testing ML Predictions - Portfolio Performance${NC}"
echo "---"
curl -s -X POST "$BASE_URL/api/ml/predict-portfolio" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "total_value": 100000,
      "assets": [
        {"symbol": "AAPL", "allocation": 0.4, "expected_return": 0.10, "volatility": 0.20, "type": "stock"},
        {"symbol": "BND", "allocation": 0.6, "expected_return": 0.04, "volatility": 0.05, "type": "bond"}
      ]
    },
    "timeframe_days": 365
  }' | python -m json.tool
echo ""
echo ""

echo -e "${BLUE}7. Testing ML Predictions - Recommendations${NC}"
echo "---"
curl -s -X POST "$BASE_URL/api/ml/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_profile": {
      "age": 35,
      "risk_tolerance": "moderate",
      "monthly_expenses": 5000,
      "emergency_fund": 15000,
      "retirement_contributions": 0.08
    }
  }' | python -m json.tool
echo ""
echo ""

echo -e "${GREEN}=========================================="
echo "Testing Complete!"
echo "==========================================${NC}"
