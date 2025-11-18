#!/usr/bin/env python
"""
Direct API endpoint testing without running a server
Tests the endpoint functions directly
"""
import asyncio
import json
from datetime import datetime


async def test_conversational_endpoints():
    """Test conversational AI endpoints"""
    print("\n" + "=" * 60)
    print("TESTING CONVERSATIONAL AI ENDPOINTS")
    print("=" * 60)

    from api.conversational_endpoints import parse_goal, generate_narrative, health_check
    from api.conversational_endpoints import ParseGoalRequest, GenerateNarrativeRequest

    # Test 1: Parse Goal
    print("\n1. Parse Goal Endpoint")
    request = ParseGoalRequest(
        user_input="I want to retire at 55 with $1.5 million",
        user_context={"age": 30, "income": 80000}
    )
    result = await parse_goal(request)
    print(f"   Goal Type: {result.goal_type}")
    print(f"   Target Amount: ${result.target_amount:,.2f}" if result.target_amount else "   Target Amount: Not specified")
    print(f"   Timeframe: {result.timeframe_years} years" if result.timeframe_years else "   Timeframe: Not specified")
    print(f"   Risk Tolerance: {result.risk_tolerance}")

    # Test 2: Generate Narrative
    print("\n2. Generate Narrative Endpoint")
    narrative_request = GenerateNarrativeRequest(
        plan={
            "goal_type": "retirement",
            "target_amount": 1500000,
            "timeframe_years": 25,
            "risk_tolerance": "moderate"
        }
    )
    narrative_result = await generate_narrative(narrative_request)
    print("   Narrative Preview:")
    print("   " + narrative_result.narrative[:200] + "...")

    # Test 3: Health Check
    print("\n3. Health Check")
    health = await health_check()
    print(f"   Status: {health['status']}")
    print(f"   Agent ID: {health['agent_id']}")

    print("\n✓ Conversational endpoints working!")


async def test_risk_endpoints():
    """Test risk detection endpoints"""
    print("\n" + "=" * 60)
    print("TESTING RISK DETECTION ENDPOINTS")
    print("=" * 60)

    from api.risk_endpoints import (
        analyze_transaction_risk,
        detect_fraud,
        health_check,
        TransactionRiskRequest,
        FraudCheckRequest
    )

    # Test 1: Transaction Risk Analysis
    print("\n1. Transaction Risk Analysis")
    risk_request = TransactionRiskRequest(
        user_id="test_user_456",
        lookback_days=90
    )
    risk_result = await analyze_transaction_risk(risk_request)
    print(f"   Risk Level: {risk_result.risk_level}")
    print(f"   Risk Score: {risk_result.overall_risk_score:.2f}")
    print(f"   Transactions Analyzed: {risk_result.transaction_count}")
    print(f"   Anomalies: {len(risk_result.anomalies)}")
    print(f"   Fraud Indicators: {len(risk_result.fraud_indicators)}")

    # Test 2: Fraud Detection
    print("\n2. Fraud Detection")
    fraud_request = FraudCheckRequest(
        transactions=[
            {"amount": 5000, "merchant": "Unknown", "timestamp": datetime.utcnow().isoformat()},
            {"amount": 5100, "merchant": "Unknown", "timestamp": datetime.utcnow().isoformat()},
            {"amount": 200, "merchant": "Grocery", "timestamp": datetime.utcnow().isoformat()}
        ],
        user_profile={"average_transaction": 150}
    )
    fraud_result = await detect_fraud(fraud_request)
    print(f"   Fraud Indicators Found: {fraud_result.fraud_indicators_found}")
    print(f"   Highest Confidence: {fraud_result.highest_confidence:.2f}")
    if fraud_result.indicators:
        print(f"   Top Indicator: {fraud_result.indicators[0].get('type', 'N/A')}")

    # Test 3: Health Check
    print("\n3. Health Check")
    health = await health_check()
    print(f"   Status: {health['status']}")

    print("\n✓ Risk detection endpoints working!")


async def test_ml_endpoints():
    """Test ML prediction endpoints"""
    print("\n" + "=" * 60)
    print("TESTING ML PREDICTION ENDPOINTS")
    print("=" * 60)

    from api.ml_endpoints import (
        predict_market_trend,
        predict_portfolio_performance,
        generate_recommendations,
        predict_risk_levels,
        health_check,
        MarketPredictionRequest,
        PortfolioPredictionRequest,
        RecommendationRequest,
        RiskPredictionRequest
    )

    # Test 1: Market Prediction
    print("\n1. Market Trend Prediction")
    market_request = MarketPredictionRequest(
        symbol="AAPL",
        horizon_days=30
    )
    market_result = await predict_market_trend(market_request)
    print(f"   Symbol: {market_result.symbol}")
    print(f"   Current Price: ${market_result.current_price:.2f}")
    print(f"   Predicted Price: ${market_result.predicted_price:.2f}")
    print(f"   Trend: {market_result.trend_direction}")
    print(f"   Confidence: {market_result.confidence:.2f}")

    # Test 2: Portfolio Prediction
    print("\n2. Portfolio Performance Prediction")
    portfolio_request = PortfolioPredictionRequest(
        portfolio={
            "total_value": 100000,
            "assets": [
                {"symbol": "AAPL", "allocation": 0.5, "expected_return": 0.10, "volatility": 0.20, "type": "stock"},
                {"symbol": "BND", "allocation": 0.5, "expected_return": 0.04, "volatility": 0.05, "type": "bond"}
            ]
        },
        timeframe_days=365,
        num_simulations=1000
    )
    portfolio_result = await predict_portfolio_performance(portfolio_request)
    print(f"   Current Value: ${portfolio_result.current_value:,.2f}")
    print(f"   Expected Value (1y): ${portfolio_result.predictions['expected_value']:,.2f}")
    print(f"   Expected Return: {portfolio_result.predictions['expected_return']*100:.2f}%")
    print(f"   95th Percentile: ${portfolio_result.predictions['percentile_95']:,.2f}")
    print(f"   5th Percentile: ${portfolio_result.predictions['percentile_5']:,.2f}")

    # Test 3: Recommendations
    print("\n3. Personalized Recommendations")
    rec_request = RecommendationRequest(
        user_profile={
            "age": 35,
            "risk_tolerance": "moderate",
            "monthly_expenses": 5000,
            "emergency_fund": 15000,
            "retirement_contributions": 0.08
        }
    )
    rec_result = await generate_recommendations(rec_request)
    print(f"   Total Recommendations: {rec_result.total_count}")
    for i, rec in enumerate(rec_result.recommendations[:3], 1):
        print(f"   {i}. {rec['title']} (Priority: {rec['priority']})")

    # Test 4: Risk Predictions
    print("\n4. Risk Level Predictions")
    risk_pred_request = RiskPredictionRequest(
        portfolio={
            "total_value": 100000,
            "assets": [
                {"symbol": "AAPL", "allocation": 0.6, "type": "stock"},
                {"symbol": "BND", "allocation": 0.4, "type": "bond"}
            ]
        },
        scenarios=["market_crash", "recession", "bull_market"]
    )
    risk_pred_result = await predict_risk_levels(risk_pred_request)
    print(f"   Overall Risk Level: {risk_pred_result.overall_risk_level}")
    print(f"   Scenarios Analyzed: {len(risk_pred_result.scenarios_analyzed)}")
    for scenario, pred in list(risk_pred_result.scenarios_analyzed.items())[:2]:
        print(f"   • {scenario}: {pred['risk_level']} risk, {pred['expected_return']*100:.1f}% return")

    # Test 5: Health Check
    print("\n5. Health Check")
    health = await health_check()
    print(f"   Status: {health['status']}")
    print(f"   Models Loaded: {health['models_loaded']}")

    print("\n✓ ML prediction endpoints working!")


async def main():
    """Run all endpoint tests"""
    print("\n" + "#" * 60)
    print("# PHASE 6 API ENDPOINT TESTING (Direct)")
    print("#" * 60)

    await test_conversational_endpoints()
    await test_risk_endpoints()
    await test_ml_endpoints()

    print("\n" + "=" * 60)
    print("✓ ALL API ENDPOINTS TESTED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Run the API server: python test_api_server.py")
    print("  2. Test with curl: ./test_api_endpoints.sh")
    print("  3. Visit interactive docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
