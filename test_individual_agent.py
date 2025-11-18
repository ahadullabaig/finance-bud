#!/usr/bin/env python
"""
Quick script to test individual Phase 6 agents interactively
"""
import asyncio
from agents.conversational_agent import get_conversational_agent
from agents.graph_risk_detector import get_graph_risk_detector
from agents.ml_prediction_engine import get_ml_prediction_engine


async def test_conversational():
    """Test conversational agent with your own inputs"""
    agent = get_conversational_agent()

    # Try your own goal!
    goal = await agent.parse_natural_language_goal(
        "I'm 30 years old and want to save $100,000 for a house down payment in 5 years"
    )

    print("\n=== Parsed Goal ===")
    print(f"Goal Type: {goal['goal_type']}")
    print(f"Target Amount: ${goal.get('target_amount', 0):,.2f}")
    print(f"Timeframe: {goal.get('timeframe_years')} years")
    print(f"Risk Tolerance: {goal['risk_tolerance']}")


async def test_risk_detection():
    """Test risk detection with sample transactions"""
    detector = get_graph_risk_detector()

    # Analyze transaction risk
    report = await detector.analyze_transaction_risk("user_123", lookback_days=90)

    print("\n=== Risk Analysis ===")
    print(f"Risk Level: {report['risk_level']}")
    print(f"Risk Score: {report['overall_risk_score']:.2f}")
    print(f"Transactions Analyzed: {report['transaction_count']}")
    print(f"Anomalies Found: {len(report['anomalies'])}")
    print(f"Fraud Indicators: {len(report['fraud_indicators'])}")


async def test_ml_predictions():
    """Test ML predictions"""
    engine = get_ml_prediction_engine()

    # Predict market trend
    prediction = await engine.predict_market_trend("AAPL", horizon_days=30)

    print("\n=== Market Prediction ===")
    print(f"Symbol: {prediction['symbol']}")
    print(f"Current Price: ${prediction['current_price']:.2f}")
    print(f"Predicted Price (30d): ${prediction['predicted_price']:.2f}")
    print(f"Trend: {prediction['trend_direction']}")
    print(f"Confidence: {prediction['confidence']:.2f}")

    # Get personalized recommendations
    user_profile = {
        "age": 35,
        "risk_tolerance": "moderate",
        "monthly_expenses": 5000,
        "emergency_fund": 15000,
        "retirement_contributions": 0.08
    }

    recommendations = await engine.generate_recommendations(user_profile)

    print("\n=== Recommendations ===")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec['title']} (Priority: {rec['priority']})")
        print(f"   {rec['description']}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Phase 6 Interactive Testing")
    print("=" * 60)

    await test_conversational()
    await test_risk_detection()
    await test_ml_predictions()

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
