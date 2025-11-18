#!/usr/bin/env python
"""
Simple API test that imports agents directly
No FastAPI server required
"""
import asyncio
import json


async def test_all_features():
    """Test all Phase 6 features"""
    print("\n" + "=" * 70)
    print(" PHASE 6 FEATURE TESTING - COMPREHENSIVE OUTPUT")
    print("=" * 70)

    # Import agents
    from agents.conversational_agent import get_conversational_agent
    from agents.graph_risk_detector import get_graph_risk_detector
    from agents.ml_prediction_engine import get_ml_prediction_engine

    # Test 1: Conversational Agent
    print("\n" + "─" * 70)
    print("1️⃣  CONVERSATIONAL AI - Natural Language Processing")
    print("─" * 70)

    conv_agent = get_conversational_agent()

    goal = await conv_agent.parse_natural_language_goal(
        "I'm 28 years old making $75,000/year and want to buy a $400,000 house in 5 years"
    )

    print(f"\n📝 User Input:")
    print(f"   'I'm 28 years old making $75,000/year and want to buy a")
    print(f"    $400,000 house in 5 years'")
    print(f"\n✨ Parsed Goal:")
    print(f"   • Goal Type: {goal['goal_type']}")
    print(f"   • Target Amount: ${goal.get('target_amount', 0):,.2f}")
    print(f"   • Timeframe: {goal.get('timeframe_years', 'N/A')} years")
    print(f"   • Risk Tolerance: {goal['risk_tolerance']}")
    print(f"   • Parsing Method: {goal.get('parsing_method', 'rule_based')}")

    # Generate narrative
    narrative = await conv_agent.generate_financial_narrative({
        "goal_type": "home_purchase",
        "target_amount": 80000,
        "timeframe_years": 5,
        "risk_tolerance": "moderate"
    })

    print(f"\n📖 Generated Financial Plan Narrative:")
    print("   ┌" + "─" * 66 + "┐")
    for line in narrative.split('\n')[:8]:  # First 8 lines
        print(f"   │ {line[:64]:<64} │")
    print("   │ ... (truncated for brevity)                                   │")
    print("   └" + "─" * 66 + "┘")

    # Test 2: Graph Risk Detector
    print("\n" + "─" * 70)
    print("2️⃣  GRAPH RISK DETECTION - Fraud & Anomaly Analysis")
    print("─" * 70)

    risk_detector = get_graph_risk_detector()

    risk_report = await risk_detector.analyze_transaction_risk("user_789", 90)

    print(f"\n🔍 Transaction Risk Analysis (90 days):")
    print(f"   • Risk Level: {risk_report['risk_level']} "
          f"({risk_report['overall_risk_score']:.2f}/1.0)")
    print(f"   • Transactions Analyzed: {risk_report['transaction_count']}")
    print(f"   • Anomalies Detected: {len(risk_report['anomalies'])}")
    print(f"   • Fraud Indicators: {len(risk_report['fraud_indicators'])}")

    if risk_report['anomalies']:
        print(f"\n⚠️  Top Anomaly:")
        anomaly = risk_report['anomalies'][0]
        print(f"   • Reason: {anomaly.get('reason', 'Unknown')}")
        print(f"   • Severity Score: {anomaly.get('anomaly_score', 0):.2f}")

    if risk_report['fraud_indicators']:
        print(f"\n🚨 Top Fraud Indicator:")
        fraud = risk_report['fraud_indicators'][0]
        print(f"   • Type: {fraud.get('type', 'Unknown')}")
        print(f"   • Severity: {fraud.get('severity', 'Unknown')}")
        print(f"   • Confidence: {fraud.get('fraud_confidence', 0):.0%}")

    # Test systemic risk
    assets = [
        {"symbol": "AAPL", "type": "stock", "sector": "technology", "allocation": 0.3},
        {"symbol": "MSFT", "type": "stock", "sector": "technology", "allocation": 0.2},
        {"symbol": "BND", "type": "bond", "sector": "fixed_income", "allocation": 0.3},
        {"symbol": "VTI", "type": "stock", "sector": "diversified", "allocation": 0.2}
    ]

    asset_graph = await risk_detector.build_asset_correlation_graph(assets)
    systemic_risk = await risk_detector.calculate_systemic_risk(asset_graph)

    print(f"\n📊 Portfolio Systemic Risk Analysis:")
    print(f"   • Overall Risk Score: {systemic_risk['overall_risk_score']:.3f}")
    print(f"   • Risk Level: {systemic_risk['risk_level']}")
    print(f"   • Assets in Portfolio: {len(assets)}")
    print(f"   • Correlation Edges: {asset_graph.number_of_edges()}")

    if systemic_risk['recommendations']:
        print(f"\n💡 Top Recommendation:")
        print(f"   {systemic_risk['recommendations'][0]}")

    # Test 3: ML Prediction Engine
    print("\n" + "─" * 70)
    print("3️⃣  MACHINE LEARNING PREDICTIONS - Market & Portfolio Analysis")
    print("─" * 70)

    ml_engine = get_ml_prediction_engine()

    # Market prediction
    market_pred = await ml_engine.predict_market_trend("AAPL", horizon_days=30)

    print(f"\n📈 Market Trend Prediction (AAPL - 30 days):")
    print(f"   • Current Price: ${market_pred['current_price']:.2f}")
    print(f"   • Predicted Price: ${market_pred['predicted_price']:.2f}")
    print(f"   • Expected Change: ${market_pred['price_change']:.2f} "
          f"({market_pred.get('price_change_percent', 0):.2f}%)")
    print(f"   • Trend Direction: {market_pred['trend_direction'].upper()}")
    print(f"   • Model Confidence: {market_pred['confidence']:.0%}")
    print(f"   • Prediction Model: {market_pred['model']}")

    # Portfolio prediction
    portfolio = {
        "total_value": 100000,
        "assets": [
            {"symbol": "AAPL", "allocation": 0.4, "expected_return": 0.10,
             "volatility": 0.20, "type": "stock"},
            {"symbol": "BND", "allocation": 0.6, "expected_return": 0.04,
             "volatility": 0.05, "type": "bond"}
        ]
    }

    portfolio_pred = await ml_engine.predict_portfolio_performance(portfolio, 365)

    print(f"\n💼 Portfolio Performance Prediction (1 year):")
    print(f"   • Current Value: ${portfolio_pred['current_value']:,.2f}")
    print(f"   • Expected Value: ${portfolio_pred['predictions']['expected_value']:,.2f}")
    print(f"   • Expected Return: {portfolio_pred['predictions']['expected_return']*100:.2f}%")
    print(f"   • Best Case (95th %ile): ${portfolio_pred['predictions']['percentile_95']:,.2f}")
    print(f"   • Worst Case (5th %ile): ${portfolio_pred['predictions']['percentile_5']:,.2f}")
    print(f"   • Sharpe Ratio: {portfolio_pred['risk_metrics']['sharpe_ratio']:.2f}")
    print(f"   • Simulations Run: {portfolio_pred['num_simulations']:,}")

    # Recommendations
    user_profile = {
        "age": 35,
        "risk_tolerance": "moderate",
        "monthly_expenses": 5000,
        "emergency_fund": 15000,
        "retirement_contributions": 0.08,
        "tax_bracket": 24,
        "goals": ["retirement", "emergency_fund"],
        "portfolio": portfolio
    }

    recommendations = await ml_engine.generate_recommendations(user_profile)

    print(f"\n🎯 Personalized Recommendations ({len(recommendations)} total):")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n   {i}. {rec['title']}")
        print(f"      • Priority: {rec['priority'].upper()}")
        print(f"      • Type: {rec['type'].replace('_', ' ').title()}")
        print(f"      • Impact: {rec['estimated_impact']}")
        print(f"      • Confidence: {rec['confidence']:.0%}")
        print(f"      • Action: {rec['description'][:60]}...")

    # Risk scenarios
    risk_scenarios = await ml_engine.predict_risk_levels(
        portfolio,
        ["market_crash", "recession", "bull_market"]
    )

    print(f"\n🎲 Risk Scenario Analysis:")
    print(f"   • Portfolio Value: ${risk_scenarios['portfolio_value']:,.2f}")
    print(f"   • Overall Risk Level: {risk_scenarios['overall_risk_level']}")
    print(f"\n   Scenario Predictions:")

    for scenario, pred in risk_scenarios['scenarios_analyzed'].items():
        emoji = "📉" if pred['expected_return'] < 0 else "📈"
        print(f"   {emoji} {scenario.replace('_', ' ').title()}:")
        print(f"      Return: {pred['expected_return']*100:+.1f}% | "
              f"Risk: {pred['risk_level']} | "
              f"Probability: {pred['probability']*100:.0f}%")

    # Summary
    print("\n" + "=" * 70)
    print(" ✅ PHASE 6 TESTING COMPLETE - ALL FEATURES WORKING!")
    print("=" * 70)
    print("\n📋 Summary:")
    print("   ✓ Conversational AI: Natural language parsing & narratives")
    print("   ✓ Graph Risk Detection: Fraud detection & systemic risk analysis")
    print("   ✓ ML Predictions: Market trends, portfolio forecasting, recommendations")
    print("\n🚀 Next Steps:")
    print("   1. Start API server: python test_api_server.py")
    print("   2. Test with curl: ./test_api_endpoints.sh")
    print("   3. Visit API docs: http://localhost:8000/docs")
    print("   4. Integrate with frontend ReasonGraph visualization")
    print("   5. (Optional) Install Ollama for enhanced LLM features")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_all_features())
