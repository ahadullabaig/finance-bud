"""
Phase 6 Agents Test Script

Tests all three Phase 6 agents:
- Task 23: ConversationalAgent
- Task 24: GraphRiskDetector
- Task 25: MLPredictionEngine
"""

import asyncio
import json
from datetime import datetime, timedelta, UTC


async def test_conversational_agent():
    """Test ConversationalAgent (Task 23)"""
    print("\n" + "="*80)
    print("TESTING CONVERSATIONAL AGENT (Task 23)")
    print("="*80)

    try:
        from agents.conversational_agent import get_conversational_agent

        agent = get_conversational_agent()
        print(f"✓ Agent initialized: {agent.agent_id}")
        print(f"  Ollama available: {agent.ollama_available}")

        # Test 1: Natural language goal parsing
        print("\n1. Testing natural language goal parsing...")
        test_inputs = [
            "I want to retire at 60 with $2 million",
            "Help me save $50,000 for emergency fund in 2 years",
            "I need to pay off $30,000 in student debt aggressively"
        ]

        for user_input in test_inputs:
            print(f"\n   Input: '{user_input}'")
            result = await agent.parse_natural_language_goal(
                user_input,
                user_context={"age": 35, "income": 100000}
            )
            print(f"   → Goal type: {result.get('goal_type')}")
            print(f"   → Target amount: ${result.get('target_amount', 0):,.2f}")
            print(f"   → Timeframe: {result.get('timeframe_years', 'N/A')} years")
            print(f"   → Risk tolerance: {result.get('risk_tolerance')}")
            print(f"   → Method: {result.get('parsing_method')}")

        # Test 2: Financial narrative generation
        print("\n2. Testing financial narrative generation...")
        test_plan = {
            "goal_type": "retirement",
            "target_amount": 2000000,
            "timeframe_years": 25,
            "risk_tolerance": "moderate"
        }

        narrative = await agent.generate_financial_narrative(test_plan)
        print(f"\n   Generated Narrative:")
        print(f"   {'-'*76}")
        for line in narrative.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*76}")

        # Test 3: What-if scenario explanation
        print("\n3. Testing what-if scenario explanation...")
        scenario = {
            "type": "market_crash",
            "severity": "high",
            "description": "30% market decline"
        }
        impact = {
            "target_amount_change": -50000,
            "timeframe_change": 2
        }

        explanation = await agent.explain_what_if_scenario(scenario, impact)
        print(f"\n   Scenario Explanation:")
        print(f"   {'-'*76}")
        for line in explanation.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*76}")

        print("\n✓ ConversationalAgent tests PASSED")
        return True

    except Exception as e:
        print(f"\n✗ ConversationalAgent tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_graph_risk_detector():
    """Test GraphRiskDetector (Task 24)"""
    print("\n" + "="*80)
    print("TESTING GRAPH RISK DETECTOR (Task 24)")
    print("="*80)

    try:
        from agents.graph_risk_detector import get_graph_risk_detector

        detector = get_graph_risk_detector()
        print(f"✓ Detector initialized: {detector.agent_id}")

        # Test 1: Build transaction graph
        print("\n1. Testing transaction graph building...")
        mock_transactions = [
            {"from_account": "checking", "merchant": "Amazon", "amount": 150, "timestamp": datetime.now(UTC).isoformat()},
            {"from_account": "checking", "merchant": "Grocery", "amount": 85, "timestamp": datetime.now(UTC).isoformat()},
            {"from_account": "checking", "merchant": "Gas", "amount": 60, "timestamp": datetime.now(UTC).isoformat()},
            {"from_account": "checking", "merchant": "Restaurant", "amount": 45, "timestamp": datetime.now(UTC).isoformat()},
            {"from_account": "checking", "merchant": "Amazon", "amount": 5000, "timestamp": datetime.now(UTC).isoformat()},  # Anomaly
        ]

        graph = await detector.build_transaction_graph(mock_transactions, "test_user")
        print(f"   → Nodes: {graph.number_of_nodes()}")
        print(f"   → Edges: {graph.number_of_edges()}")
        print(f"   → Graph type: {'Directed' if graph.is_directed() else 'Undirected'}")

        # Test 2: Detect anomalies
        print("\n2. Testing anomaly detection...")
        anomalies = await detector.detect_anomalous_patterns(graph, method="statistics")
        print(f"   → Anomalies detected: {len(anomalies)}")
        for idx, anomaly in enumerate(anomalies[:3], 1):
            print(f"   {idx}. {anomaly.get('reason', 'Unknown')}")
            print(f"      Score: {anomaly.get('anomaly_score', 0):.2f}")

        # Test 3: Build asset correlation graph
        print("\n3. Testing asset correlation graph...")
        mock_assets = [
            {"symbol": "AAPL", "type": "stock", "sector": "technology", "allocation": 0.3},
            {"symbol": "MSFT", "type": "stock", "sector": "technology", "allocation": 0.2},
            {"symbol": "BND", "type": "bond", "sector": "fixed_income", "allocation": 0.3},
            {"symbol": "VTI", "type": "stock", "sector": "diversified", "allocation": 0.2}
        ]

        asset_graph = await detector.build_asset_correlation_graph(mock_assets)
        print(f"   → Asset nodes: {asset_graph.number_of_nodes()}")
        print(f"   → Correlation edges: {asset_graph.number_of_edges()}")

        # Test 4: Calculate systemic risk
        print("\n4. Testing systemic risk calculation...")
        risk_assessment = await detector.calculate_systemic_risk(asset_graph)
        print(f"   → Overall risk score: {risk_assessment['overall_risk_score']:.3f}")
        print(f"   → Risk level: {risk_assessment['risk_level']}")
        print(f"   → Recommendations:")
        for idx, rec in enumerate(risk_assessment['recommendations'][:3], 1):
            print(f"      {idx}. {rec}")

        # Test 5: Fraud detection
        print("\n5. Testing fraud detection...")
        fraud_indicators = await detector.find_fraud_indicators(mock_transactions)
        print(f"   → Fraud indicators: {len(fraud_indicators)}")
        for idx, indicator in enumerate(fraud_indicators[:3], 1):
            print(f"   {idx}. Type: {indicator.get('type')}")
            print(f"      Severity: {indicator.get('severity')}")

        # Test 6: Comprehensive transaction risk analysis
        print("\n6. Testing comprehensive transaction risk analysis...")
        risk_report = await detector.analyze_transaction_risk("test_user", 90)
        print(f"   → Overall risk score: {risk_report['overall_risk_score']:.3f}")
        print(f"   → Risk level: {risk_report['risk_level']}")
        print(f"   → Transactions analyzed: {risk_report['transaction_count']}")
        print(f"   → Anomalies: {len(risk_report['anomalies'])}")
        print(f"   → Fraud indicators: {len(risk_report['fraud_indicators'])}")

        print("\n✓ GraphRiskDetector tests PASSED")
        return True

    except Exception as e:
        print(f"\n✗ GraphRiskDetector tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ml_prediction_engine():
    """Test MLPredictionEngine (Task 25)"""
    print("\n" + "="*80)
    print("TESTING ML PREDICTION ENGINE (Task 25)")
    print("="*80)

    try:
        from agents.ml_prediction_engine import get_ml_prediction_engine

        engine = get_ml_prediction_engine()
        print(f"✓ Engine initialized: {engine.agent_id}")

        # Test 1: Market trend prediction
        print("\n1. Testing market trend prediction...")
        prediction = await engine.predict_market_trend("AAPL", horizon_days=30)
        print(f"   → Symbol: {prediction['symbol']}")
        print(f"   → Current price: ${prediction['current_price']:.2f}")
        print(f"   → Predicted price (30d): ${prediction['predicted_price']:.2f}")
        print(f"   → Price change: ${prediction['price_change']:.2f} ({prediction.get('price_change_percent', 0):.2f}%)")
        print(f"   → Trend: {prediction['trend_direction']}")
        print(f"   → Model: {prediction['model']}")
        print(f"   → Confidence: {prediction['confidence']:.2f}")

        # Test 2: Portfolio performance prediction
        print("\n2. Testing portfolio performance prediction...")
        test_portfolio = {
            "total_value": 100000,
            "assets": [
                {"symbol": "AAPL", "allocation": 0.4, "expected_return": 0.10, "volatility": 0.20, "type": "stock"},
                {"symbol": "BND", "allocation": 0.6, "expected_return": 0.04, "volatility": 0.05, "type": "bond"}
            ]
        }

        perf_prediction = await engine.predict_portfolio_performance(test_portfolio, timeframe_days=365)
        print(f"   → Current value: ${perf_prediction['current_value']:,.2f}")
        print(f"   → Expected value (1y): ${perf_prediction['predictions']['expected_value']:,.2f}")
        print(f"   → Expected return: {perf_prediction['predictions']['expected_return']*100:.2f}%")
        print(f"   → 95th percentile: ${perf_prediction['predictions']['percentile_95']:,.2f}")
        print(f"   → 5th percentile: ${perf_prediction['predictions']['percentile_5']:,.2f}")
        print(f"   → Sharpe ratio: {perf_prediction['risk_metrics']['sharpe_ratio']:.2f}")

        # Test 3: Anomaly detection
        print("\n3. Testing market anomaly detection...")
        # Generate mock data with anomalies
        mock_market_data = []
        base_price = 100
        for i in range(50):
            price = base_price + (i * 0.5)  # Gradual increase
            if i == 25:  # Add anomaly
                price = price * 1.5
            mock_market_data.append({
                "date": (datetime.now(UTC) - timedelta(days=50-i)).strftime('%Y-%m-%d'),
                "price": price,
                "volume": 2000000,
                "volatility": 0.02
            })

        anomaly_result = await engine.detect_market_anomaly(mock_market_data)
        print(f"   → Total data points: {anomaly_result['total_points']}")
        print(f"   → Anomalies detected: {anomaly_result['anomalies_detected']}")
        print(f"   → Anomaly rate: {anomaly_result['anomaly_rate']*100:.2f}%")
        print(f"   → Model: {anomaly_result['model']}")

        # Test 4: Generate recommendations
        print("\n4. Testing personalized recommendations...")
        test_user_profile = {
            "age": 35,
            "risk_tolerance": "moderate",
            "monthly_expenses": 5000,
            "emergency_fund": 20000,
            "retirement_contributions": 0.10,
            "tax_bracket": 24,
            "goals": ["retirement", "emergency_fund"],
            "portfolio": test_portfolio
        }

        recommendations = await engine.generate_recommendations(test_user_profile)
        print(f"   → Total recommendations: {len(recommendations)}")
        for idx, rec in enumerate(recommendations[:3], 1):
            print(f"\n   {idx}. {rec['title']}")
            print(f"      Type: {rec['type']}")
            print(f"      Priority: {rec['priority']}")
            print(f"      Confidence: {rec['confidence']:.2f}")
            print(f"      Impact: {rec['estimated_impact']}")

        # Test 5: Risk prediction for scenarios
        print("\n5. Testing risk level predictions...")
        scenarios = ["market_crash", "recession", "inflation_spike", "bull_market"]
        risk_predictions = await engine.predict_risk_levels(test_portfolio, scenarios)
        print(f"   → Portfolio value: ${risk_predictions['portfolio_value']:,.2f}")
        print(f"   → Overall risk level: {risk_predictions['overall_risk_level']}")
        print(f"\n   Scenario Analysis:")
        for scenario, prediction in risk_predictions['scenarios_analyzed'].items():
            print(f"   - {scenario.replace('_', ' ').title()}:")
            print(f"     Expected return: {prediction['expected_return']*100:.2f}%")
            print(f"     Risk level: {prediction['risk_level']}")
            print(f"     Probability: {prediction['probability']*100:.0f}%")

        print("\n✓ MLPredictionEngine tests PASSED")
        return True

    except Exception as e:
        print(f"\n✗ MLPredictionEngine tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all Phase 6 agent tests"""
    print("\n" + "#"*80)
    print("# PHASE 6 AGENTS TEST SUITE")
    print("# Tasks 23, 24, 25: Advanced AI Features")
    print("#"*80)

    results = {
        "conversational_agent": False,
        "graph_risk_detector": False,
        "ml_prediction_engine": False
    }

    # Run tests
    results["conversational_agent"] = await test_conversational_agent()
    results["graph_risk_detector"] = await test_graph_risk_detector()
    results["ml_prediction_engine"] = await test_ml_prediction_engine()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = all(results.values())

    for agent, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{agent.replace('_', ' ').title()}: {status}")

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL PHASE 6 TESTS PASSED!")
        print("="*80)
        print("\nPhase 6 Implementation Complete:")
        print("  • Task 23: Conversational AI (LLM-based NL parsing & narratives)")
        print("  • Task 24: Graph Risk Detection (NetworkX-based fraud & risk analysis)")
        print("  • Task 25: ML Predictions (Market trends, portfolio forecasting, anomaly detection)")
        print("\nNext Steps:")
        print("  1. Start FastAPI server to test API endpoints")
        print("  2. Integrate with frontend visualization")
        print("  3. Connect to real market data sources")
        print("  4. (Optional) Upgrade to NVIDIA NIM for production LLM")
        print("  5. (Optional) Migrate to Neo4j + cuGraph for larger graphs")
    else:
        print("✗ SOME TESTS FAILED - Review errors above")

    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
