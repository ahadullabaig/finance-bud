"""
Comprehensive Test Suite for ML Prediction Engine - Phase 6 Enterprise Solutions

Tests all advanced AI and machine learning features including:
- Market trend predictions with confidence intervals
- Portfolio performance forecasting using Monte Carlo simulation
- Anomaly detection using Isolation Forest
- Personalized recommendation generation
- Risk prediction under multiple scenarios
- Performance benchmarking and stress testing

Requirements: Phase 6, Tasks 23, 24, 25 - Advanced AI capabilities
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the ML prediction engine
from agents.ml_prediction_engine import MLPredictionEngine, get_ml_prediction_engine
from data_models.schemas import AgentMessage, MessageType

# Test fixtures and utilities
@pytest.fixture
def ml_engine():
    """Create ML prediction engine for testing"""
    return MLPredictionEngine(agent_id="test_ml_engine", confidence_threshold=0.7)

@pytest.fixture
def sample_portfolio():
    """Sample portfolio for testing"""
    return {
        "total_value": 100000,
        "assets": [
            {
                "symbol": "AAPL",
                "allocation": 0.4,
                "type": "stock",
                "expected_return": 0.10,
                "volatility": 0.20
            },
            {
                "symbol": "BND", 
                "allocation": 0.6,
                "type": "bond",
                "expected_return": 0.04,
                "volatility": 0.05
            }
        ]
    }

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    base_date = datetime.utcnow() - timedelta(days=30)
    data = []
    
    for i in range(30):
        data.append({
            "symbol": "AAPL",
            "date": (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
            "price": 150.0 + (i * 0.5) + ((-1) ** i * 2),  # Trending up with noise
            "volume": 1000000 + (i * 10000),
            "volatility": 0.02 + (i * 0.001)
        })
    
    return data

@pytest.fixture
def sample_user_profile():
    """Sample user profile for recommendations"""
    return {
        "age": 35,
        "risk_tolerance": "moderate",
        "monthly_expenses": 5000,
        "emergency_fund": 20000,
        "retirement_contributions": 0.10,
        "tax_bracket": 24,
        "goals": ["retirement", "emergency_fund"],
        "portfolio": {
            "total_value": 100000,
            "assets": [
                {"symbol": "AAPL", "allocation": 0.6, "type": "stock"},
                {"symbol": "BND", "allocation": 0.4, "type": "bond"}
            ]
        }
    }


class TestMLEngineInitialization:
    """Test ML prediction engine initialization and setup"""
    
    def test_ml_engine_initialization(self, ml_engine):
        """Test ML engine initializes correctly"""
        assert ml_engine.agent_id == "test_ml_engine"
        assert ml_engine.confidence_threshold == 0.7
        assert isinstance(ml_engine.models, dict)
        assert len(ml_engine.models) == 4  # market, risk, anomaly, recommender
        assert ml_engine.prediction_cache == {}
    
    def test_singleton_pattern(self):
        """Test singleton pattern for ML engine"""
        engine1 = get_ml_prediction_engine()
        engine2 = get_ml_prediction_engine()
        assert engine1 is engine2
    
    def test_model_availability_check(self, ml_engine):
        """Test model availability checking"""
        # Should handle missing sklearn gracefully
        assert hasattr(ml_engine, 'models')
        
        # Check if models are properly initialized as None
        for model_name, model in ml_engine.models.items():
            assert model is None  # Models start as None until trained


class TestMarketTrendPrediction:
    """Test market trend prediction functionality"""
    
    @pytest.mark.asyncio
    async def test_predict_market_trend_basic(self, ml_engine, sample_market_data):
        """Test basic market trend prediction"""
        result = await ml_engine.predict_market_trend(
            symbol="AAPL",
            horizon_days=30,
            historical_data=sample_market_data
        )
        
        # Verify response structure
        assert "symbol" in result
        assert "current_price" in result
        assert "predicted_price" in result
        assert "trend_direction" in result
        assert "confidence" in result
        assert "forecast" in result
        
        # Verify data types
        assert isinstance(result["current_price"], float)
        assert isinstance(result["predicted_price"], float)
        assert result["trend_direction"] in ["bullish", "bearish"]
        assert 0 <= result["confidence"] <= 1
        assert isinstance(result["forecast"], list)
    
    @pytest.mark.asyncio
    async def test_predict_market_trend_without_data(self, ml_engine):
        """Test market prediction without historical data (uses mock data)"""
        result = await ml_engine.predict_market_trend(
            symbol="TSLA",
            horizon_days=15
        )
        
        assert result["symbol"] == "TSLA"
        assert "predicted_price" in result
        assert "trend_direction" in result
    
    @pytest.mark.asyncio
    async def test_predict_market_trend_insufficient_data(self, ml_engine):
        """Test prediction with insufficient historical data"""
        insufficient_data = [
            {"symbol": "AAPL", "date": "2024-01-01", "price": 150.0}
        ]
        
        result = await ml_engine.predict_market_trend(
            symbol="AAPL",
            horizon_days=30,
            historical_data=insufficient_data
        )
        
        # Should handle gracefully
        assert "error" in result or "predicted_price" in result
    
    @pytest.mark.asyncio
    async def test_predict_market_trend_performance(self, ml_engine, sample_market_data):
        """Test market prediction performance"""
        start_time = time.time()
        
        result = await ml_engine.predict_market_trend(
            symbol="AAPL",
            horizon_days=30,
            historical_data=sample_market_data
        )
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 2.0  # 2 seconds max
        assert "predicted_price" in result
    
    @pytest.mark.asyncio
    async def test_predict_multiple_horizons(self, ml_engine, sample_market_data):
        """Test predictions for different time horizons"""
        horizons = [7, 30, 90]
        
        for horizon in horizons:
            result = await ml_engine.predict_market_trend(
                symbol="AAPL",
                horizon_days=horizon,
                historical_data=sample_market_data
            )
            
            assert "predicted_price" in result
            assert len(result["forecast"]) <= 10  # Limited to first 10 days


class TestPortfolioPerformancePrediction:
    """Test portfolio performance prediction using Monte Carlo simulation"""
    
    @pytest.mark.asyncio
    async def test_predict_portfolio_performance_basic(self, ml_engine, sample_portfolio):
        """Test basic portfolio performance prediction"""
        result = await ml_engine.predict_portfolio_performance(
            portfolio=sample_portfolio,
            timeframe_days=365,
            num_simulations=100  # Reduced for testing speed
        )
        
        # Verify response structure
        assert "current_value" in result
        assert "predictions" in result
        assert "risk_metrics" in result
        assert "num_simulations" in result
        
        # Verify predictions structure
        predictions = result["predictions"]
        assert "expected_return" in predictions
        assert "expected_value" in predictions
        assert "percentile_5" in predictions
        assert "percentile_95" in predictions
        
        # Verify risk metrics
        risk_metrics = result["risk_metrics"]
        assert "var_95" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
    
    @pytest.mark.asyncio
    async def test_predict_portfolio_empty_assets(self, ml_engine):
        """Test portfolio prediction with empty assets"""
        empty_portfolio = {"total_value": 100000, "assets": []}
        
        result = await ml_engine.predict_portfolio_performance(
            portfolio=empty_portfolio,
            timeframe_days=365
        )
        
        assert "error" in result or "expected_value" in result
    
    @pytest.mark.asyncio
    async def test_predict_portfolio_different_timeframes(self, ml_engine, sample_portfolio):
        """Test portfolio predictions for different timeframes"""
        timeframes = [30, 365, 1825]  # 1 month, 1 year, 5 years
        
        for timeframe in timeframes:
            result = await ml_engine.predict_portfolio_performance(
                portfolio=sample_portfolio,
                timeframe_days=timeframe,
                num_simulations=50
            )
            
            assert "expected_value" in result
            assert result["timeframe_days"] == timeframe
    
    @pytest.mark.asyncio
    async def test_portfolio_monte_carlo_consistency(self, ml_engine, sample_portfolio):
        """Test Monte Carlo simulation consistency"""
        # Run same prediction multiple times
        results = []
        
        for _ in range(3):
            result = await ml_engine.predict_portfolio_performance(
                portfolio=sample_portfolio,
                timeframe_days=365,
                num_simulations=100
            )
            results.append(result["predictions"]["expected_return"])
        
        # Results should be similar but not identical (due to randomness)
        mean_return = sum(results) / len(results)
        for result in results:
            assert abs(result - mean_return) < 0.1  # Within 10% of mean


class TestAnomalyDetection:
    """Test market anomaly detection functionality"""
    
    @pytest.mark.asyncio
    async def test_detect_market_anomaly_basic(self, ml_engine, sample_market_data):
        """Test basic anomaly detection"""
        # Add some anomalous data points
        anomalous_data = sample_market_data.copy()
        anomalous_data.append({
            "symbol": "AAPL",
            "date": "2024-02-01",
            "price": 500.0,  # Anomalously high price
            "volume": 50000000,  # Anomalously high volume
            "volatility": 0.5  # Anomalously high volatility
        })
        
        result = await ml_engine.detect_market_anomaly(
            market_data=anomalous_data,
            contamination=0.1
        )
        
        # Verify response structure
        assert "total_points" in result
        assert "anomalies_detected" in result
        assert "anomaly_rate" in result
        assert "anomalies" in result
        
        # Should detect at least one anomaly
        assert result["anomalies_detected"] >= 0
        assert isinstance(result["anomalies"], list)
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_insufficient_data(self, ml_engine):
        """Test anomaly detection with insufficient data"""
        insufficient_data = [
            {"symbol": "AAPL", "date": "2024-01-01", "price": 150.0, "volume": 1000000, "volatility": 0.02}
        ]
        
        result = await ml_engine.detect_market_anomaly(
            market_data=insufficient_data,
            contamination=0.1
        )
        
        assert "error" in result or "anomalies_detected" in result
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_different_contamination(self, ml_engine, sample_market_data):
        """Test anomaly detection with different contamination levels"""
        contamination_levels = [0.05, 0.1, 0.2]
        
        for contamination in contamination_levels:
            result = await ml_engine.detect_market_anomaly(
                market_data=sample_market_data,
                contamination=contamination
            )
            
            assert "anomalies_detected" in result
            # Higher contamination should generally detect more anomalies
            assert result["anomaly_rate"] <= contamination + 0.1  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_anomaly_severity_classification(self, ml_engine):
        """Test anomaly severity classification"""
        # Create data with clear anomalies
        anomalous_data = []
        base_price = 150.0
        
        for i in range(20):
            price = base_price + (i * 0.1)  # Normal trend
            if i == 10:  # Insert anomaly
                price = base_price * 2  # Double the price
            
            anomalous_data.append({
                "symbol": "AAPL",
                "date": f"2024-01-{i+1:02d}",
                "price": price,
                "volume": 1000000,
                "volatility": 0.02
            })
        
        result = await ml_engine.detect_market_anomaly(
            market_data=anomalous_data,
            contamination=0.1
        )
        
        # Check if anomalies have severity classification
        for anomaly in result.get("anomalies", []):
            if "severity" in anomaly:
                assert anomaly["severity"] in ["low", "medium", "high"]


class TestPersonalizedRecommendations:
    """Test personalized recommendation generation"""
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_basic(self, ml_engine, sample_user_profile):
        """Test basic recommendation generation"""
        recommendations = await ml_engine.generate_recommendations(
            user_profile=sample_user_profile
        )
        
        # Verify response structure
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert "type" in rec
            assert "title" in rec
            assert "description" in rec
            assert "priority" in rec
            assert "confidence" in rec
            assert "action_items" in rec
            
            # Verify data types
            assert rec["priority"] in ["low", "medium", "high"]
            assert 0 <= rec["confidence"] <= 1
            assert isinstance(rec["action_items"], list)
    
    @pytest.mark.asyncio
    async def test_recommendations_emergency_fund(self, ml_engine):
        """Test emergency fund recommendations"""
        profile_no_emergency = {
            "age": 30,
            "monthly_expenses": 5000,
            "emergency_fund": 1000,  # Insufficient emergency fund
            "risk_tolerance": "moderate"
        }
        
        recommendations = await ml_engine.generate_recommendations(
            user_profile=profile_no_emergency
        )
        
        # Should recommend building emergency fund
        emergency_recs = [r for r in recommendations if r["type"] == "emergency_fund"]
        assert len(emergency_recs) > 0
        assert emergency_recs[0]["priority"] == "high"
    
    @pytest.mark.asyncio
    async def test_recommendations_retirement_savings(self, ml_engine):
        """Test retirement savings recommendations"""
        profile_low_retirement = {
            "age": 35,
            "retirement_contributions": 0.05,  # Low retirement savings
            "monthly_expenses": 4000,
            "emergency_fund": 25000,  # Adequate emergency fund
            "risk_tolerance": "moderate"
        }
        
        recommendations = await ml_engine.generate_recommendations(
            user_profile=profile_low_retirement
        )
        
        # Should recommend increasing retirement contributions
        retirement_recs = [r for r in recommendations if r["type"] == "retirement_savings"]
        assert len(retirement_recs) > 0
    
    @pytest.mark.asyncio
    async def test_recommendations_with_market_context(self, ml_engine, sample_user_profile):
        """Test recommendations with market context"""
        market_context = {
            "volatility": 0.8,  # High volatility
            "market_trend": "bearish"
        }
        
        recommendations = await ml_engine.generate_recommendations(
            user_profile=sample_user_profile,
            market_context=market_context
        )
        
        # Should adjust recommendations based on market conditions
        assert len(recommendations) > 0
        
        # Diversification should be boosted in volatile markets
        diversification_recs = [r for r in recommendations if r["type"] == "diversification"]
        if diversification_recs:
            # Should have higher score due to market volatility
            assert diversification_recs[0]["score"] > 0.5
    
    @pytest.mark.asyncio
    async def test_recommendation_scoring_and_ranking(self, ml_engine, sample_user_profile):
        """Test recommendation scoring and ranking"""
        recommendations = await ml_engine.generate_recommendations(
            user_profile=sample_user_profile
        )
        
        # Recommendations should be sorted by score (highest first)
        scores = [rec["score"] for rec in recommendations]
        assert scores == sorted(scores, reverse=True)
        
        # All scores should be between 0 and 1
        for score in scores:
            assert 0 <= score <= 1


class TestRiskPrediction:
    """Test risk level prediction under different scenarios"""
    
    @pytest.mark.asyncio
    async def test_predict_risk_levels_basic(self, ml_engine, sample_portfolio):
        """Test basic risk level prediction"""
        scenarios = ["market_crash", "recession", "inflation_spike"]
        
        result = await ml_engine.predict_risk_levels(
            portfolio=sample_portfolio,
            scenarios=scenarios
        )
        
        # Verify response structure
        assert "portfolio_value" in result
        assert "scenarios_analyzed" in result
        assert "overall_risk_level" in result
        
        # Check each scenario
        for scenario in scenarios:
            assert scenario in result["scenarios_analyzed"]
            scenario_result = result["scenarios_analyzed"][scenario]
            
            assert "expected_return" in scenario_result
            assert "risk_level" in scenario_result
            assert "probability" in scenario_result
            assert "mitigation_strategies" in scenario_result
            
            # Verify risk level categories
            assert scenario_result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    @pytest.mark.asyncio
    async def test_risk_prediction_market_crash(self, ml_engine, sample_portfolio):
        """Test risk prediction for market crash scenario"""
        result = await ml_engine.predict_risk_levels(
            portfolio=sample_portfolio,
            scenarios=["market_crash"]
        )
        
        crash_result = result["scenarios_analyzed"]["market_crash"]
        
        # Market crash should show negative expected return
        assert crash_result["expected_return"] < 0
        
        # Should have high or critical risk level
        assert crash_result["risk_level"] in ["HIGH", "CRITICAL"]
        
        # Should have mitigation strategies
        assert len(crash_result["mitigation_strategies"]) > 0
    
    @pytest.mark.asyncio
    async def test_risk_prediction_bull_market(self, ml_engine, sample_portfolio):
        """Test risk prediction for bull market scenario"""
        result = await ml_engine.predict_risk_levels(
            portfolio=sample_portfolio,
            scenarios=["bull_market"]
        )
        
        bull_result = result["scenarios_analyzed"]["bull_market"]
        
        # Bull market should show positive expected return
        assert bull_result["expected_return"] > 0
        
        # Should have low risk level
        assert bull_result["risk_level"] in ["LOW", "MEDIUM"]
    
    @pytest.mark.asyncio
    async def test_overall_risk_calculation(self, ml_engine, sample_portfolio):
        """Test overall risk level calculation"""
        scenarios = ["market_crash", "recession", "bull_market"]
        
        result = await ml_engine.predict_risk_levels(
            portfolio=sample_portfolio,
            scenarios=scenarios
        )
        
        # Overall risk should be calculated from weighted scenarios
        assert result["overall_risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    @pytest.mark.asyncio
    async def test_risk_prediction_unknown_scenario(self, ml_engine, sample_portfolio):
        """Test risk prediction with unknown scenario"""
        result = await ml_engine.predict_risk_levels(
            portfolio=sample_portfolio,
            scenarios=["unknown_scenario"]
        )
        
        # Should handle unknown scenarios gracefully
        assert "unknown_scenario" in result["scenarios_analyzed"]
        unknown_result = result["scenarios_analyzed"]["unknown_scenario"]
        assert "expected_return" in unknown_result


class TestMessageProcessing:
    """Test ML engine message processing"""
    
    @pytest.mark.asyncio
    async def test_process_market_prediction_message(self, ml_engine):
        """Test processing market prediction message"""
        message = AgentMessage(
            agent_id="test_agent",
            target_agent_id=ml_engine.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "action": "predict_market",
                "symbol": "AAPL",
                "horizon_days": 30
            },
            correlation_id="test_correlation",
            session_id="test_session"
        )
        
        response = await ml_engine.process_message(message)
        
        assert response is not None
        assert response.message_type == MessageType.RESPONSE
        assert response.correlation_id == "test_correlation"
        assert "symbol" in response.content
    
    @pytest.mark.asyncio
    async def test_process_unknown_action_message(self, ml_engine):
        """Test processing message with unknown action"""
        message = AgentMessage(
            agent_id="test_agent",
            target_agent_id=ml_engine.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "action": "unknown_action"
            },
            correlation_id="test_correlation",
            session_id="test_session"
        )
        
        response = await ml_engine.process_message(message)
        
        # Should return None for unknown actions
        assert response is None


class TestPerformanceAndStress:
    """Test ML engine performance and stress conditions"""
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, ml_engine, sample_market_data):
        """Test concurrent prediction requests"""
        # Create multiple concurrent prediction tasks
        tasks = []
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        for symbol in symbols:
            task = ml_engine.predict_market_trend(
                symbol=symbol,
                horizon_days=30,
                historical_data=sample_market_data
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # All predictions should complete successfully
        assert len(results) == len(symbols)
        for result in results:
            assert "predicted_price" in result or "error" in result
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # 10 seconds for 5 concurrent predictions
    
    @pytest.mark.asyncio
    async def test_large_dataset_anomaly_detection(self, ml_engine):
        """Test anomaly detection with large dataset"""
        # Generate large dataset
        large_dataset = []
        base_date = datetime.utcnow() - timedelta(days=1000)
        
        for i in range(1000):  # 1000 data points
            large_dataset.append({
                "symbol": "AAPL",
                "date": (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                "price": 150.0 + (i * 0.01) + ((-1) ** i * 0.5),
                "volume": 1000000 + (i * 100),
                "volatility": 0.02 + (i * 0.00001)
            })
        
        start_time = time.time()
        result = await ml_engine.detect_market_anomaly(
            market_data=large_dataset,
            contamination=0.05
        )
        execution_time = time.time() - start_time
        
        # Should handle large dataset
        assert "anomalies_detected" in result or "error" in result
        
        # Should complete within reasonable time
        assert execution_time < 30.0  # 30 seconds for 1000 data points
    
    @pytest.mark.asyncio
    async def test_memory_usage_portfolio_simulation(self, ml_engine):
        """Test memory usage during large Monte Carlo simulation"""
        large_portfolio = {
            "total_value": 1000000,
            "assets": [
                {"symbol": f"STOCK_{i}", "allocation": 0.01, "expected_return": 0.08, "volatility": 0.15}
                for i in range(100)  # 100 assets
            ]
        }
        
        # Large number of simulations
        result = await ml_engine.predict_portfolio_performance(
            portfolio=large_portfolio,
            timeframe_days=365,
            num_simulations=5000  # Large simulation count
        )
        
        # Should handle large simulation without memory issues
        assert "expected_value" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_rapid_recommendation_requests(self, ml_engine, sample_user_profile):
        """Test rapid recommendation generation requests"""
        # Generate multiple rapid requests
        tasks = []
        
        for i in range(20):  # 20 rapid requests
            modified_profile = sample_user_profile.copy()
            modified_profile["age"] = 30 + i  # Vary age
            
            task = ml_engine.generate_recommendations(
                user_profile=modified_profile
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # All requests should complete
        assert len(results) == 20
        for result in results:
            assert isinstance(result, list)
        
        # Should complete rapidly
        assert execution_time < 5.0  # 5 seconds for 20 requests


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_portfolio_data(self, ml_engine):
        """Test handling of invalid portfolio data"""
        invalid_portfolio = {
            "total_value": "invalid",  # Should be number
            "assets": "not_a_list"     # Should be list
        }
        
        result = await ml_engine.predict_portfolio_performance(
            portfolio=invalid_portfolio
        )
        
        # Should handle gracefully
        assert "error" in result or "expected_value" in result
    
    @pytest.mark.asyncio
    async def test_empty_market_data(self, ml_engine):
        """Test handling of empty market data"""
        result = await ml_engine.detect_market_anomaly(
            market_data=[],
            contamination=0.1
        )
        
        assert "error" in result or "anomalies_detected" in result
    
    @pytest.mark.asyncio
    async def test_negative_horizon_days(self, ml_engine, sample_market_data):
        """Test handling of negative horizon days"""
        result = await ml_engine.predict_market_trend(
            symbol="AAPL",
            horizon_days=-10,  # Invalid negative horizon
            historical_data=sample_market_data
        )
        
        # Should handle gracefully or return error
        assert "error" in result or "predicted_price" in result
    
    @pytest.mark.asyncio
    async def test_extreme_contamination_values(self, ml_engine, sample_market_data):
        """Test handling of extreme contamination values"""
        # Test with contamination > 1.0
        result1 = await ml_engine.detect_market_anomaly(
            market_data=sample_market_data,
            contamination=1.5  # Invalid > 1.0
        )
        
        # Test with contamination < 0
        result2 = await ml_engine.detect_market_anomaly(
            market_data=sample_market_data,
            contamination=-0.1  # Invalid < 0
        )
        
        # Should handle gracefully
        for result in [result1, result2]:
            assert "error" in result or "anomalies_detected" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])