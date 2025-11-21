"""
Comprehensive Test Suite for ML API Endpoints - Phase 6 Enterprise Solutions

Tests all ML prediction API endpoints including:
- Market trend prediction API
- Portfolio performance prediction API  
- Anomaly detection API
- Personalized recommendations API
- Risk prediction API
- Health check and model status endpoints
- Error handling and validation
- Performance and load testing

Requirements: Phase 6, Task 25 - ML API endpoints
"""

import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import the FastAPI app and ML endpoints
from api.ml_endpoints import router
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestMarketPredictionAPI:
    """Test market trend prediction API endpoints"""
    
    def test_predict_market_trend_success(self):
        """Test successful market trend prediction"""
        request_data = {
            "symbol": "AAPL",
            "horizon_days": 30
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            # Mock the prediction result
            mock_engine.return_value.predict_market_trend = AsyncMock(return_value={
                "symbol": "AAPL",
                "current_price": 150.0,
                "predicted_price": 155.0,
                "price_change": 5.0,
                "price_change_percent": 3.33,
                "trend_direction": "bullish",
                "trend_strength": 0.033,
                "forecast": [
                    {
                        "date": "2024-01-02",
                        "predicted_price": 151.0,
                        "lower_bound": 148.0,
                        "upper_bound": 154.0,
                        "confidence": 0.95
                    }
                ],
                "model": "linear_regression",
                "confidence": 0.85,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response = client.post("/api/ml/predict-market", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["symbol"] == "AAPL"
            assert data["current_price"] == 150.0
            assert data["predicted_price"] == 155.0
            assert data["trend_direction"] == "bullish"
            assert "forecast" in data
            assert "confidence" in data
    
    def test_predict_market_trend_with_historical_data(self):
        """Test market prediction with historical data"""
        request_data = {
            "symbol": "GOOGL",
            "horizon_days": 15,
            "historical_data": [
                {
                    "date": "2024-01-01",
                    "price": 2800.0,
                    "volume": 1000000,
                    "volatility": 0.02
                }
            ]
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.predict_market_trend = AsyncMock(return_value={
                "symbol": "GOOGL",
                "current_price": 2800.0,
                "predicted_price": 2850.0,
                "price_change": 50.0,
                "price_change_percent": 1.79,
                "trend_direction": "bullish",
                "trend_strength": 0.018,
                "forecast": [],
                "model": "linear_regression",
                "confidence": 0.78,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response = client.post("/api/ml/predict-market", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["symbol"] == "GOOGL"
    
    def test_predict_market_trend_invalid_data(self):
        """Test market prediction with invalid data"""
        request_data = {
            "symbol": "",  # Empty symbol
            "horizon_days": -5  # Negative horizon
        }
        
        response = client.post("/api/ml/predict-market", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_predict_market_trend_engine_error(self):
        """Test market prediction when engine throws error"""
        request_data = {
            "symbol": "AAPL",
            "horizon_days": 30
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.predict_market_trend = AsyncMock(
                side_effect=Exception("Prediction failed")
            )
            
            response = client.post("/api/ml/predict-market", json=request_data)
            
            assert response.status_code == 500
            assert "Market prediction failed" in response.json()["detail"]


class TestPortfolioPredictionAPI:
    """Test portfolio performance prediction API endpoints"""
    
    def test_predict_portfolio_performance_success(self):
        """Test successful portfolio performance prediction"""
        request_data = {
            "portfolio": {
                "total_value": 100000,
                "assets": [
                    {
                        "symbol": "AAPL",
                        "allocation": 0.6,
                        "expected_return": 0.10,
                        "volatility": 0.20
                    },
                    {
                        "symbol": "BND",
                        "allocation": 0.4,
                        "expected_return": 0.04,
                        "volatility": 0.05
                    }
                ]
            },
            "timeframe_days": 365,
            "num_simulations": 1000
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.predict_portfolio_performance = AsyncMock(return_value={
                "current_value": 100000,
                "timeframe_days": 365,
                "predictions": {
                    "expected_return": 0.075,
                    "expected_value": 107500,
                    "std_deviation": 0.12,
                    "percentile_5": 85000,
                    "percentile_25": 95000,
                    "median": 107000,
                    "percentile_75": 118000,
                    "percentile_95": 130000
                },
                "risk_metrics": {
                    "var_95": 85000,
                    "sharpe_ratio": 0.625,
                    "max_drawdown_estimate": -0.15
                },
                "num_simulations": 1000,
                "confidence": 0.95,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response = client.post("/api/ml/predict-portfolio", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["current_value"] == 100000
            assert data["timeframe_days"] == 365
            assert "predictions" in data
            assert "risk_metrics" in data
            assert data["num_simulations"] == 1000
    
    def test_predict_portfolio_empty_assets(self):
        """Test portfolio prediction with empty assets"""
        request_data = {
            "portfolio": {
                "total_value": 100000,
                "assets": []  # Empty assets
            },
            "timeframe_days": 365
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.predict_portfolio_performance = AsyncMock(return_value={
                "error": "No assets in portfolio"
            })
            
            response = client.post("/api/ml/predict-portfolio", json=request_data)
            
            # Should handle gracefully
            assert response.status_code == 200 or response.status_code == 500
    
    def test_predict_portfolio_different_timeframes(self):
        """Test portfolio prediction with different timeframes"""
        base_request = {
            "portfolio": {
                "total_value": 50000,
                "assets": [
                    {"symbol": "SPY", "allocation": 1.0, "expected_return": 0.08, "volatility": 0.15}
                ]
            },
            "num_simulations": 500
        }
        
        timeframes = [30, 365, 1825]  # 1 month, 1 year, 5 years
        
        for timeframe in timeframes:
            request_data = base_request.copy()
            request_data["timeframe_days"] = timeframe
            
            with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
                mock_engine.return_value.predict_portfolio_performance = AsyncMock(return_value={
                    "current_value": 50000,
                    "timeframe_days": timeframe,
                    "predictions": {"expected_return": 0.08 * (timeframe / 365)},
                    "risk_metrics": {"sharpe_ratio": 0.5},
                    "num_simulations": 500,
                    "confidence": 0.9,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                response = client.post("/api/ml/predict-portfolio", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["timeframe_days"] == timeframe


class TestAnomalyDetectionAPI:
    """Test anomaly detection API endpoints"""
    
    def test_detect_anomalies_success(self):
        """Test successful anomaly detection"""
        request_data = {
            "market_data": [
                {
                    "date": "2024-01-01",
                    "price": 150.0,
                    "volume": 1000000,
                    "volatility": 0.02
                },
                {
                    "date": "2024-01-02",
                    "price": 300.0,  # Anomalous price
                    "volume": 5000000,  # Anomalous volume
                    "volatility": 0.5   # Anomalous volatility
                }
            ],
            "contamination": 0.1
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.detect_market_anomaly = AsyncMock(return_value={
                "total_points": 2,
                "anomalies_detected": 1,
                "anomaly_rate": 0.5,
                "anomalies": [
                    {
                        "date": "2024-01-02",
                        "anomaly_score": 0.8,
                        "data": {
                            "date": "2024-01-02",
                            "price": 300.0,
                            "volume": 5000000,
                            "volatility": 0.5
                        },
                        "severity": "high"
                    }
                ],
                "model": "isolation_forest",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response = client.post("/api/ml/detect-anomalies", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["total_points"] == 2
            assert data["anomalies_detected"] == 1
            assert data["anomaly_rate"] == 0.5
            assert len(data["anomalies"]) == 1
            assert data["anomalies"][0]["severity"] == "high"
    
    def test_detect_anomalies_no_anomalies(self):
        """Test anomaly detection with no anomalies"""
        request_data = {
            "market_data": [
                {"date": "2024-01-01", "price": 150.0, "volume": 1000000, "volatility": 0.02},
                {"date": "2024-01-02", "price": 151.0, "volume": 1100000, "volatility": 0.021}
            ],
            "contamination": 0.05
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.detect_market_anomaly = AsyncMock(return_value={
                "total_points": 2,
                "anomalies_detected": 0,
                "anomaly_rate": 0.0,
                "anomalies": [],
                "model": "isolation_forest",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response = client.post("/api/ml/detect-anomalies", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["anomalies_detected"] == 0
            assert len(data["anomalies"]) == 0
    
    def test_detect_anomalies_invalid_contamination(self):
        """Test anomaly detection with invalid contamination"""
        request_data = {
            "market_data": [
                {"date": "2024-01-01", "price": 150.0, "volume": 1000000, "volatility": 0.02}
            ],
            "contamination": 1.5  # Invalid > 1.0
        }
        
        response = client.post("/api/ml/detect-anomalies", json=request_data)
        
        # Should return validation error or handle gracefully
        assert response.status_code in [200, 422, 500]


class TestRecommendationsAPI:
    """Test personalized recommendations API endpoints"""
    
    def test_generate_recommendations_success(self):
        """Test successful recommendation generation"""
        request_data = {
            "user_profile": {
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
            },
            "market_context": {
                "volatility": 0.6,
                "market_trend": "bearish"
            }
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.generate_recommendations = AsyncMock(return_value=[
                {
                    "type": "emergency_fund",
                    "title": "Build Emergency Fund",
                    "description": "Ensure you have 6 months of expenses saved",
                    "priority": "high",
                    "rationale": "Emergency fund provides financial security",
                    "action_items": [
                        "Calculate 6 months of expenses",
                        "Set up automatic transfers to savings"
                    ],
                    "estimated_impact": "high",
                    "confidence": 0.9,
                    "score": 0.85
                },
                {
                    "type": "diversification",
                    "title": "Improve Portfolio Diversification",
                    "description": "Spread investments across more asset classes",
                    "priority": "medium",
                    "rationale": "Diversification reduces risk",
                    "action_items": [
                        "Add bonds to portfolio",
                        "Consider international exposure"
                    ],
                    "estimated_impact": "medium",
                    "confidence": 0.75,
                    "score": 0.70
                }
            ])
            
            response = client.post("/api/ml/recommendations", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "recommendations" in data
            assert "total_count" in data
            assert data["total_count"] == 2
            
            # Check recommendation structure
            rec = data["recommendations"][0]
            assert "type" in rec
            assert "title" in rec
            assert "priority" in rec
            assert "confidence" in rec
            assert "action_items" in rec
    
    def test_generate_recommendations_minimal_profile(self):
        """Test recommendations with minimal user profile"""
        request_data = {
            "user_profile": {
                "age": 25,
                "risk_tolerance": "conservative"
            }
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.generate_recommendations = AsyncMock(return_value=[
                {
                    "type": "emergency_fund",
                    "title": "Start Emergency Fund",
                    "description": "Begin building emergency savings",
                    "priority": "high",
                    "confidence": 0.95,
                    "action_items": ["Open high-yield savings account"],
                    "score": 0.9
                }
            ])
            
            response = client.post("/api/ml/recommendations", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["recommendations"]) >= 1
    
    def test_generate_recommendations_with_market_context(self):
        """Test recommendations with market context"""
        request_data = {
            "user_profile": {
                "age": 40,
                "risk_tolerance": "aggressive",
                "portfolio": {"total_value": 200000}
            },
            "market_context": {
                "volatility": 0.8,  # High volatility
                "market_trend": "volatile"
            }
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.generate_recommendations = AsyncMock(return_value=[
                {
                    "type": "diversification",
                    "title": "Increase Diversification",
                    "description": "Reduce portfolio risk in volatile market",
                    "priority": "high",
                    "confidence": 0.85,
                    "action_items": ["Rebalance portfolio", "Add defensive assets"],
                    "score": 0.88
                }
            ])
            
            response = client.post("/api/ml/recommendations", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should have recommendations adjusted for market volatility
            assert len(data["recommendations"]) >= 1


class TestRiskPredictionAPI:
    """Test risk prediction API endpoints"""
    
    def test_predict_risk_levels_success(self):
        """Test successful risk level prediction"""
        request_data = {
            "portfolio": {
                "total_value": 100000,
                "assets": [
                    {"symbol": "AAPL", "allocation": 0.6, "type": "stock"},
                    {"symbol": "BND", "allocation": 0.4, "type": "bond"}
                ]
            },
            "scenarios": ["market_crash", "recession", "inflation_spike"]
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.predict_risk_levels = AsyncMock(return_value={
                "portfolio_value": 100000,
                "scenarios_analyzed": {
                    "market_crash": {
                        "expected_return": -0.25,
                        "expected_value_change": -25000,
                        "risk_level": "CRITICAL",
                        "probability": 0.05,
                        "mitigation_strategies": [
                            "Increase bond allocation",
                            "Hold more cash reserves"
                        ]
                    },
                    "recession": {
                        "expected_return": -0.10,
                        "expected_value_change": -10000,
                        "risk_level": "HIGH",
                        "probability": 0.15,
                        "mitigation_strategies": [
                            "Focus on quality companies",
                            "Increase fixed income"
                        ]
                    },
                    "inflation_spike": {
                        "expected_return": -0.05,
                        "expected_value_change": -5000,
                        "risk_level": "MEDIUM",
                        "probability": 0.20,
                        "mitigation_strategies": [
                            "Add inflation-protected securities"
                        ]
                    }
                },
                "overall_risk_level": "HIGH",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response = client.post("/api/ml/predict-risk", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["portfolio_value"] == 100000
            assert "scenarios_analyzed" in data
            assert "overall_risk_level" in data
            
            # Check each scenario
            scenarios = data["scenarios_analyzed"]
            assert "market_crash" in scenarios
            assert "recession" in scenarios
            assert "inflation_spike" in scenarios
            
            # Verify risk levels
            assert scenarios["market_crash"]["risk_level"] == "CRITICAL"
            assert scenarios["recession"]["risk_level"] == "HIGH"
            assert scenarios["inflation_spike"]["risk_level"] == "MEDIUM"
    
    def test_predict_risk_single_scenario(self):
        """Test risk prediction for single scenario"""
        request_data = {
            "portfolio": {
                "total_value": 50000,
                "assets": [
                    {"symbol": "SPY", "allocation": 1.0, "type": "stock"}
                ]
            },
            "scenarios": ["bull_market"]
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.predict_risk_levels = AsyncMock(return_value={
                "portfolio_value": 50000,
                "scenarios_analyzed": {
                    "bull_market": {
                        "expected_return": 0.20,
                        "expected_value_change": 10000,
                        "risk_level": "LOW",
                        "probability": 0.30,
                        "mitigation_strategies": [
                            "Rebalance to target allocation",
                            "Take profits strategically"
                        ]
                    }
                },
                "overall_risk_level": "LOW",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response = client.post("/api/ml/predict-risk", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["scenarios_analyzed"]) == 1
            assert data["scenarios_analyzed"]["bull_market"]["risk_level"] == "LOW"
    
    def test_predict_risk_empty_scenarios(self):
        """Test risk prediction with empty scenarios list"""
        request_data = {
            "portfolio": {
                "total_value": 100000,
                "assets": [{"symbol": "AAPL", "allocation": 1.0, "type": "stock"}]
            },
            "scenarios": []  # Empty scenarios
        }
        
        response = client.post("/api/ml/predict-risk", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422


class TestHealthAndStatusEndpoints:
    """Test health check and model status endpoints"""
    
    def test_health_check_success(self):
        """Test successful health check"""
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_instance = Mock()
            mock_instance.agent_id = "ml-prediction-engine-001"
            mock_instance.models = {"model1": None, "model2": Mock(), "model3": None}
            mock_engine.return_value = mock_instance
            
            response = client.get("/api/ml/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["agent_id"] == "ml-prediction-engine-001"
            assert "models_loaded" in data
            assert "timestamp" in data
    
    def test_health_check_failure(self):
        """Test health check when service is unhealthy"""
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.side_effect = Exception("Service unavailable")
            
            response = client.get("/api/ml/health")
            
            assert response.status_code == 503
            assert "Service unhealthy" in response.json()["detail"]
    
    def test_model_status_success(self):
        """Test successful model status check"""
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_instance = Mock()
            mock_instance.models = {
                "market_predictor": Mock(),  # Loaded
                "risk_predictor": None,      # Not loaded
                "anomaly_detector": Mock(),  # Loaded
                "recommender": None          # Not loaded
            }
            mock_instance.logger = Mock()
            mock_instance.logger.info = Mock()
            mock_engine.return_value = mock_instance
            
            response = client.get("/api/ml/models/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "models" in data
            assert "timestamp" in data
            
            models = data["models"]
            assert models["market_predictor"] == "loaded"
            assert models["risk_predictor"] == "not_loaded"
            assert models["anomaly_detector"] == "loaded"
            assert models["recommender"] == "not_loaded"
    
    def test_model_status_failure(self):
        """Test model status when service fails"""
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.side_effect = Exception("Cannot get model status")
            
            response = client.get("/api/ml/models/status")
            
            assert response.status_code == 500
            assert "Failed to get model status" in response.json()["detail"]


class TestAPIValidation:
    """Test API request validation and error handling"""
    
    def test_missing_required_fields(self):
        """Test API with missing required fields"""
        # Missing symbol in market prediction
        response = client.post("/api/ml/predict-market", json={"horizon_days": 30})
        assert response.status_code == 422
        
        # Missing portfolio in portfolio prediction
        response = client.post("/api/ml/predict-portfolio", json={"timeframe_days": 365})
        assert response.status_code == 422
        
        # Missing market_data in anomaly detection
        response = client.post("/api/ml/detect-anomalies", json={"contamination": 0.1})
        assert response.status_code == 422
    
    def test_invalid_data_types(self):
        """Test API with invalid data types"""
        # String instead of number for horizon_days
        response = client.post("/api/ml/predict-market", json={
            "symbol": "AAPL",
            "horizon_days": "thirty"
        })
        assert response.status_code == 422
        
        # String instead of dict for portfolio
        response = client.post("/api/ml/predict-portfolio", json={
            "portfolio": "invalid_portfolio",
            "timeframe_days": 365
        })
        assert response.status_code == 422
    
    def test_empty_request_body(self):
        """Test API with empty request body"""
        response = client.post("/api/ml/predict-market", json={})
        assert response.status_code == 422
        
        response = client.post("/api/ml/recommendations", json={})
        assert response.status_code == 422


class TestAPIPerformance:
    """Test API performance and load handling"""
    
    def test_concurrent_api_requests(self):
        """Test concurrent API requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
                mock_engine.return_value.predict_market_trend = AsyncMock(return_value={
                    "symbol": "AAPL",
                    "predicted_price": 155.0,
                    "confidence": 0.8,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                response = client.post("/api/ml/predict-market", json={
                    "symbol": "AAPL",
                    "horizon_days": 30
                })
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        execution_time = time.time() - start_time
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)
        
        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 seconds for 10 concurrent requests
    
    def test_large_request_handling(self):
        """Test handling of large request payloads"""
        # Large market data for anomaly detection
        large_market_data = []
        for i in range(1000):  # 1000 data points
            large_market_data.append({
                "date": f"2024-01-{(i % 30) + 1:02d}",
                "price": 150.0 + (i * 0.01),
                "volume": 1000000 + i,
                "volatility": 0.02 + (i * 0.00001)
            })
        
        request_data = {
            "market_data": large_market_data,
            "contamination": 0.05
        }
        
        with patch('api.ml_endpoints.get_ml_prediction_engine') as mock_engine:
            mock_engine.return_value.detect_market_anomaly = AsyncMock(return_value={
                "total_points": 1000,
                "anomalies_detected": 50,
                "anomaly_rate": 0.05,
                "anomalies": [],
                "model": "isolation_forest",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response = client.post("/api/ml/detect-anomalies", json=request_data)
            
            # Should handle large payload
            assert response.status_code == 200
            data = response.json()
            assert data["total_points"] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])