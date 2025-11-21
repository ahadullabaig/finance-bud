"""
ML Prediction API Endpoints - Phase 6, Task 25

FastAPI endpoints for machine learning predictions and insights.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.ml_prediction_engine import get_ml_prediction_engine


router = APIRouter(prefix="/api/ml", tags=["ml"])


# Request/Response Models
class MarketPredictionRequest(BaseModel):
    """Request for market trend prediction"""
    symbol: str = Field(..., description="Stock/asset symbol")
    horizon_days: int = Field(30, description="Prediction horizon in days")
    historical_data: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optional historical data"
    )


class MarketPredictionResponse(BaseModel):
    """Market prediction response"""
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    trend_direction: str
    trend_strength: float
    forecast: List[Dict[str, Any]]
    model: str
    confidence: float
    timestamp: str


class PortfolioPredictionRequest(BaseModel):
    """Request for portfolio performance prediction"""
    portfolio: Dict[str, Any] = Field(..., description="Portfolio composition")
    timeframe_days: int = Field(365, description="Prediction timeframe")
    num_simulations: int = Field(1000, description="Number of Monte Carlo simulations")


class PortfolioPredictionResponse(BaseModel):
    """Portfolio prediction response"""
    current_value: float
    timeframe_days: int
    predictions: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    num_simulations: int
    confidence: float
    timestamp: str


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    market_data: List[Dict[str, Any]] = Field(..., description="Time series market data")
    contamination: float = Field(0.1, description="Expected proportion of anomalies")


class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response"""
    total_points: int
    anomalies_detected: int
    anomaly_rate: float
    anomalies: List[Dict[str, Any]]
    model: str
    timestamp: str


class RecommendationRequest(BaseModel):
    """Request for personalized recommendations"""
    user_profile: Dict[str, Any] = Field(..., description="User financial profile")
    market_context: Optional[Dict[str, Any]] = Field(None, description="Current market conditions")


class RecommendationResponse(BaseModel):
    """Recommendation response"""
    recommendations: List[Dict[str, Any]]
    total_count: int
    timestamp: str


class RiskPredictionRequest(BaseModel):
    """Request for risk level predictions"""
    portfolio: Dict[str, Any] = Field(..., description="Portfolio composition")
    scenarios: List[str] = Field(..., description="Scenario names to analyze")


class RiskPredictionResponse(BaseModel):
    """Risk prediction response"""
    portfolio_value: float
    scenarios_analyzed: Dict[str, Any]
    overall_risk_level: str
    timestamp: str


# Endpoints
@router.post("/predict-market", response_model=MarketPredictionResponse)
async def predict_market_trend(request: MarketPredictionRequest):
    """
    Predict market trend using linear regression and ensemble methods.

    Example:
    ```json
    {
        "symbol": "AAPL",
        "horizon_days": 30
    }
    ```

    Returns predictions with confidence intervals.
    """
    try:
        engine = get_ml_prediction_engine()

        result = await engine.predict_market_trend(
            request.symbol,
            request.horizon_days,
            request.historical_data
        )

        return MarketPredictionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Market prediction failed: {str(e)}"
        )


@router.post("/predict-portfolio", response_model=PortfolioPredictionResponse)
async def predict_portfolio_performance(request: PortfolioPredictionRequest):
    """
    Predict portfolio performance using Monte Carlo simulation.

    Example:
    ```json
    {
        "portfolio": {
            "total_value": 100000,
            "assets": [
                {
                    "symbol": "AAPL",
                    "allocation": 0.4,
                    "expected_return": 0.10,
                    "volatility": 0.20
                },
                {
                    "symbol": "BND",
                    "allocation": 0.6,
                    "expected_return": 0.04,
                    "volatility": 0.05
                }
            ]
        },
        "timeframe_days": 365,
        "num_simulations": 1000
    }
    ```

    Returns expected returns with uncertainty quantification.
    """
    try:
        engine = get_ml_prediction_engine()

        result = await engine.predict_portfolio_performance(
            request.portfolio,
            request.timeframe_days,
            request.num_simulations
        )

        return PortfolioPredictionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio prediction failed: {str(e)}"
        )


@router.post("/detect-anomalies", response_model=AnomalyDetectionResponse)
async def detect_market_anomalies(request: AnomalyDetectionRequest):
    """
    Detect anomalies in market data using Isolation Forest.

    Example:
    ```json
    {
        "market_data": [
            {
                "date": "2024-01-01",
                "price": 150.0,
                "volume": 1000000,
                "volatility": 0.02
            }
        ],
        "contamination": 0.1
    }
    ```

    Returns detected anomalies with severity scores.
    """
    try:
        engine = get_ml_prediction_engine()

        result = await engine.detect_market_anomaly(
            request.market_data,
            request.contamination
        )

        return AnomalyDetectionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@router.post("/recommendations", response_model=RecommendationResponse)
async def generate_recommendations(request: RecommendationRequest):
    """
    Generate personalized financial recommendations.

    Example:
    ```json
    {
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
                "assets": [...]
            }
        },
        "market_context": {
            "volatility": 0.6,
            "market_trend": "bearish"
        }
    }
    ```

    Returns ranked recommendations with action items.
    """
    try:
        engine = get_ml_prediction_engine()

        recommendations = await engine.generate_recommendations(
            request.user_profile,
            request.market_context
        )

        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {str(e)}"
        )


@router.post("/predict-risk", response_model=RiskPredictionResponse)
async def predict_risk_levels(request: RiskPredictionRequest):
    """
    Predict risk levels under different scenarios.

    Example:
    ```json
    {
        "portfolio": {
            "total_value": 100000,
            "assets": [
                {"symbol": "AAPL", "allocation": 0.6, "type": "stock"},
                {"symbol": "BND", "allocation": 0.4, "type": "bond"}
            ]
        },
        "scenarios": ["market_crash", "recession", "inflation_spike"]
    }
    ```

    Returns risk predictions for each scenario.
    """
    try:
        engine = get_ml_prediction_engine()

        result = await engine.predict_risk_levels(
            request.portfolio,
            request.scenarios
        )

        return RiskPredictionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Risk prediction failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        engine = get_ml_prediction_engine()
        return {
            "status": "healthy",
            "agent_id": engine.agent_id,
            "models_loaded": len([m for m in engine.models.values() if m is not None]),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@router.get("/models/status")
async def get_model_status():
    """Get status of all ML models"""
    try:
        engine = get_ml_prediction_engine()

        return {
            "models": {
                name: "loaded" if model is not None else "not_loaded"
                for name, model in engine.models.items()
            },
            "sklearn_available": engine.logger.info("sklearn check"),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )
