"""
Graph Risk Detection API Endpoints - Phase 6, Task 24

FastAPI endpoints for graph-based risk and fraud detection.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.graph_risk_detector import get_graph_risk_detector


router = APIRouter(prefix="/api/risk", tags=["risk"])


# Request/Response Models
class TransactionRiskRequest(BaseModel):
    """Request for transaction risk analysis"""
    user_id: str = Field(..., description="User identifier")
    lookback_days: int = Field(90, description="Days of transaction history to analyze")


class TransactionRiskResponse(BaseModel):
    """Transaction risk analysis response"""
    user_id: str
    analysis_period_days: int
    overall_risk_score: float
    risk_level: str
    transaction_count: int
    graph_stats: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    fraud_indicators: List[Dict[str, Any]]
    timestamp: str


class BuildGraphRequest(BaseModel):
    """Request to build transaction or asset graph"""
    graph_type: str = Field(..., description="Type of graph: 'transaction' or 'asset'")
    data: List[Dict[str, Any]] = Field(..., description="Transaction or asset data")
    user_id: Optional[str] = Field(None, description="User ID for transaction graphs")


class BuildGraphResponse(BaseModel):
    """Graph build response"""
    graph_type: str
    nodes: int
    edges: int
    density: float
    timestamp: str


class SystemicRiskRequest(BaseModel):
    """Request for systemic risk analysis"""
    assets: List[Dict[str, Any]] = Field(..., description="Portfolio assets")
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Pre-computed correlation matrix"
    )


class SystemicRiskResponse(BaseModel):
    """Systemic risk analysis response"""
    overall_risk_score: float
    risk_level: str
    metrics: Dict[str, Any]
    recommendations: List[str]


class FraudCheckRequest(BaseModel):
    """Request for fraud detection"""
    transactions: List[Dict[str, Any]] = Field(..., description="Transactions to analyze")
    user_profile: Optional[Dict[str, Any]] = Field(None, description="User profile for context")


class FraudCheckResponse(BaseModel):
    """Fraud detection response"""
    fraud_indicators_found: int
    indicators: List[Dict[str, Any]]
    highest_confidence: float
    timestamp: str


# Endpoints
@router.post("/analyze-transactions", response_model=TransactionRiskResponse)
async def analyze_transaction_risk(request: TransactionRiskRequest):
    """
    Comprehensive transaction risk analysis using graph algorithms.

    Analyzes:
    - Transaction patterns
    - Anomalous behavior
    - Potential fraud indicators

    Example:
    ```json
    {
        "user_id": "user_12345",
        "lookback_days": 90
    }
    ```
    """
    try:
        detector = get_graph_risk_detector()

        result = await detector.analyze_transaction_risk(
            request.user_id,
            request.lookback_days
        )

        return TransactionRiskResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transaction risk analysis failed: {str(e)}"
        )


@router.post("/build-graph", response_model=BuildGraphResponse)
async def build_graph(request: BuildGraphRequest):
    """
    Build transaction or asset correlation graph.

    Example:
    ```json
    {
        "graph_type": "transaction",
        "data": [
            {
                "from_account": "checking",
                "merchant": "Amazon",
                "amount": 150.50,
                "timestamp": "2024-01-15T10:30:00"
            }
        ],
        "user_id": "user_12345"
    }
    ```
    """
    try:
        detector = get_graph_risk_detector()

        if request.graph_type == "transaction":
            if not request.user_id:
                raise HTTPException(
                    status_code=400,
                    detail="user_id required for transaction graphs"
                )

            graph = await detector.build_transaction_graph(
                request.data,
                request.user_id
            )

        elif request.graph_type == "asset":
            graph = await detector.build_asset_correlation_graph(
                request.data
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown graph type: {request.graph_type}"
            )

        return BuildGraphResponse(
            graph_type=request.graph_type,
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            density=float(graph.number_of_edges() / max(1, graph.number_of_nodes())),
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph building failed: {str(e)}"
        )


@router.post("/systemic-risk", response_model=SystemicRiskResponse)
async def analyze_systemic_risk(request: SystemicRiskRequest):
    """
    Calculate systemic risk from asset correlations.

    Uses graph topology to identify:
    - Contagion risk
    - Sector concentration
    - Critical assets

    Example:
    ```json
    {
        "assets": [
            {
                "symbol": "AAPL",
                "type": "stock",
                "sector": "technology",
                "allocation": 0.3
            },
            {
                "symbol": "MSFT",
                "type": "stock",
                "sector": "technology",
                "allocation": 0.2
            }
        ]
    }
    ```
    """
    try:
        detector = get_graph_risk_detector()

        # Build correlation graph
        graph = await detector.build_asset_correlation_graph(
            request.assets,
            request.correlation_matrix
        )

        # Calculate systemic risk
        result = await detector.calculate_systemic_risk(graph)

        return SystemicRiskResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Systemic risk analysis failed: {str(e)}"
        )


@router.post("/detect-fraud", response_model=FraudCheckResponse)
async def detect_fraud(request: FraudCheckRequest):
    """
    Detect potential fraud indicators in transactions.

    Checks for:
    - Unusual velocity
    - Amount anomalies
    - Pattern deviations

    Example:
    ```json
    {
        "transactions": [
            {
                "amount": 5000,
                "merchant": "Unknown",
                "timestamp": "2024-01-15T02:30:00"
            }
        ],
        "user_profile": {
            "average_transaction": 150,
            "typical_merchants": ["Amazon", "Grocery Store"]
        }
    }
    ```
    """
    try:
        detector = get_graph_risk_detector()

        indicators = await detector.find_fraud_indicators(
            request.transactions,
            request.user_profile
        )

        highest_confidence = max(
            (ind.get('fraud_confidence', 0) for ind in indicators),
            default=0.0
        )

        return FraudCheckResponse(
            fraud_indicators_found=len(indicators),
            indicators=indicators,
            highest_confidence=highest_confidence,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fraud detection failed: {str(e)}"
        )


@router.get("/graph-stats/{user_id}")
async def get_graph_stats(
    user_id: str,
    graph_type: str = Query("transaction", description="Graph type: transaction or asset")
):
    """
    Get statistics for existing graph.

    Returns node/edge counts, density, and other metrics.
    """
    try:
        detector = get_graph_risk_detector()

        if graph_type == "transaction":
            graph = detector.transaction_graph
        elif graph_type == "asset":
            graph = detector.asset_correlation_graph
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown graph type: {graph_type}"
            )

        return {
            "graph_type": graph_type,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": float(graph.number_of_edges() / max(1, graph.number_of_nodes())),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph stats: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        detector = get_graph_risk_detector()
        return {
            "status": "healthy",
            "agent_id": detector.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
