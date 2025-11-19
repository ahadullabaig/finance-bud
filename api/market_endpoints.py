"""
Market Data & Information Retrieval API Endpoints

Handles market data fetching and trigger detection.
"""

from fastapi import APIRouter
from typing import Dict, Any, Optional
import time

from data_models.schemas import AgentMessage, MessageType
from api.utils import create_api_response, get_request_id

router = APIRouter(
    prefix="/api/v1/market",
    tags=["market"],
    responses={404: {"description": "Not found"}},
)

# This will be injected by main.py
_agents = {}


def set_agents(agents: Dict[str, Any]):
    """Set agent instances (called from main.py)"""
    global _agents
    _agents = agents


@router.get("/data")
async def get_market_data(symbols: Optional[str] = None, refresh: bool = False):
    """
    Get real-time market data for specified symbols.

    Args:
        symbols: Comma-separated list of stock symbols (e.g., "AAPL,GOOGL,MSFT")
        refresh: Force refresh from external APIs (bypasses cache)

    Returns:
        Market data including prices, volatility, and indicators
    """
    start_time = time.time()
    request_id = get_request_id()

    try:
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=_agents["information_retrieval"].agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "market_data_request": {
                    "symbols": symbols.split(",") if symbols else [],
                    "refresh": refresh
                }
            },
            correlation_id=request_id,
            session_id=request_id,
            trace_id=request_id
        )

        response = await _agents["information_retrieval"].process_message(message)

        execution_time = time.time() - start_time
        return create_api_response(
            success=True,
            data=response.payload,
            request_id=request_id,
            execution_time=execution_time
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return create_api_response(
            success=False,
            error=str(e),
            error_code="MARKET_DATA_FETCH_FAILED",
            request_id=request_id,
            execution_time=execution_time
        )


@router.post("/triggers/detect")
async def detect_triggers(request: Dict[str, Any]):
    """
    Detect market triggers and events that may require plan adjustments.

    Analyzes market conditions for volatility spikes, significant changes,
    and other events that could impact financial plans.
    """
    start_time = time.time()
    request_id = get_request_id()

    try:
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=_agents["information_retrieval"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"trigger_detection": request},
            correlation_id=request_id,
            session_id=request_id,
            trace_id=request_id
        )

        response = await _agents["information_retrieval"].process_message(message)

        execution_time = time.time() - start_time
        return create_api_response(
            success=True,
            data=response.payload,
            request_id=request_id,
            execution_time=execution_time
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return create_api_response(
            success=False,
            error=str(e),
            error_code="TRIGGER_DETECTION_FAILED",
            request_id=request_id,
            execution_time=execution_time
        )
