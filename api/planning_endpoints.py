"""
Planning Agent API Endpoints

Handles financial plan generation and optimization.
"""

from fastapi import APIRouter
from typing import Dict, Any
import time

from data_models.schemas import AgentMessage, MessageType
from api.utils import create_api_response, get_request_id

router = APIRouter(
    prefix="/api/v1/planning",
    tags=["planning"],
    responses={404: {"description": "Not found"}},
)

# This will be injected by main.py
_agents = {}


def set_agents(agents: Dict[str, Any]):
    """Set agent instances (called from main.py)"""
    global _agents
    _agents = agents


@router.post("/generate")
async def generate_plan(request: Dict[str, Any]):
    """
    Generate a comprehensive financial plan.

    Uses the Planning Agent to create multi-path strategies with
    constraint-based optimization and risk-adjusted returns.
    """
    start_time = time.time()
    request_id = get_request_id()

    try:
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=_agents["planning"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"planning_request": request},
            correlation_id=request_id,
            session_id=request.get("session_id", request_id),
            trace_id=request_id
        )

        response = await _agents["planning"].process_message(message)

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
            error_code="PLAN_GENERATION_FAILED",
            request_id=request_id,
            execution_time=execution_time
        )
