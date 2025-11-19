"""
Orchestration Agent API Endpoints

Handles goal submission and workflow status tracking.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time

from data_models.schemas import AgentMessage, MessageType, Priority
from api.utils import create_api_response, get_request_id

router = APIRouter(
    prefix="/api/v1/orchestration",
    tags=["orchestration"],
    responses={404: {"description": "Not found"}},
)

# This will be injected by main.py
_agents = {}


def set_agents(agents: Dict[str, Any]):
    """Set agent instances (called from main.py)"""
    global _agents
    _agents = agents


@router.post("/goals")
async def submit_goal(request: Dict[str, Any]):
    """
    Submit a financial goal for processing.

    The orchestration agent coordinates the workflow across all agents
    to process the user's financial goal.
    """
    start_time = time.time()
    request_id = get_request_id()

    try:
        # Create agent message
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=_agents["orchestration"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"user_goal": request.get("user_goal"), **request},
            correlation_id=request_id,
            session_id=request.get("session_id", request_id),
            trace_id=request_id,
            priority=Priority[request.get("priority", "MEDIUM").upper()]
        )

        # Process message
        response = await _agents["orchestration"].process_message(message)

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
            error_code="GOAL_SUBMISSION_FAILED",
            request_id=request_id,
            execution_time=execution_time
        )


@router.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    Get the status of a workflow by ID.

    Returns current progress, state, and step information.
    """
    return create_api_response(
        success=True,
        data={
            "workflow_id": workflow_id,
            "status": "in_progress",
            "progress": 0.6,
            "current_step": "verification",
            "steps_completed": 3,
            "total_steps": 5
        }
    )
