"""
Conversational AI API Endpoints - Phase 6, Task 23

FastAPI endpoints for natural language financial planning interactions.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.conversational_agent import get_conversational_agent


router = APIRouter(prefix="/api/conversational", tags=["conversational"])


# Request/Response Models
class ParseGoalRequest(BaseModel):
    """Request to parse natural language goal"""
    user_input: str = Field(..., description="Natural language description of financial goal")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context (age, income, etc.)")


class ParseGoalResponse(BaseModel):
    """Parsed goal response"""
    goal_type: str
    target_amount: Optional[float] = None
    timeframe_years: Optional[int] = None
    risk_tolerance: Optional[str] = None
    parsed_at: str
    raw_input: str
    parsing_method: str


class GenerateNarrativeRequest(BaseModel):
    """Request to generate financial narrative"""
    plan: Dict[str, Any] = Field(..., description="Structured financial plan")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class GenerateNarrativeResponse(BaseModel):
    """Generated narrative response"""
    narrative: str
    generated_at: str


class ExplainScenarioRequest(BaseModel):
    """Request to explain what-if scenario"""
    scenario: Dict[str, Any] = Field(..., description="Scenario description")
    impact: Dict[str, Any] = Field(..., description="Quantified impact")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ExplainScenarioResponse(BaseModel):
    """Scenario explanation response"""
    explanation: str
    scenario_type: str
    generated_at: str


# Endpoints
@router.post("/parse-goal", response_model=ParseGoalResponse)
async def parse_goal(request: ParseGoalRequest):
    """
    Parse natural language financial goal into structured format.

    Example:
    ```json
    {
        "user_input": "I want to retire at 60 with $2 million",
        "user_context": {"age": 35, "income": 100000}
    }
    ```
    """
    try:
        agent = get_conversational_agent()

        result = await agent.parse_natural_language_goal(
            request.user_input,
            request.user_context
        )

        return ParseGoalResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Goal parsing failed: {str(e)}")


@router.post("/generate-narrative", response_model=GenerateNarrativeResponse)
async def generate_narrative(request: GenerateNarrativeRequest):
    """
    Generate human-readable narrative from financial plan.

    Example:
    ```json
    {
        "plan": {
            "goal_type": "retirement",
            "target_amount": 2000000,
            "timeframe_years": 25
        }
    }
    ```
    """
    try:
        agent = get_conversational_agent()

        narrative = await agent.generate_financial_narrative(
            request.plan,
            request.context
        )

        return GenerateNarrativeResponse(
            narrative=narrative,
            generated_at=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Narrative generation failed: {str(e)}")


@router.post("/explain-scenario", response_model=ExplainScenarioResponse)
async def explain_scenario(request: ExplainScenarioRequest):
    """
    Explain what-if scenario and its impact on financial plan.

    Example:
    ```json
    {
        "scenario": {
            "type": "market_crash",
            "severity": "high",
            "description": "30% market decline"
        },
        "impact": {
            "target_amount_change": -50000,
            "timeframe_change": 2
        }
    }
    ```
    """
    try:
        agent = get_conversational_agent()

        explanation = await agent.explain_what_if_scenario(
            request.scenario,
            request.impact,
            request.context
        )

        return ExplainScenarioResponse(
            explanation=explanation,
            scenario_type=request.scenario.get('type', 'unknown'),
            generated_at=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario explanation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        agent = get_conversational_agent()
        return {
            "status": "healthy",
            "agent_id": agent.agent_id,
            "ollama_available": agent.ollama_available,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
