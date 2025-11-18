"""
FinPilot VP-MAS Backend API Server

FastAPI application providing REST endpoints for all VP-MAS agents.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from datetime import datetime
from typing import Dict, Any
import uvicorn
import logging

from api.contracts import APIResponse, APIError, APIVersion
from agents.mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent
)
from agents.verifier import VerificationAgent
from data_models.schemas import AgentMessage, MessageType, Priority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import conversational endpoints with error handling
try:
    from api.conversational_endpoints import router as conversational_router
    CONVERSATIONAL_AVAILABLE = True
    logger.info("✅ Conversational endpoints imported successfully")
except ImportError as e:
    CONVERSATIONAL_AVAILABLE = False
    conversational_router = None
    logger.warning(f"⚠️ Conversational endpoints unavailable: {e}")
except Exception as e:
    CONVERSATIONAL_AVAILABLE = False
    conversational_router = None
    logger.error(f"❌ Failed to import conversational endpoints: {e}")


# Agent instances
agents = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup"""
    print("🚀 Starting FinPilot VP-MAS Backend...")
    
    # Initialize agents (using real VerificationAgent for Person D tasks)
    agents["orchestration"] = MockOrchestrationAgent()
    agents["planning"] = MockPlanningAgent()
    agents["information_retrieval"] = MockInformationRetrievalAgent()
    agents["verification"] = VerificationAgent()
    
    # Initialize conversational agent with error handling
    try:
        from agents.conversational_agent import get_conversational_agent
        agents["conversational"] = get_conversational_agent()
        logger.info("✅ Conversational agent initialized successfully")
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import conversational agent: {e}")
        agents["conversational"] = None
    except Exception as e:
        logger.error(f"❌ Failed to initialize conversational agent: {e}")
        agents["conversational"] = None
    
    # Log conversational endpoints availability
    if CONVERSATIONAL_AVAILABLE:
        logger.info("✅ Conversational endpoints are available and ready")
    else:
        logger.warning("⚠️ Conversational endpoints are not available - chatbot functionality disabled")
    
    # Log agent initialization status
    agent_status = {}
    for name, agent in agents.items():
        if agent is not None:
            agent_status[name] = "initialized"
        else:
            agent_status[name] = "failed"
    
    logger.info(f"Agent initialization status: {agent_status}")
    print("✅ All agents initialized successfully")
    yield
    
    # Cleanup
    print("🛑 Shutting down agents...")


# Create FastAPI app
app = FastAPI(
    title="FinPilot VP-MAS API",
    description="Verifiable Planning Multi-Agent System for Financial Planning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include conversational router if available
if CONVERSATIONAL_AVAILABLE and conversational_router:
    app.include_router(conversational_router)
    logger.info("✅ Conversational router included at /api/conversational")
else:
    logger.warning("⚠️ Conversational router not included - endpoints unavailable")


# Middleware for request tracking
@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    """Add request ID and timing to all requests"""
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    
    response = await call_next(request)
    
    execution_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Execution-Time"] = str(execution_time)
    
    return response


def create_api_response(success: bool, data: Any = None, error: str = None, 
                       error_code: str = None, request_id: str = None,
                       execution_time: float = 0) -> Dict:
    """Create standardized API response"""
    return {
        "success": success,
        "data": data,
        "error": error,
        "error_code": error_code,
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id or f"req_{int(time.time() * 1000)}",
        "execution_time": execution_time,
        "api_version": "v1"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agents": {
            name: agent.get_health_status() if agent is not None else "unavailable"
            for name, agent in agents.items()
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    endpoints = {
        "health": "/health",
        "docs": "/docs",
        "orchestration": "/api/v1/orchestration",
        "planning": "/api/v1/planning",
        "market": "/api/v1/market",
        "verification": "/api/v1/verification"
    }
    
    # Add conversational endpoints if available
    if CONVERSATIONAL_AVAILABLE:
        endpoints["conversational"] = "/api/conversational"
    
    return {
        "service": "FinPilot VP-MAS API",
        "version": "1.0.0",
        "status": "running",
        "conversational_available": CONVERSATIONAL_AVAILABLE,
        "conversational_agent_status": "initialized" if agents.get("conversational") is not None else "unavailable",
        "endpoints": endpoints
    }


if __name__ == "__main__":
    print("🚀 Starting FinPilot VP-MAS Backend Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Orchestration Agent Endpoints
@app.post("/api/v1/orchestration/goals")
async def submit_goal(request: Dict[str, Any]):
    """Submit a financial goal for processing"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        # Create agent message
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=agents["orchestration"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"user_goal": request.get("user_goal"), **request},
            correlation_id=request_id,
            session_id=request.get("session_id", request_id),
            trace_id=request_id,
            priority=Priority[request.get("priority", "MEDIUM").upper()]
        )
        
        # Process message
        response = await agents["orchestration"].process_message(message)
        
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


@app.get("/api/v1/orchestration/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
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


# Planning Agent Endpoints
@app.post("/api/v1/planning/generate")
async def generate_plan(request: Dict[str, Any]):
    """Generate financial plan"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=agents["planning"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"planning_request": request},
            correlation_id=request_id,
            session_id=request.get("session_id", request_id),
            trace_id=request_id
        )
        
        response = await agents["planning"].process_message(message)
        
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


# Market Data Endpoints
@app.get("/api/v1/market/data")
async def get_market_data(symbols: str = None, refresh: bool = False):
    """Get market data"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=agents["information_retrieval"].agent_id,
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
        
        response = await agents["information_retrieval"].process_message(message)
        
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


@app.post("/api/v1/market/triggers/detect")
async def detect_triggers(request: Dict[str, Any]):
    """Detect market triggers"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=agents["information_retrieval"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"trigger_detection": request},
            correlation_id=request_id,
            session_id=request_id,
            trace_id=request_id
        )
        
        response = await agents["information_retrieval"].process_message(message)
        
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


# Verification Agent Endpoints
@app.post("/api/v1/verification/verify")
async def verify_plan(request: Dict[str, Any]):
    """Verify financial plan"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=agents["verification"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"verification_request": request},
            correlation_id=request_id,
            session_id=request.get("session_id", request_id),
            trace_id=request_id
        )
        
        response = await agents["verification"].process_message(message)
        
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
            error_code="VERIFICATION_FAILED",
            request_id=request_id,
            execution_time=execution_time
        )