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

from api.contracts import APIResponse, APIError, APIVersion
from agents.mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent
)
from agents.verifier import VerificationAgent
from data_models.schemas import AgentMessage, MessageType, Priority


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
            name: agent.get_health_status()
            for name, agent in agents.items()
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "FinPilot VP-MAS API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "orchestration": "/api/v1/orchestration",
            "planning": "/api/v1/planning",
            "market": "/api/v1/market",
            "verification": "/api/v1/verification"
        }
    }


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



# ReasonGraph Visualization Endpoints
@app.post("/api/v1/reasongraph/generate")
async def generate_reason_graph(request: Dict[str, Any]):
    """Generate ReasonGraph visualization data from planning and verification traces"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        from utils.reason_graph_mapper import ReasonGraphMapper
        
        mapper = ReasonGraphMapper()
        graphs = []
        
        # Map planning trace if provided
        if "planning_response" in request:
            planning_graph = mapper.map_planning_trace(request["planning_response"])
            graphs.append(planning_graph)
        
        # Map verification trace if provided
        if "verification_response" in request:
            verification_graph = mapper.map_verification_trace(
                request["verification_response"],
                graphs[0] if graphs else None
            )
            graphs.append(verification_graph)
        
        # Map CMVL trace if provided
        if "cmvl_response" in request:
            cmvl_graph = mapper.map_cmvl_trace(
                request["cmvl_response"],
                graphs[-1] if graphs else None
            )
            graphs.append(cmvl_graph)
        
        # Merge all graphs
        final_graph = mapper.merge_graphs(*graphs) if len(graphs) > 1 else graphs[0] if graphs else {}
        
        execution_time = time.time() - start_time
        return create_api_response(
            success=True,
            data=final_graph,
            request_id=request_id,
            execution_time=execution_time
        )
    
    except Exception as e:
        execution_time = time.time() - start_time
        return create_api_response(
            success=False,
            error=str(e),
            error_code="REASONGRAPH_GENERATION_FAILED",
            request_id=request_id,
            execution_time=execution_time
        )


@app.post("/api/v1/demo/complete-workflow")
async def demo_complete_workflow(request: Dict[str, Any]):
    """
    Demo endpoint that runs a complete workflow:
    1. Submit goal to orchestration
    2. Generate plan
    3. Verify plan
    4. Generate ReasonGraph
    """
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    session_id = f"session_{int(time.time() * 1000)}"
    
    try:
        from utils.reason_graph_mapper import ReasonGraphMapper
        
        user_goal = request.get("user_goal", "Save $100,000 for retirement in 10 years")
        
        # Step 1: Orchestration
        orch_message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=agents["orchestration"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"user_goal": user_goal},
            correlation_id=request_id,
            session_id=session_id,
            trace_id=request_id
        )
        orch_response = await agents["orchestration"].process_message(orch_message)
        
        # Step 2: Planning
        planning_message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=agents["planning"].agent_id,
            message_type=MessageType.REQUEST,
            payload={"planning_request": {"user_goal": user_goal, "time_horizon": 120}},
            correlation_id=request_id,
            session_id=session_id,
            trace_id=request_id
        )
        planning_response = await agents["planning"].process_message(planning_message)
        
        # Step 3: Verification
        verification_message = AgentMessage(
            agent_id="api_gateway",
            target_agent_id=agents["verification"].agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "verification_request": {
                    "plan_id": "demo_plan_001",
                    "plan_steps": planning_response.payload.get("plan_steps", []),
                    "verification_level": "comprehensive"
                }
            },
            correlation_id=request_id,
            session_id=session_id,
            trace_id=request_id
        )
        verification_response = await agents["verification"].process_message(verification_message)
        
        # Step 4: Generate ReasonGraph
        mapper = ReasonGraphMapper()
        planning_graph = mapper.map_planning_trace(planning_response.payload)
        verification_graph = mapper.map_verification_trace(verification_response.payload, planning_graph)
        final_graph = mapper.merge_graphs(planning_graph, verification_graph)
        
        execution_time = time.time() - start_time
        
        return create_api_response(
            success=True,
            data={
                "orchestration": orch_response.payload,
                "planning": planning_response.payload,
                "verification": verification_response.payload,
                "reason_graph": final_graph,
                "session_id": session_id
            },
            request_id=request_id,
            execution_time=execution_time
        )
    
    except Exception as e:
        execution_time = time.time() - start_time
        return create_api_response(
            success=False,
            error=str(e),
            error_code="DEMO_WORKFLOW_FAILED",
            request_id=request_id,
            execution_time=execution_time
        )
