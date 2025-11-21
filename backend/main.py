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

from config import settings
from api.contracts import APIResponse, APIError, APIVersion
from api.utils import create_api_response
from agents.factory import get_agent_factory
from data_models.schemas import AgentMessage, MessageType, Priority

# Import API routers
from api import (
    orchestration_endpoints,
    planning_endpoints,
    market_endpoints,
    verification_endpoints
)

# Configure logging
logging.basicConfig(
    level=settings.log_level.upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    """Initialize agents on startup using AgentFactory"""
    logger.info("🚀 Starting FinPilot VP-MAS Backend...")

    # Create all agents using factory
    factory = get_agent_factory()
    initialized_agents = factory.create_all_agents()
    agents.update(initialized_agents)

    # Log conversational endpoints availability
    if CONVERSATIONAL_AVAILABLE:
        logger.info("✅ Conversational endpoints are available and ready")
    else:
        logger.warning("⚠️ Conversational endpoints are not available - chatbot functionality disabled")

    # Perform health check
    health_status = factory.health_check()
    logger.info(f"Agent health status: {health_status}")

    # Inject agents into API routers
    inject_agents_into_routers()
    logger.info("✅ Agents injected into API routers")
    logger.info("✅ All agents initialized successfully")

    yield

    # Cleanup
    logger.info("🛑 Shutting down agents...")
    factory.shutdown()


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Inject agents into routers
def inject_agents_into_routers():
    """Inject agent instances into all routers"""
    orchestration_endpoints.set_agents(agents)
    planning_endpoints.set_agents(agents)
    market_endpoints.set_agents(agents)
    verification_endpoints.set_agents(agents)


# Include all routers
app.include_router(orchestration_endpoints.router)
app.include_router(planning_endpoints.router)
app.include_router(market_endpoints.router)
app.include_router(verification_endpoints.router)

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


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with detailed agent status"""
    factory = get_agent_factory()
    agent_health = factory.health_check()

    # Determine overall system health
    all_healthy = all(status.get("healthy", False) for status in agent_health.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "agents": agent_health,
        "conversational_endpoints_available": CONVERSATIONAL_AVAILABLE
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
    logger.info("🚀 Starting FinPilot VP-MAS Backend Server...")
    logger.info(f"📍 API will be available at: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"📚 API Documentation: http://{settings.api_host}:{settings.api_port}/docs")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level=settings.log_level
    )
