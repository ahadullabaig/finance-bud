#!/usr/bin/env python
"""
Test FastAPI server for Phase 6 endpoints

Run with: python test_api_server.py
Then test with: curl commands or visit http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import Phase 6 routers
from api.conversational_endpoints import router as conversational_router
from api.risk_endpoints import router as risk_router
from api.ml_endpoints import router as ml_router


# Create FastAPI app
app = FastAPI(
    title="FinPilot Multi-Agent System - Phase 6 Test API",
    description="Test API for Phase 6 Advanced AI Features",
    version="6.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Phase 6 routers
app.include_router(conversational_router)
app.include_router(risk_router)
app.include_router(ml_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FinPilot Phase 6 API",
        "version": "6.0.0",
        "docs": "Visit /docs for interactive API documentation",
        "endpoints": {
            "conversational": "/api/conversational/*",
            "risk": "/api/risk/*",
            "ml": "/api/ml/*"
        }
    }


@app.get("/health")
async def health():
    """Overall health check"""
    return {
        "status": "healthy",
        "phase": 6,
        "features": [
            "conversational_ai",
            "graph_risk_detection",
            "ml_predictions"
        ]
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Starting FinPilot Phase 6 Test API Server")
    print("=" * 60)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nEndpoints:")
    print("  • Conversational AI: http://localhost:8000/api/conversational/")
    print("  • Risk Detection: http://localhost:8000/api/risk/")
    print("  • ML Predictions: http://localhost:8000/api/ml/")
    print("\n" + "=" * 60)
    print("\nPress Ctrl+C to stop the server\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
