#!/usr/bin/env python
"""
Simple FastAPI server for Phase 6 endpoints
Runs without database dependencies for frontend testing

Run with: python simple_api_server.py
Then visit: http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import Phase 6 routers directly
from api.conversational_endpoints import router as conversational_router
from api.risk_endpoints import router as risk_router
from api.ml_endpoints import router as ml_router


# Create FastAPI app
app = FastAPI(
    title="FinPilot Multi-Agent System - Phase 6 API",
    description="Advanced AI Features: Conversational AI, Graph Risk Detection, ML Predictions",
    version="6.0.0"
)

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Phase 6 routers
app.include_router(conversational_router, prefix="/api")
app.include_router(risk_router, prefix="/api")
app.include_router(ml_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FinPilot Phase 6 API - Advanced AI Features",
        "version": "6.0.0",
        "status": "running",
        "docs": "http://localhost:8000/docs",
        "features": {
            "conversational_ai": {
                "description": "Natural language goal parsing and financial narratives",
                "endpoints": "/api/conversational/*"
            },
            "graph_risk_detection": {
                "description": "Transaction anomaly and fraud detection",
                "endpoints": "/api/risk/*"
            },
            "ml_predictions": {
                "description": "Market trends and portfolio forecasting",
                "endpoints": "/api/ml/*"
            }
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "phase": 6,
        "agents": {
            "conversational_agent": "operational",
            "graph_risk_detector": "operational",
            "ml_prediction_engine": "operational"
        }
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 FinPilot Phase 6 API Server Starting...")
    print("=" * 70)
    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("\n🤖 Available AI Agents:")
    print("   • Conversational AI: http://localhost:8000/api/conversational/")
    print("   • Risk Detection: http://localhost:8000/api/risk/")
    print("   • ML Predictions: http://localhost:8000/api/ml/")
    print("\n🌐 Frontend: http://localhost:3000")
    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop the server\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
