#!/usr/bin/env python
"""
FinPilot Phase 6 API Server
Standalone server for Phase 6 Advanced AI Features

Run with: python run_api_server.py
Visit: http://localhost:8000/docs for interactive API documentation
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import Phase 6 routers directly (bypassing api/__init__.py to avoid database deps)
import api.conversational_endpoints as conversational_module
import api.risk_endpoints as risk_module
import api.ml_endpoints as ml_module


# Create FastAPI app
app = FastAPI(
    title="FinPilot Multi-Agent System API",
    description="""
# FinPilot Phase 6: Advanced AI Features

This API provides access to three advanced AI agents:

## 🤖 Conversational AI Agent (Task 23)
- Natural language goal parsing
- Financial narrative generation
- What-if scenario explanations

## 🛡️ Graph Risk Detector (Task 24)
- Transaction anomaly detection
- Fraud pattern recognition
- Portfolio risk analysis

## 📈 ML Prediction Engine (Task 25)
- Market trend forecasting
- Portfolio performance predictions
- Personalized recommendations
    """,
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Phase 6 routers
app.include_router(conversational_module.router)
app.include_router(risk_module.router)
app.include_router(ml_module.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "app": "FinPilot Multi-Agent System",
        "version": "6.0.0",
        "phase": 6,
        "status": "running",
        "documentation": {
            "swagger": "http://localhost:8000/docs",
            "redoc": "http://localhost:8000/redoc"
        },
        "agents": {
            "conversational_ai": {
                "status": "operational",
                "description": "Natural language understanding and financial narratives",
                "endpoints": "/api/conversational/*"
            },
            "graph_risk_detector": {
                "status": "operational",
                "description": "Transaction anomaly and fraud detection",
                "endpoints": "/api/risk/*"
            },
            "ml_prediction_engine": {
                "status": "operational",
                "description": "Market forecasting and portfolio predictions",
                "endpoints": "/api/ml/*"
            }
        },
        "frontend": "http://localhost:3000"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Test agent imports
        from agents.conversational_agent import get_conversational_agent
        from agents.graph_risk_detector import get_graph_risk_detector
        from agents.ml_prediction_engine import get_ml_prediction_engine

        # Initialize agents to verify they're working
        conversational = get_conversational_agent()
        risk_detector = get_graph_risk_detector()
        ml_engine = get_ml_prediction_engine()

        return {
            "status": "healthy",
            "phase": 6,
            "timestamp": str(Path(__file__).stat().st_mtime),
            "agents": {
                "conversational_agent": {
                    "id": conversational.agent_id,
                    "status": "ready",
                    "ollama": conversational.ollama_available
                },
                "graph_risk_detector": {
                    "id": risk_detector.agent_id,
                    "status": "ready"
                },
                "ml_prediction_engine": {
                    "id": ml_engine.agent_id,
                    "status": "ready"
                }
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 FinPilot Phase 6 API Server")
    print("=" * 80)
    print("\n📖 Interactive API Documentation:")
    print("   → Swagger UI:  http://localhost:8000/docs")
    print("   → ReDoc:       http://localhost:8000/redoc")
    print("\n❤️  Health Check:")
    print("   → Status:      http://localhost:8000/health")
    print("\n🤖 AI Agent Endpoints:")
    print("   → Conversational AI:    http://localhost:8000/api/conversational/")
    print("   → Graph Risk Detection: http://localhost:8000/api/risk/")
    print("   → ML Predictions:       http://localhost:8000/api/ml/")
    print("\n🌐 Frontend Application:")
    print("   → React App:   http://localhost:3000")
    print("\n" + "=" * 80)
    print("\n💡 Tip: Visit /docs for interactive API testing")
    print("🛑 Press Ctrl+C to stop the server\n")

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
