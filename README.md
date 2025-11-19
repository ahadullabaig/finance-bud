# FinPilot - Verifiable Planning Multi-Agent System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Node](https://img.shields.io/badge/Node-18+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)
![React](https://img.shields.io/badge/React-18-cyan.svg)

**An enterprise-grade Verifiable Planning Multi-Agent System for adaptive financial planning with natural language processing, real-time market integration, and continuous monitoring.**

<div align="center">

**[🎮 Live Demo](https://finance-bud.vercel.app/)** •
**[📺 Video Tour](https://youtu.be/hKzpLhGA75k)** •
**[📚 Full Documentation](docs/)** •
**[🔧 API Docs](https://ecell-production.up.railway.app/docs)**

</div>

---

## Table of Contents

- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
  - [Your First Request](#your-first-request)
- [What is FinPilot?](#-what-is-finpilot)
- [Live Demo & Features](#-live-demo--features)
- [Architecture](#-architecture)
  - [Multi-Agent System](#multi-agent-system)
  - [System Diagram](#system-diagram)
  - [Communication Flow](#communication-flow)
- [Technology Stack](#-technology-stack)
- [Development Guide](#-development-guide)
  - [Backend Development](#backend-development)
  - [Frontend Development](#frontend-development)
  - [Testing](#testing)
- [API Documentation](#-api-documentation)
  - [Core Endpoints](#core-endpoints)
  - [Code Examples](#code-examples)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🚀 Quick Start

Get FinPilot running locally in under 5 minutes.

### Prerequisites

- **Python 3.11+** - [Download](https://www.python.org/downloads/)
- **Node.js 18+** - [Download](https://nodejs.org/)
- **Git** - [Download](https://git-scm.com/)
- **Optional**: [Ollama](https://ollama.ai/) for local LLM support

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/finance-bud.git
cd finance-bud

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
npm install
```

### Running Locally

**Terminal 1 - Backend:**
```bash
python main.py
# API server: http://localhost:8000
# API docs: http://localhost:8000/docs
```

**Terminal 2 - Frontend:**
```bash
npm run dev
# Development server: http://localhost:3000
```

### Your First Request

Verify the backend is running:
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "timestamp": "..."}
```

Submit a financial goal:
```bash
curl -X POST http://localhost:8000/api/v1/orchestration/goals \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I want to retire at 60 with $2 million",
    "user_context": {"age": 35, "income": 100000}
  }'
```

Open http://localhost:3000 to interact with the UI.

---

## 💡 What is FinPilot?

FinPilot is a **Verifiable Planning Multi-Agent System (VP-MAS)** that provides intelligent, adaptive financial planning through natural language interactions. Unlike traditional financial planning tools, FinPilot uses a sophisticated multi-agent architecture to:

- **Understand your goals** in natural language ("I want to retire at 60 with $2M")
- **Generate optimal plans** using advanced algorithms (Guided Search Module, Thought of Search)
- **Continuously monitor** your financial situation and market conditions
- **Adapt automatically** when life events or market changes occur
- **Explain every decision** with transparent reasoning graphs and audit trails

**Why FinPilot?**

| Feature | Traditional Tools | FinPilot |
|---------|------------------|----------|
| Goal Input | Forms & dropdowns | Natural language |
| Planning | Static rules | AI-powered multi-agent system |
| Monitoring | Manual check-ins | Continuous real-time monitoring (CMVL) |
| Adaptation | User must re-plan | Automatic trigger detection & re-planning |
| Transparency | Black box | Complete reasoning graphs & audit trails |
| Technology | Monolithic | Distributed multi-agent architecture |

---

## 🎮 Live Demo & Features

**Try it live:** [finance-bud.vercel.app](https://finance-bud.vercel.app/)
**Watch demo:** [YouTube Video Tour](https://youtu.be/hKzpLhGA75k)

### Core Capabilities

#### 🧠 Advanced Planning Intelligence
- **Natural Language Processing**: Convert conversational goals into structured financial plans
- **Multi-Constraint Optimization**: Balance budget ratios, safety funds, debt limits, and tax implications
- **Guided Search Module (GSM)**: Explore multiple strategy paths using Thought of Search (ToS) heuristics
- **Scenario Support**: Retirement, emergency funds, debt payoff, investment portfolios, education savings

#### 🔄 Continuous Monitoring & Verification Loop (CMVL)
- **Real-Time Monitoring**: Track plan execution and external conditions 24/7
- **Trigger Detection**: Market volatility, life events (job loss, medical emergencies), economic changes
- **Automatic Re-Planning**: Generate updated plans when triggers are detected
- **User Approval Workflows**: Present plan changes with clear before/after comparisons

#### 🎨 Transparency & Explainability
- **ReasonGraph Visualization**: Interactive D3.js graphs showing decision-making processes
- **Agent Communication Traces**: Complete audit trails with correlation IDs
- **Decision Explanations**: Human-readable narratives for every recommendation
- **Performance Metrics**: Track agent response times and system health

#### 🔒 Enterprise-Grade Reliability
- **Type-Safe Communication**: Pydantic schemas ensure data integrity across agents
- **Circuit Breakers**: Graceful degradation when services are unavailable
- **Comprehensive Testing**: 80%+ test coverage (unit, integration, E2E)
- **Performance Monitoring**: Request tracking, logging, and health checks

---

## 🏗️ Architecture

### Multi-Agent System

FinPilot consists of **6 specialized agents** that communicate through structured message passing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION AGENT (OA)                     │
│           Workflow Coordination • Trigger Management            │
└────────┬────────────────────────────────────────────────┬───────┘
         │                                                │
    ┌────▼─────┐  ┌──────────────┐  ┌────────────┐  ┌─────▼─────┐
    │ Planning │  │ Information  │  │Verification│  │ Execution │
    │  Agent   │  │  Retrieval   │  │   Agent    │  │   Agent   │
    │   (PA)   │  │  Agent (IRA) │  │    (VA)    │  │   (EA)    │
    │          │  │              │  │            │  │           │
    │ GSM+ToS  │  │ Market Data  │  │ Constraint │  │  Ledger   │
    │ Planning │  │ Integration  │  │ Validation │  │   Mgmt    │
    └────┬─────┘  └──────┬───────┘  └─────┬──────┘  └─────┬─────┘
         │               │                │               │
         └───────────────┴────────────────┴───────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Conversational Agent   │
                    │         (CA)            │
                    │  NLP • Goal Parsing •   │
                    │  Narrative Generation   │
                    └─────────────────────────┘
```

#### Agent Responsibilities

1. **Orchestration Agent (OA)** - `agents/orchestration_agent.py`
   - Mission control for workflow coordination
   - Trigger management and goal parsing
   - Circuit breakers and reliability management
   - Agent lifecycle and health monitoring

2. **Planning Agent (PA)** - `agents/planning_agent.py`
   - **Guided Search Module (GSM)**: Multi-path strategy exploration
   - **Thought of Search (ToS)**: Advanced heuristic algorithms
   - Constraint-based optimization
   - Multi-scenario plan generation

3. **Information Retrieval Agent (IRA)** - `agents/retriever.py`
   - Real-time market data integration (yfinance, Alpha Vantage, Barchart)
   - External API management
   - Trigger detection (volatility, economic indicators)
   - Data freshness validation

4. **Verification Agent (VA)** - `agents/verifier.py`
   - Constraint satisfaction validation
   - CMVL workflow coordination
   - Regulatory compliance checking
   - Plan risk assessment

5. **Execution Agent (EA)** - `agents/execution_agent.py`
   - Plan execution and action implementation
   - Ledger management and transaction tracking
   - Progress monitoring
   - Execution audit trails

6. **Conversational Agent (CA)** - `agents/conversational_agent.py`
   - Natural language processing (Ollama integration)
   - Goal parsing and intent extraction
   - Narrative generation for plans
   - Hardcoded fallbacks for offline operation

### System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│        React + TypeScript + D3.js + Recharts                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                             │
│                    api/endpoints.py                              │
└────────────────────────────┬─────────────────────────────────────┘
                             │ AgentMessage
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   MULTI-AGENT SYSTEM                             │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐      │
│  │  OA  │◄─┤  PA  │  │ IRA  │  │  VA  │  │  EA  │  │  CA  │      │
│  └───┬──┘  └───┬──┘  └───┬──┘  └───┬──┘  └───┬──┘  └──────┘      │
│      │         │         │         │         │                   │
│      └─────────┴─────────┴─────────┴─────────┘                   │
│              Structured Message Passing                          │
│            (Pydantic schemas + correlation IDs)                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  Market  │      │  Ollama  │      │ Database │
    │   APIs   │      │   LLM    │      │(Optional)│
    └──────────┘      └──────────┘      └──────────┘
```

### Communication Flow

All agents communicate via **AgentMessage** objects defined in `data_models/schemas.py`:

```python
class AgentMessage(BaseModel):
    correlation_id: str          # Track messages across agent boundaries
    sender_agent: str            # Source agent identifier
    recipient_agent: str         # Target agent identifier
    message_type: MessageType    # REQUEST | RESPONSE | NOTIFICATION
    payload: Dict[str, Any]      # Typed data payload
    performance_metrics: Optional[PerformanceMetrics]
    timestamp: datetime
```

**Key Design Principles:**
- **Type Safety**: Pydantic validation ensures data integrity
- **Traceability**: Correlation IDs enable end-to-end tracking
- **Performance**: Metrics collected at each hop
- **Reliability**: Circuit breakers prevent cascade failures

---

## 🛠️ Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI + Uvicorn | High-performance async API server |
| **Language** | Python 3.11+ | Core implementation |
| **Validation** | Pydantic v2 | Type-safe data models |
| **AI/ML** | Ollama, LangChain | LLM integration & agent orchestration |
| **Graph Analysis** | NetworkX, scikit-learn | Risk detection & analysis |
| **Market Data** | yfinance, Alpha Vantage, Barchart | Real-time financial data |
| **Testing** | pytest, pytest-asyncio | Comprehensive test suite |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | React 18 + TypeScript | Type-safe UI development |
| **Build Tool** | Vite 6.3+ | Fast builds & HMR |
| **UI Library** | Radix UI | Accessible component primitives |
| **Styling** | Tailwind CSS | Utility-first styling |
| **Visualization** | D3.js, Recharts | ReasonGraph & financial charts |
| **Testing** | Playwright | End-to-end testing |

### DevOps & Infrastructure
- **CI/CD**: GitHub Actions
- **Deployment**: Vercel (frontend), Railway (backend)
- **Monitoring**: FastAPI health checks, structured logging
- **Configuration**: Environment variables, api_config.json

---

## 👨‍💻 Development Guide

### Backend Development

#### Start Development Server
```bash
python main.py
# API available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

#### Common Commands
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_agents.py -v              # Agent unit tests
pytest tests/test_integration.py -v          # Integration tests
pytest -m asyncio                            # Async tests only
pytest -m integration                        # Integration tests only

# Test individual agents
python test_agents_cli.py

# Check test coverage
pytest tests/ --cov=agents --cov-report=html
```

#### Project-Specific Test Files
```bash
# Comprehensive agent validation
python run_tests.py

# CMVL workflow testing
pytest tests/test_cmvl_advanced.py -v

# API endpoint testing
pytest tests/api/ -v
```

### Frontend Development

#### Start Development Server
```bash
npm run dev
# Development server: http://localhost:3000
# API proxy: /api → http://localhost:8000
```

#### Common Commands
```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Run E2E tests
npm run test:e2e              # Headless mode
npm run test:e2e:ui           # Interactive UI mode
npm run test:e2e:headed       # Headed browser mode
npm run test:e2e:debug        # Debug mode

# View test report
npm run test:e2e:report

# Generate test code
npm run test:e2e:codegen
```

### Testing

FinPilot has comprehensive test coverage across all layers:

#### Backend Tests (pytest)
```bash
# Unit tests - Individual agent functionality
pytest tests/unit/ -v

# Integration tests - Multi-agent communication
pytest tests/integration/ -v

# Performance tests - Benchmarks
pytest tests/performance/ -v

# API tests - Endpoint validation
pytest tests/api/ -v

# Scenario tests - End-to-end workflows
pytest tests/scenarios/ -v
```

#### Frontend Tests (Playwright)
```bash
# User journey tests
npm run test:e2e -- e2e/tests/user-journey.spec.ts

# API integration tests
npm run test:e2e -- e2e/tests/backend-integration.spec.ts

# Interactive debugging
npm run test:e2e:ui
```

#### Test Data
Consistent test data is available in `tests/mock_data.py` for all backend tests.

---

## 📡 API Documentation

### Core Endpoints

#### Orchestration & Goal Management
```http
POST /api/v1/orchestration/goals
```
Submit a financial goal for processing.

**Request:**
```json
{
  "user_input": "I want to retire at 60 with $2 million",
  "user_context": {
    "age": 35,
    "income": 100000,
    "current_savings": 50000
  }
}
```

**Response:**
```json
{
  "goal_id": "goal_12345",
  "parsed_goal": {
    "objective": "retirement",
    "target_age": 60,
    "target_amount": 2000000
  },
  "correlation_id": "corr_abc123",
  "status": "accepted"
}
```

#### Planning & Strategy Generation
```http
POST /api/v1/planning/generate
```
Generate a financial plan for a parsed goal.

**Request:**
```json
{
  "goal_id": "goal_12345",
  "constraints": {
    "risk_tolerance": "moderate",
    "time_horizon_years": 25,
    "monthly_contribution_limit": 2000
  }
}
```

**Response:**
```json
{
  "plan_id": "plan_67890",
  "strategies": [
    {
      "name": "Balanced Growth Strategy",
      "allocation": {
        "stocks": 0.6,
        "bonds": 0.3,
        "cash": 0.1
      },
      "monthly_contribution": 1500,
      "projected_value": 2100000,
      "confidence": 0.85
    }
  ],
  "reason_graph": { ... },
  "performance_metrics": { ... }
}
```

#### Market Data Retrieval
```http
GET /api/v1/market/data?symbols=SPY,AGG&period=1mo
```
Fetch real-time market data.

**Response:**
```json
{
  "data": {
    "SPY": {
      "current_price": 450.25,
      "change_percent": 1.2,
      "volatility": 0.15
    },
    "AGG": {
      "current_price": 98.50,
      "change_percent": -0.3,
      "volatility": 0.05
    }
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### Plan Verification
```http
POST /api/v1/verification/verify
```
Validate a plan against constraints and regulations.

**Request:**
```json
{
  "plan_id": "plan_67890",
  "verification_mode": "comprehensive"
}
```

**Response:**
```json
{
  "is_valid": true,
  "violations": [],
  "warnings": [
    "High equity allocation for age group"
  ],
  "compliance_score": 0.92
}
```

#### Health Check
```http
GET /health
```
System health status.

**Response:**
```json
{
  "status": "healthy",
  "agents": {
    "orchestration_agent": "online",
    "planning_agent": "online",
    "retriever_agent": "online",
    "verification_agent": "online",
    "execution_agent": "online",
    "conversational_agent": "online"
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Code Examples

#### Python Client
```python
import requests

# Submit a goal
response = requests.post(
    "http://localhost:8000/api/v1/orchestration/goals",
    json={
        "user_input": "I want to save $50,000 for a down payment in 5 years",
        "user_context": {
            "age": 28,
            "income": 75000,
            "current_savings": 10000
        }
    }
)
goal_id = response.json()["goal_id"]

# Generate a plan
response = requests.post(
    "http://localhost:8000/api/v1/planning/generate",
    json={
        "goal_id": goal_id,
        "constraints": {
            "risk_tolerance": "conservative",
            "time_horizon_years": 5
        }
    }
)
plan = response.json()
print(f"Recommended monthly contribution: ${plan['strategies'][0]['monthly_contribution']}")
```

#### JavaScript/TypeScript Client
```typescript
// Submit a goal
const goalResponse = await fetch('http://localhost:8000/api/v1/orchestration/goals', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_input: "I want to pay off $30,000 in student loans in 3 years",
    user_context: {
      age: 25,
      income: 60000,
      current_debt: 30000
    }
  })
});
const { goal_id } = await goalResponse.json();

// Generate a plan
const planResponse = await fetch('http://localhost:8000/api/v1/planning/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    goal_id,
    constraints: {
      risk_tolerance: "moderate",
      time_horizon_years: 3
    }
  })
});
const plan = await planResponse.json();
console.log('Debt payoff strategy:', plan.strategies[0]);
```

#### cURL Examples
```bash
# Submit goal
curl -X POST http://localhost:8000/api/v1/orchestration/goals \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I want to retire at 60 with $2 million",
    "user_context": {"age": 35, "income": 100000}
  }'

# Check health
curl http://localhost:8000/health

# Get market data
curl "http://localhost:8000/api/v1/market/data?symbols=SPY,AGG&period=1mo"
```

### Conversational AI Endpoints

```http
POST /api/conversational/parse-goal       # Parse natural language goals
POST /api/conversational/generate-narrative  # Generate plan narratives
POST /api/conversational/explain-scenario    # Explain what-if scenarios
GET  /api/conversational/health              # Conversational agent status
```

See full API documentation at `/docs` when running locally.

---

## 📁 Project Structure

```
/finance-bud
├── agents/                         # Multi-agent system implementation
│   ├── base_agent.py                  # Base class for all agents
│   ├── orchestration_agent.py         # Workflow coordination
│   ├── planning_agent.py              # GSM and ToS algorithms
│   ├── retriever.py                   # Market data integration
│   ├── verifier.py                    # Constraint validation
│   ├── execution_agent.py             # Plan execution
│   ├── conversational_agent.py        # Natural language processing
│   ├── cmvl_workflow.py               # CMVL implementation
│   ├── graph_risk_detector.py         # Risk detection (NetworkX)
│   └── communication.py               # Agent messaging framework
│
├── api/                            # FastAPI REST endpoints
│   ├── endpoints.py                   # Core agent endpoints
│   ├── conversational_endpoints.py    # Chatbot API routes
│   ├── risk_endpoints.py              # Risk detection endpoints
│   ├── execution_endpoints.py         # Execution agent endpoints
│   └── ml_endpoints.py                # Machine learning endpoints
│
├── components/                     # React UI components
│   ├── ReasonGraph.tsx                # D3.js decision visualization
│   └── ui/                           # Radix UI component library
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       └── ...
│
├── views/                          # React application views
│   ├── DashboardView.tsx              # Main dashboard
│   ├── ReasonGraphView.tsx            # Decision visualization
│   ├── LiveDemoView.tsx               # Interactive demonstrations
│   └── ...
│
├── data_models/                    # Pydantic schemas
│   └── schemas.py                     # Type-safe data contracts
│
├── tests/                          # Comprehensive test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── performance/                  # Performance benchmarks
│   ├── api/                         # API endpoint tests
│   ├── scenarios/                   # E2E workflow tests
│   └── mock_data.py                 # Shared test data
│
├── e2e/                            # Playwright E2E tests
│   ├── tests/
│   │   ├── user-journey.spec.ts
│   │   ├── backend-integration.spec.ts
│   │   └── ...
│   └── fixtures/                    # Test fixtures
│
├── utils/                          # Shared utilities
│   ├── logger.py                     # Structured logging
│   ├── reason_graph_mapper.py        # Decision tree mapping
│   └── ...
│
├── docs/                           # Documentation
│   ├── architecture/                 # System design docs
│   ├── development/                  # Dev guides
│   ├── guides/                      # Integration guides
│   └── QUICKSTART.md                # 5-minute deployment guide
│
├── main.py                         # FastAPI application server
├── requirements.txt                # Python dependencies
├── package.json                    # Node.js dependencies
├── vite.config.ts                  # Vite configuration
├── playwright.config.ts            # Playwright configuration
├── pytest.ini                      # Pytest configuration
├── .env.example                    # Environment variable template
└── CLAUDE.md                       # Project instructions for AI

```

**Key Directories:**

- **agents/**: Core multi-agent system - each agent is a separate module
- **api/**: FastAPI REST endpoints organized by functionality
- **components/**: Reusable React components (UI library, visualizations)
- **views/**: Full-page React views
- **data_models/**: ALL inter-agent data contracts (Pydantic schemas)
- **tests/**: Comprehensive test suite (80%+ coverage)
- **docs/**: Detailed documentation (architecture, development, deployment)

---

## 🚀 Deployment

FinPilot is production-ready with multiple deployment options:

### Live Deployments

- **Frontend**: [finance-bud.vercel.app](https://finance-bud.vercel.app/) (Vercel)
- **Backend API**: [ecell-production.up.railway.app](https://ecell-production.up.railway.app/) (Railway)

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Agent Configuration
USE_MOCK_AGENTS=false
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434

# External APIs (optional)
ALPHA_VANTAGE_API_KEY=your_key_here
YFINANCE_ENABLED=true

# Database (optional)
DATABASE_URL=postgresql://...

# Performance
ENABLE_CACHING=true
REDIS_URL=redis://localhost:6379
```

### Platform-Specific Deployment

#### Vercel (Frontend)
```bash
# Deploy to Vercel
npm run build
vercel deploy
```
Configuration: `vercel.json` (already configured)

#### Railway (Backend)
```bash
# Deploy to Railway
railway up
```
Configuration: `railway.json` (already configured)

#### Render
Configuration: `render.yaml` (already configured)

#### Fly.io
Configuration: `fly.toml` (already configured)

#### Heroku
Configuration: `Procfile` (already configured)

### Docker (Coming Soon)

```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Production Checklist

- [ ] Set all environment variables in `.env`
- [ ] Configure external API keys (Alpha Vantage, etc.)
- [ ] Set up database (PostgreSQL/Supabase) if persistence needed
- [ ] Configure Redis for caching (optional but recommended)
- [ ] Set up Ollama or other LLM service
- [ ] Run all tests: `pytest tests/ -v && npm run test:e2e`
- [ ] Build frontend: `npm run build`
- [ ] Configure CORS for your frontend domain
- [ ] Set up monitoring and logging
- [ ] Enable health check endpoints

See [docs/development/deployment.md](docs/development/deployment.md) for detailed instructions.

---

## 🔧 Troubleshooting

### Common Issues

#### Backend Won't Start

**Problem**: `ModuleNotFoundError` when running `python main.py`

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/finance-bud

# Reinstall dependencies
pip install -r requirements.txt

# If using a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Frontend Can't Connect to Backend

**Problem**: API requests fail with CORS errors

**Solution**:
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check proxy configuration in `vite.config.ts`:
   ```typescript
   server: {
     proxy: {
       '/api': 'http://127.0.0.1:8000'  // Update port if needed
     }
   }
   ```
3. Restart frontend dev server: `npm run dev`

#### Ollama/LLM Not Working

**Problem**: Conversational agent returns errors

**Solution**:
FinPilot includes hardcoded fallbacks and works without Ollama. To enable Ollama:
```bash
# Install Ollama
pip install ollama

# Set environment variable
export OLLAMA_ENABLED=true
export OLLAMA_BASE_URL=http://localhost:11434

# Restart backend
python main.py
```

If Ollama is unavailable, the system automatically falls back to rule-based responses.

#### Tests Failing

**Problem**: pytest or Playwright tests fail

**Solution**:
```bash
# Backend tests
pytest tests/ -v --tb=short  # Short traceback for easier debugging

# Clear pytest cache
pytest --cache-clear

# Frontend E2E tests
npx playwright install  # Install browsers
npm run test:e2e:debug  # Debug mode
```

#### Port Already in Use

**Problem**: `Address already in use` error

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000  # On macOS/Linux
netstat -ano | findstr :8000  # On Windows

# Kill the process or change port in main.py:
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port
```

#### Performance Issues

**Problem**: Slow response times

**Solution**:
1. Enable caching: Set `ENABLE_CACHING=true` in `.env`
2. Use mock agents for development: `USE_MOCK_AGENTS=true`
3. Check external API rate limits (Alpha Vantage, yfinance)
4. Monitor agent performance in logs

### Getting Help

- **Documentation**: Check [docs/](docs/) for detailed guides
- **Issues**: Report bugs at [GitHub Issues](https://github.com/yourusername/finance-bud/issues)
- **Discord**: Join our community (link coming soon)
- **Email**: support@finpilot.io (coming soon)

---

## 🤝 Contributing

We welcome contributions from the community! FinPilot is built with a spec-driven development approach.

### Development Process

1. **Fork & Clone**
   ```bash
   git clone https://github.com/yourusername/finance-bud.git
   cd finance-bud
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow existing code style (TypeScript, Python type hints)
   - Add tests for new features
   - Update documentation as needed

4. **Run Tests**
   ```bash
   # Backend
   pytest tests/ -v

   # Frontend
   npm run test:e2e
   ```

5. **Commit & Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Describe your changes
   - Reference any related issues
   - Ensure all CI checks pass

### Code Quality Standards

- **Type Safety**: Full TypeScript (frontend) and Pydantic (backend)
- **Testing**: Minimum 80% test coverage for new code
- **Documentation**: Comprehensive docstrings and comments
- **Performance**: Optimized agent communication and response times
- **Linting**: Code must pass linters (pylint, eslint)

### Areas for Contribution

- 🎨 **UI/UX Improvements**: Better visualizations, responsive design
- 🤖 **Agent Enhancements**: New planning algorithms, better heuristics
- 📊 **Data Integrations**: Additional market data sources
- 🧪 **Testing**: Increase test coverage, add edge cases
- 📚 **Documentation**: Tutorials, guides, API examples
- 🌍 **Internationalization**: Multi-language support
- ⚡ **Performance**: Optimization and caching strategies

---

## 🗺️ Roadmap

### Current Version: 1.0 (Production-Ready)

- ✅ 6-agent VP-MAS architecture
- ✅ CMVL workflow with trigger detection
- ✅ Natural language goal processing
- ✅ ReasonGraph visualization
- ✅ Comprehensive testing (unit, integration, E2E)
- ✅ Multi-platform deployment ready

### Version 1.1 (Q1 2025)

- 🔄 **Enhanced CMVL**: Predictive monitoring and proactive adjustments
- 📱 **Mobile Optimization**: Responsive design improvements
- 🔗 **Bank API Integration**: Plaid/Yodlee for account linking
- 📊 **Advanced Analytics**: Historical performance tracking

### Version 1.2 (Q2 2025)

- 🤖 **Multi-LLM Support**: OpenAI, Anthropic Claude, local models
- 🧠 **Reinforcement Learning**: Portfolio optimization with RL
- 🔐 **Enhanced Security**: OAuth, encryption, audit logging
- 🌐 **Internationalization**: Multi-language and multi-currency support

### Version 2.0 (Q3-Q4 2025)

- 🚀 **Graph Neural Networks**: Heavy GNN-based risk detection (GPU-accelerated)
- 🤝 **Collaborative Planning**: Multi-user household planning
- 📈 **Predictive Analytics**: Market trend prediction with ML
- 🔌 **Plugin System**: Extensible architecture for third-party integrations

### Research Areas

- **Reinforcement Learning**: Advanced portfolio optimization
- **Graph Neural Networks**: Large-scale risk detection
- **Natural Language**: Improved conversational capabilities with RAG
- **Federated Learning**: Privacy-preserving collaborative models

---

## 📄 License

See [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgments

- **Kiro AI**: AI-assisted development platform used for spec-driven development
- **FastAPI**: High-performance async web framework
- **React & Vite**: Modern frontend tooling
- **Radix UI**: Accessible component primitives
- **D3.js**: Powerful data visualization
- **Ollama**: Local LLM capabilities
- **Open Source Community**: All the amazing libraries that make this possible

---

<div align="center">

**Built with ❤️ by the FinPilot Team**

[🌐 Live Demo](https://finance-bud.vercel.app/) •
[📚 Documentation](docs/) •
[🐛 Report Bug](https://github.com/yourusername/finance-bud/issues) •
[💡 Request Feature](https://github.com/yourusername/finance-bud/issues)

</div>
