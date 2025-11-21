# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinPilot is a Verifiable Planning Multi-Agent System (VP-MAS) for financial planning. It uses a sophisticated multi-agent architecture to provide adaptive financial planning with natural language processing, real-time market integration, and continuous monitoring.

## Project Structure

```
finance-bud/
├── backend/                    # Python FastAPI backend
│   ├── agents/                 # Agent implementations
│   ├── api/                    # FastAPI endpoints
│   ├── data_models/            # Pydantic schemas
│   ├── tests/                  # Python tests
│   ├── utils/                  # Utilities
│   ├── main.py                 # Backend entry point
│   ├── config.py               # Settings
│   └── requirements.txt        # Python dependencies
├── frontend/                   # React/Vite frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── views/              # Page views
│   │   ├── styles/             # CSS files
│   │   ├── lib/                # Utilities
│   │   ├── App.tsx             # App root
│   │   └── main.tsx            # Entry point
│   ├── e2e/                    # Playwright tests
│   ├── index.html              # HTML entry
│   ├── package.json            # Node dependencies
│   └── vite.config.ts          # Vite config
├── docs/                       # Documentation
└── .github/workflows/          # CI/CD
```

### Core Agents

The system consists of six specialized agents that communicate through structured message passing:

1. **Orchestration Agent (OA)** - `backend/agents/orchestration_agent.py`: Mission control for workflow coordination, trigger management, and goal parsing. Manages circuit breakers and system reliability.

2. **Planning Agent (PA)** - `backend/agents/planning_agent.py`: Advanced financial planning using Guided Search Module (GSM) and Thought of Search (ToS) algorithms. Generates multi-path strategies with constraint-based optimization.

3. **Information Retrieval Agent (IRA)** - `backend/agents/retriever.py`: Real-time market data integration and external API management. Interfaces with financial APIs like yfinance and Alpha Vantage.

4. **Verification Agent (VA)** - `backend/agents/verifier.py`: Constraint satisfaction and plan validation with CMVL (Continuous Monitoring and Verification Loop) capabilities.

5. **Execution Agent (EA)** - `backend/agents/execution_agent.py`: Plan execution, ledger management, and action implementation.

6. **Conversational Agent (CA)** - `backend/agents/conversational_agent.py`: Natural language processing for goal parsing and narrative generation. Uses Ollama for local LLM with hardcoded fallbacks when unavailable.

### Key Architectural Concepts

**Agent Communication**: All agents extend `BaseAgent` (backend/agents/base_agent.py) and communicate via `AgentMessage` objects defined in `backend/data_models/schemas.py`. Messages include correlation IDs for tracking, performance metrics, and structured payloads.

**CMVL Workflow** (backend/agents/cmvl_workflow.py): The Continuous Monitoring and Verification Loop is the core workflow that monitors financial plans, detects triggers (market volatility, life events), and coordinates re-planning across agents.

**Data Contracts**: All inter-agent communication uses Pydantic models from `backend/data_models/schemas.py`. These provide type safety, validation, and comprehensive documentation.

**Reason Graph**: Decision-making processes are tracked through `ReasoningTrace` and `DecisionPoint` objects, which are visualized in the frontend using D3.js (frontend/src/components/ReasonGraph.tsx).

## Development Commands

### Backend Development

Start the FastAPI development server (with hot reload):
```bash
cd backend
PYTHONPATH=. python main.py
# Or use the runner script:
./run.sh
# API available at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

Run all backend tests:
```bash
cd backend
PYTHONPATH=. pytest tests/ -v
```

Run specific test categories:
```bash
cd backend
PYTHONPATH=. pytest tests/unit/test_agents.py -v       # Agent unit tests
PYTHONPATH=. pytest tests/integration/ -v               # Integration tests
PYTHONPATH=. pytest tests/performance/ -v               # Performance tests
PYTHONPATH=. pytest -m asyncio                          # Only async tests
PYTHONPATH=. pytest -m integration                      # Only integration tests
```

### Frontend Development

Start Vite development server (port 3000, proxies /api to localhost:8000):
```bash
cd frontend
npm install  # First time only
npm run dev
```

Build for production:
```bash
cd frontend
npm run build
```

Run E2E tests with Playwright:
```bash
cd frontend
npm run test:e2e              # Headless mode
npm run test:e2e:ui           # Interactive UI mode
npm run test:e2e:headed       # Headed browser mode
npm run test:e2e:debug        # Debug mode
npm run test:e2e:report       # View test report
npm run test:e2e:codegen      # Generate test code
```

## Architecture Guidelines

### Agent Implementation

When creating or modifying agents:
- Extend `BaseAgent` from `backend/agents/base_agent.py`
- Use `AgentMessage` for all inter-agent communication
- Include `correlation_id` in messages for tracking across agent boundaries
- Add performance metrics using `PerformanceMetrics` objects
- Log to structured logs with correlation IDs

### Data Model Changes

When modifying data contracts in `backend/data_models/schemas.py`:
- Use Pydantic v2 syntax with Field(...) for all attributes
- Add comprehensive docstrings and validation
- Update both request and response models
- Ensure backward compatibility or coordinate breaking changes across all agents
- Run all tests to verify schema changes don't break agent communication

### Frontend Components

React components follow these patterns:
- TypeScript with strict typing
- Radix UI primitives for accessible components (in `frontend/src/components/ui/`)
- Tailwind CSS for styling
- D3.js for data visualizations (especially ReasonGraph)
- Recharts for financial charts

Component organization:
- `frontend/src/components/` - Reusable components (ReasonGraph, etc.)
- `frontend/src/components/ui/` - Radix UI-based primitives (buttons, cards, etc.)
- `frontend/src/views/` - Full page views (DashboardView, LiveDemoView, etc.)

### API Endpoints

API routes are organized by functionality in the `backend/api/` directory:
- `backend/api/orchestration_endpoints.py` - Orchestration agent endpoints
- `backend/api/conversational_endpoints.py` - Conversational AI endpoints
- `backend/api/risk_endpoints.py` - Risk detection endpoints
- `backend/api/execution_endpoints.py` - Execution agent endpoints
- `backend/api/ml_endpoints.py` - Machine learning endpoints

All endpoints follow FastAPI patterns with Pydantic request/response models.

### Testing Strategy

**Backend Tests** (pytest):
- Unit tests: `backend/tests/unit/` - Individual agent functionality in isolation
- Integration tests: `backend/tests/integration/` - Multi-agent communication and workflows
- Mock data: Use `backend/tests/mock_data.py` for consistent test data
- Async tests: Mark with `@pytest.mark.asyncio`
- Performance tests: `backend/tests/performance/` - Use `pytest-benchmark` for validation

**Frontend Tests** (Playwright):
- E2E tests live in `frontend/e2e/tests/`
- Test interactive scenarios: goal submission, plan visualization, CMVL workflows
- Use fixtures from `frontend/e2e/fixtures/` for test data
- Page object pattern for maintainability

### Risk Detection - Light vs Heavy

The codebase includes two approaches:
- **Light (default)**: `backend/agents/graph_risk_detector.py` using NetworkX and scikit-learn for CPU-friendly, explainable detection
- **Heavy (production path)**: GPU-accelerated with NVIDIA cuGraph + GNN (not implemented, see docs/PHASE_6_IMPLEMENTATION_SUMMARY.md for migration guidance)

For production, use a hybrid approach: heavy GNN for high-throughput inference, light for explainability and fallback.

## Important Files

- `backend/main.py` - FastAPI application entry point, initializes all agents
- `backend/data_models/schemas.py` - ALL data contracts for agent communication
- `backend/agents/base_agent.py` - Base class all agents must extend
- `backend/agents/communication.py` - Agent communication framework with circuit breakers
- `backend/agents/cmvl_workflow.py` - CMVL workflow orchestration
- `frontend/src/App.tsx` - React application root
- `frontend/vite.config.ts` - Vite configuration with proxy setup
- `backend/pytest.ini` - Pytest configuration with markers
- `frontend/playwright.config.ts` - Playwright E2E test configuration

## Configuration

Backend API configuration:
- Default port: 8000
- CORS enabled for frontend development
- Optional: Create `backend/api_config.json` from `backend/api_config.example.json` for external API keys
- **Important**: Always set `PYTHONPATH=.` when running from backend directory

Frontend proxy configuration:
- Development server runs on port 3000
- `/api` requests are proxied to `http://127.0.0.1:8000`
- Change proxy target in `frontend/vite.config.ts` if backend runs on different port

Environment variables:
- Copy `.env.example` to `.env` for local configuration
- API keys for financial services (optional, graceful fallback to mocks)
- Ollama configuration for local LLM

## Dependencies

Python 3.11+ required. Key dependencies:
- FastAPI + Uvicorn for backend
- Pydantic v2 for data validation
- LangChain for agent orchestration
- NetworkX for graph-based risk detection
- Pytest + pytest-asyncio for testing
- Optional: Ollama for local LLM, yfinance for market data

Node 18+ required. Key dependencies:
- React 18 + TypeScript
- Vite for build tooling
- Radix UI for accessible components
- D3.js for visualizations
- Recharts for charts
- Playwright for E2E testing

## Common Workflows

### Running the Full Stack Locally

Terminal 1 - Backend:
```bash
cd backend
PYTHONPATH=. uvicorn main:app --reload --port 8000
# Or: ./run.sh
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Visit http://localhost:3000 for the UI, http://localhost:8000/docs for API documentation.

### Adding a New Agent Feature

1. Update data contracts in `backend/data_models/schemas.py` if needed
2. Modify the agent class (e.g., `backend/agents/planning_agent.py`)
3. Update the API endpoint (e.g., `backend/api/planning_endpoints.py`)
4. Add unit tests in `backend/tests/unit/`
5. Add integration test in `backend/tests/integration/`
6. Run tests: `cd backend && PYTHONPATH=. pytest tests/ -v`
7. Update frontend if UI changes needed

### Testing CMVL Workflow

The CMVL workflow can be tested end-to-end:
```bash
cd backend
PYTHONPATH=. pytest tests/integration/test_cmvl_advanced.py -v
```

Frontend CMVL visualization available in frontend/src/views/LiveDemoView.tsx with trigger simulation.
