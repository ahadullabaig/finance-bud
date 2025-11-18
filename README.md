
# FinPilot - Advanced Multi-Agent Financial Planner

🧩 **PROJECT**: FinPilot — Advanced Verifiable Multi-Agent Financial Planner  
🎯 **GOAL**: Sophisticated Verifiable Planning Multi-Agent System (VP-MAS) for adaptive financial planning with Supabase backend

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Supabase account

### Setup

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd finance-bud
   npm install
   pip install -r requirements.txt
   ```

2. **Configure Supabase**
   ```bash
   python setup_supabase.py
   ```
   Follow the printed instructions to set up your Supabase project.

3. **Environment Variables**
   Update `.env` with your Supabase credentials:
   ```env
   SUPABASE_URL=your-project-url
   SUPABASE_ANON_KEY=your-anon-key
   SUPABASE_SERVICE_KEY=your-service-key
   ```

4. **Database Setup**
   - Go to Supabase SQL Editor
   - Run the migration from `supabase/migrations/001_initial_schema.sql`

5. **Start Development**
   ```bash
   npm run dev
   ```

## 🏗️ Architecture

### Multi-Agent System
- **Orchestration Agent (OA)**: Workflow coordination
- **Planning Agent (PA)**: Financial plan generation with Guided Search
- **Information Retrieval Agent (IRA)**: Market data and intelligence
- **Verification Agent (VA)**: Plan validation and compliance
- **Execution Agent (EA)**: Plan execution and monitoring

### Technology Stack
- **Frontend**: React + TypeScript + Vite
- **Backend**: Python + FastAPI + Pydantic
- **Database**: Supabase (PostgreSQL)
- **Real-time**: Supabase Realtime
- **Authentication**: Supabase Auth
- **APIs**: Alpha Vantage, Yahoo Finance, IEX Cloud

## 📊 Features

- ✅ **Real-time Financial Planning**: Live market data integration
- ✅ **Multi-Agent Coordination**: Sophisticated agent communication
- ✅ **Guided Search (ToS)**: Advanced planning algorithms
- ✅ **Continuous Verification (CMVL)**: Real-time plan validation
- ✅ **ReasonGraph Visualization**: Transparent decision making
- ✅ **Compliance Tracking**: Regulatory requirement monitoring
- ✅ **Risk Assessment**: Comprehensive risk profiling
- ✅ **Tax Optimization**: Tax-efficient planning strategies

## 🔧 Development

### Project Structure
```
/finpilot
  /agents          # Multi-agent system
  /api             # REST API endpoints
  /components      # React UI components
  /data_models     # Pydantic schemas
  /lib             # Frontend utilities
  /supabase        # Database operations
  /utils           # Shared utilities
  /views           # React views
```

### Key Commands
```bash
# Frontend development
npm run dev
npm run build

# Backend testing
pytest tests/
python -m pytest tests/ -v

# Code quality
black .
flake8 .
mypy .
```

## 🔐 Security

- Row Level Security (RLS) enabled
- JWT-based authentication
- API rate limiting
- Input validation with Pydantic
- Secure environment variable management

## 📈 Monitoring

- Real-time agent communication logs
- Performance metrics tracking
- Market data quality monitoring
- Plan execution audit trails

## ⚖️ Risk Detection — light and heavy implementations

- This project includes two approaches for graph-based risk detection:
   - Light (default): `GraphRiskDetector` — a CPU-friendly, explainable implementation using NetworkX and scikit-learn. This is the detector included and used by default in this repository. It lives at `agents/graph_risk_detector.py` and powers the `/api/risk` endpoints (see `api/risk_endpoints.py`).
   - Heavy (production path): NVIDIA/cuGraph + GNN — a GPU-accelerated graph processing and Graph Neural Network (GNN) approach for large-scale, high-sensitivity detection. The repo includes upgrade notes and an interface-ready design; migrating to this requires GPU infra and trained models (see `PHASE_6_IMPLEMENTATION_SUMMARY.md` for migration guidance).

Notes:
- By default this repository provides the lightweight NetworkX detector for local development, CI, and demos.
- Recommended production approach is a hybrid: use the heavy GNN detector for high-throughput inference and the light detector for explainability, fallback, and analyst-facing explanations.
- To add or switch to a GPU/GNN implementation, implement the same interface (e.g., `BaseGraphRiskDetector`) and provide a runtime selection (env var/config) that chooses `networkx|gnn|hybrid`.


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

  
