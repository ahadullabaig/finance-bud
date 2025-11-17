
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

  
