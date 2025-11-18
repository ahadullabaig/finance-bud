
# FinPilot - Verifiable Planning Multi-Agent System (VP-MAS)

🧩 **PROJECT**: FinPilot — Advanced Verifiable Multi-Agent Financial Planner  
🎯 **GOAL**: Sophisticated VP-MAS for adaptive financial planning with natural language processing and real-time market integration  
🤖 **POWERED BY**: Kiro AI-driven development with comprehensive spec-driven architecture

## 🚀 Quick Start

### Prerequisites
- **Node.js 18+** and npm
- **Python 3.11+**
- **Optional**: Ollama for local LLM (conversational AI)

### Setup

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd finpilot
   npm install
   pip install -r requirements.txt
   ```

2. **Start Backend API Server**
   ```bash
   python main.py
   # API available at: http://localhost:8000
   # Documentation: http://localhost:8000/docs
   ```

3. **Start Frontend Development Server**
   ```bash
   npm run dev
   # Frontend available at: http://localhost:5173
   ```

4. **Optional: Enable Conversational AI**
   ```bash
   # Install Ollama for local LLM support
   pip install ollama
   # The system gracefully falls back to rule-based responses if unavailable
   ```

## 🏗️ VP-MAS Architecture

### Core Multi-Agent System
- **🎯 Orchestration Agent (OA)**: Mission control for workflow coordination and trigger management
- **🧠 Planning Agent (PA)**: Advanced financial planning with Guided Search Module (GSM) and Thought of Search (ToS) algorithms
- **🌐 Information Retrieval Agent (IRA)**: Real-time market data integration and external API management
- **✅ Verification Agent (VA)**: Constraint satisfaction and plan validation with CMVL (Continuous Monitoring and Verification Loop)
- **⚡ Execution Agent (EA)**: Plan execution, ledger management, and action implementation
- **💬 Conversational Agent (CA)**: Natural language processing for goal parsing and narrative generation

### Technology Stack
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS + Radix UI
- **Backend**: Python + FastAPI + Pydantic + Uvicorn
- **AI/ML**: Ollama (local LLM) + Rule-based fallbacks
- **Visualization**: D3.js + Recharts for ReasonGraph and financial charts
- **Testing**: Playwright (E2E) + Pytest (Backend) + Comprehensive agent testing
- **Development**: Kiro AI-assisted development with spec-driven workflows

## 🎯 Key Features & Capabilities

### 🧠 Advanced Planning Intelligence
- ✅ **Natural Language Goal Processing**: "I want to retire at 60 with $2M" → Structured financial plan
- ✅ **Guided Search Module (GSM)**: Multi-path strategy exploration with ToS heuristics
- ✅ **Constraint Satisfaction**: Budget ratios, safety funds, debt limits, tax implications
- ✅ **Multi-Scenario Planning**: Retirement, emergency funds, investment, debt payoff, education

### 🔄 Continuous Monitoring & Adaptation
- ✅ **CMVL Workflow**: Real-time plan monitoring and automatic re-verification
- ✅ **Market Trigger Detection**: Volatility spikes, economic changes, sector rotations
- ✅ **Life Event Handling**: Job loss, medical emergencies, business disruption
- ✅ **Adaptive Re-planning**: Automatic plan adjustments with user approval workflows

### 🎨 Transparency & Visualization
- ✅ **ReasonGraph**: Interactive D3.js visualization of decision-making processes
- ✅ **Agent Communication Tracing**: Complete audit trails of inter-agent messages
- ✅ **Before/After Comparisons**: Visual plan changes with detailed explanations
- ✅ **Real-time Dashboard**: Live monitoring of agent activities and system health

### 💬 Conversational AI Interface
- ✅ **Goal Parsing**: Natural language → Structured financial objectives
- ✅ **Narrative Generation**: Plan explanations in human-readable format
- ✅ **Scenario Explanations**: What-if analysis with risk assessments
- ✅ **Hardcoded Fallbacks**: Comprehensive responses when AI services are unavailable

### 🔒 Enterprise-Grade Reliability
- ✅ **Graceful Error Handling**: Circuit breakers and fallback mechanisms
- ✅ **Type-Safe Communication**: Pydantic schemas for all agent interactions
- ✅ **Comprehensive Testing**: Unit, integration, and E2E test coverage
- ✅ **Performance Monitoring**: Request tracking, execution timing, health checks

## 📁 Project Structure

```
/finpilot
├── .kiro/specs/                    # Kiro AI development specifications
│   ├── finpilot-multi-agent-system/   # Core VP-MAS architecture spec
│   ├── complete-cmvl-workflow/         # CMVL implementation spec
│   └── chatbot-api-integration/        # Conversational AI integration spec
├── agents/                         # Multi-agent system implementation
│   ├── orchestration_agent.py         # Workflow coordination
│   ├── planning_agent.py              # GSM and ToS algorithms
│   ├── retriever.py                   # Market data integration
│   ├── verifier.py                    # Constraint validation
│   ├── execution_agent.py             # Plan execution
│   └── conversational_agent.py        # Natural language processing
├── api/                           # FastAPI REST endpoints
│   ├── endpoints.py                   # Core agent endpoints
│   └── conversational_endpoints.py    # Chatbot API routes
├── components/                    # React UI components
│   ├── ReasonGraph.tsx               # D3.js decision visualization
│   └── ui/                          # Radix UI component library
├── data_models/                   # Pydantic schemas
│   └── schemas.py                    # Type-safe data contracts
├── tests/                         # Comprehensive test suite
│   ├── test_agents.py               # Agent unit tests
│   ├── test_integration.py         # Multi-agent integration tests
│   └── ui/                         # Playwright E2E tests
├── utils/                         # Shared utilities
│   ├── logger.py                    # Structured logging
│   └── reason_graph_mapper.py       # Decision tree mapping
├── views/                         # React application views
│   ├── DashboardView.tsx            # Main dashboard
│   ├── ReasonGraphView.tsx          # Decision visualization
│   └── LiveDemoView.tsx             # Interactive demonstrations
└── main.py                        # FastAPI application server
```

## 🔧 Development Workflow

### Backend Development
```bash
# Start API server with hot reload
python main.py

# Run comprehensive test suite
pytest tests/ -v

# Test specific agent functionality
python test_agents_cli.py

# Check API endpoints
curl http://localhost:8000/health
```

### Frontend Development
```bash
# Start development server
npm run dev

# Build for production
npm run build

# Run E2E tests
npm run test:e2e

# Interactive test debugging
npm run test:e2e:ui
```

### Kiro AI Development
```bash
# View current specs
ls .kiro/specs/

# Execute spec tasks (through Kiro IDE)
# Open .kiro/specs/*/tasks.md and click "Start task"
```

## 🌐 API Endpoints

### Core Agent Endpoints
- `POST /api/v1/orchestration/goals` - Submit financial goals
- `POST /api/v1/planning/generate` - Generate financial plans
- `GET /api/v1/market/data` - Fetch market data
- `POST /api/v1/verification/verify` - Validate plans
- `GET /health` - System health check

### Conversational AI Endpoints
- `POST /api/conversational/parse-goal` - Parse natural language goals
- `POST /api/conversational/generate-narrative` - Generate plan narratives
- `POST /api/conversational/explain-scenario` - Explain what-if scenarios
- `GET /api/conversational/health` - Conversational agent status

## 🎮 Interactive Demo Scenarios

### Scenario 1: Retirement Planning
```json
{
  "user_input": "I want to retire at 60 with $2 million",
  "user_context": {"age": 35, "income": 100000}
}
```

### Scenario 2: Life Event Adaptation
```json
{
  "trigger": "job_loss",
  "severity": "high",
  "user_context": {"emergency_fund": 15000, "monthly_expenses": 4000}
}
```

### Scenario 3: Market Volatility Response
```json
{
  "market_event": "volatility_spike",
  "severity": 0.35,
  "affected_assets": ["stocks", "crypto"]
}
```

## 🔍 Monitoring & Observability

### Real-time Monitoring
- **Agent Health**: Individual agent status and performance metrics
- **Workflow Tracing**: Complete request lifecycle tracking with correlation IDs
- **Market Data Quality**: API response times and data freshness indicators
- **CMVL Performance**: Trigger detection and response time monitoring

### Logging Structure
```
/logs
├── agents/           # Individual agent logs
├── system/          # System-wide events
├── performance/     # Performance metrics
└── audit/          # Financial decision audit trails
```

## 🧪 Testing Strategy

### Comprehensive Test Coverage
- **Unit Tests**: Individual agent functionality (80%+ coverage)
- **Integration Tests**: Multi-agent communication and workflows
- **E2E Tests**: Complete user scenarios with Playwright
- **Performance Tests**: Load testing and benchmark validation
- **Mock Testing**: Offline development with realistic data

### Test Execution
```bash
# Run all tests
python run_tests.py

# Specific test categories
pytest tests/test_agents.py          # Agent unit tests
pytest tests/test_integration.py     # Integration tests
npm run test:e2e                     # E2E tests
```

## 🚀 Deployment & Production

### Production Readiness
- ✅ **Environment Configuration**: Production-ready settings
- ✅ **Error Handling**: Comprehensive exception management
- ✅ **Performance Optimization**: Efficient agent communication
- ✅ **Security**: Input validation and secure API design
- ✅ **Monitoring**: Health checks and performance metrics

### Deployment Options
- **Local Development**: `python main.py` + `npm run dev`
- **Docker**: Containerized deployment (see `Dockerfile`)
- **Cloud Platforms**: Railway, Render, or AWS deployment ready

## 🤝 Contributing

### Development Process
1. **Spec-Driven Development**: Create or update Kiro specs in `.kiro/specs/`
2. **Implementation**: Follow spec tasks and requirements
3. **Testing**: Comprehensive test coverage for all changes
4. **Documentation**: Update README and API documentation
5. **Review**: Code review and integration testing

### Code Quality Standards
- **Type Safety**: Full TypeScript (frontend) and Pydantic (backend)
- **Testing**: Minimum 80% test coverage
- **Documentation**: Comprehensive docstrings and comments
- **Performance**: Optimized agent communication and response times

## 📊 Performance Benchmarks

### System Performance
- **Goal Processing**: < 30 seconds for complex multi-constraint scenarios
- **CMVL Response**: < 5 minutes for complete workflow cycles
- **API Response**: < 2 seconds for standard endpoints
- **Market Data**: < 5 minutes for trigger detection and response

### Scalability Metrics
- **Concurrent Users**: Designed for multi-user scenarios
- **Agent Communication**: Efficient message passing with correlation tracking
- **Memory Usage**: Optimized for production deployment
- **Database Performance**: Efficient data storage and retrieval

## 🎯 Roadmap & Future Enhancements

### Planned Features
- 🔄 **Enhanced CMVL**: Predictive monitoring and proactive adjustments
- 🤖 **Advanced AI**: Integration with additional LLM providers
- 📱 **Mobile Support**: Responsive design and mobile optimization
- 🔗 **External Integrations**: Bank APIs and financial institution connections
- 📈 **Advanced Analytics**: Machine learning for personalized recommendations

### Research Areas
- **Reinforcement Learning**: Portfolio optimization algorithms
- **Graph Neural Networks**: Advanced risk detection
- **Natural Language**: Improved conversational capabilities
- **Predictive Analytics**: Market trend prediction and analysis

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

**Built with Kiro AI** - Leveraging AI-assisted development for sophisticated financial planning systems

  
