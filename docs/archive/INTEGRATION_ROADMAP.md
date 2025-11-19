# FinPilot Integration Roadmap & CI/CD Resolution Plan

> **Backend Developer's Guide to System Integration**
> Last Updated: 2025-11-18
> Status: 🚧 In Progress

---

## 📋 Table of Contents

- [Phase 0: Critical CI/CD Fixes](#phase-0-critical-cicd-fixes)
- [Phase 1: Backend Foundation](#phase-1-backend-foundation)
- [Phase 2: Database & Infrastructure](#phase-2-database--infrastructure)
- [Phase 3: Agent System Integration](#phase-3-agent-system-integration)
- [Phase 4: API Layer Integration](#phase-4-api-layer-integration)
- [Phase 5: External Services Integration](#phase-5-external-services-integration)
- [Phase 6: Testing & Validation](#phase-6-testing--validation)
- [Phase 7: Production Readiness](#phase-7-production-readiness)

---

## 🎯 Integration Goals

1. ✅ Fix all CI/CD pipeline failures
2. ✅ Ensure all test suites pass (Python backend focus)
3. ✅ Establish working database layer
4. ✅ Integrate all agents with proper communication
5. ✅ Connect external APIs (Alpha Vantage, Yahoo Finance, etc.)
6. ✅ Implement proper error handling and logging
7. ✅ Achieve production-grade reliability

---

## Phase 0: Critical CI/CD Fixes

**Priority:** 🔴 CRITICAL
**Timeline:** Week 1, Days 1-2
**Owner:** Backend Team Lead

### 0.1 Pytest Configuration & Async Compatibility

- [x] Fix pytest.ini section header (`[tool:pytest]` → `[pytest]`)
- [x] Add async configuration (`asyncio_default_fixture_loop_scope = function`)
- [ ] Install pytest-timeout plugin for timeout config
  ```bash
  pip install pytest-timeout
  ```
- [ ] Verify pytest mode is AUTO instead of STRICT
- [ ] Test configuration:
  ```bash
  pytest --collect-only  # Should show all tests
  ```

### 0.2 Data Models Test Suite (COMPLETED ✅)

- [x] Fix import errors in `data_models/test_schemas.py`
- [x] Migrate Pydantic V1 validators to V2
- [x] Fix computed field validators
- [x] All 12 tests passing
- [ ] Add coverage reporting
  ```bash
  pytest data_models/test_schemas.py --cov=data_models --cov-report=html
  ```

### 0.3 Agent Tests - Async Fixture Resolution

**File:** `tests/test_agents.py`
**Issue:** 26 tests failing due to async fixture incompatibility

#### Tasks:

- [ ] **Identify all async fixtures**
  ```bash
  grep -n "@pytest.fixture" tests/test_agents.py
  ```

- [ ] **Fix `orchestration_agent` fixture**
  ```python
  # Before:
  @pytest.fixture
  def orchestration_agent():
      # ...

  # After:
  @pytest.fixture
  @pytest.mark.asyncio
  async def orchestration_agent():
      from agents.orchestration_agent import OrchestrationAgent
      agent = OrchestrationAgent(agent_id="test_oa")
      await agent.initialize()
      yield agent
      await agent.cleanup()
  ```

- [ ] **Fix all test functions to be async**
  ```python
  # Before:
  def test_orchestration_agent_initialization(orchestration_agent):
      result = orchestration_agent.process_request(...)

  # After:
  @pytest.mark.asyncio
  async def test_orchestration_agent_initialization(orchestration_agent):
      result = await orchestration_agent.process_request(...)
  ```

- [ ] **Fix these specific fixtures:**
  - [ ] `orchestration_agent` fixture
  - [ ] `planning_agent` fixture
  - [ ] `ira_agent` fixture (Information Retrieval Agent)
  - [ ] `verification_agent` fixture
  - [ ] `execution_agent` fixture
  - [ ] `mock_communication_bus` fixture

- [ ] **Update test assertions to handle async returns**
  ```python
  # Handle coroutines properly
  result = await agent.handle_message(message)
  assert result.status == "success"
  ```

- [ ] **Run agent tests and verify passing**
  ```bash
  pytest tests/test_agents.py -v --tb=short
  ```

### 0.4 Integration Tests - Async Fixture Resolution

**File:** `tests/test_integration.py`
**Issue:** 6 tests with async fixture errors

#### Tasks:

- [ ] **Fix `integrated_system` fixture**
  ```python
  @pytest.fixture
  @pytest.mark.asyncio
  async def integrated_system():
      # Initialize all agents
      agents = {
          'orchestration': OrchestrationAgent(...),
          'planning': PlanningAgent(...),
          'ira': InformationRetrievalAgent(...),
          'verification': VerificationAgent(...),
          'execution': ExecutionAgent(...)
      }

      # Initialize communication bus
      comm_bus = CommunicationBus()
      await comm_bus.initialize()

      # Connect all agents
      for agent in agents.values():
          await agent.connect(comm_bus)

      yield {'agents': agents, 'bus': comm_bus}

      # Cleanup
      await comm_bus.shutdown()
      for agent in agents.values():
          await agent.cleanup()
  ```

- [ ] **Convert all integration tests to async**
  - [ ] `test_complete_planning_workflow`
  - [ ] `test_cmvl_trigger_scenario`
  - [ ] `test_concurrent_user_sessions`
  - [ ] `test_agent_failure_recovery`
  - [ ] `test_end_to_end_financial_planning`
  - [ ] `test_performance_under_load`

- [ ] **Run integration tests**
  ```bash
  pytest tests/test_integration.py -v --tb=short
  ```

### 0.5 CI/CD Pipeline Validation

- [ ] **Update `.github/workflows/ci-cd.yml` if needed**
  - [ ] Ensure Python version matches (currently 3.10, 3.11, 3.12 matrix)
  - [ ] Add Python 3.14 to matrix (current development version)
  - [ ] Verify all test paths are correct

- [ ] **Test locally what CI/CD will run**
  ```bash
  # Data model tests
  pytest data_models/test_schemas.py -v --cov=data_models --cov-report=xml

  # Agent tests
  pytest tests/test_agents.py -v --cov=agents --cov-report=xml

  # Integration tests
  pytest tests/test_integration.py -v --cov-append --cov-report=xml
  ```

- [ ] **Fix any remaining failures before committing**

### 0.6 Code Quality Tools Setup

- [ ] **Install code quality tools**
  ```bash
  pip install black flake8 mypy bandit safety isort
  ```

- [ ] **Run Black (code formatting)**
  ```bash
  black . --check --diff
  black .  # Apply formatting
  ```

- [ ] **Run Flake8 (linting)**
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  flake8 . --count --max-line-length=100 --statistics
  ```

- [ ] **Run MyPy (type checking)**
  ```bash
  mypy data_models/ agents/ api/ --ignore-missing-imports
  ```

- [ ] **Run Bandit (security analysis)**
  ```bash
  bandit -r . -f json -o bandit-report.json
  ```

- [ ] **Run Safety (dependency security check)**
  ```bash
  safety check --json --output safety-report.json
  ```

- [ ] **Fix critical issues found by tools**

### Phase 0 Completion Criteria

- [ ] All data model tests passing (12/12) ✅
- [ ] All agent tests passing (26/26)
- [ ] All integration tests passing (6/6)
- [ ] Black formatting applied
- [ ] Flake8 critical issues resolved
- [ ] No HIGH/CRITICAL security vulnerabilities
- [ ] CI/CD pipeline runs green locally

---

## Phase 1: Backend Foundation

**Priority:** 🔴 HIGH
**Timeline:** Week 1, Days 3-5
**Owner:** Backend Developer

### 1.1 Python Environment Setup

- [ ] **Create virtual environment**
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # Linux/Mac
  # or
  venv\Scripts\activate  # Windows
  ```

- [ ] **Install all dependencies**
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt 2>&1 | tee install.log
  ```

- [ ] **Handle dependency conflicts**
  - [ ] Check install.log for errors
  - [ ] Note packages that failed to install
  - [ ] Create `requirements-optional.txt` for optional deps
  - [ ] Document system dependencies needed (TA-Lib, etc.)

- [ ] **Verify core dependencies installed**
  ```bash
  python -c "import fastapi, pydantic, aiohttp, redis; print('Core deps OK')"
  ```

### 1.2 Configuration Management

- [ ] **Create environment configuration**
  ```bash
  cp .env.example .env
  ```

- [ ] **Set up development environment variables**
  ```bash
  # Edit .env
  ENVIRONMENT=development
  LOG_LEVEL=DEBUG
  API_VERSION=v1

  # Database (local for now)
  DATABASE_URL=postgresql://localhost:5432/finpilot_dev

  # Redis (local)
  REDIS_URL=redis://localhost:6379/0

  # JWT Configuration
  JWT_SECRET=$(openssl rand -hex 32)
  JWT_ALGORITHM=HS256
  JWT_EXPIRATION_MINUTES=60

  # Encryption
  ENCRYPTION_KEY=$(openssl rand -hex 32)
  ```

- [ ] **Create config loader module**
  ```python
  # lib/config.py
  from pydantic_settings import BaseSettings
  from typing import Optional

  class Settings(BaseSettings):
      environment: str = "development"
      log_level: str = "INFO"
      database_url: Optional[str] = None
      redis_url: Optional[str] = None
      jwt_secret: str
      encryption_key: str

      class Config:
          env_file = ".env"

  settings = Settings()
  ```

- [ ] **Validate configuration loading**
  ```bash
  python -c "from lib.config import settings; print(settings.environment)"
  ```

### 1.3 Logging Setup

- [ ] **Create centralized logging configuration**
  ```python
  # lib/logging_config.py
  import logging
  import structlog
  from pythonjsonlogger import jsonlogger

  def setup_logging(log_level: str = "INFO"):
      structlog.configure(
          processors=[
              structlog.stdlib.filter_by_level,
              structlog.stdlib.add_logger_name,
              structlog.stdlib.add_log_level,
              structlog.stdlib.PositionalArgumentsFormatter(),
              structlog.processors.TimeStamper(fmt="iso"),
              structlog.processors.StackInfoRenderer(),
              structlog.processors.format_exc_info,
              structlog.processors.UnicodeDecoder(),
              structlog.processors.JSONRenderer()
          ],
          context_class=dict,
          logger_factory=structlog.stdlib.LoggerFactory(),
          cache_logger_on_first_use=True,
      )

      logging.basicConfig(
          level=log_level,
          format="%(message)s",
      )
  ```

- [ ] **Add logging to all agents**
  ```python
  # In each agent
  import structlog
  logger = structlog.get_logger(__name__)

  # Usage
  logger.info("agent_initialized", agent_id=self.agent_id)
  logger.error("processing_failed", error=str(e), agent_id=self.agent_id)
  ```

### 1.4 Error Handling Framework

- [ ] **Create custom exception hierarchy**
  ```python
  # lib/exceptions.py
  class FinPilotException(Exception):
      """Base exception for FinPilot"""
      pass

  class AgentException(FinPilotException):
      """Agent-related errors"""
      pass

  class CommunicationException(FinPilotException):
      """Agent communication errors"""
      pass

  class ValidationException(FinPilotException):
      """Data validation errors"""
      pass

  class ExternalAPIException(FinPilotException):
      """External API integration errors"""
      pass

  class DatabaseException(FinPilotException):
      """Database operation errors"""
      pass
  ```

- [ ] **Implement error handlers in FastAPI**
  ```python
  # api/error_handlers.py
  from fastapi import Request, status
  from fastapi.responses import JSONResponse

  @app.exception_handler(FinPilotException)
  async def finpilot_exception_handler(request: Request, exc: FinPilotException):
      return JSONResponse(
          status_code=status.HTTP_400_BAD_REQUEST,
          content={
              "error": exc.__class__.__name__,
              "message": str(exc),
              "path": str(request.url)
          }
      )
  ```

### 1.5 Database Connection Pool

- [ ] **Set up database connection manager**
  ```python
  # database/connection.py
  from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
  from sqlalchemy.orm import sessionmaker
  from lib.config import settings

  engine = create_async_engine(
      settings.database_url,
      echo=settings.environment == "development",
      pool_size=10,
      max_overflow=20,
      pool_pre_ping=True
  )

  async_session = sessionmaker(
      engine, class_=AsyncSession, expire_on_commit=False
  )

  async def get_db():
      async with async_session() as session:
          yield session
  ```

- [ ] **Test database connection**
  ```python
  # test_db_connection.py
  import asyncio
  from database.connection import get_db

  async def test_connection():
      async for db in get_db():
          result = await db.execute("SELECT 1")
          print("Database connected:", result.scalar())

  asyncio.run(test_connection())
  ```

### Phase 1 Completion Criteria

- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] .env file configured with development settings
- [ ] Configuration loading working
- [ ] Logging configured and tested
- [ ] Error handling framework in place
- [ ] Database connection pool working

---

## Phase 2: Database & Infrastructure

**Priority:** 🟡 HIGH
**Timeline:** Week 2, Days 1-3
**Owner:** Backend Developer + DevOps

### 2.1 Database Decision & Setup

- [ ] **DECISION: Choose database approach**
  - Option A: Full Supabase (recommended for speed)
  - Option B: Self-hosted PostgreSQL + custom auth
  - Option C: Hybrid (Supabase auth + custom PostgreSQL)

  **Selected:** _____________

#### If Supabase (Option A):

- [ ] **Create Supabase project**
  - Go to https://supabase.com
  - Create new project
  - Note: Project URL, Anon Key, Service Key

- [ ] **Install Supabase CLI**
  ```bash
  npm install -g supabase
  supabase init
  ```

- [ ] **Link to remote project**
  ```bash
  supabase link --project-ref YOUR_PROJECT_REF
  ```

- [ ] **Migrate existing SQL**
  ```bash
  # Copy existing migration
  cp database/migrations/001_initial_schema.sql supabase/migrations/20251118000000_initial_schema.sql

  # Push to Supabase
  supabase db push
  ```

- [ ] **Update .env with Supabase credentials**
  ```bash
  SUPABASE_URL=https://xxxxx.supabase.co
  SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  DATABASE_URL=postgresql://postgres:[PASSWORD]@db.xxxxx.supabase.co:5432/postgres
  ```

#### If Self-Hosted (Option B):

- [ ] **Start PostgreSQL with Docker**
  ```bash
  docker run -d \
    --name finpilot-postgres \
    -e POSTGRES_PASSWORD=dev_password \
    -e POSTGRES_DB=finpilot_dev \
    -p 5432:5432 \
    postgres:15-alpine
  ```

- [ ] **Run migrations**
  ```bash
  alembic upgrade head
  ```

### 2.2 Redis Cache Setup

- [ ] **Start Redis locally**
  ```bash
  docker run -d \
    --name finpilot-redis \
    -p 6379:6379 \
    redis:7-alpine
  ```

- [ ] **Create Redis client wrapper**
  ```python
  # database/cache.py
  import aioredis
  from lib.config import settings

  class CacheManager:
      def __init__(self):
          self.redis = None

      async def connect(self):
          self.redis = await aioredis.from_url(
              settings.redis_url,
              encoding="utf-8",
              decode_responses=True
          )

      async def get(self, key: str):
          return await self.redis.get(key)

      async def set(self, key: str, value: str, expire: int = 3600):
          await self.redis.set(key, value, ex=expire)

      async def delete(self, key: str):
          await self.redis.delete(key)

  cache = CacheManager()
  ```

- [ ] **Test Redis connection**
  ```bash
  python -c "import asyncio; from database.cache import cache; asyncio.run(cache.connect())"
  ```

### 2.3 Database Models & Migrations

- [ ] **Review existing migration**
  ```bash
  cat database/migrations/001_initial_schema.sql
  ```

- [ ] **Create SQLAlchemy models matching schema**
  ```python
  # database/models.py
  from sqlalchemy import Column, String, DateTime, Decimal, JSON
  from sqlalchemy.ext.declarative import declarative_base
  from datetime import datetime

  Base = declarative_base()

  class User(Base):
      __tablename__ = "users"

      user_id = Column(String, primary_key=True)
      email = Column(String, unique=True, nullable=False)
      created_at = Column(DateTime, default=datetime.utcnow)
      # ... other fields

  class FinancialPlan(Base):
      __tablename__ = "financial_plans"

      plan_id = Column(String, primary_key=True)
      user_id = Column(String, nullable=False)
      plan_data = Column(JSON, nullable=False)
      created_at = Column(DateTime, default=datetime.utcnow)
      # ... other fields
  ```

- [ ] **Set up Alembic for migrations**
  ```bash
  alembic init alembic
  ```

- [ ] **Configure alembic.ini**
  ```ini
  sqlalchemy.url = postgresql://localhost:5432/finpilot_dev
  ```

- [ ] **Create initial migration**
  ```bash
  alembic revision --autogenerate -m "Initial schema"
  alembic upgrade head
  ```

### 2.4 Database Repository Pattern

- [ ] **Create repository base class**
  ```python
  # database/repositories/base.py
  from typing import Generic, TypeVar, Type, Optional, List
  from sqlalchemy.ext.asyncio import AsyncSession
  from sqlalchemy import select

  T = TypeVar('T')

  class BaseRepository(Generic[T]):
      def __init__(self, model: Type[T], session: AsyncSession):
          self.model = model
          self.session = session

      async def get_by_id(self, id: str) -> Optional[T]:
          result = await self.session.execute(
              select(self.model).where(self.model.id == id)
          )
          return result.scalar_one_or_none()

      async def get_all(self) -> List[T]:
          result = await self.session.execute(select(self.model))
          return result.scalars().all()

      async def create(self, entity: T) -> T:
          self.session.add(entity)
          await self.session.commit()
          await self.session.refresh(entity)
          return entity

      async def update(self, entity: T) -> T:
          await self.session.commit()
          await self.session.refresh(entity)
          return entity

      async def delete(self, id: str) -> bool:
          entity = await self.get_by_id(id)
          if entity:
              await self.session.delete(entity)
              await self.session.commit()
              return True
          return False
  ```

- [ ] **Create specific repositories**
  ```python
  # database/repositories/user_repository.py
  from database.repositories.base import BaseRepository
  from database.models import User

  class UserRepository(BaseRepository[User]):
      async def get_by_email(self, email: str) -> Optional[User]:
          result = await self.session.execute(
              select(User).where(User.email == email)
          )
          return result.scalar_one_or_none()
  ```

### Phase 2 Completion Criteria

- [ ] Database running (Supabase or PostgreSQL)
- [ ] Redis cache running
- [ ] Migrations executed successfully
- [ ] SQLAlchemy models created
- [ ] Repository pattern implemented
- [ ] Database connection tested
- [ ] Cache operations tested

---

## Phase 3: Agent System Integration

**Priority:** 🔴 CRITICAL
**Timeline:** Week 2, Days 4-7
**Owner:** Backend Developer (Agent Systems)

### 3.1 Communication Bus Implementation

- [ ] **Enhance CommunicationBus with Redis backend**
  ```python
  # agents/communication_bus.py
  import asyncio
  from typing import Dict, Callable, List
  from database.cache import cache
  import json

  class CommunicationBus:
      def __init__(self):
          self.subscribers: Dict[str, List[Callable]] = {}
          self.redis = None

      async def initialize(self):
          await cache.connect()
          self.redis = cache.redis

      async def publish(self, channel: str, message: dict):
          # Publish to Redis for distributed systems
          await self.redis.publish(
              channel,
              json.dumps(message)
          )

          # Also notify local subscribers
          if channel in self.subscribers:
              for callback in self.subscribers[channel]:
                  asyncio.create_task(callback(message))

      async def subscribe(self, channel: str, callback: Callable):
          if channel not in self.subscribers:
              self.subscribers[channel] = []
          self.subscribers[channel].append(callback)

      async def request_reply(self, channel: str, message: dict, timeout: int = 30):
          correlation_id = message.get('correlation_id')
          reply_channel = f"{channel}:reply:{correlation_id}"

          # Create one-time subscriber for reply
          reply_future = asyncio.Future()

          async def reply_handler(msg):
              reply_future.set_result(msg)

          await self.subscribe(reply_channel, reply_handler)
          await self.publish(channel, message)

          try:
              reply = await asyncio.wait_for(reply_future, timeout=timeout)
              return reply
          except asyncio.TimeoutError:
              raise CommunicationException(f"No reply received within {timeout}s")
  ```

- [ ] **Test communication bus**
  ```python
  # Test in test_communication_framework.py
  @pytest.mark.asyncio
  async def test_bus_pub_sub():
      bus = CommunicationBus()
      await bus.initialize()

      received = []

      async def handler(msg):
          received.append(msg)

      await bus.subscribe("test_channel", handler)
      await bus.publish("test_channel", {"test": "data"})

      await asyncio.sleep(0.1)
      assert len(received) == 1
      assert received[0]["test"] == "data"
  ```

### 3.2 Base Agent Enhancement

- [ ] **Add database access to BaseAgent**
  ```python
  # agents/base_agent.py
  from database.connection import get_db
  from database.repositories.user_repository import UserRepository

  class BaseAgent:
      def __init__(self, agent_id: str):
          self.agent_id = agent_id
          self.bus = None
          self.db_session = None
          self.logger = structlog.get_logger(self.__class__.__name__)

      async def initialize(self):
          # Get database session
          async for session in get_db():
              self.db_session = session
              break

          self.logger.info("agent_initialized", agent_id=self.agent_id)

      async def connect_bus(self, bus: CommunicationBus):
          self.bus = bus
          await bus.subscribe(f"agent:{self.agent_id}", self.handle_message)

      async def handle_message(self, message: dict):
          raise NotImplementedError("Subclass must implement handle_message")

      async def send_message(self, target_agent: str, message: dict):
          message['from'] = self.agent_id
          await self.bus.publish(f"agent:{target_agent}", message)

      async def cleanup(self):
          if self.db_session:
              await self.db_session.close()
  ```

- [ ] **Update all agents to use enhanced BaseAgent**
  - [ ] OrchestrationAgent
  - [ ] PlanningAgent
  - [ ] InformationRetrievalAgent
  - [ ] VerificationAgent
  - [ ] ExecutionAgent

### 3.3 Orchestration Agent Integration

- [ ] **Implement workflow coordination**
  ```python
  # agents/orchestration_agent.py
  from agents.base_agent import BaseAgent
  from data_models.schemas import AgentMessage, MessageType, Priority

  class OrchestrationAgent(BaseAgent):
      async def handle_user_goal(self, goal: str, user_id: str):
          self.logger.info("processing_user_goal", goal=goal, user_id=user_id)

          # Create plan request
          plan_request = {
              'message_type': MessageType.REQUEST,
              'goal': goal,
              'user_id': user_id,
              'correlation_id': generate_correlation_id()
          }

          # Send to Planning Agent
          plan_response = await self.bus.request_reply(
              "agent:planning",
              plan_request,
              timeout=60
          )

          # Send plan for verification
          verification_request = {
              'message_type': MessageType.REQUEST,
              'plan': plan_response['plan'],
              'correlation_id': plan_request['correlation_id']
          }

          verification_response = await self.bus.request_reply(
              "agent:verification",
              verification_request,
              timeout=30
          )

          if verification_response['status'] == 'approved':
              # Send to Execution Agent
              execution_request = {
                  'message_type': MessageType.REQUEST,
                  'plan': plan_response['plan'],
                  'correlation_id': plan_request['correlation_id']
              }

              await self.send_message("execution", execution_request)

          return {
              'plan': plan_response['plan'],
              'verification': verification_response
          }
  ```

- [ ] **Test orchestration flow**

### 3.4 Information Retrieval Agent (IRA) Integration

- [ ] **Implement market data caching**
  ```python
  # agents/retriever.py (Information Retrieval Agent)
  from agents.base_agent import BaseAgent
  from agents.external_apis import AlphaVantageAPI, YahooFinanceAPI
  from database.cache import cache

  class InformationRetrievalAgent(BaseAgent):
      def __init__(self, agent_id: str):
          super().__init__(agent_id)
          self.alpha_vantage = AlphaVantageAPI()
          self.yahoo_finance = YahooFinanceAPI()

      async def get_market_data(self, symbol: str):
          # Check cache first
          cache_key = f"market_data:{symbol}"
          cached = await cache.get(cache_key)

          if cached:
              self.logger.info("cache_hit", symbol=symbol)
              return json.loads(cached)

          # Fetch from external API
          try:
              data = await self.alpha_vantage.get_quote(symbol)

              # Cache for 5 minutes
              await cache.set(cache_key, json.dumps(data), expire=300)

              return data
          except Exception as e:
              self.logger.error("market_data_fetch_failed", symbol=symbol, error=str(e))
              # Fallback to Yahoo Finance
              return await self.yahoo_finance.get_quote(symbol)

      async def handle_message(self, message: dict):
          if message.get('request_type') == 'market_data':
              symbol = message.get('symbol')
              data = await self.get_market_data(symbol)

              # Reply with data
              reply_channel = f"agent:{message['from']}:reply:{message['correlation_id']}"
              await self.bus.publish(reply_channel, {
                  'status': 'success',
                  'data': data
              })
  ```

### 3.5 Agent Lifecycle Management

- [ ] **Create agent manager**
  ```python
  # agents/agent_manager.py
  from typing import Dict
  from agents.base_agent import BaseAgent
  from agents.orchestration_agent import OrchestrationAgent
  from agents.communication_bus import CommunicationBus

  class AgentManager:
      def __init__(self):
          self.agents: Dict[str, BaseAgent] = {}
          self.bus = CommunicationBus()

      async def initialize(self):
          await self.bus.initialize()

          # Create all agents
          self.agents['orchestration'] = OrchestrationAgent("orchestration_001")
          self.agents['planning'] = PlanningAgent("planning_001")
          self.agents['ira'] = InformationRetrievalAgent("ira_001")
          self.agents['verification'] = VerificationAgent("verification_001")
          self.agents['execution'] = ExecutionAgent("execution_001")

          # Initialize all agents
          for agent in self.agents.values():
              await agent.initialize()
              await agent.connect_bus(self.bus)

      async def shutdown(self):
          for agent in self.agents.values():
              await agent.cleanup()

          await self.bus.shutdown()

      def get_agent(self, agent_id: str) -> BaseAgent:
          return self.agents.get(agent_id)

  # Global instance
  agent_manager = AgentManager()
  ```

### 3.6 Health Monitoring

- [ ] **Implement agent health checks**
  ```python
  # agents/base_agent.py (add to BaseAgent)
  async def health_check(self) -> dict:
      return {
          'agent_id': self.agent_id,
          'status': 'healthy',
          'uptime': time.time() - self.start_time,
          'messages_processed': self.message_count,
          'last_activity': self.last_activity
      }
  ```

- [ ] **Create health monitoring endpoint**

### Phase 3 Completion Criteria

- [ ] Communication bus working with Redis
- [ ] All 5 agents initialized successfully
- [ ] Agent-to-agent messaging working
- [ ] Request-reply pattern implemented
- [ ] IRA can fetch and cache market data
- [ ] Orchestration agent coordinates workflow
- [ ] Health checks implemented
- [ ] All agent tests passing (26/26)

---

## Phase 4: API Layer Integration

**Priority:** 🟡 HIGH
**Timeline:** Week 3, Days 1-3
**Owner:** Backend API Developer

### 4.1 FastAPI Application Setup

- [ ] **Create main FastAPI app**
  ```python
  # main.py
  from fastapi import FastAPI
  from fastapi.middleware.cors import CORSMiddleware
  from api import endpoints, ml_endpoints, risk_endpoints, conversational_endpoints
  from agents.agent_manager import agent_manager
  from lib.logging_config import setup_logging
  from lib.config import settings

  app = FastAPI(
      title="FinPilot VP-MAS API",
      description="Advanced Multi-Agent Financial Planning System",
      version="0.1.0"
  )

  # CORS
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["http://localhost:5173"],  # Vite dev server
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  @app.on_event("startup")
  async def startup_event():
      setup_logging(settings.log_level)
      await agent_manager.initialize()

  @app.on_event("shutdown")
  async def shutdown_event():
      await agent_manager.shutdown()

  # Include routers
  app.include_router(endpoints.router, prefix="/api/v1", tags=["core"])
  app.include_router(ml_endpoints.router, prefix="/api/v1/ml", tags=["ml"])
  app.include_router(risk_endpoints.router, prefix="/api/v1/risk", tags=["risk"])
  app.include_router(conversational_endpoints.router, prefix="/api/v1/chat", tags=["chat"])

  @app.get("/health")
  async def health_check():
      agent_health = {}
      for agent_id, agent in agent_manager.agents.items():
          agent_health[agent_id] = await agent.health_check()

      return {
          'status': 'healthy',
          'agents': agent_health
      }
  ```

- [ ] **Test API startup**
  ```bash
  uvicorn main:app --reload --port 8000
  curl http://localhost:8000/health
  ```

### 4.2 Core Endpoints Implementation

- [ ] **Planning endpoints**
  ```python
  # api/endpoints.py
  from fastapi import APIRouter, Depends, HTTPException
  from data_models.schemas import EnhancedPlanRequest, FinancialPlan
  from agents.agent_manager import agent_manager

  router = APIRouter()

  @router.post("/plan", response_model=FinancialPlan)
  async def create_financial_plan(request: EnhancedPlanRequest):
      try:
          orchestration = agent_manager.get_agent('orchestration')
          result = await orchestration.handle_user_goal(
              goal=request.user_goal,
              user_id=request.user_id
          )
          return result
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))

  @router.get("/plan/{plan_id}")
  async def get_plan(plan_id: str):
      # Retrieve from database
      pass

  @router.put("/plan/{plan_id}")
  async def update_plan(plan_id: str, updates: dict):
      # Update plan
      pass
  ```

- [ ] **Market data endpoints**
  ```python
  @router.get("/market/{symbol}")
  async def get_market_data(symbol: str):
      ira = agent_manager.get_agent('ira')
      data = await ira.get_market_data(symbol)
      return data
  ```

### 4.3 WebSocket Support for Real-time Updates

- [ ] **Implement WebSocket endpoint**
  ```python
  # api/websocket.py
  from fastapi import WebSocket, WebSocketDisconnect
  from typing import Dict, Set

  class ConnectionManager:
      def __init__(self):
          self.active_connections: Dict[str, Set[WebSocket]] = {}

      async def connect(self, websocket: WebSocket, user_id: str):
          await websocket.accept()
          if user_id not in self.active_connections:
              self.active_connections[user_id] = set()
          self.active_connections[user_id].add(websocket)

      def disconnect(self, websocket: WebSocket, user_id: str):
          self.active_connections[user_id].discard(websocket)

      async def send_personal_message(self, message: dict, user_id: str):
          if user_id in self.active_connections:
              for connection in self.active_connections[user_id]:
                  await connection.send_json(message)

  manager = ConnectionManager()

  @app.websocket("/ws/{user_id}")
  async def websocket_endpoint(websocket: WebSocket, user_id: str):
      await manager.connect(websocket, user_id)
      try:
          while True:
              data = await websocket.receive_text()
              # Process incoming messages
              await manager.send_personal_message(
                  {"message": f"Received: {data}"},
                  user_id
              )
      except WebSocketDisconnect:
          manager.disconnect(websocket, user_id)
  ```

### 4.4 API Authentication & Authorization

- [ ] **Implement JWT authentication**
  ```python
  # api/auth.py
  from fastapi import Depends, HTTPException, status
  from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
  import jwt
  from lib.config import settings

  security = HTTPBearer()

  def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
      try:
          token = credentials.credentials
          payload = jwt.decode(
              token,
              settings.jwt_secret,
              algorithms=[settings.jwt_algorithm]
          )
          return payload
      except jwt.ExpiredSignatureError:
          raise HTTPException(
              status_code=status.HTTP_401_UNAUTHORIZED,
              detail="Token has expired"
          )
      except jwt.InvalidTokenError:
          raise HTTPException(
              status_code=status.HTTP_401_UNAUTHORIZED,
              detail="Invalid token"
          )

  # Use in endpoints
  @router.get("/protected")
  async def protected_route(token_data = Depends(verify_token)):
      return {"user": token_data}
  ```

- [ ] **Create login endpoint**

### 4.5 API Documentation

- [ ] **Configure OpenAPI/Swagger**
  ```python
  # main.py
  app = FastAPI(
      title="FinPilot VP-MAS API",
      description="""
      Advanced Multi-Agent Financial Planning System

      ## Features
      * Multi-agent coordination
      * Real-time market data
      * AI-powered planning
      * Risk assessment
      """,
      version="0.1.0",
      docs_url="/docs",
      redoc_url="/redoc"
  )
  ```

- [ ] **Add response models to all endpoints**
- [ ] **Add examples to request/response schemas**

### 4.6 Rate Limiting

- [ ] **Implement rate limiting middleware**
  ```python
  # api/middleware/rate_limit.py
  from slowapi import Limiter, _rate_limit_exceeded_handler
  from slowapi.util import get_remote_address
  from slowapi.errors import RateLimitExceeded

  limiter = Limiter(key_func=get_remote_address)
  app.state.limiter = limiter
  app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

  @app.get("/api/v1/plan")
  @limiter.limit("10/minute")
  async def create_plan(request: Request):
      # endpoint logic
      pass
  ```

### Phase 4 Completion Criteria

- [ ] FastAPI app starts successfully
- [ ] All endpoint routers integrated
- [ ] Health check endpoint working
- [ ] WebSocket connections working
- [ ] JWT authentication implemented
- [ ] Rate limiting configured
- [ ] OpenAPI documentation accessible
- [ ] All endpoints tested manually

---

## Phase 5: External Services Integration

**Priority:** 🟡 MEDIUM
**Timeline:** Week 3, Days 4-5
**Owner:** Backend Integration Developer

### 5.1 API Key Management

- [ ] **Acquire API keys**
  - [ ] Alpha Vantage: https://www.alphavantage.co/support/#api-key
  - [ ] Yahoo Finance: Use yfinance library (no key needed)
  - [ ] IEX Cloud: https://iexcloud.io/

- [ ] **Store in .env**
  ```bash
  ALPHA_VANTAGE_API_KEY=YOUR_KEY_HERE
  IEX_CLOUD_API_KEY=YOUR_KEY_HERE
  ```

- [ ] **Create API key rotation mechanism**
  ```python
  # agents/api_config.py
  class APIKeyManager:
      def __init__(self):
          self.keys = {
              'alpha_vantage': [settings.alpha_vantage_api_key],
              'iex_cloud': [settings.iex_cloud_api_key]
          }
          self.current_index = {
              'alpha_vantage': 0,
              'iex_cloud': 0
          }

      def get_key(self, service: str) -> str:
          keys = self.keys.get(service, [])
          if not keys:
              raise ValueError(f"No API keys configured for {service}")

          idx = self.current_index[service]
          key = keys[idx]

          # Rotate to next key
          self.current_index[service] = (idx + 1) % len(keys)

          return key
  ```

### 5.2 Alpha Vantage Integration

- [ ] **Implement Alpha Vantage client**
  ```python
  # agents/external_apis.py
  import aiohttp
  from lib.exceptions import ExternalAPIException

  class AlphaVantageAPI:
      BASE_URL = "https://www.alphavantage.co/query"

      def __init__(self):
          self.api_key = settings.alpha_vantage_api_key
          self.session = None

      async def _request(self, params: dict):
          if not self.session:
              self.session = aiohttp.ClientSession()

          params['apikey'] = self.api_key

          async with self.session.get(self.BASE_URL, params=params) as response:
              if response.status != 200:
                  raise ExternalAPIException(f"Alpha Vantage API error: {response.status}")

              data = await response.json()

              if "Error Message" in data:
                  raise ExternalAPIException(data["Error Message"])

              return data

      async def get_quote(self, symbol: str):
          params = {
              'function': 'GLOBAL_QUOTE',
              'symbol': symbol
          }
          return await self._request(params)

      async def get_time_series_daily(self, symbol: str):
          params = {
              'function': 'TIME_SERIES_DAILY',
              'symbol': symbol,
              'outputsize': 'compact'
          }
          return await self._request(params)
  ```

- [ ] **Add retry logic with exponential backoff**
  ```python
  from tenacity import retry, stop_after_attempt, wait_exponential

  @retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(multiplier=1, min=4, max=10)
  )
  async def get_quote_with_retry(self, symbol: str):
      return await self.get_quote(symbol)
  ```

### 5.3 Yahoo Finance Integration

- [ ] **Implement Yahoo Finance client**
  ```python
  import yfinance as yf

  class YahooFinanceAPI:
      async def get_quote(self, symbol: str):
          ticker = yf.Ticker(symbol)
          info = ticker.info

          return {
              'symbol': symbol,
              'price': info.get('regularMarketPrice'),
              'change': info.get('regularMarketChange'),
              'volume': info.get('regularMarketVolume'),
              'market_cap': info.get('marketCap')
          }

      async def get_historical_data(self, symbol: str, period: str = "1mo"):
          ticker = yf.Ticker(symbol)
          hist = ticker.history(period=period)
          return hist.to_dict('records')
  ```

### 5.4 API Fallback Strategy

- [ ] **Implement fallback chain**
  ```python
  # agents/retriever.py
  async def get_market_data_with_fallback(self, symbol: str):
      # Try Alpha Vantage first
      try:
          return await self.alpha_vantage.get_quote(symbol)
      except ExternalAPIException as e:
          self.logger.warning("alpha_vantage_failed", error=str(e))

      # Fallback to Yahoo Finance
      try:
          return await self.yahoo_finance.get_quote(symbol)
      except Exception as e:
          self.logger.error("all_apis_failed", error=str(e))
          raise ExternalAPIException("Unable to fetch market data from any source")
  ```

### 5.5 Rate Limiting for External APIs

- [ ] **Implement API rate limiter**
  ```python
  # lib/rate_limiter.py
  import asyncio
  from datetime import datetime, timedelta

  class RateLimiter:
      def __init__(self, calls: int, period: int):
          self.calls = calls
          self.period = period
          self.timestamps = []

      async def acquire(self):
          now = datetime.now()

          # Remove old timestamps
          cutoff = now - timedelta(seconds=self.period)
          self.timestamps = [ts for ts in self.timestamps if ts > cutoff]

          if len(self.timestamps) >= self.calls:
              # Wait until oldest timestamp expires
              sleep_time = (self.timestamps[0] - cutoff).total_seconds()
              await asyncio.sleep(sleep_time)

          self.timestamps.append(now)

  # Usage in API client
  class AlphaVantageAPI:
      def __init__(self):
          self.rate_limiter = RateLimiter(calls=5, period=60)  # 5 calls per minute

      async def _request(self, params: dict):
          await self.rate_limiter.acquire()
          # ... rest of request logic
  ```

### 5.6 External API Monitoring

- [ ] **Track API usage and errors**
  ```python
  # lib/api_metrics.py
  from prometheus_client import Counter, Histogram

  api_requests_total = Counter(
      'api_requests_total',
      'Total API requests',
      ['service', 'endpoint', 'status']
  )

  api_request_duration = Histogram(
      'api_request_duration_seconds',
      'API request duration',
      ['service', 'endpoint']
  )
  ```

### Phase 5 Completion Criteria

- [ ] All API keys acquired and stored
- [ ] Alpha Vantage integration working
- [ ] Yahoo Finance integration working
- [ ] Fallback strategy implemented
- [ ] Rate limiting configured
- [ ] Retry logic with exponential backoff
- [ ] API monitoring metrics in place
- [ ] External API tests passing

---

## Phase 6: Testing & Validation

**Priority:** 🔴 CRITICAL
**Timeline:** Week 4, Days 1-3
**Owner:** Backend QA Developer

### 6.1 Unit Test Coverage

- [ ] **Run coverage analysis**
  ```bash
  pytest --cov=agents --cov=api --cov=data_models --cov-report=html
  ```

- [ ] **Achieve minimum coverage targets**
  - [ ] Data models: >90% coverage
  - [ ] Agents: >80% coverage
  - [ ] API endpoints: >85% coverage
  - [ ] Database repositories: >90% coverage

- [ ] **Write missing tests for uncovered code**

### 6.2 Integration Test Suite

- [ ] **Fix all integration tests**
  ```bash
  pytest tests/test_integration.py -v
  ```

- [ ] **Add new integration scenarios**
  - [ ] Multi-agent workflow coordination
  - [ ] Database + Agent interaction
  - [ ] External API integration with mocking
  - [ ] WebSocket real-time updates
  - [ ] Error handling and recovery

### 6.3 Performance Testing

- [ ] **Run performance benchmarks**
  ```bash
  pytest tests/test_performance.py --benchmark-only
  ```

- [ ] **Establish baselines**
  - [ ] Agent message processing: < 100ms
  - [ ] API endpoint response: < 500ms
  - [ ] Database query: < 50ms
  - [ ] External API (cached): < 10ms

- [ ] **Load testing with Locust**
  ```python
  # locustfile.py
  from locust import HttpUser, task, between

  class FinPilotUser(HttpUser):
      wait_time = between(1, 3)

      @task
      def create_plan(self):
          self.client.post("/api/v1/plan", json={
              "user_id": "test_user",
              "user_goal": "Save $10,000",
              # ... other fields
          })

      @task(2)
      def get_market_data(self):
          self.client.get("/api/v1/market/AAPL")
  ```

  ```bash
  locust -f locustfile.py --host=http://localhost:8000
  ```

### 6.4 End-to-End Testing

- [ ] **Create E2E test scenarios**
  ```python
  # tests/test_e2e.py
  import pytest
  from httpx import AsyncClient

  @pytest.mark.asyncio
  async def test_complete_planning_workflow():
      async with AsyncClient(base_url="http://localhost:8000") as client:
          # 1. Create a plan
          response = await client.post("/api/v1/plan", json={
              "user_id": "test_user",
              "user_goal": "Retire in 20 years with $1M",
              "current_state": {...},
              # ... other fields
          })
          assert response.status_code == 200
          plan_id = response.json()['plan_id']

          # 2. Get plan status
          response = await client.get(f"/api/v1/plan/{plan_id}")
          assert response.status_code == 200

          # 3. Verify market data was fetched
          # 4. Verify plan was saved to database
          # 5. Verify all agents communicated correctly
  ```

### 6.5 Security Testing

- [ ] **Run security scans**
  ```bash
  # Bandit for Python security
  bandit -r . -ll -f json -o bandit-report.json

  # Safety for dependency vulnerabilities
  safety check --json --output safety-report.json

  # OWASP dependency check
  pip install pip-audit
  pip-audit
  ```

- [ ] **Fix all HIGH and CRITICAL vulnerabilities**

- [ ] **Test authentication/authorization**
  - [ ] Invalid JWT tokens rejected
  - [ ] Expired tokens rejected
  - [ ] Rate limiting prevents abuse
  - [ ] SQL injection prevention
  - [ ] XSS prevention in API responses

### 6.6 Test Automation in CI/CD

- [ ] **Verify CI/CD runs all tests**
  ```yaml
  # .github/workflows/ci-cd.yml should run:
  # 1. Data model tests ✓
  # 2. Agent tests ✓
  # 3. Integration tests ✓
  # 4. Performance tests ✓
  # 5. Security scans ✓
  ```

- [ ] **Add test result reporting**
  ```yaml
  - name: Publish Test Results
    uses: EnricoMi/publish-unit-test-result-action@v2
    if: always()
    with:
      files: test-results/**/*.xml
  ```

### Phase 6 Completion Criteria

- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Performance benchmarks established
- [ ] E2E tests covering critical paths
- [ ] No HIGH/CRITICAL security vulnerabilities
- [ ] Test coverage >80% overall
- [ ] CI/CD pipeline green
- [ ] Test results published in CI/CD

---

## Phase 7: Production Readiness

**Priority:** 🟢 MEDIUM
**Timeline:** Week 4, Days 4-5
**Owner:** Backend + DevOps

### 7.1 Docker Containerization

- [ ] **Create backend Dockerfile**
  ```dockerfile
  # docker/Dockerfile.backend
  FROM python:3.11-slim

  WORKDIR /app

  # Install system dependencies
  RUN apt-get update && apt-get install -y \
      gcc \
      postgresql-client \
      && rm -rf /var/lib/apt/lists/*

  # Install Python dependencies
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  # Copy application
  COPY . .

  # Run migrations on startup
  CMD alembic upgrade head && \
      uvicorn main:app --host 0.0.0.0 --port 8000
  ```

- [ ] **Create docker-compose for local development**
  ```yaml
  # docker-compose.yml
  version: '3.8'

  services:
    postgres:
      image: postgres:15-alpine
      environment:
        POSTGRES_DB: finpilot_dev
        POSTGRES_PASSWORD: dev_password
      ports:
        - "5432:5432"
      volumes:
        - postgres_data:/var/lib/postgresql/data

    redis:
      image: redis:7-alpine
      ports:
        - "6379:6379"

    backend:
      build:
        context: .
        dockerfile: docker/Dockerfile.backend
      ports:
        - "8000:8000"
      environment:
        DATABASE_URL: postgresql://postgres:dev_password@postgres:5432/finpilot_dev
        REDIS_URL: redis://redis:6379/0
      depends_on:
        - postgres
        - redis
      volumes:
        - .:/app
      command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  volumes:
    postgres_data:
  ```

- [ ] **Test Docker setup**
  ```bash
  docker-compose up -d
  docker-compose logs -f backend
  curl http://localhost:8000/health
  ```

### 7.2 Monitoring & Observability

- [ ] **Add Prometheus metrics**
  ```python
  # lib/metrics.py
  from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

  # Metrics
  http_requests_total = Counter(
      'http_requests_total',
      'Total HTTP requests',
      ['method', 'endpoint', 'status']
  )

  http_request_duration = Histogram(
      'http_request_duration_seconds',
      'HTTP request duration',
      ['method', 'endpoint']
  )

  agent_messages_processed = Counter(
      'agent_messages_processed_total',
      'Total messages processed by agents',
      ['agent_id', 'message_type']
  )

  active_agents = Gauge(
      'active_agents',
      'Number of active agents',
      ['agent_type']
  )

  # Add to FastAPI
  from fastapi import FastAPI
  metrics_app = make_asgi_app()
  app.mount("/metrics", metrics_app)
  ```

- [ ] **Add structured logging**
  ```python
  # All logs should be JSON formatted for parsing
  logger.info(
      "request_processed",
      method=request.method,
      path=request.url.path,
      status_code=response.status_code,
      duration=duration
  )
  ```

### 7.3 Configuration for Multiple Environments

- [ ] **Create environment-specific configs**
  ```bash
  # .env.development
  ENVIRONMENT=development
  LOG_LEVEL=DEBUG
  DATABASE_URL=postgresql://localhost:5432/finpilot_dev

  # .env.staging
  ENVIRONMENT=staging
  LOG_LEVEL=INFO
  DATABASE_URL=postgresql://staging-db:5432/finpilot_staging

  # .env.production
  ENVIRONMENT=production
  LOG_LEVEL=WARNING
  DATABASE_URL=postgresql://prod-db:5432/finpilot_prod
  ```

- [ ] **Load config based on environment**
  ```python
  import os
  from dotenv import load_dotenv

  env = os.getenv('ENVIRONMENT', 'development')
  load_dotenv(f'.env.{env}')
  ```

### 7.4 Database Backup & Recovery

- [ ] **Set up automated backups**
  ```bash
  # backup.sh
  #!/bin/bash
  DATE=$(date +%Y%m%d_%H%M%S)
  BACKUP_FILE="backup_${DATE}.sql"

  pg_dump $DATABASE_URL > /backups/$BACKUP_FILE

  # Upload to S3 or backup service
  aws s3 cp /backups/$BACKUP_FILE s3://finpilot-backups/
  ```

- [ ] **Test database restore**
  ```bash
  psql $DATABASE_URL < backup_file.sql
  ```

### 7.5 Deployment Documentation

- [ ] **Create deployment guide**
  ```markdown
  # DEPLOYMENT.md

  ## Prerequisites
  - Docker & Docker Compose
  - PostgreSQL 15+
  - Redis 7+
  - API keys for external services

  ## Deployment Steps
  1. Clone repository
  2. Copy .env.example to .env.production
  3. Fill in production credentials
  4. Run migrations: `alembic upgrade head`
  5. Start services: `docker-compose -f docker-compose.prod.yml up -d`
  6. Verify health: `curl https://api.finpilot.com/health`

  ## Monitoring
  - Metrics: https://metrics.finpilot.com
  - Logs: Check CloudWatch/Datadog
  - Alerts: PagerDuty integration
  ```

### 7.6 CI/CD Deployment Pipeline

- [ ] **Add deployment job to CI/CD**
  ```yaml
  # .github/workflows/ci-cd.yml
  deploy:
    runs-on: ubuntu-latest
    needs: [test-python, test-frontend, code-quality]
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to staging
      run: |
        echo "Deploying to staging..."
        # Deploy to staging server
        ssh deploy@staging.finpilot.com "cd /app && git pull && docker-compose up -d --build"

    - name: Run smoke tests
      run: |
        curl https://staging.finpilot.com/health
        # Run critical path tests

    - name: Deploy to production
      if: success()
      run: |
        echo "Deploying to production..."
        # Deploy to production
  ```

### Phase 7 Completion Criteria

- [ ] Docker images built successfully
- [ ] docker-compose working for local dev
- [ ] Prometheus metrics exposed
- [ ] Structured logging implemented
- [ ] Multi-environment configuration
- [ ] Database backup strategy in place
- [ ] Deployment documentation complete
- [ ] CI/CD deployment pipeline configured

---

## 📊 Integration Progress Tracker

### Overall Progress

```
Phase 0: CI/CD Fixes          [████████░░] 80%  (Data models fixed, agents/integration pending)
Phase 1: Backend Foundation   [░░░░░░░░░░]  0%
Phase 2: Database Setup       [░░░░░░░░░░]  0%
Phase 3: Agent Integration    [░░░░░░░░░░]  0%
Phase 4: API Layer           [░░░░░░░░░░]  0%
Phase 5: External Services   [░░░░░░░░░░]  0%
Phase 6: Testing             [██░░░░░░░░] 20%  (Data model tests passing)
Phase 7: Production          [░░░░░░░░░░]  0%

TOTAL PROGRESS: [██░░░░░░░░] 15%
```

### Critical Path Items (Must Complete in Order)

1. ✅ Fix data model tests (DONE)
2. ⏳ Fix agent tests (IN PROGRESS - async fixtures)
3. ⏳ Fix integration tests (BLOCKED by #2)
4. ⏳ Install all dependencies
5. ⏳ Set up database (Supabase vs PostgreSQL decision needed)
6. ⏳ Implement agent communication bus
7. ⏳ Connect external APIs
8. ⏳ Create FastAPI endpoints
9. ⏳ Achieve 80% test coverage
10. ⏳ Docker containerization

---

## 🚨 Known Blockers & Risks

### Current Blockers

1. **Pytest Async Compatibility** 🔴
   - Impact: 32 tests failing
   - Blocker for: CI/CD pipeline, integration testing
   - Owner: Backend Developer
   - ETA: Week 1, Day 2

2. **Database Architecture Decision** 🟡
   - Impact: Can't proceed with data persistence
   - Blocker for: Agent state management, API persistence
   - Owner: Tech Lead + Backend
   - ETA: Week 2, Day 1

3. **Missing API Keys** 🟡
   - Impact: Can't test external integrations
   - Blocker for: IRA testing, market data features
   - Owner: Project Manager (to acquire)
   - ETA: Week 2, Day 4

### Risks

1. **Pydantic V2 Migration** (MITIGATED ✅)
   - Risk: Breaking changes in validators
   - Mitigation: Already fixed in Phase 0

2. **External API Rate Limits** 🟡
   - Risk: Free tier APIs may have insufficient quotas
   - Mitigation: Implement aggressive caching, fallback APIs

3. **Performance at Scale** 🟡
   - Risk: Unknown performance characteristics under load
   - Mitigation: Performance testing in Phase 6

---

## 📞 Support & Escalation

### Contact Points

- **Backend Lead:** [Your Name]
- **DevOps:** [DevOps Contact]
- **Frontend:** [Frontend Contact]
- **Project Manager:** [PM Contact]

### When to Escalate

- Any blocker lasting >2 days
- Security vulnerabilities (HIGH/CRITICAL)
- Test failures preventing merge to main
- Production incidents

---

## 🎯 Success Metrics

### Definition of Done

- [ ] All CI/CD pipeline jobs passing ✅
- [ ] Test coverage >80%
- [ ] All HIGH/CRITICAL security issues resolved
- [ ] API documentation complete (OpenAPI)
- [ ] Health checks implemented
- [ ] Monitoring & logging in place
- [ ] Docker containers working
- [ ] Deployment guide complete
- [ ] Performance baselines established
- [ ] No known production blockers

### Go-Live Checklist

- [ ] All integration phases complete
- [ ] Load testing passed (1000+ concurrent users)
- [ ] Security audit passed
- [ ] Backup/recovery tested
- [ ] Monitoring dashboards configured
- [ ] On-call rotation established
- [ ] Incident response plan documented
- [ ] Rollback procedure tested

---

**Last Updated:** 2025-11-18
**Document Owner:** Backend Development Team
**Next Review:** After Phase 0 completion
