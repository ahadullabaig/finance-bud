# FinPilot Multi-Agent System Architecture

## Visual System Overview

**Version:** 1.0
**Last Updated:** 2025-11-18

---

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   Web App    │  │  Mobile App  │  │  API Client  │  │  Admin Dashboard  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────────┬─────────┘  │
│         │                  │                  │                    │            │
│         └──────────────────┴──────────────────┴────────────────────┘            │
│                                    │                                            │
└────────────────────────────────────┼────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                             API GATEWAY / ROUTER                                │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │  FastAPI Server (Port 8000)                                            │   │
│  │  • REST API endpoints                                                  │   │
│  │  • WebSocket support (future: real-time updates)                       │   │
│  │  • Authentication & Authorization                                      │   │
│  │  • Rate limiting & throttling                                          │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION & ROUTING LAYER                           │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                    ORCHESTRATION AGENT (OA)                            │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────┐  │   │
│  │  │SessionManager│  │  GoalParser  │  │  TaskDelegator              │  │   │
│  │  └──────────────┘  └──────────────┘  └─────────────────────────────┘  │   │
│  │  ┌──────────────┐  ┌──────────────────────────────────────────────┐  │   │
│  │  │TriggerMonitor│  │  ConcurrentTriggerHandler                    │  │   │
│  │  └──────────────┘  └──────────────────────────────────────────────┘  │   │
│  │                                                                        │   │
│  │  Responsibilities:                                                    │   │
│  │  • Parse user goals                                                   │   │
│  │  • Manage sessions (persistent)                                       │   │
│  │  • Delegate to specialized agents                                     │   │
│  │  • Coordinate workflows                                               │   │
│  │  • Handle CMVL triggers                                               │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└─────┬───────┬────────┬────────┬────────┬────────┬──────────────────────────────┘
      │       │        │        │        │        │
      ▼       ▼        ▼        ▼        ▼        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SPECIALIZED AGENT LAYER                                 │
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐ │
│  │ CONVERSATIONAL  │  │  INFORMATION    │  │      PLANNING AGENT (PA)        │ │
│  │   AGENT (CA)    │  │  RETRIEVAL      │  │      "ULTRATHINK"               │ │
│  │                 │  │  AGENT (IRA)    │  │                                 │ │
│  │ • Ollama LLM    │  │                 │  │  ┌──────────────────────────┐   │ │
│  │ • NLP parsing   │  │ • Market data   │  │  │  Guided Search Module    │   │ │
│  │ • Goal extract  │  │ • Alpha Vantage │  │  │  (ToS Algorithm)         │   │ │
│  │ • Fallback      │  │ • Barchart API  │  │  └──────────────────────────┘   │ │
│  │   rule-based    │  │ • Interest rate │  │  ┌──────────────────────────┐   │ │
│  │                 │  │ • Inflation     │  │  │  Multi-Path Generation   │   │ │
│  │                 │  │ • Volatility    │  │  │  • Conservative          │   │ │
│  │                 │  │   monitoring    │  │  │  • Balanced              │   │ │
│  │                 │  │ • Triggers      │  │  │  • Aggressive            │   │ │
│  │                 │  │                 │  │  │  • Tax-Optimized         │   │ │
│  │                 │  │                 │  │  │  • Growth-Focused        │   │ │
│  │                 │  │                 │  │  │  • Income-Focused        │   │ │
│  │                 │  │                 │  │  │  • Risk-Parity           │   │ │
│  └─────────────────┘  └─────────────────┘  │  └──────────────────────────┘   │ │
│                                            │  ┌──────────────────────────┐   │ │
│  ┌─────────────────┐  ┌─────────────────┐  │  │  Beam Search Optimizer   │   │ │
│  │  VERIFICATION   │  │   EXECUTION     │  │  │  • Constraint filtering  │   │ │
│  │   AGENT (VA)    │  │   AGENT (EA)    │  │  │  • Rejection sampling    │   │ │
│  │                 │  │                 │  │  │  • Path pruning          │   │ │
│  │ • Constraint    │  │ • Transaction   │  │  └──────────────────────────┘   │ │
│  │   satisfaction  │  │   processing    │  │  ┌──────────────────────────┐   │ │
│  │ • Regulatory    │  │ • Ledger mgt.   │  │  │  Reasoning Trace Gen.    │   │ │
│  │   compliance    │  │ • Audit trail   │  │  │  • Explored paths        │   │ │
│  │ • Tax rules     │  │ • Rollback      │  │  │  • Pruned paths          │   │ │
│  │ • Risk assess.  │  │ • Tax report    │  │  │  • Decision points       │   │ │
│  │ • CMVL monitor  │  │                 │  │  │  • Alternatives rejected │   │ │
│  │                 │  │                 │  │  └──────────────────────────┘   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────────┘ │
│                                                                                  │
└───────────┬──────────────────────────────────────────────────────┬──────────────┘
            │                                                      │
            ▼                                                      ▼
┌────────────────────────────────────────┐  ┌────────────────────────────────────┐
│    AGENT COMMUNICATION FRAMEWORK       │  │  ADVANCED ML/AI COMPONENTS         │
│  ┌──────────────────────────────────┐  │  │  ┌──────────────────────────────┐  │
│  │       Message Router             │  │  │  │  NVIDIA NIM Engine          │  │
│  │  • Priority queuing              │  │  │  │  (Alternative LLM)          │  │
│  │  • Circuit breakers              │  │  │  └──────────────────────────────┘  │
│  │  • Correlation tracking          │  │  │  ┌──────────────────────────────┐  │
│  └──────────────────────────────────┘  │  │  │  GNN Risk Detector          │  │
│  ┌──────────────────────────────────┐  │  │  │  (Graph Neural Network)     │  │
│  │  Persistent Message Queue        │  │  │  └──────────────────────────────┘  │
│  │  • File-based persistence        │  │  │  ┌──────────────────────────────┐  │
│  │  • Dead letter queue             │  │  │  │  RL Portfolio Optimizer     │  │
│  │  • Backpressure handling         │  │  │  │  (Reinforcement Learning)   │  │
│  │  • Automatic retry               │  │  │  └──────────────────────────────┘  │
│  └──────────────────────────────────┘  │  │  ┌──────────────────────────────┐  │
│  ┌──────────────────────────────────┐  │  │  │  AI Financial Coach         │  │
│  │  Session Persistence Manager     │  │  │  └──────────────────────────────┘  │
│  │  • JSON/Pickle storage           │  │  │  ┌──────────────────────────────┐  │
│  │  • Automatic expiration          │  │  │  │  Anomaly Detector           │  │
│  │  • Crash recovery                │  │  │  └──────────────────────────────┘  │
│  └──────────────────────────────────┘  │  └────────────────────────────────────┘
└────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          DATA & STORAGE LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Message     │  │   Session    │  │  Reasoning   │  │  Dead Letter     │   │
│  │  Queues      │  │   Storage    │  │  Traces      │  │  Queue           │   │
│  │  (./data/    │  │  (./data/    │  │  (in-memory) │  │  (./data/        │   │
│  │   queues/)   │  │   sessions/) │  │              │  │   queues/*dlq)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                      Cache Layer (Redis) - Optional                      │  │
│  │  • Market data caching                                                   │  │
│  │  • API response caching                                                  │  │
│  │  • Session caching (fast access)                                         │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL SERVICES & APIs                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ Alpha Vantage  │  │  Barchart API  │  │  Massive API   │  │  Ollama LLM  │ │
│  │ (Economic data)│  │  (Market data) │  │ (Alt. data)    │  │  (Local)     │ │
│  └────────────────┘  └────────────────┘  └────────────────┘  └──────────────┘ │
│  ┌────────────────┐  ┌────────────────────────────────────────────────────────┐ │
│  │  NVIDIA NIM    │  │         External Data Sources (Future)                │ │
│  │  (Cloud LLM)   │  │  • News APIs, • Regulatory databases,                 │ │
│  │  (Optional)    │  │  • Social sentiment, • Economic indicators            │ │
│  └────────────────┘  └────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. UltraThink ToS Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   ULTRATHINK THOUGHT OF SEARCH (ToS)                     │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: User Goal + Constraints
  │
  ▼
┌─────────────────────────────────────┐
│  1. GOAL DECOMPOSITION              │
│  ┌───────────────────────────────┐  │
│  │ • Parse financial goal        │  │
│  │ • Extract time horizon        │  │
│  │ • Identify constraints        │  │
│  │ • Set milestones              │  │
│  └───────────────────────────────┘  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. INITIALIZE BEAM WITH STRATEGY TEMPLATES                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ Conservative │ │   Balanced   │ │  Aggressive  │ │Tax-Optimized │  │
│  │  Strategy    │ │   Strategy   │ │   Strategy   │ │  Strategy    │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                   │
│  │Growth-Focused│ │Income-Focused│ │ Risk-Parity  │                   │
│  │  Strategy    │ │   Strategy   │ │   Strategy   │                   │
│  └──────────────┘ └──────────────┘ └──────────────┘                   │
│                                                                         │
│  Beam Width: 7 strategies                                              │
└─────────────┬───────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. BEAM SEARCH ITERATIONS (Max 10 iterations)                          │
│                                                                          │
│  For each iteration:                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  For each strategy in beam:                                        │ │
│  │    │                                                               │ │
│  │    ├─ Expand: Generate child strategies                           │ │
│  │    │         (variations in allocation, risk, timeline)           │ │
│  │    │                                                               │ │
│  │    ├─ Evaluate: Score each child                                  │ │
│  │    │    ┌──────────────────────────────────────────────┐         │ │
│  │    │    │ Score = w₁·Risk-Adj-Return                  │         │ │
│  │    │    │       + w₂·Constraint-Satisfaction           │         │ │
│  │    │    │       + w₃·Tax-Efficiency                    │         │ │
│  │    │    │       + w₄·Feasibility                       │         │ │
│  │    │    └──────────────────────────────────────────────┘         │ │
│  │    │                                                               │ │
│  │    ├─ Constraint Check: Does it satisfy hard constraints?        │ │
│  │    │    YES ─→ Continue                                          │ │
│  │    │    NO  ─→ PRUNE (add to pruned_paths)                      │ │
│  │    │                                                               │ │
│  │    ├─ Rejection Sampling: Will it violate future constraints?    │ │
│  │    │    NO  ─→ Continue                                          │ │
│  │    │    YES ─→ PRUNE (add to pruned_paths)                      │ │
│  │    │                                                               │ │
│  │    └─ Add to beam if score > threshold                           │ │
│  │                                                                    │ │
│  │  Keep only TOP K (beam_width) strategies                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Termination:                                                           │
│  • Max iterations reached, OR                                           │
│  • Best score > 0.95, OR                                                │
│  • No improvement in last 3 iterations                                  │
└─────────────┬────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. FINAL RANKING & SELECTION                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Sort explored_paths by combined_score (descending)               │ │
│  │                                                                    │ │
│  │  Top Strategies:                                                  │ │
│  │  1. Balanced        (score: 0.92) ← BEST                         │ │
│  │  2. Tax-Optimized   (score: 0.88)                                │ │
│  │  3. Conservative    (score: 0.85)                                │ │
│  │  4. Growth-Focused  (score: 0.81)                                │ │
│  │  5. Income-Focused  (score: 0.78)                                │ │
│  │                                                                    │ │
│  │  Pruned Strategies:                                               │ │
│  │  • Aggressive       (reason: Exceeds risk tolerance)             │ │
│  │  • High-Leverage    (reason: Predicted liquidity violation)      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────┬────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. GENERATE REASONING TRACE                                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  ReasoningTrace {                                                 │ │
│  │    explored_paths: [5 paths with details]                        │ │
│  │    pruned_paths: [2 paths with reasons]                          │ │
│  │    decision_points: [                                            │ │
│  │      {                                                            │ │
│  │        decision_type: "strategy_selection",                      │ │
│  │        options_considered: 7,                                    │ │
│  │        chosen_option: "balanced",                                │ │
│  │        rationale: "Best risk-adjusted return",                   │ │
│  │        alternatives_rejected: [                                  │ │
│  │          {option: "aggressive", reason: "Too risky"},            │ │
│  │          {option: "conservative", reason: "Lower returns"}       │ │
│  │        ]                                                          │ │
│  │      }                                                            │ │
│  │    ],                                                             │ │
│  │    confidence_metrics: { ... },                                  │ │
│  │    performance_data: { execution_time: 1.2s }                    │ │
│  │  }                                                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────┬────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Top 5 Strategies + Reasoning Trace                             │
│  → Sent to VerificationAgent for validation                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Request Flow (End-to-End)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER REQUEST FLOW                                 │
└──────────────────────────────────────────────────────────────────────────┘

[1] USER SUBMITS GOAL
    "I want to save $50,000 for a house in 5 years"
    │
    ▼
[2] API GATEWAY (FastAPI)
    POST /api/planning/create
    │
    ├─ Create session_id
    ├─ User authentication
    └─ Forward to OrchestrationAgent
    │
    ▼
[3] ORCHESTRATION AGENT (OA)
    │
    ├─ Create PersistentSession
    │  └─ session_id: "abc123"
    │     user_id: "user_456"
    │     state: { goal: "...", status: "parsing" }
    │
    ├─ (Optional) Send to ConversationalAgent for NLP parsing
    │  │
    │  └─ [3a] CONVERSATIONAL AGENT (CA)
    │         │
    │         ├─ Ollama LLM: Parse natural language
    │         ├─ Extract: {goal_type: "savings", amount: 50000, years: 5}
    │         └─ Return structured goal
    │
    ├─ Create workflow: "financial_planning"
    │
    └─ Delegate to PlanningAgent
       │
       ▼
[4] PLANNING AGENT (PA) - "ULTRATHINK"
    │
    ├─ Run ToS Algorithm:
    │  ├─ Generate 7+ strategies
    │  ├─ Beam search optimization
    │  ├─ Constraint filtering
    │  └─ Prune invalid paths
    │
    ├─ Create ReasoningTrace:
    │  ├─ explored_paths: 5 strategies
    │  ├─ pruned_paths: 2 strategies
    │  └─ decision_points: [...]
    │
    ├─ Select best path: "Balanced Strategy"
    │
    └─ Return to OrchestrationAgent
       │
       ▼
[5] ORCHESTRATION AGENT (OA)
    │
    ├─ Update session:
    │  └─ state.status = "verification"
    │     state.planning_result = {...}
    │
    └─ Delegate to VerificationAgent
       │
       ▼
[6] VERIFICATION AGENT (VA)
    │
    ├─ For each strategy (top 4):
    │  │
    │  ├─ Check constraints:
    │  │  ├─ Risk tolerance: OK
    │  │  ├─ Liquidity: OK
    │  │  └─ Regulatory: OK
    │  │
    │  ├─ Tax validation: OK
    │  │
    │  └─ Risk assessment:
    │     └─ Risk score: 0.65 (moderate)
    │
    ├─ Generate VerificationReport:
    │  ├─ Strategy 1 (Balanced): APPROVED (92% confidence)
    │  ├─ Strategy 2 (Tax-Opt): APPROVED (88% confidence)
    │  ├─ Strategy 3 (Conserv.): APPROVED (85% confidence)
    │  └─ Strategy 4 (Aggress.): CONDITIONAL (requires ack)
    │
    └─ Return to OrchestrationAgent
       │
       ▼
[7] ORCHESTRATION AGENT (OA)
    │
    ├─ Update session:
    │  └─ state.status = "verified"
    │     state.verification_result = {...}
    │
    ├─ Select best approved: "Balanced Strategy"
    │
    └─ Delegate to ExecutionAgent
       │
       ▼
[8] EXECUTION AGENT (EA)
    │
    ├─ Process plan steps:
    │  ├─ Open savings account
    │  ├─ Set up auto-deposit: $710/month
    │  ├─ Allocate 60% to index funds
    │  ├─ Allocate 40% to bonds
    │  └─ Schedule quarterly rebalancing
    │
    ├─ Update ledger:
    │  └─ Record all transactions
    │
    ├─ Generate audit trail
    │
    └─ Return ExecutionLog
       │
       ▼
[9] ORCHESTRATION AGENT (OA)
    │
    ├─ Update session:
    │  └─ state.status = "completed"
    │     state.execution_result = {...}
    │
    ├─ Generate user-facing summary
    │
    └─ Return to API Gateway
       │
       ▼
[10] API GATEWAY
     │
     └─ Return JSON response:
        {
          "session_id": "abc123",
          "status": "completed",
          "selected_strategy": "balanced",
          "monthly_contribution": 710,
          "expected_return": "5.0%",
          "confidence": 0.92,
          "execution_summary": {...},
          "reasoning_graph_url": "/api/reasoning-graph/abc123"
        }
        │
        ▼
[11] USER RECEIVES RESULT
     └─ View recommended plan
        View alternatives
        Explore reasoning graph
        Execute plan


┌──────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL PROCESS: CMVL MONITORING                      │
└──────────────────────────────────────────────────────────────────────────┘

[CONTINUOUS] INFORMATION RETRIEVAL AGENT (IRA)
    │
    ├─ Every 5 minutes:
    │  ├─ Fetch market data (Barchart, Alpha Vantage)
    │  ├─ Calculate volatility
    │  └─ Check for triggers
    │
    ├─ IF TRIGGER DETECTED (e.g., market drop 15%):
    │  │
    │  └─ Create TriggerEvent
    │     └─ Send to OrchestrationAgent
    │        │
    │        ▼
    │     ORCHESTRATION AGENT
    │        │
    │        └─ Activate CMVL for affected sessions
    │           │
    │           ▼
    │        VERIFICATION AGENT (CMVL Mode)
    │           │
    │           ├─ Re-verify current plan
    │           │
    │           ├─ IF STILL VALID:
    │           │  └─ Update confidence, notify user
    │           │
    │           └─ IF VIOLATED:
    │              └─ Trigger dynamic replanning
    │                 │
    │                 └─ Back to PlanningAgent (ToS)
    │                    └─ Generate new strategies
    │                       └─ Continue flow from step [4]
    │
    └─ Continue monitoring...
```

---

## 4. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                       │
└──────────────────────────────────────────────────────────────────────────┘

USER INPUT
  │
  ├─→ API Request
  │      │
  │      └─→ OrchestrationAgent
  │             │
  │             ├─→ Session Creation
  │             │      │
  │             │      └─→ SessionPersistenceManager
  │             │             │
  │             │             └─→ ./data/sessions/session_abc123.json
  │             │
  │             └─→ Message Queue
  │                    │
  │                    └─→ PersistentMessageQueue
  │                           │
  │                           └─→ ./data/queues/OA_queue.json
  │
  └─→ Agent Messages
         │
         ├─→ Planning Request
         │      │
         │      └─→ PlanningAgent
         │             │
         │             ├─→ ToS Algorithm (in-memory)
         │             │      │
         │             │      ├─→ Explored Paths []
         │             │      └─→ Pruned Paths []
         │             │
         │             └─→ ReasoningTrace (in-memory)
         │                    │
         │                    └─→ ReasonGraphMapper
         │                           │
         │                           └─→ Graph Data (nodes, edges)
         │                                  │
         │                                  └─→ Frontend Visualization
         │
         ├─→ Verification Request
         │      │
         │      └─→ VerificationAgent
         │             │
         │             └─→ VerificationReport
         │
         ├─→ Execution Request
         │      │
         │      └─→ ExecutionAgent
         │             │
         │             ├─→ Ledger Updates
         │             └─→ ExecutionLog
         │
         └─→ Market Data Request
                │
                └─→ InformationRetrievalAgent
                       │
                       ├─→ Alpha Vantage API
                       │      │
                       │      └─→ Economic Data (interest, inflation)
                       │
                       ├─→ Barchart API
                       │      │
                       │      └─→ Market Prices
                       │
                       └─→ Cache (Redis - optional)
```

---

## 5. Message Persistence Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MESSAGE LIFECYCLE WITH PERSISTENCE                     │
└──────────────────────────────────────────────────────────────────────────┘

[1] MESSAGE SENT
    AgentMessage created
    │
    └─→ PersistentMessageQueue.put(message)
        │
        ├─ Check queue capacity
        │  │
        │  ├─ IF FULL:
        │  │  └─→ Apply backpressure strategy
        │  │      ├─ DROP_OLDEST: Remove oldest message
        │  │      ├─ DROP_NEWEST: Reject new message
        │  │      ├─ BLOCK: Wait for space
        │  │      └─ REJECT: Return false
        │  │
        │  └─ IF SPACE:
        │     └─→ Continue
        │
        ├─ Create PersistedMessage
        │  └─ {
        │      message: AgentMessage,
        │      message_id: "msg_123",
        │      status: PENDING,
        │      retry_count: 0,
        │      max_retries: 3
        │    }
        │
        ├─ Add to in-memory queue
        │  └─ asyncio.Queue.put()
        │
        ├─ Add to pending_messages dict
        │  └─ pending_messages["msg_123"] = PersistedMessage
        │
        └─ Persist to disk (background, every 1s)
           └─ ./data/queues/agent_id_queue.json

[2] MESSAGE RETRIEVED
    message = await PersistentMessageQueue.get()
    │
    └─ Status: PROCESSING

[3] MESSAGE PROCESSING
    try:
        result = await agent.handle_message(message)
        └─→ [SUCCESS PATH]
    except Exception as e:
        └─→ [FAILURE PATH]

[4a] SUCCESS PATH
     │
     └─→ PersistentMessageQueue.mark_delivered("msg_123")
         │
         ├─ Update status: DELIVERED
         ├─ Remove from pending_messages
         ├─ Update metrics: messages_delivered++
         └─ Persist to disk

[4b] FAILURE PATH
     │
     └─→ PersistentMessageQueue.mark_failed("msg_123", error_msg)
         │
         ├─ Increment retry_count
         ├─ Update error_message
         │
         ├─ IF retry_count < max_retries:
         │  │
         │  ├─ Status: PENDING (retry)
         │  └─ Re-queue message
         │     └─→ Back to [1]
         │
         └─ IF retry_count >= max_retries:
            │
            ├─ Status: DEAD_LETTER
            ├─ Move to dead_letter_queue[]
            ├─ Remove from pending_messages
            ├─ Update metrics: messages_failed++
            └─ Persist to DLQ file
               └─ ./data/queues/agent_id_dlq.json

[5] SYSTEM CRASH
    │
    └─→ All in-memory state lost
        BUT files persisted:
        ├─ agent_id_queue.json (pending messages)
        └─ agent_id_dlq.json (failed messages)

[6] SYSTEM RESTART
    │
    └─→ PersistentMessageQueue.__init__()
        │
        ├─ Load from queue.json
        │  └─ Restore pending_messages
        │     └─→ Re-add to asyncio.Queue
        │
        └─ Load from dlq.json
           └─ Restore dead_letter_queue
              │
              └─ Optional: replay_dlq()
                 └─→ Retry failed messages
```

---

## 6. Reasoning Graph Visualization

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    REASONING GRAPH STRUCTURE                              │
└──────────────────────────────────────────────────────────────────────────┘

                            [User Goal]
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
         [Explored Path 1]  [Explored Path 2]  [Pruned Path 1]
         "Conservative"     "Balanced" ✓       "Aggressive" ✗
         Score: 0.85        Score: 0.92        Reason: Exceeds
         Status: approved   Status: approved   risk tolerance
                 │               │
                 │               ▼
                 │       [Decision Point]
                 │       "Strategy Selection"
                 │       Chosen: "Balanced"
                 │       Alternatives: 2
                 │               │
                 │               ├─→ [Rejected Alt 1]
                 │               │   "Conservative"
                 │               │   Reason: Lower returns
                 │               │
                 │               └─→ [Rejected Alt 2]
                 │                   "Tax-Optimized"
                 │                   Reason: Complexity
                 │               │
                 └───────────────┼───────────────┘
                                 │
                                 ▼
                        [Final Selection]
                        "Balanced Strategy"
                        Confidence: 92%
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
            [Step 1]        [Step 2]        [Step 3]
            "Open           "Auto-          "Allocate
            Account"        deposit"        Assets"


LEGEND:
━━━  Solid edge (accepted path)
╌╌╌  Dashed edge (rejected path)
✓    Approved status
✗    Rejected/Pruned status
◯    Alternative status
```

---

## 7. Technology Stack

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          TECHNOLOGY STACK                                 │
└──────────────────────────────────────────────────────────────────────────┘

BACKEND:
├─ Python 3.9+
├─ FastAPI (Web framework)
├─ Pydantic (Data validation)
├─ AsyncIO (Concurrency)
├─ Ollama (Local LLM - llama3.2:3b)
└─ aiohttp (Async HTTP client)

AGENTS:
├─ Custom multi-agent framework
├─ Message-based communication
├─ Circuit breaker pattern
├─ Persistent message queues
└─ Session management

ALGORITHMS:
├─ Thought of Search (ToS) - Beam search
├─ Constraint satisfaction
├─ Rejection sampling
├─ Monte Carlo simulation
└─ Risk-adjusted optimization

STORAGE:
├─ File-based (JSON/Pickle)
│  ├─ Message queues
│  ├─ Sessions
│  └─ Dead letter queue
├─ Redis (Optional caching)
└─ In-memory (Reasoning traces)

EXTERNAL APIs:
├─ Alpha Vantage (Economic data)
├─ Barchart (Market data)
├─ Massive API (Alternative data)
└─ NVIDIA NIM (Optional LLM)

FRONTEND:
├─ React / Next.js
├─ TypeScript
├─ D3.js (Graph visualization)
├─ Framer Motion (Animations)
└─ WebSocket (Real-time updates - future)

MONITORING:
├─ Structured logging
├─ Performance metrics
├─ Correlation tracking
└─ Prometheus + Grafana (future)

TESTING:
├─ pytest (Unit tests)
├─ pytest-asyncio (Async tests)
└─ Mock APIs
```

---

## 8. Deployment Architecture (Future)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT (FUTURE)                         │
└──────────────────────────────────────────────────────────────────────────┘

                         [Load Balancer]
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   [API Server 1]        [API Server 2]        [API Server 3]
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
[Agent Cluster 1]     [Agent Cluster 2]     [Agent Cluster 3]
   OA, PA, VA            OA, PA, VA            OA, PA, VA
   IRA, EA, CA           IRA, EA, CA           IRA, EA, CA
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   [PostgreSQL]           [Redis]              [S3 Storage]
   (Sessions,            (Cache,              (Backups,
    Transactions)         Queues)              Archives)
```

---

**Document Status:** ✅ Complete
**Related Documents:**
- `AGENT_INTEGRATION_REPORT.md` - Full analysis
- `ULTRATHINK_TECHNICAL_DEEP_DIVE.md` - ToS algorithm details
- `INTEGRATION_GUIDE.md` - Developer integration guide
