# Integration Guide: Enhanced Features

## Quick Start Guide for New Persistence and Communication Features

**Version:** 1.0
**Date:** 2025-11-18

---

## Table of Contents

1. [Overview](#overview)
2. [Message Persistence Integration](#message-persistence-integration)
3. [Session Persistence Integration](#session-persistence-integration)
4. [Enhanced Reasoning Graph](#enhanced-reasoning-graph)
5. [Real-Time Visualization](#real-time-visualization)
6. [Testing Guide](#testing-guide)
7. [Migration Path](#migration-path)

---

## 1. Overview

This guide covers integration of the newly implemented features:

✅ **Message Persistence** - Survive system crashes, dead letter queue
✅ **Session Persistence** - State survives restarts
✅ **Backpressure Handling** - Prevent queue overflow
✅ **Enhanced Reasoning Graph** - Full ToS visualization
✅ **Real-Time Updates** - Live agent progress (ready for WebSocket)

---

## 2. Message Persistence Integration

### 2.1 Basic Setup

Replace the standard `asyncio.Queue` in your agents with `PersistentMessageQueue`:

**Before:**
```python
class MyAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "MyAgent")
        self.message_queue = asyncio.Queue()  # ❌ Lost on crash
```

**After:**
```python
from agents.message_persistence import PersistentMessageQueue, BackpressureStrategy

class MyAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "MyAgent")
        self.message_queue = PersistentMessageQueue(
            agent_id=agent_id,
            storage_path="./data/message_queues",
            max_queue_size=1000,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
            enable_persistence=True
        )  # ✅ Survives crashes!
```

### 2.2 Sending Messages

```python
async def send_message(self, message: AgentMessage):
    """
    Send message with automatic persistence
    """
    # Put message in queue with backpressure handling
    success = await self.message_queue.put(
        message=message,
        block=False,  # Don't wait if queue full
        timeout=5.0   # Max 5s wait if blocking
    )

    if not success:
        self.logger.warning(f"Message dropped due to backpressure: {message.message_id}")
```

### 2.3 Receiving Messages

```python
async def process_messages(self):
    """
    Process messages with delivery tracking
    """
    while self.status == "running":
        # Get message from queue
        message = await self.message_queue.get(timeout=1.0)

        if message is None:
            continue

        try:
            # Process the message
            result = await self._handle_message(message)

            # Mark as successfully delivered
            self.message_queue.mark_delivered(message.message_id)

        except Exception as e:
            # Mark as failed (will retry or move to DLQ)
            self.message_queue.mark_failed(
                message_id=message.message_id,
                error=str(e)
            )
```

### 2.4 Monitoring Queue Health

```python
def check_queue_health(self):
    """
    Monitor queue metrics
    """
    metrics = self.message_queue.get_metrics()

    print(f"Queue Size: {metrics['queue_size']}/{metrics['max_queue_size']}")
    print(f"Utilization: {metrics['utilization']:.1%}")
    print(f"Pending: {metrics['pending_messages']}")
    print(f"DLQ Size: {metrics['dlq_size']}")
    print(f"Delivered: {metrics['messages_delivered']}")
    print(f"Failed: {metrics['messages_failed']}")
    print(f"Dropped: {metrics['messages_dropped']}")

    # Alert if queue is filling up
    if metrics['utilization'] > 0.8:
        self.logger.warning("Queue utilization high! Consider scaling.")
```

### 2.5 Dead Letter Queue Management

```python
async def process_dead_letter_queue(self):
    """
    Periodically review and retry DLQ messages
    """
    # Get all DLQ messages
    dlq_messages = list(self.message_queue.dead_letter_queue)

    for msg_dict in dlq_messages:
        message_id = msg_dict['message_id']
        error = msg_dict['error_message']
        retry_count = msg_dict['retry_count']

        # Analyze failure
        if "timeout" in error.lower():
            # Retry timeout errors
            await self.message_queue.replay_dlq(
                filter_fn=lambda m: "timeout" in m.get('error_message', '').lower()
            )

        elif retry_count >= 3:
            # Log persistent failures
            self.logger.error(f"Message {message_id} failed {retry_count} times: {error}")
            # Could send alert, move to manual review, etc.
```

### 2.6 Graceful Shutdown

```python
async def shutdown(self):
    """
    Shutdown with message persistence
    """
    self.logger.info("Shutting down agent...")

    # Stop processing new messages
    self.status = "stopping"

    # Wait for in-flight messages
    await asyncio.sleep(0.5)

    # Persist all pending messages
    await self.message_queue.shutdown()

    self.logger.info("Shutdown complete. All messages persisted.")
```

### 2.7 Backpressure Strategies

Choose the right strategy for your use case:

```python
# 1. DROP_OLDEST - Good for real-time data (market prices)
queue = PersistentMessageQueue(
    agent_id="market_agent",
    backpressure_strategy=BackpressureStrategy.DROP_OLDEST
)
# Old market prices become stale anyway

# 2. DROP_NEWEST - Good when order matters
queue = PersistentMessageQueue(
    agent_id="execution_agent",
    backpressure_strategy=BackpressureStrategy.DROP_NEWEST
)
# Don't want to skip earlier transaction requests

# 3. BLOCK - Good for critical messages
queue = PersistentMessageQueue(
    agent_id="verification_agent",
    backpressure_strategy=BackpressureStrategy.BLOCK,
    max_queue_size=100
)
# All verification requests are important

# 4. REJECT - Good when caller should handle it
queue = PersistentMessageQueue(
    agent_id="api_agent",
    backpressure_strategy=BackpressureStrategy.REJECT
)
# Return 429 Too Many Requests to client

# 5. DROP_LOWEST_PRIORITY - Good for mixed workloads
queue = PersistentMessageQueue(
    agent_id="orchestrator",
    backpressure_strategy=BackpressureStrategy.DROP_LOWEST_PRIORITY
)
# Critical messages preserved, low-priority dropped
```

---

## 3. Session Persistence Integration

### 3.1 Initialize Session Manager

```python
from agents.session_persistence import get_session_manager

# Get global session manager (singleton)
session_manager = get_session_manager(
    storage_path="./data/sessions",
    default_ttl_hours=24,
    cleanup_interval_minutes=60,
    storage_format="json"  # or "pickle"
)
```

### 3.2 Create Sessions

```python
async def handle_new_user_request(user_id: str, goal: str):
    """
    Create session for new planning request
    """
    # Generate session ID
    session_id = str(uuid4())

    # Create persistent session
    session = session_manager.create_session(
        session_id=session_id,
        user_id=user_id,
        initial_state={
            "goal": goal,
            "status": "planning",
            "created_at": datetime.utcnow().isoformat()
        },
        metadata={
            "source": "web_app",
            "user_agent": "Mozilla/5.0...",
            "ip_address": "192.168.1.1"
        },
        ttl_hours=24
    )

    print(f"Session created: {session_id}")
    print(f"Expires: {session.expires_at}")

    return session
```

### 3.3 Retrieve and Update Sessions

```python
async def continue_planning_session(session_id: str):
    """
    Continue existing planning session
    """
    # Retrieve session
    session = session_manager.get_session(session_id)

    if session is None:
        raise ValueError("Session not found or expired")

    # Access state
    current_status = session.state.get("status")
    user_goal = session.state.get("goal")

    print(f"Resuming session for goal: {user_goal}")
    print(f"Current status: {current_status}")

    # Update state
    session_manager.update_session(
        session_id=session_id,
        state_updates={
            "status": "verification",
            "plan_generated_at": datetime.utcnow().isoformat(),
            "selected_strategy": "balanced"
        },
        extend_ttl_hours=24  # Extend expiration
    )

    return session
```

### 3.4 Store Complex State

```python
async def store_planning_results(session_id: str, planning_result):
    """
    Store complex planning results in session
    """
    session_manager.update_session(
        session_id=session_id,
        state_updates={
            # Store planning result
            "planning_result": {
                "best_path": planning_result.best_path.dict(),
                "alternatives": [p.dict() for p in planning_result.explored_paths],
                "reasoning_trace_id": planning_result.reasoning_trace.trace_id
            },

            # Store verification status
            "verification_status": "pending",

            # Store metrics
            "performance": {
                "planning_time_ms": planning_result.execution_time_ms,
                "paths_explored": len(planning_result.explored_paths),
                "paths_pruned": len(planning_result.pruned_paths)
            }
        }
    )
```

### 3.5 Handle Session Expiration

```python
def check_session_validity(session_id: str) -> bool:
    """
    Check if session is still valid
    """
    session = session_manager.get_session(session_id)

    if session is None:
        return False

    if session.is_expired():
        print(f"Session {session_id} expired at {session.expires_at}")
        return False

    # Check custom business logic
    if session.state.get("status") == "completed":
        # Completed sessions can be deleted
        session_manager.delete_session(session_id)
        return False

    return True
```

### 3.6 User Session Management

```python
def get_all_user_sessions(user_id: str):
    """
    Get all active sessions for a user
    """
    sessions = session_manager.get_user_sessions(user_id)

    print(f"User {user_id} has {len(sessions)} active sessions:")

    for session in sessions:
        goal = session.state.get("goal", "Unknown")
        status = session.state.get("status", "Unknown")
        age = datetime.utcnow() - session.created_at

        print(f"  - {session.session_id}")
        print(f"    Goal: {goal}")
        print(f"    Status: {status}")
        print(f"    Age: {age}")
        print(f"    Expires: {session.expires_at}")
```

### 3.7 Crash Recovery Example

```python
async def recover_from_crash():
    """
    Recover all sessions after system restart
    """
    # SessionPersistenceManager automatically loads sessions from disk
    session_manager = get_session_manager()

    # Get metrics
    metrics = session_manager.get_metrics()

    print(f"Recovered {metrics['active_sessions']} active sessions")
    print(f"Cleaned up {metrics['expired_sessions']} expired sessions")

    # Resume processing for each session
    for session_id, session_data in session_manager.sessions.items():
        status = session_data.state.get("status")

        if status == "planning":
            # Resume planning
            await resume_planning(session_id)

        elif status == "verification":
            # Resume verification
            await resume_verification(session_id)

        elif status == "execution":
            # Check execution status
            await check_execution_status(session_id)
```

---

## 4. Enhanced Reasoning Graph

### 4.1 Generate Enhanced Graph Data

```python
from utils.reason_graph_mapper import ReasonGraphMapper

async def generate_reasoning_graph(planning_result):
    """
    Generate enhanced reasoning graph with full ToS visualization
    """
    # Prepare response with reasoning trace
    planning_response = {
        "session_id": planning_result.session_id,
        "plan_id": planning_result.plan_id,
        "selected_strategy": planning_result.best_path.strategy,
        "confidence_score": planning_result.confidence_score,

        # Include full reasoning trace (KEY!)
        "reasoning_trace": {
            "explored_paths": [p.dict() for p in planning_result.explored_paths],
            "pruned_paths": [p.dict() for p in planning_result.pruned_paths],
            "decision_points": [dp.dict() for dp in planning_result.decision_points],
            "trace_id": planning_result.reasoning_trace.trace_id
        },

        # Plan steps for implementation
        "plan_steps": [s.dict() for s in planning_result.plan_steps]
    }

    # Map to graph format
    graph_data = ReasonGraphMapper.map_planning_trace(planning_response)

    print(f"Graph generated:")
    print(f"  Nodes: {len(graph_data['nodes'])}")
    print(f"  Edges: {len(graph_data['edges'])}")
    print(f"  Explored paths: {graph_data['metadata']['total_explored_paths']}")
    print(f"  Pruned paths: {graph_data['metadata']['total_pruned_paths']}")
    print(f"  Decision points: {graph_data['metadata']['total_decision_points']}")

    return graph_data
```

### 4.2 API Endpoint for Graph Data

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/api/reasoning-graph/{session_id}")
async def get_reasoning_graph(session_id: str):
    """
    API endpoint to fetch reasoning graph data
    """
    # Get session
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get planning result from session state
    planning_result = session.state.get("planning_result")

    if not planning_result:
        raise HTTPException(status_code=404, detail="Planning not complete")

    # Generate graph
    graph_data = ReasonGraphMapper.map_planning_trace(planning_result)

    return {
        "session_id": session_id,
        "graph": graph_data,
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "total_nodes": len(graph_data['nodes']),
            "total_edges": len(graph_data['edges'])
        }
    }
```

### 4.3 Frontend Integration

```typescript
// Fetch reasoning graph data
async function loadReasoningGraph(sessionId: string) {
  const response = await fetch(`/api/reasoning-graph/${sessionId}`);
  const data = await response.json();

  // data.graph contains:
  // - nodes: Array of graph nodes
  //   - explored paths (approved/alternative status)
  //   - pruned paths (rejected status)
  //   - decision points
  //   - plan steps
  // - edges: Connections between nodes
  //   - solid edges for accepted paths
  //   - dashed edges for pruned paths
  // - metadata: Statistics

  return data.graph;
}

// Use with ReasonGraph component
import { ReasonGraph } from '../components/ReasonGraph';

function PlanningResults({ sessionId }) {
  const [graphData, setGraphData] = useState(null);

  useEffect(() => {
    loadReasoningGraph(sessionId).then(setGraphData);
  }, [sessionId]);

  if (!graphData) return <div>Loading reasoning graph...</div>;

  return (
    <div>
      <h2>UltraThink Decision Process</h2>
      <ReasonGraph data={graphData} />

      <div className="stats">
        <p>Explored Paths: {graphData.metadata.total_explored_paths}</p>
        <p>Pruned Paths: {graphData.metadata.total_pruned_paths}</p>
        <p>Decision Points: {graphData.metadata.total_decision_points}</p>
      </div>
    </div>
  );
}
```

---

## 5. Real-Time Visualization

### 5.1 WebSocket Backend (Future)

```python
from fastapi import WebSocket, WebSocketDisconnect

class ReasoningBroadcaster:
    """
    Broadcast real-time reasoning updates
    """
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)

    async def broadcast_event(self, session_id: str, event: dict):
        """Send event to all connected clients for this session"""
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(event)
                except:
                    pass

broadcaster = ReasoningBroadcaster()

@app.websocket("/ws/reasoning/{session_id}")
async def reasoning_websocket(websocket: WebSocket, session_id: str):
    await broadcaster.connect(session_id, websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(session_id, websocket)
```

### 5.2 Emit Events from Planning Agent

```python
class PlanningAgent(BaseAgent):
    def __init__(self, *args, broadcaster=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.broadcaster = broadcaster

    async def _explore_path(self, node: SearchNode, session_id: str):
        """
        Explore path with real-time updates
        """
        # Emit "exploring" event
        if self.broadcaster:
            await self.broadcaster.broadcast_event(session_id, {
                "type": "path_exploring",
                "node_id": node.id,
                "strategy": node.strategy,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Explore the path
        result = await self._evaluate_path(node)

        # Emit "explored" or "pruned" event
        if node.is_pruned:
            await self.broadcaster.broadcast_event(session_id, {
                "type": "path_pruned",
                "node_id": node.id,
                "reason": node.pruning_reason,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            await self.broadcaster.broadcast_event(session_id, {
                "type": "path_explored",
                "node_id": node.id,
                "score": node.combined_score,
                "timestamp": datetime.utcnow().isoformat()
            })

        return result
```

### 5.3 Frontend WebSocket Integration

```typescript
function ReasonGraphLive({ sessionId }: { sessionId: string }) {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket(`ws://localhost:8000/ws/reasoning/${sessionId}`);

    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);

      if (update.type === 'path_exploring') {
        // Highlight node being explored
        setActiveNodeId(update.node_id);

      } else if (update.type === 'path_explored') {
        // Add explored node to graph
        setNodes(prev => [...prev, {
          id: update.node_id,
          status: 'approved',
          score: update.score,
          // ... other properties
        }]);
        setActiveNodeId(null);

      } else if (update.type === 'path_pruned') {
        // Add pruned node to graph
        setNodes(prev => [...prev, {
          id: update.node_id,
          status: 'rejected',
          reason: update.reason,
          // ... other properties
        }]);
        setActiveNodeId(null);
      }
    };

    return () => ws.close();
  }, [sessionId]);

  return <ReasonGraphVisualization nodes={nodes} activeNodeId={activeNodeId} />;
}
```

---

## 6. Testing Guide

### 6.1 Test Message Persistence

```python
import pytest
from agents.message_persistence import PersistentMessageQueue

@pytest.mark.asyncio
async def test_message_persistence_recovery():
    """
    Test that messages survive system restart
    """
    agent_id = "test_agent"

    # Create queue and add messages
    queue1 = PersistentMessageQueue(
        agent_id=agent_id,
        storage_path="./test_data/queues",
        enable_persistence=True
    )

    # Add test messages
    for i in range(10):
        msg = AgentMessage(
            message_id=f"msg_{i}",
            agent_id=agent_id,
            # ... other fields
        )
        await queue1.put(msg)

    # Shutdown (persists to disk)
    await queue1.shutdown()

    # Simulate system restart - create new queue instance
    queue2 = PersistentMessageQueue(
        agent_id=agent_id,
        storage_path="./test_data/queues",
        enable_persistence=True
    )

    # Check messages were recovered
    metrics = queue2.get_metrics()
    assert metrics['pending_messages'] == 10

    # Can retrieve messages
    msg = await queue2.get()
    assert msg is not None
```

### 6.2 Test Backpressure

```python
@pytest.mark.asyncio
async def test_backpressure_drop_oldest():
    """
    Test DROP_OLDEST backpressure strategy
    """
    queue = PersistentMessageQueue(
        agent_id="test",
        max_queue_size=5,
        backpressure_strategy=BackpressureStrategy.DROP_OLDEST
    )

    # Fill queue to capacity
    messages = []
    for i in range(5):
        msg = create_test_message(i)
        messages.append(msg)
        await queue.put(msg)

    # Add one more - should drop oldest
    new_msg = create_test_message(100)
    await queue.put(new_msg)

    # Get all messages
    retrieved = []
    for _ in range(5):
        msg = await queue.get(timeout=0.1)
        if msg:
            retrieved.append(msg)

    # Should have messages 1-4 and 100 (message 0 dropped)
    retrieved_ids = [m.message_id for m in retrieved]
    assert "msg_0" not in retrieved_ids
    assert "msg_100" in retrieved_ids
```

### 6.3 Test Session Persistence

```python
def test_session_persistence_recovery():
    """
    Test session recovery after restart
    """
    # Create manager and session
    manager1 = SessionPersistenceManager(
        storage_path="./test_data/sessions"
    )

    session = manager1.create_session(
        session_id="test_session",
        user_id="user_123",
        initial_state={"goal": "test goal"}
    )

    # Shutdown (persists to disk)
    manager1.shutdown()

    # Simulate restart - create new manager
    manager2 = SessionPersistenceManager(
        storage_path="./test_data/sessions"
    )

    # Session should be recovered
    recovered = manager2.get_session("test_session")
    assert recovered is not None
    assert recovered.user_id == "user_123"
    assert recovered.state["goal"] == "test goal"
```

### 6.4 Test Enhanced Reasoning Graph

```python
def test_enhanced_reasoning_graph_mapping():
    """
    Test that enhanced mapper captures pruned paths
    """
    # Create mock planning response with reasoning trace
    planning_response = {
        "session_id": "test",
        "reasoning_trace": {
            "explored_paths": [
                {"path_type": "conservative", "combined_score": 0.85},
                {"path_type": "balanced", "combined_score": 0.92},
            ],
            "pruned_paths": [
                {
                    "path_type": "aggressive",
                    "combined_score": 0.45,
                    "pruning_reason": "Exceeds risk tolerance"
                }
            ],
            "decision_points": [
                {
                    "decision_type": "strategy_selection",
                    "alternatives_rejected": [
                        {"option": "aggressive", "reason": "Too risky"}
                    ]
                }
            ]
        }
    }

    # Map to graph
    graph_data = ReasonGraphMapper.map_planning_trace(planning_response)

    # Verify pruned paths are included
    pruned_nodes = [
        n for n in graph_data['nodes']
        if n.get('status') == 'rejected'
    ]

    assert len(pruned_nodes) > 0
    assert any('aggressive' in n['label'].lower() for n in pruned_nodes)

    # Verify decision points
    decision_nodes = [
        n for n in graph_data['nodes']
        if n.get('type') == 'decision'
    ]

    assert len(decision_nodes) > 0

    # Verify metadata
    assert graph_data['metadata']['total_pruned_paths'] == 1
```

---

## 7. Migration Path

### 7.1 Gradual Migration Strategy

**Phase 1: Add Persistence (No Breaking Changes)**
```python
# Step 1: Keep existing queue, add persistence alongside
class MyAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "MyAgent")

        # Keep old queue
        self.message_queue = asyncio.Queue()

        # Add new persistent queue (parallel)
        self.persistent_queue = PersistentMessageQueue(
            agent_id=agent_id,
            enable_persistence=True
        )

        # Feature flag for gradual rollout
        self.use_persistent_queue = os.getenv("USE_PERSISTENT_QUEUE", "false") == "true"

    async def get_message(self):
        if self.use_persistent_queue:
            return await self.persistent_queue.get()
        else:
            return await self.message_queue.get()
```

**Phase 2: Test in Staging**
```bash
# Enable in staging environment
export USE_PERSISTENT_QUEUE=true

# Monitor metrics
curl http://localhost:8000/metrics | grep queue
```

**Phase 3: Full Migration**
```python
# Remove old queue entirely
class MyAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "MyAgent")
        self.message_queue = PersistentMessageQueue(
            agent_id=agent_id,
            enable_persistence=True
        )
```

### 7.2 Data Migration

If you have existing session data in a different format:

```python
async def migrate_legacy_sessions():
    """
    Migrate sessions from old format to new persistent format
    """
    # Load legacy sessions (e.g., from Redis)
    legacy_sessions = await redis_client.keys("session:*")

    # Get new session manager
    session_manager = get_session_manager()

    migrated_count = 0

    for legacy_key in legacy_sessions:
        # Parse legacy session
        legacy_data = await redis_client.get(legacy_key)
        session_id = legacy_key.split(":")[1]

        # Create in new format
        session_manager.create_session(
            session_id=session_id,
            user_id=legacy_data["user_id"],
            initial_state=legacy_data["state"],
            metadata={"migrated_from": "redis"}
        )

        migrated_count += 1

    print(f"Migrated {migrated_count} sessions")
```

### 7.3 Rollback Plan

If issues arise, you can rollback:

```python
# Disable new features with environment variables
USE_PERSISTENT_QUEUE=false
USE_SESSION_PERSISTENCE=false

# Or with feature flags in code
if not feature_flags.is_enabled("persistent_queue"):
    # Use old implementation
    self.message_queue = asyncio.Queue()
```

---

## 8. Production Checklist

Before deploying to production:

### 8.1 Configuration

```bash
# Set appropriate storage paths
export MESSAGE_QUEUE_PATH=/var/lib/finpilot/queues
export SESSION_STORAGE_PATH=/var/lib/finpilot/sessions

# Configure queue sizes based on load testing
export MAX_QUEUE_SIZE=5000

# Set TTLs based on use case
export SESSION_TTL_HOURS=48

# Configure backpressure
export BACKPRESSURE_STRATEGY=DROP_OLDEST
```

### 8.2 Monitoring

```python
# Add Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram

messages_sent = Counter('messages_sent_total', 'Total messages sent')
messages_failed = Counter('messages_failed_total', 'Total messages failed')
queue_size = Gauge('queue_size', 'Current queue size', ['agent_id'])
message_latency = Histogram('message_latency_seconds', 'Message processing latency')

# Update metrics in code
async def send_message(self, message):
    messages_sent.inc()
    with message_latency.time():
        await self.message_queue.put(message)
    queue_size.labels(agent_id=self.agent_id).set(self.message_queue.qsize())
```

### 8.3 Alerts

```yaml
# alerting_rules.yml
groups:
  - name: finpilot_agents
    rules:
      - alert: HighQueueUtilization
        expr: queue_utilization > 0.8
        for: 5m
        annotations:
          summary: "Queue {{ $labels.agent_id }} is {{ $value }}% full"

      - alert: HighMessageFailureRate
        expr: rate(messages_failed_total[5m]) > 0.1
        annotations:
          summary: "Message failure rate: {{ $value }}/s"

      - alert: DLQBuildup
        expr: dlq_size > 100
        annotations:
          summary: "Dead letter queue has {{ $value }} messages"
```

### 8.4 Backup and Recovery

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR=/backup/finpilot/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Backup message queues
cp -r /var/lib/finpilot/queues $BACKUP_DIR/

# Backup sessions
cp -r /var/lib/finpilot/sessions $BACKUP_DIR/

# Upload to S3
aws s3 sync $BACKUP_DIR s3://finpilot-backups/$(date +%Y%m%d)/
```

---

## 9. Troubleshooting

### Common Issues

**Q: Messages not persisting to disk**
```python
# Check write permissions
import os
print(os.access("./data/message_queues", os.W_OK))

# Check disk space
import shutil
stats = shutil.disk_usage("./data")
print(f"Free space: {stats.free / (1024**3):.2f} GB")

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
```

**Q: Sessions not loading after restart**
```python
# Check file format
import json
with open("./data/sessions/session_abc123.json") as f:
    data = json.load(f)
    print(data)

# Verify datetime format
from datetime import datetime
datetime.fromisoformat(data['created_at'])  # Should not raise error
```

**Q: Queue filling up too fast**
```python
# Check producer/consumer balance
metrics = queue.get_metrics()
print(f"Received: {metrics['messages_received']}")
print(f"Delivered: {metrics['messages_delivered']}")
print(f"Rate: {metrics['messages_received'] - metrics['messages_delivered']}/s")

# If imbalance, either:
# 1. Scale up consumers (add more agent instances)
# 2. Adjust backpressure strategy
# 3. Increase max_queue_size
```

---

## 10. Next Steps

1. **Read the full reports:**
   - `AGENT_INTEGRATION_REPORT.md` - Complete system analysis
   - `ULTRATHINK_TECHNICAL_DEEP_DIVE.md` - ToS algorithm details

2. **Start with small changes:**
   - Add persistence to one agent first
   - Test thoroughly in development
   - Gradually roll out to other agents

3. **Monitor metrics:**
   - Set up Prometheus + Grafana
   - Create dashboards for queue health
   - Configure alerts

4. **Plan WebSocket integration:**
   - Set up FastAPI WebSocket endpoint
   - Add event emission to agents
   - Connect frontend ReasonGraphLive

---

**Document Version:** 1.0
**Status:** ✅ Ready for Production
**Support:** See `AGENT_INTEGRATION_REPORT.md` for detailed architecture
