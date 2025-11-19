# Phase 6 Quick Start Guide

Get started with FinPilot's Advanced AI Features in 5 minutes!

---

## Prerequisites

✅ Python 3.10+
✅ Dependencies installed (`pip install -r requirements.txt`)
✅ Phase 1-5 complete (existing agents working)

---

## Quick Test (No Setup Required)

The system works out-of-the-box with fallback implementations:

```bash
# Run comprehensive test suite
python test_phase6_agents.py
```

**Expected output**: All tests passing ✅

---

## Option 1: Use with Rule-Based Parsing (No LLM)

All agents work immediately with intelligent fallbacks:

```python
import asyncio
from agents.conversational_agent import get_conversational_agent
from agents.graph_risk_detector import get_graph_risk_detector
from agents.ml_prediction_engine import get_ml_prediction_engine

# Example 1: Parse financial goal (rule-based)
async def demo_parsing():
    agent = get_conversational_agent()
    result = await agent.parse_natural_language_goal(
        "I want to save $100,000 for a house down payment in 5 years"
    )
    print(f"Goal type: {result['goal_type']}")
    print(f"Amount: ${result['target_amount']:,.2f}")

# Example 2: Detect transaction anomalies
async def demo_risk_detection():
    detector = get_graph_risk_detector()
    report = await detector.analyze_transaction_risk("user_123", lookback_days=90)
    print(f"Risk level: {report['risk_level']}")
    print(f"Anomalies: {len(report['anomalies'])}")

# Example 3: Predict market trends
async def demo_ml_prediction():
    engine = get_ml_prediction_engine()
    prediction = await engine.predict_market_trend("AAPL", horizon_days=30)
    print(f"Current: ${prediction['current_price']:.2f}")
    print(f"Predicted: ${prediction['predicted_price']:.2f}")

# Run examples
asyncio.run(demo_parsing())
asyncio.run(demo_risk_detection())
asyncio.run(demo_ml_prediction())
```

---

## Option 2: Enable LLM Features (Recommended)

For enhanced natural language understanding:

### Step 1: Install Ollama

**Linux/macOS**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**:
Download from https://ollama.com/download

### Step 2: Pull Model

```bash
# Lightweight model (3B parameters, ~2GB)
ollama pull llama3.2:3b

# Or larger model for better accuracy (8B parameters, ~5GB)
ollama pull llama3.2:8b
```

### Step 3: Start Ollama Service

```bash
# Start in background
ollama serve &

# Verify
ollama list
```

### Step 4: Test LLM Parsing

```python
import asyncio
from agents.conversational_agent import get_conversational_agent

async def test_llm():
    agent = get_conversational_agent(model_name="llama3.2:3b")

    # Now uses LLM for parsing
    result = await agent.parse_natural_language_goal(
        "I'm 28 years old, make $80k/year, and want to retire early at 55 "
        "with enough to travel the world. I'm comfortable with moderate risk."
    )

    print(f"Parsing method: {result['parsing_method']}")  # Should show 'llm'
    print(f"Goal: {result['goal_type']}")
    print(f"Risk: {result['risk_tolerance']}")

asyncio.run(test_llm())
```

---

## API Usage

### Start FastAPI Server

First, register the new routers in your main FastAPI app:

**Edit your main app file** (e.g., `main.py` or `api/app.py`):

```python
from fastapi import FastAPI
from api.conversational_endpoints import router as conversational_router
from api.risk_endpoints import router as risk_router
from api.ml_endpoints import router as ml_router

app = FastAPI(title="FinPilot Multi-Agent System")

# Register Phase 6 routers
app.include_router(conversational_router)
app.include_router(risk_router)
app.include_router(ml_router)

# ... other routers ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Start server**:
```bash
python main.py
# or
uvicorn main:app --reload
```

### API Examples

#### 1. Parse Financial Goal

```bash
curl -X POST http://localhost:8000/api/conversational/parse-goal \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I want to buy a house in 3 years and need $80,000 for down payment",
    "user_context": {"age": 32, "income": 90000}
  }'
```

#### 2. Analyze Transaction Risk

```bash
curl -X POST http://localhost:8000/api/risk/analyze-transactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "lookback_days": 90
  }'
```

#### 3. Predict Market Trend

```bash
curl -X POST http://localhost:8000/api/ml/predict-market \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "horizon_days": 30
  }'
```

#### 4. Get Personalized Recommendations

```bash
curl -X POST http://localhost:8000/api/ml/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_profile": {
      "age": 35,
      "risk_tolerance": "moderate",
      "monthly_expenses": 5000,
      "emergency_fund": 15000,
      "retirement_contributions": 0.08
    }
  }'
```

#### 5. Detect Fraud Patterns

```bash
curl -X POST http://localhost:8000/api/risk/detect-fraud \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"amount": 5000, "merchant": "Unknown", "timestamp": "2024-01-15T02:30:00"},
      {"amount": 5100, "merchant": "Unknown", "timestamp": "2024-01-15T02:35:00"}
    ],
    "user_profile": {"average_transaction": 150}
  }'
```

---

## Integration with Frontend

### Example React Component

```typescript
// ReasonGraphWithAI.tsx
import { useState } from 'react';

export function ReasonGraphWithAI() {
  const [userGoal, setUserGoal] = useState('');
  const [parsedGoal, setParsedGoal] = useState(null);

  const handleParseGoal = async () => {
    const response = await fetch('http://localhost:8000/api/conversational/parse-goal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_input: userGoal,
        user_context: { age: 35, income: 100000 }
      })
    });

    const result = await response.json();
    setParsedGoal(result);
  };

  return (
    <div>
      <textarea
        value={userGoal}
        onChange={(e) => setUserGoal(e.target.value)}
        placeholder="Describe your financial goal in plain English..."
      />
      <button onClick={handleParseGoal}>Parse Goal</button>

      {parsedGoal && (
        <div>
          <h3>Understood:</h3>
          <p>Goal: {parsedGoal.goal_type}</p>
          <p>Amount: ${parsedGoal.target_amount?.toLocaleString()}</p>
          <p>Timeline: {parsedGoal.timeframe_years} years</p>
        </div>
      )}
    </div>
  );
}
```

---

## Common Use Cases

### Use Case 1: Natural Language Financial Planning

```python
# User describes goal in plain English
user_input = "I'm 25, make $60k, want to buy a $400k house in 7 years"

# Parse goal
goal = await conversational_agent.parse_natural_language_goal(user_input)

# Use parsed goal in planning
plan = await planning_agent.create_plan(goal)

# Generate explanation
narrative = await conversational_agent.generate_financial_narrative(plan)

# Show to user
print(narrative)
```

### Use Case 2: Fraud Detection Pipeline

```python
# Get user transactions
transactions = await execution_agent.get_transactions(user_id, days=90)

# Build graph
graph = await risk_detector.build_transaction_graph(transactions, user_id)

# Detect anomalies
anomalies = await risk_detector.detect_anomalous_patterns(graph)

# Check fraud
fraud_indicators = await risk_detector.find_fraud_indicators(transactions)

# Alert if high risk
if fraud_indicators and max(i['fraud_confidence'] for i in fraud_indicators) > 0.8:
    send_alert(user_id, fraud_indicators)
```

### Use Case 3: Portfolio Optimization with Predictions

```python
# Get current portfolio
portfolio = await execution_agent.get_portfolio(user_id)

# Predict future performance
prediction = await ml_engine.predict_portfolio_performance(portfolio, days=365)

# Analyze systemic risk
risk_graph = await risk_detector.build_asset_correlation_graph(portfolio['assets'])
systemic_risk = await risk_detector.calculate_systemic_risk(risk_graph)

# Get recommendations
recommendations = await ml_engine.generate_recommendations(
    user_profile,
    market_context={'volatility': systemic_risk['metrics']['density']}
)

# Optimize plan
optimized_plan = await planning_agent.optimize_with_predictions(
    current_plan,
    prediction,
    systemic_risk,
    recommendations
)
```

---

## Performance Tips

### 1. Cache Predictions

```python
import functools
from datetime import datetime, timedelta

@functools.lru_cache(maxsize=100)
def cached_market_prediction(symbol: str, date: str):
    # Cache predictions for same day
    return ml_engine.predict_market_trend(symbol, 30)
```

### 2. Batch Processing

```python
# Process multiple symbols in parallel
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
predictions = await asyncio.gather(*[
    ml_engine.predict_market_trend(symbol, 30)
    for symbol in symbols
])
```

### 3. Incremental Graph Updates

```python
# Update existing graph instead of rebuilding
new_transactions = get_new_transactions(since_last_update)
risk_detector.transaction_graph = update_graph(
    risk_detector.transaction_graph,
    new_transactions
)
```

---

## Monitoring & Debugging

### Check Agent Health

```bash
# Conversational Agent
curl http://localhost:8000/api/conversational/health

# Risk Detector
curl http://localhost:8000/api/risk/health

# ML Engine
curl http://localhost:8000/api/ml/health
```

### View Logs

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Agents log automatically
agent = get_conversational_agent()
# Will see detailed logs of operations
```

### Test Individual Components

```python
# Test just parsing
result = await conversational_agent.parse_natural_language_goal("test goal")

# Test just graph building
graph = await risk_detector.build_transaction_graph([], "test_user")

# Test just prediction
prediction = await ml_engine.predict_market_trend("AAPL", 30)
```

---

## Troubleshooting

### Issue: "Ollama not available"
**Solution**: System falls back to rule-based parsing (works fine). To enable LLM, install Ollama.

### Issue: "No module named 'networkx'"
**Solution**: `pip install networkx`

### Issue: "No module named 'sklearn'"
**Solution**: `pip install scikit-learn`

### Issue: API endpoints return 404
**Solution**: Make sure routers are registered in main FastAPI app

### Issue: Predictions seem inaccurate
**Solution**: This is MVP. Accuracy improves with:
- More historical data
- Better trained models
- Parameter tuning

---

## Next Steps

1. ✅ Run tests: `python test_phase6_agents.py`
2. 🔧 Register API routers in FastAPI app
3. 🚀 Start server and test endpoints
4. 🎨 Integrate with frontend
5. 📊 Connect to real market data (IRA)
6. 🔮 (Optional) Set up Ollama for LLM features
7. 📈 (Optional) Upgrade to NVIDIA NIM/cuGraph for production

---

## Getting Help

- **Documentation**: See `PHASE_6_IMPLEMENTATION_SUMMARY.md`
- **Tests**: Run `python test_phase6_agents.py` to verify setup
- **API Docs**: Visit `http://localhost:8000/docs` when server running
- **Logs**: Check console output for detailed operation logs

---

**Ready to use Phase 6 AI features!** 🚀

For production deployment, see upgrade paths in the implementation summary.
