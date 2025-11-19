# FinPilot MVP - Quick Deployment Guide

Your API is **ready to deploy** in under 10 minutes! Choose your preferred platform:

## ✅ Current Status

- ✅ API running successfully on http://localhost:8000
- ✅ All 4 agents initialized (Orchestration, Planning, IRA, Verification)
- ✅ All 12 data model tests passing
- ✅ Health check endpoint working
- ✅ Mock agents functional for rapid testing

## 🚀 Deploy Options (Choose One)

### Option 1: Render (Easiest - Recommended)

1. **Push code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy to Render**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Render will auto-detect the `render.yaml` config
   - Click "Create Web Service"
   - **Done!** Your API will be live in ~5 minutes

3. **Access your API**
   - Render will provide a URL like: `https://finpilot-api.onrender.com`
   - Test: `https://finpilot-api.onrender.com/health`

### Option 2: Railway (Fast & Simple)

1. **Install Railway CLI** (optional)
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Deploy**
   ```bash
   railway init
   railway up
   ```

   OR use the web UI:
   - Go to https://railway.app
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repo
   - Railway auto-detects Python and deploys

3. **Access your API**
   - Railway provides a URL like: `https://finpilot.up.railway.app`

### Option 3: Fly.io (Global Edge Deployment)

1. **Install Fly CLI**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and Deploy**
   ```bash
   fly auth login
   fly launch  # Uses fly.toml config
   fly deploy
   ```

3. **Access your API**
   - Fly provides a URL like: `https://finpilot-mvp.fly.dev`

### Option 4: Heroku (Classic)

1. **Deploy**
   ```bash
   heroku create finpilot-api
   git push heroku main
   ```

2. **Uses the Procfile** automatically

## 📋 Environment Variables (Optional)

If you're using external APIs or databases, set these in your deployment platform:

```bash
# Optional - for future integrations
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ALPHA_VANTAGE_API_KEY=your_api_key
```

For now, the MVP runs without these (using mock data).

## 🧪 Test Your Deployed API

Once deployed, test these endpoints:

```bash
# Health check
curl https://your-app-url.com/health

# Root endpoint
curl https://your-app-url.com/

# Submit a goal (test orchestration)
curl -X POST https://your-app-url.com/api/v1/orchestration/goals \
  -H "Content-Type: application/json" \
  -d '{"user_goal": "Save $50,000 for a house down payment"}'

# Generate a plan
curl -X POST https://your-app-url.com/api/v1/planning/generate \
  -H "Content-Type: application/json" \
  -d '{"planning_request": {"user_goal": "Retire in 10 years"}}'

# Demo complete workflow
curl -X POST https://your-app-url.com/api/v1/demo/complete-workflow \
  -H "Content-Type: application/json" \
  -d '{"user_goal": "Build emergency fund of $10,000"}'
```

## 📚 API Documentation

Once deployed, access interactive API docs:
- Swagger UI: `https://your-app-url.com/docs`
- ReDoc: `https://your-app-url.com/redoc`

## 🔥 What's Working Now

✅ **Agents:**
- Orchestration Agent (coordinates workflow)
- Planning Agent (generates financial plans)
- Information Retrieval Agent (market data, triggers)
- Verification Agent (validates plans with comprehensive rules)

✅ **Endpoints:**
- `/health` - System health check
- `/api/v1/orchestration/goals` - Submit user goals
- `/api/v1/planning/generate` - Generate financial plans
- `/api/v1/market/data` - Get market data
- `/api/v1/verification/verify` - Verify plans
- `/api/v1/demo/complete-workflow` - Full workflow demo
- `/api/v1/reasongraph/generate` - Generate reasoning visualizations

✅ **Features:**
- Request tracking (X-Request-ID headers)
- Execution time monitoring
- Agent health monitoring
- CORS enabled for frontend integration
- Structured error responses

## 🎯 Next Steps (Post-MVP)

After your MVP is deployed, you can gradually add:
1. Real database (Supabase/PostgreSQL)
2. External API integrations (Alpha Vantage, Yahoo Finance)
3. Redis caching for performance
4. Authentication & user management
5. Frontend integration
6. More comprehensive tests

## 💡 Pro Tips

1. **Use Render for demos** - Free tier, auto-deploys on git push
2. **Railway for rapid iteration** - Excellent DX, quick deploys
3. **Fly.io for production** - Global CDN, excellent performance
4. **Monitor health endpoint** - Set up alerts on `/health` endpoint

## 🆘 Troubleshooting

**Server won't start?**
- Check logs in your deployment platform
- Verify Python version is 3.11+
- Ensure all packages in requirements-minimal.txt install correctly

**Import errors?**
- The app needs the full codebase structure
- Make sure you've committed all agent files

**Need help?**
- Check `/health` endpoint for agent status
- Review logs for specific error messages
- API docs at `/docs` show all available endpoints

---

**You're all set!** Pick a platform above and deploy in the next 10 minutes. 🚀
