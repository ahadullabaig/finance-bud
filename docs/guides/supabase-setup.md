# Supabase Setup Guide for FinPilot VP-MAS

This guide will help you set up Supabase as the backend for your FinPilot multi-agent financial planning system.

## Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- Supabase account (free tier available)

## Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Click "New Project"
3. Choose your organization
4. Set project name: `finpilot-vpmas`
5. Choose a region close to you
6. Set a strong database password (save this!)
7. Click "Create new project"

## Step 2: Get Your Project Credentials

1. Go to **Settings > API** in your Supabase dashboard
2. Copy the following values:
   - **Project URL** (looks like: `https://your-project-id.supabase.co`)
   - **anon public** key (starts with `eyJ...`)
   - **service_role secret** key (starts with `eyJ...`) - Keep this secure!

## Step 3: Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Update your `.env` file with your Supabase credentials:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
   SUPABASE_SERVICE_KEY=your-service-key-here
   ```

## Step 4: Set Up Database Schema

1. Go to **SQL Editor** in your Supabase dashboard
2. Copy the contents of `supabase/migrations/001_initial_schema.sql`
3. Paste and run the SQL to create all tables and policies

## Step 5: Configure Row Level Security (RLS)

The migration script automatically enables RLS and creates policies. Verify in your dashboard:

1. Go to **Authentication > Policies**
2. Ensure all tables have appropriate policies
3. Test with a user account to verify access

## Step 6: Install Dependencies

### Frontend Dependencies
```bash
npm install
```

### Backend Dependencies
```bash
pip install -r requirements.txt
```

## Step 7: Test Connection

Run the setup script to verify everything is working:

```bash
python setup_supabase.py
```

Test the connection:
```bash
python -c "from supabase.config import get_supabase_client; print('✅ Connected!')"
```

## Step 8: Start Development

```bash
# Start frontend
npm run dev

# Start backend (in another terminal)
python -m uvicorn api.main:app --reload
```

## Database Tables Overview

The system creates the following tables:

### Core Tables
- `user_profiles` - User account information
- `financial_states` - User financial snapshots
- `plans` - Financial plans with metadata
- `plan_steps` - Individual plan actions

### Agent Communication
- `agent_messages` - Inter-agent communication
- `execution_logs` - Action execution tracking
- `verification_reports` - Plan verification results

### Market Data
- `market_data` - Real-time market information
- `trigger_events` - Market/life event triggers

### Analytics
- `reasoning_traces` - Decision-making traces for ReasonGraph

## Security Features

### Row Level Security (RLS)
- Users can only access their own data
- Service role can access all data for system operations
- Policies automatically filter data by user ID

### Authentication
- Built-in Supabase Auth with email/password
- JWT tokens for API authentication
- Session management with automatic refresh

### API Security
- Rate limiting on all endpoints
- Input validation with Pydantic
- CORS configuration for frontend access

## Real-time Features

### Supabase Realtime
- Live updates for plan changes
- Real-time agent communication
- Market data streaming
- Collaborative planning sessions

### WebSocket Integration
- Automatic reconnection
- Event filtering by user
- Optimistic updates with conflict resolution

## Monitoring and Analytics

### Built-in Analytics
- User activity tracking
- Performance metrics
- Error monitoring
- Usage analytics

### Custom Metrics
- Agent performance tracking
- Planning success rates
- Market data quality
- System health monitoring

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check your environment variables
   - Verify project URL and keys
   - Ensure network connectivity

2. **Permission Denied**
   - Verify RLS policies are correct
   - Check user authentication
   - Ensure proper API key usage

3. **Migration Errors**
   - Check SQL syntax in migration file
   - Verify database permissions
   - Look for conflicting table names

### Debug Commands

```bash
# Test database connection
python -c "from supabase import get_supabase_client; client = get_supabase_client(); print(client.health_check())"

# Check environment variables
python -c "import os; print('URL:', os.getenv('SUPABASE_URL')); print('Key:', os.getenv('SUPABASE_ANON_KEY')[:20] + '...')"

# Test authentication
python -c "from supabase import get_supabase_client; client = get_supabase_client(); print(client.auth.get_session())"
```

## Production Deployment

### Environment Setup
- Use environment variables for all secrets
- Enable SSL/TLS for all connections
- Configure proper CORS origins
- Set up monitoring and alerting

### Performance Optimization
- Enable connection pooling
- Configure appropriate indexes
- Set up caching strategies
- Monitor query performance

### Backup and Recovery
- Enable automated backups
- Test restore procedures
- Document recovery processes
- Set up monitoring alerts

## Support and Resources

- [Supabase Documentation](https://supabase.com/docs)
- [Python Client Reference](https://supabase.com/docs/reference/python)
- [JavaScript Client Reference](https://supabase.com/docs/reference/javascript)
- [Community Discord](https://discord.supabase.com)

## Next Steps

1. Complete the basic setup following this guide
2. Run the test suite to verify functionality
3. Explore the demo scenarios
4. Customize for your specific requirements
5. Deploy to production when ready

Your FinPilot system is now ready to use Supabase as its backend!