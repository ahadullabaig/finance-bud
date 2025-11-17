-- FinPilot VP-MAS Database Schema
-- Initial migration for Supabase database setup

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table (extends Supabase auth.users)
CREATE TABLE public.user_profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    email TEXT NOT NULL,
    full_name TEXT,
    risk_tolerance TEXT DEFAULT 'moderate',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Financial states table
CREATE TABLE public.financial_states (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    total_assets DECIMAL(15,2) DEFAULT 0,
    total_liabilities DECIMAL(15,2) DEFAULT 0,
    net_worth DECIMAL(15,2) GENERATED ALWAYS AS (total_assets - total_liabilities) STORED,
    monthly_income DECIMAL(15,2) DEFAULT 0,
    monthly_expenses DECIMAL(15,2) DEFAULT 0,
    monthly_cash_flow DECIMAL(15,2) GENERATED ALWAYS AS (monthly_income - monthly_expenses) STORED,
    risk_tolerance TEXT DEFAULT 'moderate',
    tax_filing_status TEXT DEFAULT 'single',
    estimated_tax_rate DECIMAL(5,4) DEFAULT 0.22,
    as_of_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Plans table
CREATE TABLE public.plans (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    plan_data JSONB NOT NULL,
    status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'completed', 'cancelled')),
    confidence_score DECIMAL(3,2) DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    correlation_id UUID NOT NULL,
    session_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Plan steps table
CREATE TABLE public.plan_steps (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    plan_id UUID REFERENCES public.plans(id) ON DELETE CASCADE,
    step_id UUID DEFAULT uuid_generate_v4(),
    sequence_number INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    description TEXT NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    target_date TIMESTAMP WITH TIME ZONE NOT NULL,
    rationale TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    risk_level TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market data table
CREATE TABLE public.market_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    data_id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source TEXT NOT NULL,
    market_volatility DECIMAL(5,4) DEFAULT 0,
    interest_rates JSONB DEFAULT '{}'::jsonb,
    sector_trends JSONB DEFAULT '{}'::jsonb,
    economic_sentiment DECIMAL(3,2) DEFAULT 0 CHECK (economic_sentiment >= -1 AND economic_sentiment <= 1),
    collection_method TEXT,
    refresh_frequency INTEGER DEFAULT 300,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent messages table
CREATE TABLE public.agent_messages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    message_id UUID DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    target_agent_id TEXT,
    message_type TEXT NOT NULL CHECK (message_type IN ('request', 'response', 'notification', 'error', 'heartbeat')),
    payload JSONB NOT NULL,
    correlation_id UUID NOT NULL,
    session_id UUID,
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('critical', 'high', 'medium', 'low')),
    trace_id UUID,
    retry_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Execution logs table
CREATE TABLE public.execution_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    log_id UUID DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    operation_name TEXT NOT NULL,
    execution_status TEXT NOT NULL CHECK (execution_status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
    input_data JSONB DEFAULT '{}'::jsonb,
    output_data JSONB DEFAULT '{}'::jsonb,
    execution_time DECIMAL(10,6) DEFAULT 0,
    session_id UUID,
    correlation_id UUID NOT NULL,
    trace_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Verification reports table
CREATE TABLE public.verification_reports (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    report_id UUID DEFAULT uuid_generate_v4(),
    plan_id UUID REFERENCES public.plans(id) ON DELETE CASCADE,
    verification_status TEXT NOT NULL CHECK (verification_status IN ('approved', 'rejected', 'conditional', 'pending')),
    constraints_checked INTEGER DEFAULT 0,
    constraints_passed INTEGER DEFAULT 0,
    constraint_violations JSONB DEFAULT '[]'::jsonb,
    overall_risk_score DECIMAL(3,2) DEFAULT 0 CHECK (overall_risk_score >= 0 AND overall_risk_score <= 1),
    approval_rationale TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    verification_time DECIMAL(10,6) DEFAULT 0,
    verifier_agent_id TEXT NOT NULL,
    correlation_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.financial_states ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.plan_steps ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.verification_reports ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own profile" ON public.user_profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.user_profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can view own financial states" ON public.financial_states
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own financial states" ON public.financial_states
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own plans" ON public.plans
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own plans" ON public.plans
    FOR ALL USING (auth.uid() = user_id);