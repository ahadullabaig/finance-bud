"""
FinPilot Multi-Agent System - Shared Constants

Centralized constants used across all agents.

Requirements: 9.1, 9.4
"""

from typing import Dict

# ============================================================================
# API CONFIGURATION
# ============================================================================

# API Providers
API_PROVIDERS = {
    'BARCHART': 'barchart',
    'ALPHA_VANTAGE': 'alphavantage',
    'MASSIVE': 'massive',
    'MOCK': 'mock'
}

# API Endpoints
BARCHART_BASE_URL = 'https://marketdata.websol.barchart.com'
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'
MASSIVE_BASE_URL = 'https://api.massivedata.com/v1'

# API Rate Limits (requests per minute)
RATE_LIMITS = {
    'barchart': 60,
    'alphavantage': 5,  # Free tier
    'massive': 100,
    'mock': 1000
}

# API Timeout (seconds)
API_TIMEOUT_SECONDS = 30

# Circuit Breaker Configuration
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT_DURATION = 60  # seconds
CIRCUIT_BREAKER_EXPECTED_EXCEPTION = Exception

# Retry Configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# Redis Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None

# Cache TTL (seconds)
CACHE_TTL = {
    'market_data': 60,          # 1 minute for real-time data
    'historical_data': 3600,    # 1 hour for historical data
    'news': 300,                # 5 minutes for news
    'fundamentals': 86400,      # 24 hours for fundamentals
    'market_context': 120       # 2 minutes for market context
}

# Cache Key Prefixes
CACHE_KEY_PREFIX = {
    'market_data': 'finpilot:market:',
    'trigger': 'finpilot:trigger:',
    'plan': 'finpilot:plan:',
    'verification': 'finpilot:verify:'
}

# ============================================================================
# MARKET DATA THRESHOLDS
# ============================================================================

# Volatility Thresholds
VOLATILITY_THRESHOLDS = {
    'low': 10.0,        # VIX < 10
    'normal': 20.0,     # VIX 10-20
    'high': 30.0,       # VIX 20-30
    'extreme': 40.0     # VIX > 30
}

# Market Change Thresholds (percentage)
MARKET_CHANGE_THRESHOLDS = {
    'minor': 1.0,       # < 1% change
    'moderate': 2.5,    # 1-2.5% change
    'significant': 5.0, # 2.5-5% change
    'major': 10.0,      # > 5% change
    'crash': 15.0       # > 15% intraday drop
}

# Trigger Detection Thresholds
TRIGGER_THRESHOLDS = {
    'volatility_spike': 25.0,          # VIX increase threshold
    'market_crash': -10.0,              # % drop to trigger crash
    'volume_surge': 2.0,                # Multiple of average volume
    'correlation_break': 0.3,           # Correlation coefficient drop
    'interest_rate_change': 0.25        # % change in interest rates
}

# ============================================================================
# SEVERITY LEVELS
# ============================================================================

# Severity Score Ranges (0-100)
SEVERITY_SCORES = {
    'low': (0, 25),
    'medium': (25, 50),
    'high': (50, 75),
    'critical': (75, 100)
}

# Event Impact Multipliers
EVENT_IMPACT_MULTIPLIERS = {
    'market_crash': 3.0,
    'regulatory_change': 2.0,
    'job_loss': 2.5,
    'family_emergency': 2.0,
    'volatility_spike': 1.5,
    'interest_rate_change': 1.8
}

# ============================================================================
# PLANNING CONFIGURATION
# ============================================================================

# Planning Horizons (years)
PLANNING_HORIZONS = {
    'short_term': 1,
    'medium_term': 5,
    'long_term': 10,
    'retirement': 30
}

# Planning Algorithm Parameters
PLANNING_PARAMS = {
    'max_paths_to_explore': 100,
    'max_search_depth': 10,
    'min_success_probability': 0.7,
    'constraint_weight': 0.4,
    'return_weight': 0.3,
    'risk_weight': 0.3
}

# Guided Search (ToS) Parameters
GUIDED_SEARCH_PARAMS = {
    'beam_width': 5,                    # Number of paths to keep
    'heuristic_weights': {
        'information_gain': 0.3,
        'state_similarity': 0.2,
        'constraint_complexity': 0.2,
        'risk_adjusted_return': 0.3
    },
    'pruning_threshold': 0.3,           # Prune paths with score < threshold
    'max_alternatives': 5                # Max alternative plans to generate
}

# ============================================================================
# VERIFICATION CONFIGURATION
# ============================================================================

# Verification Thresholds
VERIFICATION_THRESHOLDS = {
    'min_confidence_score': 0.8,
    'max_constraint_violations': 0,
    'max_risk_score': 85.0,
    'min_success_probability': 0.75
}

# Compliance Rules
COMPLIANCE_RULES = {
    'max_single_stock_allocation': 0.15,    # 15% max in single stock
    'min_emergency_fund_months': 6,          # 6 months of expenses
    'max_debt_to_income': 0.4,              # 40% max DTI ratio
    'min_diversification_score': 0.6        # Minimum portfolio diversity
}

# ============================================================================
# REGULATORY CONFIGURATION
# ============================================================================

# Tax Brackets (India - Example FY 2024-25)
TAX_BRACKETS_INDIA = [
    (0, 300000, 0.0),           # No tax up to 3L
    (300000, 700000, 0.05),     # 5% from 3L to 7L
    (700000, 1000000, 0.10),    # 10% from 7L to 10L
    (1000000, 1200000, 0.15),   # 15% from 10L to 12L
    (1200000, 1500000, 0.20),   # 20% from 12L to 15L
    (1500000, float('inf'), 0.30)  # 30% above 15L
]

# Tax Saving Limits (India)
TAX_SAVING_LIMITS = {
    '80C': 150000,      # ELSS, PPF, etc.
    '80D': 25000,       # Health insurance (50k for senior citizens)
    '80CCD1B': 50000,   # NPS additional deduction
    'HRA': None,        # Varies by salary
    'HOME_LOAN': 200000 # Interest deduction
}

# Regulatory Jurisdictions
SUPPORTED_JURISDICTIONS = [
    'INDIA',
    'USA',
    'UK',
    'EU'
]

# ============================================================================
# RISK CONFIGURATION
# ============================================================================

# Risk Tolerance Profiles
RISK_PROFILES = {
    'conservative': {
        'max_equity': 0.3,
        'min_fixed_income': 0.6,
        'max_volatility': 10.0,
        'max_drawdown': 10.0
    },
    'moderate': {
        'max_equity': 0.6,
        'min_fixed_income': 0.3,
        'max_volatility': 15.0,
        'max_drawdown': 20.0
    },
    'aggressive': {
        'max_equity': 0.8,
        'min_fixed_income': 0.1,
        'max_volatility': 20.0,
        'max_drawdown': 30.0
    },
    'very_aggressive': {
        'max_equity': 0.95,
        'min_fixed_income': 0.05,
        'max_volatility': 30.0,
        'max_drawdown': 40.0
    }
}

# Asset Class Expected Returns (Annual %)
EXPECTED_RETURNS = {
    'equity_large_cap': 12.0,
    'equity_mid_cap': 14.0,
    'equity_small_cap': 16.0,
    'debt_govt': 7.0,
    'debt_corporate': 8.0,
    'gold': 8.0,
    'real_estate': 10.0,
    'cash': 4.0
}

# Asset Class Volatilities (Annual %)
ASSET_VOLATILITIES = {
    'equity_large_cap': 18.0,
    'equity_mid_cap': 22.0,
    'equity_small_cap': 28.0,
    'debt_govt': 5.0,
    'debt_corporate': 7.0,
    'gold': 15.0,
    'real_estate': 12.0,
    'cash': 1.0
}

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# Agent Names
AGENT_NAMES = {
    'ORCHESTRATOR': 'OrchestratorAgent',
    'PLANNER': 'PlanningAgent',
    'RETRIEVER': 'InformationRetrievalAgent',
    'VERIFIER': 'VerificationAgent',
    'EXECUTOR': 'ExecutionAgent'
}

# Agent Communication Timeouts (seconds)
AGENT_TIMEOUTS = {
    'orchestrator': 5.0,
    'planner': 30.0,      # Planning can take time
    'retriever': 10.0,
    'verifier': 15.0,
    'executor': 20.0
}

# Agent Retry Configuration
AGENT_RETRY = {
    'max_retries': 3,
    'retry_delay': 1.0,
    'exponential_backoff': True
}

# ============================================================================
# MONITORING & PERFORMANCE
# ============================================================================

# Performance SLAs (milliseconds)
PERFORMANCE_SLA = {
    'simple_query': 1000,           # 1 second
    'complex_query': 3000,          # 3 seconds
    'plan_generation': 5000,        # 5 seconds
    'verification': 2000,           # 2 seconds
    'api_call': 500                 # 500ms
}

# Health Check Intervals (seconds)
HEALTH_CHECK_INTERVAL = 60

# Metrics Collection
METRICS_ENABLED = True
METRICS_COLLECTION_INTERVAL = 30  # seconds

# ============================================================================
# MOCK DATA CONFIGURATION
# ============================================================================

# Mock Market Scenarios
MOCK_SCENARIOS = {
    'normal': {
        'volatility': 15.0,
        'market_change': 0.5,
        'sentiment': 'neutral'
    },
    'crash': {
        'volatility': 45.0,
        'market_change': -12.0,
        'sentiment': 'bearish'
    },
    'bull': {
        'volatility': 12.0,
        'market_change': 8.0,
        'sentiment': 'bullish'
    },
    'bear': {
        'volatility': 25.0,
        'market_change': -5.0,
        'sentiment': 'bearish'
    },
    'volatility_spike': {
        'volatility': 35.0,
        'market_change': -3.0,
        'sentiment': 'uncertain'
    }
}

# Mock Indices
MOCK_INDICES = [
    'NIFTY50',
    'SENSEX',
    'BANKNIFTY',
    'NIFTYIT',
    'SPY',      # S&P 500
    'QQQ'       # NASDAQ
]

# ============================================================================
# FILE PATHS
# ============================================================================

# Log Files
LOG_DIR = 'logs'
LOG_FILES = {
    'orchestrator': f'{LOG_DIR}/orchestrator.log',
    'planner': f'{LOG_DIR}/planner.log',
    'retriever': f'{LOG_DIR}/retriever.log',
    'verifier': f'{LOG_DIR}/verifier.log',
    'executor': f'{LOG_DIR}/executor.log',
    'system': f'{LOG_DIR}/system.log'
}

# Data Directories
DATA_DIR = 'data'
MOCK_DATA_DIR = f'{DATA_DIR}/mock'
HISTORICAL_DATA_DIR = f'{DATA_DIR}/historical'

# Configuration Files
CONFIG_DIR = 'config'
API_CONFIG_FILE = f'{CONFIG_DIR}/api_config.json'

# ============================================================================
# MESSAGE TYPES
# ============================================================================

# Inter-Agent Message Types
MESSAGE_TYPES = {
    'PLAN_REQUEST': 'plan_request',
    'PLAN_RESPONSE': 'plan_response',
    'VERIFY_REQUEST': 'verify_request',
    'VERIFY_RESPONSE': 'verify_response',
    'MARKET_DATA_REQUEST': 'market_data_request',
    'MARKET_DATA_RESPONSE': 'market_data_response',
    'TRIGGER_EVENT': 'trigger_event',
    'EXECUTE_REQUEST': 'execute_request',
    'EXECUTE_RESPONSE': 'execute_response',
    'HEALTH_CHECK': 'health_check',
    'ERROR': 'error'
}

# ============================================================================
# ERROR CODES
# ============================================================================

ERROR_CODES = {
    'API_ERROR': 'ERR_API_001',
    'VALIDATION_ERROR': 'ERR_VAL_002',
    'CONSTRAINT_VIOLATION': 'ERR_CON_003',
    'TIMEOUT': 'ERR_TIME_004',
    'CIRCUIT_OPEN': 'ERR_CIRC_005',
    'CACHE_ERROR': 'ERR_CACHE_006',
    'AGENT_COMM_ERROR': 'ERR_AGENT_007',
    'PLAN_REJECTED': 'ERR_PLAN_008',
    'INSUFFICIENT_DATA': 'ERR_DATA_009',
    'REGULATORY_VIOLATION': 'ERR_REG_010'
}

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # API
    'API_PROVIDERS',
    'RATE_LIMITS',
    'API_TIMEOUT_SECONDS',
    'CIRCUIT_BREAKER_FAILURE_THRESHOLD',
    'CIRCUIT_BREAKER_TIMEOUT_DURATION',

    # Cache
    'CACHE_TTL',
    'CACHE_KEY_PREFIX',

    # Thresholds
    'VOLATILITY_THRESHOLDS',
    'MARKET_CHANGE_THRESHOLDS',
    'TRIGGER_THRESHOLDS',
    'SEVERITY_SCORES',

    # Planning
    'PLANNING_HORIZONS',
    'PLANNING_PARAMS',
    'GUIDED_SEARCH_PARAMS',

    # Verification
    'VERIFICATION_THRESHOLDS',
    'COMPLIANCE_RULES',

    # Regulatory
    'TAX_BRACKETS_INDIA',
    'TAX_SAVING_LIMITS',

    # Risk
    'RISK_PROFILES',
    'EXPECTED_RETURNS',
    'ASSET_VOLATILITIES',

    # Agents
    'AGENT_NAMES',
    'AGENT_TIMEOUTS',

    # Monitoring
    'PERFORMANCE_SLA',
    'METRICS_ENABLED',

    # Mock
    'MOCK_SCENARIOS',
    'MOCK_INDICES',

    # Paths
    'LOG_FILES',
    'API_CONFIG_FILE',

    # Messages
    'MESSAGE_TYPES',
    'ERROR_CODES'
]
