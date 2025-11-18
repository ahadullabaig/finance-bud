"""
FinPilot Utilities Package

Shared utilities for logging, constants, and common functionality.
"""

from utils.logger import (
    StructuredLogger,
    # get_logger,
    # LoggingStandards,
    # correlation_id_var
)

from utils.constants import (
    # API
    API_PROVIDERS,
    RATE_LIMITS,
    API_TIMEOUT_SECONDS,

    # Cache
    CACHE_TTL,
    CACHE_KEY_PREFIX,

    # Thresholds
    VOLATILITY_THRESHOLDS,
    MARKET_CHANGE_THRESHOLDS,
    TRIGGER_THRESHOLDS,
    SEVERITY_SCORES,

    # Planning
    PLANNING_HORIZONS,
    PLANNING_PARAMS,
    GUIDED_SEARCH_PARAMS,

    # Verification
    VERIFICATION_THRESHOLDS,
    COMPLIANCE_RULES,

    # Risk
    RISK_PROFILES,
    EXPECTED_RETURNS,

    # Agents
    AGENT_NAMES,
    AGENT_TIMEOUTS,

    # Mock
    MOCK_SCENARIOS,
    MOCK_INDICES,

    # Messages
    MESSAGE_TYPES,
    ERROR_CODES
)

__all__ = [
    # Logger
    'StructuredLogger',
    # 'get_logger',
    # 'LoggingStandards',
    # 'correlation_id_var',

    # API Constants
    'API_PROVIDERS',
    'RATE_LIMITS',
    'API_TIMEOUT_SECONDS',

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

    # Risk
    'RISK_PROFILES',
    'EXPECTED_RETURNS',

    # Agents
    'AGENT_NAMES',
    'AGENT_TIMEOUTS',

    # Mock
    'MOCK_SCENARIOS',
    'MOCK_INDICES',

    # Messages
    'MESSAGE_TYPES',
    'ERROR_CODES'
]
