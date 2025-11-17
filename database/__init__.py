"""
Supabase integration module for FinPilot VP-MAS

Provides centralized access to Supabase configuration, database operations,
and client management for the multi-agent financial planning system.
"""

from .config import get_supabase_client, get_supabase_config, SupabaseConfig, SupabaseClient
from .database import (
    DatabaseManager, db,
    FinancialStateDB, PlansDB, PlanStepsDB, MarketDataDB,
    AgentMessagesDB, ExecutionLogsDB, VerificationReportsDB
)

__all__ = [
    'get_supabase_client',
    'get_supabase_config', 
    'SupabaseConfig',
    'SupabaseClient',
    'DatabaseManager',
    'db',
    'FinancialStateDB',
    'PlansDB', 
    'PlanStepsDB',
    'MarketDataDB',
    'AgentMessagesDB',
    'ExecutionLogsDB',
    'VerificationReportsDB'
]