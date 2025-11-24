"""
FinPilot Multi-Agent System - Agent Package

This package contains all production agent implementations for the VP-MAS system.

NOTE: Mock agents have been moved to tests/mocks/mock_interfaces.py
Mock agents should only be imported in test code, not in production.
"""

from .base_agent import BaseAgent
from .communication import AgentCommunicationFramework
from .orchestration_agent import OrchestrationAgent

__all__ = [
    'BaseAgent',
    'AgentCommunicationFramework',
    'OrchestrationAgent',
]