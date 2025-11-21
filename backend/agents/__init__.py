"""
FinPilot Multi-Agent System - Agent Package

This package contains all agent implementations and mock interfaces for the VP-MAS system.
Supports independent development with comprehensive mocking capabilities.
"""

from .mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent,
    MockVerificationAgent,
    MockExecutionAgent
)

from .base_agent import BaseAgent
from .communication import AgentCommunicationFramework
from .orchestration_agent import OrchestrationAgent

__all__ = [
    'BaseAgent',
    'AgentCommunicationFramework',
    'OrchestrationAgent',
    'MockOrchestrationAgent',
    'MockPlanningAgent', 
    'MockInformationRetrievalAgent',
    'MockVerificationAgent',
    'MockExecutionAgent'
]