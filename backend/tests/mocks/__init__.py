"""
Mock Agents for Testing

This module contains mock implementations of all VP-MAS agents.
Use ONLY for testing and development, NOT for production.

Mock agents provide fast, deterministic behavior for unit and integration tests.
"""

from .mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent,
    MockVerificationAgent,
    MockExecutionAgent
)

__all__ = [
    "MockOrchestrationAgent",
    "MockPlanningAgent",
    "MockInformationRetrievalAgent",
    "MockVerificationAgent",
    "MockExecutionAgent",
]
