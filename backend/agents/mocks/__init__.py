"""
Mock Agents Module

Mock implementations of all VP-MAS agents for independent development and testing.
Currently re-exports from parent mock_interfaces.py for backward compatibility.

Future: Each mock agent will be in its own file:
- orchestration.py
- planning.py
- retrieval.py
- verification.py
- execution.py
"""

# Import from parent directory for backward compatibility
from ..mock_interfaces import (
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
