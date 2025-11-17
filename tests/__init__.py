"""
FinPilot Multi-Agent System - Test Suite

Comprehensive testing framework for VP-MAS agents, communication protocols,
and integration scenarios with mock data and performance benchmarks.

Requirements: 11.1, 11.2, 11.3
"""

# Import only existing modules
try:
    from .test_agents import *
except ImportError:
    pass

try:
    from .test_integration import *
except ImportError:
    pass

try:
    from .test_performance import *
except ImportError:
    pass

try:
    from .mock_data import *
except ImportError:
    pass

__all__ = [
    'TestAgentCommunication',
    'TestMockAgents',
    'TestOrchestrationAgent'
]