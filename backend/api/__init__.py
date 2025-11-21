"""
FinPilot Multi-Agent System - API Package

Defines API endpoint contracts, documentation standards,
and communication protocols for the VP-MAS system.

Requirements: 9.4, 9.5
"""

from .contracts import APIContracts
from .documentation import APIDocumentationStandards
from .endpoints import AgentEndpoints, endpoint_registry

__all__ = [
    'APIContracts',
    'APIDocumentationStandards', 
    'AgentEndpoints',
    'endpoint_registry'
]