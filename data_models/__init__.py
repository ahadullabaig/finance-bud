# FinPilot Multi-Agent System Data Models
# Comprehensive Pydantic data contracts for agent communication

# Import enums from dedicated module (better organization)
from .enums import (
    AgentType, MessageType, Priority, ExecutionStatus, VerificationStatus,
    SeverityLevel, MarketEventType,
    ConstraintType, ConstraintPriority,
    RiskLevel, ComplianceLevel,
    LifeEventType, WorkflowState, UrgencyLevel
)

# Import all models
from .schemas import *

__all__ = [
    # Enums (extracted to enums.py for better organization)
    "AgentType", "MessageType", "Priority", "ExecutionStatus", "VerificationStatus",
    "SeverityLevel", "MarketEventType",
    "ConstraintType", "ConstraintPriority",
    "RiskLevel", "ComplianceLevel",
    "LifeEventType", "WorkflowState", "UrgencyLevel",

    # Core Communication Models
    "AgentMessage",
    "EnhancedPlanRequest",
    "PlanStep",
    "VerificationReport",

    # Market and Trigger Models
    "MarketData",
    "TriggerEvent",
    "ComplexMarketTrigger",

    # Financial Models
    "FinancialState",
    "Constraint",
    "ExecutionLog",

    # Reasoning and Search Models
    "ReasoningTrace",
    "SearchPath",

    # Risk and Compliance Models
    "RiskProfile",
    "TaxContext",
    "RegulatoryRequirement",
    "ComplianceStatus",

    # Performance and Metrics
    "PerformanceMetrics",
    "AuditTrail"
]