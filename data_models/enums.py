"""
FinPilot Data Model Enums

All enum types used across the VP-MAS system, extracted for better organization.
"""

from enum import Enum


# ============================================================================
# AGENT & COMMUNICATION ENUMS
# ============================================================================

class AgentType(str, Enum):
    """Agent types in the VP-MAS system"""
    ORCHESTRATION = "orchestration"
    PLANNING = "planning"
    INFORMATION_RETRIEVAL = "information_retrieval"
    VERIFICATION = "verification"
    EXECUTION = "execution"


class MessageType(str, Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class Priority(str, Enum):
    """Message and task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExecutionStatus(str, Enum):
    """Status of execution operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationStatus(str, Enum):
    """Verification result status"""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    PENDING = "pending"


# ============================================================================
# MARKET & TRIGGER ENUMS
# ============================================================================

class SeverityLevel(str, Enum):
    """Severity levels for events and triggers"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketEventType(str, Enum):
    """Types of market events that can trigger CMVL workflow"""
    VOLATILITY_SPIKE = "volatility_spike"
    SIGNIFICANT_DROP = "significant_drop"
    SIGNIFICANT_GAIN = "significant_gain"
    CORRELATION_BREAKDOWN = "correlation_breakdown"


# ============================================================================
# FINANCIAL & CONSTRAINT ENUMS
# ============================================================================

class ConstraintType(str, Enum):
    """Types of financial planning constraints"""
    BUDGET = "budget"
    LIQUIDITY = "liquidity"
    RISK = "risk"
    TAX = "tax"
    TIME = "time"
    REGULATORY = "regulatory"
    PERSONAL = "personal"


class ConstraintPriority(str, Enum):
    """Priority levels for constraint satisfaction"""
    MANDATORY = "mandatory"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


# ============================================================================
# RISK & COMPLIANCE ENUMS
# ============================================================================

class RiskLevel(str, Enum):
    """Risk tolerance and assessment levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ComplianceLevel(str, Enum):
    """Regulatory compliance levels"""
    FULL_COMPLIANCE = "full_compliance"
    PARTIAL_COMPLIANCE = "partial_compliance"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"


# ============================================================================
# LIFE EVENTS & CMVL ENUMS
# ============================================================================

class LifeEventType(str, Enum):
    """Types of life events that may require plan adjustments"""
    JOB_LOSS = "job_loss"
    MEDICAL_EMERGENCY = "medical_emergency"
    FAMILY_CHANGE = "family_change"
    INCOME_CHANGE = "income_change"
    MAJOR_EXPENSE = "major_expense"
    BUSINESS_DISRUPTION = "business_disruption"
    MARKET_EVENT = "market_event"
    REGULATORY_CHANGE = "regulatory_change"


class WorkflowState(str, Enum):
    """States of CMVL workflow execution"""
    MONITORING = "monitoring"
    TRIGGER_DETECTED = "trigger_detected"
    TRIGGER_CLASSIFIED = "trigger_classified"
    PLANNING = "planning"
    VERIFICATION = "verification"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTION = "execution"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UrgencyLevel(str, Enum):
    """Urgency levels for triggers and adjustments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMEDIATE = "immediate"
