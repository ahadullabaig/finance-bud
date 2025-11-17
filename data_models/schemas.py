"""
FinPilot Multi-Agent System - Comprehensive Pydantic Data Contracts

This module defines all data models for inter-agent communication, financial planning,
market data, and system operations. All models include comprehensive validation,
documentation, and support for advanced features like correlation tracking,
performance metrics, and regulatory compliance.

Requirements covered: 6.1, 6.2, 9.1, 9.3, 28.1, 28.4, 28.5
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import uuid4

# Supabase integration
try:
    from ..supabase.database import db
except ImportError:
    db = None


# ============================================================================
# ENUMS AND CONSTANTS
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
# CORE COMMUNICATION MODELS

class PerformanceMetrics(BaseModel):
    """
    Performance metrics for tracking agent and system performance.
    Used for monitoring, optimization, and debugging purposes.
    """
    execution_time: float = Field(..., description="Execution time in seconds")
    memory_usage: float = Field(..., description="Memory usage in MB")
    api_calls: int = Field(default=0, description="Number of API calls made")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    error_count: int = Field(default=0, description="Number of errors encountered")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Success rate (0-1)")
    throughput: float = Field(default=0.0, description="Operations per second")
    latency_p50: float = Field(default=0.0, description="50th percentile latency in ms")
    latency_p95: float = Field(default=0.0, description="95th percentile latency in ms")
    latency_p99: float = Field(default=0.0, description="99th percentile latency in ms")


class AgentMessage(BaseModel):
    """
    Enhanced agent message for inter-agent communication with correlation tracking,
    performance metrics, and comprehensive metadata for debugging and monitoring.
    
    Used by all agents for structured communication in the VP-MAS system.
    """
    message_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique message identifier")
    agent_id: str = Field(..., description="ID of the sending agent")
    target_agent_id: Optional[str] = Field(None, description="ID of the target agent (None for broadcast)")
    message_type: MessageType = Field(..., description="Type of message being sent")
    payload: Dict[str, Any] = Field(..., description="Message payload data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message creation timestamp")
    correlation_id: str = Field(..., description="Correlation ID for tracking related messages")
    session_id: str = Field(..., description="Session ID for user interaction tracking")
    priority: Priority = Field(default=Priority.MEDIUM, description="Message priority level")
    trace_id: str = Field(..., description="Distributed tracing identifier")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics for this message")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    expires_at: Optional[datetime] = Field(None, description="Message expiration timestamp")


class EnhancedPlanRequest(BaseModel):
    """
    Enhanced planning request with comprehensive context, constraints, and metadata.
    Used by the Orchestration Agent to request financial plans from the Planning Agent.
    
    Supports complex multi-constraint scenarios, regulatory requirements, and tax optimization.
    """
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique request identifier")
    user_id: str = Field(..., description="User identifier for the planning request")
    user_goal: str = Field(..., description="Natural language description of the financial goal")
    current_state: Dict[str, Any] = Field(..., description="Current financial state of the user")
    constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Financial and regulatory constraints")
    trigger_type: Optional[str] = Field(None, description="Type of trigger that initiated this request")
    scenario_context: Optional[Dict[str, Any]] = Field(None, description="Additional scenario context")
    risk_profile: Dict[str, Any] = Field(..., description="User's risk tolerance and preferences")
    regulatory_requirements: List[Dict[str, Any]] = Field(default_factory=list, description="Applicable regulatory requirements")
    tax_considerations: Dict[str, Any] = Field(..., description="Tax context and optimization preferences")
    time_horizon: int = Field(..., gt=0, description="Planning time horizon in months")
    optimization_preferences: Dict[str, Any] = Field(default_factory=dict, description="User optimization preferences")
    correlation_id: str = Field(..., description="Correlation ID for tracking")
    session_id: str = Field(..., description="Session ID for user interaction")
    priority: Priority = Field(default=Priority.MEDIUM, description="Request priority")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Request creation timestamp")
    
    @validator('time_horizon')
    def validate_time_horizon(cls, v):
        if v > 600:  # 50 years
            raise ValueError('Time horizon cannot exceed 600 months (50 years)')
        return v


class PlanStep(BaseModel):
    """Individual step in a financial plan with detailed metadata"""
    step_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique step identifier")
    sequence_number: int = Field(..., ge=1, description="Order of this step in the plan")
    action_type: str = Field(..., description="Type of financial action")
    description: str = Field(..., description="Human-readable description of the step")
    amount: Decimal = Field(..., description="Monetary amount for this step")
    target_date: datetime = Field(..., description="Target execution date")
    rationale: str = Field(..., description="Explanation of why this step is recommended")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    risk_level: str = Field(..., description="Risk level associated with this step")


class VerificationReport(BaseModel):
    """Comprehensive verification report with detailed analysis"""
    report_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique report identifier")
    plan_id: str = Field(..., description="ID of the plan being verified")
    verification_status: VerificationStatus = Field(..., description="Overall verification status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Verification completion timestamp")
    constraints_checked: int = Field(..., description="Total number of constraints checked")
    constraints_passed: int = Field(..., description="Number of constraints that passed")
    constraint_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed constraint violations")
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    approval_rationale: str = Field(..., description="Rationale for approval/rejection decision")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in verification results")
    verification_time: float = Field(..., description="Time taken for verification in seconds")
    verifier_agent_id: str = Field(..., description="ID of the agent that performed verification")
    correlation_id: str = Field(..., description="Correlation ID for tracking")


# Market Data Models
class SeverityLevel(str, Enum):
    """Severity levels for market events and triggers"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MarketEventType(str, Enum):
    """Types of market events"""
    VOLATILITY_SPIKE = "volatility_spike"
    MARKET_CRASH = "market_crash"
    MARKET_RECOVERY = "market_recovery"
    SECTOR_ROTATION = "sector_rotation"
    INTEREST_RATE_CHANGE = "interest_rate_change"
    REGULATORY_CHANGE = "regulatory_change"


class MarketData(BaseModel):
    """Comprehensive market data with predictive indicators"""
    data_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique data identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data collection timestamp")
    source: str = Field(..., description="Data source")
    market_volatility: float = Field(..., ge=0.0, description="Current market volatility index")
    interest_rates: Dict[str, float] = Field(..., description="Interest rates by type")
    sector_trends: Dict[str, float] = Field(..., description="Sector performance trends")
    economic_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Economic sentiment score")
    collection_method: str = Field(..., description="Method used to collect this data")
    refresh_frequency: int = Field(..., description="Data refresh frequency in seconds")


class TriggerEvent(BaseModel):
    """Market or life event trigger with severity assessment"""
    trigger_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique trigger identifier")
    trigger_type: str = Field(..., description="Type of trigger")
    event_type: MarketEventType = Field(..., description="Specific event classification")
    severity: SeverityLevel = Field(..., description="Severity level of the trigger")
    description: str = Field(..., description="Human-readable description of the trigger event")
    source_data: Dict[str, Any] = Field(..., description="Raw data that triggered the event")
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="When the trigger was detected")
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Estimated impact score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in trigger detection")
    detector_agent_id: str = Field(..., description="ID of agent that detected this trigger")
    correlation_id: str = Field(..., description="Correlation ID for tracking")


# Financial Models
class ConstraintType(str, Enum):
    """Types of financial constraints"""
    BUDGET = "budget"
    RISK = "risk"
    LIQUIDITY = "liquidity"
    REGULATORY = "regulatory"
    TAX = "tax"
    TIME = "time"
    GOAL = "goal"
    COMPLIANCE = "compliance"


class ConstraintPriority(str, Enum):
    """Priority levels for constraints"""
    MANDATORY = "mandatory"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class FinancialState(BaseModel):
    """Comprehensive financial state with regulatory compliance and tax context"""
    state_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique state identifier")
    user_id: str = Field(..., description="User identifier")
    as_of_date: datetime = Field(default_factory=datetime.utcnow, description="Date of this financial snapshot")
    total_assets: Decimal = Field(..., ge=0, description="Total asset value")
    total_liabilities: Decimal = Field(..., ge=0, description="Total liability value")
    net_worth: Decimal = Field(..., description="Net worth (assets - liabilities)")
    monthly_income: Decimal = Field(..., ge=0, description="Monthly gross income")
    monthly_expenses: Decimal = Field(..., ge=0, description="Monthly expenses")
    monthly_cash_flow: Decimal = Field(..., description="Monthly net cash flow")
    risk_tolerance: str = Field(..., description="Risk tolerance level")
    tax_filing_status: str = Field(..., description="Tax filing status")
    estimated_tax_rate: float = Field(..., ge=0.0, le=1.0, description="Estimated effective tax rate")
    
    @validator('net_worth', always=True)
    def calculate_net_worth(cls, v, values):
        if 'total_assets' in values and 'total_liabilities' in values:
            return values['total_assets'] - values['total_liabilities']
        return v
    
    @validator('monthly_cash_flow', always=True)
    def calculate_cash_flow(cls, v, values):
        if 'monthly_income' in values and 'monthly_expenses' in values:
            return values['monthly_income'] - values['monthly_expenses']
        return v


class Constraint(BaseModel):
    """Advanced financial constraint with regulatory compliance and tax context"""
    constraint_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique constraint identifier")
    name: str = Field(..., description="Human-readable constraint name")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    priority: ConstraintPriority = Field(..., description="Constraint priority level")
    description: str = Field(..., description="Detailed description of the constraint")
    validation_rule: str = Field(..., description="Rule for validating this constraint")
    threshold_value: Union[float, int, str] = Field(..., description="Threshold value for constraint")
    comparison_operator: str = Field(..., description="Comparison operator")
    violation_severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM, description="Severity of violating this constraint")
    created_by: str = Field(..., description="Agent or system that created this constraint")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Constraint creation timestamp")


class ExecutionLog(BaseModel):
    """Comprehensive execution log with regulatory compliance and audit trail"""
    log_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique log entry identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Log entry timestamp")
    agent_id: str = Field(..., description="ID of the agent performing the action")
    action_type: str = Field(..., description="Type of action being performed")
    operation_name: str = Field(..., description="Name of the specific operation")
    execution_status: ExecutionStatus = Field(..., description="Status of the execution")
    input_data: Dict[str, Any] = Field(..., description="Input data for the operation")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data from the operation")
    execution_time: float = Field(..., description="Execution time in seconds")
    session_id: str = Field(..., description="Session ID for tracking")
    correlation_id: str = Field(..., description="Correlation ID for related operations")
    trace_id: str = Field(..., description="Distributed tracing identifier")


# Reasoning Models
class DecisionPoint(BaseModel):
    """Individual decision point in the reasoning process"""
    decision_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique decision identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Decision timestamp")
    decision_type: str = Field(..., description="Type of decision made")
    options_considered: List[Dict[str, Any]] = Field(..., description="Options that were considered")
    chosen_option: Dict[str, Any] = Field(..., description="Option that was chosen")
    rationale: str = Field(..., description="Reasoning behind the decision")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the decision")


class SearchPath(BaseModel):
    """Detailed search path with visualization metadata for ReasonGraph display"""
    path_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique path identifier")
    search_session_id: str = Field(..., description="ID of the search session this path belongs to")
    path_type: str = Field(..., description="Type of search path")
    sequence_steps: List[Dict[str, Any]] = Field(..., description="Ordered sequence of steps in this path")
    decision_points: List[DecisionPoint] = Field(default_factory=list, description="Key decision points along the path")
    total_cost: float = Field(..., description="Total cost/effort for this path")
    expected_value: float = Field(..., description="Expected value/benefit of this path")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score for this path")
    feasibility_score: float = Field(..., ge=0.0, le=1.0, description="Feasibility assessment score")
    combined_score: float = Field(..., description="Combined heuristic score")
    constraint_satisfaction_score: float = Field(..., ge=0.0, le=1.0, description="Overall constraint satisfaction")
    path_status: str = Field(..., description="Status of this path")
    exploration_time: float = Field(..., description="Time spent exploring this path in seconds")
    created_by_agent: str = Field(..., description="Agent that created this search path")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Path creation timestamp")


class ReasoningTrace(BaseModel):
    """Detailed reasoning trace with visualization metadata for comprehensive transparency"""
    trace_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique trace identifier")
    session_id: str = Field(..., description="Session ID this trace belongs to")
    agent_id: str = Field(..., description="ID of the agent that generated this trace")
    operation_type: str = Field(..., description="Type of operation being traced")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Trace start timestamp")
    end_time: Optional[datetime] = Field(None, description="Trace completion timestamp")
    total_duration: Optional[float] = Field(None, description="Total trace duration in seconds")
    decision_points: List[DecisionPoint] = Field(default_factory=list, description="All decision points in the trace")
    search_paths: List[str] = Field(default_factory=list, description="IDs of search paths explored")
    final_decision: str = Field(..., description="Final decision or recommendation")
    decision_rationale: str = Field(..., description="Comprehensive rationale for the final decision")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the final decision")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics for this trace")
    correlation_id: str = Field(..., description="Correlation ID for tracking related traces")


# Risk and Compliance Models
class RiskLevel(str, Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE_CONSERVATIVE = "moderate_conservative"
    MODERATE = "moderate"
    MODERATE_AGGRESSIVE = "moderate_aggressive"
    AGGRESSIVE = "aggressive"


class ComplianceLevel(str, Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    EXEMPT = "exempt"


class RiskProfile(BaseModel):
    """Comprehensive risk profile with behavioral analysis and stress testing"""
    profile_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique risk profile identifier")
    user_id: str = Field(..., description="User identifier")
    assessment_date: datetime = Field(default_factory=datetime.utcnow, description="Date of risk assessment")
    overall_risk_tolerance: RiskLevel = Field(..., description="Overall risk tolerance level")
    risk_capacity: float = Field(..., ge=0.0, le=1.0, description="Financial capacity to take risk")
    risk_perception: float = Field(..., ge=0.0, le=1.0, description="User's perception of risk")
    risk_composure: float = Field(..., ge=0.0, le=1.0, description="Ability to stay calm during volatility")
    investment_horizon: int = Field(..., gt=0, description="Investment time horizon in months")
    liquidity_needs: Dict[str, float] = Field(..., description="Liquidity needs by time period")
    volatility_comfort: float = Field(..., ge=0.0, le=1.0, description="Comfort with portfolio volatility")
    loss_tolerance: float = Field(..., ge=0.0, le=1.0, description="Maximum acceptable loss percentage")
    investment_experience: str = Field(..., description="Level of investment experience")
    financial_knowledge: float = Field(..., ge=0.0, le=1.0, description="Self-assessed financial knowledge")
    decision_making_style: str = Field(..., description="Decision-making style")
    primary_goals: List[str] = Field(..., description="Primary financial goals")
    goal_priorities: Dict[str, int] = Field(..., description="Priority ranking of goals")
    assessment_method: str = Field(..., description="Method used for risk assessment")
    next_review_date: datetime = Field(..., description="Date for next risk profile review")


class TaxContext(BaseModel):
    """Comprehensive tax context with optimization opportunities and compliance tracking"""
    context_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique tax context identifier")
    user_id: str = Field(..., description="User identifier")
    tax_year: int = Field(..., description="Tax year this context applies to")
    filing_status: str = Field(..., description="Tax filing status")
    number_of_dependents: int = Field(default=0, ge=0, description="Number of dependents")
    state_of_residence: str = Field(..., description="State of residence for tax purposes")
    estimated_agi: Decimal = Field(..., ge=0, description="Estimated Adjusted Gross Income")
    marginal_tax_rate: float = Field(..., ge=0.0, le=1.0, description="Marginal federal tax rate")
    effective_tax_rate: float = Field(..., ge=0.0, le=1.0, description="Effective federal tax rate")
    state_tax_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="State tax rate")
    standard_deduction: Decimal = Field(..., ge=0, description="Standard deduction amount")
    estimated_tax_liability: Decimal = Field(..., description="Estimated total tax liability")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class RegulatoryRequirement(BaseModel):
    """Regulatory requirement with compliance tracking and impact assessment"""
    requirement_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique requirement identifier")
    regulation_name: str = Field(..., description="Name of the regulation")
    regulatory_body: str = Field(..., description="Regulatory body that issued the requirement")
    title: str = Field(..., description="Title of the regulatory requirement")
    description: str = Field(..., description="Detailed description of the requirement")
    category: str = Field(..., description="Category of regulation")
    applicable_entities: List[str] = Field(..., description="Types of entities this requirement applies to")
    compliance_level: ComplianceLevel = Field(..., description="Required compliance level")
    mandatory_actions: List[str] = Field(..., description="Mandatory actions for compliance")
    effective_date: datetime = Field(..., description="Date when requirement becomes effective")
    business_impact: str = Field(..., description="Assessment of business impact")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update to this requirement")


class ComplianceStatus(BaseModel):
    """Comprehensive compliance status tracking with audit trail and remediation plans"""
    status_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique status identifier")
    entity_id: str = Field(..., description="ID of entity being assessed for compliance")
    requirement_id: str = Field(..., description="ID of regulatory requirement")
    compliance_level: ComplianceLevel = Field(..., description="Current compliance level")
    assessment_date: datetime = Field(default_factory=datetime.utcnow, description="Date of compliance assessment")
    assessor_id: str = Field(..., description="ID of agent or person who performed assessment")
    compliance_risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall compliance risk score")
    remediation_required: bool = Field(default=False, description="Whether remediation is required")
    next_review_date: datetime = Field(..., description="Date for next compliance review")
    monitoring_frequency: str = Field(..., description="Frequency of ongoing monitoring")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in compliance assessment")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class AuditTrail(BaseModel):
    """Comprehensive audit trail for compliance and debugging purposes"""
    audit_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique audit entry identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Audit entry timestamp")
    user_id: Optional[str] = Field(None, description="User ID associated with this audit entry")
    session_id: str = Field(..., description="Session ID for tracking user interactions")
    agent_id: str = Field(..., description="ID of agent that performed the action")
    correlation_id: str = Field(..., description="Correlation ID for tracking related actions")
    action_type: str = Field(..., description="Type of action performed")
    action_description: str = Field(..., description="Detailed description of the action")
    authorization_level: str = Field(..., description="Authorization level required/used")
    action_result: str = Field(..., description="Result of the action")
    environment: str = Field(default="production", description="Environment where action occurred")
    system_version: str = Field(default="1.0.0", description="System version at time of action")