"""
FinPilot Custom Exception Hierarchy

Provides structured, traceable exceptions for all system components.
"""

from typing import Optional, Dict, Any


class FinPilotException(Exception):
    """
    Base exception for all FinPilot errors.

    All custom exceptions inherit from this base class to enable
    centralized exception handling and logging.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FinPilot exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for API responses
            details: Additional error context and debugging information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "details": self.details
        }


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(FinPilotException):
    """Raised when input validation fails"""
    pass


class ConstraintViolationError(ValidationError):
    """Raised when financial constraints are violated"""
    pass


class InvalidGoalError(ValidationError):
    """Raised when a financial goal is invalid or malformed"""
    pass


class InvalidPlanError(ValidationError):
    """Raised when a financial plan is invalid"""
    pass


# ============================================================================
# Agent Errors
# ============================================================================

class AgentError(FinPilotException):
    """Base exception for agent-related errors"""
    pass


class AgentInitializationError(AgentError):
    """Raised when agent initialization fails"""
    pass


class AgentCommunicationError(AgentError):
    """Raised when inter-agent communication fails"""
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent operation times out"""
    pass


class AgentNotAvailableError(AgentError):
    """Raised when required agent is not available"""
    pass


# ============================================================================
# Planning Errors
# ============================================================================

class PlanningError(FinPilotException):
    """Base exception for planning-related errors"""
    pass


class GoalDecompositionError(PlanningError):
    """Raised when goal decomposition fails"""
    pass


class OptimizationError(PlanningError):
    """Raised when optimization algorithms fail to converge"""
    pass


class InfeasiblePlanError(PlanningError):
    """Raised when no feasible plan can be generated"""
    pass


class SearchSpaceExhaustedError(PlanningError):
    """Raised when search space is exhausted without finding solution"""
    pass


# ============================================================================
# Verification Errors
# ============================================================================

class VerificationError(FinPilotException):
    """Base exception for verification-related errors"""
    pass


class ConstraintCheckError(VerificationError):
    """Raised when constraint verification fails"""
    pass


class ComplianceError(VerificationError):
    """Raised when regulatory compliance check fails"""
    pass


class RiskAssessmentError(VerificationError):
    """Raised when risk assessment fails"""
    pass


# ============================================================================
# External Service Errors
# ============================================================================

class ExternalServiceError(FinPilotException):
    """Base exception for external service errors"""
    pass


class APIError(ExternalServiceError):
    """Raised when external API call fails"""
    pass


class MarketDataError(ExternalServiceError):
    """Raised when market data retrieval fails"""
    pass


class LLMServiceError(ExternalServiceError):
    """Raised when LLM service (Ollama/NVIDIA NIM) fails"""
    pass


class DatabaseError(ExternalServiceError):
    """Raised when database operations fail"""
    pass


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(FinPilotException):
    """Raised when configuration is invalid or missing"""
    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid"""
    pass


# ============================================================================
# Authentication & Authorization Errors
# ============================================================================

class AuthenticationError(FinPilotException):
    """Raised when authentication fails"""
    pass


class AuthorizationError(FinPilotException):
    """Raised when authorization/permissions check fails"""
    pass


class RateLimitExceededError(FinPilotException):
    """Raised when API rate limit is exceeded"""
    pass


# ============================================================================
# Data Errors
# ============================================================================

class DataError(FinPilotException):
    """Base exception for data-related errors"""
    pass


class DataNotFoundError(DataError):
    """Raised when requested data is not found"""
    pass


class DataCorruptionError(DataError):
    """Raised when data is corrupted or inconsistent"""
    pass


class SerializationError(DataError):
    """Raised when data serialization/deserialization fails"""
    pass


# ============================================================================
# System Errors
# ============================================================================

class SystemError(FinPilotException):
    """Base exception for system-level errors"""
    pass


class ResourceExhaustedError(SystemError):
    """Raised when system resources are exhausted"""
    pass


class CircuitBreakerOpenError(SystemError):
    """Raised when circuit breaker is open"""
    pass


class MaintenanceModeError(SystemError):
    """Raised when system is in maintenance mode"""
    pass


# ============================================================================
# Utility Functions
# ============================================================================

def handle_exception(exc: Exception) -> Dict[str, Any]:
    """
    Convert any exception to standardized error dictionary.

    Args:
        exc: Exception to handle

    Returns:
        Standardized error dictionary
    """
    if isinstance(exc, FinPilotException):
        return exc.to_dict()

    # Handle standard Python exceptions
    return {
        "error": str(exc),
        "error_code": "INTERNAL_ERROR",
        "error_type": exc.__class__.__name__,
        "details": {}
    }


def is_retryable(exc: Exception) -> bool:
    """
    Determine if exception indicates a retryable error.

    Args:
        exc: Exception to check

    Returns:
        True if error is retryable, False otherwise
    """
    retryable_types = (
        AgentTimeoutError,
        AgentCommunicationError,
        APIError,
        MarketDataError,
        DatabaseError,
        ResourceExhaustedError,
    )

    return isinstance(exc, retryable_types)
