"""
API Contracts for FinPilot VP-MAS

Defines standardized API contracts for all agent interactions,
including request/response schemas, error handling, and validation rules.

Requirements: 9.4, 9.5
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from data_models.schemas import (
    AgentMessage, MessageType, Priority, ExecutionStatus,
    EnhancedPlanRequest, VerificationReport, MarketData, TriggerEvent
)


class APIVersion(str, Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"


class HTTPMethod(str, Enum):
    """HTTP methods for API endpoints"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIResponse(BaseModel):
    """Standardized API response format"""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data payload")
    error: Optional[str] = Field(None, description="Error message if request failed")
    error_code: Optional[str] = Field(None, description="Specific error code for debugging")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str = Field(..., description="Unique request identifier for tracking")
    execution_time: float = Field(..., description="Request execution time in seconds")
    api_version: APIVersion = Field(default=APIVersion.V1, description="API version used")


class APIError(BaseModel):
    """Standardized API error format"""
    error_code: str = Field(..., description="Unique error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: str = Field(..., description="Request ID that caused the error")
    suggested_action: Optional[str] = Field(None, description="Suggested action to resolve error")


class PaginationParams(BaseModel):
    """Standardized pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field(default="asc", pattern="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseModel):
    """Standardized paginated response format"""
    items: List[Dict[str, Any]] = Field(..., description="List of items for current page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    current_page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class APIContracts:
    """
    Defines API contracts for all VP-MAS agent interactions.
    
    Provides standardized request/response formats, validation rules,
    and documentation for all agent endpoints.
    """
    
    # Orchestration Agent Contracts
    ORCHESTRATION_CONTRACTS = {
        "submit_goal": {
            "method": HTTPMethod.POST,
            "path": "/api/v1/orchestration/goals",
            "description": "Submit a financial goal for processing",
            "request_schema": {
                "type": "object",
                "required": ["user_goal", "user_id"],
                "properties": {
                    "user_goal": {
                        "type": "string",
                        "description": "Natural language financial goal",
                        "example": "Save $100,000 for retirement in 10 years"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Unique user identifier",
                        "example": "user_12345"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "default": "medium",
                        "description": "Goal processing priority"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for goal processing"
                    }
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Unique workflow identifier"
                    },
                    "session_id": {
                        "type": "string", 
                        "description": "Session identifier for tracking"
                    },
                    "estimated_completion": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Estimated completion time"
                    },
                    "workflow_steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent": {"type": "string"},
                                "action": {"type": "string"},
                                "estimated_duration": {"type": "number"}
                            }
                        }
                    }
                }
            },
            "error_codes": {
                "INVALID_GOAL": "Goal format is invalid or cannot be parsed",
                "USER_NOT_FOUND": "Specified user ID does not exist",
                "WORKFLOW_CREATION_FAILED": "Failed to create workflow",
                "RATE_LIMIT_EXCEEDED": "Too many requests from user"
            }
        },
        
        "get_workflow_status": {
            "method": HTTPMethod.GET,
            "path": "/api/v1/orchestration/workflows/{workflow_id}",
            "description": "Get status of a workflow",
            "path_params": {
                "workflow_id": {
                    "type": "string",
                    "description": "Unique workflow identifier"
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed", "cancelled"]
                    },
                    "progress": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Completion progress (0-1)"
                    },
                    "current_step": {"type": "string"},
                    "steps_completed": {"type": "integer"},
                    "total_steps": {"type": "integer"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"}
                }
            }
        }
    }
    
    # Planning Agent Contracts
    PLANNING_CONTRACTS = {
        "generate_plan": {
            "method": HTTPMethod.POST,
            "path": "/api/v1/planning/generate",
            "description": "Generate financial plan using advanced algorithms",
            "request_schema": {
                "type": "object",
                "required": ["planning_request"],
                "properties": {
                    "planning_request": {
                        "$ref": "#/components/schemas/EnhancedPlanRequest"
                    },
                    "algorithm_preferences": {
                        "type": "object",
                        "properties": {
                            "search_depth": {"type": "integer", "minimum": 1, "maximum": 10},
                            "max_strategies": {"type": "integer", "minimum": 3, "maximum": 10},
                            "optimization_focus": {
                                "type": "string",
                                "enum": ["return", "risk", "tax_efficiency", "balanced"]
                            }
                        }
                    }
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "selected_strategy": {"type": "string"},
                    "plan_steps": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/PlanStep"}
                    },
                    "search_paths": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/SearchPath"}
                    },
                    "reasoning_trace": {"$ref": "#/components/schemas/ReasoningTrace"},
                    "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "alternative_strategies": {"type": "integer"},
                    "processing_time": {"type": "number"}
                }
            }
        },
        
        "get_search_paths": {
            "method": HTTPMethod.GET,
            "path": "/api/v1/planning/search-paths/{session_id}",
            "description": "Get detailed search paths for a planning session",
            "path_params": {
                "session_id": {"type": "string", "description": "Planning session ID"}
            },
            "query_params": {
                "include_pruned": {"type": "boolean", "default": False},
                "strategy_filter": {"type": "string", "enum": ["conservative", "balanced", "aggressive"]}
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "search_paths": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/SearchPath"}
                    },
                    "total_paths_explored": {"type": "integer"},
                    "paths_pruned": {"type": "integer"},
                    "exploration_time": {"type": "number"}
                }
            }
        }
    }
    
    # Information Retrieval Agent Contracts
    IRA_CONTRACTS = {
        "get_market_data": {
            "method": HTTPMethod.GET,
            "path": "/api/v1/market/data",
            "description": "Get comprehensive market data",
            "query_params": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbols to fetch data for"
                },
                "data_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["prices", "volatility", "trends", "sentiment"]},
                    "default": ["prices", "volatility"]
                },
                "refresh": {"type": "boolean", "default": False, "description": "Force refresh from source"}
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "market_data": {"$ref": "#/components/schemas/MarketData"},
                    "data_quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "collection_time": {"type": "number"},
                    "data_sources": {"type": "array", "items": {"type": "string"}},
                    "cache_status": {"type": "string", "enum": ["hit", "miss", "refresh"]}
                }
            }
        },
        
        "detect_triggers": {
            "method": HTTPMethod.POST,
            "path": "/api/v1/market/triggers/detect",
            "description": "Detect market triggers and events",
            "request_schema": {
                "type": "object",
                "properties": {
                    "monitoring_config": {
                        "type": "object",
                        "properties": {
                            "volatility_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                            "sentiment_threshold": {"type": "number", "minimum": -1, "maximum": 1},
                            "sector_change_threshold": {"type": "number"}
                        }
                    },
                    "user_preferences": {
                        "type": "object",
                        "properties": {
                            "risk_tolerance": {"type": "string", "enum": ["low", "medium", "high"]},
                            "notification_frequency": {"type": "string", "enum": ["immediate", "hourly", "daily"]}
                        }
                    }
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "triggers_detected": {"type": "integer"},
                    "triggers": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/TriggerEvent"}
                    },
                    "monitoring_active": {"type": "boolean"},
                    "next_check": {"type": "string", "format": "date-time"}
                }
            }
        }
    }
    
    # Verification Agent Contracts
    VERIFICATION_CONTRACTS = {
        "verify_plan": {
            "method": HTTPMethod.POST,
            "path": "/api/v1/verification/verify",
            "description": "Verify financial plan against constraints",
            "request_schema": {
                "type": "object",
                "required": ["plan_id", "plan_steps"],
                "properties": {
                    "plan_id": {"type": "string"},
                    "plan_steps": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/PlanStep"}
                    },
                    "verification_level": {
                        "type": "string",
                        "enum": ["basic", "comprehensive", "regulatory"],
                        "default": "comprehensive"
                    },
                    "custom_constraints": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Constraint"}
                    }
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "verification_report": {"$ref": "#/components/schemas/VerificationReport"},
                    "step_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_id": {"type": "string"},
                                "violations": {"type": "array"},
                                "compliance_score": {"type": "number"},
                                "verification_time": {"type": "number"}
                            }
                        }
                    },
                    "recommendations": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        
        "start_cmvl": {
            "method": HTTPMethod.POST,
            "path": "/api/v1/verification/cmvl/start",
            "description": "Start Continuous Monitoring and Verification Loop",
            "request_schema": {
                "type": "object",
                "required": ["trigger_event"],
                "properties": {
                    "trigger_event": {"$ref": "#/components/schemas/TriggerEvent"},
                    "monitoring_config": {
                        "type": "object",
                        "properties": {
                            "frequency": {"type": "string", "enum": ["real_time", "5_minutes", "hourly"]},
                            "auto_remediation": {"type": "boolean", "default": True},
                            "escalation_threshold": {"type": "string", "enum": ["medium", "high", "critical"]}
                        }
                    }
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "cmvl_id": {"type": "string"},
                    "cmvl_activated": {"type": "boolean"},
                    "monitoring_frequency": {"type": "string"},
                    "verification_actions": {"type": "array", "items": {"type": "string"}},
                    "estimated_completion": {"type": "string", "format": "date-time"},
                    "auto_remediation": {"type": "boolean"}
                }
            }
        }
    }
    
    # Execution Agent Contracts
    EXECUTION_CONTRACTS = {
        "execute_plan": {
            "method": HTTPMethod.POST,
            "path": "/api/v1/execution/execute",
            "description": "Execute approved financial plan",
            "request_schema": {
                "type": "object",
                "required": ["plan_id", "plan_steps"],
                "properties": {
                    "plan_id": {"type": "string"},
                    "plan_steps": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/PlanStep"}
                    },
                    "execution_mode": {
                        "type": "string",
                        "enum": ["simulation", "live", "staged"],
                        "default": "simulation"
                    },
                    "confirmation_required": {"type": "boolean", "default": True}
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "execution_id": {"type": "string"},
                    "execution_completed": {"type": "boolean"},
                    "steps_executed": {"type": "integer"},
                    "total_steps": {"type": "integer"},
                    "execution_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_id": {"type": "string"},
                                "status": {"type": "string", "enum": ["completed", "failed", "pending"]},
                                "transaction_id": {"type": "string"},
                                "amount_executed": {"type": "number"},
                                "fees": {"type": "number"},
                                "execution_time": {"type": "number"}
                            }
                        }
                    },
                    "portfolio_updated": {"type": "boolean"},
                    "execution_time": {"type": "number"}
                }
            }
        },
        
        "get_portfolio": {
            "method": HTTPMethod.GET,
            "path": "/api/v1/execution/portfolio/{user_id}",
            "description": "Get current portfolio state",
            "path_params": {
                "user_id": {"type": "string", "description": "User identifier"}
            },
            "query_params": {
                "include_history": {"type": "boolean", "default": False},
                "date_range": {"type": "string", "description": "Date range for historical data"}
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string"},
                    "user_id": {"type": "string"},
                    "current_value": {"type": "number"},
                    "asset_allocation": {
                        "type": "object",
                        "additionalProperties": {"type": "number"}
                    },
                    "performance_metrics": {
                        "type": "object",
                        "properties": {
                            "total_return": {"type": "number"},
                            "ytd_return": {"type": "number"},
                            "volatility": {"type": "number"},
                            "sharpe_ratio": {"type": "number"}
                        }
                    },
                    "last_updated": {"type": "string", "format": "date-time"}
                }
            }
        }
    }
    
    @classmethod
    def get_all_contracts(cls) -> Dict[str, Dict]:
        """Get all API contracts for documentation generation"""
        return {
            "orchestration": cls.ORCHESTRATION_CONTRACTS,
            "planning": cls.PLANNING_CONTRACTS,
            "information_retrieval": cls.IRA_CONTRACTS,
            "verification": cls.VERIFICATION_CONTRACTS,
            "execution": cls.EXECUTION_CONTRACTS
        }
    
    @classmethod
    def validate_contract(cls, agent_type: str, endpoint: str, data: Dict) -> bool:
        """Validate data against contract schema"""
        contracts = cls.get_all_contracts()
        
        if agent_type not in contracts:
            return False
        
        if endpoint not in contracts[agent_type]:
            return False
        
        # In a real implementation, this would use jsonschema validation
        # For now, return True as a placeholder
        return True
    
    @classmethod
    def get_error_codes(cls, agent_type: str, endpoint: str) -> Dict[str, str]:
        """Get error codes for a specific endpoint"""
        contracts = cls.get_all_contracts()
        
        if agent_type in contracts and endpoint in contracts[agent_type]:
            return contracts[agent_type][endpoint].get("error_codes", {})
        
        return {}


# Common error codes used across all agents
COMMON_ERROR_CODES = {
    "INVALID_REQUEST": "Request format is invalid",
    "MISSING_REQUIRED_FIELD": "Required field is missing from request",
    "INVALID_FIELD_VALUE": "Field value is invalid or out of range",
    "AUTHENTICATION_FAILED": "Authentication credentials are invalid",
    "AUTHORIZATION_FAILED": "User is not authorized for this operation",
    "RATE_LIMIT_EXCEEDED": "Request rate limit exceeded",
    "INTERNAL_SERVER_ERROR": "Internal server error occurred",
    "SERVICE_UNAVAILABLE": "Service is temporarily unavailable",
    "TIMEOUT": "Request timed out",
    "VALIDATION_ERROR": "Data validation failed",
    "RESOURCE_NOT_FOUND": "Requested resource was not found",
    "CONFLICT": "Request conflicts with current resource state",
    "PRECONDITION_FAILED": "Request precondition was not met"
}


# HTTP status code mappings
HTTP_STATUS_CODES = {
    "success": 200,
    "created": 201,
    "accepted": 202,
    "no_content": 204,
    "bad_request": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "not_found": 404,
    "method_not_allowed": 405,
    "conflict": 409,
    "precondition_failed": 412,
    "unprocessable_entity": 422,
    "too_many_requests": 429,
    "internal_server_error": 500,
    "service_unavailable": 503,
    "gateway_timeout": 504
}