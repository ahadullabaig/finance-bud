"""
API Utilities

Shared utility functions for API endpoints.
"""

import time
from datetime import datetime
from typing import Any, Optional

from api.contracts import APIResponse, APIVersion


def get_request_id() -> str:
    """Generate a unique request ID"""
    return f"req_{int(time.time() * 1000)}"


def create_api_response(
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    error_code: Optional[str] = None,
    request_id: Optional[str] = None,
    execution_time: float = 0
) -> APIResponse:
    """
    Create standardized API response using Pydantic model.

    Args:
        success: Whether the request was successful
        data: Response data (if successful)
        error: Error message (if failed)
        error_code: Error code (if failed)
        request_id: Unique request identifier
        execution_time: Request execution time in seconds

    Returns:
        Standardized APIResponse Pydantic model
    """
    return APIResponse(
        success=success,
        data=data,
        error=error,
        error_code=error_code,
        timestamp=datetime.utcnow(),
        request_id=request_id or get_request_id(),
        execution_time=execution_time,
        api_version=APIVersion.V1
    )
