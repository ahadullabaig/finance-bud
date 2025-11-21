"""
Message Validator - ML Prediction Engine Test Fixes

Provides comprehensive validation for AgentMessage objects to ensure proper
Pydantic field compliance and graceful error handling for inter-agent communication.

Requirements: 2.1, 2.2, 2.4 - Message processing validation
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4

from data_models.schemas import AgentMessage, MessageType, Priority


class MessageValidator:
    """
    Validates and creates proper AgentMessage objects with all required Pydantic fields.
    
    Handles missing fields by providing sensible defaults and ensures complete
    AgentMessage creation for reliable inter-agent communication.
    """
    
    @staticmethod
    def validate_agent_message(message_data: dict) -> AgentMessage:
        """
        Create a valid AgentMessage from potentially incomplete message data.
        
        Args:
            message_data: Dictionary containing message data (may be incomplete)
            
        Returns:
            AgentMessage: Properly validated AgentMessage with all required fields
            
        Raises:
            ValueError: If critical required fields are missing and cannot be defaulted
        """
        # Extract and validate required fields
        agent_id = message_data.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id is required and cannot be defaulted")
        
        # Handle message_type with validation
        message_type_str = message_data.get("message_type")
        if message_type_str:
            try:
                message_type = MessageType(message_type_str)
            except ValueError:
                # Default to REQUEST if invalid message type provided
                message_type = MessageType.REQUEST
        else:
            message_type = MessageType.REQUEST
        
        # Ensure payload is present (required field)
        payload = message_data.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"content": payload} if payload else {}
        
        # Generate correlation_id if missing (required field)
        correlation_id = message_data.get("correlation_id")
        if not correlation_id:
            correlation_id = str(uuid4())
        
        # Generate session_id if missing (required field)
        session_id = message_data.get("session_id")
        if not session_id:
            session_id = str(uuid4())
        
        # Generate trace_id if missing (required field)
        trace_id = message_data.get("trace_id")
        if not trace_id:
            trace_id = str(uuid4())
        
        # Handle optional fields with defaults
        target_agent_id = message_data.get("target_agent_id")
        
        # Handle priority with validation
        priority_str = message_data.get("priority")
        if priority_str:
            try:
                priority = Priority(priority_str)
            except ValueError:
                priority = Priority.MEDIUM
        else:
            priority = Priority.MEDIUM
        
        # Handle timestamp
        timestamp = message_data.get("timestamp")
        if not timestamp:
            timestamp = datetime.utcnow()
        elif isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.utcnow()
        
        # Handle other optional fields
        message_id = message_data.get("message_id", str(uuid4()))
        performance_metrics = message_data.get("performance_metrics")
        retry_count = message_data.get("retry_count", 0)
        expires_at = message_data.get("expires_at")
        
        # Create and return the validated AgentMessage
        return AgentMessage(
            message_id=message_id,
            agent_id=agent_id,
            target_agent_id=target_agent_id,
            message_type=message_type,
            payload=payload,
            timestamp=timestamp,
            correlation_id=correlation_id,
            session_id=session_id,
            priority=priority,
            trace_id=trace_id,
            performance_metrics=performance_metrics,
            retry_count=retry_count,
            expires_at=expires_at
        )
    
    @staticmethod
    def create_error_response(
        error_type: str, 
        message: str, 
        original_message: Optional[AgentMessage] = None,
        agent_id: str = "ml-prediction-engine"
    ) -> Dict[str, Any]:
        """
        Create a standardized error response for validation failures.
        
        Args:
            error_type: Type of error (e.g., "validation_error", "missing_field")
            message: Human-readable error message
            original_message: Original message that caused the error (if available)
            agent_id: ID of the agent creating the error response
            
        Returns:
            Dict containing standardized error response
        """
        error_response = {
            "success": False,
            "error": message,
            "error_type": error_type,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id
        }
        
        # Include correlation information if original message is available
        if original_message:
            error_response.update({
                "correlation_id": original_message.correlation_id,
                "session_id": original_message.session_id,
                "original_message_id": original_message.message_id
            })
        
        return error_response
    
    @staticmethod
    def validate_message_content(content: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate message content structure and required fields.
        
        Args:
            content: Message content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(content, dict):
            return False, "Message content must be a dictionary"
        
        # Check for action field in request messages
        if "action" in content:
            action = content["action"]
            if not isinstance(action, str) or not action.strip():
                return False, "Action must be a non-empty string"
        
        return True, ""
    
    @staticmethod
    def create_response_message(
        original_message: AgentMessage,
        response_content: Dict[str, Any],
        responding_agent_id: str
    ) -> AgentMessage:
        """
        Create a proper response message from an original request.
        
        Args:
            original_message: The original request message
            response_content: Content for the response
            responding_agent_id: ID of the agent sending the response
            
        Returns:
            AgentMessage: Properly formatted response message
        """
        return AgentMessage(
            agent_id=responding_agent_id,
            target_agent_id=original_message.agent_id,
            message_type=MessageType.RESPONSE,
            payload=response_content,
            correlation_id=original_message.correlation_id,
            session_id=original_message.session_id,
            trace_id=original_message.trace_id,
            priority=original_message.priority
        )
    
    @staticmethod
    def handle_unknown_action(
        action: str,
        original_message: AgentMessage,
        responding_agent_id: str
    ) -> Dict[str, Any]:
        """
        Create appropriate error response for unknown actions.
        
        Args:
            action: The unknown action that was requested
            original_message: The original message containing the unknown action
            responding_agent_id: ID of the agent handling the request
            
        Returns:
            Dict containing error response for unknown action
        """
        return MessageValidator.create_error_response(
            error_type="unknown_action",
            message=f"Unknown action '{action}'. Supported actions: predict_market, predict_portfolio, detect_anomaly, generate_recommendations, predict_risk",
            original_message=original_message,
            agent_id=responding_agent_id
        )