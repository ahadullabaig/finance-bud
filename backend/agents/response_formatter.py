"""
Response Formatter for ML Prediction Engine

Provides standardized response formatting across all prediction methods
to ensure consistent structure and backward compatibility.

Requirements: Phase 6, Task 25 - ML Prediction Engine Test Fixes
"""

from typing import Dict, Any, List, Optional
import logging


class ResponseFormatter:
    """
    Standardizes response formats for ML prediction engine methods.
    
    Ensures consistent structure across different prediction types while
    maintaining backward compatibility with existing response formats.
    """
    
    @staticmethod
    def format_portfolio_response(
        current_value: float,
        predictions: Dict[str, Any],
        risk_metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format portfolio prediction response with standardized structure.
        
        Args:
            current_value: Current portfolio value
            predictions: Prediction results dictionary
            risk_metrics: Optional risk metrics
            metadata: Optional metadata (timeframe, simulations, etc.)
            error: Optional error message
            
        Returns:
            Standardized portfolio response with top-level "expected_value" field
        """
        # Extract expected_value from predictions for top-level placement
        expected_value = predictions.get("expected_value", current_value)
        
        response = {
            "success": error is None,
            "current_value": current_value,
            "expected_value": expected_value,  # Top-level field as required
            "predictions": predictions,  # Maintain nested structure for backward compatibility
        }
        
        # Add optional sections
        if risk_metrics:
            response["risk_metrics"] = risk_metrics
            
        if metadata:
            response.update(metadata)  # Add metadata fields directly to response
            
        if error:
            response["error"] = error
            
        return response
    
    @staticmethod
    def format_market_trend_response(
        symbol: str,
        current_price: float,
        predicted_price: float,
        trend_data: Dict[str, Any],
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format market trend prediction response.
        
        Args:
            symbol: Asset symbol
            current_price: Current market price
            predicted_price: Predicted future price
            trend_data: Additional trend analysis data
            error: Optional error message
            
        Returns:
            Standardized market trend response
        """
        response = {
            "success": error is None,
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": predicted_price,
        }
        
        # Add trend data
        response.update(trend_data)
        
        if error:
            response["error"] = error
            
        return response
    
    @staticmethod
    def format_anomaly_response(
        total_points: int,
        anomalies: List[Dict[str, Any]],
        detection_metadata: Dict[str, Any],
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format anomaly detection response.
        
        Args:
            total_points: Total number of data points analyzed
            anomalies: List of detected anomalies
            detection_metadata: Metadata about detection process
            error: Optional error message
            
        Returns:
            Standardized anomaly detection response
        """
        response = {
            "success": error is None,
            "total_points": total_points,
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / total_points if total_points > 0 else 0,
            "anomalies": anomalies,
        }
        
        # Add detection metadata
        response.update(detection_metadata)
        
        if error:
            response["error"] = error
            
        return response
    
    @staticmethod
    def format_error_response(
        error_type: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format standardized error response.
        
        Args:
            error_type: Type of error (validation_error, processing_error, etc.)
            error_message: Human-readable error message
            details: Optional additional error details
            
        Returns:
            Standardized error response
        """
        response = {
            "success": False,
            "error": error_message,
            "error_type": error_type
        }
        
        if details:
            response["details"] = details
            
        return response
    
    @staticmethod
    def validate_response_structure(
        response: Dict[str, Any],
        required_fields: List[str]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that response contains all required fields.
        
        Args:
            response: Response dictionary to validate
            required_fields: List of required field names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_fields = []
        
        for field in required_fields:
            if field not in response:
                missing_fields.append(field)
        
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            return False, error_msg
            
        return True, None
    
    @staticmethod
    def validate_portfolio_response(response: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate portfolio prediction response structure.
        
        Args:
            response: Portfolio response to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["success", "current_value", "expected_value", "predictions"]
        return ResponseFormatter.validate_response_structure(response, required_fields)
    
    @staticmethod
    def ensure_backward_compatibility(
        response: Dict[str, Any],
        legacy_format: bool = True
    ) -> Dict[str, Any]:
        """
        Ensure response maintains backward compatibility with existing code.
        
        Args:
            response: Response to modify for compatibility
            legacy_format: Whether to maintain legacy field structure
            
        Returns:
            Response with backward compatibility ensured
        """
        if not legacy_format:
            return response
            
        # Ensure nested expected_value exists in predictions for backward compatibility
        if "predictions" in response and "expected_value" in response:
            if "expected_value" not in response["predictions"]:
                response["predictions"]["expected_value"] = response["expected_value"]
                
        return response