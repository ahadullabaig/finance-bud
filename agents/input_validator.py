"""
Input Validation Framework for ML Prediction Engine

This module provides comprehensive input validation for the ML Prediction Engine
to prevent errors and ensure data integrity. It validates portfolio data structures,
parameter ranges, and message formats according to the requirements.

Requirements: 3.1, 3.2, 3.3, 3.5
"""

from typing import Dict, Any, List, Tuple, Union
import logging
from datetime import datetime


class InputValidator:
    """
    Comprehensive input validation for ML prediction engine operations.
    
    Provides validation methods for:
    - Portfolio data structure validation
    - Parameter range validation for ML models
    - Message format validation
    - Error response generation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"finpilot.agents.{self.__class__.__name__}")
    
    @staticmethod
    def validate_portfolio_data(portfolio: Union[Dict[str, Any], List, str]) -> Tuple[bool, str]:
        """
        Validate portfolio data structure to prevent 'str' object attribute errors.
        
        Args:
            portfolio: Portfolio data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(portfolio, dict):
            return False, f"Portfolio must be a dictionary, got {type(portfolio).__name__}"
        
        # Check for required fields
        if 'total_value' not in portfolio:
            return False, "Portfolio missing required field 'total_value'"
        
        # Validate total_value is numeric
        total_value = portfolio.get('total_value')
        if not isinstance(total_value, (int, float)):
            return False, f"Portfolio 'total_value' must be numeric, got {type(total_value).__name__}"
        
        if total_value < 0:
            return False, "Portfolio 'total_value' cannot be negative"
        
        # Check assets field
        assets = portfolio.get('assets', [])
        if not isinstance(assets, list):
            return False, f"Portfolio 'assets' must be a list, got {type(assets).__name__}"
        
        # Validate each asset
        for i, asset in enumerate(assets):
            if not isinstance(asset, dict):
                return False, f"Asset at index {i} must be a dictionary, got {type(asset).__name__}"
            
            # Check required asset fields
            required_fields = ['allocation']
            for field in required_fields:
                if field not in asset:
                    return False, f"Asset at index {i} missing required field '{field}'"
            
            # Validate allocation is numeric
            allocation = asset.get('allocation')
            if not isinstance(allocation, (int, float)):
                return False, f"Asset at index {i} 'allocation' must be numeric, got {type(allocation).__name__}"
            
            if allocation < 0 or allocation > 1:
                return False, f"Asset at index {i} 'allocation' must be between 0 and 1, got {allocation}"
        
        return True, ""
    
    @staticmethod
    def validate_horizon_days(horizon_days: Union[int, float, str]) -> Tuple[bool, str]:
        """
        Validate horizon days parameter to ensure positive values.
        
        Args:
            horizon_days: Horizon days parameter to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if it's numeric
        if not isinstance(horizon_days, (int, float)):
            return False, f"Horizon days must be numeric, got {type(horizon_days).__name__}"
        
        # Convert to int if float
        if isinstance(horizon_days, float):
            if not horizon_days.is_integer():
                return False, f"Horizon days must be a whole number, got {horizon_days}"
            horizon_days = int(horizon_days)
        
        # Check if positive
        if horizon_days <= 0:
            return False, f"Horizon days must be positive, got {horizon_days}"
        
        # Check reasonable upper bound (10 years)
        if horizon_days > 3650:
            return False, f"Horizon days cannot exceed 3650 (10 years), got {horizon_days}"
        
        return True, ""
    
    @staticmethod
    def validate_contamination_value(contamination: Union[float, int, str]) -> Tuple[bool, str]:
        """
        Validate contamination parameter for sklearn IsolationForest compatibility.
        
        Args:
            contamination: Contamination parameter to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if it's numeric
        if not isinstance(contamination, (int, float)):
            return False, f"Contamination must be numeric, got {type(contamination).__name__}"
        
        # Check sklearn IsolationForest constraints: (0.0, 0.5]
        if contamination <= 0.0:
            return False, f"Contamination must be greater than 0.0, got {contamination}"
        
        if contamination > 0.5:
            return False, f"Contamination must be less than or equal to 0.5, got {contamination}"
        
        return True, ""
    
    @staticmethod
    def validate_market_data(market_data: Union[List, Dict, str]) -> Tuple[bool, str]:
        """
        Validate market data structure for anomaly detection.
        
        Args:
            market_data: Market data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(market_data, list):
            return False, f"Market data must be a list, got {type(market_data).__name__}"
        
        if len(market_data) == 0:
            return False, "Market data cannot be empty"
        
        # Need at least 10 data points for meaningful anomaly detection
        if len(market_data) < 10:
            return False, f"Market data needs at least 10 data points for anomaly detection, got {len(market_data)}"
        
        # Validate each data point
        required_fields = ['price', 'volume', 'volatility']
        for i, data_point in enumerate(market_data):
            if not isinstance(data_point, dict):
                return False, f"Market data point at index {i} must be a dictionary, got {type(data_point).__name__}"
            
            for field in required_fields:
                if field not in data_point:
                    return False, f"Market data point at index {i} missing required field '{field}'"
                
                value = data_point[field]
                if not isinstance(value, (int, float)):
                    return False, f"Market data point at index {i} field '{field}' must be numeric, got {type(value).__name__}"
        
        return True, ""
    
    @staticmethod
    def validate_num_simulations(num_simulations: Union[int, float, str]) -> Tuple[bool, str]:
        """
        Validate number of simulations parameter for Monte Carlo.
        
        Args:
            num_simulations: Number of simulations to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(num_simulations, (int, float)):
            return False, f"Number of simulations must be numeric, got {type(num_simulations).__name__}"
        
        if isinstance(num_simulations, float):
            if not num_simulations.is_integer():
                return False, f"Number of simulations must be a whole number, got {num_simulations}"
            num_simulations = int(num_simulations)
        
        if num_simulations <= 0:
            return False, f"Number of simulations must be positive, got {num_simulations}"
        
        # Reasonable upper bound to prevent performance issues
        if num_simulations > 100000:
            return False, f"Number of simulations cannot exceed 100,000, got {num_simulations}"
        
        return True, ""
    
    @staticmethod
    def validate_timeframe_days(timeframe_days: Union[int, float, str]) -> Tuple[bool, str]:
        """
        Validate timeframe days parameter for portfolio predictions.
        
        Args:
            timeframe_days: Timeframe days to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(timeframe_days, (int, float)):
            return False, f"Timeframe days must be numeric, got {type(timeframe_days).__name__}"
        
        if isinstance(timeframe_days, float):
            if not timeframe_days.is_integer():
                return False, f"Timeframe days must be a whole number, got {timeframe_days}"
            timeframe_days = int(timeframe_days)
        
        if timeframe_days <= 0:
            return False, f"Timeframe days must be positive, got {timeframe_days}"
        
        # Reasonable upper bound (50 years)
        if timeframe_days > 18250:
            return False, f"Timeframe days cannot exceed 18,250 (50 years), got {timeframe_days}"
        
        return True, ""
    
    @staticmethod
    def create_error_response(error_type: str, message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create standardized error response for validation failures.
        
        Args:
            error_type: Type of error (e.g., 'validation_error', 'parameter_error')
            message: Error message
            details: Optional additional error details
            
        Returns:
            Standardized error response dictionary
        """
        return {
            "success": False,
            "error": message,
            "error_type": error_type,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @classmethod
    def validate_all_portfolio_inputs(
        cls,
        portfolio: Any,
        timeframe_days: Any = None,
        num_simulations: Any = None
    ) -> Tuple[bool, str]:
        """
        Validate all inputs for portfolio performance prediction.
        
        Args:
            portfolio: Portfolio data to validate
            timeframe_days: Optional timeframe days to validate
            num_simulations: Optional number of simulations to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate portfolio data
        is_valid, error_msg = cls.validate_portfolio_data(portfolio)
        if not is_valid:
            return False, error_msg
        
        # Validate timeframe if provided
        if timeframe_days is not None:
            is_valid, error_msg = cls.validate_timeframe_days(timeframe_days)
            if not is_valid:
                return False, error_msg
        
        # Validate simulations if provided
        if num_simulations is not None:
            is_valid, error_msg = cls.validate_num_simulations(num_simulations)
            if not is_valid:
                return False, error_msg
        
        return True, ""
    
    @classmethod
    def validate_all_market_inputs(
        cls,
        symbol: Any = None,
        horizon_days: Any = None,
        historical_data: Any = None
    ) -> Tuple[bool, str]:
        """
        Validate all inputs for market trend prediction.
        
        Args:
            symbol: Optional symbol to validate
            horizon_days: Optional horizon days to validate
            historical_data: Optional historical data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate symbol if provided
        if symbol is not None and not isinstance(symbol, str):
            return False, f"Symbol must be a string, got {type(symbol).__name__}"
        
        # Validate horizon days if provided
        if horizon_days is not None:
            is_valid, error_msg = cls.validate_horizon_days(horizon_days)
            if not is_valid:
                return False, error_msg
        
        # Validate historical data if provided
        if historical_data is not None:
            if not isinstance(historical_data, list):
                return False, f"Historical data must be a list, got {type(historical_data).__name__}"
            
            if len(historical_data) > 0:
                # Check first few data points for structure
                for i, data_point in enumerate(historical_data[:3]):
                    if not isinstance(data_point, dict):
                        return False, f"Historical data point at index {i} must be a dictionary, got {type(data_point).__name__}"
        
        return True, ""
    
    @classmethod
    def validate_all_anomaly_inputs(
        cls,
        market_data: Any,
        contamination: Any = None
    ) -> Tuple[bool, str]:
        """
        Validate all inputs for anomaly detection.
        
        Args:
            market_data: Market data to validate
            contamination: Optional contamination parameter to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate market data
        is_valid, error_msg = cls.validate_market_data(market_data)
        if not is_valid:
            return False, error_msg
        
        # Validate contamination if provided
        if contamination is not None:
            is_valid, error_msg = cls.validate_contamination_value(contamination)
            if not is_valid:
                return False, error_msg
        
        return True, ""