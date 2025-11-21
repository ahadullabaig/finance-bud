"""
Comprehensive Logging and Monitoring Standards for FinPilot VP-MAS

Provides structured logging, performance monitoring, and audit trail
capabilities across all agents with correlation tracking and compliance support.

Requirements: 9.4, 9.5, 11.2
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, List
from uuid import uuid4
from contextlib import contextmanager
from functools import wraps

from data_models.schemas import ExecutionLog, ExecutionStatus, PerformanceMetrics


class StructuredLogger:
    """
    Structured logger with correlation tracking, performance monitoring,
    and compliance-ready audit trails for VP-MAS agents.
    """
    
    def __init__(self, name: str, agent_id: str = None):
        self.name = name
        self.agent_id = agent_id or "unknown"
        self.logger = self._setup_logger()
        self.execution_logs: List[ExecutionLog] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Set up structured logger with custom formatter"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler with structured format
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Custom formatter for structured logging
            formatter = StructuredFormatter()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler for persistent logs
            file_handler = logging.FileHandler(f"logs/{self.name}.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log(logging.INFO, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with structured data"""
        # Add agent context
        kwargs.update({
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "log_level": logging.getLevelName(level)
        })
        
        # Create structured log entry
        log_data = {
            "message": message,
            "context": kwargs
        }
        
        # Log with structured data as extra
        self.logger.log(level, message, extra={"structured_data": log_data})
    
    def log_execution(
        self,
        operation_name: str,
        status: ExecutionStatus,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any] = None,
        execution_time: float = 0.0,
        correlation_id: str = None,
        session_id: str = None,
        trace_id: str = None,
        error_details: str = None
    ) -> ExecutionLog:
        """Log execution with comprehensive audit trail"""
        
        log_entry = ExecutionLog(
            agent_id=self.agent_id,
            action_type="execution",
            operation_name=operation_name,
            execution_status=status,
            input_data=input_data,
            output_data=output_data or {},
            execution_time=execution_time,
            session_id=session_id or str(uuid4()),
            correlation_id=correlation_id or str(uuid4()),
            trace_id=trace_id or str(uuid4())
        )
        
        # Store execution log
        self.execution_logs.append(log_entry)
        
        # Log structured execution data
        self.info(
            f"Execution: {operation_name}",
            operation=operation_name,
            status=status.value,
            execution_time=execution_time,
            correlation_id=correlation_id,
            session_id=session_id,
            trace_id=trace_id,
            input_size=len(str(input_data)),
            output_size=len(str(output_data)) if output_data else 0,
            error_details=error_details
        )
        
        return log_entry
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics for monitoring"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        self.performance_metrics[operation].append(duration)
        
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration=duration,
            **metrics
        )
    
    def log_agent_communication(
        self,
        direction: str,  # "sent" or "received"
        message_type: str,
        target_agent: str,
        correlation_id: str,
        session_id: str,
        payload_size: int,
        processing_time: float = None
    ):
        """Log agent-to-agent communication"""
        self.info(
            f"Agent Communication: {direction} {message_type}",
            direction=direction,
            message_type=message_type,
            target_agent=target_agent,
            correlation_id=correlation_id,
            session_id=session_id,
            payload_size=payload_size,
            processing_time=processing_time
        )
    
    def log_market_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        impact_score: float,
        confidence_score: float,
        source_data: Dict[str, Any]
    ):
        """Log market events and triggers"""
        self.info(
            f"Market Event: {event_type}",
            event_type=event_type,
            severity=severity,
            description=description,
            impact_score=impact_score,
            confidence_score=confidence_score,
            source_data_size=len(str(source_data))
        )
    
    def log_financial_transaction(
        self,
        transaction_type: str,
        amount: float,
        asset: str,
        user_id: str,
        session_id: str,
        compliance_status: str,
        risk_score: float = None
    ):
        """Log financial transactions for audit compliance"""
        self.info(
            f"Financial Transaction: {transaction_type}",
            transaction_type=transaction_type,
            amount=amount,
            asset=asset,
            user_id=user_id,
            session_id=session_id,
            compliance_status=compliance_status,
            risk_score=risk_score,
            audit_required=True
        )
    
    def log_constraint_violation(
        self,
        constraint_name: str,
        violation_type: str,
        severity: str,
        user_id: str,
        plan_id: str,
        remediation_suggested: str = None
    ):
        """Log constraint violations for compliance monitoring"""
        self.warning(
            f"Constraint Violation: {constraint_name}",
            constraint_name=constraint_name,
            violation_type=violation_type,
            severity=severity,
            user_id=user_id,
            plan_id=plan_id,
            remediation_suggested=remediation_suggested,
            compliance_alert=True
        )
    
    def get_execution_logs(self, correlation_id: str = None) -> List[ExecutionLog]:
        """Get execution logs, optionally filtered by correlation ID"""
        if correlation_id:
            return [log for log in self.execution_logs if log.correlation_id == correlation_id]
        return self.execution_logs.copy()
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics summary"""
        summary = {}
        
        for operation, durations in self.performance_metrics.items():
            if durations:
                summary[operation] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations)
                }
        
        return summary
    
    def export_audit_trail(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Export audit trail for compliance reporting"""
        audit_trail = []
        
        for log in self.execution_logs:
            if start_date and log.timestamp < start_date:
                continue
            if end_date and log.timestamp > end_date:
                continue
            
            audit_trail.append({
                "timestamp": log.timestamp.isoformat(),
                "agent_id": log.agent_id,
                "operation": log.operation_name,
                "status": log.execution_status.value,
                "correlation_id": log.correlation_id,
                "session_id": log.session_id,
                "execution_time": log.execution_time,
                "input_data_hash": hash(str(log.input_data)),
                "output_data_hash": hash(str(log.output_data))
            })
        
        return audit_trail


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        # Base log format
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Add structured data if available
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data.get("context", {}))
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class PerformanceMonitor:
    """Performance monitoring decorator and context manager"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def monitor_execution(self, operation_name: str):
        """Decorator for monitoring function execution"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                correlation_id = kwargs.get('correlation_id', str(uuid4()))
                session_id = kwargs.get('session_id', str(uuid4()))
                
                try:
                    # Log execution start
                    self.logger.log_execution(
                        operation_name=operation_name,
                        status=ExecutionStatus.IN_PROGRESS,
                        input_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                        correlation_id=correlation_id,
                        session_id=session_id
                    )
                    
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Log successful execution
                    self.logger.log_execution(
                        operation_name=operation_name,
                        status=ExecutionStatus.COMPLETED,
                        input_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                        output_data={"result_type": type(result).__name__},
                        execution_time=execution_time,
                        correlation_id=correlation_id,
                        session_id=session_id
                    )
                    
                    # Log performance metrics
                    self.logger.log_performance(operation_name, execution_time)
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    # Log failed execution
                    self.logger.log_execution(
                        operation_name=operation_name,
                        status=ExecutionStatus.FAILED,
                        input_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                        output_data={"error": str(e)},
                        execution_time=execution_time,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        error_details=str(e)
                    )
                    
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                correlation_id = kwargs.get('correlation_id', str(uuid4()))
                session_id = kwargs.get('session_id', str(uuid4()))
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    self.logger.log_execution(
                        operation_name=operation_name,
                        status=ExecutionStatus.COMPLETED,
                        input_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                        output_data={"result_type": type(result).__name__},
                        execution_time=execution_time,
                        correlation_id=correlation_id,
                        session_id=session_id
                    )
                    
                    self.logger.log_performance(operation_name, execution_time)
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    self.logger.log_execution(
                        operation_name=operation_name,
                        status=ExecutionStatus.FAILED,
                        input_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                        execution_time=execution_time,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        error_details=str(e)
                    )
                    
                    raise
            
            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @contextmanager
    def monitor_context(self, operation_name: str, correlation_id: str = None, session_id: str = None):
        """Context manager for monitoring code blocks"""
        start_time = time.time()
        correlation_id = correlation_id or str(uuid4())
        session_id = session_id or str(uuid4())
        
        try:
            self.logger.log_execution(
                operation_name=operation_name,
                status=ExecutionStatus.IN_PROGRESS,
                input_data={"context": "started"},
                correlation_id=correlation_id,
                session_id=session_id
            )
            
            yield
            
            execution_time = time.time() - start_time
            self.logger.log_execution(
                operation_name=operation_name,
                status=ExecutionStatus.COMPLETED,
                input_data={"context": "started"},
                output_data={"context": "completed"},
                execution_time=execution_time,
                correlation_id=correlation_id,
                session_id=session_id
            )
            
            self.logger.log_performance(operation_name, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_execution(
                operation_name=operation_name,
                status=ExecutionStatus.FAILED,
                input_data={"context": "started"},
                execution_time=execution_time,
                correlation_id=correlation_id,
                session_id=session_id,
                error_details=str(e)
            )
            raise


class LoggingStandards:
    """
    Centralized logging standards and utilities for VP-MAS system.
    
    Provides consistent logging patterns, correlation tracking,
    and compliance-ready audit trails across all agents.
    """
    
    @staticmethod
    def create_agent_logger(agent_id: str, agent_type: str) -> StructuredLogger:
        """Create standardized logger for an agent"""
        logger_name = f"finpilot.agents.{agent_type}.{agent_id}"
        return StructuredLogger(logger_name, agent_id)
    
    @staticmethod
    def create_system_logger(component: str) -> StructuredLogger:
        """Create standardized logger for system components"""
        logger_name = f"finpilot.system.{component}"
        return StructuredLogger(logger_name, f"system_{component}")
    
    @staticmethod
    def setup_log_directories():
        """Set up log directory structure"""
        import os
        
        log_dirs = [
            "logs",
            "logs/agents",
            "logs/system",
            "logs/audit",
            "logs/performance"
        ]
        
        for log_dir in log_dirs:
            os.makedirs(log_dir, exist_ok=True)
    
    @staticmethod
    def get_correlation_context(correlation_id: str = None, session_id: str = None) -> Dict[str, str]:
        """Get correlation context for logging"""
        return {
            "correlation_id": correlation_id or str(uuid4()),
            "session_id": session_id or str(uuid4()),
            "trace_id": str(uuid4())
        }


# Global logger instances for common use
system_logger = LoggingStandards.create_system_logger("core")
communication_logger = LoggingStandards.create_system_logger("communication")
performance_logger = LoggingStandards.create_system_logger("performance")

# Set up log directories on import
LoggingStandards.setup_log_directories()