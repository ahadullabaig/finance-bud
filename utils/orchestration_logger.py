"""
Comprehensive Orchestration Decision Logger

Provides detailed logging for all orchestration decisions, workflow coordination,
trigger handling, and agent management activities with structured output for
debugging and audit purposes.

Requirements: 6.1, 6.2, 6.5, 11.1, 11.2, 2.1, 2.2, 10.2
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
from enum import Enum

from data_models.schemas import (
    AgentMessage, TriggerEvent, PerformanceMetrics, AuditTrail
)


class DecisionType(str, Enum):
    """Types of orchestration decisions"""
    WORKFLOW_INITIATION = "workflow_initiation"
    AGENT_SELECTION = "agent_selection"
    TASK_DELEGATION = "task_delegation"
    TRIGGER_RESPONSE = "trigger_response"
    PRIORITY_ESCALATION = "priority_escalation"
    RESOURCE_ALLOCATION = "resource_allocation"
    ERROR_RECOVERY = "error_recovery"
    CIRCUIT_BREAKER_ACTION = "circuit_breaker_action"
    LOAD_BALANCING = "load_balancing"
    CMVL_ACTIVATION = "cmvl_activation"


class DecisionOutcome(str, Enum):
    """Outcomes of orchestration decisions"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class OrchestrationDecision:
    """Represents a single orchestration decision with full context"""
    
    def __init__(
        self,
        decision_type: DecisionType,
        decision_context: Dict[str, Any],
        correlation_id: str = None,
        session_id: str = None
    ):
        self.decision_id = str(uuid4())
        self.decision_type = decision_type
        self.decision_context = decision_context
        self.correlation_id = correlation_id or str(uuid4())
        self.session_id = session_id or str(uuid4())
        
        # Decision tracking
        self.timestamp = datetime.utcnow()
        self.decision_maker = "orchestration_agent"
        self.options_considered: List[Dict[str, Any]] = []
        self.chosen_option: Optional[Dict[str, Any]] = None
        self.rationale: str = ""
        self.confidence_score: float = 0.0
        
        # Execution tracking
        self.execution_start_time: Optional[datetime] = None
        self.execution_end_time: Optional[datetime] = None
        self.outcome: Optional[DecisionOutcome] = None
        self.performance_metrics: Optional[PerformanceMetrics] = None
        
        # Impact tracking
        self.affected_agents: List[str] = []
        self.affected_workflows: List[str] = []
        self.resource_impact: Dict[str, Any] = {}
        
        # Error handling
        self.errors: List[Dict[str, Any]] = []
        self.recovery_actions: List[Dict[str, Any]] = []
    
    def add_option(self, option_name: str, option_data: Dict[str, Any], score: float = 0.0) -> None:
        """Add a considered option to the decision"""
        self.options_considered.append({
            "option_name": option_name,
            "option_data": option_data,
            "score": score,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def set_chosen_option(self, option_name: str, option_data: Dict[str, Any], rationale: str, confidence: float) -> None:
        """Set the chosen option with rationale"""
        self.chosen_option = {
            "option_name": option_name,
            "option_data": option_data,
            "chosen_at": datetime.utcnow().isoformat()
        }
        self.rationale = rationale
        self.confidence_score = confidence
    
    def start_execution(self) -> None:
        """Mark the start of decision execution"""
        self.execution_start_time = datetime.utcnow()
    
    def complete_execution(self, outcome: DecisionOutcome, metrics: PerformanceMetrics = None) -> None:
        """Mark the completion of decision execution"""
        self.execution_end_time = datetime.utcnow()
        self.outcome = outcome
        self.performance_metrics = metrics
    
    def add_error(self, error_type: str, error_message: str, error_context: Dict[str, Any] = None) -> None:
        """Add an error that occurred during decision execution"""
        self.errors.append({
            "error_id": str(uuid4()),
            "error_type": error_type,
            "error_message": error_message,
            "error_context": error_context or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_recovery_action(self, action_type: str, action_description: str, action_result: str) -> None:
        """Add a recovery action taken in response to errors"""
        self.recovery_actions.append({
            "action_id": str(uuid4()),
            "action_type": action_type,
            "action_description": action_description,
            "action_result": action_result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for logging"""
        execution_time = None
        if self.execution_start_time and self.execution_end_time:
            execution_time = (self.execution_end_time - self.execution_start_time).total_seconds()
        
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "decision_maker": self.decision_maker,
            "decision_context": self.decision_context,
            "options_considered": self.options_considered,
            "chosen_option": self.chosen_option,
            "rationale": self.rationale,
            "confidence_score": self.confidence_score,
            "execution_start_time": self.execution_start_time.isoformat() if self.execution_start_time else None,
            "execution_end_time": self.execution_end_time.isoformat() if self.execution_end_time else None,
            "execution_time_seconds": execution_time,
            "outcome": self.outcome.value if self.outcome else None,
            "performance_metrics": self.performance_metrics.dict() if self.performance_metrics else None,
            "affected_agents": self.affected_agents,
            "affected_workflows": self.affected_workflows,
            "resource_impact": self.resource_impact,
            "errors": self.errors,
            "recovery_actions": self.recovery_actions
        }


class OrchestrationLogger:
    """
    Comprehensive logger for orchestration decisions and activities.
    
    Provides structured logging with multiple output formats, filtering,
    and analysis capabilities for debugging and audit purposes.
    """
    
    def __init__(self, log_level: str = "INFO", enable_file_logging: bool = True):
        self.logger = self._setup_logger(log_level)
        self.enable_file_logging = enable_file_logging
        
        # Decision tracking
        self.active_decisions: Dict[str, OrchestrationDecision] = {}
        self.completed_decisions: List[OrchestrationDecision] = []
        self.decision_history_limit = 1000
        
        # Performance tracking
        self.decision_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "average_decision_time": 0.0,
            "decision_types": {}
        }
        
        # Audit trail
        self.audit_entries: List[AuditTrail] = []
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Set up structured logger for orchestration decisions"""
        logger = logging.getLogger("finpilot.orchestration.decisions")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for detailed logs
            if self.enable_file_logging:
                file_handler = logging.FileHandler('logs/orchestration_decisions.log')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def start_decision(
        self,
        decision_type: DecisionType,
        decision_context: Dict[str, Any],
        correlation_id: str = None,
        session_id: str = None
    ) -> str:
        """Start tracking a new orchestration decision"""
        decision = OrchestrationDecision(
            decision_type=decision_type,
            decision_context=decision_context,
            correlation_id=correlation_id,
            session_id=session_id
        )
        
        self.active_decisions[decision.decision_id] = decision
        
        self.logger.info(
            f"Started decision {decision.decision_id}: {decision_type.value}",
            extra={
                'decision_id': decision.decision_id,
                'decision_type': decision_type.value,
                'correlation_id': decision.correlation_id,
                'session_id': decision.session_id,
                'context': decision_context
            }
        )
        
        return decision.decision_id
    
    def add_decision_option(
        self,
        decision_id: str,
        option_name: str,
        option_data: Dict[str, Any],
        score: float = 0.0
    ) -> None:
        """Add a considered option to an active decision"""
        if decision_id not in self.active_decisions:
            self.logger.warning(f"Decision {decision_id} not found for adding option")
            return
        
        decision = self.active_decisions[decision_id]
        decision.add_option(option_name, option_data, score)
        
        self.logger.debug(
            f"Added option '{option_name}' to decision {decision_id} (score: {score})",
            extra={
                'decision_id': decision_id,
                'option_name': option_name,
                'option_score': score,
                'correlation_id': decision.correlation_id
            }
        )
    
    def make_decision(
        self,
        decision_id: str,
        chosen_option: str,
        option_data: Dict[str, Any],
        rationale: str,
        confidence: float,
        affected_agents: List[str] = None,
        affected_workflows: List[str] = None
    ) -> None:
        """Record the final decision choice"""
        if decision_id not in self.active_decisions:
            self.logger.warning(f"Decision {decision_id} not found for making decision")
            return
        
        decision = self.active_decisions[decision_id]
        decision.set_chosen_option(chosen_option, option_data, rationale, confidence)
        decision.affected_agents = affected_agents or []
        decision.affected_workflows = affected_workflows or []
        decision.start_execution()
        
        self.logger.info(
            f"Decision made for {decision_id}: '{chosen_option}' (confidence: {confidence:.2f})",
            extra={
                'decision_id': decision_id,
                'chosen_option': chosen_option,
                'rationale': rationale,
                'confidence_score': confidence,
                'affected_agents': affected_agents,
                'affected_workflows': affected_workflows,
                'correlation_id': decision.correlation_id
            }
        )
    
    def complete_decision(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        performance_metrics: PerformanceMetrics = None,
        resource_impact: Dict[str, Any] = None
    ) -> None:
        """Complete a decision and move it to history"""
        if decision_id not in self.active_decisions:
            self.logger.warning(f"Decision {decision_id} not found for completion")
            return
        
        decision = self.active_decisions[decision_id]
        decision.complete_execution(outcome, performance_metrics)
        decision.resource_impact = resource_impact or {}
        
        # Move to completed decisions
        self.completed_decisions.append(decision)
        del self.active_decisions[decision_id]
        
        # Update metrics
        self._update_decision_metrics(decision)
        
        # Maintain history limit
        if len(self.completed_decisions) > self.decision_history_limit:
            self.completed_decisions = self.completed_decisions[-self.decision_history_limit:]
        
        self.logger.info(
            f"Completed decision {decision_id}: {outcome.value}",
            extra={
                'decision_id': decision_id,
                'outcome': outcome.value,
                'execution_time': (decision.execution_end_time - decision.execution_start_time).total_seconds() if decision.execution_start_time and decision.execution_end_time else None,
                'correlation_id': decision.correlation_id
            }
        )
        
        # Create audit trail entry
        self._create_audit_entry(decision)
    
    def log_decision_error(
        self,
        decision_id: str,
        error_type: str,
        error_message: str,
        error_context: Dict[str, Any] = None
    ) -> None:
        """Log an error that occurred during decision execution"""
        if decision_id not in self.active_decisions:
            self.logger.warning(f"Decision {decision_id} not found for error logging")
            return
        
        decision = self.active_decisions[decision_id]
        decision.add_error(error_type, error_message, error_context)
        
        self.logger.error(
            f"Error in decision {decision_id}: {error_type} - {error_message}",
            extra={
                'decision_id': decision_id,
                'error_type': error_type,
                'error_message': error_message,
                'error_context': error_context,
                'correlation_id': decision.correlation_id
            }
        )
    
    def log_recovery_action(
        self,
        decision_id: str,
        action_type: str,
        action_description: str,
        action_result: str
    ) -> None:
        """Log a recovery action taken in response to decision errors"""
        if decision_id not in self.active_decisions:
            self.logger.warning(f"Decision {decision_id} not found for recovery action logging")
            return
        
        decision = self.active_decisions[decision_id]
        decision.add_recovery_action(action_type, action_description, action_result)
        
        self.logger.info(
            f"Recovery action for decision {decision_id}: {action_type} - {action_result}",
            extra={
                'decision_id': decision_id,
                'action_type': action_type,
                'action_description': action_description,
                'action_result': action_result,
                'correlation_id': decision.correlation_id
            }
        )
    
    def log_workflow_coordination(
        self,
        workflow_id: str,
        coordination_type: str,
        agents_involved: List[str],
        coordination_result: str,
        correlation_id: str = None
    ) -> None:
        """Log workflow coordination activities"""
        self.logger.info(
            f"Workflow coordination: {coordination_type} for workflow {workflow_id}",
            extra={
                'workflow_id': workflow_id,
                'coordination_type': coordination_type,
                'agents_involved': agents_involved,
                'coordination_result': coordination_result,
                'correlation_id': correlation_id or str(uuid4())
            }
        )
    
    def log_trigger_handling(
        self,
        trigger_event: TriggerEvent,
        handling_decision: str,
        response_actions: List[str],
        response_time_ms: float
    ) -> None:
        """Log trigger event handling decisions"""
        self.logger.info(
            f"Trigger handled: {trigger_event.event_type.value} - {handling_decision}",
            extra={
                'trigger_id': trigger_event.trigger_id,
                'trigger_type': trigger_event.trigger_type,
                'event_type': trigger_event.event_type.value,
                'severity': trigger_event.severity.value,
                'handling_decision': handling_decision,
                'response_actions': response_actions,
                'response_time_ms': response_time_ms,
                'correlation_id': trigger_event.correlation_id
            }
        )
    
    def log_agent_communication(
        self,
        message: AgentMessage,
        communication_result: str,
        response_time_ms: float = None
    ) -> None:
        """Log agent communication activities"""
        self.logger.debug(
            f"Agent communication: {message.agent_id} -> {message.target_agent_id} ({message.message_type.value})",
            extra={
                'message_id': message.message_id,
                'sender_agent': message.agent_id,
                'target_agent': message.target_agent_id,
                'message_type': message.message_type.value,
                'communication_result': communication_result,
                'response_time_ms': response_time_ms,
                'correlation_id': message.correlation_id,
                'session_id': message.session_id
            }
        )
    
    def get_decision_history(
        self,
        decision_type: DecisionType = None,
        correlation_id: str = None,
        session_id: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get decision history with optional filtering"""
        decisions = self.completed_decisions
        
        # Apply filters
        if decision_type:
            decisions = [d for d in decisions if d.decision_type == decision_type]
        
        if correlation_id:
            decisions = [d for d in decisions if d.correlation_id == correlation_id]
        
        if session_id:
            decisions = [d for d in decisions if d.session_id == session_id]
        
        # Sort by timestamp (most recent first) and limit
        decisions = sorted(decisions, key=lambda d: d.timestamp, reverse=True)[:limit]
        
        return [decision.to_dict() for decision in decisions]
    
    def get_decision_metrics(self) -> Dict[str, Any]:
        """Get comprehensive decision metrics"""
        return {
            **self.decision_metrics,
            "active_decisions": len(self.active_decisions),
            "completed_decisions": len(self.completed_decisions),
            "decision_success_rate": (
                self.decision_metrics["successful_decisions"] / 
                max(1, self.decision_metrics["total_decisions"])
            )
        }
    
    def export_decision_log(
        self,
        output_file: str,
        format_type: str = "json",
        include_active: bool = False
    ) -> None:
        """Export decision log to file"""
        decisions_to_export = self.completed_decisions.copy()
        
        if include_active:
            decisions_to_export.extend(self.active_decisions.values())
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_decisions": len(decisions_to_export),
            "metrics": self.get_decision_metrics(),
            "decisions": [decision.to_dict() for decision in decisions_to_export]
        }
        
        with open(output_file, 'w') as f:
            if format_type.lower() == "json":
                json.dump(export_data, f, indent=2, default=str)
            else:
                # CSV format for basic decision data
                import csv
                writer = csv.writer(f)
                writer.writerow([
                    "decision_id", "decision_type", "timestamp", "outcome",
                    "execution_time", "confidence_score", "correlation_id"
                ])
                
                for decision in decisions_to_export:
                    writer.writerow([
                        decision.decision_id,
                        decision.decision_type.value,
                        decision.timestamp.isoformat(),
                        decision.outcome.value if decision.outcome else "pending",
                        (decision.execution_end_time - decision.execution_start_time).total_seconds() 
                        if decision.execution_start_time and decision.execution_end_time else None,
                        decision.confidence_score,
                        decision.correlation_id
                    ])
        
        self.logger.info(f"Exported {len(decisions_to_export)} decisions to {output_file}")
    
    def _update_decision_metrics(self, decision: OrchestrationDecision) -> None:
        """Update decision performance metrics"""
        self.decision_metrics["total_decisions"] += 1
        
        if decision.outcome == DecisionOutcome.SUCCESS:
            self.decision_metrics["successful_decisions"] += 1
        elif decision.outcome in [DecisionOutcome.FAILURE, DecisionOutcome.TIMEOUT]:
            self.decision_metrics["failed_decisions"] += 1
        
        # Update decision type metrics
        decision_type = decision.decision_type.value
        if decision_type not in self.decision_metrics["decision_types"]:
            self.decision_metrics["decision_types"][decision_type] = {
                "count": 0,
                "success_count": 0,
                "average_time": 0.0
            }
        
        type_metrics = self.decision_metrics["decision_types"][decision_type]
        type_metrics["count"] += 1
        
        if decision.outcome == DecisionOutcome.SUCCESS:
            type_metrics["success_count"] += 1
        
        # Update average execution time
        if decision.execution_start_time and decision.execution_end_time:
            execution_time = (decision.execution_end_time - decision.execution_start_time).total_seconds()
            
            # Update overall average
            total_decisions = self.decision_metrics["total_decisions"]
            current_avg = self.decision_metrics["average_decision_time"]
            self.decision_metrics["average_decision_time"] = (
                (current_avg * (total_decisions - 1) + execution_time) / total_decisions
            )
            
            # Update type-specific average
            type_count = type_metrics["count"]
            type_avg = type_metrics["average_time"]
            type_metrics["average_time"] = (
                (type_avg * (type_count - 1) + execution_time) / type_count
            )
    
    def _create_audit_entry(self, decision: OrchestrationDecision) -> None:
        """Create audit trail entry for completed decision"""
        audit_entry = AuditTrail(
            user_id=None,  # System decision
            session_id=decision.session_id,
            agent_id=decision.decision_maker,
            correlation_id=decision.correlation_id,
            action_type=decision.decision_type.value,
            action_description=f"Orchestration decision: {decision.rationale}",
            authorization_level="system",
            action_result=decision.outcome.value if decision.outcome else "unknown"
        )
        
        self.audit_entries.append(audit_entry)
        
        # Maintain audit history limit
        if len(self.audit_entries) > self.decision_history_limit:
            self.audit_entries = self.audit_entries[-self.decision_history_limit:]


# Global orchestration logger instance
orchestration_logger = OrchestrationLogger()


# Convenience functions for common logging operations
def log_workflow_start(workflow_id: str, workflow_type: str, user_id: str, correlation_id: str = None) -> str:
    """Log workflow initiation decision"""
    return orchestration_logger.start_decision(
        DecisionType.WORKFLOW_INITIATION,
        {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "user_id": user_id
        },
        correlation_id=correlation_id
    )


def log_agent_selection(agents_available: List[str], selected_agent: str, selection_criteria: Dict[str, Any], correlation_id: str = None) -> str:
    """Log agent selection decision"""
    decision_id = orchestration_logger.start_decision(
        DecisionType.AGENT_SELECTION,
        {
            "agents_available": agents_available,
            "selection_criteria": selection_criteria
        },
        correlation_id=correlation_id
    )
    
    # Add all available agents as options
    for agent in agents_available:
        orchestration_logger.add_decision_option(
            decision_id,
            agent,
            {"agent_id": agent},
            score=1.0 if agent == selected_agent else 0.5
        )
    
    return decision_id


def log_trigger_response(trigger_event: TriggerEvent, response_strategy: str, rationale: str) -> None:
    """Log trigger response decision"""
    orchestration_logger.log_trigger_handling(
        trigger_event,
        response_strategy,
        ["cmvl_activation", "agent_coordination", "plan_adjustment"],
        response_time_ms=50.0  # Typical response time
    )


def log_priority_escalation(original_priority: str, new_priority: str, escalation_reason: str, correlation_id: str = None) -> str:
    """Log priority escalation decision"""
    return orchestration_logger.start_decision(
        DecisionType.PRIORITY_ESCALATION,
        {
            "original_priority": original_priority,
            "new_priority": new_priority,
            "escalation_reason": escalation_reason
        },
        correlation_id=correlation_id
    )