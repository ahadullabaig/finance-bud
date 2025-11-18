"""
Comprehensive Logging, Tracing and Performance Tracking System

Implements detailed logging, reasoning trace generation, and performance metrics
for the Planning Agent and Guided Search Module as required by Task 12.

Requirements: 3.3, 7.5, 10.1, 7.1, 7.2, 8.1, 8.2, 11.1, 11.3
"""

import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import threading
from collections import defaultdict, deque

from data_models.schemas import (
    PerformanceMetrics, AgentMessage, ExecutionStatus,
    SearchPath, ReasoningTrace, DecisionPoint
)


class LogLevel(str, Enum):
    """Enhanced logging levels for detailed tracing"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DecisionType(str, Enum):
    """Types of decisions made during planning"""
    STRATEGY_SELECTION = "strategy_selection"
    PATH_EXPLORATION = "path_exploration"
    CONSTRAINT_HANDLING = "constraint_handling"
    HEURISTIC_EVALUATION = "heuristic_evaluation"
    PATH_PRUNING = "path_pruning"
    MILESTONE_PLANNING = "milestone_planning"
    SEQUENCE_OPTIMIZATION = "sequence_optimization"
    REJECTION_SAMPLING = "rejection_sampling"
    FINAL_SELECTION = "final_selection"


@dataclass
class PathExplorationLog:
    """Detailed log entry for path exploration"""
    exploration_id: str
    session_id: str
    strategy: str
    start_time: datetime
    end_time: Optional[datetime] = None
    nodes_explored: int = 0
    nodes_pruned: int = 0
    paths_generated: int = 0
    constraint_violations: int = 0
    heuristic_evaluations: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    exploration_depth: int = 0
    pruning_reasons: List[str] = None
    performance_bottlenecks: List[str] = None
    
    def __post_init__(self):
        if self.pruning_reasons is None:
            self.pruning_reasons = []
        if self.performance_bottlenecks is None:
            self.performance_bottlenecks = []


@dataclass
class DecisionLog:
    """Detailed log entry for individual decisions"""
    decision_id: str
    session_id: str
    decision_type: DecisionType
    timestamp: datetime
    context: Dict[str, Any]
    options_considered: List[Dict[str, Any]]
    chosen_option: Dict[str, Any]
    rationale: str
    confidence_score: float
    execution_time_ms: float
    memory_impact_mb: float
    alternative_paths: List[str] = None
    constraint_impacts: List[str] = None
    
    def __post_init__(self):
        if self.alternative_paths is None:
            self.alternative_paths = []
        if self.constraint_impacts is None:
            self.constraint_impacts = []


@dataclass
class PerformanceSnapshot:
    """System performance snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    active_threads: int
    open_files: int
    network_connections: int
    disk_io_read_mb: float
    disk_io_write_mb: float


class ComprehensiveLogger:
    """
    Comprehensive logging system for detailed tracing and performance monitoring.
    
    Provides:
    - Verbose logs for each reasoning step and decision point
    - Performance metrics tracking with optimization insights
    - Multi-layered decision documentation
    - Path exploration analysis and debugging tools
    - Real-time performance monitoring
    """
    
    def __init__(self, logger_name: str = "planning_agent", log_level: LogLevel = LogLevel.INFO):
        self.logger_name = logger_name
        self.log_level = log_level
        
        # Set up structured logging
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.value))
        
        # Create formatters for different log types
        self._setup_formatters()
        
        # Performance tracking
        self.performance_snapshots: deque = deque(maxlen=1000)
        self.decision_logs: Dict[str, List[DecisionLog]] = defaultdict(list)
        self.exploration_logs: Dict[str, List[PathExplorationLog]] = defaultdict(list)
        
        # Real-time metrics
        self.metrics_lock = threading.Lock()
        self.current_metrics = {
            "total_decisions": 0,
            "total_explorations": 0,
            "average_decision_time": 0.0,
            "average_exploration_time": 0.0,
            "constraint_violation_rate": 0.0,
            "path_pruning_rate": 0.0,
            "memory_efficiency": 0.0,
            "cpu_efficiency": 0.0
        }
        
        # Start performance monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()
    
    def _setup_formatters(self):
        """Set up different formatters for various log types"""
        # Standard formatter
        standard_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # JSON formatter for structured logs
        self.json_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(standard_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(f'logs/{self.logger_name}_detailed.log')
        file_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(file_handler)
    
    def log_decision_point(
        self,
        session_id: str,
        decision_type: DecisionType,
        context: Dict[str, Any],
        options_considered: List[Dict[str, Any]],
        chosen_option: Dict[str, Any],
        rationale: str,
        confidence_score: float,
        execution_time_ms: float = 0.0,
        alternative_paths: List[str] = None,
        constraint_impacts: List[str] = None
    ) -> str:
        """
        Log a detailed decision point with comprehensive metadata.
        
        Returns the decision_id for correlation tracking.
        """
        decision_id = str(uuid4())
        
        # Capture memory impact
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        decision_log = DecisionLog(
            decision_id=decision_id,
            session_id=session_id,
            decision_type=decision_type,
            timestamp=datetime.utcnow(),
            context=context,
            options_considered=options_considered,
            chosen_option=chosen_option,
            rationale=rationale,
            confidence_score=confidence_score,
            execution_time_ms=execution_time_ms,
            memory_impact_mb=memory_before,
            alternative_paths=alternative_paths or [],
            constraint_impacts=constraint_impacts or []
        )
        
        # Store decision log
        self.decision_logs[session_id].append(decision_log)
        
        # Log structured decision
        self.logger.info(
            f"DECISION_POINT: {decision_type.value} | "
            f"Session: {session_id} | "
            f"Confidence: {confidence_score:.3f} | "
            f"Options: {len(options_considered)} | "
            f"Time: {execution_time_ms:.2f}ms | "
            f"Rationale: {rationale[:100]}..."
        )
        
        # Update metrics
        with self.metrics_lock:
            self.current_metrics["total_decisions"] += 1
            self._update_average_decision_time(execution_time_ms)
        
        return decision_id
    
    def log_path_exploration_start(
        self,
        session_id: str,
        strategy: str,
        initial_context: Dict[str, Any]
    ) -> str:
        """Start logging path exploration with detailed tracking"""
        exploration_id = str(uuid4())
        
        exploration_log = PathExplorationLog(
            exploration_id=exploration_id,
            session_id=session_id,
            strategy=strategy,
            start_time=datetime.utcnow(),
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent()
        )
        
        self.exploration_logs[session_id].append(exploration_log)
        
        self.logger.info(
            f"PATH_EXPLORATION_START: {strategy} | "
            f"Session: {session_id} | "
            f"Exploration: {exploration_id} | "
            f"Context: {json.dumps(initial_context, default=str)[:200]}..."
        )
        
        return exploration_id
    
    def log_path_exploration_step(
        self,
        session_id: str,
        exploration_id: str,
        step_type: str,
        step_data: Dict[str, Any],
        performance_data: Dict[str, float] = None
    ):
        """Log individual steps during path exploration"""
        # Find the exploration log
        exploration_log = None
        for log in self.exploration_logs[session_id]:
            if log.exploration_id == exploration_id:
                exploration_log = log
                break
        
        if exploration_log:
            # Update exploration metrics
            if step_type == "node_explored":
                exploration_log.nodes_explored += 1
            elif step_type == "node_pruned":
                exploration_log.nodes_pruned += 1
                if "reason" in step_data:
                    exploration_log.pruning_reasons.append(step_data["reason"])
            elif step_type == "path_generated":
                exploration_log.paths_generated += 1
            elif step_type == "constraint_violation":
                exploration_log.constraint_violations += 1
            elif step_type == "heuristic_evaluation":
                exploration_log.heuristic_evaluations += 1
        
        self.logger.debug(
            f"PATH_EXPLORATION_STEP: {step_type} | "
            f"Exploration: {exploration_id} | "
            f"Data: {json.dumps(step_data, default=str)[:150]}..."
        )
        
        # Log performance bottlenecks
        if performance_data and exploration_log:
            if performance_data.get("execution_time", 0) > 100:  # > 100ms
                exploration_log.performance_bottlenecks.append(
                    f"{step_type}: {performance_data['execution_time']:.2f}ms"
                )
    
    def log_path_exploration_end(
        self,
        session_id: str,
        exploration_id: str,
        final_results: Dict[str, Any],
        performance_summary: Dict[str, float]
    ):
        """Complete path exploration logging with final metrics"""
        # Find and update exploration log
        exploration_log = None
        for log in self.exploration_logs[session_id]:
            if log.exploration_id == exploration_id:
                exploration_log = log
                break
        
        if exploration_log:
            exploration_log.end_time = datetime.utcnow()
            exploration_log.exploration_depth = final_results.get("max_depth", 0)
            
            # Calculate exploration efficiency
            total_time = (exploration_log.end_time - exploration_log.start_time).total_seconds()
            efficiency = exploration_log.paths_generated / max(total_time, 0.001)
            
            self.logger.info(
                f"PATH_EXPLORATION_END: {exploration_log.strategy} | "
                f"Exploration: {exploration_id} | "
                f"Duration: {total_time:.2f}s | "
                f"Nodes: {exploration_log.nodes_explored} | "
                f"Paths: {exploration_log.paths_generated} | "
                f"Pruned: {exploration_log.nodes_pruned} | "
                f"Efficiency: {efficiency:.2f} paths/sec"
            )
            
            # Update global metrics
            with self.metrics_lock:
                self.current_metrics["total_explorations"] += 1
                self._update_average_exploration_time(total_time)
                self._update_pruning_rate(exploration_log)
    
    def log_constraint_evaluation(
        self,
        session_id: str,
        constraint_id: str,
        constraint_type: str,
        evaluation_result: bool,
        violation_details: Dict[str, Any] = None,
        mitigation_suggestions: List[str] = None
    ):
        """Log detailed constraint evaluation results"""
        self.logger.info(
            f"CONSTRAINT_EVAL: {constraint_type} | "
            f"ID: {constraint_id} | "
            f"Result: {'SATISFIED' if evaluation_result else 'VIOLATED'} | "
            f"Session: {session_id}"
        )
        
        if not evaluation_result and violation_details:
            self.logger.warning(
                f"CONSTRAINT_VIOLATION: {constraint_id} | "
                f"Details: {json.dumps(violation_details, default=str)} | "
                f"Suggestions: {mitigation_suggestions or []}"
            )
    
    def log_heuristic_evaluation(
        self,
        session_id: str,
        heuristic_type: str,
        input_state: Dict[str, Any],
        heuristic_score: float,
        calculation_details: Dict[str, Any],
        execution_time_ms: float
    ):
        """Log detailed heuristic evaluation with calculation breakdown"""
        self.logger.debug(
            f"HEURISTIC_EVAL: {heuristic_type} | "
            f"Score: {heuristic_score:.4f} | "
            f"Time: {execution_time_ms:.2f}ms | "
            f"Session: {session_id}"
        )
        
        # Log calculation details for debugging
        if self.log_level in [LogLevel.TRACE, LogLevel.DEBUG]:
            self.logger.debug(
                f"HEURISTIC_DETAILS: {heuristic_type} | "
                f"Input: {json.dumps(input_state, default=str)[:200]}... | "
                f"Calculation: {json.dumps(calculation_details, default=str)[:200]}..."
            )
    
    def log_performance_optimization_insight(
        self,
        session_id: str,
        optimization_type: str,
        current_performance: Dict[str, float],
        optimization_suggestion: str,
        expected_improvement: Dict[str, float]
    ):
        """Log performance optimization insights and recommendations"""
        self.logger.info(
            f"OPTIMIZATION_INSIGHT: {optimization_type} | "
            f"Session: {session_id} | "
            f"Suggestion: {optimization_suggestion} | "
            f"Expected Improvement: {json.dumps(expected_improvement, default=str)}"
        )
    
    def generate_reasoning_trace(
        self,
        session_id: str,
        operation_type: str,
        final_decision: str,
        confidence_score: float
    ) -> ReasoningTrace:
        """
        Generate comprehensive reasoning trace with multi-layered decision documentation.
        
        Returns detailed ReasoningTrace for ReasonGraph visualization.
        """
        # Collect all decision points for this session
        decision_points = []
        
        for decision_log in self.decision_logs.get(session_id, []):
            decision_point = DecisionPoint(
                decision_type=decision_log.decision_type.value,
                options_considered=decision_log.options_considered,
                chosen_option=decision_log.chosen_option,
                rationale=decision_log.rationale,
                confidence_score=decision_log.confidence_score,
                timestamp=decision_log.timestamp,
                execution_time_ms=decision_log.execution_time_ms,
                alternative_paths=decision_log.alternative_paths,
                constraint_impacts=decision_log.constraint_impacts
            )
            decision_points.append(decision_point)
        
        # Collect exploration data
        exploration_data = []
        for exploration_log in self.exploration_logs.get(session_id, []):
            exploration_data.append({
                "exploration_id": exploration_log.exploration_id,
                "strategy": exploration_log.strategy,
                "nodes_explored": exploration_log.nodes_explored,
                "nodes_pruned": exploration_log.nodes_pruned,
                "paths_generated": exploration_log.paths_generated,
                "constraint_violations": exploration_log.constraint_violations,
                "pruning_reasons": exploration_log.pruning_reasons,
                "performance_bottlenecks": exploration_log.performance_bottlenecks,
                "duration_seconds": (
                    (exploration_log.end_time - exploration_log.start_time).total_seconds()
                    if exploration_log.end_time else 0
                )
            })
        
        # Generate comprehensive performance metrics
        performance_metrics = self._generate_session_performance_metrics(session_id)
        
        # Create detailed rationale
        detailed_rationale = self._generate_detailed_rationale(
            session_id, decision_points, exploration_data
        )
        
        reasoning_trace = ReasoningTrace(
            session_id=session_id,
            agent_id=self.logger_name,
            operation_type=operation_type,
            start_time=min(
                (log.timestamp for log in self.decision_logs.get(session_id, [])),
                default=datetime.utcnow()
            ),
            end_time=datetime.utcnow(),
            decision_points=decision_points,
            search_paths=[],  # Will be populated by caller
            final_decision=final_decision,
            decision_rationale=detailed_rationale,
            confidence_score=confidence_score,
            performance_metrics=performance_metrics,
            exploration_metadata=exploration_data,
            optimization_insights=self._generate_optimization_insights(session_id)
        )
        
        return reasoning_trace
    
    def _generate_session_performance_metrics(self, session_id: str) -> PerformanceMetrics:
        """Generate comprehensive performance metrics for a session"""
        decision_logs = self.decision_logs.get(session_id, [])
        exploration_logs = self.exploration_logs.get(session_id, [])
        
        # Calculate timing metrics
        total_decision_time = sum(log.execution_time_ms for log in decision_logs)
        total_exploration_time = sum(
            (log.end_time - log.start_time).total_seconds() * 1000
            for log in exploration_logs
            if log.end_time
        )
        
        # Calculate efficiency metrics
        total_nodes_explored = sum(log.nodes_explored for log in exploration_logs)
        total_paths_generated = sum(log.paths_generated for log in exploration_logs)
        total_nodes_pruned = sum(log.nodes_pruned for log in exploration_logs)
        
        return PerformanceMetrics(
            execution_time=total_decision_time + total_exploration_time,
            memory_usage=max(
                (log.memory_impact_mb for log in decision_logs),
                default=0.0
            ),
            api_calls=0,  # Would be tracked separately
            cache_hits=0,  # Would be tracked separately
            cache_misses=0,  # Would be tracked separately
            error_count=0,  # Would be tracked separately
            success_rate=1.0 if total_paths_generated > 0 else 0.0,
            throughput=total_paths_generated / max(total_exploration_time / 1000, 0.001),
            latency_p50=total_decision_time / max(len(decision_logs), 1),
            latency_p95=max(
                (log.execution_time_ms for log in decision_logs),
                default=0.0
            ),
            latency_p99=max(
                (log.execution_time_ms for log in decision_logs),
                default=0.0
            )
        )
    
    def _generate_detailed_rationale(
        self,
        session_id: str,
        decision_points: List[DecisionPoint],
        exploration_data: List[Dict[str, Any]]
    ) -> str:
        """Generate detailed multi-layered rationale for the reasoning process"""
        rationale_parts = []
        
        # Strategy exploration summary
        strategies_explored = set(exp["strategy"] for exp in exploration_data)
        rationale_parts.append(
            f"Explored {len(strategies_explored)} strategic approaches: {', '.join(strategies_explored)}."
        )
        
        # Decision analysis
        decision_types = [dp.decision_type for dp in decision_points]
        rationale_parts.append(
            f"Made {len(decision_points)} key decisions across {len(set(decision_types))} decision categories."
        )
        
        # Performance analysis
        total_nodes = sum(exp["nodes_explored"] for exp in exploration_data)
        total_paths = sum(exp["paths_generated"] for exp in exploration_data)
        pruning_rate = sum(exp["nodes_pruned"] for exp in exploration_data) / max(total_nodes, 1)
        
        rationale_parts.append(
            f"Explored {total_nodes} nodes, generated {total_paths} paths, "
            f"with {pruning_rate:.1%} pruning efficiency."
        )
        
        # Constraint handling
        constraint_decisions = [dp for dp in decision_points if dp.decision_type == "constraint_handling"]
        if constraint_decisions:
            avg_confidence = sum(dp.confidence_score for dp in constraint_decisions) / len(constraint_decisions)
            rationale_parts.append(
                f"Constraint handling decisions made with {avg_confidence:.1%} average confidence."
            )
        
        # Performance bottlenecks
        all_bottlenecks = []
        for exp in exploration_data:
            all_bottlenecks.extend(exp.get("performance_bottlenecks", []))
        
        if all_bottlenecks:
            rationale_parts.append(
                f"Identified {len(all_bottlenecks)} performance bottlenecks for optimization."
            )
        
        return " ".join(rationale_parts)
    
    def _generate_optimization_insights(self, session_id: str) -> List[Dict[str, Any]]:
        """Generate optimization insights based on session performance"""
        insights = []
        
        exploration_logs = self.exploration_logs.get(session_id, [])
        decision_logs = self.decision_logs.get(session_id, [])
        
        # Analyze exploration efficiency
        for exploration_log in exploration_logs:
            if exploration_log.end_time:
                duration = (exploration_log.end_time - exploration_log.start_time).total_seconds()
                efficiency = exploration_log.paths_generated / max(duration, 0.001)
                
                if efficiency < 1.0:  # Less than 1 path per second
                    insights.append({
                        "type": "exploration_efficiency",
                        "strategy": exploration_log.strategy,
                        "issue": "Low path generation efficiency",
                        "current_rate": efficiency,
                        "recommendation": "Consider increasing beam width or optimizing heuristics",
                        "expected_improvement": "2-3x faster exploration"
                    })
        
        # Analyze decision timing
        slow_decisions = [log for log in decision_logs if log.execution_time_ms > 100]
        if slow_decisions:
            insights.append({
                "type": "decision_timing",
                "issue": f"{len(slow_decisions)} slow decisions detected",
                "recommendation": "Cache heuristic calculations or simplify decision logic",
                "expected_improvement": "50% faster decision making"
            })
        
        # Analyze pruning effectiveness
        total_explored = sum(log.nodes_explored for log in exploration_logs)
        total_pruned = sum(log.nodes_pruned for log in exploration_logs)
        
        if total_explored > 0:
            pruning_rate = total_pruned / total_explored
            if pruning_rate < 0.3:  # Less than 30% pruning
                insights.append({
                    "type": "pruning_efficiency",
                    "issue": "Low pruning rate may indicate inefficient search",
                    "current_rate": pruning_rate,
                    "recommendation": "Tighten pruning thresholds or improve heuristics",
                    "expected_improvement": "30-50% reduction in search space"
                })
        
        return insights
    
    def _monitor_performance(self):
        """Background thread for continuous performance monitoring"""
        while self.monitoring_active:
            try:
                # Capture system metrics
                process = psutil.Process()
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.utcnow(),
                    cpu_percent=psutil.cpu_percent(),
                    memory_mb=process.memory_info().rss / 1024 / 1024,
                    memory_percent=process.memory_percent(),
                    active_threads=process.num_threads(),
                    open_files=len(process.open_files()),
                    network_connections=len(process.connections()),
                    disk_io_read_mb=process.io_counters().read_bytes / 1024 / 1024,
                    disk_io_write_mb=process.io_counters().write_bytes / 1024 / 1024
                )
                
                self.performance_snapshots.append(snapshot)
                
                # Update efficiency metrics
                with self.metrics_lock:
                    self.current_metrics["memory_efficiency"] = 100.0 - snapshot.memory_percent
                    self.current_metrics["cpu_efficiency"] = 100.0 - snapshot.cpu_percent
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _update_average_decision_time(self, new_time: float):
        """Update running average of decision times"""
        current_avg = self.current_metrics["average_decision_time"]
        total_decisions = self.current_metrics["total_decisions"]
        
        if total_decisions == 1:
            self.current_metrics["average_decision_time"] = new_time
        else:
            self.current_metrics["average_decision_time"] = (
                (current_avg * (total_decisions - 1) + new_time) / total_decisions
            )
    
    def _update_average_exploration_time(self, new_time: float):
        """Update running average of exploration times"""
        current_avg = self.current_metrics["average_exploration_time"]
        total_explorations = self.current_metrics["total_explorations"]
        
        if total_explorations == 1:
            self.current_metrics["average_exploration_time"] = new_time
        else:
            self.current_metrics["average_exploration_time"] = (
                (current_avg * (total_explorations - 1) + new_time) / total_explorations
            )
    
    def _update_pruning_rate(self, exploration_log: PathExplorationLog):
        """Update pruning rate metrics"""
        if exploration_log.nodes_explored > 0:
            session_pruning_rate = exploration_log.nodes_pruned / exploration_log.nodes_explored
            
            # Update global pruning rate (simple moving average)
            current_rate = self.current_metrics["path_pruning_rate"]
            total_explorations = self.current_metrics["total_explorations"]
            
            if total_explorations == 1:
                self.current_metrics["path_pruning_rate"] = session_pruning_rate
            else:
                self.current_metrics["path_pruning_rate"] = (
                    (current_rate * (total_explorations - 1) + session_pruning_rate) / total_explorations
                )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        with self.metrics_lock:
            return self.current_metrics.copy()
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for a specific session"""
        decision_logs = self.decision_logs.get(session_id, [])
        exploration_logs = self.exploration_logs.get(session_id, [])
        
        return {
            "session_id": session_id,
            "total_decisions": len(decision_logs),
            "total_explorations": len(exploration_logs),
            "decision_types": list(set(log.decision_type.value for log in decision_logs)),
            "strategies_explored": list(set(log.strategy for log in exploration_logs)),
            "total_nodes_explored": sum(log.nodes_explored for log in exploration_logs),
            "total_paths_generated": sum(log.paths_generated for log in exploration_logs),
            "total_constraint_violations": sum(log.constraint_violations for log in exploration_logs),
            "average_confidence": (
                sum(log.confidence_score for log in decision_logs) / max(len(decision_logs), 1)
            ),
            "performance_bottlenecks": [
                bottleneck
                for log in exploration_logs
                for bottleneck in log.performance_bottlenecks
            ],
            "optimization_insights": self._generate_optimization_insights(session_id)
        }
    
    def export_session_data(self, session_id: str, format: str = "json") -> str:
        """Export comprehensive session data for analysis"""
        session_data = {
            "session_summary": self.get_session_summary(session_id),
            "decision_logs": [asdict(log) for log in self.decision_logs.get(session_id, [])],
            "exploration_logs": [asdict(log) for log in self.exploration_logs.get(session_id, [])],
            "performance_snapshots": [
                asdict(snapshot) for snapshot in list(self.performance_snapshots)[-100:]
            ],
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        if format == "json":
            return json.dumps(session_data, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_session(self, session_id: str):
        """Clean up session data to free memory"""
        if session_id in self.decision_logs:
            del self.decision_logs[session_id]
        if session_id in self.exploration_logs:
            del self.exploration_logs[session_id]
    
    def shutdown(self):
        """Shutdown the logger and cleanup resources"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Log final metrics
        self.logger.info(f"Logger shutdown. Final metrics: {self.get_current_metrics()}")


# Global logger instance for easy access
comprehensive_logger = ComprehensiveLogger()