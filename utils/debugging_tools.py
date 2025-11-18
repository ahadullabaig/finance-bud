"""
Debugging Tools for Path Exploration Analysis

Provides comprehensive debugging utilities for analyzing planning agent performance,
path exploration, constraint satisfaction, and reasoning traces.

Requirements: Task 12 - debugging tools for path exploration analysis
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from agents.planning_agent import SearchPath, PlanningState, SearchNode, HeuristicType
from data_models.schemas import ReasoningTrace, DecisionPoint, PerformanceMetrics


@dataclass
class PathExplorationAnalysis:
    """Analysis results for path exploration debugging"""
    session_id: str
    total_paths_explored: int
    successful_paths: int
    pruned_paths: int
    average_exploration_time: float
    constraint_satisfaction_rates: Dict[str, float]
    heuristic_performance: Dict[str, Dict[str, float]]
    bottlenecks_identified: List[str]
    optimization_recommendations: List[str]


@dataclass
class ConstraintAnalysisResult:
    """Results of constraint satisfaction analysis"""
    constraint_id: str
    constraint_type: str
    satisfaction_rate: float
    violation_frequency: int
    average_violation_severity: float
    impact_on_path_selection: float
    recommendations: List[str]


@dataclass
class HeuristicPerformanceAnalysis:
    """Analysis of heuristic performance and optimization opportunities"""
    heuristic_name: str
    average_execution_time: float
    total_calls: int
    accuracy_score: float
    optimization_potential: float
    bottleneck_severity: str
    recommendations: List[str]


class PathExplorationDebugger:
    """
    Comprehensive debugging tool for analyzing planning agent path exploration.
    
    Provides detailed analysis of:
    - Search tree exploration patterns
    - Heuristic performance and accuracy
    - Constraint satisfaction bottlenecks
    - Performance optimization opportunities
    - Decision point analysis
    """
    
    def __init__(self):
        self.analysis_cache = {}
        self.performance_history = []
        self.constraint_violation_patterns = {}
        self.heuristic_accuracy_tracking = {}
    
    def analyze_path_exploration_session(
        self, 
        session_id: str, 
        planning_agent, 
        search_paths: List[SearchPath],
        reasoning_trace: ReasoningTrace
    ) -> PathExplorationAnalysis:
        """
        Perform comprehensive analysis of a path exploration session.
        
        Returns detailed analysis including performance bottlenecks,
        optimization opportunities, and constraint satisfaction patterns.
        """
        
        start_time = time.time()
        
        # Analyze search performance
        search_performance = self._analyze_search_performance(
            session_id, planning_agent, search_paths
        )
        
        # Analyze constraint satisfaction patterns
        constraint_analysis = self._analyze_constraint_satisfaction(
            search_paths, reasoning_trace
        )
        
        # Analyze heuristic performance
        heuristic_analysis = self._analyze_heuristic_performance(
            planning_agent, session_id
        )
        
        # Identify bottlenecks and optimization opportunities
        bottlenecks = self._identify_performance_bottlenecks(
            search_performance, constraint_analysis, heuristic_analysis
        )
        
        optimization_recommendations = self._generate_optimization_recommendations(
            bottlenecks, search_performance, constraint_analysis
        )
        
        analysis_time = time.time() - start_time
        
        analysis = PathExplorationAnalysis(
            session_id=session_id,
            total_paths_explored=len(search_paths),
            successful_paths=len([p for p in search_paths if p.path_status == "completed"]),
            pruned_paths=len([p for p in search_paths if "pruned" in p.path_status]),
            average_exploration_time=sum([p.exploration_time for p in search_paths]) / max(len(search_paths), 1),
            constraint_satisfaction_rates=constraint_analysis,
            heuristic_performance=heuristic_analysis,
            bottlenecks_identified=bottlenecks,
            optimization_recommendations=optimization_recommendations
        )
        
        # Cache analysis for future reference
        self.analysis_cache[session_id] = analysis
        
        return analysis
    
    def _analyze_search_performance(
        self, 
        session_id: str, 
        planning_agent, 
        search_paths: List[SearchPath]
    ) -> Dict[str, Any]:
        """Analyze search algorithm performance metrics"""
        
        gsm_session_data = planning_agent.gsm.active_sessions.get(session_id, {})
        
        performance_metrics = {
            "total_execution_time": gsm_session_data.get("start_time", 0),
            "nodes_explored": gsm_session_data.get("nodes_explored", 0),
            "paths_generated": len(search_paths),
            "average_path_score": sum([p.combined_score for p in search_paths]) / max(len(search_paths), 1),
            "score_variance": self._calculate_score_variance(search_paths),
            "exploration_efficiency": self._calculate_exploration_efficiency(search_paths, gsm_session_data),
            "memory_usage_pattern": self._analyze_memory_usage_pattern(planning_agent),
            "search_depth_distribution": self._analyze_search_depth_distribution(search_paths)
        }
        
        return performance_metrics
    
    def _analyze_constraint_satisfaction(
        self, 
        search_paths: List[SearchPath], 
        reasoning_trace: ReasoningTrace
    ) -> Dict[str, float]:
        """Analyze constraint satisfaction patterns across paths"""
        
        if not search_paths:
            return {}
        
        # Calculate satisfaction rates by constraint type
        satisfaction_rates = {}
        
        # Extract constraint information from reasoning trace
        constraint_decision_points = [
            dp for dp in reasoning_trace.decision_points 
            if dp.decision_type == "constraint_handling"
        ]
        
        if constraint_decision_points:
            constraint_analysis = constraint_decision_points[0].constraint_analysis
            if constraint_analysis:
                satisfaction_rates["overall"] = constraint_analysis.get("satisfaction_rate", 0.0)
                satisfaction_rates["mandatory_constraints"] = constraint_analysis.get("mandatory_constraints", 0)
                satisfaction_rates["high_priority_constraints"] = constraint_analysis.get("high_priority_constraints", 0)
        
        # Calculate path-level satisfaction statistics
        path_satisfaction_scores = [p.constraint_satisfaction_score for p in search_paths]
        satisfaction_rates.update({
            "average_path_satisfaction": sum(path_satisfaction_scores) / len(path_satisfaction_scores),
            "min_path_satisfaction": min(path_satisfaction_scores),
            "max_path_satisfaction": max(path_satisfaction_scores),
            "satisfaction_variance": self._calculate_variance(path_satisfaction_scores)
        })
        
        return satisfaction_rates
    
    def _analyze_heuristic_performance(
        self, 
        planning_agent, 
        session_id: str
    ) -> Dict[str, Dict[str, float]]:
        """Analyze individual heuristic performance and accuracy"""
        
        heuristic_performance = {}
        
        for heuristic_type, stats in planning_agent.heuristic_performance_stats.items():
            if stats["call_count"] > 0:
                avg_time = stats["total_time"] / stats["call_count"]
                
                # Calculate performance metrics
                performance_data = {
                    "average_execution_time": avg_time,
                    "total_calls": stats["call_count"],
                    "total_time": stats["total_time"],
                    "calls_per_second": stats["call_count"] / max(stats["total_time"], 0.001),
                    "relative_cost": avg_time / max(sum([s["total_time"] for s in planning_agent.heuristic_performance_stats.values()]), 0.001)
                }
                
                heuristic_performance[heuristic_type.value] = performance_data
        
        return heuristic_performance
    
    def _identify_performance_bottlenecks(
        self, 
        search_performance: Dict[str, Any], 
        constraint_analysis: Dict[str, float], 
        heuristic_analysis: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Identify performance bottlenecks in the planning process"""
        
        bottlenecks = []
        
        # Check execution time bottlenecks
        total_time = search_performance.get("total_execution_time", 0)
        if total_time > 10.0:
            bottlenecks.append("High total execution time - consider search optimization")
        
        # Check exploration efficiency
        efficiency = search_performance.get("exploration_efficiency", 1.0)
        if efficiency < 0.5:
            bottlenecks.append("Low exploration efficiency - many paths pruned or failed")
        
        # Check constraint satisfaction bottlenecks
        avg_satisfaction = constraint_analysis.get("average_path_satisfaction", 1.0)
        if avg_satisfaction < 0.6:
            bottlenecks.append("Low constraint satisfaction - constraints may be too restrictive")
        
        # Check heuristic performance bottlenecks
        for heuristic_name, perf_data in heuristic_analysis.items():
            if perf_data.get("relative_cost", 0) > 0.3:
                bottlenecks.append(f"Heuristic {heuristic_name} consuming excessive computation time")
        
        # Check memory usage patterns
        memory_pattern = search_performance.get("memory_usage_pattern", {})
        if memory_pattern.get("peak_usage", 0) > 500:  # MB
            bottlenecks.append("High memory usage - consider reducing search space or caching")
        
        return bottlenecks
    
    def _generate_optimization_recommendations(
        self, 
        bottlenecks: List[str], 
        search_performance: Dict[str, Any], 
        constraint_analysis: Dict[str, float]
    ) -> List[str]:
        """Generate specific optimization recommendations based on analysis"""
        
        recommendations = []
        
        # Performance-based recommendations
        if "High total execution time" in str(bottlenecks):
            recommendations.append("Reduce beam search width or search depth limit")
            recommendations.append("Implement early termination conditions")
            recommendations.append("Consider parallel path exploration")
        
        if "Low exploration efficiency" in str(bottlenecks):
            recommendations.append("Adjust pruning thresholds to be less aggressive")
            recommendations.append("Improve heuristic accuracy to reduce false pruning")
            recommendations.append("Implement adaptive search strategies")
        
        # Constraint-based recommendations
        avg_satisfaction = constraint_analysis.get("average_path_satisfaction", 1.0)
        if avg_satisfaction < 0.6:
            recommendations.append("Consider constraint relaxation or prioritization")
            recommendations.append("Implement constraint negotiation mechanisms")
            recommendations.append("Add constraint violation recovery strategies")
        
        # Heuristic-based recommendations
        score_variance = search_performance.get("score_variance", 0)
        if score_variance < 0.1:
            recommendations.append("Increase heuristic diversity to explore more varied paths")
            recommendations.append("Adjust heuristic weights for better discrimination")
        
        # Memory-based recommendations
        memory_pattern = search_performance.get("memory_usage_pattern", {})
        if memory_pattern.get("peak_usage", 0) > 500:
            recommendations.append("Implement state compression or caching strategies")
            recommendations.append("Use lazy evaluation for expensive computations")
        
        return recommendations
    
    def _calculate_score_variance(self, search_paths: List[SearchPath]) -> float:
        """Calculate variance in path scores"""
        if len(search_paths) < 2:
            return 0.0
        
        scores = [p.combined_score for p in search_paths]
        mean_score = sum(scores) / len(scores)
        variance = sum([(score - mean_score) ** 2 for score in scores]) / len(scores)
        return variance
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum([(val - mean_val) ** 2 for val in values]) / len(values)
        return variance
    
    def _calculate_exploration_efficiency(
        self, 
        search_paths: List[SearchPath], 
        session_data: Dict[str, Any]
    ) -> float:
        """Calculate exploration efficiency (successful paths / total exploration effort)"""
        
        successful_paths = len([p for p in search_paths if p.path_status == "completed"])
        nodes_explored = session_data.get("nodes_explored", len(search_paths))
        
        if nodes_explored == 0:
            return 1.0
        
        return successful_paths / nodes_explored
    
    def _analyze_memory_usage_pattern(self, planning_agent) -> Dict[str, Any]:
        """Analyze memory usage patterns during planning"""
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "current_usage": current_memory,
                "peak_usage": current_memory,  # Would need tracking for actual peak
                "cache_size": len(planning_agent.search_performance_cache),
                "session_count": len(planning_agent.planning_sessions)
            }
        except ImportError:
            return {"current_usage": 0, "peak_usage": 0, "cache_size": 0, "session_count": 0}
    
    def _analyze_search_depth_distribution(self, search_paths: List[SearchPath]) -> Dict[str, Any]:
        """Analyze distribution of search depths across paths"""
        
        # Simplified depth analysis based on sequence steps
        depths = [len(p.sequence_steps) for p in search_paths]
        
        if not depths:
            return {"average_depth": 0, "max_depth": 0, "min_depth": 0}
        
        return {
            "average_depth": sum(depths) / len(depths),
            "max_depth": max(depths),
            "min_depth": min(depths),
            "depth_variance": self._calculate_variance([float(d) for d in depths])
        }
    
    def generate_debugging_report(
        self, 
        session_id: str, 
        analysis: PathExplorationAnalysis
    ) -> str:
        """Generate comprehensive debugging report for path exploration session"""
        
        report_lines = []
        
        report_lines.append(f"=== PATH EXPLORATION DEBUG REPORT ===")
        report_lines.append(f"Session ID: {session_id}")
        report_lines.append(f"Analysis Timestamp: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Performance Summary
        report_lines.append("PERFORMANCE SUMMARY:")
        report_lines.append(f"  Total Paths Explored: {analysis.total_paths_explored}")
        report_lines.append(f"  Successful Paths: {analysis.successful_paths}")
        report_lines.append(f"  Pruned Paths: {analysis.pruned_paths}")
        report_lines.append(f"  Average Exploration Time: {analysis.average_exploration_time:.3f}s")
        report_lines.append("")
        
        # Constraint Analysis
        report_lines.append("CONSTRAINT SATISFACTION ANALYSIS:")
        for constraint_type, rate in analysis.constraint_satisfaction_rates.items():
            report_lines.append(f"  {constraint_type}: {rate:.3f}")
        report_lines.append("")
        
        # Heuristic Performance
        report_lines.append("HEURISTIC PERFORMANCE:")
        for heuristic_name, perf_data in analysis.heuristic_performance.items():
            report_lines.append(f"  {heuristic_name}:")
            report_lines.append(f"    Avg Time: {perf_data.get('average_execution_time', 0):.6f}s")
            report_lines.append(f"    Total Calls: {perf_data.get('total_calls', 0)}")
            report_lines.append(f"    Relative Cost: {perf_data.get('relative_cost', 0):.3f}")
        report_lines.append("")
        
        # Bottlenecks
        if analysis.bottlenecks_identified:
            report_lines.append("IDENTIFIED BOTTLENECKS:")
            for bottleneck in analysis.bottlenecks_identified:
                report_lines.append(f"  - {bottleneck}")
            report_lines.append("")
        
        # Optimization Recommendations
        if analysis.optimization_recommendations:
            report_lines.append("OPTIMIZATION RECOMMENDATIONS:")
            for recommendation in analysis.optimization_recommendations:
                report_lines.append(f"  - {recommendation}")
            report_lines.append("")
        
        report_lines.append("=== END DEBUG REPORT ===")
        
        return "\n".join(report_lines)
    
    def export_analysis_data(
        self, 
        session_id: str, 
        analysis: PathExplorationAnalysis, 
        format: str = "json"
    ) -> str:
        """Export analysis data in specified format for external analysis"""
        
        if format.lower() == "json":
            import json
            return json.dumps(asdict(analysis), indent=2, default=str)
        elif format.lower() == "csv":
            # Simplified CSV export for key metrics
            csv_lines = []
            csv_lines.append("metric,value")
            csv_lines.append(f"session_id,{session_id}")
            csv_lines.append(f"total_paths_explored,{analysis.total_paths_explored}")
            csv_lines.append(f"successful_paths,{analysis.successful_paths}")
            csv_lines.append(f"average_exploration_time,{analysis.average_exploration_time}")
            
            return "\n".join(csv_lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class ConstraintSatisfactionAnalyzer:
    """
    Specialized analyzer for constraint satisfaction patterns and edge cases.
    
    Provides detailed analysis of:
    - Constraint violation patterns
    - Edge case handling
    - Constraint interaction effects
    - Satisfaction optimization opportunities
    """
    
    def __init__(self):
        self.violation_history = {}
        self.edge_case_patterns = {}
        self.constraint_interaction_matrix = {}
    
    def analyze_constraint_violations(
        self, 
        constraints: List[Constraint], 
        search_paths: List[SearchPath],
        financial_states: List[Dict[str, Any]]
    ) -> List[ConstraintAnalysisResult]:
        """Analyze constraint violation patterns across multiple scenarios"""
        
        results = []
        
        for constraint in constraints:
            # Calculate satisfaction statistics
            satisfaction_scores = []
            violation_count = 0
            violation_severities = []
            
            for path in search_paths:
                if path.constraint_satisfaction_score < 1.0:
                    violation_count += 1
                    severity = 1.0 - path.constraint_satisfaction_score
                    violation_severities.append(severity)
                
                satisfaction_scores.append(path.constraint_satisfaction_score)
            
            # Calculate metrics
            satisfaction_rate = sum(satisfaction_scores) / max(len(satisfaction_scores), 1)
            avg_violation_severity = sum(violation_severities) / max(len(violation_severities), 1)
            
            # Analyze impact on path selection
            impact_score = self._calculate_constraint_impact(constraint, search_paths)
            
            # Generate recommendations
            recommendations = self._generate_constraint_recommendations(
                constraint, satisfaction_rate, violation_count, avg_violation_severity
            )
            
            result = ConstraintAnalysisResult(
                constraint_id=constraint.constraint_id,
                constraint_type=constraint.constraint_type.value if hasattr(constraint.constraint_type, 'value') else str(constraint.constraint_type),
                satisfaction_rate=satisfaction_rate,
                violation_frequency=violation_count,
                average_violation_severity=avg_violation_severity,
                impact_on_path_selection=impact_score,
                recommendations=recommendations
            )
            
            results.append(result)
        
        return results
    
    def _calculate_constraint_impact(
        self, 
        constraint: Constraint, 
        search_paths: List[SearchPath]
    ) -> float:
        """Calculate how much a constraint impacts path selection"""
        
        if len(search_paths) < 2:
            return 0.0
        
        # Simplified impact calculation based on score differences
        # In a full implementation, this would analyze actual constraint influence
        scores = [p.constraint_satisfaction_score for p in search_paths]
        score_variance = sum([(s - sum(scores)/len(scores))**2 for s in scores]) / len(scores)
        
        return min(score_variance * 10, 1.0)  # Normalize to [0, 1]
    
    def _generate_constraint_recommendations(
        self, 
        constraint: Constraint, 
        satisfaction_rate: float, 
        violation_count: int, 
        avg_violation_severity: float
    ) -> List[str]:
        """Generate recommendations for constraint optimization"""
        
        recommendations = []
        
        if satisfaction_rate < 0.5:
            recommendations.append("Consider relaxing constraint threshold or priority")
            recommendations.append("Implement constraint negotiation mechanisms")
        
        if violation_count > 5:
            recommendations.append("High violation frequency - review constraint feasibility")
        
        if avg_violation_severity > 0.7:
            recommendations.append("Severe violations detected - implement early warning system")
        
        if constraint.constraint_type == ConstraintType.BUDGET and satisfaction_rate < 0.8:
            recommendations.append("Budget constraint issues - consider income optimization strategies")
        
        if constraint.constraint_type == ConstraintType.LIQUIDITY and satisfaction_rate < 0.7:
            recommendations.append("Liquidity constraint issues - prioritize emergency fund building")
        
        return recommendations
    
    def detect_edge_cases(
        self, 
        financial_states: List[Dict[str, Any]], 
        constraints: List[Constraint]
    ) -> List[Dict[str, Any]]:
        """Detect edge cases in financial states and constraint combinations"""
        
        edge_cases = []
        
        for i, state in enumerate(financial_states):
            # Check for zero income edge case
            if state.get('monthly_income', 0) == 0:
                edge_cases.append({
                    "type": "zero_income",
                    "state_index": i,
                    "description": "Zero monthly income detected",
                    "severity": "high",
                    "affected_constraints": [c.constraint_id for c in constraints if c.constraint_type == ConstraintType.BUDGET]
                })
            
            # Check for negative net worth edge case
            net_worth = state.get('total_assets', 0) - state.get('total_liabilities', 0)
            if net_worth < 0:
                edge_cases.append({
                    "type": "negative_net_worth",
                    "state_index": i,
                    "description": f"Negative net worth: ${net_worth:,.2f}",
                    "severity": "high",
                    "affected_constraints": [c.constraint_id for c in constraints]
                })
            
            # Check for extreme expense ratios
            monthly_income = state.get('monthly_income', 0)
            monthly_expenses = state.get('monthly_expenses', 0)
            if monthly_income > 0:
                expense_ratio = monthly_expenses / monthly_income
                if expense_ratio > 0.95:
                    edge_cases.append({
                        "type": "extreme_expense_ratio",
                        "state_index": i,
                        "description": f"Expense ratio: {expense_ratio:.1%}",
                        "severity": "medium",
                        "affected_constraints": [c.constraint_id for c in constraints if c.constraint_type == ConstraintType.BUDGET]
                    })
            
            # Check for minimal assets
            total_assets = state.get('total_assets', 0)
            if total_assets < 1000:
                edge_cases.append({
                    "type": "minimal_assets",
                    "state_index": i,
                    "description": f"Very low assets: ${total_assets:,.2f}",
                    "severity": "medium",
                    "affected_constraints": [c.constraint_id for c in constraints if c.constraint_type == ConstraintType.LIQUIDITY]
                })
        
        return edge_cases


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite for planning algorithms.
    
    Provides standardized benchmarks for:
    - Search algorithm performance
    - Heuristic calculation speed
    - Memory usage patterns
    - Scalability analysis
    """
    
    def __init__(self):
        self.benchmark_results = {}
        self.baseline_metrics = {}
        self.performance_history = []
    
    def run_comprehensive_benchmarks(self, planning_agent) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks"""
        
        benchmark_results = {}
        
        # Search algorithm benchmarks
        benchmark_results["search_performance"] = self._benchmark_search_algorithms(planning_agent)
        
        # Heuristic performance benchmarks
        benchmark_results["heuristic_performance"] = self._benchmark_heuristic_calculations(planning_agent)
        
        # Memory usage benchmarks
        benchmark_results["memory_performance"] = self._benchmark_memory_usage(planning_agent)
        
        # Scalability benchmarks
        benchmark_results["scalability_analysis"] = self._benchmark_scalability(planning_agent)
        
        # Concurrent performance benchmarks
        benchmark_results["concurrent_performance"] = self._benchmark_concurrent_execution(planning_agent)
        
        # Store results for historical analysis
        self.benchmark_results[datetime.now().isoformat()] = benchmark_results
        
        return benchmark_results
    
    def _benchmark_search_algorithms(self, planning_agent) -> Dict[str, Any]:
        """Benchmark search algorithm performance under various conditions"""
        
        results = {}
        
        # Simple scenario benchmark
        simple_start = time.time()
        simple_paths = planning_agent.gsm.search_optimal_paths(
            initial_state={"total_assets": 100000, "monthly_income": 5000},
            goal="simple benchmark",
            constraints=[],
            time_horizon=12,
            strategies=[SearchStrategy.BALANCED]
        )
        simple_time = time.time() - simple_start
        
        results["simple_scenario"] = {
            "execution_time": simple_time,
            "paths_generated": len(simple_paths),
            "time_per_path": simple_time / max(len(simple_paths), 1)
        }
        
        # Complex scenario benchmark
        complex_constraints = [
            Constraint(
                constraint_id=f"benchmark_constraint_{i}",
                constraint_type=ConstraintType.BUDGET,
                description=f"Benchmark constraint {i}",
                priority=ConstraintPriority.MEDIUM,
                threshold_value=0.5 + (i * 0.02)
            )
            for i in range(10)
        ]
        
        complex_start = time.time()
        complex_paths = planning_agent.gsm.search_optimal_paths(
            initial_state={"total_assets": 500000, "monthly_income": 12000},
            goal="complex benchmark",
            constraints=complex_constraints,
            time_horizon=60,
            strategies=[SearchStrategy.CONSERVATIVE, SearchStrategy.BALANCED, SearchStrategy.AGGRESSIVE]
        )
        complex_time = time.time() - complex_start
        
        results["complex_scenario"] = {
            "execution_time": complex_time,
            "paths_generated": len(complex_paths),
            "time_per_path": complex_time / max(len(complex_paths), 1),
            "constraints_processed": len(complex_constraints)
        }
        
        return results
    
    def _benchmark_heuristic_calculations(self, planning_agent) -> Dict[str, Any]:
        """Benchmark individual heuristic calculation performance"""
        
        results = {}
        gsm = planning_agent.gsm
        
        test_states = [
            {"total_assets": 50000, "investments": 30000, "cash": 20000},
            {"total_assets": 200000, "investments": 120000, "cash": 80000},
            {"total_assets": 1000000, "investments": 700000, "cash": 300000}
        ]
        
        heuristic_methods = {
            "risk_adjusted_return": lambda state: gsm._calculate_risk_adjusted_return(state, SearchStrategy.BALANCED),
            "liquidity_score": lambda state: gsm._calculate_liquidity_score(state),
            "diversification_score": lambda state: gsm._calculate_diversification_score(state),
            "tax_efficiency": lambda state: gsm._calculate_tax_efficiency(state, SearchStrategy.TAX_OPTIMIZED)
        }
        
        for heuristic_name, heuristic_func in heuristic_methods.items():
            execution_times = []
            
            for state in test_states:
                start_time = time.time()
                score = heuristic_func(state)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            
            results[heuristic_name] = {
                "average_time": sum(execution_times) / len(execution_times),
                "max_time": max(execution_times),
                "min_time": min(execution_times),
                "total_calls": len(execution_times)
            }
        
        return results
    
    def _benchmark_memory_usage(self, planning_agent) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute memory-intensive operations
            large_constraints = [
                Constraint(
                    constraint_id=f"memory_constraint_{i}",
                    constraint_type=ConstraintType.BUDGET,
                    description=f"Memory test constraint {i}",
                    priority=ConstraintPriority.LOW,
                    threshold_value=0.1 + (i * 0.01)
                )
                for i in range(50)
            ]
            
            paths = planning_agent.gsm.search_optimal_paths(
                initial_state={"total_assets": 100000, "monthly_income": 5000},
                goal="memory benchmark",
                constraints=large_constraints,
                time_horizon=60,
                strategies=[SearchStrategy.BALANCED, SearchStrategy.CONSERVATIVE]
            )
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "baseline_memory_mb": baseline_memory,
                "peak_memory_mb": peak_memory,
                "memory_increase_mb": peak_memory - baseline_memory,
                "paths_generated": len(paths),
                "memory_per_path_mb": (peak_memory - baseline_memory) / max(len(paths), 1)
            }
            
        except ImportError:
            return {"error": "psutil not available for memory benchmarking"}
    
    def _benchmark_scalability(self, planning_agent) -> Dict[str, Any]:
        """Benchmark scalability with increasing problem complexity"""
        
        results = {}
        
        # Test with increasing constraint counts
        constraint_counts = [1, 5, 10, 20, 30]
        
        for count in constraint_counts:
            constraints = [
                Constraint(
                    constraint_id=f"scale_constraint_{i}",
                    constraint_type=ConstraintType.BUDGET,
                    description=f"Scale test constraint {i}",
                    priority=ConstraintPriority.MEDIUM,
                    threshold_value=0.5 + (i * 0.01)
                )
                for i in range(count)
            ]
            
            start_time = time.time()
            paths = planning_agent.gsm.search_optimal_paths(
                initial_state={"total_assets": 100000, "monthly_income": 5000},
                goal=f"scalability test {count} constraints",
                constraints=constraints,
                time_horizon=24,
                strategies=[SearchStrategy.BALANCED]
            )
            execution_time = time.time() - start_time
            
            results[f"constraints_{count}"] = {
                "execution_time": execution_time,
                "paths_generated": len(paths),
                "time_per_constraint": execution_time / count
            }
        
        return results
    
    def _benchmark_concurrent_execution(self, planning_agent) -> Dict[str, Any]:
        """Benchmark concurrent execution performance"""
        
        import concurrent.futures
        import threading
        
        def planning_task(task_id):
            start_time = time.time()
            paths = planning_agent.gsm.search_optimal_paths(
                initial_state={"total_assets": 100000 + (task_id * 10000), "monthly_income": 5000},
                goal=f"concurrent benchmark {task_id}",
                constraints=[],
                time_horizon=12,
                strategies=[SearchStrategy.BALANCED]
            )
            execution_time = time.time() - start_time
            
            return {
                "task_id": task_id,
                "execution_time": execution_time,
                "paths_generated": len(paths)
            }
        
        # Test with different concurrency levels
        concurrency_levels = [1, 3, 5, 8]
        results = {}
        
        for num_workers in concurrency_levels:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                start_time = time.time()
                
                futures = [
                    executor.submit(planning_task, task_id) 
                    for task_id in range(num_workers)
                ]
                
                task_results = [future.result() for future in concurrent.futures.as_completed(futures)]
                total_time = time.time() - start_time
            
            results[f"workers_{num_workers}"] = {
                "total_time": total_time,
                "average_task_time": sum([r["execution_time"] for r in task_results]) / len(task_results),
                "throughput": len(task_results) / total_time,
                "efficiency": sum([r["execution_time"] for r in task_results]) / total_time
            }
        
        return results
  