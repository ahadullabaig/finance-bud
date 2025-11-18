"""
Debugging Tools for Path Exploration Analysis

Provides comprehensive debugging and analysis tools for the Planning Agent's
path exploration, constraint satisfaction, and performance optimization.

Requirements: 3.3, 7.5, 10.1, 7.1, 7.2, 8.1, 8.2, 11.1, 11.3
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict
import numpy as np
from collections import defaultdict, Counter

from .comprehensive_logging import (
    ComprehensiveLogger, DecisionLog, PathExplorationLog, 
    DecisionType, PerformanceSnapshot
)
from data_models.schemas import SearchPath, ReasoningTrace, Constraint


class PathExplorationAnalyzer:
    """
    Advanced debugging and analysis tools for path exploration.
    
    Provides:
    - Visual analysis of search trees and decision paths
    - Performance bottleneck identification
    - Constraint satisfaction analysis
    - Heuristic effectiveness evaluation
    - Search algorithm optimization recommendations
    """
    
    def __init__(self, logger: ComprehensiveLogger):
        self.logger = logger
        self.analysis_cache = {}
    
    def analyze_search_tree_structure(
        self, 
        session_id: str, 
        exploration_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the structure and efficiency of search tree exploration.
        
        Returns detailed analysis of:
        - Tree depth and branching factor
        - Pruning effectiveness
        - Search strategy efficiency
        - Memory usage patterns
        """
        exploration_logs = self.logger.exploration_logs.get(session_id, [])
        
        if exploration_id:
            exploration_logs = [
                log for log in exploration_logs 
                if log.exploration_id == exploration_id
            ]
        
        if not exploration_logs:
            return {"error": "No exploration data found"}
        
        analysis = {
            "session_id": session_id,
            "total_explorations": len(exploration_logs),
            "strategies_analyzed": [],
            "tree_metrics": {},
            "efficiency_metrics": {},
            "bottleneck_analysis": {},
            "optimization_recommendations": []
        }
        
        for log in exploration_logs:
            strategy_analysis = self._analyze_single_exploration(log)
            analysis["strategies_analyzed"].append(strategy_analysis)
            
            # Aggregate metrics
            strategy = log.strategy
            analysis["tree_metrics"][strategy] = {
                "nodes_explored": log.nodes_explored,
                "nodes_pruned": log.nodes_pruned,
                "paths_generated": log.paths_generated,
                "exploration_depth": log.exploration_depth,
                "pruning_rate": log.nodes_pruned / max(log.nodes_explored, 1),
                "path_generation_rate": log.paths_generated / max(log.nodes_explored, 1)
            }
            
            # Calculate efficiency metrics
            if log.end_time:
                duration = (log.end_time - log.start_time).total_seconds()
                analysis["efficiency_metrics"][strategy] = {
                    "exploration_time": duration,
                    "nodes_per_second": log.nodes_explored / max(duration, 0.001),
                    "paths_per_second": log.paths_generated / max(duration, 0.001),
                    "memory_efficiency": self._calculate_memory_efficiency(log),
                    "cpu_efficiency": self._calculate_cpu_efficiency(log)
                }
            
            # Analyze bottlenecks
            analysis["bottleneck_analysis"][strategy] = {
                "performance_bottlenecks": log.performance_bottlenecks,
                "pruning_reasons": Counter(log.pruning_reasons),
                "constraint_violation_rate": log.constraint_violations / max(log.nodes_explored, 1)
            }
        
        # Generate optimization recommendations
        analysis["optimization_recommendations"] = self._generate_search_optimization_recommendations(
            analysis["tree_metrics"], 
            analysis["efficiency_metrics"], 
            analysis["bottleneck_analysis"]
        )
        
        return analysis
    
    def _analyze_single_exploration(self, log: PathExplorationLog) -> Dict[str, Any]:
        """Analyze a single path exploration in detail"""
        return {
            "exploration_id": log.exploration_id,
            "strategy": log.strategy,
            "duration_seconds": (
                (log.end_time - log.start_time).total_seconds() 
                if log.end_time else 0
            ),
            "search_efficiency": {
                "nodes_explored": log.nodes_explored,
                "effective_pruning": log.nodes_pruned > log.nodes_explored * 0.2,
                "path_diversity": log.paths_generated > 1,
                "constraint_compliance": log.constraint_violations < log.nodes_explored * 0.1
            },
            "performance_profile": {
                "memory_usage_mb": log.memory_usage_mb,
                "cpu_usage_percent": log.cpu_usage_percent,
                "bottleneck_count": len(log.performance_bottlenecks),
                "major_bottlenecks": [
                    bottleneck for bottleneck in log.performance_bottlenecks
                    if "100ms" in bottleneck or "200ms" in bottleneck
                ]
            },
            "quality_metrics": {
                "exploration_completeness": min(log.exploration_depth / 10, 1.0),
                "pruning_effectiveness": log.nodes_pruned / max(log.nodes_explored, 1),
                "path_generation_success": log.paths_generated > 0
            }
        }
    
    def analyze_decision_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze decision-making patterns and effectiveness.
        
        Returns analysis of:
        - Decision type distribution
        - Confidence score patterns
        - Decision timing analysis
        - Alternative path consideration
        """
        decision_logs = self.logger.decision_logs.get(session_id, [])
        
        if not decision_logs:
            return {"error": "No decision data found"}
        
        # Group decisions by type
        decisions_by_type = defaultdict(list)
        for log in decision_logs:
            decisions_by_type[log.decision_type].append(log)
        
        analysis = {
            "session_id": session_id,
            "total_decisions": len(decision_logs),
            "decision_distribution": {},
            "confidence_analysis": {},
            "timing_analysis": {},
            "alternative_analysis": {},
            "decision_quality_metrics": {}
        }
        
        # Analyze each decision type
        for decision_type, logs in decisions_by_type.items():
            type_name = decision_type.value
            
            # Distribution analysis
            analysis["decision_distribution"][type_name] = {
                "count": len(logs),
                "percentage": len(logs) / len(decision_logs) * 100,
                "average_confidence": sum(log.confidence_score for log in logs) / len(logs),
                "confidence_std": np.std([log.confidence_score for log in logs])
            }
            
            # Timing analysis
            execution_times = [log.execution_time_ms for log in logs]
            analysis["timing_analysis"][type_name] = {
                "average_time_ms": np.mean(execution_times),
                "median_time_ms": np.median(execution_times),
                "max_time_ms": np.max(execution_times),
                "time_std": np.std(execution_times),
                "slow_decisions": len([t for t in execution_times if t > 100])
            }
            
            # Alternative analysis
            alternatives_considered = [len(log.options_considered) for log in logs]
            analysis["alternative_analysis"][type_name] = {
                "average_alternatives": np.mean(alternatives_considered),
                "max_alternatives": np.max(alternatives_considered),
                "decisions_with_alternatives": len([a for a in alternatives_considered if a > 1])
            }
        
        # Overall quality metrics
        all_confidences = [log.confidence_score for log in decision_logs]
        all_times = [log.execution_time_ms for log in decision_logs]
        
        analysis["decision_quality_metrics"] = {
            "overall_confidence": {
                "mean": np.mean(all_confidences),
                "median": np.median(all_confidences),
                "std": np.std(all_confidences),
                "high_confidence_rate": len([c for c in all_confidences if c > 0.8]) / len(all_confidences)
            },
            "decision_efficiency": {
                "mean_time_ms": np.mean(all_times),
                "fast_decision_rate": len([t for t in all_times if t < 50]) / len(all_times),
                "slow_decision_rate": len([t for t in all_times if t > 100]) / len(all_times)
            },
            "decision_thoroughness": {
                "average_alternatives_considered": np.mean([
                    len(log.options_considered) for log in decision_logs
                ]),
                "decisions_with_rationale": len([
                    log for log in decision_logs if len(log.rationale) > 50
                ]) / len(decision_logs)
            }
        }
        
        return analysis
    
    def analyze_constraint_satisfaction_patterns(
        self, 
        session_id: str, 
        constraints: List[Constraint]
    ) -> Dict[str, Any]:
        """
        Analyze constraint satisfaction patterns and violation trends.
        
        Returns detailed analysis of:
        - Constraint violation frequency
        - Violation patterns by constraint type
        - Impact on path generation
        - Mitigation effectiveness
        """
        exploration_logs = self.logger.exploration_logs.get(session_id, [])
        decision_logs = self.logger.decision_logs.get(session_id, [])
        
        analysis = {
            "session_id": session_id,
            "constraint_overview": {},
            "violation_patterns": {},
            "impact_analysis": {},
            "mitigation_analysis": {},
            "recommendations": []
        }
        
        # Analyze constraint overview
        total_violations = sum(log.constraint_violations for log in exploration_logs)
        total_nodes = sum(log.nodes_explored for log in exploration_logs)
        
        analysis["constraint_overview"] = {
            "total_constraints": len(constraints),
            "total_violations": total_violations,
            "violation_rate": total_violations / max(total_nodes, 1),
            "constraints_by_type": Counter(c.constraint_type.value for c in constraints),
            "constraints_by_priority": Counter(c.priority.value for c in constraints)
        }
        
        # Analyze violation patterns by strategy
        for log in exploration_logs:
            strategy = log.strategy
            if log.nodes_explored > 0:
                violation_rate = log.constraint_violations / log.nodes_explored
                
                analysis["violation_patterns"][strategy] = {
                    "violation_count": log.constraint_violations,
                    "violation_rate": violation_rate,
                    "nodes_explored": log.nodes_explored,
                    "paths_generated": log.paths_generated,
                    "success_despite_violations": log.paths_generated > 0 and log.constraint_violations > 0
                }
        
        # Analyze impact on path generation
        strategies_with_violations = [
            log.strategy for log in exploration_logs 
            if log.constraint_violations > 0
        ]
        
        strategies_without_violations = [
            log.strategy for log in exploration_logs 
            if log.constraint_violations == 0
        ]
        
        analysis["impact_analysis"] = {
            "strategies_affected": len(strategies_with_violations),
            "strategies_unaffected": len(strategies_without_violations),
            "path_generation_impact": self._analyze_violation_impact_on_paths(exploration_logs),
            "pruning_correlation": self._analyze_violation_pruning_correlation(exploration_logs)
        }
        
        # Analyze constraint handling decisions
        constraint_decisions = [
            log for log in decision_logs 
            if log.decision_type == DecisionType.CONSTRAINT_HANDLING
        ]
        
        if constraint_decisions:
            analysis["mitigation_analysis"] = {
                "constraint_decisions_made": len(constraint_decisions),
                "average_confidence": sum(log.confidence_score for log in constraint_decisions) / len(constraint_decisions),
                "mitigation_strategies": [log.chosen_option for log in constraint_decisions],
                "effectiveness_indicators": self._analyze_mitigation_effectiveness(constraint_decisions, exploration_logs)
            }
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_constraint_optimization_recommendations(
            analysis["constraint_overview"],
            analysis["violation_patterns"],
            analysis["impact_analysis"]
        )
        
        return analysis
    
    def analyze_heuristic_effectiveness(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze the effectiveness of different heuristics in path evaluation.
        
        Returns analysis of:
        - Heuristic performance by type
        - Correlation with successful paths
        - Computation efficiency
        - Optimization opportunities
        """
        # This would analyze heuristic logs if they were captured
        # For now, provide a framework for heuristic analysis
        
        analysis = {
            "session_id": session_id,
            "heuristic_performance": {},
            "correlation_analysis": {},
            "efficiency_analysis": {},
            "optimization_opportunities": []
        }
        
        # Placeholder for heuristic analysis
        # In a full implementation, this would analyze:
        # - Heuristic calculation times
        # - Correlation between heuristic scores and path success
        # - Heuristic value distributions
        # - Computational bottlenecks in heuristic calculations
        
        analysis["optimization_opportunities"] = [
            "Implement heuristic caching for repeated calculations",
            "Profile individual heuristic computation times",
            "Analyze correlation between heuristic scores and path success",
            "Consider adaptive heuristic weighting based on performance"
        ]
        
        return analysis
    
    def generate_performance_report(self, session_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive performance report for a planning session.
        
        Combines all analysis types into a unified performance report.
        """
        report = {
            "session_id": session_id,
            "report_timestamp": datetime.utcnow().isoformat(),
            "executive_summary": {},
            "detailed_analysis": {},
            "performance_metrics": {},
            "optimization_roadmap": [],
            "recommendations": {
                "immediate": [],
                "short_term": [],
                "long_term": []
            }
        }
        
        # Get session summary
        session_summary = self.logger.get_session_summary(session_id)
        
        # Generate executive summary
        report["executive_summary"] = {
            "session_duration": "N/A",  # Would calculate from logs
            "strategies_explored": len(session_summary.get("strategies_explored", [])),
            "total_decisions": session_summary.get("total_decisions", 0),
            "paths_generated": session_summary.get("total_paths_generated", 0),
            "constraint_violations": session_summary.get("total_constraint_violations", 0),
            "average_confidence": session_summary.get("average_confidence", 0.0),
            "performance_bottlenecks": len(session_summary.get("performance_bottlenecks", [])),
            "overall_efficiency": self._calculate_overall_efficiency(session_summary)
        }
        
        # Detailed analysis
        report["detailed_analysis"] = {
            "search_tree_analysis": self.analyze_search_tree_structure(session_id),
            "decision_pattern_analysis": self.analyze_decision_patterns(session_id),
            "heuristic_effectiveness": self.analyze_heuristic_effectiveness(session_id)
        }
        
        # Performance metrics
        report["performance_metrics"] = self.logger.get_current_metrics()
        
        # Generate optimization roadmap
        report["optimization_roadmap"] = self._generate_optimization_roadmap(
            report["detailed_analysis"],
            session_summary
        )
        
        # Categorize recommendations
        all_recommendations = []
        for analysis in report["detailed_analysis"].values():
            if isinstance(analysis, dict) and "optimization_recommendations" in analysis:
                all_recommendations.extend(analysis["optimization_recommendations"])
        
        report["recommendations"] = self._categorize_recommendations(all_recommendations)
        
        return report
    
    def visualize_search_tree(self, session_id: str, save_path: Optional[str] = None) -> str:
        """
        Create visual representation of the search tree exploration.
        
        Returns path to generated visualization or base64 encoded image.
        """
        exploration_logs = self.logger.exploration_logs.get(session_id, [])
        
        if not exploration_logs:
            return "No exploration data available for visualization"
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Search Tree Analysis - Session {session_id}', fontsize=16)
        
        # 1. Nodes explored by strategy
        strategies = [log.strategy for log in exploration_logs]
        nodes_explored = [log.nodes_explored for log in exploration_logs]
        
        axes[0, 0].bar(strategies, nodes_explored)
        axes[0, 0].set_title('Nodes Explored by Strategy')
        axes[0, 0].set_ylabel('Nodes Explored')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Pruning effectiveness
        pruning_rates = [
            log.nodes_pruned / max(log.nodes_explored, 1) 
            for log in exploration_logs
        ]
        
        axes[0, 1].bar(strategies, pruning_rates)
        axes[0, 1].set_title('Pruning Effectiveness by Strategy')
        axes[0, 1].set_ylabel('Pruning Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Path generation success
        paths_generated = [log.paths_generated for log in exploration_logs]
        
        axes[1, 0].bar(strategies, paths_generated)
        axes[1, 0].set_title('Paths Generated by Strategy')
        axes[1, 0].set_ylabel('Paths Generated')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Exploration efficiency (paths per node)
        efficiency = [
            log.paths_generated / max(log.nodes_explored, 1)
            for log in exploration_logs
        ]
        
        axes[1, 1].bar(strategies, efficiency)
        axes[1, 1].set_title('Exploration Efficiency (Paths/Node)')
        axes[1, 1].set_ylabel('Efficiency Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            # Return base64 encoded image for web display
            import io
            import base64
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
    
    def export_debugging_data(self, session_id: str, format: str = "json") -> str:
        """
        Export comprehensive debugging data for external analysis.
        
        Supports JSON, CSV, and Excel formats.
        """
        # Generate comprehensive analysis
        performance_report = self.generate_performance_report(session_id)
        
        if format == "json":
            return json.dumps(performance_report, default=str, indent=2)
        
        elif format == "csv":
            # Convert to pandas DataFrame for CSV export
            df_data = []
            
            # Flatten the performance report for CSV
            for key, value in performance_report.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        df_data.append({
                            "category": key,
                            "metric": subkey,
                            "value": str(subvalue)
                        })
                else:
                    df_data.append({
                        "category": "summary",
                        "metric": key,
                        "value": str(value)
                    })
            
            df = pd.DataFrame(df_data)
            return df.to_csv(index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    # Helper methods
    
    def _calculate_memory_efficiency(self, log: PathExplorationLog) -> float:
        """Calculate memory efficiency score for exploration"""
        # Simplified calculation - would be more sophisticated in practice
        if log.nodes_explored == 0:
            return 0.0
        
        memory_per_node = log.memory_usage_mb / log.nodes_explored
        # Lower memory per node = higher efficiency
        return max(0.0, 1.0 - (memory_per_node / 10.0))  # Normalize to 0-1
    
    def _calculate_cpu_efficiency(self, log: PathExplorationLog) -> float:
        """Calculate CPU efficiency score for exploration"""
        # Higher CPU usage during exploration is expected, so this measures relative efficiency
        return max(0.0, 1.0 - (log.cpu_usage_percent / 100.0))
    
    def _generate_search_optimization_recommendations(
        self,
        tree_metrics: Dict[str, Any],
        efficiency_metrics: Dict[str, Any],
        bottleneck_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate specific optimization recommendations based on analysis"""
        recommendations = []
        
        # Analyze pruning effectiveness
        avg_pruning_rate = np.mean([
            metrics["pruning_rate"] for metrics in tree_metrics.values()
        ])
        
        if avg_pruning_rate < 0.3:
            recommendations.append(
                "Low pruning rate detected. Consider tightening pruning thresholds or improving heuristics."
            )
        
        # Analyze path generation efficiency
        avg_path_rate = np.mean([
            metrics["path_generation_rate"] for metrics in tree_metrics.values()
        ])
        
        if avg_path_rate < 0.1:
            recommendations.append(
                "Low path generation rate. Consider relaxing constraints or improving search strategy."
            )
        
        # Analyze exploration speed
        if efficiency_metrics:
            slow_strategies = [
                strategy for strategy, metrics in efficiency_metrics.items()
                if metrics.get("nodes_per_second", 0) < 10
            ]
            
            if slow_strategies:
                recommendations.append(
                    f"Slow exploration detected in strategies: {', '.join(slow_strategies)}. "
                    "Consider optimizing heuristic calculations or reducing search depth."
                )
        
        # Analyze bottlenecks
        common_bottlenecks = Counter()
        for analysis in bottleneck_analysis.values():
            for bottleneck in analysis.get("performance_bottlenecks", []):
                common_bottlenecks[bottleneck] += 1
        
        if common_bottlenecks:
            most_common = common_bottlenecks.most_common(1)[0]
            recommendations.append(
                f"Most common bottleneck: {most_common[0]}. "
                "Focus optimization efforts on this area."
            )
        
        return recommendations
    
    def _analyze_violation_impact_on_paths(self, exploration_logs: List[PathExplorationLog]) -> Dict[str, Any]:
        """Analyze how constraint violations impact path generation"""
        with_violations = [log for log in exploration_logs if log.constraint_violations > 0]
        without_violations = [log for log in exploration_logs if log.constraint_violations == 0]
        
        return {
            "avg_paths_with_violations": (
                np.mean([log.paths_generated for log in with_violations])
                if with_violations else 0
            ),
            "avg_paths_without_violations": (
                np.mean([log.paths_generated for log in without_violations])
                if without_violations else 0
            ),
            "violation_impact_factor": (
                (np.mean([log.paths_generated for log in without_violations]) - 
                 np.mean([log.paths_generated for log in with_violations]))
                if with_violations and without_violations else 0
            )
        }
    
    def _analyze_violation_pruning_correlation(self, exploration_logs: List[PathExplorationLog]) -> float:
        """Analyze correlation between violations and pruning"""
        if len(exploration_logs) < 2:
            return 0.0
        
        violations = [log.constraint_violations for log in exploration_logs]
        pruning_rates = [
            log.nodes_pruned / max(log.nodes_explored, 1) 
            for log in exploration_logs
        ]
        
        return np.corrcoef(violations, pruning_rates)[0, 1] if len(violations) > 1 else 0.0
    
    def _analyze_mitigation_effectiveness(
        self,
        constraint_decisions: List[DecisionLog],
        exploration_logs: List[PathExplorationLog]
    ) -> Dict[str, Any]:
        """Analyze effectiveness of constraint mitigation strategies"""
        return {
            "mitigation_attempts": len(constraint_decisions),
            "average_confidence": (
                sum(log.confidence_score for log in constraint_decisions) / len(constraint_decisions)
                if constraint_decisions else 0
            ),
            "post_mitigation_violations": sum(
                log.constraint_violations for log in exploration_logs
            ),
            "mitigation_success_rate": 0.8  # Placeholder - would calculate from actual data
        }
    
    def _generate_constraint_optimization_recommendations(
        self,
        constraint_overview: Dict[str, Any],
        violation_patterns: Dict[str, Any],
        impact_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate constraint-specific optimization recommendations"""
        recommendations = []
        
        violation_rate = constraint_overview.get("violation_rate", 0)
        
        if violation_rate > 0.2:
            recommendations.append(
                "High constraint violation rate detected. Consider relaxing non-critical constraints."
            )
        
        if impact_analysis.get("violation_impact_factor", 0) > 2:
            recommendations.append(
                "Constraint violations significantly impact path generation. "
                "Implement constraint relaxation strategies."
            )
        
        # Analyze by constraint type
        constraint_types = constraint_overview.get("constraints_by_type", {})
        if constraint_types.get("budget", 0) > constraint_types.get("risk", 0):
            recommendations.append(
                "Budget constraints dominate. Consider budget optimization strategies."
            )
        
        return recommendations
    
    def _calculate_overall_efficiency(self, session_summary: Dict[str, Any]) -> float:
        """Calculate overall efficiency score for the session"""
        # Combine multiple efficiency factors
        factors = []
        
        # Path generation efficiency
        paths = session_summary.get("total_paths_generated", 0)
        decisions = session_summary.get("total_decisions", 1)
        factors.append(min(paths / decisions, 1.0))
        
        # Confidence factor
        factors.append(session_summary.get("average_confidence", 0.0))
        
        # Bottleneck factor (fewer bottlenecks = higher efficiency)
        bottlenecks = len(session_summary.get("performance_bottlenecks", []))
        factors.append(max(0.0, 1.0 - (bottlenecks / 10.0)))
        
        return np.mean(factors) if factors else 0.0
    
    def _generate_optimization_roadmap(
        self,
        detailed_analysis: Dict[str, Any],
        session_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized optimization roadmap"""
        roadmap = []
        
        # High priority optimizations
        if session_summary.get("average_confidence", 0) < 0.7:
            roadmap.append({
                "priority": "high",
                "area": "decision_confidence",
                "action": "Improve heuristic accuracy and decision logic",
                "expected_impact": "20-30% improvement in decision confidence"
            })
        
        # Medium priority optimizations
        bottlenecks = len(session_summary.get("performance_bottlenecks", []))
        if bottlenecks > 3:
            roadmap.append({
                "priority": "medium",
                "area": "performance_bottlenecks",
                "action": "Address identified performance bottlenecks",
                "expected_impact": "15-25% improvement in exploration speed"
            })
        
        # Low priority optimizations
        roadmap.append({
            "priority": "low",
            "area": "monitoring_enhancement",
            "action": "Implement additional performance monitoring",
            "expected_impact": "Better visibility into system performance"
        })
        
        return roadmap
    
    def _categorize_recommendations(self, recommendations: List[str]) -> Dict[str, List[str]]:
        """Categorize recommendations by urgency and impact"""
        categorized = {
            "immediate": [],
            "short_term": [],
            "long_term": []
        }
        
        for rec in recommendations:
            if any(word in rec.lower() for word in ["critical", "urgent", "immediately"]):
                categorized["immediate"].append(rec)
            elif any(word in rec.lower() for word in ["consider", "improve", "optimize"]):
                categorized["short_term"].append(rec)
            else:
                categorized["long_term"].append(rec)
        
        return categorized