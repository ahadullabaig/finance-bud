"""
ReasonGraph Data Mapper

Converts planning and verification data into ReasonGraph visualization format.
Person D - Task 21
"""

from typing import Dict, List, Any
from datetime import datetime


class ReasonGraphMapper:
    """Maps agent reasoning traces to ReasonGraph visualization format"""
    
    @staticmethod
    def map_planning_trace(planning_response: Dict) -> Dict:
        """
        Convert planning agent response to ReasonGraph format
        
        Args:
            planning_response: Response from planning agent with search paths
            
        Returns:
            ReasonGraph data structure with nodes and edges
        """
        nodes = []
        edges = []
        
        # Add root node for user goal
        root_id = "goal_root"
        nodes.append({
            "id": root_id,
            "type": "planning",
            "label": "User Goal",
            "status": "explored",
            "confidence": 1.0,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Add nodes for each search path
        search_paths = planning_response.get("search_paths", [])
        for i, path in enumerate(search_paths):
            path_id = f"path_{i}"
            strategy = path.get("strategy", f"Strategy {i+1}")
            combined_score = path.get("combined_score", 0.5)
            status = path.get("status", "explored")
            
            nodes.append({
                "id": path_id,
                "type": "alternative",
                "label": strategy,
                "status": status,
                "confidence": combined_score,
                "rationale": f"Risk: {path.get('risk_score', 0):.2f}, Return: {path.get('expected_return', 0):.2%}",
                "metadata": path
            })
            
            edges.append({
                "source": root_id,
                "target": path_id,
                "weight": combined_score
            })
        
        # Add selected strategy node
        selected_strategy = planning_response.get("selected_strategy")
        if selected_strategy:
            selected_id = "selected_strategy"
            nodes.append({
                "id": selected_id,
                "type": "decision",
                "label": f"Selected: {selected_strategy}",
                "status": "approved",
                "confidence": planning_response.get("confidence_score", 0.8)
            })
            
            # Find and connect the selected path
            for i, path in enumerate(search_paths):
                if path.get("strategy") == selected_strategy:
                    edges.append({
                        "source": f"path_{i}",
                        "target": selected_id,
                        "weight": 1.0
                    })
                    break
        
        # Add plan steps as nodes
        plan_steps = planning_response.get("plan_steps", [])
        prev_step_id = selected_id if selected_strategy else root_id
        
        for i, step in enumerate(plan_steps):
            step_id = f"step_{i}"
            nodes.append({
                "id": step_id,
                "type": "planning",
                "label": step.get("action_type", f"Step {i+1}"),
                "status": "explored",
                "confidence": step.get("confidence_score", 0.8),
                "rationale": step.get("rationale", ""),
                "metadata": step
            })
            
            edges.append({
                "source": prev_step_id,
                "target": step_id,
                "weight": 1.0
            })
            
            prev_step_id = step_id
        
        return {
            "nodes": nodes,
            "edges": edges,
            "session_id": planning_response.get("session_id", "unknown"),
            "plan_id": planning_response.get("plan_id")
        }
    
    @staticmethod
    def map_verification_trace(verification_response: Dict, planning_graph: Dict = None) -> Dict:
        """
        Convert verification agent response to ReasonGraph format
        
        Args:
            verification_response: Response from verification agent
            planning_graph: Optional existing planning graph to extend
            
        Returns:
            ReasonGraph data structure with verification nodes
        """
        if planning_graph:
            nodes = planning_graph["nodes"].copy()
            edges = planning_graph["edges"].copy()
        else:
            nodes = []
            edges = []
        
        verification_report = verification_response.get("verification_report", {})
        step_results = verification_response.get("step_results", [])
        
        # Add verification root node
        verification_id = "verification_root"
        status_map = {
            "approved": "approved",
            "rejected": "rejected",
            "conditional": "conditional"
        }
        
        verification_status = verification_report.get("verification_status", "pending")
        mapped_status = status_map.get(verification_status.lower(), "explored")
        
        nodes.append({
            "id": verification_id,
            "type": "verification",
            "label": "Verification Check",
            "status": mapped_status,
            "confidence": verification_report.get("confidence_score", 0.8),
            "rationale": verification_report.get("approval_rationale", ""),
            "metadata": verification_report
        })
        
        # Connect verification to plan steps
        for i, step_result in enumerate(step_results):
            step_id = f"step_{i}"
            verification_step_id = f"verification_step_{i}"
            
            # Determine status based on violations
            violations = step_result.get("violations", [])
            if not violations:
                step_status = "approved"
            elif any(v.get("severity") in ["critical", "high"] for v in violations):
                step_status = "rejected"
            else:
                step_status = "conditional"
            
            nodes.append({
                "id": verification_step_id,
                "type": "verification",
                "label": f"Verify: {step_result.get('action_type', f'Step {i+1}')}",
                "status": step_status,
                "confidence": step_result.get("compliance_score", 0.8),
                "rationale": f"{len(violations)} violations found" if violations else "All checks passed",
                "metadata": step_result
            })
            
            # Connect to corresponding plan step if it exists
            if any(n["id"] == step_id for n in nodes):
                edges.append({
                    "source": step_id,
                    "target": verification_step_id,
                    "weight": 1.0
                })
            
            # Connect to verification root
            edges.append({
                "source": verification_step_id,
                "target": verification_id,
                "weight": step_result.get("compliance_score", 0.8)
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "session_id": verification_response.get("session_id", "unknown"),
            "plan_id": verification_report.get("plan_id")
        }
    
    @staticmethod
    def map_cmvl_trace(cmvl_response: Dict, existing_graph: Dict = None) -> Dict:
        """
        Convert CMVL trigger and response to ReasonGraph format
        
        Args:
            cmvl_response: Response from CMVL trigger
            existing_graph: Optional existing graph to extend
            
        Returns:
            ReasonGraph data structure with CMVL nodes
        """
        if existing_graph:
            nodes = existing_graph["nodes"].copy()
            edges = existing_graph["edges"].copy()
        else:
            nodes = []
            edges = []
        
        # Add CMVL trigger node
        cmvl_id = f"cmvl_{cmvl_response.get('cmvl_id', 'unknown')}"
        nodes.append({
            "id": cmvl_id,
            "type": "verification",
            "label": "CMVL Trigger",
            "status": "conditional",
            "confidence": 0.9,
            "rationale": f"Severity: {cmvl_response.get('trigger_severity', 'unknown')}",
            "metadata": cmvl_response
        })
        
        # Add verification action nodes
        verification_actions = cmvl_response.get("verification_actions", [])
        for i, action in enumerate(verification_actions):
            action_id = f"{cmvl_id}_action_{i}"
            nodes.append({
                "id": action_id,
                "type": "verification",
                "label": action.replace("_", " ").title(),
                "status": "explored",
                "confidence": 0.85
            })
            
            edges.append({
                "source": cmvl_id,
                "target": action_id,
                "weight": 1.0
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "session_id": cmvl_response.get("session_id", "unknown"),
            "cmvl_id": cmvl_response.get("cmvl_id")
        }
    
    @staticmethod
    def merge_graphs(*graphs: Dict) -> Dict:
        """
        Merge multiple ReasonGraph structures
        
        Args:
            *graphs: Variable number of graph dictionaries
            
        Returns:
            Merged ReasonGraph structure
        """
        merged_nodes = []
        merged_edges = []
        seen_node_ids = set()
        seen_edge_pairs = set()
        
        session_id = None
        plan_id = None
        
        for graph in graphs:
            if not graph:
                continue
            
            # Merge nodes (avoid duplicates)
            for node in graph.get("nodes", []):
                if node["id"] not in seen_node_ids:
                    merged_nodes.append(node)
                    seen_node_ids.add(node["id"])
            
            # Merge edges (avoid duplicates)
            for edge in graph.get("edges", []):
                edge_pair = (edge["source"], edge["target"])
                if edge_pair not in seen_edge_pairs:
                    merged_edges.append(edge)
                    seen_edge_pairs.add(edge_pair)
            
            # Capture session and plan IDs
            if not session_id:
                session_id = graph.get("session_id")
            if not plan_id:
                plan_id = graph.get("plan_id")
        
        return {
            "nodes": merged_nodes,
            "edges": merged_edges,
            "session_id": session_id or "unknown",
            "plan_id": plan_id
        }
