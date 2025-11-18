#!/usr/bin/env python3
"""
Debug beam search implementation
"""

from agents.planning_agent import GuidedSearchModule, SearchStrategy, PlanningState
from data_models.schemas import Constraint

def debug_beam_search():
    gsm = GuidedSearchModule()
    
    initial_state = {
        'total_assets': 100000,
        'total_liabilities': 20000,
        'monthly_income': 8000,
        'monthly_expenses': 5000,
        'cash': 15000,
        'investments': 60000,
        'emergency_fund': 25000
    }
    
    print("=== Debugging Beam Search ===")
    
    # Test heuristic calculation
    heuristic_score = gsm._calculate_heuristic_score(
        initial_state, "retirement", [], SearchStrategy.BALANCED
    )
    print(f"Initial heuristic score: {heuristic_score}")
    
    # Create initial planning state
    initial_planning_state = PlanningState(
        state_id="initial",
        financial_state=initial_state.copy(),
        constraints_satisfied=[],
        constraints_violated=[],
        path_cost=0.0,
        heuristic_score=heuristic_score
    )
    
    print(f"Initial planning state total score: {initial_planning_state.total_score}")
    
    # Test action generation
    actions = gsm._generate_possible_actions(initial_planning_state, SearchStrategy.BALANCED, 240)
    print(f"Generated {len(actions)} actions")
    
    # Test node expansion
    from agents.planning_agent import SearchNode
    initial_node = SearchNode(state=initial_planning_state, children=[])
    
    children = gsm._expand_node(initial_node, "retirement", [], SearchStrategy.BALANCED, 240)
    print(f"Expanded to {len(children)} children")
    
    non_pruned = [child for child in children if not child.is_pruned]
    print(f"Non-pruned children: {len(non_pruned)}")
    
    if non_pruned:
        best_child = max(non_pruned, key=lambda n: n.state.total_score)
        print(f"Best child score: {best_child.state.total_score}")
        
        # Test goal state detection
        is_goal = gsm._is_goal_state(best_child.state, "retirement", [])
        print(f"Is goal state: {is_goal}")
    
    # Test full beam search
    print("\n=== Testing Full Beam Search ===")
    beam_paths = gsm._beam_search_optimization(
        initial_state, "retirement", [], 240, SearchStrategy.BALANCED
    )
    print(f"Beam search generated {len(beam_paths)} paths")
    
    if beam_paths:
        for i, path in enumerate(beam_paths):
            print(f"Path {i+1}: {path.path_type}, Score: {path.combined_score:.3f}, Steps: {len(path.sequence_steps)}")

if __name__ == '__main__':
    debug_beam_search()