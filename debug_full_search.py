#!/usr/bin/env python3
"""
Debug full search pipeline
"""

from agents.planning_agent import GuidedSearchModule, SearchStrategy
from data_models.schemas import Constraint

def debug_full_search():
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
    
    constraints = []
    strategies = [SearchStrategy.CONSERVATIVE, SearchStrategy.BALANCED, SearchStrategy.AGGRESSIVE]
    
    print("=== Debugging Full Search Pipeline ===")
    
    # Test strategy exploration
    for strategy in strategies:
        print(f"\n--- Testing {strategy.value} strategy ---")
        
        strategy_paths = gsm._explore_strategy_paths_with_optimization(
            initial_state, "retirement", constraints, 240, strategy, "debug_session"
        )
        print(f"Strategy paths generated: {len(strategy_paths)}")
        
        if strategy_paths:
            for i, path in enumerate(strategy_paths):
                print(f"  Path {i+1}: {path.path_type}, Score: {path.combined_score:.3f}")
    
    # Test full search
    print(f"\n=== Testing Full Search ===")
    all_paths = gsm.search_optimal_paths(
        initial_state=initial_state,
        goal="retirement",
        constraints=constraints,
        time_horizon=240,
        strategies=strategies,
        session_id="debug_full_session"
    )
    
    print(f"Full search generated {len(all_paths)} paths")
    
    if all_paths:
        for i, path in enumerate(all_paths):
            print(f"Path {i+1}: {path.path_type}, Score: {path.combined_score:.3f}, Steps: {len(path.sequence_steps)}")
    else:
        print("No paths generated - investigating...")
        
        # Check session tracking
        if "debug_full_session" in gsm.active_sessions:
            session_data = gsm.active_sessions["debug_full_session"]
            print(f"Session data: {session_data}")

if __name__ == '__main__':
    debug_full_search()