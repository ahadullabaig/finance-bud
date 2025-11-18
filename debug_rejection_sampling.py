#!/usr/bin/env python3
"""
Debug rejection sampling
"""

from agents.planning_agent import GuidedSearchModule, SearchStrategy
from data_models.schemas import Constraint

def debug_rejection_sampling():
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
    
    print("=== Debugging Rejection Sampling ===")
    
    # Generate some paths first
    all_paths = []
    for strategy in [SearchStrategy.BALANCED, SearchStrategy.AGGRESSIVE]:
        strategy_paths = gsm._explore_strategy_paths_with_optimization(
            initial_state, "retirement", constraints, 240, strategy, "debug_session"
        )
        all_paths.extend(strategy_paths)
    
    print(f"Generated {len(all_paths)} paths before rejection sampling")
    
    # Test rejection sampling
    filtered_paths = gsm._apply_rejection_sampling(all_paths, constraints)
    print(f"After rejection sampling: {len(filtered_paths)} paths")
    
    # Test individual path violation prediction
    if all_paths:
        test_path = all_paths[0]
        violation_prob = gsm._predict_constraint_violations(test_path, constraints)
        print(f"Violation probability for test path: {violation_prob}")
        
        accept = gsm._rejection_sampling_accept(violation_prob)
        print(f"Rejection sampling accept: {accept}")
    
    # Test ranking
    if filtered_paths:
        ranked_paths = gsm._rank_paths_with_sequence_optimization(filtered_paths, constraints, 240)
        print(f"After ranking: {len(ranked_paths)} paths")
        
        for i, path in enumerate(ranked_paths[:3]):
            print(f"Ranked path {i+1}: {path.path_type}, Score: {path.combined_score:.3f}")

if __name__ == '__main__':
    debug_rejection_sampling()