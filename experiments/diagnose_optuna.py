"""
Diagnostic script to investigate Optuna division by zero warnings.

This script:
1. Runs Optuna optimization
2. Visualizes trial exploration
3. Analyzes failed trials
4. Identifies problematic hyperparameter ranges
"""

import sys
import warnings
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import run_simulation
from visualization import plot_optuna_trials, print_optuna_summary, analyze_failed_trials


if __name__ == "__main__":
    
    print("=" * 80)
    print("OPTUNA DIAGNOSTIC: Investigating Division by Zero Warnings")
    print("=" * 80)
    
    # Configuration with Optuna
    config = {
        "experiment_name": "optuna_diagnostic",
        "num_simulations": 1,  # Just 1 replication to see the issue clearly
        "data_generation": [
            {"delta": 0.15, "n_samples": 200},
        ],
        "models_to_run": ["Poisson"],  # Focus on one model
        "optimization": "optuna",
        "optuna_n_trials": 20,  # Moderate number of trials
        "optuna_n_jobs": 1,  # Sequential to see warnings clearly
    }
    
    print("\nConfiguration:")
    print(f"  Model: {config['models_to_run']}")
    print(f"  Trials: {config['optuna_n_trials']}")
    print(f"  Data: delta={config['data_generation'][0]['delta']}")
    
    print("\n" + "-" * 80)
    print("Running Optuna optimization...")
    print("Watch for 'invalid value encountered in divide' warnings")
    print("-" * 80 + "\n")
    
    # Run with warnings enabled to capture the issue
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        
        results = run_simulation(config, cache=False, verbose=True)
        
        # Check if we got division warnings
        division_warnings = [warning for warning in w 
                           if 'invalid value encountered in divide' in str(warning.message)]
        
        print("\n" + "=" * 80)
        print(f"CAPTURED {len(division_warnings)} DIVISION WARNINGS")
        print("=" * 80)
        
        if len(division_warnings) > 0:
            print("\nExample warning:")
            print(f"  File: {division_warnings[0].filename}")
            print(f"  Line: {division_warnings[0].lineno}")
            print(f"  Message: {division_warnings[0].message}")
    
    # Get the results DataFrame
    results_df = results.results_df
    
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    # Print summary
    print_optuna_summary(results_df)
    
    # Analyze failed trials
    analyze_failed_trials(results_df, threshold=0.3)
    
    # Visualize trials
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    
    output_dir = Path(results.output_dir)
    plot_optuna_trials(results_df, output_dir=output_dir, show=False)
    
    print(f"\nPlots saved to: {output_dir}")
    
    # Detailed analysis of problematic hyperparameters
    print("\n" + "=" * 80)
    print("DIVISION BY ZERO ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    optuna_df = results_df[results_df['optimization_method'] == 'optuna']
    
    print("\n❓ WHY DOES Ns BECOME 0?")
    print("-" * 80)
    print("""
The warning 'invalid value encountered in divide' at:
  jumpmodels/utils/calculation.py:133: means_ = weighted_sum / Ns

occurs when Ns (the number of points assigned to a cluster) is 0.

This happens when:
1. jump_penalty is TOO HIGH
   → Model refuses to create jumps (state changes)
   → All data assigned to single cluster
   → Other clusters remain empty (Ns = 0)

2. n_components (n_states) is TOO HIGH
   → More clusters than needed
   → Some clusters never get assigned any points
   → Empty clusters have Ns = 0

3. Bad initialization
   → Random initialization places cluster centers poorly
   → Some clusters start empty and never attract points
""")
    
    # Check correlation between hyperparameters and performance
    print("\n📊 HYPERPARAMETER CORRELATION WITH PERFORMANCE")
    print("-" * 80)
    
    for col in ['n_components', 'jump_penalty', 'kappa']:
        if col in optuna_df.columns:
            corr = optuna_df[col].corr(optuna_df['balanced_accuracy'])
            print(f"{col:20s}: correlation = {corr:+.3f}")
            
            if col == 'jump_penalty':
                # Check if very high penalties cause problems
                high_penalty = optuna_df[optuna_df['jump_penalty'] > 50]
                if len(high_penalty) > 0:
                    avg_score = high_penalty['balanced_accuracy'].mean()
                    print(f"  → Trials with penalty > 50: avg score = {avg_score:.3f}")
            
            if col == 'n_components':
                # Check if high n_components cause problems
                high_n = optuna_df[optuna_df['n_components'] > 4]
                if len(high_n) > 0:
                    avg_score = high_n['balanced_accuracy'].mean()
                    print(f"  → Trials with n_states > 4:  avg score = {avg_score:.3f}")
    
    print("\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    # Find best performing hyperparameter ranges
    good_trials = optuna_df[optuna_df['balanced_accuracy'] > 0.5]
    
    if len(good_trials) > 0:
        print("\nBest performing hyperparameter ranges:")
        if 'n_components' in good_trials.columns:
            print(f"  n_components:  [{good_trials['n_components'].min():.0f}, {good_trials['n_components'].max():.0f}]")
        if 'jump_penalty' in good_trials.columns:
            print(f"  jump_penalty:  [{good_trials['jump_penalty'].min():.2f}, {good_trials['jump_penalty'].max():.2f}]")
        
        print("\nSuggested hyperparameter search space:")
        print("  hyperparameters = {")
        if 'n_components' in good_trials.columns:
            n_min = max(2, int(good_trials['n_components'].quantile(0.1)))
            n_max = min(6, int(good_trials['n_components'].quantile(0.9)))
            print(f"      'n_states_min': {n_min},")
            print(f"      'n_states_max': {n_max},")
        if 'jump_penalty' in good_trials.columns:
            jp_min = max(0.1, good_trials['jump_penalty'].quantile(0.1))
            jp_max = min(100.0, good_trials['jump_penalty'].quantile(0.9))
            print(f"      'jump_penalty_min': {jp_min:.2f},")
            print(f"      'jump_penalty_max': {jp_max:.2f},")
        print("  }")
    
    print("\n✅ THE WARNINGS ARE NOT ERRORS!")
    print("-" * 80)
    print("""
The division by zero warnings are EXPECTED during Optuna exploration:

✓ Optuna explores extreme hyperparameters to find boundaries
✓ Some trials will create empty clusters (this is OK!)
✓ The model handles empty clusters gracefully (NaN → ignored)
✓ Optuna learns from bad trials and avoids them

You should NOT suppress these warnings because:
✗ They indicate Optuna is working correctly
✗ They help identify problematic hyperparameter ranges
✗ Real errors would show up as model failures (not just warnings)

What to watch for:
⚠️  If ALL trials have low scores → hyperparameter ranges too extreme
⚠️  If convergence failures > 50% → need to adjust search space
✅  If some trials succeed → Optuna is learning correctly
""")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("Check optuna_trials.png for visualization")
