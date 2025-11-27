"""
Comparison study: Grid Search vs Optuna Optimization

This experiment compares the performance and efficiency of two hyperparameter
optimization methods:
- Grid Search: Exhaustive search over hyperparameter grid
- Optuna: Bayesian optimization for efficient hyperparameter search

The study evaluates:
- Performance metrics (balanced accuracy, F1 scores, etc.)
- Execution time and computational efficiency
- Number of evaluations needed to find good hyperparameters
"""

from simulation import run_simulation
from visualization import visualize_results, compare_optimization_methods
from pathlib import Path
import time


def main():
    """Run the grid vs optuna comparison study."""
    
    print("=" * 80)
    print("GRID SEARCH vs OPTUNA COMPARISON STUDY")
    print("=" * 80)
    print("\nThis experiment will:")
    print("  1. Run simulations with Grid Search optimization")
    print("  2. Run simulations with Optuna optimization")
    print("  3. Compare performance metrics and execution time")
    print("  4. Generate comparative visualizations")
    print("\n" + "=" * 80)
    
    # Base configuration (shared between both methods)
    base_config = {
        "num_simulations": 3,  # Small number for quick comparison
        
        # Test configurations
        "data_generation": [
            {
                "n_samples": 200,
                "n_states": 3,
                "n_informative": 15,
                "distribution_type": "Poisson",
                "delta": 0.15,
            },
            {
                "n_samples": 200,
                "n_states": 3,
                "n_informative": 15,
                "distribution_type": "Poisson",
                "delta": 0.07,
            },
        ],
        
        "models_to_run": ["Gaussian", "Poisson", "PoissonKL"],
        "optimize_metric": "composite_score",
    }
    
    # ========================================================================
    # 1. Grid Search Configuration
    # ========================================================================
    grid_config = {
        **base_config,
        "experiment_name": "optimization_comparison_grid",
        "optimization": "grid",
        "hyperparameters": {
            "n_states_values": [2, 3, 4],
            "jump_penalty_min": 0.1,
            "jump_penalty_max": 100.0,
            "jump_penalty_num": 3, 
        },
    }
    
    print("\n" + "=" * 80)
    print("RUNNING GRID SEARCH")
    print("=" * 80)
    print(f"Hyperparameter space:")
    print(f"  n_states: {grid_config['hyperparameters']['n_states_values']}")
    print(f"  jump_penalty: {grid_config['hyperparameters']['jump_penalty_num']} values")
    print(f"  Total grid points: {len(grid_config['hyperparameters']['n_states_values']) * grid_config['hyperparameters']['jump_penalty_num']}")
    
    start_time = time.time()
    grid_results = run_simulation(grid_config, cache=False, verbose=True)
    grid_time = time.time() - start_time
    
    print(f"\n✓ Grid search completed in {grid_time:.1f}s ({grid_time/60:.1f} min)")
    
    # ========================================================================
    # 2. Optuna Configuration
    # ========================================================================
    optuna_config = {
        **base_config,
        "experiment_name": "optimization_comparison_optuna",
        "optimization": "optuna",
        "optuna_n_trials": 12,  # Trials per model (12 × 3 models = 36 trials per task)
        "optuna_n_jobs": -1,  # Run 3 trials in parallel for faster optimization
    }
    
    print("\n" + "=" * 80)
    print("RUNNING OPTUNA OPTIMIZATION")
    print("=" * 80)
    print(f"Bayesian optimization:")
    print(f"  n_trials per model: {optuna_config['optuna_n_trials']}")
    print(f"  Parallel trials: {optuna_config['optuna_n_jobs']}")
    print(f"  Total trials per task: {optuna_config['optuna_n_trials'] * len(base_config['models_to_run'])}")
    print(f"  Search strategy: TPE (Tree-structured Parzen Estimator)")
    
    start_time = time.time()
    optuna_results = run_simulation(optuna_config, cache=False, verbose=True)
    optuna_time = time.time() - start_time
    
    print(f"\n✓ Optuna completed in {optuna_time:.1f}s ({optuna_time/60:.1f} min)")
    
    # ========================================================================
    # 3. Compare Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARING RESULTS")
    print("=" * 80)
    
    # Print quick summary
    grid_timing = grid_results.get_timing_info()
    optuna_timing = optuna_results.get_timing_info()
    
    print(f"\nGrid Search:")
    print(f"  Time: {grid_timing['total_time']:.1f}s")
    print(f"  Evaluations: {grid_timing['n_evaluations']}")
    print(f"  Best accuracy: {grid_results.best_df['balanced_accuracy'].max():.4f}")
    
    print(f"\nOptuna:")
    print(f"  Time: {optuna_timing['total_time']:.1f}s")
    print(f"  Evaluations: {optuna_timing['n_evaluations']}")
    print(f"  Best accuracy: {optuna_results.best_df['balanced_accuracy'].max():.4f}")
    
    print(f"\nSpeedup: {grid_timing['total_time'] / optuna_timing['total_time']:.2f}x "
          f"({'Optuna faster' if optuna_timing['total_time'] < grid_timing['total_time'] else 'Grid faster'})")
    
    # ========================================================================
    # 4. Create Comparison Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Individual visualizations
    print("\nGenerating Grid Search visualizations...")
    visualize_results(grid_results)
    
    print("\nGenerating Optuna visualizations...")
    visualize_results(optuna_results)
    
    # Comparison visualizations
    print("\nGenerating comparison plots...")
    comparison_dir = compare_optimization_methods(
        {
            'Grid Search': grid_results,
            'Optuna': optuna_results,
        },
        output_dir='experiments/optimization_comparison',
        metrics=['balanced_accuracy', 'feature_f1', 'chamfer_distance']
    )
    
    # ========================================================================
    # 5. Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  Grid Search: {grid_results.output_dir}")
    print(f"  Optuna: {optuna_results.output_dir}")
    print(f"  Comparison: {comparison_dir}")
    print("\nKey Findings:")
    
    # Determine winner for each metric
    grid_perf = grid_results.get_performance_summary()
    optuna_perf = optuna_results.get_performance_summary()
    
    for metric in ['balanced_accuracy', 'feature_f1']:
        grid_val = grid_perf[grid_perf['metric'] == metric]['mean'].values[0]
        optuna_val = optuna_perf[optuna_perf['metric'] == metric]['mean'].values[0]
        winner = 'Grid' if grid_val > optuna_val else 'Optuna'
        diff = abs(grid_val - optuna_val)
        print(f"  {metric}: {winner} wins by {diff:.4f}")
    
    print(f"  Execution time: {'Optuna' if optuna_timing['total_time'] < grid_timing['total_time'] else 'Grid'} "
          f"({abs(grid_timing['total_time'] - optuna_timing['total_time']):.1f}s faster)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
