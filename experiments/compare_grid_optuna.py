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
    
    # Calculate grid size to match Optuna trials
    n_states_values = [2, 3, 4]
    jump_penalty_num = 4  # 4 jump penalty values
    kappa_num = 1  # 1 kappa value (default behavior in grid)
    
    # Total grid points per model
    grid_points_per_model = len(n_states_values) * jump_penalty_num * kappa_num
    
    print(f"\nConfigured to test {grid_points_per_model} hyperparameter combinations per model")
    print(f"  n_states: {len(n_states_values)} values {n_states_values}")
    print(f"  jump_penalty: {jump_penalty_num} values")
    print(f"  kappa: {kappa_num} value(s)")
    
    # ========================================================================
    # 1. Grid Search Configuration
    # ========================================================================
    grid_config = {
        **base_config,
        "experiment_name": "optimization_comparison_grid",
        "optimization": "grid",
        "grid_n_jobs": -1,  # Parallelize hyperparameter search across all cores
        "hyperparameters": {
            "n_states_values": n_states_values,
            "jump_penalty_min": 0.1,
            "jump_penalty_max": 100.0,
            "jump_penalty_num": jump_penalty_num,
            "kappa_num": kappa_num,
        },
    }
    
    print("\n" + "=" * 80)
    print("RUNNING GRID SEARCH")
    print("=" * 80)
    print(f"Hyperparameter space:")
    print(f"  n_states: {grid_config['hyperparameters']['n_states_values']}")
    print(f"  jump_penalty: {grid_config['hyperparameters']['jump_penalty_num']} values")
    print(f"  kappa: {grid_config['hyperparameters']['kappa_num']} value(s)")
    print(f"  Total grid points per model: {grid_points_per_model}")
    print(f"  Models: {len(base_config['models_to_run'])}")
    print(f"  Total evaluations per task: {grid_points_per_model * len(base_config['models_to_run'])}")
    print(f"  Parallel hyperparameter search: {grid_config['grid_n_jobs']} jobs")
    
    start_time = time.time()
    grid_results = run_simulation(grid_config, cache=False, verbose=True)
    grid_time = time.time() - start_time
    
    print(f"\n✓ Grid search completed in {grid_time:.1f}s ({grid_time/60:.1f} min)")
    
    # ========================================================================
    # 2. Optuna Configuration (matching grid size)
    # ========================================================================
    optuna_config = {
        **base_config,
        "experiment_name": "optimization_comparison_optuna",
        "optimization": "optuna",
        "optuna_n_trials": grid_points_per_model,  # Match grid size!
        "optuna_n_jobs": -1,  # Run trials in parallel for faster optimization
    }
    
    print("\n" + "=" * 80)
    print("RUNNING OPTUNA OPTIMIZATION")
    print("=" * 80)
    print(f"Bayesian optimization:")
    print(f"  n_trials per model: {optuna_config['optuna_n_trials']} (MATCHES GRID SIZE)")
    print(f"  Parallel trials: {optuna_config['optuna_n_jobs']}")
    print(f"  Models: {len(base_config['models_to_run'])}")
    print(f"  Total evaluations per task: {optuna_config['optuna_n_trials'] * len(base_config['models_to_run'])}")
    print(f"  Search strategy: TPE (Tree-structured Parzen Estimator)")
    
    start_time = time.time()
    optuna_results = run_simulation(optuna_config, cache=False, verbose=True)
    optuna_time = time.time() - start_time
    
    print(f"\n✓ Optuna completed in {optuna_time:.1f}s ({optuna_time/60:.1f} min)")
    
    # ========================================================================
    # 3. Compare Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARING RESULTS (EQUAL NUMBER OF EVALUATIONS)")
    print("=" * 80)
    
    # Print quick summary
    grid_timing = grid_results.get_timing_info()
    optuna_timing = optuna_results.get_timing_info()
    
    # For Optuna, calculate actual computation time from trial durations
    # (wall-clock time is misleading due to parallel execution)
    from visualization.optuna_plots import load_optuna_trials
    optuna_trials = load_optuna_trials(optuna_results.output_dir)
    
    # Count actual trials run (not just best results saved)
    if optuna_trials is not None:
        n_optuna_trials = len(optuna_trials)
        n_optuna_tasks = optuna_timing['n_configs'] * base_config['num_simulations']  # configs × replications
        n_expected_trials = n_optuna_tasks * grid_points_per_model * len(base_config['models_to_run'])
    else:
        n_optuna_trials = optuna_timing['n_evaluations']
        n_expected_trials = n_optuna_trials
    
    # For Grid, count actual hyperparameter evaluations
    # grid_df has one row per best result per config/rep/model, but each tested grid_points_per_model hyperparams
    n_grid_tasks = grid_timing['n_configs'] * base_config['num_simulations']
    n_grid_evaluations = n_grid_tasks * grid_points_per_model * len(base_config['models_to_run'])
    
    if optuna_trials is not None and 'duration' in optuna_trials.columns:
        # Sum all trial durations to get total computation time
        optuna_computation_time = optuna_trials['duration'].sum()
        optuna_wall_time = optuna_timing['total_time']
        optuna_parallel_speedup = optuna_computation_time / optuna_wall_time if optuna_wall_time > 0 else 1
        
        print(f"\nGrid Search (exhaustive, parallel hyperparameters):")
        print(f"  Wall-clock time: {grid_timing['total_time']:.1f}s ({grid_timing['total_time']/60:.1f} min)")
        print(f"  Total hyperparameter evaluations: {n_grid_evaluations}")
        print(f"  Tasks (config × replication): {n_grid_tasks}")
        print(f"  Evaluations per task: {grid_points_per_model * len(base_config['models_to_run'])}")
        print(f"  Time per evaluation (wall-clock): {grid_timing['total_time']/n_grid_evaluations:.2f}s")
        print(f"  Best accuracy: {grid_results.best_df['balanced_accuracy'].max():.4f}")
        print(f"  Best composite score: {grid_results.best_df['composite_score'].max():.4f}")
        
        print(f"\nOptuna (Bayesian, parallel trials):")
        print(f"  Wall-clock time: {optuna_wall_time:.1f}s ({optuna_wall_time/60:.1f} min)")
        print(f"  Computation time: {optuna_computation_time:.1f}s ({optuna_computation_time/60:.1f} min) [sum of all trials]")
        print(f"  Parallel speedup: {optuna_parallel_speedup:.2f}x")
        print(f"  Total trials run: {n_optuna_trials}")
        print(f"  Tasks (config × replication): {n_optuna_tasks}")
        print(f"  Trials per task: {grid_points_per_model * len(base_config['models_to_run'])}")
        print(f"  Time per trial (computation): {optuna_computation_time/n_optuna_trials:.2f}s")
        print(f"  Best accuracy: {optuna_results.best_df['balanced_accuracy'].max():.4f}")
        print(f"  Best composite score: {optuna_results.best_df['composite_score'].max():.4f}")
        
        # Calculate speedup/efficiency using wall-clock time (fair for both parallel methods)
        time_ratio = grid_timing['total_time'] / optuna_wall_time
        computation_ratio = n_grid_evaluations * (grid_timing['total_time']/n_grid_evaluations) / optuna_computation_time
    else:
        # Fallback if trial data not available
        print(f"\nGrid Search (exhaustive):")
        print(f"  Time: {grid_timing['total_time']:.1f}s ({grid_timing['total_time']/60:.1f} min)")
        print(f"  Evaluations: {grid_timing['n_evaluations']}")
        print(f"  Time per evaluation: {grid_timing['time_per_evaluation']:.2f}s")
        print(f"  Best accuracy: {grid_results.best_df['balanced_accuracy'].max():.4f}")
        print(f"  Best composite score: {grid_results.best_df['composite_score'].max():.4f}")
        
        print(f"\nOptuna (Bayesian):")
        print(f"  Time: {optuna_timing['total_time']:.1f}s ({optuna_timing['total_time']/60:.1f} min)")
        print(f"  Evaluations: {optuna_timing['n_evaluations']}")
        print(f"  Time per evaluation: {optuna_timing['time_per_evaluation']:.2f}s")
        print(f"  Best accuracy: {optuna_results.best_df['balanced_accuracy'].max():.4f}")
        print(f"  Best composite score: {optuna_results.best_df['composite_score'].max():.4f}")
        
        time_ratio = grid_timing['total_time'] / optuna_timing['total_time']
        computation_ratio = time_ratio
        optuna_computation_time = optuna_timing['total_time']
        optuna_wall_time = optuna_timing['total_time']
        n_optuna_trials = optuna_timing['n_evaluations']
        n_grid_evaluations = grid_timing['n_evaluations']
    
    perf_ratio = optuna_results.best_df['composite_score'].max() / grid_results.best_df['composite_score'].max()
    
    print(f"\nComparison (Both using parallel execution):")
    print(f"  Wall-clock time ratio (Grid/Optuna): {time_ratio:.2f}x")
    if optuna_trials is not None:
        print(f"  → Optuna took {abs(1-time_ratio)*100:.1f}% {'MORE' if time_ratio < 1 else 'LESS'} wall-clock time")
        print(f"  ")
        print(f"  Parallelization efficiency:")
        print(f"    Grid: Parallelized {grid_points_per_model} hyperparameters across cores")
        print(f"    Optuna: Ran ~{optuna_parallel_speedup:.0f} trials simultaneously")
        print(f"  ")
        print(f"  Note: Both methods ran the same number of model evaluations ({grid_points_per_model} per model)")
        print(f"        but with different parallelization strategies")
    print(f"  Performance ratio (Optuna/Grid): {perf_ratio:.2f}x")
    print(f"  Winner (wall-clock): {'Optuna' if time_ratio > 1 else 'Grid'} by {abs(time_ratio - 1)*100:.1f}%")
    print(f"  Winner (performance): {'Optuna' if perf_ratio > 1 else 'Grid'} by {abs(perf_ratio - 1)*100:.1f}%")
    
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
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS (EQUAL BUDGET COMPARISON)")
    print("=" * 80)
    
    print(f"\nBoth methods used {grid_points_per_model} evaluations per model")
    print(f"Total evaluations per task: {grid_points_per_model * len(base_config['models_to_run'])}")
    
    # Determine winner for each metric
    grid_perf = grid_results.get_performance_summary()
    optuna_perf = optuna_results.get_performance_summary()
    
    print(f"\nPerformance comparison:")
    for metric in ['balanced_accuracy', 'composite_score', 'feature_f1']:
        if metric in grid_perf['metric'].values and metric in optuna_perf['metric'].values:
            grid_val = grid_perf[grid_perf['metric'] == metric]['mean'].values[0]
            optuna_val = optuna_perf[optuna_perf['metric'] == metric]['mean'].values[0]
            winner = 'Grid' if grid_val > optuna_val else 'Optuna'
            diff = abs(grid_val - optuna_val)
            pct_diff = (diff / max(grid_val, optuna_val)) * 100
            print(f"  {metric:20s}: {winner:6s} wins by {diff:.4f} ({pct_diff:.1f}%)")
    
    print(f"\nEfficiency comparison:")
    if optuna_trials is not None:
        print(f"  Wall-clock times (what you actually waited):")
        print(f"    Grid:   {grid_timing['total_time']:.1f}s ({grid_timing['total_time']/60:.1f} min)")
        print(f"    Optuna: {optuna_wall_time:.1f}s ({optuna_wall_time/60:.1f} min)")
        print(f"    Winner: {'Optuna' if time_ratio > 1 else 'Grid'} by {abs(grid_timing['total_time'] - optuna_wall_time):.1f}s")
        print(f"  ")
        print(f"  Total computation (sum of all parallel work):")
        print(f"    Grid:   ~{grid_timing['total_time']:.1f}s (approximate, no individual timing)")
        print(f"    Optuna: {optuna_computation_time:.1f}s ({optuna_computation_time/60:.1f} min) [measured from trials]")
        print(f"  ")
        print(f"  Per-evaluation time:")
        print(f"    Grid:   {grid_timing['total_time']/n_grid_evaluations:.2f}s per hyperparameter eval (wall-clock)")
        print(f"    Optuna: {optuna_computation_time/n_optuna_trials:.2f}s per trial (computation)")
    else:
        print(f"  Total time: {'Optuna' if optuna_timing['total_time'] < grid_timing['total_time'] else 'Grid'} "
              f"faster by {abs(grid_timing['total_time'] - optuna_timing['total_time']):.1f}s "
              f"({abs(time_ratio - 1)*100:.1f}%)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
