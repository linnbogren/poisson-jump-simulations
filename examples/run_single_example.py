"""
Example: run a single simulation and print aggregated results.

This example creates a minimal ExperimentConfig in 'single' mode,
runs one replication on a small simulated dataset using grid search,
and prints the aggregated results to stdout.

Run with:
    poetry run python examples\run_single_example.py
"""
from pathlib import Path
from datetime import datetime

from simulation.config import ExperimentConfig, SimulationConfig
from simulation.runner import run_simulation
from simulation.results import ResultManager


def main():
    out_dir = Path("examples/single_run_results")

    # Create a simple data configuration for a single run
    # Testing with Poisson data - will gracefully skip failed hyperparameter combinations
    data_cfg = SimulationConfig(
        n_samples=100,
        n_states=3,
        n_informative=15,
        n_total_features=15,
        delta=0.5,
        lambda_0=10.0,
        persistence=0.97,
        distribution_type="Poisson",  # Test Poisson with error handling
        random_seed=42,
    )

    # Create experiment configuration (single mode)
    # Note: Some hyperparameter combinations may fail convergence and will be skipped automatically
    exp_config = ExperimentConfig(
        name="single_example",
        mode="single",
        data=data_cfg,  # Required for single mode
        optimization_method="grid",
        n_replications=1,
        parallel=False,
        single_thread=True,
        model_names=["Gaussian", "Poisson"],  # Try both models
        optimize_metric="balanced_accuracy",
        quick_test=True,
        hyperparameter_grid={
            "n_states_values": [2, 3, 4],
            "jump_penalty_min": 0.1,
            "jump_penalty_max": 100.0,
            "jump_penalty_num": 7,
            "jump_penalty_scale": "log",
            "kappa_min": 1.0,  # Some combinations will fail, but that's OK
            "kappa_max_type": "sqrt_P",
            "kappa_num": 14,
            "quick_test": True,
        }
    )

    print("Starting single simulation run...")
    start = datetime.now()

    # Run simulation (will write results under out_dir)
    results_df = run_simulation(
        experiment_config=exp_config,
        data_configs=[data_cfg],
        output_dir=str(out_dir),
        n_jobs=1,
        save_models=True,
        verbose=True,
    )

    end = datetime.now()
    print(f"Simulation finished in {(end - start).total_seconds():.2f}s")

    # Print aggregated results
    print("\nAggregated results (first rows):")
    print(results_df.to_string(index=False, max_colwidth=15))

    # Show how to load results programmatically via ResultManager
    print(f"\n{'='*80}")
    print("Loading results via ResultManager...")
    print(f"{'='*80}")
    
    manager = ResultManager(out_dir)
    
    # Show experiment metadata
    if manager._metadata:
        print("\nExperiment metadata:")
        for k, v in manager._metadata.get('experiment', {}).items():
            print(f"  {k}: {v}")
        
        # Show execution info
        exec_info = manager._metadata.get('execution', {})
        if exec_info:
            print("\nExecution info:")
            print(f"  Status: {exec_info.get('status', 'unknown')}")
            print(f"  Execution time: {exec_info.get('execution_time_seconds', 0):.2f}s")
    
    # Load best results again
    best = manager.load_best_results()
    print(f"\nLoaded {len(best)} best model result(s)")
    print(f"Best balanced accuracy: {best['balanced_accuracy'].max():.4f}")
    print(f"\n{'='*80}")
    print(f"Results saved to: {out_dir.absolute()}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
