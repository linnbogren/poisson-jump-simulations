"""
Example showing how to compare different data generation parameters.

This demonstrates comparing Poisson vs Gaussian distributions
across different delta values.
"""

from simulation import run_simulation
from visualization import visualize_results


def main():
    """Main function to run the comparison example."""
    # Define a comparison experiment
    config = {
        "experiment_name": "poisson_vs_gaussian_comparison",
        "num_simulations": 2,
        
        # Test multiple distributions and delta values
        "data_generation": [
            {
                "n_samples": 200,
                "n_states": 2,
                "distribution_type": "Poisson",
                "delta": delta,
            }
            for delta in [0.1, 0.2, 0.3]
        ] + [
            {
                "n_samples": 200,
                "n_states": 2,
                "distribution_type": "Gaussian",
                "delta": delta,
            }
            for delta in [0.1, 0.2, 0.3]
        ],
        
        # Compare multiple models
        "models_to_run": ["Gaussian", "Poisson", "PoissonKL"],
        
        # Optimize on balanced accuracy
        "optimize_metric": "balanced_accuracy",
        
        # Enable grid search for model selection
        "hyperparameters": {
            "n_states_values": [2, 3],
            "jump_penalty_min": 0.1,
            "jump_penalty_max": 100.0,
            "jump_penalty_num": 3,
        },
    }

    # Run the simulation
    print("Running comparison simulation (this may take a while)...")
    results = run_simulation(config, verbose=True)

    # Print summary
    print(f"\nSimulation complete!")
    print(f"Total configurations tested: {len(results.best_df)}")
    print(f"Grid search evaluations: {len(results.grid_df)}")

    # Show which model performed best for each condition
    print("\nBest model by condition:")
    summary = results.get_best_models("balanced_accuracy")
    for idx, row in summary.iterrows():
        print(f"  {row['distribution_type']:12s} delta={row['delta']:.1f}: "
              f"{row['model_name']:15s} (BAC={row['balanced_accuracy']:.3f})")

    # Generate comprehensive visualizations
    print("\nGenerating visualizations...")
    visualize_results(results)

    print(f"\nDone! Results saved to: {results.output_dir}")


if __name__ == "__main__":
    main()
