"""
Simple example of the new simplified API.

This demonstrates the minimal code needed to run a simulation
and visualize the results.
"""

from simulation import run_simulation
from visualization import visualize_results


def main():
    """Main function to run the example."""
    # Define a simple configuration
    config = {
        "experiment_name": "simple_test",
        "num_simulations": 2,
        "data_generation": {
            "n_samples": 200,
            "n_states": 2,
        },
        "models_to_run": ["Gaussian", "Poisson", "PoissonKL"],
        "optimize_metric": "composite_score",
    }

    # Run the simulation (results are automatically cached)
    print("Running simulation...")
    results = run_simulation(config)

    # Access results
    print(f"\nBest results shape: {results.best_df.shape}")
    print(f"Grid search results shape: {results.grid_df.shape}")
    print(f"\nBest models by composite score")
    print(results.get_best_models("composite_score").head())

    # Automatically generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(results)

    print("\nDone! Check the plots/ directory for visualizations.")


if __name__ == "__main__":
    main()
