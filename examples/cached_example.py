"""
Example demonstrating automatic caching of simulation results.

This shows how the framework avoids re-running identical experiments.
"""

from simulation import run_simulation
from visualization import visualize_results
import time


def main():
    """Main function to demonstrate caching."""
    # Define a configuration
    config = {
        "experiment_name": "caching_demo",
        "num_simulations": 3,
        "data_generation": {
            "n_samples": 500,
            "n_states": 3,
            "distribution_type": "Poisson",
            "delta": 0.15,
        },
        "models_to_run": ["Gaussian", "Poisson", "PoissonKL"],
        "optimize_metric": "composite_score",
    }

    # First run - will execute the simulation
    print("=" * 60)
    print("FIRST RUN (will execute simulation)")
    print("=" * 60)
    start = time.time()
    results1 = run_simulation(config, verbose=True, save_models=True)
    elapsed1 = time.time() - start
    print(f"\nFirst run took {elapsed1:.2f} seconds")

    # Second run - should load from cache
    print("\n" + "=" * 60)
    print("SECOND RUN (should load from cache)")
    print("=" * 60)
    start = time.time()
    results2 = run_simulation(config, verbose=True, save_models=True)
    elapsed2 = time.time() - start
    print(f"\nSecond run took {elapsed2:.2f} seconds")

    # Verify results are identical
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Results are identical: {results1.best_df.equals(results2.best_df)}")
    print(f"Speedup from caching: {elapsed1 / elapsed2:.1f}x faster")

    # Third run - disable cache to force re-execution
    print("\n" + "=" * 60)
    print("THIRD RUN (cache disabled, will re-execute)")
    print("=" * 60)
    start = time.time()
    results3 = run_simulation(config, cache=False, verbose=True, save_models=True)
    elapsed3 = time.time() - start
    print(f"\nThird run took {elapsed3:.2f} seconds")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"First run (no cache):    {elapsed1:6.2f}s")
    print(f"Second run (cached):     {elapsed2:6.2f}s  ({elapsed1/elapsed2:4.1f}x faster)")
    print(f"Third run (cache off):   {elapsed3:6.2f}s")
    print("\nCaching works! Identical configurations are loaded instantly.")

    # Visualize the results
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    visualize_results(results1)
    print(f"\nResults saved to: {results1.output_dir}")


if __name__ == "__main__":
    main()
