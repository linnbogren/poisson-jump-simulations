"""
Example showing how to compare different data generation parameters.

This demonstrates comparing Poisson vs Gaussian distributions
across different delta values.
"""

from simulation import run_simulation
from visualization import visualize_results, plot_stacked_states, create_all_tables
from pathlib import Path
import pickle


def plot_stacked_per_delta(results):
    """Plot stacked time series for each delta value and P combination using saved models."""
    # Get unique combinations of delta and n_total_features
    unique_combos = results.best_df[['delta', 'n_total_features']].drop_duplicates()
    
    plots_dir = results.output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    for _, row in unique_combos.iterrows():
        delta_val = row['delta']
        P_val = int(row['n_total_features'])
        
        print(f"  Creating stacked plot for delta={delta_val:.2f}, P={P_val}...")
        
        # Get the config_id for this combination
        config_rows = results.best_df[
            (results.best_df['delta'] == delta_val) & 
            (results.best_df['n_total_features'] == P_val)
        ]
        if len(config_rows) == 0:
            continue
            
        # Get config parameters from first row
        config_row = config_rows.iloc[0]
        
        # Find the matching config index
        for config_id, config_dict in enumerate(results.configs.to_dict('records')):
            if (abs(config_dict.get('delta', 0) - delta_val) < 0.001 and
                config_dict.get('n_total_features', 0) == P_val):
                break
        else:
            print(f"    Warning: Could not find config for delta={delta_val}, P={P_val}")
            continue
        
        # Load regenerated data for this config (use replication 0)
        try:
            from simulation.data_generation import generate_data
            from simulation.config import SimulationConfig
            
            # Recreate config
            sim_config = SimulationConfig(
                n_samples=int(config_row['n_samples']),
                n_states=int(config_row['n_states']),
                n_informative=int(config_row['n_informative']),
                n_total_features=int(config_row['n_total_features']),
                delta=float(config_row['delta']),
                lambda_0=float(config_row['lambda_0']),
                persistence=float(config_row['persistence']),
                distribution_type=config_row['distribution_type'],
                correlated_noise=bool(config_row.get('correlated_noise', False)),
                random_seed=int(config_row.get('random_seed', 42))  # Use same seed as first replication
            )
            
            # Generate data
            X, true_states, _ = generate_data(sim_config)
            
            # Convert to DataFrame
            import pandas as pd
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
            
        except Exception as e:
            print(f"    Warning: Could not generate data for delta={delta_val}: {e}")
            continue
        
        # Load saved models for each model type
        models_dict = {}
        model_names = results.models
        
        # Use the actual random_seed from the config
        seed = int(config_row.get('random_seed', 42))
        
        for model_name in model_names:
            try:
                # Construct filename using the same pattern as saving
                model_filename = (
                    f"model_{model_name}_"
                    f"seed{seed}_"
                    f"P{P_val}_"
                    f"delta{delta_val}.pkl"
                )
                model_path = results.output_dir / "models" / model_filename
                
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                models_dict[model_name] = model
            except FileNotFoundError:
                print(f"    Warning: Model file not found: {model_filename}")
                continue
        
        if not models_dict:
            print(f"    Warning: No models found for delta={delta_val}, P={P_val}")
            continue
        
        # Create stacked plot
        save_path = plots_dir / f'stacked_states_delta_{delta_val:.2f}_P_{P_val}.png'
        
        try:
            fig = plot_stacked_states(
                X=X_df,
                models_dict=models_dict,
                true_states=true_states,
                feature_to_plot=feature_names[0],  # Plot first feature
                figsize=(12, 10),
                save_path=save_path
            )
            print(f"    ✓ Saved: {save_path.name}")
        except Exception as e:
            print(f"    Error creating plot for delta={delta_val}, P={P_val}: {e}")


def main():
    """Main function to run the comparison example."""
    # Define a comparison experiment
    config = {
        "experiment_name": "poisson_delta_comparison",
        "num_simulations": 5,
        
        # Test multiple delta values and feature counts
        "data_generation": [
            {
                "n_samples": 400,
                "n_states": 3,
                "distribution_type": "Poisson",
                "delta": delta,
                "n_total_features": P,
                "n_informative": 15,  # Keep informative features constant
            }
            for delta in [0.05, 0.1, 0.15, 0.2]
            for P in [15, 60, 100]
        ] ,
        
        # Compare multiple models
        "models_to_run": ["Gaussian", "Poisson", "PoissonKL"],
        
        # Optimize on balanced accuracy
        "optimize_metric": "composite_score",
        
        # Enable grid search for model selection
        "hyperparameters": {
            "n_states_values": [2, 3, 4],
            "jump_penalty_min": 0.1,
            "jump_penalty_max": 100.0,
            "jump_penalty_num": 3,
        },
    }

    # Run the simulation
    print("Running comparison simulation (this may take a while)...")
    results = run_simulation(config, verbose=True, save_models=True)

    # Print summary
    print(f"\nSimulation complete!")
    print(f"Total configurations tested: {len(results.best_df)}")
    print(f"Grid search evaluations: {len(results.grid_df)}")

    # Show which model performed best for each condition

    print("\nBest model by condition:")
    summary = results.get_best_models("composite_score")
    for idx, row in summary.iterrows():
        print(f"  {row['distribution_type']:12s} delta={row['delta']:.2f} P={int(row['n_total_features']):3d}: "
              f"{row['model_name']:15s} (Composite Score={row['composite_score']:.3f})")

    # Generate comprehensive visualizations
    print("\nGenerating visualizations...")
    visualize_results(results)

    # Plot stacked time series for each delta value
    print("\nGenerating stacked time series plots per delta...")
    plot_stacked_per_delta(results)

    # Create LaTeX tables
    print("\nGenerating LaTeX tables...")
    create_all_tables(results, output_dir="tables", alpha=0.05)

    print(f"\nDone! Results saved to: {results.output_dir}")


if __name__ == "__main__":
    main()
