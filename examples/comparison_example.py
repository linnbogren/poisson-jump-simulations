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
        
        # Find the best performing replication across all models for this delta/P to use for data
        # We want to pick the seed that has the best overall performance
        all_models_for_combo = results.best_df[
            (results.best_df['delta'] == delta_val) & 
            (results.best_df['n_total_features'] == P_val)
        ]
        
        if len(all_models_for_combo) == 0:
            print(f"    Warning: No models found for delta={delta_val}, P={P_val}")
            continue
        
        # Group by seed and get average composite score to find best replication
        seed_performance = all_models_for_combo.groupby('random_seed')['composite_score'].mean()
        best_seed = int(seed_performance.idxmax())
        data_seed = best_seed
        
        print(f"    Using replication with seed {data_seed} (avg composite={seed_performance[best_seed]:.3f})")
        
        # Load regenerated data using the correct seed
        try:
            from simulation.data_generation import generate_data
            from simulation.config import SimulationConfig
            
            # Recreate config with the CORRECT seed that was used for training
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
                random_seed=data_seed  # Use the ACTUAL seed from the best replication
            )
            
            # Generate data with the correct seed
            X, true_states, _ = generate_data(sim_config)
            
            # X is already a DataFrame with columns like 'informative_1', 'informative_2', 'noise_1', etc.
            # Rename to feature_0, feature_1, ... for consistency
            import pandas as pd
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = X.copy()
            X_df.columns = feature_names
            
        except Exception as e:
            print(f"    Warning: Could not generate data for delta={delta_val}: {e}")
            continue
        
        # Load saved models for each model type
        # IMPORTANT: Load models that were ALL trained on the SAME data (same seed)
        # We use data_seed which was determined above
        models_dict = {}
        model_metrics = {}
        model_names = results.models
        
        for model_name in model_names:
            # Get results for this delta/P/model/seed combination
            model_rows = results.best_df[
                (results.best_df['delta'] == delta_val) & 
                (results.best_df['n_total_features'] == P_val) &
                (results.best_df['model_name'] == model_name) &
                (results.best_df['random_seed'] == data_seed)  # Same seed as data!
            ]
            
            if len(model_rows) == 0:
                print(f"    Warning: No results found for {model_name} with seed {data_seed}")
                continue
            
            # Should only be one row since we filtered by seed
            best_row = model_rows.iloc[0]
            
            # Store metrics for this model
            model_metrics[model_name] = {
                'composite_score': best_row.get('composite_score', 0),
                'balanced_accuracy': best_row.get('balanced_accuracy', 0),
                'breakpoint_error': best_row.get('breakpoint_error', 0),
                'bic': best_row.get('bic', 0)
            }
            
            try:
                # Construct filename using the same pattern as saving
                model_filename = (
                    f"model_{model_name}_"
                    f"seed{data_seed}_"
                    f"P{P_val}_"
                    f"delta{delta_val}.pkl"
                )
                model_path = results.output_dir / "models" / model_filename
                
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                models_dict[model_name] = model
                print(f"    ✓ Loaded {model_name} (seed={data_seed}, composite={best_row['composite_score']:.3f})")
            except FileNotFoundError:
                print(f"    Warning: Model file not found: {model_filename}")
                continue
        
        if not models_dict:
            print(f"    Warning: No models found for delta={delta_val}, P={P_val}")
            continue
        
        # Create stacked plot
        save_path = plots_dir / f'stacked_states_delta_{delta_val:.2f}_P_{P_val}.png'
        
        try:
            # Plot the first informative feature (feature_0) with metrics
            fig = plot_stacked_states(
                X=X_df,
                models_dict=models_dict,
                true_states=true_states,
                feature_to_plot='feature_0',  # Plot first feature
                model_metrics=model_metrics,  # Include performance metrics
                figsize=(12, 10),
                save_path=save_path
            )
            print(f"    ✓ Saved plot: {save_path.name}")
        except Exception as e:
            print(f"    Error creating plot for delta={delta_val}, P={P_val}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the comparison example."""
    # Define a comparison experiment
    config = {
        "experiment_name": "poisson_delta_comparison",
        "num_simulations": 1,
        
        # Test multiple delta values and feature counts
        "data_generation": [
            {
                "n_samples": 500,
                "n_states": 3,
                "distribution_type": "Poisson",
                "delta": delta,
                "n_total_features": P,
                "n_informative": 15,  # Keep informative features constant
            }
            for delta in [0.05, 0.1, 0.15, 0.2, 0.25]
            for P in [15, 60, 100, 300]
        ] ,
        
        # Compare multiple models
        "models_to_run": ["Gaussian", "Poisson", "PoissonKL"],
        
        # Optimize on balanced accuracy
        "optimize_metric": "bic",
        
        # Enable grid search for model selection
        "hyperparameters": {
            "n_states_values": [2, 3, 4],
            "jump_penalty_min": 0.1,
            "jump_penalty_max": 100.0,
            "jump_penalty_num": 5,
            "kappa_min": 1.0,
            "kappa_max_type": "sqrt_P",  # Max kappa = sqrt(P)
            "kappa_num": 7,  # Number of kappa values to test
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
