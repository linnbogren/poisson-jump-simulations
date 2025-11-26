"""Visualize results from a single simulation run.

This script loads results from examples/single_run_results and creates
visualizations of the model performance and data.

Usage:
    python examples/visualize_single_run.py
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.results import ResultManager
from visualization import (
    plot_time_series_with_breakpoints,
    plot_stacked_states,
    plot_model_comparison_bars,
    plot_hyperparameter_heatmap,
    setup_plotting_style,
)


def main():
    """Load and visualize single run results."""
    
    # Setup plotting style
    setup_plotting_style()
    
    # Load results
    results_path = Path(__file__).parent / "single_run_results"
    
    if not results_path.exists():
        print(f"Error: Results directory not found at {results_path}")
        print("Please run examples/run_single_example.py first.")
        return
    
    print("="*80)
    print("Loading results from single run...")
    print("="*80)
    
    manager = ResultManager(results_path)
    
    # Load best results
    best_results = manager.load_best_results()
    grid_results = manager.load_grid_results()
    
    # Print summary
    print(f"\nExperiment: {manager._metadata.get('experiment', {}).get('name', 'Single Run')}")
    print(f"Models tested: {best_results['model_name'].nunique()}")
    print(f"Best model evaluations: {len(best_results)}")
    print(f"Total evaluations: {len(grid_results)}")
    
    # Get best results by model
    print(f"\nBest results by balanced_accuracy:")
    for model_name in best_results['model_name'].unique():
        model_df = best_results[best_results['model_name'] == model_name]
        best_bac = model_df['balanced_accuracy'].max()
        print(f"  {model_name}: {best_bac:.4f}")
    
    # Create output directory for plots
    output_dir = results_path / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Model comparison bars for key metrics
    print("\n" + "="*80)
    print("Creating model comparison plots...")
    print("="*80)
    
    metrics_to_plot = [
        ('balanced_accuracy', 'Model Performance: Balanced Accuracy'),
        ('feature_f1', 'Model Performance: Feature F1 Score'),
        ('chamfer_distance', 'Model Performance: Chamfer Distance'),
    ]
    
    for metric, plot_title in metrics_to_plot:
        if metric in grid_results.columns:
            fig = plot_model_comparison_bars(
                grid_results,
                metric=metric,
                title=plot_title
            )
            filename = f"comparison_{metric}.png"
            fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir / filename}")
            plt.close(fig)
    
    # 2. Hyperparameter heatmaps for each model
    print("\n" + "="*80)
    print("Creating hyperparameter heatmaps...")
    print("="*80)
    
    # For heatmaps, we need the grid search results with proper hyperparameter columns
    # Check if grid_results has the necessary columns
    required_cols = ['best_jump_penalty', 'best_max_feats']
    
    model_names = grid_results['model_name'].unique()
    for model_name in model_names:
        model_results = grid_results[
            grid_results['model_name'] == model_name
        ]
        
        # Only create heatmap if we have enough data points and the required columns
        if len(model_results) > 1 and all(col in model_results.columns for col in required_cols):
            # Check if there's variation in the hyperparameters
            n_unique_x = model_results['best_jump_penalty'].nunique()
            n_unique_y = model_results['best_max_feats'].nunique()
            
            if n_unique_x > 1 and n_unique_y > 1:
                try:
                    fig2 = plot_hyperparameter_heatmap(
                        model_results,
                        param_x='best_jump_penalty',
                        param_y='best_max_feats',
                        metric='balanced_accuracy',
                        title=f'{model_name} Hyperparameter Performance'
                    )
                    filename = f"hyperparameters_{model_name.lower()}.png"
                    fig2.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
                    print(f"Saved: {output_dir / filename}")
                    plt.close(fig2)
                except Exception as e:
                    print(f"Could not create heatmap for {model_name}: {e}")
            else:
                print(f"Skipping heatmap for {model_name} - not enough hyperparameter variation")
        else:
            print(f"Skipping heatmap for {model_name} - insufficient data or missing columns")
    
    # 3. Time series visualization (if models are saved)
    print("\n" + "="*80)
    print("Creating time series visualizations...")
    print("="*80)
    
    # Check if models directory exists and has models
    models_dir = results_path / "models"
    if models_dir.exists() and list(models_dir.glob("*.pkl")):
        print("Found saved models!")
        
        # Get best result for each model
        for model_name in best_results['model_name'].unique():
            print(f"\nVisualizing {model_name} predictions...")
            
            # Get best result for this model
            model_best = best_results[best_results['model_name'] == model_name].iloc[0]
            
            # Load the model (using naming from run_single_example.py)
            # Try to find the model file
            model_files = list(models_dir.glob(f"model_{model_name}_*.pkl"))
            if not model_files:
                print(f"No model file found for {model_name}, skipping...")
                continue
            
            import pickle
            with open(model_files[0], 'rb') as f:
                model = pickle.load(f)
            
            # Regenerate the data with the same config
            from simulation.data_generation import generate_data
            from simulation.config import SimulationConfig
            
            config = SimulationConfig(
                n_samples=int(model_best['n_samples']),
                n_states=int(model_best['n_states']),
                n_informative=int(model_best['n_informative']),
                n_noise=int(model_best['n_noise']),
                n_total_features=int(model_best['n_total_features']),
                delta=float(model_best['delta']),
                lambda_0=float(model_best['lambda_0']),
                persistence=float(model_best['persistence']),
                distribution_type=str(model_best['distribution_type']),
                correlated_noise=bool(model_best['correlated_noise']),
                random_seed=int(model_best['random_seed']),
            )
            
            X, states, breakpoints = generate_data(config)
            
            # Plot time series with first 3 informative features
            feature_cols = [col for col in X.columns if col.startswith('informative_')][:3]
            
            fig3 = plot_time_series_with_breakpoints(
                X[feature_cols],
                model,
                actual_breakpoints=breakpoints,
                title=f'{model_name}: Time Series with Breakpoints'
            )
            filename = f"timeseries_{model_name.lower()}.png"
            fig3.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir / filename}")
            
            # Plot stacked states
            fig4 = plot_stacked_states(
                X,
                models_dict={model_name: model},
                true_states=states,
                feature_to_plot='informative_1'
            )
            filename = f"states_{model_name.lower()}.png"
            fig4.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir / filename}")
            
            plt.close('all')  # Close figures to save memory
    else:
        print("No saved models found. Skipping time series visualizations.")
        print("To save models, run with save_models=True in run_single_example.py")
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
