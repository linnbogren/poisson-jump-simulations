"""Run a full simulation experiment with multiple configurations and replications.

This script allows you to:
1. Define multiple data generation configurations
2. Run simulations with multiple replications
3. Automatically generate comprehensive visualizations
4. Compare results across different configurations

Usage:
    python examples/run_experiment.py
    
Customize the EXPERIMENT_CONFIGS dictionary below to define your experiment.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.config import SimulationConfig, ExperimentConfig, HyperparameterGridConfig
from simulation.runner import run_simulation
from simulation.results import ResultManager
from visualization import (
    plot_model_comparison_bars,
    plot_parameter_sensitivity,
    plot_metric_comparison_grid,
    plot_aggregated_results_overview,
    plot_correlation_matrix,
    plot_time_series_with_breakpoints,
    plot_stacked_states,
    setup_plotting_style,
)


###############################################################################
# EXPERIMENT CONFIGURATION
###############################################################################

# Define your experiment configurations here
EXPERIMENT_CONFIGS = {
    'experiment_name': 'poisson_vs_gaussian_comparison',
    
    # Execution settings
    'n_replications': 2,  # Number of independent runs per configuration
    'optimization': 'grid',  # 'grid' or 'optuna'
    'n_jobs': -1,  # Number of parallel jobs (-1 = all cores)
    'save_models': True,  # Save fitted models for visualization
    
    # Models to compare
    'model_names': ['Gaussian', 'Poisson', 'PoissonKL'],
    
    # Hyperparameter grid (for grid search)
    'grid_config': {
        'n_components': [2, 3],
        'jump_penalty': [1, 10, 100],
        'kappa_min': [1.0, 2.0],
        'kappa_max': [20.0],
    },
    
    # OR: Optuna settings (if optimization='optuna')
    'optuna_config': {
        'n_trials': 100,
        'n_components_range': (2, 4),
        'jump_penalty_range': (0.01, 1000.0),
        'kappa_range': (1.0, 30.0),
    },
    
    # Metric to optimize
    'optimize_metric': 'composite_score',
    
    # Data generation configurations to test
    'data_configs': [
        # Configuration 1: Poisson data, high jump rate
        {
            'n_samples': 200,
            'n_states': 3,
            'n_informative': 15,
            'n_noise': 5,
            'delta': 0.15,  # High jump rate
            'lambda_0': 10.0,
            'persistence': 0.97,
            'distribution_type': 'Poisson',
            'correlated_noise': False,
            'random_seed': 42,
        },
        
        # Configuration 2: Poisson data, low jump rate
        {
            'n_samples': 200,
            'n_states': 3,
            'n_informative': 15,
            'n_noise': 5,
            'delta': 0.07,  # Low jump rate
            'lambda_0': 10.0,
            'persistence': 0.97,
            'distribution_type': 'Poisson',
            'correlated_noise': False,
            'random_seed': 42,
        }
    ],
}


###############################################################################
# VISUALIZATION CONFIGURATION
###############################################################################

VISUALIZATION_CONFIG = {
    'create_comparison_plots': True,
    'create_heatmaps': True,
    'create_time_series': True,  # Requires save_models=True
    'create_correlation_matrix': True,
    'create_overview_plot': True,
    
    # Metrics to visualize
    'comparison_metrics': [
        'balanced_accuracy',
        'feature_f1',
        'chamfer_distance',
        'breakpoint_count_error',
    ],
    
    # For heatmaps
    'heatmap_params': [
        ('best_jump_penalty', 'best_n_components'),
    ],
    
    # Plot settings
    'dpi': 300,
    'figsize_default': (10, 6),
    'figsize_large': (14, 10),
}


###############################################################################
# MAIN EXECUTION
###############################################################################

def create_visualizations(results_path: Path, config: dict):
    """Create all visualizations for the experiment results."""
    
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    # Setup plotting style
    setup_plotting_style()
    
    # Load results
    manager = ResultManager(results_path)
    best_results = manager.load_best_results()
    grid_results = manager.load_grid_results()
    
    # Create plots directory
    plots_dir = results_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    viz_config = VISUALIZATION_CONFIG
    
    # 1. Model comparison plots
    if viz_config['create_comparison_plots']:
        print("\n" + "-"*80)
        print("Creating model comparison plots...")
        print("-"*80)
        
        for metric in viz_config['comparison_metrics']:
            if metric in best_results.columns:
                try:
                    fig = plot_model_comparison_bars(
                        best_results,
                        metric=metric,
                        title=f'Model Performance: {metric.replace("_", " ").title()}'
                    )
                    filename = f"comparison_{metric}.png"
                    fig.savefig(plots_dir / filename, dpi=viz_config['dpi'], bbox_inches='tight')
                    print(f"  ✓ Saved: {filename}")
                    plt.close(fig)
                except Exception as e:
                    print(f"  ✗ Error creating {metric} comparison: {e}")
    
    # 2. Aggregated overview plot
    if viz_config['create_overview_plot']:
        print("\n" + "-"*80)
        print("Creating overview plot...")
        print("-"*80)
        
        try:
            fig = plot_aggregated_results_overview(
                best_results,
                key_metrics=['balanced_accuracy', 'feature_f1', 'chamfer_distance'],
                figsize=viz_config['figsize_large']
            )
            fig.savefig(plots_dir / "overview.png", dpi=viz_config['dpi'], bbox_inches='tight')
            print("  ✓ Saved: overview.png")
            plt.close(fig)
        except Exception as e:
            print(f"  ✗ Error creating overview: {e}")
    
    # 3. Correlation matrix
    if viz_config['create_correlation_matrix']:
        print("\n" + "-"*80)
        print("Creating correlation matrix...")
        print("-"*80)
        
        try:
            metric_cols = [col for col in best_results.columns 
                          if col not in ['n_samples', 'n_states', 'n_informative', 'n_noise',
                                        'n_total_features', 'delta', 'lambda_0', 'persistence',
                                        'distribution_type', 'correlated_noise', 'random_seed',
                                        'model_name', 'best_n_components', 'best_jump_penalty',
                                        'best_max_feats']]
            
            fig = plot_correlation_matrix(
                best_results[metric_cols],
                title='Metric Correlations',
                figsize=viz_config['figsize_large']
            )
            fig.savefig(plots_dir / "correlation_matrix.png", dpi=viz_config['dpi'], bbox_inches='tight')
            print("  ✓ Saved: correlation_matrix.png")
            plt.close(fig)
        except Exception as e:
            print(f"  ✗ Error creating correlation matrix: {e}")
    
    # 4. Parameter sensitivity plots
    if viz_config['create_comparison_plots'] and len(config['data_configs']) > 1:
        print("\n" + "-"*80)
        print("Creating parameter sensitivity plots...")
        print("-"*80)
        
        # Plot how performance varies with delta (jump rate)
        if 'delta' in best_results.columns:
            try:
                fig = plot_parameter_sensitivity(
                    best_results,
                    parameter='delta',
                    metric='balanced_accuracy',
                    title='Performance vs Jump Rate (delta)'
                )
                fig.savefig(plots_dir / "sensitivity_delta.png", dpi=viz_config['dpi'], bbox_inches='tight')
                print("  ✓ Saved: sensitivity_delta.png")
                plt.close(fig)
            except Exception as e:
                print(f"  ✗ Error creating delta sensitivity: {e}")
        
        # Plot how performance varies with distribution type
        if 'distribution_type' in best_results.columns:
            try:
                fig = plot_parameter_sensitivity(
                    best_results,
                    parameter='distribution_type',
                    metric='balanced_accuracy',
                    title='Performance vs Distribution Type'
                )
                fig.savefig(plots_dir / "sensitivity_distribution.png", dpi=viz_config['dpi'], bbox_inches='tight')
                print("  ✓ Saved: sensitivity_distribution.png")
                plt.close(fig)
            except Exception as e:
                print(f"  ✗ Error creating distribution sensitivity: {e}")
    
    # 5. Stacked time series visualization
    if viz_config['create_time_series'] and config['save_models']:
        print("\n" + "-"*80)
        print("Creating stacked time series visualization...")
        print("-"*80)
        
        models_dir = results_path / "models"
        if models_dir.exists() and list(models_dir.glob("*.pkl")):
            try:
                import pickle
                from simulation.data_generation import generate_data
                from simulation.models import GaussianJumpModel, PoissonJumpModel, PoissonKLJumpModel
                
                # Get best hyperparameters for each model from the first data configuration
                first_config_results = best_results.iloc[0]
                data_config_params = {
                    'n_samples': int(first_config_results['n_samples']),
                    'n_states': int(first_config_results['n_states']),
                    'n_informative': int(first_config_results['n_informative']),
                    'n_noise': int(first_config_results['n_noise']),
                    'n_total_features': int(first_config_results['n_total_features']),
                    'delta': float(first_config_results['delta']),
                    'lambda_0': float(first_config_results['lambda_0']),
                    'persistence': float(first_config_results['persistence']),
                    'distribution_type': str(first_config_results['distribution_type']),
                    'correlated_noise': bool(first_config_results['correlated_noise']),
                    'random_seed': int(first_config_results['random_seed']),
                }
                
                # Regenerate data (same for all models)
                sim_config = SimulationConfig(**data_config_params)
                X, states, breakpoints = generate_data(sim_config)
                
                # Refit models with best hyperparameters
                models_dict = {}
                
                for model_name in best_results['model_name'].unique():
                    # Get best hyperparameters for this model
                    model_best = best_results[best_results['model_name'] == model_name].iloc[0]
                    
                    n_components = int(model_best['best_n_components'])
                    jump_penalty = float(model_best['best_jump_penalty'])
                    
                    # Create and fit model with best hyperparameters
                    if model_name == 'Gaussian':
                        model = GaussianJumpModel(
                            n_components=n_components,
                            jump_penalty=jump_penalty,
                            random_state=42
                        )
                    elif model_name == 'Poisson':
                        model = PoissonJumpModel(
                            n_components=n_components,
                            jump_penalty=jump_penalty,
                            random_state=42
                        )
                    elif model_name == 'PoissonKL':
                        model = PoissonKLJumpModel(
                            n_components=n_components,
                            jump_penalty=jump_penalty,
                            random_state=42
                        )
                    else:
                        print(f"  ⚠ Unknown model: {model_name}, skipping...")
                        continue
                    
                    # Fit the model
                    model.fit(X)
                    models_dict[model_name] = model
                    print(f"  • Fitted {model_name} with n_components={n_components}, jump_penalty={jump_penalty}")
                
                if models_dict:
                    # Select first informative feature
                    informative_cols = [col for col in X.columns if col.startswith('informative_')]
                    if informative_cols:
                        feature_to_plot = informative_cols[0]
                        
                        # Create stacked plot
                        fig = plot_stacked_states(
                            X,
                            models_dict,
                            true_states=states,
                            feature_to_plot=feature_to_plot,
                            figsize=(14, 10)
                        )
                        
                        filename = "timeseries_stacked_comparison.png"
                        fig.savefig(plots_dir / filename, dpi=viz_config['dpi'], bbox_inches='tight')
                        print(f"  ✓ Saved: {filename}")
                        plt.close(fig)
                else:
                    print("  ⚠ No models fitted successfully.")
                    
            except Exception as e:
                print(f"  ✗ Error creating stacked time series: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  ⚠ No saved models found. Skipping time series plots.")
    
    print("\n" + "="*80)
    print(f"All plots saved to: {plots_dir}")
    print("="*80)


def main():
    """Run the full experiment."""
    
    print("="*80)
    print("SIMULATION EXPERIMENT")
    print("="*80)
    
    config = EXPERIMENT_CONFIGS
    
    # Display experiment info
    print(f"\nExperiment: {config['experiment_name']}")
    print(f"Replications: {config['n_replications']}")
    print(f"Configurations: {len(config['data_configs'])}")
    print(f"Models: {', '.join(config['model_names'])}")
    print(f"Optimization: {config['optimization']}")
    
    if config['optimization'] == 'grid':
        grid_size = (
            len(config['grid_config']['n_components']) *
            len(config['grid_config']['jump_penalty']) *
            len(config['grid_config']['kappa_min']) *
            len(config['grid_config']['kappa_max'])
        )
        print(f"Grid size: {grid_size} combinations per model")
    else:
        print(f"Optuna trials: {config['optuna_config']['n_trials']} per model")
    
    # Check for existing results
    experiment_base_dir = Path(__file__).parent / "experiment_results"
    existing_dirs = []
    if experiment_base_dir.exists():
        pattern = f"{config['experiment_name']}_*"
        existing_dirs = sorted(experiment_base_dir.glob(pattern), reverse=True)
    
    # If results exist, ask what to do
    skip_simulation = False
    out_dir = None
    
    if existing_dirs:
        latest_dir = existing_dirs[0]
        print(f"\n{'='*80}")
        print(f"Found existing results: {latest_dir.name}")
        print(f"{'='*80}")
        
        # Check if it has the expected files
        has_best = (latest_dir / "best_results.csv").exists()
        has_grid = (latest_dir / "grid_results.csv").exists()
        
        if has_best and has_grid:
            response = input("\nRe-run simulation? [y/N]: ")
            if response.lower() != 'y':
                skip_simulation = True
                out_dir = latest_dir
                print(f"\nUsing existing results from: {out_dir}")
                
                # Ask about recreating plots
                response = input("Recreate plots? [y/N]: ")
                if response.lower() == 'y':
                    create_visualizations(out_dir, config)
                    print("\n" + "="*80)
                    print("PLOTS RECREATED")
                    print("="*80)
                    print(f"\nPlots saved to: {out_dir / 'plots'}")
                    print("="*80)
                    return
                else:
                    print("\nNo action taken. Exiting.")
                    return
    
    # Create new output directory if running simulation
    if not skip_simulation:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(__file__).parent / "experiment_results" / f"{config['experiment_name']}_{timestamp}"
        print(f"\nResults will be saved to: {out_dir}")
    
    # Convert data configs to SimulationConfig objects
    data_configs = [SimulationConfig(**cfg) for cfg in config['data_configs']]
    
    # Confirm before running
    if not skip_simulation:
        total_fits = (
            len(data_configs) * 
            config['n_replications'] * 
            len(config['model_names'])
        )
        if config['optimization'] == 'grid':
            total_fits *= grid_size
        else:
            total_fits *= config['optuna_config']['n_trials']
        
        print(f"\nTotal model fits: ~{total_fits}")
        
        response = input("\nProceed with experiment? [y/N]: ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            return
    
    # Run simulation
    if not skip_simulation:
        print("\n" + "="*80)
        print("Running Simulations")
        print("="*80)
        
        start_time = datetime.now()
        
        # Run each data configuration
        all_results = []
        for i, data_cfg in enumerate(data_configs, 1):
            print(f"\n{'='*80}")
            print(f"Configuration {i}/{len(data_configs)}")
            print(f"{'='*80}")
            
            # Create experiment config for this data configuration
            if config['optimization'] == 'grid':
                n_components = config['grid_config']['n_components']
                jump_penalties = config['grid_config']['jump_penalty']
                kappa_min_vals = config['grid_config']['kappa_min']
                kappa_max_vals = config['grid_config']['kappa_max']
                
                hyperparameter_grid = {
                    'n_states_values': n_components,
                    'jump_penalty_min': min(jump_penalties),
                    'jump_penalty_max': max(jump_penalties),
                    'jump_penalty_num': len(jump_penalties),
                    'jump_penalty_scale': "log" if len(jump_penalties) > 2 else "linear",
                    'kappa_min': min(kappa_min_vals),
                    'kappa_max_type': "fixed",
                    'kappa_max_fixed': max(kappa_max_vals),
                    'kappa_num': len(kappa_min_vals),
                }
            else:
                hyperparameter_grid = None
            
            exp_config = ExperimentConfig(
                name=f"{config['experiment_name']}_config{i}",
                mode='single',
                data=data_cfg,
                n_replications=config['n_replications'],
                parallel=(config['n_jobs'] != 1),
                single_thread=(config['n_jobs'] == 1),
                model_names=config['model_names'],
                optimization_method=config['optimization'],
                optimize_metric=config['optimize_metric'],
                hyperparameter_grid=hyperparameter_grid,
                optuna_n_trials=config.get('optuna_config', {}).get('n_trials', 100),
                output_dir=str(out_dir),
            )
            
            # Run simulation for this configuration
            config_results = run_simulation(
                experiment_config=exp_config,
                data_configs=[data_cfg],
                output_dir=str(out_dir),
                n_jobs=config['n_jobs'],
                save_models=config['save_models'],
                verbose=True,
            )
            
            all_results.append(config_results)
        
        # Combine all results
        import pandas as pd
        results_df = pd.concat(all_results, ignore_index=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"Simulation Complete!")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"{'='*80}")
        
        # Display summary
        print(f"\nResults summary:")
        print(f"  Total evaluations: {len(results_df)}")
        print(f"  Best balanced accuracy: {results_df['balanced_accuracy'].max():.4f}")
        print(f"  Mean balanced accuracy: {results_df['balanced_accuracy'].mean():.4f}")
        
        # Create visualizations
        create_visualizations(out_dir, config)
        
        # Final summary
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {out_dir.absolute()}")
        print(f"\nNext steps:")
        print(f"  1. View plots in: {out_dir / 'plots'}")
        print(f"  2. Load results with: ResultManager('{out_dir}')")
        print(f"  3. Analyze with notebooks or custom scripts")
        print("="*80)


if __name__ == '__main__':
    main()
