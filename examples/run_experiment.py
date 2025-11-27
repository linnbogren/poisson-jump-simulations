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
from visualization import create_experiment_visualizations


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
                    create_experiment_visualizations(out_dir, VISUALIZATION_CONFIG)
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
        create_experiment_visualizations(out_dir, VISUALIZATION_CONFIG)
        
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
