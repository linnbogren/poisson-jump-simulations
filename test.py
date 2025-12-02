"""
Test Optuna visualization tools
"""

from simulation import run_simulation
from visualization import plot_optuna_trials, print_optuna_summary

if __name__ == "__main__":
    # Run a quick Optuna optimization
    config = {
        'experiment_name': 'optuna_viz_test',
        'num_simulations': 1,
        'data_generation': [{'delta': 0.15}],
        'models_to_run': ['Poisson'],
        'optimization': 'optuna',
        'optuna_n_trials': 10,
        'optuna_n_jobs': 1
    }
    
    print(f"DEBUG: Config optuna_n_trials = {config['optuna_n_trials']}")
    print("Running Optuna optimization...")
    results = run_simulation(config, cache=False, verbose=True)
    
    print("\n" + "="*80)
    print("OPTUNA TRIAL ANALYSIS")
    print("="*80)
    
    # Print summary and create visualization using output directory
    print_optuna_summary(output_dir=results.output_dir)
    
    # Create visualization
    print("\nCreating trial visualization plots...")
    plot_optuna_trials(output_dir=results.output_dir, show=True)
    
    print(f"\nPlots saved to: {results.output_dir}")
