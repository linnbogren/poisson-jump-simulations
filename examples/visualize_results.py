"""Example script demonstrating visualization capabilities.

This script shows how to use the visualization package to create various
plots for analyzing simulation results.

Usage:
    python examples/visualize_results.py              # Interactive mode
    python examples/visualize_results.py --all        # Run all examples
    python examples/visualize_results.py --example 1  # Run specific example
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.config import SimulationConfig, load_config
from simulation.data_generation import generate_data
from simulation.models import ModelWrapper
from simulation.results import ResultManager

# Import visualization functions
from visualization import (
    # Time series
    plot_time_series_with_breakpoints,
    plot_simulated_from_regimes,
    plot_stacked_states,
    
    # Comparison
    plot_model_comparison_bars,
    plot_parameter_sensitivity,
    plot_metric_comparison_grid,
    plot_hyperparameter_heatmap,
    
    # Results
    plot_metric_distributions,
    plot_correlation_matrix,
    create_summary_table,
    plot_summary_table,
    plot_aggregated_results_overview,
    
    # Utilities
    setup_plotting_style,
)


def example_1_time_series_visualization():
    """Example 1: Visualize time series with breakpoints."""
    print("\n" + "="*60)
    print("Example 1: Time Series Visualization")
    print("="*60)
    
    # Create simple data
    config = SimulationConfig(
        n_samples=300,
        n_states=3,
        n_informative=10,
        n_total_features=10,
        delta=0.5,
        lambda_0=10.0,
        persistence=0.97,
        distribution_type="Poisson",
    )
    
    X, states, breakpoints = generate_data(config)
    
    # Fit a model
    model_wrapper = ModelWrapper(
        model_name='Poisson',
        n_components=3,
        max_feats=8,
        jump_penalty=1.0,
    )
    
    model_wrapper.fit(X)
    model = model_wrapper.model
    
    # Plot 1: Time series with breakpoints
    print("\nCreating time series plot with breakpoints...")
    fig1 = plot_time_series_with_breakpoints(
        X[['informative_1', 'informative_2']],  # Plot first 2 features
        model,
        actual_breakpoints=breakpoints,
        title='Jump Model: Time Series with Breakpoints'
    )
    
    # Plot 2: Stacked states view
    print("Creating stacked states plot...")
    fig2 = plot_stacked_states(
        X,
        models_dict={'Sparse Jump Model': model},
        true_states=states,
        feature_to_plot='informative_1'
    )
    
    # Plot 3: Simulated data from regimes (use full dataset)
    print("Creating simulated data plot...")
    fig3 = plot_simulated_from_regimes(
        X,  # Use full dataset
        model,
        title='Simulated Data from Predicted Regimes'
    )
    
    print("\n✓ Created 3 time series visualizations")
    print("Note: In interactive mode, plots would be displayed.")
    print("      Use save_path parameter to save figures.")
    plt.close('all')


def example_2_model_comparison():
    """Example 2: Compare multiple models."""
    print("\n" + "="*60)
    print("Example 2: Model Comparison")
    print("="*60)
    
    # Create mock results dataframe
    np.random.seed(42)
    n_replications = 30
    
    models = ['SparseJumpPoisson', 'SparseJumpGaussian']
    metrics = ['balanced_accuracy', 'composite_score', 'breakpoint_f1', 
              'chamfer_distance']
    
    results_list = []
    for model in models:
        for rep in range(n_replications):
            # Generate realistic-looking metrics
            base_performance = 0.75 if 'Poisson' in model else 0.70
            result = {
                'model_name': model,
                'replication': rep,
                'balanced_accuracy': base_performance + np.random.normal(0, 0.05),
                'composite_score': base_performance + 0.05 + np.random.normal(0, 0.04),
                'breakpoint_f1': base_performance - 0.1 + np.random.normal(0, 0.06),
                'chamfer_distance': 15 + np.random.normal(0, 3),
                'n_selected_features': np.random.randint(5, 12),
            }
            results_list.append(result)
    
    results_df = pd.DataFrame(results_list)
    
    # Plot 1: Bar chart comparison
    print("\nCreating model comparison bar chart...")
    fig1 = plot_model_comparison_bars(
        results_df,
        metric='balanced_accuracy',
        title='Model Performance Comparison'
    )
    
    # Plot 2: Metric comparison grid
    print("Creating metric comparison grid...")
    fig2 = plot_metric_comparison_grid(
        results_df,
        metrics=['balanced_accuracy', 'composite_score', 'breakpoint_f1'],
        ncols=3
    )
    
    # Plot 3: Metric distributions
    print("Creating metric distribution plots...")
    fig3 = plot_metric_distributions(
        results_df,
        metrics=['balanced_accuracy', 'composite_score'],
        plot_type='violin'
    )
    
    print("\n✓ Created 3 comparison visualizations")
    plt.close('all')


def example_3_parameter_sensitivity():
    """Example 3: Parameter sensitivity analysis."""
    print("\n" + "="*60)
    print("Example 3: Parameter Sensitivity Analysis")
    print("="*60)
    
    # Create mock results with parameter variations
    np.random.seed(42)
    
    deltas = [0.2, 0.5, 0.8]
    lambdas = [5.0, 10.0, 15.0]
    gammas = [0.5, 1.0, 2.0, 4.0]
    kappas = [5, 8, 12]
    
    results_list = []
    for delta in deltas:
        for lambda_0 in lambdas:
            for _ in range(5):  # 5 replications each
                # Performance improves with higher delta
                base = 0.6 + 0.2 * delta
                results_list.append({
                    'model_name': 'SparseJumpPoisson',
                    'delta': delta,
                    'lambda_0': lambda_0,
                    'balanced_accuracy': base + np.random.normal(0, 0.03),
                    'composite_score': base + 0.05 + np.random.normal(0, 0.03),
                })
    
    results_df = pd.DataFrame(results_list)
    
    # Plot 1: Single parameter sensitivity
    print("\nCreating parameter sensitivity plot for delta...")
    fig1 = plot_parameter_sensitivity(
        results_df,
        parameter='delta',
        metric='balanced_accuracy',
        title='Performance vs Delta'
    )
    
    # Create hyperparameter grid results
    hp_results = []
    for gamma in gammas:
        for kappa in kappas:
            for _ in range(3):
                # Higher gamma and kappa generally better
                score = 0.6 + 0.05 * np.log(gamma) + 0.02 * kappa + np.random.normal(0, 0.02)
                hp_results.append({
                    'model_name': 'SparseJumpPoisson',
                    'gamma': gamma,
                    'kappa': kappa,
                    'balanced_accuracy': np.clip(score, 0, 1),
                })
    
    hp_df = pd.DataFrame(hp_results)
    
    # Plot 2: Hyperparameter heatmap
    print("Creating hyperparameter heatmap...")
    fig2 = plot_hyperparameter_heatmap(
        hp_df,
        param_x='gamma',
        param_y='kappa',
        metric='balanced_accuracy',
        model_name='SparseJumpPoisson'
    )
    
    print("\n✓ Created 2 parameter sensitivity visualizations")
    plt.close('all')


def example_4_result_aggregation():
    """Example 4: Aggregate result analysis."""
    print("\n" + "="*60)
    print("Example 4: Result Aggregation and Analysis")
    print("="*60)
    
    # Create comprehensive mock results
    np.random.seed(42)
    n_replications = 50
    
    models = ['SparseJumpPoisson', 'SparseJumpGaussian']
    
    results_list = []
    for model in models:
        for rep in range(n_replications):
            base = 0.75 if 'Poisson' in model else 0.70
            result = {
                'model_name': model,
                'replication': rep,
                'balanced_accuracy': base + np.random.normal(0, 0.05),
                'composite_score': base + 0.05 + np.random.normal(0, 0.04),
                'breakpoint_f1': base - 0.1 + np.random.normal(0, 0.06),
                'chamfer_distance': 15 + np.random.normal(0, 3),
            }
            results_list.append(result)
    
    results_df = pd.DataFrame(results_list)
    
    # Create summary table
    print("\nCreating summary statistics table...")
    summary = create_summary_table(
        results_df,
        metrics=['balanced_accuracy', 'composite_score', 'breakpoint_f1'],
        include_std=True
    )
    print("\nSummary Statistics:")
    print(summary)
    
    # Plot 1: Summary table visualization
    print("\nCreating summary table plot...")
    fig1 = plot_summary_table(summary)
    
    # Plot 2: Correlation matrix
    print("Creating correlation matrix...")
    fig2 = plot_correlation_matrix(
        results_df,
        metrics=['balanced_accuracy', 'composite_score', 'breakpoint_f1', 
                'chamfer_distance']
    )
    
    # Plot 3: Comprehensive overview
    print("Creating comprehensive overview dashboard...")
    fig3 = plot_aggregated_results_overview(
        results_df,
        key_metrics=['balanced_accuracy', 'composite_score', 'breakpoint_f1']
    )
    
    print("\n✓ Created 3 aggregated result visualizations")
    plt.close('all')


def example_5_load_and_visualize_saved_results():
    """Example 5: Load and visualize saved results."""
    print("="*60)
    print("Example 5: Load and Visualize Saved Results")
    print("="*60)
    
    # Check if we have any saved results
    results_dir = Path("results")
    
    if not results_dir.exists() or not any(results_dir.iterdir()):
        print("\nNo saved results found. Run a simulation first!")
        print("Example: poetry run python examples/run_single_example.py")
        return
    
    results_manager = ResultManager(results_dir=results_dir)
    
    if not results_dir.exists() or not any(results_dir.iterdir()):
        print("\nNo saved results found. Run a simulation first!")
        print("Example: poetry run python examples/run_single_example.py")
        return
    
    # Try to load results
    try:
        # Find most recent experiment
        experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        if not experiment_dirs:
            print("\nNo experiment directories found.")
            return
        
        latest_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
        print(f"\nLoading results from: {latest_dir}")
        
        # Load best results
        best_results = results_manager.load_best_results(latest_dir)
        
        if best_results is None or best_results.empty:
            print("No results found in directory.")
            return
        
        print(f"\nLoaded {len(best_results)} results")
        print("\nColumns:", best_results.columns.tolist())
        print("\nFirst few rows:")
        print(best_results.head())
        
        # Create visualizations
        if 'balanced_accuracy' in best_results.columns:
            print("\nCreating model comparison...")
            fig1 = plot_model_comparison_bars(
                best_results,
                metric='balanced_accuracy',
                model_column='distance_metric',
                title='Loaded Results: Model Comparison'
            )
        
        print("\n✓ Visualized saved results")
        plt.close('all')
        
    except Exception as e:
        print(f"\nError loading results: {e}")
        print("This is expected if no simulations have been run yet.")


def main():
    """Run all visualization examples."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualization examples')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    parser.add_argument('--example', type=int, choices=[1,2,3,4,5], help='Run specific example')
    parser.add_argument('--non-interactive', action='store_true', help='Non-interactive mode (no plots shown)')
    args = parser.parse_args()
    
    # Set non-interactive backend if requested
    if args.non_interactive:
        import matplotlib
        matplotlib.use('Agg')
    
    print("\n" + "="*70)
    print("VISUALIZATION PACKAGE EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates the visualization capabilities.")
    print("\nExamples:")
    print("  1. Time series visualization")
    print("  2. Model comparison")
    print("  3. Parameter sensitivity")
    print("  4. Result aggregation")
    print("  5. Load and visualize saved results")
    
    # Setup plotting style
    setup_plotting_style(context='notebook', font_scale=1.0)
    
    # Run examples
    examples = [
        example_1_time_series_visualization,
        example_2_model_comparison,
        example_3_parameter_sensitivity,
        example_4_result_aggregation,
        example_5_load_and_visualize_saved_results,
    ]
    
    if args.all:
        print("\nRunning all examples...")
        for example in examples:
            example()
    elif args.example is not None:
        print(f"\nRunning example {args.example}...")
        examples[args.example - 1]()
    else:
        # Interactive mode
        print("\n" + "-"*70)
        choice = input("\nRun all examples? (y/n, or enter number 1-5 for specific): ").strip().lower()
        
        if choice == 'y' or choice == 'yes':
            for example in examples:
                example()
        elif choice.isdigit() and 1 <= int(choice) <= 5:
            examples[int(choice) - 1]()
        else:
            print("Invalid choice. Use --all or --example <n> for non-interactive mode.")
            return
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nTip: Check the visualization/ package for more functions:")
    print("  - visualization.time_series")
    print("  - visualization.comparison")
    print("  - visualization.results")
    print("  - visualization.utils")


if __name__ == "__main__":
    main()
