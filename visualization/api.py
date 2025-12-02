"""
Simplified Visualization API

This module provides automatic visualization creation with smart defaults
based on experiment configuration and available data.

Main function:
- visualize_results(): Auto-detects what to plot and creates comprehensive visualizations
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd


def visualize_results(
    results: Union['SimulationResults', str, Path],
    config: Optional[Dict] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create visualization suite for simulation results.
    
    Automatically infers what to plot based on the experiment configuration
    and results structure. Can be customized with optional config.
    
    Parameters
    ----------
    results : SimulationResults, str, or Path
        Results object or path to results directory
    config : dict, optional
        Visualization configuration (overrides auto-detection):
        - metrics: List[str] - Metrics to compare (default: auto-detect from optimize_metric)
        - vary_parameters: List[str] - Parameters to show sensitivity for (default: auto-detect)
        - create_time_series: bool (default: True if models saved)
        - create_heatmaps: bool (default: True if grid search used)
        - create_sensitivity: bool (default: True if multiple configs)
        - create_comparison_plots: bool (default: True)
        - create_overview_plot: bool (default: True)
        - create_correlation_matrix: bool (default: True)
        - dpi: int (default: 300)
    output_dir : str or Path, optional
        Override output directory (default: results.path / 'plots')
        
    Returns
    -------
    Path
        Directory where plots were saved
        
    Examples
    --------
    >>> # Automatic visualization
    >>> visualize_results(results)
    
    >>> # Custom metrics
    >>> visualize_results(results, config={
    ...     'metrics': ['balanced_accuracy', 'feature_f1'],
    ...     'create_time_series': False,
    ... })
    
    Notes
    -----
    Auto-detection logic:
    - Reads metadata.json to get optimization_method and optimize_metric
    - Detects which parameters vary across data_configs
    - Creates appropriate plots based on what data is available
    """
    # Import here to avoid circular dependencies
    from simulation.api import SimulationResults
    from simulation.results import ResultManager
    from .experiment_plots import create_experiment_visualizations
    
    # Handle different input types
    if isinstance(results, (str, Path)):
        results = SimulationResults(results)
    
    # Determine output directory
    if output_dir is None:
        output_dir = results.path / 'plots'
    else:
        output_dir = Path(output_dir)
    
    # Build visualization config with auto-detection
    viz_config = _build_visualization_config(results, config)
    
    # Create visualizations
    create_experiment_visualizations(results.path, viz_config)
    
    return Path(output_dir)


def _build_visualization_config(
    results: 'SimulationResults',
    user_config: Optional[Dict] = None
) -> Dict:
    """
    Build visualization config with auto-detection and user overrides.
    
    Parameters
    ----------
    results : SimulationResults
        Results object
    user_config : dict, optional
        User-provided configuration overrides
        
    Returns
    -------
    dict
        Complete visualization configuration
    """
    # Start with defaults
    viz_config = {
        'create_comparison_plots': True,
        'create_overview_plot': True,
        'create_correlation_matrix': True,
        'create_unsupervised_plots': False,  # Auto-detect below
        'create_heatmaps': False,
        'create_time_series': False,
        'create_sensitivity': False,
        'dpi': 300,
        'figsize_default': (10, 6),
        'figsize_large': (14, 10),
    }
    
    # Auto-detect metrics to plot
    viz_config['comparison_metrics'] = _detect_metrics_to_plot(results)
    
    # Check if unsupervised metrics are available
    unsupervised_metrics = ['bic', 'aic', 'silhouette']
    has_unsupervised = any(
        metric in results.best_df.columns and results.best_df[metric].notna().any()
        for metric in unsupervised_metrics
    )
    if has_unsupervised:
        viz_config['create_unsupervised_plots'] = True
    
    # Auto-detect varying parameters
    varying_params = _detect_varying_parameters(results)
    if varying_params:
        viz_config['create_sensitivity'] = True
        viz_config['vary_parameters'] = varying_params
    
    # Check if models are saved
    models_dir = results.path / 'models'
    if models_dir.exists() and any(models_dir.iterdir()):
        viz_config['create_time_series'] = True
    
    # Check if grid search was used
    if results.metadata.get('optimization_method') == 'grid':
        viz_config['create_heatmaps'] = True
        viz_config['heatmap_params'] = [('best_jump_penalty', 'best_n_components')]
    
    # Apply user overrides
    if user_config:
        viz_config.update(user_config)
    
    return viz_config


def _detect_metrics_to_plot(results: 'SimulationResults') -> List[str]:
    """
    Auto-detect which metrics to include in comparison plots.
    
    Prioritizes the optimization metric and adds common important metrics.
    """
    metrics = []
    
    # Add optimization metric first
    optimize_metric = results.metadata.get('optimize_metric', 'balanced_accuracy')
    metrics.append(optimize_metric)
    
    # Add other important metrics
    important_metrics = [
        'balanced_accuracy',
        'composite_score',
        'feature_f1',
        'chamfer_distance',
        'breakpoint_count_error',
        # Unsupervised metrics (if available)
        'bic',
        'aic',
        'silhouette',
    ]
    
    for metric in important_metrics:
        if metric not in metrics and metric in results.best_df.columns:
            # Check if metric has non-null values
            if results.best_df[metric].notna().any():
                metrics.append(metric)
    
    # Limit to 6 metrics to avoid too many plots (increased from 4 to accommodate unsupervised)
    return metrics[:6]


def _detect_varying_parameters(results: 'SimulationResults') -> List[str]:
    """
    Detect which parameters vary across data configurations.
    
    Returns list of parameter names that have more than one unique value.
    """
    config_df = results.configs
    
    if len(config_df) <= 1:
        return []
    
    varying_params = []
    
    # Parameters to check
    params_to_check = [
        'delta', 'n_samples', 'n_states', 'n_informative',
        'n_total_features', 'lambda_0', 'persistence',
        'distribution_type', 'correlated_noise'
    ]
    
    for param in params_to_check:
        if param in config_df.columns:
            if config_df[param].nunique() > 1:
                varying_params.append(param)
    
    return varying_params


def create_comparison_plot(
    results: 'SimulationResults',
    metric: str = 'balanced_accuracy',
    save_path: Optional[Path] = None
) -> Any:
    """
    Create single comparison plot for a specific metric.
    
    Parameters
    ----------
    results : SimulationResults
        Results object
    metric : str, default='balanced_accuracy'
        Metric to plot
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    from .comparison import plot_model_comparison_bars
    
    fig = plot_model_comparison_bars(
        results.best_df,
        metric=metric,
        title=f'Model Performance: {metric.replace("_", " ").title()}',
        save_path=save_path
    )
    
    return fig


def create_overview_dashboard(
    results: 'SimulationResults',
    save_path: Optional[Path] = None
) -> Any:
    """
    Create comprehensive overview dashboard.
    
    Parameters
    ----------
    results : SimulationResults
        Results object
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    from .results import plot_aggregated_results_overview
    
    fig = plot_aggregated_results_overview(
        results.best_df,
        key_metrics=['balanced_accuracy', 'feature_f1', 'chamfer_distance'],
        figsize=(14, 10),
        save_path=save_path
    )
    
    return fig


def compare_optimization_methods(
    results_dict: Dict[str, 'SimulationResults'],
    output_dir: Optional[Union[str, Path]] = None,
    metrics: Optional[List[str]] = None
) -> Path:
    """
    Compare performance and timing across different optimization methods.
    
    Creates visualizations comparing:
    - Performance metrics (accuracy, F1, etc.)
    - Execution time and efficiency
    - Number of evaluations vs performance
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to SimulationResults objects.
        Example: {'Grid': grid_results, 'Optuna': optuna_results}
    output_dir : str or Path, optional
        Directory to save comparison plots. If None, uses current directory.
    metrics : list of str, optional
        Metrics to compare. If None, uses default set.
        
    Returns
    -------
    Path
        Directory where comparison plots were saved
        
    Examples
    --------
    >>> grid_results = run_simulation(config_grid)
    >>> optuna_results = run_simulation(config_optuna)
    >>> compare_optimization_methods({
    ...     'Grid Search': grid_results,
    ...     'Optuna': optuna_results
    ... })
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set output directory
    if output_dir is None:
        output_dir = Path('comparison_results')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default metrics
    if metrics is None:
        metrics = ['balanced_accuracy', 'feature_f1', 'chamfer_distance', 
                  'breakpoint_count_error', 'composite_score']
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION METHOD COMPARISON")
    print("=" * 80)
    
    # Collect timing and performance data
    comparison_data = []
    for method_name, results in results_dict.items():
        timing = results.get_timing_info()
        perf = results.get_performance_summary(metrics)
        
        print(f"\n{method_name}:")
        print(f"  Total time: {timing['total_time']:.1f}s ({timing['total_time']/60:.1f} min)")
        print(f"  Evaluations: {timing['n_evaluations']}")
        print(f"  Time per evaluation: {timing['time_per_evaluation']:.3f}s")
        
        for _, row in perf.iterrows():
            comparison_data.append({
                'method': method_name,
                'metric': row['metric'],
                'mean': row['mean'],
                'std': row['std'],
                'min': row['min'],
                'max': row['max'],
                'total_time': timing['total_time'],
                'n_evaluations': timing['n_evaluations'],
                'time_per_eval': timing['time_per_evaluation'],
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create comparison plots
    print("\nCreating comparison visualizations...")
    
    # 1. Performance comparison
    available_metrics = [m for m in metrics if m in comp_df['metric'].values]
    n_metrics = len(available_metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        metric_data = comp_df[comp_df['metric'] == metric]
        
        x_pos = np.arange(len(metric_data))
        ax.bar(x_pos, metric_data['mean'], yerr=metric_data['std'], 
               capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_data['method'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    perf_path = output_dir / 'performance_comparison.png'
    plt.savefig(perf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {perf_path.name}")
    
    # 2. Timing comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Total execution time
    timing_data = comp_df.groupby('method').first()
    methods = timing_data.index.tolist()
    times = timing_data['total_time'].values
    
    ax1.bar(range(len(methods)), times / 60, alpha=0.7)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Time (minutes)')
    ax1.set_title('Total Execution Time')
    ax1.grid(axis='y', alpha=0.3)
    
    # Number of evaluations
    n_evals = timing_data['n_evaluations'].values
    ax2.bar(range(len(methods)), n_evals, alpha=0.7, color='orange')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Number of Evaluations')
    ax2.set_title('Total Hyperparameter Evaluations')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    timing_path = output_dir / 'timing_comparison.png'
    plt.savefig(timing_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {timing_path.name}")
    
    # 3. Efficiency plot (performance vs time)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        metric_data = comp_df[comp_df['metric'] == metric]
        
        for method in metric_data['method'].unique():
            method_data = metric_data[metric_data['method'] == method]
            ax.scatter(method_data['total_time'] / 60, method_data['mean'], 
                      s=100, label=method, alpha=0.7)
            ax.errorbar(method_data['total_time'] / 60, method_data['mean'],
                       yerr=method_data['std'], fmt='none', alpha=0.3)
        
        ax.set_xlabel('Execution Time (minutes)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Time')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    efficiency_path = output_dir / 'efficiency_comparison.png'
    plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {efficiency_path.name}")
    
    # Save comparison data to CSV
    csv_path = output_dir / 'comparison_summary.csv'
    comp_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path.name}")
    
    print("\n" + "=" * 80)
    print(f"All comparison plots saved to: {output_dir}")
    print("=" * 80)
    
    return output_dir

