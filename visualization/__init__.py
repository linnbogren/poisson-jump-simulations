"""Visualization package for Poisson Jump simulations.

This package provides comprehensive visualization tools for analyzing
simulation results, including:
- Time series and state sequence visualizations
- Model performance comparisons
- Parameter sensitivity analyses
- Result aggregation and distribution plots
"""

# Import utility functions
from .utils import (
    setup_plotting_style,
    get_state_colors,
    get_model_color,
    get_metric_color,
    save_figure,
    format_parameter_name,
    format_metric_name,
)

# Import time series visualization functions
from .time_series import (
    plot_time_series_with_breakpoints,
    plot_simulated_from_regimes,
    plot_stacked_states,
    plot_multiple_series_comparison,
)

# Import comparison visualization functions
from .comparison import (
    plot_model_comparison_bars,
    plot_parameter_sensitivity,
    plot_multiple_parameter_sensitivity,
    plot_hyperparameter_heatmap,
    plot_metric_comparison_grid,
    plot_performance_vs_complexity,
)

# Import result aggregation functions
from .results import (
    plot_metric_distributions,
    plot_correlation_matrix,
    create_summary_table,
    plot_summary_table,
    plot_pairwise_metric_scatter,
    plot_metric_evolution,
    plot_aggregated_results_overview,
)

__all__ = [
    # Utilities
    'setup_plotting_style',
    'get_state_colors',
    'get_model_color',
    'get_metric_color',
    'save_figure',
    'format_parameter_name',
    'format_metric_name',
    
    # Time series
    'plot_time_series_with_breakpoints',
    'plot_simulated_from_regimes',
    'plot_stacked_states',
    'plot_multiple_series_comparison',
    
    # Comparison
    'plot_model_comparison_bars',
    'plot_parameter_sensitivity',
    'plot_multiple_parameter_sensitivity',
    'plot_hyperparameter_heatmap',
    'plot_metric_comparison_grid',
    'plot_performance_vs_complexity',
    
    # Results
    'plot_metric_distributions',
    'plot_correlation_matrix',
    'create_summary_table',
    'plot_summary_table',
    'plot_pairwise_metric_scatter',
    'plot_metric_evolution',
    'plot_aggregated_results_overview',
]
