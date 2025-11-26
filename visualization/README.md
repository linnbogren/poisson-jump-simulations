# Visualization Package

Comprehensive visualization tools for analyzing Poisson Jump simulation results.

## Overview

The visualization package provides four main modules:

1. **utils.py** - Common plotting utilities and configurations
2. **time_series.py** - Time series and state sequence visualizations
3. **comparison.py** - Model performance comparisons
4. **results.py** - Result aggregation and distribution plots

## Quick Start

```python
from visualization import (
    plot_time_series_with_breakpoints,
    plot_stacked_states,
    plot_model_comparison_bars,
    plot_parameter_sensitivity,
    plot_metric_distributions,
)

# Visualize time series with breakpoints
fig = plot_time_series_with_breakpoints(
    X, model, actual_breakpoints=[100, 200],
    title='Jump Model Results'
)

# Compare models
fig = plot_model_comparison_bars(
    results_df, metric='balanced_accuracy'
)

# Parameter sensitivity
fig = plot_parameter_sensitivity(
    results_df, parameter='delta', metric='balanced_accuracy'
)
```

## Module Details

### visualization.utils

Common utilities used across all visualization modules:

- `setup_plotting_style()` - Configure matplotlib/seaborn styles
- `get_state_colors()` - Get color mappings for states
- `get_model_color()` - Get color for specific model
- `save_figure()` - Save figures with standard settings
- `format_parameter_name()` - Format parameter names with LaTeX
- `format_metric_name()` - Format metric names for display

### visualization.time_series

Time series and state sequence visualizations:

- `plot_time_series_with_breakpoints()` - Plot time series with predicted/true breakpoints
- `plot_simulated_from_regimes()` - Simulate and plot data from predicted regimes
- `plot_stacked_states()` - Stacked plot showing data and state assignments
- `plot_multiple_series_comparison()` - Compare multiple time series

### visualization.comparison

Model performance comparison functions:

- `plot_model_comparison_bars()` - Bar chart comparing models across a metric
- `plot_parameter_sensitivity()` - Line plot showing parameter sensitivity
- `plot_multiple_parameter_sensitivity()` - Grid of sensitivity plots
- `plot_hyperparameter_heatmap()` - Heatmap of hyperparameter performance
- `plot_metric_comparison_grid()` - Grid comparing models across metrics
- `plot_performance_vs_complexity()` - Performance vs complexity trade-off

### visualization.results

Result aggregation and analysis:

- `plot_metric_distributions()` - Violin/box/strip plots of metric distributions
- `plot_correlation_matrix()` - Correlation heatmap of metrics
- `create_summary_table()` - Generate summary statistics table
- `plot_summary_table()` - Visualize summary table
- `plot_pairwise_metric_scatter()` - Scatter plot of two metrics
- `plot_metric_evolution()` - Metric evolution over replications
- `plot_aggregated_results_overview()` - Comprehensive dashboard

## Examples

See `examples/visualize_results.py` for comprehensive examples:

```bash
poetry run python examples/visualize_results.py
```

This demonstrates:
1. Time series visualization
2. Model comparison
3. Parameter sensitivity analysis
4. Result aggregation
5. Loading and visualizing saved results

## Color Schemes

### State Colors
- State 0: Red (#e74c3c)
- State 1: Blue (#3498db)
- State 2: Green (#2ecc71)
- State 3: Orange (#f39c12)
- Additional states use colormap

### Model Colors
- SparseJumpPoisson: Green (#2ecc71)
- SparseJumpGaussian: Blue (#3498db)
- JumpPoisson: Red (#e74c3c)
- JumpGaussian: Orange (#f39c12)

## Customization

All plot functions support:
- `figsize` - Figure size (width, height)
- `save_path` - Path to save figure
- `title` - Custom plot title

Example:
```python
fig = plot_model_comparison_bars(
    results_df,
    metric='balanced_accuracy',
    figsize=(12, 6),
    title='My Custom Title',
    save_path='results/figures/comparison.png'
)
```

## Typical Workflow

1. **Run simulation** and save results
2. **Load results** using ResultManager
3. **Create visualizations**:
   - Time series plots for individual runs
   - Comparison plots for multiple models
   - Parameter sensitivity for grid searches
   - Aggregated overview for final analysis
4. **Save figures** for reports/papers

```python
from simulation.results import ResultManager
from visualization import plot_aggregated_results_overview

# Load results
manager = ResultManager()
results_df = manager.load_best_results('results/my_experiment')

# Create overview
fig = plot_aggregated_results_overview(
    results_df,
    key_metrics=['balanced_accuracy', 'composite_score', 'breakpoint_f1'],
    save_path='results/figures/overview.png'
)
```

## Dependencies

- matplotlib >= 3.7
- seaborn >= 0.12
- numpy
- pandas

All dependencies are managed via `pyproject.toml`.
