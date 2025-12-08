"""Model comparison and parameter sensitivity visualization functions.

This module provides functions for comparing model performance across metrics
and visualizing parameter sensitivity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Union, Any
from pathlib import Path

from .utils import (
    get_model_color,
    get_metric_color,
    format_metric_name,
    format_parameter_name,
    save_figure,
    add_grid,
    add_value_labels,
)


def plot_model_comparison_bars(
    results_df: pd.DataFrame,
    metric: str = 'balanced_accuracy',
    model_column: str = 'model_name',
    show_error_bars: bool = True,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create bar chart comparing models across a metric.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns [model_column, metric, ...].
    metric : str, default='balanced_accuracy'
        Metric to compare.
    model_column : str, default='model_name'
        Column name containing model names.
    show_error_bars : bool, default=True
        Whether to show standard error bars.
    title : str, optional
        Plot title. If None, auto-generated.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
        
    Examples
    --------
    >>> fig = plot_model_comparison_bars(
    ...     results_df, metric='balanced_accuracy',
    ...     title='Model Performance Comparison'
    ... )
    """
    # Calculate statistics
    stats = results_df.groupby(model_column)[metric].agg(['mean', 'std', 'sem']).reset_index()
    stats = stats.sort_values('mean', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors for each model
    colors = [get_model_color(model) for model in stats[model_column]]
    
    # Create bars
    x_pos = np.arange(len(stats))
    bars = ax.bar(x_pos, stats['mean'], 
                  yerr=stats['sem'] if show_error_bars else None,
                  color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
    
    # Customize
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats[model_column], rotation=45, ha='right')
    ax.set_ylabel(format_metric_name(metric), fontsize=16)
    ax.set_xlabel('Model', fontsize=16)
    
    add_grid(ax, alpha=0.3)
    
    # Add value labels
    add_value_labels(ax, bars, format_str='{:.3f}')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_parameter_sensitivity(
    results_df: pd.DataFrame,
    parameter: str,
    metric: str = 'balanced_accuracy',
    model_column: str = 'model_name',
    models: Optional[List[str]] = None,
    show_ci: bool = True,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot parameter sensitivity across models.
    
    Shows how a metric varies with a parameter for different models.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    parameter : str
        Parameter name (e.g., 'delta', 'lambda_0').
    metric : str, default='balanced_accuracy'
        Metric to plot on y-axis.
    model_column : str, default='model_name'
        Column containing model names.
    models : list, optional
        List of models to include. If None, uses all.
    show_ci : bool, default=True
        Whether to show confidence intervals.
    title : str, optional
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    # Filter models if specified
    if models is not None:
        results_df = results_df[results_df[model_column].isin(models)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique models
    unique_models = results_df[model_column].unique()
    
    for model_name in unique_models:
        model_data = results_df[results_df[model_column] == model_name]
        
        # Group by parameter and calculate statistics
        stats = model_data.groupby(parameter)[metric].agg(['mean', 'std', 'sem']).reset_index()
        stats = stats.sort_values(parameter)
        
        color = get_model_color(model_name)
        
        # Plot mean line
        ax.plot(stats[parameter], stats['mean'], 
               marker='o', label=model_name, color=color, 
               linewidth=2, markersize=8)
        
        # Add confidence interval
        if show_ci:
            ax.fill_between(stats[parameter], 
                           stats['mean'] - 1.96 * stats['sem'],
                           stats['mean'] + 1.96 * stats['sem'],
                           alpha=0.2, color=color)
    
    # Customize
    ax.set_xlabel(format_parameter_name(parameter), fontsize=16)
    ax.set_ylabel(format_metric_name(metric), fontsize=16)
    
    add_grid(ax)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_multiple_parameter_sensitivity(
    results_df: pd.DataFrame,
    parameters: List[str],
    metric: str = 'balanced_accuracy',
    model_column: str = 'model_name',
    models: Optional[List[str]] = None,
    ncols: int = 2,
    figsize: tuple = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot sensitivity to multiple parameters in a grid.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    parameters : list of str
        List of parameter names to plot.
    metric : str
        Metric to plot.
    model_column : str
        Column containing model names.
    models : list, optional
        Models to include.
    ncols : int, default=2
        Number of columns in subplot grid.
    figsize : tuple, default=(14, 10)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    n_params = len(parameters)
    nrows = (n_params + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_params > 1 else [axes]
    
    # Filter models if specified
    if models is not None:
        results_df = results_df[results_df[model_column].isin(models)]
    
    unique_models = results_df[model_column].unique()
    
    for idx, parameter in enumerate(parameters):
        ax = axes[idx]
        
        for model_name in unique_models:
            model_data = results_df[results_df[model_column] == model_name]
            
            # Check if parameter exists in data
            if parameter not in model_data.columns:
                continue
            
            stats = model_data.groupby(parameter)[metric].agg(['mean', 'sem']).reset_index()
            stats = stats.sort_values(parameter)
            
            color = get_model_color(model_name)
            
            ax.plot(stats[parameter], stats['mean'], 
                   marker='o', label=model_name, color=color, linewidth=2)
            ax.fill_between(stats[parameter], 
                           stats['mean'] - 1.96 * stats['sem'],
                           stats['mean'] + 1.96 * stats['sem'],
                           alpha=0.2, color=color)
        
        ax.set_xlabel(format_parameter_name(parameter), fontsize=11)
        ax.set_ylabel(format_metric_name(metric), fontsize=11)
        ax.set_title(format_parameter_name(parameter), fontsize=12)
        add_grid(ax, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=14)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    # Main title removed per user request
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_hyperparameter_heatmap(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = 'balanced_accuracy',
    model_name: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create heatmap showing metric across two hyperparameters.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    param_x : str
        Hyperparameter for x-axis (e.g., 'gamma').
    param_y : str
        Hyperparameter for y-axis (e.g., 'kappa').
    metric : str, default='balanced_accuracy'
        Metric to visualize.
    model_name : str, optional
        Filter to specific model.
    title : str, optional
        Plot title.
    figsize : tuple, default=(10, 8)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    # Filter to model if specified
    if model_name is not None:
        results_df = results_df[results_df['model_name'] == model_name]
    
    # Create pivot table
    pivot = results_df.pivot_table(
        values=metric,
        index=param_y,
        columns=param_x,
        aggfunc='mean'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', 
               cbar_kws={'label': format_metric_name(metric)},
               ax=ax, linewidths=0.5)
    
    ax.set_xlabel(format_parameter_name(param_x), fontsize=12)
    ax.set_ylabel(format_parameter_name(param_y), fontsize=12)
    
    if title is None:
        title = f'Hyperparameter Grid: {format_metric_name(metric)}'
        if model_name:
            title += f' ({model_name})'
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_metric_comparison_grid(
    results_df: pd.DataFrame,
    metrics: List[str],
    model_column: str = 'model_name',
    ncols: int = 2,
    figsize: tuple = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create grid of bar charts comparing models across multiple metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    metrics : list of str
        List of metrics to compare.
    model_column : str
        Column containing model names.
    ncols : int, default=2
        Number of columns in grid.
    figsize : tuple, default=(14, 10)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    n_metrics = len(metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Calculate statistics
        stats = results_df.groupby(model_column)[metric].agg(['mean', 'sem']).reset_index()
        stats = stats.sort_values('mean', ascending=False)
        
        # Get colors
        colors = [get_model_color(model) for model in stats[model_column]]
        
        # Create bars
        x_pos = np.arange(len(stats))
        bars = ax.bar(x_pos, stats['mean'], yerr=stats['sem'],
                     color=colors, alpha=0.8, capsize=3, 
                     edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats[model_column], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(format_metric_name(metric), fontsize=16)
        # Subplot titles removed per user request
        add_grid(ax, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # Main title removed per user request
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_performance_vs_complexity(
    results_df: pd.DataFrame,
    complexity_metric: str = 'n_selected_features',
    performance_metric: str = 'balanced_accuracy',
    model_column: str = 'model_name',
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot performance vs model complexity trade-off.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    complexity_metric : str, default='n_selected_features'
        Metric representing model complexity.
    performance_metric : str, default='balanced_accuracy'
        Performance metric.
    model_column : str
        Column containing model names.
    title : str, optional
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_models = results_df[model_column].unique()
    
    for model_name in unique_models:
        model_data = results_df[results_df[model_column] == model_name]
        
        color = get_model_color(model_name)
        
        ax.scatter(model_data[complexity_metric], 
                  model_data[performance_metric],
                  color=color, label=model_name, 
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(format_metric_name(complexity_metric), fontsize=12)
    ax.set_ylabel(format_metric_name(performance_metric), fontsize=12)
    
    if title is None:
        title = 'Performance vs Complexity Trade-off'
    ax.set_title(title, fontsize=14, pad=20)
    
    add_grid(ax)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig
