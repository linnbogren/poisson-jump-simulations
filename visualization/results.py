"""Result aggregation and distribution visualization functions.

This module provides functions for analyzing and visualizing aggregated results,
including metric distributions, summary statistics, and correlation analyses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Union, Any
from pathlib import Path

from .utils import (
    get_model_color,
    format_metric_name,
    format_parameter_name,
    save_figure,
    add_grid,
)


def plot_metric_distributions(
    results_df: pd.DataFrame,
    metrics: List[str],
    model_column: str = 'model_name',
    plot_type: str = 'violin',
    title: Optional[str] = None,
    figsize: tuple = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot distribution of metrics across models.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    metrics : list of str
        List of metrics to plot.
    model_column : str, default='model_name'
        Column containing model names.
    plot_type : str, default='violin'
        Type of plot: 'violin', 'box', or 'strip'.
    title : str, optional
        Plot title.
    figsize : tuple, default=(14, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        if plot_type == 'violin':
            sns.violinplot(data=results_df, x=model_column, y=metric, ax=ax,
                          palette=[get_model_color(m) for m in results_df[model_column].unique()])
        elif plot_type == 'box':
            sns.boxplot(data=results_df, x=model_column, y=metric, ax=ax,
                       palette=[get_model_color(m) for m in results_df[model_column].unique()])
        elif plot_type == 'strip':
            sns.stripplot(data=results_df, x=model_column, y=metric, ax=ax,
                         palette=[get_model_color(m) for m in results_df[model_column].unique()],
                         alpha=0.6)
        
        ax.set_xlabel('Model', fontsize=16)
        ax.set_ylabel(format_metric_name(metric), fontsize=16)
        # Subplot titles removed per user request
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        add_grid(ax, alpha=0.3)
    
    # Main title removed per user request
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_correlation_matrix(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = 'Metric Correlation Matrix',
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot correlation matrix of metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    metrics : list of str, optional
        Metrics to include. If None, uses all numeric columns.
    title : str
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
    if metrics is None:
        # Use all numeric columns
        metrics = results_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Compute correlation matrix
    corr = results_df[metrics].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, vmin=-1, vmax=1,
               square=True, linewidths=0.5, ax=ax,
               cbar_kws={'label': 'Correlation'})
    
    # Format labels
    ax.set_xticklabels([format_metric_name(m) for m in metrics], 
                      rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels([format_metric_name(m) for m in metrics], 
                      rotation=0, fontsize=14)
    
    # Title removed per user request
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def create_summary_table(
    results_df: pd.DataFrame,
    metrics: List[str],
    model_column: str = 'model_name',
    include_std: bool = True,
    include_median: bool = False,
) -> pd.DataFrame:
    """Create summary statistics table for models.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    metrics : list of str
        Metrics to summarize.
    model_column : str, default='model_name'
        Column containing model names.
    include_std : bool, default=True
        Include standard deviation.
    include_median : bool, default=False
        Include median values.
        
    Returns
    -------
    pd.DataFrame
        Summary table with multi-level columns.
    """
    # Aggregation functions
    agg_funcs = ['mean', 'count']
    if include_std:
        agg_funcs.append('std')
    if include_median:
        agg_funcs.append('median')
    
    # Group and aggregate
    summary = results_df.groupby(model_column)[metrics].agg(agg_funcs)
    
    # Round values
    summary = summary.round(4)
    
    return summary


def plot_summary_table(
    summary_df: pd.DataFrame,
    title: str = 'Model Performance Summary',
    figsize: tuple = (14, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Visualize summary table as a formatted plot.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics dataframe (from create_summary_table).
    title : str
        Plot title.
    figsize : tuple, default=(14, 8)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_df.values,
                    rowLabels=summary_df.index,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style row labels
    for i in range(1, len(summary_df) + 1):
        table[(i, -1)].set_facecolor('#ecf0f1')
        table[(i, -1)].set_text_props(weight='bold')
    
    # Title removed per user request
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_pairwise_metric_scatter(
    results_df: pd.DataFrame,
    metric_x: str,
    metric_y: str,
    model_column: str = 'model_name',
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create scatter plot of two metrics colored by model.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    metric_x : str
        Metric for x-axis.
    metric_y : str
        Metric for y-axis.
    model_column : str, default='model_name'
        Column containing model names.
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
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_models = results_df[model_column].unique()
    
    for model_name in unique_models:
        model_data = results_df[results_df[model_column] == model_name]
        color = get_model_color(model_name)
        
        ax.scatter(model_data[metric_x], model_data[metric_y],
                  color=color, label=model_name, alpha=0.6, 
                  s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(format_metric_name(metric_x), fontsize=16)
    ax.set_ylabel(format_metric_name(metric_y), fontsize=16)
    
    add_grid(ax)
    # Title removed per user request
    ax.legend(loc='best', fontsize=14)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_metric_evolution(
    results_df: pd.DataFrame,
    metric: str,
    time_column: str = 'replication',
    model_column: str = 'model_name',
    rolling_window: Optional[int] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot metric evolution over replications/time.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    metric : str
        Metric to plot.
    time_column : str, default='replication'
        Column representing time/order.
    model_column : str, default='model_name'
        Column containing model names.
    rolling_window : int, optional
        Window size for rolling average.
    title : str, optional
        Plot title.
    figsize : tuple, default=(12, 6)
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
        model_data = results_df[results_df[model_column] == model_name].sort_values(time_column)
        color = get_model_color(model_name)
        
        if rolling_window is not None:
            # Plot rolling average
            values = model_data[metric].rolling(window=rolling_window, center=True).mean()
            ax.plot(model_data[time_column], values, 
                   label=f'{model_name} (rolling avg)', 
                   color=color, linewidth=2)
        else:
            # Plot raw values
            ax.plot(model_data[time_column], model_data[metric],
                   label=model_name, color=color, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel(format_parameter_name(time_column), fontsize=16)
    ax.set_ylabel(format_metric_name(metric), fontsize=16)
    
    add_grid(ax)
    # Title removed per user request
    ax.legend(loc='best', fontsize=14)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_aggregated_results_overview(
    results_df: pd.DataFrame,
    key_metrics: List[str] = ['balanced_accuracy', 'composite_score', 'breakpoint_f1'],
    model_column: str = 'model_name',
    figsize: tuple = (16, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create comprehensive overview dashboard of results.
    
    Combines multiple visualizations into a single figure.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    key_metrics : list of str
        Key metrics to highlight.
    model_column : str, default='model_name'
        Column containing model names.
    figsize : tuple, default=(16, 10)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Bar chart comparison for first metric
    ax1 = fig.add_subplot(gs[0, 0])
    stats = results_df.groupby(model_column)[key_metrics[0]].agg(['mean', 'sem']).reset_index()
    stats = stats.sort_values('mean', ascending=False)
    colors = [get_model_color(m) for m in stats[model_column]]
    x_pos = np.arange(len(stats))
    ax1.bar(x_pos, stats['mean'], yerr=stats['sem'], color=colors, alpha=0.8, capsize=3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stats[model_column], rotation=45, ha='right', fontsize=14)
    ax1.set_ylabel(format_metric_name(key_metrics[0]), fontsize=16)
    # Subplot titles removed per user request
    add_grid(ax1, alpha=0.3)
    
    # 2. Violin plot for second metric
    ax2 = fig.add_subplot(gs[0, 1])
    unique_models = results_df[model_column].unique()
    palette = [get_model_color(m) for m in unique_models]
    sns.violinplot(data=results_df, x=model_column, y=key_metrics[1], hue=model_column, 
                   ax=ax2, palette=palette, legend=False)
    ax2.set_xlabel('Model', fontsize=16)
    ax2.set_ylabel(format_metric_name(key_metrics[1]), fontsize=16)
    # Subplot titles removed per user request
    ax2.tick_params(axis='x', rotation=45, labelsize=14)
    add_grid(ax2, alpha=0.3)
    
    # 3. Box plot for third metric
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=results_df, x=model_column, y=key_metrics[2], hue=model_column,
                ax=ax3, palette=palette, legend=False)
    ax3.set_xlabel('Model', fontsize=16)
    ax3.set_ylabel(format_metric_name(key_metrics[2]), fontsize=16)
    # Subplot titles removed per user request
    ax3.tick_params(axis='x', rotation=45, labelsize=14)
    add_grid(ax3, alpha=0.3)
    
    # 4. Scatter plot: metric 1 vs metric 2
    ax4 = fig.add_subplot(gs[1, 0])
    for model_name in unique_models:
        model_data = results_df[results_df[model_column] == model_name]
        ax4.scatter(model_data[key_metrics[0]], model_data[key_metrics[1]],
                   color=get_model_color(model_name), label=model_name, alpha=0.6, s=30)
    ax4.set_xlabel(format_metric_name(key_metrics[0]), fontsize=16)
    ax4.set_ylabel(format_metric_name(key_metrics[1]), fontsize=16)
    # Subplot titles removed per user request
    add_grid(ax4)
    ax4.legend(loc='best', fontsize=14)
    
    # 5. Scatter plot: metric 1 vs metric 3
    ax5 = fig.add_subplot(gs[1, 1])
    for model_name in unique_models:
        model_data = results_df[results_df[model_column] == model_name]
        ax5.scatter(model_data[key_metrics[0]], model_data[key_metrics[2]],
                   color=get_model_color(model_name), label=model_name, alpha=0.6, s=30)
    ax5.set_xlabel(format_metric_name(key_metrics[0]), fontsize=16)
    ax5.set_ylabel(format_metric_name(key_metrics[2]), fontsize=16)
    # Subplot titles removed per user request
    add_grid(ax5)
    
    # 6. Summary statistics text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
    for metric in key_metrics[:3]:
        summary_text += f"{format_metric_name(metric)}:\n"
        for model_name in unique_models:
            model_data = results_df[results_df[model_column] == model_name]
            mean_val = model_data[metric].mean()
            std_val = model_data[metric].std()
            summary_text += f"  {model_name}: {mean_val:.3f} ± {std_val:.3f}\n"
        summary_text += "\n"
    
    ax6.text(0.1, 0.9, summary_text, fontsize=14, family='monospace',
            verticalalignment='top', transform=ax6.transAxes)
    
    # Main title removed per user request
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_unsupervised_metrics(
    results_df: pd.DataFrame,
    model_column: str = 'model_name',
    figsize: tuple = (15, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """
    Create visualization for unsupervised (label-free) metrics.
    
    Plots BIC, AIC, and Silhouette coefficient distributions across models.
    These metrics can be used for model selection on real data without ground truth.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with unsupervised metrics
    model_column : str, default='model_name'
        Column containing model names
    figsize : tuple, default=(15, 5)
        Figure size
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure or None
        The created figure, or None if no unsupervised metrics are available
        
    Notes
    -----
    - BIC and AIC: Lower is better (model selection criteria)
    - Silhouette: Higher is better (clustering quality, range [-1, 1])
    """
    # Check if unsupervised metrics are available
    unsupervised_metrics = ['bic', 'aic', 'silhouette']
    available_metrics = [m for m in unsupervised_metrics 
                        if m in results_df.columns and results_df[m].notna().any()]
    
    if not available_metrics:
        return None
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    unique_models = results_df[model_column].unique()
    palette = [get_model_color(m) for m in unique_models]
    
    metric_info = {
        'bic': {'name': 'BIC', 'direction': '(lower is better)'},
        'aic': {'name': 'AIC', 'direction': '(lower is better)'},
        'silhouette': {'name': 'Silhouette Coefficient', 'direction': '(higher is better)'}
    }
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Create violin plot
        sns.violinplot(data=results_df, x=model_column, y=metric, hue=model_column,
                      ax=ax, palette=palette, legend=False)
        
        # Add mean markers
        for i, model in enumerate(unique_models):
            model_data = results_df[results_df[model_column] == model]
            mean_val = model_data[metric].mean()
            ax.plot(i, mean_val, 'D', color='white', markersize=6, 
                   markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        
        info = metric_info[metric]
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(f"{info['name']}", fontsize=11)
        ax.set_title(f"{info['name']} {info['direction']}", fontsize=12, weight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        add_grid(ax, alpha=0.3)
        
        # Add horizontal line at 0 for silhouette
        if metric == 'silhouette':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Main title removed per user request
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_supervised_vs_unsupervised_correlation(
    results_df: pd.DataFrame,
    supervised_metric: str = 'balanced_accuracy',
    unsupervised_metrics: Optional[List[str]] = None,
    model_column: str = 'model_name',
    figsize: tuple = (15, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """
    Plot correlation between supervised and unsupervised metrics.
    
    This helps identify which unsupervised metrics best predict supervised performance,
    useful for model selection on real data where ground truth is unavailable.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with both supervised and unsupervised metrics
    supervised_metric : str, default='balanced_accuracy'
        Supervised metric to correlate against (e.g., 'balanced_accuracy', 'composite_score')
    unsupervised_metrics : list of str, optional
        Unsupervised metrics to correlate. If None, uses ['bic', 'aic', 'silhouette']
    model_column : str, default='model_name'
        Column containing model names
    figsize : tuple, default=(15, 5)
        Figure size
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure or None
        The created figure, or None if metrics are not available
    """
    if unsupervised_metrics is None:
        unsupervised_metrics = ['bic', 'aic', 'silhouette']
    
    # Check availability
    if supervised_metric not in results_df.columns:
        return None
    
    available_unsup = [m for m in unsupervised_metrics 
                      if m in results_df.columns and results_df[m].notna().any()]
    
    if not available_unsup:
        return None
    
    n_metrics = len(available_unsup)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    unique_models = results_df[model_column].unique()
    
    for idx, unsup_metric in enumerate(available_unsup):
        ax = axes[idx]
        
        # Scatter plot for each model
        for model in unique_models:
            model_data = results_df[results_df[model_column] == model]
            # Filter out NaN values
            valid_data = model_data[[supervised_metric, unsup_metric]].dropna()
            
            if len(valid_data) > 0:
                ax.scatter(valid_data[unsup_metric], valid_data[supervised_metric],
                         label=model, alpha=0.6, s=50, color=get_model_color(model))
        
        # Add correlation coefficient
        valid_data = results_df[[supervised_metric, unsup_metric]].dropna()
        if len(valid_data) > 1:
            corr = valid_data[supervised_metric].corr(valid_data[unsup_metric])
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(format_metric_name(unsup_metric), fontsize=16)
        ax.set_ylabel(format_metric_name(supervised_metric), fontsize=16)
        # Subplot titles removed per user request
        ax.legend(fontsize=14, loc='best')
        add_grid(ax, alpha=0.3)
    
    # Main title removed per user request
    plt.tight_layout()
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig
