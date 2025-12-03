"""Time series visualization functions.

This module provides functions for visualizing time series data, state sequences,
and breakpoint detection results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union, Any, Tuple
from pathlib import Path
from typing import Iterable

from .utils import (
    get_state_colors,
    get_model_color,
    format_parameter_name,
    save_figure,
    add_grid,
    setup_plotting_style,
)


def plot_time_series_with_breakpoints(
    X: pd.DataFrame,
    model: Any,
    actual_breakpoints: Optional[Union[int, List[int], np.ndarray]] = None,
    title: str = 'Model Results',
    figsize: Optional[tuple] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot time series with predicted and actual breakpoints.
    
    Visualizes the fitted model results showing both predicted regime changes
    and actual breakpoints (if provided) overlaid on the time series data.
    
    Parameters
    ----------
    X : pd.DataFrame
        The input time series data with features as columns.
    model : object
        Fitted model with labels_ attribute (JumpModel or SparseJumpModel).
    actual_breakpoints : int, list, or np.ndarray, optional
        True breakpoint indices for comparison. Can be a single value or list.
    title : str, default='Model Results'
        Plot title.
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    save_path : str or Path, optional
        Path to save figure. If None, figure is not saved.
        
    Returns
    -------
    plt.Figure
        The created figure.
        
    Examples
    --------
    >>> fig = plot_time_series_with_breakpoints(
    ...     X, model, actual_breakpoints=[100, 200],
    ...     title='Jump Model Results'
    ... )
    """
    # Normalize actual_breakpoints to list
    if actual_breakpoints is not None:
        if isinstance(actual_breakpoints, (int, float)):
            actual_breakpoints = [int(actual_breakpoints)]
        else:
            actual_breakpoints = list(actual_breakpoints)
    
    labels = model.labels_.to_numpy() if hasattr(model.labels_, 'to_numpy') else np.array(model.labels_)
    
    # Find predicted change points
    change_point_indices = np.where(labels[:-1] != labels[1:])[0] + 1
    
    # Convert to index values
    if isinstance(X.index, pd.DatetimeIndex):
        change_point_values = X.index[change_point_indices]
        if actual_breakpoints is not None:
            valid_indices = [bp for bp in actual_breakpoints if bp < len(X.index)]
            breakpoint_vals = X.index[valid_indices]
        else:
            breakpoint_vals = []
    else:
        change_point_values = change_point_indices
        breakpoint_vals = actual_breakpoints if actual_breakpoints is not None else []
    
    features = X.columns
    n_features = len(features)
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (15, 4 * n_features)
    
    # Create subplots
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    if n_features == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=18)
    
    base_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, feature in enumerate(features):
        ax = axes[i]
        color = base_colors[i % len(base_colors)]
        
        # Plot time series
        ax.plot(X.index, X[feature], label=f'{feature} Data', color=color, zorder=2, linewidth=1.5)
        
        # Plot actual breakpoints
        for j, bp in enumerate(breakpoint_vals):
            label_actual = 'True Breakpoint' if i == 0 and j == 0 else ""
            ax.axvline(x=bp, color='#2ecc71', linestyle='-', linewidth=2.5, 
                      label=label_actual, zorder=3, alpha=0.7)
        
        # Plot predicted change points
        for j, cp in enumerate(change_point_values):
            label_pred = 'Predicted Breakpoint' if i == 0 and j == 0 else ""
            ax.axvline(x=cp, color='#e74c3c', linestyle='--', linewidth=2, 
                      label=label_pred, zorder=3)
        
        ax.set_ylabel('Value', fontsize=12)
        
        # Add feature weights if available
        title_text = f'{feature.capitalize()}'
        if hasattr(model, 'feat_weights') and model.feat_weights is not None:
            weight = (model.feat_weights.iloc[i] if isinstance(model.feat_weights, pd.Series) 
                     else model.feat_weights[i])
            title_text += f" (Weight: {weight:.4f})"
        
        ax.set_title(title_text, fontsize=14)
        add_grid(ax)
        ax.legend(loc='best', fontsize=10)
    
    axes[-1].set_xlabel('Time Step', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_simulated_from_regimes(
    X: pd.DataFrame,
    model: Any,
    title: str = 'Simulated Data from Predicted Regimes',
    figsize: Optional[tuple] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Generate and plot simulated data from predicted regimes.
    
    Creates a new time series by sampling from the distributions of the
    predicted states from a fitted model.
    
    Parameters
    ----------
    X : pd.DataFrame
        Original data (used for index, columns, and shape).
    model : object
        Fitted model with labels_, centers_, and distribution attributes.
    title : str, default='Simulated Data from Predicted Regimes'
        Plot title.
    figsize : tuple, optional
        Figure size. If None, auto-calculated.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    # Extract model parameters
    labels = model.labels_.to_numpy() if hasattr(model.labels_, 'to_numpy') else np.array(model.labels_)
    centers = model.centers_
    distribution = model.distribution
    n_samples, n_features = X.shape
    
    # For Gaussian, estimate standard deviation
    stds = None
    if distribution == "Gaussian":
        stds = np.array([X[labels == i].std(axis=0).fillna(1.0) for i in range(model.n_components)])
    
    # Simulate data
    simulated_data = np.zeros_like(X)
    for t in range(n_samples):
        state = labels[t]
        params = centers[state]
        
        if distribution == "Poisson":
            simulated_data[t, :] = np.random.poisson(lam=params)
        elif distribution == "Gaussian":
            state_stds = stds[state]
            simulated_data[t, :] = np.random.normal(loc=params, scale=state_stds)
        else:
            raise NotImplementedError(f"Simulation not implemented for '{distribution}' distribution.")
    
    simulated_df = pd.DataFrame(simulated_data, index=X.index, columns=X.columns)
    
    # Find change points
    change_point_indices = np.where(labels[:-1] != labels[1:])[0] + 1
    if isinstance(X.index, pd.DatetimeIndex):
        change_point_values = X.index[change_point_indices]
    else:
        change_point_values = change_point_indices
    
    # Create plot
    if figsize is None:
        figsize = (15, 4 * n_features)
    
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    if n_features == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=18)
    
    base_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, feature in enumerate(simulated_df.columns):
        ax = axes[i]
        color = base_colors[i % len(base_colors)]
        
        # Plot simulated data
        ax.plot(simulated_df.index, simulated_df[feature], 
               label=f'Simulated {feature}', color=color, zorder=2, linewidth=1.5)
        
        # Plot predicted change points
        for j, cp in enumerate(change_point_values):
            label_pred = 'Predicted Breakpoint' if i == 0 and j == 0 else ""
            ax.axvline(x=cp, color='#e74c3c', linestyle='--', linewidth=2, 
                      label=label_pred, zorder=3)
        
        ax.set_ylabel('Simulated Value', fontsize=12)
        ax.set_title(f'Simulated {feature.capitalize()} from Predicted Regimes', fontsize=14)
        add_grid(ax)
        ax.legend(loc='best', fontsize=10)
    
    axes[-1].set_xlabel('Time Step', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_stacked_states(
    X: pd.DataFrame,
    models_dict: Dict[str, Any],
    true_states: Optional[Union[np.ndarray, pd.Series]] = None,
    feature_to_plot: Optional[Union[str, List[str]]] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create stacked plot showing time series and state assignments.
    
    Similar to visualizations in academic papers, this shows:
    - Top panels: Time series data (one or more features)
    - Subsequent panels: State assignments from different models as horizontal bars
    
    Model labels are automatically permuted to best align with true states if provided.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input time series data.
    models_dict : dict
        Dictionary of {model_name: model} where each model has labels_ attribute.
        Example: {'Jump Model': jm, 'Sparse Jump': sjm}
    true_states : np.ndarray or pd.Series, optional
        True state sequence if available.
    feature_to_plot : str or list, optional
        Feature name(s) to plot. If None, plots all features.
    figsize : tuple, default=(12, 10)
        Figure size (width, height).
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
        
    Examples
    --------
    >>> models = {
    ...     'Sparse Jump Poisson': sjm_poisson,
    ...     'Sparse Jump Gaussian': sjm_gaussian,
    ... }
    >>> fig = plot_stacked_states(X, models, true_states=states, 
    ...                           feature_to_plot='feature_0')
    """
    # Import the BAC permutation function from metrics module
    import sys
    from pathlib import Path
    
    # Add parent directory to path to allow absolute import
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from simulation.metrics import compute_bac_best_permutation
    
    # Determine features to plot
    if feature_to_plot is None:
        features = X.columns.tolist()
    elif isinstance(feature_to_plot, str):
        features = [feature_to_plot]
    else:
        features = list(feature_to_plot)
    
    # Calculate number of panels
    n_data_panels = len(features)
    n_model_panels = len(models_dict)
    n_panels = n_data_panels + n_model_panels
    if true_states is not None:
        n_panels += 1
    
    # Create figure
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]
    
    fig.suptitle('Time Series and State Assignments', fontsize=16, y=0.995)
    
    data_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    panel_idx = 0
    
    # Plot time series data
    for i, feature in enumerate(features):
        ax = axes[panel_idx]
        ax.plot(X.index, X[feature], color=data_colors[i % len(data_colors)], 
               linewidth=1.5, alpha=0.8)
        ax.set_ylabel(feature, fontsize=10)
        add_grid(ax, alpha=0.3)
        ax.tick_params(axis='y', labelsize=9)
        panel_idx += 1
    
    def plot_state_bars(ax: plt.Axes, labels: Union[np.ndarray, pd.Series], 
                       label_text: str, color_map: Optional[Dict] = None):
        """Helper function to plot horizontal state bars."""
        labels_array = labels.to_numpy() if hasattr(labels, 'to_numpy') else np.array(labels)
        n_states = len(np.unique(labels_array))
        
        # Get color map
        if color_map is None:
            color_map = get_state_colors(n_states)
        
        # Plot horizontal bars for each state segment
        current_state = labels_array[0]
        start_idx = 0
        
        for t in range(1, len(labels_array) + 1):
            if t == len(labels_array) or labels_array[t] != current_state:
                # Plot bar for previous segment
                y_pos = current_state + 1  # 1-indexed for display
                ax.barh(y=y_pos, width=t - start_idx, left=start_idx, 
                       height=0.8, color=color_map[current_state], 
                       edgecolor='white', linewidth=0.5, align='center')
                
                if t < len(labels_array):
                    current_state = labels_array[t]
                    start_idx = t
        
        ax.set_ylabel(label_text, fontsize=10, rotation=0, ha='right', va='center')
        ax.set_ylim(0.5, n_states + 0.5)
        ax.set_yticks(range(1, n_states + 1))
        ax.set_yticklabels([f'S{i}' for i in range(1, n_states + 1)], fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.tick_params(left=True, labelsize=8)
    
    # Get true states as numpy array
    true_states_array = None
    if true_states is not None:
        true_states_array = true_states.to_numpy() if hasattr(true_states, 'to_numpy') else np.array(true_states)
    
    # Plot true states if provided
    if true_states_array is not None:
        ax = axes[panel_idx]
        plot_state_bars(ax, true_states_array, 'True\nStates')
        panel_idx += 1
    
    # Plot model predictions with best permutation
    for model_name, model in models_dict.items():
        ax = axes[panel_idx]
        
        # Get model labels
        model_labels = model.labels_.to_numpy() if hasattr(model.labels_, 'to_numpy') else np.array(model.labels_)
        
        # If true states are available, find best permutation
        if true_states_array is not None:
            _, permuted_labels = compute_bac_best_permutation(
                true_states_array, 
                model_labels,
                return_permuted=True
            )
            plot_state_bars(ax, permuted_labels, model_name)
        else:
            plot_state_bars(ax, model_labels, model_name)
        
        panel_idx += 1
    
    # Set x-axis label
    axes[-1].set_xlabel('Time Step', fontsize=11)
    axes[-1].tick_params(axis='x', labelsize=9)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def plot_multiple_series_comparison(
    data_dict: Dict[str, pd.DataFrame],
    title: str = 'Time Series Comparison',
    figsize: tuple = (14, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot multiple time series for comparison.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of {label: dataframe} to plot.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The created figure.
    """
    n_series = len(data_dict)
    
    fig, axes = plt.subplots(n_series, 1, figsize=figsize, sharex=True)
    if n_series == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, (label, df) in enumerate(data_dict.items()):
        ax = axes[idx]
        
        # Plot all columns
        for col_idx, col in enumerate(df.columns):
            ax.plot(df.index, df[col], 
                   color=colors[col_idx % len(colors)],
                   label=col, linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel(label, fontsize=11)
        add_grid(ax)
        if len(df.columns) <= 5:  # Only show legend if not too many columns
            ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Time Step', fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    if save_path is not None:
        save_figure(fig, save_path)
    
    return fig


def filter_models(
    models_dict: Dict[str, Any],
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    order: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    rename: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Filter and order a models dictionary for plotting.

    Parameters
    ----------
    models_dict : dict
        Mapping of display name -> fitted model (must have `labels_`).
    include : list of str, optional
        Keep entries whose names contain any of these substrings. If None, keep all.
    exclude : list of str, optional
        Drop entries whose names contain any of these substrings.
    order : list of str, optional
        Desired name order (by exact match or substring priority). Others follow.
    limit : int, optional
        Keep only the first N entries after filtering and ordering.
    rename : dict, optional
        Mapping old name -> new display name.

    Returns
    -------
    dict
        A new dict with filtered, ordered, and optionally renamed keys.
    """
    items = list(models_dict.items())

    def matches_any(name: str, patterns: Iterable[str]) -> bool:
        return any(p.lower() in name.lower() for p in patterns)

    if include:
        items = [(k, v) for k, v in items if matches_any(k, include)]
    if exclude:
        items = [(k, v) for k, v in items if not matches_any(k, exclude)]

    if order:
        priority = []
        rest = []
        used = set()
        for pat in order:
            for i, (k, v) in enumerate(items):
                if i in used:
                    continue
                if (pat == k) or (pat.lower() in k.lower()):
                    priority.append((k, v))
                    used.add(i)
        for i, kv in enumerate(items):
            if i not in used:
                rest.append(kv)
        items = priority + rest

    if limit is not None and limit >= 0:
        items = items[:limit]

    if rename:
        items = [(rename.get(k, k), v) for k, v in items]

    return {k: v for k, v in items}


def plot_stacked_states_from_results(
    X: pd.DataFrame,
    results: Dict[str, Any],
    feature_to_plot: Optional[Union[str, List[str]]] = None,
    include_models: Optional[Iterable[str]] = None,
    exclude_models: Optional[Iterable[str]] = None,
    order: Optional[Iterable[str]] = None,
    name_format: str = "{model} (K={n_components})",
    figsize: tuple = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Convenience wrapper to plot stacked states from `fit_on_real_data` results.

    Parameters
    ----------
    X : pd.DataFrame
        Input time series data.
    results : dict
        The return dict from `simulation.api.fit_on_real_data`.
    feature_to_plot : str or list, optional
        Feature name(s) to plot on the top panel(s).
    include_models : list of str, optional
        Only include models whose formatted names contain any of these substrings.
    exclude_models : list of str, optional
        Exclude models whose formatted names contain any of these substrings.
    order : list of str, optional
        Desired order for model display names (exact or substring priority).
    name_format : str, default="{model} (K={n_components})"
        Format string for each model panel name.
    figsize : tuple, default=(12, 10)
        Figure size for the plot.
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    best_models = results.get("best_models", {})
    built: Dict[str, Any] = {}
    for mname, payload in best_models.items():
        model_obj = payload.get("model", None) if isinstance(payload, dict) else None
        if model_obj is None:
            continue
        ncomp = payload.get("n_components", getattr(model_obj, "n_components", None))
        display_name = name_format.format(model=mname, n_components=ncomp)
        built[display_name] = model_obj

    filtered = filter_models(
        built,
        include=include_models,
        exclude=exclude_models,
        order=order,
    )

    return plot_stacked_states(
        X=X,
        models_dict=filtered,
        true_states=None,
        feature_to_plot=feature_to_plot,
        figsize=figsize,
        save_path=save_path,
    )
