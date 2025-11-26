"""Common plotting utilities for visualization package.

This module provides shared plotting configurations, color schemes, and utility
functions used across all visualization modules.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Union


# Default color schemes
STATE_COLORS = {
    0: '#e74c3c',  # Red
    1: '#3498db',  # Blue
    2: '#2ecc71',  # Green
    3: '#f39c12',  # Orange
    4: '#9b59b6',  # Purple
    5: '#1abc9c',  # Teal
    6: '#e67e22',  # Dark Orange
    7: '#34495e',  # Dark Gray
}

MODEL_COLORS = {
    'SparseJumpPoisson': '#2ecc71',      # Green
    'SparseJumpGaussian': '#3498db',     # Blue
    'JumpPoisson': '#e74c3c',            # Red
    'JumpGaussian': '#f39c12',           # Orange
    'True': '#000000',                   # Black for ground truth
}

METRIC_COLORS = {
    'balanced_accuracy': '#3498db',
    'composite_score': '#2ecc71',
    'breakpoint_f1': '#e74c3c',
    'chamfer_distance': '#9b59b6',
    'breakpoint_error': '#f39c12',
}

# Seaborn style settings
DEFAULT_STYLE = 'whitegrid'
DEFAULT_PALETTE = 'Set2'
DEFAULT_CONTEXT = 'notebook'


def setup_plotting_style(
    style: str = DEFAULT_STYLE,
    palette: str = DEFAULT_PALETTE,
    context: str = DEFAULT_CONTEXT,
    font_scale: float = 1.0
) -> None:
    """Configure matplotlib and seaborn plotting styles.
    
    Parameters
    ----------
    style : str, default='whitegrid'
        Seaborn style name.
    palette : str, default='Set2'
        Seaborn color palette.
    context : str, default='notebook'
        Seaborn context (paper, notebook, talk, poster).
    font_scale : float, default=1.0
        Font scale multiplier.
    """
    sns.set_style(style)
    sns.set_palette(palette)
    sns.set_context(context, font_scale=font_scale)
    
    # Additional matplotlib settings
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#cccccc',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })


def get_state_colors(n_states: int) -> Dict[int, str]:
    """Get color mapping for states.
    
    Parameters
    ----------
    n_states : int
        Number of states to generate colors for.
        
    Returns
    -------
    Dict[int, str]
        Mapping from state index to hex color code.
    """
    if n_states <= len(STATE_COLORS):
        return {i: STATE_COLORS[i] for i in range(n_states)}
    
    # Generate additional colors using colormap if needed
    cmap = plt.colormaps.get_cmap('Set2')
    return {i: cmap(i / max(n_states - 1, 1)) for i in range(n_states)}


def get_model_color(model_name: str) -> str:
    """Get color for a specific model.
    
    Parameters
    ----------
    model_name : str
        Name of the model.
        
    Returns
    -------
    str
        Hex color code or matplotlib color.
    """
    return MODEL_COLORS.get(model_name, '#95a5a6')  # Default gray


def get_metric_color(metric_name: str) -> str:
    """Get color for a specific metric.
    
    Parameters
    ----------
    metric_name : str
        Name of the metric.
        
    Returns
    -------
    str
        Hex color code.
    """
    return METRIC_COLORS.get(metric_name, '#95a5a6')  # Default gray


def save_figure(
    fig: plt.Figure,
    filepath: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight',
    **kwargs
) -> None:
    """Save figure to file with standard settings.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    filepath : str or Path
        Path where to save the figure.
    dpi : int, default=300
        Resolution in dots per inch.
    bbox_inches : str, default='tight'
        Bounding box setting.
    **kwargs
        Additional arguments passed to fig.savefig().
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Figure saved to: {filepath}")


def format_parameter_name(param_name: str) -> str:
    """Format parameter name for display in plots.
    
    Parameters
    ----------
    param_name : str
        Parameter name (e.g., 'delta', 'lambda_0').
        
    Returns
    -------
    str
        Formatted name with LaTeX if appropriate.
    """
    latex_mapping = {
        'delta': r'$\delta$',
        'lambda_0': r'$\lambda_0$',
        'persistence': r'Persistence',
        'n_informative': r'$n_{informative}$',
        'n_total_features': r'$n_{features}$',
        'n_samples': r'$n_{samples}$',
        'n_states': r'$n_{states}$',
        'gamma': r'$\gamma$',
        'kappa': r'$\kappa$',
        'P': r'$P$',
    }
    return latex_mapping.get(param_name, param_name.replace('_', ' ').title())


def format_metric_name(metric_name: str) -> str:
    """Format metric name for display in plots.
    
    Parameters
    ----------
    metric_name : str
        Metric name (e.g., 'balanced_accuracy').
        
    Returns
    -------
    str
        Formatted name.
    """
    name_mapping = {
        'balanced_accuracy': 'Balanced Accuracy',
        'composite_score': 'Composite Score',
        'breakpoint_f1': 'Breakpoint F1',
        'chamfer_distance': 'Chamfer Distance',
        'feature_selection_tpr': 'Feature Selection TPR',
        'feature_selection_fpr': 'Feature Selection FPR',
        'selection_stability': 'Selection Stability',
        'n_jumps_true': 'True Jumps',
        'n_jumps_est': 'Estimated Jumps',
        'breakpoint_error': 'Breakpoint Error',
    }
    return name_mapping.get(metric_name, metric_name.replace('_', ' ').title())


def add_grid(ax: plt.Axes, alpha: float = 0.3, linestyle: str = '--') -> None:
    """Add grid to axes with standard settings.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to add grid to.
    alpha : float, default=0.3
        Grid transparency.
    linestyle : str, default='--'
        Grid line style.
    """
    ax.grid(True, alpha=alpha, linestyle=linestyle)


def add_value_labels(
    ax: plt.Axes,
    bars,
    format_str: str = '{:.3f}',
    fontsize: int = 9,
    offset: float = 0.01
) -> None:
    """Add value labels on top of bars.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes containing the bars.
    bars
        Bar container from ax.bar() or ax.barh().
    format_str : str, default='{:.3f}'
        Format string for values.
    fontsize : int, default=9
        Font size for labels.
    offset : float, default=0.01
        Offset from bar top (as fraction of y-range).
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + offset * y_range,
                format_str.format(height),
                ha='center',
                va='bottom',
                fontsize=fontsize
            )


def create_legend_outside(
    ax: plt.Axes,
    loc: str = 'center left',
    bbox_to_anchor: tuple = (1, 0.5),
    **kwargs
) -> None:
    """Create legend outside the plot area.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to add legend to.
    loc : str, default='center left'
        Legend location.
    bbox_to_anchor : tuple, default=(1, 0.5)
        Bounding box anchor.
    **kwargs
        Additional arguments passed to ax.legend().
    """
    ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, **kwargs)


def set_axis_limits(
    ax: plt.Axes,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    margin: float = 0.05
) -> None:
    """Set axis limits with optional margin.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to modify.
    xlim : tuple, optional
        X-axis limits (min, max).
    ylim : tuple, optional
        Y-axis limits (min, max).
    margin : float, default=0.05
        Margin to add (as fraction of range).
    """
    if xlim is not None:
        x_range = xlim[1] - xlim[0]
        ax.set_xlim(xlim[0] - margin * x_range, xlim[1] + margin * x_range)
    
    if ylim is not None:
        y_range = ylim[1] - ylim[0]
        ax.set_ylim(ylim[0] - margin * y_range, ylim[1] + margin * y_range)


# Initialize default plotting style on import
setup_plotting_style()
