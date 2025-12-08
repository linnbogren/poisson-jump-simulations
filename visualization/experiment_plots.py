"""Experiment-level visualization creation.

This module provides high-level functions to create comprehensive visualization
suites for completed experiments, loading results and models as needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union

from .comparison import (
    plot_model_comparison_bars,
    plot_parameter_sensitivity,
)
from .results import (
    plot_correlation_matrix,
    plot_aggregated_results_overview,
)
from .time_series import plot_stacked_states
from .utils import setup_plotting_style


class ExperimentVisualizer:
    """Create visualizations for a completed experiment.
    
    This class handles all visualization logic for experiments, loading
    results and models from disk and creating publication-ready plots.
    
    Parameters
    ----------
    results_manager : ResultManager
        Manager for loading experiment results.
    output_dir : Path
        Directory where plots will be saved.
    config : dict, optional
        Visualization configuration options.
        
    Examples
    --------
    >>> from simulation.results import ResultManager
    >>> manager = ResultManager("results/my_experiment")
    >>> visualizer = ExperimentVisualizer(manager, manager.results_dir / "plots")
    >>> visualizer.create_all_plots()
    """
    
    def __init__(self, results_manager, output_dir: Path, config: Optional[Dict] = None):
        """Initialize the visualizer."""
        self.manager = results_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'create_comparison_plots': True,
            'create_heatmaps': True,
            'create_time_series': True,
            'create_correlation_matrix': True,
            'create_overview_plot': True,
            
            # Metrics to visualize
            'comparison_metrics': [
                'balanced_accuracy',
                'feature_f1',
                'chamfer_distance',
                'breakpoint_count_error',
            ],
            
            # Plot settings
            'dpi': 300,
            'figsize_default': (10, 6),
            'figsize_large': (14, 10),
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Setup plotting style
        setup_plotting_style()
        
        # Load results
        self.best_results = self.manager.load_best_results()
        self.grid_results = self.manager.load_grid_results()
    
    def create_all_plots(self):
        """Create all configured plots for the experiment."""
        print("\n" + "="*80)
        print("Creating Visualizations")
        print("="*80)
        
        self.create_comparison_plots()
        self.create_overview_plot()
        self.create_correlation_matrix()
        self.create_unsupervised_metrics_plots()
        self.create_sensitivity_plots()
        self.create_stacked_time_series()
        
        print("\n" + "="*80)
        print(f"All plots saved to: {self.output_dir}")
        print("="*80)
    
    def create_comparison_plots(self):
        """Create model comparison bar charts."""
        if not self.config['create_comparison_plots']:
            return
        
        print("\n" + "-"*80)
        print("Creating model comparison plots...")
        print("-"*80)
        
        for metric in self.config['comparison_metrics']:
            if metric in self.best_results.columns:
                try:
                    fig = plot_model_comparison_bars(
                        self.best_results,
                        metric=metric,
                        title=f'Model Performance: {metric.replace("_", " ").title()}'
                    )
                    filename = f"comparison_{metric}.png"
                    fig.savefig(
                        self.output_dir / filename, 
                        dpi=self.config['dpi'], 
                        bbox_inches='tight'
                    )
                    print(f"  ✓ Saved: {filename}")
                    plt.close(fig)
                except Exception as e:
                    print(f"  ✗ Error creating {metric} comparison: {e}")
    
    def create_overview_plot(self):
        """Create aggregated overview plot."""
        if not self.config['create_overview_plot']:
            return
        
        print("\n" + "-"*80)
        print("Creating overview plot...")
        print("-"*80)
        
        try:
            fig = plot_aggregated_results_overview(
                self.best_results,
                key_metrics=['balanced_accuracy', 'feature_f1', 'chamfer_distance'],
                figsize=self.config['figsize_large']
            )
            fig.savefig(
                self.output_dir / "overview.png", 
                dpi=self.config['dpi'], 
                bbox_inches='tight'
            )
            print("  ✓ Saved: overview.png")
            plt.close(fig)
        except Exception as e:
            print(f"  ✗ Error creating overview: {e}")
    
    def create_correlation_matrix(self):
        """Create metric correlation matrix."""
        if not self.config['create_correlation_matrix']:
            return
        
        print("\n" + "-"*80)
        print("Creating correlation matrix...")
        print("-"*80)
        
        try:
            # Use specific metrics from config if provided
            if 'correlation_metrics' in self.config:
                # Filter to only metrics that exist in the dataframe
                metric_cols = [col for col in self.config['correlation_metrics'] 
                              if col in self.best_results.columns]
            else:
                # Fall back to auto-detection (exclude non-metric columns)
                exclude_cols = [
                    'n_samples', 'n_states', 'n_informative', 'n_noise',
                    'n_total_features', 'delta', 'lambda_0', 'persistence',
                    'distribution_type', 'correlated_noise', 'random_seed',
                    'model_name', 'best_n_components', 'best_jump_penalty',
                    'best_max_feats'
                ]
                metric_cols = [col for col in self.best_results.columns 
                              if col not in exclude_cols]
            
            # Only plot if we have at least 2 metrics
            if len(metric_cols) >= 2:
                fig = plot_correlation_matrix(
                    self.best_results[metric_cols],
                    metrics=metric_cols,
                    title='Metric Correlations',
                    figsize=self.config['figsize_large']
                )
                fig.savefig(
                    self.output_dir / "correlation_matrix.png", 
                    dpi=self.config['dpi'], 
                    bbox_inches='tight'
                )
                print("  ✓ Saved: correlation_matrix.png")
                plt.close(fig)
            else:
                print("  ⚠ Skipped correlation matrix (insufficient metrics available)")
        except Exception as e:
            print(f"  ✗ Error creating correlation matrix: {e}")
    
    def create_unsupervised_metrics_plots(self):
        """Create plots for unsupervised (label-free) metrics."""
        if not self.config.get('create_unsupervised_plots', False):
            return
        
        print("\n" + "-"*80)
        print("Creating unsupervised metrics plots...")
        print("-"*80)
        
        try:
            # Import unsupervised visualization functions
            from .results import (
                plot_unsupervised_metrics,
                plot_supervised_vs_unsupervised_correlation
            )
            
            # 1. Unsupervised metrics distribution plot
            fig = plot_unsupervised_metrics(
                self.best_results,
                model_column='model_name',
                figsize=(15, 5)
            )
            if fig is not None:
                fig.savefig(
                    self.output_dir / "unsupervised_metrics.png",
                    dpi=self.config['dpi'],
                    bbox_inches='tight'
                )
                print("  ✓ Saved: unsupervised_metrics.png")
                plt.close(fig)
            
            # 2. Correlation plots between supervised and unsupervised metrics
            supervised_metrics = ['balanced_accuracy', 'composite_score', 'feature_f1']
            for sup_metric in supervised_metrics:
                if sup_metric in self.best_results.columns:
                    fig = plot_supervised_vs_unsupervised_correlation(
                        self.best_results,
                        supervised_metric=sup_metric,
                        model_column='model_name',
                        figsize=(15, 5)
                    )
                    if fig is not None:
                        filename = f"unsupervised_vs_{sup_metric}.png"
                        fig.savefig(
                            self.output_dir / filename,
                            dpi=self.config['dpi'],
                            bbox_inches='tight'
                        )
                        print(f"  ✓ Saved: {filename}")
                        plt.close(fig)
        
        except Exception as e:
            print(f"  ✗ Error creating unsupervised metrics plots: {e}")
    
    def create_sensitivity_plots(self):
        """Create parameter sensitivity plots."""
        if not self.config['create_comparison_plots']:
            return
        
        # Only create if multiple configurations exist
        if len(self.best_results) <= 1:
            return
        
        print("\n" + "-"*80)
        print("Creating parameter sensitivity plots...")
        print("-"*80)
        
        # Delta sensitivity
        if 'delta' in self.best_results.columns and self.best_results['delta'].nunique() > 1:
            try:
                fig = plot_parameter_sensitivity(
                    self.best_results,
                    parameter='delta',
                    metric='balanced_accuracy',
                    title='Performance vs Jump Rate (delta)'
                )
                fig.savefig(
                    self.output_dir / "sensitivity_delta.png", 
                    dpi=self.config['dpi'], 
                    bbox_inches='tight'
                )
                print("  ✓ Saved: sensitivity_delta.png")
                plt.close(fig)
            except Exception as e:
                print(f"  ✗ Error creating delta sensitivity: {e}")
        
        # Distribution type sensitivity
        if 'distribution_type' in self.best_results.columns and \
           self.best_results['distribution_type'].nunique() > 1:
            try:
                fig = plot_parameter_sensitivity(
                    self.best_results,
                    parameter='distribution_type',
                    metric='balanced_accuracy',
                    title='Performance vs Distribution Type'
                )
                fig.savefig(
                    self.output_dir / "sensitivity_distribution.png", 
                    dpi=self.config['dpi'], 
                    bbox_inches='tight'
                )
                print("  ✓ Saved: sensitivity_distribution.png")
                plt.close(fig)
            except Exception as e:
                print(f"  ✗ Error creating distribution sensitivity: {e}")
    
    def create_stacked_time_series(self):
        """Create stacked time series visualization using saved models."""
        if not self.config['create_time_series']:
            return
        
        print("\n" + "-"*80)
        print("Creating stacked time series visualization...")
        print("-"*80)
        
        try:
            # Get first result to determine which configuration to plot
            first_result = self.best_results.iloc[0]
            config_id = 0  # First configuration
            seed = int(first_result['random_seed'])
            delta = float(first_result['delta'])
            
            # Load data for this configuration
            X, states, breakpoints = self.manager.load_data_for_config(config_id)
            
            # Load models for this seed/delta combination
            models_dict = self.manager.load_models_for_visualization(
                seed=seed,
                delta=delta
            )
            
            if not models_dict:
                print("  ⚠ No models found for visualization.")
                return
            
            # Print loaded models
            for model_name, model in models_dict.items():
                print(f"  ✓ Loaded {model_name}")
                print(f"    - n_components: {model.n_components}")
                print(f"    - jump_penalty: {model.jump_penalty}")
                if hasattr(model, 'max_feats'):
                    print(f"    - max_feats: {model.max_feats}")
            
            # Select first informative feature
            informative_cols = [col for col in X.columns if col.startswith('informative_')]
            if not informative_cols:
                informative_cols = X.columns.tolist()
            
            feature_to_plot = informative_cols[0]
            
            # Create stacked plot
            fig = plot_stacked_states(
                X,
                models_dict,
                true_states=states,
                feature_to_plot=feature_to_plot,
                figsize=(14, 10)
            )
            
            filename = "timeseries_stacked_comparison.png"
            fig.savefig(
                self.output_dir / filename, 
                dpi=self.config['dpi'], 
                bbox_inches='tight'
            )
            print(f"  ✓ Saved: {filename}")
            plt.close(fig)
            
        except Exception as e:
            print(f"  ✗ Error creating stacked time series: {e}")
            import traceback
            traceback.print_exc()


def create_experiment_visualizations(
    results_dir: Union[str, Path],
    config: Optional[Dict] = None
) -> Path:
    """Create all visualizations for an experiment.
    
    Convenience function to create visualizations without manually
    instantiating the ResultManager and ExperimentVisualizer.
    
    Parameters
    ----------
    results_dir : str or Path
        Path to experiment results directory.
    config : dict, optional
        Visualization configuration options.
        
    Returns
    -------
    Path
        Path to the plots directory.
        
    Examples
    --------
    >>> plots_dir = create_experiment_visualizations(
    ...     "results/my_experiment_20251127_084934"
    ... )
    """
    import sys
    from pathlib import Path
    
    # Add parent to path for absolute imports
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from simulation.results import ResultManager
    
    results_dir = Path(results_dir)
    manager = ResultManager(results_dir)
    
    plots_dir = results_dir / "plots"
    visualizer = ExperimentVisualizer(manager, plots_dir, config)
    visualizer.create_all_plots()
    
    return plots_dir
