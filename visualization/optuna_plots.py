"""
Optuna Trial Visualization

Functions to visualize Optuna hyperparameter optimization results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union


def load_optuna_trials(results_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load Optuna trial data from results directory.
    
    Parameters
    ----------
    results_path : str or Path
        Path to results directory
        
    Returns
    -------
    pd.DataFrame or None
        Optuna trials data if available, None otherwise
    """
    results_path = Path(results_path)
    optuna_trials_file = results_path / "grid_search" / "optuna_trials.csv"
    
    if optuna_trials_file.exists():
        return pd.read_csv(optuna_trials_file)
    return None


def plot_optuna_trials(
    results_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    metric: str = 'value',
    show: bool = True
):
    """
    Visualize Optuna trial history and hyperparameter exploration.
    
    Creates 4 plots:
    1. Trial history (metric over time)
    2. Hyperparameter distributions
    3. Hyperparameter vs metric scatter plots
    4. Best trial highlight
    
    Parameters
    ----------
    results_df : pd.DataFrame, optional
        Optuna trials DataFrame. If None, will try to load from output_dir.
    output_dir : str, optional
        Directory to save plots (and load trials from if results_df is None)
    metric : str, default='value'
        Metric column name (use 'value' for Optuna trial data)
    show : bool, default=True
        Whether to display plots
    """
    # Try to load Optuna trials from file if not provided
    if results_df is None and output_dir is not None:
        results_df = load_optuna_trials(output_dir)
        if results_df is None:
            print(f"No Optuna trial data found in {output_dir}")
            return
    
    if results_df is None:
        print("No results data provided")
        return
    
    # Filter to successful trials only
    optuna_df = results_df[results_df['state'] == 'COMPLETE'].copy() if 'state' in results_df.columns else results_df.copy()
    
    if len(optuna_df) == 0:
        print("No completed trials found in DataFrame")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Get unique models
    models = optuna_df['model_name'].unique()
    
    for idx, model_name in enumerate(models):
        model_df = optuna_df[optuna_df['model_name'] == model_name].copy()
        
        # Sort by trial number (if available) or index
        if 'trial_number' in model_df.columns:
            model_df = model_df.sort_values('trial_number')
        
        # Plot 1: Trial History
        ax1 = fig.add_subplot(gs[0, idx % 2])
        
        # Plot all trials
        ax1.scatter(range(len(model_df)), model_df[metric], 
                   alpha=0.5, s=50, label='All trials')
        
        # Plot running best
        running_best = model_df[metric].cummax()
        ax1.plot(range(len(model_df)), running_best, 
                'r-', linewidth=2, label='Best so far')
        
        # Highlight best trial
        best_idx = model_df[metric].idxmax()
        best_trial_num = model_df.loc[best_idx].name if 'trial_number' not in model_df.columns else model_df.loc[best_idx, 'trial_number']
        ax1.scatter([model_df.index.get_loc(best_idx)], 
                   [model_df.loc[best_idx, metric]],
                   color='gold', s=200, marker='*', 
                   edgecolors='black', linewidths=2,
                   label='Best trial', zorder=5)
        
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f'{model_name}: Trial History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hyperparameter Exploration
        ax2 = fig.add_subplot(gs[1, idx % 2])
        
        # Get hyperparameter columns
        hyperparam_cols = ['n_components', 'jump_penalty']
        if 'kappa' in model_df.columns:
            hyperparam_cols.append('kappa')
        
        # Create scatter plot of hyperparameters
        if 'n_components' in model_df.columns and 'jump_penalty' in model_df.columns:
            scatter = ax2.scatter(
                model_df['n_components'],
                model_df['jump_penalty'],
                c=model_df[metric],
                s=100,
                cmap='viridis',
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5
            )
            
            # Highlight best
            ax2.scatter(
                [model_df.loc[best_idx, 'n_components']],
                [model_df.loc[best_idx, 'jump_penalty']],
                color='red', s=300, marker='*',
                edgecolors='black', linewidths=2,
                label='Best', zorder=5
            )
            
            ax2.set_xlabel('n_components (n_states)')
            ax2.set_ylabel('jump_penalty (log scale)')
            ax2.set_yscale('log')
            ax2.set_title(f'{model_name}: Hyperparameter Exploration')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label(metric.replace('_', ' ').title())
        
        # Plot 3: Individual hyperparameter distributions
        ax3 = fig.add_subplot(gs[2, idx % 2])
        
        # Box plots for each hyperparameter
        hyperparam_data = []
        hyperparam_names = []
        
        for col in hyperparam_cols:
            if col in model_df.columns:
                if col == 'jump_penalty':
                    # Use log scale for jump_penalty
                    hyperparam_data.append(np.log10(model_df[col]))
                    hyperparam_names.append(f'{col}\n(log10)')
                else:
                    hyperparam_data.append(model_df[col])
                    hyperparam_names.append(col)
        
        if hyperparam_data:
            bp = ax3.boxplot(hyperparam_data, labels=hyperparam_names,
                           patch_artist=True)
            
            # Color boxes by correlation with metric
            for i, (patch, col) in enumerate(zip(bp['boxes'], hyperparam_cols[:len(hyperparam_data)])):
                if col in model_df.columns:
                    corr = model_df[col].corr(model_df[metric])
                    color = plt.cm.RdYlGn((corr + 1) / 2)  # Map -1,1 to 0,1
                    patch.set_facecolor(color)
            
            ax3.set_title(f'{model_name}: Hyperparameter Distributions')
            ax3.set_ylabel('Value')
            ax3.grid(True, alpha=0.3, axis='y')
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / 'optuna_trials.png', dpi=300, bbox_inches='tight')
        print(f"Saved Optuna trial plots to {output_path / 'optuna_trials.png'}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_optuna_summary(
    results_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    metric: str = 'value'
):
    """
    Print summary statistics for Optuna trials.
    
    Parameters
    ----------
    results_df : pd.DataFrame, optional
        Optuna trials DataFrame. If None, will try to load from output_dir.
    output_dir : str, optional
        Directory to load trials from if results_df is None
    metric : str, default='value'
        Metric to summarize (use 'value' for Optuna trial data)
    """
    # Try to load Optuna trials from file if not provided
    if results_df is None and output_dir is not None:
        results_df = load_optuna_trials(output_dir)
        if results_df is None:
            print(f"No Optuna trial data found in {output_dir}")
            return
    
    if results_df is None:
        print("No results data provided")
        return
    
    # Filter to successful trials only
    optuna_df = results_df[results_df['state'] == 'COMPLETE'].copy() if 'state' in results_df.columns else results_df.copy()
    
    if len(optuna_df) == 0:
        print("No completed trials found")
        return
    
    print("\n" + "=" * 80)
    print("OPTUNA TRIAL SUMMARY")
    print("=" * 80)
    
    for model_name in optuna_df['model_name'].unique():
        model_df = optuna_df[optuna_df['model_name'] == model_name]
        
        print(f"\n{model_name}:")
        print(f"  Total trials: {len(model_df)}")
        print(f"  {metric}:")
        print(f"    Best:   {model_df[metric].max():.4f}")
        print(f"    Worst:  {model_df[metric].min():.4f}")
        print(f"    Mean:   {model_df[metric].mean():.4f}")
        print(f"    Std:    {model_df[metric].std():.4f}")
        
        # Best hyperparameters
        best_idx = model_df[metric].idxmax()
        best_trial = model_df.loc[best_idx]
        
        print(f"  Best hyperparameters:")
        if 'n_components' in best_trial:
            print(f"    n_components:  {best_trial['n_components']:.0f}")
        if 'jump_penalty' in best_trial:
            print(f"    jump_penalty:  {best_trial['jump_penalty']:.4f}")
        if 'kappa' in best_trial:
            print(f"    kappa:         {best_trial['kappa']:.4f}")
        
        # Check for problematic trials (very low scores)
        bad_trials = model_df[model_df[metric] < 0.2]
        if len(bad_trials) > 0:
            print(f"\n  ⚠️  WARNING: {len(bad_trials)} trials with {metric} < 0.2")
            print(f"      This suggests extreme hyperparameters causing model failure")
            print(f"      Example bad hyperparameters:")
            worst_idx = model_df[metric].idxmin()
            worst_trial = model_df.loc[worst_idx]
            if 'n_components' in worst_trial:
                print(f"        n_components:  {worst_trial['n_components']:.0f}")
            if 'jump_penalty' in worst_trial:
                print(f"        jump_penalty:  {worst_trial['jump_penalty']:.4f}")
            if 'kappa' in worst_trial:
                print(f"        kappa:         {worst_trial['kappa']:.4f}")
    
    print("\n" + "=" * 80)


def analyze_failed_trials(results_df: pd.DataFrame, metric: str = 'balanced_accuracy', threshold: float = 0.2):
    """
    Analyze trials that performed poorly to identify problematic hyperparameter ranges.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from Optuna optimization
    metric : str, default='balanced_accuracy'
        Metric to analyze
    threshold : float, default=0.2
        Threshold below which trials are considered "failed"
    """
    # Use all results (assumes this is from an Optuna run)
    optuna_df = results_df.copy()
    
    if len(optuna_df) == 0:
        print("No results found")
        return
    
    print("\n" + "=" * 80)
    print("FAILED TRIAL ANALYSIS")
    print("=" * 80)
    print(f"Threshold: {metric} < {threshold}\n")
    
    for model_name in optuna_df['model_name'].unique():
        model_df = optuna_df[optuna_df['model_name'] == model_name]
        failed = model_df[model_df[metric] < threshold]
        
        if len(failed) == 0:
            print(f"{model_name}: No failed trials ✓")
            continue
        
        print(f"\n{model_name}:")
        print(f"  Failed trials: {len(failed)}/{len(model_df)} ({100*len(failed)/len(model_df):.1f}%)")
        
        # Analyze hyperparameter ranges
        if 'n_components' in failed.columns:
            print(f"  n_components in failed trials:")
            print(f"    Range: [{failed['n_components'].min():.0f}, {failed['n_components'].max():.0f}]")
            print(f"    Mean:  {failed['n_components'].mean():.1f}")
            
            # Compare to successful trials
            success = model_df[model_df[metric] >= threshold]
            if len(success) > 0:
                print(f"  n_components in successful trials:")
                print(f"    Range: [{success['n_components'].min():.0f}, {success['n_components'].max():.0f}]")
                print(f"    Mean:  {success['n_components'].mean():.1f}")
        
        if 'jump_penalty' in failed.columns:
            print(f"  jump_penalty in failed trials:")
            print(f"    Range: [{failed['jump_penalty'].min():.4f}, {failed['jump_penalty'].max():.4f}]")
            print(f"    Mean:  {failed['jump_penalty'].mean():.4f}")
            
            success = model_df[model_df[metric] >= threshold]
            if len(success) > 0:
                print(f"  jump_penalty in successful trials:")
                print(f"    Range: [{success['jump_penalty'].min():.4f}, {success['jump_penalty'].max():.4f}]")
                print(f"    Mean:  {success['jump_penalty'].mean():.4f}")
        
        # Suggest constraints
        success = model_df[model_df[metric] >= threshold]
        if len(success) > 0:
            print(f"\n  💡 Suggested hyperparameter constraints:")
            if 'n_components' in success.columns:
                n_comp_min = max(2, int(success['n_components'].quantile(0.1)))
                n_comp_max = int(success['n_components'].quantile(0.9))
                print(f"     n_components: [{n_comp_min}, {n_comp_max}]")
            if 'jump_penalty' in success.columns:
                jp_min = success['jump_penalty'].quantile(0.1)
                jp_max = success['jump_penalty'].quantile(0.9)
                print(f"     jump_penalty: [{jp_min:.4f}, {jp_max:.4f}]")
    
    print("\n" + "=" * 80)
