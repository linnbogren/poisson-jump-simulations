"""
Generate LaTeX-formatted tables from simulation results.

This module creates publication-ready LaTeX tables showing model performance
across different configurations. Tables automatically adapt to varying parameters
in the experiment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from scipy import stats


def format_mean_std(mean: float, std: float, decimals: int = 2, bold: bool = False) -> str:
    """
    Format mean and std as 'mean (std)' with specified decimal places.
    
    Parameters
    ----------
    mean : float
        Mean value
    std : float
        Standard deviation
    decimals : int, default=2
        Number of decimal places
    bold : bool, default=False
        Whether to format in bold (for LaTeX)
        
    Returns
    -------
    str
        Formatted string 'mean (std)'
    """
    fmt = f"{{:.{decimals}f}}"
    mean_str = fmt.format(mean)
    std_str = fmt.format(std)
    result = f"{mean_str} ({std_str})"
    
    if bold:
        result = f"\\textbf{{{result}}}"
    
    return result


def detect_varying_parameters(results_df: pd.DataFrame) -> List[str]:
    """
    Detect which parameters vary across configurations.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
        
    Returns
    -------
    list of str
        List of parameter names that vary
    """
    params_to_check = [
        'delta', 'n_samples', 'n_states', 'n_informative',
        'n_total_features', 'lambda_0', 'persistence',
        'distribution_type', 'correlated_noise'
    ]
    
    varying = []
    for param in params_to_check:
        if param in results_df.columns:
            if results_df[param].nunique() > 1:
                varying.append(param)
    
    return varying


def create_model_comparison_table(
    results_df: pd.DataFrame,
    metrics: Optional[List[tuple]] = None,
    output_file: Optional[Union[str, Path]] = None,
    alpha: float = 0.05
) -> str:
    """
    Create a comprehensive LaTeX table comparing all models across metrics.
    Bold values indicate statistically significant best performance.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from simulation
    metrics : list of tuple, optional
        List of (column_name, display_name, decimals, higher_is_better).
        If None, auto-detects available metrics.
    output_file : str or Path, optional
        If provided, save table to this file
    alpha : float, default=0.05
        Significance level for statistical tests
        
    Returns
    -------
    str
        LaTeX table code
    """
    models = sorted(results_df['model_name'].unique())
    
    # Auto-detect metrics if not provided
    if metrics is None:
        metrics = []
        available_metrics = [
            ('balanced_accuracy', 'BAC', 2, True),
            ('composite_score', 'Composite', 2, True),
            ('feature_f1', 'F1', 2, True),
            ('feature_recall', 'Recall', 2, True),
            ('breakpoint_f1', 'BP-F1', 2, True),
            ('chamfer_distance', 'Chamfer', 2, False),
            ('bic', 'BIC', 1, False),
            ('aic', 'AIC', 1, False),
            ('silhouette', 'Silhouette', 3, True),
        ]
        
        for metric_col, display_name, decimals, higher_is_better in available_metrics:
            if metric_col in results_df.columns and results_df[metric_col].notna().any():
                metrics.append((metric_col, display_name, decimals, higher_is_better))
    
    # Build column specification
    n_cols = len(metrics)
    col_spec = 'l' + 'r' * n_cols
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Model Comparison: Mean (Std) across all configurations}")
    latex.append("  \\label{tab:model_comparison}")
    latex.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex.append("    \\toprule")
    
    # Header row
    header = "    Model"
    for _, display_name, _, _ in metrics:
        header += f" & {display_name}"
    latex.append(header + " \\\\")
    latex.append("    \\midrule")
    
    # Determine significantly best models for each metric
    metric_winners = {}
    for metric_col, _, _, higher_is_better in metrics:
        data_groups = {}
        for model in models:
            model_data = results_df[results_df['model_name'] == model][metric_col].values
            model_data = model_data[~np.isnan(model_data)]
            data_groups[model] = model_data
        
        winners = set()
        for i, model1 in enumerate(models):
            data1 = data_groups[model1]
            if len(data1) == 0:
                continue
            
            is_sig_better = True
            mean1 = np.mean(data1)
            
            for j, model2 in enumerate(models):
                if i == j:
                    continue
                
                data2 = data_groups[model2]
                if len(data2) == 0:
                    continue
                
                mean2 = np.mean(data2)
                
                # Check if model1 is better
                if higher_is_better:
                    if mean1 <= mean2:
                        is_sig_better = False
                        break
                else:
                    if mean1 >= mean2:
                        is_sig_better = False
                        break
                
                # Test significance
                try:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    if higher_is_better:
                        p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2
                    else:
                        p_value_one_tailed = p_value / 2 if t_stat < 0 else 1 - p_value / 2
                    
                    if p_value_one_tailed > alpha:
                        is_sig_better = False
                        break
                except:
                    is_sig_better = False
                    break
            
            if is_sig_better and len(data1) > 0:
                winners.add(model1)
        
        metric_winners[metric_col] = winners
    
    # Build table rows
    for model in models:
        model_data = results_df[results_df['model_name'] == model]
        row = f"    {model}"
        
        for metric_col, _, decimals, _ in metrics:
            values = model_data[metric_col].values
            values = values[~np.isnan(values)]
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                is_bold = model in metric_winners[metric_col]
                row += f" & {format_mean_std(mean_val, std_val, decimals, bold=is_bold)}"
            else:
                row += " & ---"
        
        latex.append(row + " \\\\")
    
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("  \\begin{tablenotes}")
    latex.append("    \\small")
    latex.append(f"    \\item Note: Bold values indicate statistically significant best performance ($p < {alpha}$).")
    latex.append("  \\end{tablenotes}")
    latex.append("\\end{table}")
    
    latex_code = "\n".join(latex)
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(latex_code)
    
    return latex_code


def create_performance_by_params_table(
    results_df: pd.DataFrame,
    param1: str,
    param2: Optional[str] = None,
    metric: str = 'balanced_accuracy',
    metric_name: Optional[str] = None,
    higher_is_better: bool = True,
    output_file: Optional[Union[str, Path]] = None,
    alpha: float = 0.05
) -> str:
    """
    Create a LaTeX table showing performance by varying parameters.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from simulation
    param1 : str
        Primary grouping parameter (rows)
    param2 : str, optional
        Secondary parameter (columns). If None, uses models as columns.
    metric : str, default='balanced_accuracy'
        Column name of the metric to display
    metric_name : str, optional
        Display name for the metric. If None, uses metric column name.
    higher_is_better : bool, default=True
        Whether higher values are better for this metric
    output_file : str or Path, optional
        If provided, save table to this file
    alpha : float, default=0.05
        Significance level for statistical tests
        
    Returns
    -------
    str
        LaTeX table code
    """
    if metric_name is None:
        metric_name = metric.replace('_', ' ').title()
    
    # Get unique values
    param1_values = sorted(results_df[param1].unique())
    models = sorted(results_df['model_name'].unique())
    
    if param2:
        param2_values = sorted(results_df[param2].unique())
        col_spec = 'l' + 'c' * len(param2_values)
    else:
        col_spec = 'l' + 'c' * len(models)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    
    if param2:
        latex.append(f"  \\caption{{{metric_name} by {param1.replace('_', ' ').title()} and {param2.replace('_', ' ').title()}}}")
    else:
        latex.append(f"  \\caption{{{metric_name} by {param1.replace('_', ' ').title()}}}")
    
    label_suffix = metric.replace('_', '')
    latex.append(f"  \\label{{tab:{label_suffix}_by_{param1}}}")
    latex.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex.append("    \\toprule")
    
    # Header row
    header = f"    {param1.replace('_', ' ').title()}"
    if param2:
        for val in param2_values:
            header += f" & {param2.replace('_', ' ')} = {val}"
    else:
        for model in models:
            header += f" & {model}"
    latex.append(header + " \\\\")
    latex.append("    \\midrule")
    
    # Data rows
    for p1_val in param1_values:
        if param2:
            # Show each model separately for this param1 value
            # Add section header for this param1 value
            latex.append(f"    \\multicolumn{{{len(param2_values)+1}}}{{l}}{{$\\mathbf{{{param1} = {p1_val}}}$}} \\\\")
            
            # For each param2 value, find which model(s) are significantly best
            winners_by_p2 = {}
            for p2_val in param2_values:
                # Get data for each model at this param1/param2 combination
                model_data_groups = {}
                for model in models:
                    mask = (results_df[param1] == p1_val) & \
                           (results_df[param2] == p2_val) & \
                           (results_df['model_name'] == model)
                    values = results_df[mask][metric].values
                    values = values[~np.isnan(values)]
                    model_data_groups[model] = values
                
                # Find significantly best model(s) for this configuration
                winners = set()
                for i, model1 in enumerate(models):
                    data1 = model_data_groups[model1]
                    if len(data1) == 0:
                        continue
                    
                    is_sig_better = True
                    mean1 = np.mean(data1)
                    
                    for j, model2 in enumerate(models):
                        if i == j:
                            continue
                        
                        data2 = model_data_groups[model2]
                        if len(data2) == 0:
                            continue
                        
                        mean2 = np.mean(data2)
                        
                        if higher_is_better:
                            if mean1 <= mean2:
                                is_sig_better = False
                                break
                        else:
                            if mean1 >= mean2:
                                is_sig_better = False
                                break
                        
                        try:
                            t_stat, p_value = stats.ttest_ind(data1, data2)
                            if higher_is_better:
                                p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2
                            else:
                                p_value_one_tailed = p_value / 2 if t_stat < 0 else 1 - p_value / 2
                            
                            if p_value_one_tailed > alpha:
                                is_sig_better = False
                                break
                        except:
                            is_sig_better = False
                            break
                    
                    if is_sig_better:
                        winners.add(model1)
                
                winners_by_p2[p2_val] = winners
            
            # Now create rows for each model
            for model in models:
                row = f"    {model}"
                
                for p2_val in param2_values:
                    mask = (results_df[param1] == p1_val) & \
                           (results_df[param2] == p2_val) & \
                           (results_df['model_name'] == model)
                    values = results_df[mask][metric].values
                    values = values[~np.isnan(values)]
                    
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        is_bold = model in winners_by_p2[p2_val]
                        row += f" & {format_mean_std(mean_val, std_val, bold=is_bold)}"
                    else:
                        row += " & ---"
                
                latex.append(row + " \\\\")
            
            # Add separator between param1 groups (except after last)
            if p1_val != param1_values[-1]:
                latex.append("    \\midrule")
        else:
            # One row per param1 value, columns are models
            row = f"    ${param1} = {p1_val}$"
            
            # Get data for each model
            data_groups = {}
            for model in models:
                mask = (results_df[param1] == p1_val) & (results_df['model_name'] == model)
                values = results_df[mask][metric].values
                values = values[~np.isnan(values)]
                data_groups[model] = values
            
            # Find significantly best model(s) for this configuration
            winners = set()
            for i, model1 in enumerate(models):
                data1 = data_groups[model1]
                if len(data1) == 0:
                    continue
                
                is_sig_better = True
                mean1 = np.mean(data1)
                
                for j, model2 in enumerate(models):
                    if i == j:
                        continue
                    
                    data2 = data_groups[model2]
                    if len(data2) == 0:
                        continue
                    
                    mean2 = np.mean(data2)
                    
                    if higher_is_better:
                        if mean1 <= mean2:
                            is_sig_better = False
                            break
                    else:
                        if mean1 >= mean2:
                            is_sig_better = False
                            break
                    
                    try:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        if higher_is_better:
                            p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2
                        else:
                            p_value_one_tailed = p_value / 2 if t_stat < 0 else 1 - p_value / 2
                        
                        if p_value_one_tailed > alpha:
                            is_sig_better = False
                            break
                    except:
                        is_sig_better = False
                        break
                
                if is_sig_better:
                    winners.add(model1)
            
            # Add values for each model
            for model in models:
                values = data_groups[model]
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    is_bold = model in winners
                    row += f" & {format_mean_std(mean_val, std_val, bold=is_bold)}"
                else:
                    row += " & ---"
            
            latex.append(row + " \\\\")
    
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    
    latex.append("  \\begin{tablenotes}")
    latex.append("    \\small")
    latex.append(f"    \\item Note: Bold values indicate statistically significant best model for each configuration ($p < {alpha}$).")
    latex.append("  \\end{tablenotes}")
    
    latex.append("\\end{table}")
    
    latex_code = "\n".join(latex)
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(latex_code)
    
    return latex_code


def create_hyperparameter_table(
    results_df: pd.DataFrame,
    output_file: Optional[Union[str, Path]] = None
) -> str:
    """
    Create a LaTeX table showing best hyperparameter selections.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from simulation
    output_file : str or Path, optional
        If provided, save table to this file
        
    Returns
    -------
    str
        LaTeX table code
    """
    models = sorted(results_df['model_name'].unique())
    
    # Check which hyperparameter columns exist
    has_n_components = 'best_n_components' in results_df.columns
    has_jump_penalty = 'best_jump_penalty' in results_df.columns
    has_max_feats = 'best_max_feats' in results_df.columns
    
    n_cols = 1 + sum([has_n_components, has_jump_penalty, has_max_feats])
    col_spec = 'l' + 'c' * (n_cols - 1)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Best Hyperparameters: Mean (Std) of selected values}")
    latex.append("  \\label{tab:hyperparameters}")
    latex.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex.append("    \\toprule")
    
    # Header row
    header = "    Model"
    if has_n_components:
        header += " & States ($K$)"
    if has_jump_penalty:
        header += " & Jump Penalty ($\\lambda$)"
    if has_max_feats:
        header += " & Max Features ($\\kappa^2$)"
    latex.append(header + " \\\\")
    latex.append("    \\midrule")
    
    for model in models:
        model_data = results_df[results_df['model_name'] == model]
        row = f"    {model}"
        
        if has_n_components:
            k_values = model_data['best_n_components'].values
            k_values = k_values[~np.isnan(k_values)]
            if len(k_values) > 0:
                row += f" & {format_mean_std(np.mean(k_values), np.std(k_values), 1)}"
            else:
                row += " & ---"
        
        if has_jump_penalty:
            lambda_values = model_data['best_jump_penalty'].values
            lambda_values = lambda_values[~np.isnan(lambda_values)]
            if len(lambda_values) > 0:
                row += f" & {format_mean_std(np.mean(lambda_values), np.std(lambda_values), 1)}"
            else:
                row += " & ---"
        
        if has_max_feats:
            kappa_values = model_data['best_max_feats'].values
            kappa_values = kappa_values[~np.isnan(kappa_values)]
            if len(kappa_values) > 0:
                row += f" & {format_mean_std(np.mean(kappa_values), np.std(kappa_values), 1)}"
            else:
                row += " & ---"
        
        latex.append(row + " \\\\")
    
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    
    latex_code = "\n".join(latex)
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(latex_code)
    
    return latex_code


def create_all_tables(
    results: Union[pd.DataFrame, str, Path, 'SimulationResults'],
    output_dir: Union[str, Path] = "tables",
    alpha: float = 0.05,
    verbose: bool = True
) -> Path:
    """
    Create all relevant LaTeX tables based on experiment configuration.
    
    Automatically detects varying parameters and creates appropriate tables.
    
    Parameters
    ----------
    results : pd.DataFrame, str, Path, or SimulationResults
        Results dataframe or path to results directory
    output_dir : str or Path, default="tables"
        Directory to save the LaTeX table files
    alpha : float, default=0.05
        Significance level for statistical tests
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    Path
        Directory where tables were saved
        
    Examples
    --------
    >>> from visualization import create_all_tables
    >>> create_all_tables(results, output_dir="results/tables")
    """
    # Handle different input types
    if isinstance(results, (str, Path)):
        from simulation.api import SimulationResults
        results_obj = SimulationResults(results)
        results_df = results_obj.best_df
        output_base = results_obj.path
    elif hasattr(results, 'best_df'):
        results_df = results.best_df
        output_base = results.path
    else:
        results_df = results
        output_base = Path.cwd()
    
    # Create output directory
    output_path = output_base / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 80)
        print("Creating LaTeX Tables from Simulation Results")
        print("=" * 80)
        print(f"\nResults: {len(results_df)} rows")
        print(f"Models: {results_df['model_name'].unique().tolist()}")
        print(f"Significance level: α = {alpha}")
    
    # Detect varying parameters
    varying_params = detect_varying_parameters(results_df)
    
    if verbose:
        print(f"Varying parameters: {varying_params}")
    
    table_count = 1
    
    # Always create model comparison table
    if verbose:
        print(f"\n{table_count}. Creating Model Comparison Table...")
    create_model_comparison_table(
        results_df,
        output_file=output_path / f"{table_count:02d}_model_comparison.tex",
        alpha=alpha
    )
    table_count += 1
    
    # Create tables for each varying parameter
    key_metrics = [
        ('balanced_accuracy', 'Balanced Accuracy', True),
        ('composite_score', 'Composite Score', True),
        ('feature_f1', 'Feature F1', True),
        ('feature_recall', 'Feature Recall', True),
        ('breakpoint_f1', 'Breakpoint F1', True),
        ('chamfer_distance', 'Chamfer Distance', False),
        ('breakpoint_error', 'Breakpoint Error', False),
        ('bic', 'BIC', False),
        ('aic', 'AIC', False),
    ]
    
    for metric, metric_name, higher_is_better in key_metrics:
        if metric not in results_df.columns or not results_df[metric].notna().any():
            continue
        
        if len(varying_params) == 1:
            # Single varying parameter
            param = varying_params[0]
            if verbose:
                print(f"\n{table_count}. Creating {metric_name} by {param} Table...")
            create_performance_by_params_table(
                results_df,
                param1=param,
                metric=metric,
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                output_file=output_path / f"{table_count:02d}_{metric}_by_{param}.tex",
                alpha=alpha
            )
            table_count += 1
        
        elif len(varying_params) >= 2:
            # Two varying parameters - create grid table
            param1, param2 = varying_params[:2]
            if verbose:
                print(f"\n{table_count}. Creating {metric_name} by {param1} and {param2} Table...")
            create_performance_by_params_table(
                results_df,
                param1=param1,
                param2=param2,
                metric=metric,
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                output_file=output_path / f"{table_count:02d}_{metric}_by_{param1}_{param2}.tex",
                alpha=alpha
            )
            table_count += 1
    
    # Create hyperparameter table
    if verbose:
        print(f"\n{table_count}. Creating Hyperparameter Table...")
    create_hyperparameter_table(
        results_df,
        output_file=output_path / f"{table_count:02d}_hyperparameters.tex"
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print("All LaTeX tables created!")
        print("=" * 80)
        print(f"\nTables saved to: {output_path}/")
        print("\nYou can include these in your LaTeX document with:")
        print(f"  \\input{{{output_path}/01_model_comparison.tex}}")
        print(f"\nNote: Bold values indicate statistically significant best performance (p < {alpha})")
    
    return output_path
