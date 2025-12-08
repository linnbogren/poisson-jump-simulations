"""
Simulation Runner for Sparse Jump Model Evaluation

This module orchestrates the complete simulation study with support for:
- Parallel or sequential execution
- Grid search or Optuna hyperparameter optimization
- Multiple replications
- Model comparison across distributions
- Incremental result saving for resumability

Key functions:
- run_single_replication: Worker function for one data replication
- run_simulation: Main orchestration function
- select_best_models: Find best hyperparameters from grid search
"""

import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime
import warnings
from joblib import Parallel, delayed
import logging

# Configure logging for suppressed warnings
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Suppress specific division warnings at module level (before multiprocessing)
# This works across process boundaries unlike warnings.catch_warnings()
warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in divide',
    category=RuntimeWarning
)
warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in scalar subtract',
    category=RuntimeWarning
)

from .config import SimulationConfig, GridSearchResult, ExperimentConfig
from .data_generation import generate_data
from .hyperparameters import (
    create_hyperparameter_grid,
    create_optuna_study,
    suggest_hyperparameters
)
from .models import (
    fit_and_evaluate,
    results_to_grid_search_result,
    grid_results_to_dataframe
)
from .metrics import extract_breakpoints
from .results import (
    save_experiment_metadata,
    update_metadata_on_completion
)


###############################################################################
# Single Replication Worker
###############################################################################

def run_single_replication_grid(args: Tuple) -> List[GridSearchResult]:
    """
    Worker function for grid search on a single replication.
    
    Fits ALL model types (Gaussian, Poisson, PoissonKL) on the SAME
    generated data for fair comparison.
    
    Parameters
    ----------
    args : Tuple
        (config, hyperparameter_grid, model_names, save_models, optimize_metric, grid_n_jobs)
    
    Returns
    -------
    List[GridSearchResult]
        Results for all models and hyperparameter combinations
        If save_models=True, returns (results, best_models_dict)
    """
    config, hyperparameter_grid, model_names, save_models, optimize_metric, grid_n_jobs = args
    
    # Generate data ONCE - all models use same data
    X, states, breakpoints = generate_data(config)
    
    # Fit all models on same data
    all_results = []
    best_models = {}  # If saving, track best model for each type
    best_scores = {}  # Track best metric value for each model type
    failed_convergence = {}  # Track convergence failures
    
    # Initialize best scores based on optimization direction
    initial_score = float('inf') if optimize_metric in ['bic', 'aic'] else -1.0
    
    for model_name in model_names:
        best_scores[model_name] = initial_score
        failed_convergence[model_name] = 0
        
        # Helper function for parallel execution
        def fit_single_params(params):
            """Fit model with single hyperparameter combination."""
            result = fit_and_evaluate(
                X, states, config, model_name, params,
                return_model=save_models
            )
            return result, params
        
        # Parallelize grid search if grid_n_jobs > 1
        if grid_n_jobs == 1:
            # Sequential execution (original behavior)
            model_results = []
            for params in hyperparameter_grid:
                result = fit_and_evaluate(
                    X, states, config, model_name, params,
                    return_model=save_models
                )
                model_results.append((result, params))
        else:
            # Parallel execution across hyperparameters
            model_results = Parallel(n_jobs=grid_n_jobs, backend='loky')(
                delayed(fit_single_params)(params)
                for params in hyperparameter_grid
            )
        
        # Process results (same for both sequential and parallel)
        for result, params in model_results:
            if not result.get('success', False):
                # Track failure type
                if result.get('convergence_failed', False):
                    failed_convergence[model_name] += 1
                continue
            
            # Convert to GridSearchResult
            grid_result = results_to_grid_search_result(result, config)
            all_results.append(grid_result)
            
            # Track best model if saving (using configured metric)
            if save_models:
                # Check if optimize_metric is in results (model may have failed/not converged)
                if optimize_metric in result:
                    metric_value = result[optimize_metric]
                    # BIC and AIC should be minimized (lower is better)
                    is_better = (
                        (metric_value < best_scores[model_name]) if optimize_metric in ['bic', 'aic']
                        else (metric_value > best_scores[model_name])
                    )
                    if is_better:
                        best_scores[model_name] = metric_value
                        best_models[model_name] = {
                            'model': result.get('model'),
                            'wrapper': result.get('wrapper'),
                            'result': grid_result
                        }
                # If metric not in result but model succeeded, that's an error
                elif result.get('success', False):
                    raise KeyError(
                        f"Metric '{optimize_metric}' not found in successful model results. "
                        f"Available metrics: {list(result.keys())}"
                    )
    
    if save_models and best_models:
        return (all_results, best_models, failed_convergence)
    return (all_results, failed_convergence)


def run_single_replication_optuna(args: Tuple) -> List[GridSearchResult]:
    """
    Worker function for Optuna optimization on a single replication.
    
    Parameters
    ----------
    args : Tuple
        (config, n_trials, n_total_features, model_names, optimize_metric, grid_config, n_jobs)
    
    Returns
    -------
    List[GridSearchResult]
        Best results for each model type from Optuna optimization
    """
    config, n_trials, n_total_features, model_names, optimize_metric, grid_config, optuna_n_jobs = args
    
    # Generate data ONCE
    X, states, breakpoints = generate_data(config)
    
    all_results = []
    
    for model_name in model_names:
        # Define objective function for this model
        def objective(trial):
            params = suggest_hyperparameters(
                trial, 
                grid_config,
                n_total_features
            )
            result = fit_and_evaluate(X, states, config, model_name, params)
            
            if not result.get('success', False):
                return -1e6
            
            # Return metric to optimize (with fallback for failed models)
            if optimize_metric in result:
                return result[optimize_metric]
            else:
                # Metric missing even though model succeeded - this is a config error
                raise KeyError(
                    f"Metric '{optimize_metric}' not found in successful model results. "
                    f"Available metrics: {list(result.keys())}"
                )
        
        # Run Optuna optimization with parallel trials
        study = create_optuna_study(
            objective,
            grid_config=grid_config,
            n_total_features=n_total_features,
            n_trials=n_trials,
            seed=config.random_seed,
            direction='maximize',
            n_jobs=optuna_n_jobs
        )
        
        # Re-evaluate best trial to get full results
        best_params = {
            'n_components': int(study.best_params['n_components']),
            'jump_penalty': float(study.best_params['jump_penalty']),
            'max_feats': float(study.best_params['kappa'] ** 2)
        }
        
        best_result = fit_and_evaluate(X, states, config, model_name, best_params)
        
        if best_result.get('success', False):
            grid_result = results_to_grid_search_result(best_result, config)
            # Add trial information to the result
            grid_result.trial_number = len(study.trials)
            grid_result.study = study  # Store study for later analysis
            all_results.append(grid_result)
    
    return all_results


###############################################################################
# Main Simulation Runner
###############################################################################

def run_simulation(
    experiment_config: ExperimentConfig,
    data_configs: List[SimulationConfig],
    output_dir: str = "results",
    n_jobs: int = -1,
    save_models: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run complete simulation study with parallel or sequential execution.
    
    This is the main entry point for running simulations. Supports:
    - Grid search or Optuna optimization
    - Parallel or sequential execution
    - Multiple replications
    - Incremental saving for resumability
    - Model saving for later analysis
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration (optimization method, n_trials, etc.)
    data_configs : List[SimulationConfig]
        List of data generation configurations to test
    output_dir : str, default="results"
        Directory to save results
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all CPUs, 1 for sequential
    save_models : bool, default=False
        If True, save best fitted models to disk
    verbose : bool, default=True
        Print progress information
    
    Returns
    -------
    pd.DataFrame
        Best results for each configuration (one row per config/model combo)
        
    Examples
    --------
    >>> from simulation import ExperimentConfig, SimulationConfig
    >>> exp_config = ExperimentConfig(
    ...     optimization_method='grid',
    ...     quick_test=True,
    ...     n_replications=10
    ... )
    >>> data_configs = create_data_config_grid(quick_test=True)
    >>> results = run_simulation(exp_config, data_configs, n_jobs=4)
    """
    # Record start time
    start_time = datetime.now()
    
    # Setup parallelization strategy
    # If using parallel grid/optuna optimization, disable outer parallelization to avoid nesting
    n_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
    
    # Check if inner parallelization is enabled
    has_inner_parallelization = (
        (experiment_config.optimization_method == "grid" and experiment_config.grid_n_jobs != 1) or
        (experiment_config.optimization_method == "optuna" and experiment_config.optuna_n_jobs != 1)
    )
    
    # Disable outer parallelization if inner parallelization is active
    if has_inner_parallelization and n_workers > 1:
        if verbose:
            print(f"⚠️  Parallel {experiment_config.optimization_method} optimization detected")
            print(f"   Disabling replication-level parallelization to avoid nesting conflicts")
            print(f"   → Hyperparameters will be searched in parallel instead")
        use_parallel = False
        n_workers = 1
    else:
        use_parallel = n_workers > 1 and not experiment_config.single_thread
    
    if verbose:
        print(f"{'='*80}")
        print("Sparse Jump Model Simulation Study")
        print(f"{'='*80}")
        print(f"Optimization: {experiment_config.optimization_method}")
        print(f"Execution: {'Parallel' if use_parallel else 'Sequential'}")
        if use_parallel:
            print(f"Workers: {n_workers}")
        elif has_inner_parallelization:
            jobs_param = experiment_config.grid_n_jobs if experiment_config.optimization_method == "grid" else experiment_config.optuna_n_jobs
            print(f"Parallel {experiment_config.optimization_method} optimization: {jobs_param} jobs")
        print(f"Configurations: {len(data_configs)}")
        print(f"Models: {len(experiment_config.model_names)}")
        print(f"Replications: {experiment_config.n_replications}")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "aggregated").mkdir(exist_ok=True)
    (output_path / "grid_search").mkdir(exist_ok=True)
    incremental_dir = output_path / "grid_search" / "incremental"
    incremental_dir.mkdir(parents=True, exist_ok=True)
    
    if save_models:
        (output_path / "models").mkdir(exist_ok=True)
        if verbose:
            print(f"Model saving ENABLED -> {output_path / 'models'}")
    
    # Save experiment metadata
    save_experiment_metadata(
        output_dir=output_path,
        experiment_config=experiment_config,
        data_configs=data_configs,
        start_time=start_time,
        n_workers=n_workers if use_parallel else 1
    )
    
    if verbose:
        print(f"Metadata saved to {output_path / 'metadata.json'}")
    
    # Prepare tasks
    all_tasks = []
    
    for data_config in data_configs:
        # Create hyperparameter grid for this configuration
        if experiment_config.optimization_method == "grid":
            hyperparam_grid = create_hyperparameter_grid(
                experiment_config.hyperparameter_grid,
                data_config.n_total_features
            )
        else:
            hyperparam_grid = None  # Not used for Optuna
        
        for rep in range(experiment_config.n_replications):
            # Create unique config for this replication
            config = SimulationConfig(
                n_samples=data_config.n_samples,
                n_states=data_config.n_states,
                n_informative=data_config.n_informative,
                n_total_features=data_config.n_total_features,
                delta=data_config.delta,
                lambda_0=data_config.lambda_0,
                persistence=data_config.persistence,
                distribution_type=data_config.distribution_type,
                correlated_noise=data_config.correlated_noise,
                noise_correlation=data_config.noise_correlation,
                nb_dispersion=data_config.nb_dispersion,
                random_seed=data_config.random_seed + rep * 1000
            )
            
            if experiment_config.optimization_method == "grid":
                task = (config, hyperparam_grid, experiment_config.model_names, 
                       save_models, experiment_config.optimize_metric, experiment_config.grid_n_jobs)
            else:  # Optuna
                task = (
                    config,
                    experiment_config.optuna_n_trials,
                    data_config.n_total_features,
                    experiment_config.model_names,
                    experiment_config.optimize_metric,
                    experiment_config.hyperparameter_grid,
                    experiment_config.optuna_n_jobs
                )
            
            all_tasks.append(task)
    
    if verbose:
        print(f"Total tasks: {len(all_tasks)}")
        if experiment_config.optimization_method == "grid":
            grid_size = len(hyperparam_grid) if hyperparam_grid else 0
            print(f"Grid size: {grid_size}")
            print(f"Total model fits: {len(all_tasks) * len(experiment_config.model_names) * grid_size:,}")
    
    # Check for existing results
    existing_files = sorted(incremental_dir.glob("batch_*.pkl"))
    if existing_files:
        if verbose:
            print(f"\nFound {len(existing_files)} existing batches")
            print(f"Resuming from task {len(existing_files) + 1}...")
        tasks_to_run = all_tasks[len(existing_files):]
        batch_offset = len(existing_files)
    else:
        tasks_to_run = all_tasks
        batch_offset = 0
    
    if len(tasks_to_run) == 0:
        if verbose:
            print("All tasks completed! Proceeding to aggregation...")
    else:
        # Select worker function
        if experiment_config.optimization_method == "grid":
            worker_fn = run_single_replication_grid
        else:
            worker_fn = run_single_replication_optuna
        
        # Run tasks
        if use_parallel:
            if verbose:
                print(f"\nRunning {len(tasks_to_run)} tasks in parallel...")
            
            with Pool(processes=n_workers) as pool:
                results_iter = pool.imap_unordered(worker_fn, tasks_to_run)
                
                batch_idx = batch_offset
                for result in tqdm(results_iter, total=len(tasks_to_run),
                                 desc="Simulations", initial=batch_offset,
                                 disable=not verbose):
                    # TODO: Re-enable exception handling after testing
                    # try:
                    # Unpack result tuple
                    if isinstance(result, tuple) and len(result) == 3:
                        # With models: (grid_results, best_models_dict, failed_conv)
                        grid_results, best_models_dict, failed_conv = result
                    elif isinstance(result, tuple) and len(result) == 2:
                        # Without models: (grid_results, failed_conv)
                        grid_results, failed_conv = result
                        best_models_dict = None
                    else:
                        # Legacy format or error
                        grid_results = result
                        best_models_dict = None
                        failed_conv = {}
                    
                    # Handle model saving
                    if save_models and best_models_dict:
                        # Save models
                        for model_name, model_info in best_models_dict.items():
                            grid_res = model_info['result']
                            model_filename = (
                                f"model_{model_name}_"
                                f"seed{grid_res.config.random_seed}_"
                                f"P{grid_res.config.n_total_features}_"
                                f"delta{grid_res.config.delta}.pkl"
                            )
                            with open(output_path / "models" / model_filename, 'wb') as f:
                                pickle.dump(model_info['model'], f)
                    
                    # Save batch
                    batch_file = incremental_dir / f"batch_{batch_idx:06d}.pkl"
                    with open(batch_file, 'wb') as f:
                        pickle.dump(grid_results, f)
                    
                    batch_idx += 1
                    
                    # except Exception as e:
                    #     if verbose:
                    #         print(f"\nError in task: {e}")
                    #     batch_idx += 1
                    #     continue
        
        else:  # Sequential execution
            if verbose:
                print(f"\nRunning {len(tasks_to_run)} tasks sequentially...")
            
            batch_idx = batch_offset
            for task in tqdm(tasks_to_run, desc="Simulations", disable=not verbose):
                # TODO: Re-enable exception handling after testing
                # try:
                result = worker_fn(task)
                
                # Unpack result tuple
                if isinstance(result, tuple) and len(result) == 3:
                    # With models: (grid_results, best_models_dict, failed_conv)
                    grid_results, best_models_dict, failed_conv = result
                elif isinstance(result, tuple) and len(result) == 2:
                    # Without models: (grid_results, failed_conv)
                    grid_results, failed_conv = result
                    best_models_dict = None
                else:
                    # Legacy format or error
                    grid_results = result
                    best_models_dict = None
                    failed_conv = {}
                
                # Handle model saving
                if save_models and best_models_dict:
                    # Save models
                    for model_name, model_info in best_models_dict.items():
                        grid_res = model_info['result']
                        model_filename = (
                            f"model_{model_name}_"
                            f"seed{grid_res.config.random_seed}_"
                            f"P{grid_res.config.n_total_features}_"
                            f"delta{grid_res.config.delta}.pkl"
                        )
                        with open(output_path / "models" / model_filename, 'wb') as f:
                            pickle.dump(model_info['model'], f)
                
                # Save batch
                batch_file = incremental_dir / f"batch_{batch_idx:06d}.pkl"
                with open(batch_file, 'wb') as f:
                    pickle.dump(grid_results, f)
                
                batch_idx += 1
                
                # except Exception as e:
                #     if verbose:
                #         print(f"\nError in task: {e}")
                #     batch_idx += 1
                #     continue        # Print info about suppressed warnings
        if verbose and experiment_config.optimization_method == "optuna":
            print(f"\n💡 Note: Division warnings from extreme hyperparameter exploration were suppressed")
            print(f"   (This is normal - Optuna tests boundary values during optimization)")
    
    # Aggregate results
    if verbose:
        print(f"\n{'='*80}")
        print("Aggregating results...")
        print(f"{'='*80}")
    
    all_batch_files = sorted(incremental_dir.glob("batch_*.pkl"))
    if verbose:
        print(f"Loading {len(all_batch_files)} batches...")
    
    # Process in chunks to manage memory
    chunk_size = 100
    all_grid_dfs = []
    
    for chunk_start in range(0, len(all_batch_files), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_batch_files))
        
        chunk_results = []
        for batch_file in all_batch_files[chunk_start:chunk_end]:
            # TODO: Re-enable exception handling after testing
            # try:
            with open(batch_file, 'rb') as f:
                batch_results = pickle.load(f)
                chunk_results.extend(batch_results)
            # except Exception as e:
            #     if verbose:
            #         print(f"Warning: Could not load {batch_file.name}: {e}")
        
        if chunk_results:
            chunk_df = grid_results_to_dataframe(chunk_results)
            all_grid_dfs.append(chunk_df)
        
        del chunk_results
    
    # Combine all chunks
    grid_df = pd.concat(all_grid_dfs, ignore_index=True)
    del all_grid_dfs
    
    # Extract and save Optuna trial data if using Optuna
    if experiment_config.optimization_method == "optuna":
        if verbose:
            print("Extracting Optuna trial data...")
        
        # Collect all Optuna studies from the batch results
        optuna_trials_data = []
        for batch_file in all_batch_files:
            with open(batch_file, 'rb') as f:
                batch_results = pickle.load(f)
                for result in batch_results:
                    if hasattr(result, 'study'):
                        study = result.study
                        # Extract trial data
                        for trial in study.trials:
                            trial_data = {
                                'model_name': result.model_name,
                                'trial_number': trial.number,
                                'value': trial.value,
                                'n_components': trial.params.get('n_components'),
                                'jump_penalty': trial.params.get('jump_penalty'),
                                'kappa': trial.params.get('kappa'),
                                'state': trial.state.name,
                                'duration': trial.duration.total_seconds() if trial.duration else None,
                            }
                            # Add configuration identifiers
                            trial_data['delta'] = result.config.delta
                            trial_data['n_states'] = result.config.n_states
                            trial_data['random_seed'] = result.config.random_seed
                            optuna_trials_data.append(trial_data)
        
        if optuna_trials_data:
            optuna_trials_df = pd.DataFrame(optuna_trials_data)
            optuna_trials_path = output_path / "grid_search" / "optuna_trials.csv"
            optuna_trials_df.to_csv(optuna_trials_path, index=False)
            
            if verbose:
                print(f"Saved {len(optuna_trials_df)} Optuna trials to optuna_trials.csv")
    
    if verbose:
        print(f"Total evaluations: {len(grid_df)}")
    
    # Save full grid search results
    grid_df.to_csv(output_path / "grid_search" / "all_grid_results.csv", index=False)
    
    # Select best models
    if verbose:
        print(f"Selecting best models (optimizing {experiment_config.optimize_metric})...")
    
    best_df = select_best_models(grid_df, metric=experiment_config.optimize_metric)
    
    # Save best results
    best_df.to_csv(output_path / "aggregated" / "all_results.csv", index=False)
    
    # Clean up incremental files
    if verbose:
        print("Cleaning up incremental files...")
    for f in incremental_dir.glob("batch_*.pkl"):
        f.unlink()
    
    # Update metadata with completion info
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    update_metadata_on_completion(output_path, end_time, execution_time)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Complete! Results saved to {output_dir}")
        print(f"Grid evaluations: {len(grid_df)}")
        print(f"Best models: {len(best_df)}")
        print(f"Execution time: {execution_time:.1f}s ({execution_time/60:.1f} min)")
        print(f"{'='*80}")
    
    return best_df


def select_best_models(grid_df: pd.DataFrame, metric: str = 'balanced_accuracy') -> pd.DataFrame:
    """
    Select best model for each (config, model_name) combination.
    
    Automatically filters out failed runs (those missing the optimization metric).
    
    Parameters
    ----------
    grid_df : pd.DataFrame
        DataFrame with all grid search results
    metric : str, default='balanced_accuracy'
        Metric to use for selecting best models.
        Options: 'balanced_accuracy', 'composite_score', 'f1_breakpoint', etc.
    
    Returns
    -------
    pd.DataFrame
        Best results with renamed hyperparameter columns
    """
    group_cols = [
        'n_samples', 'n_states', 'n_informative', 'n_noise', 'n_total_features',
        'delta', 'lambda_0', 'persistence', 'distribution_type',
        'correlated_noise', 'random_seed', 'model_name'
    ]
    
    # Validate metric exists
    if metric not in grid_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results. Available: {grid_df.columns.tolist()}")
    
    # Filter out failed runs (those with NaN/missing metric values)
    # Failed runs don't have metrics computed
    valid_results = grid_df[grid_df[metric].notna()].copy()
    
    n_failed = len(grid_df) - len(valid_results)
    if n_failed > 0:
        print(f"  Filtered out {n_failed} failed runs (missing {metric})")
    
    if len(valid_results) == 0:
        raise ValueError(f"No valid results found! All runs are missing '{metric}'")
    
    # Find best (highest metric value) for each group
    idx = valid_results.groupby(group_cols)[metric].idxmax()
    best_df = valid_results.loc[idx].copy()
    
    # Rename hyperparameters
    best_df = best_df.rename(columns={
        'n_components': 'best_n_components',
        'jump_penalty': 'best_jump_penalty',
        'max_feats': 'best_max_feats'
    })
    
    # Keep n_selected_total for analysis (number of features actually selected by the model)
    # Note: This is NOT dropped anymore - it's useful for understanding feature selection behavior
    
    return best_df
