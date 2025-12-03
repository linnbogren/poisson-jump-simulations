"""
Simplified API for Sparse Jump Model Simulations

This module provides a high-level API for running simulations and managing results:
- run_simulation(): Main entry point for running experiments with automatic caching
- SimulationResults: Container class for accessing and exploring results

Example
-------
>>> from simulation import run_simulation
>>> 
>>> config = {
>>>     'name': 'my_experiment',
>>>     'data_configs': [{'delta': 0.1}, {'delta': 0.2}],
>>>     'n_replications': 10,
>>> }
>>> 
>>> results = run_simulation(config, cache=True)
>>> results.summary()
>>> print(results.best_df.head())
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class SimulationResults:
    """
    Container for simulation outputs.
    
    Provides convenient access to results DataFrames, metadata,
    and utility methods for exploring and analyzing results.
    
    Attributes
    ----------
    path : Path
        Directory containing results
    best_df : pd.DataFrame
        Best model results (one row per config/model combination)
    grid_df : pd.DataFrame
        All grid search results (all hyperparameter combinations)
    metadata : Dict
        Experiment metadata including configuration and timing
        
    Examples
    --------
    >>> results = SimulationResults(Path("results/my_experiment_abc123"))
    >>> results.summary()
    >>> print(f"Best accuracy: {results.best_df['balanced_accuracy'].max()}")
    """
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize SimulationResults from a results directory.
        
        Parameters
        ----------
        path : str or Path
            Path to results directory
        """
        self.path = Path(path)
        
        if not self.path.exists():
            raise FileNotFoundError(f"Results directory not found: {self.path}")
        
        # Load results
        self.best_df = self._load_best_results()
        self.grid_df = self._load_grid_results()
        self.metadata = self._load_metadata()
    
    def _load_best_results(self) -> pd.DataFrame:
        """Load best model results."""
        best_path = self.path / "aggregated" / "all_results.csv"
        if not best_path.exists():
            raise FileNotFoundError(f"Best results not found: {best_path}")
        return pd.read_csv(best_path)
    
    def _load_grid_results(self) -> pd.DataFrame:
        """Load grid search results."""
        grid_path = self.path / "grid_search" / "all_grid_results.csv"
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid results not found: {grid_path}")
        return pd.read_csv(grid_path)
    
    def _load_metadata(self) -> Dict:
        """Load experiment metadata."""
        metadata_path = self.path / "metadata.json"
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    @property
    def models(self) -> List[str]:
        """Get list of model names in results."""
        return self.best_df['model_name'].unique().tolist()
    
    @property
    def configs(self) -> pd.DataFrame:
        """Get data configuration summary."""
        config_cols = [
            'n_samples', 'n_states', 'n_informative', 'n_total_features',
            'delta', 'lambda_0', 'persistence', 'distribution_type',
            'correlated_noise'
        ]
        available_cols = [c for c in config_cols if c in self.best_df.columns]
        return self.best_df[available_cols].drop_duplicates().reset_index(drop=True)
    
    def summary(self) -> None:
        """Print summary statistics of the experiment."""
        print("=" * 80)
        print("SIMULATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"\nExperiment: {self.metadata.get('experiment_name', 'Unknown')}")
        print(f"Results directory: {self.path}")
        print(f"\nModels: {', '.join(self.models)}")
        print(f"Configurations: {len(self.configs)}")
        print(f"Replications: {self.metadata.get('n_replications', 'Unknown')}")
        
        if 'execution_time' in self.metadata:
            exec_time = self.metadata['execution_time']
            print(f"Execution time: {exec_time:.1f}s ({exec_time/60:.1f} min)")
        
        print(f"\n{'Metric':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)
        
        metrics = ['balanced_accuracy', 'feature_f1', 'chamfer_distance']
        for metric in metrics:
            if metric in self.best_df.columns:
                values = self.best_df[metric]
                print(f"{metric:<30} {values.mean():<12.4f} {values.std():<12.4f} "
                      f"{values.min():<12.4f} {values.max():<12.4f}")
        
        print("=" * 80)
    
    def get_config(self, config_id: int = 0) -> Dict:
        """
        Get data configuration parameters for a specific configuration.
        
        Parameters
        ----------
        config_id : int, default=0
            Configuration index
            
        Returns
        -------
        dict
            Configuration parameters
        """
        if config_id >= len(self.configs):
            raise ValueError(f"config_id {config_id} out of range (max: {len(self.configs)-1})")
        
        return self.configs.iloc[config_id].to_dict()
    
    def load_model(self, config_id: int = 0, model_name: str = 'Poisson', 
                   replication: int = 0) -> Any:
        """
        Load a fitted model from disk.
        
        Parameters
        ----------
        config_id : int, default=0
            Configuration ID
        model_name : str, default='Poisson'
            Model name
        replication : int, default=0
            Replication number
            
        Returns
        -------
        model
            Fitted model object
            
        Raises
        ------
        FileNotFoundError
            If model file doesn't exist
        """
        import pickle
        
        model_path = self.path / "models" / f"config{config_id}_{model_name}_{replication}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Models may not have been saved (set save_models=True in run_simulation)"
            )
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def get_best_models(self, metric: str = 'balanced_accuracy', 
                       ascending: bool = False) -> pd.DataFrame:
        """
        Get best performing models sorted by specified metric.
        
        Parameters
        ----------
        metric : str, default='balanced_accuracy'
            Metric to sort by (e.g., 'balanced_accuracy', 'feature_f1', 
            'composite_score', 'chamfer_distance')
        ascending : bool, default=False
            If True, sort ascending (lower is better). 
            If False, sort descending (higher is better).
            
        Returns
        -------
        pd.DataFrame
            Best results sorted by metric
            
        Examples
        --------
        >>> # Get models with highest balanced accuracy
        >>> best = results.get_best_models('balanced_accuracy')
        >>> 
        >>> # Get models with lowest chamfer distance
        >>> best = results.get_best_models('chamfer_distance', ascending=True)
        """
        if metric not in self.best_df.columns:
            available = [c for c in self.best_df.columns if not c.startswith('_')]
            raise ValueError(
                f"Metric '{metric}' not found in results. "
                f"Available metrics: {', '.join(available)}"
            )
        
        return self.best_df.sort_values(metric, ascending=ascending)
    
    def get_timing_info(self) -> Dict[str, float]:
        """
        Get execution timing information.
        
        Returns
        -------
        dict
            Dictionary with timing information:
            - total_time: Total execution time in seconds
            - time_per_config: Average time per configuration
            - time_per_evaluation: Average time per grid evaluation
            
        Examples
        --------
        >>> timing = results.get_timing_info()
        >>> print(f"Total: {timing['total_time']:.1f}s")
        >>> print(f"Per config: {timing['time_per_config']:.2f}s")
        """
        total_time = self.metadata.get('execution_time_seconds', 
                                      self.metadata.get('execution_time', 0))
        n_configs = len(self.configs)
        n_evaluations = len(self.grid_df)
        
        return {
            'total_time': total_time,
            'time_per_config': total_time / n_configs if n_configs > 0 else 0,
            'time_per_evaluation': total_time / n_evaluations if n_evaluations > 0 else 0,
            'n_configs': n_configs,
            'n_evaluations': n_evaluations,
            'optimization_method': self.metadata.get('optimization_method', 'unknown'),
        }
    
    def get_performance_summary(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get summary statistics for performance metrics.
        
        Parameters
        ----------
        metrics : list of str, optional
            Metrics to include. If None, uses default set.
            
        Returns
        -------
        pd.DataFrame
            Summary statistics (mean, std, min, max) for each metric
            
        Examples
        --------
        >>> summary = results.get_performance_summary()
        >>> print(summary)
        """
        if metrics is None:
            metrics = ['balanced_accuracy', 'feature_f1', 'chamfer_distance', 
                      'breakpoint_count_error', 'composite_score']
        
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in self.best_df.columns]
        
        if not available_metrics:
            raise ValueError(f"None of the requested metrics found in results")
        
        summary_data = []
        for metric in available_metrics:
            values = self.best_df[metric]
            summary_data.append({
                'metric': metric,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
            })
        
        return pd.DataFrame(summary_data)
    
    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return self.path


def run_simulation(
    config: Dict,
    cache: bool = True,
    output_dir: str = "results",
    save_models: bool = False,
    verbose: bool = True
) -> SimulationResults:
    """
    Run simulation study with automatic caching.
    
    This is the main entry point for running experiments. It:
    1. Hashes the configuration
    2. Checks for cached results (if cache=True)
    3. Runs the simulation if needed
    4. Returns a SimulationResults object
    
    Parameters
    ----------
    config : dict
        Simplified configuration with keys:
        - experiment_name or name: str (required) - Experiment name
        - data_generation or data_configs: Dict or List[Dict] (required) - Data configurations
        - models_to_run or models: List[str] (default: ['Gaussian', 'Poisson', 'PoissonKL'])
        - num_simulations or n_replications: int (default: 10)
        - hyperparameters or hyperparameter_grid: Dict (optional)
        - n_jobs: int (default: -1, use all cores)
        - optimization: str (default: 'grid')
        - optimize_metric: str (default: 'balanced_accuracy')
    cache : bool, default=True
        Check for existing results and load if found
    output_dir : str, default="results"
        Base directory for results
    save_models : bool, default=False
        Save fitted models to disk (required for time series visualization)
    verbose : bool, default=True
        Print progress information
        
    Returns
    -------
    SimulationResults
        Object containing results and metadata
        
    Examples
    --------
    >>> config = {
    ...     'experiment_name': 'simple_test',
    ...     'data_generation': {'T': 1000, 'n_states': 2},
    ...     'models_to_run': ['AR', 'Gaussian_HMM'],
    ...     'num_simulations': 10,
    ... }
    >>> results = run_simulation(config)
    >>> results.summary()
    """
    from .cache import hash_config, find_cached_results, save_config_hash
    from .config import dict_to_experiment_config
    from .runner import run_simulation as run_simulation_core
    from copy import deepcopy
    
    # Normalize config keys to internal format
    config = deepcopy(config)
    
    # Handle experiment name (experiment_name -> name)
    if 'experiment_name' in config and 'name' not in config:
        config['name'] = config.pop('experiment_name')
    
    # Handle data generation (data_generation -> data_configs)
    if 'data_generation' in config and 'data_configs' not in config:
        data_gen = config.pop('data_generation')
        config['data_configs'] = [data_gen] if isinstance(data_gen, dict) else data_gen
    
    # Handle models (models_to_run -> models)
    if 'models_to_run' in config and 'models' not in config:
        config['models'] = config.pop('models_to_run')
    
    # Handle num simulations (num_simulations -> n_replications)
    if 'num_simulations' in config and 'n_replications' not in config:
        config['n_replications'] = config.pop('num_simulations')
    
    # Handle hyperparameters (hyperparameters -> hyperparameter_grid)
    if 'hyperparameters' in config and 'hyperparameter_grid' not in config:
        config['hyperparameter_grid'] = config.pop('hyperparameters')
    
    # Validate config
    if 'name' not in config:
        raise ValueError("config must include 'experiment_name' or 'name'")
    if 'data_configs' not in config or not config['data_configs']:
        raise ValueError("config must include 'data_generation' or 'data_configs'")
    
    # Compute config hash
    config_hash = hash_config(config)
    
    if verbose:
        print("=" * 80)
        print("SIMULATION EXECUTION")
        print("=" * 80)
        print(f"Experiment: {config['name']}")
        print(f"Config hash: {config_hash[:16]}...")
    
    # Check for cached results
    if cache:
        cached_path = find_cached_results(config_hash, output_dir)
        if cached_path:
            if verbose:
                print(f"\n✓ Found cached results: {cached_path.name}")
                print("  Loading existing results (use cache=False to re-run)")
                print("=" * 80)
            return SimulationResults(cached_path)
    
    if verbose:
        if cache:
            print("\n✗ No cached results found, running simulation...")
        else:
            print("\n  Cache disabled, running simulation...")
    
    # Convert to ExperimentConfig and get simulation configs with defaults applied
    exp_config, sim_configs = dict_to_experiment_config(config)
    
    # Use the simulation configs that were created with defaults
    if exp_config.mode == 'single':
        data_configs = [exp_config.data]
    else:
        data_configs = sim_configs
    
    # Create output directory with hash
    result_path = Path(output_dir) / f"{config['name']}_{config_hash[:8]}"
    
    if verbose:
        print(f"  Output directory: {result_path}")
        print("=" * 80)
    
    # Run simulation
    run_simulation_core(
        experiment_config=exp_config,
        data_configs=data_configs,
        output_dir=str(result_path),
        n_jobs=config.get('n_jobs', -1),
        save_models=save_models,
        verbose=verbose
    )
    
    # Save config hash for future lookups
    save_config_hash(result_path, config_hash, config)
    
    if verbose:
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {result_path}")
        print("=" * 80)
    
    return SimulationResults(result_path)


def fit_on_real_data(
    X: pd.DataFrame,
    models: Optional[List[str]] = None,
    n_components_range: Optional[List[int]] = None,
    optimize_metric: str = 'bic',
    optimization_method: str = 'grid',
    n_trials: int = 50,
    hyperparameter_grid: Optional[Dict] = None,
    n_jobs: int = -1,
    save_models: bool = True,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fit sparse jump models on real data using unsupervised optimization.
    
    This function is for analyzing real data where ground truth labels are not available.
    It fits models and optimizes hyperparameters using unsupervised metrics (BIC, AIC, Silhouette).
    
    Parameters
    ----------
    X : pd.DataFrame
        Time series data (n_samples, n_features)
    models : list of str, optional
        Models to fit. Default: ['Gaussian', 'Poisson', 'PoissonKL']
    n_components_range : list of int, optional
        Range of number of states to try. Default: [2, 3, 4, 5]
    optimize_metric : str, default='bic'
        Metric to optimize. Options: 'bic', 'aic', 'silhouette'
        - 'bic': Bayesian Information Criterion (lower is better)
        - 'aic': Akaike Information Criterion (lower is better)  
        - 'silhouette': Silhouette coefficient (higher is better)
    optimization_method : str, default='grid'
        Optimization method: 'grid' or 'optuna'
        - 'grid': Exhaustive grid search over hyperparameters
        - 'optuna': Bayesian optimization (faster, fewer evaluations)
    n_trials : int, default=50
        Number of Optuna trials (only used if optimization_method='optuna')
    hyperparameter_grid : dict, optional
        Custom hyperparameter grid/ranges. If None, uses reasonable defaults.
        For grid search: Keys 'kappa_range', 'jump_penalty_range' as (min, max, num_points)
        For optuna: Keys 'kappa_min', 'kappa_max', 'jump_penalty_min', 'jump_penalty_max'
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    save_models : bool, default=True
        Save fitted models for later visualization
    output_dir : str, optional
        Directory to save results. If None, creates timestamped directory
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'results_df': DataFrame with all model results
        - 'best_models': Dict mapping model names to best fitted ModelWrapper objects
        - 'output_dir': Path where results were saved
        - 'optimization_method': Method used ('grid' or 'optuna')
        - 'studies': (optuna only) Dict of Optuna study objects per model
        
    Examples
    --------
    >>> import pandas as pd
    >>> from simulation import fit_on_real_data
    >>> 
    >>> # Load your data
    >>> X = pd.read_csv('my_data.csv')
    >>> 
    >>> # Grid search (exhaustive)
    >>> results = fit_on_real_data(
    ...     X, 
    ...     models=['Poisson', 'PoissonKL'],
    ...     optimize_metric='bic',
    ...     optimization_method='grid',
    ...     n_components_range=[2, 3, 4]
    ... )
    >>> 
    >>> # Optuna optimization (faster)
    >>> results = fit_on_real_data(
    ...     X,
    ...     optimize_metric='silhouette',
    ...     optimization_method='optuna',
    ...     n_trials=100
    ... )
    >>> 
    >>> # Access best model
    >>> best_poisson = results['best_models']['Poisson']
    >>> print(f"Best Poisson BIC: {best_poisson.evaluate_unsupervised(X)['bic']:.2f}")
    >>> 
    >>> # For Optuna, access study for analysis
    >>> if 'studies' in results:
    ...     study = results['studies']['Poisson']
    ...     print(f"Best trial: {study.best_trial.number}")
    >>> 
    >>> # Visualize stacked time series
    >>> from visualization import plot_stacked_time_series
    >>> plot_stacked_time_series(X, best_poisson.model, save_path='results_plot.png')
    """
    import time
    import numpy as np
    from joblib import Parallel, delayed
    from .models import ModelWrapper
    
    # Validate optimization method
    if optimization_method not in ['grid', 'optuna']:
        raise ValueError(f"optimization_method must be 'grid' or 'optuna', got '{optimization_method}'")
    
    # Set defaults
    if models is None:
        models = ['Gaussian', 'Poisson', 'PoissonKL']
    
    if n_components_range is None:
        n_components_range = [2, 3, 4, 5]
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"real_data_results_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 80)
        print("FITTING SPARSE JUMP MODELS ON REAL DATA")
        print("=" * 80)
        print(f"Data shape: {X.shape}")
        print(f"Models: {', '.join(models)}")
        print(f"Optimization method: {optimization_method}")
        print(f"Optimizing: {optimize_metric} ({'minimize' if optimize_metric in ['bic', 'aic'] else 'maximize'})")
        print(f"Number of states to try: {n_components_range}")
        if optimization_method == 'optuna':
            print(f"Trials per model/state: {n_trials}")
        print("=" * 80)
    
    start_time = time.time()
    
    if optimization_method == 'grid':
        results = _fit_real_data_grid(
            X, models, n_components_range, optimize_metric,
            hyperparameter_grid, n_jobs, save_models, output_dir, verbose
        )
    else:  # optuna
        results = _fit_real_data_optuna(
            X, models, n_components_range, optimize_metric,
            n_trials, hyperparameter_grid, n_jobs, save_models, output_dir, verbose
        )
    
    elapsed = time.time() - start_time
    results['execution_time'] = elapsed
    results['optimization_method'] = optimization_method
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"COMPLETE - Execution time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)
    
    return results


def _fit_real_data_grid(
    X: pd.DataFrame,
    models: List[str],
    n_components_range: List[int],
    optimize_metric: str,
    hyperparameter_grid: Optional[Dict],
    n_jobs: int,
    save_models: bool,
    output_dir: Path,
    verbose: bool
) -> Dict[str, Any]:
    """Grid search implementation for fit_on_real_data."""
    import numpy as np
    from joblib import Parallel, delayed
    from .models import ModelWrapper
    from itertools import product
    
    # Create hyperparameter grid manually
    if hyperparameter_grid is None:
        # Default grid for real data
        kappa_min = 1.0
        kappa_max = min(5.0, np.sqrt(X.shape[1]))  # Up to sqrt(n_features) or 5
        kappa_values = np.linspace(kappa_min, kappa_max, 3)  # 3 values
        
        jump_penalty_values = np.logspace(-1, 0.5, 3)  # [0.1, 0.56, 3.16]
    else:
        # Parse custom grid
        kappa_range = hyperparameter_grid.get('kappa_range', (1.0, 5.0, 3))
        jump_penalty_range = hyperparameter_grid.get('jump_penalty_range', (0.1, 2.0, 3))
        
        # kappa_range: (min, max, num_points)
        if len(kappa_range) == 3:
            kappa_min, kappa_max, num_points = kappa_range
            kappa_values = np.linspace(kappa_min, kappa_max, int(num_points))
        else:
            raise ValueError("kappa_range must be (min, max, num_points)")
        
        if len(jump_penalty_range) == 3:
            jp_min, jp_max, num_points = jump_penalty_range
            jump_penalty_values = np.linspace(jp_min, jp_max, int(num_points))
        else:
            raise ValueError("jump_penalty_range must be (min, max, num_points)")
    
    # Convert kappa to max_feats (κ²)
    max_feats_values = kappa_values ** 2
    
    # Create grid combinations
    hp_grid = [
        {'max_feats': mf, 'jump_penalty': jp}
        for mf, jp in product(max_feats_values, jump_penalty_values)
    ]
    
    if verbose:
        print(f"Hyperparameter combinations per state: {len(hp_grid)}")
        print(f"Total evaluations: {len(models) * len(n_components_range) * len(hp_grid)}")
        print("=" * 80)
    if verbose:
        print(f"Hyperparameter combinations per state: {len(hp_grid)}")
        print(f"Total evaluations: {len(models) * len(n_components_range) * len(hp_grid)}")
        print("=" * 80)
    
    # Function to fit a single model configuration
    def fit_single_config(model_name, n_components, params):
        try:
            wrapper = ModelWrapper(
                model_name=model_name,
                n_components=n_components,
                max_feats=params['max_feats'],
                jump_penalty=params['jump_penalty'],
                verbose=0
            )
            wrapper.fit(X)
            results = wrapper.evaluate_unsupervised(X, return_predictions=False)
            results['n_components'] = n_components
            return wrapper, results, True
        except Exception as e:
            # Return failure marker with error details
            if verbose:
                print(f"    ✗ Failed: {model_name} K={n_components} κ²={params['max_feats']:.1f} λ={params['jump_penalty']:.2f}: {str(e)[:50]}")
            return None, {
                'model_name': model_name,
                'n_components': n_components,
                'hyperparameters': params,
                'error': str(e),
                'bic': np.inf,
                'aic': np.inf,
                'silhouette': -1.0,
            }, False
    
    # Fit all model configurations
    all_results = []
    all_wrappers = []
    
    for model_name in models:
        if verbose:
            print(f"\nFitting {model_name} models...")
        
        for n_comp in n_components_range:
            # Parallel execution across hyperparameters
            if n_jobs != 1:
                results_list = Parallel(n_jobs=n_jobs)(
                    delayed(fit_single_config)(model_name, n_comp, params)
                    for params in hp_grid
                )
            else:
                # Sequential
                results_list = [
                    fit_single_config(model_name, n_comp, params)
                    for params in hp_grid
                ]
            
            # Collect results
            for wrapper, result, success in results_list:
                all_results.append(result)
                if success:
                    all_wrappers.append((model_name, n_comp, wrapper))
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Find best model for each model type
    best_models = {}
    is_minimize = optimize_metric in ['bic', 'aic']
    
    for model_name in models:
        model_results = results_df[results_df['model_name'] == model_name]
        
        if is_minimize:
            best_idx = model_results[optimize_metric].idxmin()
        else:
            best_idx = model_results[optimize_metric].idxmax()
        
        best_row = model_results.loc[best_idx]
        
        # Find corresponding wrapper
        best_wrapper = None
        for m_name, n_comp, wrapper in all_wrappers:
            if (m_name == model_name and 
                n_comp == best_row['n_components'] and
                wrapper.hyperparameters == best_row['hyperparameters']):
                best_wrapper = wrapper
                break
        
        best_models[model_name] = {
            'model': best_wrapper.model if best_wrapper else None,
            'wrapper': best_wrapper,
            'n_components': int(best_row['n_components']),
            'hyperparameters': best_row['hyperparameters'],
        }
        
        if verbose:
            print(f"\n  Best {model_name}:")
            print(f"    States: {best_row['n_components']}")
            print(f"    {optimize_metric.upper()}: {best_row[optimize_metric]:.2f}")
            print(f"    BIC: {best_row['bic']:.2f}, AIC: {best_row['aic']:.2f}, Silhouette: {best_row['silhouette']:.3f}")
    
    # Save results
    results_df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Save best models
    if save_models:
        models_dir = output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        import pickle
        for model_name, model_info in best_models.items():
            if model_info and model_info.get('wrapper') is not None:
                model_path = models_dir / f'best_{model_name}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['wrapper'].model, f)
    
    return {
        'results_df': results_df,
        'best_models': best_models,
        'output_dir': output_dir,
    }


def _fit_real_data_optuna(
    X: pd.DataFrame,
    models: List[str],
    n_components_range: List[int],
    optimize_metric: str,
    n_trials: int,
    hyperparameter_ranges: Optional[Dict],
    n_jobs: int,
    save_models: bool,
    output_dir: Path,
    verbose: bool
) -> Dict[str, Any]:
    """Optuna optimization implementation for fit_on_real_data."""
    import numpy as np
    import optuna
    from .models import ModelWrapper
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Parse hyperparameter ranges
    if hyperparameter_ranges is None:
        kappa_min = 1.0
        kappa_max = min(5.0, np.sqrt(X.shape[1]))
        jump_penalty_min = 0.1
        jump_penalty_max = 2.0
    else:
        kappa_min = hyperparameter_ranges.get('kappa_min', 1.0)
        kappa_max = hyperparameter_ranges.get('kappa_max', 5.0)
        jump_penalty_min = hyperparameter_ranges.get('jump_penalty_min', 0.1)
        jump_penalty_max = hyperparameter_ranges.get('jump_penalty_max', 2.0)
    
    # Determine optimization direction
    direction = 'minimize' if optimize_metric in ['bic', 'aic'] else 'maximize'
    
    all_results = []
    all_wrappers = []
    studies = {}
    
    for model_name in models:
        if verbose:
            print(f"\nOptimizing {model_name} models...")
        
        for n_comp in n_components_range:
            if verbose:
                print(f"  States={n_comp}: Running {n_trials} trials...", end='', flush=True)
            
            # Define objective function for this model/n_comp combination
            def objective(trial):
                # Suggest hyperparameters
                kappa = trial.suggest_float('kappa', kappa_min, kappa_max)
                jump_penalty = trial.suggest_float('jump_penalty', jump_penalty_min, jump_penalty_max, log=True)
                max_feats = kappa ** 2
                
                try:
                    wrapper = ModelWrapper(
                        model_name=model_name,
                        n_components=n_comp,
                        max_feats=max_feats,
                        jump_penalty=jump_penalty,
                        verbose=0
                    )
                    wrapper.fit(X)
                    results = wrapper.evaluate_unsupervised(X, return_predictions=False)
                    
                    # Store wrapper for later retrieval
                    trial.set_user_attr('wrapper', wrapper)
                    trial.set_user_attr('results', results)
                    
                    return results[optimize_metric]
                except Exception as e:
                    # Return worst possible value
                    if direction == 'minimize':
                        return float('inf')
                    else:
                        return -1.0
            
            # Create and run study
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
            
            # Store study
            study_key = f"{model_name}_K{n_comp}"
            studies[study_key] = study
            
            # Get best trial
            if study.best_trial is not None:
                best_trial = study.best_trial
                best_wrapper = best_trial.user_attrs.get('wrapper')
                best_results = best_trial.user_attrs.get('results')
                
                if best_results is not None:
                    best_results['n_components'] = n_comp
                    all_results.append(best_results)
                    if best_wrapper is not None:
                        all_wrappers.append((model_name, n_comp, best_wrapper))
                
                if verbose:
                    print(f" Best {optimize_metric}={study.best_value:.2f}")
            else:
                if verbose:
                    print(f" All trials failed")
                # Add failure marker
                all_results.append({
                    'model_name': model_name,
                    'n_components': n_comp,
                    'bic': np.inf,
                    'aic': np.inf,
                    'silhouette': -1.0,
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Find best model for each model type across all n_components
    best_models = {}
    is_minimize = optimize_metric in ['bic', 'aic']
    
    for model_name in models:
        model_results = results_df[results_df['model_name'] == model_name]
        
        if len(model_results) == 0:
            best_models[model_name] = None
            continue
        
        if is_minimize:
            best_idx = model_results[optimize_metric].idxmin()
        else:
            best_idx = model_results[optimize_metric].idxmax()
        
        best_row = model_results.loc[best_idx]
        
        # Find corresponding wrapper
        best_wrapper = None
        for m_name, n_comp, wrapper in all_wrappers:
            if (m_name == model_name and 
                n_comp == best_row['n_components'] and
                wrapper.hyperparameters == best_row['hyperparameters']):
                best_wrapper = wrapper
                break
        
        best_models[model_name] = {
            'model': best_wrapper.model if best_wrapper else None,
            'wrapper': best_wrapper,
            'n_components': int(best_row['n_components']),
            'hyperparameters': best_row['hyperparameters'],
        }
        
        if verbose:
            print(f"\n  Best {model_name} overall:")
            print(f"    States: {best_row['n_components']}")
            print(f"    {optimize_metric.upper()}: {best_row[optimize_metric]:.2f}")
            print(f"    BIC: {best_row['bic']:.2f}, AIC: {best_row['aic']:.2f}, Silhouette: {best_row['silhouette']:.3f}")
    
    # Save results
    results_df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Save best models
    if save_models:
        models_dir = output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        import pickle
        for model_name, model_info in best_models.items():
            if model_info and model_info.get('wrapper') is not None:
                model_path = models_dir / f'best_{model_name}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['wrapper'].model, f)
    
    return {
        'results_df': results_df,
        'best_models': best_models,
        'output_dir': output_dir,
        'studies': studies,  # Include Optuna studies for further analysis
    }
