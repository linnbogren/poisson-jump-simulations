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
