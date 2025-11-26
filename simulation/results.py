"""
Result Storage and Retrieval Utilities

This module provides efficient loading, filtering, and aggregation of simulation results.

Key components:
- ResultManager: Query and load results efficiently
- Metadata tracking: Save/load experiment metadata
- Experiment listing: Browse all experiments
"""

import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Tuple
from datetime import datetime
import sys


class ResultManager:
    """
    Manage simulation results with efficient loading and querying.
    
    Provides methods to load and filter results without loading entire datasets
    into memory, along with utilities for aggregation and analysis.
    
    Examples
    --------
    >>> rm = ResultManager("results/my_experiment_2025-11-26")
    >>> summary = rm.get_summary()
    >>> best_results = rm.load_best_results()
    >>> top_configs = rm.load_grid_results(
    ...     metric_filter={'balanced_accuracy': ('>', 0.8)}
    ... )
    """
    
    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize result manager.
        
        Parameters
        ----------
        results_dir : str or Path
            Path to results directory (contains aggregated/, grid_search/, etc.)
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        self._metadata = self._load_metadata()
        self._data_configs = self._load_data_configs()
    
    def _load_metadata(self) -> Dict:
        """Load experiment metadata if available."""
        metadata_path = self.results_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_data_configs(self) -> Optional[pd.DataFrame]:
        """Load data configuration details if available."""
        config_path = self.results_dir / "data_configs.csv"
        if config_path.exists():
            return pd.read_csv(config_path)
        return None
    
    def load_best_results(self) -> pd.DataFrame:
        """
        Load best model results (fastest to load).
        
        Returns
        -------
        pd.DataFrame
            Best model results for each configuration and model
        """
        best_path = self.results_dir / "aggregated" / "all_results.csv"
        if not best_path.exists():
            raise FileNotFoundError(f"Best results not found: {best_path}")
        return pd.read_csv(best_path)
    
    def load_grid_results(self, 
                         models: Optional[List[str]] = None,
                         configs: Optional[List[int]] = None,
                         metric_filter: Optional[Dict[str, Tuple[str, float]]] = None,
                         nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load grid search results with optional filtering.
        
        Parameters
        ----------
        models : list of str, optional
            Filter by model names
        configs : list of int, optional
            Filter by config IDs
        metric_filter : dict, optional
            Filter by metrics, e.g. {'balanced_accuracy': ('>', 0.8)}
            Supported operators: '>', '<', '>=', '<=', '=='
        nrows : int, optional
            Load only first N rows (for quick inspection)
            
        Returns
        -------
        pd.DataFrame
            Filtered grid search results
            
        Examples
        --------
        >>> # Load only successful runs
        >>> good_results = rm.load_grid_results(
        ...     metric_filter={'balanced_accuracy': ('>', 0.8)}
        ... )
        >>> 
        >>> # Load results for specific models
        >>> poisson_results = rm.load_grid_results(
        ...     models=['SparseJumpPoisson']
        ... )
        """
        grid_path = self.results_dir / "grid_search" / "all_grid_results.csv"
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid results not found: {grid_path}")
        
        # Load data (potentially large file)
        df = pd.read_csv(grid_path, nrows=nrows)
        
        # Apply filters
        if models:
            df = df[df['model_name'].isin(models)]
        if configs:
            df = df[df['config_id'].isin(configs)]
        if metric_filter:
            for metric, (op, threshold) in metric_filter.items():
                if metric not in df.columns:
                    raise ValueError(f"Metric '{metric}' not found in results")
                
                if op == '>':
                    df = df[df[metric] > threshold]
                elif op == '<':
                    df = df[df[metric] < threshold]
                elif op == '>=':
                    df = df[df[metric] >= threshold]
                elif op == '<=':
                    df = df[df[metric] <= threshold]
                elif op == '==':
                    df = df[df[metric] == threshold]
                else:
                    raise ValueError(f"Unsupported operator: {op}")
        
        return df
    
    def load_model(self, config_id: int, model_name: str, replication: int = 0):
        """
        Load a fitted model from disk.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        model_name : str
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
        model_path = self.results_dir / "models" / f"{config_id}_{model_name}_{replication}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the experiment.
        
        Returns
        -------
        dict
            Summary information including experiment details and mean metrics
            
        Examples
        --------
        >>> summary = rm.get_summary()
        >>> print(f"Experiment: {summary['experiment_name']}")
        >>> print(f"Mean BAC: {summary['mean_bac']:.3f}")
        """
        best_df = self.load_best_results()
        
        # Try to load grid results for total evaluations count
        try:
            grid_path = self.results_dir / "grid_search" / "all_grid_results.csv"
            if grid_path.exists():
                # Use pandas to count rows without loading entire file
                n_total_evals = sum(1 for _ in open(grid_path)) - 1  # -1 for header
            else:
                n_total_evals = None
        except Exception:
            n_total_evals = None
        
        # Calculate summary statistics
        summary = {
            'experiment_name': self._metadata.get('name', 'unknown'),
            'timestamp': self._metadata.get('timestamp', 'unknown'),
            'config_file': self._metadata.get('config_file', 'unknown'),
            'optimization_method': self._metadata.get('optimization_method', 'unknown'),
            'n_configurations': best_df['config_id'].nunique() if 'config_id' in best_df.columns else 0,
            'n_models': best_df['model_name'].nunique() if 'model_name' in best_df.columns else 0,
            'n_replications': self._metadata.get('n_replications', 'unknown'),
            'n_total_evaluations': n_total_evals,
            'models': best_df['model_name'].unique().tolist() if 'model_name' in best_df.columns else [],
            'parallel': self._metadata.get('parallel', 'unknown'),
            'n_workers': self._metadata.get('n_workers', 'unknown'),
        }
        
        # Add metric averages
        metric_cols = [col for col in best_df.columns if col not in 
                      ['config_id', 'model_name', 'replication', 'fit_time', 
                       'n_components', 'jump_penalty', 'max_feats']]
        
        for metric in metric_cols:
            try:
                summary[f'mean_{metric}'] = float(best_df[metric].mean())
                summary[f'std_{metric}'] = float(best_df[metric].std())
            except Exception:
                continue
        
        return summary
    
    def aggregate_by_config(self, metric: str = 'balanced_accuracy') -> pd.DataFrame:
        """
        Aggregate results by data configuration.
        
        Returns mean ± std for metrics across replications.
        
        Parameters
        ----------
        metric : str, default='balanced_accuracy'
            Primary metric to aggregate
            
        Returns
        -------
        pd.DataFrame
            Aggregated results with mean, std, min, max for each metric
            
        Examples
        --------
        >>> agg_df = rm.aggregate_by_config('composite_score')
        >>> print(agg_df[['config_id', 'model_name', 'composite_score_mean']])
        """
        best_df = self.load_best_results()
        
        # Identify metric columns
        metric_cols = [col for col in best_df.columns 
                      if col not in ['config_id', 'model_name', 'replication', 
                                   'n_components', 'jump_penalty', 'max_feats']]
        
        # Build aggregation dictionary
        agg_funcs = {}
        for col in metric_cols:
            if col == 'fit_time':
                agg_funcs[col] = ['mean', 'std']
            else:
                agg_funcs[col] = ['mean', 'std', 'min', 'max']
        
        # Aggregate
        agg_df = best_df.groupby(['config_id', 'model_name']).agg(agg_funcs).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in agg_df.columns.values]
        
        return agg_df
    
    def find_best_hyperparameters(self, config_id: int, model_name: str, 
                                 metric: str = 'balanced_accuracy',
                                 top_k: int = 10) -> pd.DataFrame:
        """
        Find best hyperparameters for a specific configuration.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        model_name : str
            Model name
        metric : str, default='balanced_accuracy'
            Metric to optimize
        top_k : int, default=10
            Number of top results to return
            
        Returns
        -------
        pd.DataFrame
            Top hyperparameter configurations sorted by metric
            
        Examples
        --------
        >>> best_params = rm.find_best_hyperparameters(
        ...     config_id=5, 
        ...     model_name='SparseJumpPoisson',
        ...     metric='composite_score'
        ... )
        """
        grid_df = self.load_grid_results(models=[model_name], configs=[config_id])
        if len(grid_df) == 0:
            raise ValueError(f"No results found for config_id={config_id}, model={model_name}")
        
        return grid_df.nlargest(top_k, metric)
    
    def get_config_details(self, config_id: int) -> Dict[str, Any]:
        """
        Get data generation parameters for a specific configuration.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
            
        Returns
        -------
        dict
            Configuration parameters
            
        Raises
        ------
        ValueError
            If config_id not found
        """
        if self._data_configs is None:
            raise FileNotFoundError("data_configs.csv not found")
        
        config_row = self._data_configs[self._data_configs['config_id'] == config_id]
        if len(config_row) == 0:
            raise ValueError(f"config_id {config_id} not found")
        
        return config_row.iloc[0].to_dict()
    
    def compare_models(self, metric: str = 'balanced_accuracy') -> pd.DataFrame:
        """
        Compare model performance across all configurations.
        
        Parameters
        ----------
        metric : str, default='balanced_accuracy'
            Metric to compare
            
        Returns
        -------
        pd.DataFrame
            Model comparison with mean and std for metric
        """
        best_df = self.load_best_results()
        
        comparison = best_df.groupby('model_name')[metric].agg(['mean', 'std', 'min', 'max']).reset_index()
        comparison = comparison.sort_values('mean', ascending=False)
        
        return comparison


def save_experiment_metadata(
    output_dir: Union[str, Path], 
    experiment_config,
    data_configs: List,
    start_time: Optional[datetime] = None,
    n_workers: Optional[int] = None
) -> None:
    """
    Save experiment metadata for easier retrieval and reproducibility.
    
    Parameters
    ----------
    output_dir : str or Path
        Output directory
    experiment_config : ExperimentConfig
        Experiment configuration
    data_configs : list of SimulationConfig
        Data configurations
    start_time : datetime, optional
        Experiment start time
    n_workers : int, optional
        Number of parallel workers used
        
    Examples
    --------
    >>> save_experiment_metadata(
    ...     "results/my_experiment",
    ...     experiment_config,
    ...     data_configs,
    ...     start_time=datetime.now()
    ... )
    """
    output_path = Path(output_dir)
    
    # Build metadata dictionary
    metadata = {
        'name': experiment_config.name,
        'timestamp': start_time.isoformat() if start_time else datetime.now().isoformat(),
        'optimization_method': experiment_config.optimization_method,
        'n_replications': experiment_config.n_replications,
        'n_configurations': len(data_configs),
        'models': experiment_config.model_names,
        'parallel': not experiment_config.single_thread,
        'n_workers': n_workers,
    }
    
    # Add config file path if available
    if hasattr(experiment_config, 'config_file') and experiment_config.config_file:
        metadata['config_file'] = str(experiment_config.config_file)
    
    # Add hyperparameter grid info if grid search
    if experiment_config.optimization_method == "grid" and experiment_config.hyperparameter_grid:
        grid = experiment_config.hyperparameter_grid
        metadata['hyperparameter_grid'] = {
            'n_states_values': grid.n_states_values,
            'jump_penalty_min': grid.jump_penalty_min,
            'jump_penalty_max': grid.jump_penalty_max,
            'jump_penalty_num': grid.jump_penalty_num,
            'jump_penalty_scale': grid.jump_penalty_scale,
            'kappa_min': grid.kappa_min,
            'kappa_max_type': grid.kappa_max_type,
            'kappa_num': grid.kappa_num,
            'quick_test': grid.quick_test
        }
    
    # Add optuna info if applicable
    if experiment_config.optimization_method == "optuna" and experiment_config.optuna_config:
        metadata['optuna'] = {
            'n_trials': experiment_config.optuna_config.n_trials,
            'direction': experiment_config.optuna_config.direction,
            'metric': experiment_config.optuna_config.metric
        }
    
    # Add Python environment info
    metadata['environment'] = {
        'python_version': sys.version.split()[0],
    }
    
    # Try to get package versions
    try:
        import numpy
        metadata['environment']['numpy_version'] = numpy.__version__
    except Exception:
        pass
    
    try:
        import pandas
        metadata['environment']['pandas_version'] = pandas.__version__
    except Exception:
        pass
    
    # Save metadata
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save data config details
    config_details = []
    for i, config in enumerate(data_configs):
        config_details.append({
            'config_id': i,
            'n_samples': config.n_samples,
            'n_states': config.n_states,
            'n_total_features': config.n_total_features,
            'n_informative': config.n_informative,
            'delta': config.delta,
            'lambda_0': config.lambda_0,
            'distribution_type': config.distribution_type,
            'correlated_noise': config.correlated_noise,
            'noise_correlation': config.noise_correlation if config.correlated_noise else None,
            'nb_dispersion': config.nb_dispersion if config.distribution_type == 'NegativeBinomial' else None,
            'random_seed': config.random_seed
        })
    
    pd.DataFrame(config_details).to_csv(output_path / "data_configs.csv", index=False)


def update_metadata_on_completion(
    output_dir: Union[str, Path],
    end_time: datetime,
    execution_time_seconds: float
) -> None:
    """
    Update metadata with completion information.
    
    Parameters
    ----------
    output_dir : str or Path
        Output directory
    end_time : datetime
        Experiment end time
    execution_time_seconds : float
        Total execution time in seconds
    """
    output_path = Path(output_dir)
    metadata_path = output_path / "metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['completed_at'] = end_time.isoformat()
        metadata['execution_time_seconds'] = execution_time_seconds
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def list_experiments(results_base_dir: Union[str, Path] = "results") -> pd.DataFrame:
    """
    List all experiments in the results directory.
    
    Parameters
    ----------
    results_base_dir : str or Path, default="results"
        Base results directory
        
    Returns
    -------
    pd.DataFrame
        DataFrame with experiment metadata for all experiments
        
    Examples
    --------
    >>> experiments = list_experiments()
    >>> print(experiments[['name', 'timestamp', 'n_replications']])
    >>> 
    >>> # Filter to recent experiments
    >>> recent = experiments[experiments['timestamp'] > '2025-11-20']
    """
    results_path = Path(results_base_dir)
    
    if not results_path.exists():
        return pd.DataFrame()
    
    experiments = []
    
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            metadata_file = exp_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata['directory'] = str(exp_dir)
                    experiments.append(metadata)
                except Exception as e:
                    # Skip corrupted metadata files
                    print(f"Warning: Could not load metadata from {exp_dir}: {e}")
                    continue
    
    if not experiments:
        return pd.DataFrame()
    
    return pd.DataFrame(experiments)


def save_results_compressed(df: pd.DataFrame, filepath: Union[str, Path],
                           format: str = 'csv.gz') -> None:
    """
    Save results in compressed format for large datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    filepath : str or Path
        Output file path (extension will be added if needed)
    format : str, default='csv.gz'
        Format: 'csv.gz' (gzip compressed CSV) or 'parquet'
        
    Examples
    --------
    >>> save_results_compressed(grid_df, "all_grid_results", format='csv.gz')
    >>> save_results_compressed(grid_df, "all_grid_results", format='parquet')
    """
    filepath = Path(filepath)
    
    if format == 'csv.gz':
        if not str(filepath).endswith('.csv.gz'):
            filepath = filepath.with_suffix('.csv.gz')
        df.to_csv(filepath, compression='gzip', index=False)
    elif format == 'parquet':
        if not str(filepath).endswith('.parquet'):
            filepath = filepath.with_suffix('.parquet')
        df.to_parquet(filepath, index=False, compression='snappy')
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv.gz' or 'parquet'")


def load_results_compressed(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load compressed results.
    
    Parameters
    ----------
    filepath : str or Path
        Path to compressed file (.csv.gz or .parquet)
        
    Returns
    -------
    pd.DataFrame
        Loaded results
        
    Examples
    --------
    >>> df = load_results_compressed("all_grid_results.csv.gz")
    >>> df = load_results_compressed("all_grid_results.parquet")
    """
    filepath = Path(filepath)
    
    if str(filepath).endswith('.csv.gz'):
        return pd.read_csv(filepath, compression='gzip')
    elif str(filepath).endswith('.parquet'):
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
