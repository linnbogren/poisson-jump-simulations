"""
Configuration Management for Poisson Jump Simulations

This module handles:
- Configuration dataclasses
- YAML config file loading and validation
- CLI argument parsing and merging with config files
"""

import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any, Union


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    n_samples: int = 500
    n_states: int = 3
    n_informative: int = 15
    n_noise: int = 0  # Derived from n_total_features
    n_total_features: int = 15
    delta: float = 0.2
    lambda_0: float = 10.0
    persistence: float = 0.97
    distribution_type: str = "Poisson"  # "Poisson", "NegativeBinomial", "Gaussian"
    correlated_noise: bool = False
    noise_correlation: float = 0.1
    nb_dispersion: float = 2.0  # For Negative Binomial
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Calculate derived parameters and validate."""
        self.n_noise = self.n_total_features - self.n_informative
        assert self.n_noise >= 0, "n_total_features must be >= n_informative"
        assert 0 <= self.delta , "delta must be greater than 0"
        assert self.n_states >= 2, "Must have at least 2 states"
        assert self.distribution_type.lower() in ["poisson", "negativebinomial", "gaussian"], \
            f"Invalid distribution_type: {self.distribution_type}"


@dataclass
class ReplicationResult:
    """Results from a single replication (best model only)."""
    config: SimulationConfig
    model_name: str
    balanced_accuracy: float
    accuracy: float
    n_jumps_true: int
    n_jumps_estimated: int
    feature_f1: float
    feature_precision: float
    feature_recall: float
    n_selected_noise: int
    poisson_deviance: float
    computation_time: float
    selected_features: List[int]
    true_states: np.ndarray
    predicted_states: np.ndarray
    best_hyperparams: Dict  # Hyperparameters of the best model
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        d = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        d['true_states'] = self.true_states.tolist() if hasattr(self.true_states, 'tolist') else self.true_states
        d['predicted_states'] = self.predicted_states.tolist() if hasattr(self.predicted_states, 'tolist') else self.predicted_states
        return d


@dataclass
class GridSearchResult:
    """Results from all models in a grid search."""
    config: SimulationConfig
    model_name: str
    hyperparameters: Dict  # The hyperparameters for this model
    balanced_accuracy: float
    accuracy: float
    n_jumps_true: int
    n_jumps_estimated: int
    breakpoint_error: int
    n_breakpoints_true: int  # Number of true breakpoints
    n_breakpoints_estimated: int  # Number of estimated breakpoints
    breakpoint_count_error: int  # Absolute difference in breakpoint counts
    chamfer_distance: float  # Chamfer distance between true and estimated breakpoints
    composite_score: float  # Composite score combining BAC and breakpoint F1
    breakpoint_f1: float  # Breakpoint detection F1 score
    breakpoint_precision: float  # Breakpoint detection precision
    breakpoint_recall: float  # Breakpoint detection recall
    feature_f1: float
    feature_precision: float
    feature_recall: float
    n_selected_noise: int
    n_selected_total: int
    computation_time: float
    # Unsupervised metrics (for real data without ground truth)
    bic: Optional[float] = None  # Bayesian Information Criterion (lower is better)
    aic: Optional[float] = None  # Akaike Information Criterion (lower is better)
    silhouette: Optional[float] = None  # Mean silhouette coefficient (higher is better)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    type: str  # "SparseJumpModel", "JumpModel"
    distance_metric: str = "poisson"  # "poisson", "gaussian", "kullback_leibler"


@dataclass
class HyperparameterGridConfig:
    """Configuration for hyperparameter grid generation."""
    # Number of states
    n_states_values: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # Jump penalty (lambda)
    jump_penalty_min: float = 0.1
    jump_penalty_max: float = 100.0
    jump_penalty_num: int = 7  # Number of log-spaced values
    jump_penalty_scale: str = "log"  # "log" or "linear"
    
    # Feature selection (kappa -> max_feats)
    kappa_min: float = 1.0
    kappa_max_type: str = "sqrt_P"  # "sqrt_P", "P", or "fixed"
    kappa_max_fixed: Optional[float] = None  # Only used if kappa_max_type="fixed"
    kappa_num: int = 14  # Number of linearly spaced kappa values
    
    # Quick test mode (overrides above with min/center/max)
    quick_test: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DataGridConfig:
    """Configuration for data generation grid."""
    # Feature counts
    n_total_features_values: List[int] = field(default_factory=lambda: [15, 30, 60, 150, 300])
    
    # Effect sizes
    delta_values: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.5])
    
    # Distribution types
    distribution_types: List[str] = field(default_factory=lambda: ["Poisson", "NegativeBinomial", "Gaussian"])
    
    # Correlated noise
    correlated_noise_values: List[bool] = field(default_factory=lambda: [False, True])
    
    # Only test correlated noise with certain distributions
    correlated_noise_distributions: List[str] = field(default_factory=lambda: ["Poisson"])
    
    # Fixed parameters (same across all grid points)
    n_samples: int = 500
    n_states: int = 3
    n_informative: int = 15
    lambda_0: float = 10.0
    persistence: float = 0.97
    noise_correlation: float = 0.1
    nb_dispersion: float = 2.0
    base_seed: int = 42
    
    # Quick test mode (overrides above with min/center/max)
    quick_test: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Main experiment configuration loaded from YAML."""
    # Experiment metadata
    name: str
    mode: str  # "single" or "grid"
    n_replications: int = 100
    parallel: bool = True
    single_thread: bool = False  # Force sequential execution even if parallel=True
    n_workers: Optional[int] = None  # None = use all CPUs
    random_seed: Optional[int] = None
    output_dir: str = "results"
    
    # Data configuration (for single mode)
    data: Optional[SimulationConfig] = None
    
    # Data grid configuration (for grid mode)
    data_grid: Optional[DataGridConfig] = None
    
    # Model names to run
    model_names: List[str] = field(default_factory=lambda: ["Gaussian", "Poisson", "PoissonKL"])
    
    # Hyperparameter optimization
    optimization_method: str = "grid"  # "grid" or "optuna"
    optimize_metric: str = "balanced_accuracy"  # Metric to optimize
    hyperparameter_grid: Optional[HyperparameterGridConfig] = None
    optuna_n_trials: int = 20  # For Optuna optimization
    optuna_n_jobs: int = -1  # Parallel trials for Optuna (1=sequential, -1=all cores)
    grid_n_jobs: int = -1  # Parallel hyperparameter search for Grid (1=sequential, -1=all cores)
    
    # Quick test mode (applies to both hyperparameter and data grids)
    quick_test: bool = False
    
    # Visualization settings
    visualization: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.mode in ["single", "grid"], f"Mode must be 'single' or 'grid', got: {self.mode}"
        assert self.optimization_method in ["grid", "optuna"], \
            f"optimization_method must be 'grid' or 'optuna', got: {self.optimization_method}"
        
        if self.mode == "single":
            assert self.data is not None, "data config required for single mode"
        elif self.mode == "grid":
            assert self.data_grid is not None, "data_grid config required for grid mode"
        
        # Create default hyperparameter grid config if not provided
        if self.hyperparameter_grid is None:
            self.hyperparameter_grid = HyperparameterGridConfig(quick_test=self.quick_test)
        
        # Create default data grid config if not provided and in grid mode
        if self.mode == "grid" and self.data_grid is None:
            self.data_grid = DataGridConfig(quick_test=self.quick_test)
        
        # Convert dicts to config objects if needed
        if isinstance(self.hyperparameter_grid, dict):
            self.hyperparameter_grid = HyperparameterGridConfig(**self.hyperparameter_grid)
        
        if isinstance(self.data_grid, dict):
            self.data_grid = DataGridConfig(**self.data_grid)


def load_config(yaml_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.
    
    Parameters:
    -----------
    yaml_path : str or Path
        Path to YAML configuration file
        
    Returns:
    --------
    ExperimentConfig
        Loaded and validated configuration
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract sections
    exp_config = config_dict.get('experiment', {})
    data_config = config_dict.get('data', None)
    data_grid_config = config_dict.get('data_grid', None)
    hyperparam_grid_config = config_dict.get('hyperparameter_grid', None)
    viz_config = config_dict.get('visualization', {})
    
    # Convert data config to SimulationConfig
    if data_config:
        data_obj = SimulationConfig(**data_config)
    else:
        data_obj = None
    
    # Convert data grid config
    if data_grid_config:
        data_grid_obj = DataGridConfig(**data_grid_config)
    else:
        data_grid_obj = None
    
    # Convert hyperparameter grid config
    if hyperparam_grid_config:
        hyperparam_grid_obj = HyperparameterGridConfig(**hyperparam_grid_config)
    else:
        hyperparam_grid_obj = None
    
    # Create experiment config
    exp = ExperimentConfig(
        name=exp_config['name'],
        mode=exp_config.get('mode', 'single'),
        n_replications=exp_config.get('n_replications', 100),
        parallel=exp_config.get('parallel', True),
        single_thread=exp_config.get('single_thread', False),
        n_workers=exp_config.get('n_workers', None),
        random_seed=exp_config.get('random_seed', None),
        output_dir=exp_config.get('output_dir', 'results'),
        data=data_obj,
        data_grid=data_grid_obj,
        model_names=exp_config.get('model_names', ["Gaussian", "Poisson", "PoissonKL"]),
        optimization_method=exp_config.get('optimization_method', 'grid'),
        optimize_metric=exp_config.get('optimize_metric', 'balanced_accuracy'),
        hyperparameter_grid=hyperparam_grid_obj,
        optuna_n_trials=exp_config.get('optuna_n_trials', 20),
        optuna_n_jobs=exp_config.get('optuna_n_jobs', 1),
        grid_n_jobs=exp_config.get('grid_n_jobs', 1),
        quick_test=exp_config.get('quick_test', False),
        visualization=viz_config
    )
    
    return exp


def save_config(config: ExperimentConfig, output_path: Union[str, Path]) -> None:
    """
    Save experiment configuration to YAML file.
    
    Parameters:
    -----------
    config : ExperimentConfig
        Configuration to save
    output_path : str or Path
        Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    config_dict = asdict(config)
    
    # Write YAML
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Parameters:
    -----------
    config : ExperimentConfig
        Configuration to validate
        
    Returns:
    --------
    List[str]
        List of validation messages (empty if valid)
    """
    messages = []
    
    # Check data configuration
    if config.mode == "single" and config.data is None:
        messages.append("ERROR: Single mode requires 'data' configuration")
    
    if config.mode == "grid" and config.data_grid is None:
        messages.append("ERROR: Grid mode requires 'data_grid' configuration")
    
    # Check models
    if not config.model_names:
        messages.append("WARNING: No model_names specified")
    
    # Check hyperparameters
    if config.hyperparameter_grid is None:
        messages.append("WARNING: No hyperparameter_grid specified, using defaults")
    
    # Check visualization
    if config.visualization.get('enabled', False):
        plot_types = config.visualization.get('plot_types', [])
        if not plot_types:
            messages.append("WARNING: Visualization enabled but no plot_types specified")
    
    return messages


###############################################################################
# Simplified API Helper Functions
###############################################################################

def apply_defaults(config: Dict) -> Dict:
    """
    Apply default values to simplified config dict.
    
    Parameters
    ----------
    config : dict
        User-provided configuration
        
    Returns
    -------
    dict
        Configuration with defaults applied
        
    Default Values
    --------------
    - models: ['Gaussian', 'Poisson', 'PoissonKL']
    - n_replications: 10
    - n_jobs: -1 (all cores)
    - optimization: 'grid'
    - optimize_metric: 'balanced_accuracy'
    - quick_test: False
    - optuna_n_trials: 20 (for Optuna optimization)
    - hyperparameter_grid: HyperparameterGridConfig() (with defaults)
    
    For data_configs:
    - n_samples: 200
    - n_states: 3
    - n_informative: 15
    - n_total_features: 15
    - lambda_0: 10.0
    - persistence: 0.97
    - distribution_type: 'Poisson'
    - correlated_noise: False
    - random_seed: 42
    """
    from copy import deepcopy
    
    config = deepcopy(config)
    
    # Top-level defaults
    config.setdefault('models', ['Gaussian', 'Poisson', 'PoissonKL'])
    config.setdefault('n_replications', 2)
    config.setdefault('n_jobs', -1)
    config.setdefault('optimization', 'grid')
    config.setdefault('optimize_metric', 'composite_score')
    config.setdefault('quick_test', True)  # Use small grid by default
    config.setdefault('optuna_n_trials', 20)  # Reasonable default for Optuna
    config.setdefault('optuna_n_jobs', -1)  # Use all cores for parallel trials
    config.setdefault('grid_n_jobs', -1)  # Use all cores for parallel hyperparameter search
    
    # Data config defaults
    data_defaults = {
        'n_samples': 200,
        'n_states': 3,
        'n_informative': 15,
        'n_total_features': 15,
        'lambda_0': 10.0,
        'persistence': 0.97,
        'distribution_type': 'Poisson',
        'correlated_noise': False,
        'random_seed': 42,
    }
    
    for i, data_cfg in enumerate(config.get('data_configs', [])):
        for key, default_val in data_defaults.items():
            data_cfg.setdefault(key, default_val)
    
    # Hyperparameter grid defaults
    if 'hyperparameter_grid' not in config:
        config['hyperparameter_grid'] = {}
    
    return config


def dict_to_experiment_config(config: Dict) -> tuple[ExperimentConfig, list]:
    """
    Convert simplified dict config to ExperimentConfig.
    
    Parameters
    ----------
    config : dict
        Simplified configuration with keys:
        - name: str (required)
        - data_configs: List[Dict] (required)
        - models: List[str]
        - n_replications: int
        - hyperparameter_grid: Dict
        - n_jobs: int
        - optimization: str
        - optimize_metric: str
        
    Returns
    -------
    tuple[ExperimentConfig, list[SimulationConfig]]
        Fully configured experiment object and list of simulation configs
    """
    # Apply defaults
    config = apply_defaults(config)
    
    # Validate required fields
    if 'name' not in config:
        raise ValueError("config must include 'name'")
    if 'data_configs' not in config or not config['data_configs']:
        raise ValueError("config must include non-empty 'data_configs'")
    
    # Create simulation configs
    sim_configs = [SimulationConfig(**dc) for dc in config['data_configs']]
    
    # Create hyperparameter grid config
    hyperparam_cfg = HyperparameterGridConfig(
        **config.get('hyperparameter_grid', {}),
        quick_test=config.get('quick_test', False)
    )
    
    # Determine mode
    mode = 'grid' if len(sim_configs) > 1 else 'single'
    
    # For grid mode, create a minimal data_grid config (won't be used, but required by validation)
    data_grid_cfg = None
    if mode == 'grid':
        data_grid_cfg = DataGridConfig(quick_test=config.get('quick_test', False))
    
    # Create experiment config
    exp_config = ExperimentConfig(
        name=config['name'],
        mode=mode,
        n_replications=config['n_replications'],
        parallel=(config['n_jobs'] != 1),
        single_thread=(config['n_jobs'] == 1),
        n_workers=None if config['n_jobs'] == -1 else abs(config['n_jobs']),
        model_names=config['models'],
        optimization_method=config['optimization'],
        optimize_metric=config['optimize_metric'],
        hyperparameter_grid=hyperparam_cfg,
        optuna_n_trials=config.get('optuna_n_trials', 20),
        optuna_n_jobs=config.get('optuna_n_jobs', -1),
        grid_n_jobs=config.get('grid_n_jobs', -1),
        quick_test=config.get('quick_test', False),
        data=sim_configs[0] if mode == 'single' else None,
        data_grid=data_grid_cfg,
        output_dir='results',
    )
    
    return exp_config, sim_configs


def normalize_config_for_hashing(config: Dict) -> Dict:
    """
    Normalize config for deterministic hashing.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Normalized configuration
        
    Normalization Steps
    -------------------
    - Sort all lists and dict keys
    - Remove None values
    - Round floats to 6 decimal places
    - Convert all strings to lowercase
    - Remove fields that don't affect results (n_jobs, output paths, etc.)
    """
    from copy import deepcopy
    import json
    
    config = deepcopy(config)
    
    # Fields to exclude from hash (don't affect results)
    exclude_fields = {'n_jobs', 'output_dir', 'verbose'}
    
    def normalize_value(val):
        if val is None:
            return None
        elif isinstance(val, float):
            return round(val, 6)
        elif isinstance(val, str):
            return val.lower()
        elif isinstance(val, list):
            return sorted([normalize_value(v) for v in val], key=str)
        elif isinstance(val, dict):
            return {k: normalize_value(v) for k, v in sorted(val.items())
                    if k not in exclude_fields and v is not None}
        else:
            return val
    
    normalized = normalize_value(config)
    
    # Convert to JSON string for consistent representation
    return json.loads(json.dumps(normalized, sort_keys=True))
