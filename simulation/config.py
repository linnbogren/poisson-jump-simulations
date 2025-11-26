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
    optuna_n_trials: int = 100  # For Optuna optimization
    
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
        optuna_n_trials=exp_config.get('optuna_n_trials', 100),
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
