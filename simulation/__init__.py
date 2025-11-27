"""
Simplified simulation API for Sparse Jump Model experiments.

Quick Start
-----------
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
"""

# Legacy API (still available for advanced users)
from .config import (
    SimulationConfig,
    ReplicationResult,
    GridSearchResult,
    ModelConfig,
    HyperparameterGridConfig,
    DataGridConfig,
    ExperimentConfig,
    load_config,
    save_config,
    validate_config
)

# Import advanced runner function
from .runner import run_simulation as run_simulation_advanced

# Import ResultManager
from .results import ResultManager

# New simplified API (recommended - imported last to take precedence)
from .api import run_simulation, SimulationResults

from .data_generation import (
    generate_hmm_transition_matrix,
    compute_state_lambdas,
    sample_state_sequence,
    generate_poisson_hmm_data,
    generate_negative_binomial_hmm_data,
    generate_gaussian_hmm_data,
    generate_correlated_noise,
    generate_data
)

from .metrics import (
    compute_bac_best_permutation,
    compute_breakpoint_error,
    extract_breakpoints,
    compute_chamfer_distance,
    compute_breakpoint_f1,
    compute_composite_score,
    compute_feature_selection_metrics,
    get_selected_features
)

from .hyperparameters import (
    create_hyperparameter_grid,
    create_data_config_grid,
    create_optuna_study,
    suggest_hyperparameters,
    optuna_objective_wrapper,
    get_search_space_info
)

from .models import (
    ModelWrapper,
    fit_and_evaluate,
    results_to_grid_search_result,
    grid_results_to_dataframe
)

from .runner import (
    run_single_replication_grid,
    run_single_replication_optuna,
    select_best_models
)

from .results import (
    ResultManager,
    save_experiment_metadata,
    update_metadata_on_completion,
    list_experiments,
    save_results_compressed,
    load_results_compressed
)

__all__ = [
    # Simplified API (recommended)
    'run_simulation',
    'SimulationResults',
    
    # Configuration classes
    'SimulationConfig',
    'ExperimentConfig',
    'HyperparameterGridConfig',
    'DataGridConfig',
    'load_config',
    'save_config',
    'validate_config',
    
    # Legacy advanced API
    'run_simulation_advanced',  # Alias for runner.run_simulation
    'ResultManager',
    
    # Result classes
    'ReplicationResult',
    'GridSearchResult',
    'ModelConfig',
    # Data generation functions
    'generate_hmm_transition_matrix',
    'compute_state_lambdas',
    'sample_state_sequence',
    'generate_poisson_hmm_data',
    'generate_negative_binomial_hmm_data',
    'generate_gaussian_hmm_data',
    'generate_correlated_noise',
    'generate_data',
    # Metrics functions
    'compute_bac_best_permutation',
    'compute_breakpoint_error',
    'extract_breakpoints',
    'compute_chamfer_distance',
    'compute_breakpoint_f1',
    'compute_composite_score',
    'compute_feature_selection_metrics',
    'get_selected_features',
    # Hyperparameter functions
    'create_hyperparameter_grid',
    'create_data_config_grid',
    'create_optuna_study',
    'suggest_hyperparameters',
    'optuna_objective_wrapper',
    'get_search_space_info',
    # Model classes and functions
    'ModelWrapper',
    'fit_and_evaluate',
    'results_to_grid_search_result',
    'grid_results_to_dataframe',
    # Runner functions
    'run_single_replication_grid',
    'run_single_replication_optuna',
    'select_best_models',
    # Result management
    'ResultManager',
    'save_experiment_metadata',
    'update_metadata_on_completion',
    'list_experiments',
    'save_results_compressed',
    'load_results_compressed',
]
