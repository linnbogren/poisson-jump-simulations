"""
Hyperparameter Management for Sparse Jump Model Simulations

This module provides config-driven hyperparameter grid generation and
Optuna-based optimization for the simulation study.

ALL hyperparameter values come from configuration files - NO hardcoded values!

Key components:
- Grid search: Generate grids from HyperparameterGridConfig
- Optuna optimization: Bayesian search with config-defined ranges
- Data config grids: Generate from DataGridConfig
"""

import numpy as np
import optuna
from typing import List, Dict, Callable, Optional
from itertools import product
import logging

from .config import SimulationConfig, HyperparameterGridConfig, DataGridConfig

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


###############################################################################
# Grid Search Hyperparameters
###############################################################################

def create_hyperparameter_grid(
    grid_config: HyperparameterGridConfig,
    n_total_features: int
) -> List[Dict]:
    """
    Create hyperparameter grid from configuration.
    
    ALL values come from grid_config - no hardcoded parameters!
    
    Parameters
    ----------
    grid_config : HyperparameterGridConfig
        Configuration specifying grid parameters
    n_total_features : int
        Total number of features (P), used to calculate kappa_max if needed
    
    Returns
    -------
    List[Dict]
        List of hyperparameter dictionaries with n_components, jump_penalty, max_feats
        
    Examples
    --------
    >>> config = HyperparameterGridConfig(
    ...     n_states_values=[2, 3, 4],
    ...     jump_penalty_min=0.1,
    ...     jump_penalty_max=100.0,
    ...     jump_penalty_num=7
    ... )
    >>> grid = create_hyperparameter_grid(config, n_total_features=60)
    """
    if grid_config.quick_test:
        # Quick test: Use only min, center, max for each hyperparameter
        n_states_values = grid_config.n_states_values
        
        # Jump penalty: min, geometric center, max
        jump_penalty_values = [
            grid_config.jump_penalty_min,
            np.sqrt(grid_config.jump_penalty_min * grid_config.jump_penalty_max),  # geometric mean
            grid_config.jump_penalty_max
        ]
        
        # Kappa: min, arithmetic center, max
        kappa_max = _calculate_kappa_max(grid_config, n_total_features)
        kappa_values = [
            grid_config.kappa_min,
            (grid_config.kappa_min + kappa_max) / 2,
            kappa_max
        ]
    else:
        # Full grid from config
        n_states_values = grid_config.n_states_values
        
        # Jump penalty values
        if grid_config.jump_penalty_scale == "log":
            jump_penalty_values = np.logspace(
                np.log10(grid_config.jump_penalty_min),
                np.log10(grid_config.jump_penalty_max),
                grid_config.jump_penalty_num
            )
        else:  # linear
            jump_penalty_values = np.linspace(
                grid_config.jump_penalty_min,
                grid_config.jump_penalty_max,
                grid_config.jump_penalty_num
            )
        
        # Kappa values (always linear)
        kappa_max = _calculate_kappa_max(grid_config, n_total_features)
        kappa_values = np.linspace(
            grid_config.kappa_min,
            kappa_max,
            grid_config.kappa_num
        )
    
    # Convert kappa to max_feats (square of kappa)
    max_feats_values = np.array(kappa_values) ** 2
    
    # Create grid
    grid = []
    for n_states, gamma, max_feats in product(n_states_values, jump_penalty_values, max_feats_values):
        grid.append({
            'n_components': n_states,
            'jump_penalty': gamma,
            'max_feats': max_feats
        })
    
    return grid


def _calculate_kappa_max(grid_config: HyperparameterGridConfig, n_total_features: int) -> float:
    """
    Calculate maximum kappa value based on configuration.
    
    Parameters
    ----------
    grid_config : HyperparameterGridConfig
        Grid configuration
    n_total_features : int
        Total number of features (P)
    
    Returns
    -------
    float
        Maximum kappa value
    """
    if grid_config.kappa_max_type == "sqrt_P":
        return np.sqrt(n_total_features)
    elif grid_config.kappa_max_type == "P":
        return float(n_total_features)
    elif grid_config.kappa_max_type == "fixed":
        if grid_config.kappa_max_fixed is None:
            raise ValueError("kappa_max_fixed must be set when kappa_max_type='fixed'")
        return grid_config.kappa_max_fixed
    else:
        raise ValueError(f"Unknown kappa_max_type: {grid_config.kappa_max_type}")


###############################################################################
# Data Configuration Grids
###############################################################################

def create_data_config_grid(data_grid_config: DataGridConfig) -> List[SimulationConfig]:
    """
    Create grid of data generation configurations from config.
    
    ALL values come from data_grid_config - no hardcoded parameters!
    
    Parameters
    ----------
    data_grid_config : DataGridConfig
        Configuration specifying data generation parameter ranges
    
    Returns
    -------
    List[SimulationConfig]
        List of simulation configurations
    """
    if data_grid_config.quick_test:
        # Quick test: Use first, middle, last values
        n_total_features_values = _get_quick_values(data_grid_config.n_total_features_values)
        delta_values = _get_quick_values(data_grid_config.delta_values)
        distribution_types = [data_grid_config.distribution_types[0]] if data_grid_config.distribution_types else ["Poisson"]
        correlated_noise_values = [False]
    else:
        # Full grid from config
        n_total_features_values = data_grid_config.n_total_features_values
        delta_values = data_grid_config.delta_values
        distribution_types = data_grid_config.distribution_types
        correlated_noise_values = data_grid_config.correlated_noise_values
    
    configs = []
    config_id = 0
    
    for F, delta, dist_type, corr_noise in product(
        n_total_features_values, delta_values, distribution_types, correlated_noise_values
    ):
        # Only test correlated noise with specified distributions
        if corr_noise and dist_type not in data_grid_config.correlated_noise_distributions:
            continue
            
        config = SimulationConfig(
            n_samples=data_grid_config.n_samples,
            n_states=data_grid_config.n_states,
            n_informative=data_grid_config.n_informative,
            n_total_features=F,
            delta=delta,
            lambda_0=data_grid_config.lambda_0,
            persistence=data_grid_config.persistence,
            distribution_type=dist_type,
            correlated_noise=corr_noise,
            noise_correlation=data_grid_config.noise_correlation,
            nb_dispersion=data_grid_config.nb_dispersion,
            random_seed=data_grid_config.base_seed + config_id
        )
        configs.append(config)
        config_id += 1
    
    return configs


def _get_quick_values(values: List) -> List:
    """
    Get min, center, max values from a list for quick testing.
    
    Parameters
    ----------
    values : List
        List of values
    
    Returns
    -------
    List
        [min, center, max] or original list if <= 3 elements
    """
    if len(values) <= 3:
        return values
    return [values[0], values[len(values)//2], values[-1]]


###############################################################################
# Optuna Optimization
###############################################################################

def create_optuna_study(
    objective_fn: Callable,
    grid_config: HyperparameterGridConfig,
    n_total_features: int,
    n_trials: int = 100,
    seed: Optional[int] = None,
    direction: str = 'maximize',
    n_jobs: int = 1
) -> optuna.Study:
    """
    Create and run an Optuna study for hyperparameter optimization.
    
    Search space defined by grid_config - no hardcoded values!
    
    Parameters
    ----------
    objective_fn : Callable
        Objective function to optimize
    grid_config : HyperparameterGridConfig
        Configuration defining the search space
    n_total_features : int
        Total number of features (P)
    n_trials : int, default=100
        Number of optimization trials
    seed : int, optional
        Random seed
    direction : str, default='maximize'
        'maximize' or 'minimize'
    n_jobs : int, default=1
        Number of parallel jobs for trials. 1 for sequential, -1 for all cores.
        Setting >1 enables parallel trial evaluation for faster optimization.
    
    Returns
    -------
    optuna.Study
        Completed study with optimization results
        
    Notes
    -----
    Parallel trials (n_jobs > 1) can significantly speed up optimization when
    you have few replications. For example, with n_jobs=4, 4 trials run
    simultaneously, potentially giving ~4x speedup.
    """
    # Create study with TPE sampler
    sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else optuna.samplers.TPESampler()
    study = optuna.create_study(direction=direction, sampler=sampler)
    
    # Run optimization with parallel trials if requested
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)
    
    return study


def suggest_hyperparameters(
    trial: optuna.Trial,
    grid_config: HyperparameterGridConfig,
    n_total_features: int
) -> Dict:
    """
    Suggest hyperparameters for a single Optuna trial.
    
    Search space defined by grid_config - no hardcoded values!
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    grid_config : HyperparameterGridConfig
        Configuration defining the search space
    n_total_features : int
        Total number of features (P)
    
    Returns
    -------
    Dict
        Dictionary with n_components, jump_penalty, max_feats
    """
    # Number of states (discrete)
    n_components = trial.suggest_categorical('n_components', grid_config.n_states_values)
    
    # Jump penalty
    if grid_config.jump_penalty_scale == "log":
        jump_penalty = trial.suggest_float(
            'jump_penalty',
            grid_config.jump_penalty_min,
            grid_config.jump_penalty_max,
            log=True
        )
    else:
        jump_penalty = trial.suggest_float(
            'jump_penalty',
            grid_config.jump_penalty_min,
            grid_config.jump_penalty_max
        )
    
    # Feature selection parameter (sample kappa, then square)
    kappa_max = _calculate_kappa_max(grid_config, n_total_features)
    kappa = trial.suggest_float('kappa', grid_config.kappa_min, kappa_max)
    max_feats = kappa ** 2
    
    return {
        'n_components': int(n_components),
        'jump_penalty': float(jump_penalty),
        'max_feats': float(max_feats)
    }


def optuna_objective_wrapper(
    trial: optuna.Trial,
    evaluate_fn: Callable[[Dict], float],
    grid_config: HyperparameterGridConfig,
    n_total_features: int
) -> float:
    """
    Wrapper for Optuna objective function.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    evaluate_fn : Callable
        Function that takes hyperparameters and returns score
    grid_config : HyperparameterGridConfig
        Configuration defining search space
    n_total_features : int
        Total number of features
    
    Returns
    -------
    float
        Evaluation score (-1e6 if evaluation fails)
    """
    try:
        params = suggest_hyperparameters(trial, grid_config, n_total_features)
        score = evaluate_fn(params)
        
        if score is None or np.isnan(score) or np.isinf(score):
            return -1e6
        return float(score)
        
    except Exception as e:
        trial.set_user_attr('error', str(e))
        return -1e6


###############################################################################
# Utility Functions
###############################################################################

def get_search_space_info(
    grid_config: HyperparameterGridConfig,
    n_total_features: int
) -> Dict:
    """
    Get information about the hyperparameter search space.
    
    Parameters
    ----------
    grid_config : HyperparameterGridConfig
        Grid configuration
    n_total_features : int
        Total number of features
    
    Returns
    -------
    Dict
        Information about the search space
    """
    kappa_max = _calculate_kappa_max(grid_config, n_total_features)
    
    if grid_config.quick_test:
        n_jump = 3
        n_kappa = 3
    else:
        n_jump = grid_config.jump_penalty_num
        n_kappa = grid_config.kappa_num
    
    info = {
        'grid_size': len(grid_config.n_states_values) * n_jump * n_kappa,
        'n_states_values': grid_config.n_states_values,
        'n_jump_penalty_values': n_jump,
        'n_kappa_values': n_kappa,
        'kappa_range': (grid_config.kappa_min, kappa_max),
        'jump_penalty_range': (grid_config.jump_penalty_min, grid_config.jump_penalty_max),
        'jump_penalty_scale': grid_config.jump_penalty_scale
    }
    
    return info
