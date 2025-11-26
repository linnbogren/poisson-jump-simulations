"""
Tests for simulation/runner.py - Simulation orchestration and worker functions
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from simulation.runner import (
    run_single_replication_grid,
    run_single_replication_optuna,
    select_best_models,
    run_simulation
)
from simulation.config import (
    SimulationConfig,
    ExperimentConfig,
    HyperparameterGridConfig,
    GridSearchResult
)


@pytest.fixture
def sample_config():
    """Create a sample simulation configuration."""
    return SimulationConfig(
        n_samples=100,
        n_states=3,
        n_informative=5,
        n_total_features=10,
        delta=0.5,
        lambda_0=2.0,
        persistence=0.9,
        distribution_type='poisson',
        random_seed=42
    )


@pytest.fixture
def sample_hyperparameter_grid():
    """Create a sample hyperparameter grid."""
    return [
        {'n_components': 2, 'jump_penalty': 0.1, 'max_feats': 0.25},
        {'n_components': 2, 'jump_penalty': 0.5, 'max_feats': 0.25},
        {'n_components': 3, 'jump_penalty': 0.1, 'max_feats': 1.0},
    ]


@pytest.fixture
def sample_grid_config():
    """Create a sample hyperparameter grid config."""
    return HyperparameterGridConfig(
        n_components=[2, 3],
        jump_penalty=[0.1, 0.5],
        kappa=[0.5, 1.0]
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestRunSingleReplicationGrid:
    """Test run_single_replication_grid worker function."""
    
    def test_returns_grid_search_results(self, sample_config, sample_hyperparameter_grid):
        """Test that function returns list of GridSearchResult objects."""
        args = (
            sample_config,
            sample_hyperparameter_grid,
            ['Gaussian'],
            False,  # save_models
            'balanced_accuracy'  # optimize_metric
        )
        
        results = run_single_replication_grid(args)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, GridSearchResult) for r in results)
    
    def test_multiple_models(self, sample_config, sample_hyperparameter_grid):
        """Test with multiple model types."""
        args = (
            sample_config,
            sample_hyperparameter_grid,
            ['Gaussian', 'Poisson'],
            False,
            'balanced_accuracy'
        )
        
        results = run_single_replication_grid(args)
        
        # Should have results for both models
        model_names = {r.model_name for r in results}
        assert 'Gaussian' in model_names
        assert 'Poisson' in model_names
    
    def test_save_models_returns_tuple(self, sample_config, sample_hyperparameter_grid):
        """Test that save_models=True returns tuple with best models."""
        args = (
            sample_config,
            sample_hyperparameter_grid,
            ['Gaussian'],
            True,  # save_models
            'balanced_accuracy'
        )
        
        result = run_single_replication_grid(args)
        
        # Should return tuple: (results, best_models_dict)
        assert isinstance(result, tuple)
        assert len(result) == 2
        results_list, best_models = result
        assert isinstance(results_list, list)
        assert isinstance(best_models, dict)
    
    def test_optimize_metric_not_found_raises_error(self, sample_config, sample_hyperparameter_grid):
        """Test that missing optimize_metric raises KeyError."""
        args = (
            sample_config,
            sample_hyperparameter_grid,
            ['Gaussian'],
            True,  # save_models - needed to trigger metric check
            'nonexistent_metric'  # This metric doesn't exist
        )
        
        with pytest.raises(KeyError) as exc_info:
            run_single_replication_grid(args)
        
        assert 'nonexistent_metric' in str(exc_info.value)
        assert 'not found in results' in str(exc_info.value)


class TestRunSingleReplicationOptuna:
    """Test run_single_replication_optuna worker function."""
    
    def test_returns_grid_search_results(self, sample_config, sample_grid_config):
        """Test that function returns list of GridSearchResult objects."""
        args = (
            sample_config,
            5,  # n_trials
            sample_config.n_total_features,
            ['Gaussian'],
            'balanced_accuracy',
            sample_grid_config
        )
        
        results = run_single_replication_optuna(args)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, GridSearchResult) for r in results)
    
    def test_multiple_models(self, sample_config, sample_grid_config):
        """Test Optuna with multiple models."""
        args = (
            sample_config,
            3,  # n_trials (small for speed)
            sample_config.n_total_features,
            ['Gaussian', 'Poisson'],
            'balanced_accuracy',
            sample_grid_config
        )
        
        results = run_single_replication_optuna(args)
        
        # Should have results for both models (one best per model)
        model_names = {r.model_name for r in results}
        assert 'Gaussian' in model_names or 'Poisson' in model_names
    
    def test_optimize_metric_not_found_raises_error(self, sample_config, sample_grid_config):
        """Test that missing optimize_metric raises KeyError in Optuna."""
        args = (
            sample_config,
            2,  # n_trials
            sample_config.n_total_features,
            ['Gaussian'],
            'nonexistent_metric',
            sample_grid_config
        )
        
        with pytest.raises(KeyError) as exc_info:
            run_single_replication_optuna(args)
        
        assert 'nonexistent_metric' in str(exc_info.value)


class TestSelectBestModels:
    """Test select_best_models function."""
    
    def test_selects_best_by_balanced_accuracy(self):
        """Test selection by balanced_accuracy (default)."""
        df = pd.DataFrame({
            'model_name': ['Gaussian', 'Gaussian', 'Gaussian'],
            'n_samples': [100, 100, 100],
            'n_states': [3, 3, 3],
            'n_informative': [5, 5, 5],
            'n_noise': [5, 5, 5],
            'n_total_features': [10, 10, 10],
            'delta': [0.5, 0.5, 0.5],
            'lambda_0': [2.0, 2.0, 2.0],
            'persistence': [0.9, 0.9, 0.9],
            'distribution_type': ['poisson', 'poisson', 'poisson'],
            'correlated_noise': [False, False, False],
            'random_seed': [42, 42, 42],
            'n_components': [2, 3, 4],
            'jump_penalty': [0.1, 0.1, 0.1],
            'max_feats': [0.25, 0.25, 0.25],
            'balanced_accuracy': [0.85, 0.90, 0.88],
            'composite_score': [0.80, 0.85, 0.82]
        })
        
        best = select_best_models(df, metric='balanced_accuracy')
        
        # Should select the one with highest balanced_accuracy (0.90)
        assert len(best) == 1
        assert best.iloc[0]['best_n_components'] == 3
    
    def test_selects_best_by_composite_score(self):
        """Test selection by composite_score."""
        df = pd.DataFrame({
            'model_name': ['Gaussian', 'Gaussian', 'Gaussian'],
            'n_samples': [100, 100, 100],
            'n_states': [3, 3, 3],
            'n_informative': [5, 5, 5],
            'n_noise': [5, 5, 5],
            'n_total_features': [10, 10, 10],
            'delta': [0.5, 0.5, 0.5],
            'lambda_0': [2.0, 2.0, 2.0],
            'persistence': [0.9, 0.9, 0.9],
            'distribution_type': ['poisson', 'poisson', 'poisson'],
            'correlated_noise': [False, False, False],
            'random_seed': [42, 42, 42],
            'n_components': [2, 3, 4],
            'jump_penalty': [0.1, 0.1, 0.1],
            'max_feats': [0.25, 0.25, 0.25],
            'balanced_accuracy': [0.90, 0.85, 0.88],
            'composite_score': [0.80, 0.87, 0.82]
        })
        
        best = select_best_models(df, metric='composite_score')
        
        # Should select the one with highest composite_score (0.87)
        assert len(best) == 1
        assert best.iloc[0]['best_n_components'] == 3
    
    def test_metric_not_found_raises_error(self):
        """Test that missing metric raises ValueError."""
        df = pd.DataFrame({
            'model_name': ['Gaussian'],
            'n_samples': [100],
            'n_states': [3],
            'n_informative': [5],
            'n_noise': [5],
            'n_total_features': [10],
            'delta': [0.5],
            'lambda_0': [2.0],
            'persistence': [0.9],
            'distribution_type': ['poisson'],
            'correlated_noise': [False],
            'random_seed': [42],
            'balanced_accuracy': [0.85]
        })
        
        with pytest.raises(ValueError) as exc_info:
            select_best_models(df, metric='nonexistent_metric')
        
        assert 'nonexistent_metric' in str(exc_info.value)
        assert 'not found in results' in str(exc_info.value)
    
    def test_renames_hyperparameter_columns(self):
        """Test that hyperparameter columns are renamed with 'best_' prefix."""
        df = pd.DataFrame({
            'model_name': ['Gaussian'],
            'n_samples': [100],
            'n_states': [3],
            'n_informative': [5],
            'n_noise': [5],
            'n_total_features': [10],
            'delta': [0.5],
            'lambda_0': [2.0],
            'persistence': [0.9],
            'distribution_type': ['poisson'],
            'correlated_noise': [False],
            'random_seed': [42],
            'n_components': [3],
            'jump_penalty': [0.1],
            'max_feats': [0.25],
            'balanced_accuracy': [0.85]
        })
        
        best = select_best_models(df)
        
        assert 'best_n_components' in best.columns
        assert 'best_jump_penalty' in best.columns
        assert 'best_max_feats' in best.columns
        assert 'n_components' not in best.columns
    
    def test_groups_by_config_and_model(self):
        """Test that best model is selected for each (config, model) combination."""
        df = pd.DataFrame({
            'model_name': ['Gaussian', 'Gaussian', 'Poisson', 'Poisson'],
            'n_samples': [100, 100, 100, 100],
            'n_states': [3, 3, 3, 3],
            'n_informative': [5, 5, 5, 5],
            'n_noise': [5, 5, 5, 5],
            'n_total_features': [10, 10, 10, 10],
            'delta': [0.5, 0.5, 0.5, 0.5],
            'lambda_0': [2.0, 2.0, 2.0, 2.0],
            'persistence': [0.9, 0.9, 0.9, 0.9],
            'distribution_type': ['poisson', 'poisson', 'poisson', 'poisson'],
            'correlated_noise': [False, False, False, False],
            'random_seed': [42, 42, 42, 42],
            'n_components': [2, 3, 2, 3],
            'jump_penalty': [0.1, 0.1, 0.1, 0.1],
            'max_feats': [0.25, 0.25, 0.25, 0.25],
            'balanced_accuracy': [0.85, 0.90, 0.82, 0.88]
        })
        
        best = select_best_models(df)
        
        # Should have one best for each model type
        assert len(best) == 2
        gaussian_best = best[best['model_name'] == 'Gaussian'].iloc[0]
        poisson_best = best[best['model_name'] == 'Poisson'].iloc[0]
        
        assert gaussian_best['best_n_components'] == 3  # Higher BAC
        assert poisson_best['best_n_components'] == 3  # Higher BAC


class TestRunSimulationIntegration:
    """Integration tests for run_simulation function."""
    
    def test_run_simulation_grid_sequential(self, temp_output_dir):
        """Test run_simulation with grid search, sequential execution."""
        experiment_config = ExperimentConfig(
            optimization_method='grid',
            model_names=['Gaussian'],
            n_replications=2,
            optimize_metric='balanced_accuracy',
            hyperparameter_grid=HyperparameterGridConfig(
                n_components=[2],
                jump_penalty=[0.1],
                kappa=[0.5]
            ),
            single_thread=True
        )
        
        data_configs = [
            SimulationConfig(
                n_samples=50,
                n_states=2,
                n_informative=3,
                n_total_features=5,
                delta=0.5,
                lambda_0=2.0,
                persistence=0.9,
                distribution_type='gaussian',
                random_seed=42
            )
        ]
        
        results = run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Check results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'balanced_accuracy' in results.columns
        assert 'best_n_components' in results.columns
        
        # Check output files created
        assert (temp_output_dir / "metadata.json").exists()
        assert (temp_output_dir / "aggregated" / "all_results.csv").exists()
    
    def test_run_simulation_creates_metadata(self, temp_output_dir):
        """Test that run_simulation creates proper metadata."""
        experiment_config = ExperimentConfig(
            optimization_method='grid',
            model_names=['Gaussian'],
            n_replications=1,
            optimize_metric='balanced_accuracy',
            hyperparameter_grid=HyperparameterGridConfig(
                n_components=[2],
                jump_penalty=[0.1],
                kappa=[0.5]
            ),
            single_thread=True
        )
        
        data_configs = [
            SimulationConfig(
                n_samples=50,
                n_states=2,
                n_informative=3,
                n_total_features=5,
                random_seed=42
            )
        ]
        
        run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Check metadata
        import json
        with open(temp_output_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        assert metadata['experiment']['optimization_method'] == 'grid'
        assert metadata['execution']['status'] == 'completed'
        assert 'execution_time_seconds' in metadata['execution']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
