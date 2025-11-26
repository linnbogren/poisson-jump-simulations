"""
Integration tests for the complete simulation pipeline.

Tests end-to-end workflows:
- YAML config loading -> simulation execution -> results validation
- Grid search vs Optuna optimization
- Different optimize_metric values
- Config-driven behavior (no hardcoded values)
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import yaml
from pathlib import Path

from simulation.config import load_config
from simulation.runner import run_simulation
from simulation.hyperparameters import create_data_config_grid
from simulation.results import ResultManager


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_grid_config_yaml():
    """Sample grid search YAML configuration."""
    return {
        'experiment': {
            'optimization_method': 'grid',
            'model_names': ['Gaussian'],
            'n_replications': 2,
            'optimize_metric': 'balanced_accuracy',
            'single_thread': True
        },
        'hyperparameters': {
            'n_components': [2, 3],
            'jump_penalty': [0.1],
            'kappa': [0.5]
        },
        'data_grid': {
            'n_samples': [50],
            'n_states': [2],
            'n_informative': [3],
            'n_total_features': [5],
            'delta': [0.5],
            'lambda_0': [2.0],
            'persistence': [0.9],
            'distribution_type': ['gaussian'],
            'random_seed': [42]
        }
    }


@pytest.fixture
def sample_optuna_config_yaml():
    """Sample Optuna optimization YAML configuration."""
    return {
        'experiment': {
            'optimization_method': 'optuna',
            'model_names': ['Gaussian'],
            'n_replications': 1,
            'optuna_n_trials': 3,
            'optimize_metric': 'balanced_accuracy',
            'single_thread': True
        },
        'hyperparameters': {
            'n_components': [2, 3, 4],
            'jump_penalty': [0.01, 1.0],
            'kappa': [0.3, 1.5]
        },
        'data_grid': {
            'n_samples': [50],
            'n_states': [2],
            'n_informative': [3],
            'n_total_features': [5],
            'delta': [0.5],
            'lambda_0': [2.0],
            'persistence': [0.9],
            'distribution_type': ['gaussian'],
            'random_seed': [42]
        }
    }


class TestEndToEndGridSearch:
    """End-to-end tests for grid search pipeline."""
    
    def test_yaml_to_results_pipeline(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test complete pipeline: YAML -> load_config -> run_simulation -> results."""
        # Save YAML config
        config_file = temp_config_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_grid_config_yaml, f)
        
        # Load config
        experiment_config, data_configs = load_config(config_file)
        
        # Verify config loaded correctly
        assert experiment_config.optimization_method == 'grid'
        assert experiment_config.optimize_metric == 'balanced_accuracy'
        assert len(data_configs) > 0
        
        # Run simulation
        results = run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'balanced_accuracy' in results.columns
        assert 'model_name' in results.columns
        
        # Verify output files
        assert (temp_output_dir / "metadata.json").exists()
        assert (temp_output_dir / "data_configs.csv").exists()
        assert (temp_output_dir / "aggregated" / "all_results.csv").exists()
        assert (temp_output_dir / "grid_search" / "all_grid_results.csv").exists()
    
    def test_different_optimize_metrics(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test that different optimize_metric values work correctly."""
        # Test with composite_score
        config_yaml = sample_grid_config_yaml.copy()
        config_yaml['experiment']['optimize_metric'] = 'composite_score'
        
        config_file = temp_config_dir / "test_composite.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        results = run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Should still work and include composite_score
        assert 'composite_score' in results.columns
        assert len(results) > 0
    
    def test_multiple_models(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test simulation with multiple model types."""
        config_yaml = sample_grid_config_yaml.copy()
        config_yaml['experiment']['model_names'] = ['Gaussian', 'Poisson']
        
        config_file = temp_config_dir / "test_multi_model.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        results = run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Should have results for both models
        model_names = results['model_name'].unique()
        assert 'Gaussian' in model_names
        assert 'Poisson' in model_names


class TestEndToEndOptunaSearch:
    """End-to-end tests for Optuna optimization pipeline."""
    
    def test_yaml_to_results_optuna(self, temp_config_dir, temp_output_dir, sample_optuna_config_yaml):
        """Test complete Optuna pipeline: YAML -> run_simulation -> results."""
        config_file = temp_config_dir / "test_optuna.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_optuna_config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        assert experiment_config.optimization_method == 'optuna'
        assert experiment_config.optuna_n_trials == 3
        
        results = run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'balanced_accuracy' in results.columns
        
        # Optuna should have found best hyperparameters
        assert 'best_n_components' in results.columns
        assert 'best_jump_penalty' in results.columns


class TestResultManagerIntegration:
    """Test ResultManager with actual simulation outputs."""
    
    def test_load_and_filter_results(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test loading and filtering results using ResultManager."""
        # Run a simulation
        config_file = temp_config_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_grid_config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Use ResultManager to load results
        manager = ResultManager(temp_output_dir)
        results = manager.load_results(temp_output_dir)
        
        assert results is not None
        assert len(results) > 0
        
        # Filter by metric
        filtered = manager.filter_by_metric(results, 'balanced_accuracy', min_value=0.0)
        assert len(filtered) > 0
    
    def test_list_and_load_experiments(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test listing experiments and loading their metadata."""
        config_file = temp_config_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_grid_config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # List experiments
        manager = ResultManager(temp_output_dir)
        experiments = manager.list_experiments()
        
        assert len(experiments) == 1
        assert experiments[0]['optimization_method'] == 'grid'
        
        # Load metadata
        metadata = manager.load_experiment_metadata(temp_output_dir)
        assert metadata['experiment']['optimize_metric'] == 'balanced_accuracy'


class TestConfigDrivenBehavior:
    """Test that all behavior is config-driven with no hardcoded values."""
    
    def test_hyperparameters_from_config(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test that hyperparameters come from config, not hardcoded."""
        # Set specific hyperparameter values
        config_yaml = sample_grid_config_yaml.copy()
        config_yaml['hyperparameters'] = {
            'n_components': [5],  # Unusual value
            'jump_penalty': [0.123],  # Specific value
            'kappa': [0.789]
        }
        
        config_file = temp_config_dir / "test_hyperparam.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        results = run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Check that our specific values were used
        assert results.iloc[0]['best_n_components'] == 5
        assert abs(results.iloc[0]['best_jump_penalty'] - 0.123) < 0.001
    
    def test_data_params_from_config(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test that data generation parameters come from config."""
        config_yaml = sample_grid_config_yaml.copy()
        config_yaml['data_grid'] = {
            'n_samples': [123],  # Specific value
            'n_states': [7],
            'n_informative': [11],
            'n_total_features': [15],
            'delta': [0.789],
            'lambda_0': [3.45],
            'persistence': [0.876],
            'distribution_type': ['negative_binomial'],
            'random_seed': [999]
        }
        
        config_file = temp_config_dir / "test_dataparam.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        results = run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Check that our specific data config values were used
        assert results.iloc[0]['n_samples'] == 123
        assert results.iloc[0]['n_states'] == 7
        assert results.iloc[0]['n_informative'] == 11
        assert abs(results.iloc[0]['delta'] - 0.789) < 0.001


class TestErrorHandling:
    """Test error handling in the pipeline."""
    
    def test_invalid_metric_raises_error(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test that specifying invalid optimize_metric raises clear error."""
        config_yaml = sample_grid_config_yaml.copy()
        config_yaml['experiment']['optimize_metric'] = 'totally_fake_metric'
        
        config_file = temp_config_dir / "test_bad_metric.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        with pytest.raises(KeyError) as exc_info:
            run_simulation(
                experiment_config=experiment_config,
                data_configs=data_configs,
                output_dir=str(temp_output_dir),
                n_jobs=1,
                verbose=False
            )
        
        # Should have a clear error message
        assert 'totally_fake_metric' in str(exc_info.value)
        assert 'not found' in str(exc_info.value)


class TestIncrementalSaving:
    """Test incremental saving and resumability."""
    
    def test_incremental_batches_created(self, temp_config_dir, temp_output_dir, sample_grid_config_yaml):
        """Test that incremental batch files are created during execution."""
        config_file = temp_config_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_grid_config_yaml, f)
        
        experiment_config, data_configs = load_config(config_file)
        
        run_simulation(
            experiment_config=experiment_config,
            data_configs=data_configs,
            output_dir=str(temp_output_dir),
            n_jobs=1,
            verbose=False
        )
        
        # Incremental files should be cleaned up after completion
        incremental_dir = temp_output_dir / "grid_search" / "incremental"
        batch_files = list(incremental_dir.glob("batch_*.pkl"))
        assert len(batch_files) == 0  # Should be cleaned up


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
