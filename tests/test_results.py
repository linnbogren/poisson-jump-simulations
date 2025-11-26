"""
Tests for simulation/results.py - ResultManager and metadata utilities
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from simulation.results import (
    ResultManager,
    save_experiment_metadata,
    update_metadata_on_completion,
    list_experiments,
    save_results_compressed,
    load_results_compressed
)
from simulation.config import (
    SimulationConfig,
    ExperimentConfig,
    HyperparameterGridConfig,
    DataGridConfig
)


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for test results."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_experiment_config():
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        name="test_experiment",
        mode='grid',
        optimization_method='grid',
        model_names=['Gaussian', 'Poisson'],
        n_replications=5,
        optimize_metric='balanced_accuracy',
        hyperparameter_grid=HyperparameterGridConfig(
            n_states_values=[2, 3],
            jump_penalty_min=0.1,
            jump_penalty_max=0.5,
            jump_penalty_num=2,
            kappa_min=0.5,
            kappa_max_type='fixed',
            kappa_max_fixed=1.0,
            kappa_num=2
        ),
        data_grid=DataGridConfig()
    )


@pytest.fixture
def sample_data_configs():
    """Create sample data configurations."""
    return [
        SimulationConfig(
            n_samples=100,
            n_states=3,
            n_informative=5,
            n_total_features=10,
            delta=0.5,
            lambda_0=2.0,
            persistence=0.97,
            distribution_type='poisson',
            random_seed=42
        ),
        SimulationConfig(
            n_samples=200,
            n_states=4,
            n_informative=8,
            n_total_features=20,
            delta=1.0,
            lambda_0=3.0,
            persistence=0.97,
            distribution_type='gaussian',
            random_seed=123
        )
    ]


@pytest.fixture
def sample_results_df():
    """Create a sample results DataFrame."""
    return pd.DataFrame({
        'model_name': ['Gaussian', 'Poisson', 'Gaussian', 'Poisson'],
        'n_components': [2, 2, 3, 3],
        'jump_penalty': [0.1, 0.1, 0.5, 0.5],
        'balanced_accuracy': [0.85, 0.82, 0.88, 0.86],
        'composite_score': [0.80, 0.78, 0.83, 0.81],
        'n_samples': [100, 100, 200, 200],
        'delta': [0.5, 0.5, 1.0, 1.0]
    })


class TestSaveExperimentMetadata:
    """Test save_experiment_metadata function."""
    
    def test_creates_metadata_file(self, temp_results_dir, sample_experiment_config, sample_data_configs):
        """Test that metadata.json is created."""
        save_experiment_metadata(
            output_dir=temp_results_dir,
            experiment_config=sample_experiment_config,
            data_configs=sample_data_configs,
            start_time=datetime.now(),
            n_workers=4
        )
        
        metadata_file = temp_results_dir / "metadata.json"
        assert metadata_file.exists()
    
    def test_metadata_structure(self, temp_results_dir, sample_experiment_config, sample_data_configs):
        """Test that metadata has correct structure."""
        start_time = datetime.now()
        save_experiment_metadata(
            output_dir=temp_results_dir,
            experiment_config=sample_experiment_config,
            data_configs=sample_data_configs,
            start_time=start_time,
            n_workers=4
        )
        
        with open(temp_results_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Check required keys
        assert 'name' in metadata
        assert 'timestamp' in metadata
        assert 'optimization_method' in metadata
        assert 'environment' in metadata
        
        # Check experiment details
        assert metadata['optimization_method'] == 'grid'
        assert metadata['n_replications'] == 5
        assert metadata['n_workers'] == 4
    
    def test_data_configs_csv_created(self, temp_results_dir, sample_experiment_config, sample_data_configs):
        """Test that data_configs.csv is created."""
        save_experiment_metadata(
            output_dir=temp_results_dir,
            experiment_config=sample_experiment_config,
            data_configs=sample_data_configs,
            start_time=datetime.now(),
            n_workers=1
        )
        
        csv_file = temp_results_dir / "data_configs.csv"
        assert csv_file.exists()
        
        # Check CSV content
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert 'n_samples' in df.columns
        assert 'distribution_type' in df.columns


class TestUpdateMetadataOnCompletion:
    """Test update_metadata_on_completion function."""
    
    def test_updates_existing_metadata(self, temp_results_dir, sample_experiment_config, sample_data_configs):
        """Test that completion info is added to metadata."""
        # First create metadata
        start_time = datetime.now()
        save_experiment_metadata(
            output_dir=temp_results_dir,
            experiment_config=sample_experiment_config,
            data_configs=sample_data_configs,
            start_time=start_time,
            n_workers=1
        )
        
        # Update with completion info
        end_time = datetime.now()
        execution_time = 123.45
        update_metadata_on_completion(temp_results_dir, end_time, execution_time)
        
        # Check updated metadata
        with open(temp_results_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        assert 'completed_at' in metadata
        assert metadata['execution_time_seconds'] == 123.45


class TestListExperiments:
    """Test list_experiments function."""
    
    def test_lists_experiments(self, temp_results_dir):
        """Test that experiments are listed correctly."""
        # Create some experiment directories
        (temp_results_dir / "exp1" / "metadata.json").parent.mkdir(parents=True)
        (temp_results_dir / "exp2" / "metadata.json").parent.mkdir(parents=True)
        (temp_results_dir / "exp1" / "metadata.json").write_text('{"name": "test1"}')
        (temp_results_dir / "exp2" / "metadata.json").write_text('{"name": "test2"}')
        
        experiments = list_experiments(temp_results_dir)
        assert len(experiments) == 2


class TestSaveLoadResultsCompressed:
    """Test save_results_compressed and load_results_compressed functions."""
    
    def test_save_and_load_csv_gz(self, temp_results_dir, sample_results_df):
        """Test saving and loading CSV.GZ format."""
        output_file = temp_results_dir / "results.csv.gz"
        
        # Save
        save_results_compressed(sample_results_df, output_file, format='csv.gz')
        assert output_file.exists()
        
        # Load
        loaded_df = load_results_compressed(output_file)
        pd.testing.assert_frame_equal(loaded_df, sample_results_df)
    
    def test_save_and_load_parquet(self, temp_results_dir, sample_results_df):
        """Test saving and loading Parquet format."""
        pytest.skip("pyarrow not installed - optional dependency")
        output_file = temp_results_dir / "results.parquet"
        
        # Save
        save_results_compressed(sample_results_df, output_file, format='parquet')
        assert output_file.exists()
        
        # Load
        loaded_df = load_results_compressed(output_file)
        pd.testing.assert_frame_equal(loaded_df, sample_results_df)
    
    def test_auto_detect_format(self, temp_results_dir, sample_results_df):
        """Test automatic format detection from filename."""
        # CSV.GZ
        csv_file = temp_results_dir / "test.csv.gz"
        save_results_compressed(sample_results_df, csv_file)  # No format specified
        loaded = load_results_compressed(csv_file)
        pd.testing.assert_frame_equal(loaded, sample_results_df)
    
    def test_auto_detect_format_parquet(self, temp_results_dir, sample_results_df):
        """Test automatic format detection for parquet."""
        pytest.importorskip("pyarrow")  # Skip if pyarrow not installed
        
        # Parquet
        parquet_file = temp_results_dir / "test.parquet"
        save_results_compressed(sample_results_df, parquet_file)
        loaded = load_results_compressed(parquet_file)
        pd.testing.assert_frame_equal(loaded, sample_results_df)


class TestResultManager:
    """Test ResultManager class."""
    
    def test_initialization(self, temp_results_dir):
        """Test ResultManager initialization."""
        # Create metadata file so directory is valid
        (temp_results_dir / "metadata.json").write_text('{}')
        manager = ResultManager(temp_results_dir)
        assert manager.results_dir == temp_results_dir
        assert temp_results_dir.exists()
    
    def test_load_best_results(self, temp_results_dir, sample_results_df):
        """Test loading best results CSV."""
        # Create metadata and aggregated directory
        (temp_results_dir / "metadata.json").write_text('{}')
        aggregated_dir = temp_results_dir / "aggregated"
        aggregated_dir.mkdir()
        sample_results_df.to_csv(aggregated_dir / "all_results.csv", index=False)
        
        manager = ResultManager(temp_results_dir)
        results = manager.load_best_results()
        
        assert results is not None
        assert len(results) == len(sample_results_df)
    
    def test_get_summary(self, temp_results_dir, sample_experiment_config, sample_data_configs):
        """Test getting experiment summary."""
        save_experiment_metadata(
            output_dir=temp_results_dir,
            experiment_config=sample_experiment_config,
            data_configs=sample_data_configs,
            start_time=datetime.now(),
            n_workers=1
        )
        
        # Create minimal results
        aggregated_dir = temp_results_dir / "aggregated"
        aggregated_dir.mkdir()
        sample_df = pd.DataFrame({'balanced_accuracy': [0.85, 0.82]})
        sample_df.to_csv(aggregated_dir / "all_results.csv", index=False)
        
        manager = ResultManager(temp_results_dir)
        summary = manager.get_summary()
        
        assert summary is not None
        assert 'optimization_method' in summary
        assert summary['optimization_method'] == 'grid'


class TestResultManagerAggregation:
    """Test ResultManager aggregation methods."""
    
    def test_aggregate_by_config(self, temp_results_dir, sample_experiment_config, sample_data_configs):
        """Test aggregating results by configuration."""
        # Create metadata
        save_experiment_metadata(
            output_dir=temp_results_dir,
            experiment_config=sample_experiment_config,
            data_configs=sample_data_configs,
            start_time=datetime.now(),
            n_workers=1
        )
        
        # Create results with multiple replications
        df = pd.DataFrame({
            'config_id': [0, 0, 0, 1, 1, 1],
            'model_name': ['Gaussian'] * 6,
            'replication': [1, 2, 3, 1, 2, 3],
            'n_samples': [100, 100, 100, 200, 200, 200],
            'delta': [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            'balanced_accuracy': [0.85, 0.86, 0.84, 0.90, 0.91, 0.89],
            'random_seed': [1, 2, 3, 1, 2, 3]
        })
        
        aggregated_dir = temp_results_dir / "aggregated"
        aggregated_dir.mkdir()
        df.to_csv(aggregated_dir / "best_results.csv", index=False)
        
        manager = ResultManager(temp_results_dir)
        agg = manager.aggregate_by_config(metric='balanced_accuracy')
        
        assert agg is not None
        assert len(agg) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
