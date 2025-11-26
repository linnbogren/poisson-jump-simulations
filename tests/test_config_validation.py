"""
Tests for Configuration Loading and Validation

This test file validates that:
1. All example YAML configs load correctly
2. Config dataclasses are properly constructed
3. Hyperparameter grids are generated from config
4. Data config grids are generated from config
5. No hardcoded values are used
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.config import (
    SimulationConfig,
    HyperparameterGridConfig,
    DataGridConfig,
    ExperimentConfig,
    load_config,
    validate_config
)
from simulation.hyperparameters import (
    create_hyperparameter_grid,
    create_data_config_grid,
    get_search_space_info
)


class TestConfigLoading:
    """Test loading configurations from YAML files."""
    
    def test_load_quick_test_config(self):
        """Test loading quick_test.yaml configuration."""
        config_path = Path(__file__).parent.parent / "configs" / "examples" / "quick_test.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        config = load_config(config_path)
        
        # Check experiment settings
        assert config.name == "quick_test_experiment"
        assert config.mode == "grid"
        assert config.n_replications == 5
        assert config.parallel == True
        assert config.quick_test == True
        assert config.optimization_method == "grid"
        assert config.model_names == ["Gaussian", "Poisson", "PoissonKL"]
        
        # Check hyperparameter grid config
        assert config.hyperparameter_grid is not None
        assert isinstance(config.hyperparameter_grid, HyperparameterGridConfig)
        assert config.hyperparameter_grid.n_states_values == [2, 3, 4]
        assert config.hyperparameter_grid.jump_penalty_min == 0.1
        assert config.hyperparameter_grid.jump_penalty_max == 100.0
        assert config.hyperparameter_grid.jump_penalty_num == 7
        assert config.hyperparameter_grid.jump_penalty_scale == "log"
        assert config.hyperparameter_grid.kappa_min == 1.0
        assert config.hyperparameter_grid.kappa_max_type == "sqrt_P"
        assert config.hyperparameter_grid.quick_test == True
        
        # Check data grid config
        assert config.data_grid is not None
        assert isinstance(config.data_grid, DataGridConfig)
        assert config.data_grid.n_total_features_values == [15, 30, 60, 150, 300]
        assert config.data_grid.delta_values == [0.05, 0.1, 0.2, 0.5]
        assert config.data_grid.distribution_types == ["Poisson", "NegativeBinomial"]
        assert config.data_grid.n_samples == 500
        assert config.data_grid.n_states == 3
        assert config.data_grid.n_informative == 15
        assert config.data_grid.quick_test == True
        
        print("✓ quick_test.yaml loaded successfully")
    
    def test_load_full_study_config(self):
        """Test loading full_study.yaml configuration."""
        config_path = Path(__file__).parent.parent / "configs" / "examples" / "full_study.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        config = load_config(config_path)
        
        # Check experiment settings
        assert config.name == "full_simulation_study"
        assert config.mode == "grid"
        assert config.n_replications == 100
        assert config.quick_test == False
        assert config.optimization_method == "grid"
        
        # Check hyperparameter grid config - should NOT be in quick test mode
        assert config.hyperparameter_grid.quick_test == False
        assert config.hyperparameter_grid.jump_penalty_num == 7
        assert config.hyperparameter_grid.kappa_num == 14
        
        # Check data grid config - should be full grid
        assert config.data_grid.quick_test == False
        assert len(config.data_grid.n_total_features_values) == 5
        assert len(config.data_grid.delta_values) == 4
        
        print("✓ full_study.yaml loaded successfully")
    
    def test_load_optuna_config(self):
        """Test loading optuna_example.yaml configuration."""
        config_path = Path(__file__).parent.parent / "configs" / "examples" / "optuna_example.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        config = load_config(config_path)
        
        # Check Optuna-specific settings
        assert config.optimization_method == "optuna"
        assert config.optuna_n_trials == 100
        assert config.optimize_metric == "composite_score"
        
        # Check that hyperparameter grid still defines search space
        assert config.hyperparameter_grid is not None
        assert config.hyperparameter_grid.jump_penalty_scale == "log"
        assert config.hyperparameter_grid.kappa_max_type == "sqrt_P"
        
        print("✓ optuna_example.yaml loaded successfully")


class TestHyperparameterGridGeneration:
    """Test hyperparameter grid generation from config."""
    
    def test_quick_test_grid_generation(self):
        """Test that quick test mode generates 3x3x3 grid."""
        grid_config = HyperparameterGridConfig(
            n_states_values=[2, 3, 4],
            jump_penalty_min=0.1,
            jump_penalty_max=100.0,
            jump_penalty_num=7,
            jump_penalty_scale="log",
            kappa_min=1.0,
            kappa_max_type="sqrt_P",
            kappa_num=14,
            quick_test=True
        )
        
        grid = create_hyperparameter_grid(grid_config, n_total_features=60)
        
        # Quick test should be 3 states × 3 jump_penalty × 3 kappa = 27
        assert len(grid) == 27, f"Expected 27 combinations, got {len(grid)}"
        
        # Check that all required keys are present
        for params in grid:
            assert 'n_components' in params
            assert 'jump_penalty' in params
            assert 'max_feats' in params
            assert params['n_components'] in [2, 3, 4]
            assert params['jump_penalty'] >= 0.1
            assert params['jump_penalty'] <= 100.0
            assert params['max_feats'] >= 1.0  # kappa_min^2
        
        print(f"✓ Quick test grid: {len(grid)} combinations")
    
    def test_full_grid_generation(self):
        """Test that full grid mode generates 3x7x14 grid."""
        grid_config = HyperparameterGridConfig(
            n_states_values=[2, 3, 4],
            jump_penalty_min=0.1,
            jump_penalty_max=100.0,
            jump_penalty_num=7,
            jump_penalty_scale="log",
            kappa_min=1.0,
            kappa_max_type="sqrt_P",
            kappa_num=14,
            quick_test=False
        )
        
        grid = create_hyperparameter_grid(grid_config, n_total_features=60)
        
        # Full grid should be 3 states × 7 jump_penalty × 14 kappa = 294
        assert len(grid) == 294, f"Expected 294 combinations, got {len(grid)}"
        
        print(f"✓ Full grid: {len(grid)} combinations")
    
    def test_kappa_max_sqrt_P(self):
        """Test that kappa_max_type='sqrt_P' works correctly."""
        grid_config = HyperparameterGridConfig(
            n_states_values=[3],
            jump_penalty_min=1.0,
            jump_penalty_max=1.0,
            jump_penalty_num=1,
            kappa_min=1.0,
            kappa_max_type="sqrt_P",
            kappa_num=3,
            quick_test=False
        )
        
        # For n_total_features=100, sqrt(100) = 10
        grid = create_hyperparameter_grid(grid_config, n_total_features=100)
        
        # Should have 1 × 1 × 3 = 3 combinations
        assert len(grid) == 3
        
        # Max max_feats should be 10^2 = 100
        max_feats_values = [p['max_feats'] for p in grid]
        assert max(max_feats_values) == pytest.approx(100.0, rel=1e-2)
        assert min(max_feats_values) == pytest.approx(1.0, rel=1e-2)
        
        print("✓ kappa_max_type='sqrt_P' works correctly")
    
    def test_kappa_max_fixed(self):
        """Test that kappa_max_type='fixed' works correctly."""
        grid_config = HyperparameterGridConfig(
            n_states_values=[3],
            jump_penalty_min=1.0,
            jump_penalty_max=1.0,
            jump_penalty_num=1,
            kappa_min=1.0,
            kappa_max_type="fixed",
            kappa_max_fixed=5.0,
            kappa_num=3,
            quick_test=False
        )
        
        grid = create_hyperparameter_grid(grid_config, n_total_features=100)
        
        # Max max_feats should be 5^2 = 25
        max_feats_values = [p['max_feats'] for p in grid]
        assert max(max_feats_values) == pytest.approx(25.0, rel=1e-2)
        
        print("✓ kappa_max_type='fixed' works correctly")
    
    def test_linear_jump_penalty_scale(self):
        """Test that jump_penalty_scale='linear' works correctly."""
        grid_config = HyperparameterGridConfig(
            n_states_values=[3],
            jump_penalty_min=1.0,
            jump_penalty_max=10.0,
            jump_penalty_num=10,
            jump_penalty_scale="linear",
            kappa_min=1.0,
            kappa_max_type="sqrt_P",
            kappa_num=1,
            quick_test=False
        )
        
        grid = create_hyperparameter_grid(grid_config, n_total_features=60)
        
        # Should have 1 × 10 × 1 = 10 combinations
        assert len(grid) == 10
        
        # Jump penalties should be linearly spaced
        jump_penalties = sorted(set(p['jump_penalty'] for p in grid))
        assert len(jump_penalties) == 10
        
        # Check roughly equal spacing
        diffs = [jump_penalties[i+1] - jump_penalties[i] for i in range(len(jump_penalties)-1)]
        assert all(abs(d - 1.0) < 0.01 for d in diffs)  # Should be ~1.0 apart
        
        print("✓ jump_penalty_scale='linear' works correctly")


class TestDataConfigGridGeneration:
    """Test data configuration grid generation from config."""
    
    def test_quick_test_data_grid(self):
        """Test that quick test mode uses min/center/max values."""
        data_grid_config = DataGridConfig(
            n_total_features_values=[15, 30, 60, 150, 300],
            delta_values=[0.05, 0.1, 0.2, 0.5],
            distribution_types=["Poisson", "NegativeBinomial"],
            correlated_noise_values=[False, True],
            correlated_noise_distributions=["Poisson"],
            quick_test=True
        )
        
        configs = create_data_config_grid(data_grid_config)
        
        # Quick test: 3 features × 3 deltas × 1 distribution × 1 noise = 9
        # (only Poisson is used in quick test, no correlated noise in quick mode)
        assert len(configs) == 9, f"Expected 9 configs, got {len(configs)}"
        
        # Check that only first, middle, last values are used
        feature_counts = sorted(set(c.n_total_features for c in configs))
        assert feature_counts == [15, 60, 300]  # first, middle, last
        
        delta_values = sorted(set(c.delta for c in configs))
        assert delta_values == [0.05, 0.2, 0.5]  # first, middle, last
        
        print(f"✓ Quick test data grid: {len(configs)} configurations")
    
    def test_full_data_grid(self):
        """Test that full grid uses all specified values."""
        data_grid_config = DataGridConfig(
            n_total_features_values=[15, 30, 60],
            delta_values=[0.1, 0.2],
            distribution_types=["Poisson", "NegativeBinomial"],
            correlated_noise_values=[False, True],
            correlated_noise_distributions=["Poisson"],
            quick_test=False
        )
        
        configs = create_data_config_grid(data_grid_config)
        
        # Full grid: 3 features × 2 deltas × 2 distributions × (1 + correlated for Poisson)
        # = 3 × 2 × 2 (base) + 3 × 2 × 1 (Poisson with correlated noise)
        # = 12 + 6 = 18
        expected = 3 * 2 * 2 + 3 * 2 * 1  # Base + Poisson with correlated noise
        assert len(configs) == expected, f"Expected {expected} configs, got {len(configs)}"
        
        print(f"✓ Full data grid: {len(configs)} configurations")
    
    def test_correlated_noise_only_with_poisson(self):
        """Test that correlated noise is only tested with Poisson."""
        data_grid_config = DataGridConfig(
            n_total_features_values=[60],
            delta_values=[0.2],
            distribution_types=["Poisson", "NegativeBinomial"],
            correlated_noise_values=[False, True],
            correlated_noise_distributions=["Poisson"],
            quick_test=False
        )
        
        configs = create_data_config_grid(data_grid_config)
        
        # Check that NegativeBinomial never has correlated noise
        for config in configs:
            if config.distribution_type == "NegativeBinomial":
                assert config.correlated_noise == False
            elif config.distribution_type == "Poisson":
                # Poisson should have both True and False
                pass
        
        # Should have configs with correlated noise = True
        assert any(c.correlated_noise for c in configs), "No configs with correlated noise found"
        
        print("✓ Correlated noise only applied to Poisson")
    
    def test_unique_random_seeds(self):
        """Test that each config gets a unique random seed."""
        data_grid_config = DataGridConfig(
            n_total_features_values=[15, 30],
            delta_values=[0.1, 0.2],
            distribution_types=["Poisson"],
            correlated_noise_values=[False],
            base_seed=42,
            quick_test=False
        )
        
        configs = create_data_config_grid(data_grid_config)
        
        # All seeds should be unique
        seeds = [c.random_seed for c in configs]
        assert len(seeds) == len(set(seeds)), "Random seeds are not unique!"
        
        # All seeds should be >= base_seed
        assert all(s >= data_grid_config.base_seed for s in seeds)
        
        print("✓ All configs have unique random seeds")


class TestSearchSpaceInfo:
    """Test search space information utility."""
    
    def test_search_space_info_quick(self):
        """Test search space info for quick test mode."""
        grid_config = HyperparameterGridConfig(quick_test=True)
        
        info = get_search_space_info(grid_config, n_total_features=60)
        
        assert info['grid_size'] == 27  # 3 × 3 × 3
        assert info['n_states_values'] == [2, 3, 4]
        assert info['n_jump_penalty_values'] == 3
        assert info['n_kappa_values'] == 3
        assert info['jump_penalty_scale'] == "log"
        
        print(f"✓ Search space info (quick): {info['grid_size']} combinations")
    
    def test_search_space_info_full(self):
        """Test search space info for full grid mode."""
        grid_config = HyperparameterGridConfig(quick_test=False)
        
        info = get_search_space_info(grid_config, n_total_features=60)
        
        assert info['grid_size'] == 294  # 3 × 7 × 14
        assert info['n_jump_penalty_values'] == 7
        assert info['n_kappa_values'] == 14
        
        print(f"✓ Search space info (full): {info['grid_size']} combinations")


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_valid_config(self):
        """Test that valid config passes validation."""
        config_path = Path(__file__).parent.parent / "configs" / "examples" / "quick_test.yaml"
        config = load_config(config_path)
        
        messages = validate_config(config)
        
        # Should have no error messages
        errors = [m for m in messages if m.startswith("ERROR")]
        assert len(errors) == 0, f"Valid config has errors: {errors}"
        
        print("✓ Valid config passes validation")
    
    def test_all_example_configs_valid(self):
        """Test that all example configs are valid."""
        examples_dir = Path(__file__).parent.parent / "configs" / "examples"
        yaml_files = list(examples_dir.glob("*.yaml"))
        
        assert len(yaml_files) > 0, "No example YAML files found"
        
        for yaml_file in yaml_files:
            config = load_config(yaml_file)
            messages = validate_config(config)
            errors = [m for m in messages if m.startswith("ERROR")]
            
            assert len(errors) == 0, f"{yaml_file.name} has errors: {errors}"
            print(f"✓ {yaml_file.name} is valid")


class TestNoHardcodedValues:
    """Test that no values are hardcoded in the functions."""
    
    def test_different_configs_produce_different_grids(self):
        """Test that changing config values changes the grid."""
        # Config 1: Default
        config1 = HyperparameterGridConfig(
            jump_penalty_min=0.1,
            jump_penalty_max=100.0,
            quick_test=True
        )
        grid1 = create_hyperparameter_grid(config1, n_total_features=60)
        
        # Config 2: Different range
        config2 = HyperparameterGridConfig(
            jump_penalty_min=1.0,  # DIFFERENT
            jump_penalty_max=10.0,  # DIFFERENT
            quick_test=True
        )
        grid2 = create_hyperparameter_grid(config2, n_total_features=60)
        
        # Grids should be different
        jump_penalties_1 = sorted(set(p['jump_penalty'] for p in grid1))
        jump_penalties_2 = sorted(set(p['jump_penalty'] for p in grid2))
        
        assert jump_penalties_1 != jump_penalties_2, \
            "Different configs produced same jump penalties - values might be hardcoded!"
        
        # Config 2 should have smaller range
        assert max(jump_penalties_2) < max(jump_penalties_1)
        assert min(jump_penalties_2) > min(jump_penalties_1)
        
        print("✓ Different configs produce different grids (not hardcoded)")
    
    def test_config_controls_grid_size(self):
        """Test that config controls the grid size, not hardcoded values."""
        # Small grid
        config_small = HyperparameterGridConfig(
            n_states_values=[3],  # 1 value
            jump_penalty_min=1.0,
            jump_penalty_max=1.0,
            jump_penalty_num=1,  # 1 value
            kappa_num=1,  # 1 value
            quick_test=False
        )
        grid_small = create_hyperparameter_grid(config_small, n_total_features=60)
        assert len(grid_small) == 1  # 1 × 1 × 1
        
        # Large grid
        config_large = HyperparameterGridConfig(
            n_states_values=[2, 3, 4, 5, 6],  # 5 values
            jump_penalty_num=10,  # 10 values
            kappa_num=20,  # 20 values
            quick_test=False
        )
        grid_large = create_hyperparameter_grid(config_large, n_total_features=60)
        assert len(grid_large) == 1000  # 5 × 10 × 20
        
        print("✓ Config controls grid size (not hardcoded)")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("CONFIGURATION VALIDATION TESTS")
    print("="*80 + "\n")
    
    test_classes = [
        TestConfigLoading,
        TestHyperparameterGridGeneration,
        TestDataConfigGridGeneration,
        TestSearchSpaceInfo,
        TestConfigValidation,
        TestNoHardcodedValues
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 80)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name} FAILED: {e}")
            except Exception as e:
                print(f"✗ {method_name} ERROR: {e}")
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("="*80 + "\n")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Configuration system is working correctly.")
        print("✅ NO hardcoded values detected")
        print("✅ All YAML configs load successfully")
        print("✅ Grid generation works from config")
        return 0
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
