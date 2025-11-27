# Examples Directory

This directory contains example scripts demonstrating how to use the Poisson Jump Simulations framework.

## Examples

All examples use the simplified API - just specify your configuration and run!

### 1. `simple_example.py` - Basic Usage ⭐ START HERE
The simplest possible example showing how to run a simulation and visualize results.

```bash
python examples/simple_example.py
```

**What it demonstrates:**
- Basic configuration setup
- Running a simulation with automatic caching
- Accessing results (best_df, grid_df)
- Auto-generating visualizations

**Runtime:** ~1 minute (cached: <1 second)

---

### 2. `comparison_example.py` - Comparing Configurations
Shows how to compare multiple data generation configurations.

```bash
python examples/comparison_example.py
```

**What it demonstrates:**
- Multiple data configurations in a single experiment
- Comparing Poisson vs Gaussian distributions
- Comparing different effect sizes (delta values)
- Custom hyperparameter grids
- Analyzing results by condition

**Runtime:** ~3-5 minutes

---

### 3. `cached_example.py` - Automatic Caching
Demonstrates the automatic caching system and performance benefits.

```bash
python examples/cached_example.py
```

**What it demonstrates:**
- How caching works automatically
- Performance comparison (cached vs uncached)
- Cache invalidation (cache=False)
- Verifying cached results match original

**Runtime:** ~2 minutes (runs simulation 3 times to demo caching)

---

## Utility Scripts

### 4. `check_data_generation.py` - Data Generation Diagnostics
Utility script to visualize and verify data generation is working correctly.

```bash
python examples/check_data_generation.py
```

**What it demonstrates:**
- Generating data from different distributions
- Visualizing state sequences and breakpoints
- Diagnostic plots for troubleshooting
- Feature correlations

**Use when:** You want to verify data generation or troubleshoot issues.

---

## Results Directory Structure

After running examples, you'll find results in:

```
examples/
├── results/                    # Results from examples
│   └── <experiment_name>_<hash>/
│       ├── metadata.json       # Experiment configuration and timing
│       ├── data_configs.csv    # Data generation parameters
│       ├── aggregated/
│       │   └── all_results.csv # Best model results
│       ├── grid_search/
│       │   └── all_grid_results.csv  # All hyperparameter combinations
│       ├── models/             # Saved model files (if save_models=True)
│       └── plots/              # Auto-generated visualizations
```

---

## Common Patterns

### Pattern 1: Quick Test
```python
config = {
    "experiment_name": "quick_test",
    "data_generation": {"n_samples": 200, "n_states": 2},
    "quick_test": True,  # Small grid (default)
}
results = run_simulation(config)
```

### Pattern 2: Multiple Configurations
```python
config = {
    "experiment_name": "comparison",
    "data_generation": [
        {"delta": 0.1, "distribution_type": "Poisson"},
        {"delta": 0.2, "distribution_type": "Poisson"},
        {"delta": 0.1, "distribution_type": "Gaussian"},
    ],
}
```

### Pattern 3: Full Grid Search
```python
config = {
    "experiment_name": "full_grid",
    "quick_test": False,  # Use full hyperparameter grid
    "hyperparameters": {
        "n_states_values": [2, 3, 4],
        "jump_penalty_num": 7,  # 7 log-spaced values
        "kappa_num": 14,        # 14 linearly-spaced values
    },
}
```

### Pattern 4: Accessing Results
```python
results = run_simulation(config)

# DataFrames
print(results.best_df)      # Best models
print(results.grid_df)      # All hyperparameter combinations

# Summary statistics
results.summary()

# Sort by metric
best = results.get_best_models("balanced_accuracy")
worst_distance = results.get_best_models("chamfer_distance", ascending=True)

# Visualize
visualize_results(results)
```

---

## Tips

1. **Start with simple_example.py** - Understand the basics first
2. **Use caching** - Identical configs load instantly from cache
3. **Use quick_test** - Default quick_test=True is fast for development
4. **Check plots/** - Auto-generated visualizations are very informative
5. **Read metadata.json** - Contains all experiment details and timing

---

## Getting Help

See the main [QUICK_START.md](../QUICK_START.md) for detailed API documentation.

For questions about specific configuration options:
```python
from simulation import SimulationConfig
help(SimulationConfig)
```
