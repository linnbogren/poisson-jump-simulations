# poisson-jump-simulations

Simulation testing and real-data analysis for sparse jump models (Gaussian, Poisson, and PoissonKL distributions).

## Features

- **Synthetic Data Simulations**: Test models on generated data with known ground truth
- **Real Data Analysis**: Fit models to unlabeled real-world data using unsupervised metrics (BIC, AIC, Silhouette)
- **Hyperparameter Optimization**: Grid search or Bayesian optimization (Optuna)
- **Visualization**: Stacked time series plots showing model-detected state assignments

## Quick Start: Real Data Analysis

### Basic Usage

```python
import pandas as pd
from simulation import fit_on_real_data
from visualization import plot_stacked_states_from_results

# Load your data (rows = time steps, columns = features)
X = pd.read_csv("your_data.csv", index_col=0)

# Fit models with automatic hyperparameter optimization
results = fit_on_real_data(
    X=X,
    models=["Gaussian", "Poisson", "PoissonKL"],
    n_components_range=[2, 3, 4],
    optimize_metric="bic",  # or "aic", "silhouette"
    optimization_method="optuna",  # or "grid"
    n_trials=50,  # for optuna
    n_jobs=-1,
    verbose=True
)

# Visualize results
fig = plot_stacked_states_from_results(
    X=X,
    results=results,
    feature_to_plot="feature_0",  # show one feature
    save_path="stacked_states.png"
)

# Access best models
best_gaussian = results["best_models"]["Gaussian"]
print(f"Best Gaussian model: {best_gaussian['n_components']} states")
print(f"BIC: {results['results_df'].loc[0, 'bic']:.2f}")
```

### Command-Line Example

```bash
python examples/plot_stacked_states_real_data.py \
    --input data.csv \
    --index-col 0 \
    --feature feature_0 \
    --method optuna \
    --trials 50 \
    --metric bic \
    --states 2 3 4 \
    --save-path results.png
```

Or use synthetic fallback data for testing:
```bash
python examples/plot_stacked_states_real_data.py --method grid
```

## API Reference

### `fit_on_real_data()`

Main function for real data analysis without ground truth labels.

**Parameters**:
- `X` (DataFrame): Time series data (n_samples × n_features)
- `models` (list): Model types to fit, e.g., `["Gaussian", "Poisson", "PoissonKL"]`
- `n_components_range` (list): Number of states to try, e.g., `[2, 3, 4]`
- `optimize_metric` (str): Optimization criterion - `"bic"`, `"aic"`, or `"silhouette"`
- `optimization_method` (str): `"grid"` (exhaustive) or `"optuna"` (Bayesian)
- `n_trials` (int): Number of Optuna trials per model/state combination (if using optuna)
- `hyperparameter_grid` (dict): Custom hyperparameter ranges (optional)
- `n_jobs` (int): Parallel jobs (-1 = all CPUs)
- `save_models` (bool): Save fitted models to disk
- `output_dir` (str): Directory for saving results
- `verbose` (bool): Print progress

**Returns**: Dict with keys:
- `"best_models"`: Dict of best model for each type
- `"results_df"`: DataFrame with all evaluation results
- `"output_dir"`: Path to saved results
- `"studies"` (optuna only): Optuna study objects for analysis

### Hyperparameter Specification

**For Grid Search**:
```python
hyperparameter_grid = {
    "kappa_range": (1.0, 6.0, 4),  # (min, max, num_points)
    "jump_penalty_range": (0.1, 100, 4),
}
```

**For Optuna**:
```python
hyperparameter_grid = {
    "kappa_min": 1.0,
    "kappa_max": 6.0,
    "jump_penalty_min": 0.1,
    "jump_penalty_max": 100,
}
```

## Choosing Optimization Metrics

- **BIC** (Bayesian Information Criterion): Penalizes model complexity, good for identifying true number of states
- **AIC** (Akaike Information Criterion): Less penalty for complexity than BIC
- **Silhouette**: Measures cluster separation quality (higher = better separated states)

Lower is better for BIC/AIC, higher is better for Silhouette.

## Optimization Methods

| Method | Speed | Coverage | Best For |
|--------|-------|----------|----------|
| Grid | Slower | Complete | Small grids, thorough search |
| Optuna | Faster | Intelligent sampling | Large search spaces, quick results |


## Visualization

The stacked plot shows:
1. **Top panel(s)**: Your time series data
2. **Model panels**: Horizontal bars showing detected state assignments for each model

Different colors represent different states. Models automatically align their state labels for visual comparison.

## Documentation

- [REAL_DATA_USAGE.md](REAL_DATA_USAGE.md) - Detailed real-data analysis guide
- [CONFIG_DRIVEN_DESIGN.md](CONFIG_DRIVEN_DESIGN.md) - Configuration system for simulations
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Development roadmap

## Installation

```bash
poetry install
```

## License

See [LICENSE](LICENSE) file.
