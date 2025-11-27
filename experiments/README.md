# Optimization Method Comparison Study

This experiment compares **Grid Search** vs **Optuna** (Bayesian optimization) for hyperparameter tuning in the Sparse Jump Model framework.

## Running the Study

```bash
python experiments/compare_grid_optuna.py
```

## What It Does

1. **Grid Search Run**
   - Exhaustive search over hyperparameter combinations
   - Tests all combinations of n_states and jump_penalty values
   - Consistent but potentially slower

2. **Optuna Run**
   - Bayesian optimization (TPE algorithm)
   - Intelligently explores hyperparameter space
   - Potentially faster with similar performance

3. **Comparison Analysis**
   - Performance metrics comparison
   - Execution time analysis
   - Efficiency visualization (performance vs time)

## Key Metrics Compared

- **Balanced Accuracy**: Classification performance
- **Feature F1**: Feature selection quality
- **Chamfer Distance**: Time series similarity
- **Execution Time**: Total computation time
- **Evaluations**: Number of hyperparameter combinations tested

## Output

Results are saved to:
- `results/optimization_comparison_grid_[hash]/` - Grid search results
- `results/optimization_comparison_optuna_[hash]/` - Optuna results  
- `experiments/optimization_comparison/` - Comparison plots and summary

### Comparison Visualizations

1. **performance_comparison.png** - Bar plots of metrics by method
2. **timing_comparison.png** - Total time and number of evaluations
3. **efficiency_comparison.png** - Performance vs execution time scatter plots
4. **comparison_summary.csv** - Detailed statistics table

## Customization

Edit `compare_grid_optuna.py` to:
- Increase `num_simulations` for more robust estimates (currently 3 for speed)
- Add more data configurations to test different scenarios
- Adjust hyperparameter search spaces
- Change `optuna_n_trials` to control Optuna's search budget

## New API Functions Used

### From `simulation.api`
- `results.get_timing_info()` - Extract timing statistics
- `results.get_performance_summary(metrics)` - Summary statistics for metrics

### From `visualization.api`
- `compare_optimization_methods(results_dict)` - Create comparison visualizations
