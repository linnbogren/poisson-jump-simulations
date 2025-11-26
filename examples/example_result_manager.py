"""
Example: Using ResultManager to Analyze Simulation Results

This example demonstrates how to use the ResultManager class to load,
filter, and analyze simulation results efficiently.
"""

from simulation import ResultManager, list_experiments

# ============================================================================
# Example 1: List all experiments
# ============================================================================
print("="*80)
print("EXAMPLE 1: List All Experiments")
print("="*80)

experiments = list_experiments("results")
if len(experiments) > 0:
    print(experiments[['name', 'timestamp', 'n_replications', 'optimization_method']])
else:
    print("No experiments found in results/")

# ============================================================================
# Example 2: Load and explore a specific experiment
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 2: Explore a Specific Experiment")
print("="*80)

# Load experiment (replace with actual directory)
# rm = ResultManager("results/grid_search_2025-11-26_14-30")

# Get summary
# summary = rm.get_summary()
# print("\nExperiment Summary:")
# for key, value in summary.items():
#     print(f"  {key}: {value}")

# ============================================================================
# Example 3: Load and filter results
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 3: Load and Filter Results")
print("="*80)

# Load best results (fastest)
# best_df = rm.load_best_results()
# print(f"\nBest results: {len(best_df)} rows")
# print(best_df.head())

# Load grid results with filtering
# good_results = rm.load_grid_results(
#     models=['SparseJumpPoisson'],
#     metric_filter={'balanced_accuracy': ('>', 0.8)}
# )
# print(f"\nFiltered grid results: {len(good_results)} rows")

# ============================================================================
# Example 4: Find best hyperparameters
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 4: Find Best Hyperparameters")
print("="*80)

# Find best hyperparameters for a specific configuration
# best_params = rm.find_best_hyperparameters(
#     config_id=5,
#     model_name='SparseJumpPoisson',
#     metric='composite_score',
#     top_k=5
# )
# print("\nTop 5 hyperparameter configurations:")
# print(best_params[['n_components', 'jump_penalty', 'max_feats', 'composite_score']])

# ============================================================================
# Example 5: Aggregate results by configuration
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 5: Aggregate Results Across Replications")
print("="*80)

# Aggregate to get mean ± std across replications
# agg_df = rm.aggregate_by_config('balanced_accuracy')
# print("\nAggregated results (mean ± std):")
# print(agg_df[['config_id', 'model_name', 'balanced_accuracy_mean', 'balanced_accuracy_std']])

# ============================================================================
# Example 6: Compare models
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 6: Compare Model Performance")
print("="*80)

# Compare models across all configurations
# comparison = rm.compare_models('composite_score')
# print("\nModel comparison:")
# print(comparison)

# ============================================================================
# Example 7: Get configuration details
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 7: Get Data Configuration Details")
print("="*80)

# Get details for a specific configuration
# config_details = rm.get_config_details(config_id=5)
# print("\nConfiguration 5 details:")
# for key, value in config_details.items():
#     print(f"  {key}: {value}")

# ============================================================================
# Example 8: Load a fitted model
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 8: Load a Fitted Model")
print("="*80)

# Load best model for a configuration
# model = rm.load_model(config_id=5, model_name='SparseJumpPoisson', replication=0)
# print(f"\nLoaded model: {type(model)}")
# print(f"Model centers shape: {model.centers_.shape}")

print("\n" + "="*80)
print("Examples completed!")
print("="*80)
print("\nTo use these examples, replace the commented lines with actual")
print("experiment directories and run the script.")
