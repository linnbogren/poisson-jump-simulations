"""Quick test of visualization package."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.config import SimulationConfig
from simulation.data_generation import generate_data
from simulation.models import ModelWrapper
from visualization import plot_model_comparison_bars
import matplotlib.pyplot as plt

print("Testing visualization package...")

# Test 1: Mock data comparison
print("\n1. Creating mock results dataframe...")
np.random.seed(42)
results = []
for model in ['SparseJumpPoisson', 'SparseJumpGaussian']:
    for i in range(20):
        results.append({
            'model_name': model,
            'balanced_accuracy': 0.75 + np.random.normal(0, 0.05),
            'composite_score': 0.80 + np.random.normal(0, 0.04),
        })

results_df = pd.DataFrame(results)
print(f"   Created {len(results_df)} mock results")

# Test 2: Create comparison plot
print("\n2. Creating model comparison plot...")
fig = plot_model_comparison_bars(
    results_df,
    metric='balanced_accuracy',
    title='Test: Model Comparison'
)
print("   ✓ Plot created successfully")
plt.close(fig)

# Test 3: Time series data
print("\n3. Generating time series data...")
config = SimulationConfig(
    n_samples=300,
    n_states=3,
    n_informative=10,
    n_total_features=10,
    delta=0.5,
    lambda_0=10.0,
    persistence=0.97,
    distribution_type="Poisson",
)

X, states, breakpoints = generate_data(config)
print(f"   Generated data shape: {X.shape}")
print(f"   Columns: {X.columns.tolist()[:3]}...")

# Test 4: Fit model
print("\n4. Fitting model...")
model_wrapper = ModelWrapper(
    model_name='Poisson',
    n_components=3,
    max_feats=8,
    jump_penalty=1.0,
)

model_wrapper.fit(X)
print(f"   ✓ Model fitted")

# Test 5: Time series plot
print("\n5. Creating time series plot...")
from visualization import plot_time_series_with_breakpoints

fig = plot_time_series_with_breakpoints(
    X[['informative_1', 'informative_2']],
    model_wrapper.model,
    actual_breakpoints=breakpoints,
    title='Test Plot'
)
print("   ✓ Time series plot created")
plt.close(fig)

# Test 6: Stacked states
print("\n6. Creating stacked states plot...")
from visualization import plot_stacked_states

fig = plot_stacked_states(
    X,
    models_dict={'Test Model': model_wrapper.model},
    true_states=states,
    feature_to_plot='informative_1'
)
print("   ✓ Stacked states plot created")
plt.close(fig)

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nVisualization package is working correctly!")
