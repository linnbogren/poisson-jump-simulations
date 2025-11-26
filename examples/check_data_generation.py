"""
Verify Data Generation with Visualizations

This script generates data from each distribution type and creates
diagnostic plots to verify the data generation is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.config import SimulationConfig
from simulation.data_generation import generate_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def visualize_distribution(config: SimulationConfig, title_prefix: str):
    """Generate and visualize data for a given configuration."""
    
    # Generate data
    X, states, breakpoints = generate_data(config)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{title_prefix} - {config.distribution_type} HMM', fontsize=16, fontweight='bold')
    
    # 1. State sequence with breakpoints
    ax = axes[0, 0]
    ax.plot(states, 'k-', linewidth=1.5, label='True States')
    for bp in breakpoints:
        ax.axvline(bp, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('State Sequence with Breakpoints (red)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Time series of informative features
    ax = axes[0, 1]
    n_informative = min(3, config.n_informative)  # Show first 3
    for i in range(n_informative):
        ax.plot(X.iloc[:, i], alpha=0.7, label=f'Feature {i+1}')
    # Shade background by state
    for state in range(config.n_states):
        state_mask = states == state
        if state_mask.any():
            indices = np.where(state_mask)[0]
            for idx in indices:
                ax.axvspan(idx-0.5, idx+0.5, alpha=0.1, color=f'C{state}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'First {n_informative} Informative Features Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution of informative features by state
    ax = axes[1, 0]
    for state in range(config.n_states):
        state_data = X.iloc[states == state, :config.n_informative].values.flatten()
        ax.hist(state_data, bins=30, alpha=0.5, label=f'State {state}', density=True)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Informative Features by State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distribution of noise features (should be similar across states)
    ax = axes[1, 1]
    if config.n_noise > 0:
        noise_data = X.iloc[:, config.n_informative:].values.flatten()
        ax.hist(noise_data, bins=30, alpha=0.7, color='gray', density=True)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Noise Features (all states)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No noise features', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('No Noise Features')
    
    # 5. Mean values by state for informative features
    ax = axes[2, 0]
    means_by_state = []
    for state in range(config.n_states):
        state_mean = X.iloc[states == state, :config.n_informative].mean(axis=0).values
        means_by_state.append(state_mean)
    
    x_pos = np.arange(config.n_informative)
    width = 0.8 / config.n_states
    for state in range(config.n_states):
        offset = (state - config.n_states/2 + 0.5) * width
        ax.bar(x_pos + offset, means_by_state[state], width, 
               alpha=0.8, label=f'State {state}')
    
    ax.set_xlabel('Informative Feature')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Values by State (Informative Features)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'F{i+1}' for i in range(config.n_informative)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Correlation heatmap
    ax = axes[2, 1]
    corr = X.corr()
    sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8},
                vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    return fig


def main():
    """Generate diagnostic plots for each distribution type."""
    
    # Create output directory
    output_dir = Path('data_generation_checks')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating diagnostic plots for data generation...")
    print("=" * 60)
    
    # Common configuration
    base_config = {
        'n_samples': 200,
        'n_states': 3,
        'n_informative': 5,
        'n_total_features': 8,  # 5 informative + 3 noise
        'delta': 0.5,
        'lambda_0': 5.0,
        'persistence': 0.95,
        'random_seed': 42
    }
    
    # 1. Poisson Distribution
    print("\n1. Testing Poisson HMM...")
    poisson_config = SimulationConfig(
        **base_config,
        distribution_type='Poisson',
        correlated_noise=False
    )
    fig = visualize_distribution(poisson_config, "Test 1")
    fig.savefig(output_dir / '1_poisson_hmm.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / '1_poisson_hmm.png'}")
    plt.close(fig)
    
    # 2. Poisson with Correlated Noise
    print("\n2. Testing Poisson HMM with Correlated Noise...")
    poisson_corr_config = SimulationConfig(
        **base_config,
        distribution_type='Poisson',
        correlated_noise=True,
        noise_correlation=0.6
    )
    fig = visualize_distribution(poisson_corr_config, "Test 2")
    fig.savefig(output_dir / '2_poisson_correlated_noise.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / '2_poisson_correlated_noise.png'}")
    plt.close(fig)
    
    # 3. Negative Binomial (Overdispersed)
    print("\n3. Testing Negative Binomial HMM...")
    nb_config = SimulationConfig(
        **base_config,
        distribution_type='NegativeBinomial',
        nb_dispersion=3.0,
        correlated_noise=False
    )
    fig = visualize_distribution(nb_config, "Test 3")
    fig.savefig(output_dir / '3_negative_binomial_hmm.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / '3_negative_binomial_hmm.png'}")
    plt.close(fig)
    
    # 4. Gaussian HMM
    print("\n4. Testing Gaussian HMM...")
    gaussian_config = SimulationConfig(
        **base_config,
        distribution_type='Gaussian',
        correlated_noise=False
    )
    fig = visualize_distribution(gaussian_config, "Test 4")
    fig.savefig(output_dir / '4_gaussian_hmm.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / '4_gaussian_hmm.png'}")
    plt.close(fig)
    
    # 5. Gaussian with Correlated Noise
    print("\n5. Testing Gaussian HMM with Correlated Noise...")
    gaussian_corr_config = SimulationConfig(
        **base_config,
        distribution_type='Gaussian',
        correlated_noise=True,
        noise_correlation=0.6
    )
    fig = visualize_distribution(gaussian_corr_config, "Test 5")
    fig.savefig(output_dir / '5_gaussian_correlated_noise.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / '5_gaussian_correlated_noise.png'}")
    plt.close(fig)
    
    # 6. Weak Signal (small delta)
    print("\n6. Testing Weak Signal (δ=0.2)...")
    weak_signal_config = SimulationConfig(
        n_samples=200,
        n_states=3,
        n_informative=5,
        n_total_features=8,
        delta=0.2,
        lambda_0=5.0,
        persistence=0.95,
        distribution_type='Poisson',
        correlated_noise=False,
        random_seed=42
    )
    fig = visualize_distribution(weak_signal_config, "Test 6")
    fig.savefig(output_dir / '6_weak_signal.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / '6_weak_signal.png'}")
    plt.close(fig)
    
    # 7. Strong Signal (large delta)
    print("\n7. Testing Strong Signal (δ=0.8)...")
    strong_signal_config = SimulationConfig(
        n_samples=200,
        n_states=3,
        n_informative=5,
        n_total_features=8,
        delta=0.8,
        lambda_0=5.0,
        persistence=0.95,
        distribution_type='Poisson',
        correlated_noise=False,
        random_seed=42
    )
    fig = visualize_distribution(strong_signal_config, "Test 7")
    fig.savefig(output_dir / '7_strong_signal.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / '7_strong_signal.png'}")
    plt.close(fig)
    
    # 8. Two-state HMM
    print("\n8. Testing Two-State HMM...")
    two_state_config = SimulationConfig(
        n_samples=200,
        n_states=2,
        n_informative=5,
        n_total_features=8,
        delta=0.6,
        lambda_0=5.0,
        persistence=0.95,
        distribution_type='Poisson',
        correlated_noise=False,
        random_seed=42
    )
    fig = visualize_distribution(two_state_config, "Test 8")
    fig.savefig(output_dir / '8_two_state_hmm.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / '8_two_state_hmm.png'}")
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print(f"✓ All diagnostic plots saved to: {output_dir.absolute()}")
    print("\nKey things to check:")
    print("  1. States should be clearly separated in the distributions")
    print("  2. Informative features should have different means by state")
    print("  3. Noise features should have similar distributions across states")
    print("  4. Breakpoints (red lines) should align with state changes")
    print("  5. Correlated noise should show correlation in heatmap")
    print("  6. Gaussian data should look continuous, Poisson discrete")


if __name__ == '__main__':
    main()
