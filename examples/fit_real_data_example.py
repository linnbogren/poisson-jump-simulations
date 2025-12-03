"""
Example: Fitting Sparse Jump Models to Real Data

This example demonstrates how to use the simulation package on real data
where ground truth labels are not available. The workflow includes:

1. Loading/creating data
2. Fitting multiple models (Gaussian, Poisson, PoissonKL) 
3. Optimizing hyperparameters using unsupervised metrics (BIC/AIC/Silhouette)
4. Visualizing results with stacked time series plots

The API is designed to be simple and requires minimal configuration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the package (assuming it's installed or in PYTHONPATH)
from simulation import fit_on_real_data
from visualization import plot_stacked_time_series


def load_or_create_example_data():
    """
    Load your real data here, or use this example generator.
    
    Replace this with your actual data loading:
    >>> X = pd.read_csv('my_data.csv')
    >>> X = pd.read_excel('my_data.xlsx')
    >>> X = pd.read_parquet('my_data.parquet')
    """
    # Example: Generate some synthetic data for demonstration
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    # Create data with 3 hidden regimes
    states = np.zeros(n_samples, dtype=int)
    states[150:300] = 1
    states[300:] = 2
    
    # Generate Poisson data with different rates per state
    X = np.zeros((n_samples, n_features))
    for t in range(n_samples):
        if states[t] == 0:
            X[t, :10] = np.random.poisson(5, size=10)
            X[t, 10:] = np.random.poisson(10, size=10)
        elif states[t] == 1:
            X[t, :10] = np.random.poisson(15, size=10)
            X[t, 10:] = np.random.poisson(10, size=10)
        else:
            X[t, :10] = np.random.poisson(10, size=10)
            X[t, 10:] = np.random.poisson(10, size=10)
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
    
    return X_df


def main():
    """Main workflow for fitting models to real data."""
    
    print("="*80)
    print("SPARSE JUMP MODELS ON REAL DATA - EXAMPLE")
    print("="*80)
    
    # =========================================================================
    # Step 1: Load your data
    # =========================================================================
    print("\n1. Loading data...")
    X = load_or_create_example_data()
    print(f"   Data shape: {X.shape}")
    print(f"   Data range: [{X.min().min():.1f}, {X.max().max():.1f}]")
    
    # =========================================================================
    # Step 2: Fit models with BIC optimization
    # =========================================================================
    print("\n2. Fitting models optimized for BIC...")
    results_bic = fit_on_real_data(
        X,
        models=['Gaussian', 'Poisson', 'PoissonKL'],
        n_components_range=[2, 3, 4],  # Try 2-4 states
        optimize_metric='bic',  # Minimize BIC
        n_jobs=-1,  # Use all cores
        output_dir='real_data_results_bic',
        verbose=True
    )
    
    print("\n   Best models (BIC):")
    for model_name, wrapper in results_bic['best_models'].items():
        if wrapper is not None:
            metrics = wrapper.evaluate_unsupervised(X)
            print(f"     {model_name:12s}: BIC={metrics['bic']:7.1f}, "
                  f"States={metrics['hyperparameters']['n_components']}, "
                  f"Selected features={metrics['n_selected_total']}")
    
    # =========================================================================
    # Step 3: Fit models with AIC optimization
    # =========================================================================
    print("\n3. Fitting models optimized for AIC...")
    results_aic = fit_on_real_data(
        X,
        models=['Gaussian', 'Poisson', 'PoissonKL'],
        n_components_range=[2, 3, 4],
        optimize_metric='aic',  # Minimize AIC
        n_jobs=-1,
        output_dir='real_data_results_aic',
        verbose=True
    )
    
    # =========================================================================
    # Step 4: Fit models with Silhouette optimization
    # =========================================================================
    print("\n4. Fitting models optimized for Silhouette...")
    results_silhouette = fit_on_real_data(
        X,
        models=['Gaussian', 'Poisson', 'PoissonKL'],
        n_components_range=[2, 3, 4],
        optimize_metric='silhouette',  # Maximize Silhouette
        n_jobs=-1,
        output_dir='real_data_results_silhouette',
        verbose=True
    )
    
    # =========================================================================
    # Step 5: Visualize results with stacked time series
    # =========================================================================
    print("\n5. Creating visualizations...")
    
    output_plots = Path('real_data_plots')
    output_plots.mkdir(exist_ok=True)
    
    # Plot for each optimization criterion
    for criterion, results in [
        ('BIC', results_bic),
        ('AIC', results_aic),
        ('Silhouette', results_silhouette)
    ]:
        print(f"\n   Creating plots for {criterion} optimization...")
        
        for model_name, wrapper in results['best_models'].items():
            if wrapper is None:
                continue
            
            # Create stacked time series plot
            fig = plot_stacked_time_series(
                X,
                wrapper.model,
                selected_features=wrapper.get_selected_features()[:10],  # Show top 10
                title=f'{model_name} Model (optimized for {criterion})',
                figsize=(14, 8),
            )
            
            # Save plot
            plot_path = output_plots / f'{criterion.lower()}_{model_name.lower()}_timeseries.png'
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"     ✓ Saved: {plot_path.name}")
    
    # =========================================================================
    # Step 6: Compare all results
    # =========================================================================
    print("\n6. Comparing results across optimization criteria...")
    
    comparison_data = []
    for criterion, results in [
        ('BIC', results_bic),
        ('AIC', results_aic),
        ('Silhouette', results_silhouette)
    ]:
        for model_name, wrapper in results['best_models'].items():
            if wrapper is None:
                continue
            metrics = wrapper.evaluate_unsupervised(X)
            comparison_data.append({
                'Criterion': criterion,
                'Model': model_name,
                'BIC': metrics['bic'],
                'AIC': metrics['aic'],
                'Silhouette': metrics['silhouette'],
                'States': metrics['hyperparameters']['n_components'],
                'Features': metrics['n_selected_total'],
                'Breakpoints': metrics['n_breakpoints_estimated'],
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = output_plots / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n   ✓ Saved comparison to: {comparison_path}")
    
    # =========================================================================
    # Step 7: Access fitted models for further analysis
    # =========================================================================
    print("\n7. Example: Accessing fitted models...")
    
    # Get best Poisson model optimized for BIC
    best_poisson = results_bic['best_models']['Poisson']
    
    if best_poisson is not None:
        # Get predicted states
        states = best_poisson.get_states()
        print(f"\n   Predicted states (first 20): {states[:20]}")
        
        # Get selected features
        selected_features = best_poisson.get_selected_features()
        print(f"   Selected features: {selected_features}")
        
        # Get model parameters
        print(f"   Model centers shape: {best_poisson.model.centers_.shape}")
        print(f"   Hyperparameters: {best_poisson.hyperparameters}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print(f"Results saved to: {output_plots}")
    print("="*80)


if __name__ == '__main__':
    main()
