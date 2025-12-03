"""Example: Plot stacked states for real data runs (no true labels).

Usage
-----
Run with a CSV path:
    powershell> python examples/plot_stacked_states_real_data.py --input data.csv --index-col 0 --feature feature_0 --method optuna --trials 50

Or run without input to use a synthetic fallback:
    powershell> python examples/plot_stacked_states_real_data.py --method grid

This script optimizes models on unlabeled data (BIC/AIC/Silhouette),
then draws a stacked time-series + state bars plot for one feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from simulation import fit_on_real_data
from visualization import plot_stacked_states_from_results


def load_data(input_path: str | None, index_col: int | str | None) -> pd.DataFrame:
    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        kwargs = {}
        if index_col is not None:
            kwargs["index_col"] = index_col
        df = pd.read_csv(p, **kwargs)
        return df

    # Fallback synthetic dataset (no labels): piecewise-constant means with noise
    rng = np.random.default_rng(42)
    n, p = 300, 6
    segs = [0, 120, 210, n]
    means = [
        np.array([2.0, 3.0, 1.0, 0.5, 1.5, 2.5]),
        np.array([5.0, 1.0, 2.0, 3.0, 0.8, 1.2]),
        np.array([1.0, 2.5, 3.5, 2.0, 2.2, 0.5]),
    ]
    X = np.empty((n, p), dtype=float)
    for s in range(len(segs) - 1):
        a, b = segs[s], segs[s + 1]
        X[a:b] = rng.normal(loc=means[s], scale=0.7, size=(b - a, p))
    cols = [f"feature_{i}" for i in range(p)]
    idx = np.arange(n)
    return pd.DataFrame(X, index=idx, columns=cols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stacked states for real/unlabeled data.")
    parser.add_argument("--input", type=str, default=None, help="Path to CSV with features. If omitted, uses synthetic data.")
    parser.add_argument("--index-col", type=str, default=None, help="Index column name or integer for CSV loading.")
    parser.add_argument("--feature", type=str, default=None, help="Single feature to show on the top panel. Defaults to first column.")
    parser.add_argument("--models", type=str, nargs="*", default=["Gaussian", "Poisson", "PoissonKL"], help="Models to evaluate.")
    parser.add_argument("--states", type=int, nargs="*", default=[2, 3, 4], help="State counts to try (e.g., 2 3 4).")
    parser.add_argument("--metric", type=str, default="bic", choices=["bic", "aic", "silhouette"], help="Optimization metric.")
    parser.add_argument("--method", type=str, default="optuna", choices=["grid", "optuna"], help="Optimization method.")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials per model/state (if method=optuna).")
    parser.add_argument("--save-path", type=str, default="examples/stacked_states_real.png", help="Where to save the figure.")
    parser.add_argument("--include", type=str, nargs="*", default=None, help="Include models by substring filter (optional).")
    parser.add_argument("--exclude", type=str, nargs="*", default=None, help="Exclude models by substring filter (optional).")
    parser.add_argument("--order", type=str, nargs="*", default=None, help="Display order by exact/substring priority (optional).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X = load_data(args.input, args.index_col)
    feature = args.feature or X.columns[0]

    # Default hyperparameter specifications
    if args.method == "optuna":
        hyperparams = {
            "kappa_min": 1.0,
            "kappa_max": 6.0,
            "jump_penalty_min": 0.1,
            "jump_penalty_max": 100,
        }
    else:  # grid
        hyperparams = {
            "kappa_range": (1.0, 6.0, 4),            # 4 points between 1 and 6
            "jump_penalty_range": (0.1, 100, 4),    # 4 points log/lin spaced inside implementation
        }

    print("Fitting models on unlabeled data...")
    res = fit_on_real_data(
        X=X,
        models=args.models,
        n_components_range=args.states,
        optimize_metric=args.metric,
        optimization_method=args.method,
        n_trials=args.trials,
        hyperparameter_grid=hyperparams,
        n_jobs=-1,
        save_models=False,
        output_dir=None,
        verbose=True,
    )

    print("Creating stacked plot (no true labels)...")
    fig = plot_stacked_states_from_results(
        X=X,
        results=res,
        feature_to_plot=feature,
        include_models=args.include,
        exclude_models=args.exclude,
        order=args.order,
        save_path=args.save_path,
        figsize=(12, 8),
    )
    print(f"Saved figure to: {args.save_path}")


if __name__ == "__main__":
    main()
