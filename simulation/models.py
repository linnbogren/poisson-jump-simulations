"""
Model Wrapper and Evaluation for Sparse Jump Model Simulations

This module provides a unified interface for working with Sparse Jump Models
and evaluating them against ground truth.

Key components:
- ModelWrapper: Unified interface for model fitting and evaluation
- Model evaluation: Comprehensive metric computation
- Result conversion: Convert model outputs to result dataclasses
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict

from jumpmodels.sparse_jump import SparseJumpModel

from .config import SimulationConfig, GridSearchResult, ReplicationResult
from .metrics import (
    compute_bac_best_permutation,
    compute_breakpoint_error,
    extract_breakpoints,
    compute_chamfer_distance,
    compute_breakpoint_f1,
    compute_composite_score,
    compute_feature_selection_metrics,
    get_selected_features,
    # Unsupervised metrics
    compute_bic,
    compute_aic,
    compute_silhouette_coefficient
)



class ModelWrapper:
    """
    Wrapper class for Sparse Jump Models providing unified interface.
    
    This class handles:
    - Model initialization with hyperparameters
    - Fitting with timing
    - Prediction and state extraction
    - Feature selection extraction
    - Comprehensive evaluation against ground truth
    
    Parameters
    ----------
    model_name : str
        Distribution type: "Gaussian", "Poisson", or "PoissonKL"
    n_components : int
        Number of hidden states
    max_feats : float
        Maximum number of features for selection (κ²)
    jump_penalty : float
        Penalty for state transitions (λ)
    max_iter : int, default=50
        Maximum iterations for EM algorithm
    tol_w : float, default=1e-4
        Convergence tolerance for feature weights
    n_init_jm : int, default=10
        Number of initializations for jump model
    random_state : int, optional
        Random seed for reproducibility
    verbose : int, default=0
        Verbosity level (0 = silent, 1 = progress, 2 = detailed)
        
    Attributes
    ----------
    model : SparseJumpModel
        The underlying fitted model
    fit_time : float
        Time taken to fit the model (seconds)
    """
    
    def __init__(
        self,
        model_name: str,
        n_components: int,
        max_feats: float,
        jump_penalty: float,
        max_iter: int = 50,
        tol_w: float = 1e-4,
        n_init_jm: int = 10,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.model_name = model_name
        self.hyperparameters = {
            'n_components': n_components,
            'max_feats': max_feats,
            'jump_penalty': jump_penalty
        }
        
        self.model = SparseJumpModel(
            n_components=n_components,
            max_feats=max_feats,
            jump_penalty=jump_penalty,
            distribution=model_name,
            max_iter=max_iter,
            tol_w=tol_w,
            n_init_jm=n_init_jm,
            verbose=verbose,
            random_state=random_state
        )
        
        self.fit_time = None
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'ModelWrapper':
        """
        Fit the model to data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data (n_samples, n_features)
        
        Returns
        -------
        self : ModelWrapper
            Fitted model wrapper
            
        Raises
        ------
        Exception
            If model fitting fails
        """
        start_time = time.time()
        self.model.fit(X)
        self.fit_time = time.time() - start_time
        self._is_fitted = True
        return self
    
    def get_states(self) -> np.ndarray:
        """
        Get predicted states from fitted model.
        
        Returns
        -------
        np.ndarray
            Predicted state sequence
            
        Raises
        ------
        RuntimeError
            If model hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting states")
        
        # Handle both Series and array returns from model
        labels = self.model.labels_
        return labels.values if hasattr(labels, 'values') else labels
    
    def get_selected_features(self) -> List[int]:
        """
        Get indices of selected features.
        
        Returns
        -------
        List[int]
            Indices of features selected by the model
            
        Raises
        ------
        RuntimeError
            If model hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting features")
        
        return get_selected_features(self.model)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        true_states: np.ndarray,
        config: SimulationConfig,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate fitted model against ground truth.
        
        Computes all metrics defined in the simulation study:
        - Balanced accuracy (BAC) with label permutation
        - Breakpoint detection (F1, precision, recall, Chamfer distance)
        - Feature selection (F1, precision, recall)
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data
        true_states : np.ndarray
            Ground truth state sequence
        config : SimulationConfig
            Data generation configuration
        return_predictions : bool, default=False
            If True, include predicted states in returned dict
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all computed metrics
            
        Raises
        ------
        RuntimeError
            If model hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Get predictions
        pred_states = self.get_states()
        
        # Segmentation metrics
        bac, pred_perm = compute_bac_best_permutation(
            true_states, pred_states, return_permuted=True
        )
        
        # Breakpoint metrics
        true_breakpoints = extract_breakpoints(true_states)
        estimated_breakpoints = extract_breakpoints(pred_perm)
        
        bp_f1, bp_precision, bp_recall = compute_breakpoint_f1(
            true_breakpoints, estimated_breakpoints
        )
        chamfer_dist = compute_chamfer_distance(true_breakpoints, estimated_breakpoints)
        
        # Jump/breakpoint count metrics
        n_jumps_true, n_jumps_est, breakpoint_error = compute_breakpoint_error(
            true_states, pred_perm
        )
        
        # Composite score
        css = compute_composite_score(bac, bp_f1)
        
        # Feature selection metrics
        selected_features = self.get_selected_features()
        feat_metrics = compute_feature_selection_metrics(
            selected_features, config.n_informative, config.n_total_features
        )
        
        # Unsupervised / label-free metrics (for real data without ground truth)
        # Use appropriate distribution for silhouette coefficient
        bic = compute_bic(self.model, X, pred_perm)
        aic = compute_aic(self.model, X, pred_perm)
        silhouette = compute_silhouette_coefficient(self.model, X, pred_perm, distribution=self.model_name)
        
        # Compile results
        results = {
            'model_name': self.model_name,
            'hyperparameters': self.hyperparameters.copy(),
            'fit_time': self.fit_time,
            # Segmentation
            'balanced_accuracy': float(bac),
            'n_jumps_true': int(n_jumps_true),
            'n_jumps_estimated': int(n_jumps_est),
            'breakpoint_error': int(breakpoint_error),
            # Breakpoints
            'n_breakpoints_true': len(true_breakpoints),
            'n_breakpoints_estimated': len(estimated_breakpoints),
            'breakpoint_count_error': abs(len(true_breakpoints) - len(estimated_breakpoints)),
            'breakpoint_f1': float(bp_f1),
            'breakpoint_precision': float(bp_precision),
            'breakpoint_recall': float(bp_recall),
            'chamfer_distance': float(chamfer_dist),
            # Composite
            'composite_score': float(css),
            # Feature selection
            'feature_f1': float(feat_metrics['f1']),
            'feature_precision': float(feat_metrics['precision']),
            'feature_recall': float(feat_metrics['recall']),
            'n_selected_noise': int(feat_metrics['n_selected_noise']),
            'n_selected_total': len(selected_features),
            # Unsupervised metrics (for real data)
            'bic': float(bic),
            'aic': float(aic),
            'silhouette': float(silhouette),
        }
        
        # Optionally include predictions
        if return_predictions:
            results['true_states'] = true_states
            results['predicted_states'] = pred_states
            results['predicted_states_permuted'] = pred_perm
            results['selected_features'] = selected_features
        
        return results
    
    def evaluate_unsupervised(
        self,
        X: pd.DataFrame,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate fitted model using only unsupervised metrics.
        
        This method is for real data where ground truth labels are not available.
        Computes model selection criteria (BIC, AIC) and clustering quality (Silhouette).
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data
        return_predictions : bool, default=False
            If True, include predicted states and selected features in returned dict
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing unsupervised metrics:
            - bic: Bayesian Information Criterion (lower is better)
            - aic: Akaike Information Criterion (lower is better)
            - silhouette: Silhouette coefficient (higher is better, range [-1, 1])
            - n_breakpoints_estimated: Number of detected breakpoints
            - n_selected_total: Number of selected features
            
        Raises
        ------
        RuntimeError
            If model hasn't been fitted yet
            
        Examples
        --------
        >>> wrapper = ModelWrapper('Poisson', n_components=3, max_feats=10, jump_penalty=1.0)
        >>> wrapper.fit(X_real)
        >>> results = wrapper.evaluate_unsupervised(X_real)
        >>> print(f"BIC: {results['bic']:.2f}, Silhouette: {results['silhouette']:.3f}")
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Get predictions
        pred_states = self.get_states()
        
        # Count breakpoints/jumps
        estimated_breakpoints = extract_breakpoints(pred_states)
        n_jumps_est = len(estimated_breakpoints)
        
        # Feature selection
        selected_features = self.get_selected_features()
        
        # Unsupervised metrics
        bic = compute_bic(self.model, X, pred_states)
        aic = compute_aic(self.model, X, pred_states)
        silhouette = compute_silhouette_coefficient(
            self.model, X, pred_states, distribution=self.model_name
        )
        
        # Compile results
        results = {
            'model_name': self.model_name,
            'hyperparameters': self.hyperparameters.copy(),
            'fit_time': self.fit_time,
            # Unsupervised metrics
            'bic': float(bic),
            'aic': float(aic),
            'silhouette': float(silhouette),
            # Descriptive statistics
            'n_jumps_estimated': int(n_jumps_est),
            'n_breakpoints_estimated': len(estimated_breakpoints),
            'n_selected_total': len(selected_features),
        }
        
        # Optionally include predictions
        if return_predictions:
            results['predicted_states'] = pred_states
            results['selected_features'] = selected_features
        
        return results


def fit_and_evaluate(
    X: pd.DataFrame,
    true_states: np.ndarray,
    config: SimulationConfig,
    model_name: str,
    hyperparameters: Dict[str, Any],
    return_model: bool = False,
    trial_number: Optional[int] = None, # THESE TWO WERE REMOVED?
    replication: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a single model and evaluate it.
    
    Convenience function that creates a ModelWrapper, fits it, and evaluates.
    Useful for grid search and parallel processing.
    
    Parameters
    ----------
    X : pd.DataFrame
        Time series data
    true_states : np.ndarray
        Ground truth state sequence
    config : SimulationConfig
        Data generation configuration
    model_name : str
        Distribution type: "Gaussian", "Poisson", or "PoissonKL"
    hyperparameters : Dict[str, Any]
        Model hyperparameters (n_components, max_feats, jump_penalty)
    return_model : bool, default=False
        If True, include fitted model in returned dict
    trial_number : int, optional
        Optuna trial number (for warning logging)
    replication : int, optional
        Replication number (for warning logging)
    Returns
    -------
    Dict[str, Any]
        Evaluation results. If return_model=True, includes 'model' key.
        If fitting fails, returns dict with 'success': False
        
    Examples
    --------
    >>> results = fit_and_evaluate(
    ...     X, states, config, "Poisson",
    ...     {'n_components': 3, 'max_feats': 10, 'jump_penalty': 1.0}
    ... )
    >>> print(results['balanced_accuracy'])
     >>> print(f"Warnings: {len(results['warning_capture'].warnings)}") # THIS AND THE CONFIGURATION WAS REMOVED
    """
    # Prepare data config for warning logging
    data_config_dict = {
        'delta': config.delta,
        'n_samples': config.n_samples,
        'n_states': config.n_states,
        'n_informative': config.n_informative,
        'n_total_features': config.n_total_features,
    }
    try:
        # Create and fit model
        wrapper = ModelWrapper(
            model_name=model_name,
            n_components=hyperparameters['n_components'],
            max_feats=hyperparameters['max_feats'],
            jump_penalty=hyperparameters['jump_penalty'],
            random_state=config.random_seed,
            verbose=0
        )
        
        wrapper.fit(X)
        
        # Evaluate
        results = wrapper.evaluate(X, true_states, config)
        results['success'] = True
        results['convergence_failed'] = False
        
        if return_model:
            results['model'] = wrapper.model
            results['wrapper'] = wrapper
        
        return results
        
    except ValueError as e:
        # Catch specific convergence/initialization failures from the jump model
        error_msg = str(e)
        if 'initialization' in error_msg.lower() or 'converge' in error_msg.lower() or 'finite loss' in error_msg.lower():
            # This is an expected convergence failure - return failure marker
            return {
                'success': False,
                'convergence_failed': True,
                'error': error_msg,
                'error_type': 'ConvergenceError',
                'model_name': model_name,
                'hyperparameters': hyperparameters,
            }
        else:
            # Unexpected ValueError - re-raise it
            raise


def results_to_grid_search_result(
    results: Dict[str, Any],
    config: SimulationConfig
) -> GridSearchResult:
    """
    Convert evaluation results to GridSearchResult dataclass.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Evaluation results from fit_and_evaluate or ModelWrapper.evaluate
    config : SimulationConfig
        Data generation configuration
    
    Returns
    -------
    GridSearchResult
        Structured result object
    """
    return GridSearchResult(
        config=config,
        model_name=results['model_name'],
        hyperparameters=results['hyperparameters'],
        balanced_accuracy=results['balanced_accuracy'],
        accuracy=results.get('accuracy', results['balanced_accuracy']),  # Use BAC if accuracy not computed
        n_jumps_true=results['n_jumps_true'],
        n_jumps_estimated=results['n_jumps_estimated'],
        breakpoint_error=results['breakpoint_error'],
        n_breakpoints_true=results['n_breakpoints_true'],
        n_breakpoints_estimated=results['n_breakpoints_estimated'],
        breakpoint_count_error=results['breakpoint_count_error'],
        chamfer_distance=results['chamfer_distance'],
        composite_score=results['composite_score'],
        breakpoint_f1=results['breakpoint_f1'],
        breakpoint_precision=results['breakpoint_precision'],
        breakpoint_recall=results['breakpoint_recall'],
        feature_f1=results['feature_f1'],
        feature_precision=results['feature_precision'],
        feature_recall=results['feature_recall'],
        n_selected_noise=results['n_selected_noise'],
        n_selected_total=results['n_selected_total'],
        computation_time=results['fit_time'],
        # Unsupervised metrics
        bic=results.get('bic', None),
        aic=results.get('aic', None),
        silhouette=results.get('silhouette', None)
    )


def grid_results_to_dataframe(results: List[GridSearchResult]) -> pd.DataFrame:
    """
    Convert list of grid search results to pandas DataFrame.
    
    SINGLE SOURCE - consolidates implementations from:
    - simulation_runner_parallel.py
    - simulation_runner.py
    
    Parameters
    ----------
    results : List[GridSearchResult]
        List of grid search results
    
    Returns
    -------
    pd.DataFrame
        Flattened results with one row per result
    """
    rows = []
    for result in results:
        row = {
            # Config parameters
            'n_samples': result.config.n_samples,
            'n_states': result.config.n_states,
            'n_informative': result.config.n_informative,
            'n_noise': result.config.n_noise,
            'n_total_features': result.config.n_total_features,
            'delta': result.config.delta,
            'lambda_0': result.config.lambda_0,
            'persistence': result.config.persistence,
            'distribution_type': result.config.distribution_type,
            'correlated_noise': result.config.correlated_noise,
            'random_seed': result.config.random_seed,
            
            # Model
            'model_name': result.model_name,
            
            # Hyperparameters
            'n_components': result.hyperparameters['n_components'],
            'jump_penalty': result.hyperparameters['jump_penalty'],
            'max_feats': result.hyperparameters['max_feats'],
            
            # Results
            'balanced_accuracy': result.balanced_accuracy,
            'accuracy': result.accuracy,
            'n_jumps_true': result.n_jumps_true,
            'n_jumps_estimated': result.n_jumps_estimated,
            'breakpoint_error': result.breakpoint_error,
            'n_breakpoints_true': result.n_breakpoints_true,
            'n_breakpoints_estimated': result.n_breakpoints_estimated,
            'breakpoint_count_error': result.breakpoint_count_error,
            'chamfer_distance': result.chamfer_distance,
            'composite_score': result.composite_score,
            'breakpoint_f1': result.breakpoint_f1,
            'breakpoint_precision': result.breakpoint_precision,
            'breakpoint_recall': result.breakpoint_recall,
            'feature_f1': result.feature_f1,
            'feature_precision': result.feature_precision,
            'feature_recall': result.feature_recall,
            'n_selected_noise': result.n_selected_noise,
            'n_selected_total': result.n_selected_total,
            'computation_time': result.computation_time,
            # Unsupervised metrics
            'bic': result.bic,
            'aic': result.aic,
            'silhouette': result.silhouette,
        }
        rows.append(row)
    
    return pd.DataFrame(rows)
