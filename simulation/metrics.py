"""
Evaluation Metrics for Poisson Jump Simulations

This module provides all evaluation metrics used in the simulation study.

Key metrics:
- Balanced Accuracy with label permutation
- Persistence reliability (jump count accuracy)
- Chamfer distance (breakpoint alignment)
- Feature selection metrics (precision, recall, F1)
- Poisson deviance
- Composite scores
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from typing import Tuple, List, Dict, Optional
from itertools import permutations

# Suppress sklearn warnings about y_pred containing classes not in y_true
# This is expected during hyperparameter search when models don't identify all states
warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true', category=UserWarning)


###############################################################################
# State Sequence Evaluation
###############################################################################

def compute_bac_best_permutation(y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 return_permuted: bool = False):
    """
    Compute balanced accuracy with best label permutation.
    
    Since clustering/segmentation algorithms can assign arbitrary labels to states,
    we need to find the permutation of predicted labels that maximizes agreement
    with true labels.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True state labels
    y_pred : np.ndarray
        Predicted state labels
    return_permuted : bool
        If True, also return the permuted predictions
        
    Returns:
    --------
    float or tuple
        If return_permuted=False: best BAC score
        If return_permuted=True: (best BAC score, permuted predictions)
    """
    import warnings
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get unique labels
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # If different number of states, compute BAC with available states
    if len(unique_true) != len(unique_pred):
        # Fall back to standard BAC (will be suboptimal but won't crash)
        # Suppress the sklearn warning about mismatched classes
        # TODO: Re-enable warnings after testing
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
        bac = balanced_accuracy_score(y_true, y_pred)
        if return_permuted:
            return bac, y_pred
        return bac
    
    K = len(unique_true)
    
    # Try all permutations of predicted labels
    best_bac = 0.0
    best_y_pred_perm = y_pred.copy()
    
    for perm in permutations(range(K)):
        # Create mapping from predicted labels to permuted labels
        label_map = {unique_pred[i]: unique_true[perm[i]] for i in range(K)}
        
        # Remap predicted labels
        y_pred_remapped = np.array([label_map.get(label, label) for label in y_pred])
        
        # Compute BAC for this permutation
        bac = balanced_accuracy_score(y_true, y_pred_remapped)
        
        if bac > best_bac:
            best_bac = bac
            best_y_pred_perm = y_pred_remapped.copy()
    
    if return_permuted:
        return best_bac, best_y_pred_perm
    return best_bac


def compute_breakpoint_error(true_states: np.ndarray,
                                    predicted_states: np.ndarray) -> Tuple[int, int, int]:
    """
    Compute breakpoint error: |J_est - J_true|.
    
    Parameters:
    -----------
    true_states : np.ndarray
        True state sequence.
    predicted_states : np.ndarray
        Predicted state sequence.
        
    Returns:
    --------
    n_jumps_true : int
        Number of true state changes.
    n_jumps_estimated : int
        Number of estimated state changes.
    breakpoint_error : int
        Absolute difference.
    """
    true_jumps = np.sum(true_states[:-1] != true_states[1:])
    pred_jumps = np.sum(predicted_states[:-1] != predicted_states[1:])
    
    return int(true_jumps), int(pred_jumps), int(abs(true_jumps - pred_jumps))


###############################################################################
# Breakpoint Evaluation
###############################################################################

def extract_breakpoints(state_sequence: np.ndarray) -> np.ndarray:
    """
    Extract breakpoint indices from a state sequence.
    
    A breakpoint occurs at index i when state[i] != state[i+1].
    
    Parameters:
    -----------
    state_sequence : np.ndarray
        Sequence of state labels.
        
    Returns:
    --------
    np.ndarray
        Indices where state changes occur.
    """
    # Find where state changes occur
    changes = np.where(state_sequence[:-1] != state_sequence[1:])[0]
    # The breakpoint is at the index before the change
    return changes


def compute_chamfer_distance(true_breakpoints: np.ndarray,
                             estimated_breakpoints: np.ndarray) -> float:
    """
    Compute Chamfer distance between true and estimated breakpoints.
    
    The Chamfer distance measures the average minimum distance from each point
    in one set to the nearest point in the other set, in both directions.
    
    CD(A, B) = mean(min_b∈B ||a - b||) + mean(min_a∈A ||b - a||)
    
    Lower values indicate better alignment of breakpoints.
    
    Parameters:
    -----------
    true_breakpoints : np.ndarray
        Indices of true state change points (breakpoints).
    estimated_breakpoints : np.ndarray
        Indices of estimated state change points.
        
    Returns:
    --------
    float
        Chamfer distance. Returns 0.0 if both sets are empty.
    """
    # Handle edge cases
    if len(true_breakpoints) == 0 and len(estimated_breakpoints) == 0:
        return 0.0
    if len(true_breakpoints) == 0:
        return float('inf')  # No true breakpoints but model found some
    if len(estimated_breakpoints) == 0:
        return float('inf')  # True breakpoints exist but none were found
    
    # Convert to numpy arrays if needed
    true_bp = np.asarray(true_breakpoints, dtype=float)
    est_bp = np.asarray(estimated_breakpoints, dtype=float)
    
    # For each true breakpoint, find distance to nearest estimated breakpoint
    distances_true_to_est = []
    for t in true_bp:
        min_dist = np.min(np.abs(est_bp - t))
        distances_true_to_est.append(min_dist)
    
    # For each estimated breakpoint, find distance to nearest true breakpoint
    distances_est_to_true = []
    for e in est_bp:
        min_dist = np.min(np.abs(true_bp - e))
        distances_est_to_true.append(min_dist)
    
    # Chamfer distance is the mean of both directional distances
    chamfer_dist = np.mean(distances_true_to_est) + np.mean(distances_est_to_true)
    
    return float(chamfer_dist)


def compute_breakpoint_f1(true_breakpoints: np.ndarray,
                          estimated_breakpoints: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute F1 score based on breakpoint count accuracy.
    
    This metric evaluates how well the model predicts the NUMBER of breakpoints,
    without considering their temporal positions. It treats breakpoint detection
    as a count prediction problem.
    
    F1_BP = 1 - |n_true - n_est| / max(n_true, n_est)
    
    This gives:
    - F1_BP = 1.0 when counts match perfectly
    - F1_BP = 0.0 when one count is zero and the other is non-zero
    - F1_BP decreases linearly with absolute count error
    
    Precision and Recall are set equal to F1_BP for consistency.
    
    Parameters:
    -----------
    true_breakpoints : np.ndarray
        Indices of true state change points (breakpoints).
    estimated_breakpoints : np.ndarray
        Indices of estimated state change points.
        
    Returns:
    --------
    Tuple[float, float, float]
        (F1_BP, Precision_BP, Recall_BP)
        All values in range [0, 1], where 1.0 is perfect.
    """
    n_true = len(true_breakpoints)
    n_est = len(estimated_breakpoints)
    
    # Handle edge case: both zero
    if n_true == 0 and n_est == 0:
        return 1.0, 1.0, 1.0  # Perfect - no breakpoints to detect
    
    # Compute count-based F1 score
    max_count = max(n_true, n_est)
    if max_count == 0:
        f1_bp = 1.0
    else:
        f1_bp = 1.0 - abs(n_true - n_est) / max_count
    
    # Set precision and recall equal to F1 for consistency
    # (since we're not doing position-based matching)
    precision_bp = f1_bp
    recall_bp = f1_bp
    
    return float(f1_bp), float(precision_bp), float(recall_bp)


###############################################################################
# Composite Scores
###############################################################################

def compute_composite_score(bac: float, f1_bp: float) -> float:
    """
    Compute Composite Segmentation Score (CSS).
    
    Combines state-wise classification performance (BAC) with event-wise
    detection accuracy (F1_BP) using harmonic mean. This imposes a strict
    penalty structure: the model must achieve high performance in BOTH domains.
    
    CSS = 2 * (BAC * F1_BP) / (BAC + F1_BP)
    
    Parameters:
    -----------
    bac : float
        Balanced Accuracy score [0, 1].
    f1_bp : float
        Breakpoint F1 score [0, 1].
        
    Returns:
    --------
    float
        Composite Segmentation Score in range [0, 1], where 1.0 is perfect.
    """
    if bac + f1_bp == 0:
        return 0.0
    
    css = 2 * (bac * f1_bp) / (bac + f1_bp)
    return float(css)


###############################################################################
# Feature Selection Metrics
###############################################################################

def compute_feature_selection_metrics(selected_features: List[int],
                                      n_informative: int,
                                      n_total: int) -> Dict[str, float]:
    """
    Compute feature selection quality metrics.
    
    Parameters:
    -----------
    selected_features : List[int]
        Indices of selected features (0-indexed).
    n_informative : int
        Number of truly informative features (first n_informative indices).
    n_total : int
        Total number of features.
        
    Returns:
    --------
    dict
        Dictionary with precision, recall, f1, and n_selected_noise.
    """
    # Ground truth: first n_informative features are informative
    true_informative = set(range(n_informative))
    selected_set = set(selected_features)
    
    # True positives, false positives, false negatives
    tp = len(selected_set & true_informative)
    fp = len(selected_set - true_informative)
    fn = len(true_informative - selected_set)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_selected_noise': fp
    }


def get_selected_features(model) -> List[int]:
    """
    Extract selected features from a fitted model.
    
    Parameters:
    -----------
    model : fitted model
        Model with feat_weights attribute (SparseJumpModel).
        
    Returns:
    --------
    List[int]
        Indices of features with non-zero weights.
    """
    if hasattr(model, 'feat_weights'):
        weights = model.feat_weights.values if hasattr(model.feat_weights, 'values') else model.feat_weights
        return np.where(weights > 1e-10)[0].tolist()
    else:
        # Non-sparse model: all features selected
        return list(range(model.centers_.shape[1]))


###############################################################################
# Unsupervised / Label-Free Metrics (for real data)
###############################################################################

def compute_bic(model, X: np.ndarray, predicted_states: np.ndarray) -> float:
    """
    Compute Bayesian Information Criterion (BIC) for model selection.
    
    BIC penalizes model complexity more strongly than AIC and is preferred
    for segmentation to avoid over-segmentation.
    
    BIC = k * ln(n) - 2 * ln(L_hat)
    
    where:
    - n: number of observations (time steps)
    - L_hat: maximized likelihood (Poisson likelihood)
    - k: number of free parameters
        k = K * P_active + (K-1) + |J|
        - K: number of states
        - P_active: number of non-zero weighted features
        - (K-1): state proportions
        - |J|: number of detected jumps
    
    Lower BIC is better.
    
    Parameters
    ----------
    model : fitted model
        Model with centers_ and feat_weights attributes
    X : np.ndarray
        Data matrix (n_samples, n_features)
    predicted_states : np.ndarray
        Predicted state sequence
        
    Returns
    -------
    float
        BIC value (lower is better)
        
    """
    n = len(X)  # number of time steps
    
    # Get model parameters
    K = len(np.unique(predicted_states))  # number of states
    P_active = len(get_selected_features(model))  # active features
    n_jumps = np.sum(predicted_states[:-1] != predicted_states[1:])  # detected jumps
    
    # Count free parameters
    k = K * P_active + (K - 1) + n_jumps
    
    # Compute log-likelihood using the model's actual distribution
    log_likelihood = _compute_log_likelihood(model, X, predicted_states)
    
    # BIC = k * ln(n) - 2 * ln(L)
    bic = k * np.log(n) - 2 * log_likelihood
    
    return float(bic)


def compute_aic(model, X: np.ndarray, predicted_states: np.ndarray) -> float:
    """
    Compute Akaike Information Criterion (AIC) for model selection.
    
    AIC has a lighter complexity penalty than BIC and may be more sensitive
    in high-noise regimes.
    
    AIC = 2k - 2 * ln(L_hat)
    
    where k is the number of free parameters (same as BIC) and L_hat is
    the maximized likelihood.
    
    Lower AIC is better.
    
    Parameters
    ----------
    model : fitted model
        Model with centers_ and feat_weights attributes
    X : np.ndarray
        Data matrix (n_samples, n_features)
    predicted_states : np.ndarray
        Predicted state sequence
        
    Returns
    -------
    float
        AIC value (lower is better)
        
    """
    # Get model parameters
    K = len(np.unique(predicted_states))
    P_active = len(get_selected_features(model))
    n_jumps = np.sum(predicted_states[:-1] != predicted_states[1:])
    
    # Count free parameters
    k = K * P_active + (K - 1) + n_jumps
    
    # Compute log-likelihood using the model's actual distribution
    log_likelihood = _compute_log_likelihood(model, X, predicted_states)
    
    # AIC = 2k - 2 * ln(L)
    aic = 2 * k - 2 * log_likelihood
    
    return float(aic)


def compute_silhouette_coefficient(model, X: np.ndarray, 
                                   predicted_states: np.ndarray,
                                   distribution: str = "Poisson") -> float:
    """
    Compute mean Silhouette Coefficient for state clustering quality.
    
    Treats inferred states as clusters and assesses within-state cohesion
    versus between-state separation using the model's dissimilarity measure.
    
    For each time point t with state s_t:
        s(t) = (b(t) - a(t)) / max(a(t), b(t))
    
    where:
    - a(t): average dissimilarity to other points in same state
    - b(t): minimum average dissimilarity to points in any other state
    
    We use a centroid-based approximation to reduce O(T^2) complexity,
    computing distances to state centroids instead of all pairwise distances.
    
    Higher silhouette (closer to 1.0) indicates more distinctive, coherent states.
    
    Parameters
    ----------
    model : fitted model
        Model with centers_ and feat_weights attributes
    X : np.ndarray
        Data matrix (n_samples, n_features)
    predicted_states : np.ndarray
        Predicted state sequence
    distribution : str, default="Poisson"
        Distribution type: "Gaussian", "Poisson", or "PoissonKL"
        Determines which dissimilarity measure to use
        
    Returns
    -------
    float
        Mean silhouette coefficient in [-1, 1] (higher is better)
        
    """
    X = np.asarray(X)
    predicted_states = np.asarray(predicted_states)
    
    # Get centroids and weights from model
    centroids = model.centers_
    if hasattr(model, 'feat_weights'):
        weights = model.feat_weights.values if hasattr(model.feat_weights, 'values') else model.feat_weights
    else:
        weights = np.ones(X.shape[1])
    
    unique_states = np.unique(predicted_states)
    n_states = len(unique_states)
    
    if n_states <= 1:
        # Silhouette undefined for single cluster
        return 0.0
    
    silhouette_scores = []
    
    for t in range(len(X)):
        y_t = X[t]
        state_t = predicted_states[t]
        
        # Compute distance to own centroid (a(t))
        state_idx = np.where(unique_states == state_t)[0][0]
        a_t = _compute_weighted_dissimilarity(y_t, centroids[state_idx], weights, distribution)
        
        # Compute distances to other state centroids (for b(t))
        other_distances = []
        for i, state in enumerate(unique_states):
            if state != state_t:
                dist = _compute_weighted_dissimilarity(y_t, centroids[i], weights, distribution)
                other_distances.append(dist)
        
        if len(other_distances) == 0:
            b_t = 0.0
        else:
            b_t = min(other_distances)
        
        # Compute silhouette for this point
        if max(a_t, b_t) == 0:
            s_t = 0.0
        else:
            s_t = (b_t - a_t) / max(a_t, b_t)
        
        silhouette_scores.append(s_t)
    
    return float(np.mean(silhouette_scores))


###############################################################################
# Helper Functions for Unsupervised Metrics
###############################################################################

def _compute_log_likelihood(model, X: np.ndarray, 
                           predicted_states: np.ndarray) -> float:
    """
    Compute log-likelihood for the fitted model using its distribution type.
    
    Returns the maximized log-likelihood (ignoring additive constants).
    
    Note: This is used for BIC/AIC computation. Uses the model's centroids
    as the fitted parameters.
    """
    # Determine the model's distribution type
    distribution = getattr(model, 'distribution', 'Poisson')
    
    if distribution == 'Gaussian':
        return _compute_gaussian_log_likelihood(model, X, predicted_states)
    else:
        # Both Poisson and PoissonKL use Poisson likelihood
        # (PoissonKL has additional KL penalty in training, but likelihood is still Poisson)
        return _compute_poisson_log_likelihood(model, X, predicted_states)


def _compute_poisson_log_likelihood(model, X: np.ndarray, 
                                    predicted_states: np.ndarray) -> float:
    """
    Compute Poisson log-likelihood for the fitted model.
    
    Returns the maximized log-likelihood (ignoring additive constants).
    
    Note: This is used for BIC/AIC computation. Uses the model's centroids
    as the fitted parameters.
    """
    X = np.asarray(X)
    predicted_states = np.asarray(predicted_states)
    
    # Get centroids and weights
    centroids = model.centers_
    if hasattr(model, 'feat_weights'):
        weights = model.feat_weights.values if hasattr(model.feat_weights, 'values') else model.feat_weights
    else:
        weights = np.ones(X.shape[1])
    
    log_likelihood = 0.0
    unique_states = np.unique(predicted_states)
    
    for t in range(len(X)):
        y_t = X[t]
        state_t = predicted_states[t]
        state_idx = np.where(unique_states == state_t)[0][0]
        mu_t = centroids[state_idx]
        
        # Poisson log-likelihood (ignoring factorial constant):
        # ln(L) = sum_f w_f * (y_f * ln(mu_f) - mu_f)
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        mu_safe = np.maximum(mu_t, epsilon)
        
        ll_t = np.sum(weights * (y_t * np.log(mu_safe) - mu_t))
        log_likelihood += ll_t
    
    return float(log_likelihood)


def _compute_gaussian_log_likelihood(model, X: np.ndarray, 
                                     predicted_states: np.ndarray) -> float:
    """
    Compute Gaussian log-likelihood for the fitted model.
    
    Returns the maximized log-likelihood (ignoring additive constants).
    """
    X = np.asarray(X)
    predicted_states = np.asarray(predicted_states)
    
    # Get centroids (means) and weights
    centroids = model.centers_
    if hasattr(model, 'feat_weights'):
        weights = model.feat_weights.values if hasattr(model.feat_weights, 'values') else model.feat_weights
    else:
        weights = np.ones(X.shape[1])
    
    # Estimate variance for each state
    unique_states = np.unique(predicted_states)
    variances = np.zeros_like(centroids)
    
    for state_idx, state in enumerate(unique_states):
        state_mask = predicted_states == state
        if np.sum(state_mask) > 0:
            X_state = X[state_mask]
            mu_state = centroids[state_idx]
            # Weighted variance estimation
            variances[state_idx] = np.sum(weights * np.var(X_state, axis=0)) / np.sum(weights)
            # Avoid zero variance
            variances[state_idx] = max(variances[state_idx], 1e-6)
    
    log_likelihood = 0.0
    
    for t in range(len(X)):
        y_t = X[t]
        state_t = predicted_states[t]
        state_idx = np.where(unique_states == state_t)[0][0]
        mu_t = centroids[state_idx]
        sigma2_t = variances[state_idx]
        
        # Gaussian log-likelihood (ignoring constant term):
        # ln(L) = -0.5 * sum_f w_f * ((y_f - mu_f)^2 / sigma^2 + ln(sigma^2))
        ll_t = -0.5 * np.sum(weights * (((y_t - mu_t)**2 / sigma2_t) + np.log(sigma2_t)))
        log_likelihood += ll_t
    
    return float(log_likelihood)


def _compute_weighted_dissimilarity(y: np.ndarray, mu: np.ndarray, 
                                     weights: np.ndarray, 
                                     distribution: str = "Poisson") -> float:
    """
    Compute weighted dissimilarity between observation and centroid.
    
    This function unifies all distribution-specific distance computations
    for use in silhouette coefficient and other metrics.
    
    Parameters
    ----------
    y : np.ndarray
        Observation vector
    mu : np.ndarray
        Centroid vector
    weights : np.ndarray
        Feature weights
    distribution : str, default="Poisson"
        Distribution type: "Gaussian", "Poisson", or "PoissonKL"
        
    Returns
    -------
    float
        Weighted dissimilarity measure
        
    Notes
    -----
    For Gaussian: Uses weighted squared Euclidean distance
    For Poisson: Uses weighted Poisson negative log-likelihood
    For PoissonKL: Uses weighted KL divergence between Poisson distributions
    """
    epsilon = 1e-10
    distribution = distribution.lower()
    
    if distribution == "gaussian":
        # Weighted squared Euclidean distance
        # d(y, mu) = sum_f w_f * (y_f - mu_f)^2
        squared_diff = (y - mu) ** 2
        return float(np.sum(weights * squared_diff))
    
    elif distribution == "poisson":
        # Weighted Poisson negative log-likelihood (ignoring constants)
        # d(y, mu) = sum_f w_f * (mu_f - y_f * ln(mu_f))
        mu_safe = np.maximum(mu, epsilon)
        nll = mu_safe - y * np.log(mu_safe)
        return float(np.sum(weights * nll))
    
    elif distribution == "poissonkl":
        # Weighted KL divergence: D_KL(Pois(y) || Pois(mu))
        # d(y, mu) = sum_f w_f * (y_f * ln(y_f / mu_f) + mu_f - y_f)
        mu_safe = np.maximum(mu, epsilon)
        y_safe = np.maximum(y, epsilon)
        kl = y_safe * np.log(y_safe / mu_safe) + mu_safe - y_safe
        return float(np.sum(weights * kl))
    
    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            f"Must be one of: 'Gaussian', 'Poisson', 'PoissonKL'"
        )

