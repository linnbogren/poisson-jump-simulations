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

