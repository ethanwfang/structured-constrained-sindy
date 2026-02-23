"""
SINDy with cross-validated threshold selection.

This module provides hyperparameter-tuned SINDy that uses cross-validation
to automatically select the optimal STLS threshold, addressing reviewer
concerns about fixed threshold comparisons.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold

from .sindy import sindy_stls, DEFAULT_STLS_THRESHOLD


# Default threshold candidates for grid search
DEFAULT_THRESHOLDS = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]


def cross_validate_threshold(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    thresholds: Optional[List[float]] = None,
    n_folds: int = 5,
    max_iter: int = 10,
    metric: str = "reconstruction",
    random_state: int = 42,
) -> Tuple[float, dict]:
    """
    Find optimal STLS threshold via cross-validation.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    thresholds : List[float], optional
        Candidate thresholds to evaluate. Default: [0.01, 0.03, ..., 0.50]
    n_folds : int, optional
        Number of cross-validation folds (default: 5).
    max_iter : int, optional
        STLS iterations per fold (default: 10).
    metric : str, optional
        Selection metric: "reconstruction" (MSE) or "sparsity" (reconstruction + penalty).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    best_threshold : float
        Threshold with best cross-validation score.
    cv_results : dict
        Dictionary with per-threshold scores and other metrics.

    Examples
    --------
    >>> Theta = np.random.randn(500, 20)
    >>> x_dot = np.random.randn(500, 3)
    >>> best_thresh, cv_results = cross_validate_threshold(Theta, x_dot)
    >>> print(f"Best threshold: {best_thresh}")
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    n_samples = Theta.shape[0]

    results = {
        "thresholds": thresholds,
        "mean_scores": [],
        "std_scores": [],
        "mean_sparsity": [],
        "fold_scores": [],
    }

    for threshold in thresholds:
        fold_scores = []
        fold_sparsity = []

        for train_idx, val_idx in kf.split(Theta):
            Theta_train, Theta_val = Theta[train_idx], Theta[val_idx]
            x_dot_train, x_dot_val = x_dot[train_idx], x_dot[val_idx]

            # Fit on training fold
            xi, _ = sindy_stls(Theta_train, x_dot_train, threshold=threshold, max_iter=max_iter)

            # Evaluate on validation fold
            x_dot_pred = Theta_val @ xi.T
            mse = np.mean((x_dot_pred - x_dot_val) ** 2)

            # Compute sparsity (fraction of zero coefficients)
            sparsity = np.mean(np.abs(xi) < 1e-10)

            if metric == "reconstruction":
                score = -mse  # Higher is better
            elif metric == "sparsity":
                # Balance reconstruction and sparsity (like BIC)
                score = -mse - 0.01 * (1 - sparsity)  # Penalize non-sparse solutions
            else:
                score = -mse

            fold_scores.append(score)
            fold_sparsity.append(sparsity)

        results["mean_scores"].append(np.mean(fold_scores))
        results["std_scores"].append(np.std(fold_scores))
        results["mean_sparsity"].append(np.mean(fold_sparsity))
        results["fold_scores"].append(fold_scores)

    # Select best threshold
    best_idx = np.argmax(results["mean_scores"])
    best_threshold = thresholds[best_idx]

    results["best_threshold"] = best_threshold
    results["best_score"] = results["mean_scores"][best_idx]
    results["best_sparsity"] = results["mean_sparsity"][best_idx]

    return best_threshold, results


def sindy_tuned(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    thresholds: Optional[List[float]] = None,
    n_folds: int = 5,
    max_iter: int = 10,
    metric: str = "reconstruction",
    random_state: int = 42,
) -> Tuple[np.ndarray, float, dict]:
    """
    SINDy with cross-validated threshold selection.

    Automatically selects the optimal STLS threshold using k-fold
    cross-validation, then fits the final model on all data.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    thresholds : List[float], optional
        Candidate thresholds to evaluate.
    n_folds : int, optional
        Number of cross-validation folds (default: 5).
    max_iter : int, optional
        STLS iterations (default: 10).
    metric : str, optional
        Selection metric: "reconstruction" or "sparsity".
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xi : np.ndarray
        Coefficient matrix with shape [n_vars, n_terms].
    elapsed_time : float
        Total computation time in seconds.
    cv_results : dict
        Cross-validation results including best threshold.

    Examples
    --------
    >>> Theta = np.random.randn(500, 20)
    >>> x_dot = np.random.randn(500, 3)
    >>> xi, elapsed, cv_results = sindy_tuned(Theta, x_dot)
    >>> print(f"Selected threshold: {cv_results['best_threshold']}")
    """
    t_start = time.time()

    # Find best threshold via cross-validation
    best_threshold, cv_results = cross_validate_threshold(
        Theta, x_dot, thresholds, n_folds, max_iter, metric, random_state
    )

    # Fit final model with best threshold
    xi, _ = sindy_stls(Theta, x_dot, threshold=best_threshold, max_iter=max_iter)

    elapsed = time.time() - t_start
    cv_results["total_time"] = elapsed

    return xi, elapsed, cv_results


def get_system_class_threshold(
    system_class: str,
    threshold_lookup: Optional[dict] = None,
) -> float:
    """
    Get pre-computed optimal threshold for a system class.

    For fair comparison, thresholds should be tuned on TRAINING systems only,
    then applied to test systems of the same class.

    Parameters
    ----------
    system_class : str
        System class name (e.g., "oscillator", "chaotic", "biological").
    threshold_lookup : dict, optional
        Pre-computed thresholds per class. If None, returns default.

    Returns
    -------
    threshold : float
        Recommended threshold for this system class.
    """
    # Default thresholds tuned on training systems (placeholder values)
    default_lookup = {
        "oscillator": 0.10,  # Van der Pol, Duffing, etc.
        "chaotic": 0.08,     # Lorenz, Rossler (typically need lower threshold)
        "biological": 0.12,  # Lotka-Volterra, Glycolysis
        "polynomial": 0.10,  # Generic polynomial systems
        "default": 0.10,
    }

    if threshold_lookup is None:
        threshold_lookup = default_lookup

    return threshold_lookup.get(system_class, threshold_lookup.get("default", DEFAULT_STLS_THRESHOLD))


def sindy_with_class_threshold(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    system_class: str,
    threshold_lookup: Optional[dict] = None,
    max_iter: int = 10,
) -> Tuple[np.ndarray, float, float]:
    """
    SINDy with pre-computed class-based threshold.

    Uses thresholds that were tuned on training systems of the same class,
    avoiding data leakage from test systems.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    system_class : str
        System class name.
    threshold_lookup : dict, optional
        Pre-computed thresholds per class.
    max_iter : int, optional
        STLS iterations (default: 10).

    Returns
    -------
    xi : np.ndarray
        Coefficient matrix with shape [n_vars, n_terms].
    elapsed_time : float
        Computation time in seconds.
    threshold_used : float
        The threshold that was applied.
    """
    threshold = get_system_class_threshold(system_class, threshold_lookup)
    xi, elapsed = sindy_stls(Theta, x_dot, threshold=threshold, max_iter=max_iter)
    return xi, elapsed, threshold
