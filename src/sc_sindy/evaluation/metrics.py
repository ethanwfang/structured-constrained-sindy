"""
Comprehensive evaluation metrics for SC-SINDy.

This module provides additional metrics beyond the core metrics module,
specifically designed for evaluating the fair (non-oracle) evaluation pipeline.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..metrics import compute_coefficient_error, compute_structure_metrics


def compute_all_metrics(
    xi_pred: np.ndarray,
    xi_true: np.ndarray,
    network_probs: Optional[np.ndarray] = None,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficients with shape [n_vars, n_terms].
    xi_true : np.ndarray
        True coefficients with shape [n_vars, n_terms].
    network_probs : np.ndarray, optional
        Network probability predictions with shape [n_vars, n_terms].
        If provided, also computes network prediction quality metrics.
    tol : float
        Tolerance for determining active terms.

    Returns
    -------
    metrics : Dict[str, Any]
        Dictionary containing all computed metrics.
    """
    # Structure metrics
    structure_metrics = compute_structure_metrics(xi_pred, xi_true, tol=tol)

    # Coefficient error
    coef_error = compute_coefficient_error(xi_pred, xi_true)

    # Combine into results
    results = {
        # Structure recovery
        "f1": structure_metrics["f1"],
        "precision": structure_metrics["precision"],
        "recall": structure_metrics["recall"],
        "accuracy": structure_metrics.get("accuracy", 0.0),
        # Coefficient accuracy
        "coefficient_mae": coef_error,
        "coefficient_nrmse": compute_normalized_coefficient_error(xi_pred, xi_true),
        # Sparsity
        "n_predicted_terms": int(np.sum(np.abs(xi_pred) > tol)),
        "n_true_terms": int(np.sum(np.abs(xi_true) > tol)),
        "sparsity_ratio": compute_sparsity_ratio(xi_pred, xi_true, tol),
    }

    # Network prediction quality (if provided)
    if network_probs is not None:
        true_structure = np.abs(xi_true) > tol
        pred_structure = network_probs > 0.5

        net_metrics = compute_structure_metrics(
            pred_structure.astype(float), true_structure.astype(float)
        )
        results["network_f1"] = net_metrics["f1"]
        results["network_precision"] = net_metrics["precision"]
        results["network_recall"] = net_metrics["recall"]

        # Calibration metrics
        results["network_mean_prob_true"] = float(
            np.mean(network_probs[true_structure]) if np.any(true_structure) else 0.0
        )
        results["network_mean_prob_false"] = float(
            np.mean(network_probs[~true_structure]) if np.any(~true_structure) else 0.0
        )

    return results


def compute_normalized_coefficient_error(
    xi_pred: np.ndarray, xi_true: np.ndarray, eps: float = 1e-10
) -> float:
    """
    Compute normalized root mean squared error for coefficients.

    NRMSE = RMSE / range(xi_true)

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficients.
    xi_true : np.ndarray
        True coefficients.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    nrmse : float
        Normalized RMSE.
    """
    rmse = np.sqrt(np.mean((xi_pred - xi_true) ** 2))
    value_range = np.max(np.abs(xi_true)) - np.min(np.abs(xi_true)) + eps
    return float(rmse / value_range)


def compute_sparsity_ratio(xi_pred: np.ndarray, xi_true: np.ndarray, tol: float = 1e-6) -> float:
    """
    Compute ratio of predicted sparsity to true sparsity.

    Values > 1 indicate over-sparse predictions (missing terms).
    Values < 1 indicate under-sparse predictions (spurious terms).

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficients.
    xi_true : np.ndarray
        True coefficients.
    tol : float
        Tolerance for active term detection.

    Returns
    -------
    ratio : float
        Sparsity ratio.
    """
    n_pred_zeros = np.sum(np.abs(xi_pred) <= tol)
    n_true_zeros = np.sum(np.abs(xi_true) <= tol)

    if n_true_zeros == 0:
        return 0.0

    return float(n_pred_zeros / n_true_zeros)


def compute_success_rate(results_list: list, f1_threshold: float = 0.8) -> Dict[str, float]:
    """
    Compute success rate (fraction of trials with F1 above threshold).

    Parameters
    ----------
    results_list : list
        List of EvaluationResult objects.
    f1_threshold : float
        F1 threshold for considering a trial successful.

    Returns
    -------
    rates : Dict[str, float]
        Success rates for standard and SC-SINDy.
    """
    if not results_list:
        return {"standard": 0.0, "sc": 0.0}

    std_successes = sum(1 for r in results_list if r.standard_f1 >= f1_threshold)
    sc_successes = sum(1 for r in results_list if r.sc_f1 >= f1_threshold)
    n_trials = len(results_list)

    return {
        "standard": std_successes / n_trials,
        "sc": sc_successes / n_trials,
    }


def compute_improvement_statistics(results_list: list) -> Dict[str, float]:
    """
    Compute statistics about SC-SINDy improvement over standard SINDy.

    Parameters
    ----------
    results_list : list
        List of EvaluationResult objects.

    Returns
    -------
    stats : Dict[str, float]
        Improvement statistics.
    """
    if not results_list:
        return {}

    improvements = [r.f1_improvement for r in results_list]
    speedups = [r.speedup for r in results_list]

    return {
        "mean_f1_improvement": float(np.mean(improvements)),
        "std_f1_improvement": float(np.std(improvements)),
        "median_f1_improvement": float(np.median(improvements)),
        "pct_improved": float(np.mean([1 if i > 0 else 0 for i in improvements])),
        "pct_degraded": float(np.mean([1 if i < 0 else 0 for i in improvements])),
        "mean_speedup": float(np.mean(speedups)),
        "median_speedup": float(np.median(speedups)),
    }


def results_to_dataframe(results_list: list):
    """
    Convert list of EvaluationResult objects to pandas DataFrame.

    Parameters
    ----------
    results_list : list
        List of EvaluationResult objects.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with all results.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for results_to_dataframe()")

    records = [r.to_dict() for r in results_list]
    return pd.DataFrame(records)
