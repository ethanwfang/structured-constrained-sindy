"""
Coefficient error metrics.

This module provides metrics for evaluating how accurately SINDy algorithms
recover the true coefficient values.
"""

from typing import Dict

import numpy as np


def compute_coefficient_error(xi_pred: np.ndarray, xi_true: np.ndarray) -> float:
    """
    Compute mean absolute error between predicted and true coefficients.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix with shape [n_vars, n_terms].
    xi_true : np.ndarray
        True coefficient matrix with shape [n_vars, n_terms].

    Returns
    -------
    mae : float
        Mean absolute error.

    Examples
    --------
    >>> xi_pred = np.array([[0, 1.0, 0], [0, 0, 2.0]])
    >>> xi_true = np.array([[0, 1.0, 0], [0, 0, 1.5]])
    >>> mae = compute_coefficient_error(xi_pred, xi_true)
    """
    return np.mean(np.abs(xi_pred - xi_true))


def compute_coefficient_rmse(xi_pred: np.ndarray, xi_true: np.ndarray) -> float:
    """
    Compute root mean squared error between coefficients.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.

    Returns
    -------
    rmse : float
        Root mean squared error.
    """
    return np.sqrt(np.mean((xi_pred - xi_true) ** 2))


def compute_relative_coefficient_error(
    xi_pred: np.ndarray, xi_true: np.ndarray, tol: float = 1e-6
) -> float:
    """
    Compute mean relative error for non-zero true coefficients.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.
    tol : float, optional
        Tolerance for identifying non-zero coefficients.

    Returns
    -------
    mre : float
        Mean relative error (only for true non-zero coefficients).
    """
    active_mask = np.abs(xi_true) > tol

    if not np.any(active_mask):
        return 0.0

    relative_errors = np.abs(xi_pred[active_mask] - xi_true[active_mask]) / np.abs(
        xi_true[active_mask]
    )
    return np.mean(relative_errors)


def compute_active_coefficient_error(
    xi_pred: np.ndarray, xi_true: np.ndarray, tol: float = 1e-6
) -> Dict[str, float]:
    """
    Compute coefficient errors only for truly active terms.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.
    tol : float, optional
        Tolerance for identifying active terms.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary with mae, rmse, and max_error for active terms.
    """
    active_mask = np.abs(xi_true) > tol

    if not np.any(active_mask):
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "max_error": 0.0,
            "n_active": 0,
        }

    errors = xi_pred[active_mask] - xi_true[active_mask]

    return {
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors**2)),
        "max_error": np.max(np.abs(errors)),
        "n_active": int(np.sum(active_mask)),
    }


def coefficient_correlation(xi_pred: np.ndarray, xi_true: np.ndarray) -> float:
    """
    Compute Pearson correlation between predicted and true coefficients.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.

    Returns
    -------
    correlation : float
        Pearson correlation coefficient.
    """
    pred_flat = xi_pred.flatten()
    true_flat = xi_true.flatten()

    if np.std(pred_flat) < 1e-10 or np.std(true_flat) < 1e-10:
        return 0.0

    return np.corrcoef(pred_flat, true_flat)[0, 1]


def compute_coefficient_metrics(
    xi_pred: np.ndarray, xi_true: np.ndarray, tol: float = 1e-6
) -> Dict[str, float]:
    """
    Compute comprehensive coefficient accuracy metrics.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.
    tol : float, optional
        Tolerance for identifying active terms.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary with all coefficient metrics.
    """
    active_metrics = compute_active_coefficient_error(xi_pred, xi_true, tol)

    return {
        "mae": compute_coefficient_error(xi_pred, xi_true),
        "rmse": compute_coefficient_rmse(xi_pred, xi_true),
        "relative_error": compute_relative_coefficient_error(xi_pred, xi_true, tol),
        "correlation": coefficient_correlation(xi_pred, xi_true),
        "active_mae": active_metrics["mae"],
        "active_rmse": active_metrics["rmse"],
        "active_max_error": active_metrics["max_error"],
    }


def per_equation_coefficient_error(xi_pred: np.ndarray, xi_true: np.ndarray) -> np.ndarray:
    """
    Compute coefficient MAE for each equation separately.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix with shape [n_vars, n_terms].
    xi_true : np.ndarray
        True coefficient matrix.

    Returns
    -------
    errors : np.ndarray
        MAE for each equation with shape [n_vars].
    """
    return np.mean(np.abs(xi_pred - xi_true), axis=1)
