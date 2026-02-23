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


def compute_comprehensive_coefficient_metrics(
    xi_pred: np.ndarray,
    xi_true: np.ndarray,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute comprehensive coefficient metrics including median and success rate.

    This addresses reviewer concerns about unclear metrics by providing:
    - Both mean and median (robust to outliers)
    - Both all-term and active-term metrics
    - Success rate (fraction of coefficients within 2x of true value)

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix with shape [n_vars, n_terms].
    xi_true : np.ndarray
        True coefficient matrix with shape [n_vars, n_terms].
    tol : float
        Tolerance for identifying active terms (default: 1e-6).

    Returns
    -------
    Dict[str, float]
        Comprehensive metrics including:
        - mae_all: Mean absolute error over all terms
        - mae_active: Mean absolute error over active terms only
        - median_ae_active: Median absolute error over active terms (robust)
        - rmse_active: RMSE over active terms
        - max_error: Maximum absolute error over active terms
        - success_rate: Fraction of active terms within 2x of true value
        - n_active: Number of active terms
        - n_total: Total number of terms
    """
    # All-term metrics
    mae_all = float(np.mean(np.abs(xi_pred - xi_true)))

    # Active-term metrics
    active_mask = np.abs(xi_true) > tol
    n_active = int(np.sum(active_mask))
    n_total = xi_true.size

    if n_active == 0:
        return {
            "mae_all": mae_all,
            "mae_active": 0.0,
            "median_ae_active": 0.0,
            "rmse_active": 0.0,
            "max_error": 0.0,
            "success_rate": 1.0,  # No active terms means trivial success
            "n_active": 0,
            "n_total": n_total,
        }

    # Errors on active terms
    errors = np.abs(xi_pred[active_mask] - xi_true[active_mask])
    true_vals = np.abs(xi_true[active_mask])

    # Success rate: coefficient is within 2x of true value
    # |pred - true| < |true| means relative error < 100%
    # This is equivalent to pred being within [0, 2*true] for positive coefficients
    success_mask = errors < true_vals
    success_rate = float(np.mean(success_mask))

    return {
        "mae_all": mae_all,
        "mae_active": float(np.mean(errors)),
        "median_ae_active": float(np.median(errors)),
        "rmse_active": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "success_rate": success_rate,
        "n_active": n_active,
        "n_total": n_total,
    }


def compute_lorenz_coefficient_breakdown(
    xi_pred: np.ndarray,
    xi_true: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-parameter coefficient errors for the Lorenz system.

    The Lorenz system has three key parameters:
    - sigma (σ): Prandtl number, appears in dx/dt = σ(y - x)
    - rho (ρ): Rayleigh number, appears in dy/dt = x(ρ - z) - y
    - beta (β): Geometric factor, appears in dz/dt = xy - βz

    For polynomial library order 3 with variables [x, y, z]:
    Library: [1, x, y, z, x², xy, xz, y², yz, z², x³, x²y, x²z, xy², xyz, xz², y³, y²z, yz², z³]
    Indices:  0  1  2  3   4   5   6   7   8   9   10  11   12   13   14   15   16   17   18   19

    Lorenz equations:
    - dx/dt = -σx + σy     → terms: x (idx 1, coef -σ), y (idx 2, coef +σ)
    - dy/dt = ρx - y - xz  → terms: x (idx 1, coef ρ), y (idx 2, coef -1), xz (idx 6, coef -1)
    - dz/dt = xy - βz      → terms: xy (idx 5, coef 1), z (idx 3, coef -β)

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficients with shape [3, n_terms].
    xi_true : np.ndarray
        True coefficients with shape [3, n_terms].

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per-parameter breakdown with predicted, true, error, and relative error.
    """
    results = {}

    # Sigma: coefficient of y in equation 0 (should be +10)
    # Also appears as coefficient of x in equation 0 (should be -10)
    sigma_pred_y = xi_pred[0, 2] if xi_pred.shape[1] > 2 else np.nan
    sigma_true_y = xi_true[0, 2] if xi_true.shape[1] > 2 else np.nan
    sigma_pred_x = -xi_pred[0, 1] if xi_pred.shape[1] > 1 else np.nan  # Negate because coef is -σ
    sigma_true_x = -xi_true[0, 1] if xi_true.shape[1] > 1 else np.nan

    results["sigma"] = {
        "predicted": float(sigma_pred_y),
        "true": float(sigma_true_y),
        "error": float(abs(sigma_pred_y - sigma_true_y)) if not np.isnan(sigma_pred_y) else np.nan,
        "relative_error": float(abs(sigma_pred_y - sigma_true_y) / abs(sigma_true_y))
            if not np.isnan(sigma_pred_y) and abs(sigma_true_y) > 1e-10 else np.nan,
    }

    # Rho: coefficient of x in equation 1 (should be +28)
    rho_pred = xi_pred[1, 1] if xi_pred.shape[1] > 1 else np.nan
    rho_true = xi_true[1, 1] if xi_true.shape[1] > 1 else np.nan

    results["rho"] = {
        "predicted": float(rho_pred),
        "true": float(rho_true),
        "error": float(abs(rho_pred - rho_true)) if not np.isnan(rho_pred) else np.nan,
        "relative_error": float(abs(rho_pred - rho_true) / abs(rho_true))
            if not np.isnan(rho_pred) and abs(rho_true) > 1e-10 else np.nan,
    }

    # Beta: coefficient of z in equation 2 (should be -8/3 ≈ -2.667)
    beta_pred = -xi_pred[2, 3] if xi_pred.shape[1] > 3 else np.nan  # Negate because coef is -β
    beta_true = -xi_true[2, 3] if xi_true.shape[1] > 3 else np.nan

    results["beta"] = {
        "predicted": float(beta_pred),
        "true": float(beta_true),
        "error": float(abs(beta_pred - beta_true)) if not np.isnan(beta_pred) else np.nan,
        "relative_error": float(abs(beta_pred - beta_true) / abs(beta_true))
            if not np.isnan(beta_pred) and abs(beta_true) > 1e-10 else np.nan,
    }

    return results
