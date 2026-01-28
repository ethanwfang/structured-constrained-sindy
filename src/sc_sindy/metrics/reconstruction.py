"""
Trajectory reconstruction metrics.

This module provides metrics for evaluating how well discovered models
reproduce the original trajectory dynamics.
"""

from typing import Callable, Dict, Optional

import numpy as np
from scipy.integrate import odeint


def compute_reconstruction_error(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """
    Compute RMSE between true and predicted trajectories.

    Parameters
    ----------
    x_true : np.ndarray
        True trajectory with shape [n_samples, n_vars].
    x_pred : np.ndarray
        Predicted trajectory with shape [n_samples, n_vars].

    Returns
    -------
    rmse : float
        Root mean squared error.

    Examples
    --------
    >>> x_true = np.random.randn(100, 2)
    >>> x_pred = x_true + 0.1 * np.random.randn(100, 2)
    >>> rmse = compute_reconstruction_error(x_true, x_pred)
    """
    return np.sqrt(np.mean((x_true - x_pred) ** 2))


def compute_derivative_reconstruction_error(
    x_dot_true: np.ndarray, x_dot_pred: np.ndarray
) -> float:
    """
    Compute RMSE between true and predicted derivatives.

    Parameters
    ----------
    x_dot_true : np.ndarray
        True derivatives with shape [n_samples, n_vars].
    x_dot_pred : np.ndarray
        Predicted derivatives with shape [n_samples, n_vars].

    Returns
    -------
    rmse : float
        Root mean squared error.
    """
    return np.sqrt(np.mean((x_dot_true - x_dot_pred) ** 2))


def compute_normalized_error(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """
    Compute normalized RMSE (relative to data range).

    Parameters
    ----------
    x_true : np.ndarray
        True trajectory.
    x_pred : np.ndarray
        Predicted trajectory.

    Returns
    -------
    nrmse : float
        Normalized RMSE.
    """
    rmse = compute_reconstruction_error(x_true, x_pred)
    data_range = np.max(x_true) - np.min(x_true)
    return rmse / data_range if data_range > 0 else rmse


def compute_r2_score(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).

    Parameters
    ----------
    x_true : np.ndarray
        True values.
    x_pred : np.ndarray
        Predicted values.

    Returns
    -------
    r2 : float
        R-squared value.
    """
    ss_res = np.sum((x_true - x_pred) ** 2)
    ss_tot = np.sum((x_true - np.mean(x_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def integrate_discovered_model(
    xi: np.ndarray, x0: np.ndarray, t: np.ndarray, build_library_fn: Callable, poly_order: int = 3
) -> np.ndarray:
    """
    Integrate the discovered model forward in time.

    Parameters
    ----------
    xi : np.ndarray
        Discovered coefficient matrix with shape [n_vars, n_terms].
    x0 : np.ndarray
        Initial condition.
    t : np.ndarray
        Time points for integration.
    build_library_fn : Callable
        Function to build feature library from state.
    poly_order : int, optional
        Polynomial order for library (default: 3).

    Returns
    -------
    trajectory : np.ndarray
        Integrated trajectory with shape [n_times, n_vars].
    """

    def model_derivatives(state, t):
        x_curr = state.reshape(1, -1)
        Theta, _ = build_library_fn(x_curr, poly_order)
        return (Theta @ xi.T).flatten()

    try:
        trajectory = odeint(model_derivatives, x0, t)
    except Exception:
        trajectory = np.full((len(t), len(x0)), np.nan)

    return trajectory


def compute_forward_prediction_error(
    xi: np.ndarray,
    x_true: np.ndarray,
    t: np.ndarray,
    build_library_fn: Callable,
    poly_order: int = 3,
    n_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute error of forward prediction from discovered model.

    Parameters
    ----------
    xi : np.ndarray
        Discovered coefficient matrix.
    x_true : np.ndarray
        True trajectory.
    t : np.ndarray
        Time points.
    build_library_fn : Callable
        Function to build feature library.
    poly_order : int, optional
        Polynomial order for library.
    n_steps : int, optional
        Number of steps to predict. If None, predicts full trajectory.

    Returns
    -------
    metrics : Dict[str, float]
        Forward prediction error metrics.
    """
    x0 = x_true[0]

    if n_steps is not None:
        t_pred = t[:n_steps]
        x_target = x_true[:n_steps]
    else:
        t_pred = t
        x_target = x_true

    x_pred = integrate_discovered_model(xi, x0, t_pred, build_library_fn, poly_order)

    # Check for numerical issues
    if np.any(np.isnan(x_pred)) or np.any(np.isinf(x_pred)):
        return {
            "rmse": np.inf,
            "nrmse": np.inf,
            "r2": -np.inf,
            "valid": False,
        }

    return {
        "rmse": compute_reconstruction_error(x_target, x_pred),
        "nrmse": compute_normalized_error(x_target, x_pred),
        "r2": compute_r2_score(x_target, x_pred),
        "valid": True,
    }


def compute_lyapunov_time_error(
    xi: np.ndarray,
    x_true: np.ndarray,
    t: np.ndarray,
    build_library_fn: Callable,
    error_threshold: float = 1.0,
    poly_order: int = 3,
) -> float:
    """
    Compute time until prediction error exceeds threshold (Lyapunov time proxy).

    Parameters
    ----------
    xi : np.ndarray
        Discovered coefficient matrix.
    x_true : np.ndarray
        True trajectory.
    t : np.ndarray
        Time points.
    build_library_fn : Callable
        Function to build feature library.
    error_threshold : float, optional
        Error threshold for "failure" (default: 1.0).
    poly_order : int, optional
        Polynomial order.

    Returns
    -------
    lyapunov_time : float
        Time until error exceeds threshold (or max time if never exceeded).
    """
    x0 = x_true[0]
    x_pred = integrate_discovered_model(xi, x0, t, build_library_fn, poly_order)

    if np.any(np.isnan(x_pred)) or np.any(np.isinf(x_pred)):
        return 0.0

    errors = np.sqrt(np.sum((x_true - x_pred) ** 2, axis=1))

    exceeded = np.where(errors > error_threshold)[0]
    if len(exceeded) > 0:
        return t[exceeded[0]] - t[0]
    else:
        return t[-1] - t[0]


def compute_reconstruction_metrics(
    xi: np.ndarray, Theta: np.ndarray, x_dot_true: np.ndarray
) -> Dict[str, float]:
    """
    Compute all reconstruction metrics using library matrix.

    Parameters
    ----------
    xi : np.ndarray
        Discovered coefficient matrix [n_vars, n_terms].
    Theta : np.ndarray
        Feature library matrix [n_samples, n_terms].
    x_dot_true : np.ndarray
        True derivatives [n_samples, n_vars].

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary with reconstruction metrics.
    """
    x_dot_pred = Theta @ xi.T

    return {
        "derivative_rmse": compute_derivative_reconstruction_error(x_dot_true, x_dot_pred),
        "derivative_nrmse": compute_normalized_error(x_dot_true, x_dot_pred),
        "derivative_r2": compute_r2_score(x_dot_true, x_dot_pred),
    }
