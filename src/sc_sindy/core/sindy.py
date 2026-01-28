"""
Standard SINDy implementation using Sequential Thresholded Least Squares (STLS).

This module provides the core STLS algorithm for sparse regression,
which is the foundation of the SINDy method.
"""

import time
from typing import Tuple

import numpy as np

# Default STLS threshold
DEFAULT_STLS_THRESHOLD = 0.1


def sindy_stls(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    threshold: float = DEFAULT_STLS_THRESHOLD,
    max_iter: int = 10,
    normalize: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Sequential Thresholded Least Squares (Standard SINDy).

    Iteratively performs least squares regression and thresholds small
    coefficients to promote sparsity.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    threshold : float, optional
        Coefficients below this value are set to zero (default: 0.1).
    max_iter : int, optional
        Maximum number of iterations (default: 10).
    normalize : bool, optional
        Whether to normalize columns of Theta (default: False).

    Returns
    -------
    xi : np.ndarray
        Coefficient matrix with shape [n_vars, n_terms].
    elapsed_time : float
        Computation time in seconds.

    Examples
    --------
    >>> Theta = np.random.randn(100, 10)
    >>> x_dot = np.random.randn(100, 2)
    >>> xi, elapsed = sindy_stls(Theta, x_dot, threshold=0.1)
    >>> print(xi.shape)
    (2, 10)
    """
    t_start = time.time()

    # Handle normalization
    if normalize:
        norms = np.linalg.norm(Theta, axis=0)
        norms[norms == 0] = 1.0
        Theta_normalized = Theta / norms
    else:
        Theta_normalized = Theta
        norms = np.ones(Theta.shape[1])

    n_vars = x_dot.shape[1]
    n_terms = Theta.shape[1]
    xi = np.zeros((n_vars, n_terms))

    for i in range(n_vars):
        xi[i, :] = _stls_single(Theta_normalized, x_dot[:, i], threshold, max_iter)

    # Rescale if normalized
    if normalize:
        xi = xi / norms

    return xi, time.time() - t_start


def _stls_single(Theta: np.ndarray, y: np.ndarray, threshold: float, max_iter: int) -> np.ndarray:
    """
    STLS for single output variable.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    y : np.ndarray
        Target variable with shape [n_samples].
    threshold : float
        Sparsity threshold.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    xi : np.ndarray
        Coefficient vector with shape [n_terms].
    """
    xi = np.linalg.lstsq(Theta, y, rcond=None)[0]

    for _ in range(max_iter):
        small_inds = np.abs(xi) < threshold
        xi[small_inds] = 0
        big_inds = ~small_inds

        if np.sum(big_inds) == 0:
            break

        xi[big_inds] = np.linalg.lstsq(Theta[:, big_inds], y, rcond=None)[0]

    return xi


def sindy_ridge(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    alpha: float = 0.01,
    threshold: float = DEFAULT_STLS_THRESHOLD,
    max_iter: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    SINDy with ridge regression regularization.

    Uses L2 regularization in the least squares step before thresholding.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    alpha : float, optional
        Ridge regularization parameter (default: 0.01).
    threshold : float, optional
        Sparsity threshold (default: 0.1).
    max_iter : int, optional
        Maximum iterations (default: 10).

    Returns
    -------
    xi : np.ndarray
        Coefficient matrix with shape [n_vars, n_terms].
    elapsed_time : float
        Computation time in seconds.
    """
    t_start = time.time()

    n_vars = x_dot.shape[1]
    n_terms = Theta.shape[1]
    xi = np.zeros((n_vars, n_terms))

    # Precompute for ridge regression
    ThetaTTheta = Theta.T @ Theta
    I = np.eye(n_terms)

    for i in range(n_vars):
        xi[i, :] = _ridge_stls_single(
            Theta, x_dot[:, i], ThetaTTheta, I, alpha, threshold, max_iter
        )

    return xi, time.time() - t_start


def _ridge_stls_single(
    Theta: np.ndarray,
    y: np.ndarray,
    ThetaTTheta: np.ndarray,
    I: np.ndarray,
    alpha: float,
    threshold: float,
    max_iter: int,
) -> np.ndarray:
    """Ridge STLS for single output variable."""
    # Initial ridge regression
    xi = np.linalg.solve(ThetaTTheta + alpha * I, Theta.T @ y)

    for _ in range(max_iter):
        small_inds = np.abs(xi) < threshold
        xi[small_inds] = 0
        big_inds = ~small_inds

        if np.sum(big_inds) == 0:
            break

        # Ridge regression on active terms only
        Theta_active = Theta[:, big_inds]
        TT_active = Theta_active.T @ Theta_active
        I_active = np.eye(np.sum(big_inds))
        xi[big_inds] = np.linalg.solve(TT_active + alpha * I_active, Theta_active.T @ y)

    return xi
