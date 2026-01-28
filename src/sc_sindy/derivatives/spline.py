"""
Spline-based methods for derivative computation.

This module provides smoothing spline methods for computing derivatives
from noisy trajectory data.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline, splev, splrep


def compute_derivatives_spline(x: np.ndarray, t: np.ndarray, smooth: float = 0.01) -> np.ndarray:
    """
    Compute derivatives using spline interpolation.

    Uses smoothing splines to reduce noise before differentiation.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, n_vars].
    t : np.ndarray
        Time points with shape [n_samples].
    smooth : float, optional
        Smoothing parameter (default: 0.01). Larger values produce
        smoother fits but may lose detail.

    Returns
    -------
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> x = np.sin(t).reshape(-1, 1) + 0.1 * np.random.randn(100, 1)
    >>> x_dot = compute_derivatives_spline(x, t, smooth=0.1)
    """
    x_dot = np.zeros_like(x)

    for i in range(x.shape[1]):
        spline = UnivariateSpline(t, x[:, i], s=smooth)
        x_dot[:, i] = spline.derivative()(t)

    return x_dot


def compute_derivatives_bspline(
    x: np.ndarray, t: np.ndarray, degree: int = 3, smooth: Optional[float] = None
) -> np.ndarray:
    """
    Compute derivatives using B-splines.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, n_vars].
    t : np.ndarray
        Time points with shape [n_samples].
    degree : int, optional
        Degree of the B-spline (default: 3 for cubic).
    smooth : float, optional
        Smoothing condition. If None, uses interpolating spline.

    Returns
    -------
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    """
    x_dot = np.zeros_like(x)

    for i in range(x.shape[1]):
        tck = splrep(t, x[:, i], k=degree, s=smooth)
        x_dot[:, i] = splev(t, tck, der=1)

    return x_dot


def compute_derivatives_adaptive(
    x: np.ndarray, t: np.ndarray, noise_estimate: Optional[float] = None
) -> np.ndarray:
    """
    Compute derivatives with adaptive smoothing based on noise estimate.

    Automatically selects smoothing parameter based on estimated noise level.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, n_vars].
    t : np.ndarray
        Time points with shape [n_samples].
    noise_estimate : float, optional
        Estimated noise standard deviation. If None, estimated from data.

    Returns
    -------
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    """
    n_samples, n_vars = x.shape
    x_dot = np.zeros_like(x)

    for i in range(n_vars):
        # Estimate noise if not provided
        if noise_estimate is None:
            # Use difference of adjacent points as noise proxy
            diff = np.diff(x[:, i])
            noise_est = np.std(diff) / np.sqrt(2)
        else:
            noise_est = noise_estimate

        # Adaptive smoothing: s ~ n * noise^2
        smooth = n_samples * noise_est**2

        spline = UnivariateSpline(t, x[:, i], s=smooth)
        x_dot[:, i] = spline.derivative()(t)

    return x_dot


def smooth_and_differentiate(
    x: np.ndarray, t: np.ndarray, smooth: float = 0.01, n_derivatives: int = 1
) -> Tuple[np.ndarray, ...]:
    """
    Smooth data and compute multiple derivatives.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, n_vars].
    t : np.ndarray
        Time points with shape [n_samples].
    smooth : float, optional
        Smoothing parameter (default: 0.01).
    n_derivatives : int, optional
        Number of derivatives to compute (default: 1).

    Returns
    -------
    results : tuple
        Tuple containing (x_smooth, x_dot, x_ddot, ...) up to n_derivatives.

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> x = np.sin(t).reshape(-1, 1)
    >>> x_smooth, x_dot, x_ddot = smooth_and_differentiate(x, t, n_derivatives=2)
    """
    results = []
    n_vars = x.shape[1]

    # Smoothed data
    x_smooth = np.zeros_like(x)
    splines = []

    for i in range(n_vars):
        spline = UnivariateSpline(t, x[:, i], s=smooth)
        splines.append(spline)
        x_smooth[:, i] = spline(t)

    results.append(x_smooth)

    # Derivatives
    for d in range(1, n_derivatives + 1):
        x_deriv = np.zeros_like(x)
        for i in range(n_vars):
            x_deriv[:, i] = splines[i].derivative(n=d)(t)
        results.append(x_deriv)

    return tuple(results)
