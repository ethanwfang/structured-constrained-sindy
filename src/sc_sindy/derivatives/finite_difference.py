"""
Finite difference methods for derivative computation.

This module provides various finite difference schemes for computing
time derivatives from trajectory data.
"""

import numpy as np


def compute_derivatives_finite_diff(x: np.ndarray, dt: float, order: int = 2) -> np.ndarray:
    """
    Compute derivatives using finite differences.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, n_vars].
    dt : float
        Time step between samples.
    order : int, optional
        Order of finite difference scheme (2 or 4, default: 2).

    Returns
    -------
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 10, 100).reshape(-1, 1))
    >>> dt = 0.1
    >>> x_dot = compute_derivatives_finite_diff(x, dt, order=2)

    Notes
    -----
    Order 2: Central difference with forward/backward at boundaries
        - Interior: (x[i+1] - x[i-1]) / (2*dt)
        - Boundaries: One-sided 2nd order

    Order 4: 4th order central difference
        - Interior: (-x[i+2] + 8*x[i+1] - 8*x[i-1] + x[i-2]) / (12*dt)
        - Boundaries: Forward/backward difference
    """
    n_vars = x.shape[1]
    x_dot = np.zeros_like(x)

    for i in range(n_vars):
        if order == 2:
            # Central difference for interior points
            x_dot[1:-1, i] = (x[2:, i] - x[:-2, i]) / (2 * dt)
            # Forward difference at start (2nd order)
            x_dot[0, i] = (-3 * x[0, i] + 4 * x[1, i] - x[2, i]) / (2 * dt)
            # Backward difference at end (2nd order)
            x_dot[-1, i] = (3 * x[-1, i] - 4 * x[-2, i] + x[-3, i]) / (2 * dt)

        elif order == 4:
            # 4th order central difference for interior
            x_dot[2:-2, i] = (-x[4:, i] + 8 * x[3:-1, i] - 8 * x[1:-3, i] + x[:-4, i]) / (12 * dt)
            # Lower order at boundaries
            x_dot[:2, i] = (x[1:3, i] - x[0:2, i]) / dt
            x_dot[-2:, i] = (x[-2:, i] - x[-3:-1, i]) / dt

        else:
            raise ValueError(f"Order {order} not supported. Use 2 or 4.")

    return x_dot


def compute_second_derivative(x: np.ndarray, dt: float, order: int = 2) -> np.ndarray:
    """
    Compute second derivatives using finite differences.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, n_vars].
    dt : float
        Time step between samples.
    order : int, optional
        Order of finite difference scheme (2 or 4, default: 2).

    Returns
    -------
    x_ddot : np.ndarray
        Second time derivatives with shape [n_samples, n_vars].
    """
    n_vars = x.shape[1]
    x_ddot = np.zeros_like(x)

    for i in range(n_vars):
        if order == 2:
            # Central second difference for interior
            x_ddot[1:-1, i] = (x[2:, i] - 2 * x[1:-1, i] + x[:-2, i]) / (dt**2)
            # Boundaries using forward/backward
            x_ddot[0, i] = (x[2, i] - 2 * x[1, i] + x[0, i]) / (dt**2)
            x_ddot[-1, i] = (x[-1, i] - 2 * x[-2, i] + x[-3, i]) / (dt**2)

        elif order == 4:
            # 4th order central second difference
            x_ddot[2:-2, i] = (
                -x[4:, i] + 16 * x[3:-1, i] - 30 * x[2:-2, i] + 16 * x[1:-3, i] - x[:-4, i]
            ) / (12 * dt**2)
            # Lower order at boundaries
            x_ddot[:2, i] = (x[2:4, i] - 2 * x[1:3, i] + x[:2, i]) / (dt**2)
            x_ddot[-2:, i] = (x[-2:, i] - 2 * x[-3:-1, i] + x[-4:-2, i]) / (dt**2)

        else:
            raise ValueError(f"Order {order} not supported. Use 2 or 4.")

    return x_ddot


def compute_derivatives_nonuniform(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute derivatives for non-uniformly sampled data.

    Uses three-point Lagrange interpolation for derivative estimation.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, n_vars].
    t : np.ndarray
        Time points with shape [n_samples].

    Returns
    -------
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    """
    n_samples, n_vars = x.shape
    x_dot = np.zeros_like(x)

    for i in range(n_vars):
        for j in range(1, n_samples - 1):
            # Three-point Lagrange formula
            h1 = t[j] - t[j - 1]
            h2 = t[j + 1] - t[j]

            x_dot[j, i] = (
                -h2 / (h1 * (h1 + h2)) * x[j - 1, i]
                + (h2 - h1) / (h1 * h2) * x[j, i]
                + h1 / (h2 * (h1 + h2)) * x[j + 1, i]
            )

        # Boundary points using forward/backward difference
        h = t[1] - t[0]
        x_dot[0, i] = (x[1, i] - x[0, i]) / h
        h = t[-1] - t[-2]
        x_dot[-1, i] = (x[-1, i] - x[-2, i]) / h

    return x_dot
