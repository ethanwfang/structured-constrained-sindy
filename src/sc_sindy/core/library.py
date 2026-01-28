"""
Feature library construction for SINDy.

This module provides functions to build polynomial feature libraries
for 2D, 3D, and N-dimensional dynamical systems.
"""

from typing import List, Tuple

import numpy as np


def build_library_2d(x: np.ndarray, poly_order: int = 3) -> Tuple[np.ndarray, List[str]]:
    """
    Build polynomial library for 2D systems.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, 2].
    poly_order : int, optional
        Maximum polynomial order (default: 3).

    Returns
    -------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    names : List[str]
        Names of library terms.

    Examples
    --------
    >>> x = np.random.randn(100, 2)
    >>> Theta, names = build_library_2d(x, poly_order=2)
    >>> print(names)
    ['1', 'x', 'y', 'xx', 'xy', 'yy']
    """
    T = x.shape[0]
    x1, x2 = x[:, 0], x[:, 1]

    terms = [np.ones(T)]
    names = ["1"]

    # Order 1
    terms.extend([x1, x2])
    names.extend(["x", "y"])

    # Order 2
    if poly_order >= 2:
        terms.extend([x1**2, x1 * x2, x2**2])
        names.extend(["xx", "xy", "yy"])

    # Order 3
    if poly_order >= 3:
        terms.extend([x1**3, x1**2 * x2, x1 * x2**2, x2**3])
        names.extend(["xxx", "xxy", "xyy", "yyy"])

    # Order 4
    if poly_order >= 4:
        terms.extend([x1**4, x1**3 * x2, x1**2 * x2**2, x1 * x2**3, x2**4])
        names.extend(["xxxx", "xxxy", "xxyy", "xyyy", "yyyy"])

    # Order 5
    if poly_order >= 5:
        terms.extend([x1**5, x1**4 * x2, x1**3 * x2**2, x1**2 * x2**3, x1 * x2**4, x2**5])
        names.extend(["xxxxx", "xxxxy", "xxxyy", "xxyyy", "xyyyy", "yyyyy"])

    Theta = np.column_stack(terms)
    return Theta, names


def build_library_3d(x: np.ndarray, poly_order: int = 2) -> Tuple[np.ndarray, List[str]]:
    """
    Build polynomial library for 3D systems.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, 3].
    poly_order : int, optional
        Maximum polynomial order (default: 2).

    Returns
    -------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    names : List[str]
        Names of library terms.

    Examples
    --------
    >>> x = np.random.randn(100, 3)
    >>> Theta, names = build_library_3d(x, poly_order=2)
    >>> print(names)
    ['1', 'x', 'y', 'z', 'xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    """
    T = x.shape[0]
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

    terms = [np.ones(T)]
    names = ["1"]

    # Order 1
    terms.extend([x1, x2, x3])
    names.extend(["x", "y", "z"])

    # Order 2
    if poly_order >= 2:
        terms.extend([x1**2, x1 * x2, x1 * x3, x2**2, x2 * x3, x3**2])
        names.extend(["xx", "xy", "xz", "yy", "yz", "zz"])

    # Order 3
    if poly_order >= 3:
        terms.extend(
            [
                x1**3,
                x1**2 * x2,
                x1**2 * x3,
                x1 * x2**2,
                x1 * x2 * x3,
                x1 * x3**2,
                x2**3,
                x2**2 * x3,
                x2 * x3**2,
                x3**3,
            ]
        )
        names.extend(["xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz"])

    Theta = np.column_stack(terms)
    return Theta, names


def build_library_nd(x: np.ndarray, poly_order: int = 2) -> Tuple[np.ndarray, List[str]]:
    """
    Build polynomial library for N-dimensional systems.

    Uses itertools to generate all polynomial combinations up to the specified order.

    Parameters
    ----------
    x : np.ndarray
        State data with shape [n_samples, n_dims].
    poly_order : int, optional
        Maximum polynomial order (default: 2).

    Returns
    -------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    names : List[str]
        Names of library terms.
    """
    from itertools import combinations_with_replacement

    n_samples, n_dims = x.shape
    var_names = [chr(ord("x") + i) if i < 3 else f"x{i}" for i in range(n_dims)]

    terms = [np.ones(n_samples)]
    names = ["1"]

    for order in range(1, poly_order + 1):
        for combo in combinations_with_replacement(range(n_dims), order):
            term = np.ones(n_samples)
            name_parts = []
            for idx in combo:
                term *= x[:, idx]
                name_parts.append(var_names[idx])
            terms.append(term)
            names.append("".join(name_parts))

    Theta = np.column_stack(terms)
    return Theta, names
