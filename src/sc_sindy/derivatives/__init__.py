"""
Derivative computation methods.

This module provides various numerical differentiation methods
for computing time derivatives from trajectory data.
"""

from .finite_difference import (
    compute_derivatives_finite_diff,
    compute_derivatives_nonuniform,
    compute_second_derivative,
)
from .spline import (
    compute_derivatives_adaptive,
    compute_derivatives_bspline,
    compute_derivatives_spline,
    smooth_and_differentiate,
)

__all__ = [
    # Finite difference methods
    "compute_derivatives_finite_diff",
    "compute_second_derivative",
    "compute_derivatives_nonuniform",
    # Spline methods
    "compute_derivatives_spline",
    "compute_derivatives_bspline",
    "compute_derivatives_adaptive",
    "smooth_and_differentiate",
]
