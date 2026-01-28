"""
Core SINDy algorithms.

This module provides the fundamental algorithms for Sparse Identification
of Nonlinear Dynamics (SINDy), including standard STLS and the
Structure-Constrained variant.
"""

from .library import (
    build_library_2d,
    build_library_3d,
    build_library_nd,
)
from .sindy import (
    DEFAULT_STLS_THRESHOLD,
    sindy_ridge,
    sindy_stls,
)
from .structure_constrained import (
    DEFAULT_STRUCTURE_THRESHOLD,
    get_recommended_threshold,
    sindy_structure_constrained,
    sindy_structure_constrained_soft,
)

__all__ = [
    # Standard SINDy
    "sindy_stls",
    "sindy_ridge",
    "DEFAULT_STLS_THRESHOLD",
    # Structure-Constrained SINDy
    "sindy_structure_constrained",
    "sindy_structure_constrained_soft",
    "get_recommended_threshold",
    "DEFAULT_STRUCTURE_THRESHOLD",
    # Library construction
    "build_library_2d",
    "build_library_3d",
    "build_library_nd",
]
