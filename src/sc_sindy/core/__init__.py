"""
Core SINDy algorithms.

This module provides the fundamental algorithms for Sparse Identification
of Nonlinear Dynamics (SINDy), including standard STLS, Structure-Constrained,
and Ensemble variants.
"""

from .ensemble import (
    EnsembleResult,
    compute_inclusion_probabilities,
    ensemble_sindy,
    ensemble_sindy_library_bagging,
)
from .ensemble_structure_constrained import (
    EnsembleSCResult,
    ensemble_structure_constrained_sindy,
    get_uncertainty_report,
    probability_fusion,
    structure_weighted_ensemble,
    two_stage_ensemble,
)
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
    # Ensemble SINDy
    "ensemble_sindy",
    "ensemble_sindy_library_bagging",
    "compute_inclusion_probabilities",
    "EnsembleResult",
    # Ensemble Structure-Constrained SINDy
    "ensemble_structure_constrained_sindy",
    "two_stage_ensemble",
    "structure_weighted_ensemble",
    "probability_fusion",
    "get_uncertainty_report",
    "EnsembleSCResult",
    # Library construction
    "build_library_2d",
    "build_library_3d",
    "build_library_nd",
]
