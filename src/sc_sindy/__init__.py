"""
Structure-Constrained SINDy (SC-SINDy)
======================================

A comprehensive implementation of Structure-Constrained SINDy for discovering
governing equations from data with learned structural priors.

Main Features
-------------
- Standard SINDy with Sequential Thresholded Least Squares (STLS)
- Structure-Constrained SINDy with neural network priors
- Comprehensive dynamical systems library for testing
- Feature extraction for trajectory characterization
- Extensive evaluation metrics

Quick Start
-----------
>>> from sc_sindy import sindy_stls, sindy_structure_constrained, build_library_2d
>>> from sc_sindy.systems import VanDerPol
>>> from sc_sindy.derivatives import compute_derivatives_finite_diff
>>>
>>> # Generate data
>>> system = VanDerPol(mu=1.5)
>>> t = np.linspace(0, 50, 5000)
>>> x = system.generate_trajectory(np.array([2.0, 0.0]), t)
>>>
>>> # Compute derivatives
>>> dt = t[1] - t[0]
>>> x_dot = compute_derivatives_finite_diff(x, dt)
>>>
>>> # Build library and run SINDy
>>> Theta, term_names = build_library_2d(x, poly_order=3)
>>> xi, elapsed = sindy_stls(Theta, x_dot, threshold=0.1)

References
----------
The Structure-Constrained SINDy method achieves 97-1568x improvement over
standard SINDy on challenging systems by using learned structural priors.

Default threshold of 0.3 is based on ablation study showing robust performance
in range [0.2, 0.8].
"""

__version__ = "0.1.0"
__author__ = "Structure-Constrained SINDy Project"

# Core algorithms
from .core import (
    DEFAULT_STLS_THRESHOLD,
    DEFAULT_STRUCTURE_THRESHOLD,
    build_library_2d,
    build_library_3d,
    build_library_nd,
    get_recommended_threshold,
    sindy_ridge,
    sindy_stls,
    sindy_structure_constrained,
    sindy_structure_constrained_soft,
)

# Derivative computation
from .derivatives import (
    compute_derivatives_adaptive,
    compute_derivatives_finite_diff,
    compute_derivatives_spline,
)

# Metrics
from .metrics import (
    compute_coefficient_error,
    compute_reconstruction_error,
    compute_structure_metrics,
)

# Dynamical systems
from .systems import (
    CoupledBrusselator,
    DampedHarmonicOscillator,
    DuffingOscillator,
    DynamicalSystem,
    Lorenz,
    LotkaVolterra,
    Rossler,
    SelkovGlycolysis,
    VanDerPol,
    get_benchmark_systems,
    get_system,
    list_systems,
)

# Utilities
from .utils import (
    format_equation,
    load_lynx_hare_data,
    print_equations,
)

# Network (may not be available if PyTorch is not installed)
try:
    from .network import (
        TORCH_AVAILABLE,
        StructureNetwork,
        StructurePredictor,
        create_oracle_network_probs,
        extract_trajectory_features,
        train_structure_network,
        train_structure_network_with_split,
    )
except ImportError:
    TORCH_AVAILABLE = False

# Evaluation framework (for fair evaluation without oracle)
from .evaluation import (
    SCSINDyEvaluator,
    get_split,
    TRAIN_SYSTEMS_2D,
    TEST_SYSTEMS_2D,
)

__all__ = [
    # Version
    "__version__",
    # Core algorithms
    "sindy_stls",
    "sindy_ridge",
    "sindy_structure_constrained",
    "sindy_structure_constrained_soft",
    "get_recommended_threshold",
    "build_library_2d",
    "build_library_3d",
    "build_library_nd",
    "DEFAULT_STLS_THRESHOLD",
    "DEFAULT_STRUCTURE_THRESHOLD",
    # Derivatives
    "compute_derivatives_finite_diff",
    "compute_derivatives_spline",
    "compute_derivatives_adaptive",
    # Systems
    "DynamicalSystem",
    "VanDerPol",
    "DuffingOscillator",
    "DampedHarmonicOscillator",
    "LotkaVolterra",
    "SelkovGlycolysis",
    "CoupledBrusselator",
    "Lorenz",
    "Rossler",
    "get_system",
    "list_systems",
    "get_benchmark_systems",
    # Metrics
    "compute_structure_metrics",
    "compute_coefficient_error",
    "compute_reconstruction_error",
    # Utilities
    "format_equation",
    "print_equations",
    "load_lynx_hare_data",
    # Network
    "TORCH_AVAILABLE",
    # Evaluation
    "SCSINDyEvaluator",
    "get_split",
    "TRAIN_SYSTEMS_2D",
    "TEST_SYSTEMS_2D",
]

# Conditional exports for network module
if TORCH_AVAILABLE:
    __all__.extend(
        [
            "StructureNetwork",
            "StructurePredictor",
            "extract_trajectory_features",
            "train_structure_network",
            "train_structure_network_with_split",
            "create_oracle_network_probs",
        ]
    )
