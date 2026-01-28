"""
Dynamical systems library.

This module provides a collection of dynamical systems for testing
and benchmarking SINDy algorithms.
"""

from .base import DynamicalSystem
from .biological import (
    CoupledBrusselator,
    LotkaVolterra,
    SelkovGlycolysis,
    SIRModel,
)
from .chaotic import (
    ChenSystem,
    DoublePendulum,
    Lorenz,
    Rossler,
)
from .oscillators import (
    DampedHarmonicOscillator,
    DuffingOscillator,
    ForcedOscillator,
    VanDerPol,
)
from .registry import (
    SYSTEM_CATEGORIES,
    SYSTEM_REGISTRY,
    get_2d_benchmark_systems,
    get_benchmark_systems,
    get_system,
    list_systems,
    system_info,
)

__all__ = [
    # Base class
    "DynamicalSystem",
    # Oscillators
    "VanDerPol",
    "DuffingOscillator",
    "DampedHarmonicOscillator",
    "ForcedOscillator",
    # Biological
    "LotkaVolterra",
    "SelkovGlycolysis",
    "CoupledBrusselator",
    "SIRModel",
    # Chaotic
    "Lorenz",
    "Rossler",
    "ChenSystem",
    "DoublePendulum",
    # Registry
    "get_system",
    "list_systems",
    "get_benchmark_systems",
    "get_2d_benchmark_systems",
    "system_info",
    "SYSTEM_REGISTRY",
    "SYSTEM_CATEGORIES",
]
