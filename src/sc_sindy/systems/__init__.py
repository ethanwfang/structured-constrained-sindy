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
from .canonical import (
    CubicOscillator,
    HopfNormalForm,
    LinearOscillator,
    QuadraticOscillator,
    RayleighOscillator,
)
from .chaotic import (
    ChenSystem,
    DoublePendulum,
    Lorenz,
    Rossler,
)
from .chaotic_3d import (
    AizawaAttractor,
    HalvorsenAttractor,
    RabinovichFabrikant,
    SprottB,
    SprottD,
    ThomasAttractor,
)
from .coupled_4d import (
    CoupledDuffing,
    CoupledFitzHughNagumo,
    CoupledVanDerPol,
    Cubic4DSystem,
    HyperchaoticLorenz,
    HyperchaoticRossler,
    LorenzExtended4D,
    LotkaVolterra4D,
    MixedCoupledOscillator,
    SimpleQuadratic4D,
)
from .ecological import (
    CompetitiveExclusion,
    MutualismModel,
    PredatorPreyTypeII,
    SISEpidemic,
    SimplePredatorPrey,
)
from .neural import (
    FitzHughNagumo,
    HindmarshRose2D,
    MorrisLecar,
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
    # Ecological (with xy interaction)
    "CompetitiveExclusion",
    "MutualismModel",
    "SISEpidemic",
    "PredatorPreyTypeII",
    "SimplePredatorPrey",
    # Neural
    "FitzHughNagumo",
    "MorrisLecar",
    "HindmarshRose2D",
    # Canonical
    "HopfNormalForm",
    "CubicOscillator",
    "QuadraticOscillator",
    "RayleighOscillator",
    "LinearOscillator",
    # Chaotic
    "Lorenz",
    "Rossler",
    "ChenSystem",
    "DoublePendulum",
    # Chaotic 3D (new)
    "ThomasAttractor",
    "HalvorsenAttractor",
    "SprottB",
    "SprottD",
    "RabinovichFabrikant",
    "AizawaAttractor",
    # Coupled 4D (new)
    "CoupledVanDerPol",
    "CoupledDuffing",
    "HyperchaoticLorenz",
    "HyperchaoticRossler",
    "LotkaVolterra4D",
    "CoupledFitzHughNagumo",
    "MixedCoupledOscillator",
    "LorenzExtended4D",
    "SimpleQuadratic4D",
    "Cubic4DSystem",
    # Registry
    "get_system",
    "list_systems",
    "get_benchmark_systems",
    "get_2d_benchmark_systems",
    "system_info",
    "SYSTEM_REGISTRY",
    "SYSTEM_CATEGORIES",
]
