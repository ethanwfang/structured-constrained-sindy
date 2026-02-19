"""
System registry for easy access to dynamical systems.

This module provides a registry pattern for accessing systems by name
and utilities for working with multiple systems.
"""

from typing import Dict, List, Optional, Type

from .base import DynamicalSystem
from .biological import CoupledBrusselator, LotkaVolterra, SelkovGlycolysis, SIRModel
from .canonical import (
    CubicOscillator,
    HopfNormalForm,
    LinearOscillator,
    QuadraticOscillator,
    RayleighOscillator,
)
from .chaotic import ChenSystem, DoublePendulum, Lorenz, Rossler
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
    HyperchaoticLorenz,
    HyperchaoticRossler,
    LotkaVolterra4D,
)
from .ecological import (
    CompetitiveExclusion,
    MutualismModel,
    PredatorPreyTypeII,
    SISEpidemic,
    SimplePredatorPrey,
)
from .neural import FitzHughNagumo, HindmarshRose2D, MorrisLecar
from .oscillators import DampedHarmonicOscillator, DuffingOscillator, ForcedOscillator, VanDerPol

# Registry of all available systems
SYSTEM_REGISTRY: Dict[str, Type[DynamicalSystem]] = {
    # Oscillators
    "vanderpol": VanDerPol,
    "duffing": DuffingOscillator,
    "damped_harmonic": DampedHarmonicOscillator,
    "forced_oscillator": ForcedOscillator,
    # Biological
    "lotka_volterra": LotkaVolterra,
    "selkov": SelkovGlycolysis,
    "brusselator": CoupledBrusselator,
    "sir": SIRModel,
    # Ecological (with xy interaction)
    "competitive_exclusion": CompetitiveExclusion,
    "mutualism": MutualismModel,
    "sis_epidemic": SISEpidemic,
    "predator_prey_type2": PredatorPreyTypeII,
    "simple_predator_prey": SimplePredatorPrey,
    # Neural
    "fitzhugh_nagumo": FitzHughNagumo,
    "morris_lecar": MorrisLecar,
    "hindmarsh_rose": HindmarshRose2D,
    # Canonical
    "hopf": HopfNormalForm,
    "cubic_oscillator": CubicOscillator,
    "quadratic_oscillator": QuadraticOscillator,
    "rayleigh": RayleighOscillator,
    "linear_oscillator": LinearOscillator,
    # Chaotic
    "lorenz": Lorenz,
    "rossler": Rossler,
    "chen": ChenSystem,
    "double_pendulum": DoublePendulum,
    # Chaotic 3D (new)
    "thomas": ThomasAttractor,
    "halvorsen": HalvorsenAttractor,
    "sprott_b": SprottB,
    "sprott_d": SprottD,
    "rabinovich_fabrikant": RabinovichFabrikant,
    "aizawa": AizawaAttractor,
    # Coupled 4D (new)
    "coupled_vanderpol": CoupledVanDerPol,
    "coupled_duffing": CoupledDuffing,
    "hyperchaotic_lorenz": HyperchaoticLorenz,
    "hyperchaotic_rossler": HyperchaoticRossler,
    "lotka_volterra_4d": LotkaVolterra4D,
    "coupled_fitzhugh_nagumo": CoupledFitzHughNagumo,
}


# Categorized systems
SYSTEM_CATEGORIES = {
    "oscillators": [
        "vanderpol", "duffing", "damped_harmonic", "forced_oscillator",
        "cubic_oscillator", "quadratic_oscillator", "rayleigh", "linear_oscillator",
    ],
    "biological": ["lotka_volterra", "selkov", "brusselator", "sir"],
    "ecological": [
        "competitive_exclusion", "mutualism", "sis_epidemic",
        "predator_prey_type2", "simple_predator_prey",
    ],
    "neural": ["fitzhugh_nagumo", "morris_lecar", "hindmarsh_rose"],
    "canonical": ["hopf", "cubic_oscillator", "quadratic_oscillator", "rayleigh", "linear_oscillator"],
    "chaotic": [
        "lorenz", "rossler", "chen", "double_pendulum",
        "thomas", "halvorsen", "sprott_b", "sprott_d", "rabinovich_fabrikant", "aizawa",
        "hyperchaotic_lorenz", "hyperchaotic_rossler",
    ],
    "2d": [
        "vanderpol", "duffing", "damped_harmonic", "forced_oscillator",
        "lotka_volterra", "selkov", "brusselator",
        "competitive_exclusion", "mutualism", "sis_epidemic",
        "predator_prey_type2", "simple_predator_prey",
        "fitzhugh_nagumo", "morris_lecar", "hindmarsh_rose",
        "hopf", "cubic_oscillator", "quadratic_oscillator", "rayleigh", "linear_oscillator",
    ],
    "3d": [
        "lorenz", "rossler", "chen", "sir",
        "thomas", "halvorsen", "sprott_b", "sprott_d", "rabinovich_fabrikant", "aizawa",
    ],
    "4d": [
        "coupled_vanderpol", "coupled_duffing",
        "hyperchaotic_lorenz", "hyperchaotic_rossler",
        "lotka_volterra_4d", "coupled_fitzhugh_nagumo",
    ],
    # Systems with xy bilinear interaction term
    "xy_interaction": [
        "lotka_volterra", "competitive_exclusion", "mutualism", "sis_epidemic",
        "predator_prey_type2", "simple_predator_prey", "morris_lecar",
    ],
    # Systems with bilinear interactions (any dimension)
    "bilinear": [
        "lotka_volterra", "competitive_exclusion", "mutualism", "sis_epidemic",
        "predator_prey_type2", "simple_predator_prey", "morris_lecar",
        "lorenz", "chen", "sir", "sprott_b", "rabinovich_fabrikant", "aizawa",
        "hyperchaotic_lorenz", "hyperchaotic_rossler", "lotka_volterra_4d",
    ],
}


def get_system(name: str, **params) -> DynamicalSystem:
    """
    Get a dynamical system by name.

    Parameters
    ----------
    name : str
        System name (case-insensitive).
    **params
        Parameters to pass to the system constructor.

    Returns
    -------
    system : DynamicalSystem
        Instantiated system.

    Raises
    ------
    ValueError
        If system name is not found.

    Examples
    --------
    >>> system = get_system("vanderpol", mu=2.0)
    >>> system = get_system("lorenz")  # Default parameters
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower not in SYSTEM_REGISTRY:
        available = ", ".join(sorted(SYSTEM_REGISTRY.keys()))
        raise ValueError(f"Unknown system '{name}'. Available: {available}")

    return SYSTEM_REGISTRY[name_lower](**params)


def list_systems(category: Optional[str] = None) -> List[str]:
    """
    List available systems.

    Parameters
    ----------
    category : str, optional
        Filter by category (e.g., "oscillators", "2d", "chaotic").

    Returns
    -------
    systems : List[str]
        List of system names.

    Examples
    --------
    >>> list_systems()  # All systems
    >>> list_systems("2d")  # Only 2D systems
    """
    if category is None:
        return sorted(SYSTEM_REGISTRY.keys())

    category_lower = category.lower()
    if category_lower not in SYSTEM_CATEGORIES:
        available = ", ".join(sorted(SYSTEM_CATEGORIES.keys()))
        raise ValueError(f"Unknown category '{category}'. Available: {available}")

    return SYSTEM_CATEGORIES[category_lower]


def get_benchmark_systems() -> List[DynamicalSystem]:
    """
    Get standard benchmark systems for testing.

    Returns
    -------
    systems : List[DynamicalSystem]
        List of systems commonly used for benchmarking.
    """
    return [
        VanDerPol(mu=1.5),
        SelkovGlycolysis(a=0.1, b=0.6),
        CoupledBrusselator(A=1.0, B=2.5),
        LotkaVolterra(alpha=1.0, beta=0.5, delta=0.5, gamma=1.0),
        Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0),
    ]


def get_2d_benchmark_systems() -> List[DynamicalSystem]:
    """
    Get 2D benchmark systems for testing.

    Returns
    -------
    systems : List[DynamicalSystem]
        List of 2D systems for benchmarking.
    """
    return [
        VanDerPol(mu=1.5),
        SelkovGlycolysis(a=0.1, b=0.6),
        CoupledBrusselator(A=1.0, B=2.5),
        LotkaVolterra(alpha=1.0, beta=0.5, delta=0.5, gamma=1.0),
        DuffingOscillator(alpha=1.0, beta=1.0, delta=0.2),
    ]


def system_info(name: str) -> Dict:
    """
    Get information about a system.

    Parameters
    ----------
    name : str
        System name.

    Returns
    -------
    info : dict
        Dictionary with system information.
    """
    system_class = SYSTEM_REGISTRY.get(name.lower())
    if system_class is None:
        raise ValueError(f"Unknown system '{name}'")

    # Create instance with default parameters
    system = system_class()

    return {
        "name": system.name,
        "dimension": system.dim,
        "parameters": system.params,
        "class": system_class.__name__,
        "docstring": system_class.__doc__,
    }
