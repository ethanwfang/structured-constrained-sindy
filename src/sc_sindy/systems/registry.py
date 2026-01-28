"""
System registry for easy access to dynamical systems.

This module provides a registry pattern for accessing systems by name
and utilities for working with multiple systems.
"""

from typing import Dict, List, Optional, Type

from .base import DynamicalSystem
from .biological import CoupledBrusselator, LotkaVolterra, SelkovGlycolysis, SIRModel
from .chaotic import ChenSystem, DoublePendulum, Lorenz, Rossler
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
    # Chaotic
    "lorenz": Lorenz,
    "rossler": Rossler,
    "chen": ChenSystem,
    "double_pendulum": DoublePendulum,
}


# Categorized systems
SYSTEM_CATEGORIES = {
    "oscillators": ["vanderpol", "duffing", "damped_harmonic", "forced_oscillator"],
    "biological": ["lotka_volterra", "selkov", "brusselator", "sir"],
    "chaotic": ["lorenz", "rossler", "chen", "double_pendulum"],
    "2d": ["vanderpol", "duffing", "damped_harmonic", "lotka_volterra", "selkov", "brusselator"],
    "3d": ["lorenz", "rossler", "chen", "sir"],
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
