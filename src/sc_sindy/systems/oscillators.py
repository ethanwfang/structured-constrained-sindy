"""
Oscillator dynamical systems.

This module provides various oscillator systems including Van der Pol,
Duffing, and damped harmonic oscillators.
"""

from typing import List

import numpy as np

from .base import DynamicalSystem


class VanDerPol(DynamicalSystem):
    """
    Van der Pol oscillator.

    Equations:
        dx/dt = y
        dy/dt = mu*(1 - x^2)*y - x

    In second-order form: x'' - mu*(1 - x^2)*x' + x = 0

    Parameters
    ----------
    mu : float, optional
        Nonlinearity parameter (default: 1.5).

    Notes
    -----
    - mu = 0: Simple harmonic oscillator
    - mu > 0: Limit cycle oscillations
    - Larger mu leads to more relaxation-type oscillations
    """

    def __init__(self, mu: float = 1.5):
        super().__init__("Van der Pol", 2, {"mu": mu})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        mu = self.params["mu"]
        return np.array([y, mu * (1 - x**2) * y - x])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: y
        dy/dt uses: x, y, x^2*y (written as xxy)
        """
        mask = np.zeros((2, len(term_names)), dtype=bool)

        if "y" in term_names:
            mask[0, term_names.index("y")] = True

        for term in ["x", "y", "xxy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Return actual coefficient values."""
        mu = self.params["mu"]
        xi = np.zeros((2, len(term_names)))

        if "y" in term_names:
            xi[0, term_names.index("y")] = 1.0

        if "x" in term_names:
            xi[1, term_names.index("x")] = -1.0
        if "y" in term_names:
            xi[1, term_names.index("y")] = mu
        if "xxy" in term_names:
            xi[1, term_names.index("xxy")] = -mu

        return xi


class DuffingOscillator(DynamicalSystem):
    """
    Duffing oscillator.

    Equations:
        dx/dt = y
        dy/dt = -delta*y - alpha*x - beta*x^3

    Parameters
    ----------
    alpha : float, optional
        Linear stiffness (default: 1.0).
    beta : float, optional
        Nonlinear stiffness (default: 1.0).
    delta : float, optional
        Damping coefficient (default: 0.2).

    Notes
    -----
    - beta > 0: Hardening spring
    - beta < 0: Softening spring
    - Can exhibit chaotic behavior when driven
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, delta: float = 0.2):
        super().__init__("Duffing Oscillator", 2, {"alpha": alpha, "beta": beta, "delta": delta})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        return np.array([y, -p["delta"] * y - p["alpha"] * x - p["beta"] * x**3])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: y
        dy/dt uses: x, y, x^3 (written as xxx)
        """
        mask = np.zeros((2, len(term_names)), dtype=bool)

        if "y" in term_names:
            mask[0, term_names.index("y")] = True

        for term in ["x", "y", "xxx"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Return actual coefficient values."""
        p = self.params
        xi = np.zeros((2, len(term_names)))

        if "y" in term_names:
            xi[0, term_names.index("y")] = 1.0

        if "x" in term_names:
            xi[1, term_names.index("x")] = -p["alpha"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["delta"]
        if "xxx" in term_names:
            xi[1, term_names.index("xxx")] = -p["beta"]

        return xi


class DampedHarmonicOscillator(DynamicalSystem):
    """
    Damped harmonic oscillator.

    Equations:
        dx/dt = y
        dy/dt = -omega0^2 * x - 2*zeta*omega0 * y

    In second-order form: x'' + 2*zeta*omega0*x' + omega0^2*x = 0

    Parameters
    ----------
    omega0 : float, optional
        Natural frequency (default: 2*pi for 1 Hz).
    zeta : float, optional
        Damping ratio (default: 0.1).

    Notes
    -----
    - zeta < 1: Underdamped (oscillates while decaying)
    - zeta = 1: Critically damped
    - zeta > 1: Overdamped (no oscillation)
    """

    def __init__(self, omega0: float = 2 * np.pi, zeta: float = 0.1):
        super().__init__("Damped Harmonic Oscillator", 2, {"omega0": omega0, "zeta": zeta})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        return np.array([y, -p["omega0"] ** 2 * x - 2 * p["zeta"] * p["omega0"] * y])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: y
        dy/dt uses: x, y (linear system)
        """
        mask = np.zeros((2, len(term_names)), dtype=bool)

        if "y" in term_names:
            mask[0, term_names.index("y")] = True

        for term in ["x", "y"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Return actual coefficient values."""
        p = self.params
        xi = np.zeros((2, len(term_names)))

        if "y" in term_names:
            xi[0, term_names.index("y")] = 1.0

        if "x" in term_names:
            xi[1, term_names.index("x")] = -p["omega0"] ** 2
        if "y" in term_names:
            xi[1, term_names.index("y")] = -2 * p["zeta"] * p["omega0"]

        return xi


class ForcedOscillator(DynamicalSystem):
    """
    Forced harmonic oscillator with sinusoidal driving.

    Equations:
        dx/dt = y
        dy/dt = -omega0^2 * x - 2*zeta*omega0 * y + A*sin(omega*t)

    Parameters
    ----------
    omega0 : float, optional
        Natural frequency (default: 1.0).
    zeta : float, optional
        Damping ratio (default: 0.1).
    A : float, optional
        Forcing amplitude (default: 1.0).
    omega : float, optional
        Forcing frequency (default: 1.0).

    Notes
    -----
    This system is non-autonomous (explicitly depends on time).
    The library should include sin(omega*t) and cos(omega*t) terms.
    """

    def __init__(self, omega0: float = 1.0, zeta: float = 0.1, A: float = 1.0, omega: float = 1.0):
        super().__init__(
            "Forced Oscillator", 2, {"omega0": omega0, "zeta": zeta, "A": A, "omega": omega}
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        forcing = p["A"] * np.sin(p["omega"] * t)
        return np.array([y, -p["omega0"] ** 2 * x - 2 * p["zeta"] * p["omega0"] * y + forcing])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: y
        dy/dt uses: x, y, sin(t) or constant forcing approximation
        """
        mask = np.zeros((2, len(term_names)), dtype=bool)

        if "y" in term_names:
            mask[0, term_names.index("y")] = True

        for term in ["x", "y"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        return mask
