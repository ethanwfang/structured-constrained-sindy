"""
Canonical dynamical systems from physics and mathematics.

This module provides well-known dynamical systems that serve as
canonical examples in nonlinear dynamics and bifurcation theory.
"""

from typing import List

import numpy as np

from .base import DynamicalSystem


class HopfNormalForm(DynamicalSystem):
    """
    Hopf bifurcation normal form.

    The simplest system exhibiting a Hopf bifurcation.

    Equations (in Cartesian coordinates):
        dx/dt = mu*x - omega*y - (x^2 + y^2)*x
              = mu*x - omega*y - x^3 - x*y^2  (terms: x, y, xxx, xyy)
        dy/dt = omega*x + mu*y - (x^2 + y^2)*y
              = omega*x + mu*y - x^2*y - y^3  (terms: x, y, xxy, yyy)

    Parameters
    ----------
    mu : float
        Bifurcation parameter (default: 0.1). mu > 0 gives limit cycle.
    omega : float
        Angular frequency (default: 1.0).
    """

    def __init__(self, mu: float = 0.1, omega: float = 1.0):
        super().__init__("Hopf Normal Form", 2, {"mu": mu, "omega": omega})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        r_sq = x**2 + y**2
        dx = p["mu"] * x - p["omega"] * y - r_sq * x
        dy = p["omega"] * x + p["mu"] * y - r_sq * y
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: x, y, xxx, xyy; dy/dt uses: x, y, xxy, yyy"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["x", "y", "xxx", "xyy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["x", "y", "xxy", "yyy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "x" in term_names:
            xi[0, term_names.index("x")] = p["mu"]
        if "y" in term_names:
            xi[0, term_names.index("y")] = -p["omega"]
        if "xxx" in term_names:
            xi[0, term_names.index("xxx")] = -1.0
        if "xyy" in term_names:
            xi[0, term_names.index("xyy")] = -1.0
        if "x" in term_names:
            xi[1, term_names.index("x")] = p["omega"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = p["mu"]
        if "xxy" in term_names:
            xi[1, term_names.index("xxy")] = -1.0
        if "yyy" in term_names:
            xi[1, term_names.index("yyy")] = -1.0
        return xi


class CubicOscillator(DynamicalSystem):
    """
    Cubic nonlinear oscillator.

    A simple oscillator with cubic nonlinearity (like Duffing but different form).

    Equations:
        dx/dt = y  (terms: y)
        dy/dt = -omega^2*x - delta*y - beta*x^3  (terms: x, y, xxx)

    Parameters
    ----------
    omega : float
        Natural frequency (default: 1.0).
    delta : float
        Damping coefficient (default: 0.1).
    beta : float
        Cubic nonlinearity strength (default: 0.5).
    """

    def __init__(self, omega: float = 1.0, delta: float = 0.1, beta: float = 0.5):
        super().__init__(
            "Cubic Oscillator", 2, {"omega": omega, "delta": delta, "beta": beta}
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        dx = y
        dy = -(p["omega"] ** 2) * x - p["delta"] * y - p["beta"] * x**3
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: y; dy/dt uses: x, y, xxx"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        if "y" in term_names:
            mask[0, term_names.index("y")] = True
        for term in ["x", "y", "xxx"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "y" in term_names:
            xi[0, term_names.index("y")] = 1.0
        if "x" in term_names:
            xi[1, term_names.index("x")] = -(p["omega"] ** 2)
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["delta"]
        if "xxx" in term_names:
            xi[1, term_names.index("xxx")] = -p["beta"]
        return xi


class QuadraticOscillator(DynamicalSystem):
    """
    Oscillator with quadratic nonlinearity.

    Equations:
        dx/dt = y  (terms: y)
        dy/dt = -omega^2*x - delta*y + alpha*x^2  (terms: x, y, xx)

    Parameters
    ----------
    omega : float
        Natural frequency (default: 1.0).
    delta : float
        Damping coefficient (default: 0.15).
    alpha : float
        Quadratic nonlinearity (default: 0.3).
    """

    def __init__(self, omega: float = 1.0, delta: float = 0.15, alpha: float = 0.3):
        super().__init__(
            "Quadratic Oscillator", 2, {"omega": omega, "delta": delta, "alpha": alpha}
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        dx = y
        dy = -(p["omega"] ** 2) * x - p["delta"] * y + p["alpha"] * x**2
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: y; dy/dt uses: x, y, xx"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        if "y" in term_names:
            mask[0, term_names.index("y")] = True
        for term in ["x", "y", "xx"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "y" in term_names:
            xi[0, term_names.index("y")] = 1.0
        if "x" in term_names:
            xi[1, term_names.index("x")] = -(p["omega"] ** 2)
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["delta"]
        if "xx" in term_names:
            xi[1, term_names.index("xx")] = p["alpha"]
        return xi


class RayleighOscillator(DynamicalSystem):
    """
    Rayleigh oscillator with velocity-dependent damping.

    Equations:
        dx/dt = y  (terms: y)
        dy/dt = -x + mu*(1 - y^2)*y = -x + mu*y - mu*y^3  (terms: x, y, yyy)

    Parameters
    ----------
    mu : float
        Nonlinear damping parameter (default: 1.0).
    """

    def __init__(self, mu: float = 1.0):
        super().__init__("Rayleigh Oscillator", 2, {"mu": mu})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        dx = y
        dy = -x + p["mu"] * (1 - y**2) * y
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: y; dy/dt uses: x, y, yyy"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        if "y" in term_names:
            mask[0, term_names.index("y")] = True
        for term in ["x", "y", "yyy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "y" in term_names:
            xi[0, term_names.index("y")] = 1.0
        if "x" in term_names:
            xi[1, term_names.index("x")] = -1.0
        if "y" in term_names:
            xi[1, term_names.index("y")] = p["mu"]
        if "yyy" in term_names:
            xi[1, term_names.index("yyy")] = -p["mu"]
        return xi


class LinearOscillator(DynamicalSystem):
    """
    Simple linear oscillator (harmonic oscillator with damping).

    Equations:
        dx/dt = y  (terms: y)
        dy/dt = -omega^2*x - 2*zeta*omega*y  (terms: x, y)

    Parameters
    ----------
    omega : float
        Natural frequency (default: 1.0).
    zeta : float
        Damping ratio (default: 0.1).
    """

    def __init__(self, omega: float = 1.0, zeta: float = 0.1):
        super().__init__("Linear Oscillator", 2, {"omega": omega, "zeta": zeta})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        dx = y
        dy = -(p["omega"] ** 2) * x - 2 * p["zeta"] * p["omega"] * y
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: y; dy/dt uses: x, y"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        if "y" in term_names:
            mask[0, term_names.index("y")] = True
        for term in ["x", "y"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "y" in term_names:
            xi[0, term_names.index("y")] = 1.0
        if "x" in term_names:
            xi[1, term_names.index("x")] = -(p["omega"] ** 2)
        if "y" in term_names:
            xi[1, term_names.index("y")] = -2 * p["zeta"] * p["omega"]
        return xi
