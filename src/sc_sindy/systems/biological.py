"""
Biological dynamical systems.

This module provides dynamical systems from biology and biochemistry,
including predator-prey models and metabolic oscillators.
"""

from typing import List

import numpy as np

from .base import DynamicalSystem


class LotkaVolterra(DynamicalSystem):
    """
    Lotka-Volterra predator-prey model.

    Equations:
        dx/dt = alpha*x - beta*x*y   (prey growth - predation)
        dy/dt = delta*x*y - gamma*y   (predator growth - death)

    Parameters
    ----------
    alpha : float, optional
        Prey birth rate (default: 1.0).
    beta : float, optional
        Predation rate (default: 0.5).
    delta : float, optional
        Predator reproduction rate (default: 0.5).
    gamma : float, optional
        Predator death rate (default: 1.0).

    Notes
    -----
    Classical model for population dynamics. Exhibits closed orbits
    in phase space (conservative system).
    """

    def __init__(
        self, alpha: float = 1.0, beta: float = 0.5, delta: float = 0.5, gamma: float = 1.0
    ):
        super().__init__(
            "Lotka-Volterra", 2, {"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma}
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        return np.array([p["alpha"] * x - p["beta"] * x * y, p["delta"] * x * y - p["gamma"] * y])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: x, xy
        dy/dt uses: y, xy
        """
        mask = np.zeros((2, len(term_names)), dtype=bool)

        for term in ["x", "xy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True

        for term in ["y", "xy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Return actual coefficient values."""
        p = self.params
        xi = np.zeros((2, len(term_names)))

        if "x" in term_names:
            xi[0, term_names.index("x")] = p["alpha"]
        if "xy" in term_names:
            xi[0, term_names.index("xy")] = -p["beta"]

        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["gamma"]
        if "xy" in term_names:
            xi[1, term_names.index("xy")] = p["delta"]

        return xi


class SelkovGlycolysis(DynamicalSystem):
    """
    Selkov model for glycolytic oscillations.

    Equations:
        dx/dt = -x + a*y + x^2*y
        dy/dt = b - a*y - x^2*y

    Parameters
    ----------
    a : float, optional
        Feedback parameter (default: 0.1).
    b : float, optional
        Input rate (default: 0.6).

    Notes
    -----
    Models oscillations in glycolysis pathway. Can exhibit limit cycles
    for appropriate parameter values.
    """

    def __init__(self, a: float = 0.1, b: float = 0.6):
        super().__init__("Selkov Glycolysis", 2, {"a": a, "b": b})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        return np.array([-x + p["a"] * y + x**2 * y, p["b"] - p["a"] * y - x**2 * y])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: x, y, x^2*y (xxy)
        dy/dt uses: 1 (constant), y, x^2*y (xxy)
        """
        mask = np.zeros((2, len(term_names)), dtype=bool)

        for term in ["x", "y", "xxy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True

        for term in ["1", "y", "xxy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Return actual coefficient values."""
        p = self.params
        xi = np.zeros((2, len(term_names)))

        if "x" in term_names:
            xi[0, term_names.index("x")] = -1.0
        if "y" in term_names:
            xi[0, term_names.index("y")] = p["a"]
        if "xxy" in term_names:
            xi[0, term_names.index("xxy")] = 1.0

        if "1" in term_names:
            xi[1, term_names.index("1")] = p["b"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["a"]
        if "xxy" in term_names:
            xi[1, term_names.index("xxy")] = -1.0

        return xi


class CoupledBrusselator(DynamicalSystem):
    """
    Brusselator model for chemical oscillations.

    Equations:
        dx/dt = A + x^2*y - (B+1)*x
        dy/dt = B*x - x^2*y

    Parameters
    ----------
    A : float, optional
        Input concentration (default: 1.0).
    B : float, optional
        Control parameter (default: 2.5).

    Notes
    -----
    Model for autocatalytic chemical reactions. Shows Hopf bifurcation
    when B > 1 + A^2.
    """

    def __init__(self, A: float = 1.0, B: float = 2.5):
        super().__init__("Coupled Brusselator", 2, {"A": A, "B": B})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        return np.array([p["A"] + x**2 * y - (p["B"] + 1) * x, p["B"] * x - x**2 * y])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: 1 (constant), x, x^2*y (xxy)
        dy/dt uses: x, x^2*y (xxy)
        """
        mask = np.zeros((2, len(term_names)), dtype=bool)

        for term in ["1", "x", "xxy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True

        for term in ["x", "xxy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Return actual coefficient values."""
        p = self.params
        xi = np.zeros((2, len(term_names)))

        if "1" in term_names:
            xi[0, term_names.index("1")] = p["A"]
        if "x" in term_names:
            xi[0, term_names.index("x")] = -(p["B"] + 1)
        if "xxy" in term_names:
            xi[0, term_names.index("xxy")] = 1.0

        if "x" in term_names:
            xi[1, term_names.index("x")] = p["B"]
        if "xxy" in term_names:
            xi[1, term_names.index("xxy")] = -1.0

        return xi


class SIRModel(DynamicalSystem):
    """
    SIR epidemiological model.

    Equations:
        dS/dt = -beta*S*I          (susceptibles getting infected)
        dI/dt = beta*S*I - gamma*I  (infection spread - recovery)
        dR/dt = gamma*I             (recovery)

    Parameters
    ----------
    beta : float, optional
        Infection rate (default: 0.3).
    gamma : float, optional
        Recovery rate (default: 0.1).

    Notes
    -----
    Note: This is a 3D system but S + I + R = N (constant), so effectively 2D.
    We track all three for completeness.
    """

    def __init__(self, beta: float = 0.3, gamma: float = 0.1):
        super().__init__("SIR Model", 3, {"beta": beta, "gamma": gamma})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        S, I, R = state
        p = self.params
        return np.array([-p["beta"] * S * I, p["beta"] * S * I - p["gamma"] * I, p["gamma"] * I])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dS/dt uses: S*I (xy if x=S, y=I)
        dI/dt uses: S*I, I
        dR/dt uses: I
        """
        mask = np.zeros((3, len(term_names)), dtype=bool)

        # Map: x=S, y=I, z=R
        if "xy" in term_names:
            mask[0, term_names.index("xy")] = True  # -beta*S*I

        for term in ["xy", "y"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        if "y" in term_names:
            mask[2, term_names.index("y")] = True

        return mask
