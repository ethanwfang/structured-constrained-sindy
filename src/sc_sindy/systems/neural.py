"""
Neural and excitable dynamical systems.

This module provides dynamical systems from neuroscience and excitable media,
including simplified models of neural activity.
"""

from typing import List

import numpy as np

from .base import DynamicalSystem


class FitzHughNagumo(DynamicalSystem):
    """
    FitzHugh-Nagumo model for neural excitability.

    A simplified 2D reduction of the Hodgkin-Huxley model.

    Equations:
        dv/dt = v - v^3/3 - w + I  (terms: 1, x, xxx)
        dw/dt = epsilon*(v + a - b*w)  (terms: 1, x, y)

    Parameters
    ----------
    a : float
        Recovery parameter (default: 0.7).
    b : float
        Recovery time scale (default: 0.8).
    epsilon : float
        Time scale separation (default: 0.08).
    I : float
        External current (default: 0.5).
    """

    def __init__(
        self, a: float = 0.7, b: float = 0.8, epsilon: float = 0.08, I: float = 0.5
    ):
        super().__init__(
            "FitzHugh-Nagumo",
            2,
            {"a": a, "b": b, "epsilon": epsilon, "I": I},
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        v, w = state
        p = self.params
        dv = v - v**3 / 3 - w + p["I"]
        dw = p["epsilon"] * (v + p["a"] - p["b"] * w)
        return np.array([dv, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dv/dt uses: 1, x, y, xxx; dw/dt uses: 1, x, y"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["1", "x", "y", "xxx"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["1", "x", "y"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "1" in term_names:
            xi[0, term_names.index("1")] = p["I"]
        if "x" in term_names:
            xi[0, term_names.index("x")] = 1.0
        if "y" in term_names:
            xi[0, term_names.index("y")] = -1.0
        if "xxx" in term_names:
            xi[0, term_names.index("xxx")] = -1.0 / 3.0
        if "1" in term_names:
            xi[1, term_names.index("1")] = p["epsilon"] * p["a"]
        if "x" in term_names:
            xi[1, term_names.index("x")] = p["epsilon"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["epsilon"] * p["b"]
        return xi


class MorrisLecar(DynamicalSystem):
    """
    Simplified Morris-Lecar model for neural oscillations.

    A polynomial approximation of the Morris-Lecar equations.

    Equations (simplified polynomial form):
        dV/dt = -g_L*V - g_Ca*V + g_K*V*w + I  (terms: 1, x, xy)
        dw/dt = phi*(w_inf - w)/tau_w ≈ a + b*V - c*w  (terms: 1, x, y)

    Parameters
    ----------
    g_L : float
        Leak conductance (default: 0.5).
    g_Ca : float
        Calcium conductance approximation (default: 0.4).
    g_K : float
        Potassium conductance (default: 0.8).
    I : float
        External current (default: 0.4).
    a : float
        Recovery offset (default: 0.1).
    b : float
        Recovery V-dependence (default: 0.2).
    c : float
        Recovery decay (default: 0.3).
    """

    def __init__(
        self,
        g_L: float = 0.5,
        g_Ca: float = 0.4,
        g_K: float = 0.8,
        I: float = 0.4,
        a: float = 0.1,
        b: float = 0.2,
        c: float = 0.3,
    ):
        super().__init__(
            "Morris-Lecar",
            2,
            {"g_L": g_L, "g_Ca": g_Ca, "g_K": g_K, "I": I, "a": a, "b": b, "c": c},
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        V, w = state
        p = self.params
        dV = -(p["g_L"] + p["g_Ca"]) * V + p["g_K"] * V * w + p["I"]
        dw = p["a"] + p["b"] * V - p["c"] * w
        return np.array([dV, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dV/dt uses: 1, x, xy; dw/dt uses: 1, x, y"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["1", "x", "xy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["1", "x", "y"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "1" in term_names:
            xi[0, term_names.index("1")] = p["I"]
        if "x" in term_names:
            xi[0, term_names.index("x")] = -(p["g_L"] + p["g_Ca"])
        if "xy" in term_names:
            xi[0, term_names.index("xy")] = p["g_K"]
        if "1" in term_names:
            xi[1, term_names.index("1")] = p["a"]
        if "x" in term_names:
            xi[1, term_names.index("x")] = p["b"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["c"]
        return xi


class HindmarshRose2D(DynamicalSystem):
    """
    Simplified 2D Hindmarsh-Rose model for neural bursting.

    Equations:
        dx/dt = y - a*x^3 + b*x^2 - z + I  (simplified: terms: 1, y, xx, xxx)
        dy/dt = c - d*x^2 - y  (terms: 1, xx, y)

    For 2D version, z is treated as constant (slow variable frozen).

    Parameters
    ----------
    a : float
        Cubic coefficient (default: 1.0).
    b : float
        Quadratic coefficient (default: 3.0).
    c : float
        y-dynamics constant (default: 1.0).
    d : float
        y-dynamics x^2 coefficient (default: 5.0).
    I : float
        External current (default: 2.0).
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 3.0,
        c: float = 1.0,
        d: float = 5.0,
        I: float = 2.0,
    ):
        super().__init__(
            "Hindmarsh-Rose 2D",
            2,
            {"a": a, "b": b, "c": c, "d": d, "I": I},
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        dx = y - p["a"] * x**3 + p["b"] * x**2 + p["I"]
        dy = p["c"] - p["d"] * x**2 - y
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: 1, y, xx, xxx; dy/dt uses: 1, xx, y"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["1", "y", "xx", "xxx"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["1", "xx", "y"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "1" in term_names:
            xi[0, term_names.index("1")] = p["I"]
        if "y" in term_names:
            xi[0, term_names.index("y")] = 1.0
        if "xx" in term_names:
            xi[0, term_names.index("xx")] = p["b"]
        if "xxx" in term_names:
            xi[0, term_names.index("xxx")] = -p["a"]
        if "1" in term_names:
            xi[1, term_names.index("1")] = p["c"]
        if "xx" in term_names:
            xi[1, term_names.index("xx")] = -p["d"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -1.0
        return xi
