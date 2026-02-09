"""
Ecological and epidemiological dynamical systems.

This module provides dynamical systems with bilinear interaction terms (xy),
which are common in ecology (competition, mutualism) and epidemiology (disease spread).

These systems are critical for training networks to recognize interaction dynamics.
"""

from typing import List

import numpy as np

from .base import DynamicalSystem


class CompetitiveExclusion(DynamicalSystem):
    """
    Competitive exclusion model (two-species competition).

    Equations:
        dx/dt = r1*x*(1 - x/K1 - a12*y/K1)
              = r1*x - (r1/K1)*x^2 - (r1*a12/K1)*x*y
        dy/dt = r2*y*(1 - y/K2 - a21*x/K2)
              = r2*y - (r2/K2)*y^2 - (r2*a21/K2)*x*y

    Simplified (normalized K1=K2=1):
        dx/dt = r1*x - r1*x^2 - r1*a12*x*y  (terms: x, xx, xy)
        dy/dt = r2*y - r2*y^2 - r2*a21*x*y  (terms: y, yy, xy)

    Parameters
    ----------
    r1 : float
        Growth rate of species 1 (default: 1.0).
    r2 : float
        Growth rate of species 2 (default: 0.8).
    a12 : float
        Competition coefficient (effect of y on x) (default: 0.5).
    a21 : float
        Competition coefficient (effect of x on y) (default: 0.6).
    """

    def __init__(
        self, r1: float = 1.0, r2: float = 0.8, a12: float = 0.5, a21: float = 0.6
    ):
        super().__init__(
            "Competitive Exclusion",
            2,
            {"r1": r1, "r2": r2, "a12": a12, "a21": a21},
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        dx = p["r1"] * x * (1 - x - p["a12"] * y)
        dy = p["r2"] * y * (1 - y - p["a21"] * x)
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: x, xx, xy; dy/dt uses: y, yy, xy"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["x", "xx", "xy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["y", "yy", "xy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "x" in term_names:
            xi[0, term_names.index("x")] = p["r1"]
        if "xx" in term_names:
            xi[0, term_names.index("xx")] = -p["r1"]
        if "xy" in term_names:
            xi[0, term_names.index("xy")] = -p["r1"] * p["a12"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = p["r2"]
        if "yy" in term_names:
            xi[1, term_names.index("yy")] = -p["r2"]
        if "xy" in term_names:
            xi[1, term_names.index("xy")] = -p["r2"] * p["a21"]
        return xi


class MutualismModel(DynamicalSystem):
    """
    Mutualism model (two-species facilitation).

    Equations:
        dx/dt = r1*x*(1 - x/K1) + b12*x*y
              = r1*x - (r1/K1)*x^2 + b12*x*y  (terms: x, xx, xy)
        dy/dt = r2*y*(1 - y/K2) + b21*x*y
              = r2*y - (r2/K2)*y^2 + b21*x*y  (terms: y, yy, xy)

    Parameters
    ----------
    r1 : float
        Growth rate of species 1 (default: 0.5).
    r2 : float
        Growth rate of species 2 (default: 0.4).
    b12 : float
        Mutualism benefit (effect of y on x) (default: 0.3).
    b21 : float
        Mutualism benefit (effect of x on y) (default: 0.25).
    K1 : float
        Carrying capacity of species 1 (default: 1.0).
    K2 : float
        Carrying capacity of species 2 (default: 1.0).
    """

    def __init__(
        self,
        r1: float = 0.5,
        r2: float = 0.4,
        b12: float = 0.3,
        b21: float = 0.25,
        K1: float = 1.0,
        K2: float = 1.0,
    ):
        super().__init__(
            "Mutualism Model",
            2,
            {"r1": r1, "r2": r2, "b12": b12, "b21": b21, "K1": K1, "K2": K2},
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        dx = p["r1"] * x * (1 - x / p["K1"]) + p["b12"] * x * y
        dy = p["r2"] * y * (1 - y / p["K2"]) + p["b21"] * x * y
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: x, xx, xy; dy/dt uses: y, yy, xy"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["x", "xx", "xy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["y", "yy", "xy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "x" in term_names:
            xi[0, term_names.index("x")] = p["r1"]
        if "xx" in term_names:
            xi[0, term_names.index("xx")] = -p["r1"] / p["K1"]
        if "xy" in term_names:
            xi[0, term_names.index("xy")] = p["b12"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = p["r2"]
        if "yy" in term_names:
            xi[1, term_names.index("yy")] = -p["r2"] / p["K2"]
        if "xy" in term_names:
            xi[1, term_names.index("xy")] = p["b21"]
        return xi


class SISEpidemic(DynamicalSystem):
    """
    SIS epidemic model (Susceptible-Infected-Susceptible).

    Unlike SIR, individuals return to susceptible after recovery.
    With S + I = N (constant), we can write as 2D system.

    Equations (normalized N=1):
        dS/dt = -beta*S*I + gamma*I = -beta*S*I + gamma*(1-S)
              = gamma - gamma*S - beta*S*I  (terms: 1, x, xy where x=S, y=I)
        dI/dt = beta*S*I - gamma*I
              = beta*S*I - gamma*I  (terms: y, xy)

    But since S + I = 1, we use I as independent variable:
        dI/dt = beta*(1-I)*I - gamma*I
              = beta*I - beta*I^2 - gamma*I
              = (beta-gamma)*I - beta*I^2  (terms: y, yy)

    For 2D with interaction, we track both but recognize the constraint:
        dx/dt = -beta*x*y + gamma*y  (terms: y, xy)
        dy/dt = beta*x*y - gamma*y   (terms: y, xy)

    Parameters
    ----------
    beta : float
        Infection rate (default: 0.5).
    gamma : float
        Recovery rate (default: 0.2).
    """

    def __init__(self, beta: float = 0.5, gamma: float = 0.2):
        super().__init__("SIS Epidemic", 2, {"beta": beta, "gamma": gamma})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        S, I = state
        p = self.params
        dS = -p["beta"] * S * I + p["gamma"] * I
        dI = p["beta"] * S * I - p["gamma"] * I
        return np.array([dS, dI])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dS/dt uses: y, xy; dI/dt uses: y, xy"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["y", "xy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["y", "xy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "y" in term_names:
            xi[0, term_names.index("y")] = p["gamma"]
        if "xy" in term_names:
            xi[0, term_names.index("xy")] = -p["beta"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["gamma"]
        if "xy" in term_names:
            xi[1, term_names.index("xy")] = p["beta"]
        return xi


class PredatorPreyTypeII(DynamicalSystem):
    """
    Predator-prey model with Holling Type II functional response.

    The predation rate saturates at high prey density.

    Equations:
        dx/dt = r*x*(1 - x/K) - a*x*y/(1 + a*h*x)
        dy/dt = e*a*x*y/(1 + a*h*x) - d*y

    For small x (linear approximation of functional response):
        dx/dt ≈ r*x - (r/K)*x^2 - a*x*y  (terms: x, xx, xy)
        dy/dt ≈ e*a*x*y - d*y            (terms: y, xy)

    Parameters
    ----------
    r : float
        Prey growth rate (default: 1.0).
    K : float
        Prey carrying capacity (default: 1.0).
    a : float
        Attack rate (default: 0.5).
    e : float
        Conversion efficiency (default: 0.6).
    d : float
        Predator death rate (default: 0.3).
    h : float
        Handling time (default: 0.1).
    """

    def __init__(
        self,
        r: float = 1.0,
        K: float = 1.0,
        a: float = 0.5,
        e: float = 0.6,
        d: float = 0.3,
        h: float = 0.1,
    ):
        super().__init__(
            "Predator-Prey Type II",
            2,
            {"r": r, "K": K, "a": a, "e": e, "d": d, "h": h},
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        functional_response = p["a"] * x / (1 + p["a"] * p["h"] * x)
        dx = p["r"] * x * (1 - x / p["K"]) - functional_response * y
        dy = p["e"] * functional_response * y - p["d"] * y
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        Approximate structure for small x (linearized functional response).
        dx/dt uses: x, xx, xy; dy/dt uses: y, xy
        """
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["x", "xx", "xy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["y", "xy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Coefficients for linearized (small x) approximation."""
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "x" in term_names:
            xi[0, term_names.index("x")] = p["r"]
        if "xx" in term_names:
            xi[0, term_names.index("xx")] = -p["r"] / p["K"]
        if "xy" in term_names:
            xi[0, term_names.index("xy")] = -p["a"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["d"]
        if "xy" in term_names:
            xi[1, term_names.index("xy")] = p["e"] * p["a"]
        return xi


class SimplePredatorPrey(DynamicalSystem):
    """
    Simple predator-prey with logistic prey growth.

    Simpler than Lotka-Volterra Type II, uses basic mass-action kinetics
    but with logistic prey growth for stability.

    Equations:
        dx/dt = r*x - (r/K)*x^2 - a*x*y  (terms: x, xx, xy)
        dy/dt = b*x*y - d*y              (terms: y, xy)

    Parameters
    ----------
    r : float
        Prey growth rate (default: 1.0).
    K : float
        Prey carrying capacity (default: 2.0).
    a : float
        Predation rate (default: 0.4).
    b : float
        Predator growth from consumption (default: 0.3).
    d : float
        Predator death rate (default: 0.2).
    """

    def __init__(
        self,
        r: float = 1.0,
        K: float = 2.0,
        a: float = 0.4,
        b: float = 0.3,
        d: float = 0.2,
    ):
        super().__init__(
            "Simple Predator-Prey",
            2,
            {"r": r, "K": K, "a": a, "b": b, "d": d},
        )

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        p = self.params
        dx = p["r"] * x - (p["r"] / p["K"]) * x**2 - p["a"] * x * y
        dy = p["b"] * x * y - p["d"] * y
        return np.array([dx, dy])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """dx/dt uses: x, xx, xy; dy/dt uses: y, xy"""
        mask = np.zeros((2, len(term_names)), dtype=bool)
        for term in ["x", "xx", "xy"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True
        for term in ["y", "xy"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True
        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        p = self.params
        xi = np.zeros((2, len(term_names)))
        if "x" in term_names:
            xi[0, term_names.index("x")] = p["r"]
        if "xx" in term_names:
            xi[0, term_names.index("xx")] = -p["r"] / p["K"]
        if "xy" in term_names:
            xi[0, term_names.index("xy")] = -p["a"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -p["d"]
        if "xy" in term_names:
            xi[1, term_names.index("xy")] = p["b"]
        return xi
