"""
Chaotic dynamical systems.

This module provides famous chaotic systems including the Lorenz attractor,
Rossler system, and Chen system.
"""

from typing import List

import numpy as np

from .base import DynamicalSystem


class Lorenz(DynamicalSystem):
    """
    Lorenz system (Lorenz attractor).

    Equations:
        dx/dt = sigma*(y - x)
        dy/dt = x*(rho - z) - y
        dz/dt = x*y - beta*z

    Parameters
    ----------
    sigma : float, optional
        Prandtl number (default: 10.0).
    rho : float, optional
        Rayleigh number (default: 28.0).
    beta : float, optional
        Geometric factor (default: 8/3).

    Notes
    -----
    Classic chaotic system discovered by Edward Lorenz in 1963. Shows
    sensitive dependence on initial conditions (butterfly effect).
    Standard parameters (10, 28, 8/3) produce chaotic behavior.
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        super().__init__("Lorenz", 3, {"sigma": sigma, "rho": rho, "beta": beta})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state
        p = self.params
        return np.array([p["sigma"] * (y - x), x * (p["rho"] - z) - y, x * y - p["beta"] * z])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: x, y
        dy/dt uses: x, y, xz
        dz/dt uses: z, xy
        """
        mask = np.zeros((3, len(term_names)), dtype=bool)

        # dx/dt = sigma*(y - x)
        for term in ["x", "y"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True

        # dy/dt = x*(rho - z) - y = rho*x - xz - y
        for term in ["x", "y", "xz"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        # dz/dt = xy - beta*z
        for term in ["z", "xy"]:
            if term in term_names:
                mask[2, term_names.index(term)] = True

        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Return actual coefficient values."""
        p = self.params
        xi = np.zeros((3, len(term_names)))

        # dx/dt
        if "x" in term_names:
            xi[0, term_names.index("x")] = -p["sigma"]
        if "y" in term_names:
            xi[0, term_names.index("y")] = p["sigma"]

        # dy/dt
        if "x" in term_names:
            xi[1, term_names.index("x")] = p["rho"]
        if "y" in term_names:
            xi[1, term_names.index("y")] = -1.0
        if "xz" in term_names:
            xi[1, term_names.index("xz")] = -1.0

        # dz/dt
        if "z" in term_names:
            xi[2, term_names.index("z")] = -p["beta"]
        if "xy" in term_names:
            xi[2, term_names.index("xy")] = 1.0

        return xi


class Rossler(DynamicalSystem):
    """
    Rossler system.

    Equations:
        dx/dt = -y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x - c)

    Parameters
    ----------
    a : float, optional
        Parameter a (default: 0.2).
    b : float, optional
        Parameter b (default: 0.2).
    c : float, optional
        Parameter c (default: 5.7).

    Notes
    -----
    Proposed by Otto Rossler in 1976. Simpler than Lorenz but still
    exhibits chaotic behavior. Has a distinctive single-band structure.
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        super().__init__("Rossler", 3, {"a": a, "b": b, "c": c})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state
        p = self.params
        return np.array([-y - z, x + p["a"] * y, p["b"] + z * (x - p["c"])])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: y, z
        dy/dt uses: x, y
        dz/dt uses: 1 (constant), z, xz
        """
        mask = np.zeros((3, len(term_names)), dtype=bool)

        # dx/dt = -y - z
        for term in ["y", "z"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True

        # dy/dt = x + a*y
        for term in ["x", "y"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        # dz/dt = b + z*(x - c) = b + xz - c*z
        for term in ["1", "z", "xz"]:
            if term in term_names:
                mask[2, term_names.index(term)] = True

        return mask

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """Return actual coefficient values."""
        p = self.params
        xi = np.zeros((3, len(term_names)))

        # dx/dt
        if "y" in term_names:
            xi[0, term_names.index("y")] = -1.0
        if "z" in term_names:
            xi[0, term_names.index("z")] = -1.0

        # dy/dt
        if "x" in term_names:
            xi[1, term_names.index("x")] = 1.0
        if "y" in term_names:
            xi[1, term_names.index("y")] = p["a"]

        # dz/dt
        if "1" in term_names:
            xi[2, term_names.index("1")] = p["b"]
        if "z" in term_names:
            xi[2, term_names.index("z")] = -p["c"]
        if "xz" in term_names:
            xi[2, term_names.index("xz")] = 1.0

        return xi


class ChenSystem(DynamicalSystem):
    """
    Chen system (Chen attractor).

    Equations:
        dx/dt = a*(y - x)
        dy/dt = (c - a)*x - x*z + c*y
        dz/dt = x*y - b*z

    Parameters
    ----------
    a : float, optional
        Parameter a (default: 35.0).
    b : float, optional
        Parameter b (default: 3.0).
    c : float, optional
        Parameter c (default: 28.0).

    Notes
    -----
    Discovered by Chen Guanrong in 1999. Related to Lorenz system but
    has different dynamical properties.
    """

    def __init__(self, a: float = 35.0, b: float = 3.0, c: float = 28.0):
        super().__init__("Chen System", 3, {"a": a, "b": b, "c": c})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state
        p = self.params
        return np.array(
            [p["a"] * (y - x), (p["c"] - p["a"]) * x - x * z + p["c"] * y, x * y - p["b"] * z]
        )

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        dx/dt uses: x, y
        dy/dt uses: x, y, xz
        dz/dt uses: z, xy
        """
        mask = np.zeros((3, len(term_names)), dtype=bool)

        # dx/dt
        for term in ["x", "y"]:
            if term in term_names:
                mask[0, term_names.index(term)] = True

        # dy/dt
        for term in ["x", "y", "xz"]:
            if term in term_names:
                mask[1, term_names.index(term)] = True

        # dz/dt
        for term in ["z", "xy"]:
            if term in term_names:
                mask[2, term_names.index(term)] = True

        return mask


class DoublePendulum(DynamicalSystem):
    """
    Double pendulum system (simplified, small angle approximation).

    For the full nonlinear double pendulum, the equations are complex.
    This is a linearized version for demonstration.

    Equations (simplified):
        dtheta1/dt = omega1
        domega1/dt = -g/L1 * theta1 - coupling terms
        dtheta2/dt = omega2
        domega2/dt = -g/L2 * theta2 - coupling terms

    Parameters
    ----------
    L1 : float, optional
        Length of first pendulum (default: 1.0).
    L2 : float, optional
        Length of second pendulum (default: 1.0).
    m1 : float, optional
        Mass of first pendulum (default: 1.0).
    m2 : float, optional
        Mass of second pendulum (default: 1.0).
    g : float, optional
        Gravitational acceleration (default: 9.81).
    """

    def __init__(
        self, L1: float = 1.0, L2: float = 1.0, m1: float = 1.0, m2: float = 1.0, g: float = 9.81
    ):
        super().__init__("Double Pendulum", 4, {"L1": L1, "L2": L2, "m1": m1, "m2": m2, "g": g})

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        theta1, omega1, theta2, omega2 = state
        p = self.params
        L1, L2, m1, m2, g = p["L1"], p["L2"], p["m1"], p["m2"], p["g"]

        # Simplified linearized equations
        # Full nonlinear equations would require sin/cos terms
        delta = theta2 - theta1
        den = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2

        # Approximate (small angle)
        domega1 = (
            m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(theta2) * np.cos(delta)
            + m2 * L2 * omega2**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta1)
        ) / den

        domega2 = (
            -m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
            - (m1 + m2) * L1 * omega1**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta2)
        ) / (L2 * den / L1)

        return np.array([omega1, domega1, omega2, domega2])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """Approximate structure for small angle case."""
        # This is a 4D system - would need custom library
        mask = np.zeros((4, len(term_names)), dtype=bool)
        return mask
