"""
Additional 3D chaotic systems for factorized network training.

This module provides diverse 3D polynomial dynamical systems with
different structural patterns (bilinear, quadratic, cubic terms).
"""

from typing import Dict, List

import numpy as np

from .base import DynamicalSystem


class ThomasAttractor(DynamicalSystem):
    """
    Thomas' cyclically symmetric attractor (polynomial approximation).

    The original equations use sin() functions, but we approximate with
    Taylor series: sin(x) ≈ x - x³/6 for small x.

    Equations:
        dx/dt = sin(y) - bx ≈ y - bx - y³/6
        dy/dt = sin(z) - bz ≈ z - by - z³/6
        dz/dt = sin(x) - bz ≈ x - bz - x³/6

    Key terms: x, y, z (linear), yyy, zzz, xxx (cubic)
    """

    def __init__(self, b: float = 0.208186):
        params = {"b": b}
        super().__init__(name="ThomasAttractor", dim=3, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state
        b = self.params["b"]

        # Taylor approximation of sin
        dx = y - b * x - (y**3) / 6
        dy = z - b * y - (z**3) / 6
        dz = x - b * z - (x**3) / 6

        return np.array([dx, dy, dz])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((3, n_terms), dtype=bool)

        # dx/dt = y - bx - y³/6: terms [x, y, yyy]
        for j, name in enumerate(term_names):
            if name == "x":
                structure[0, j] = True
            if name == "y":
                structure[0, j] = True
            if name == "yyy":
                structure[0, j] = True

        # dy/dt = z - by - z³/6: terms [y, z, zzz]
        for j, name in enumerate(term_names):
            if name == "y":
                structure[1, j] = True
            if name == "z":
                structure[1, j] = True
            if name == "zzz":
                structure[1, j] = True

        # dz/dt = x - bz - x³/6: terms [x, z, xxx]
        for j, name in enumerate(term_names):
            if name == "x":
                structure[2, j] = True
            if name == "z":
                structure[2, j] = True
            if name == "xxx":
                structure[2, j] = True

        return structure


class HalvorsenAttractor(DynamicalSystem):
    """
    Halvorsen's cyclically symmetric chaotic attractor.

    Equations:
        dx/dt = -ax - 4y - 4z - y²
        dy/dt = -ay - 4z - 4x - z²
        dz/dt = -az - 4x - 4y - x²

    Key terms: x, y, z (linear), xx, yy, zz (quadratic, self-interaction)
    """

    def __init__(self, a: float = 1.89):
        params = {"a": a}
        super().__init__(name="HalvorsenAttractor", dim=3, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state
        a = self.params["a"]

        dx = -a * x - 4 * y - 4 * z - y**2
        dy = -a * y - 4 * z - 4 * x - z**2
        dz = -a * z - 4 * x - 4 * y - x**2

        return np.array([dx, dy, dz])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((3, n_terms), dtype=bool)

        # dx/dt: terms [x, y, z, yy]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "z", "yy"]:
                structure[0, j] = True

        # dy/dt: terms [x, y, z, zz]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "z", "zz"]:
                structure[1, j] = True

        # dz/dt: terms [x, y, z, xx]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "z", "xx"]:
                structure[2, j] = True

        return structure


class SprottB(DynamicalSystem):
    """
    Sprott Case B - minimal chaotic system.

    One of the simplest chaotic systems with only 5 terms total.

    Equations:
        dx/dt = yz
        dy/dt = x - y
        dz/dt = 1 - xy

    Key terms: yz (bilinear), x, y (linear), 1 (constant), xy (bilinear)
    """

    def __init__(self):
        params = {}
        super().__init__(name="SprottB", dim=3, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state

        dx = y * z
        dy = x - y
        dz = 1 - x * y

        return np.array([dx, dy, dz])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((3, n_terms), dtype=bool)

        # dx/dt = yz: terms [yz]
        for j, name in enumerate(term_names):
            if name == "yz":
                structure[0, j] = True

        # dy/dt = x - y: terms [x, y]
        for j, name in enumerate(term_names):
            if name in ["x", "y"]:
                structure[1, j] = True

        # dz/dt = 1 - xy: terms [1, xy]
        for j, name in enumerate(term_names):
            if name in ["1", "xy"]:
                structure[2, j] = True

        return structure


class SprottD(DynamicalSystem):
    """
    Sprott Case D - minimal quadratic chaotic system.

    Equations:
        dx/dt = -y
        dy/dt = x + z
        dz/dt = xz + 3y²

    Key terms: y (linear), x, z (linear), xz (bilinear), yy (quadratic)
    """

    def __init__(self):
        params = {}
        super().__init__(name="SprottD", dim=3, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state

        dx = -y
        dy = x + z
        dz = x * z + 3 * y**2

        return np.array([dx, dy, dz])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((3, n_terms), dtype=bool)

        # dx/dt = -y: terms [y]
        for j, name in enumerate(term_names):
            if name == "y":
                structure[0, j] = True

        # dy/dt = x + z: terms [x, z]
        for j, name in enumerate(term_names):
            if name in ["x", "z"]:
                structure[1, j] = True

        # dz/dt = xz + 3y²: terms [xz, yy]
        for j, name in enumerate(term_names):
            if name in ["xz", "yy"]:
                structure[2, j] = True

        return structure


class RabinovichFabrikant(DynamicalSystem):
    """
    Rabinovich-Fabrikant equations.

    Rich bilinear structure with multiple cross-term interactions.

    Equations:
        dx/dt = y(z - 1 + x²) + γx = yz - y + x²y + γx
        dy/dt = x(3z + 1 - x²) + γy = 3xz + x - x³ + γy
        dz/dt = -2z(α + xy) = -2αz - 2xyz

    Key terms: Many bilinear (xy, xz, yz) and cubic (xxy, xyz, xxx)
    """

    def __init__(self, gamma: float = 0.87, alpha: float = 1.1):
        params = {"gamma": gamma, "alpha": alpha}
        super().__init__(name="RabinovichFabrikant", dim=3, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state
        gamma = self.params["gamma"]
        alpha = self.params["alpha"]

        dx = y * (z - 1 + x**2) + gamma * x
        dy = x * (3 * z + 1 - x**2) + gamma * y
        dz = -2 * z * (alpha + x * y)

        return np.array([dx, dy, dz])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((3, n_terms), dtype=bool)

        # dx/dt = yz - y + x²y + γx: terms [x, y, yz, xxy]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "yz", "xxy"]:
                structure[0, j] = True

        # dy/dt = 3xz + x - x³ + γy: terms [x, y, xz, xxx]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "xz", "xxx"]:
                structure[1, j] = True

        # dz/dt = -2αz - 2xyz: terms [z, xyz]
        for j, name in enumerate(term_names):
            if name in ["z", "xyz"]:
                structure[2, j] = True

        return structure


class AizawaAttractor(DynamicalSystem):
    """
    Aizawa attractor (Langford attractor variant).

    Equations:
        dx/dt = (z - b)x - dy
        dy/dt = dx + (z - b)y
        dz/dt = c + az - z³/3 - (x² + y²)(1 + ez) + fzx³

    Simplified polynomial form (keeping main structure):
        dx/dt = zx - bx - dy
        dy/dt = dx + zy - by
        dz/dt = c + az - z³/3 - x² - y² - ezx² - ezy² + fzx³

    Key terms: xz, yz (bilinear), x, y, z (linear), xx, yy, zzz (higher order)
    """

    def __init__(
        self, a: float = 0.95, b: float = 0.7, c: float = 0.6, d: float = 3.5, e: float = 0.25, f: float = 0.1
    ):
        params = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}
        super().__init__(name="AizawaAttractor", dim=3, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        d = self.params["d"]
        e = self.params["e"]
        f = self.params["f"]

        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a * z - (z**3) / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3

        return np.array([dx, dy, dz])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((3, n_terms), dtype=bool)

        # dx/dt = xz - bx - dy: terms [x, y, xz]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "xz"]:
                structure[0, j] = True

        # dy/dt = dx + yz - by: terms [x, y, yz]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "yz"]:
                structure[1, j] = True

        # dz/dt = c + az - z³/3 - x² - y² - ezx² - ezy²: terms [1, z, zzz, xx, yy, xxz, yyz]
        # Note: xxz and yyz may not be in poly_order=3 library, so we mark what's available
        for j, name in enumerate(term_names):
            if name in ["1", "z", "zzz", "xx", "yy"]:
                structure[2, j] = True
            # Higher order terms if they exist
            if name in ["xxz", "yyz", "xxxz"]:
                structure[2, j] = True

        return structure
