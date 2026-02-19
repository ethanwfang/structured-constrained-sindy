"""
4D coupled dynamical systems for factorized network training.

This module provides diverse 4D polynomial dynamical systems including
coupled oscillators and hyperchaotic systems.

Note: For 4D systems, the state variables are (x, y, z, w) and the
polynomial library includes terms like xy, xz, xw, yz, yw, zw, etc.
"""

from typing import Dict, List

import numpy as np

from .base import DynamicalSystem


class CoupledVanDerPol(DynamicalSystem):
    """
    Two coupled Van der Pol oscillators.

    State: [x1, x2, x3, x4] = [position1, velocity1, position2, velocity2]

    Equations:
        dx1/dt = x2
        dx2/dt = μ(1 - x1²)x2 - x1 + k(x3 - x1)
        dx3/dt = x4
        dx4/dt = μ(1 - x3²)x4 - x3 + k(x1 - x3)

    In (x, y, z, w) notation:
        dx/dt = y
        dy/dt = μy - μx²y - x + k(z - x) = μy - x - kx + kz - μxxy
        dz/dt = w
        dw/dt = μw - μz²w - z + k(x - z) = μw - z - kz + kx - μzzw

    Key terms: y, w (linear), xxy, zzw (cubic coupling)
    """

    def __init__(self, mu: float = 1.0, k: float = 0.5):
        params = {"mu": mu, "k": k}
        super().__init__(name="CoupledVanDerPol", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        mu = self.params["mu"]
        k = self.params["k"]

        dx = y
        dy = mu * (1 - x**2) * y - x + k * (z - x)
        dz = w
        dw = mu * (1 - z**2) * w - z + k * (x - z)

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = y: terms [y]
        for j, name in enumerate(term_names):
            if name == "y":
                structure[0, j] = True

        # dy/dt = μy - x - kx + kz - μx²y: terms [x, y, z, xxy]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "z", "xxy"]:
                structure[1, j] = True

        # dz/dt = w: terms [w]
        for j, name in enumerate(term_names):
            if name == "w":
                structure[2, j] = True

        # dw/dt = μw - z - kz + kx - μz²w: terms [x, z, w, zzw]
        for j, name in enumerate(term_names):
            if name in ["x", "z", "w", "zzw"]:
                structure[3, j] = True

        return structure


class CoupledDuffing(DynamicalSystem):
    """
    Two coupled Duffing oscillators.

    State: [x1, x2, x3, x4] = [position1, velocity1, position2, velocity2]

    Equations:
        dx1/dt = x2
        dx2/dt = -δx2 - αx1 - βx1³ + k(x3 - x1)
        dx3/dt = x4
        dx4/dt = -δx4 - αx3 - βx3³ + k(x1 - x3)

    In (x, y, z, w) notation:
        dx/dt = y
        dy/dt = -δy - αx - βx³ + kz - kx
        dz/dt = w
        dw/dt = -δw - αz - βz³ + kx - kz

    Key terms: y, w (linear), xxx, zzz (cubic)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, delta: float = 0.3, k: float = 0.2):
        params = {"alpha": alpha, "beta": beta, "delta": delta, "k": k}
        super().__init__(name="CoupledDuffing", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        delta = self.params["delta"]
        k = self.params["k"]

        dx = y
        dy = -delta * y - alpha * x - beta * x**3 + k * (z - x)
        dz = w
        dw = -delta * w - alpha * z - beta * z**3 + k * (x - z)

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = y: terms [y]
        for j, name in enumerate(term_names):
            if name == "y":
                structure[0, j] = True

        # dy/dt = -δy - αx - kx + kz - βx³: terms [x, y, z, xxx]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "z", "xxx"]:
                structure[1, j] = True

        # dz/dt = w: terms [w]
        for j, name in enumerate(term_names):
            if name == "w":
                structure[2, j] = True

        # dw/dt = -δw - αz - kz + kx - βz³: terms [x, z, w, zzz]
        for j, name in enumerate(term_names):
            if name in ["x", "z", "w", "zzz"]:
                structure[3, j] = True

        return structure


class HyperchaoticLorenz(DynamicalSystem):
    """
    4D Hyperchaotic Lorenz system.

    Extension of the classic Lorenz system to 4D with hyperchaotic behavior.

    Equations:
        dx/dt = σ(y - x) + w
        dy/dt = rx - y - xz
        dz/dt = xy - bz
        dw/dt = -yw - dw

    Key terms: x, y, z, w (linear), xz, xy, yw (bilinear)
    """

    def __init__(self, sigma: float = 10.0, r: float = 28.0, b: float = 8 / 3, d: float = 1.3):
        params = {"sigma": sigma, "r": r, "b": b, "d": d}
        super().__init__(name="HyperchaoticLorenz", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        sigma = self.params["sigma"]
        r = self.params["r"]
        b = self.params["b"]
        d = self.params["d"]

        dx = sigma * (y - x) + w
        dy = r * x - y - x * z
        dz = x * y - b * z
        dw = -y * w - d * w

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = σ(y - x) + w: terms [x, y, w]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "w"]:
                structure[0, j] = True

        # dy/dt = rx - y - xz: terms [x, y, xz]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "xz"]:
                structure[1, j] = True

        # dz/dt = xy - bz: terms [z, xy]
        for j, name in enumerate(term_names):
            if name in ["z", "xy"]:
                structure[2, j] = True

        # dw/dt = -yw - dw: terms [w, yw]
        for j, name in enumerate(term_names):
            if name in ["w", "yw"]:
                structure[3, j] = True

        return structure


class HyperchaoticRossler(DynamicalSystem):
    """
    4D Hyperchaotic Rossler system.

    Extension of the Rossler system to 4D.

    Equations:
        dx/dt = -y - z
        dy/dt = x + ay + w
        dz/dt = b + xz
        dw/dt = -cz + dw

    Key terms: x, y, z, w (linear), xz (bilinear), 1 (constant)
    """

    def __init__(self, a: float = 0.25, b: float = 3.0, c: float = 0.5, d: float = 0.05):
        params = {"a": a, "b": b, "c": c, "d": d}
        super().__init__(name="HyperchaoticRossler", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        d = self.params["d"]

        dx = -y - z
        dy = x + a * y + w
        dz = b + x * z
        dw = -c * z + d * w

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = -y - z: terms [y, z]
        for j, name in enumerate(term_names):
            if name in ["y", "z"]:
                structure[0, j] = True

        # dy/dt = x + ay + w: terms [x, y, w]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "w"]:
                structure[1, j] = True

        # dz/dt = b + xz: terms [1, xz]
        for j, name in enumerate(term_names):
            if name in ["1", "xz"]:
                structure[2, j] = True

        # dw/dt = -cz + dw: terms [z, w]
        for j, name in enumerate(term_names):
            if name in ["z", "w"]:
                structure[3, j] = True

        return structure


class LotkaVolterra4D(DynamicalSystem):
    """
    4-species competitive Lotka-Volterra system.

    Models competition between 4 species with bilinear interactions.

    Equations:
        dx/dt = x(r1 - a11*x - a12*y - a13*z - a14*w)
        dy/dt = y(r2 - a21*x - a22*y - a23*z - a24*w)
        dz/dt = z(r3 - a31*x - a32*y - a33*z - a34*w)
        dw/dt = w(r4 - a41*x - a42*y - a43*z - a44*w)

    Expanding:
        dx/dt = r1*x - a11*xx - a12*xy - a13*xz - a14*xw
        dy/dt = r2*y - a21*xy - a22*yy - a23*yz - a24*yw
        dz/dt = r3*z - a31*xz - a32*yz - a33*zz - a34*zw
        dw/dt = r4*w - a41*xw - a42*yw - a43*zw - a44*ww

    Key terms: Many bilinear interactions (xy, xz, xw, yz, yw, zw)
    """

    def __init__(
        self,
        r: tuple = (1.0, 0.72, 1.53, 1.27),
        a11: float = 1.0,
        a12: float = 1.09,
        a13: float = 1.52,
        a14: float = 0.0,
        a21: float = 0.0,
        a22: float = 1.0,
        a23: float = 0.44,
        a24: float = 1.36,
        a31: float = 2.33,
        a32: float = 0.0,
        a33: float = 1.0,
        a34: float = 0.47,
        a41: float = 1.21,
        a42: float = 0.51,
        a43: float = 0.35,
        a44: float = 1.0,
    ):
        params = {
            "r": r,
            "a11": a11,
            "a12": a12,
            "a13": a13,
            "a14": a14,
            "a21": a21,
            "a22": a22,
            "a23": a23,
            "a24": a24,
            "a31": a31,
            "a32": a32,
            "a33": a33,
            "a34": a34,
            "a41": a41,
            "a42": a42,
            "a43": a43,
            "a44": a44,
        }
        super().__init__(name="LotkaVolterra4D", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        r = self.params["r"]
        p = self.params

        dx = x * (r[0] - p["a11"] * x - p["a12"] * y - p["a13"] * z - p["a14"] * w)
        dy = y * (r[1] - p["a21"] * x - p["a22"] * y - p["a23"] * z - p["a24"] * w)
        dz = z * (r[2] - p["a31"] * x - p["a32"] * y - p["a33"] * z - p["a34"] * w)
        dw = w * (r[3] - p["a41"] * x - p["a42"] * y - p["a43"] * z - p["a44"] * w)

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt: terms [x, xx, xy, xz, xw]
        for j, name in enumerate(term_names):
            if name in ["x", "xx", "xy", "xz", "xw"]:
                structure[0, j] = True

        # dy/dt: terms [y, xy, yy, yz, yw]
        for j, name in enumerate(term_names):
            if name in ["y", "xy", "yy", "yz", "yw"]:
                structure[1, j] = True

        # dz/dt: terms [z, xz, yz, zz, zw]
        for j, name in enumerate(term_names):
            if name in ["z", "xz", "yz", "zz", "zw"]:
                structure[2, j] = True

        # dw/dt: terms [w, xw, yw, zw, ww]
        for j, name in enumerate(term_names):
            if name in ["w", "xw", "yw", "zw", "ww"]:
                structure[3, j] = True

        return structure


class CoupledFitzHughNagumo(DynamicalSystem):
    """
    Two coupled FitzHugh-Nagumo neurons.

    State: [v1, w1, v2, w2] = [voltage1, recovery1, voltage2, recovery2]

    Equations:
        dv1/dt = v1 - v1³/3 - w1 + k(v2 - v1)
        dw1/dt = ε(v1 + a - bw1)
        dv2/dt = v2 - v2³/3 - w2 + k(v1 - v2)
        dw2/dt = ε(v2 + a - bw2)

    In (x, y, z, w) notation:
        dx/dt = x - x³/3 - y + k(z - x)
        dy/dt = ε(x + a - by)
        dz/dt = z - z³/3 - w + k(x - z)
        dw/dt = ε(z + a - bw)

    Key terms: x, y, z, w (linear), xxx, zzz (cubic), 1 (constant)
    """

    def __init__(self, a: float = 0.7, b: float = 0.8, epsilon: float = 0.08, k: float = 0.3):
        params = {"a": a, "b": b, "epsilon": epsilon, "k": k}
        super().__init__(name="CoupledFitzHughNagumo", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        a = self.params["a"]
        b = self.params["b"]
        epsilon = self.params["epsilon"]
        k = self.params["k"]

        dx = x - (x**3) / 3 - y + k * (z - x)
        dy = epsilon * (x + a - b * y)
        dz = z - (z**3) / 3 - w + k * (x - z)
        dw = epsilon * (z + a - b * w)

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = x - x³/3 - y + kz - kx: terms [x, y, z, xxx]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "z", "xxx"]:
                structure[0, j] = True

        # dy/dt = ε(x + a - by): terms [1, x, y]
        for j, name in enumerate(term_names):
            if name in ["1", "x", "y"]:
                structure[1, j] = True

        # dz/dt = z - z³/3 - w + kx - kz: terms [x, z, w, zzz]
        for j, name in enumerate(term_names):
            if name in ["x", "z", "w", "zzz"]:
                structure[2, j] = True

        # dw/dt = ε(z + a - bw): terms [1, z, w]
        for j, name in enumerate(term_names):
            if name in ["1", "z", "w"]:
                structure[3, j] = True

        return structure


class MixedCoupledOscillator(DynamicalSystem):
    """
    Mixed coupled oscillator: Van der Pol + Duffing with bidirectional coupling.

    Combines a Van der Pol oscillator (first pair) with a Duffing oscillator
    (second pair) through asymmetric coupling terms.

    State: [x1, x2, x3, x4] = [VdP position, VdP velocity, Duffing pos, Duffing vel]

    Equations:
        dx/dt = y
        dy/dt = μ(1 - x²)y - x + k₁z
        dz/dt = w
        dw/dt = -δw - αz - βz³ + k₂x

    Key terms: y, w (linear), x, z (linear), xxy (Van der Pol), zzz (Duffing)
    """

    def __init__(
        self,
        mu: float = 1.0,
        alpha: float = 1.0,
        beta: float = 0.5,
        delta: float = 0.3,
        k1: float = 0.3,
        k2: float = 0.2,
    ):
        params = {
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "k1": k1,
            "k2": k2,
        }
        super().__init__(name="MixedCoupledOscillator", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        p = self.params

        dx = y
        dy = p["mu"] * (1 - x**2) * y - x + p["k1"] * z
        dz = w
        dw = -p["delta"] * w - p["alpha"] * z - p["beta"] * z**3 + p["k2"] * x

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = y
        for j, name in enumerate(term_names):
            if name == "y":
                structure[0, j] = True

        # dy/dt = μy - μx²y - x + k₁z: terms [x, y, z, xxy]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "z", "xxy"]:
                structure[1, j] = True

        # dz/dt = w
        for j, name in enumerate(term_names):
            if name == "w":
                structure[2, j] = True

        # dw/dt = -δw - αz + k₂x - βz³: terms [x, z, w, zzz]
        for j, name in enumerate(term_names):
            if name in ["x", "z", "w", "zzz"]:
                structure[3, j] = True

        return structure


class LorenzExtended4D(DynamicalSystem):
    """
    4D extended Lorenz system with additional variable.

    Extends the classic Lorenz attractor with a fourth variable
    that couples to the existing dynamics.

    Equations:
        dx/dt = σ(y - x)
        dy/dt = x(r - z) - y
        dz/dt = xy - bz
        dw/dt = -cw + xy  (additional equation coupling to xy)

    This adds structural diversity by having w coupled through xy.
    """

    def __init__(self, sigma: float = 10.0, r: float = 28.0, b: float = 8 / 3, c: float = 0.5):
        params = {"sigma": sigma, "r": r, "b": b, "c": c}
        super().__init__(name="LorenzExtended4D", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        sigma = self.params["sigma"]
        r = self.params["r"]
        b = self.params["b"]
        c = self.params["c"]

        dx = sigma * (y - x)
        dy = x * r - x * z - y
        dz = x * y - b * z
        dw = -c * w + x * y

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = σ(y - x): terms [x, y]
        for j, name in enumerate(term_names):
            if name in ["x", "y"]:
                structure[0, j] = True

        # dy/dt = rx - xz - y: terms [x, y, xz]
        for j, name in enumerate(term_names):
            if name in ["x", "y", "xz"]:
                structure[1, j] = True

        # dz/dt = xy - bz: terms [z, xy]
        for j, name in enumerate(term_names):
            if name in ["z", "xy"]:
                structure[2, j] = True

        # dw/dt = -cw + xy: terms [w, xy]
        for j, name in enumerate(term_names):
            if name in ["w", "xy"]:
                structure[3, j] = True

        return structure


class SimpleQuadratic4D(DynamicalSystem):
    """
    Simple 4D system with sparse quadratic structure.

    A minimal 4D system with clear sparse structure for training.
    Each equation has only 2-3 active terms.

    Equations:
        dx/dt = -x + yz
        dy/dt = -y + xz
        dz/dt = -z + xw
        dw/dt = -w + xy

    Key terms: All linear terms present, with unique bilinear terms per equation.
    """

    def __init__(self, a: float = 1.0):
        params = {"a": a}
        super().__init__(name="SimpleQuadratic4D", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        a = self.params["a"]

        dx = -a * x + y * z
        dy = -a * y + x * z
        dz = -a * z + x * w
        dw = -a * w + x * y

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = -x + yz: terms [x, yz]
        for j, name in enumerate(term_names):
            if name in ["x", "yz"]:
                structure[0, j] = True

        # dy/dt = -y + xz: terms [y, xz]
        for j, name in enumerate(term_names):
            if name in ["y", "xz"]:
                structure[1, j] = True

        # dz/dt = -z + xw: terms [z, xw]
        for j, name in enumerate(term_names):
            if name in ["z", "xw"]:
                structure[2, j] = True

        # dw/dt = -w + xy: terms [w, xy]
        for j, name in enumerate(term_names):
            if name in ["w", "xy"]:
                structure[3, j] = True

        return structure


class Cubic4DSystem(DynamicalSystem):
    """
    4D system with cubic nonlinearities.

    Tests the network's ability to identify cubic terms in 4D.

    Equations:
        dx/dt = -x³ + y
        dy/dt = x - y³
        dz/dt = -z³ + w
        dw/dt = z - w³

    Key terms: Linear (x, y, z, w) and cubic (xxx, yyy, zzz, www).
    """

    def __init__(self, alpha: float = 0.5):
        params = {"alpha": alpha}
        super().__init__(name="Cubic4DSystem", dim=4, params=params)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z, w = state
        a = self.params["alpha"]

        dx = -a * x**3 + y
        dy = x - a * y**3
        dz = -a * z**3 + w
        dw = z - a * w**3

        return np.array([dx, dy, dz, dw])

    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        n_terms = len(term_names)
        structure = np.zeros((4, n_terms), dtype=bool)

        # dx/dt = -x³ + y: terms [y, xxx]
        for j, name in enumerate(term_names):
            if name in ["y", "xxx"]:
                structure[0, j] = True

        # dy/dt = x - y³: terms [x, yyy]
        for j, name in enumerate(term_names):
            if name in ["x", "yyy"]:
                structure[1, j] = True

        # dz/dt = -z³ + w: terms [w, zzz]
        for j, name in enumerate(term_names):
            if name in ["w", "zzz"]:
                structure[2, j] = True

        # dw/dt = z - w³: terms [z, www]
        for j, name in enumerate(term_names):
            if name in ["z", "www"]:
                structure[3, j] = True

        return structure
