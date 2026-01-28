"""
Base class for dynamical systems.

This module provides the abstract base class that all dynamical systems
should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from scipy.integrate import odeint


class DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems.

    All dynamical systems should inherit from this class and implement
    the required methods.

    Parameters
    ----------
    name : str
        Human-readable name of the system.
    dim : int
        Dimension of the state space.
    params : dict
        Dictionary of system parameters.

    Attributes
    ----------
    name : str
        System name.
    dim : int
        State space dimension.
    params : dict
        System parameters.
    """

    def __init__(self, name: str, dim: int, params: Dict):
        self.name = name
        self.dim = dim
        self.params = params

    @abstractmethod
    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute time derivatives of the state.

        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        t : float
            Current time (may be unused for autonomous systems).

        Returns
        -------
        derivatives : np.ndarray
            Time derivatives of each state variable.
        """
        raise NotImplementedError

    @abstractmethod
    def get_true_structure(self, term_names: List[str]) -> np.ndarray:
        """
        Return binary mask of true active terms.

        Parameters
        ----------
        term_names : List[str]
            List of library term names.

        Returns
        -------
        mask : np.ndarray
            Boolean mask with shape [n_vars, n_terms] indicating active terms.
        """
        raise NotImplementedError

    def generate_trajectory(
        self, x0: np.ndarray, t: np.ndarray, noise_level: float = 0.0
    ) -> np.ndarray:
        """
        Generate trajectory with optional noise.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state vector.
        t : np.ndarray
            Time points for integration.
        noise_level : float, optional
            Noise level as fraction of signal standard deviation (default: 0.0).

        Returns
        -------
        x : np.ndarray
            State trajectory with shape [n_times, n_vars].

        Examples
        --------
        >>> system = VanDerPol(mu=1.5)
        >>> t = np.linspace(0, 50, 5000)
        >>> x = system.generate_trajectory(np.array([2.0, 0.0]), t, noise_level=0.05)
        """
        x = odeint(self.derivatives, x0, t)

        if noise_level > 0:
            noise = noise_level * np.std(x, axis=0) * np.random.randn(*x.shape)
            x = x + noise

        return x

    def get_true_coefficients(self, term_names: List[str]) -> np.ndarray:
        """
        Get true coefficient values for library terms.

        Default implementation returns structure mask as float. Override in
        subclasses to return actual coefficient values.

        Parameters
        ----------
        term_names : List[str]
            List of library term names.

        Returns
        -------
        coefficients : np.ndarray
            Coefficient matrix with shape [n_vars, n_terms].
        """
        return self.get_true_structure(term_names).astype(float)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
