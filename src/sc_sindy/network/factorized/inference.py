"""
Inference utilities for factorized structure networks.

This module provides a high-level predictor class for using trained
factorized networks to predict equation structure.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .factorized_network import FactorizedStructureNetwork, FactorizedStructureNetworkV2
from .term_representation import get_library_terms


class FactorizedPredictor:
    """
    High-level predictor for factorized structure networks.

    Provides a simple interface for predicting equation structure from
    trajectories of any dimension.

    Parameters
    ----------
    model : nn.Module
        Trained FactorizedStructureNetwork or FactorizedStructureNetworkV2.
    default_poly_order : int, optional
        Default polynomial order to use (default: 3).

    Examples
    --------
    >>> # Load trained model
    >>> predictor = FactorizedPredictor.load("model.pt")
    >>>
    >>> # Predict on 2D trajectory
    >>> x_2d = np.random.randn(1000, 2)
    >>> probs = predictor.predict(x_2d, poly_order=3)
    >>> print(probs.shape)  # [2, 10]
    >>>
    >>> # Predict on 3D trajectory (same model!)
    >>> x_3d = np.random.randn(1000, 3)
    >>> probs = predictor.predict(x_3d, poly_order=2)
    >>> print(probs.shape)  # [3, 10]
    """

    def __init__(
        self,
        model,
        default_poly_order: int = 3,
    ):
        self.model = model
        self.default_poly_order = default_poly_order

        # Put model in eval mode
        if TORCH_AVAILABLE:
            self.model.eval()

    def predict(
        self,
        x: np.ndarray,
        poly_order: Optional[int] = None,
        return_term_names: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Predict structure probabilities from trajectory.

        Parameters
        ----------
        x : np.ndarray
            Trajectory with shape [T, n_vars].
        poly_order : int, optional
            Maximum polynomial order. If None, uses default.
        return_term_names : bool, optional
            If True, also return term names (default: False).

        Returns
        -------
        probs : np.ndarray
            Structure probabilities with shape [n_vars, n_terms].
        term_names : List[str], optional
            Term names if return_term_names=True.
        """
        if poly_order is None:
            poly_order = self.default_poly_order

        n_vars = x.shape[1]

        # Get predictions
        probs = self.model.predict_structure(x, n_vars=n_vars, poly_order=poly_order)

        if return_term_names:
            term_names = get_library_terms(n_vars, poly_order)
            return probs, term_names
        return probs

    def predict_with_threshold(
        self,
        x: np.ndarray,
        threshold: float = 0.3,
        poly_order: Optional[int] = None,
    ) -> Dict:
        """
        Predict structure and apply threshold.

        Parameters
        ----------
        x : np.ndarray
            Trajectory with shape [T, n_vars].
        threshold : float, optional
            Probability threshold for active terms (default: 0.3).
        poly_order : int, optional
            Maximum polynomial order.

        Returns
        -------
        result : Dict
            Dictionary with:
            - 'probs': Raw probabilities [n_vars, n_terms]
            - 'active_mask': Boolean mask [n_vars, n_terms]
            - 'active_terms': List of active term names per equation
            - 'term_names': All term names
        """
        probs, term_names = self.predict(
            x, poly_order=poly_order, return_term_names=True
        )

        active_mask = probs > threshold

        # Get active terms per equation
        n_vars = x.shape[1]
        active_terms = []
        for eq_idx in range(n_vars):
            terms = [term_names[j] for j in range(len(term_names)) if active_mask[eq_idx, j]]
            active_terms.append(terms)

        return {
            "probs": probs,
            "active_mask": active_mask,
            "active_terms": active_terms,
            "term_names": term_names,
        }

    def predict_batch(
        self,
        trajectories: list,
        poly_order: Optional[int] = None,
    ) -> list:
        """
        Predict structure for multiple trajectories.

        Note: Trajectories must have the same dimension.

        Parameters
        ----------
        trajectories : list
            List of trajectory arrays, each with shape [T, n_vars].
        poly_order : int, optional
            Maximum polynomial order.

        Returns
        -------
        probs_list : list
            List of probability arrays [n_vars, n_terms].
        """
        return [self.predict(x, poly_order=poly_order) for x in trajectories]

    def save(self, model_path: str, config_path: Optional[str] = None):
        """
        Save predictor to files.

        Parameters
        ----------
        model_path : str
            Path for model checkpoint.
        config_path : str, optional
            Path for config JSON. If None, uses model_path with .json extension.
        """
        if config_path is None:
            config_path = str(Path(model_path).with_suffix(".json"))

        # Save model
        self.model.save(model_path)

        # Save config
        config = {
            "default_poly_order": self.default_poly_order,
            "model_type": type(self.model).__name__,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
    ) -> "FactorizedPredictor":
        """
        Load predictor from files.

        Parameters
        ----------
        model_path : str
            Path to model checkpoint.
        config_path : str, optional
            Path to config JSON. If None, uses model_path with .json extension.

        Returns
        -------
        predictor : FactorizedPredictor
            Loaded predictor.
        """
        if config_path is None:
            config_path = str(Path(model_path).with_suffix(".json"))

        # Load config
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            default_poly_order = config.get("default_poly_order", 3)
            model_type = config.get("model_type", "FactorizedStructureNetworkV2")
        else:
            default_poly_order = 3
            model_type = "FactorizedStructureNetworkV2"

        # Load model
        if model_type == "FactorizedStructureNetwork":
            model = FactorizedStructureNetwork.load(model_path)
        else:
            model = FactorizedStructureNetworkV2.load(model_path)

        return cls(model=model, default_poly_order=default_poly_order)

    @classmethod
    def from_training_results(
        cls,
        model,
        config: Dict,
    ) -> "FactorizedPredictor":
        """
        Create predictor from training results.

        Parameters
        ----------
        model : nn.Module
            Trained model.
        config : Dict
            Training configuration dict.

        Returns
        -------
        predictor : FactorizedPredictor
            Configured predictor.
        """
        return cls(
            model=model,
            default_poly_order=config.get("poly_order", 3),
        )


def predict_structure_for_sindy(
    predictor: FactorizedPredictor,
    x: np.ndarray,
    dt: float,
    poly_order: int = 3,
) -> np.ndarray:
    """
    Predict structure probabilities in format compatible with SC-SINDy.

    This is a convenience function that matches the interface expected
    by sindy_structure_constrained().

    Parameters
    ----------
    predictor : FactorizedPredictor
        Trained predictor.
    x : np.ndarray
        Trajectory with shape [T, n_vars].
    dt : float
        Time step (not used, for API compatibility).
    poly_order : int, optional
        Maximum polynomial order.

    Returns
    -------
    probs : np.ndarray
        Structure probabilities with shape [n_vars, n_terms].
    """
    return predictor.predict(x, poly_order=poly_order)
