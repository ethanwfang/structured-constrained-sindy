"""
Inference utilities for trained structure network.

This module provides the StructurePredictor class that wraps a trained
model for easy prediction from trajectories, including feature normalization.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .feature_extraction import extract_trajectory_features
from .structure_network import StructureNetwork


class StructurePredictor:
    """
    Wrapper for structure network inference.

    Handles feature extraction, normalization, and prediction reshaping
    for easy use in the evaluation pipeline.

    Parameters
    ----------
    model : StructureNetwork
        Trained structure network model.
    n_vars : int
        Number of state variables (dimension of the system).
    n_terms : int
        Number of library terms.
    feature_mean : np.ndarray, optional
        Mean values for feature normalization.
    feature_std : np.ndarray, optional
        Standard deviation for feature normalization.

    Examples
    --------
    >>> predictor = StructurePredictor.load("model.pt", "config.json")
    >>> probs = predictor.predict_from_trajectory(x, dt)
    >>> print(probs.shape)  # (n_vars, n_terms)
    """

    def __init__(
        self,
        model: StructureNetwork,
        n_vars: int,
        n_terms: int,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.n_vars = n_vars
        self.n_terms = n_terms
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # Ensure std doesn't have zeros
        if self.feature_std is not None:
            self.feature_std = np.where(self.feature_std < 1e-10, 1.0, self.feature_std)

    def predict_from_trajectory(
        self,
        x: np.ndarray,
        dt: float,
        return_features: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict structure probabilities from trajectory.

        Parameters
        ----------
        x : np.ndarray
            Trajectory with shape [n_samples, n_vars].
        dt : float
            Time step between samples.
        return_features : bool
            If True, also return extracted features.

        Returns
        -------
        probs : np.ndarray
            Structure probabilities with shape [n_vars, n_terms].
        features : np.ndarray, optional
            Extracted and normalized features (if return_features=True).
        """
        # Extract features
        features = extract_trajectory_features(x, dt)

        # Normalize if stats available
        if self.feature_mean is not None and self.feature_std is not None:
            features_normalized = (features - self.feature_mean) / self.feature_std
        else:
            features_normalized = features

        # Handle NaN/Inf in features
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions
        probs_flat = self.model.predict(features_normalized.reshape(1, -1))

        # Reshape to [n_vars, n_terms]
        probs = probs_flat.reshape(self.n_vars, self.n_terms)

        if return_features:
            return probs, features_normalized
        return probs

    def predict_batch(
        self,
        trajectories: list,
        dt: float,
    ) -> np.ndarray:
        """
        Predict structure for multiple trajectories.

        Parameters
        ----------
        trajectories : list
            List of trajectories, each with shape [n_samples, n_vars].
        dt : float
            Time step.

        Returns
        -------
        probs : np.ndarray
            Predictions with shape [n_trajectories, n_vars, n_terms].
        """
        all_features = []

        for x in trajectories:
            features = extract_trajectory_features(x, dt)
            if self.feature_mean is not None:
                features = (features - self.feature_mean) / self.feature_std
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            all_features.append(features)

        features_batch = np.array(all_features)
        probs_flat = self.model.predict(features_batch)

        return probs_flat.reshape(-1, self.n_vars, self.n_terms)

    def save(self, model_path: Union[str, Path], config_path: Union[str, Path]):
        """
        Save predictor model and configuration.

        Parameters
        ----------
        model_path : str or Path
            Path to save the model weights.
        config_path : str or Path
            Path to save the configuration JSON.
        """
        model_path = Path(model_path)
        config_path = Path(config_path)

        # Save model
        self.model.save(str(model_path))

        # Save config
        config = {
            "n_vars": self.n_vars,
            "n_terms": self.n_terms,
            "feature_mean": self.feature_mean.tolist() if self.feature_mean is not None else None,
            "feature_std": self.feature_std.tolist() if self.feature_std is not None else None,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls, model_path: Union[str, Path], config_path: Union[str, Path]
    ) -> "StructurePredictor":
        """
        Load predictor from saved model and config.

        Parameters
        ----------
        model_path : str or Path
            Path to the saved model weights.
        config_path : str or Path
            Path to the configuration JSON.

        Returns
        -------
        predictor : StructurePredictor
            Loaded predictor ready for inference.
        """
        model_path = Path(model_path)
        config_path = Path(config_path)

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load model
        model = StructureNetwork.load(str(model_path))

        # Parse normalization stats
        feature_mean = (
            np.array(config["feature_mean"]) if config.get("feature_mean") is not None else None
        )
        feature_std = (
            np.array(config["feature_std"]) if config.get("feature_std") is not None else None
        )

        return cls(
            model=model,
            n_vars=config["n_vars"],
            n_terms=config["n_terms"],
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    @classmethod
    def from_training_results(
        cls,
        model: StructureNetwork,
        n_vars: int,
        n_terms: int,
        training_data: list,
    ) -> "StructurePredictor":
        """
        Create predictor from model and compute normalization stats from training data.

        Parameters
        ----------
        model : StructureNetwork
            Trained model.
        n_vars : int
            Number of state variables.
        n_terms : int
            Number of library terms.
        training_data : list
            List of (features, labels) tuples used for training.

        Returns
        -------
        predictor : StructurePredictor
            Predictor with normalization stats computed from training data.
        """
        # Extract all features
        all_features = np.array([features for features, _ in training_data])

        # Compute normalization stats
        feature_mean = np.mean(all_features, axis=0)
        feature_std = np.std(all_features, axis=0)
        feature_std = np.where(feature_std < 1e-10, 1.0, feature_std)

        return cls(
            model=model,
            n_vars=n_vars,
            n_terms=n_terms,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    def get_config(self) -> Dict:
        """Get predictor configuration."""
        return {
            "n_vars": self.n_vars,
            "n_terms": self.n_terms,
            "has_normalization": self.feature_mean is not None,
            "n_features": len(self.feature_mean) if self.feature_mean is not None else None,
        }
