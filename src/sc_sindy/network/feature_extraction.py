"""
Feature extraction for trajectory characterization.

This module provides functions to extract meaningful features from
trajectory data for use with the Structure Network.
"""

from typing import Optional

import numpy as np
from scipy.signal import correlate, welch
from scipy.stats import kurtosis, skew


def extract_trajectory_features(
    x: np.ndarray, dt: float, include_cross_features: bool = True
) -> np.ndarray:
    """
    Extract comprehensive features from trajectory for Structure Network.

    Features include:
    - Statistical moments (mean, std, skewness, kurtosis)
    - Energy metrics (kinetic energy proxy)
    - Temporal features (autocorrelation, spectral peaks)
    - Phase space geometry (attractor dimension proxy)

    Parameters
    ----------
    x : np.ndarray
        State trajectory with shape [n_samples, n_vars].
    dt : float
        Time step between samples.
    include_cross_features : bool, optional
        Whether to include cross-variable features (default: True).

    Returns
    -------
    features : np.ndarray
        Feature vector.

    Examples
    --------
    >>> x = np.random.randn(1000, 2)
    >>> dt = 0.01
    >>> features = extract_trajectory_features(x, dt)
    """
    features = []
    n_vars = x.shape[1]

    for i in range(n_vars):
        xi = x[:, i]

        # Basic statistics
        features.extend([np.mean(xi), np.std(xi), skew(xi), kurtosis(xi)])

        # Energy (mean squared value)
        features.append(np.mean(xi**2))

        # Temporal correlation (characteristic time)
        autocorr = correlate(xi - np.mean(xi), xi - np.mean(xi), mode="same")
        if autocorr.max() > 0:
            autocorr = autocorr / autocorr.max()
        features.append(np.sum(autocorr > 0.5) * dt)

        # Spectral features
        n_samples = len(xi)
        nperseg = min(256, n_samples // 4)
        if nperseg >= 4:
            freqs, psd = welch(xi, fs=1 / dt, nperseg=nperseg)
            if len(psd) > 0:
                peak_idx = np.argmax(psd)
                features.append(freqs[peak_idx])
                features.append(psd[peak_idx])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])

    # Cross-variable features
    if include_cross_features and n_vars >= 2:
        # Correlation between variables
        corr_matrix = np.corrcoef(x.T)
        upper_tri = corr_matrix[np.triu_indices(n_vars, k=1)]
        features.extend(upper_tri.tolist())

        # Phase space volume proxy
        features.append(np.prod([np.std(x[:, i]) for i in range(min(n_vars, 3))]))

        # Cross-correlation lag
        for i in range(min(n_vars - 1, 2)):
            cross_corr = correlate(x[:, i], x[:, i + 1], mode="same")
            if cross_corr.max() > 0:
                cross_corr = cross_corr / cross_corr.max()
            lag = np.argmax(cross_corr) - len(cross_corr) // 2
            features.append(lag * dt)

    return np.array(features)


def extract_features_batch(
    trajectories: list, dt: float, include_cross_features: bool = True
) -> np.ndarray:
    """
    Extract features from multiple trajectories.

    Parameters
    ----------
    trajectories : list
        List of trajectory arrays, each with shape [n_samples, n_vars].
    dt : float
        Time step.
    include_cross_features : bool, optional
        Whether to include cross-variable features.

    Returns
    -------
    features : np.ndarray
        Feature matrix with shape [n_trajectories, n_features].
    """
    feature_list = []
    for traj in trajectories:
        features = extract_trajectory_features(traj, dt, include_cross_features)
        feature_list.append(features)
    return np.array(feature_list)


def get_feature_names(n_vars: int, include_cross_features: bool = True) -> list:
    """
    Get names for extracted features.

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    include_cross_features : bool, optional
        Whether cross-variable features are included.

    Returns
    -------
    names : list
        List of feature names.
    """
    var_names = [f"x{i}" for i in range(n_vars)]
    names = []

    # Per-variable features
    for var in var_names:
        names.extend(
            [
                f"{var}_mean",
                f"{var}_std",
                f"{var}_skewness",
                f"{var}_kurtosis",
                f"{var}_energy",
                f"{var}_autocorr_time",
                f"{var}_peak_freq",
                f"{var}_peak_psd",
            ]
        )

    # Cross-variable features
    if include_cross_features and n_vars >= 2:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                names.append(f"corr_{var_names[i]}_{var_names[j]}")
        names.append("phase_volume")
        for i in range(min(n_vars - 1, 2)):
            names.append(f"cross_lag_{var_names[i]}_{var_names[i+1]}")

    return names


def compute_feature_dimension(n_vars: int, include_cross_features: bool = True) -> int:
    """
    Compute the dimension of the feature vector.

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    include_cross_features : bool, optional
        Whether cross-variable features are included.

    Returns
    -------
    dim : int
        Feature vector dimension.
    """
    # 8 features per variable
    dim = 8 * n_vars

    if include_cross_features and n_vars >= 2:
        # Correlation pairs
        dim += n_vars * (n_vars - 1) // 2
        # Phase volume
        dim += 1
        # Cross-correlation lags
        dim += min(n_vars - 1, 2)

    return dim


def normalize_features(
    features: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
) -> tuple:
    """
    Normalize features to zero mean and unit variance.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix with shape [n_samples, n_features].
    mean : np.ndarray, optional
        Pre-computed mean for each feature. If None, computed from data.
    std : np.ndarray, optional
        Pre-computed std for each feature. If None, computed from data.

    Returns
    -------
    normalized : np.ndarray
        Normalized features.
    mean : np.ndarray
        Mean values used for normalization.
    std : np.ndarray
        Standard deviation values used for normalization.
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std[std < 1e-10] = 1.0  # Avoid division by zero

    normalized = (features - mean) / std
    return normalized, mean, std
