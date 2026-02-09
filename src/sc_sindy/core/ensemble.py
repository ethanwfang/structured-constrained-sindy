"""
Ensemble-SINDy implementation using bootstrap aggregating.

This module provides the Ensemble-SINDy (E-SINDy) algorithm that uses
bootstrap aggregating (bagging/bragging) for robust sparse model discovery.

References
----------
Fasel et al. (2022) "Ensemble-SINDy: Robust sparse model discovery in the
low-data, high-noise limit, with active learning and control"
Proceedings of the Royal Society A, 478(2260).
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .sindy import _stls_single

# Default parameters
DEFAULT_N_BOOTSTRAP = 100
DEFAULT_STLS_THRESHOLD = 0.1
DEFAULT_INCLUSION_THRESHOLD = 0.5


@dataclass
class EnsembleResult:
    """Container for Ensemble-SINDy results.

    Attributes
    ----------
    xi : np.ndarray
        Aggregated coefficient matrix [n_vars, n_terms].
    inclusion_probs : np.ndarray
        Inclusion probabilities for each term [n_vars, n_terms].
    xi_mean : np.ndarray
        Mean coefficients across ensemble [n_vars, n_terms].
    xi_median : np.ndarray
        Median coefficients across ensemble [n_vars, n_terms].
    xi_std : np.ndarray
        Standard deviation of coefficients [n_vars, n_terms].
    confidence_intervals : np.ndarray
        95% confidence intervals [n_vars, n_terms, 2] (lower, upper).
    ensemble_coeffs : np.ndarray
        All ensemble coefficients [n_bootstrap, n_vars, n_terms].
    elapsed_time : float
        Computation time in seconds.
    """

    xi: np.ndarray
    inclusion_probs: np.ndarray
    xi_mean: np.ndarray
    xi_median: np.ndarray
    xi_std: np.ndarray
    confidence_intervals: np.ndarray
    ensemble_coeffs: np.ndarray
    elapsed_time: float


def ensemble_sindy(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    stls_threshold: float = DEFAULT_STLS_THRESHOLD,
    inclusion_threshold: float = DEFAULT_INCLUSION_THRESHOLD,
    aggregation: Literal["bagging", "bragging"] = "bragging",
    random_state: Optional[int] = None,
) -> EnsembleResult:
    """
    Ensemble-SINDy using bootstrap aggregating.

    Creates multiple SINDy models from bootstrap samples and aggregates
    them to improve robustness in low-data, high-noise scenarios.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 100).
    stls_threshold : float, optional
        STLS threshold for each bootstrap model (default: 0.1).
    inclusion_threshold : float, optional
        Minimum inclusion probability to keep a term in final model (default: 0.5).
    aggregation : {"bagging", "bragging"}, optional
        Aggregation method: "bagging" uses mean, "bragging" uses median (default).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : EnsembleResult
        Container with aggregated coefficients, inclusion probabilities,
        confidence intervals, and ensemble statistics.

    Examples
    --------
    >>> Theta = np.random.randn(100, 10)
    >>> x_dot = np.random.randn(100, 2)
    >>> result = ensemble_sindy(Theta, x_dot, n_bootstrap=100)
    >>> print(f"Inclusion probs shape: {result.inclusion_probs.shape}")
    Inclusion probs shape: (2, 10)
    """
    t_start = time.time()

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = Theta.shape[0]
    n_terms = Theta.shape[1]
    n_vars = x_dot.shape[1]

    # Store all bootstrap results
    ensemble_coeffs = np.zeros((n_bootstrap, n_vars, n_terms))

    for b in range(n_bootstrap):
        # Bootstrap sample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        Theta_boot = Theta[idx]
        x_dot_boot = x_dot[idx]

        # Run STLS on bootstrap sample
        for i in range(n_vars):
            ensemble_coeffs[b, i, :] = _stls_single(
                Theta_boot, x_dot_boot[:, i], stls_threshold, max_iter=10
            )

    # Compute statistics
    result = _aggregate_ensemble(
        ensemble_coeffs, inclusion_threshold, aggregation, time.time() - t_start
    )

    return result


def ensemble_sindy_library_bagging(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_terms_sample: Optional[int] = None,
    stls_threshold: float = DEFAULT_STLS_THRESHOLD,
    inclusion_threshold: float = DEFAULT_INCLUSION_THRESHOLD,
    aggregation: Literal["bagging", "bragging"] = "bragging",
    random_state: Optional[int] = None,
) -> EnsembleResult:
    """
    Ensemble-SINDy using library bagging.

    Samples library terms instead of data rows for each bootstrap iteration.
    This can be more efficient for large libraries.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix with shape [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives with shape [n_samples, n_vars].
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 100).
    n_terms_sample : int, optional
        Number of terms to sample per bootstrap. If None, uses sqrt(n_terms).
    stls_threshold : float, optional
        STLS threshold for each bootstrap model (default: 0.1).
    inclusion_threshold : float, optional
        Minimum inclusion probability for final model (default: 0.5).
    aggregation : {"bagging", "bragging"}, optional
        Aggregation method (default: "bragging").
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : EnsembleResult
        Container with aggregated results.
    """
    t_start = time.time()

    if random_state is not None:
        np.random.seed(random_state)

    n_terms = Theta.shape[1]
    n_vars = x_dot.shape[1]

    if n_terms_sample is None:
        n_terms_sample = max(3, int(np.sqrt(n_terms)))

    # Store all bootstrap results
    ensemble_coeffs = np.zeros((n_bootstrap, n_vars, n_terms))

    for b in range(n_bootstrap):
        # Sample library terms without replacement
        term_idx = np.random.choice(n_terms, n_terms_sample, replace=False)
        Theta_reduced = Theta[:, term_idx]

        # Run STLS on reduced library
        for i in range(n_vars):
            xi_reduced = _stls_single(
                Theta_reduced, x_dot[:, i], stls_threshold, max_iter=10
            )
            ensemble_coeffs[b, i, term_idx] = xi_reduced

    # Compute statistics
    result = _aggregate_ensemble(
        ensemble_coeffs, inclusion_threshold, aggregation, time.time() - t_start
    )

    return result


def _aggregate_ensemble(
    ensemble_coeffs: np.ndarray,
    inclusion_threshold: float,
    aggregation: str,
    elapsed_time: float,
) -> EnsembleResult:
    """
    Aggregate ensemble results.

    Parameters
    ----------
    ensemble_coeffs : np.ndarray
        All ensemble coefficients [n_bootstrap, n_vars, n_terms].
    inclusion_threshold : float
        Minimum inclusion probability to keep term.
    aggregation : str
        "bagging" or "bragging".
    elapsed_time : float
        Computation time.

    Returns
    -------
    result : EnsembleResult
        Aggregated results.
    """
    n_bootstrap, n_vars, n_terms = ensemble_coeffs.shape

    # Compute inclusion probabilities (fraction of models with non-zero coefficient)
    inclusion_probs = np.mean(np.abs(ensemble_coeffs) > 1e-10, axis=0)

    # Compute statistics
    xi_mean = np.mean(ensemble_coeffs, axis=0)
    xi_median = np.median(ensemble_coeffs, axis=0)
    xi_std = np.std(ensemble_coeffs, axis=0)

    # 95% confidence intervals
    confidence_intervals = np.zeros((n_vars, n_terms, 2))
    confidence_intervals[:, :, 0] = np.percentile(ensemble_coeffs, 2.5, axis=0)
    confidence_intervals[:, :, 1] = np.percentile(ensemble_coeffs, 97.5, axis=0)

    # Select aggregation method
    if aggregation == "bagging":
        xi_agg = xi_mean.copy()
    else:  # bragging
        xi_agg = xi_median.copy()

    # Zero out terms below inclusion threshold
    xi_agg[inclusion_probs < inclusion_threshold] = 0.0

    return EnsembleResult(
        xi=xi_agg,
        inclusion_probs=inclusion_probs,
        xi_mean=xi_mean,
        xi_median=xi_median,
        xi_std=xi_std,
        confidence_intervals=confidence_intervals,
        ensemble_coeffs=ensemble_coeffs,
        elapsed_time=elapsed_time,
    )


def compute_inclusion_probabilities(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    stls_threshold: float = DEFAULT_STLS_THRESHOLD,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Compute inclusion probabilities via bootstrap.

    This is a lightweight function that only returns inclusion probabilities
    without full ensemble statistics.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives [n_samples, n_vars].
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 100).
    stls_threshold : float, optional
        STLS threshold (default: 0.1).
    random_state : int, optional
        Random seed.

    Returns
    -------
    inclusion_probs : np.ndarray
        Inclusion probabilities [n_vars, n_terms].
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = Theta.shape[0]
    n_terms = Theta.shape[1]
    n_vars = x_dot.shape[1]

    # Count how many times each term is active
    active_counts = np.zeros((n_vars, n_terms))

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        Theta_boot = Theta[idx]
        x_dot_boot = x_dot[idx]

        for i in range(n_vars):
            xi = _stls_single(Theta_boot, x_dot_boot[:, i], stls_threshold, max_iter=10)
            active_counts[i, :] += (np.abs(xi) > 1e-10).astype(float)

    return active_counts / n_bootstrap
