"""
Ensemble Structure-Constrained SINDy implementation.

This module combines SC-SINDy's learned structural priors with Ensemble-SINDy's
bootstrap robustness for improved equation discovery.

The combination leverages two independent sources of evidence:
1. SC-SINDy: P(term active | trajectory features) - learned from training systems
2. E-SINDy: P(term active | bootstrap statistics) - computed from data

These probabilities can be fused to provide more robust structure recovery
and uncertainty quantification.
"""

import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from .ensemble import EnsembleResult, _aggregate_ensemble
from .sindy import _stls_single
from .structure_constrained import DEFAULT_STLS_THRESHOLD, DEFAULT_STRUCTURE_THRESHOLD


@dataclass
class EnsembleSCResult:
    """Container for Ensemble-SC-SINDy results.

    Attributes
    ----------
    xi : np.ndarray
        Final aggregated coefficient matrix [n_vars, n_terms].
    structure_probs : np.ndarray
        Network structure probabilities [n_vars, n_terms].
    ensemble_probs : np.ndarray
        Bootstrap inclusion probabilities [n_vars, n_terms].
    combined_probs : np.ndarray
        Fused probabilities [n_vars, n_terms].
    xi_mean : np.ndarray
        Mean coefficients across ensemble [n_vars, n_terms].
    xi_median : np.ndarray
        Median coefficients across ensemble [n_vars, n_terms].
    xi_std : np.ndarray
        Standard deviation of coefficients [n_vars, n_terms].
    confidence_intervals : np.ndarray
        95% confidence intervals [n_vars, n_terms, 2].
    ensemble_coeffs : np.ndarray
        All ensemble coefficients [n_bootstrap, n_vars, n_terms].
    elapsed_time : float
        Computation time in seconds.
    """

    xi: np.ndarray
    structure_probs: np.ndarray
    ensemble_probs: np.ndarray
    combined_probs: np.ndarray
    xi_mean: np.ndarray
    xi_median: np.ndarray
    xi_std: np.ndarray
    confidence_intervals: np.ndarray
    ensemble_coeffs: np.ndarray
    elapsed_time: float


def probability_fusion(
    p_structure: np.ndarray,
    p_ensemble: np.ndarray,
    method: Literal["product", "average", "weighted", "noisy_or", "max", "min"] = "product",
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Fuse structure and ensemble probabilities.

    Parameters
    ----------
    p_structure : np.ndarray
        Network structure probabilities [n_vars, n_terms].
    p_ensemble : np.ndarray
        Bootstrap inclusion probabilities [n_vars, n_terms].
    method : str, optional
        Fusion method:
        - "product": p_s * p_e (AND-like, conservative)
        - "average": (p_s + p_e) / 2
        - "weighted": alpha * p_s + (1 - alpha) * p_e
        - "noisy_or": 1 - (1 - p_s) * (1 - p_e) (OR-like, permissive)
        - "max": max(p_s, p_e)
        - "min": min(p_s, p_e)
        Default is "product".
    alpha : float, optional
        Weight for structure probs when method="weighted" (default: 0.5).

    Returns
    -------
    p_combined : np.ndarray
        Fused probabilities [n_vars, n_terms].
    """
    if method == "product":
        return p_structure * p_ensemble
    elif method == "average":
        return (p_structure + p_ensemble) / 2
    elif method == "weighted":
        return alpha * p_structure + (1 - alpha) * p_ensemble
    elif method == "noisy_or":
        return 1 - (1 - p_structure) * (1 - p_ensemble)
    elif method == "max":
        return np.maximum(p_structure, p_ensemble)
    elif method == "min":
        return np.minimum(p_structure, p_ensemble)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def ensemble_structure_constrained_sindy(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    network_probs: np.ndarray,
    n_bootstrap: int = 100,
    structure_threshold: float = DEFAULT_STRUCTURE_THRESHOLD,
    stls_threshold: float = DEFAULT_STLS_THRESHOLD,
    fusion_method: Literal["product", "average", "weighted", "noisy_or"] = "product",
    fusion_alpha: float = 0.5,
    final_threshold: float = 0.3,
    aggregation: Literal["bagging", "bragging"] = "bragging",
    random_state: Optional[int] = None,
) -> EnsembleSCResult:
    """
    Ensemble Structure-Constrained SINDy.

    Combines SC-SINDy's learned structural priors with Ensemble-SINDy's
    bootstrap robustness. This method:
    1. Runs SC-SINDy on each bootstrap sample
    2. Computes ensemble statistics and inclusion probabilities
    3. Fuses network priors with ensemble statistics
    4. Applies final thresholding based on fused probabilities

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives [n_samples, n_vars].
    network_probs : np.ndarray
        Network predictions for term inclusion [n_vars, n_terms].
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 100).
    structure_threshold : float, optional
        SC-SINDy structure threshold for each bootstrap (default: 0.3).
    stls_threshold : float, optional
        STLS threshold for refinement (default: 0.1).
    fusion_method : str, optional
        Method to fuse structure and ensemble probabilities (default: "product").
    fusion_alpha : float, optional
        Weight for weighted fusion (default: 0.5).
    final_threshold : float, optional
        Threshold on fused probabilities for final model (default: 0.3).
    aggregation : {"bagging", "bragging"}, optional
        Aggregation method (default: "bragging").
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : EnsembleSCResult
        Container with all results including fused probabilities.

    Examples
    --------
    >>> Theta = np.random.randn(100, 10)
    >>> x_dot = np.random.randn(100, 2)
    >>> network_probs = np.random.rand(2, 10)
    >>> result = ensemble_structure_constrained_sindy(
    ...     Theta, x_dot, network_probs, n_bootstrap=50
    ... )
    >>> print(f"Combined probs shape: {result.combined_probs.shape}")
    Combined probs shape: (2, 10)
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

        # Run SC-SINDy on bootstrap sample
        for i in range(n_vars):
            # Stage 1: Network-guided coarse filtering
            active_mask = network_probs[i, :] > structure_threshold

            if not np.any(active_mask):
                continue

            # Stage 2: STLS on reduced library
            Theta_reduced = Theta_boot[:, active_mask]
            xi_reduced = _stls_single(
                Theta_reduced, x_dot_boot[:, i], stls_threshold, max_iter=10
            )
            ensemble_coeffs[b, i, active_mask] = xi_reduced

    # Compute ensemble statistics
    ensemble_probs = np.mean(np.abs(ensemble_coeffs) > 1e-10, axis=0)
    xi_mean = np.mean(ensemble_coeffs, axis=0)
    xi_median = np.median(ensemble_coeffs, axis=0)
    xi_std = np.std(ensemble_coeffs, axis=0)

    # 95% confidence intervals
    confidence_intervals = np.zeros((n_vars, n_terms, 2))
    confidence_intervals[:, :, 0] = np.percentile(ensemble_coeffs, 2.5, axis=0)
    confidence_intervals[:, :, 1] = np.percentile(ensemble_coeffs, 97.5, axis=0)

    # Fuse probabilities
    combined_probs = probability_fusion(
        network_probs, ensemble_probs, method=fusion_method, alpha=fusion_alpha
    )

    # Aggregate coefficients
    if aggregation == "bagging":
        xi_agg = xi_mean.copy()
    else:  # bragging
        xi_agg = xi_median.copy()

    # Apply final threshold based on fused probabilities
    xi_agg[combined_probs < final_threshold] = 0.0

    return EnsembleSCResult(
        xi=xi_agg,
        structure_probs=network_probs,
        ensemble_probs=ensemble_probs,
        combined_probs=combined_probs,
        xi_mean=xi_mean,
        xi_median=xi_median,
        xi_std=xi_std,
        confidence_intervals=confidence_intervals,
        ensemble_coeffs=ensemble_coeffs,
        elapsed_time=time.time() - t_start,
    )


def two_stage_ensemble(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    network_probs: np.ndarray,
    structure_threshold: float = 0.1,  # Lower default for pre-filtering
    n_bootstrap: int = 100,
    stls_threshold: float = DEFAULT_STLS_THRESHOLD,
    inclusion_threshold: float = 0.5,
    aggregation: Literal["bagging", "bragging"] = "bragging",
    random_state: Optional[int] = None,
) -> EnsembleSCResult:
    """
    Two-stage Ensemble: SC-SINDy filtering followed by E-SINDy.

    Stage 1: Use SC-SINDy to pre-filter library based on network probabilities
    Stage 2: Run standard E-SINDy on the reduced library

    This reduces the search space for E-SINDy, making it more efficient
    and potentially more accurate. Since SC-SINDy acts only as a pre-filter,
    a lower threshold (default: 0.1) is used to avoid filtering out true terms.
    E-SINDy then handles the final selection via bootstrap statistics.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives [n_samples, n_vars].
    network_probs : np.ndarray
        Network predictions for term inclusion [n_vars, n_terms].
    structure_threshold : float, optional
        Threshold for Stage 1 pre-filtering (default: 0.1). Use a lower value
        than standard SC-SINDy since this is just a pre-filter; E-SINDy will
        handle the final selection.
    n_bootstrap : int, optional
        Number of bootstrap samples for Stage 2 (default: 100).
    stls_threshold : float, optional
        STLS threshold (default: 0.1).
    inclusion_threshold : float, optional
        E-SINDy inclusion threshold (default: 0.5).
    aggregation : {"bagging", "bragging"}, optional
        Aggregation method (default: "bragging").
    random_state : int, optional
        Random seed.

    Returns
    -------
    result : EnsembleSCResult
        Results with ensemble statistics on reduced library.

    Notes
    -----
    The key insight is that SC-SINDy's role here is to reduce the search space,
    not to make final decisions. A permissive threshold (0.1) lets through terms
    that might be relevant, and E-SINDy's bootstrap aggregation then robustly
    identifies which terms are truly active.
    """
    t_start = time.time()

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = Theta.shape[0]
    n_terms = Theta.shape[1]
    n_vars = x_dot.shape[1]

    # Stage 1: Determine active terms per variable based on network probs
    active_masks = network_probs > structure_threshold

    # Store all bootstrap results
    ensemble_coeffs = np.zeros((n_bootstrap, n_vars, n_terms))

    for b in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_samples, n_samples, replace=True)
        Theta_boot = Theta[idx]
        x_dot_boot = x_dot[idx]

        for i in range(n_vars):
            active_mask = active_masks[i, :]

            if not np.any(active_mask):
                continue

            # Run STLS on reduced library (Stage 2)
            Theta_reduced = Theta_boot[:, active_mask]
            xi_reduced = _stls_single(
                Theta_reduced, x_dot_boot[:, i], stls_threshold, max_iter=10
            )
            ensemble_coeffs[b, i, active_mask] = xi_reduced

    # Compute ensemble statistics
    ensemble_probs = np.mean(np.abs(ensemble_coeffs) > 1e-10, axis=0)
    xi_mean = np.mean(ensemble_coeffs, axis=0)
    xi_median = np.median(ensemble_coeffs, axis=0)
    xi_std = np.std(ensemble_coeffs, axis=0)

    # 95% confidence intervals
    confidence_intervals = np.zeros((n_vars, n_terms, 2))
    confidence_intervals[:, :, 0] = np.percentile(ensemble_coeffs, 2.5, axis=0)
    confidence_intervals[:, :, 1] = np.percentile(ensemble_coeffs, 97.5, axis=0)

    # Combined probs: structure filtering means ensemble probs are conditional
    # P(active | structure > threshold) * P(structure > threshold)
    combined_probs = ensemble_probs * active_masks.astype(float)

    # Aggregate coefficients
    if aggregation == "bagging":
        xi_agg = xi_mean.copy()
    else:
        xi_agg = xi_median.copy()

    # Apply inclusion threshold
    xi_agg[ensemble_probs < inclusion_threshold] = 0.0

    return EnsembleSCResult(
        xi=xi_agg,
        structure_probs=network_probs,
        ensemble_probs=ensemble_probs,
        combined_probs=combined_probs,
        xi_mean=xi_mean,
        xi_median=xi_median,
        xi_std=xi_std,
        confidence_intervals=confidence_intervals,
        ensemble_coeffs=ensemble_coeffs,
        elapsed_time=time.time() - t_start,
    )


def structure_weighted_ensemble(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    network_probs: np.ndarray,
    n_bootstrap: int = 100,
    stls_threshold: float = DEFAULT_STLS_THRESHOLD,
    inclusion_threshold: float = 0.5,
    aggregation: Literal["bagging", "bragging"] = "bragging",
    random_state: Optional[int] = None,
) -> EnsembleSCResult:
    """
    Structure-weighted Ensemble-SINDy.

    Uses network probabilities as weights when aggregating ensemble members.
    Terms with higher network probability get more weight in the final model.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix [n_samples, n_terms].
    x_dot : np.ndarray
        Time derivatives [n_samples, n_vars].
    network_probs : np.ndarray
        Network predictions for term inclusion [n_vars, n_terms].
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 100).
    stls_threshold : float, optional
        STLS threshold (default: 0.1).
    inclusion_threshold : float, optional
        Minimum inclusion probability (default: 0.5).
    aggregation : {"bagging", "bragging"}, optional
        Base aggregation before weighting (default: "bragging").
    random_state : int, optional
        Random seed.

    Returns
    -------
    result : EnsembleSCResult
        Results with structure-weighted aggregation.
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
        # Bootstrap sample
        idx = np.random.choice(n_samples, n_samples, replace=True)
        Theta_boot = Theta[idx]
        x_dot_boot = x_dot[idx]

        # Run standard STLS (no filtering)
        for i in range(n_vars):
            ensemble_coeffs[b, i, :] = _stls_single(
                Theta_boot, x_dot_boot[:, i], stls_threshold, max_iter=10
            )

    # Compute ensemble statistics
    ensemble_probs = np.mean(np.abs(ensemble_coeffs) > 1e-10, axis=0)
    xi_mean = np.mean(ensemble_coeffs, axis=0)
    xi_median = np.median(ensemble_coeffs, axis=0)
    xi_std = np.std(ensemble_coeffs, axis=0)

    # 95% confidence intervals
    confidence_intervals = np.zeros((n_vars, n_terms, 2))
    confidence_intervals[:, :, 0] = np.percentile(ensemble_coeffs, 2.5, axis=0)
    confidence_intervals[:, :, 1] = np.percentile(ensemble_coeffs, 97.5, axis=0)

    # Combined probabilities
    combined_probs = probability_fusion(network_probs, ensemble_probs, method="product")

    # Aggregate with structure weighting
    if aggregation == "bagging":
        xi_base = xi_mean.copy()
    else:
        xi_base = xi_median.copy()

    # Weight by network probabilities
    # Higher network prob = less shrinkage
    xi_agg = xi_base * network_probs

    # Apply inclusion threshold on combined probs
    xi_agg[combined_probs < inclusion_threshold] = 0.0

    return EnsembleSCResult(
        xi=xi_agg,
        structure_probs=network_probs,
        ensemble_probs=ensemble_probs,
        combined_probs=combined_probs,
        xi_mean=xi_mean,
        xi_median=xi_median,
        xi_std=xi_std,
        confidence_intervals=confidence_intervals,
        ensemble_coeffs=ensemble_coeffs,
        elapsed_time=time.time() - t_start,
    )


def get_uncertainty_report(result: EnsembleSCResult, term_names: list) -> str:
    """
    Generate a human-readable uncertainty report.

    Parameters
    ----------
    result : EnsembleSCResult
        Results from ensemble method.
    term_names : list
        Names of library terms.

    Returns
    -------
    report : str
        Formatted report string.
    """
    n_vars = result.xi.shape[0]
    lines = []
    lines.append("=" * 60)
    lines.append("Ensemble SC-SINDy Uncertainty Report")
    lines.append("=" * 60)

    for i in range(n_vars):
        lines.append(f"\nEquation {i + 1} (dx{i + 1}/dt):")
        lines.append("-" * 40)

        # Find active terms
        active_idx = np.where(np.abs(result.xi[i, :]) > 1e-10)[0]

        if len(active_idx) == 0:
            lines.append("  No active terms")
            continue

        for j in active_idx:
            coef = result.xi[i, j]
            ci_low = result.confidence_intervals[i, j, 0]
            ci_high = result.confidence_intervals[i, j, 1]
            std = result.xi_std[i, j]
            p_struct = result.structure_probs[i, j]
            p_ens = result.ensemble_probs[i, j]
            p_comb = result.combined_probs[i, j]

            lines.append(f"  {term_names[j]:>6}: {coef:+.4f}")
            lines.append(f"         95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
            lines.append(f"         std: {std:.4f}")
            lines.append(f"         P(struct): {p_struct:.3f}, P(ens): {p_ens:.3f}, P(comb): {p_comb:.3f}")

    lines.append("\n" + "=" * 60)
    lines.append(f"Computation time: {result.elapsed_time:.3f}s")

    return "\n".join(lines)
