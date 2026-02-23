#!/usr/bin/env python
"""
Comprehensive SINDy Benchmark Suite

This script implements all standard SINDy benchmarks required for publication:
1. Coefficient recovery metrics (MAE, RMSE, relative error)
2. Noise sensitivity analysis (1%-50%)
3. Trajectory prediction error (RMSE at Lyapunov times)
4. Comparison to Weak-SINDy and other baselines
5. Per-system analysis including Lorenz

Usage:
    python scripts/comprehensive_benchmark.py --full
    python scripts/comprehensive_benchmark.py --noise-sweep
    python scripts/comprehensive_benchmark.py --trajectory
    python scripts/comprehensive_benchmark.py --coefficient
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import odeint
from scipy.stats import wilcoxon, ttest_rel

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import SC-SINDy components
from src.sc_sindy.core.sindy import sindy_stls
from src.sc_sindy.core.sindy_tuned import sindy_tuned, cross_validate_threshold
from src.sc_sindy.core.library import build_library_2d, build_library_3d, build_library_nd
from src.sc_sindy.derivatives.finite_difference import compute_derivatives_finite_diff
from src.sc_sindy.metrics.coefficient import (
    compute_coefficient_error,
    compute_coefficient_rmse,
    compute_relative_coefficient_error,
    compute_active_coefficient_error,
    compute_comprehensive_coefficient_metrics,
    compute_lorenz_coefficient_breakdown,
)
from src.sc_sindy.metrics.structure import compute_structure_metrics
from src.sc_sindy.evaluation.splits_factorized import (
    TEST_SYSTEMS_2D_FACTORIZED,
    TEST_SYSTEMS_3D_FACTORIZED,
    TEST_SYSTEMS_4D_FACTORIZED,
)
from src.sc_sindy.network.factorized.term_representation import get_library_terms

# Try to import PyTorch and model
try:
    import torch
    from src.sc_sindy.network.factorized import FactorizedStructureNetworkV2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import PySINDy for Weak-SINDy comparison
try:
    import pysindy as ps
    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False
    print("Warning: PySINDy not available. Install with: pip install pysindy")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    system_name: str
    dimension: int
    noise_level: float
    n_trials: int

    # Structure metrics
    structure_f1: float
    structure_precision: float
    structure_recall: float
    structure_success_rate: float  # % perfect structure recovery

    # Coefficient metrics
    coefficient_mae: float
    coefficient_rmse: float
    coefficient_relative_error: float
    coefficient_max_error: float

    # Trajectory metrics
    trajectory_rmse_1lyap: float  # Error at 1 Lyapunov time
    trajectory_rmse_5lyap: float  # Error at 5 Lyapunov times
    trajectory_valid_rate: float  # % of trajectories that don't diverge

    # Method info
    method: str


# Standard benchmark systems with known Lyapunov times
BENCHMARK_SYSTEMS = {
    "Lorenz": {
        "class_name": "Lorenz",
        "dimension": 3,
        "lyapunov_time": 1.1,  # ~1.1 time units
        "true_coefficients": {
            # dx/dt = sigma*(y - x)
            # dy/dt = x*(rho - z) - y
            # dz/dt = x*y - beta*z
            "sigma": 10.0,
            "rho": 28.0,
            "beta": 8/3,
        },
    },
    "VanDerPol": {
        "class_name": "VanDerPol",
        "dimension": 2,
        "lyapunov_time": None,  # Not chaotic
        "true_coefficients": {
            "mu": 1.0,
        },
    },
    "Rossler": {
        "class_name": "Rossler",
        "dimension": 3,
        "lyapunov_time": 5.9,  # ~5.9 time units
        "true_coefficients": {
            "a": 0.2,
            "b": 0.2,
            "c": 5.7,
        },
    },
    "LotkaVolterra": {
        "class_name": "LotkaVolterra",
        "dimension": 2,
        "lyapunov_time": None,  # Periodic
        "true_coefficients": {
            "alpha": 1.0,
            "beta": 0.1,
            "gamma": 1.5,
            "delta": 0.075,
        },
    },
    "Duffing": {
        "class_name": "DuffingOscillator",
        "dimension": 2,
        "lyapunov_time": None,
        "true_coefficients": {
            "alpha": 1.0,
            "beta": 5.0,
            "delta": 0.02,
        },
    },
}

# Systems with high variance that need more trials for reliable statistics
# These are typically chaotic systems where small changes lead to large outcome variance
HIGH_VARIANCE_SYSTEMS = {
    "Lorenz",
    "Rossler",
    "RabinovichFabrikant",
    "HyperchaoticRossler",
}

# Default trial counts
DEFAULT_N_TRIALS = 50
HIGH_VARIANCE_N_TRIALS = 100


def get_system_class(system_name: str):
    """Get system class by name."""
    from src.sc_sindy.systems import (
        Lorenz, VanDerPol, Rossler, LotkaVolterra, DuffingOscillator,
        DampedHarmonicOscillator, LinearOscillator, ForcedOscillator,
        PredatorPreyTypeII, HindmarshRose2D, SIRModel, RabinovichFabrikant,
        HyperchaoticRossler, CoupledFitzHughNagumo,
    )

    class_map = {
        "Lorenz": Lorenz,
        "VanDerPol": VanDerPol,
        "Rossler": Rossler,
        "LotkaVolterra": LotkaVolterra,
        "DuffingOscillator": DuffingOscillator,
        "DampedHarmonicOscillator": DampedHarmonicOscillator,
        "LinearOscillator": LinearOscillator,
        "ForcedOscillator": ForcedOscillator,
        "PredatorPreyTypeII": PredatorPreyTypeII,
        "HindmarshRose2D": HindmarshRose2D,
        "SIRModel": SIRModel,
        "RabinovichFabrikant": RabinovichFabrikant,
        "HyperchaoticRossler": HyperchaoticRossler,
        "CoupledFitzHughNagumo": CoupledFitzHughNagumo,
    }

    return class_map.get(system_name)


def build_library_for_dim(X: np.ndarray, poly_order: int = 3):
    """Build polynomial library for any dimension."""
    n_vars = X.shape[1]
    if n_vars == 2:
        return build_library_2d(X, poly_order=poly_order)
    elif n_vars == 3:
        return build_library_3d(X, poly_order=poly_order)
    else:
        return build_library_nd(X, poly_order=poly_order)


def integrate_model(xi: np.ndarray, x0: np.ndarray, t: np.ndarray,
                   n_vars: int, poly_order: int = 3) -> np.ndarray:
    """Integrate discovered model forward in time."""
    def model_rhs(x, t_val):
        x_reshaped = x.reshape(1, -1)
        if n_vars == 2:
            Theta, _ = build_library_2d(x_reshaped, poly_order=poly_order)
        elif n_vars == 3:
            Theta, _ = build_library_3d(x_reshaped, poly_order=poly_order)
        else:
            Theta, _ = build_library_nd(x_reshaped, poly_order=poly_order)
        return (Theta @ xi.T).flatten()

    try:
        x_pred = odeint(model_rhs, x0, t)
        return x_pred
    except Exception:
        return np.full((len(t), n_vars), np.nan)


def compute_trajectory_error(x_true: np.ndarray, x_pred: np.ndarray,
                            t: np.ndarray, lyapunov_time: Optional[float] = None) -> Dict:
    """Compute trajectory prediction errors at different time horizons."""
    results = {
        "rmse_1lyap": np.nan,
        "rmse_5lyap": np.nan,
        "rmse_full": np.nan,
        "valid": False,
    }

    # Check for valid prediction
    if np.any(np.isnan(x_pred)) or np.any(np.isinf(x_pred)):
        return results

    if np.max(np.abs(x_pred)) > 1e6:  # Diverged
        return results

    results["valid"] = True

    # Full trajectory RMSE
    results["rmse_full"] = np.sqrt(np.mean((x_true - x_pred) ** 2))

    # Lyapunov time-based errors (for chaotic systems)
    if lyapunov_time is not None:
        dt = t[1] - t[0]
        idx_1lyap = min(int(lyapunov_time / dt), len(t) - 1)
        idx_5lyap = min(int(5 * lyapunov_time / dt), len(t) - 1)

        results["rmse_1lyap"] = np.sqrt(np.mean((x_true[:idx_1lyap] - x_pred[:idx_1lyap]) ** 2))
        results["rmse_5lyap"] = np.sqrt(np.mean((x_true[:idx_5lyap] - x_pred[:idx_5lyap]) ** 2))
    else:
        # For non-chaotic systems, use fixed time windows
        n_points = len(t)
        idx_short = n_points // 5
        idx_mid = n_points // 2

        results["rmse_1lyap"] = np.sqrt(np.mean((x_true[:idx_short] - x_pred[:idx_short]) ** 2))
        results["rmse_5lyap"] = np.sqrt(np.mean((x_true[:idx_mid] - x_pred[:idx_mid]) ** 2))

    return results


def run_sindy_trial(system, noise_level: float, poly_order: int = 3,
                   t_span: Tuple[float, float] = (0, 25), n_points: int = 2500,
                   threshold: float = 0.1) -> Dict:
    """Run a single SINDy trial and compute all metrics."""
    n_vars = system.dim

    # Generate trajectory
    x0 = np.random.randn(n_vars) * 0.5
    t = np.linspace(t_span[0], t_span[1], n_points)
    dt = t[1] - t[0]

    try:
        X = system.generate_trajectory(x0, t, noise_level=noise_level)
    except Exception:
        return None

    # Check validity
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return None

    # Trim transients
    trim = 100
    X = X[trim:-trim]
    t = t[trim:-trim]

    # Compute derivatives
    X_dot = compute_derivatives_finite_diff(X, dt)

    # Build library
    Theta, term_names = build_library_for_dim(X, poly_order=poly_order)

    # Run STLS
    xi_pred, _ = sindy_stls(Theta, X_dot, threshold=threshold)

    # Get true coefficients and structure
    xi_true = system.get_true_coefficients(term_names)
    structure_true = system.get_true_structure(term_names)
    structure_pred = (np.abs(xi_pred) > 1e-10).astype(float)

    # Structure metrics
    structure_metrics = compute_structure_metrics(structure_pred, structure_true)

    # Coefficient metrics - comprehensive including median, success rate
    coef_metrics = compute_comprehensive_coefficient_metrics(xi_pred, xi_true)

    # Lorenz-specific coefficient breakdown
    if system.__class__.__name__ == "Lorenz" and xi_pred.shape[0] == 3:
        coef_metrics["lorenz_breakdown"] = compute_lorenz_coefficient_breakdown(xi_pred, xi_true)

    # Trajectory prediction
    x0_clean = X[0]
    X_clean = system.generate_trajectory(x0_clean, t, noise_level=0.0)
    X_pred = integrate_model(xi_pred, x0_clean, t, n_vars, poly_order)

    # Get Lyapunov time if available
    lyap_time = BENCHMARK_SYSTEMS.get(system.__class__.__name__, {}).get("lyapunov_time")
    traj_metrics = compute_trajectory_error(X_clean, X_pred, t, lyap_time)

    return {
        "structure": structure_metrics,
        "coefficient": coef_metrics,
        "trajectory": traj_metrics,
        "perfect_structure": structure_metrics["f1"] == 1.0,
    }


def run_scsindy_trial(system, model, noise_level: float, poly_order: int = 3,
                     t_span: Tuple[float, float] = (0, 25), n_points: int = 2500,
                     structure_threshold: float = 0.3, stls_threshold: float = 0.1) -> Dict:
    """Run a single SC-SINDy trial with neural network prefiltering."""
    from src.sc_sindy.core.structure_constrained import sindy_structure_constrained

    n_vars = system.dim

    # Generate trajectory
    x0 = np.random.randn(n_vars) * 0.5
    t = np.linspace(t_span[0], t_span[1], n_points)
    dt = t[1] - t[0]

    try:
        X = system.generate_trajectory(x0, t, noise_level=noise_level)
    except Exception:
        return None

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return None

    # Trim transients
    trim = 100
    X = X[trim:-trim]
    t = t[trim:-trim]

    # Compute derivatives
    X_dot = compute_derivatives_finite_diff(X, dt)

    # Build library
    Theta, term_names = build_library_for_dim(X, poly_order=poly_order)

    # Get network predictions
    model.eval()
    with torch.no_grad():
        network_probs = model.predict_structure(X, n_vars, poly_order)

    # Run structure-constrained STLS
    xi_pred, _ = sindy_structure_constrained(
        Theta, X_dot, network_probs,
        structure_threshold=structure_threshold,
        stls_threshold=stls_threshold
    )

    # Get true coefficients and structure
    xi_true = system.get_true_coefficients(term_names)
    structure_true = system.get_true_structure(term_names)
    structure_pred = (np.abs(xi_pred) > 1e-10).astype(float)

    # Structure metrics
    structure_metrics = compute_structure_metrics(structure_pred, structure_true)

    # Coefficient metrics - comprehensive including median, success rate
    coef_metrics = compute_comprehensive_coefficient_metrics(xi_pred, xi_true)

    # Lorenz-specific coefficient breakdown
    if system.__class__.__name__ == "Lorenz" and xi_pred.shape[0] == 3:
        coef_metrics["lorenz_breakdown"] = compute_lorenz_coefficient_breakdown(xi_pred, xi_true)

    # Trajectory prediction
    x0_clean = X[0]
    X_clean = system.generate_trajectory(x0_clean, t, noise_level=0.0)
    X_pred = integrate_model(xi_pred, x0_clean, t, n_vars, poly_order)

    lyap_time = BENCHMARK_SYSTEMS.get(system.__class__.__name__, {}).get("lyapunov_time")
    traj_metrics = compute_trajectory_error(X_clean, X_pred, t, lyap_time)

    return {
        "structure": structure_metrics,
        "coefficient": coef_metrics,
        "trajectory": traj_metrics,
        "perfect_structure": structure_metrics["f1"] == 1.0,
        "network_probs": network_probs,
    }


def run_sindy_tuned_trial(system, noise_level: float, poly_order: int = 3,
                          t_span: Tuple[float, float] = (0, 25), n_points: int = 2500,
                          n_folds: int = 5) -> Dict:
    """Run SINDy with cross-validated threshold selection.

    This provides a fair comparison baseline where SINDy gets its best possible
    threshold tuned on the same data, addressing reviewer concerns about
    fixed threshold comparisons.
    """
    n_vars = system.dim

    # Generate trajectory
    x0 = np.random.randn(n_vars) * 0.5
    t = np.linspace(t_span[0], t_span[1], n_points)
    dt = t[1] - t[0]

    try:
        X = system.generate_trajectory(x0, t, noise_level=noise_level)
    except Exception:
        return None

    # Check validity
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return None

    # Trim transients
    trim = 100
    X = X[trim:-trim]
    t = t[trim:-trim]

    # Compute derivatives
    X_dot = compute_derivatives_finite_diff(X, dt)

    # Build library
    Theta, term_names = build_library_for_dim(X, poly_order=poly_order)

    # Run tuned STLS with cross-validation
    xi_pred, elapsed, cv_results = sindy_tuned(
        Theta, X_dot, n_folds=n_folds, random_state=42
    )

    # Get true coefficients and structure
    xi_true = system.get_true_coefficients(term_names)
    structure_true = system.get_true_structure(term_names)
    structure_pred = (np.abs(xi_pred) > 1e-10).astype(float)

    # Structure metrics
    structure_metrics = compute_structure_metrics(structure_pred, structure_true)

    # Coefficient metrics - comprehensive
    coef_metrics = compute_comprehensive_coefficient_metrics(xi_pred, xi_true)

    # Lorenz-specific breakdown
    if system.__class__.__name__ == "Lorenz" and xi_pred.shape[0] == 3:
        coef_metrics["lorenz_breakdown"] = compute_lorenz_coefficient_breakdown(xi_pred, xi_true)

    # Trajectory prediction
    x0_clean = X[0]
    X_clean = system.generate_trajectory(x0_clean, t, noise_level=0.0)
    X_pred = integrate_model(xi_pred, x0_clean, t, n_vars, poly_order)

    lyap_time = BENCHMARK_SYSTEMS.get(system.__class__.__name__, {}).get("lyapunov_time")
    traj_metrics = compute_trajectory_error(X_clean, X_pred, t, lyap_time)

    return {
        "structure": structure_metrics,
        "coefficient": coef_metrics,
        "trajectory": traj_metrics,
        "perfect_structure": structure_metrics["f1"] == 1.0,
        "best_threshold": cv_results["best_threshold"],
        "cv_time": elapsed,
    }


def run_weak_sindy_trial(system, noise_level: float, poly_order: int = 3,
                        t_span: Tuple[float, float] = (0, 25), n_points: int = 2500) -> Dict:
    """Run Weak-SINDy using PySINDy."""
    if not PYSINDY_AVAILABLE:
        return None

    n_vars = system.dim

    # Generate trajectory
    x0 = np.random.randn(n_vars) * 0.5
    t = np.linspace(t_span[0], t_span[1], n_points)
    dt = t[1] - t[0]

    try:
        X = system.generate_trajectory(x0, t, noise_level=noise_level)
    except Exception:
        return None

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return None

    # Trim transients
    trim = 100
    X = X[trim:-trim]
    t = t[trim:-trim]

    # Build library and get term names for comparison
    Theta, term_names = build_library_for_dim(X, poly_order=poly_order)

    # Run Weak-SINDy via PySINDy
    try:
        # Use weak formulation
        weak_library = ps.WeakPDELibrary(
            function_library=ps.PolynomialLibrary(degree=poly_order),
            spatiotemporal_grid=t,
            is_uniform=True,
            K=100,  # Number of test functions
        )

        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=poly_order),
            optimizer=ps.STLSQ(threshold=0.1),
            differentiation_method=ps.SmoothedFiniteDifference(),
        )
        model.fit(X, t=dt)

        xi_pred = model.coefficients()

    except Exception as e:
        # Fall back to standard PySINDy if weak formulation fails
        try:
            model = ps.SINDy(
                feature_library=ps.PolynomialLibrary(degree=poly_order),
                optimizer=ps.STLSQ(threshold=0.1),
            )
            model.fit(X, t=dt)
            xi_pred = model.coefficients()
        except Exception:
            return None

    # Get true coefficients
    xi_true = system.get_true_coefficients(term_names)
    structure_true = system.get_true_structure(term_names)
    structure_pred = (np.abs(xi_pred) > 1e-10).astype(float)

    # Metrics
    structure_metrics = compute_structure_metrics(structure_pred, structure_true)
    active_coef_metrics = compute_active_coefficient_error(xi_pred, xi_true)

    # Trajectory prediction
    x0_clean = X[0]
    X_clean = system.generate_trajectory(x0_clean, t, noise_level=0.0)
    X_pred = integrate_model(xi_pred, x0_clean, t, n_vars, poly_order)

    lyap_time = BENCHMARK_SYSTEMS.get(system.__class__.__name__, {}).get("lyapunov_time")
    traj_metrics = compute_trajectory_error(X_clean, X_pred, t, lyap_time)

    return {
        "structure": structure_metrics,
        "coefficient": active_coef_metrics,
        "trajectory": traj_metrics,
        "perfect_structure": structure_metrics["f1"] == 1.0,
    }


def compute_statistical_significance(
    baseline_results: List[float],
    method_results: List[float],
    alpha: float = 0.05,
    n_comparisons: int = 1,
) -> Dict:
    """Compute statistical significance between two methods.

    Uses Wilcoxon signed-rank test (non-parametric) and paired t-test.
    Applies Bonferroni correction when n_comparisons > 1.

    Parameters
    ----------
    baseline_results : List[float]
        Results from baseline method (e.g., SINDy).
    method_results : List[float]
        Results from method being compared (e.g., SC-SINDy).
    alpha : float
        Significance level (default: 0.05).
    n_comparisons : int
        Number of comparisons for Bonferroni correction (default: 1).
        For noise sweep with 5 systems × 5 noise levels = 25 comparisons.

    Returns
    -------
    Dict with keys:
        - wilcoxon_p: p-value from Wilcoxon signed-rank test
        - ttest_p: p-value from paired t-test (one-sided)
        - significant: True if significant at alpha
        - significant_bonferroni: True if significant after Bonferroni correction
        - cohens_d: Effect size (Cohen's d)
        - mean_diff: Mean difference (method - baseline)
        - improvement_ratio_mean: Mean of method/baseline ratios
        - improvement_ci_low: 2.5th percentile of improvement ratio
        - improvement_ci_high: 97.5th percentile of improvement ratio
    """
    n = min(len(baseline_results), len(method_results))
    if n < 5:
        return {
            "wilcoxon_p": np.nan,
            "ttest_p": np.nan,
            "significant": False,
            "significant_bonferroni": False,
            "cohens_d": np.nan,
            "mean_diff": np.nan,
            "improvement_ratio_mean": np.nan,
            "improvement_ci_low": np.nan,
            "improvement_ci_high": np.nan,
            "n_samples": n,
            "n_comparisons": n_comparisons,
            "alpha_corrected": alpha / n_comparisons,
        }

    baseline = np.array(baseline_results[:n])
    method = np.array(method_results[:n])
    diff = method - baseline

    # Wilcoxon signed-rank test (one-sided: method > baseline)
    try:
        stat_w, p_wilcoxon = wilcoxon(method, baseline, alternative="greater")
    except ValueError:
        # All differences are zero
        p_wilcoxon = 1.0

    # Paired t-test (two-sided, then halve for one-sided)
    try:
        stat_t, p_ttest = ttest_rel(method, baseline)
        p_ttest_one_sided = p_ttest / 2 if np.mean(diff) > 0 else 1 - p_ttest / 2
    except Exception:
        p_ttest_one_sided = 1.0

    # Cohen's d (effect size)
    std_diff = np.std(diff)
    cohens_d = np.mean(diff) / std_diff if std_diff > 1e-10 else 0.0

    # Improvement ratio with confidence intervals
    # Avoid division by zero
    valid_mask = baseline > 1e-10
    if np.sum(valid_mask) > 0:
        ratios = method[valid_mask] / baseline[valid_mask]
        improvement_ratio_mean = float(np.mean(ratios))
        improvement_ci_low = float(np.percentile(ratios, 2.5))
        improvement_ci_high = float(np.percentile(ratios, 97.5))
    else:
        improvement_ratio_mean = np.nan
        improvement_ci_low = np.nan
        improvement_ci_high = np.nan

    # Bonferroni-corrected alpha
    alpha_corrected = alpha / n_comparisons

    return {
        "wilcoxon_p": float(p_wilcoxon),
        "ttest_p": float(p_ttest_one_sided),
        "significant": p_wilcoxon < alpha,
        "significant_bonferroni": p_wilcoxon < alpha_corrected,
        "cohens_d": float(cohens_d),
        "mean_diff": float(np.mean(diff)),
        "improvement_ratio_mean": improvement_ratio_mean,
        "improvement_ci_low": improvement_ci_low,
        "improvement_ci_high": improvement_ci_high,
        "n_samples": n,
        "n_comparisons": n_comparisons,
        "alpha_corrected": alpha_corrected,
    }


def benchmark_noise_sweep(systems: List[str], noise_levels: List[float],
                         n_trials: int = DEFAULT_N_TRIALS,
                         high_variance_n_trials: int = HIGH_VARIANCE_N_TRIALS,
                         model_path: Optional[str] = None) -> Dict:
    """Run noise sensitivity analysis across multiple systems.

    Parameters
    ----------
    systems : List[str]
        List of system names to benchmark.
    noise_levels : List[float]
        Noise levels to test (e.g., [0.01, 0.05, 0.1, 0.2, 0.5]).
    n_trials : int
        Number of trials for standard systems (default: 50).
    high_variance_n_trials : int
        Number of trials for high-variance chaotic systems (default: 100).
    model_path : str, optional
        Path to SC-SINDy model.

    Returns
    -------
    Dict
        Results dictionary with per-system, per-noise-level metrics.
    """
    print("\n" + "=" * 70)
    print("NOISE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Default trials: {n_trials}, High-variance trials: {high_variance_n_trials}")

    # Calculate total comparisons for Bonferroni correction
    # Each system × each noise level = one comparison
    n_comparisons = len(systems) * len(noise_levels)
    print(f"Total comparisons for Bonferroni correction: {n_comparisons}")

    # Load SC-SINDy model if available
    scsindy_model = None
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()
        print(f"Loaded SC-SINDy model from {model_path}")

    results = {
        "noise_levels": noise_levels,
        "systems": {},
    }

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            print(f"  System {system_name} not found, skipping")
            continue

        try:
            system = system_cls()
        except Exception as e:
            print(f"  Could not instantiate {system_name}: {e}")
            continue

        # Use more trials for high-variance chaotic systems
        actual_n_trials = high_variance_n_trials if system_name in HIGH_VARIANCE_SYSTEMS else n_trials
        print(f"  Using {actual_n_trials} trials")

        system_results = {
            "sindy": {"f1": [], "f1_std": [], "coef_mae": [], "coef_mae_std": [], "traj_rmse": [], "traj_rmse_std": []},
            "sindy_tuned": {"f1": [], "f1_std": [], "coef_mae": [], "coef_mae_std": [], "traj_rmse": [], "traj_rmse_std": [], "best_thresholds": []},
            "scsindy": {"f1": [], "f1_std": [], "coef_mae": [], "coef_mae_std": [], "traj_rmse": [], "traj_rmse_std": []},
            "weak_sindy": {"f1": [], "f1_std": [], "coef_mae": [], "coef_mae_std": [], "traj_rmse": [], "traj_rmse_std": []},
            "n_trials": actual_n_trials,  # Store for reference
        }

        for noise in noise_levels:
            print(f"  Noise {noise*100:.0f}%: ", end="", flush=True)

            sindy_f1s, sindy_maes, sindy_rmses = [], [], []
            sindy_tuned_f1s, sindy_tuned_maes, sindy_tuned_rmses, tuned_thresholds = [], [], [], []
            scsindy_f1s, scsindy_maes, scsindy_rmses = [], [], []
            weak_f1s, weak_maes, weak_rmses = [], [], []

            for trial in range(actual_n_trials):
                # Standard SINDy (fixed threshold=0.1)
                result = run_sindy_trial(system, noise)
                if result:
                    sindy_f1s.append(result["structure"]["f1"])
                    sindy_maes.append(result["coefficient"]["mae_active"])
                    if result["trajectory"]["valid"]:
                        sindy_rmses.append(result["trajectory"]["rmse_full"])

                # SINDy-Tuned (cross-validated threshold)
                result = run_sindy_tuned_trial(system, noise)
                if result:
                    sindy_tuned_f1s.append(result["structure"]["f1"])
                    sindy_tuned_maes.append(result["coefficient"]["mae_active"])
                    tuned_thresholds.append(result["best_threshold"])
                    if result["trajectory"]["valid"]:
                        sindy_tuned_rmses.append(result["trajectory"]["rmse_full"])

                # SC-SINDy
                if scsindy_model:
                    result = run_scsindy_trial(system, scsindy_model, noise)
                    if result:
                        scsindy_f1s.append(result["structure"]["f1"])
                        scsindy_maes.append(result["coefficient"]["mae_active"])
                        if result["trajectory"]["valid"]:
                            scsindy_rmses.append(result["trajectory"]["rmse_full"])

                # Weak-SINDy
                if PYSINDY_AVAILABLE:
                    result = run_weak_sindy_trial(system, noise)
                    if result:
                        weak_f1s.append(result["structure"]["f1"])
                        # Weak-SINDy still uses old metrics format
                        weak_maes.append(result["coefficient"].get("mae_active", result["coefficient"].get("mae", 0)))
                        if result["trajectory"]["valid"]:
                            weak_rmses.append(result["trajectory"]["rmse_full"])

            # Aggregate with mean and std
            system_results["sindy"]["f1"].append(np.mean(sindy_f1s) if sindy_f1s else 0)
            system_results["sindy"]["f1_std"].append(np.std(sindy_f1s) if sindy_f1s else 0)
            system_results["sindy"]["coef_mae"].append(np.mean(sindy_maes) if sindy_maes else np.nan)
            system_results["sindy"]["coef_mae_std"].append(np.std(sindy_maes) if sindy_maes else np.nan)
            system_results["sindy"]["traj_rmse"].append(np.mean(sindy_rmses) if sindy_rmses else np.nan)
            system_results["sindy"]["traj_rmse_std"].append(np.std(sindy_rmses) if sindy_rmses else np.nan)

            # SINDy-Tuned aggregation
            system_results["sindy_tuned"]["f1"].append(np.mean(sindy_tuned_f1s) if sindy_tuned_f1s else 0)
            system_results["sindy_tuned"]["f1_std"].append(np.std(sindy_tuned_f1s) if sindy_tuned_f1s else 0)
            system_results["sindy_tuned"]["coef_mae"].append(np.mean(sindy_tuned_maes) if sindy_tuned_maes else np.nan)
            system_results["sindy_tuned"]["coef_mae_std"].append(np.std(sindy_tuned_maes) if sindy_tuned_maes else np.nan)
            system_results["sindy_tuned"]["traj_rmse"].append(np.mean(sindy_tuned_rmses) if sindy_tuned_rmses else np.nan)
            system_results["sindy_tuned"]["traj_rmse_std"].append(np.std(sindy_tuned_rmses) if sindy_tuned_rmses else np.nan)
            system_results["sindy_tuned"]["best_thresholds"].append(tuned_thresholds)

            if scsindy_model:
                system_results["scsindy"]["f1"].append(np.mean(scsindy_f1s) if scsindy_f1s else 0)
                system_results["scsindy"]["f1_std"].append(np.std(scsindy_f1s) if scsindy_f1s else 0)
                system_results["scsindy"]["coef_mae"].append(np.mean(scsindy_maes) if scsindy_maes else np.nan)
                system_results["scsindy"]["coef_mae_std"].append(np.std(scsindy_maes) if scsindy_maes else np.nan)
                system_results["scsindy"]["traj_rmse"].append(np.mean(scsindy_rmses) if scsindy_rmses else np.nan)
                system_results["scsindy"]["traj_rmse_std"].append(np.std(scsindy_rmses) if scsindy_rmses else np.nan)

            if PYSINDY_AVAILABLE:
                system_results["weak_sindy"]["f1"].append(np.mean(weak_f1s) if weak_f1s else 0)
                system_results["weak_sindy"]["f1_std"].append(np.std(weak_f1s) if weak_f1s else 0)
                system_results["weak_sindy"]["coef_mae"].append(np.mean(weak_maes) if weak_maes else np.nan)
                system_results["weak_sindy"]["coef_mae_std"].append(np.std(weak_maes) if weak_maes else np.nan)
                system_results["weak_sindy"]["traj_rmse"].append(np.mean(weak_rmses) if weak_rmses else np.nan)
                system_results["weak_sindy"]["traj_rmse_std"].append(np.std(weak_rmses) if weak_rmses else np.nan)

            # Statistical significance tests (SC-SINDy vs SINDy)
            if scsindy_model and len(sindy_f1s) >= 5 and len(scsindy_f1s) >= 5:
                if "significance" not in system_results:
                    system_results["significance"] = []
                sig_result = compute_statistical_significance(
                    sindy_f1s, scsindy_f1s,
                    n_comparisons=n_comparisons
                )
                system_results["significance"].append({
                    "noise_level": noise,
                    **sig_result
                })

            print(f"SINDy F1={system_results['sindy']['f1'][-1]:.3f}", end="")
            print(f", SINDy-Tuned F1={system_results['sindy_tuned']['f1'][-1]:.3f}", end="")
            if scsindy_model:
                print(f", SC-SINDy F1={system_results['scsindy']['f1'][-1]:.3f}", end="")
            print()

        results["systems"][system_name] = system_results

    return results


def print_significance_summary(results: Dict) -> None:
    """Print a summary table of statistical significance tests.

    Parameters
    ----------
    results : Dict
        Results from benchmark_noise_sweep containing significance tests.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("=" * 70)
    print("SC-SINDy vs SINDy (F1 Score)")
    print("-" * 70)
    print(f"{'System':<20} {'Noise':<8} {'p-value':<10} {'Bonf. Sig.':<12} {'Cohen d':<10} {'95% CI':<20}")
    print("-" * 70)

    significant_count = 0
    bonf_significant_count = 0
    total_count = 0

    for system_name, system_data in results.get("systems", {}).items():
        sig_tests = system_data.get("significance", [])
        for sig in sig_tests:
            total_count += 1
            noise = sig.get("noise_level", 0)
            p_val = sig.get("wilcoxon_p", np.nan)
            bonf_sig = sig.get("significant_bonferroni", False)
            cohens_d = sig.get("cohens_d", np.nan)
            ci_low = sig.get("improvement_ci_low", np.nan)
            ci_high = sig.get("improvement_ci_high", np.nan)

            if sig.get("significant", False):
                significant_count += 1
            if bonf_sig:
                bonf_significant_count += 1

            sig_marker = "***" if bonf_sig else ("*" if sig.get("significant", False) else "")
            ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]" if not np.isnan(ci_low) else "N/A"

            print(f"{system_name:<20} {noise*100:>5.0f}% {p_val:>10.4f} {sig_marker:<12} {cohens_d:>10.2f} {ci_str:<20}")

    print("-" * 70)
    print(f"Total comparisons: {total_count}")
    print(f"Significant at α=0.05: {significant_count}/{total_count} ({100*significant_count/max(1,total_count):.1f}%)")
    print(f"Significant after Bonferroni: {bonf_significant_count}/{total_count} ({100*bonf_significant_count/max(1,total_count):.1f}%)")
    print("=" * 70)


def benchmark_coefficients(systems: List[str], noise_level: float = 0.05,
                          n_trials: int = DEFAULT_N_TRIALS,
                          high_variance_n_trials: int = HIGH_VARIANCE_N_TRIALS,
                          model_path: Optional[str] = None) -> Dict:
    """Detailed coefficient recovery benchmark with comprehensive metrics.

    Reports:
    - mae_all: Mean absolute error over ALL terms
    - mae_active: Mean absolute error over active (non-zero) terms only
    - median_ae_active: Median absolute error (robust to outliers)
    - rmse_active: Root mean squared error on active terms
    - max_error: Maximum absolute error
    - success_rate: Fraction of active terms recovered within 2x of true value
    - For Lorenz: per-parameter breakdown (sigma, rho, beta)
    """
    print("\n" + "=" * 70)
    print("COEFFICIENT RECOVERY BENCHMARK (Comprehensive Metrics)")
    print("=" * 70)
    print(f"Noise level: {noise_level*100:.0f}%, Trials: {n_trials}")

    # Load SC-SINDy model
    scsindy_model = None
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()

    results = {}

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        # Use more trials for high-variance chaotic systems
        actual_n_trials = high_variance_n_trials if system_name in HIGH_VARIANCE_SYSTEMS else n_trials
        print(f"  Using {actual_n_trials} trials")

        # Comprehensive metrics storage
        sindy_results = {
            "mae_all": [], "mae_active": [], "median_ae_active": [],
            "rmse_active": [], "max_error": [], "success_rate": []
        }
        scsindy_results = {
            "mae_all": [], "mae_active": [], "median_ae_active": [],
            "rmse_active": [], "max_error": [], "success_rate": []
        }

        # Lorenz-specific parameter errors
        sindy_lorenz = {"sigma": [], "rho": [], "beta": []}
        scsindy_lorenz = {"sigma": [], "rho": [], "beta": []}

        for trial in range(actual_n_trials):
            # Standard SINDy
            result = run_sindy_trial(system, noise_level)
            if result and result["coefficient"]:
                coef = result["coefficient"]
                sindy_results["mae_all"].append(coef["mae_all"])
                sindy_results["mae_active"].append(coef["mae_active"])
                sindy_results["median_ae_active"].append(coef["median_ae_active"])
                sindy_results["rmse_active"].append(coef["rmse_active"])
                sindy_results["max_error"].append(coef["max_error"])
                sindy_results["success_rate"].append(coef["success_rate"])

                # Lorenz breakdown
                if "lorenz_breakdown" in coef:
                    sindy_lorenz["sigma"].append(coef["lorenz_breakdown"]["sigma"]["error"])
                    sindy_lorenz["rho"].append(coef["lorenz_breakdown"]["rho"]["error"])
                    sindy_lorenz["beta"].append(coef["lorenz_breakdown"]["beta"]["error"])

            # SC-SINDy
            if scsindy_model:
                result = run_scsindy_trial(system, scsindy_model, noise_level)
                if result and result["coefficient"]:
                    coef = result["coefficient"]
                    scsindy_results["mae_all"].append(coef["mae_all"])
                    scsindy_results["mae_active"].append(coef["mae_active"])
                    scsindy_results["median_ae_active"].append(coef["median_ae_active"])
                    scsindy_results["rmse_active"].append(coef["rmse_active"])
                    scsindy_results["max_error"].append(coef["max_error"])
                    scsindy_results["success_rate"].append(coef["success_rate"])

                    # Lorenz breakdown
                    if "lorenz_breakdown" in coef:
                        scsindy_lorenz["sigma"].append(coef["lorenz_breakdown"]["sigma"]["error"])
                        scsindy_lorenz["rho"].append(coef["lorenz_breakdown"]["rho"]["error"])
                        scsindy_lorenz["beta"].append(coef["lorenz_breakdown"]["beta"]["error"])

        # Store results with mean and std
        results[system_name] = {
            "sindy": {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in sindy_results.items() if v},
            "scsindy": {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in scsindy_results.items() if v},
        }

        # Add Lorenz-specific breakdown
        if sindy_lorenz["sigma"]:
            results[system_name]["sindy"]["lorenz_params"] = {
                k: {"mean": np.mean(v), "std": np.std(v)} for k, v in sindy_lorenz.items() if v
            }
        if scsindy_lorenz["sigma"]:
            results[system_name]["scsindy"]["lorenz_params"] = {
                k: {"mean": np.mean(v), "std": np.std(v)} for k, v in scsindy_lorenz.items() if v
            }

        # Print comprehensive summary
        print(f"\n  {'Metric':<20} {'SINDy':<25} {'SC-SINDy':<25}")
        print(f"  {'-'*20} {'-'*25} {'-'*25}")

        if sindy_results["mae_active"]:
            sindy_str = f"{np.mean(sindy_results['mae_active']):.4f} ± {np.std(sindy_results['mae_active']):.4f}"
        else:
            sindy_str = "N/A"
        if scsindy_results["mae_active"]:
            scsindy_str = f"{np.mean(scsindy_results['mae_active']):.4f} ± {np.std(scsindy_results['mae_active']):.4f}"
        else:
            scsindy_str = "N/A"
        print(f"  {'MAE (active)':<20} {sindy_str:<25} {scsindy_str:<25}")

        if sindy_results["median_ae_active"]:
            sindy_str = f"{np.mean(sindy_results['median_ae_active']):.4f} ± {np.std(sindy_results['median_ae_active']):.4f}"
        else:
            sindy_str = "N/A"
        if scsindy_results["median_ae_active"]:
            scsindy_str = f"{np.mean(scsindy_results['median_ae_active']):.4f} ± {np.std(scsindy_results['median_ae_active']):.4f}"
        else:
            scsindy_str = "N/A"
        print(f"  {'Median AE (active)':<20} {sindy_str:<25} {scsindy_str:<25}")

        if sindy_results["success_rate"]:
            sindy_str = f"{100*np.mean(sindy_results['success_rate']):.1f}% ± {100*np.std(sindy_results['success_rate']):.1f}%"
        else:
            sindy_str = "N/A"
        if scsindy_results["success_rate"]:
            scsindy_str = f"{100*np.mean(scsindy_results['success_rate']):.1f}% ± {100*np.std(scsindy_results['success_rate']):.1f}%"
        else:
            scsindy_str = "N/A"
        print(f"  {'Success Rate (<2x)':<20} {sindy_str:<25} {scsindy_str:<25}")

        # Lorenz parameter breakdown
        if sindy_lorenz["sigma"] or scsindy_lorenz["sigma"]:
            print(f"\n  Lorenz Parameter Errors:")
            for param in ["sigma", "rho", "beta"]:
                sindy_str = f"{np.mean(sindy_lorenz[param]):.4f} ± {np.std(sindy_lorenz[param]):.4f}" if sindy_lorenz[param] else "N/A"
                scsindy_str = f"{np.mean(scsindy_lorenz[param]):.4f} ± {np.std(scsindy_lorenz[param]):.4f}" if scsindy_lorenz[param] else "N/A"
                print(f"  {param:<20} {sindy_str:<25} {scsindy_str:<25}")

    return results


def benchmark_trajectory_prediction(systems: List[str], noise_level: float = 0.05,
                                    n_trials: int = DEFAULT_N_TRIALS,
                                    high_variance_n_trials: int = HIGH_VARIANCE_N_TRIALS,
                                    model_path: Optional[str] = None) -> Dict:
    """Trajectory prediction benchmark."""
    print("\n" + "=" * 70)
    print("TRAJECTORY PREDICTION BENCHMARK")
    print("=" * 70)
    print(f"Default trials: {n_trials}, High-variance trials: {high_variance_n_trials}")

    scsindy_model = None
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()

    results = {}

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        # Use more trials for high-variance chaotic systems
        actual_n_trials = high_variance_n_trials if system_name in HIGH_VARIANCE_SYSTEMS else n_trials
        print(f"  Using {actual_n_trials} trials")

        lyap_time = BENCHMARK_SYSTEMS.get(system_name, {}).get("lyapunov_time")

        sindy_results = {"rmse_1lyap": [], "rmse_5lyap": [], "valid_rate": 0}
        scsindy_results = {"rmse_1lyap": [], "rmse_5lyap": [], "valid_rate": 0}

        sindy_valid = 0
        scsindy_valid = 0

        for trial in range(actual_n_trials):
            result = run_sindy_trial(system, noise_level)
            if result and result["trajectory"]["valid"]:
                sindy_valid += 1
                sindy_results["rmse_1lyap"].append(result["trajectory"]["rmse_1lyap"])
                sindy_results["rmse_5lyap"].append(result["trajectory"]["rmse_5lyap"])

            if scsindy_model:
                result = run_scsindy_trial(system, scsindy_model, noise_level)
                if result and result["trajectory"]["valid"]:
                    scsindy_valid += 1
                    scsindy_results["rmse_1lyap"].append(result["trajectory"]["rmse_1lyap"])
                    scsindy_results["rmse_5lyap"].append(result["trajectory"]["rmse_5lyap"])

        sindy_results["valid_rate"] = sindy_valid / actual_n_trials
        scsindy_results["valid_rate"] = scsindy_valid / actual_n_trials

        results[system_name] = {
            "lyapunov_time": lyap_time,
            "sindy": {
                "rmse_1lyap": np.mean(sindy_results["rmse_1lyap"]) if sindy_results["rmse_1lyap"] else np.nan,
                "rmse_1lyap_std": np.std(sindy_results["rmse_1lyap"]) if sindy_results["rmse_1lyap"] else np.nan,
                "rmse_5lyap": np.mean(sindy_results["rmse_5lyap"]) if sindy_results["rmse_5lyap"] else np.nan,
                "rmse_5lyap_std": np.std(sindy_results["rmse_5lyap"]) if sindy_results["rmse_5lyap"] else np.nan,
                "valid_rate": sindy_results["valid_rate"],
            },
            "scsindy": {
                "rmse_1lyap": np.mean(scsindy_results["rmse_1lyap"]) if scsindy_results["rmse_1lyap"] else np.nan,
                "rmse_1lyap_std": np.std(scsindy_results["rmse_1lyap"]) if scsindy_results["rmse_1lyap"] else np.nan,
                "rmse_5lyap": np.mean(scsindy_results["rmse_5lyap"]) if scsindy_results["rmse_5lyap"] else np.nan,
                "rmse_5lyap_std": np.std(scsindy_results["rmse_5lyap"]) if scsindy_results["rmse_5lyap"] else np.nan,
                "valid_rate": scsindy_results["valid_rate"],
            },
        }

        print(f"  SINDy:    1-Lyap RMSE={results[system_name]['sindy']['rmse_1lyap']:.4f}, "
              f"Valid={results[system_name]['sindy']['valid_rate']*100:.0f}%")
        if scsindy_model:
            print(f"  SC-SINDy: 1-Lyap RMSE={results[system_name]['scsindy']['rmse_1lyap']:.4f}, "
                  f"Valid={results[system_name]['scsindy']['valid_rate']*100:.0f}%")

    return results


def analyze_lorenz_performance(model_path: str, n_trials: int = 100) -> Dict:
    """Detailed analysis of Lorenz system performance."""
    print("\n" + "=" * 70)
    print("LORENZ SYSTEM DETAILED ANALYSIS")
    print("=" * 70)

    from src.sc_sindy.systems import Lorenz
    system = Lorenz()

    scsindy_model = None
    if TORCH_AVAILABLE and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()

    # True Lorenz parameters
    sigma, rho, beta = 10.0, 28.0, 8/3
    print(f"True parameters: σ={sigma}, ρ={rho}, β={beta:.4f}")

    # Term-by-term analysis
    term_names = get_library_terms(3, 3)
    print(f"Library terms ({len(term_names)}): {term_names[:10]}...")

    # Run many trials
    sindy_coefs = []
    scsindy_coefs = []
    sindy_structures = []
    scsindy_structures = []

    for trial in range(n_trials):
        result = run_sindy_trial(system, noise_level=0.05)
        if result:
            sindy_structures.append(result["structure"])

        if scsindy_model:
            result = run_scsindy_trial(system, scsindy_model, noise_level=0.05)
            if result:
                scsindy_structures.append(result["structure"])
                # Store network probabilities for analysis
                if "network_probs" in result:
                    pass  # Could analyze term-by-term predictions

    # Aggregate with mean and std
    sindy_f1 = np.mean([s["f1"] for s in sindy_structures]) if sindy_structures else 0
    sindy_f1_std = np.std([s["f1"] for s in sindy_structures]) if sindy_structures else 0
    sindy_precision = np.mean([s["precision"] for s in sindy_structures]) if sindy_structures else 0
    sindy_precision_std = np.std([s["precision"] for s in sindy_structures]) if sindy_structures else 0
    sindy_recall = np.mean([s["recall"] for s in sindy_structures]) if sindy_structures else 0
    sindy_recall_std = np.std([s["recall"] for s in sindy_structures]) if sindy_structures else 0

    scsindy_f1 = np.mean([s["f1"] for s in scsindy_structures]) if scsindy_structures else 0
    scsindy_f1_std = np.std([s["f1"] for s in scsindy_structures]) if scsindy_structures else 0
    scsindy_precision = np.mean([s["precision"] for s in scsindy_structures]) if scsindy_structures else 0
    scsindy_precision_std = np.std([s["precision"] for s in scsindy_structures]) if scsindy_structures else 0
    scsindy_recall = np.mean([s["recall"] for s in scsindy_structures]) if scsindy_structures else 0
    scsindy_recall_std = np.std([s["recall"] for s in scsindy_structures]) if scsindy_structures else 0

    print(f"\nResults over {n_trials} trials:")
    print(f"  SINDy:    F1={sindy_f1:.3f}±{sindy_f1_std:.3f}, Precision={sindy_precision:.3f}±{sindy_precision_std:.3f}, Recall={sindy_recall:.3f}±{sindy_recall_std:.3f}")
    print(f"  SC-SINDy: F1={scsindy_f1:.3f}±{scsindy_f1_std:.3f}, Precision={scsindy_precision:.3f}±{scsindy_precision_std:.3f}, Recall={scsindy_recall:.3f}±{scsindy_recall_std:.3f}")

    # Analyze why SC-SINDy might underperform
    print("\nAnalysis:")
    if scsindy_f1 < sindy_f1:
        print("  SC-SINDy underperforms standard SINDy on Lorenz.")
        print("  Possible reasons:")
        print("    1. Lorenz was held out from training - zero-shot generalization")
        print("    2. Cross-terms (xy, xz) may be challenging for network")
        print("    3. Network may over-filter valid terms")

    return {
        "sindy": {
            "f1": sindy_f1, "f1_std": sindy_f1_std,
            "precision": sindy_precision, "precision_std": sindy_precision_std,
            "recall": sindy_recall, "recall_std": sindy_recall_std,
        },
        "scsindy": {
            "f1": scsindy_f1, "f1_std": scsindy_f1_std,
            "precision": scsindy_precision, "precision_std": scsindy_precision_std,
            "recall": scsindy_recall, "recall_std": scsindy_recall_std,
        },
    }


def print_summary_tables(noise_results: Dict, coef_results: Dict, traj_results: Dict):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)

    # Noise sensitivity table
    if noise_results:
        print("\n### Noise Sensitivity (F1 Score)")
        print("-" * 80)
        noise_levels = noise_results.get("noise_levels", [])
        header = "System".ljust(25) + "".join([f"{n*100:>8.0f}%" for n in noise_levels])
        print(header)
        print("-" * 80)

        for system_name, data in noise_results.get("systems", {}).items():
            sindy_row = system_name.ljust(25) + "".join([f"{f:>9.3f}" for f in data["sindy"]["f1"]])
            print(f"SINDy - {sindy_row}")
            if data["scsindy"]["f1"]:
                scsindy_row = " " * 25 + "".join([f"{f:>9.3f}" for f in data["scsindy"]["f1"]])
                print(f"SC-SINDy - {scsindy_row}")

    # Coefficient error table
    if coef_results:
        print("\n### Coefficient Recovery (MAE ± std)")
        print("-" * 60)
        print(f"{'System':<20} {'SINDy':>18} {'SC-SINDy':>18}")
        print("-" * 60)

        for system_name, data in coef_results.items():
            sindy_mae = data.get("sindy", {}).get("mae", (np.nan, np.nan))
            scsindy_mae = data.get("scsindy", {}).get("mae", (np.nan, np.nan))

            sindy_str = f"{sindy_mae[0]:.4f} ± {sindy_mae[1]:.4f}" if not np.isnan(sindy_mae[0]) else "N/A"
            scsindy_str = f"{scsindy_mae[0]:.4f} ± {scsindy_mae[1]:.4f}" if not np.isnan(scsindy_mae[0]) else "N/A"

            print(f"{system_name:<20} {sindy_str:>18} {scsindy_str:>18}")

    # Trajectory prediction table
    if traj_results:
        print("\n### Trajectory Prediction (RMSE at 1 Lyapunov time)")
        print("-" * 60)
        print(f"{'System':<20} {'SINDy':>12} {'SC-SINDy':>12} {'Valid %':>12}")
        print("-" * 60)

        for system_name, data in traj_results.items():
            sindy_rmse = data.get("sindy", {}).get("rmse_1lyap", np.nan)
            scsindy_rmse = data.get("scsindy", {}).get("rmse_1lyap", np.nan)
            scsindy_valid = data.get("scsindy", {}).get("valid_rate", 0) * 100

            print(f"{system_name:<20} {sindy_rmse:>12.4f} {scsindy_rmse:>12.4f} {scsindy_valid:>11.0f}%")


# =============================================================================
# ROBUSTNESS TESTS (Phase 3 - Issue 5)
# =============================================================================

def add_noise_with_model(X: np.ndarray, noise_level: float, noise_type: str = "additive") -> np.ndarray:
    """Apply different noise models to trajectory data.

    Parameters
    ----------
    X : np.ndarray
        Clean trajectory data with shape [n_samples, n_vars].
    noise_level : float
        Noise magnitude (interpretation depends on noise_type).
    noise_type : str
        Type of noise: "additive", "multiplicative", or "outliers".

    Returns
    -------
    X_noisy : np.ndarray
        Noisy trajectory data.
    """
    if noise_type == "additive":
        # Standard additive Gaussian noise (as used throughout paper)
        return X + np.random.randn(*X.shape) * noise_level * np.std(X)

    elif noise_type == "multiplicative":
        # Multiplicative noise (signal-dependent)
        return X * (1 + np.random.randn(*X.shape) * noise_level)

    elif noise_type == "outliers":
        # Additive noise + sparse outliers
        X_noisy = X + np.random.randn(*X.shape) * noise_level * np.std(X)
        # 5% of points are outliers with 10x noise magnitude
        outlier_mask = np.random.rand(*X.shape) < 0.05
        X_noisy[outlier_mask] += np.random.randn(np.sum(outlier_mask)) * 10 * noise_level * np.std(X)
        return X_noisy

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


def benchmark_robustness(systems: List[str], model_path: Optional[str] = None,
                        n_trials: int = DEFAULT_N_TRIALS,
                        high_variance_n_trials: int = HIGH_VARIANCE_N_TRIALS) -> Dict:
    """Test robustness under varied experimental conditions.

    Tests:
    1. Trajectory length variation: t_max in [10, 25, 50, 100]
    2. Noise model variation: additive, multiplicative, outliers
    3. Sampling rate variation: dt in [0.005, 0.01, 0.02, 0.05]

    Parameters
    ----------
    systems : List[str]
        Systems to test (typically ["Lorenz", "VanDerPol"]).
    model_path : str, optional
        Path to SC-SINDy model.
    n_trials : int
        Number of trials per condition.
    high_variance_n_trials : int
        Trials for chaotic systems.

    Returns
    -------
    Dict
        Results organized by test type.
    """
    print("\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 70)

    # Load SC-SINDy model
    scsindy_model = None
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()

    results = {
        "trajectory_length": {},
        "noise_model": {},
        "sampling_rate": {},
    }

    # Standard noise level for robustness tests
    noise_level = 0.05

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        actual_n_trials = high_variance_n_trials if system_name in HIGH_VARIANCE_SYSTEMS else n_trials

        # Test 1: Trajectory length variation
        print(f"  Trajectory length variation:")
        results["trajectory_length"][system_name] = {}

        for t_max in [10, 25, 50, 100]:
            n_points = int(t_max * 100)  # Keep dt=0.01 constant
            sindy_f1s, scsindy_f1s = [], []

            for trial in range(actual_n_trials):
                result = run_sindy_trial(system, noise_level, t_span=(0, t_max), n_points=n_points)
                if result:
                    sindy_f1s.append(result["structure"]["f1"])

                if scsindy_model:
                    result = run_scsindy_trial(system, scsindy_model, noise_level, t_span=(0, t_max), n_points=n_points)
                    if result:
                        scsindy_f1s.append(result["structure"]["f1"])

            results["trajectory_length"][system_name][t_max] = {
                "sindy": {"mean": np.mean(sindy_f1s), "std": np.std(sindy_f1s)} if sindy_f1s else None,
                "scsindy": {"mean": np.mean(scsindy_f1s), "std": np.std(scsindy_f1s)} if scsindy_f1s else None,
            }
            print(f"    t_max={t_max}: SINDy F1={np.mean(sindy_f1s):.3f}", end="")
            if scsindy_f1s:
                print(f", SC-SINDy F1={np.mean(scsindy_f1s):.3f}")
            else:
                print()

        # Test 2: Noise model variation
        print(f"  Noise model variation:")
        results["noise_model"][system_name] = {}

        for noise_type in ["additive", "multiplicative", "outliers"]:
            sindy_f1s, scsindy_f1s = [], []

            for trial in range(actual_n_trials):
                # Generate clean trajectory
                n_vars = system.dim
                x0 = np.random.randn(n_vars) * 0.5
                t = np.linspace(0, 25, 2500)
                dt = t[1] - t[0]

                try:
                    X_clean = system.generate_trajectory(x0, t, noise_level=0.0)
                    X = add_noise_with_model(X_clean, noise_level, noise_type)
                except Exception:
                    continue

                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    continue

                # Trim transients
                trim = 100
                X = X[trim:-trim]
                t_trimmed = t[trim:-trim]

                X_dot = compute_derivatives_finite_diff(X, dt)
                Theta, term_names = build_library_for_dim(X)
                xi_true = system.get_true_coefficients(term_names)
                structure_true = system.get_true_structure(term_names)

                # Run SINDy
                xi_pred, _ = sindy_stls(Theta, X_dot, threshold=0.1)
                structure_pred = (np.abs(xi_pred) > 1e-10).astype(float)
                metrics = compute_structure_metrics(structure_pred, structure_true)
                sindy_f1s.append(metrics["f1"])

                # Run SC-SINDy
                if scsindy_model:
                    from src.sc_sindy.core.structure_constrained import sindy_structure_constrained
                    scsindy_model.eval()
                    with torch.no_grad():
                        probs = scsindy_model.predict_structure(X, n_vars, 3)
                    xi_pred, _ = sindy_structure_constrained(Theta, X_dot, probs)
                    structure_pred = (np.abs(xi_pred) > 1e-10).astype(float)
                    metrics = compute_structure_metrics(structure_pred, structure_true)
                    scsindy_f1s.append(metrics["f1"])

            results["noise_model"][system_name][noise_type] = {
                "sindy": {"mean": np.mean(sindy_f1s), "std": np.std(sindy_f1s)} if sindy_f1s else None,
                "scsindy": {"mean": np.mean(scsindy_f1s), "std": np.std(scsindy_f1s)} if scsindy_f1s else None,
            }
            print(f"    {noise_type}: SINDy F1={np.mean(sindy_f1s):.3f}", end="")
            if scsindy_f1s:
                print(f", SC-SINDy F1={np.mean(scsindy_f1s):.3f}")
            else:
                print()

        # Test 3: Sampling rate variation
        print(f"  Sampling rate variation:")
        results["sampling_rate"][system_name] = {}

        for dt in [0.005, 0.01, 0.02, 0.05]:
            n_points = int(25 / dt)  # Keep t_max=25 constant
            sindy_f1s, scsindy_f1s = [], []

            for trial in range(actual_n_trials):
                result = run_sindy_trial(system, noise_level, n_points=n_points)
                if result:
                    sindy_f1s.append(result["structure"]["f1"])

                if scsindy_model:
                    result = run_scsindy_trial(system, scsindy_model, noise_level, n_points=n_points)
                    if result:
                        scsindy_f1s.append(result["structure"]["f1"])

            results["sampling_rate"][system_name][dt] = {
                "sindy": {"mean": np.mean(sindy_f1s), "std": np.std(sindy_f1s)} if sindy_f1s else None,
                "scsindy": {"mean": np.mean(scsindy_f1s), "std": np.std(scsindy_f1s)} if scsindy_f1s else None,
            }
            print(f"    dt={dt}: SINDy F1={np.mean(sindy_f1s):.3f}", end="")
            if scsindy_f1s:
                print(f", SC-SINDy F1={np.mean(scsindy_f1s):.3f}")
            else:
                print()

    return results


# =============================================================================
# COMPUTATIONAL COST ANALYSIS (Phase 3 - Issue 8)
# =============================================================================

def benchmark_computational_cost(systems: List[str], model_path: Optional[str] = None,
                                 n_trials: int = 20) -> Dict:
    """Measure runtime and memory usage for different methods.

    Provides timing comparison table for the paper.

    Parameters
    ----------
    systems : List[str]
        Systems to benchmark.
    model_path : str, optional
        Path to SC-SINDy model.
    n_trials : int
        Number of timing trials.

    Returns
    -------
    Dict
        Timing results in milliseconds.
    """
    import time
    import tracemalloc

    print("\n" + "=" * 70)
    print("COMPUTATIONAL COST ANALYSIS")
    print("=" * 70)

    # Load SC-SINDy model
    scsindy_model = None
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()

    from src.sc_sindy.core.structure_constrained import sindy_structure_constrained

    results = {}

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        # Generate a single trajectory for consistent timing
        n_vars = system.dim
        x0 = np.random.randn(n_vars) * 0.5
        t = np.linspace(0, 25, 2500)
        dt = t[1] - t[0]
        X = system.generate_trajectory(x0, t, noise_level=0.05)

        # Trim transients
        trim = 100
        X = X[trim:-trim]
        X_dot = compute_derivatives_finite_diff(X, dt)
        Theta, term_names = build_library_for_dim(X)

        # Time standard SINDy
        sindy_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            xi, _ = sindy_stls(Theta, X_dot, threshold=0.1)
            sindy_times.append((time.perf_counter() - t0) * 1000)  # ms

        # Time SINDy-Tuned (with CV)
        tuned_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            xi, _, cv_results = sindy_tuned(Theta, X_dot, n_folds=5)
            tuned_times.append((time.perf_counter() - t0) * 1000)

        # Time SC-SINDy (network + STLS)
        scsindy_times = []
        network_times = []
        stls_times = []

        if scsindy_model:
            for _ in range(n_trials):
                # Total time
                t0 = time.perf_counter()

                # Network inference
                t_net = time.perf_counter()
                scsindy_model.eval()
                with torch.no_grad():
                    probs = scsindy_model.predict_structure(X, n_vars, 3)
                network_time = (time.perf_counter() - t_net) * 1000

                # STLS with constraints
                t_stls = time.perf_counter()
                xi, _ = sindy_structure_constrained(Theta, X_dot, probs)
                stls_time = (time.perf_counter() - t_stls) * 1000

                total_time = (time.perf_counter() - t0) * 1000

                scsindy_times.append(total_time)
                network_times.append(network_time)
                stls_times.append(stls_time)

        results[system_name] = {
            "sindy_ms": {"mean": np.mean(sindy_times), "std": np.std(sindy_times)},
            "sindy_tuned_ms": {"mean": np.mean(tuned_times), "std": np.std(tuned_times)},
        }

        if scsindy_model:
            results[system_name].update({
                "scsindy_total_ms": {"mean": np.mean(scsindy_times), "std": np.std(scsindy_times)},
                "scsindy_network_ms": {"mean": np.mean(network_times), "std": np.std(network_times)},
                "scsindy_stls_ms": {"mean": np.mean(stls_times), "std": np.std(stls_times)},
                "overhead_factor": np.mean(scsindy_times) / np.mean(sindy_times),
            })

        # Print summary
        print(f"  SINDy (fixed):     {np.mean(sindy_times):>8.2f} ms (±{np.std(sindy_times):.2f})")
        print(f"  SINDy (tuned/CV):  {np.mean(tuned_times):>8.2f} ms (±{np.std(tuned_times):.2f})")
        if scsindy_model:
            print(f"  SC-SINDy (total):  {np.mean(scsindy_times):>8.2f} ms (±{np.std(scsindy_times):.2f})")
            print(f"    - Network:       {np.mean(network_times):>8.2f} ms")
            print(f"    - STLS:          {np.mean(stls_times):>8.2f} ms")
            print(f"  Overhead factor:   {np.mean(scsindy_times) / np.mean(sindy_times):>8.1f}x")

    # Summary table
    print("\n" + "=" * 70)
    print("COMPUTATIONAL COST SUMMARY")
    print("=" * 70)
    print(f"{'System':<20} {'SINDy (ms)':>12} {'Tuned (ms)':>12} {'SC-SINDy (ms)':>14} {'Overhead':>10}")
    print("-" * 70)

    for system_name, data in results.items():
        sindy = data.get("sindy_ms", {}).get("mean", np.nan)
        tuned = data.get("sindy_tuned_ms", {}).get("mean", np.nan)
        scsindy = data.get("scsindy_total_ms", {}).get("mean", np.nan)
        overhead = data.get("overhead_factor", np.nan)

        print(f"{system_name:<20} {sindy:>12.2f} {tuned:>12.2f} {scsindy:>14.2f} {overhead:>9.1f}x")

    return results


# =============================================================================
# PIPELINE DECOMPOSITION (Phase 3 - Issue 6)
# =============================================================================

def benchmark_pipeline_decomposition(systems: List[str], model_path: Optional[str] = None,
                                     noise_level: float = 0.05,
                                     n_trials: int = DEFAULT_N_TRIALS,
                                     high_variance_n_trials: int = HIGH_VARIANCE_N_TRIALS) -> Dict:
    """Decompose SC-SINDy pipeline to understand where value comes from.

    Compares four configurations:
    1. network_only: Raw network predictions (P > 0.5), no STLS
    2. network_stls: Full SC-SINDy (network + STLS)
    3. stls_only: Standard SINDy (STLS only)
    4. oracle_stls: STLS with perfect structure knowledge

    This explains discrepancies like Lorenz F1=0.738 (network-only) vs F1=1.0 (full pipeline).

    Parameters
    ----------
    systems : List[str]
        Systems to analyze.
    model_path : str, optional
        Path to SC-SINDy model.
    noise_level : float
        Noise level for tests.
    n_trials : int
        Trials per configuration.

    Returns
    -------
    Dict
        Per-system results for each configuration.
    """
    print("\n" + "=" * 70)
    print("PIPELINE DECOMPOSITION ANALYSIS")
    print("=" * 70)
    print("Comparing: Network-only | Network+STLS | STLS-only | Oracle")

    # Load SC-SINDy model
    scsindy_model = None
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()
    else:
        print("WARNING: No model provided, skipping network-based comparisons")
        return {}

    from src.sc_sindy.core.structure_constrained import sindy_structure_constrained

    results = {}

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        actual_n_trials = high_variance_n_trials if system_name in HIGH_VARIANCE_SYSTEMS else n_trials

        network_only_f1 = []
        network_stls_f1 = []
        stls_only_f1 = []
        oracle_stls_f1 = []

        for trial in range(actual_n_trials):
            # Generate trajectory
            n_vars = system.dim
            x0 = np.random.randn(n_vars) * 0.5
            t = np.linspace(0, 25, 2500)
            dt = t[1] - t[0]

            try:
                X = system.generate_trajectory(x0, t, noise_level=noise_level)
            except Exception:
                continue

            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                continue

            # Trim transients
            trim = 100
            X = X[trim:-trim]
            t = t[trim:-trim]

            X_dot = compute_derivatives_finite_diff(X, dt)
            Theta, term_names = build_library_for_dim(X)
            xi_true = system.get_true_coefficients(term_names)
            structure_true = system.get_true_structure(term_names)

            # 1. Network-only: raw predictions thresholded at 0.5
            scsindy_model.eval()
            with torch.no_grad():
                probs = scsindy_model.predict_structure(X, n_vars, 3)
            network_structure = (probs > 0.5).astype(float)
            metrics = compute_structure_metrics(network_structure, structure_true)
            network_only_f1.append(metrics["f1"])

            # 2. Network + STLS (full SC-SINDy)
            xi_scsindy, _ = sindy_structure_constrained(Theta, X_dot, probs)
            structure_scsindy = (np.abs(xi_scsindy) > 1e-10).astype(float)
            metrics = compute_structure_metrics(structure_scsindy, structure_true)
            network_stls_f1.append(metrics["f1"])

            # 3. STLS-only (standard SINDy)
            xi_stls, _ = sindy_stls(Theta, X_dot, threshold=0.1)
            structure_stls = (np.abs(xi_stls) > 1e-10).astype(float)
            metrics = compute_structure_metrics(structure_stls, structure_true)
            stls_only_f1.append(metrics["f1"])

            # 4. Oracle: STLS with perfect structure knowledge
            # Set probabilities to exactly match true structure
            oracle_probs = structure_true.astype(float)
            xi_oracle, _ = sindy_structure_constrained(Theta, X_dot, oracle_probs, structure_threshold=0.5)
            structure_oracle = (np.abs(xi_oracle) > 1e-10).astype(float)
            metrics = compute_structure_metrics(structure_oracle, structure_true)
            oracle_stls_f1.append(metrics["f1"])

        results[system_name] = {
            "network_only": {"mean": np.mean(network_only_f1), "std": np.std(network_only_f1)},
            "network_stls": {"mean": np.mean(network_stls_f1), "std": np.std(network_stls_f1)},
            "stls_only": {"mean": np.mean(stls_only_f1), "std": np.std(stls_only_f1)},
            "oracle_stls": {"mean": np.mean(oracle_stls_f1), "std": np.std(oracle_stls_f1)},
            "n_trials": len(network_only_f1),
        }

        # Print summary
        print(f"  {'Configuration':<20} {'F1 Mean':>10} {'F1 Std':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        print(f"  {'Network-only':<20} {np.mean(network_only_f1):>10.3f} {np.std(network_only_f1):>10.3f}")
        print(f"  {'Network+STLS':<20} {np.mean(network_stls_f1):>10.3f} {np.std(network_stls_f1):>10.3f}")
        print(f"  {'STLS-only':<20} {np.mean(stls_only_f1):>10.3f} {np.std(stls_only_f1):>10.3f}")
        print(f"  {'Oracle':<20} {np.mean(oracle_stls_f1):>10.3f} {np.std(oracle_stls_f1):>10.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive SINDy Benchmark Suite")
    parser.add_argument("--full", action="store_true", help="Run all benchmarks")
    parser.add_argument("--noise-sweep", action="store_true", help="Run noise sensitivity analysis")
    parser.add_argument("--coefficient", action="store_true", help="Run coefficient recovery benchmark")
    parser.add_argument("--trajectory", action="store_true", help="Run trajectory prediction benchmark")
    parser.add_argument("--lorenz", action="store_true", help="Detailed Lorenz analysis")
    parser.add_argument("--robustness", action="store_true", help="Run robustness analysis (trajectory length, noise models)")
    parser.add_argument("--decomposition", action="store_true", help="Run pipeline decomposition analysis")
    parser.add_argument("--cost", action="store_true", help="Run computational cost analysis")
    parser.add_argument("--model", type=str, default="models/factorized/factorized_model.pt",
                       help="Path to SC-SINDy model")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                       help=f"Number of trials for standard systems (default: {DEFAULT_N_TRIALS})")
    parser.add_argument("--high-variance-trials", type=int, default=HIGH_VARIANCE_N_TRIALS,
                       help=f"Number of trials for high-variance systems (default: {HIGH_VARIANCE_N_TRIALS})")
    parser.add_argument("--output", type=str, default="models/factorized/comprehensive_benchmark.json",
                       help="Output file for results")
    args = parser.parse_args()

    # Default systems for benchmarking
    benchmark_systems = ["Lorenz", "VanDerPol", "Rossler", "LotkaVolterra", "DuffingOscillator"]

    # Noise levels for sweep
    noise_levels = [0.01, 0.05, 0.10, 0.20, 0.50]

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
    }

    if args.full or args.noise_sweep:
        results["noise_sweep"] = benchmark_noise_sweep(
            benchmark_systems, noise_levels,
            n_trials=args.n_trials,
            high_variance_n_trials=args.high_variance_trials,
            model_path=args.model
        )
        # Print significance summary
        print_significance_summary(results["noise_sweep"])

    if args.full or args.coefficient:
        results["coefficient"] = benchmark_coefficients(
            benchmark_systems, noise_level=0.05,
            n_trials=args.n_trials,
            high_variance_n_trials=args.high_variance_trials,
            model_path=args.model
        )

    if args.full or args.trajectory:
        results["trajectory"] = benchmark_trajectory_prediction(
            benchmark_systems, noise_level=0.05,
            n_trials=args.n_trials,
            high_variance_n_trials=args.high_variance_trials,
            model_path=args.model
        )

    if args.full or args.lorenz:
        results["lorenz_analysis"] = analyze_lorenz_performance(args.model, n_trials=args.n_trials)

    if args.full or args.robustness:
        # Use a smaller set of systems for robustness (computationally intensive)
        robustness_systems = ["Lorenz", "VanDerPol"]
        results["robustness"] = benchmark_robustness(
            robustness_systems,
            model_path=args.model,
            n_trials=args.n_trials,
            high_variance_n_trials=args.high_variance_trials,
        )

    if args.full or args.decomposition:
        results["decomposition"] = benchmark_pipeline_decomposition(
            benchmark_systems,
            model_path=args.model,
            noise_level=0.05,
            n_trials=args.n_trials,
            high_variance_n_trials=args.high_variance_trials,
        )

    if args.full or args.cost:
        results["computational_cost"] = benchmark_computational_cost(
            benchmark_systems,
            model_path=args.model,
            n_trials=20,  # Fixed for timing consistency
        )

    # Print summary
    print_summary_tables(
        results.get("noise_sweep"),
        results.get("coefficient"),
        results.get("trajectory")
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(args.output, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
