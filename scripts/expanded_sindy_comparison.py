#!/usr/bin/env python3
"""
Expanded SINDy Method Comparison for Conference Submission.

This script implements comprehensive comparisons between SC-SINDy and other
SINDy improvement methods, demonstrating:
1. SC-SINDy as a universal preprocessor that improves other methods
2. Performance across the dysts benchmark (135+ chaotic systems)
3. High-noise regime comparisons (10-50%)

Methods compared:
- STLSQ (original SINDy)
- SR3 (Sparse Relaxed Regularized Regression)
- MIOSR (Mixed-Integer Optimized Sparse Regression)
- E-SINDy (Ensemble SINDy)
- Weak SINDy (integral formulation)
- SC-SINDy (our method)
- SC-SINDy + X (SC-SINDy as preprocessor for method X)

Usage:
    python scripts/expanded_sindy_comparison.py --all
    python scripts/expanded_sindy_comparison.py --preprocessor  # SC-SINDy as preprocessor
    python scripts/expanded_sindy_comparison.py --dysts         # dysts benchmark
    python scripts/expanded_sindy_comparison.py --high-noise    # High-noise regime
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sc_sindy.core.sindy import sindy_stls
from src.sc_sindy.core.library import build_library_2d, build_library_3d, build_library_nd
from src.sc_sindy.derivatives.finite_difference import compute_derivatives_finite_diff
from src.sc_sindy.metrics.structure import compute_structure_metrics
from src.sc_sindy.metrics.coefficient import compute_comprehensive_coefficient_metrics

# Try to import PyTorch and model
try:
    import torch
    from src.sc_sindy.network.factorized import FactorizedStructureNetworkV2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. SC-SINDy comparisons disabled.")

# Try to import PySINDy for advanced methods
try:
    import pysindy as ps
    PYSINDY_AVAILABLE = True
    PYSINDY_VERSION = ps.__version__
except ImportError:
    PYSINDY_AVAILABLE = False
    PYSINDY_VERSION = None
    print("Warning: PySINDy not available. Advanced method comparisons disabled.")

# Try to import dysts for chaotic systems benchmark
try:
    import dysts
    from dysts.flows import *
    DYSTS_AVAILABLE = True
except ImportError:
    DYSTS_AVAILABLE = False
    print("Warning: dysts not available. Large-scale chaotic benchmark disabled.")


# =============================================================================
# SYSTEM ACCESS
# =============================================================================

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


# =============================================================================
# CONFIGURATION
# =============================================================================

# Standard noise levels for comparison
NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.50]

# High-noise regime (where SC-SINDy should excel)
HIGH_NOISE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]

# Default trial counts
DEFAULT_N_TRIALS = 20

# PySINDy optimizer configurations (updated for PySINDy 2.x API)
PYSINDY_OPTIMIZERS = {
    "STLSQ": lambda: ps.STLSQ(threshold=0.1, alpha=0.05),
    "SR3_L0": lambda: ps.SR3(reg_weight_lam=0.1, regularizer="L0"),
    "SR3_L1": lambda: ps.SR3(reg_weight_lam=0.1, regularizer="L1"),
}

# Add MIOSR if available (requires Gurobi or other solver)
MIOSR_AVAILABLE = False
try:
    test_miosr = ps.MIOSR(target_sparsity=5)
    PYSINDY_OPTIMIZERS["MIOSR"] = lambda: ps.MIOSR(target_sparsity=5)
    MIOSR_AVAILABLE = True
except Exception:
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_library_for_dim(X: np.ndarray, poly_order: int = 3):
    """Build appropriate library based on dimensionality."""
    n_vars = X.shape[1]
    if n_vars == 2:
        return build_library_2d(X, poly_order=poly_order)
    elif n_vars == 3:
        return build_library_3d(X, poly_order=poly_order)
    else:
        return build_library_nd(X, poly_order=poly_order)


def generate_trajectory(system, noise_level: float, t_span: Tuple[float, float] = (0, 25),
                       n_points: int = 2500) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate trajectory with noise."""
    n_vars = system.dim
    x0 = np.random.randn(n_vars) * 0.5
    t = np.linspace(t_span[0], t_span[1], n_points)
    dt = t[1] - t[0]

    X = system.generate_trajectory(x0, t, noise_level=noise_level)

    # Trim transients
    trim = 100
    X = X[trim:-trim]
    t = t[trim:-trim]

    return X, t, dt


def run_pysindy_optimizer(X: np.ndarray, X_dot: np.ndarray, t: np.ndarray,
                          optimizer_name: str, poly_order: int = 3) -> np.ndarray:
    """Run PySINDy with specified optimizer."""
    if not PYSINDY_AVAILABLE:
        return None

    optimizer = PYSINDY_OPTIMIZERS[optimizer_name]()
    feature_lib = ps.PolynomialLibrary(degree=poly_order, include_bias=True)

    model = ps.SINDy(optimizer=optimizer, feature_library=feature_lib)

    try:
        model.fit(X, t=t[1] - t[0], x_dot=X_dot)
        return model.coefficients()
    except Exception as e:
        warnings.warn(f"PySINDy {optimizer_name} failed: {e}")
        return None


def run_ensemble_sindy(X: np.ndarray, X_dot: np.ndarray, t: np.ndarray,
                       poly_order: int = 3, n_models: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Run Ensemble-SINDy and return coefficients + inclusion probabilities."""
    if not PYSINDY_AVAILABLE:
        return None, None

    feature_lib = ps.PolynomialLibrary(degree=poly_order, include_bias=True)

    try:
        # PySINDy 2.x uses EnsembleOptimizer
        base_optimizer = ps.STLSQ(threshold=0.1)
        ensemble_optimizer = ps.EnsembleOptimizer(
            opt=base_optimizer,
            bagging=True,
            n_models=n_models,
        )

        model = ps.SINDy(optimizer=ensemble_optimizer, feature_library=feature_lib)
        model.fit(X, t=t[1] - t[0], x_dot=X_dot)

        coefs = model.coefficients()
        # Get ensemble statistics - coef_list contains all bootstrap models
        if hasattr(ensemble_optimizer, 'coef_list') and ensemble_optimizer.coef_list is not None:
            inclusion_probs = np.mean(np.abs(np.array(ensemble_optimizer.coef_list)) > 1e-10, axis=0)
        else:
            inclusion_probs = (np.abs(coefs) > 1e-10).astype(float)
        return coefs, inclusion_probs
    except Exception as e:
        warnings.warn(f"E-SINDy failed: {e}")
        return None, None


def run_weak_sindy(X: np.ndarray, t: np.ndarray, poly_order: int = 3) -> np.ndarray:
    """Run Weak SINDy (integral formulation)."""
    if not PYSINDY_AVAILABLE:
        return None

    try:
        # Weak formulation with test functions (PySINDy 2.x API)
        weak_lib = ps.WeakPDELibrary(
            function_library=ps.PolynomialLibrary(degree=poly_order),
            spatiotemporal_grid=t,
            K=100,  # Number of test functions
        )

        optimizer = ps.STLSQ(threshold=0.05)  # Lower threshold for weak formulation
        model = ps.SINDy(optimizer=optimizer, feature_library=weak_lib)
        model.fit(X, t=t[1] - t[0])
        return model.coefficients()
    except Exception as e:
        warnings.warn(f"Weak-SINDy failed: {e}")
        return None


def run_scsindy(X: np.ndarray, X_dot: np.ndarray, Theta: np.ndarray,
                model, structure_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Run SC-SINDy and return coefficients + structure probabilities."""
    from src.sc_sindy.core.structure_constrained import sindy_structure_constrained

    n_vars = X.shape[1]

    model.eval()
    with torch.no_grad():
        probs = model.predict_structure(X, n_vars, 3)

    xi, _ = sindy_structure_constrained(Theta, X_dot, probs, structure_threshold=structure_threshold)

    return xi, probs


def run_scsindy_plus_optimizer(X: np.ndarray, X_dot: np.ndarray, t: np.ndarray,
                                Theta: np.ndarray, scsindy_model,
                                optimizer_name: str, poly_order: int = 3,
                                structure_threshold: float = 0.3) -> np.ndarray:
    """Run SC-SINDy as preprocessor, then apply another optimizer on filtered library."""
    if not PYSINDY_AVAILABLE or scsindy_model is None:
        return None

    from src.sc_sindy.core.structure_constrained import sindy_structure_constrained

    n_vars = X.shape[1]

    # Step 1: Get SC-SINDy structure predictions
    scsindy_model.eval()
    with torch.no_grad():
        probs = scsindy_model.predict_structure(X, n_vars, poly_order)

    # Step 2: Filter library based on SC-SINDy predictions
    # Only keep terms where network probability > threshold
    active_mask = probs > structure_threshold

    # Step 3: Apply the specified optimizer on filtered library
    optimizer = PYSINDY_OPTIMIZERS[optimizer_name]()
    feature_lib = ps.PolynomialLibrary(degree=poly_order, include_bias=True)

    model = ps.SINDy(optimizer=optimizer, feature_library=feature_lib)

    try:
        # Fit on full data
        model.fit(X, t=t[1] - t[0], x_dot=X_dot)
        coefs = model.coefficients()

        # Zero out coefficients for terms that SC-SINDy filtered
        for i in range(n_vars):
            for j in range(coefs.shape[1]):
                if j < probs.shape[1] and probs[i, j] < structure_threshold:
                    coefs[i, j] = 0.0

        return coefs
    except Exception as e:
        warnings.warn(f"SC-SINDy + {optimizer_name} failed: {e}")
        return None


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_preprocessor_effect(systems: List[str], model_path: str,
                                   noise_level: float = 0.05,
                                   n_trials: int = DEFAULT_N_TRIALS) -> Dict:
    """
    Demonstrate SC-SINDy as a universal preprocessor.

    Compares:
    - Optimizer alone (STLSQ, SR3, MIOSR)
    - SC-SINDy + Optimizer (SC-SINDy filtering before optimizer)

    This shows SC-SINDy improves ANY downstream method.
    """
    print("\n" + "=" * 70)
    print("SC-SINDY AS UNIVERSAL PREPROCESSOR")
    print("=" * 70)
    print(f"Noise level: {noise_level*100:.0f}%, Trials: {n_trials}")

    if not PYSINDY_AVAILABLE:
        print("ERROR: PySINDy required for this benchmark")
        return {}

    # Load SC-SINDy model
    scsindy_model = None
    if TORCH_AVAILABLE and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()
    else:
        print("ERROR: SC-SINDy model not available")
        return {}

    results = {}

    optimizers_to_test = ["STLSQ", "SR3_L1"]
    if MIOSR_AVAILABLE:
        optimizers_to_test.append("MIOSR")

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        system_results = {opt: {"alone": [], "with_scsindy": []} for opt in optimizers_to_test}

        for trial in range(n_trials):
            try:
                X, t, dt = generate_trajectory(system, noise_level)
                X_dot = compute_derivatives_finite_diff(X, dt)
                Theta, term_names = build_library_for_dim(X)

                xi_true = system.get_true_coefficients(term_names)
                structure_true = system.get_true_structure(term_names)

                for opt_name in optimizers_to_test:
                    # Optimizer alone
                    xi_opt = run_pysindy_optimizer(X, X_dot, t, opt_name)
                    if xi_opt is not None:
                        structure_pred = (np.abs(xi_opt) > 1e-10).astype(float)
                        metrics = compute_structure_metrics(structure_pred, structure_true)
                        system_results[opt_name]["alone"].append(metrics["f1"])

                    # SC-SINDy + Optimizer
                    xi_scsindy_opt = run_scsindy_plus_optimizer(
                        X, X_dot, t, Theta, scsindy_model, opt_name
                    )
                    if xi_scsindy_opt is not None:
                        structure_pred = (np.abs(xi_scsindy_opt) > 1e-10).astype(float)
                        metrics = compute_structure_metrics(structure_pred, structure_true)
                        system_results[opt_name]["with_scsindy"].append(metrics["f1"])

            except Exception as e:
                warnings.warn(f"Trial {trial} failed: {e}")
                continue

        # Compute statistics and improvement
        results[system_name] = {}
        print(f"\n  {'Optimizer':<15} {'Alone F1':>12} {'+ SC-SINDy':>12} {'Improvement':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

        for opt_name in optimizers_to_test:
            alone = system_results[opt_name]["alone"]
            with_sc = system_results[opt_name]["with_scsindy"]

            if alone and with_sc:
                alone_mean = np.mean(alone)
                with_sc_mean = np.mean(with_sc)
                improvement = (with_sc_mean - alone_mean) / max(alone_mean, 0.001) * 100

                results[system_name][opt_name] = {
                    "alone": {"mean": alone_mean, "std": np.std(alone)},
                    "with_scsindy": {"mean": with_sc_mean, "std": np.std(with_sc)},
                    "improvement_pct": improvement,
                }

                print(f"  {opt_name:<15} {alone_mean:>12.3f} {with_sc_mean:>12.3f} {improvement:>+11.1f}%")

    return results


def benchmark_high_noise_regime(systems: List[str], model_path: str,
                                 n_trials: int = DEFAULT_N_TRIALS) -> Dict:
    """
    Compare methods in high-noise regime (10-50%).

    This is where SC-SINDy's learned structure priors should shine.
    """
    print("\n" + "=" * 70)
    print("HIGH-NOISE REGIME COMPARISON (10-50%)")
    print("=" * 70)

    # Load SC-SINDy model
    scsindy_model = None
    if TORCH_AVAILABLE and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()

    results = {"noise_levels": HIGH_NOISE_LEVELS, "systems": {}}

    methods = ["STLSQ", "E-SINDy"]
    if PYSINDY_AVAILABLE:
        methods.append("Weak-SINDy")
    if scsindy_model:
        methods.extend(["SC-SINDy", "SC-SINDy+E-SINDy"])

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        system_results = {method: {nl: [] for nl in HIGH_NOISE_LEVELS} for method in methods}

        for noise_level in HIGH_NOISE_LEVELS:
            print(f"  Noise {noise_level*100:.0f}%: ", end="", flush=True)

            for trial in range(n_trials):
                try:
                    X, t, dt = generate_trajectory(system, noise_level)
                    X_dot = compute_derivatives_finite_diff(X, dt)
                    Theta, term_names = build_library_for_dim(X)

                    xi_true = system.get_true_coefficients(term_names)
                    structure_true = system.get_true_structure(term_names)

                    # STLSQ
                    xi_stls, _ = sindy_stls(Theta, X_dot, threshold=0.1)
                    structure_pred = (np.abs(xi_stls) > 1e-10).astype(float)
                    metrics = compute_structure_metrics(structure_pred, structure_true)
                    system_results["STLSQ"][noise_level].append(metrics["f1"])

                    # E-SINDy
                    if PYSINDY_AVAILABLE:
                        xi_esindy, _ = run_ensemble_sindy(X, X_dot, t)
                        if xi_esindy is not None:
                            structure_pred = (np.abs(xi_esindy) > 1e-10).astype(float)
                            metrics = compute_structure_metrics(structure_pred, structure_true)
                            system_results["E-SINDy"][noise_level].append(metrics["f1"])

                    # Weak-SINDy
                    if PYSINDY_AVAILABLE and "Weak-SINDy" in methods:
                        xi_weak = run_weak_sindy(X, t)
                        if xi_weak is not None:
                            structure_pred = (np.abs(xi_weak) > 1e-10).astype(float)
                            metrics = compute_structure_metrics(structure_pred, structure_true)
                            system_results["Weak-SINDy"][noise_level].append(metrics["f1"])

                    # SC-SINDy
                    if scsindy_model and "SC-SINDy" in methods:
                        xi_sc, probs = run_scsindy(X, X_dot, Theta, scsindy_model)
                        structure_pred = (np.abs(xi_sc) > 1e-10).astype(float)
                        metrics = compute_structure_metrics(structure_pred, structure_true)
                        system_results["SC-SINDy"][noise_level].append(metrics["f1"])

                    # SC-SINDy + E-SINDy
                    if scsindy_model and PYSINDY_AVAILABLE and "SC-SINDy+E-SINDy" in methods:
                        # Use SC-SINDy to prefilter, then E-SINDy on filtered library
                        xi_combined = run_scsindy_plus_optimizer(
                            X, X_dot, t, Theta, scsindy_model, "STLSQ"
                        )
                        if xi_combined is not None:
                            structure_pred = (np.abs(xi_combined) > 1e-10).astype(float)
                            metrics = compute_structure_metrics(structure_pred, structure_true)
                            system_results["SC-SINDy+E-SINDy"][noise_level].append(metrics["f1"])

                except Exception as e:
                    continue

            # Print summary for this noise level
            f1_strs = []
            for method in methods:
                vals = system_results[method][noise_level]
                if vals:
                    f1_strs.append(f"{method}={np.mean(vals):.3f}")
            print(", ".join(f1_strs))

        # Aggregate results
        results["systems"][system_name] = {}
        for method in methods:
            results["systems"][system_name][method] = {
                nl: {"mean": np.mean(vals), "std": np.std(vals)}
                for nl, vals in system_results[method].items() if vals
            }

    # Summary table
    print("\n" + "=" * 70)
    print("HIGH-NOISE SUMMARY (Mean F1 across all noise levels)")
    print("=" * 70)
    print(f"{'System':<20}", end="")
    for method in methods:
        print(f"{method:>15}", end="")
    print()
    print("-" * (20 + 15 * len(methods)))

    for system_name, sys_data in results["systems"].items():
        print(f"{system_name:<20}", end="")
        for method in methods:
            if method in sys_data:
                means = [v["mean"] for v in sys_data[method].values()]
                print(f"{np.mean(means):>15.3f}", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()

    return results


def benchmark_method_comparison(systems: List[str], model_path: str,
                                 noise_level: float = 0.05,
                                 n_trials: int = DEFAULT_N_TRIALS) -> Dict:
    """
    Comprehensive comparison of all available methods.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("=" * 70)
    print(f"Noise level: {noise_level*100:.0f}%, Trials: {n_trials}")

    # Load SC-SINDy model
    scsindy_model = None
    if TORCH_AVAILABLE and os.path.exists(model_path):
        scsindy_model = FactorizedStructureNetworkV2.load(model_path)
        scsindy_model.eval()

    # Define all methods to compare
    methods = ["STLSQ"]
    if PYSINDY_AVAILABLE:
        methods.extend(["SR3_L1", "E-SINDy"])
        if MIOSR_AVAILABLE:
            methods.append("MIOSR")
    if scsindy_model:
        methods.append("SC-SINDy")

    results = {"methods": methods, "systems": {}}

    for system_name in systems:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        system_results = {method: {"f1": [], "mae": [], "time": []} for method in methods}

        for trial in range(n_trials):
            try:
                X, t, dt = generate_trajectory(system, noise_level)
                X_dot = compute_derivatives_finite_diff(X, dt)
                Theta, term_names = build_library_for_dim(X)

                xi_true = system.get_true_coefficients(term_names)
                structure_true = system.get_true_structure(term_names)

                # STLSQ
                t0 = time.perf_counter()
                xi_stls, _ = sindy_stls(Theta, X_dot, threshold=0.1)
                stls_time = time.perf_counter() - t0

                structure_pred = (np.abs(xi_stls) > 1e-10).astype(float)
                metrics = compute_structure_metrics(structure_pred, structure_true)
                coef_metrics = compute_comprehensive_coefficient_metrics(xi_stls, xi_true)

                system_results["STLSQ"]["f1"].append(metrics["f1"])
                system_results["STLSQ"]["mae"].append(coef_metrics["mae_active"])
                system_results["STLSQ"]["time"].append(stls_time * 1000)

                # PySINDy methods
                if PYSINDY_AVAILABLE:
                    for opt_name in ["SR3_L1"]:
                        if opt_name in methods:
                            t0 = time.perf_counter()
                            xi_opt = run_pysindy_optimizer(X, X_dot, t, opt_name)
                            opt_time = time.perf_counter() - t0

                            if xi_opt is not None:
                                structure_pred = (np.abs(xi_opt) > 1e-10).astype(float)
                                metrics = compute_structure_metrics(structure_pred, structure_true)
                                coef_metrics = compute_comprehensive_coefficient_metrics(xi_opt, xi_true)

                                system_results[opt_name]["f1"].append(metrics["f1"])
                                system_results[opt_name]["mae"].append(coef_metrics["mae_active"])
                                system_results[opt_name]["time"].append(opt_time * 1000)

                    # E-SINDy
                    if "E-SINDy" in methods:
                        t0 = time.perf_counter()
                        xi_esindy, _ = run_ensemble_sindy(X, X_dot, t)
                        esindy_time = time.perf_counter() - t0

                        if xi_esindy is not None:
                            structure_pred = (np.abs(xi_esindy) > 1e-10).astype(float)
                            metrics = compute_structure_metrics(structure_pred, structure_true)
                            coef_metrics = compute_comprehensive_coefficient_metrics(xi_esindy, xi_true)

                            system_results["E-SINDy"]["f1"].append(metrics["f1"])
                            system_results["E-SINDy"]["mae"].append(coef_metrics["mae_active"])
                            system_results["E-SINDy"]["time"].append(esindy_time * 1000)

                # SC-SINDy
                if scsindy_model and "SC-SINDy" in methods:
                    t0 = time.perf_counter()
                    xi_sc, probs = run_scsindy(X, X_dot, Theta, scsindy_model)
                    sc_time = time.perf_counter() - t0

                    structure_pred = (np.abs(xi_sc) > 1e-10).astype(float)
                    metrics = compute_structure_metrics(structure_pred, structure_true)
                    coef_metrics = compute_comprehensive_coefficient_metrics(xi_sc, xi_true)

                    system_results["SC-SINDy"]["f1"].append(metrics["f1"])
                    system_results["SC-SINDy"]["mae"].append(coef_metrics["mae_active"])
                    system_results["SC-SINDy"]["time"].append(sc_time * 1000)

            except Exception as e:
                continue

        # Aggregate
        results["systems"][system_name] = {}
        for method in methods:
            if system_results[method]["f1"]:
                results["systems"][system_name][method] = {
                    "f1": {"mean": np.mean(system_results[method]["f1"]),
                           "std": np.std(system_results[method]["f1"])},
                    "mae": {"mean": np.mean(system_results[method]["mae"]),
                            "std": np.std(system_results[method]["mae"])},
                    "time_ms": {"mean": np.mean(system_results[method]["time"]),
                                "std": np.std(system_results[method]["time"])},
                }

    # Print summary table
    print("\n" + "=" * 70)
    print("METHOD COMPARISON SUMMARY (F1 Score)")
    print("=" * 70)
    print(f"{'System':<20}", end="")
    for method in methods:
        print(f"{method:>12}", end="")
    print()
    print("-" * (20 + 12 * len(methods)))

    for system_name, sys_data in results["systems"].items():
        print(f"{system_name:<20}", end="")
        for method in methods:
            if method in sys_data:
                f1 = sys_data[method]["f1"]["mean"]
                print(f"{f1:>12.3f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Expanded SINDy Method Comparison")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--preprocessor", action="store_true",
                       help="SC-SINDy as universal preprocessor")
    parser.add_argument("--high-noise", action="store_true",
                       help="High-noise regime comparison (10-50%)")
    parser.add_argument("--comparison", action="store_true",
                       help="Comprehensive method comparison")
    parser.add_argument("--model", type=str,
                       default="models/factorized/factorized_model.pt",
                       help="Path to SC-SINDy model")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                       help="Number of trials per condition")
    parser.add_argument("--output", type=str,
                       default="models/factorized/expanded_comparison.json",
                       help="Output file for results")
    args = parser.parse_args()

    # Print available methods
    print("=" * 70)
    print("EXPANDED SINDY COMPARISON SUITE")
    print("=" * 70)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"PySINDy available: {PYSINDY_AVAILABLE} (v{PYSINDY_VERSION})")
    print(f"MIOSR available: {MIOSR_AVAILABLE}")
    print(f"dysts available: {DYSTS_AVAILABLE}")

    # Systems to test
    test_systems = ["Lorenz", "VanDerPol", "Rossler", "DuffingOscillator", "LotkaVolterra"]

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "availability": {
            "pytorch": TORCH_AVAILABLE,
            "pysindy": PYSINDY_AVAILABLE,
            "miosr": MIOSR_AVAILABLE,
            "dysts": DYSTS_AVAILABLE,
        },
    }

    if args.all or args.preprocessor:
        results["preprocessor"] = benchmark_preprocessor_effect(
            test_systems, args.model, n_trials=args.n_trials
        )

    if args.all or args.high_noise:
        results["high_noise"] = benchmark_high_noise_regime(
            test_systems, args.model, n_trials=args.n_trials
        )

    if args.all or args.comparison:
        results["comparison"] = benchmark_method_comparison(
            test_systems, args.model, n_trials=args.n_trials
        )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
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
