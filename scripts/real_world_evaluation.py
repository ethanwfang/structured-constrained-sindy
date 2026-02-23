#!/usr/bin/env python3
"""
Real-World Dataset Evaluation: SC-SINDy as Prefilter

This script evaluates SC-SINDy as a prefilter for other SINDy methods on real-world datasets.
The goal is to demonstrate that SC-SINDy + Method outperforms Method alone.

Datasets with citations:
1. Lynx-Hare (Brunton et al., PNAS 2016)
2. Pendulum Real Video (Gao & Kutz, Proc. Royal Soc. A 2024)
3. Oscillator Video (Stollnitz, 2023)
4. Double Pendulum (Goldstein, Classical Mechanics)

Usage:
    python scripts/real_world_evaluation.py --all
    python scripts/real_world_evaluation.py --dataset lynx_hare
    python scripts/real_world_evaluation.py --dataset pendulum
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sc_sindy import (
    sindy_stls,
    build_library_2d,
    compute_derivatives_spline,
    compute_derivatives_finite_diff,
)
from sc_sindy.metrics import compute_structure_metrics

# Try to import PySINDy for method comparison
try:
    import pysindy as ps
    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False
    print("Warning: PySINDy not available. Install with: pip install pysindy")

# Try to import SC-SINDy network
try:
    import torch
    from sc_sindy.network.factorized import FactorizedStructureNetworkV2
    from sc_sindy.core.structure_constrained import sindy_structure_constrained
    SCSINDY_AVAILABLE = True
except ImportError:
    SCSINDY_AVAILABLE = False
    print("Warning: SC-SINDy network not available")


# =============================================================================
# Dataset Citations
# =============================================================================

CITATIONS = {
    "lynx_hare": {
        "bibtex": """@article{brunton2016discovering,
  title={Discovering governing equations from data by sparse identification of nonlinear dynamical systems},
  author={Brunton, Steven L and Proctor, Joshua L and Kutz, J Nathan},
  journal={Proceedings of the National Academy of Sciences},
  volume={113},
  number={15},
  pages={3932--3937},
  year={2016},
  doi={10.1073/pnas.1517384113}
}""",
        "text": "Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. PNAS, 113(15), 3932-3937.",
        "also_used": [
            "Fasel et al. (2022) E-SINDy, Proc. Royal Soc. A",
            "Messenger & Bortz (2021) Weak-SINDy, Multiscale Model. Simul."
        ]
    },
    "pendulum_real": {
        "bibtex": """@article{gao2024bayesian,
  title={Bayesian autoencoders for data-driven discovery of coordinates, governing equations and fundamental constants},
  author={Gao, Liyao Mars and Kutz, J Nathan},
  journal={Proceedings of the Royal Society A},
  volume={480},
  number={2286},
  pages={20230506},
  year={2024},
  doi={10.1098/rspa.2023.0506}
}""",
        "text": "Gao, L. M., & Kutz, J. N. (2024). Bayesian autoencoders for data-driven discovery of coordinates, governing equations and fundamental constants. Proc. Royal Soc. A, 480(2286), 20230506."
    },
    "oscillator_video": {
        "bibtex": """@misc{stollnitz2023pysindy,
  title={Using PySINDy to discover equations from experimental data},
  author={Stollnitz, Bea},
  year={2023},
  url={https://bea.stollnitz.com/blog/oscillator-pysindy/}
}""",
        "text": "Stollnitz, B. (2023). Using PySINDy to discover equations from experimental data. https://bea.stollnitz.com/blog/oscillator-pysindy/"
    },
    "double_pendulum": {
        "bibtex": """@article{champion2019data,
  title={Data-driven discovery of coordinates and governing equations},
  author={Champion, Kathleen and Lusch, Bethany and Kutz, J Nathan and Brunton, Steven L},
  journal={Proceedings of the National Academy of Sciences},
  volume={116},
  number={45},
  pages={22445--22451},
  year={2019},
  doi={10.1073/pnas.1906995116}
}""",
        "text": "Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L. (2019). Data-driven discovery of coordinates and governing equations. PNAS, 116(45), 22445-22451."
    }
}


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_lynx_hare(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load Lynx-Hare population data from Hudson Bay Company records.

    Source: Brunton et al. (2016) PNAS - Original SINDy paper

    Returns:
        X: Trajectory data [n_samples, 2] (hare, lynx populations)
        t: Time vector (years)
        metadata: Dataset information including expected terms
    """
    csv_path = data_dir / "raw" / "lynx_hare.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Lynx-Hare data not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize populations (important for SINDy)
    X = df[["hare", "lynx"]].values.astype(float)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    t = df["year"].values.astype(float)
    t = t - t[0]  # Start from 0

    metadata = {
        "name": "Lynx-Hare Population",
        "n_samples": len(t),
        "n_vars": 2,
        "dt": 1.0,  # Annual data
        "variable_names": ["Hare", "Lynx"],
        "expected_dynamics": "Lotka-Volterra predator-prey",
        "expected_terms": ["x", "y", "xy"],  # Linear + interaction
        "normalization": {"mean": X_mean.tolist(), "std": X_std.tolist()},
        "citation": CITATIONS["lynx_hare"]
    }

    return X_norm, t, metadata


def load_pendulum_real(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load real pendulum video data from Gao & Kutz (2024).

    If real data not available, generates synthetic pendulum with realistic noise.

    Source: Gao, L. M., & Kutz, J. N. (2024) Proc. Royal Soc. A
    """
    real_data_path = data_dir / "real_world" / "pendulum_real.npy"

    if real_data_path.exists():
        # Load real data if available
        data = np.load(real_data_path, allow_pickle=True).item()
        X = data["X"]
        t = data["t"]
        is_real = True
    else:
        # Generate synthetic pendulum with realistic parameters
        print("  Note: Using synthetic pendulum (real video data not downloaded)")
        print("  To use real data, download from: https://github.com/gaoliyao/BayesianSindyAutoencoder")

        # Pendulum parameters (from Gao et al. paper)
        g_L = 9.876 / 1.0  # g/L ratio discovered in paper
        damping = 0.1

        def pendulum_rhs(state, t):
            theta, omega = state
            return [omega, -g_L * np.sin(theta) - damping * omega]

        from scipy.integrate import odeint
        t = np.linspace(0, 14, 390)  # 14 seconds, 390 frames (as in paper)
        x0 = [0.5, 0.0]  # Initial angle ~30 degrees
        X = odeint(pendulum_rhs, x0, t)

        # Add realistic measurement noise (vision tracking)
        noise_level = 0.02
        X += noise_level * np.random.randn(*X.shape)
        is_real = False

    metadata = {
        "name": "Pendulum (Real Video)" if is_real else "Pendulum (Synthetic)",
        "n_samples": len(t),
        "n_vars": 2,
        "dt": t[1] - t[0],
        "variable_names": ["theta", "omega"],
        "expected_dynamics": "Damped simple pendulum",
        "expected_terms": ["y", "sin(x)"],  # omega, -g/L*sin(theta)
        "is_real_data": is_real,
        "citation": CITATIONS["pendulum_real"]
    }

    return X, t, metadata


def load_oscillator_video(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load oscillator video tracking data from Stollnitz (2023).

    Source: Bea Stollnitz blog post on PySINDy
    """
    real_data_path = data_dir / "real_world" / "oscillator_video.npy"

    if real_data_path.exists():
        data = np.load(real_data_path, allow_pickle=True).item()
        X = data["X"]
        t = data["t"]
        is_real = True
    else:
        print("  Note: Using synthetic oscillator (video data not downloaded)")
        print("  To use real data, download from: https://github.com/bstollnitz/sindy")

        # Damped harmonic oscillator (separate x and y)
        omega_x, omega_y = 2.0, 3.0  # Different frequencies
        damping_x, damping_y = 0.1, 0.15

        t = np.linspace(0, 14, 425)  # 425 frames as in blog

        # Generate x and y independently
        x = np.exp(-damping_x * t) * np.cos(omega_x * t)
        y = np.exp(-damping_y * t) * np.sin(omega_y * t)
        X = np.column_stack([x, y])

        # Add tracking noise
        X += 0.03 * np.random.randn(*X.shape)
        is_real = False

    metadata = {
        "name": "Oscillator (Video)" if is_real else "Oscillator (Synthetic)",
        "n_samples": len(t),
        "n_vars": 2,
        "dt": t[1] - t[0],
        "variable_names": ["x", "y"],
        "expected_dynamics": "Damped harmonic oscillator",
        "expected_terms": ["x", "y"],
        "is_real_data": is_real,
        "citation": CITATIONS["oscillator_video"]
    }

    return X, t, metadata


def load_double_pendulum(data_dir: Path, chaos: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate double pendulum trajectory.

    This is typically generated rather than from real data, but represents
    a standard benchmark for chaotic dynamics discovery.

    Source: Champion et al. (2019) PNAS
    """
    from scipy.integrate import odeint

    # Double pendulum parameters
    g = 9.81
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    def double_pendulum_rhs(state, t):
        th1, th2, w1, w2 = state

        delta = th2 - th1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        den2 = (L2 / L1) * den1

        dw1 = (m2 * L1 * w1 ** 2 * np.sin(delta) * np.cos(delta) +
               m2 * g * np.sin(th2) * np.cos(delta) +
               m2 * L2 * w2 ** 2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(th1)) / den1

        dw2 = (-m2 * L2 * w2 ** 2 * np.sin(delta) * np.cos(delta) +
               (m1 + m2) * g * np.sin(th1) * np.cos(delta) -
               (m1 + m2) * L1 * w1 ** 2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(th2)) / den2

        return [w1, w2, dw1, dw2]

    # Initial conditions (chaotic vs regular)
    if chaos:
        x0 = [np.pi/2, np.pi/2, 0.0, 0.0]  # Chaotic regime
    else:
        x0 = [0.1, 0.1, 0.0, 0.0]  # Small angle (nearly linear)

    t = np.linspace(0, 20, 2000)
    X = odeint(double_pendulum_rhs, x0, t)

    # Add measurement noise
    X += 0.01 * np.random.randn(*X.shape)

    metadata = {
        "name": "Double Pendulum (Chaotic)" if chaos else "Double Pendulum (Regular)",
        "n_samples": len(t),
        "n_vars": 4,
        "dt": t[1] - t[0],
        "variable_names": ["theta1", "theta2", "omega1", "omega2"],
        "expected_dynamics": "Coupled nonlinear oscillator",
        "is_chaotic": chaos,
        "citation": CITATIONS["double_pendulum"]
    }

    return X, t, metadata


# =============================================================================
# SINDy Methods
# =============================================================================

def run_stlsq(Theta: np.ndarray, X_dot: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Run standard STLSQ."""
    xi, _ = sindy_stls(Theta, X_dot, threshold=threshold)
    return xi


def run_sr3(X: np.ndarray, dt: float, threshold: float = 0.1) -> Optional[np.ndarray]:
    """Run SR3 via PySINDy (PySINDy 2.x API)."""
    if not PYSINDY_AVAILABLE:
        return None

    try:
        # PySINDy 2.x API: use reg_weight_lam and regularizer instead of threshold/thresholder
        model = ps.SINDy(
            optimizer=ps.SR3(reg_weight_lam=threshold, regularizer="L1"),
            feature_library=ps.PolynomialLibrary(degree=3)
        )
        model.fit(X, t=dt)
        return model.coefficients()
    except Exception as e:
        print(f"    SR3 failed: {e}")
        return None


def run_esindy(X: np.ndarray, dt: float, n_models: int = 20) -> Optional[np.ndarray]:
    """Run Ensemble-SINDy via PySINDy."""
    if not PYSINDY_AVAILABLE:
        return None

    try:
        base_optimizer = ps.STLSQ(threshold=0.1)
        ensemble_optimizer = ps.EnsembleOptimizer(
            opt=base_optimizer,
            bagging=True,
            n_models=n_models,
        )
        model = ps.SINDy(
            optimizer=ensemble_optimizer,
            feature_library=ps.PolynomialLibrary(degree=3)
        )
        model.fit(X, t=dt)
        return model.coefficients()
    except Exception as e:
        print(f"    E-SINDy failed: {e}")
        return None


def run_weak_sindy(X: np.ndarray, dt: float) -> Optional[np.ndarray]:
    """Run Weak-SINDy via PySINDy."""
    if not PYSINDY_AVAILABLE:
        return None

    try:
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.1),
            feature_library=ps.PolynomialLibrary(degree=3),
            differentiation_method=ps.SmoothedFiniteDifference()
        )
        model.fit(X, t=dt)
        return model.coefficients()
    except Exception as e:
        print(f"    Weak-SINDy failed: {e}")
        return None


# =============================================================================
# SC-SINDy Prefilter
# =============================================================================

def load_scsindy_model(model_path: Path):
    """Load trained SC-SINDy model."""
    if not SCSINDY_AVAILABLE:
        return None

    if not model_path.exists():
        print(f"  Warning: SC-SINDy model not found at {model_path}")
        return None

    try:
        model = FactorizedStructureNetworkV2.load(str(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"  Error loading SC-SINDy model: {e}")
        return None


def predict_structure(model, X: np.ndarray, poly_order: int = 3) -> Optional[np.ndarray]:
    """Use SC-SINDy to predict structure probabilities."""
    if model is None:
        return None

    try:
        n_vars = X.shape[1]

        # Ensure X is a numpy array (not tensor)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Use model's predict_structure method which handles stats extraction internally
        probs = model.predict_structure(X, n_vars=n_vars, poly_order=poly_order)

        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        return probs
    except Exception as e:
        print(f"  Error in structure prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def apply_scsindy_prefilter(
    Theta: np.ndarray,
    X_dot: np.ndarray,
    structure_probs: np.ndarray,
    threshold: float = 0.3
) -> np.ndarray:
    """Apply SC-SINDy structure as prefilter, then run STLS."""
    # Run structure-constrained STLS with probability-based filtering
    xi, _ = sindy_structure_constrained(
        Theta, X_dot, structure_probs,
        structure_threshold=threshold,
        stls_threshold=0.1,
        max_iter=10
    )

    return xi


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_structure_recovery(
    xi_pred: np.ndarray,
    expected_terms: List[str],
    library_labels: List[str],
    threshold: float = 0.01
) -> Dict[str, float]:
    """
    Evaluate structure recovery against expected terms.

    For real-world data, we compare against expected physics-based terms.
    """
    # Get predicted active terms
    active_mask = np.abs(xi_pred) > threshold
    predicted_terms = set()
    for i in range(xi_pred.shape[0]):
        for j in range(xi_pred.shape[1]):
            if active_mask[i, j]:
                predicted_terms.add(library_labels[j])

    # Expected terms
    expected_set = set(expected_terms)

    # Compute metrics
    true_positives = len(predicted_terms & expected_set)
    false_positives = len(predicted_terms - expected_set)
    false_negatives = len(expected_set - predicted_terms)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_terms": list(predicted_terms),
        "expected_terms": expected_terms,
        "n_predicted": len(predicted_terms),
        "n_expected": len(expected_set)
    }


def evaluate_coefficient_sparsity(xi: np.ndarray, threshold: float = 0.01) -> Dict[str, float]:
    """Evaluate sparsity of discovered coefficients."""
    active = np.abs(xi) > threshold
    n_active = np.sum(active)
    n_total = xi.size
    sparsity = 1.0 - (n_active / n_total)

    return {
        "n_active_terms": int(n_active),
        "n_total_terms": int(n_total),
        "sparsity": float(sparsity)
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_dataset(
    name: str,
    X: np.ndarray,
    t: np.ndarray,
    metadata: Dict,
    scsindy_model,
    n_trials: int = 5
) -> Dict:
    """
    Evaluate all methods on a dataset, comparing Method vs SC-SINDy+Method.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {metadata['name']}")
    print(f"  Samples: {metadata['n_samples']}, Variables: {metadata['n_vars']}")
    print(f"  Citation: {metadata['citation']['text'][:80]}...")
    print(f"{'='*60}")

    results = {
        "dataset": name,
        "metadata": metadata,
        "methods": {}
    }

    dt = metadata["dt"]
    n_vars = metadata["n_vars"]
    n_samples = metadata["n_samples"]

    # Compute derivatives
    if n_samples < 100:
        # Sparse data: use spline smoothing with explicit time array
        t_array = np.arange(n_samples) * dt
        try:
            X_dot = compute_derivatives_spline(X, dt, t=t_array)
        except Exception:
            # Fallback to finite difference with smoothing
            from scipy.ndimage import gaussian_filter1d
            X_smooth = gaussian_filter1d(X, sigma=1, axis=0)
            X_dot = compute_derivatives_finite_diff(X_smooth, dt)
    else:
        X_dot = compute_derivatives_finite_diff(X, dt)

    # Build library (2D for now)
    if n_vars == 2:
        Theta, labels = build_library_2d(X)
    else:
        # For higher dimensions, use PySINDy library
        print(f"  Note: {n_vars}D system - using PySINDy library")
        Theta, labels = None, None

    # Get SC-SINDy structure predictions
    structure_probs = None
    if scsindy_model is not None and n_vars <= 4:
        structure_probs = predict_structure(scsindy_model, X, poly_order=3)
        if structure_probs is not None:
            print(f"  SC-SINDy structure predicted: {structure_probs.shape}")

    expected_terms = metadata.get("expected_terms", [])

    # Methods to evaluate
    methods = {
        "STLSQ": lambda: run_stlsq(Theta, X_dot, threshold=0.1) if Theta is not None else None,
        "SR3": lambda: run_sr3(X, dt, threshold=0.1),
        "E-SINDy": lambda: run_esindy(X, dt, n_models=20),
        "Weak-SINDy": lambda: run_weak_sindy(X, dt),
    }

    for method_name, method_fn in methods.items():
        print(f"\n  {method_name}:")

        method_results = {
            "alone": {"runs": []},
            "with_scsindy": {"runs": []}
        }

        for trial in range(n_trials):
            # Method alone
            try:
                start = time.time()
                xi_alone = method_fn()
                time_alone = (time.time() - start) * 1000

                if xi_alone is not None and Theta is not None:
                    metrics_alone = evaluate_structure_recovery(xi_alone, expected_terms, labels)
                    metrics_alone["time_ms"] = time_alone
                    method_results["alone"]["runs"].append(metrics_alone)
            except Exception as e:
                print(f"    Trial {trial+1} failed (alone): {e}")

            # SC-SINDy + Method
            if structure_probs is not None and Theta is not None:
                try:
                    start = time.time()
                    xi_with = apply_scsindy_prefilter(Theta, X_dot, structure_probs, threshold=0.3)
                    time_with = (time.time() - start) * 1000

                    metrics_with = evaluate_structure_recovery(xi_with, expected_terms, labels)
                    metrics_with["time_ms"] = time_with
                    method_results["with_scsindy"]["runs"].append(metrics_with)
                except Exception as e:
                    print(f"    Trial {trial+1} failed (with SC-SINDy): {e}")

        # Aggregate results
        for variant in ["alone", "with_scsindy"]:
            runs = method_results[variant]["runs"]
            if runs:
                method_results[variant]["mean_f1"] = np.mean([r["f1"] for r in runs])
                method_results[variant]["std_f1"] = np.std([r["f1"] for r in runs])
                method_results[variant]["mean_precision"] = np.mean([r["precision"] for r in runs])
                method_results[variant]["mean_recall"] = np.mean([r["recall"] for r in runs])
                method_results[variant]["mean_time_ms"] = np.mean([r["time_ms"] for r in runs])

        # Compute improvement
        if method_results["alone"].get("mean_f1") and method_results["with_scsindy"].get("mean_f1"):
            alone_f1 = method_results["alone"]["mean_f1"]
            with_f1 = method_results["with_scsindy"]["mean_f1"]
            improvement = ((with_f1 - alone_f1) / alone_f1 * 100) if alone_f1 > 0 else 0
            method_results["improvement_pct"] = improvement

            print(f"    Alone:        F1={alone_f1:.3f}")
            print(f"    +SC-SINDy:    F1={with_f1:.3f} ({improvement:+.1f}%)")
        elif method_results["alone"].get("mean_f1"):
            print(f"    Alone:        F1={method_results['alone']['mean_f1']:.3f}")
            print(f"    +SC-SINDy:    N/A (structure prediction failed)")
        else:
            print(f"    Method failed or not applicable")

        results["methods"][method_name] = method_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SC-SINDy as prefilter on real-world data")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--dataset", type=str, choices=["lynx_hare", "pendulum", "oscillator", "double_pendulum"],
                        help="Specific dataset to evaluate")
    parser.add_argument("--model", type=str, default="models/factorized/factorized_model.pt",
                        help="Path to SC-SINDy model")
    parser.add_argument("--n-trials", type=int, default=5, help="Number of trials per method")
    parser.add_argument("--output", type=str, default="models/factorized/real_world_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    model_path = project_root / args.model

    print("=" * 60)
    print("Real-World Dataset Evaluation: SC-SINDy as Prefilter")
    print("=" * 60)
    print(f"\nGoal: Show that SC-SINDy + Method outperforms Method alone")
    print(f"Model: {model_path}")
    print(f"Trials per method: {args.n_trials}")

    # Load SC-SINDy model
    scsindy_model = load_scsindy_model(model_path)
    if scsindy_model is None:
        print("\nWarning: SC-SINDy model not loaded. Will only evaluate baseline methods.")

    # Dataset loaders
    datasets = {
        "lynx_hare": lambda: load_lynx_hare(data_dir),
        "pendulum": lambda: load_pendulum_real(data_dir),
        "oscillator": lambda: load_oscillator_video(data_dir),
        "double_pendulum": lambda: load_double_pendulum(data_dir),
    }

    # Determine which datasets to run
    if args.all:
        dataset_names = list(datasets.keys())
    elif args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = ["lynx_hare"]  # Default

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "n_trials": args.n_trials,
            "datasets": dataset_names
        },
        "citations": CITATIONS,
        "results": {}
    }

    # Evaluate each dataset
    for name in dataset_names:
        try:
            X, t, metadata = datasets[name]()
            results = evaluate_dataset(name, X, t, metadata, scsindy_model, args.n_trials)
            all_results["results"][name] = results
        except Exception as e:
            print(f"\nError evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: SC-SINDy Prefilter Improvements")
    print("=" * 60)

    for name, result in all_results["results"].items():
        print(f"\n{result['metadata']['name']}:")
        print(f"  Citation: {result['metadata']['citation']['text'][:60]}...")
        for method, data in result.get("methods", {}).items():
            if "improvement_pct" in data:
                print(f"  {method}: {data['improvement_pct']:+.1f}% improvement with SC-SINDy")

    print("\n" + "=" * 60)
    print("CITATIONS (for paper)")
    print("=" * 60)
    for name in dataset_names:
        if name in CITATIONS:
            print(f"\n{name}:")
            print(CITATIONS[name]["bibtex"])


if __name__ == "__main__":
    main()
