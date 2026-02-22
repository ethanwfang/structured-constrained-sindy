#!/usr/bin/env python
"""
Threshold Sensitivity Analysis for SC-SINDy

This script evaluates the sensitivity of SC-SINDy performance to the
structure_threshold parameter and provides guidelines for selection.

Usage:
    python scripts/threshold_sensitivity.py --model models/factorized/factorized_model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sc_sindy.core.sindy import sindy_stls
from src.sc_sindy.core.structure_constrained import sindy_structure_constrained
from src.sc_sindy.core.library import build_library_nd
from src.sc_sindy.derivatives import compute_derivatives_finite_diff

try:
    import torch
    from src.sc_sindy.network.factorized import FactorizedStructureNetworkV2
    from src.sc_sindy.network.factorized.trajectory_encoder import extract_per_variable_stats
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Test systems
TEST_SYSTEMS = [
    ("Lorenz", 3),
    ("VanDerPol", 2),
    ("DuffingOscillator", 2),
]


def get_system_class(name: str):
    """Get system class by name."""
    from src.sc_sindy.systems import (
        Lorenz, VanDerPol, DuffingOscillator,
    )
    systems = {
        "Lorenz": Lorenz,
        "VanDerPol": VanDerPol,
        "DuffingOscillator": DuffingOscillator,
    }
    return systems.get(name)


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_threshold(
    model,
    system,
    threshold: float,
    n_trials: int = 20,
    noise_level: float = 0.05,
    poly_order: int = 3,
) -> Dict:
    """Evaluate SC-SINDy at a specific threshold."""
    f1_scores = []
    precisions = []
    recalls = []

    n_vars = system.n_dims

    for trial in range(n_trials):
        try:
            # Generate data
            x0 = system.sample_initial_condition()
            t = np.linspace(0, 10, 500)
            X = system.generate_trajectory(x0, t, noise_level=noise_level)
            X_dot = compute_derivatives_finite_diff(X, t)

            # Build library
            Theta = build_library_nd(X, poly_order)

            # Get structure predictions
            stats = extract_per_variable_stats(X)
            with torch.no_grad():
                probs = model.predict_structure(
                    torch.tensor(stats, dtype=torch.float32).unsqueeze(0),
                    n_vars=n_vars,
                    poly_order=poly_order,
                )
            probs_np = probs.squeeze().numpy()

            # Apply threshold
            structure_pred = (probs_np >= threshold).astype(float)

            # Run STLS with structure constraint
            xi_pred = sindy_structure_constrained(
                Theta, X_dot, structure_pred, threshold=0.1
            )

            # Compare to ground truth
            xi_true = system.get_coefficients(poly_order)
            structure_true = (np.abs(xi_true) > 1e-10).astype(float)
            structure_discovered = (np.abs(xi_pred) > 1e-10).astype(float)

            # Compute metrics
            tp = np.sum(structure_discovered * structure_true)
            fp = np.sum(structure_discovered * (1 - structure_true))
            fn = np.sum((1 - structure_discovered) * structure_true)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = compute_f1(precision, recall)

            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        except Exception:
            continue

    if not f1_scores:
        return {"f1": 0, "precision": 0, "recall": 0, "n_valid": 0}

    return {
        "f1": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "n_valid": len(f1_scores),
    }


def run_threshold_sweep(
    model_path: str,
    thresholds: List[float] = None,
    n_trials: int = 20,
    output_path: Optional[str] = None,
) -> Dict:
    """Run threshold sensitivity analysis."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("=" * 60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load model
    model = FactorizedStructureNetworkV2.load(model_path)
    model.eval()
    print(f"Loaded model from {model_path}")

    results = {
        "thresholds": thresholds,
        "systems": {},
    }

    for system_name, n_dims in TEST_SYSTEMS:
        print(f"\n--- {system_name} ---")
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        system = system_cls()
        system_results = {"f1": [], "precision": [], "recall": []}

        for thresh in thresholds:
            print(f"  Threshold {thresh:.1f}: ", end="", flush=True)
            result = evaluate_threshold(model, system, thresh, n_trials)
            system_results["f1"].append(result["f1"])
            system_results["precision"].append(result["precision"])
            system_results["recall"].append(result["recall"])
            print(f"F1={result['f1']:.3f}, P={result['precision']:.3f}, R={result['recall']:.3f}")

        results["systems"][system_name] = system_results

    # Compute optimal threshold
    all_f1s = np.zeros(len(thresholds))
    for sys_results in results["systems"].values():
        all_f1s += np.array(sys_results["f1"])
    all_f1s /= len(results["systems"])
    optimal_idx = np.argmax(all_f1s)
    results["optimal_threshold"] = thresholds[optimal_idx]
    results["mean_f1_by_threshold"] = all_f1s.tolist()

    print(f"\nOptimal threshold: {results['optimal_threshold']:.1f}")
    print(f"Mean F1 at optimal: {all_f1s[optimal_idx]:.3f}")

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Threshold sensitivity analysis")
    parser.add_argument(
        "--model",
        type=str,
        default="models/factorized/factorized_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials per threshold",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/factorized/threshold_sensitivity.json",
        help="Output path for results",
    )
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for this script")
        sys.exit(1)

    run_threshold_sweep(
        model_path=args.model,
        n_trials=args.n_trials,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
