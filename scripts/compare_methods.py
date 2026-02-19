"""
Method comparison script: SC-SINDy vs E-SINDy vs Standard SINDy.

This script compares structure prediction accuracy across different methods:
1. SC-SINDy (Factorized Network): Our neural network approach
2. E-SINDy: Ensemble SINDy with library bagging
3. Standard SINDy: Single SINDy fit

Key framing: SC-SINDy acts as a preprocessing step that improves upon
E-SINDy and standard SINDy by providing better structure predictions
before STLS refinement.

Usage:
    python scripts/compare_methods.py --n_traj 20
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
except ImportError:
    torch = None

from src.sc_sindy.evaluation.splits_factorized import (
    TEST_SYSTEMS_2D_FACTORIZED as TEST_SYSTEMS_2D,
    TEST_SYSTEMS_3D_FACTORIZED as TEST_SYSTEMS_3D,
    TEST_SYSTEMS_4D_FACTORIZED as TEST_SYSTEMS_4D,
)
from src.sc_sindy.network.factorized.term_representation import get_library_terms


def evaluate_method(
    method_name: str,
    predict_fn,
    test_systems: List,
    n_trajectories: int = 20,
    poly_order: int = 3,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a structure prediction method on test systems.

    Parameters
    ----------
    method_name : str
        Name of the method for logging.
    predict_fn : callable
        Function that takes (X, t) and returns structure probabilities.
    test_systems : List
        List of system classes to evaluate on.
    n_trajectories : int
        Number of trajectories per system.
    poly_order : int
        Maximum polynomial order.
    threshold : float
        Probability threshold for structure prediction.
    verbose : bool
        Print progress.

    Returns
    -------
    results : Dict
        Evaluation results.
    """
    results = []

    for system_cls in test_systems:
        try:
            system = system_cls()
        except Exception as e:
            if verbose:
                print(f"  Skipping {system_cls.__name__}: {e}")
            continue

        n_vars = system.dim
        term_names = get_library_terms(n_vars, poly_order)
        true_structure = system.get_true_structure(term_names)

        precisions, recalls, f1s = [], [], []

        for traj_idx in range(n_trajectories):
            try:
                x0 = np.random.randn(n_vars) * 2
                t = np.linspace(0, 50, 5000)
                trajectory = system.generate_trajectory(x0, t, noise_level=0.05)

                # Trim transients
                trim = 100
                trajectory = trajectory[trim:-trim]
                t = t[trim:-trim]

                if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
                    continue

                # Get predictions
                probs = predict_fn(trajectory, t)

                if probs is None or np.any(np.isnan(probs)):
                    continue

                pred_structure = (probs > threshold).astype(float)

                # Compute metrics
                true_pos = (pred_structure * true_structure).sum()
                pred_pos = pred_structure.sum()
                actual_pos = true_structure.sum()

                precision = true_pos / pred_pos if pred_pos > 0 else 0
                recall = true_pos / actual_pos if actual_pos > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            except Exception as e:
                continue

        if len(f1s) > 0:
            results.append({
                "system": system_cls.__name__,
                "dim": n_vars,
                "precision": float(np.mean(precisions)),
                "recall": float(np.mean(recalls)),
                "f1": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "n_samples": len(f1s),
            })
            if verbose:
                print(f"  {system_cls.__name__}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Aggregate
    all_f1s = [r["f1"] for r in results]
    all_precisions = [r["precision"] for r in results]
    all_recalls = [r["recall"] for r in results]

    return {
        "method": method_name,
        "per_system": results,
        "mean_f1": float(np.mean(all_f1s)) if all_f1s else 0,
        "std_f1": float(np.std(all_f1s)) if all_f1s else 0,
        "mean_precision": float(np.mean(all_precisions)) if all_precisions else 0,
        "mean_recall": float(np.mean(all_recalls)) if all_recalls else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare structure prediction methods")
    parser.add_argument("--n_traj", type=int, default=20, help="Trajectories per test system")
    parser.add_argument("--poly_order", type=int, default=3, help="Maximum polynomial order")
    parser.add_argument("--threshold", type=float, default=0.5, help="Structure threshold")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained SC-SINDy model")
    parser.add_argument("--n_bootstraps", type=int, default=50, help="E-SINDy bootstrap iterations")
    parser.add_argument("--output_dir", type=str, default="models/factorized",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("METHOD COMPARISON: SC-SINDy vs E-SINDy vs Standard SINDy")
    print("=" * 70)
    print(f"Trajectories per system: {args.n_traj}")
    print(f"Polynomial order: {args.poly_order}")
    print(f"Structure threshold: {args.threshold}")

    # Combine all test systems
    all_test_systems = TEST_SYSTEMS_2D + TEST_SYSTEMS_3D + TEST_SYSTEMS_4D

    all_results = []

    # Method 1: SC-SINDy (Factorized Network)
    print("\n" + "=" * 70)
    print("Evaluating SC-SINDy (Factorized Network)...")
    print("=" * 70)

    if torch is not None:
        from src.sc_sindy.network.factorized.factorized_network import FactorizedStructureNetworkV2

        # Load or create model
        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading model from {args.model_path}")
            model = FactorizedStructureNetworkV2.load(args.model_path)
        else:
            print("No model path provided or model not found.")
            print("Using untrained model (results will be poor).")
            model = FactorizedStructureNetworkV2(latent_dim=64, use_relative_eq_encoder=True)

        model.eval()

        def sc_sindy_predict(X, t):
            return model.predict_structure(X, poly_order=args.poly_order)

        sc_sindy_results = evaluate_method(
            method_name="SC-SINDy",
            predict_fn=sc_sindy_predict,
            test_systems=all_test_systems,
            n_trajectories=args.n_traj,
            poly_order=args.poly_order,
            threshold=args.threshold,
        )
        all_results.append(sc_sindy_results)
    else:
        print("PyTorch not available, skipping SC-SINDy.")

    # Method 2: E-SINDy
    print("\n" + "=" * 70)
    print("Evaluating E-SINDy (Ensemble SINDy)...")
    print("=" * 70)

    try:
        from src.sc_sindy.baselines.e_sindy import ESINDyBaseline

        e_sindy = ESINDyBaseline(
            n_bootstraps=args.n_bootstraps,
            poly_order=args.poly_order,
        )

        def e_sindy_predict(X, t):
            return e_sindy.predict_structure(X, t=t)

        e_sindy_results = evaluate_method(
            method_name="E-SINDy",
            predict_fn=e_sindy_predict,
            test_systems=all_test_systems,
            n_trajectories=args.n_traj,
            poly_order=args.poly_order,
            threshold=args.threshold,
        )
        all_results.append(e_sindy_results)

    except ImportError as e:
        print(f"E-SINDy not available: {e}")

    # Method 3: Standard SINDy
    print("\n" + "=" * 70)
    print("Evaluating Standard SINDy...")
    print("=" * 70)

    try:
        from src.sc_sindy.baselines.e_sindy import StandardSINDyBaseline

        std_sindy = StandardSINDyBaseline(
            poly_order=args.poly_order,
        )

        def std_sindy_predict(X, t):
            # Standard SINDy returns binary, not probabilities
            # We treat non-zero coefficients as probability 1.0
            return std_sindy.predict_structure(X, t=t)

        std_sindy_results = evaluate_method(
            method_name="Standard SINDy",
            predict_fn=std_sindy_predict,
            test_systems=all_test_systems,
            n_trajectories=args.n_traj,
            poly_order=args.poly_order,
            threshold=0.5,  # Binary, so threshold doesn't matter
        )
        all_results.append(std_sindy_results)

    except ImportError as e:
        print(f"Standard SINDy not available: {e}")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'F1':>10} {'Precision':>12} {'Recall':>10}")
    print("-" * 70)

    for result in sorted(all_results, key=lambda x: x["mean_f1"], reverse=True):
        print(f"{result['method']:<20} {result['mean_f1']:>10.4f} "
              f"{result['mean_precision']:>12.4f} {result['mean_recall']:>10.4f}")

    # Compute improvement over baselines
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)

    sc_sindy_f1 = next((r["mean_f1"] for r in all_results if r["method"] == "SC-SINDy"), None)

    for result in all_results:
        if result["method"] != "SC-SINDy" and sc_sindy_f1 is not None:
            baseline_f1 = result["mean_f1"]
            if baseline_f1 > 0:
                improvement = (sc_sindy_f1 - baseline_f1) / baseline_f1 * 100
                sign = "+" if improvement > 0 else ""
                print(f"SC-SINDy vs {result['method']}: {sign}{improvement:.1f}%")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"method_comparison_{timestamp}.json")

    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
