#!/usr/bin/env python3
"""
Learning Curves Analysis for SC-SINDy.

This script generates learning curves showing how model performance (F1 score)
varies with training set size. This addresses reviewer concerns about:
1. Sample efficiency of the approach
2. Whether more training data would improve results
3. Convergence of the learning process

Usage:
    python scripts/plot_learning_curves.py --model models/factorized/factorized_model.pt
    python scripts/plot_learning_curves.py --retrain  # Retrains models at each size
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sc_sindy.systems.registry import get_system_class
from src.sc_sindy.core.sindy import sindy_stls
from src.sc_sindy.core.library import build_library_2d, build_library_3d, build_library_nd
from src.sc_sindy.derivatives.finite_difference import compute_derivatives_finite_diff
from src.sc_sindy.metrics.structure import compute_structure_metrics
from src.sc_sindy.evaluation.splits_factorized import (
    TEST_SYSTEMS_2D_FACTORIZED,
    TEST_SYSTEMS_3D_FACTORIZED,
)

# Try to import PyTorch and model
try:
    import torch
    from src.sc_sindy.network.factorized import FactorizedStructureNetworkV2
    from src.sc_sindy.network.factorized.training import train_factorized_network
    from src.sc_sindy.network.factorized.data_generation import (
        generate_training_samples,
        create_training_dataloader,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Training sizes to evaluate
DEFAULT_TRAINING_SIZES = [100, 250, 500, 1000, 2500, 5000]

# Test systems for evaluation
TEST_SYSTEMS = ["Lorenz", "VanDerPol", "LotkaVolterra"]


def build_library_for_dim(X: np.ndarray, poly_order: int = 3):
    """Build appropriate library based on dimensionality."""
    n_vars = X.shape[1]
    if n_vars == 2:
        return build_library_2d(X, poly_order=poly_order)
    elif n_vars == 3:
        return build_library_3d(X, poly_order=poly_order)
    else:
        return build_library_nd(X, poly_order=poly_order)


def evaluate_model_on_test_systems(
    model,
    test_systems: List[str],
    n_trials: int = 20,
    noise_level: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a trained model on test systems.

    Parameters
    ----------
    model : FactorizedStructureNetworkV2
        Trained model.
    test_systems : List[str]
        Systems to evaluate on.
    n_trials : int
        Trials per system.
    noise_level : float
        Noise level for evaluation.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per-system F1 scores with mean and std.
    """
    from src.sc_sindy.core.structure_constrained import sindy_structure_constrained

    results = {}
    model.eval()

    for system_name in test_systems:
        system_cls = get_system_class(system_name)
        if system_cls is None:
            continue

        try:
            system = system_cls()
        except Exception:
            continue

        f1_scores = []

        for trial in range(n_trials):
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

            # Get network predictions
            with torch.no_grad():
                probs = model.predict_structure(X, n_vars, 3)

            # Run SC-SINDy
            xi_pred, _ = sindy_structure_constrained(Theta, X_dot, probs)
            structure_pred = (np.abs(xi_pred) > 1e-10).astype(float)

            metrics = compute_structure_metrics(structure_pred, structure_true)
            f1_scores.append(metrics["f1"])

        if f1_scores:
            results[system_name] = {
                "mean": float(np.mean(f1_scores)),
                "std": float(np.std(f1_scores)),
                "n_trials": len(f1_scores),
            }

    return results


def compute_learning_curve_with_retraining(
    training_sizes: List[int] = None,
    test_systems: List[str] = None,
    n_seeds: int = 3,
    n_eval_trials: int = 20,
    noise_level: float = 0.05,
    device: str = "cpu",
) -> Dict:
    """Compute learning curves by retraining at each size.

    This is the proper way to generate learning curves: train separate models
    at each training set size and evaluate on the same test set.

    Parameters
    ----------
    training_sizes : List[int]
        Training set sizes to evaluate.
    test_systems : List[str]
        Test systems for evaluation.
    n_seeds : int
        Random seeds per training size (for error bars).
    n_eval_trials : int
        Evaluation trials per system.
    noise_level : float
        Noise level for evaluation.
    device : str
        Device for training.

    Returns
    -------
    Dict
        Learning curve data.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for retraining")

    if training_sizes is None:
        training_sizes = DEFAULT_TRAINING_SIZES
    if test_systems is None:
        test_systems = TEST_SYSTEMS

    print("\n" + "=" * 70)
    print("LEARNING CURVES (with retraining)")
    print("=" * 70)
    print(f"Training sizes: {training_sizes}")
    print(f"Test systems: {test_systems}")
    print(f"Seeds per size: {n_seeds}")

    results = {
        "training_sizes": training_sizes,
        "test_systems": test_systems,
        "per_size": {},
    }

    for size in training_sizes:
        print(f"\n--- Training size: {size} ---")

        size_results = {
            "f1_per_seed": [],
            "f1_per_system": {sys: [] for sys in test_systems},
        }

        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}: ", end="", flush=True)

            # Generate training data
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Create a small model for this training size
            model = FactorizedStructureNetworkV2(
                traj_hidden_dims=[64, 32],
                term_embed_dim=32,
                attention_dim=64,
                output_hidden_dims=[64, 32],
            )

            try:
                # Generate training samples (simplified - using a subset of systems)
                train_samples = generate_training_samples(
                    n_samples=size,
                    noise_levels=[0.0, 0.01, 0.05],
                    poly_orders=[2, 3],
                )

                # Create dataloader
                dataloader = create_training_dataloader(train_samples, batch_size=min(32, size // 4))

                # Train (reduced epochs for speed)
                n_epochs = min(50, max(10, size // 100))
                history = train_factorized_network(
                    model,
                    dataloader,
                    n_epochs=n_epochs,
                    learning_rate=1e-3,
                    device=device,
                    verbose=False,
                )

                # Evaluate
                eval_results = evaluate_model_on_test_systems(
                    model, test_systems, n_eval_trials, noise_level
                )

                # Aggregate F1 across systems
                f1_values = [r["mean"] for r in eval_results.values() if "mean" in r]
                avg_f1 = np.mean(f1_values) if f1_values else 0.0

                size_results["f1_per_seed"].append(avg_f1)
                for sys_name, sys_result in eval_results.items():
                    if sys_name in size_results["f1_per_system"]:
                        size_results["f1_per_system"][sys_name].append(sys_result.get("mean", 0))

                print(f"Avg F1 = {avg_f1:.3f}")

            except Exception as e:
                print(f"Training failed: {e}")
                size_results["f1_per_seed"].append(0.0)

        # Aggregate across seeds
        results["per_size"][size] = {
            "f1_mean": float(np.mean(size_results["f1_per_seed"])),
            "f1_std": float(np.std(size_results["f1_per_seed"])),
            "per_system": {
                sys: {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std": float(np.std(vals)) if vals else 0.0,
                }
                for sys, vals in size_results["f1_per_system"].items()
            },
        }

    # Print summary table
    print("\n" + "=" * 70)
    print("LEARNING CURVE SUMMARY")
    print("=" * 70)
    print(f"{'Size':>8} {'F1 Mean':>10} {'F1 Std':>10}")
    print("-" * 30)
    for size in training_sizes:
        data = results["per_size"].get(size, {})
        print(f"{size:>8} {data.get('f1_mean', 0):>10.3f} {data.get('f1_std', 0):>10.3f}")

    return results


def compute_learning_curve_from_saved_models(
    model_paths: Dict[int, str],
    test_systems: List[str] = None,
    n_eval_trials: int = 20,
    noise_level: float = 0.05,
) -> Dict:
    """Compute learning curves from pre-saved models at different sizes.

    Parameters
    ----------
    model_paths : Dict[int, str]
        Mapping from training size to model path.
    test_systems : List[str]
        Test systems for evaluation.
    n_eval_trials : int
        Evaluation trials per system.
    noise_level : float
        Noise level for evaluation.

    Returns
    -------
    Dict
        Learning curve data.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    if test_systems is None:
        test_systems = TEST_SYSTEMS

    print("\n" + "=" * 70)
    print("LEARNING CURVES (from saved models)")
    print("=" * 70)

    results = {
        "training_sizes": sorted(model_paths.keys()),
        "test_systems": test_systems,
        "per_size": {},
    }

    for size, model_path in sorted(model_paths.items()):
        print(f"\nEvaluating model trained on {size} samples...")

        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue

        model = FactorizedStructureNetworkV2.load(model_path)
        eval_results = evaluate_model_on_test_systems(
            model, test_systems, n_eval_trials, noise_level
        )

        f1_values = [r["mean"] for r in eval_results.values() if "mean" in r]
        avg_f1 = np.mean(f1_values) if f1_values else 0.0

        results["per_size"][size] = {
            "f1_mean": avg_f1,
            "f1_std": 0.0,  # Single model, no std across seeds
            "per_system": eval_results,
        }

        print(f"  Avg F1 = {avg_f1:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Learning Curves for SC-SINDy")
    parser.add_argument("--retrain", action="store_true",
                       help="Retrain models at each size (slow but proper)")
    parser.add_argument("--model", type=str, default="models/factorized/factorized_model.pt",
                       help="Path to pre-trained model (for single-model evaluation)")
    parser.add_argument("--sizes", type=str, default="100,250,500,1000,2500,5000",
                       help="Comma-separated training sizes")
    parser.add_argument("--n-seeds", type=int, default=3,
                       help="Number of random seeds per training size")
    parser.add_argument("--n-eval-trials", type=int, default=20,
                       help="Evaluation trials per system")
    parser.add_argument("--output", type=str, default="models/factorized/learning_curves.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for training (cpu/cuda)")
    args = parser.parse_args()

    # Parse training sizes
    training_sizes = [int(s.strip()) for s in args.sizes.split(",")]

    if args.retrain:
        # Full learning curve with retraining
        results = compute_learning_curve_with_retraining(
            training_sizes=training_sizes,
            n_seeds=args.n_seeds,
            n_eval_trials=args.n_eval_trials,
            device=args.device,
        )
    else:
        # Single model evaluation (just demonstrates capability)
        if not os.path.exists(args.model):
            print(f"Model not found: {args.model}")
            print("Use --retrain to train models at each size")
            return

        model = FactorizedStructureNetworkV2.load(args.model)
        eval_results = evaluate_model_on_test_systems(
            model, TEST_SYSTEMS, args.n_eval_trials, noise_level=0.05
        )

        results = {
            "note": "Single model evaluation (use --retrain for proper learning curves)",
            "model_path": args.model,
            "evaluation": eval_results,
        }

        print("\nSingle Model Evaluation:")
        for sys_name, sys_result in eval_results.items():
            print(f"  {sys_name}: F1 = {sys_result['mean']:.3f} +/- {sys_result['std']:.3f}")

    # Save results
    results["timestamp"] = datetime.now().isoformat()
    results["config"] = vars(args)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
