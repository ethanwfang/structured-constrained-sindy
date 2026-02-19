#!/usr/bin/env python
"""
Evaluate the factorized structure network on test and held-out systems.

This script loads a trained factorized model and evaluates it on:
1. Test systems (unseen during training)
2. Held-out systems (completely excluded from train/test)

Usage:
    python scripts/evaluate_factorized.py [OPTIONS]

Options:
    --model PATH          Path to trained model (default: models/factorized/factorized_model.pt)
    --trajectories INT    Trajectories per system for evaluation (default: 20)
    --include-heldout     Also evaluate on held-out systems
    --output PATH         Output file for results (optional)
    --threshold FLOAT     Classification threshold (default: 0.5)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from sc_sindy.evaluation.splits_factorized import (
    get_factorized_test_systems,
    get_factorized_heldout_systems,
    get_factorized_train_systems,
)
from sc_sindy.network.factorized import (
    FactorizedStructureNetworkV2,
    get_library_terms,
)
from sc_sindy.network.factorized.training import generate_training_sample


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    """Compute precision, recall, F1 for structure prediction."""
    pred_binary = predictions > threshold

    tp = np.sum(pred_binary & targets)
    fp = np.sum(pred_binary & ~targets)
    fn = np.sum(~pred_binary & targets)
    tn = np.sum(~pred_binary & ~targets)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_system(model, system_cls, poly_order=3, n_trajectories=20, threshold=0.5):
    """Evaluate model on a single system."""
    try:
        system = system_cls()
    except Exception as e:
        return None, f"Could not instantiate: {e}"

    dim = system.dim
    all_metrics = []
    term_names = get_library_terms(dim, poly_order)

    for _ in range(n_trajectories):
        sample = generate_training_sample(
            system=system,
            poly_order=poly_order,
            t_span=(0, 50),
            n_points=5000,
            noise_level=0.05,
        )

        if sample is None:
            continue

        # Get predictions
        with torch.no_grad():
            stats_tensor = torch.FloatTensor(sample.stats).unsqueeze(0)
            probs = model.forward(stats_tensor, dim, poly_order)
            if probs.dim() == 3:
                probs = probs.squeeze(0)
            predictions = probs.numpy()

        metrics = compute_metrics(predictions, sample.structure.astype(bool), threshold)
        all_metrics.append(metrics)

    if not all_metrics:
        return None, "No valid trajectories generated"

    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in ["precision", "recall", "f1", "accuracy"]
    }
    avg_metrics["n_samples"] = len(all_metrics)

    return avg_metrics, None


def print_evaluation_table(results_by_dim, title):
    """Print a formatted table of results."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'System':<30} {'Dim':>4} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    print("-" * 70)

    all_f1 = []

    for dim in sorted(results_by_dim.keys()):
        for system_name, metrics in results_by_dim[dim]:
            if metrics is not None:
                print(f"{system_name:<30} {dim:>4} "
                      f"{metrics['f1']:>8.3f} "
                      f"{metrics['precision']:>8.3f} "
                      f"{metrics['recall']:>8.3f} "
                      f"{metrics['accuracy']:>8.3f}")
                all_f1.append(metrics["f1"])
            else:
                print(f"{system_name:<30} {dim:>4} {'FAILED':>8}")

    print("-" * 70)

    # Per-dimension averages
    for dim in sorted(results_by_dim.keys()):
        dim_f1 = [m["f1"] for _, m in results_by_dim[dim] if m is not None]
        if dim_f1:
            print(f"{f'{dim}D Average':<30} {dim:>4} {np.mean(dim_f1):>8.3f}")

    if all_f1:
        print(f"{'Overall Average':<30} {'':>4} {np.mean(all_f1):>8.3f}")

    print("=" * 70)

    return all_f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate factorized structure network")
    parser.add_argument("--model", type=str, default="models/factorized/factorized_model.pt",
                        help="Path to trained model")
    parser.add_argument("--trajectories", type=int, default=20,
                        help="Trajectories per system")
    parser.add_argument("--include-heldout", action="store_true",
                        help="Also evaluate on held-out systems")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    parser.add_argument("--poly-order", type=int, default=None,
                        help="Polynomial order (default: from model)")
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run train_factorized_full.py first to train a model.")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")

    latent_dim = checkpoint.get("latent_dim", 64)
    poly_order = args.poly_order or checkpoint.get("poly_order", 3)

    model = FactorizedStructureNetworkV2(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded (latent_dim={latent_dim}, poly_order={poly_order})")

    # Print training info if available
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"\nModel trained with:")
        print(f"  Epochs: {config.get('epochs', 'N/A')}")
        print(f"  Train samples: {config.get('n_train_samples', 'N/A')}")

        if "train_systems" in config:
            for dim, systems in config["train_systems"].items():
                print(f"  {dim}D train systems: {len(systems)}")

    # Get test and held-out systems
    test_systems = get_factorized_test_systems()
    heldout_systems = get_factorized_heldout_systems()

    # Evaluate on test systems
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SYSTEMS")
    print("(These systems have distinct structures from training)")
    print("=" * 70)

    test_results = {}
    for dim in sorted(test_systems.keys()):
        if not test_systems[dim]:
            continue

        print(f"\nEvaluating {dim}D test systems...")
        dim_results = []

        for system_cls in test_systems[dim]:
            system_name = system_cls.__name__
            print(f"  {system_name}...", end=" ", flush=True)

            metrics, error = evaluate_system(
                model, system_cls,
                poly_order=poly_order,
                n_trajectories=args.trajectories,
                threshold=args.threshold,
            )

            if metrics:
                print(f"F1={metrics['f1']:.3f}")
                dim_results.append((system_name, metrics))
            else:
                print(f"FAILED: {error}")
                dim_results.append((system_name, None))

        test_results[dim] = dim_results

    test_f1 = print_evaluation_table(test_results, "TEST SYSTEMS RESULTS")

    # Evaluate on held-out systems
    heldout_results = {}
    if args.include_heldout:
        has_heldout = any(len(v) > 0 for v in heldout_systems.values())
        if has_heldout:
            print("\n" + "=" * 70)
            print("EVALUATING ON HELD-OUT SYSTEMS")
            print("(Completely excluded from training and testing)")
            print("=" * 70)

            for dim in sorted(heldout_systems.keys()):
                if not heldout_systems[dim]:
                    continue

                print(f"\nEvaluating {dim}D held-out systems...")
                dim_results = []

                for system_cls in heldout_systems[dim]:
                    system_name = system_cls.__name__
                    print(f"  {system_name}...", end=" ", flush=True)

                    metrics, error = evaluate_system(
                        model, system_cls,
                        poly_order=poly_order,
                        n_trajectories=args.trajectories,
                        threshold=args.threshold,
                    )

                    if metrics:
                        print(f"F1={metrics['f1']:.3f}")
                        dim_results.append((system_name, metrics))
                    else:
                        print(f"FAILED: {error}")
                        dim_results.append((system_name, None))

                heldout_results[dim] = dim_results

            heldout_f1 = print_evaluation_table(heldout_results, "HELD-OUT SYSTEMS RESULTS")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if test_f1:
        print(f"Test Systems:     Avg F1 = {np.mean(test_f1):.3f} (n={len(test_f1)} systems)")

    if heldout_results:
        heldout_f1_list = []
        for dim_results in heldout_results.values():
            heldout_f1_list.extend([m["f1"] for _, m in dim_results if m is not None])
        if heldout_f1_list:
            print(f"Held-out Systems: Avg F1 = {np.mean(heldout_f1_list):.3f} (n={len(heldout_f1_list)} systems)")

    # Highlight key benchmark results
    print("\nKey Benchmarks:")
    for dim, results in test_results.items():
        for name, metrics in results:
            if name == "Lorenz" and metrics:
                print(f"  Lorenz (never trained): F1 = {metrics['f1']:.3f}")
            if name == "SIRModel" and metrics:
                print(f"  SIR Model: F1 = {metrics['f1']:.3f}")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results = {
            "test_results": {
                str(dim): [(name, metrics) for name, metrics in results]
                for dim, results in test_results.items()
            },
            "heldout_results": {
                str(dim): [(name, metrics) for name, metrics in results]
                for dim, results in heldout_results.items()
            } if heldout_results else {},
            "config": {
                "model_path": str(model_path),
                "poly_order": poly_order,
                "threshold": args.threshold,
                "n_trajectories": args.trajectories,
            },
        }

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
