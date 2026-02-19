#!/usr/bin/env python
"""
Train the factorized structure network on diverse 2D, 3D, and 4D systems.

This script trains the dimension-agnostic factorized network using the
train/test splits defined in splits_factorized.py.

Usage:
    python scripts/train_factorized_full.py [OPTIONS]

Options:
    --epochs INT          Number of training epochs (default: 100)
    --latent-dim INT      Latent dimension size (default: 64)
    --trajectories INT    Trajectories per system (default: 50)
    --batch-size INT      Batch size (default: 32)
    --lr FLOAT            Learning rate (default: 0.001)
    --output PATH         Output directory for model (default: models/factorized)
    --no-eval             Skip evaluation after training
    --verbose             Print detailed progress
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sc_sindy.evaluation.splits_factorized import (
    get_factorized_train_systems,
    get_factorized_test_systems,
    print_split_summary,
)
from sc_sindy.network.factorized import (
    FactorizedStructureNetworkV2,
    generate_mixed_training_data,
    train_factorized_network,
    get_library_terms,
)
from sc_sindy.network.factorized.training import generate_training_sample


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    """Compute precision, recall, F1 for structure prediction."""
    pred_binary = predictions > threshold

    tp = np.sum(pred_binary & targets)
    fp = np.sum(pred_binary & ~targets)
    fn = np.sum(~pred_binary & targets)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_on_systems(model, systems_by_dim, poly_order=3, n_trajectories=10, verbose=True):
    """
    Evaluate model on a set of systems.

    Returns metrics per system and aggregated by dimension.
    """
    import torch
    from sc_sindy.network.factorized.trajectory_encoder import extract_per_variable_stats

    model.eval()
    results = {}

    for dim, system_classes in systems_by_dim.items():
        if not system_classes:
            continue

        dim_results = []

        for system_cls in system_classes:
            system_name = system_cls.__name__
            system_metrics = []

            try:
                system = system_cls()
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not instantiate {system_name}: {e}")
                continue

            # Generate test trajectories
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

                # Compute metrics
                metrics = compute_metrics(predictions, sample.structure.astype(bool))
                system_metrics.append(metrics)

            if system_metrics:
                avg_metrics = {
                    "precision": np.mean([m["precision"] for m in system_metrics]),
                    "recall": np.mean([m["recall"] for m in system_metrics]),
                    "f1": np.mean([m["f1"] for m in system_metrics]),
                    "n_samples": len(system_metrics),
                }
                dim_results.append((system_name, avg_metrics))

                if verbose:
                    print(f"  {system_name}: F1={avg_metrics['f1']:.3f} "
                          f"(P={avg_metrics['precision']:.3f}, R={avg_metrics['recall']:.3f})")

        results[dim] = dim_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Train factorized structure network")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--trajectories", type=int, default=50, help="Trajectories per system")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--poly-order", type=int, default=3, help="Polynomial order")
    parser.add_argument("--output", type=str, default="models/factorized", help="Output directory")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("FACTORIZED STRUCTURE NETWORK TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Print split summary
    print_split_summary()
    print()

    # Get train systems
    train_systems = get_factorized_train_systems()
    test_systems = get_factorized_test_systems()

    total_train = sum(len(v) for v in train_systems.values())
    total_test = sum(len(v) for v in test_systems.values())

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Trajectories/system: {args.trajectories}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Poly order: {args.poly_order}")
    print(f"  Train systems: {total_train}")
    print(f"  Test systems: {total_test}")
    print()

    # Generate training data
    print("Generating training data...")
    samples = generate_mixed_training_data(
        systems_by_dim=train_systems,
        n_trajectories_per_system=args.trajectories,
        poly_order=args.poly_order,
        noise_levels=[0.0, 0.05, 0.10],
    )

    if len(samples) == 0:
        print("ERROR: No training samples generated!")
        sys.exit(1)

    # Count samples by dimension
    samples_by_dim = {}
    for s in samples:
        if s.n_vars not in samples_by_dim:
            samples_by_dim[s.n_vars] = 0
        samples_by_dim[s.n_vars] += 1

    print(f"\nSamples by dimension:")
    for dim in sorted(samples_by_dim.keys()):
        print(f"  {dim}D: {samples_by_dim[dim]} samples")
    print()

    # Train model
    print("Training model...")
    print("-" * 40)

    model, history = train_factorized_network(
        samples=samples,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=0.1,
        early_stopping_patience=15,
        use_v2=True,
        verbose=True,
    )

    print("-" * 40)
    print(f"Training complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")
    print()

    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    model_path = output_dir / "factorized_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "poly_order": args.poly_order,
        "history": history,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "trajectories_per_system": args.trajectories,
            "train_systems": {
                str(dim): [s.__name__ for s in systems]
                for dim, systems in train_systems.items()
            },
            "n_train_samples": len(samples),
            "samples_by_dim": samples_by_dim,
        },
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Evaluation
    if not args.no_eval:
        print()
        print("=" * 60)
        print("EVALUATION ON TEST SYSTEMS")
        print("=" * 60)
        print("(These systems were NEVER seen during training)")
        print()

        all_results = {}

        for dim in sorted(test_systems.keys()):
            if not test_systems[dim]:
                continue
            print(f"\n{dim}D Test Systems:")
            print("-" * 30)

            results = evaluate_on_systems(
                model,
                {dim: test_systems[dim]},
                poly_order=args.poly_order,
                n_trajectories=20,
                verbose=True,
            )

            if dim in results:
                all_results[dim] = results[dim]

        # Summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for dim in sorted(all_results.keys()):
            f1_scores = [m["f1"] for _, m in all_results[dim]]
            if f1_scores:
                avg_f1 = np.mean(f1_scores)
                print(f"{dim}D: Avg F1 = {avg_f1:.3f} (n={len(f1_scores)} systems)")

        # Overall
        all_f1 = []
        for dim_results in all_results.values():
            all_f1.extend([m["f1"] for _, m in dim_results])

        if all_f1:
            print(f"\nOverall Avg F1: {np.mean(all_f1):.3f}")

        # Save results
        results_path = output_dir / "evaluation_results.json"
        serializable_results = {
            str(dim): [(name, metrics) for name, metrics in dim_results]
            for dim, dim_results in all_results.items()
        }
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
