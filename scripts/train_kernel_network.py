#!/usr/bin/env python
"""
Train and evaluate the kernel-based structure network.

This script trains the dimension-agnostic kernel network using the same
train/test splits as the factorized network for fair comparison.

Usage:
    python scripts/train_kernel_network.py [OPTIONS]

Options:
    --epochs INT          Number of training epochs (default: 100)
    --embed-dim INT       Embedding dimension (default: 64)
    --kernel-type STR     Kernel type: linear, polynomial, rbf, neural, bilinear (default: neural)
    --trajectories INT    Trajectories per system (default: 30)
    --batch-size INT      Batch size (default: 32)
    --lr FLOAT            Learning rate (default: 0.001)
    --output PATH         Output directory for model (default: models/kernel)
    --no-eval             Skip evaluation after training
    --compare             Compare with factorized network results
"""

import argparse
import json
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
from sc_sindy.network.kernel import (
    KernelStructureNetwork,
    KernelType,
    generate_kernel_training_data,
    train_kernel_network,
    evaluate_kernel_network,
)


def parse_kernel_type(s: str) -> KernelType:
    """Parse kernel type string to enum."""
    mapping = {
        "linear": KernelType.LINEAR,
        "polynomial": KernelType.POLYNOMIAL,
        "rbf": KernelType.RBF,
        "neural": KernelType.NEURAL,
        "bilinear": KernelType.BILINEAR,
    }
    if s.lower() not in mapping:
        raise ValueError(f"Unknown kernel type: {s}. Choose from {list(mapping.keys())}")
    return mapping[s.lower()]


def main():
    parser = argparse.ArgumentParser(description="Train kernel structure network")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--kernel-type", type=str, default="neural",
                        choices=["linear", "polynomial", "rbf", "neural", "bilinear"],
                        help="Type of kernel function")
    parser.add_argument("--trajectories", type=int, default=30, help="Trajectories per system")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--poly-order", type=int, default=3, help="Polynomial order")
    parser.add_argument("--output", type=str, default="models/kernel", help="Output directory")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--compare", action="store_true", help="Compare with factorized results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    kernel_type = parse_kernel_type(args.kernel_type)

    print("=" * 60)
    print("KERNEL STRUCTURE NETWORK TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Kernel type: {kernel_type.value}")
    print()

    # Print split summary
    print_split_summary()
    print()

    # Get train/test systems
    train_systems = get_factorized_train_systems()
    test_systems = get_factorized_test_systems()

    total_train = sum(len(v) for v in train_systems.values())
    total_test = sum(len(v) for v in test_systems.values())

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Embedding dim: {args.embed_dim}")
    print(f"  Kernel type: {kernel_type.value}")
    print(f"  Trajectories/system: {args.trajectories}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Poly order: {args.poly_order}")
    print(f"  Train systems: {total_train}")
    print(f"  Test systems: {total_test}")
    print()

    # Generate training data
    print("Generating training data...")
    samples = generate_kernel_training_data(
        systems_by_dim=train_systems,
        n_trajectories_per_system=args.trajectories,
        poly_order=args.poly_order,
        noise_levels=[0.0, 0.05, 0.10],
        verbose=True,
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
    print("Training kernel network...")
    print("-" * 40)

    model, history = train_kernel_network(
        samples=samples,
        embed_dim=args.embed_dim,
        kernel_type=kernel_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=0.1,
        early_stopping_patience=15,
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
    model_path = output_dir / f"kernel_model_{kernel_type.value}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "embed_dim": args.embed_dim,
        "kernel_type": kernel_type.value,
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

            results = evaluate_kernel_network(
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

        # Overall average
        all_f1 = []
        for dim_results in all_results.values():
            all_f1.extend([m["f1"] for _, m in dim_results])

        if all_f1:
            print(f"\nOverall Avg F1: {np.mean(all_f1):.3f}")

        # Save results
        results_path = output_dir / f"evaluation_results_{kernel_type.value}.json"
        serializable_results = {
            str(dim): [(name, metrics) for name, metrics in dim_results]
            for dim, dim_results in all_results.items()
        }
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Compare with factorized if requested
        if args.compare:
            factorized_results_path = Path("models/factorized/evaluation_results.json")
            if factorized_results_path.exists():
                print()
                print("=" * 60)
                print("COMPARISON WITH FACTORIZED NETWORK")
                print("=" * 60)

                with open(factorized_results_path) as f:
                    factorized_results = json.load(f)

                print(f"\n{'System':<25} {'Kernel F1':>12} {'Factorized F1':>15} {'Diff':>10}")
                print("-" * 65)

                for dim in sorted(all_results.keys()):
                    kernel_systems = {name: m for name, m in all_results[dim]}
                    factorized_systems = {name: m for name, m in factorized_results.get(str(dim), [])}

                    for system_name in kernel_systems:
                        k_f1 = kernel_systems[system_name]["f1"]
                        f_f1 = factorized_systems.get(system_name, {}).get("f1", None)

                        if f_f1 is not None:
                            diff = k_f1 - f_f1
                            diff_str = f"{diff:+.3f}"
                        else:
                            diff_str = "N/A"
                            f_f1 = 0

                        print(f"{system_name:<25} {k_f1:>12.3f} {f_f1:>15.3f} {diff_str:>10}")
            else:
                print("\nNo factorized results found for comparison.")
                print(f"Expected: {factorized_results_path}")

    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
