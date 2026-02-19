"""
Zero-shot evaluation experiments for factorized structure network.

This script tests the network's ability to generalize to unseen dimensions:
- Experiment A: Train on 2D only → Test on 3D/4D
- Experiment B: Train on 2D+3D → Test on 4D
- Experiment C: Train on 3D only → Test on 2D/4D

This validates the claim that the factorized architecture enables
dimension-agnostic structure prediction.

Usage:
    python scripts/evaluate_zero_shot.py --epochs 50 --n_traj 30
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")

from src.sc_sindy.network.factorized.training import (
    generate_mixed_training_data,
    train_factorized_network,
)
from src.sc_sindy.network.factorized.factorized_network import FactorizedStructureNetworkV2
from src.sc_sindy.evaluation.splits_factorized import (
    TRAIN_SYSTEMS_2D_FACTORIZED as TRAIN_SYSTEMS_2D,
    TRAIN_SYSTEMS_3D_FACTORIZED as TRAIN_SYSTEMS_3D,
    TRAIN_SYSTEMS_4D_FACTORIZED as TRAIN_SYSTEMS_4D,
    TEST_SYSTEMS_2D_FACTORIZED as TEST_SYSTEMS_2D,
    TEST_SYSTEMS_3D_FACTORIZED as TEST_SYSTEMS_3D,
    TEST_SYSTEMS_4D_FACTORIZED as TEST_SYSTEMS_4D,
)
from src.sc_sindy.network.factorized.term_representation import get_library_terms


# Define zero-shot experiments
EXPERIMENTS = {
    "2D_only": {
        "train_dims": [2],
        "test_dims": [3, 4],
        "description": "Train on 2D systems, test on 3D and 4D (full zero-shot)",
    },
    "2D_3D": {
        "train_dims": [2, 3],
        "test_dims": [4],
        "description": "Train on 2D+3D systems, test on 4D (partial zero-shot)",
    },
    "3D_only": {
        "train_dims": [3],
        "test_dims": [2, 4],
        "description": "Train on 3D systems, test on 2D and 4D",
    },
    "full": {
        "train_dims": [2, 3, 4],
        "test_dims": [2, 3, 4],
        "description": "Train on all dimensions (baseline)",
    },
}


def evaluate_model_on_systems(
    model: FactorizedStructureNetworkV2,
    systems: List,
    n_trajectories: int = 20,
    poly_order: int = 3,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate model on a list of systems."""
    results = []

    for system_cls in systems:
        try:
            system = system_cls()
        except Exception:
            continue

        n_vars = system.dim
        term_names = get_library_terms(n_vars, poly_order)
        true_structure = system.get_true_structure(term_names)

        precisions, recalls, f1s = [], [], []

        for _ in range(n_trajectories):
            try:
                x0 = np.random.randn(n_vars) * 2
                t = np.linspace(0, 50, 5000)
                trajectory = system.generate_trajectory(x0, t, noise_level=0.05)
                trajectory = trajectory[100:-100]

                if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
                    continue

                probs = model.predict_structure(trajectory, n_vars, poly_order)
                pred_structure = (probs > threshold).astype(float)

                true_pos = (pred_structure * true_structure).sum()
                pred_pos = pred_structure.sum()
                actual_pos = true_structure.sum()

                precision = true_pos / pred_pos if pred_pos > 0 else 0
                recall = true_pos / actual_pos if actual_pos > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            except Exception:
                continue

        if len(f1s) > 0:
            results.append({
                "system": system_cls.__name__,
                "dim": n_vars,
                "precision": float(np.mean(precisions)),
                "recall": float(np.mean(recalls)),
                "f1": float(np.mean(f1s)),
                "n_samples": len(f1s),
            })

    return results


def run_zero_shot_experiment(
    experiment_name: str,
    experiment_config: Dict,
    train_systems_by_dim: Dict[int, List],
    test_systems_by_dim: Dict[int, List],
    epochs: int,
    latent_dim: int,
    n_traj: int,
    poly_order: int,
    pos_weight: float,
    seed: int,
) -> Dict:
    """Run a single zero-shot experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Description: {experiment_config['description']}")
    print(f"Train dims: {experiment_config['train_dims']}")
    print(f"Test dims: {experiment_config['test_dims']}")
    print(f"{'='*70}")

    # Build training system dict
    train_systems = {}
    for dim in experiment_config["train_dims"]:
        if dim in train_systems_by_dim:
            train_systems[dim] = train_systems_by_dim[dim]

    # Generate training data
    print("\nGenerating training data...")
    samples = generate_mixed_training_data(
        systems_by_dim=train_systems,
        n_trajectories_per_system=n_traj,
        poly_order=poly_order,
    )

    # Train model
    print("\nTraining model...")
    model, history = train_factorized_network(
        samples=samples,
        latent_dim=latent_dim,
        epochs=epochs,
        pos_weight=pos_weight,
        seed=seed,
        use_relative_eq_encoder=True,
        verbose=True,
    )

    # Evaluate on all test dimensions
    results_by_dim = {}
    for dim in experiment_config["test_dims"]:
        if dim not in test_systems_by_dim:
            continue

        print(f"\nEvaluating on {dim}D test systems...")
        is_zero_shot = dim not in experiment_config["train_dims"]

        test_results = evaluate_model_on_systems(
            model=model,
            systems=test_systems_by_dim[dim],
            poly_order=poly_order,
        )

        # Also evaluate on training systems (for comparison)
        train_results = evaluate_model_on_systems(
            model=model,
            systems=train_systems_by_dim.get(dim, []),
            poly_order=poly_order,
        )

        # Compute aggregate metrics
        all_test_f1s = [r["f1"] for r in test_results]
        all_train_f1s = [r["f1"] for r in train_results]

        results_by_dim[dim] = {
            "is_zero_shot": is_zero_shot,
            "test_systems": test_results,
            "train_systems": train_results,
            "test_mean_f1": float(np.mean(all_test_f1s)) if all_test_f1s else 0,
            "train_mean_f1": float(np.mean(all_train_f1s)) if all_train_f1s else 0,
        }

        status = "ZERO-SHOT" if is_zero_shot else "seen dim"
        print(f"  {dim}D ({status}): Test F1 = {results_by_dim[dim]['test_mean_f1']:.4f}, "
              f"Train F1 = {results_by_dim[dim]['train_mean_f1']:.4f}")

    return {
        "experiment": experiment_name,
        "config": experiment_config,
        "results_by_dim": results_by_dim,
        "n_train_samples": len(samples),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation experiments")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--poly_order", type=int, default=3, help="Maximum polynomial order")
    parser.add_argument("--n_traj", type=int, default=30, help="Trajectories per training system")
    parser.add_argument("--pos_weight", type=float, default=3.0, help="BCE positive class weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Specific experiments to run (default: all)")
    parser.add_argument("--output_dir", type=str, default="models/factorized",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("ZERO-SHOT EVALUATION EXPERIMENTS")
    print("=" * 70)
    print(f"Epochs: {args.epochs}, Latent dim: {args.latent_dim}")
    print(f"Poly order: {args.poly_order}, Trajectories: {args.n_traj}")
    print(f"Pos weight: {args.pos_weight}, Random seed: {args.seed}")

    # Setup all systems
    train_systems_by_dim = {
        2: TRAIN_SYSTEMS_2D,
        3: TRAIN_SYSTEMS_3D,
        4: TRAIN_SYSTEMS_4D,
    }
    test_systems_by_dim = {
        2: TEST_SYSTEMS_2D,
        3: TEST_SYSTEMS_3D,
        4: TEST_SYSTEMS_4D,
    }

    # Select experiments
    if args.experiments:
        experiments_to_run = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}
    else:
        experiments_to_run = EXPERIMENTS

    # Run experiments
    all_results = []
    for exp_name, exp_config in experiments_to_run.items():
        result = run_zero_shot_experiment(
            experiment_name=exp_name,
            experiment_config=exp_config,
            train_systems_by_dim=train_systems_by_dim,
            test_systems_by_dim=test_systems_by_dim,
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            n_traj=args.n_traj,
            poly_order=args.poly_order,
            pos_weight=args.pos_weight,
            seed=args.seed,
        )
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("ZERO-SHOT EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<15} {'Train Dims':<12} {'Test Dim':<10} {'Status':<12} {'F1':>8}")
    print("-" * 70)

    for result in all_results:
        exp_name = result["experiment"]
        train_dims = str(result["config"]["train_dims"])

        for dim, dim_results in result["results_by_dim"].items():
            status = "ZERO-SHOT" if dim_results["is_zero_shot"] else "seen"
            f1 = dim_results["test_mean_f1"]
            print(f"{exp_name:<15} {train_dims:<12} {dim}D{'':<7} {status:<12} {f1:>8.4f}")

    # Compute zero-shot vs seen comparison
    print("\n" + "=" * 70)
    print("ZERO-SHOT vs SEEN DIMENSION COMPARISON")
    print("=" * 70)

    zero_shot_f1s = []
    seen_f1s = []
    for result in all_results:
        for dim, dim_results in result["results_by_dim"].items():
            f1 = dim_results["test_mean_f1"]
            if dim_results["is_zero_shot"]:
                zero_shot_f1s.append(f1)
            else:
                seen_f1s.append(f1)

    if zero_shot_f1s:
        print(f"Zero-shot dimensions: Mean F1 = {np.mean(zero_shot_f1s):.4f} ± {np.std(zero_shot_f1s):.4f}")
    if seen_f1s:
        print(f"Seen dimensions:      Mean F1 = {np.mean(seen_f1s):.4f} ± {np.std(seen_f1s):.4f}")

    if zero_shot_f1s and seen_f1s:
        gap = np.mean(seen_f1s) - np.mean(zero_shot_f1s)
        print(f"Performance gap:      {gap:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"zero_shot_results_{timestamp}.json")

    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "experiments": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
