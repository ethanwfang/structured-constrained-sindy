"""
Multi-seed training script for factorized structure network.

This script trains multiple models with different random seeds to provide
statistically robust results with confidence intervals.

Usage:
    python scripts/train_factorized_multiseed.py --n_seeds 5 --epochs 100
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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


def evaluate_model(
    model: FactorizedStructureNetworkV2,
    test_systems_by_dim: Dict[int, List],
    n_trajectories: int = 20,
    poly_order: int = 3,
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate a trained model on test systems.

    Parameters
    ----------
    model : FactorizedStructureNetworkV2
        Trained model.
    test_systems_by_dim : Dict[int, List]
        Test systems grouped by dimension.
    n_trajectories : int
        Number of test trajectories per system.
    poly_order : int
        Maximum polynomial order.
    threshold : float
        Probability threshold for structure prediction.

    Returns
    -------
    results : Dict
        Evaluation results per system and aggregated.
    """
    results = {}

    for dim, system_classes in test_systems_by_dim.items():
        dim_results = []

        for system_cls in system_classes:
            try:
                system = system_cls()
            except Exception as e:
                print(f"Could not instantiate {system_cls.__name__}: {e}")
                continue

            n_vars = system.dim
            term_names = get_library_terms(n_vars, poly_order)
            true_structure = system.get_true_structure(term_names)

            precisions = []
            recalls = []
            f1s = []

            for _ in range(n_trajectories):
                try:
                    # Generate trajectory
                    x0 = np.random.randn(n_vars) * 2
                    t = np.linspace(0, 50, 5000)
                    trajectory = system.generate_trajectory(x0, t, noise_level=0.05)

                    # Trim transients
                    trajectory = trajectory[100:-100]

                    # Skip if trajectory is invalid
                    if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
                        continue

                    # Predict structure
                    probs = model.predict_structure(trajectory, n_vars, poly_order)
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

                except Exception:
                    continue

            if len(f1s) > 0:
                dim_results.append({
                    "system": system_cls.__name__,
                    "precision": float(np.mean(precisions)),
                    "recall": float(np.mean(recalls)),
                    "f1": float(np.mean(f1s)),
                    "n_samples": len(f1s),
                })

        results[dim] = dim_results

    # Compute aggregated metrics
    all_f1s = []
    for dim, dim_results in results.items():
        for sys_result in dim_results:
            all_f1s.append(sys_result["f1"])

    results["aggregate"] = {
        "mean_f1": float(np.mean(all_f1s)) if all_f1s else 0,
        "std_f1": float(np.std(all_f1s)) if all_f1s else 0,
    }

    return results


def train_and_evaluate_seed(
    seed: int,
    samples,
    epochs: int,
    latent_dim: int,
    pos_weight: float,
    use_relative_eq_encoder: bool,
    use_correlations: bool,
    test_systems_by_dim: Dict[int, List],
    poly_order: int,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """Train and evaluate a model with a specific seed."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training seed {seed}")
        print(f"{'='*60}")

    # Train model
    model, history = train_factorized_network(
        samples=samples,
        latent_dim=latent_dim,
        epochs=epochs,
        pos_weight=pos_weight,
        seed=seed,
        use_relative_eq_encoder=use_relative_eq_encoder,
        use_correlations=use_correlations,
        verbose=verbose,
    )

    # Evaluate
    if verbose:
        print(f"\nEvaluating seed {seed}...")

    eval_results = evaluate_model(
        model=model,
        test_systems_by_dim=test_systems_by_dim,
        poly_order=poly_order,
    )

    if verbose:
        print(f"Seed {seed} - Mean F1: {eval_results['aggregate']['mean_f1']:.4f}")

    return eval_results, history


def aggregate_multiseed_results(all_results: List[Dict]) -> Dict:
    """Aggregate results from multiple seeds."""

    # Collect per-system results
    system_results = {}

    for result in all_results:
        for dim, dim_results in result.items():
            if dim == "aggregate":
                continue

            for sys_result in dim_results:
                name = sys_result["system"]
                if name not in system_results:
                    system_results[name] = {
                        "dim": dim,
                        "precisions": [],
                        "recalls": [],
                        "f1s": [],
                    }
                system_results[name]["precisions"].append(sys_result["precision"])
                system_results[name]["recalls"].append(sys_result["recall"])
                system_results[name]["f1s"].append(sys_result["f1"])

    # Compute aggregated stats
    aggregated = {}
    all_f1_means = []

    for name, data in system_results.items():
        dim = data["dim"]
        if dim not in aggregated:
            aggregated[dim] = []

        f1_mean = np.mean(data["f1s"])
        f1_std = np.std(data["f1s"])
        all_f1_means.append(f1_mean)

        aggregated[dim].append({
            "system": name,
            "precision_mean": float(np.mean(data["precisions"])),
            "precision_std": float(np.std(data["precisions"])),
            "recall_mean": float(np.mean(data["recalls"])),
            "recall_std": float(np.std(data["recalls"])),
            "f1_mean": float(f1_mean),
            "f1_std": float(f1_std),
            "n_seeds": len(data["f1s"]),
        })

    # Overall aggregation
    aggregated["overall"] = {
        "mean_f1": float(np.mean(all_f1_means)),
        "std_f1": float(np.std(all_f1_means)),
        "n_systems": len(all_f1_means),
    }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training for factorized network")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per seed")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--pos_weight", type=float, default=3.0, help="Positive class weight for BCE")
    parser.add_argument("--poly_order", type=int, default=3, help="Maximum polynomial order")
    parser.add_argument("--n_traj", type=int, default=50, help="Trajectories per training system")
    parser.add_argument("--use_correlations", action="store_true", help="Use correlation features")
    parser.add_argument("--output_dir", type=str, default="models/factorized", help="Output directory")
    args = parser.parse_args()

    print("Multi-seed training for Factorized Structure Network")
    print(f"Seeds: {args.n_seeds}, Epochs: {args.epochs}, Latent dim: {args.latent_dim}")
    print(f"Pos weight: {args.pos_weight}, Use correlations: {args.use_correlations}")

    # Setup training systems
    systems_by_dim = {
        2: TRAIN_SYSTEMS_2D,
        3: TRAIN_SYSTEMS_3D,
        4: TRAIN_SYSTEMS_4D,
    }

    # Setup test systems
    test_systems_by_dim = {
        2: TEST_SYSTEMS_2D,
        3: TEST_SYSTEMS_3D,
        4: TEST_SYSTEMS_4D,
    }

    print("\nGenerating training data...")
    samples = generate_mixed_training_data(
        systems_by_dim=systems_by_dim,
        n_trajectories_per_system=args.n_traj,
        poly_order=args.poly_order,
        include_correlations=args.use_correlations,
    )

    # Train multiple seeds
    all_results = []
    all_histories = []

    for seed in range(args.n_seeds):
        results, history = train_and_evaluate_seed(
            seed=seed,
            samples=samples,
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            pos_weight=args.pos_weight,
            use_relative_eq_encoder=True,
            use_correlations=args.use_correlations,
            test_systems_by_dim=test_systems_by_dim,
            poly_order=args.poly_order,
        )
        all_results.append(results)
        all_histories.append(history)

    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATED RESULTS (mean ± std)")
    print("="*60)

    aggregated = aggregate_multiseed_results(all_results)

    for dim in sorted([d for d in aggregated.keys() if d != "overall"]):
        print(f"\n{dim}D Systems:")
        for sys_result in aggregated[dim]:
            print(f"  {sys_result['system']:30s}: F1 = {sys_result['f1_mean']:.3f} ± {sys_result['f1_std']:.3f}")

    print(f"\nOverall: F1 = {aggregated['overall']['mean_f1']:.3f} ± {aggregated['overall']['std_f1']:.3f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"multiseed_results_{timestamp}.json")

    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "per_seed_results": all_results,
            "aggregated": aggregated,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
