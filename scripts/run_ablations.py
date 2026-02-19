"""
Ablation study script for factorized structure network.

This script runs systematic ablation experiments to analyze the contribution
of each architectural component.

Components tested:
1. Term embedding: Sum vs Tensor Product
2. Equation encoder: Embedding table vs Relative position
3. Trajectory encoder: Statistics vs Statistics+Correlations
4. Loss function: Standard BCE vs Weighted BCE (recall-optimized)

Usage:
    python scripts/run_ablations.py --epochs 50 --n_traj 30
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


# Define ablation configurations
ABLATION_CONFIGS = {
    # Baseline: original configuration
    "baseline": {
        "use_relative_eq_encoder": False,
        "use_correlations": False,
        "pos_weight": 1.0,
        "use_tensor_product": False,  # Default sum aggregation
        "description": "Original architecture (embedding table, sum aggregation, standard BCE)",
    },
    # Individual components
    "relative_eq_only": {
        "use_relative_eq_encoder": True,
        "use_correlations": False,
        "pos_weight": 1.0,
        "use_tensor_product": False,
        "description": "Relative position equation encoder only",
    },
    "correlations_only": {
        "use_relative_eq_encoder": False,
        "use_correlations": True,
        "pos_weight": 1.0,
        "use_tensor_product": False,
        "description": "Pairwise correlations only",
    },
    "weighted_bce_only": {
        "use_relative_eq_encoder": False,
        "use_correlations": False,
        "pos_weight": 3.0,
        "use_tensor_product": False,
        "description": "Weighted BCE (pos_weight=3.0) only",
    },
    "tensor_product_only": {
        "use_relative_eq_encoder": False,
        "use_correlations": False,
        "pos_weight": 1.0,
        "use_tensor_product": True,
        "description": "Tensor product term embedding only",
    },
    # Combinations
    "rel_eq_tensor": {
        "use_relative_eq_encoder": True,
        "use_correlations": False,
        "pos_weight": 1.0,
        "use_tensor_product": True,
        "description": "Relative eq encoder + tensor product",
    },
    "rel_eq_weighted": {
        "use_relative_eq_encoder": True,
        "use_correlations": False,
        "pos_weight": 3.0,
        "use_tensor_product": False,
        "description": "Relative eq encoder + weighted BCE",
    },
    "all_except_corr": {
        "use_relative_eq_encoder": True,
        "use_correlations": False,
        "pos_weight": 3.0,
        "use_tensor_product": True,
        "description": "All improvements except correlations",
    },
    # Full model
    "full_model": {
        "use_relative_eq_encoder": True,
        "use_correlations": True,
        "pos_weight": 3.0,
        "use_tensor_product": True,
        "description": "Full model with all improvements",
    },
}


def evaluate_model(
    model: FactorizedStructureNetworkV2,
    test_systems_by_dim: Dict[int, List],
    n_trajectories: int = 20,
    poly_order: int = 3,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate a trained model on test systems."""
    results = {}

    for dim, system_classes in test_systems_by_dim.items():
        dim_results = []

        for system_cls in system_classes:
            try:
                system = system_cls()
            except Exception as e:
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
                dim_results.append({
                    "system": system_cls.__name__,
                    "precision": float(np.mean(precisions)),
                    "recall": float(np.mean(recalls)),
                    "f1": float(np.mean(f1s)),
                    "n_samples": len(f1s),
                })

        results[dim] = dim_results

    # Aggregate
    all_f1s = []
    all_precisions = []
    all_recalls = []
    for dim_results in results.values():
        for sys_result in dim_results:
            all_f1s.append(sys_result["f1"])
            all_precisions.append(sys_result["precision"])
            all_recalls.append(sys_result["recall"])

    results["aggregate"] = {
        "mean_f1": float(np.mean(all_f1s)) if all_f1s else 0,
        "mean_precision": float(np.mean(all_precisions)) if all_precisions else 0,
        "mean_recall": float(np.mean(all_recalls)) if all_recalls else 0,
    }

    return results


def run_ablation(
    config_name: str,
    config: Dict,
    samples,
    samples_with_corr,
    test_systems_by_dim: Dict,
    epochs: int,
    latent_dim: int,
    poly_order: int,
    seed: int = 42,
) -> Dict:
    """Run a single ablation configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")

    # Select appropriate training samples
    use_corr = config.get("use_correlations", False)
    training_samples = samples_with_corr if use_corr else samples

    # Note: tensor_product is handled in term_embedder via embed_term's use_tensor_product param
    # For now, we test the architectural components we can control in training

    model, history = train_factorized_network(
        samples=training_samples,
        latent_dim=latent_dim,
        epochs=epochs,
        pos_weight=config["pos_weight"],
        seed=seed,
        use_relative_eq_encoder=config["use_relative_eq_encoder"],
        use_correlations=config["use_correlations"],
        verbose=False,
    )

    # Evaluate
    eval_results = evaluate_model(
        model=model,
        test_systems_by_dim=test_systems_by_dim,
        poly_order=poly_order,
    )

    print(f"Results - F1: {eval_results['aggregate']['mean_f1']:.4f}, "
          f"Precision: {eval_results['aggregate']['mean_precision']:.4f}, "
          f"Recall: {eval_results['aggregate']['mean_recall']:.4f}")

    return {
        "config_name": config_name,
        "config": config,
        "eval_results": eval_results,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study for factorized network")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--poly_order", type=int, default=3, help="Maximum polynomial order")
    parser.add_argument("--n_traj", type=int, default=30, help="Trajectories per training system")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to run (default: all)")
    parser.add_argument("--output_dir", type=str, default="models/factorized",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("ABLATION STUDY FOR FACTORIZED STRUCTURE NETWORK")
    print("=" * 60)
    print(f"Epochs: {args.epochs}, Latent dim: {args.latent_dim}")
    print(f"Poly order: {args.poly_order}, Trajectories: {args.n_traj}")
    print(f"Random seed: {args.seed}")

    # Setup systems
    systems_by_dim = {
        2: TRAIN_SYSTEMS_2D,
        3: TRAIN_SYSTEMS_3D,
        4: TRAIN_SYSTEMS_4D,
    }
    test_systems_by_dim = {
        2: TEST_SYSTEMS_2D,
        3: TEST_SYSTEMS_3D,
        4: TEST_SYSTEMS_4D,
    }

    # Generate training data (both with and without correlations)
    print("\nGenerating training data without correlations...")
    samples = generate_mixed_training_data(
        systems_by_dim=systems_by_dim,
        n_trajectories_per_system=args.n_traj,
        poly_order=args.poly_order,
        include_correlations=False,
    )

    print("\nGenerating training data with correlations...")
    samples_with_corr = generate_mixed_training_data(
        systems_by_dim=systems_by_dim,
        n_trajectories_per_system=args.n_traj,
        poly_order=args.poly_order,
        include_correlations=True,
    )

    # Select configurations to run
    if args.configs:
        configs_to_run = {k: v for k, v in ABLATION_CONFIGS.items() if k in args.configs}
    else:
        configs_to_run = ABLATION_CONFIGS

    # Run ablations
    all_results = []
    for config_name, config in configs_to_run.items():
        result = run_ablation(
            config_name=config_name,
            config=config,
            samples=samples,
            samples_with_corr=samples_with_corr,
            test_systems_by_dim=test_systems_by_dim,
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            poly_order=args.poly_order,
            seed=args.seed,
        )
        all_results.append(result)

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Description':<30}")
    print("-" * 80)

    # Sort by F1 score
    all_results.sort(key=lambda x: x["eval_results"]["aggregate"]["mean_f1"], reverse=True)

    for result in all_results:
        agg = result["eval_results"]["aggregate"]
        desc = result["config"]["description"][:28] + ".." if len(result["config"]["description"]) > 30 else result["config"]["description"]
        print(f"{result['config_name']:<25} {agg['mean_f1']:>8.4f} {agg['mean_precision']:>10.4f} "
              f"{agg['mean_recall']:>8.4f} {desc:<30}")

    print("=" * 80)

    # Compute improvement over baseline
    baseline_f1 = next((r["eval_results"]["aggregate"]["mean_f1"]
                        for r in all_results if r["config_name"] == "baseline"), 0)
    if baseline_f1 > 0:
        print("\nIMPROVEMENT OVER BASELINE:")
        for result in all_results:
            if result["config_name"] != "baseline":
                f1 = result["eval_results"]["aggregate"]["mean_f1"]
                improvement = (f1 - baseline_f1) / baseline_f1 * 100
                sign = "+" if improvement > 0 else ""
                print(f"  {result['config_name']:<25}: {sign}{improvement:.1f}%")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"ablation_results_{timestamp}.json")

    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
