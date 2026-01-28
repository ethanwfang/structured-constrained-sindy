#!/usr/bin/env python3
"""
SC-SINDy Fair Evaluation Script

This script trains a structure network on TRAIN systems and evaluates
SC-SINDy using LEARNED predictions on held-out TEST systems.

Usage (in Google Colab after cloning):
    !python scripts/run_fair_evaluation.py

Or with options:
    !python scripts/run_fair_evaluation.py --n_train 200 --n_trials 20 --epochs 100
"""

import argparse
import warnings
from typing import List, Type

import numpy as np

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


def main():
    parser = argparse.ArgumentParser(description="SC-SINDy Fair Evaluation")
    parser.add_argument("--n_train", type=int, default=200, help="Training trajectories")
    parser.add_argument("--n_trials", type=int, default=20, help="Test trials per config")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")
    args = parser.parse_args()

    print("=" * 60)
    print("SC-SINDy Fair Evaluation (No Oracle)")
    print("=" * 60)

    # Import SC-SINDy
    print("\nImporting SC-SINDy...")
    from sc_sindy import (
        build_library_2d,
        compute_derivatives_finite_diff,
        compute_structure_metrics,
        format_equation,
        sindy_stls,
        sindy_structure_constrained,
    )
    from sc_sindy.evaluation import (
        TEST_SYSTEMS_2D,
        TRAIN_SYSTEMS_2D,
        print_split_info,
    )
    from sc_sindy.network import (
        StructurePredictor,
        extract_trajectory_features,
        train_structure_network,
    )
    from sc_sindy.systems import DynamicalSystem

    print("Imports successful!")

    # Show train/test split
    print("\n")
    print_split_info()

    # =========================================================================
    # Generate Training Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Generate Training Data")
    print("=" * 60)
    print(f"Training on: {[s.__name__ for s in TRAIN_SYSTEMS_2D]}")
    print(f"Will test on: {[s.__name__ for s in TEST_SYSTEMS_2D]} (held out)")

    # Get library term names
    dummy_x = np.random.randn(10, 2)
    _, term_names = build_library_2d(dummy_x)
    print(f"\nLibrary terms ({len(term_names)}): {term_names}")

    t = np.linspace(0, 50, 5000)
    dt = t[1] - t[0]
    noise_levels = [0.0, 0.05, 0.10]

    train_data = []
    for SystemClass in TRAIN_SYSTEMS_2D:
        system = SystemClass()
        print(f"  Generating data for {system.name}...")

        true_structure = system.get_true_structure(term_names)
        structure_flat = true_structure.flatten().astype(float)

        n_per_system = args.n_train // len(TRAIN_SYSTEMS_2D)
        success_count = 0

        for _ in range(n_per_system * 2):
            if success_count >= n_per_system:
                break

            x0 = np.random.randn(2) * 2
            noise = np.random.choice(noise_levels)

            try:
                x = system.generate_trajectory(x0, t, noise_level=noise)
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    continue

                x_trim = x[100:-100]
                features = extract_trajectory_features(x_trim, dt)

                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    continue

                train_data.append((features, structure_flat))
                success_count += 1
            except Exception:
                continue

        print(f"    Generated {success_count} trajectories")

    print(f"\nTotal training samples: {len(train_data)}")

    # =========================================================================
    # Train Structure Network
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Train Structure Network")
    print("=" * 60)

    # Compute normalization statistics
    all_features = np.array([f for f, _ in train_data])
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    feature_std = np.where(feature_std < 1e-10, 1.0, feature_std)

    # Normalize training data
    normalized_data = [
        ((features - feature_mean) / feature_std, labels) for features, labels in train_data
    ]

    print(f"Training for {args.epochs} epochs...")
    model, history = train_structure_network(
        normalized_data,
        epochs=args.epochs,
        batch_size=32,
        lr=0.001,
        verbose=args.verbose,
    )

    print("\nTraining complete!")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    # Create predictor
    n_vars = 2
    n_terms = len(term_names)
    predictor = StructurePredictor(
        model=model,
        n_vars=n_vars,
        n_terms=n_terms,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    # =========================================================================
    # Evaluate on Test Systems
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate on TEST Systems (Never Seen in Training)")
    print("=" * 60)

    test_noise_levels = [0.0, 0.05, 0.10]
    results = []

    for SystemClass in TEST_SYSTEMS_2D:
        system = SystemClass()
        print(f"\nEvaluating on {system.name}...")

        true_xi = system.get_true_coefficients(term_names)
        true_structure = np.abs(true_xi) > 1e-6

        for noise in test_noise_levels:
            std_f1s, sc_f1s, net_f1s = [], [], []

            for _ in range(args.n_trials):
                x0 = np.random.randn(2) * 2

                try:
                    x = system.generate_trajectory(x0, t, noise_level=noise)
                    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                        continue

                    x_trim = x[100:-100]
                    x_dot = compute_derivatives_finite_diff(x_trim, dt)
                    Theta, _ = build_library_2d(x_trim)

                    # Standard SINDy
                    xi_std, _ = sindy_stls(Theta, x_dot, threshold=0.1)

                    # SC-SINDy with LEARNED predictions
                    network_probs = predictor.predict_from_trajectory(x_trim, dt)
                    xi_sc, _ = sindy_structure_constrained(
                        Theta, x_dot, network_probs, structure_threshold=0.3
                    )

                    # Metrics
                    metrics_std = compute_structure_metrics(xi_std, true_xi)
                    metrics_sc = compute_structure_metrics(xi_sc, true_xi)

                    # Network prediction quality
                    pred_structure = network_probs > 0.5
                    net_metrics = compute_structure_metrics(
                        pred_structure.astype(float),
                        true_structure.astype(float),
                    )

                    std_f1s.append(metrics_std["f1"])
                    sc_f1s.append(metrics_sc["f1"])
                    net_f1s.append(net_metrics["f1"])

                except Exception:
                    continue

            if std_f1s:
                results.append(
                    {
                        "system": system.name,
                        "noise": noise,
                        "std_f1_mean": np.mean(std_f1s),
                        "std_f1_std": np.std(std_f1s),
                        "sc_f1_mean": np.mean(sc_f1s),
                        "sc_f1_std": np.std(sc_f1s),
                        "net_f1_mean": np.mean(net_f1s),
                        "improvement": np.mean(sc_f1s) - np.mean(std_f1s),
                    }
                )

                print(
                    f"  Noise {noise:.2f}: "
                    f"Std F1={np.mean(std_f1s):.3f}, "
                    f"SC F1={np.mean(sc_f1s):.3f}, "
                    f"Net F1={np.mean(net_f1s):.3f}, "
                    f"Improve={np.mean(sc_f1s)-np.mean(std_f1s):+.3f}"
                )

    # =========================================================================
    # Results Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'System':<25} {'Noise':<8} {'Std F1':<12} {'SC F1':<12} "
        f"{'Net F1':<10} {'Improve':<10}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['system']:<25} {r['noise']:<8.2f} "
            f"{r['std_f1_mean']:.3f}+/-{r['std_f1_std']:.2f}  "
            f"{r['sc_f1_mean']:.3f}+/-{r['sc_f1_std']:.2f}  "
            f"{r['net_f1_mean']:.3f}      "
            f"{r['improvement']:+.3f}"
        )

    # Overall statistics
    all_std = [r["std_f1_mean"] for r in results]
    all_sc = [r["sc_f1_mean"] for r in results]
    all_improve = [r["improvement"] for r in results]

    print("-" * 70)
    print(
        f"{'OVERALL':<25} {'---':<8} "
        f"{np.mean(all_std):.3f}            "
        f"{np.mean(all_sc):.3f}            "
        f"{'---':<10} "
        f"{np.mean(all_improve):+.3f}"
    )

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print(f"  - Standard SINDy average F1: {np.mean(all_std):.3f}")
    print(f"  - SC-SINDy (Learned) average F1: {np.mean(all_sc):.3f}")
    print(f"  - Average improvement: {np.mean(all_improve):+.3f}")
    if np.mean(all_improve) > 0:
        print("  - SC-SINDy IMPROVES over standard SINDy on unseen systems!")
    else:
        print("  - Network may need more training data for better generalization")
    print("=" * 70)

    # =========================================================================
    # Compare Oracle vs Learned
    # =========================================================================
    print("\n" + "=" * 60)
    print("BONUS: Oracle vs Learned Comparison")
    print("=" * 60)

    system = TEST_SYSTEMS_2D[0]()
    print(f"Testing on {system.name} with 5% noise...")

    true_xi = system.get_true_coefficients(term_names)
    true_structure = np.abs(true_xi) > 1e-6

    oracle_f1s, learned_f1s, std_f1s = [], [], []

    for _ in range(30):
        x0 = np.random.randn(2) * 2
        x = system.generate_trajectory(x0, t, noise_level=0.05)

        if np.any(np.isnan(x)):
            continue

        x_trim = x[100:-100]
        x_dot = compute_derivatives_finite_diff(x_trim, dt)
        Theta, _ = build_library_2d(x_trim)

        # Standard SINDy
        xi_std, _ = sindy_stls(Theta, x_dot, threshold=0.1)

        # Oracle (CHEATING)
        oracle_probs = true_structure.astype(float) * 0.9 + 0.05
        xi_oracle, _ = sindy_structure_constrained(
            Theta, x_dot, oracle_probs, structure_threshold=0.3
        )

        # Learned (FAIR)
        learned_probs = predictor.predict_from_trajectory(x_trim, dt)
        xi_learned, _ = sindy_structure_constrained(
            Theta, x_dot, learned_probs, structure_threshold=0.3
        )

        std_f1s.append(compute_structure_metrics(xi_std, true_xi)["f1"])
        oracle_f1s.append(compute_structure_metrics(xi_oracle, true_xi)["f1"])
        learned_f1s.append(compute_structure_metrics(xi_learned, true_xi)["f1"])

    print("\nResults (30 trials):")
    print(f"  Standard SINDy:     F1 = {np.mean(std_f1s):.3f} +/- {np.std(std_f1s):.3f}")
    print(
        f"  SC-SINDy (Oracle):  F1 = {np.mean(oracle_f1s):.3f} +/- {np.std(oracle_f1s):.3f}"
        f"  <- CHEATING!"
    )
    print(
        f"  SC-SINDy (Learned): F1 = {np.mean(learned_f1s):.3f} +/- {np.std(learned_f1s):.3f}"
        f"  <- FAIR"
    )
    print(f"\nOracle improvement: {np.mean(oracle_f1s) - np.mean(std_f1s):+.3f} (not publishable)")
    print(f"Learned improvement: {np.mean(learned_f1s) - np.mean(std_f1s):+.3f} (publishable!)")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
