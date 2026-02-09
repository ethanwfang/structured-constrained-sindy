#!/usr/bin/env python
"""
Verify Ensemble-SC-SINDy with properly trained network.

This script trains a network on the expanded library (14 systems, 5 with xy)
and tests all methods on Lynx-Hare data to verify the reported results.
"""
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sc_sindy import (
    load_lynx_hare_data,
    build_library_2d,
    compute_derivatives_finite_diff,
    sindy_stls,
    sindy_structure_constrained,
    two_stage_ensemble,
    ensemble_structure_constrained_sindy,
    structure_weighted_ensemble,
    compute_structure_metrics,
    get_uncertainty_report,
)
from sc_sindy.network import train_structure_network_with_split, StructurePredictor
from sc_sindy.evaluation import TRAIN_SYSTEMS_2D


def main():
    print("=" * 70)
    print("ENSEMBLE-SC-SINDY VERIFICATION")
    print("=" * 70)
    print(f"\nTraining on {len(TRAIN_SYSTEMS_2D)} systems:")
    for sys_cls in TRAIN_SYSTEMS_2D:
        print(f"  - {sys_cls.__name__}")

    # Step 1: Train network on expanded library
    print("\n" + "-" * 70)
    print("STEP 1: Training network on expanded library...")
    print("-" * 70)

    model, history, config = train_structure_network_with_split(
        train_system_classes=TRAIN_SYSTEMS_2D,
        dimension=2,
        n_trajectories_per_config=50,
        n_param_variations=3,
        epochs=100,
        verbose=True,
    )

    print(f"\nTraining complete.")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Term names: {config['term_names']}")

    # Step 2: Create predictor
    print("\n" + "-" * 70)
    print("STEP 2: Creating predictor...")
    print("-" * 70)

    predictor = StructurePredictor(
        model=model,
        n_vars=config['n_vars'],
        n_terms=config['n_terms'],
        feature_mean=config['feature_mean'],
        feature_std=config['feature_std']
    )

    # Step 3: Load Lynx-Hare data
    print("\n" + "-" * 70)
    print("STEP 3: Loading Lynx-Hare data...")
    print("-" * 70)

    x, years = load_lynx_hare_data()
    print(f"  Data shape: {x.shape}")
    print(f"  Years: {years[0]:.0f} - {years[-1]:.0f}")

    dt = 1.0  # yearly data
    x_dot = compute_derivatives_finite_diff(x, dt)
    Theta, term_names = build_library_2d(x, poly_order=3)
    print(f"  Library terms: {term_names}")

    # Step 4: Get network predictions
    print("\n" + "-" * 70)
    print("STEP 4: Network predictions for Lynx-Hare...")
    print("-" * 70)

    network_probs = predictor.predict_from_trajectory(x, dt)

    print("\nNetwork probability predictions:")
    print("  dH/dt (equation 1):")
    for i, name in enumerate(term_names):
        prob = network_probs[0, i]
        marker = " ***" if name == 'xy' else ""
        print(f"    {name:6s}: {prob:.2%}{marker}")

    print("\n  dL/dt (equation 2):")
    for i, name in enumerate(term_names):
        prob = network_probs[1, i]
        marker = " ***" if name == 'xy' else ""
        print(f"    {name:6s}: {prob:.2%}{marker}")

    # True Lotka-Volterra structure
    # dH/dt = alpha*H - beta*H*L  -> terms: x, xy
    # dL/dt = delta*H*L - gamma*L -> terms: y, xy
    true_xi = np.zeros((2, len(term_names)))

    # Find term indices
    x_idx = term_names.index('x')
    y_idx = term_names.index('y')
    xy_idx = term_names.index('xy')

    true_xi[0, x_idx] = 1   # x term in dH/dt (growth)
    true_xi[0, xy_idx] = 1  # xy term in dH/dt (predation)
    true_xi[1, y_idx] = 1   # y term in dL/dt (death)
    true_xi[1, xy_idx] = 1  # xy term in dL/dt (predation benefit)

    print(f"\n  True Lotka-Volterra structure:")
    print(f"    dH/dt uses: x, xy")
    print(f"    dL/dt uses: y, xy")

    # Step 5: Run all methods
    print("\n" + "=" * 70)
    print("STEP 5: VERIFICATION RESULTS")
    print("=" * 70)

    results = {}

    # Standard SINDy
    print("\n--- Standard SINDy ---")
    xi_std, time_std = sindy_stls(Theta, x_dot, threshold=0.1)
    metrics_std = compute_structure_metrics(xi_std, true_xi)
    results['Standard SINDy'] = metrics_std
    print(f"  F1: {metrics_std['f1']:.3f}")
    print(f"  Precision: {metrics_std['precision']:.3f}")
    print(f"  Recall: {metrics_std['recall']:.3f}")
    print(f"  Active terms: {metrics_std['n_active_pred']}")

    # SC-SINDy alone (threshold=0.3)
    print("\n--- SC-SINDy (threshold=0.3) ---")
    xi_sc, time_sc = sindy_structure_constrained(
        Theta, x_dot, network_probs, structure_threshold=0.3
    )
    metrics_sc = compute_structure_metrics(xi_sc, true_xi)
    results['SC-SINDy (0.3)'] = metrics_sc
    print(f"  F1: {metrics_sc['f1']:.3f}")
    print(f"  Precision: {metrics_sc['precision']:.3f}")
    print(f"  Recall: {metrics_sc['recall']:.3f}")
    print(f"  Active terms: {metrics_sc['n_active_pred']}")

    # Two-stage Ensemble (SC-SINDy -> E-SINDy) with threshold=0.1
    print("\n--- Two-Stage Ensemble (threshold=0.1) ---")
    result_two_stage = two_stage_ensemble(
        Theta, x_dot, network_probs,
        structure_threshold=0.1,
        n_bootstrap=100,
        random_state=42,
    )
    metrics_two_stage = compute_structure_metrics(result_two_stage.xi, true_xi)
    results['Two-Stage (0.1)'] = metrics_two_stage
    print(f"  F1: {metrics_two_stage['f1']:.3f}")
    print(f"  Precision: {metrics_two_stage['precision']:.3f}")
    print(f"  Recall: {metrics_two_stage['recall']:.3f}")
    print(f"  Active terms: {metrics_two_stage['n_active_pred']}")

    # Two-stage Ensemble with threshold=0.3
    print("\n--- Two-Stage Ensemble (threshold=0.3) ---")
    result_two_stage_03 = two_stage_ensemble(
        Theta, x_dot, network_probs,
        structure_threshold=0.3,
        n_bootstrap=100,
        random_state=42,
    )
    metrics_two_stage_03 = compute_structure_metrics(result_two_stage_03.xi, true_xi)
    results['Two-Stage (0.3)'] = metrics_two_stage_03
    print(f"  F1: {metrics_two_stage_03['f1']:.3f}")
    print(f"  Precision: {metrics_two_stage_03['precision']:.3f}")
    print(f"  Recall: {metrics_two_stage_03['recall']:.3f}")
    print(f"  Active terms: {metrics_two_stage_03['n_active_pred']}")

    # Full Ensemble-SC-SINDy (product fusion)
    print("\n--- Ensemble-SC-SINDy (product fusion) ---")
    result_ensemble = ensemble_structure_constrained_sindy(
        Theta, x_dot, network_probs,
        structure_threshold=0.3,
        n_bootstrap=100,
        fusion_method='product',
        random_state=42,
    )
    metrics_ensemble = compute_structure_metrics(result_ensemble.xi, true_xi)
    results['Ensemble-SC (product)'] = metrics_ensemble
    print(f"  F1: {metrics_ensemble['f1']:.3f}")
    print(f"  Precision: {metrics_ensemble['precision']:.3f}")
    print(f"  Recall: {metrics_ensemble['recall']:.3f}")
    print(f"  Active terms: {metrics_ensemble['n_active_pred']}")

    # Structure-weighted ensemble
    print("\n--- Structure-Weighted Ensemble ---")
    result_weighted = structure_weighted_ensemble(
        Theta, x_dot, network_probs,
        n_bootstrap=100,
        random_state=42,
    )
    metrics_weighted = compute_structure_metrics(result_weighted.xi, true_xi)
    results['Weighted Ensemble'] = metrics_weighted
    print(f"  F1: {metrics_weighted['f1']:.3f}")
    print(f"  Precision: {metrics_weighted['precision']:.3f}")
    print(f"  Recall: {metrics_weighted['recall']:.3f}")
    print(f"  Active terms: {metrics_weighted['n_active_pred']}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<30} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<30} {metrics['f1']:>8.3f} {metrics['precision']:>10.3f} {metrics['recall']:>8.3f}")

    print("\n" + "=" * 70)
    print("EXPECTED FROM IMPROVEMENTS_REPORT:")
    print("  SC-SINDy F1 = 0.857 (precision=1.0, recall=0.75)")
    print("=" * 70)

    # Uncertainty report for best ensemble method
    print("\n" + "-" * 70)
    print("UNCERTAINTY QUANTIFICATION (Two-Stage Ensemble)")
    print("-" * 70)
    print(get_uncertainty_report(result_two_stage, term_names))

    # Check xy probability specifically
    print("\n" + "=" * 70)
    print("KEY VERIFICATION: xy TERM PROBABILITY")
    print("=" * 70)
    xy_prob_eq1 = network_probs[0, xy_idx]
    xy_prob_eq2 = network_probs[1, xy_idx]
    print(f"  dH/dt xy probability: {xy_prob_eq1:.2%}")
    print(f"  dL/dt xy probability: {xy_prob_eq2:.2%}")

    if xy_prob_eq1 > 0.40 and xy_prob_eq2 > 0.40:
        print("\n  PASS: Network correctly identifies xy as likely important (>40%)")
    else:
        print("\n  WARNING: xy probability lower than expected (<40%)")
        print("           IMPROVEMENTS_REPORT showed ~48% for dH/dt, ~47% for dL/dt")


if __name__ == "__main__":
    main()
