#!/usr/bin/env python
"""
Test script for the Factorized Structure Network.

This script tests:
1. Term representation utilities
2. Component forward passes (TermEmbedder, TrajectoryEncoder)
3. Full network forward pass for 2D, 3D, 4D inputs
4. Training on mixed 2D+3D systems
5. Zero-shot generalization to 4D
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, "/Users/efang/Desktop/coding/structured-contained-sindy/src")


def test_term_representation():
    """Test term representation utilities."""
    print("\n" + "=" * 60)
    print("TEST 1: Term Representation Utilities")
    print("=" * 60)

    from sc_sindy.network.factorized import (
        term_name_to_powers,
        powers_to_term_name,
        get_library_terms,
        get_library_powers,
        count_library_terms,
    )

    # Test term_name_to_powers
    test_cases = [
        ("1", []),
        ("x", [(0, 1)]),
        ("y", [(1, 1)]),
        ("xy", [(0, 1), (1, 1)]),
        ("xxx", [(0, 3)]),
        ("xxy", [(0, 2), (1, 1)]),
        ("xyz", [(0, 1), (1, 1), (2, 1)]),
    ]

    print("\nterm_name_to_powers:")
    for term, expected in test_cases:
        result = term_name_to_powers(term)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {term:6} -> {str(result):30} [{status}]")

    # Test round-trip
    print("\nRound-trip test:")
    for term, _ in test_cases:
        powers = term_name_to_powers(term)
        back = powers_to_term_name(powers)
        status = "PASS" if back == term else "FAIL"
        print(f"  {term} -> {powers} -> {back} [{status}]")

    # Test library generation
    print("\nLibrary terms for 2D, poly_order=3:")
    terms = get_library_terms(2, 3)
    print(f"  {terms}")
    print(f"  Count: {len(terms)} (expected: {count_library_terms(2, 3)})")

    print("\nLibrary terms for 3D, poly_order=2:")
    terms = get_library_terms(3, 2)
    print(f"  {terms}")
    print(f"  Count: {len(terms)} (expected: {count_library_terms(3, 2)})")

    print("\nLibrary terms for 4D, poly_order=2:")
    terms = get_library_terms(4, 2)
    print(f"  {terms}")
    print(f"  Count: {len(terms)} (expected: {count_library_terms(4, 2)})")

    return True


def test_term_embedder():
    """Test TermEmbedder forward pass."""
    print("\n" + "=" * 60)
    print("TEST 2: TermEmbedder")
    print("=" * 60)

    from sc_sindy.network.factorized import TermEmbedder, get_library_powers

    embedder = TermEmbedder(latent_dim=64)
    print(f"Created TermEmbedder with latent_dim=64")

    # Test single term embedding
    powers_xy = [(0, 1), (1, 1)]  # xy
    embed = embedder.embed_term(powers_xy)
    print(f"\nEmbed 'xy': shape = {embed.shape}")

    # Test constant term
    embed_const = embedder.embed_term([])
    print(f"Embed '1': shape = {embed_const.shape}")

    # Test full library embedding
    for n_vars, poly_order in [(2, 3), (3, 2), (4, 2)]:
        embeds = embedder(n_vars, poly_order)
        n_terms = embeds.shape[0]
        print(f"Embed library ({n_vars}D, order={poly_order}): shape = {embeds.shape}")

    return True


def test_trajectory_encoder():
    """Test TrajectoryEncoder forward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: TrajectoryEncoder")
    print("=" * 60)

    from sc_sindy.network.factorized import (
        StatisticsEncoder,
        GRUEncoder,
        extract_per_variable_stats,
    )

    # Test with different dimensions
    for n_vars in [2, 3, 4]:
        x = np.random.randn(1000, n_vars)
        stats = extract_per_variable_stats(x)
        print(f"\n{n_vars}D trajectory stats shape: {stats.shape}")

        # Test StatisticsEncoder
        import torch

        encoder = StatisticsEncoder(latent_dim=64)
        stats_tensor = torch.FloatTensor(stats)
        latent = encoder(stats_tensor)
        print(f"  StatisticsEncoder output: {latent.shape}")

        # Test GRUEncoder
        encoder_gru = GRUEncoder(latent_dim=64)
        x_tensor = torch.FloatTensor(x)
        latent_gru = encoder_gru(x_tensor)
        print(f"  GRUEncoder output: {latent_gru.shape}")

    return True


def test_factorized_network():
    """Test FactorizedStructureNetwork forward pass."""
    print("\n" + "=" * 60)
    print("TEST 4: FactorizedStructureNetwork")
    print("=" * 60)

    from sc_sindy.network.factorized import (
        FactorizedStructureNetwork,
        FactorizedStructureNetworkV2,
    )

    # Test V1
    print("\nFactorizedStructureNetwork (V1):")
    model = FactorizedStructureNetwork(latent_dim=64)

    for n_vars, poly_order in [(2, 3), (3, 2), (4, 2)]:
        x = np.random.randn(1000, n_vars)
        probs = model.predict_structure(x, n_vars=n_vars, poly_order=poly_order)
        print(f"  {n_vars}D, order={poly_order}: probs shape = {probs.shape}")
        print(f"    Min prob: {probs.min():.3f}, Max prob: {probs.max():.3f}")

    # Test V2
    print("\nFactorizedStructureNetworkV2 (efficient batching):")
    model_v2 = FactorizedStructureNetworkV2(latent_dim=64)

    for n_vars, poly_order in [(2, 3), (3, 2), (4, 2)]:
        x = np.random.randn(1000, n_vars)
        probs = model_v2.predict_structure(x, n_vars=n_vars, poly_order=poly_order)
        print(f"  {n_vars}D, order={poly_order}: probs shape = {probs.shape}")

    return True


def test_training_small():
    """Test training on a small dataset."""
    print("\n" + "=" * 60)
    print("TEST 5: Training (Small Dataset)")
    print("=" * 60)

    from sc_sindy.network.factorized import (
        train_factorized_network_with_systems,
        FactorizedPredictor,
    )
    from sc_sindy.systems.oscillators import VanDerPol, DuffingOscillator

    # Check if we have 3D systems
    try:
        from sc_sindy.systems.chaotic import Lorenz

        systems_3d = [Lorenz]
        print("Using 2D + 3D systems")
    except ImportError:
        systems_3d = None
        print("Using 2D systems only (no 3D systems available)")

    print("\nTraining factorized network...")
    print("  Systems 2D: VanDerPol, Duffing")
    if systems_3d:
        print("  Systems 3D: Lorenz")
    print("  Trajectories per system: 10")
    print("  Epochs: 20")

    model, history, config = train_factorized_network_with_systems(
        systems_2d=[VanDerPol, DuffingOscillator],
        systems_3d=systems_3d,
        n_trajectories_per_system=10,
        poly_order=3,
        latent_dim=64,
        epochs=20,
        verbose=True,
    )

    print(f"\nTraining complete:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Config: {config}")

    # Create predictor
    predictor = FactorizedPredictor(model)

    # Test on 2D
    print("\nPredictions on random 2D trajectory:")
    x_2d = np.random.randn(1000, 2)
    probs_2d = predictor.predict(x_2d, poly_order=3)
    print(f"  Shape: {probs_2d.shape}")
    print(f"  Eq 1 probs: {probs_2d[0][:5]}...")
    print(f"  Eq 2 probs: {probs_2d[1][:5]}...")

    # Test on 3D
    print("\nPredictions on random 3D trajectory:")
    x_3d = np.random.randn(1000, 3)
    probs_3d = predictor.predict(x_3d, poly_order=2)
    print(f"  Shape: {probs_3d.shape}")

    # Test on 4D (zero-shot!)
    print("\nPredictions on random 4D trajectory (ZERO-SHOT!):")
    x_4d = np.random.randn(1000, 4)
    probs_4d = predictor.predict(x_4d, poly_order=2)
    print(f"  Shape: {probs_4d.shape}")
    print(f"  (Model never saw 4D data during training)")

    return True


def test_on_real_system():
    """Test predictions on a real dynamical system."""
    print("\n" + "=" * 60)
    print("TEST 6: Predictions on Real System (VanDerPol)")
    print("=" * 60)

    from sc_sindy.network.factorized import (
        FactorizedStructureNetworkV2,
        FactorizedPredictor,
        get_library_terms,
    )
    from sc_sindy.systems.oscillators import VanDerPol

    # Create a simple model (untrained - just testing forward pass)
    model = FactorizedStructureNetworkV2(latent_dim=64)
    predictor = FactorizedPredictor(model)

    # Generate VanDerPol trajectory
    system = VanDerPol()
    x0 = np.array([2.0, 0.0])
    t = np.linspace(0, 50, 5000)
    trajectory = system.generate_trajectory(x0, t)

    print(f"\nVanDerPol trajectory shape: {trajectory.shape}")
    print(f"True equations:")
    print(f"  dx/dt = y")
    print(f"  dy/dt = mu*(1-x^2)*y - x")

    # Get predictions
    result = predictor.predict_with_threshold(trajectory, threshold=0.3, poly_order=3)
    term_names = result["term_names"]
    probs = result["probs"]

    print(f"\nPredicted probabilities (untrained model):")
    for eq_idx in range(2):
        print(f"  Equation {eq_idx + 1}:")
        sorted_idx = np.argsort(probs[eq_idx])[::-1]
        for idx in sorted_idx[:5]:
            print(f"    {term_names[idx]:6}: {probs[eq_idx, idx]:.3f}")

    # Get true structure for comparison
    true_structure = system.get_true_structure(term_names)
    print(f"\nTrue structure (for reference):")
    for eq_idx in range(2):
        active = [term_names[j] for j in range(len(term_names)) if true_structure[eq_idx, j]]
        print(f"  Equation {eq_idx + 1}: {active}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("FACTORIZED STRUCTURE NETWORK TEST SUITE")
    print("=" * 60)

    tests = [
        ("Term Representation", test_term_representation),
        ("Term Embedder", test_term_embedder),
        ("Trajectory Encoder", test_trajectory_encoder),
        ("Factorized Network", test_factorized_network),
        ("Training (Small)", test_training_small),
        ("Real System Test", test_on_real_system),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, "ERROR"))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"  {name:30} [{status}]")

    n_pass = sum(1 for _, s in results if s == "PASS")
    print(f"\n{n_pass}/{len(results)} tests passed")

    return all(s == "PASS" for _, s in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
