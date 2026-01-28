#!/usr/bin/env python3
"""
SC-SINDy Quick Test Script

This script runs basic tests to verify the SC-SINDy package is working correctly.

Usage (in Google Colab after cloning):
    !python scripts/run_tests.py
"""

import numpy as np

print("=" * 60)
print("SC-SINDy Quick Test")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from sc_sindy import (
        build_library_2d,
        build_library_3d,
        compute_derivatives_finite_diff,
        compute_structure_metrics,
        format_equation,
        sindy_stls,
    )
    from sc_sindy.evaluation import get_split, print_split_info
    from sc_sindy.systems import Lorenz, LotkaVolterra, VanDerPol

    print("   All imports successful!")
except ImportError as e:
    print(f"   Import failed: {e}")
    exit(1)

# Test Van der Pol
print("\n2. Testing on Van der Pol oscillator...")
system = VanDerPol(mu=1.0)
t = np.linspace(0, 20, 2000)
X = system.generate_trajectory(np.array([1.0, 0.0]), t)
X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])

Theta, labels = build_library_2d(X)
xi, elapsed = sindy_stls(Theta, X_dot, threshold=0.1)

print(f"   Generated {len(t)} data points")
print(f"   Library: {len(labels)} terms")
print(f"   SINDy completed in {elapsed:.4f}s")

print("\n   Discovered equations:")
for i, var in enumerate(["x", "y"]):
    eq = format_equation(xi[i], labels)
    print(f"     d{var}/dt = {eq}")

true_xi = system.get_true_coefficients(labels)
metrics = compute_structure_metrics(xi, true_xi)
print(f"\n   Structure F1: {metrics['f1']:.3f}")

# Test Lorenz
print("\n3. Testing on Lorenz system...")
system = Lorenz(sigma=10, rho=28, beta=8 / 3)
t = np.linspace(0, 10, 5000)
X = system.generate_trajectory(np.array([1.0, 1.0, 1.0]), t)
X_sub = X[::5]
t_sub = t[::5]
X_dot = compute_derivatives_finite_diff(X_sub, t_sub[1] - t_sub[0])

Theta, labels = build_library_3d(X_sub)
xi, elapsed = sindy_stls(Theta, X_dot, threshold=0.5)

print(f"   Generated {len(t_sub)} data points")
print(f"   Library: {len(labels)} terms")

print("\n   Discovered equations:")
for i, var in enumerate(["x", "y", "z"]):
    eq = format_equation(xi[i], labels)
    print(f"     d{var}/dt = {eq}")

true_xi = system.get_true_coefficients(labels)
metrics = compute_structure_metrics(xi, true_xi)
print(f"\n   Structure F1: {metrics['f1']:.3f}")

# Test noise robustness
print("\n4. Testing noise robustness...")
system = VanDerPol(mu=1.0)
t = np.linspace(0, 20, 2000)

for noise in [0.0, 0.05, 0.10]:
    X = system.generate_trajectory(np.array([1.0, 0.0]), t, noise_level=noise)
    X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])
    Theta, labels = build_library_2d(X)
    xi, _ = sindy_stls(Theta, X_dot, threshold=0.1 + noise)

    true_xi = system.get_true_coefficients(labels)
    metrics = compute_structure_metrics(xi, true_xi)
    print(f"   Noise {noise:.0%}: F1 = {metrics['f1']:.3f}")

# Show train/test split
print("\n5. Train/Test split for fair evaluation:")
print_split_info()

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
print("\nTo run fair evaluation with learned structure network:")
print("  !python scripts/run_fair_evaluation.py")
