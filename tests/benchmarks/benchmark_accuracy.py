"""Accuracy benchmarks for SC-SINDy."""

import numpy as np
from typing import Type

from sc_sindy import (
    sindy_stls,
    build_library_2d,
    build_library_3d,
    compute_derivatives_finite_diff,
    compute_structure_metrics,
    compute_coefficient_error,
)
from sc_sindy.systems import (
    DynamicalSystem,
    VanDerPol,
    LotkaVolterra,
    DuffingOscillator,
    Lorenz,
    Rossler,
)


def benchmark_system(
    system_class: Type[DynamicalSystem],
    n_trials: int = 10,
    noise_level: float = 0.0,
) -> dict:
    """Benchmark accuracy on a system."""
    f1_scores = []
    mae_scores = []

    for _ in range(n_trials):
        system = system_class()
        dim = system.dim

        # Random initial condition
        x0 = np.random.randn(dim) * 0.5 + 1.0
        t = np.linspace(0, 20, 2000)
        X = system.generate_trajectory(x0, t, noise_level=noise_level)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])

        # Build library
        if dim == 2:
            Theta, labels = build_library_2d(X)
        else:
            Theta, labels = build_library_3d(X)

        # Run SINDy
        xi, _ = sindy_stls(Theta, X_dot, threshold=0.1)

        # Compute metrics
        true_xi = system.get_true_coefficients(labels)
        metrics = compute_structure_metrics(xi, true_xi, tol=0.01)
        f1_scores.append(metrics["f1"])

        mae = compute_coefficient_error(xi, true_xi)
        mae_scores.append(mae)

    return {
        "system": system_class.__name__,
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
    }


def run_all_benchmarks():
    """Run benchmarks on all systems."""
    systems_2d = [VanDerPol, LotkaVolterra, DuffingOscillator]
    systems_3d = [Lorenz, Rossler]

    print("=" * 60)
    print("SC-SINDy Accuracy Benchmarks")
    print("=" * 60)

    print("\n2D Systems:")
    print("-" * 40)
    for system_class in systems_2d:
        results = benchmark_system(system_class)
        print(
            f"{results['system']:20s} | "
            f"F1: {results['f1_mean']:.3f} +/- {results['f1_std']:.3f} | "
            f"MAE: {results['mae_mean']:.3f}"
        )

    print("\n3D Systems:")
    print("-" * 40)
    for system_class in systems_3d:
        results = benchmark_system(system_class)
        print(
            f"{results['system']:20s} | "
            f"F1: {results['f1_mean']:.3f} +/- {results['f1_std']:.3f} | "
            f"MAE: {results['mae_mean']:.3f}"
        )

    # Noise robustness
    print("\nNoise Robustness (Van der Pol):")
    print("-" * 40)
    for noise in [0.0, 0.01, 0.05, 0.10]:
        results = benchmark_system(VanDerPol, noise_level=noise)
        print(f"Noise {noise:.2f} | F1: {results['f1_mean']:.3f}")


if __name__ == "__main__":
    run_all_benchmarks()
