"""Speed benchmarks for SC-SINDy."""

import time
import numpy as np
from typing import Callable

from sc_sindy import (
    sindy_stls,
    build_library_2d,
    build_library_3d,
    compute_derivatives_finite_diff,
)
from sc_sindy.systems import VanDerPol, Lorenz


def benchmark(func: Callable, n_runs: int = 10) -> dict:
    """Run benchmark and return timing statistics."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def benchmark_sindy_2d():
    """Benchmark SINDy on 2D systems."""
    system = VanDerPol(mu=1.0)
    t = np.linspace(0, 20, 2000)
    X = system.generate_trajectory(np.array([1.0, 0.0]), t)
    X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])
    Theta, labels = build_library_2d(X)

    def run():
        sindy_stls(Theta, X_dot, threshold=0.1)

    results = benchmark(run)
    print(f"SINDy 2D: {results['mean']*1000:.2f} +/- {results['std']*1000:.2f} ms")
    return results


def benchmark_sindy_3d():
    """Benchmark SINDy on 3D systems."""
    system = Lorenz(sigma=10, rho=28, beta=8 / 3)
    t = np.linspace(0, 10, 1000)
    X = system.generate_trajectory(np.array([1.0, 1.0, 1.0]), t)
    X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])
    Theta, labels = build_library_3d(X)

    def run():
        sindy_stls(Theta, X_dot, threshold=0.1)

    results = benchmark(run)
    print(f"SINDy 3D: {results['mean']*1000:.2f} +/- {results['std']*1000:.2f} ms")
    return results


def benchmark_library_construction():
    """Benchmark library construction."""
    X_2d = np.random.randn(2000, 2)
    X_3d = np.random.randn(1000, 3)

    def run_2d():
        build_library_2d(X_2d)

    def run_3d():
        build_library_3d(X_3d)

    results_2d = benchmark(run_2d)
    results_3d = benchmark(run_3d)

    print(f"Library 2D: {results_2d['mean']*1000:.2f} ms")
    print(f"Library 3D: {results_3d['mean']*1000:.2f} ms")


if __name__ == "__main__":
    print("=" * 50)
    print("SC-SINDy Speed Benchmarks")
    print("=" * 50)

    benchmark_sindy_2d()
    benchmark_sindy_3d()
    benchmark_library_construction()
