"""End-to-end integration tests for the SC-SINDy pipeline."""

import numpy as np
import pytest

from sc_sindy import (
    sindy_stls,
    build_library_2d,
    build_library_3d,
    compute_derivatives_finite_diff,
    compute_structure_metrics,
)
from sc_sindy.systems import VanDerPol, LotkaVolterra, Lorenz, DuffingOscillator


class TestFullPipeline:
    """End-to-end tests for complete discovery pipeline."""

    @pytest.fixture
    def vanderpol_data(self):
        """Generate Van der Pol test data."""
        system = VanDerPol(mu=1.0)
        t = np.linspace(0, 20, 2000)
        X = system.generate_trajectory(np.array([1.0, 0.0]), t)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])
        return system, t, X, X_dot

    @pytest.fixture
    def lorenz_data(self):
        """Generate Lorenz test data."""
        system = Lorenz(sigma=10, rho=28, beta=8 / 3)
        t = np.linspace(0, 10, 1000)
        X = system.generate_trajectory(np.array([1.0, 1.0, 1.0]), t)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])
        return system, t, X, X_dot

    def test_vanderpol_structure_recovery(self, vanderpol_data):
        """Should recover correct Van der Pol equation structure."""
        system, t, X, X_dot = vanderpol_data
        Theta, labels = build_library_2d(X)
        xi, _ = sindy_stls(Theta, X_dot, threshold=0.1)

        # Check sparsity pattern
        true_xi = system.get_true_coefficients(labels)
        metrics = compute_structure_metrics(xi, true_xi, tol=0.01)

        assert metrics["f1"] > 0.7, f"F1 score too low: {metrics['f1']}"

    def test_lorenz_structure_recovery(self, lorenz_data):
        """Should recover correct Lorenz equation structure."""
        system, t, X, X_dot = lorenz_data
        Theta, labels = build_library_3d(X)
        xi, _ = sindy_stls(Theta, X_dot, threshold=0.1)

        true_xi = system.get_true_coefficients(labels)
        metrics = compute_structure_metrics(xi, true_xi, tol=0.01)

        assert metrics["f1"] > 0.6, f"F1 score too low: {metrics['f1']}"

    @pytest.mark.parametrize(
        "system_class,dim",
        [
            (VanDerPol, 2),
            (LotkaVolterra, 2),
            (DuffingOscillator, 2),
        ],
    )
    def test_2d_systems_pipeline(self, system_class, dim):
        """Test pipeline on various 2D systems."""
        system = system_class()
        x0 = np.random.randn(dim) * 0.5 + 1.0
        t = np.linspace(0, 20, 2000)
        X = system.generate_trajectory(x0, t)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])

        Theta, labels = build_library_2d(X)
        xi, _ = sindy_stls(Theta, X_dot, threshold=0.1)

        # Basic sanity checks
        assert xi.shape[0] == dim
        assert xi.shape[1] == len(labels)
        assert np.sum(np.abs(xi) > 0.01) > 0, "No active terms found"

    @pytest.mark.slow
    def test_noisy_data_recovery(self):
        """Test recovery with noisy data."""
        system = VanDerPol(mu=1.0)
        t = np.linspace(0, 20, 2000)
        X = system.generate_trajectory(np.array([1.0, 0.0]), t, noise_level=0.05)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])

        Theta, labels = build_library_2d(X)
        xi, _ = sindy_stls(Theta, X_dot, threshold=0.15)

        true_xi = system.get_true_coefficients(labels)
        metrics = compute_structure_metrics(xi, true_xi, tol=0.01)

        # Lower threshold for noisy data
        assert metrics["f1"] > 0.5, f"F1 score too low for noisy data: {metrics['f1']}"
