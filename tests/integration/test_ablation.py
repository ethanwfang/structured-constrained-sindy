"""Ablation study validation tests."""

import numpy as np
import pytest

from sc_sindy import (
    sindy_stls,
    build_library_2d,
    compute_derivatives_finite_diff,
    compute_structure_metrics,
)
from sc_sindy.systems import VanDerPol


class TestThresholdAblation:
    """Ablation tests for threshold sensitivity."""

    @pytest.fixture
    def test_data(self):
        """Generate test data."""
        system = VanDerPol(mu=1.0)
        t = np.linspace(0, 20, 2000)
        X = system.generate_trajectory(np.array([1.0, 0.0]), t)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])
        Theta, labels = build_library_2d(X)
        true_xi = system.get_true_coefficients(labels)
        return Theta, X_dot, true_xi

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.2, 0.5])
    def test_threshold_effect(self, test_data, threshold):
        """Higher threshold should produce sparser solutions."""
        Theta, X_dot, true_xi = test_data
        xi, _ = sindy_stls(Theta, X_dot, threshold=threshold)

        n_active = np.sum(np.abs(xi) > 0.01)
        # Higher thresholds should give fewer terms
        if threshold > 0.2:
            assert n_active < 10, "High threshold should be sparse"

    def test_threshold_monotonicity(self, test_data):
        """Sparsity should increase monotonically with threshold."""
        Theta, X_dot, _ = test_data
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
        n_active_list = []

        for threshold in thresholds:
            xi, _ = sindy_stls(Theta, X_dot, threshold=threshold)
            n_active_list.append(np.sum(np.abs(xi) > 0.01))

        # Check monotonically decreasing (allowing equal)
        for i in range(len(n_active_list) - 1):
            assert (
                n_active_list[i] >= n_active_list[i + 1]
            ), "Sparsity should increase with threshold"


class TestMethodComparison:
    """Compare different SINDy methods."""

    @pytest.mark.slow
    def test_stls_vs_ridge(self):
        """Compare STLS and Ridge regression."""
        from sc_sindy import sindy_ridge

        system = VanDerPol(mu=1.0)
        t = np.linspace(0, 20, 2000)
        X = system.generate_trajectory(np.array([1.0, 0.0]), t)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])
        Theta, labels = build_library_2d(X)

        xi_stls, _ = sindy_stls(Theta, X_dot, threshold=0.1)
        xi_ridge = sindy_ridge(Theta, X_dot, alpha=0.1)

        # STLS should be sparser
        n_stls = np.sum(np.abs(xi_stls) > 0.01)
        n_ridge = np.sum(np.abs(xi_ridge) > 0.01)

        assert n_stls <= n_ridge, "STLS should produce sparser solution"
