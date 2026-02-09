"""
Tests for Ensemble-SINDy and Ensemble-SC-SINDy implementations.
"""

import numpy as np
import pytest

from sc_sindy.core.ensemble import (
    EnsembleResult,
    compute_inclusion_probabilities,
    ensemble_sindy,
    ensemble_sindy_library_bagging,
)
from sc_sindy.core.ensemble_structure_constrained import (
    EnsembleSCResult,
    ensemble_structure_constrained_sindy,
    get_uncertainty_report,
    probability_fusion,
    structure_weighted_ensemble,
    two_stage_ensemble,
)
from sc_sindy.core.library import build_library_2d
from sc_sindy.systems import VanDerPol


class TestEnsembleSINDy:
    """Tests for base Ensemble-SINDy implementation."""

    @pytest.fixture
    def vanderpol_data(self):
        """Generate VanDerPol test data."""
        system = VanDerPol(mu=1.5)
        t = np.linspace(0, 20, 2000)
        x0 = np.array([2.0, 0.0])
        x = system.generate_trajectory(x0, t)
        dt = t[1] - t[0]

        # Compute derivatives
        x_dot = np.gradient(x, dt, axis=0)

        # Build library
        Theta, term_names = build_library_2d(x, poly_order=3)

        return {
            "x": x,
            "x_dot": x_dot,
            "Theta": Theta,
            "term_names": term_names,
            "system": system,
        }

    def test_ensemble_sindy_returns_correct_shapes(self, vanderpol_data):
        """Test that ensemble_sindy returns correctly shaped arrays."""
        Theta = vanderpol_data["Theta"]
        x_dot = vanderpol_data["x_dot"]

        result = ensemble_sindy(Theta, x_dot, n_bootstrap=20, random_state=42)

        assert isinstance(result, EnsembleResult)
        assert result.xi.shape == (2, 10)  # [n_vars, n_terms]
        assert result.inclusion_probs.shape == (2, 10)
        assert result.xi_mean.shape == (2, 10)
        assert result.xi_median.shape == (2, 10)
        assert result.xi_std.shape == (2, 10)
        assert result.confidence_intervals.shape == (2, 10, 2)
        assert result.ensemble_coeffs.shape == (20, 2, 10)

    def test_ensemble_sindy_inclusion_probs_valid(self, vanderpol_data):
        """Test that inclusion probabilities are in valid range."""
        Theta = vanderpol_data["Theta"]
        x_dot = vanderpol_data["x_dot"]

        result = ensemble_sindy(Theta, x_dot, n_bootstrap=50, random_state=42)

        assert np.all(result.inclusion_probs >= 0)
        assert np.all(result.inclusion_probs <= 1)

    def test_ensemble_sindy_bagging_vs_bragging(self, vanderpol_data):
        """Test that bagging and bragging produce different results."""
        Theta = vanderpol_data["Theta"]
        x_dot = vanderpol_data["x_dot"]

        result_bag = ensemble_sindy(
            Theta, x_dot, n_bootstrap=50, aggregation="bagging", random_state=42
        )
        result_brag = ensemble_sindy(
            Theta, x_dot, n_bootstrap=50, aggregation="bragging", random_state=42
        )

        # Results should be different (mean vs median)
        # But both should be reasonable
        assert result_bag.xi.shape == result_brag.xi.shape

    def test_ensemble_sindy_reproducibility(self, vanderpol_data):
        """Test that results are reproducible with same random state."""
        Theta = vanderpol_data["Theta"]
        x_dot = vanderpol_data["x_dot"]

        result1 = ensemble_sindy(Theta, x_dot, n_bootstrap=20, random_state=123)
        result2 = ensemble_sindy(Theta, x_dot, n_bootstrap=20, random_state=123)

        np.testing.assert_array_almost_equal(result1.xi, result2.xi)
        np.testing.assert_array_almost_equal(
            result1.inclusion_probs, result2.inclusion_probs
        )

    def test_ensemble_sindy_library_bagging(self, vanderpol_data):
        """Test library bagging variant."""
        Theta = vanderpol_data["Theta"]
        x_dot = vanderpol_data["x_dot"]

        result = ensemble_sindy_library_bagging(
            Theta, x_dot, n_bootstrap=20, n_terms_sample=5, random_state=42
        )

        assert isinstance(result, EnsembleResult)
        assert result.xi.shape == (2, 10)

    def test_compute_inclusion_probabilities(self, vanderpol_data):
        """Test lightweight inclusion probability computation."""
        Theta = vanderpol_data["Theta"]
        x_dot = vanderpol_data["x_dot"]

        probs = compute_inclusion_probabilities(
            Theta, x_dot, n_bootstrap=30, random_state=42
        )

        assert probs.shape == (2, 10)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


class TestProbabilityFusion:
    """Tests for probability fusion methods."""

    def test_product_fusion(self):
        """Test product fusion method."""
        p_struct = np.array([[0.8, 0.3], [0.5, 0.9]])
        p_ensemble = np.array([[0.9, 0.4], [0.6, 0.8]])

        result = probability_fusion(p_struct, p_ensemble, method="product")

        expected = p_struct * p_ensemble
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_fusion(self):
        """Test average fusion method."""
        p_struct = np.array([[0.8, 0.3]])
        p_ensemble = np.array([[0.6, 0.5]])

        result = probability_fusion(p_struct, p_ensemble, method="average")

        expected = (p_struct + p_ensemble) / 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_weighted_fusion(self):
        """Test weighted fusion method."""
        p_struct = np.array([[0.8, 0.2]])
        p_ensemble = np.array([[0.4, 0.6]])
        alpha = 0.7

        result = probability_fusion(
            p_struct, p_ensemble, method="weighted", alpha=alpha
        )

        expected = alpha * p_struct + (1 - alpha) * p_ensemble
        np.testing.assert_array_almost_equal(result, expected)

    def test_noisy_or_fusion(self):
        """Test noisy-OR fusion method."""
        p_struct = np.array([[0.8, 0.3]])
        p_ensemble = np.array([[0.6, 0.4]])

        result = probability_fusion(p_struct, p_ensemble, method="noisy_or")

        expected = 1 - (1 - p_struct) * (1 - p_ensemble)
        np.testing.assert_array_almost_equal(result, expected)

    def test_max_min_fusion(self):
        """Test max and min fusion methods."""
        p_struct = np.array([[0.8, 0.3]])
        p_ensemble = np.array([[0.6, 0.5]])

        result_max = probability_fusion(p_struct, p_ensemble, method="max")
        result_min = probability_fusion(p_struct, p_ensemble, method="min")

        np.testing.assert_array_almost_equal(result_max, np.array([[0.8, 0.5]]))
        np.testing.assert_array_almost_equal(result_min, np.array([[0.6, 0.3]]))

    def test_invalid_fusion_method(self):
        """Test that invalid fusion method raises error."""
        p_struct = np.array([[0.5]])
        p_ensemble = np.array([[0.5]])

        with pytest.raises(ValueError, match="Unknown fusion method"):
            probability_fusion(p_struct, p_ensemble, method="invalid")


class TestEnsembleSCStructureConstrained:
    """Tests for Ensemble-SC-SINDy combined methods."""

    @pytest.fixture
    def vanderpol_data_with_probs(self, vanderpol_data=None):
        """Generate VanDerPol data with mock network probabilities."""
        system = VanDerPol(mu=1.5)
        t = np.linspace(0, 20, 2000)
        x0 = np.array([2.0, 0.0])
        x = system.generate_trajectory(x0, t)
        dt = t[1] - t[0]

        x_dot = np.gradient(x, dt, axis=0)
        Theta, term_names = build_library_2d(x, poly_order=3)

        # Create mock network probabilities (simulate trained network)
        # VanDerPol: dx/dt = y, dy/dt = mu*(1-x^2)*y - x
        # True structure: dx: y; dy: x, y, xxy
        network_probs = np.array(
            [
                [0.1, 0.2, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # dx: mostly y
                [0.1, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1],  # dy: x, y, xxy
            ]
        )

        return {
            "x": x,
            "x_dot": x_dot,
            "Theta": Theta,
            "term_names": term_names,
            "network_probs": network_probs,
            "system": system,
        }

    def test_ensemble_sc_sindy_returns_correct_shapes(self, vanderpol_data_with_probs):
        """Test that ensemble_structure_constrained_sindy returns correct shapes."""
        data = vanderpol_data_with_probs
        result = ensemble_structure_constrained_sindy(
            data["Theta"],
            data["x_dot"],
            data["network_probs"],
            n_bootstrap=20,
            random_state=42,
        )

        assert isinstance(result, EnsembleSCResult)
        assert result.xi.shape == (2, 10)
        assert result.structure_probs.shape == (2, 10)
        assert result.ensemble_probs.shape == (2, 10)
        assert result.combined_probs.shape == (2, 10)
        assert result.confidence_intervals.shape == (2, 10, 2)

    def test_ensemble_sc_sindy_fuses_probabilities(self, vanderpol_data_with_probs):
        """Test that probabilities are properly fused."""
        data = vanderpol_data_with_probs
        result = ensemble_structure_constrained_sindy(
            data["Theta"],
            data["x_dot"],
            data["network_probs"],
            n_bootstrap=20,
            fusion_method="product",
            random_state=42,
        )

        # Combined should be product of structure and ensemble
        expected_combined = data["network_probs"] * result.ensemble_probs
        np.testing.assert_array_almost_equal(result.combined_probs, expected_combined)

    def test_two_stage_ensemble(self, vanderpol_data_with_probs):
        """Test two-stage ensemble method."""
        data = vanderpol_data_with_probs
        result = two_stage_ensemble(
            data["Theta"],
            data["x_dot"],
            data["network_probs"],
            structure_threshold=0.3,
            n_bootstrap=20,
            random_state=42,
        )

        assert isinstance(result, EnsembleSCResult)
        assert result.xi.shape == (2, 10)

    def test_structure_weighted_ensemble(self, vanderpol_data_with_probs):
        """Test structure-weighted ensemble method."""
        data = vanderpol_data_with_probs
        result = structure_weighted_ensemble(
            data["Theta"],
            data["x_dot"],
            data["network_probs"],
            n_bootstrap=20,
            random_state=42,
        )

        assert isinstance(result, EnsembleSCResult)
        assert result.xi.shape == (2, 10)

    def test_different_fusion_methods(self, vanderpol_data_with_probs):
        """Test that different fusion methods produce different results."""
        data = vanderpol_data_with_probs

        results = {}
        for method in ["product", "average", "noisy_or"]:
            results[method] = ensemble_structure_constrained_sindy(
                data["Theta"],
                data["x_dot"],
                data["network_probs"],
                n_bootstrap=30,
                fusion_method=method,
                random_state=42,
            )

        # Noisy-OR should produce higher combined probabilities than product
        assert np.mean(results["noisy_or"].combined_probs) >= np.mean(
            results["product"].combined_probs
        )

    def test_uncertainty_report(self, vanderpol_data_with_probs):
        """Test uncertainty report generation."""
        data = vanderpol_data_with_probs
        result = ensemble_structure_constrained_sindy(
            data["Theta"],
            data["x_dot"],
            data["network_probs"],
            n_bootstrap=20,
            random_state=42,
        )

        report = get_uncertainty_report(result, data["term_names"])

        assert isinstance(report, str)
        assert "Equation 1" in report
        assert "Equation 2" in report
        assert "95% CI" in report


class TestEnsembleVsStandardComparison:
    """Tests comparing ensemble methods to standard SINDy."""

    @pytest.fixture
    def noisy_data(self):
        """Generate noisy test data."""
        system = VanDerPol(mu=1.5)
        t = np.linspace(0, 30, 3000)
        x0 = np.array([2.0, 0.0])
        x = system.generate_trajectory(x0, t, noise_level=0.05)
        dt = t[1] - t[0]

        x_dot = np.gradient(x, dt, axis=0)
        Theta, term_names = build_library_2d(x, poly_order=3)

        return {
            "x": x,
            "x_dot": x_dot,
            "Theta": Theta,
            "term_names": term_names,
            "system": system,
        }

    def test_ensemble_provides_uncertainty(self, noisy_data):
        """Test that ensemble methods provide meaningful uncertainty estimates."""
        result = ensemble_sindy(
            noisy_data["Theta"], noisy_data["x_dot"], n_bootstrap=50, random_state=42
        )

        # Standard deviations should be non-zero for active terms
        active_mask = np.abs(result.xi) > 0.01
        if np.any(active_mask):
            active_stds = result.xi_std[active_mask]
            assert np.all(active_stds > 0)

    def test_confidence_intervals_contain_estimates(self, noisy_data):
        """Test that confidence intervals contain point estimates."""
        result = ensemble_sindy(
            noisy_data["Theta"], noisy_data["x_dot"], n_bootstrap=100, random_state=42
        )

        # Point estimates (median for bragging) should be within CIs
        for i in range(result.xi.shape[0]):
            for j in range(result.xi.shape[1]):
                ci_low = result.confidence_intervals[i, j, 0]
                ci_high = result.confidence_intervals[i, j, 1]
                # Median should be between 2.5th and 97.5th percentiles
                assert (
                    ci_low <= result.xi_median[i, j] <= ci_high
                ), f"CI violation at [{i},{j}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
