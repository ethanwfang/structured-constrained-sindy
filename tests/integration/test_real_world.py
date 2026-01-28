"""Tests on real-world datasets."""

import numpy as np
import pytest
from pathlib import Path

from sc_sindy import (
    sindy_stls,
    build_library_2d,
    compute_derivatives_spline,
)


class TestRealWorldData:
    """Tests on real-world datasets."""

    @pytest.fixture
    def lynx_hare_data(self):
        """Load Lynx-Hare dataset."""
        data_path = Path("data/raw/lynx_hare.csv")
        if not data_path.exists():
            pytest.skip("Lynx-Hare data not available")

        import pandas as pd

        df = pd.read_csv(data_path)
        return df

    @pytest.mark.slow
    def test_lynx_hare_discovery(self, lynx_hare_data):
        """Should discover predator-prey dynamics from Lynx-Hare data."""
        df = lynx_hare_data

        # Normalize data
        X = df[["hare", "lynx"]].values
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        dt = 1.0  # Annual data
        X_dot = compute_derivatives_spline(X, dt)

        Theta, labels = build_library_2d(X)
        xi, _ = sindy_stls(Theta, X_dot, threshold=0.1)

        # Should find interaction terms (xy)
        xy_idx = labels.index("xy") if "xy" in labels else -1
        if xy_idx >= 0:
            has_interaction = np.any(np.abs(xi[:, xy_idx]) > 0.01)
            assert has_interaction, "Should find predator-prey interaction"


class TestRobustness:
    """Robustness tests across conditions."""

    @pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.10, 0.20])
    def test_noise_tolerance(self, noise_level):
        """Performance should degrade gracefully with noise."""
        from sc_sindy.systems import VanDerPol
        from sc_sindy import compute_derivatives_finite_diff

        system = VanDerPol(mu=1.0)
        t = np.linspace(0, 20, 2000)
        X = system.generate_trajectory(np.array([1.0, 0.0]), t, noise_level=noise_level)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])

        Theta, labels = build_library_2d(X)
        xi, _ = sindy_stls(Theta, X_dot, threshold=0.1 + noise_level)

        # Should still find some structure
        n_active = np.sum(np.abs(xi) > 0.01)
        assert n_active > 0, f"No terms found at noise level {noise_level}"

    @pytest.mark.parametrize("n_samples", [50, 100, 500, 1000])
    def test_sample_efficiency(self, n_samples):
        """Should work with limited data."""
        from sc_sindy.systems import VanDerPol
        from sc_sindy import compute_derivatives_finite_diff

        system = VanDerPol(mu=1.0)
        t = np.linspace(0, n_samples * 0.01, n_samples)
        X = system.generate_trajectory(np.array([1.0, 0.0]), t)
        X_dot = compute_derivatives_finite_diff(X, t[1] - t[0])

        Theta, labels = build_library_2d(X)
        xi, _ = sindy_stls(Theta, X_dot, threshold=0.1)

        # Check basic structure
        assert xi.shape[0] == 2
        assert np.sum(np.abs(xi) > 0.01) > 0
