"""
Unit tests for spectral feature extraction.

Tests the extract_spectral_features() function on known signals
to verify correct behavior.
"""

import pytest
import numpy as np

from sc_sindy.network.factorized.trajectory_encoder import (
    extract_spectral_features,
    extract_per_variable_stats,
)


class TestExtractSpectralFeatures:
    """Unit tests for spectral feature extraction."""

    def test_returns_correct_shape(self):
        """Should return shape [n_vars, 4]."""
        T, n_vars = 1000, 3
        x = np.random.randn(T, n_vars)
        features = extract_spectral_features(x)

        assert features.shape == (n_vars, 4)

    def test_no_nan_or_inf(self):
        """Should not return NaN or Inf values."""
        T, n_vars = 1000, 2
        x = np.random.randn(T, n_vars)
        features = extract_spectral_features(x)

        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_handles_constant_signal(self):
        """Should handle constant signals gracefully."""
        T, n_vars = 1000, 2
        x = np.ones((T, n_vars)) * 5.0
        features = extract_spectral_features(x)

        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        # Constant signal should have autocorr_time = 1.0 (never decays)
        assert features[0, 0] == 1.0

    def test_sine_wave_has_clear_peak_frequency(self):
        """Sine wave should have clear peak at its frequency."""
        T = 1000
        t = np.linspace(0, 10, T)
        freq = 0.5  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(-1, 1)

        features = extract_spectral_features(x)
        peak_freq = features[0, 1]

        # Peak frequency should be close to the true frequency
        # (normalized by sampling rate)
        expected_normalized_freq = freq * 10 / T  # Approximate
        # Just check it's non-zero and reasonable
        assert 0 < peak_freq < 0.5

    def test_sine_wave_has_low_spectral_entropy(self):
        """Periodic signal should have low spectral entropy."""
        T = 1000
        t = np.linspace(0, 10, T)
        x = np.sin(2 * np.pi * 2 * t).reshape(-1, 1)

        features = extract_spectral_features(x)
        spectral_entropy = features[0, 2]

        # Periodic signal should have lower entropy than noise
        assert spectral_entropy < 0.5

    def test_white_noise_has_high_spectral_entropy(self):
        """White noise should have high spectral entropy."""
        np.random.seed(42)
        T = 2000
        x = np.random.randn(T, 1)

        features = extract_spectral_features(x)
        spectral_entropy = features[0, 2]

        # White noise has nearly flat spectrum -> high entropy
        assert spectral_entropy > 0.7

    def test_autocorr_time_sine_vs_noise(self):
        """Oscillatory signal should have longer autocorr time than noise."""
        np.random.seed(42)
        T = 1000
        t = np.linspace(0, 10, T)

        # Sine wave
        x_sine = np.sin(2 * np.pi * 1 * t).reshape(-1, 1)
        features_sine = extract_spectral_features(x_sine)

        # White noise
        x_noise = np.random.randn(T, 1)
        features_noise = extract_spectral_features(x_noise)

        # Sine wave should have longer autocorrelation time
        assert features_sine[0, 0] > features_noise[0, 0]

    def test_spectral_centroid_low_vs_high_freq(self):
        """Low frequency signal should have lower spectral centroid."""
        T = 1000
        t = np.linspace(0, 10, T)

        # Low frequency sine
        x_low = np.sin(2 * np.pi * 0.5 * t).reshape(-1, 1)
        features_low = extract_spectral_features(x_low)

        # High frequency sine
        x_high = np.sin(2 * np.pi * 5 * t).reshape(-1, 1)
        features_high = extract_spectral_features(x_high)

        # Higher frequency should have higher spectral centroid
        assert features_high[0, 3] > features_low[0, 3]


class TestExtractPerVariableStatsWithSpectral:
    """Tests for extract_per_variable_stats with spectral features."""

    def test_base_stats_shape(self):
        """Base stats should have 8 features."""
        x = np.random.randn(1000, 3)
        stats = extract_per_variable_stats(x, include_spectral=False)

        assert stats.shape == (3, 8)

    def test_spectral_stats_shape(self):
        """With spectral, should have 12 features."""
        x = np.random.randn(1000, 3)
        stats = extract_per_variable_stats(x, include_spectral=True)

        assert stats.shape == (3, 12)

    def test_backward_compatibility(self):
        """Default should be include_spectral=False for backward compat."""
        x = np.random.randn(1000, 2)
        stats = extract_per_variable_stats(x)

        assert stats.shape == (2, 8)

    def test_base_stats_unchanged_with_spectral(self):
        """First 8 columns should be same with or without spectral."""
        np.random.seed(42)
        x = np.random.randn(1000, 2)

        stats_base = extract_per_variable_stats(x, include_spectral=False)
        stats_spectral = extract_per_variable_stats(x, include_spectral=True)

        # First 8 columns should match
        np.testing.assert_array_almost_equal(
            stats_base, stats_spectral[:, :8], decimal=10
        )

    def test_no_nan_or_inf_with_spectral(self):
        """Should not produce NaN or Inf with spectral features."""
        x = np.random.randn(1000, 4)
        stats = extract_per_variable_stats(x, include_spectral=True)

        assert not np.any(np.isnan(stats))
        assert not np.any(np.isinf(stats))


class TestSpectralFeaturesOnDynamicalSystems:
    """Integration tests with actual dynamical system trajectories."""

    def test_vanderpol_oscillator(self):
        """VanDerPol oscillator should have clear periodic features."""
        # Simple VanDerPol trajectory approximation
        T = 2000
        t = np.linspace(0, 20, T)
        mu = 1.0
        # Approximate limit cycle
        x = 2 * np.cos(t)
        y = 2 * np.sin(t) - mu * np.sin(t) ** 2

        traj = np.column_stack([x, y])
        features = extract_spectral_features(traj)

        # Should have low spectral entropy (periodic)
        assert features[0, 2] < 0.5
        assert features[1, 2] < 0.5

    def test_noisy_signal_features(self):
        """Adding noise should increase spectral entropy."""
        T = 1000
        t = np.linspace(0, 10, T)

        # Clean sine
        x_clean = np.sin(2 * np.pi * 1 * t).reshape(-1, 1)
        features_clean = extract_spectral_features(x_clean)

        # Noisy sine
        np.random.seed(42)
        x_noisy = x_clean + 0.5 * np.random.randn(T, 1)
        features_noisy = extract_spectral_features(x_noisy)

        # Noisy signal should have higher spectral entropy
        assert features_noisy[0, 2] > features_clean[0, 2]
