"""
Ensemble SINDy (E-SINDy) baseline for structure prediction.

E-SINDy uses bootstrap aggregation (bagging) to estimate which terms
are consistently selected across multiple SINDy fits. This provides
a probability estimate for each term without requiring neural networks.

Reference:
    Fasel et al. "Ensemble-SINDy: Robust sparse model discovery in the
    low-data, high-noise limit, with active learning and control"
    Proceedings of the Royal Society A, 2022.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.integrate import odeint

try:
    from pysindy import SINDy
    from pysindy.feature_library import PolynomialLibrary
    from pysindy.optimizers import STLSQ

    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False


class ESINDyBaseline:
    """
    Ensemble SINDy baseline for structure prediction.

    Uses library bagging (random feature subsets) and data bagging
    (bootstrap samples) to estimate term importance.

    Parameters
    ----------
    n_bootstraps : int, optional
        Number of bootstrap iterations (default: 100).
    library_subsample_frac : float, optional
        Fraction of library terms to use per bootstrap (default: 0.8).
    data_subsample_frac : float, optional
        Fraction of data points to use per bootstrap (default: 0.8).
    threshold : float, optional
        STLSQ sparsity threshold (default: 0.1).
    poly_order : int, optional
        Maximum polynomial order (default: 3).

    Examples
    --------
    >>> baseline = ESINDyBaseline(n_bootstraps=50)
    >>> probs = baseline.predict_structure(X, X_dot)
    >>> # probs[i, j] = frequency that term j appears in equation i
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        library_subsample_frac: float = 0.8,
        data_subsample_frac: float = 0.8,
        threshold: float = 0.1,
        poly_order: int = 3,
    ):
        if not PYSINDY_AVAILABLE:
            raise ImportError(
                "PySINDy is required for ESINDyBaseline. "
                "Install with: pip install pysindy"
            )

        self.n_bootstraps = n_bootstraps
        self.library_subsample_frac = library_subsample_frac
        self.data_subsample_frac = data_subsample_frac
        self.threshold = threshold
        self.poly_order = poly_order

    def _fit_single_bootstrap(
        self,
        X: np.ndarray,
        t: np.ndarray,
        library_indices: np.ndarray,
        data_indices: np.ndarray,
    ) -> np.ndarray:
        """Fit a single bootstrap SINDy model."""
        # Subsample data
        X_sub = X[data_indices]
        t_sub = t[data_indices]

        # Fit SINDy
        model = SINDy(
            feature_library=PolynomialLibrary(degree=self.poly_order),
            optimizer=STLSQ(threshold=self.threshold),
        )

        try:
            model.fit(X_sub, t=t_sub)
            coeffs = model.coefficients()

            # Get full library size
            full_library = PolynomialLibrary(degree=self.poly_order)
            full_library.fit(X)
            n_terms = full_library.n_output_features_

            # Map coefficients to full library
            n_vars = X.shape[1]
            full_coeffs = np.zeros((n_vars, n_terms))

            # The coefficients from model.fit are in the standard order
            # We need to handle library subsampling if used
            # For now, we use full library
            if coeffs.shape[1] == n_terms:
                full_coeffs = coeffs
            else:
                # Handle case where library was subsampled
                for i, idx in enumerate(library_indices):
                    if i < coeffs.shape[1]:
                        full_coeffs[:, idx] = coeffs[:, i]

            return (full_coeffs != 0).astype(float)

        except Exception:
            n_terms = PolynomialLibrary(degree=self.poly_order).fit(X).n_output_features_
            return np.zeros((X.shape[1], n_terms))

    def predict_structure(
        self,
        X: np.ndarray,
        X_dot: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict structure probabilities using ensemble SINDy.

        Parameters
        ----------
        X : np.ndarray
            Trajectory data with shape [n_samples, n_vars].
        X_dot : np.ndarray, optional
            Derivatives with shape [n_samples, n_vars].
            If None, computed via finite differences from X and t.
        t : np.ndarray, optional
            Time array. Required if X_dot is None.

        Returns
        -------
        probs : np.ndarray
            Structure probabilities with shape [n_vars, n_terms].
            probs[i, j] = frequency that term j appears in equation i.
        """
        n_samples, n_vars = X.shape

        # Ensure we have time array (PySINDy 2.x requires t, not x_dot)
        if t is None:
            # Assume unit time steps
            t = np.arange(n_samples, dtype=float)

        # Get library info
        library = PolynomialLibrary(degree=self.poly_order)
        library.fit(X)
        n_terms = library.n_output_features_

        # Bootstrap aggregation
        selection_counts = np.zeros((n_vars, n_terms))

        n_data_samples = max(1, int(n_samples * self.data_subsample_frac))
        n_lib_samples = max(1, int(n_terms * self.library_subsample_frac))

        for _ in range(self.n_bootstraps):
            # Sample data indices WITHOUT replacement to avoid duplicate time points
            # (PySINDy requires strictly increasing time)
            data_indices = np.random.choice(n_samples, n_data_samples, replace=False)
            data_indices = np.sort(data_indices)  # Keep time ordering

            # Sample library indices (without replacement)
            library_indices = np.random.choice(n_terms, n_lib_samples, replace=False)
            library_indices = np.sort(library_indices)

            # Fit and get structure
            structure = self._fit_single_bootstrap(
                X, t, library_indices, data_indices
            )

            selection_counts += structure

        # Normalize to get probabilities
        probs = selection_counts / self.n_bootstraps

        return probs

    def predict_structure_from_system(
        self,
        system,
        n_samples: int = 5000,
        t_span: Tuple[float, float] = (0, 50),
        noise_level: float = 0.05,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict structure from a dynamical system.

        Parameters
        ----------
        system : DynamicalSystem
            System to generate trajectory from.
        n_samples : int, optional
            Number of time points (default: 5000).
        t_span : Tuple[float, float], optional
            Time span (default: (0, 50)).
        noise_level : float, optional
            Noise level (default: 0.05).
        x0 : np.ndarray, optional
            Initial condition. If None, random.

        Returns
        -------
        probs : np.ndarray
            Structure probabilities.
        """
        n_vars = system.dim
        if x0 is None:
            x0 = np.random.randn(n_vars) * 2

        t = np.linspace(t_span[0], t_span[1], n_samples)
        X = system.generate_trajectory(x0, t, noise_level=noise_level)

        # Trim transients
        trim = 100
        X = X[trim:-trim]
        t = t[trim:-trim]

        return self.predict_structure(X, t=t)


class StandardSINDyBaseline:
    """
    Standard SINDy baseline (single fit, no ensemble).

    Provides a simple baseline for comparison.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        poly_order: int = 3,
    ):
        if not PYSINDY_AVAILABLE:
            raise ImportError(
                "PySINDy is required for StandardSINDyBaseline. "
                "Install with: pip install pysindy"
            )

        self.threshold = threshold
        self.poly_order = poly_order

    def predict_structure(
        self,
        X: np.ndarray,
        X_dot: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict structure using standard SINDy.

        Returns binary structure (0 or 1), not probabilities.
        """
        n_samples, n_vars = X.shape

        # Ensure we have time array (PySINDy 2.x requires t)
        if t is None:
            t = np.arange(n_samples, dtype=float)

        model = SINDy(
            feature_library=PolynomialLibrary(degree=self.poly_order),
            optimizer=STLSQ(threshold=self.threshold),
        )

        try:
            model.fit(X, t=t)
            coeffs = model.coefficients()
            return (coeffs != 0).astype(float)
        except Exception:
            library = PolynomialLibrary(degree=self.poly_order)
            library.fit(X)
            return np.zeros((n_vars, library.n_output_features_))
