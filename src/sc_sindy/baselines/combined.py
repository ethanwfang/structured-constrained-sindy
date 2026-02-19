"""
Combined SC-SINDy + E-SINDy method.

SC-SINDy acts as a preprocessing step to filter the library,
then E-SINDy or Standard SINDy refines on the filtered terms.
"""

from typing import Optional

import numpy as np

try:
    from pysindy import SINDy
    from pysindy.feature_library import PolynomialLibrary, CustomLibrary
    from pysindy.optimizers import STLSQ

    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False


class CombinedSCSINDy:
    """
    Combined SC-SINDy + SINDy method.

    Uses SC-SINDy neural network to predict which terms are likely active,
    then applies SINDy only on those filtered terms.

    This combines:
    - SC-SINDy's learned structural priors (better precision)
    - SINDy's direct coefficient estimation (interpretable)

    Parameters
    ----------
    model : FactorizedStructureNetworkV2
        Trained SC-SINDy model.
    structure_threshold : float, optional
        Threshold for SC-SINDy structure prediction (default: 0.3).
        Lower values include more terms (higher recall).
    stls_threshold : float, optional
        STLSQ sparsity threshold for final refinement (default: 0.1).
    poly_order : int, optional
        Maximum polynomial order (default: 3).
    use_ensemble : bool, optional
        If True, use E-SINDy (ensemble). If False, use standard SINDy.
    n_bootstraps : int, optional
        Number of bootstrap iterations for E-SINDy (default: 50).
    """

    def __init__(
        self,
        model,
        structure_threshold: float = 0.3,
        stls_threshold: float = 0.1,
        poly_order: int = 3,
        use_ensemble: bool = True,
        n_bootstraps: int = 50,
    ):
        if not PYSINDY_AVAILABLE:
            raise ImportError(
                "PySINDy is required. Install with: pip install pysindy"
            )

        self.model = model
        self.structure_threshold = structure_threshold
        self.stls_threshold = stls_threshold
        self.poly_order = poly_order
        self.use_ensemble = use_ensemble
        self.n_bootstraps = n_bootstraps

    def predict_structure(
        self,
        X: np.ndarray,
        t: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict structure using combined SC-SINDy + SINDy pipeline.

        Parameters
        ----------
        X : np.ndarray
            Trajectory with shape [n_samples, n_vars].
        t : np.ndarray, optional
            Time array.

        Returns
        -------
        structure : np.ndarray
            Final structure prediction with shape [n_vars, n_terms].
        """
        n_samples, n_vars = X.shape

        if t is None:
            t = np.arange(n_samples, dtype=float)

        # Step 1: Get SC-SINDy structure predictions
        sc_probs = self.model.predict_structure(X, n_vars, self.poly_order)
        sc_mask = sc_probs > self.structure_threshold

        # Step 2: Build filtered library based on SC-SINDy predictions
        # Get full library for reference
        full_library = PolynomialLibrary(degree=self.poly_order)
        full_library.fit(X)
        n_terms = full_library.n_output_features_
        feature_names = full_library.get_feature_names()

        # Find which terms are predicted active for ANY equation
        active_terms = np.any(sc_mask, axis=0)
        active_indices = np.where(active_terms)[0]

        if len(active_indices) == 0:
            # No terms predicted, return zeros
            return np.zeros((n_vars, n_terms))

        # Step 3: Apply SINDy on filtered library
        if self.use_ensemble:
            final_structure = self._apply_ensemble_sindy(
                X, t, n_vars, n_terms, active_indices
            )
        else:
            final_structure = self._apply_standard_sindy(
                X, t, n_vars, n_terms, active_indices
            )

        return final_structure

    def _apply_standard_sindy(
        self,
        X: np.ndarray,
        t: np.ndarray,
        n_vars: int,
        n_terms: int,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """Apply standard SINDy on filtered terms."""
        # Fit SINDy with full library
        model = SINDy(
            feature_library=PolynomialLibrary(degree=self.poly_order),
            optimizer=STLSQ(threshold=self.stls_threshold),
        )

        try:
            model.fit(X, t=t)
            coeffs = model.coefficients()

            # Mask out terms not in active_indices
            filtered_coeffs = np.zeros_like(coeffs)
            filtered_coeffs[:, active_indices] = coeffs[:, active_indices]

            return (filtered_coeffs != 0).astype(float)

        except Exception:
            return np.zeros((n_vars, n_terms))

    def _apply_ensemble_sindy(
        self,
        X: np.ndarray,
        t: np.ndarray,
        n_vars: int,
        n_terms: int,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """Apply ensemble SINDy on filtered terms."""
        n_samples = X.shape[0]
        selection_counts = np.zeros((n_vars, n_terms))

        n_data_samples = max(1, int(n_samples * 0.8))

        for _ in range(self.n_bootstraps):
            # Sample data indices
            data_indices = np.random.choice(n_samples, n_data_samples, replace=False)
            data_indices = np.sort(data_indices)

            X_sub = X[data_indices]
            t_sub = t[data_indices]

            model = SINDy(
                feature_library=PolynomialLibrary(degree=self.poly_order),
                optimizer=STLSQ(threshold=self.stls_threshold),
            )

            try:
                model.fit(X_sub, t=t_sub)
                coeffs = model.coefficients()

                # Only count terms in active_indices
                for idx in active_indices:
                    if idx < coeffs.shape[1]:
                        selection_counts[:, idx] += (coeffs[:, idx] != 0).astype(float)

            except Exception:
                continue

        # Normalize to get probabilities
        probs = selection_counts / self.n_bootstraps

        return probs

    def predict_structure_with_details(
        self,
        X: np.ndarray,
        t: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Predict structure with detailed intermediate results.

        Returns
        -------
        results : dict
            Contains 'sc_sindy_probs', 'sc_sindy_mask', 'final_structure',
            'n_filtered_terms', 'n_final_terms'.
        """
        n_samples, n_vars = X.shape

        if t is None:
            t = np.arange(n_samples, dtype=float)

        # SC-SINDy predictions
        sc_probs = self.model.predict_structure(X, n_vars, self.poly_order)
        sc_mask = sc_probs > self.structure_threshold

        # Final structure
        final_structure = self.predict_structure(X, t)

        return {
            'sc_sindy_probs': sc_probs,
            'sc_sindy_mask': sc_mask,
            'final_structure': final_structure,
            'n_filtered_terms': int(np.sum(np.any(sc_mask, axis=0))),
            'n_final_terms': int(np.sum(final_structure > 0.5)),
        }
