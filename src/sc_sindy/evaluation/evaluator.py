"""
Main evaluation class for SC-SINDy without oracle access.

This module provides the SCSINDyEvaluator class for fair evaluation
using learned structure predictions instead of ground truth.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from ..core import build_library_2d, build_library_3d, sindy_stls, sindy_structure_constrained
from ..derivatives import compute_derivatives_finite_diff
from ..metrics import compute_coefficient_error, compute_structure_metrics
from ..systems import DynamicalSystem


@dataclass
class EvaluationResult:
    """Container for a single evaluation trial result."""

    system_name: str
    dimension: int
    noise_level: float
    trial_idx: int

    # Standard SINDy results
    standard_f1: float
    standard_precision: float
    standard_recall: float
    standard_coef_error: float
    standard_time: float
    standard_n_terms: int

    # SC-SINDy results
    sc_f1: float
    sc_precision: float
    sc_recall: float
    sc_coef_error: float
    sc_time: float
    sc_n_terms: int

    # Network prediction quality (structure prediction, not SINDy)
    network_f1: float = 0.0
    network_precision: float = 0.0
    network_recall: float = 0.0

    # Derived metrics
    @property
    def f1_improvement(self) -> float:
        """Absolute F1 improvement."""
        return self.sc_f1 - self.standard_f1

    @property
    def relative_f1_improvement(self) -> float:
        """Relative F1 improvement."""
        if self.standard_f1 > 0:
            return (self.sc_f1 - self.standard_f1) / self.standard_f1
        return 0.0

    @property
    def speedup(self) -> float:
        """Computational speedup ratio."""
        if self.sc_time > 0:
            return self.standard_time / self.sc_time
        return 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "system": self.system_name,
            "dimension": self.dimension,
            "noise": self.noise_level,
            "trial": self.trial_idx,
            "std_f1": self.standard_f1,
            "std_precision": self.standard_precision,
            "std_recall": self.standard_recall,
            "std_coef_error": self.standard_coef_error,
            "std_time": self.standard_time,
            "std_n_terms": self.standard_n_terms,
            "sc_f1": self.sc_f1,
            "sc_precision": self.sc_precision,
            "sc_recall": self.sc_recall,
            "sc_coef_error": self.sc_coef_error,
            "sc_time": self.sc_time,
            "sc_n_terms": self.sc_n_terms,
            "network_f1": self.network_f1,
            "network_precision": self.network_precision,
            "network_recall": self.network_recall,
            "f1_improvement": self.f1_improvement,
            "speedup": self.speedup,
        }


@dataclass
class EvaluationSummary:
    """Summary statistics across multiple evaluation trials."""

    system_name: str
    dimension: int
    noise_level: float
    n_trials: int

    # Aggregated standard SINDy metrics
    std_f1_mean: float
    std_f1_std: float
    std_precision_mean: float
    std_recall_mean: float

    # Aggregated SC-SINDy metrics
    sc_f1_mean: float
    sc_f1_std: float
    sc_precision_mean: float
    sc_recall_mean: float

    # Aggregated network metrics
    network_f1_mean: float

    # Improvement metrics
    f1_improvement_mean: float
    speedup_mean: float

    @classmethod
    def from_results(
        cls, results: List[EvaluationResult], system_name: str, noise_level: float
    ) -> "EvaluationSummary":
        """Create summary from list of results."""
        if not results:
            raise ValueError("Cannot create summary from empty results")

        std_f1s = [r.standard_f1 for r in results]
        sc_f1s = [r.sc_f1 for r in results]

        return cls(
            system_name=system_name,
            dimension=results[0].dimension,
            noise_level=noise_level,
            n_trials=len(results),
            std_f1_mean=np.mean(std_f1s),
            std_f1_std=np.std(std_f1s),
            std_precision_mean=np.mean([r.standard_precision for r in results]),
            std_recall_mean=np.mean([r.standard_recall for r in results]),
            sc_f1_mean=np.mean(sc_f1s),
            sc_f1_std=np.std(sc_f1s),
            sc_precision_mean=np.mean([r.sc_precision for r in results]),
            sc_recall_mean=np.mean([r.sc_recall for r in results]),
            network_f1_mean=np.mean([r.network_f1 for r in results]),
            f1_improvement_mean=np.mean([r.f1_improvement for r in results]),
            speedup_mean=np.mean([r.speedup for r in results]),
        )


class SCSINDyEvaluator:
    """
    Evaluator for SC-SINDy using learned structure predictions.

    This class provides fair evaluation WITHOUT oracle access by using
    a trained structure predictor instead of ground truth.

    Parameters
    ----------
    predictor : object, optional
        Trained structure predictor with `predict_from_trajectory(x, dt)` method.
        If None, will only evaluate standard SINDy.
    stls_threshold : float
        Threshold for STLS algorithm (default: 0.1).
    structure_threshold : float
        Threshold for structure network predictions (default: 0.3).

    Examples
    --------
    >>> from sc_sindy.evaluation import SCSINDyEvaluator
    >>> from sc_sindy.network import StructurePredictor
    >>> predictor = StructurePredictor.load("model.pt", "config.json")
    >>> evaluator = SCSINDyEvaluator(predictor)
    >>> results = evaluator.evaluate_system(VanDerPol(), n_trials=10)
    """

    def __init__(
        self,
        predictor: Optional[Any] = None,
        stls_threshold: float = 0.1,
        structure_threshold: float = 0.3,
    ):
        self.predictor = predictor
        self.stls_threshold = stls_threshold
        self.structure_threshold = structure_threshold

    def evaluate_system(
        self,
        system: DynamicalSystem,
        n_trials: int = 10,
        noise_levels: Optional[List[float]] = None,
        t_span: Tuple[float, float] = (0, 50),
        n_points: int = 5000,
        trim_edges: int = 100,
        verbose: bool = False,
    ) -> List[EvaluationResult]:
        """
        Evaluate both standard and SC-SINDy on a system.

        Parameters
        ----------
        system : DynamicalSystem
            System to evaluate on.
        n_trials : int
            Number of test trajectories per noise level.
        noise_levels : List[float], optional
            Noise levels to test. Default: [0.0, 0.05, 0.10, 0.15].
        t_span : Tuple[float, float]
            Time span for trajectory generation.
        n_points : int
            Number of time points.
        trim_edges : int
            Points to trim from trajectory edges.
        verbose : bool
            Print progress information.

        Returns
        -------
        results : List[EvaluationResult]
            Results for each trajectory and noise level.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.10, 0.15]

        results = []
        t = np.linspace(t_span[0], t_span[1], n_points)
        dt = t[1] - t[0]

        # Select appropriate library builder
        if system.dim == 2:
            build_library = build_library_2d
        elif system.dim == 3:
            build_library = build_library_3d
        else:
            raise ValueError(f"Unsupported dimension: {system.dim}")

        # Get term names for true coefficients
        dummy_x = np.random.randn(10, system.dim)
        _, term_names = build_library(dummy_x)
        true_xi = system.get_true_coefficients(term_names)
        true_structure = np.abs(true_xi) > 1e-6

        for noise in noise_levels:
            for trial_idx in range(n_trials):
                if verbose:
                    print(f"  {system.name} | noise={noise:.2f} | trial {trial_idx + 1}/{n_trials}")

                # Generate random initial condition
                x0 = np.random.randn(system.dim) * 2

                try:
                    # Generate trajectory
                    x = system.generate_trajectory(x0, t, noise_level=noise)

                    # Skip failed trajectories
                    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                        continue

                    # Trim edges and compute derivatives
                    if trim_edges > 0:
                        x_trim = x[trim_edges:-trim_edges]
                    else:
                        x_trim = x

                    x_dot = compute_derivatives_finite_diff(x_trim, dt)

                    # Build library
                    Theta, _ = build_library(x_trim)

                    # Standard SINDy
                    t_start = time.time()
                    xi_standard, _ = sindy_stls(Theta, x_dot, threshold=self.stls_threshold)
                    time_standard = time.time() - t_start

                    # SC-SINDy with learned predictions (if predictor available)
                    if self.predictor is not None:
                        network_probs = self.predictor.predict_from_trajectory(x_trim, dt)

                        t_start = time.time()
                        xi_sc, _ = sindy_structure_constrained(
                            Theta,
                            x_dot,
                            network_probs,
                            structure_threshold=self.structure_threshold,
                        )
                        time_sc = time.time() - t_start

                        # Network prediction quality
                        pred_structure = network_probs > 0.5
                        net_metrics = compute_structure_metrics(
                            pred_structure.astype(float), true_structure.astype(float)
                        )
                    else:
                        # No predictor - use standard SINDy results for SC fields
                        xi_sc = xi_standard
                        time_sc = time_standard
                        net_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

                    # Compute metrics
                    metrics_standard = compute_structure_metrics(xi_standard, true_xi)
                    metrics_sc = compute_structure_metrics(xi_sc, true_xi)

                    result = EvaluationResult(
                        system_name=system.name,
                        dimension=system.dim,
                        noise_level=noise,
                        trial_idx=trial_idx,
                        standard_f1=metrics_standard["f1"],
                        standard_precision=metrics_standard["precision"],
                        standard_recall=metrics_standard["recall"],
                        standard_coef_error=compute_coefficient_error(xi_standard, true_xi),
                        standard_time=time_standard,
                        standard_n_terms=int(np.sum(np.abs(xi_standard) > 1e-6)),
                        sc_f1=metrics_sc["f1"],
                        sc_precision=metrics_sc["precision"],
                        sc_recall=metrics_sc["recall"],
                        sc_coef_error=compute_coefficient_error(xi_sc, true_xi),
                        sc_time=time_sc,
                        sc_n_terms=int(np.sum(np.abs(xi_sc) > 1e-6)),
                        network_f1=net_metrics["f1"],
                        network_precision=net_metrics["precision"],
                        network_recall=net_metrics["recall"],
                    )
                    results.append(result)

                except Exception as e:
                    if verbose:
                        print(f"    Warning: Trial failed - {e}")
                    continue

        return results

    def evaluate_systems(
        self,
        systems: List[DynamicalSystem],
        n_trials: int = 10,
        noise_levels: Optional[List[float]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> List[EvaluationResult]:
        """
        Evaluate on multiple systems.

        Parameters
        ----------
        systems : List[DynamicalSystem]
            Systems to evaluate on.
        n_trials : int
            Trials per system per noise level.
        noise_levels : List[float], optional
            Noise levels to test.
        verbose : bool
            Print progress.
        **kwargs
            Additional arguments for evaluate_system.

        Returns
        -------
        results : List[EvaluationResult]
            Combined results for all systems.
        """
        all_results = []

        for i, system in enumerate(systems):
            if verbose:
                print(f"Evaluating {system.name} ({i + 1}/{len(systems)})")

            results = self.evaluate_system(
                system, n_trials=n_trials, noise_levels=noise_levels, verbose=verbose, **kwargs
            )
            all_results.extend(results)

        return all_results

    def summarize_results(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Dict[float, EvaluationSummary]]:
        """
        Create summary statistics from results.

        Parameters
        ----------
        results : List[EvaluationResult]
            Evaluation results.

        Returns
        -------
        summaries : Dict[str, Dict[float, EvaluationSummary]]
            Nested dict: system_name -> noise_level -> summary.
        """
        summaries = {}

        # Group by system and noise level
        for result in results:
            if result.system_name not in summaries:
                summaries[result.system_name] = {}

            if result.noise_level not in summaries[result.system_name]:
                summaries[result.system_name][result.noise_level] = []

            summaries[result.system_name][result.noise_level].append(result)

        # Convert to summary objects
        for system_name in summaries:
            for noise_level in summaries[system_name]:
                result_list = summaries[system_name][noise_level]
                summaries[system_name][noise_level] = EvaluationSummary.from_results(
                    result_list, system_name, noise_level
                )

        return summaries

    def print_summary(self, results: List[EvaluationResult]):
        """Print a formatted summary of results."""
        summaries = self.summarize_results(results)

        print("\n" + "=" * 70)
        print("SC-SINDy Evaluation Summary (No Oracle)")
        print("=" * 70)

        for system_name, noise_summaries in summaries.items():
            print(f"\n{system_name}:")
            print("-" * 50)
            print(
                f"{'Noise':<8} | {'Std F1':<12} | {'SC F1':<12} | {'Net F1':<10} | {'Improve':<10}"
            )
            print("-" * 50)

            for noise, summary in sorted(noise_summaries.items()):
                print(
                    f"{noise:<8.2f} | "
                    f"{summary.std_f1_mean:.3f}+/-{summary.std_f1_std:.3f} | "
                    f"{summary.sc_f1_mean:.3f}+/-{summary.sc_f1_std:.3f} | "
                    f"{summary.network_f1_mean:.3f}      | "
                    f"{summary.f1_improvement_mean:+.3f}"
                )
