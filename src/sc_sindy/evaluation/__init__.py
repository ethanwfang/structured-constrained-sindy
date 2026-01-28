"""
Evaluation framework for SC-SINDy without oracle access.

This module provides tools for fair evaluation of SC-SINDy using
learned structure predictions instead of ground truth (oracle).

Key Components
--------------
- Train/test splits at the SYSTEM level (not trajectory level)
- Evaluator class that uses trained predictor, not oracle
- Comprehensive metrics including network prediction quality

Example
-------
>>> from sc_sindy.evaluation import (
...     get_split,
...     SCSINDyEvaluator,
...     compute_all_metrics,
... )
>>> train_systems, test_systems = get_split(dimension=2)
>>> # Train predictor on train_systems, then evaluate on test_systems
>>> evaluator = SCSINDyEvaluator(predictor)
>>> results = evaluator.evaluate_systems([sys() for sys in test_systems])
"""

from .evaluator import EvaluationResult, EvaluationSummary, SCSINDyEvaluator
from .metrics import (
    compute_all_metrics,
    compute_improvement_statistics,
    compute_normalized_coefficient_error,
    compute_sparsity_ratio,
    compute_success_rate,
    results_to_dataframe,
)
from .splits import (
    TEST_SYSTEMS_2D,
    TEST_SYSTEMS_3D,
    TRAIN_SYSTEMS_2D,
    TRAIN_SYSTEMS_3D,
    get_all_systems,
    get_split,
    get_split_info,
    print_split_info,
)

__all__ = [
    # Splits
    "TRAIN_SYSTEMS_2D",
    "TEST_SYSTEMS_2D",
    "TRAIN_SYSTEMS_3D",
    "TEST_SYSTEMS_3D",
    "get_split",
    "get_all_systems",
    "get_split_info",
    "print_split_info",
    # Evaluator
    "SCSINDyEvaluator",
    "EvaluationResult",
    "EvaluationSummary",
    # Metrics
    "compute_all_metrics",
    "compute_normalized_coefficient_error",
    "compute_sparsity_ratio",
    "compute_success_rate",
    "compute_improvement_statistics",
    "results_to_dataframe",
]
