"""
Structure recovery metrics.

This module provides metrics for evaluating how well SINDy algorithms
recover the true equation structure (which terms are active).
"""

from typing import Dict

import numpy as np


def compute_structure_metrics(
    xi_pred: np.ndarray, xi_true: np.ndarray, tol: float = 1e-6
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for structure recovery.

    A term is considered 'active' if |coefficient| > tol.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix with shape [n_vars, n_terms].
    xi_true : np.ndarray
        True coefficient matrix with shape [n_vars, n_terms].
    tol : float, optional
        Tolerance for considering a term active (default: 1e-6).

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing:
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1: Harmonic mean of precision and recall
        - n_active_pred: Number of active terms predicted
        - n_active_true: Number of active terms in ground truth
        - accuracy: (TP + TN) / total

    Examples
    --------
    >>> xi_pred = np.array([[0, 1.0, 0], [0, 0, 2.0]])
    >>> xi_true = np.array([[0, 1.0, 0], [0, 0, 1.5]])
    >>> metrics = compute_structure_metrics(xi_pred, xi_true)
    >>> print(f"F1: {metrics['f1']:.3f}")
    """
    pred_active = np.abs(xi_pred) > tol
    true_active = np.abs(xi_true) > tol

    tp = np.sum(pred_active & true_active)
    fp = np.sum(pred_active & ~true_active)
    fn = np.sum(~pred_active & true_active)
    tn = np.sum(~pred_active & ~true_active)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "n_active_pred": int(np.sum(pred_active)),
        "n_active_true": int(np.sum(true_active)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def compute_per_equation_metrics(
    xi_pred: np.ndarray, xi_true: np.ndarray, tol: float = 1e-6
) -> Dict[str, Dict[str, float]]:
    """
    Compute structure metrics for each equation separately.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix with shape [n_vars, n_terms].
    xi_true : np.ndarray
        True coefficient matrix with shape [n_vars, n_terms].
    tol : float, optional
        Tolerance for considering a term active.

    Returns
    -------
    metrics : Dict[str, Dict[str, float]]
        Dictionary mapping equation index to metrics.
    """
    n_vars = xi_pred.shape[0]
    results = {}

    for i in range(n_vars):
        results[f"eq_{i}"] = compute_structure_metrics(
            xi_pred[i : i + 1, :], xi_true[i : i + 1, :], tol=tol
        )

    return results


def structure_similarity(xi_pred: np.ndarray, xi_true: np.ndarray, tol: float = 1e-6) -> float:
    """
    Compute Jaccard similarity between predicted and true structure.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.
    tol : float, optional
        Tolerance for considering a term active.

    Returns
    -------
    similarity : float
        Jaccard similarity coefficient in [0, 1].
    """
    pred_active = np.abs(xi_pred) > tol
    true_active = np.abs(xi_true) > tol

    intersection = np.sum(pred_active & true_active)
    union = np.sum(pred_active | true_active)

    return intersection / union if union > 0 else 1.0


def sparsity_ratio(xi: np.ndarray, tol: float = 1e-6) -> float:
    """
    Compute sparsity ratio (fraction of zero entries).

    Parameters
    ----------
    xi : np.ndarray
        Coefficient matrix.
    tol : float, optional
        Tolerance for considering a term zero.

    Returns
    -------
    sparsity : float
        Fraction of zero entries in [0, 1].
    """
    return np.mean(np.abs(xi) <= tol)


def compare_structures(
    xi_pred: np.ndarray, xi_true: np.ndarray, term_names: list, tol: float = 1e-6
) -> Dict[str, list]:
    """
    Compare predicted and true structures, identifying differences.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.
    term_names : list
        Names of library terms.
    tol : float, optional
        Tolerance for considering a term active.

    Returns
    -------
    comparison : Dict[str, list]
        Dictionary with:
        - correct: Correctly identified active terms
        - missed: False negatives (missed true terms)
        - spurious: False positives (incorrectly included terms)
    """
    pred_active = np.abs(xi_pred) > tol
    true_active = np.abs(xi_true) > tol

    correct = []
    missed = []
    spurious = []

    n_vars = xi_pred.shape[0]
    var_names = [f"eq{i}" for i in range(n_vars)]

    for i in range(n_vars):
        for j, term in enumerate(term_names):
            key = f"{var_names[i]}:{term}"
            if pred_active[i, j] and true_active[i, j]:
                correct.append(key)
            elif not pred_active[i, j] and true_active[i, j]:
                missed.append(key)
            elif pred_active[i, j] and not true_active[i, j]:
                spurious.append(key)

    return {
        "correct": correct,
        "missed": missed,
        "spurious": spurious,
    }
