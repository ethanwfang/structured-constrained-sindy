"""
Equation printing and formatting utilities.

This module provides functions for displaying discovered equations
in human-readable formats.
"""

from typing import List, Optional

import numpy as np


def print_equations(
    xi: np.ndarray,
    term_names: List[str],
    var_names: Optional[List[str]] = None,
    threshold: float = 1e-6,
    precision: int = 4,
) -> None:
    """
    Print discovered equations in readable format.

    Parameters
    ----------
    xi : np.ndarray
        Coefficient matrix with shape [n_vars, n_terms].
    term_names : List[str]
        Names of library terms.
    var_names : List[str], optional
        Names of state variables. If None, uses ['x', 'y', 'z', ...].
    threshold : float, optional
        Coefficients below this value are not printed (default: 1e-6).
    precision : int, optional
        Number of decimal places (default: 4).

    Examples
    --------
    >>> xi = np.array([[0, 1.0, 0], [-1.0, 1.5, -1.5]])
    >>> term_names = ['1', 'y', 'xxy']
    >>> print_equations(xi, term_names, ['x', 'y'])
    """
    n_vars = xi.shape[0]

    if var_names is None:
        var_names = [chr(ord("x") + i) if i < 3 else f"x{i}" for i in range(n_vars)]

    print("\nDiscovered Equations:")
    print("-" * 50)

    for i, var in enumerate(var_names):
        equation = format_equation(xi[i, :], term_names, threshold, precision)
        print(f"  d{var}/dt = {equation}")


def format_equation(
    coefficients: np.ndarray, term_names: List[str], threshold: float = 1e-6, precision: int = 4
) -> str:
    """
    Format a single equation as a string.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficient vector with shape [n_terms].
    term_names : List[str]
        Names of library terms.
    threshold : float, optional
        Coefficients below this value are ignored.
    precision : int, optional
        Number of decimal places.

    Returns
    -------
    equation : str
        Formatted equation string.
    """
    terms = []

    for coef, name in zip(coefficients, term_names):
        if np.abs(coef) > threshold:
            if name == "1":
                terms.append(f"{coef:.{precision}f}")
            else:
                terms.append(f"{coef:.{precision}f}*{name}")

    if len(terms) == 0:
        return "0"

    # Join terms and clean up formatting
    equation = " + ".join(terms)
    equation = equation.replace("+ -", "- ")
    equation = equation.replace("*1 ", " ")

    return equation


def equations_to_latex(
    xi: np.ndarray,
    term_names: List[str],
    var_names: Optional[List[str]] = None,
    threshold: float = 1e-6,
    precision: int = 3,
) -> List[str]:
    """
    Convert discovered equations to LaTeX format.

    Parameters
    ----------
    xi : np.ndarray
        Coefficient matrix with shape [n_vars, n_terms].
    term_names : List[str]
        Names of library terms.
    var_names : List[str], optional
        Names of state variables.
    threshold : float, optional
        Coefficients below this value are ignored.
    precision : int, optional
        Number of decimal places.

    Returns
    -------
    latex_equations : List[str]
        List of LaTeX-formatted equations.

    Examples
    --------
    >>> xi = np.array([[0, 1.0, 0], [-1.0, 1.5, -1.5]])
    >>> term_names = ['1', 'y', 'xxy']
    >>> latex = equations_to_latex(xi, term_names, ['x', 'y'])
    >>> print(latex[0])
    \\frac{dx}{dt} = 1.000 y
    """
    n_vars = xi.shape[0]

    if var_names is None:
        var_names = [chr(ord("x") + i) if i < 3 else f"x_{i}" for i in range(n_vars)]

    # Term name to LaTeX mapping
    latex_terms = {}
    for name in term_names:
        latex_name = name
        # Convert xx to x^2, xxx to x^3, etc.
        if len(name) > 1 and name[0] == name[1:2]:
            count = len(name)
            base = name[0]
            latex_name = f"{base}^{{{count}}}"
        # Convert xy to xy, xxy to x^2y, etc.
        elif len(name) > 1:
            latex_name = _term_to_latex(name)
        latex_terms[name] = latex_name

    equations = []
    for i, var in enumerate(var_names):
        terms = []
        for coef, name in zip(xi[i, :], term_names):
            if np.abs(coef) > threshold:
                latex_term = latex_terms.get(name, name)
                if name == "1":
                    terms.append(f"{coef:.{precision}f}")
                else:
                    if coef > 0:
                        terms.append(f"{coef:.{precision}f} {latex_term}")
                    else:
                        terms.append(f"{coef:.{precision}f} {latex_term}")

        if len(terms) == 0:
            rhs = "0"
        else:
            rhs = " + ".join(terms).replace("+ -", "- ")

        equations.append(f"\\frac{{d{var}}}{{dt}} = {rhs}")

    return equations


def _term_to_latex(name: str) -> str:
    """Convert polynomial term name to LaTeX."""
    if len(name) <= 1:
        return name

    # Count occurrences of each variable
    counts = {}
    for char in name:
        counts[char] = counts.get(char, 0) + 1

    # Build LaTeX representation
    parts = []
    for var in sorted(counts.keys()):
        count = counts[var]
        if count == 1:
            parts.append(var)
        else:
            parts.append(f"{var}^{{{count}}}")

    return "".join(parts)


def compare_equations(
    xi_pred: np.ndarray,
    xi_true: np.ndarray,
    term_names: List[str],
    var_names: Optional[List[str]] = None,
    threshold: float = 1e-6,
    precision: int = 4,
) -> None:
    """
    Print side-by-side comparison of predicted and true equations.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.
    term_names : List[str]
        Names of library terms.
    var_names : List[str], optional
        Names of state variables.
    threshold : float, optional
        Coefficients below this value are ignored.
    precision : int, optional
        Number of decimal places.
    """
    n_vars = xi_pred.shape[0]

    if var_names is None:
        var_names = [chr(ord("x") + i) if i < 3 else f"x{i}" for i in range(n_vars)]

    print("\nEquation Comparison:")
    print("=" * 80)

    for i, var in enumerate(var_names):
        pred_eq = format_equation(xi_pred[i, :], term_names, threshold, precision)
        true_eq = format_equation(xi_true[i, :], term_names, threshold, precision)

        print(f"\nd{var}/dt:")
        print(f"  Predicted: {pred_eq}")
        print(f"  True:      {true_eq}")


def get_active_terms(
    xi: np.ndarray, term_names: List[str], threshold: float = 1e-6
) -> List[List[str]]:
    """
    Get list of active terms for each equation.

    Parameters
    ----------
    xi : np.ndarray
        Coefficient matrix with shape [n_vars, n_terms].
    term_names : List[str]
        Names of library terms.
    threshold : float, optional
        Threshold for considering a term active.

    Returns
    -------
    active_terms : List[List[str]]
        List of active term names for each equation.
    """
    active_terms = []
    for i in range(xi.shape[0]):
        terms = [name for coef, name in zip(xi[i, :], term_names) if np.abs(coef) > threshold]
        active_terms.append(terms)
    return active_terms
