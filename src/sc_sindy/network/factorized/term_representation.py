"""
Term representation utilities for factorized networks.

This module provides functions to convert between term names (e.g., "xy", "xxx")
and structural representations (lists of (var_idx, power) tuples) that can be
embedded by neural networks regardless of dimension.
"""

from itertools import combinations_with_replacement
from typing import List, Tuple

# Type alias for power representation
# e.g., xy^2 = [(0, 1), (1, 2)] meaning x^1 * y^2
PowerTuple = Tuple[int, int]  # (variable_index, power)
PowerList = List[PowerTuple]


def term_name_to_powers(term_name: str) -> PowerList:
    """
    Convert a term name to a list of (var_idx, power) tuples.

    Parameters
    ----------
    term_name : str
        Term name like "1", "x", "xy", "xxx", "xxy", "xyz", etc.
        Variable names are single characters: x, y, z, w, v, u, ...

    Returns
    -------
    powers : PowerList
        List of (var_idx, power) tuples. Empty list for constant term "1".

    Examples
    --------
    >>> term_name_to_powers("1")
    []
    >>> term_name_to_powers("x")
    [(0, 1)]
    >>> term_name_to_powers("xy")
    [(0, 1), (1, 1)]
    >>> term_name_to_powers("xxx")
    [(0, 3)]
    >>> term_name_to_powers("xxy")
    [(0, 2), (1, 1)]
    >>> term_name_to_powers("xyz")
    [(0, 1), (1, 1), (2, 1)]
    """
    if term_name == "1":
        return []

    # Map variable characters to indices
    var_map = {"x": 0, "y": 1, "z": 2, "w": 3, "v": 4, "u": 5}

    # Count occurrences of each variable
    var_counts = {}
    for char in term_name:
        if char in var_map:
            var_idx = var_map[char]
            var_counts[var_idx] = var_counts.get(var_idx, 0) + 1

    # Convert to sorted list of (var_idx, power) tuples
    powers = [(var_idx, power) for var_idx, power in sorted(var_counts.items())]
    return powers


def powers_to_term_name(powers: PowerList) -> str:
    """
    Convert a power list back to a term name.

    Parameters
    ----------
    powers : PowerList
        List of (var_idx, power) tuples.

    Returns
    -------
    term_name : str
        Term name like "1", "x", "xy", "xxx", etc.

    Examples
    --------
    >>> powers_to_term_name([])
    '1'
    >>> powers_to_term_name([(0, 1)])
    'x'
    >>> powers_to_term_name([(0, 1), (1, 1)])
    'xy'
    >>> powers_to_term_name([(0, 3)])
    'xxx'
    >>> powers_to_term_name([(0, 2), (1, 1)])
    'xxy'
    """
    if not powers:
        return "1"

    # Map indices to variable characters
    idx_map = {0: "x", 1: "y", 2: "z", 3: "w", 4: "v", 5: "u"}

    term = ""
    for var_idx, power in sorted(powers):
        if var_idx in idx_map:
            term += idx_map[var_idx] * power
        else:
            # For higher dimensions, use x0, x1, etc. notation
            term += f"x{var_idx}" * power

    return term


def get_library_terms(n_vars: int, poly_order: int) -> List[str]:
    """
    Get all polynomial library term names for given dimension and order.

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    poly_order : int
        Maximum polynomial order.

    Returns
    -------
    terms : List[str]
        List of term names.

    Examples
    --------
    >>> get_library_terms(2, 2)
    ['1', 'x', 'y', 'xx', 'xy', 'yy']
    >>> get_library_terms(3, 1)
    ['1', 'x', 'y', 'z']
    """
    var_names = ["x", "y", "z", "w", "v", "u"][:n_vars]
    if n_vars > 6:
        var_names = [f"x{i}" for i in range(n_vars)]

    terms = ["1"]

    for order in range(1, poly_order + 1):
        for combo in combinations_with_replacement(range(n_vars), order):
            if n_vars <= 6:
                term = "".join(var_names[i] for i in combo)
            else:
                term = "".join(var_names[i] for i in combo)
            terms.append(term)

    return terms


def get_library_powers(n_vars: int, poly_order: int) -> List[PowerList]:
    """
    Get power representations for all library terms.

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    poly_order : int
        Maximum polynomial order.

    Returns
    -------
    powers_list : List[PowerList]
        List of power representations for each term.

    Examples
    --------
    >>> get_library_powers(2, 2)
    [[], [(0, 1)], [(1, 1)], [(0, 2)], [(0, 1), (1, 1)], [(1, 2)]]
    """
    terms = get_library_terms(n_vars, poly_order)
    return [term_name_to_powers(term) for term in terms]


def powers_to_tensor_index(powers: PowerList, n_vars: int, max_power: int = 5) -> List[int]:
    """
    Convert power list to a fixed-size tensor representation.

    This creates a tensor of shape [n_vars] where each element is the power
    for that variable (0 if variable not present).

    Parameters
    ----------
    powers : PowerList
        List of (var_idx, power) tuples.
    n_vars : int
        Number of variables (determines output size).
    max_power : int, optional
        Maximum power value (for bounds checking).

    Returns
    -------
    tensor_idx : List[int]
        Power for each variable, shape [n_vars].

    Examples
    --------
    >>> powers_to_tensor_index([(0, 2), (1, 1)], n_vars=3, max_power=5)
    [2, 1, 0]
    """
    result = [0] * n_vars
    for var_idx, power in powers:
        if var_idx < n_vars:
            result[var_idx] = min(power, max_power)
    return result


def count_library_terms(n_vars: int, poly_order: int) -> int:
    """
    Count the number of terms in a polynomial library.

    Uses the formula: C(n_vars + poly_order, poly_order) where C is binomial.
    This equals sum over k=0..poly_order of C(n_vars + k - 1, k).

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    poly_order : int
        Maximum polynomial order.

    Returns
    -------
    n_terms : int
        Number of library terms.

    Examples
    --------
    >>> count_library_terms(2, 3)  # 2D, order 3
    10
    >>> count_library_terms(3, 2)  # 3D, order 2
    10
    >>> count_library_terms(4, 2)  # 4D, order 2
    15
    """
    from math import comb

    total = 0
    for k in range(poly_order + 1):
        total += comb(n_vars + k - 1, k)
    return total


def get_term_total_order(powers: PowerList) -> int:
    """
    Get the total polynomial order of a term.

    Parameters
    ----------
    powers : PowerList
        Power representation of the term.

    Returns
    -------
    order : int
        Total order (sum of all powers).

    Examples
    --------
    >>> get_term_total_order([])  # constant
    0
    >>> get_term_total_order([(0, 1)])  # x
    1
    >>> get_term_total_order([(0, 2), (1, 1)])  # x^2*y
    3
    """
    return sum(power for _, power in powers)
