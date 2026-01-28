"""
Data loading and saving utilities.

This module provides functions for loading and saving data,
including real-world datasets and model checkpoints.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_lynx_hare_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load historical Lynx-Hare population data (1845-1935).

    The famous Hudson Bay Company dataset showing predator-prey dynamics.

    Returns
    -------
    x : np.ndarray
        Population data with shape [n_years, 2] where columns are [Hare, Lynx].
        Values are normalized to [0, 1] range.
    years : np.ndarray
        Year values.

    Examples
    --------
    >>> x, years = load_lynx_hare_data()
    >>> print(f"Data from {years[0]} to {years[-1]}")
    """
    years = np.array(
        [
            1845,
            1846,
            1847,
            1848,
            1849,
            1850,
            1851,
            1852,
            1853,
            1854,
            1855,
            1856,
            1857,
            1858,
            1859,
            1860,
            1861,
            1862,
            1863,
            1864,
            1865,
            1866,
            1867,
            1868,
            1869,
            1870,
            1871,
            1872,
            1873,
            1874,
            1875,
            1876,
            1877,
            1878,
            1879,
            1880,
            1881,
            1882,
            1883,
            1884,
            1885,
            1886,
            1887,
            1888,
            1889,
            1890,
            1891,
            1892,
            1893,
            1894,
            1895,
            1896,
            1897,
            1898,
            1899,
            1900,
            1901,
            1902,
            1903,
            1904,
            1905,
            1906,
            1907,
            1908,
            1909,
            1910,
            1911,
            1912,
            1913,
            1914,
            1915,
            1916,
            1917,
            1918,
            1919,
            1920,
            1921,
            1922,
            1923,
            1924,
            1925,
            1926,
            1927,
            1928,
            1929,
            1930,
            1931,
            1932,
            1933,
            1934,
            1935,
        ]
    )

    hare = np.array(
        [
            30.0,
            47.2,
            70.2,
            77.4,
            36.3,
            20.6,
            18.1,
            21.4,
            22.0,
            25.4,
            27.1,
            40.3,
            57.0,
            76.6,
            52.3,
            19.5,
            11.2,
            7.6,
            14.6,
            16.2,
            24.7,
            36.3,
            50.6,
            65.0,
            67.0,
            40.8,
            15.3,
            9.1,
            10.4,
            8.6,
            7.4,
            8.0,
            12.3,
            19.5,
            45.7,
            51.1,
            29.7,
            15.8,
            9.7,
            10.1,
            8.6,
            7.4,
            8.0,
            11.4,
            16.5,
            34.9,
            50.7,
            44.3,
            17.0,
            11.3,
            11.2,
            7.8,
            6.4,
            5.6,
            6.8,
            8.4,
            12.8,
            25.4,
            47.6,
            61.3,
            67.0,
            44.0,
            17.0,
            11.3,
            8.3,
            5.9,
            4.2,
            4.8,
            7.1,
            9.8,
            35.2,
            59.4,
            62.0,
            39.2,
            13.2,
            10.8,
            8.3,
            9.1,
            7.4,
            6.0,
            5.8,
            6.6,
            11.6,
            18.9,
            31.0,
            58.0,
            80.3,
            66.6,
            38.9,
            17.0,
            14.6,
        ]
    )

    lynx = np.array(
        [
            4.0,
            6.1,
            9.8,
            35.2,
            59.4,
            41.7,
            19.0,
            13.0,
            8.3,
            9.1,
            7.4,
            8.0,
            12.3,
            19.5,
            45.7,
            51.1,
            29.7,
            15.8,
            9.7,
            10.1,
            8.6,
            7.4,
            8.0,
            25.4,
            62.0,
            67.0,
            44.0,
            17.0,
            11.3,
            8.3,
            5.9,
            4.2,
            4.8,
            7.1,
            9.8,
            35.2,
            59.4,
            62.0,
            39.2,
            13.2,
            10.8,
            8.3,
            9.1,
            7.4,
            6.0,
            5.8,
            6.6,
            11.6,
            18.9,
            31.0,
            58.0,
            80.3,
            66.6,
            38.9,
            17.0,
            14.6,
            11.2,
            7.6,
            6.2,
            5.9,
            4.8,
            4.5,
            5.1,
            6.4,
            8.1,
            19.5,
            36.4,
            61.5,
            78.2,
            59.7,
            26.3,
            13.4,
            6.4,
            3.4,
            4.0,
            4.5,
            6.8,
            14.3,
            24.8,
            55.1,
            78.6,
            71.3,
            41.1,
            23.1,
            13.4,
            7.1,
            4.8,
            4.2,
            4.6,
            5.7,
            6.8,
        ]
    )

    # Normalize to [0, 1]
    hare = hare / 100.0
    lynx = lynx / 100.0

    return np.column_stack([hare, lynx]), years


def save_model_results(results: Dict, filepath: str, include_coefficients: bool = True) -> None:
    """
    Save SINDy results to JSON file.

    Parameters
    ----------
    results : Dict
        Dictionary containing results (coefficients, metrics, etc.).
    filepath : str
        Path to save file.
    include_coefficients : bool, optional
        Whether to include coefficient arrays.
    """

    # Convert numpy arrays to lists for JSON serialization
    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        else:
            return obj

    serializable = convert_arrays(results)

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)


def load_model_results(filepath: str) -> Dict:
    """
    Load SINDy results from JSON file.

    Parameters
    ----------
    filepath : str
        Path to results file.

    Returns
    -------
    results : Dict
        Loaded results dictionary.
    """
    with open(filepath) as f:
        results = json.load(f)

    # Convert lists back to numpy arrays where appropriate
    def convert_lists(obj):
        if isinstance(obj, dict):
            return {k: convert_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list) and len(obj) > 0:
            if isinstance(obj[0], list):
                return np.array(obj)
            elif all(isinstance(x, (int, float)) for x in obj):
                return np.array(obj)
        return obj

    return convert_lists(results)


def save_coefficients(
    xi: np.ndarray, term_names: list, filepath: str, var_names: Optional[list] = None
) -> None:
    """
    Save coefficient matrix to CSV file.

    Parameters
    ----------
    xi : np.ndarray
        Coefficient matrix with shape [n_vars, n_terms].
    term_names : list
        Names of library terms.
    filepath : str
        Path to save file.
    var_names : list, optional
        Names of state variables.
    """
    n_vars = xi.shape[0]
    if var_names is None:
        var_names = [f'd{chr(ord("x") + i)}/dt' for i in range(n_vars)]

    with open(filepath, "w") as f:
        # Header
        f.write("," + ",".join(term_names) + "\n")
        # Data rows
        for i, var in enumerate(var_names):
            row = [var] + [f"{c:.6f}" for c in xi[i, :]]
            f.write(",".join(row) + "\n")


def load_coefficients(filepath: str) -> Tuple[np.ndarray, list, list]:
    """
    Load coefficient matrix from CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file.

    Returns
    -------
    xi : np.ndarray
        Coefficient matrix.
    term_names : list
        Names of library terms.
    var_names : list
        Names of state variables.
    """
    with open(filepath) as f:
        lines = f.readlines()

    term_names = lines[0].strip().split(",")[1:]
    var_names = []
    coefficients = []

    for line in lines[1:]:
        parts = line.strip().split(",")
        var_names.append(parts[0])
        coefficients.append([float(x) for x in parts[1:]])

    return np.array(coefficients), term_names, var_names


def save_trajectory(
    x: np.ndarray, t: np.ndarray, filepath: str, var_names: Optional[list] = None
) -> None:
    """
    Save trajectory data to CSV file.

    Parameters
    ----------
    x : np.ndarray
        State trajectory with shape [n_samples, n_vars].
    t : np.ndarray
        Time points.
    filepath : str
        Path to save file.
    var_names : list, optional
        Names of state variables.
    """
    n_vars = x.shape[1]
    if var_names is None:
        var_names = [f"x{i}" for i in range(n_vars)]

    data = np.column_stack([t, x])
    header = "t," + ",".join(var_names)

    np.savetxt(filepath, data, delimiter=",", header=header, comments="")


def load_trajectory(filepath: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load trajectory data from CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file.

    Returns
    -------
    x : np.ndarray
        State trajectory.
    t : np.ndarray
        Time points.
    var_names : list
        Names of state variables.
    """
    with open(filepath) as f:
        header = f.readline().strip().split(",")

    var_names = header[1:]
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)

    t = data[:, 0]
    x = data[:, 1:]

    return x, t, var_names


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Parameters
    ----------
    path : str
        Path to directory.

    Returns
    -------
    path : Path
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
