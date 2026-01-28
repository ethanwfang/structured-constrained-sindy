"""
Visualization utilities for SINDy.

This module provides functions for plotting trajectories, phase portraits,
and comparison visualizations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_trajectory(
    x: np.ndarray,
    t: np.ndarray,
    var_names: Optional[List[str]] = None,
    title: str = "Trajectory",
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot time series of state variables.

    Parameters
    ----------
    x : np.ndarray
        State trajectory with shape [n_samples, n_vars].
    t : np.ndarray
        Time points.
    var_names : List[str], optional
        Names of state variables.
    title : str, optional
        Plot title.
    figsize : Tuple[int, int], optional
        Figure size.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if matplotlib is available.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for plotting")
        return None

    n_vars = x.shape[1]
    if var_names is None:
        var_names = [f"x{i}" for i in range(n_vars)]

    fig, axes = plt.subplots(1, n_vars, figsize=figsize)
    if n_vars == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, var_names)):
        ax.plot(t, x[:, i], "b-", linewidth=1)
        ax.set_xlabel("Time")
        ax.set_ylabel(name)
        ax.set_title(f"{name}(t)")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_phase_portrait(
    x: np.ndarray,
    var_indices: Tuple[int, int] = (0, 1),
    var_names: Optional[List[str]] = None,
    title: str = "Phase Portrait",
    figsize: Tuple[int, int] = (8, 6),
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot 2D phase portrait.

    Parameters
    ----------
    x : np.ndarray
        State trajectory with shape [n_samples, n_vars].
    var_indices : Tuple[int, int], optional
        Indices of variables to plot.
    var_names : List[str], optional
        Names of state variables.
    title : str, optional
        Plot title.
    figsize : Tuple[int, int], optional
        Figure size.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for plotting")
        return None

    i, j = var_indices
    if var_names is None:
        var_names = [f"x{k}" for k in range(x.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x[:, i], x[:, j], "b-", linewidth=1, alpha=0.7)
    ax.plot(x[0, i], x[0, j], "go", markersize=10, label="Start")
    ax.plot(x[-1, i], x[-1, j], "ro", markersize=10, label="End")
    ax.set_xlabel(var_names[i])
    ax.set_ylabel(var_names[j])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_phase_portrait_3d(
    x: np.ndarray,
    var_indices: Tuple[int, int, int] = (0, 1, 2),
    var_names: Optional[List[str]] = None,
    title: str = "3D Phase Portrait",
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot 3D phase portrait.

    Parameters
    ----------
    x : np.ndarray
        State trajectory with shape [n_samples, n_vars].
    var_indices : Tuple[int, int, int], optional
        Indices of variables to plot.
    var_names : List[str], optional
        Names of state variables.
    title : str, optional
        Plot title.
    figsize : Tuple[int, int], optional
        Figure size.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for plotting")
        return None

    i, j, k = var_indices
    if var_names is None:
        var_names = [f"x{m}" for m in range(x.shape[1])]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x[:, i], x[:, j], x[:, k], "b-", linewidth=0.5, alpha=0.7)
    ax.scatter(x[0, i], x[0, j], x[0, k], c="g", s=100, label="Start")
    ax.scatter(x[-1, i], x[-1, j], x[-1, k], c="r", s=100, label="End")
    ax.set_xlabel(var_names[i])
    ax.set_ylabel(var_names[j])
    ax.set_zlabel(var_names[k])
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_coefficient_comparison(
    xi_pred: np.ndarray,
    xi_true: np.ndarray,
    term_names: List[str],
    equation_idx: int = 0,
    title: str = "Coefficient Comparison",
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot bar chart comparing predicted and true coefficients.

    Parameters
    ----------
    xi_pred : np.ndarray
        Predicted coefficient matrix.
    xi_true : np.ndarray
        True coefficient matrix.
    term_names : List[str]
        Names of library terms.
    equation_idx : int, optional
        Which equation to plot.
    title : str, optional
        Plot title.
    figsize : Tuple[int, int], optional
        Figure size.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for plotting")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(term_names))
    width = 0.35

    ax.bar(x_pos - width / 2, xi_pred[equation_idx, :], width, label="Predicted", alpha=0.7)
    ax.bar(x_pos + width / 2, xi_true[equation_idx, :], width, label="True", alpha=0.7)

    ax.set_xlabel("Library Terms")
    ax.set_ylabel("Coefficient Value")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(term_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_threshold_ablation(
    results: Dict,
    metric: str = "f1",
    title: str = "Threshold Ablation Study",
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot threshold ablation study results.

    Parameters
    ----------
    results : Dict
        Results dictionary from threshold ablation study.
    metric : str, optional
        Metric to plot ('f1', 'mae', 'recall', etc.).
    title : str, optional
        Plot title.
    figsize : Tuple[int, int], optional
        Figure size.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for plotting")
        return None

    systems = list(results.keys())
    n_systems = len(systems)

    fig, axes = plt.subplots(1, n_systems, figsize=figsize)
    if n_systems == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))

    for idx, (ax, system_name) in enumerate(zip(axes, systems)):
        system_data = results[system_name]
        noise_levels = sorted(system_data.keys())

        for noise, color in zip(noise_levels, colors):
            data = system_data[noise]
            thresholds = [d["threshold"] for d in data]
            values = [d[metric] for d in data]
            ax.plot(
                thresholds,
                values,
                "o-",
                color=color,
                label=f"{noise:.0%} noise",
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric.upper())
        ax.set_title(system_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_reconstruction_comparison(
    x_true: np.ndarray,
    x_pred_std: np.ndarray,
    x_pred_sc: np.ndarray,
    t: np.ndarray,
    var_idx: int = 0,
    var_name: str = "x",
    title: str = "Reconstruction Comparison",
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot comparison of trajectory reconstructions.

    Parameters
    ----------
    x_true : np.ndarray
        True trajectory.
    x_pred_std : np.ndarray
        Standard SINDy reconstruction.
    x_pred_sc : np.ndarray
        Structure-Constrained SINDy reconstruction.
    t : np.ndarray
        Time points.
    var_idx : int, optional
        Variable index to plot.
    var_name : str, optional
        Variable name.
    title : str, optional
        Plot title.
    figsize : Tuple[int, int], optional
        Figure size.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for plotting")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t, x_true[:, var_idx], "k-", linewidth=2, label="True", alpha=0.8)
    ax.plot(t, x_pred_std[:, var_idx], "b--", linewidth=1.5, label="Standard SINDy", alpha=0.7)
    ax.plot(t, x_pred_sc[:, var_idx], "r:", linewidth=1.5, label="Structure-Constrained", alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel(var_name)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_metrics_comparison(
    metrics_std: Dict,
    metrics_sc: Dict,
    title: str = "Performance Comparison",
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot comparison of metrics between methods.

    Parameters
    ----------
    metrics_std : Dict
        Metrics from standard SINDy.
    metrics_sc : Dict
        Metrics from Structure-Constrained SINDy.
    title : str, optional
        Plot title.
    figsize : Tuple[int, int], optional
        Figure size.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for plotting")
        return None

    # Select metrics to plot
    metric_names = ["precision", "recall", "f1"]
    values_std = [metrics_std.get(m, 0) for m in metric_names]
    values_sc = [metrics_sc.get(m, 0) for m in metric_names]

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(metric_names))
    width = 0.35

    ax.bar(x_pos - width / 2, values_std, width, label="Standard SINDy", alpha=0.7)
    ax.bar(x_pos + width / 2, values_sc, width, label="Structure-Constrained", alpha=0.7)

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if show:
        plt.show()

    return fig
