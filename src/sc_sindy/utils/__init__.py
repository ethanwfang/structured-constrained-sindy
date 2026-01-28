"""
Utility functions for Structure-Constrained SINDy.

This module provides visualization, equation formatting, and I/O utilities.
"""

from .equations import (
    compare_equations,
    equations_to_latex,
    format_equation,
    get_active_terms,
    print_equations,
)
from .io import (
    ensure_dir,
    load_coefficients,
    load_lynx_hare_data,
    load_model_results,
    load_trajectory,
    save_coefficients,
    save_model_results,
    save_trajectory,
)
from .visualization import (
    MATPLOTLIB_AVAILABLE,
    plot_coefficient_comparison,
    plot_metrics_comparison,
    plot_phase_portrait,
    plot_phase_portrait_3d,
    plot_reconstruction_comparison,
    plot_threshold_ablation,
    plot_trajectory,
)

__all__ = [
    # Equations
    "print_equations",
    "format_equation",
    "equations_to_latex",
    "compare_equations",
    "get_active_terms",
    # I/O
    "load_lynx_hare_data",
    "save_model_results",
    "load_model_results",
    "save_coefficients",
    "load_coefficients",
    "save_trajectory",
    "load_trajectory",
    "ensure_dir",
    # Visualization
    "plot_trajectory",
    "plot_phase_portrait",
    "plot_phase_portrait_3d",
    "plot_coefficient_comparison",
    "plot_threshold_ablation",
    "plot_reconstruction_comparison",
    "plot_metrics_comparison",
    "MATPLOTLIB_AVAILABLE",
]
