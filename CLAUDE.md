# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Structure-Constrained SINDy (SC-SINDy) is a scientific computing library for discovering governing equations from trajectory data. It implements a two-stage approach combining neural network priors with sparse regression (STLS) to achieve 97-1568x improvement over standard SINDy on challenging dynamical systems.

## Common Commands

```bash
# Installation
make install              # Install in development mode
make install-dev          # Install with all dependencies + pre-commit hooks

# Testing
make test                 # Run unit tests only
make test-all            # Run unit + integration tests
make test-cov            # Generate HTML coverage report

# Code Quality
make lint                # Run ruff, black --check, mypy
make format              # Auto-format with black & ruff --fix

# Documentation
make docs                # Build Sphinx docs
```

Run a single test:
```bash
pytest tests/unit/test_sindy.py::test_function_name -v
```

## Architecture

### Two-Stage SC-SINDy Algorithm

1. **Stage 1 (Network-Guided Filtering)**: Neural network predicts term probabilities, filters out terms where P < structure_threshold (default: 0.3)
2. **Stage 2 (STLS Refinement)**: Standard STLS sparse regression on filtered library with stls_threshold (default: 0.1)

### Key Modules (`src/sc_sindy/`)

| Module | Purpose |
|--------|---------|
| `core/` | STLS algorithm (`sindy.py`) and structure-constrained variant (`structure_constrained.py`) |
| `derivatives/` | Compute derivatives via finite difference or spline interpolation |
| `systems/` | 10+ dynamical systems (oscillators, chaotic, biological) with registry pattern |
| `network/` | PyTorch neural network for structure prediction (optional dependency) |
| `metrics/` | Precision/Recall/F1 for structure, MAE for coefficients, RMSE for reconstruction |
| `evaluation/` | Fair benchmarking framework with train/test splits |

### Dynamical Systems Registry

```python
from sc_sindy import get_system, list_systems
system = get_system("vanderpol", mu=1.5)  # By name with parameters
```

Categories: Oscillators (VanDerPol, Duffing), Chaotic (Lorenz, Rössler), Biological (LotkaVolterra, Glycolysis)

### Conditional PyTorch

Neural network features are optional. Core SINDy algorithms work without PyTorch installed.

## Critical Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `structure_threshold` | 0.3 | Network filter threshold. Robust in range [0.2, 0.8]. Threshold >0.9 causes >50% recall loss |
| `stls_threshold` | 0.1 | STLS sparsity threshold |
| `max_iter` | 10 | STLS iterations |

## Code Style

- Black formatter with 100-char line length
- Ruff linting (rules: E, F, I, W)
- MyPy type checking required for function signatures
- NumPy-style docstrings

## Typical Usage Flow

```python
from sc_sindy import (
    get_system,
    compute_derivatives_finite_diff,
    build_library_2d,
    sindy_stls,
    sindy_structure_constrained,
)

# 1. Get system and generate data
system = get_system("vanderpol")
t, X = system.generate_trajectory(noise_level=0.01)

# 2. Compute derivatives
X_dot = compute_derivatives_finite_diff(X, t)

# 3. Build polynomial library
Theta = build_library_2d(X)

# 4. Run SINDy
coeffs = sindy_stls(Theta, X_dot)
# Or with structure constraints:
coeffs = sindy_structure_constrained(Theta, X_dot, structure_probs)
```
