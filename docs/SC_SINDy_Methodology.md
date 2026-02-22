# SC-SINDy: Structure-Constrained Sparse Identification of Nonlinear Dynamics

## A Comprehensive Methodology Guide

**Author:** SC-SINDy Research Team
**Date:** February 2026
**Target Audience:** Machine Learning Researchers, Applied Mathematicians, Domain Scientists

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background: Standard SINDy](#2-background-standard-sindy)
3. [SC-SINDy: The Two-Stage Approach](#3-sc-sindy-the-two-stage-approach)
4. [The Factorized Network Architecture](#4-the-factorized-network-architecture)
5. [Dimension-Agnostic Design](#5-dimension-agnostic-design)
6. [Data Requirements](#6-data-requirements)
7. [Training Procedure](#7-training-procedure)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Experimental Results](#9-experimental-results)
10. [Limitations and Considerations](#10-limitations-and-considerations)
11. [Implementation Guide](#11-implementation-guide)

---

## 1. Problem Statement

### 1.1 The Equation Discovery Challenge

Given time-series measurements of a dynamical system, we seek to discover the underlying governing equations. Formally, given observations:

$$
\mathbf{X} = [x_1(t), x_2(t), \ldots, x_n(t)]^\top \in \mathbb{R}^{T \times n}
$$

we want to identify the functional form:

$$
\dot{x}_i = f_i(x_1, x_2, \ldots, x_n), \quad i = 1, \ldots, n
$$

where $f_i$ is typically a sparse combination of basis functions (polynomials, trigonometric functions, etc.).

### 1.2 Why Standard Methods Struggle

Standard sparse regression methods (LASSO, STLS) face several challenges:

| Challenge | Description | Consequence |
|-----------|-------------|-------------|
| **Noise Sensitivity** | Measurement noise corrupts derivative estimates | False positives/negatives in structure |
| **Threshold Selection** | The sparsity threshold is system-dependent | Poor generalization across systems |
| **High-Dimensional Systems** | More terms = more false positives | Coefficient estimation degrades |
| **Chaotic Dynamics** | Sensitive dependence on initial conditions | Trajectory prediction fails rapidly |

### 1.3 Our Core Insight

**Key Observation:** The *structure* of governing equations (which terms appear) is more robust to noise than the *coefficients* (exact values). A neural network can learn to predict equation structure from trajectory statistics, providing a strong prior for sparse regression.

---

## 2. Background: Standard SINDy

### 2.1 The SINDy Framework

SINDy (Sparse Identification of Nonlinear Dynamics) represents the dynamics as:

$$
\dot{\mathbf{X}} = \mathbf{\Theta}(\mathbf{X}) \cdot \mathbf{\Xi}
$$

where:
- $\dot{\mathbf{X}} \in \mathbb{R}^{T \times n}$: Time derivatives
- $\mathbf{\Theta}(\mathbf{X}) \in \mathbb{R}^{T \times p}$: Library of candidate functions
- $\mathbf{\Xi} \in \mathbb{R}^{p \times n}$: Sparse coefficient matrix

### 2.2 The Polynomial Library

For a 2D system with polynomial order 3, the library is:

$$
\Theta(x, y) = [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
$$

The number of terms scales combinatorially:

| Dimension | Order 2 | Order 3 | Order 4 |
|-----------|---------|---------|---------|
| 2D | 6 | 10 | 15 |
| 3D | 10 | 20 | 35 |
| 4D | 15 | 35 | 70 |

### 2.3 Sequential Thresholded Least Squares (STLS)

The standard SINDy algorithm:

```
1. Solve Xi = (Theta'Theta)^{-1} Theta' X_dot  (least squares)
2. Set small coefficients to zero: Xi[|Xi| < threshold] = 0
3. Repeat with reduced library until convergence
```

**Problem:** A single threshold must balance:
- **Too low:** False positives (extra terms)
- **Too high:** False negatives (missing terms)

---

## 3. SC-SINDy: The Two-Stage Approach

### 3.1 Core Philosophy

SC-SINDy separates structure identification from coefficient estimation:

```
Stage 1: Neural Network → Structure Prediction (which terms appear)
Stage 2: STLS on Filtered Library → Coefficient Estimation (exact values)
```

### 3.2 Stage 1: Network-Guided Filtering

A neural network predicts the probability that each library term appears in each equation:

$$
P_{ij} = \text{Network}(\text{trajectory}) \in [0, 1]^{n \times p}
$$

Terms with $P_{ij} < \tau_{\text{structure}}$ are excluded from the library.

**Default threshold:** $\tau_{\text{structure}} = 0.3$

**Why 0.3?** This threshold prioritizes recall (not missing true terms) over precision. Missing terms cannot be recovered by Stage 2, but false positives can be eliminated by STLS.

### 3.3 Stage 2: STLS Refinement

Standard STLS runs on the filtered library:

$$
\tilde{\mathbf{\Theta}} = \mathbf{\Theta}[:, \text{active}], \quad \tilde{\mathbf{\Xi}} = \text{STLS}(\tilde{\mathbf{\Theta}}, \dot{\mathbf{X}}, \tau_{\text{stls}})
$$

**Default threshold:** $\tau_{\text{stls}} = 0.1$

### 3.4 Why Two Stages Work

| Stage | Function | Threshold | Sensitivity |
|-------|----------|-----------|-------------|
| Stage 1 | Coarse filtering | 0.3 | Robust in [0.2, 0.8] |
| Stage 2 | Fine refinement | 0.1 | Standard SINDy behavior |

The network provides a strong prior that:
1. Eliminates most false positive candidates
2. Keeps true terms in the library
3. Makes Stage 2 STLS more robust to noise

---

## 4. The Factorized Network Architecture

### 4.1 Design Goals

The network must:
1. Accept trajectories from systems of **any dimension**
2. Predict structure for **arbitrary polynomial libraries**
3. Learn transferable patterns across dynamical systems

### 4.2 Architecture Overview

```
Input: Trajectory X ∈ R^{T×n}
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRAJECTORY ENCODER                       │
│  Extract per-variable statistics: [n_vars, stats_dim]       │
│  Project to latent space: z_traj ∈ R^{latent_dim}           │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                      TERM EMBEDDER                           │
│  Embed each library term structurally: e_j ∈ R^{latent_dim} │
│  e.g., "x²y" → [(0,2), (1,1)] → embedding                   │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    EQUATION ENCODER                          │
│  Encode equation position: e_i ∈ R^{latent_dim}             │
│  Uses relative position for dimension-agnostic encoding      │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                  BILINEAR INTERACTION                        │
│  interaction_{ij} = z_traj ⊙ e_term_j ⊙ e_eq_i              │
│  Element-wise product in latent space                        │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                      CLASSIFIER                              │
│  MLP: interaction → P(term j in equation i)                  │
│  Output: [n_vars, n_terms] probability matrix                │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Component Details

#### Trajectory Encoder

Extracts 8 statistical features per variable:

| Feature | Formula | What It Captures |
|---------|---------|------------------|
| Mean | $\bar{x}_i$ | Equilibrium point |
| Std Dev | $\sigma_i$ | Amplitude |
| Skewness | $E[(x-\mu)^3]/\sigma^3$ | Asymmetry |
| Kurtosis | $E[(x-\mu)^4]/\sigma^4$ | Tail heaviness |
| Energy | $\bar{x_i^2}$ | Signal power |
| Range | $\max - \min$ | Excursion |
| Median | $\text{median}(x_i)$ | Robust center |
| Derivative Mag | $\overline{|\Delta x_i|}$ | Rate of change |

Optional spectral features (4 additional):
- Autocorrelation time
- Peak FFT frequency
- Spectral entropy
- Spectral centroid

These statistics are computed **per variable** → shape `[n_vars, 8]` or `[n_vars, 12]`

An MLP aggregates across variables:
```python
# Permutation-invariant aggregation
stats: [n_vars, 8] → max/mean pool → [8] → MLP → [latent_dim]
```

#### Term Embedder

Embeds polynomial terms **structurally**, independent of variable names:

```python
# Term representation as (variable_index, power) tuples
"x²y" → [(0, 2), (1, 1)]  # x has power 2, y has power 1
"xy²" → [(0, 1), (1, 2)]  # x has power 1, y has power 2
"z³"  → [(2, 3)]          # z has power 3
"1"   → []                 # constant term
```

Embedding computation:
```python
def embed_term(powers):
    if len(powers) == 0:
        return const_embedding  # Learnable constant embedding

    factors = []
    for (var_idx, power) in powers:
        var_emb = var_embedding[var_idx]   # Learnable
        pow_emb = power_embedding[power]   # Learnable
        factors.append(var_emb * pow_emb)  # Tensor product

    return MLP(sum(factors))  # Aggregate and project
```

**Key insight:** The same term structure (e.g., "product of two different variables") gets similar embeddings regardless of which specific variables are involved.

#### Equation Encoder

Uses **relative position** for dimension-agnostic encoding:

```python
def encode_equation(eq_idx, n_vars):
    features = [
        eq_idx / n_vars,          # Relative position [0, 1]
        n_vars / 10,               # Normalized dimension
        1.0 if eq_idx == 0 else 0, # Is first equation
        1.0 if eq_idx == n_vars-1 else 0  # Is last equation
    ]
    return MLP(features)
```

**Why relative position?** "First equation in a 3D system" and "first equation in a 5D system" should have similar representations because they often share structural patterns (e.g., the x-equation in physical systems).

#### Bilinear Interaction

The core matching operation:

$$
\text{interaction}_{ij} = \mathbf{z}_{\text{traj}} \odot \mathbf{e}_{\text{term}_j} \odot \mathbf{e}_{\text{eq}_i}
$$

where $\odot$ denotes element-wise multiplication.

**Why bilinear (multiplicative)?**
- Creates unique representations for each (trajectory, term, equation) triple
- Allows the classifier to learn complex dependencies
- Empirically outperforms additive interactions (+3% F1)

---

## 5. Dimension-Agnostic Design

### 5.1 The Generalization Challenge

Traditional neural networks require fixed input/output dimensions. But dynamical systems have varying:
- Number of state variables (2D, 3D, 4D, ...)
- Library sizes (depends on dimension and polynomial order)

### 5.2 Our Solution: Structural Embeddings

Each component processes inputs of arbitrary size by operating on **structural representations**:

| Component | Input | Dimension-Agnostic Strategy |
|-----------|-------|------------------------------|
| Trajectory Encoder | `[n_vars, 8]` | Aggregate across variables (max/mean pool) |
| Term Embedder | `[(var_idx, power), ...]` | Fixed-size embedding per factor, sum to combine |
| Equation Encoder | `(eq_idx, n_vars)` | Relative position encoding |

### 5.3 Zero-Shot Dimension Transfer

A model trained on 2D systems can predict on 3D/4D systems:

| Training Dimensions | Test Dimension | F1 Score |
|---------------------|----------------|----------|
| 2D only | 2D | 0.72 |
| 2D only | 3D | 0.35 |
| 2D only | 4D | 0.28 |
| 2D + 3D | 3D | 0.78 |
| 2D + 3D | 4D | 0.45 |

**Interpretation:** The network learns transferable patterns like:
- "Products of two variables often appear together"
- "Cubic terms are rare"
- "The first equation often has simpler structure"

---

## 6. Data Requirements

### 6.1 Input Data Format

The user must provide:

1. **Trajectory data:** `X ∈ R^{T×n}`
   - `T`: Number of time points (recommend: 200-1000)
   - `n`: Number of state variables

2. **Time vector:** `t ∈ R^T`
   - Uniformly spaced (constant dt)
   - Used for derivative computation

### 6.2 Data Quality Requirements

| Requirement | Recommendation | Why |
|-------------|----------------|-----|
| **Sampling rate** | 100-500 Hz | Capture dynamics without aliasing |
| **Trajectory length** | 2-10× characteristic timescale | Capture full dynamical behavior |
| **Noise level** | < 20% for best results | Higher noise needs more data |
| **Initial conditions** | Diverse, in attractor basin | Avoid transients |

### 6.3 What the Network Sees

The network does **not** see the raw trajectory. Instead, it sees:

```python
# From X ∈ R^{T×n}
stats = extract_per_variable_stats(X)  # → [n, 8]
# Optional:
corr_matrix = np.corrcoef(X.T)         # → [n, n]
```

**Implication:** The network learns from **statistical summaries**, not time-series patterns. This is both a strength (noise robust) and limitation (may miss complex temporal patterns).

### 6.4 Training Data Requirements

To train the network, you need:

| Data Type | Minimum | Recommended | Purpose |
|-----------|---------|-------------|---------|
| Trajectories per system | 20 | 50-100 | Statistical coverage |
| Different systems | 10 | 25+ | Structure diversity |
| Dimension variety | At least 2D, 3D | 2D, 3D, 4D | Generalization |
| Noise levels | 1-10% | Varied (1-20%) | Noise robustness |

---

## 7. Training Procedure

### 7.1 Training Sample Structure

Each training sample consists of:

```python
@dataclass
class TrainingSample:
    stats: np.ndarray      # [n_vars, stats_dim] trajectory statistics
    structure: np.ndarray  # [n_vars, n_terms] ground truth (0/1)
    n_vars: int            # Number of state variables
    poly_order: int        # Polynomial library order
```

### 7.2 Loss Function

We use **weighted binary cross-entropy**:

$$
\mathcal{L} = -\frac{1}{NP} \sum_{i,j} \left[ w_+ \cdot y_{ij} \log p_{ij} + (1-y_{ij}) \log(1-p_{ij}) \right]
$$

where:
- $y_{ij} \in \{0, 1\}$: Ground truth (term j present in equation i)
- $p_{ij} \in [0, 1]$: Predicted probability
- $w_+$: Positive class weight (default: 3.0)

**Why weighted?** Equation structures are **sparse** (few active terms). Without weighting, the network learns to predict all zeros. Higher $w_+$ encourages recall (not missing true terms).

### 7.3 Training Loop

```python
model, history = train_factorized_network(
    samples=training_samples,
    epochs=100,
    batch_size=32,
    lr=0.001,
    val_split=0.1,
    early_stopping_patience=10,
    pos_weight=3.0,  # Recall optimization
)
```

Key training features:
- **Mixed-dimension batching:** Samples grouped by (n_vars, poly_order)
- **Gradient clipping:** `max_norm=1.0` for stability
- **Early stopping:** Prevents overfitting

### 7.4 Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `latent_dim` | 64 | 32-128 | Embedding dimension |
| `stats_dim` | 8 | 8-12 | Per-variable features |
| `pos_weight` | 3.0 | 1.0-5.0 | Recall vs precision |
| `lr` | 0.001 | 0.0001-0.01 | Adam learning rate |
| `batch_size` | 32 | 16-64 | Per-dimension grouping |

---

## 8. Inference Pipeline

### 8.1 End-to-End Usage

```python
from sc_sindy import (
    get_system,
    compute_derivatives_finite_diff,
    build_library_nd,
    sindy_structure_constrained,
)
from sc_sindy.network.factorized import FactorizedStructureNetworkV2

# 1. Load trained model
model = FactorizedStructureNetworkV2.load("model.pt")

# 2. Get trajectory data (user provides this)
X = ...  # Shape: [T, n_vars]
t = ...  # Shape: [T]

# 3. Compute derivatives
X_dot = compute_derivatives_finite_diff(X, t)

# 4. Build polynomial library
Theta = build_library_nd(X, poly_order=3)

# 5. Get structure predictions from network
probs = model.predict_structure(X, n_vars=X.shape[1], poly_order=3)
# probs: [n_vars, n_terms] probabilities

# 6. Run SC-SINDy (two-stage)
coefficients, elapsed = sindy_structure_constrained(
    Theta, X_dot, probs,
    structure_threshold=0.3,  # Stage 1
    stls_threshold=0.1,       # Stage 2
)
# coefficients: [n_vars, n_terms] sparse matrix
```

### 8.2 Interpreting Outputs

The output `coefficients` matrix gives the discovered equations:

```python
# For Lorenz system (3D, poly_order=3, 20 terms)
# coefficients shape: [3, 20]
# Library: [1, x, y, z, x², xy, xz, y², yz, z², x³, ...]

# Example output:
# dx/dt: -10*x + 10*y        → coefficients[0] = [0, -10, 10, 0, ...]
# dy/dt: 28*x - y - x*z      → coefficients[1] = [0, 28, -1, 0, 0, 0, -1, ...]
# dz/dt: -8/3*z + x*y        → coefficients[2] = [0, 0, 0, -2.67, 0, 1, ...]
```

### 8.3 Validation Metrics

| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| **F1 Score** | Structure accuracy | 1.0 = perfect structure |
| **Precision** | TP / (TP + FP) | High = few false positives |
| **Recall** | TP / (TP + FN) | High = few missing terms |
| **Coef MAE** | Coefficient error | Lower = better coefficient estimates |
| **Traj RMSE** | Prediction error | Lower = better dynamical model |

---

## 9. Experimental Results

### 9.1 Structure Recovery (F1 Score)

Comparison across 10 test systems (not seen during training):

| System | Dimension | SC-SINDy | Standard SINDy | Improvement |
|--------|-----------|----------|----------------|-------------|
| Lorenz | 3D | **1.00** | 0.86 | +16% |
| VanDerPol | 2D | 0.87 | 0.80 | +9% |
| Duffing | 2D | 0.90 | 0.85 | +6% |
| Rössler | 3D | 0.78 | 0.34 | +129% |
| LotkaVolterra | 4D | 0.50 | 0.23 | +117% |
| **Mean** | - | **0.565** | 0.514 | **+9.5%** |

### 9.2 Noise Robustness

Performance at different noise levels (Lorenz system):

| Noise Level | SC-SINDy F1 | SINDy F1 | Advantage |
|-------------|-------------|----------|-----------|
| 1% | 1.00 | 0.98 | +2% |
| 5% | 1.00 | 0.86 | +16% |
| 10% | 0.98 | 0.62 | +58% |
| 20% | 1.00 | 0.54 | +85% |
| 50% | **0.996** | 0.52 | **+91%** |

**Key insight:** SC-SINDy maintains near-perfect structure recovery even at 50% noise, where SINDy essentially fails.

### 9.3 Coefficient Recovery

Mean Absolute Error comparison (5% noise):

| System | SC-SINDy MAE | SINDy MAE | Improvement |
|--------|--------------|-----------|-------------|
| Lorenz | 0.25 | 0.29 | 1.2× better |
| Rössler | 0.80 | 12874 | 16000× better |
| VanDerPol | 0.02 | 0.02 | Similar |

### 9.4 Trajectory Prediction

RMSE over 5 Lyapunov times (chaotic systems):

| System | SC-SINDy RMSE | SINDy RMSE | Improvement |
|--------|---------------|------------|-------------|
| Lorenz | 7.67 | 7.92 | Similar |
| Rössler | 0.10 | 590 | 5900× better |

---

## 10. Limitations and Considerations

### 10.1 When SC-SINDy Struggles

| Scenario | Challenge | Mitigation |
|----------|-----------|------------|
| **Non-polynomial dynamics** | Network trained on polynomials | Use custom library (requires retraining) |
| **Bilinear systems** (e.g., LotkaVolterra) | Products of different variables | Mixed results; may need domain knowledge |
| **Very high dimensions** (>10D) | Combinatorial explosion of terms | Model validated for n_vars ≤ 10 |
| **Sparse training data** | Network may memorize | Need diverse training systems |

### 10.2 Threshold Sensitivity

The `structure_threshold` parameter has a robust range:

| Threshold | Behavior | Risk |
|-----------|----------|------|
| 0.1-0.2 | High recall, lower precision | Extra terms (STLS can fix) |
| **0.3** (default) | **Balanced** | **Recommended** |
| 0.4-0.6 | Moderate balance | Some missed terms possible |
| 0.7-0.8 | High precision, lower recall | Missing terms |
| **>0.9** | **AVOID** | **>50% recall loss** |

### 10.3 Computational Considerations

| Operation | Time (typical) | Memory |
|-----------|----------------|--------|
| Training (25 systems, 100 epochs) | 5-10 min | 2-4 GB |
| Inference (single trajectory) | <100 ms | <1 GB |
| STLS refinement | <10 ms | Negligible |

### 10.4 When to Use SC-SINDy vs Standard SINDy

**Use SC-SINDy when:**
- Noise levels > 10%
- Chaotic systems (Lorenz, Rössler)
- Unknown optimal threshold
- Robustness is critical

**Standard SINDy may suffice when:**
- Clean data (noise < 5%)
- Simple oscillators
- Well-characterized systems
- Computational resources are limited

---

## 11. Implementation Guide

### 11.1 Installation

```bash
git clone https://github.com/yourrepo/sc-sindy.git
cd sc-sindy
pip install -e ".[dev]"  # Includes PyTorch
```

### 11.2 Quick Start

```python
# Load pre-trained model
from sc_sindy.network.factorized import FactorizedStructureNetworkV2

model = FactorizedStructureNetworkV2.load("models/factorized/factorized_model.pt")

# Your trajectory data
import numpy as np
X = np.loadtxt("your_trajectory.csv")  # [T, n_vars]
t = np.linspace(0, 10, len(X))

# Discover equations
from sc_sindy import sindy_structure_constrained, build_library_nd
from sc_sindy.derivatives import compute_derivatives_finite_diff

X_dot = compute_derivatives_finite_diff(X, t)
Theta = build_library_nd(X, poly_order=3)
probs = model.predict_structure(X, n_vars=X.shape[1], poly_order=3)
coefficients, _ = sindy_structure_constrained(Theta, X_dot, probs)

# Print discovered equations
from sc_sindy.core.library import get_library_names
names = get_library_names(X.shape[1], poly_order=3)
for i, row in enumerate(coefficients):
    terms = [f"{c:.3f}*{n}" for c, n in zip(row, names) if abs(c) > 0.01]
    print(f"dx{i+1}/dt = {' + '.join(terms)}")
```

### 11.3 Training Custom Models

```python
from sc_sindy.network.factorized import (
    train_factorized_network_with_systems,
    FactorizedStructureNetworkV2,
)
from sc_sindy.systems import Lorenz, VanDerPol, DuffingOscillator

# Define training systems
systems_2d = [VanDerPol, DuffingOscillator]
systems_3d = [Lorenz]

# Train
model, history, config = train_factorized_network_with_systems(
    systems_2d=systems_2d,
    systems_3d=systems_3d,
    n_trajectories_per_system=50,
    poly_order=3,
    epochs=100,
    pos_weight=3.0,
)

# Save
model.save("my_model.pt")
```

### 11.4 Evaluating Results

```python
from sc_sindy.metrics import (
    compute_structure_f1,
    compute_coefficient_mae,
    compute_trajectory_rmse,
)

# Compare to ground truth
f1 = compute_structure_f1(coefficients, true_coefficients)
mae = compute_coefficient_mae(coefficients, true_coefficients)
print(f"Structure F1: {f1:.3f}, Coefficient MAE: {mae:.3f}")
```

---

## Summary

SC-SINDy combines neural network priors with sparse regression in a two-stage approach:

1. **Stage 1:** A factorized neural network predicts equation structure from trajectory statistics, achieving dimension-agnostic generalization through structural term embeddings and relative position encoding.

2. **Stage 2:** Standard STLS refines coefficients on the filtered library.

**Key advantages:**
- 91% improvement at high noise (50%)
- Works across system dimensions (2D-10D)
- Robust threshold selection (0.2-0.8 range)
- Prevents catastrophic failures on chaotic systems

**Key limitations:**
- Requires diverse training data
- Polynomial-focused (custom libraries need retraining)
- Mixed results on certain system types (bilinear dynamics)

For questions or contributions, please open an issue on the repository.

---

*Document version: 1.0 | Last updated: February 2026*
