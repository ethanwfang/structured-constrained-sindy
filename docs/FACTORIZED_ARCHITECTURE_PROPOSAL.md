# Factorized Architecture for Dimension-Agnostic SC-SINDy

**Author:** [Your Name]
**Date:** February 9, 2026
**Purpose:** Technical proposal for extending SC-SINDy to arbitrary dimensions

---

## Executive Summary

This document proposes a **factorized neural network architecture** for Structure-Constrained SINDy (SC-SINDy) that enables generalization across dynamical systems of any dimension (2D, 3D, 4D+). The current architecture requires separate networks for each dimension, limiting scalability and preventing transfer learning. The proposed approach decouples trajectory encoding from term prediction, allowing a single network to handle arbitrary polynomial libraries while maintaining the interpretability that distinguishes SC-SINDy from pure neural approaches.

---

## 1. Problem Statement

### 1.1 Current Architecture Limitations

The existing SC-SINDy network architecture has a **fixed output dimension**:

```
Current: f(trajectory_features) → [p₁, p₂, ..., pₙ]
         where n = number of library terms (dimension-dependent)
```

| Dimension | Library Terms (poly order 3) | Output Size |
|-----------|------------------------------|-------------|
| 2D | 1, x, y, xy, xx, yy, xxx, xxy, xyy, yyy | 10 |
| 3D | 1, x, y, z, xy, xz, yz, xx, yy, zz, ... | 20 |
| 4D | 1, x, y, z, w, xy, xz, xw, yz, yw, zw, ... | 35 |
| 5D | ... | 56 |
| nD | ... | O(n³) for poly order 3 |

**Consequences:**
1. **Separate networks required** for each dimension
2. **No transfer learning** between dimensions (patterns learned in 2D don't help 3D)
3. **Training data inefficiency** (must collect sufficient data per dimension)
4. **Combinatorial scaling** of output layer parameters

### 1.2 The Missed Opportunity

Consider the `xy` bilinear interaction term. Our recent work demonstrated that learning this pattern from ecological systems (CompetitiveExclusion, Mutualism, SIS) enables generalization to unseen Lotka-Volterra dynamics. However, this knowledge is currently **trapped in the 2D network**.

In a 3D system, the analogous terms `xy`, `xz`, `yz` represent the same structural pattern—bilinear interaction between state variables. A dimension-agnostic architecture would recognize this automatically.

---

## 2. Proposed Solution: Factorized Architecture

### 2.1 Core Idea

**Decouple** the trajectory understanding from term prediction by learning:
1. A **trajectory encoder** that produces a fixed-size "structural fingerprint"
2. A **term embedding space** where similar structural patterns cluster together
3. A **compatibility function** that scores trajectory-term pairs

```
Proposed: g(trajectory) · h(term) → probability
          where g: ℝᵐ → ℝᵈ (trajectory encoder)
                h: term → ℝᵈ   (term embedding)
                d = latent dimension (fixed, e.g., 64)
```

### 2.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FACTORIZED SC-SINDy                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRAJECTORY BRANCH              TERM BRANCH                     │
│  ─────────────────              ───────────                     │
│                                                                 │
│  ┌─────────────────┐           ┌─────────────────┐             │
│  │ Trajectory Data │           │  Term: "xxy"    │             │
│  │ x(t), y(t), ... │           │                 │             │
│  └────────┬────────┘           └────────┬────────┘             │
│           │                             │                       │
│           ▼                             ▼                       │
│  ┌─────────────────┐           ┌─────────────────┐             │
│  │ Feature Extract │           │ Structural      │             │
│  │ (19 features)   │           │ Features        │             │
│  │ • means, stds   │           │ • degree: 3     │             │
│  │ • correlations  │           │ • x_exp: 2      │             │
│  │ • frequencies   │           │ • y_exp: 1      │             │
│  └────────┬────────┘           │ • is_mixed: 1   │             │
│           │                    └────────┬────────┘             │
│           ▼                             │                       │
│  ┌─────────────────┐                    │                       │
│  │ Trajectory MLP  │                    ▼                       │
│  │ 19 → 64 → 64    │           ┌─────────────────┐             │
│  └────────┬────────┘           │ Term MLP        │             │
│           │                    │ k → 64 → 64     │             │
│           │                    └────────┬────────┘             │
│           │                             │                       │
│           ▼                             ▼                       │
│       h_traj ∈ ℝ⁶⁴                 e_term ∈ ℝ⁶⁴                │
│           │                             │                       │
│           └──────────┬──────────────────┘                       │
│                      │                                          │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │  Dot Product  │                                  │
│              │  h · e        │                                  │
│              └───────┬───────┘                                  │
│                      │                                          │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │   Sigmoid     │                                  │
│              │   → p ∈ [0,1] │                                  │
│              └───────────────┘                                  │
│                                                                 │
│              "Probability that this term appears                │
│               in the governing equation"                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Term Structural Features

The key innovation is representing terms by their **structural properties** rather than identity:

```python
def term_to_features(term: str, max_dim: int = 6) -> np.ndarray:
    """
    Convert a polynomial term to a structural feature vector.

    Examples:
        "1"   → [0, 0, 0, 0, 0, 0, 1, 0, 0]  (constant)
        "x"   → [1, 1, 0, 0, 0, 0, 0, 1, 0]  (linear, single var)
        "xy"  → [2, 1, 1, 0, 0, 0, 0, 0, 1]  (bilinear interaction)
        "xxy" → [3, 2, 1, 0, 0, 0, 0, 0, 1]  (mixed cubic)
        "xxx" → [3, 3, 0, 0, 0, 0, 0, 1, 0]  (pure cubic)
    """
    features = []

    # Degree features
    features.append(len(term) if term != "1" else 0)  # Total degree

    # Per-variable exponents (padded to max_dim)
    var_counts = {}
    for char in term:
        if char.isalpha():
            var_counts[char] = var_counts.get(char, 0) + 1
    for var in ['x', 'y', 'z', 'w', 'v', 'u'][:max_dim]:
        features.append(var_counts.get(var, 0))

    # Structural flags
    features.append(1 if term == "1" else 0)           # Is constant
    features.append(1 if len(set(term)) == 1 else 0)   # Is pure power
    features.append(1 if len(set(term)) > 1 else 0)    # Is interaction

    return np.array(features, dtype=np.float32)
```

**Critical insight:** The term `xy` in 2D and `xy` in 3D have **identical structural features**. Similarly, `xz` in 3D has the same structure as `xy`—both are bilinear interactions. This enables automatic transfer.

---

## 3. Mathematical Formulation

### 3.1 Formal Definition

Let:
- $\mathbf{X} \in \mathbb{R}^{T \times n}$ be a trajectory with $T$ timesteps in $n$ dimensions
- $\phi(\mathbf{X}) \in \mathbb{R}^m$ be extracted trajectory features
- $\mathcal{L} = \{t_1, t_2, ..., t_k\}$ be the polynomial library terms
- $\psi(t_i) \in \mathbb{R}^p$ be structural features of term $t_i$

The factorized predictor is:

$$P(t_i \in \text{active} \mid \mathbf{X}) = \sigma\left( f_\theta(\phi(\mathbf{X}))^\top g_\omega(\psi(t_i)) \right)$$

where:
- $f_\theta: \mathbb{R}^m \rightarrow \mathbb{R}^d$ is the trajectory encoder (MLP with parameters $\theta$)
- $g_\omega: \mathbb{R}^p \rightarrow \mathbb{R}^d$ is the term encoder (MLP with parameters $\omega$)
- $\sigma$ is the sigmoid function
- $d$ is the shared latent dimension

### 3.2 Training Objective

Given training data $\mathcal{D} = \{(\mathbf{X}^{(j)}, \mathbf{y}^{(j)})\}_{j=1}^N$ where $\mathbf{y}^{(j)} \in \{0,1\}^k$ is the binary ground truth structure:

$$\mathcal{L}(\theta, \omega) = -\frac{1}{N \cdot k} \sum_{j=1}^N \sum_{i=1}^k \left[ y_i^{(j)} \log \hat{p}_i^{(j)} + (1 - y_i^{(j)}) \log(1 - \hat{p}_i^{(j)}) \right]$$

where $\hat{p}_i^{(j)} = \sigma(f_\theta(\phi(\mathbf{X}^{(j)}))^\top g_\omega(\psi(t_i)))$.

### 3.3 Extension: Bilinear Compatibility

For increased expressiveness, replace the dot product with a learned bilinear form:

$$P(t_i \in \text{active} \mid \mathbf{X}) = \sigma\left( f_\theta(\phi(\mathbf{X}))^\top \mathbf{W} \, g_\omega(\psi(t_i)) \right)$$

where $\mathbf{W} \in \mathbb{R}^{d \times d}$ is a learned compatibility matrix. This allows the model to learn asymmetric relationships between trajectory patterns and term structures.

---

## 4. Benefits for SC-SINDy

### 4.1 Dimension Agnosticism

| Property | Current Architecture | Factorized Architecture |
|----------|---------------------|------------------------|
| 2D support | ✓ (dedicated network) | ✓ |
| 3D support | Requires new network | ✓ (same network) |
| 4D+ support | Requires new network | ✓ (same network) |
| Parameter count | O(hidden × n_terms) per dim | O(hidden × latent) total |

### 4.2 Transfer Learning

**Scenario:** Train on 2D systems, apply to 3D system.

With factorized architecture:
1. Trajectory encoder learns dimension-agnostic patterns (oscillation frequency, damping, chaos indicators)
2. Term encoder learns that "bilinear interaction" terms cluster together
3. At inference on 3D data: new terms like `xz`, `yz` automatically get appropriate probabilities based on their structural similarity to learned 2D patterns

**Expected behavior:**
```
Training: 2D systems including CompetitiveExclusion (has xy)
          Network learns: "bilinear interactions often active in ecological dynamics"

Inference: 3D ecological system (e.g., 3-species competition)
          Terms xy, xz, yz all receive elevated probability
          (because they share structural features with learned xy pattern)
```

### 4.3 Sample Efficiency

Current approach: Each dimension requires full training dataset.
- 2D: ~420 trajectories from 14 systems
- 3D: Would need ~300+ trajectories from new 3D systems
- 4D: Would need ~400+ trajectories from new 4D systems

Factorized approach: Knowledge compounds across dimensions.
- Train jointly on 2D + 3D systems
- Patterns learned in 2D transfer to 3D terms
- Fewer 3D-specific examples needed

### 4.4 Interpretability

The factorized architecture provides two levels of interpretability:

**1. Trajectory Embedding Space**
- Visualize $f_\theta(\phi(\mathbf{X}))$ using t-SNE/UMAP
- Systems with similar dynamics should cluster
- Can identify what structural patterns the network has learned

**2. Term Embedding Space**
- Visualize $g_\omega(\psi(t_i))$ for all terms
- Structurally similar terms should cluster (xy, xz, yz together)
- Reveals what the network considers "similar" terms

```
Example visualization:
                    Term Embedding Space

           xxx •            • yyy
                  \        /
                   •  xyy  •
                  xxy
                    \    /
                     • xy •
                    /      \
                   x •    • y
                    \    /
                      • 1

(Terms cluster by structural similarity)
```

---

## 5. Implementation Plan

### 5.1 Phase 1: Core Architecture (1-2 weeks)

1. Implement `TermEncoder` class with structural feature extraction
2. Implement `FactorizedStructureNetwork` combining trajectory and term branches
3. Modify training loop to iterate over (trajectory, term, label) triplets
4. Unit tests for dimension-agnostic inference

### 5.2 Phase 2: Validation (1 week)

1. Train on current 2D systems
2. Verify performance matches or exceeds current architecture on 2D benchmarks
3. Ablation: compare dot product vs. bilinear compatibility

### 5.3 Phase 3: Cross-Dimension Transfer (1-2 weeks)

1. Add 3D training systems (Lorenz, Rossler, Chen already available)
2. Train jointly on 2D + 3D
3. Evaluate transfer: train on 2D only, test on 3D
4. Evaluate transfer: train on 3D only, test on 2D

### 5.4 Phase 4: Publication Experiments (2 weeks)

1. Systematic benchmarks across dimensions
2. Comparison with dimension-specific baselines
3. Visualization of learned embeddings
4. Real-world application (extend Lynx-Hare success to higher-dimensional ecological data)

---

## 6. Expected Results

### 6.1 Hypothesis 1: Maintained 2D Performance

The factorized architecture should match or exceed current 2D performance because:
- Same information is available (trajectory features, term identity via structural features)
- Additional inductive bias (similar terms should behave similarly)

**Metric:** F1 score on 2D test systems ≥ current architecture

### 6.2 Hypothesis 2: Successful Cross-Dimension Transfer

Training on 2D should provide non-trivial performance on 3D because:
- Structural patterns (linear, quadratic, bilinear) are dimension-independent
- Term embeddings encode these patterns explicitly

**Metric:** 3D F1 score with 2D-only training > random baseline (which would be ~0.3)

### 6.3 Hypothesis 3: Improved Sample Efficiency

Joint 2D+3D training should outperform dimension-specific training with limited data:

**Experiment:**
- Baseline: Train 3D-only network with N trajectories
- Proposed: Train factorized network with N/2 2D + N/2 3D trajectories
- Expected: Factorized achieves higher 3D F1

---

## 7. Comparison with Alternatives

| Approach | Dimension-Agnostic | Transfer Learning | Interpretable | Complexity |
|----------|-------------------|-------------------|---------------|------------|
| Current (fixed output) | ✗ | ✗ | ✓ | Low |
| **Factorized (proposed)** | **✓** | **✓** | **✓** | **Low-Medium** |
| Per-term prediction | ✓ | ✓ | ○ | Low |
| Graph Neural Network | ✓ | ✓ | ○ | High |
| Transformer | ✓ | ✓ | ✓ | High |
| Meta-learning (MAML) | ✓ | ✓ | ✗ | Very High |

**Why Factorized is optimal for SC-SINDy:**

1. **Maintains simplicity:** Two small MLPs, standard training
2. **Preserves interpretability:** Core SC-SINDy advantage over black-box methods
3. **Enables publication story:** "Dimension-agnostic structure learning" is a clear contribution
4. **Practical:** Can implement and validate within publication timeline

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dot product too restrictive | Medium | Medium | Use bilinear form; add MLP after dot product |
| Term features insufficient | Low | High | Iteratively add features; learn features end-to-end |
| No transfer observed | Low | High | Ensure sufficient 2D diversity; add regularization |
| Slower than current | Low | Low | Batch all terms together; GPU parallelism |

---

## 9. Conclusion

The factorized architecture represents a natural evolution of SC-SINDy that:

1. **Solves the dimension scaling problem** without sacrificing interpretability
2. **Enables transfer learning** between dimensions, improving sample efficiency
3. **Provides new interpretability tools** via embedding visualization
4. **Strengthens the publication narrative** with a clear methodological contribution

The approach is grounded in the observation that polynomial terms have intrinsic structural properties independent of the specific dimension. By explicitly encoding these properties, we allow the network to generalize patterns learned in low dimensions to high-dimensional systems—a capability not present in existing SINDy variants.

**Recommended next step:** Implement Phase 1 (core architecture) and validate on existing 2D benchmarks before proceeding with cross-dimension experiments.

---

## Appendix A: Comparison with Related Work

### A.1 HyperSINDy (Gao et al., 2022)

HyperSINDy uses a hypernetwork to generate SINDy coefficients. While powerful, it:
- Produces coefficients directly (less interpretable than structure prediction)
- Does not explicitly address dimension scaling
- Our approach: Structure prediction + explicit term encoding

### A.2 SINDy-Autoencoders (Champion et al., 2019)

SINDy-AE learns coordinates and dynamics jointly. Relation to our work:
- Orthogonal contribution (coordinate discovery vs. structure prediction)
- Could potentially combine: use factorized SC-SINDy for structure, SINDy-AE for coordinates
- Our approach focuses specifically on structure generalization

### A.3 Ensemble-SINDy (Fasel et al., 2022)

Uses statistical ensembling for robust structure selection:
- Complements learned priors (could ensemble over factorized predictions)
- Does not address dimension scaling
- Our approach: Learned priors that generalize across dimensions

---

## Appendix B: Implementation Pseudocode

```python
class FactorizedStructureNetwork(nn.Module):
    """
    Dimension-agnostic structure prediction via factorized architecture.
    """

    def __init__(self, traj_features=19, term_features=12, latent_dim=64):
        super().__init__()

        # Trajectory encoder: features → latent
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_features, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Term encoder: structural features → latent
        self.term_encoder = nn.Sequential(
            nn.Linear(term_features, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Optional: bilinear compatibility
        self.use_bilinear = True
        if self.use_bilinear:
            self.W = nn.Parameter(torch.eye(latent_dim))

    def forward(self, traj_features, term_structural_features):
        """
        Args:
            traj_features: [batch_size, 19] trajectory features
            term_structural_features: [num_terms, 12] term features

        Returns:
            probs: [batch_size, num_terms] probability each term is active
        """
        # Encode trajectory
        h_traj = self.traj_encoder(traj_features)  # [batch, latent]

        # Encode all terms
        h_terms = self.term_encoder(term_structural_features)  # [num_terms, latent]

        # Compute compatibility scores
        if self.use_bilinear:
            # h_traj @ W @ h_terms.T
            scores = torch.matmul(
                torch.matmul(h_traj, self.W),
                h_terms.T
            )
        else:
            # Simple dot product
            scores = torch.matmul(h_traj, h_terms.T)

        # Convert to probabilities
        probs = torch.sigmoid(scores)

        return probs

    def predict_for_dimension(self, traj_features, dimension, poly_order=3):
        """
        Convenience method for inference on specific dimension.
        """
        # Generate library terms for this dimension
        term_names = generate_polynomial_terms(dimension, poly_order)

        # Convert to structural features
        term_features = torch.stack([
            torch.tensor(term_to_features(t)) for t in term_names
        ])

        # Predict
        return self.forward(traj_features, term_features), term_names
```

---

*Document prepared for review. Please direct questions to [your email].*
