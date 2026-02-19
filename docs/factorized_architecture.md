# Factorized Structure Network: Dimension-Agnostic Equation Discovery

## Executive Summary

The Factorized Structure Network is a novel neural architecture for predicting the sparse structure of dynamical system equations, designed to generalize across systems of any dimension. Unlike traditional approaches that require separate models for 2D, 3D, and 4D systems, this architecture uses a factorized representation that decouples trajectory encoding from term prediction, enabling zero-shot generalization to unseen dimensions and system types.

**Key Result:** The model achieves **F1 = 0.892 on the Lorenz system** despite never being trained on it, demonstrating strong generalization to canonical benchmarks.

---

## 1. Problem Statement

### 1.1 The Dimension Barrier

Traditional structure prediction networks for SINDy face a fundamental limitation: they are dimension-specific. A network trained on 2D systems (Van der Pol, Lotka-Volterra) cannot be applied to 3D systems (Lorenz, Rossler) because:

1. **Input size varies:** A 2D trajectory has shape `[T, 2]` while 3D has `[T, 3]`
2. **Library size grows combinatorially:**
   - 2D with `poly_order=3`: 10 terms
   - 3D with `poly_order=3`: 20 terms
   - 4D with `poly_order=3`: 35 terms
3. **Output structure differs:** Predicting `[2, 10]` vs `[3, 20]` vs `[4, 35]` matrices

### 1.2 Our Solution: Factorization

We decompose the prediction problem into dimension-agnostic components:

```
P(term_j active in equation_i | trajectory X) = f(encode(X), embed(term_j), embed(eq_i))
```

Each component is designed to work for any dimension:
- **Trajectory Encoder:** Maps any `[T, n_vars]` trajectory to fixed `[latent_dim]` vector
- **Term Embedder:** Maps any polynomial term to fixed `[latent_dim]` embedding
- **Equation Embedder:** Maps equation index to fixed `[latent_dim]` embedding
- **Matching Network:** Predicts activation probability from the three embeddings

---

## 2. Architecture

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Factorized Structure Network                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Trajectory X ──► [Statistics] ──► [Encoder MLP] ──► z_traj         │
│  [T, n_vars]       [n_vars, 8]      [latent_dim]                    │
│                                                                      │
│  Term Powers ──► [Power Embed] ──► [Aggregator] ──► e_term          │
│  [(0,1), (1,2)]   [n_factors, d]    [latent_dim]                    │
│                                                                      │
│  Equation idx ──► [Embedding Table] ──► e_eq                        │
│       i           [latent_dim]                                       │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │     Bilinear Interaction: z_traj ⊙ e_term ⊙ e_eq           │    │
│  │                    ↓                                         │    │
│  │              [Classifier MLP]                                │    │
│  │                    ↓                                         │    │
│  │              P(term active)                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

#### 2.2.1 Trajectory Encoder (StatisticsEncoder)

The trajectory encoder must handle inputs of varying dimension. We achieve this through per-variable statistics extraction followed by aggregation.

**Statistics Extracted (per variable):**
| Statistic | Description |
|-----------|-------------|
| Mean | Central tendency |
| Std | Spread |
| Skewness | Asymmetry |
| Kurtosis | Tail behavior |
| Energy | Mean of x² |
| Range | max - min |
| Median | Robust center |
| Avg |dx/dt| | Derivative magnitude |

**Architecture:**
```python
stats = extract_per_variable_stats(X)  # [n_vars, 8]
stats_normalized = LayerNorm(stats)
var_embeds = SharedMLP(stats_normalized)  # [n_vars, hidden_dim]
aggregated = mean_pool(var_embeds)  # [hidden_dim]
z_traj = Projector(aggregated)  # [latent_dim]
```

**Why this works:** By processing each variable independently through a shared MLP, then aggregating, we achieve permutation invariance over variables and handle any number of them.

#### 2.2.2 Term Embedder

The term embedder creates structural representations of polynomial terms that generalize across dimensions.

**Key Insight:** A term like `x²y` can be represented as a set of (variable_index, power) pairs: `[(0, 2), (1, 1)]`. This representation is dimension-agnostic.

**Architecture:**
```python
# Learnable embeddings
variable_embedding = nn.Embedding(max_vars, embed_dim)  # Which variable
power_embedding = nn.Embedding(max_power + 1, embed_dim)  # What power

def embed_term(powers: List[Tuple[int, int]]) -> Tensor:
    """Embed a term like x²y → [(0,2), (1,1)]"""
    factor_embeds = []
    for var_idx, power in powers:
        if power > 0:
            v_embed = variable_embedding(var_idx)
            p_embed = power_embedding(power)
            factor_embeds.append(v_embed + p_embed)

    # Aggregate factors (sum for permutation invariance)
    return sum(factor_embeds) if factor_embeds else zero_embedding
```

**Examples:**
| Term | Powers Representation | Embedding |
|------|----------------------|-----------|
| `1` | `[]` | Zero vector |
| `x` | `[(0, 1)]` | var[0] + pow[1] |
| `y²` | `[(1, 2)]` | var[1] + pow[2] |
| `xy` | `[(0, 1), (1, 1)]` | (var[0] + pow[1]) + (var[1] + pow[1]) |
| `x²yz` | `[(0, 2), (1, 1), (2, 1)]` | (var[0] + pow[2]) + (var[1] + pow[1]) + (var[2] + pow[1]) |

#### 2.2.3 Matching Network

The matching network predicts whether a term should be active in a given equation.

**Architecture (V2 - Bilinear):**
```python
# Project embeddings
z_traj = traj_proj(trajectory_encoding)  # [batch, latent_dim]
e_term = term_proj(term_embeddings)      # [n_terms, latent_dim]
e_eq = eq_proj(equation_embeddings)      # [n_vars, latent_dim]

# Normalize to unit sphere (prevents explosion)
z_traj = z_traj / (z_traj.norm() + eps)
e_term = e_term / (e_term.norm() + eps)
e_eq = e_eq / (e_eq.norm() + eps)

# Bilinear interaction
interaction = z_traj[:, None, None, :] * e_term[None, None, :, :] * e_eq[None, :, None, :]
# Shape: [batch, n_vars, n_terms, latent_dim]

# Classify
probs = Classifier(interaction)  # [batch, n_vars, n_terms]
```

**Why bilinear?** The element-wise product captures multiplicative interactions between the three factors. If the trajectory "looks like" a system that should have term `xy` in equation `dx/dt`, all three embeddings will align, producing high activation.

---

## 3. Training Methodology

### 3.1 Data Generation

For each system in the training set:
1. Instantiate with default parameters
2. Generate random initial conditions: `x0 ~ N(0, 2)`
3. Integrate trajectory for `t ∈ [0, 50]` with 5000 points
4. Add measurement noise: `noise_level ∈ {0.0, 0.05, 0.10}`
5. Trim transients (first/last 100 points)
6. Extract ground truth structure from system definition

### 3.2 Mixed-Dimension Training

The key innovation is training on systems of multiple dimensions simultaneously:

```python
train_systems = {
    2: [VanDerPol, Duffing, FitzHughNagumo, ...],  # 12 systems
    3: [Rossler, Chen, Halvorsen, SprottB, ...],   # 5 systems
    4: [CoupledVanDerPol, HyperchaoticLorenz, ...] # 4 systems
}

# Generate mixed dataset
samples = []
for dim, systems in train_systems.items():
    for system_cls in systems:
        for _ in range(n_trajectories):
            sample = generate_sample(system_cls())
            samples.append(sample)  # Contains (stats, structure, n_vars)
```

### 3.3 Batching Strategy

Since samples have different dimensions, we use a custom collation strategy:

```python
def collate_by_dimension(batch):
    """Group samples by (n_vars, poly_order) within batch."""
    groups = {}
    for item in batch:
        key = (item.n_vars, item.poly_order)
        groups.setdefault(key, []).append(item)

    # Stack within each group
    return {key: stack(items) for key, items in groups.items()}
```

### 3.4 Loss Function

Binary Cross-Entropy on the structure prediction:

```python
loss = BCELoss(predicted_probs, true_structure.float())
```

### 3.5 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 16 |
| Epochs | 100 |
| Early Stopping | 15 epochs patience |
| Validation Split | 10% |
| Latent Dimension | 64 |

---

## 4. Train/Test Split Design

### 4.1 Principles

1. **No answer leakage:** Test systems have distinct structures from training
2. **Canonical benchmarks held out:** Lorenz is always in test set
3. **Structural diversity:** Training covers various term patterns (linear, bilinear, cubic)
4. **Cross-dimension evaluation:** 4D test systems are entirely novel

### 4.2 Split Summary

| Dimension | Train | Test | Held-Out |
|-----------|-------|------|----------|
| 2D | 12 | 5 | 2 |
| 3D | 5 | 3 | 2 |
| 4D | 4 | 2 | 0 |
| **Total** | **21** | **10** | **4** |

### 4.3 Detailed Splits

**2D Training Systems:**
- Oscillators: VanDerPol, Duffing, Rayleigh, Cubic
- Biological: Selkov, Brusselator
- Ecological: CompetitiveExclusion, Mutualism, SISEpidemic
- Neural: FitzHughNagumo, MorrisLecar
- Canonical: HopfNormalForm

**2D Test Systems:**
- DampedHarmonicOscillator, LinearOscillator, ForcedOscillator
- PredatorPreyTypeII, HindmarshRose2D

**3D Training Systems:**
- Rossler, Chen, Halvorsen, SprottB, Aizawa

**3D Test Systems:**
- **Lorenz** (canonical benchmark - never trained!)
- SIRModel, RabinovichFabrikant

**4D Training Systems:**
- CoupledVanDerPol, CoupledDuffing, HyperchaoticLorenz, LotkaVolterra4D

**4D Test Systems:**
- HyperchaoticRossler, CoupledFitzHughNagumo

---

## 5. Current Results

### 5.1 Test Set Performance

| System | Dim | Precision | Recall | F1 |
|--------|-----|-----------|--------|-----|
| **Lorenz** | 3D | **1.000** | 0.807 | **0.892** |
| LinearOscillator | 2D | 0.788 | 1.000 | 0.879 |
| ForcedOscillator | 2D | 0.750 | 1.000 | 0.857 |
| DampedHarmonic | 2D | 0.778 | 0.900 | 0.827 |
| PredatorPreyTypeII | 2D | 0.732 | 0.700 | 0.711 |
| CoupledFitzHughNagumo | 4D | 0.701 | 0.621 | 0.650 |
| HindmarshRose2D | 2D | 0.500 | 0.286 | 0.364 |
| SIRModel | 3D | 0.248 | 0.588 | 0.346 |
| RabinovichFabrikant | 3D | 0.327 | 0.335 | 0.329 |
| HyperchaoticRossler | 4D | 0.200 | 0.394 | 0.264 |

### 5.2 Key Observations

1. **Lorenz Generalization:** Perfect precision on Lorenz with 0.89 F1, demonstrating the model learned generalizable structural patterns rather than memorizing specific systems.

2. **2D Performance:** Strong across most 2D test systems (F1 > 0.7), suggesting good pattern learning for lower dimensions.

3. **4D Challenges:** Mixed results on 4D systems, likely due to fewer training examples (only 4 systems × 30 trajectories = 120 samples).

4. **Complex Systems:** Systems with unusual structures (HindmarshRose, RabinovichFabrikant) show lower performance, indicating room for improvement.

---

## 6. File Structure

```
src/sc_sindy/network/factorized/
├── __init__.py                 # Module exports
├── term_representation.py      # Term ↔ power utilities
├── term_embedder.py            # TermEmbedder class
├── trajectory_encoder.py       # StatisticsEncoder, GRUEncoder, HybridEncoder
├── factorized_network.py       # FactorizedStructureNetwork, V2
├── training.py                 # Mixed-dimension training
└── inference.py                # FactorizedPredictor

src/sc_sindy/systems/
├── chaotic_3d.py               # 6 new 3D systems
├── coupled_4d.py               # 6 new 4D systems
└── ...

src/sc_sindy/evaluation/
└── splits_factorized.py        # Train/test splits

scripts/
├── train_factorized_full.py    # Full training script
└── evaluate_factorized.py      # Evaluation script
```

---

## 7. Usage

### 7.1 Training

```bash
python scripts/train_factorized_full.py \
    --epochs 100 \
    --trajectories 50 \
    --latent-dim 64 \
    --batch-size 16 \
    --lr 0.0005
```

### 7.2 Evaluation

```bash
python scripts/evaluate_factorized.py \
    --model models/factorized/factorized_model.pt \
    --include-heldout
```

### 7.3 Programmatic Usage

```python
from sc_sindy.network.factorized import (
    FactorizedStructureNetworkV2,
    FactorizedPredictor,
)

# Load trained model
model = FactorizedStructureNetworkV2.load("models/factorized/factorized_model.pt")
predictor = FactorizedPredictor(model)

# Predict on any dimension
trajectory_3d = np.random.randn(1000, 3)  # Example trajectory
probs = predictor.predict(trajectory_3d, poly_order=3)
# probs.shape = [3, 20] - probabilities for each term in each equation

# Threshold to get binary structure
structure = probs > 0.5
```

---

## 8. Comparison to Baseline

| Aspect | Dimension-Specific Network | Factorized Network |
|--------|---------------------------|-------------------|
| Models needed | 1 per dimension | 1 for all |
| Training data | Dimension-specific | Mixed-dimension |
| Zero-shot transfer | No | Yes |
| Parameter count | O(n_dims × params) | O(params) |
| Lorenz (3D, unseen) | N/A (different model) | F1 = 0.892 |

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. **4D Performance Gap:** Fewer 4D training systems leads to weaker generalization
2. **Complex Dynamics:** Systems with unusual coupling (RabinovichFabrikant) are challenging
3. **Statistics-Based Encoding:** May miss temporal patterns captured by RNN-based encoders
4. **Fixed Polynomial Basis:** Limited to polynomial terms (no trigonometric, exponential)

### 9.2 Future Directions

1. **More 4D/5D Systems:** Add synthetic systems with controlled structures
2. **Hybrid Encoder:** Combine statistics with GRU for temporal awareness
3. **Attention Mechanisms:** Replace bilinear interaction with cross-attention
4. **Meta-Learning:** Few-shot adaptation to new system types
5. **Uncertainty Quantification:** Predict confidence intervals on structure

---

## 10. Conclusion

The Factorized Structure Network represents a significant advance in dimension-agnostic equation discovery. By decomposing the prediction problem into trajectory encoding, term embedding, and matching, we achieve:

- **Generalization across dimensions** (2D, 3D, 4D with single model)
- **Zero-shot transfer** to unseen system types
- **Strong benchmark performance** (F1 = 0.89 on Lorenz, never trained)

This architecture enables scalable structure prediction for dynamical systems of any dimension, paving the way for automated discovery of governing equations in complex real-world systems.
