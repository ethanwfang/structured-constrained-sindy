# Factorized Structure Network: Dimension-Agnostic Equation Discovery

## Executive Summary

The Factorized Structure Network is a novel neural architecture for predicting the sparse structure of dynamical system equations, designed to generalize across systems of any dimension. Unlike traditional approaches that require separate models for 2D, 3D, and 4D systems, this architecture uses a factorized representation that decouples trajectory encoding from term prediction, enabling zero-shot generalization to unseen dimensions and system types.

**Key Result:** The model achieves **F1 = 0.892 on the Lorenz system** despite never being trained on it, demonstrating strong generalization to canonical benchmarks.

---

## 1. Problem Statement

### 1.1 The Dimension Barrier

Traditional structure prediction networks for SINDy face a fundamental limitation: they are dimension-specific. A network trained on 2D systems (Van der Pol, Lotka-Volterra) cannot be applied to 3D systems (Lorenz, Rossler) because:

1. **Input size varies:** A 2D trajectory has shape `[T, 2]` while 3D has `[T, 3]`
2. **Library size grows combinatorially:** For a polynomial library of order $p$ with $n$ variables, the number of terms is given by the multiset coefficient:
   $$|\mathcal{L}_{n,p}| = \sum_{k=0}^{p} \binom{n + k - 1}{k} = \binom{n + p}{p}$$
   Examples:
   - 2D with `poly_order=3`: $\binom{5}{3} = 10$ terms
   - 3D with `poly_order=3`: $\binom{6}{3} = 20$ terms
   - 4D with `poly_order=3`: $\binom{7}{3} = 35$ terms
3. **Output structure differs:** Predicting $[n, |\mathcal{L}_{n,p}|]$ matrices of varying dimensions

### 1.2 Our Solution: Factorization

We decompose the prediction problem into dimension-agnostic components. Let $\mathbf{X} \in \mathbb{R}^{T \times n}$ denote a trajectory, $\theta_j \in \mathcal{L}_{n,p}$ denote the $j$-th library term, and $i \in \{1, \ldots, n\}$ denote the equation index. The structure prediction is modeled as:

$$P(S_{ij} = 1 \mid \mathbf{X}) = \sigma\left( g_\phi\left( f_{\text{enc}}(\mathbf{X}) \odot f_{\text{term}}(\theta_j) \odot f_{\text{eq}}(i, n) \right) \right)$$

where:
- $S_{ij} \in \{0, 1\}$ indicates whether term $\theta_j$ is active in equation $i$
- $f_{\text{enc}}: \mathbb{R}^{T \times n} \to \mathbb{R}^d$ is the trajectory encoder
- $f_{\text{term}}: \mathcal{L}_{n,p} \to \mathbb{R}^d$ is the term embedder
- $f_{\text{eq}}: \{1, \ldots, n\} \times \mathbb{N} \to \mathbb{R}^d$ is the equation encoder
- $g_\phi: \mathbb{R}^d \to \mathbb{R}$ is a classifier MLP
- $\odot$ denotes the Hadamard (element-wise) product
- $\sigma$ is the sigmoid function
- $d$ is the latent dimension

Each component is designed to work for any dimension:
- **Trajectory Encoder:** Maps any $\mathbf{X} \in \mathbb{R}^{T \times n}$ trajectory to fixed $\mathbb{R}^d$ vector
- **Term Embedder:** Maps any polynomial term to fixed $\mathbb{R}^d$ embedding
- **Equation Embedder:** Maps equation index and dimension to fixed $\mathbb{R}^d$ embedding
- **Matching Network:** Predicts activation probability from the Hadamard product of the three embeddings

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
│  [(0,1), (1,2)]   [n_factors, e]    [latent_dim]                    │
│                                                                      │
│  Equation idx ──► [Relative Pos Encoder] ──► e_eq                   │
│       (i, n)          [latent_dim]                                   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │     Hadamard Interaction: z_traj ⊙ e_term ⊙ e_eq           │    │
│  │                    ↓                                         │    │
│  │              [Classifier MLP]                                │    │
│  │                    ↓                                         │    │
│  │          P(S_ij = 1 | X) ∈ [0,1]                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

#### 2.2.1 Trajectory Encoder (StatisticsEncoder)

The trajectory encoder must handle inputs of varying dimension. We achieve this through per-variable statistics extraction followed by aggregation.

**Statistics Extracted (per variable $x_i \in \mathbb{R}^T$):**

| Statistic | Formula | Description |
|-----------|---------|-------------|
| Mean | $\bar{x}_i = \frac{1}{T}\sum_{t=1}^T x_i(t)$ | Central tendency |
| Std | $\sigma_i = \sqrt{\frac{1}{T}\sum_{t=1}^T (x_i(t) - \bar{x}_i)^2}$ | Spread |
| Skewness | $\gamma_i = \frac{1}{T}\sum_{t=1}^T \left(\frac{x_i(t) - \bar{x}_i}{\sigma_i}\right)^3$ | Asymmetry |
| Kurtosis | $\kappa_i = \frac{1}{T}\sum_{t=1}^T \left(\frac{x_i(t) - \bar{x}_i}{\sigma_i}\right)^4 - 3$ | Tail behavior (excess kurtosis) |
| Energy | $E_i = \frac{1}{T}\sum_{t=1}^T x_i(t)^2$ | Mean squared amplitude |
| Range | $R_i = \max_t x_i(t) - \min_t x_i(t)$ | Dynamic range |
| Median | $\text{med}(x_i)$ | Robust central tendency |
| Avg derivative | $\bar{D}_i = \frac{1}{T-1}\sum_{t=1}^{T-1} |x_i(t+1) - x_i(t)|$ | Mean derivative magnitude |

The statistics for all variables are collected into a matrix $\mathbf{S} \in \mathbb{R}^{n \times 8}$.

**Architecture:**

Let $\mathbf{s}_i \in \mathbb{R}^8$ denote the statistics for variable $i$. The encoding proceeds as:

1. **Normalization:** $\tilde{\mathbf{s}}_i = \text{LayerNorm}(\mathbf{s}_i)$
2. **Per-variable encoding:** $\mathbf{h}_i = \text{MLP}_{\text{var}}(\tilde{\mathbf{s}}_i) \in \mathbb{R}^{h}$ (shared weights across variables)
3. **Aggregation:** $\bar{\mathbf{h}} = \frac{1}{n}\sum_{i=1}^n \mathbf{h}_i$ (mean pooling)
4. **Projection:** $\mathbf{z}_{\text{traj}} = \text{MLP}_{\text{proj}}(\bar{\mathbf{h}}) \in \mathbb{R}^d$

```python
stats = extract_per_variable_stats(X)  # [n_vars, 8]
stats_normalized = LayerNorm(stats)
var_embeds = SharedMLP(stats_normalized)  # [n_vars, hidden_dim]
aggregated = mean_pool(var_embeds)  # [hidden_dim]
z_traj = Projector(aggregated)  # [latent_dim]
```

**Mathematical Properties:**
- **Permutation invariance:** The mean pooling ensures that reordering variables does not change the encoding (up to the shared MLP processing).
- **Dimension agnosticism:** The architecture handles any $n$ by processing variables independently before aggregation.

#### 2.2.2 Term Embedder

The term embedder creates structural representations of polynomial terms that generalize across dimensions.

**Key Insight:** A polynomial term can be uniquely represented by its exponent vector. For a term $\theta = x_1^{p_1} x_2^{p_2} \cdots x_n^{p_n}$, we represent it as the set of (variable index, power) pairs with non-zero powers:

$$\mathcal{P}(\theta) = \{(i, p_i) : p_i > 0\}$$

For example, $x^2 y$ is represented as $\{(0, 2), (1, 1)\}$. This representation is dimension-agnostic.

**Architecture:**

Let $\mathbf{V} \in \mathbb{R}^{n_{\max} \times e}$ and $\mathbf{P} \in \mathbb{R}^{(p_{\max}+1) \times e}$ denote learnable embedding matrices for variables and powers respectively, where $e$ is the internal embedding dimension. For a term with power representation $\mathcal{P}(\theta) = \{(i_1, p_1), \ldots, (i_k, p_k)\}$:

1. **Factor embedding:** For each factor $(i_j, p_j)$, compute:
   $$\mathbf{f}_j = \mathbf{V}_{i_j} \odot \mathbf{P}_{p_j} \in \mathbb{R}^e$$
   where $\odot$ denotes the Hadamard product (element-wise multiplication).

2. **Aggregation:** Sum all factor embeddings:
   $$\mathbf{e}_{\text{raw}} = \sum_{j=1}^k \mathbf{f}_j$$

3. **Projection:** Apply an MLP to obtain the final embedding:
   $$\mathbf{e}_{\text{term}} = \text{MLP}_{\text{proj}}(\mathbf{e}_{\text{raw}}) \in \mathbb{R}^d$$

For the constant term (empty power set), a learnable embedding $\mathbf{c} \in \mathbb{R}^e$ is used.

```python
# Learnable embeddings
variable_embedding = nn.Embedding(max_vars, embed_dim)  # V
power_embedding = nn.Embedding(max_power + 1, embed_dim)  # P

def embed_term(powers: List[Tuple[int, int]]) -> Tensor:
    """Embed a term like x^2*y -> [(0,2), (1,1)]"""
    factor_embeds = []
    for var_idx, power in powers:
        if power > 0:
            v_embed = variable_embedding(var_idx)
            p_embed = power_embedding(power)
            factor_embeds.append(v_embed * p_embed)  # Hadamard product

    # Aggregate factors (sum for permutation invariance)
    return projector(sum(factor_embeds)) if factor_embeds else projector(const_embed)
```

**Mathematical Justification:** The Hadamard product $\mathbf{V}_i \odot \mathbf{P}_p$ creates a unique encoding for each (variable, power) combination. This is preferable to addition because $\mathbf{V}_i + \mathbf{P}_p$ would yield $\mathbf{V}_0 + \mathbf{P}_2 = \mathbf{V}_1 + \mathbf{P}_1$ if embeddings happen to align, conflating distinct terms like $x^2$ and $y$.

**Examples:**
| Term | Powers Representation $\mathcal{P}(\theta)$ | Embedding Formula |
|------|----------------------|-----------|
| $1$ | $\emptyset$ | $\text{MLP}(\mathbf{c})$ |
| $x$ | $\{(0, 1)\}$ | $\text{MLP}(\mathbf{V}_0 \odot \mathbf{P}_1)$ |
| $y^2$ | $\{(1, 2)\}$ | $\text{MLP}(\mathbf{V}_1 \odot \mathbf{P}_2)$ |
| $xy$ | $\{(0, 1), (1, 1)\}$ | $\text{MLP}((\mathbf{V}_0 \odot \mathbf{P}_1) + (\mathbf{V}_1 \odot \mathbf{P}_1))$ |
| $x^2yz$ | $\{(0, 2), (1, 1), (2, 1)\}$ | $\text{MLP}((\mathbf{V}_0 \odot \mathbf{P}_2) + (\mathbf{V}_1 \odot \mathbf{P}_1) + (\mathbf{V}_2 \odot \mathbf{P}_1))$ |

#### 2.2.3 Equation Encoder (RelativeEquationEncoder)

The equation encoder must produce embeddings for equation indices that generalize across different system dimensions. A naive approach using a fixed embedding table $\mathbf{Q} \in \mathbb{R}^{n_{\max} \times d}$ fails because the meaning of "equation 2" differs between a 3D and 10D system.

**Solution: Relative Position Encoding**

We encode equations using relative positional features that are dimension-agnostic:

$$f_{\text{eq}}(i, n) = \text{MLP}\left(\left[ \frac{i}{n-1}, \frac{n}{10}, \mathbb{1}_{i=0}, \mathbb{1}_{i=n-1} \right]\right)$$

where:
- $\frac{i}{n-1} \in [0, 1]$: Relative position within the system (normalized index)
- $\frac{n}{10}$: Normalized dimension count (assuming $n \leq 10$ typically)
- $\mathbb{1}_{i=0}$: Binary indicator for first equation
- $\mathbb{1}_{i=n-1}$: Binary indicator for last equation

This encoding allows the network to learn patterns like "first equation" or "middle equation" that transfer across dimensions.

**Mathematical Properties:**
- **Dimension-agnostic:** The same encoder handles any $n$ without retraining.
- **Order-preserving:** Relative positions maintain ordering information.
- **Boundary-aware:** Binary indicators provide explicit signals for edge cases.

#### 2.2.4 Matching Network

The matching network predicts whether a term should be active in a given equation.

**Architecture (V2 - Hadamard Interaction):**

Let $\mathbf{z} \in \mathbb{R}^d$, $\mathbf{e}_j \in \mathbb{R}^d$, and $\mathbf{q}_i \in \mathbb{R}^d$ denote the trajectory encoding, term embedding for term $j$, and equation embedding for equation $i$, respectively.

1. **Projection:** Apply learned linear projections:
   $$\tilde{\mathbf{z}} = \mathbf{W}_z \mathbf{z}, \quad \tilde{\mathbf{e}}_j = \mathbf{W}_e \mathbf{e}_j, \quad \tilde{\mathbf{q}}_i = \mathbf{W}_q \mathbf{q}_i$$

2. **Normalization:** Project to unit sphere to prevent gradient explosion:
   $$\hat{\mathbf{z}} = \frac{\tilde{\mathbf{z}}}{\|\tilde{\mathbf{z}}\|_2 + \epsilon}, \quad \hat{\mathbf{e}}_j = \frac{\tilde{\mathbf{e}}_j}{\|\tilde{\mathbf{e}}_j\|_2 + \epsilon}, \quad \hat{\mathbf{q}}_i = \frac{\tilde{\mathbf{q}}_i}{\|\tilde{\mathbf{q}}_i\|_2 + \epsilon}$$
   where $\epsilon = 10^{-8}$ for numerical stability.

3. **Hadamard interaction:** Compute the element-wise product:
   $$\mathbf{m}_{ij} = \hat{\mathbf{z}} \odot \hat{\mathbf{e}}_j \odot \hat{\mathbf{q}}_i \in \mathbb{R}^d$$

4. **Classification:** Apply an MLP classifier with sigmoid output:
   $$P(S_{ij} = 1 \mid \mathbf{X}) = \sigma(\text{MLP}(\mathbf{m}_{ij}))$$

```python
# Project embeddings
z_traj = traj_proj(trajectory_encoding)  # [batch, latent_dim]
e_term = term_proj(term_embeddings)      # [n_terms, latent_dim]
e_eq = eq_proj(equation_embeddings)      # [n_vars, latent_dim]

# Normalize to unit sphere (prevents gradient explosion)
z_traj = z_traj / (z_traj.norm(dim=-1, keepdim=True) + eps)
e_term = e_term / (e_term.norm(dim=-1, keepdim=True) + eps)
e_eq = e_eq / (e_eq.norm(dim=-1, keepdim=True) + eps)

# Hadamard interaction
interaction = z_traj[:, None, None, :] * e_term[None, None, :, :] * e_eq[None, :, None, :]
# Shape: [batch, n_vars, n_terms, latent_dim]

# Classify
probs = Classifier(interaction)  # [batch, n_vars, n_terms]
```

**Mathematical Motivation:** The Hadamard product implements a form of gating or multiplicative attention. For the prediction $P(S_{ij}=1|\mathbf{X})$ to be high, all three embeddings must be "aligned" in the sense that their element-wise product yields consistently positive (or consistently negative) values that the classifier can distinguish.

This is related to tensor factorization methods: if we view the structure prediction as a 3-way tensor $\mathcal{S} \in \mathbb{R}^{B \times n \times m}$ (batch, equations, terms), the Hadamard interaction followed by an MLP approximates a Tucker decomposition where the core tensor is learned by the classifier.

**Note on terminology:** While the document refers to this as "bilinear," the operation is technically a three-way Hadamard product followed by a nonlinear classifier, not a true bilinear form $\mathbf{z}^\top \mathbf{W} \mathbf{e}$.

---

## 3. Training Methodology

### 3.1 Data Generation

For each dynamical system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ in the training set:

1. **Initial conditions:** Sample $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, 2\mathbf{I}_n)$
2. **Integration:** Numerically solve the ODE for $t \in [0, 50]$ to obtain $T = 5000$ samples
3. **Noise injection:** Add i.i.d. Gaussian noise: $\tilde{\mathbf{x}}(t) = \mathbf{x}(t) + \eta \cdot \boldsymbol{\epsilon}(t)$ where $\boldsymbol{\epsilon}(t) \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_n)$ and $\eta \in \{0, 0.05, 0.10\}$
4. **Transient removal:** Discard first and last 100 samples to remove integration artifacts
5. **Ground truth extraction:** Parse the symbolic system definition to obtain the binary structure matrix $\mathbf{S}^* \in \{0,1\}^{n \times m}$

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

The network is trained using Binary Cross-Entropy (BCE) loss on the structure prediction. For a batch of $B$ samples, let $\hat{p}_{b,i,j} = P(S_{ij}=1 | \mathbf{X}_b)$ denote the predicted probability and $s_{b,i,j} \in \{0,1\}$ denote the ground truth. The loss is:

$$\mathcal{L} = -\frac{1}{B \cdot n \cdot m} \sum_{b=1}^{B} \sum_{i=1}^{n} \sum_{j=1}^{m} \left[ s_{b,i,j} \log(\hat{p}_{b,i,j}) + (1 - s_{b,i,j}) \log(1 - \hat{p}_{b,i,j}) \right]$$

where $n$ is the number of equations (variables) and $m = |\mathcal{L}_{n,p}|$ is the number of library terms.

```python
loss = BCELoss(predicted_probs, true_structure.float())
```

**Note on class imbalance:** Structure matrices are typically sparse (most terms are inactive). However, we do not apply class weighting because: (1) the sparsity level varies across systems, and (2) empirical results show that the network learns to predict sparse structures without explicit balancing.

### 3.5 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | $\beta_1 = 0.9$, $\beta_2 = 0.999$ |
| Learning Rate | $5 \times 10^{-4}$ | Fixed, no scheduler |
| Batch Size | 16 | Grouped by dimension within batch |
| Epochs | 100 | Maximum |
| Early Stopping | 15 epochs patience | Based on validation loss |
| Validation Split | 10% | Stratified by system type |
| Latent Dimension $d$ | 64 | Shared across all components |
| Internal Embed Dim $e$ | 32 | For term embedder |
| Dropout | 0.2 | In classifier MLP |

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

### 5.1 Evaluation Metrics

For a predicted structure $\hat{\mathbf{S}}$ and ground truth $\mathbf{S}^*$, we compute:

- **Precision:** $\displaystyle \text{Prec} = \frac{|\{(i,j) : \hat{S}_{ij} = 1 \land S^*_{ij} = 1\}|}{|\{(i,j) : \hat{S}_{ij} = 1\}|}$ (fraction of predicted terms that are correct)

- **Recall:** $\displaystyle \text{Rec} = \frac{|\{(i,j) : \hat{S}_{ij} = 1 \land S^*_{ij} = 1\}|}{|\{(i,j) : S^*_{ij} = 1\}|}$ (fraction of true terms that are recovered)

- **F1 Score:** $\displaystyle \text{F1} = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ (harmonic mean of precision and recall)

The predicted structure is obtained by thresholding: $\hat{S}_{ij} = \mathbb{1}[P(S_{ij}=1|\mathbf{X}) > \tau]$ with $\tau = 0.5$.

### 5.2 Test Set Performance

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

### 5.3 Key Observations

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
3. **Attention Mechanisms:** Replace Hadamard interaction with cross-attention
4. **Meta-Learning:** Few-shot adaptation to new system types
5. **Uncertainty Quantification:** Predict confidence intervals on structure (MC Dropout is already implemented)

---

## 10. Conclusion

The Factorized Structure Network represents a significant advance in dimension-agnostic equation discovery. By decomposing the prediction problem into trajectory encoding, term embedding, and matching, we achieve:

- **Generalization across dimensions** (2D, 3D, 4D with single model)
- **Zero-shot transfer** to unseen system types
- **Strong benchmark performance** (F1 = 0.89 on Lorenz, never trained)

This architecture enables scalable structure prediction for dynamical systems of any dimension, paving the way for automated discovery of governing equations in complex real-world systems.

---

## Appendix A: Mathematical Notation Summary

| Symbol | Definition |
|--------|------------|
| $n$ | Number of state variables (system dimension) |
| $T$ | Number of time samples in trajectory |
| $p$ | Maximum polynomial order |
| $d$ | Latent embedding dimension |
| $e$ | Internal term embedding dimension |
| $\mathbf{X} \in \mathbb{R}^{T \times n}$ | Trajectory matrix |
| $\mathcal{L}_{n,p}$ | Polynomial library with $n$ variables and max order $p$ |
| $m = \|\mathcal{L}_{n,p}\|$ | Number of library terms |
| $\mathbf{S} \in \{0,1\}^{n \times m}$ | Binary structure matrix |
| $S_{ij}$ | Indicator: term $j$ active in equation $i$ |
| $\mathbf{z} \in \mathbb{R}^d$ | Trajectory encoding |
| $\mathbf{e}_j \in \mathbb{R}^d$ | Embedding for term $j$ |
| $\mathbf{q}_i \in \mathbb{R}^d$ | Embedding for equation $i$ |
| $\odot$ | Hadamard (element-wise) product |
| $\sigma(\cdot)$ | Sigmoid function: $\sigma(x) = (1 + e^{-x})^{-1}$ |

---

## Appendix B: Library Size Formula

The polynomial library $\mathcal{L}_{n,p}$ contains all monomials $x_1^{k_1} x_2^{k_2} \cdots x_n^{k_n}$ where $\sum_{i=1}^n k_i \leq p$ and $k_i \geq 0$. The number of such terms equals:

$$|\mathcal{L}_{n,p}| = \binom{n + p}{p} = \frac{(n+p)!}{n! \, p!}$$

**Derivation:** This follows from a stars-and-bars argument. We need to count non-negative integer solutions to $k_1 + k_2 + \cdots + k_n + s = p$ where $s \geq 0$ represents the "slack" (unused degree). This is equivalent to distributing $p$ indistinguishable balls into $n+1$ distinguishable bins, giving $\binom{(n+1) + p - 1}{p} = \binom{n+p}{p}$.

**Verification via direct counting:**
- $n=2, p=3$: $\binom{5}{3} = 10$ terms: $\{1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3\}$
- $n=3, p=2$: $\binom{5}{2} = 10$ terms: $\{1, x, y, z, x^2, xy, xz, y^2, yz, z^2\}$
- $n=3, p=3$: $\binom{6}{3} = 20$ terms

---

## Appendix C: Complexity Analysis

**Computational Complexity per Forward Pass:**

Let $B$ = batch size, $n$ = number of variables, $m$ = number of terms, $d$ = latent dimension.

| Component | Complexity | Dominant Operations |
|-----------|------------|---------------------|
| Statistics extraction | $O(T \cdot n)$ | Per-variable statistics |
| Trajectory encoding | $O(n \cdot d^2)$ | MLP forward pass |
| Term embedding | $O(m \cdot d^2)$ | MLP projection |
| Equation encoding | $O(n \cdot d^2)$ | MLP forward pass |
| Hadamard interaction | $O(B \cdot n \cdot m \cdot d)$ | Element-wise products |
| Classification | $O(B \cdot n \cdot m \cdot d^2)$ | MLP forward pass |

**Total:** $O(B \cdot n \cdot m \cdot d^2)$ where $m = O(n^p / p!)$

**Memory Complexity:** $O(B \cdot n \cdot m \cdot d)$ for storing interaction tensors.

The key advantage of the factorized approach is that term embeddings can be precomputed and cached for a given $(n, p)$ configuration, reducing the per-sample cost.
