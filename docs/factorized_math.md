# Mathematical Foundations of the Factorized Structure Network

## 1. Problem Formulation

### 1.1 The SINDy Structure Prediction Problem

Given a dynamical system with state **x**(t) in R^n, we seek to identify the governing equations:

```
dx/dt = f(x)
```

In SINDy, we approximate **f** as a sparse linear combination of candidate functions from a library Theta(**x**):

```
dx/dt = Theta(x) * Xi
```

where:
- **Theta(x)** in R^(T x p) is the library matrix (polynomials up to order k)
- **Xi** in R^(p x n) is the sparse coefficient matrix
- T = number of time samples, p = number of library terms, n = state dimension

**Structure Prediction Goal:** Given trajectory data **X** in R^(T x n), predict the binary support mask **S** in {0,1}^(n x p) where S_ij = 1 iff term j appears in equation i.

### 1.2 The Dimension Problem

For polynomial libraries of order k, the number of terms grows as:

```
p(n, k) = C(n+k, k) = (n+k)! / (n! * k!)
```

| Dimension n | Order k=3 | Terms p |
|-------------|-----------|---------|
| 2           | 3         | 10      |
| 3           | 3         | 20      |
| 4           | 3         | 35      |
| 5           | 3         | 56      |

A traditional neural network f_theta: R^(T x n) -> R^(n x p) has:
- Input dimension dependent on n
- Output dimension dependent on both n and p(n,k)

**This makes cross-dimension generalization impossible.**

---

## 2. The Factorization Principle

### 2.1 Core Insight: Decompose the Prediction

Instead of directly predicting the full structure matrix, we factorize the prediction into independent components:

```
P(S_ij = 1 | X) = g(phi(X), psi(term_j), eta(i))
```

where:
- **phi: R^(T x n) -> R^d** - Trajectory Encoder (dimension-agnostic)
- **psi: Term -> R^d** - Term Embedder (structure-based)
- **eta: {1,...,n} -> R^d** - Equation Embedder (index-based)
- **g: R^d x R^d x R^d -> [0,1]** - Matching Function

**Key Property:** Each component has fixed output dimension d, regardless of n.

### 2.2 Mathematical Justification

The structure prediction can be viewed as a **multi-relational prediction problem**:

```
P = {(i, j, X) : S_ij = 1}
```

We're predicting a ternary relation between:
1. Equation index i
2. Library term j
3. Trajectory X

This is analogous to **knowledge graph completion** or **tensor factorization**, where we predict missing entries in a 3-way tensor.

---

## 3. Component Mathematics

### 3.1 Trajectory Encoder: Sufficient Statistics

**Goal:** Map X in R^(T x n) to z in R^d for any n.

**Approach:** Extract per-variable statistics, then aggregate.

For each state variable x_i(t), compute statistics **s_i** in R^m:

```
s_i = [E[x_i], Std[x_i], Skew[x_i], Kurt[x_i], E[x_i^2], max-min, Median[x_i], E[|dx_i/dt|]]^T
```

This gives s_i in R^8.

**Encoding Architecture:**

```
h_i = MLP_theta(s_i)    for all i in {1, ..., n}

z_traj = MLP_phi(mean(h_1, ..., h_n))   in R^d
```

**Why This Works:**
1. **Per-variable processing:** The same MLP processes each variable's statistics
2. **Permutation invariance:** Mean pooling is symmetric over variables
3. **Dimension agnostic:** Works for any n since we aggregate over variables

**Theoretical Justification:** For ergodic dynamical systems, sample statistics converge to population moments, which characterize the invariant measure. Different dynamical systems have different invariant measures, making statistics informative for structure prediction.

### 3.2 Term Embedder: Structural Representation

**Goal:** Map any polynomial term to **e** in R^d.

**Key Insight:** A term like x_1^2 * x_3 can be represented as a **multiset** of (variable, power) pairs:

```
x_1^2 * x_3  <=>  {(1, 2), (3, 1)}
```

This representation is:
- **Dimension-agnostic:** Variable indices are just integers
- **Order-independent:** The multiset is unordered
- **Compositional:** Complex terms are built from simple factors

**Embedding Architecture:**

Given learnable embeddings:
- **V** in R^(n_max x d') - variable embeddings
- **P** in R^((k+1) x d') - power embeddings

For term t with factors {(v_1, p_1), ..., (v_m, p_m)}:

```
e_t = MLP_psi(sum_{j=1}^m (V[v_j] + P[p_j]))   in R^d
```

**Special Cases:**
- Constant term (1): Uses learned embedding **c** in R^d'
- Linear term (x_i): V[i] + P[1]
- Quadratic (x_i^2): V[i] + P[2]
- Bilinear (x_i * x_j): (V[i] + P[1]) + (V[j] + P[1])

**Why Summation for Aggregation:**
- Preserves **permutation invariance** (order of factors doesn't matter)
- Allows **compositionality** (complex terms = sum of simple factors)
- Captures **multiplicity** implicitly through power embeddings

### 3.3 Equation Embedder

**Goal:** Distinguish predictions for dx_1/dt vs dx_2/dt vs dx_3/dt.

Simple learnable embedding table:

```
e_eq^(i) = E[i]   in R^d    where E in R^(n_max x d)
```

**Interpretation:** The equation embedding captures which output variable we're predicting, allowing the model to learn that certain terms are more likely in certain equations (e.g., x terms more common in dx/dt).

### 3.4 Matching Function: Bilinear Interaction

**Goal:** Combine the three embeddings to predict term activation probability.

**V1 Architecture (Concatenation + MLP):**

```
P(S_ij = 1) = sigmoid(MLP([z_traj; e_term^(j); e_eq^(i)]))
```

**V2 Architecture (Bilinear + Classifier):**

```
z' = W_z * z_traj / ||W_z * z_traj||
e'_t = W_t * e_term / ||W_t * e_term||
e'_i = W_e * e_eq / ||W_e * e_eq||

h_ij = z' * e'_t * e'_i    (element-wise product)   in R^d

P(S_ij = 1) = sigmoid(MLP(h_ij))
```

**Why Bilinear (Hadamard Product)?**

1. **Multiplicative Interaction:** The element-wise product captures whether all three components "agree" - high values in all three produce high activation.

2. **Efficient Computation:** Can compute all (i, j) pairs in parallel:
   ```
   H = z'[B,1,1,d] * e'_t[1,1,p,d] * e'_i[1,n,1,d]   in R^(B x n x p x d)
   ```

3. **Theoretical Connection:** Related to **tensor factorization** where we decompose a 3-way tensor as sum of outer products.

**L2 Normalization:** Projecting embeddings to the unit sphere:
- Prevents gradient explosion during training
- Makes dot products interpretable as cosine similarity
- Stabilizes the bilinear interaction

---

## 4. Training Objective

### 4.1 Loss Function

Binary cross-entropy over all (equation, term) pairs:

```
L = -(1/(n*p)) * sum_{i,j} [S_ij * log(P_ij) + (1 - S_ij) * log(1 - P_ij)]
```

where P_ij = P(S_ij = 1 | X).

### 4.2 Class Imbalance

Structure matrices are highly sparse (typically 5-15% nonzero). Options:
- **Weighted BCE:** Upweight positive class
- **Focal loss:** Focus on hard negatives
- **Balanced sampling:** Sample equal positive/negative pairs

Current implementation: Standard BCE (works reasonably well due to model's inductive bias).

### 4.3 Mixed-Dimension Training

Training on systems of different dimensions simultaneously:

```
L_total = sum_{d in {2,3,4}} sum_{X in D_d} L(X, S_X)
```

**Batching Strategy:** Group samples by dimension within each batch to enable efficient tensor operations.

---

## 5. Theoretical Properties

### 5.1 Dimension Invariance

**Theorem (Informal):** The factorized architecture can process inputs of any dimension n and produce valid structure predictions for n equations with p(n,k) terms.

**Proof Sketch:**
1. Trajectory encoder: Mean pooling over n variables -> fixed R^d
2. Term embedder: Summation over variable factors -> fixed R^d
3. Equation embedder: Lookup in R^(n_max x d) -> fixed R^d
4. Matching: Fixed-size inputs -> valid probability output

### 5.2 Generalization Bounds

The model's generalization depends on:

1. **Structural similarity:** If test systems share structural patterns with training (e.g., bilinear terms xy), the term embeddings transfer.

2. **Statistical diversity:** If training trajectories cover diverse dynamical regimes, the trajectory encoder learns robust features.

3. **Dimension transfer:** Since embeddings are structural (not positional), patterns learned for x_1*x_2 in 2D can apply to x_3*x_4 in 4D.

### 5.3 Expressiveness

**What the model can learn:**
- Correlations between trajectory statistics and term presence
- Structural patterns (e.g., "cubic terms common in oscillators")
- Cross-term dependencies (e.g., "if xy present, likely x and y also present")

**What the model cannot learn (directly):**
- Exact coefficient values
- Phase space geometry
- Long-term dynamics beyond statistics

---

## 6. Connection to Related Methods

### 6.1 Tensor Factorization

The prediction can be viewed as factorizing a 3-way tensor:

```
T_ijk ~ sum_{r=1}^R z_i^(r) * e_j^(r) * q_k^(r)
```

where:
- i indexes trajectories
- j indexes terms
- k indexes equations

Our bilinear model is a **neural tensor factorization** with shared encoders.

### 6.2 Knowledge Graph Embeddings

Similar to **DistMult** or **ComplEx** for link prediction:

```
P(relation) = sigmoid(<h, r, t>)
```

Our model: P(term active) = sigmoid(<z_traj, e_term, e_eq>)

### 6.3 Set Functions (DeepSets)

The trajectory encoder follows the **DeepSets** architecture:

```
phi(X) = rho(sum_{i=1}^n f(s_i))
```

which is **universal** for permutation-invariant functions over sets.

---

## 7. Summary of Mathematical Contributions

| Component | Mathematical Principle | Why It Works |
|-----------|----------------------|--------------|
| **Trajectory Encoder** | Sufficient statistics + mean pooling | Dimension-agnostic via set function |
| **Term Embedder** | Compositional embeddings + sum aggregation | Structural representation of polynomial terms |
| **Equation Embedder** | Learned index embeddings | Distinguishes which derivative we're predicting |
| **Matching Function** | Bilinear interaction + sigmoid | Captures multiplicative agreement between embeddings |
| **Training** | Mixed-dimension BCE | Learns shared patterns across dimensions |

**Key Insight:** By decomposing the prediction into trajectory encoding, term embedding, and matching, we achieve **dimension-agnostic structure prediction** while maintaining the expressiveness needed for accurate predictions.

---

## 8. Empirical Validation

The mathematical principles are validated by:

1. **Lorenz benchmark:** F1 = 0.892 on a 3D system never seen during training
2. **Cross-dimension transfer:** Training on 2D+3D enables 4D prediction
3. **Structural generalization:** Different systems with similar term structures show consistent predictions

### 8.1 Test Results Summary

| System | Dim | Precision | Recall | F1 |
|--------|-----|-----------|--------|-----|
| **Lorenz** (never trained) | 3D | 1.000 | 0.807 | **0.892** |
| LinearOscillator | 2D | 0.788 | 1.000 | 0.879 |
| DampedHarmonic | 2D | 0.778 | 0.900 | 0.827 |
| CoupledFitzHughNagumo | 4D | 0.701 | 0.621 | 0.650 |

---

## 9. Implementation Reference

The mathematical concepts map to code as follows:

| Concept | File | Class/Function |
|---------|------|----------------|
| Trajectory Encoder | `trajectory_encoder.py` | `StatisticsEncoder` |
| Term Embedder | `term_embedder.py` | `TermEmbedder` |
| Bilinear Matching | `factorized_network.py` | `FactorizedStructureNetworkV2` |
| Mixed-dim Training | `training.py` | `train_factorized_network` |
| Statistics Extraction | `trajectory_encoder.py` | `extract_per_variable_stats` |
| Term-to-Powers | `term_representation.py` | `term_name_to_powers` |
