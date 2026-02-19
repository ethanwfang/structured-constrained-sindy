# Kernel Methods for Dimension-Agnostic Structure Prediction

## Mathematical Foundations

This document presents the mathematical foundations of the kernel-based approach to predicting sparse structure in dynamical systems equations. We show how kernel methods provide a principled framework for dimension-agnostic learning.

---

## 1. Problem Formulation

### 1.1 The Structure Prediction Problem

Given a trajectory $\mathbf{X} \in \mathbb{R}^{T \times n}$ from an $n$-dimensional dynamical system:

$$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$$

we seek to identify which terms in a polynomial library $\Theta(\mathbf{x})$ are active in each equation.

The library consists of polynomial terms up to order $p$:

$$\Theta(\mathbf{x}) = [1, x_1, x_2, \ldots, x_1^2, x_1 x_2, \ldots, x_1^p, \ldots]$$

The number of terms grows combinatorially:

$$|\Theta| = \binom{n + p}{p}$$

For example, with $p=3$: 2D → 10 terms, 3D → 20 terms, 4D → 35 terms.

### 1.2 Structure as Binary Classification

For each equation $i \in \{1, \ldots, n\}$ and term $j \in \{1, \ldots, |\Theta|\}$, we define:

$$y_{ij} = \begin{cases} 1 & \text{if term } j \text{ appears in equation } i \\ 0 & \text{otherwise} \end{cases}$$

The goal is to learn a predictor:

$$\hat{y}_{ij} = P(\text{term } j \text{ active in equation } i \mid \mathbf{X})$$

### 1.3 The Dimension Barrier

Traditional neural networks require fixed input/output dimensions, making them dimension-specific. A network trained for 2D systems cannot process 3D trajectories.

**Our solution:** Use kernel methods to define similarity in a dimension-agnostic space.

---

## 2. Kernel Methods Background

### 2.1 Kernel Functions

A kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a function that computes an inner product in some (possibly infinite-dimensional) feature space $\mathcal{H}$:

$$k(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_{\mathcal{H}}$$

where $\phi: \mathcal{X} \to \mathcal{H}$ is the feature map.

**Key property:** We can compute similarities without explicitly computing $\phi$.

### 2.2 Mercer's Theorem

A symmetric function $k(\mathbf{x}, \mathbf{x}')$ is a valid kernel if and only if the kernel matrix $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite for any finite set of points.

### 2.3 Common Kernels

| Kernel | Formula | Properties |
|--------|---------|------------|
| Linear | $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$ | Simplest, linear decision boundaries |
| Polynomial | $k(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^\top \mathbf{x}' + c)^d$ | Captures polynomial interactions |
| RBF/Gaussian | $k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ | Universal approximator |

---

## 3. Our Approach: Neural Kernels for Structure Prediction

### 3.1 Factorized Kernel Formulation

We define a kernel over three components:

1. **Trajectory representation** $\mathbf{z} \in \mathbb{R}^d$: encodes the dynamical behavior
2. **Term representation** $\mathbf{e}_t \in \mathbb{R}^d$: encodes the polynomial term type
3. **Equation representation** $\mathbf{e}_q \in \mathbb{R}^d$: encodes which equation

The kernel function computes compatibility:

$$k(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q) \in \mathbb{R}$$

The probability of term $j$ being active in equation $i$ is:

$$P(y_{ij} = 1 \mid \mathbf{X}) = \sigma\big(k(\mathbf{z}_{\mathbf{X}}, \mathbf{e}_{t_j}, \mathbf{e}_{q_i})\big)$$

where $\sigma(\cdot)$ is the sigmoid function.

### 3.2 Dimension-Agnostic Trajectory Encoding

The trajectory encoder must map trajectories of any dimension to a fixed representation. We use **permutation-invariant statistics**:

$$\mathbf{z} = g\left( \frac{1}{n} \sum_{i=1}^{n} h(\mathbf{s}_i) \right)$$

where:
- $\mathbf{s}_i \in \mathbb{R}^m$ are statistics for variable $i$ (mean, std, skewness, kurtosis, energy, range, median, derivative magnitude)
- $h: \mathbb{R}^m \to \mathbb{R}^d$ is a shared encoder (MLP)
- $g: \mathbb{R}^d \to \mathbb{R}^d$ is a projection network

**Property:** This is permutation-invariant over variables and handles any $n$.

We enhance this with cross-variable attention:

$$\mathbf{h}_i' = \text{Attention}(\mathbf{h}_i, \{\mathbf{h}_j\}_{j=1}^n)$$

which captures variable interactions while maintaining permutation equivariance.

### 3.3 Semantic Term Encoding

Instead of position-based term embeddings, we encode **semantic properties**:

A term like $x_1^2 x_2$ is represented by its **power list**: $[(1, 2), (2, 1)]$ meaning "variable 1 to power 2, variable 2 to power 1".

We extract abstract features:

| Feature | Symbol | Description |
|---------|--------|-------------|
| Total degree | $d$ | Sum of all powers: $d = \sum_i p_i$ |
| Number of variables | $v$ | Count of distinct variables |
| Max power | $p_{\max}$ | Highest individual power |
| Has self-interaction | $\mathbb{1}[\exists p_i > 1]$ | Any squared or higher term |
| Is bilinear | $\mathbb{1}[v=2 \land \forall p_i=1]$ | Exactly two variables, both linear |
| Is pure power | $\mathbb{1}[v=1]$ | Single variable term |

The term embedding:

$$\mathbf{e}_t = f_{\text{term}}(\mathbf{a}_t, d_t, v_t)$$

where $\mathbf{a}_t$ is the abstract feature vector.

**Key insight:** Terms like $x_1 x_2$ in 2D and $x_1 x_2$ in 3D have the **same** semantic features (bilinear, degree 2, 2 variables) and thus similar embeddings.

### 3.4 Equation Encoding

Equations are encoded via learnable embeddings:

$$\mathbf{e}_{q_i} = \text{Embedding}(i), \quad i \in \{1, \ldots, n_{\max}\}$$

This captures equation-specific patterns (e.g., "first equation often has damping terms").

---

## 4. Kernel Function Variants

### 4.1 Linear Kernel

The simplest kernel uses projected dot products:

$$k_{\text{linear}}(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q) = \sum_{l=1}^{d} (W_z \mathbf{z})_l \cdot (W_t \mathbf{e}_t)_l \cdot (W_q \mathbf{e}_q)_l$$

where $W_z, W_t, W_q \in \mathbb{R}^{d \times d}$ are learnable projections.

This is equivalent to a trilinear form:

$$k_{\text{linear}} = \mathbf{z}^\top W_z^\top \text{diag}(W_t \mathbf{e}_t) W_q \mathbf{e}_q$$

### 4.2 Polynomial Kernel

Extends linear with polynomial expansion:

$$k_{\text{poly}}(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q) = \left( k_{\text{linear}}(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q) + c \right)^p$$

The degree $p$ and bias $c$ are hyperparameters (or learnable).

**Implicit feature space:** For $p=2$, this implicitly computes:
$$\phi(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q) = [\text{all monomials of degree} \leq 2]$$

### 4.3 RBF Kernel

Uses distance in the combined space:

$$k_{\text{RBF}}(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q) = \exp\left( -\gamma \| W_z\mathbf{z} - (W_t\mathbf{e}_t + W_q\mathbf{e}_q) \|^2 \right)$$

where $\gamma > 0$ controls the kernel width.

**Interpretation:** High kernel value when trajectory embedding is "close" to the term+equation embedding.

**Universal approximation:** RBF kernels can approximate any continuous function given enough data.

### 4.4 Bilinear Kernel

Uses separate bilinear forms for term and equation:

$$k_{\text{bilinear}}(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q) = (\mathbf{z}^\top W_t \mathbf{e}_t) \cdot (\mathbf{z}^\top W_q \mathbf{e}_q)$$

where $W_t, W_q \in \mathbb{R}^{d \times d}$ are learnable weight matrices.

**Interpretation:** Separately measures trajectory-term and trajectory-equation compatibility, then combines multiplicatively.

### 4.5 Neural Kernel

A fully learnable kernel via neural network:

$$k_{\text{neural}}(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q) = \text{MLP}([\mathbf{z}; \mathbf{e}_t; \mathbf{e}_q])$$

where $[\cdot;\cdot;\cdot]$ denotes concatenation.

**Architecture:**
$$k_{\text{neural}} = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 [\mathbf{z}; \mathbf{e}_t; \mathbf{e}_q] + b_1) + b_2) + b_3$$

**Expressiveness:** Can learn arbitrary nonlinear compatibility functions.

**Note:** This is technically not a valid Mercer kernel, but the neural tangent kernel (NTK) perspective shows it approximates a kernel in the infinite-width limit.

---

## 5. Training Objective

### 5.1 Binary Cross-Entropy Loss

We train by minimizing the binary cross-entropy:

$$\mathcal{L} = -\frac{1}{N} \sum_{s=1}^{N} \sum_{i=1}^{n_s} \sum_{j=1}^{|\Theta_s|} \left[ y_{ij}^{(s)} \log \hat{y}_{ij}^{(s)} + (1 - y_{ij}^{(s)}) \log(1 - \hat{y}_{ij}^{(s)}) \right]$$

where:
- $N$ is the number of training trajectories
- $n_s$ is the dimension of trajectory $s$ (varies!)
- $|\Theta_s|$ is the library size for trajectory $s$
- $y_{ij}^{(s)}$ is the ground truth structure
- $\hat{y}_{ij}^{(s)} = \sigma(k(\mathbf{z}^{(s)}, \mathbf{e}_{t_j}, \mathbf{e}_{q_i}))$

### 5.2 Connection to Metric Learning

The training objective can be viewed as metric learning:

$$\mathcal{L} = \sum_{\text{active } (i,j)} -\log \sigma(k(\mathbf{z}, \mathbf{e}_{t_j}, \mathbf{e}_{q_i})) + \sum_{\text{inactive } (i,j)} -\log(1 - \sigma(k(\mathbf{z}, \mathbf{e}_{t_j}, \mathbf{e}_{q_i})))$$

This pushes:
- **Active terms:** High kernel value → embeddings "align"
- **Inactive terms:** Low kernel value → embeddings "repel"

This is similar to contrastive learning objectives like InfoNCE.

### 5.3 Gradient Flow

For the neural kernel, gradients flow through:

1. **Kernel parameters** (MLP weights): Learn the compatibility function
2. **Trajectory encoder**: Learn discriminative trajectory features
3. **Term encoder**: Learn semantic term representations
4. **Equation encoder**: Learn equation-specific patterns

All components are jointly optimized end-to-end.

---

## 6. Theoretical Properties

### 6.1 Dimension Agnosticism

**Theorem:** The kernel structure network can process trajectories of any dimension $n$ without architectural changes.

**Proof:**
- Trajectory encoding uses mean-pooling over variables: $\mathbf{z} = g(\frac{1}{n}\sum_{i=1}^n h(\mathbf{s}_i))$
- Term encoding uses semantic features independent of $n$
- Equation encoding uses indices up to $n_{\max}$
- Kernel function operates on fixed-size embeddings

Thus, the network is dimension-agnostic by construction. $\square$

### 6.2 Permutation Invariance

**Theorem:** The trajectory encoding is invariant to permutation of state variables.

**Proof:** Mean-pooling is a symmetric function:
$$\frac{1}{n}\sum_{i=1}^n h(\mathbf{s}_i) = \frac{1}{n}\sum_{i=1}^n h(\mathbf{s}_{\pi(i)})$$
for any permutation $\pi$. $\square$

**Note:** This is desirable for physical systems where variable ordering is arbitrary.

### 6.3 Generalization Across Dimensions

**Hypothesis:** Training on systems of dimensions $\{d_1, \ldots, d_k\}$ enables generalization to unseen dimension $d'$.

**Intuition:** The semantic term encoding captures **structural patterns** (bilinear coupling, cubic nonlinearity) that transfer across dimensions. A network that learns "bilinear terms indicate coupled oscillator dynamics" can apply this knowledge regardless of dimension.

**Empirical evidence:** Training on 2D systems and testing on 3D Lorenz achieves F1 > 0.4, demonstrating cross-dimension transfer.

### 6.4 Kernel Interpretation

For the linear kernel, we can interpret the learned representations:

$$P(y_{ij}=1) = \sigma\left( \sum_l z_l \cdot (e_t)_l \cdot (e_q)_l \right)$$

Each dimension $l$ represents a "feature channel" that contributes to the decision:
- $z_l$: How much the trajectory exhibits feature $l$
- $(e_t)_l$: How much term type $t$ relates to feature $l$
- $(e_q)_l$: How much equation $q$ relates to feature $l$

High probability when all three are aligned (same sign and large magnitude).

---

## 7. Comparison with Factorized Bilinear Approach

| Aspect | Factorized Network | Kernel Network |
|--------|-------------------|----------------|
| Term representation | Position-based embedding | Semantic type features |
| Interaction | Fixed bilinear: $\mathbf{z} \odot \mathbf{e}_t \odot \mathbf{e}_q$ | Learnable kernel function |
| Kernel flexibility | Single form | 5 variants (linear, poly, RBF, bilinear, neural) |
| Theoretical basis | Deep learning | Kernel methods + metric learning |
| Interpretability | Embedding alignment | Explicit kernel similarity |

### 7.1 When to Use Each

**Factorized Network:**
- Simpler architecture
- Faster training (fixed bilinear computation)
- Good when term position matters

**Kernel Network:**
- More expressive (especially neural kernel)
- Better theoretical grounding
- Good when semantic term properties matter
- Enables kernel method tools (e.g., kernel PCA, GP interpretation)

---

## 8. Extensions

### 8.1 Gaussian Process Interpretation

The kernel formulation naturally extends to a Gaussian Process:

$$P(y_{ij} = 1 \mid \mathbf{X}) = \Phi\left( \frac{\mu(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q)}{\sqrt{1 + \sigma^2(\mathbf{z}, \mathbf{e}_t, \mathbf{e}_q)}} \right)$$

where:
- $\mu$ is the GP mean function
- $\sigma^2$ is the GP variance (uncertainty)
- $\Phi$ is the probit function

This provides **uncertainty quantification** for structure predictions.

### 8.2 Multiple Kernel Learning

Combine multiple kernels adaptively:

$$k_{\text{combined}} = \sum_{m=1}^{M} \alpha_m k_m$$

where $\alpha_m \geq 0$ are learned weights.

This can combine the strengths of different kernel types (e.g., RBF for smooth patterns, polynomial for interaction detection).

### 8.3 Attention as Kernel

The attention mechanism can be viewed as a kernel:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}} \right)$$

This is related to the softmax kernel:

$$k_{\text{softmax}}(\mathbf{q}, \mathbf{k}) = \exp\left( \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d}} \right)$$

Future work could incorporate attention-based kernels for richer trajectory-term interactions.

---

## 9. Conclusion

The kernel-based approach provides a mathematically principled framework for dimension-agnostic structure prediction:

1. **Kernels enable dimension-agnostic computation** by operating on fixed-size embeddings derived from variable-size inputs.

2. **Semantic term encoding** captures structural patterns that transfer across dimensions.

3. **Multiple kernel types** offer flexibility: linear for interpretability, RBF for universality, neural for expressiveness.

4. **Training as metric learning** aligns compatible (trajectory, term, equation) tuples in embedding space.

5. **Theoretical properties** (permutation invariance, dimension agnosticism) emerge naturally from the architecture.

This approach bridges classical kernel methods with modern deep learning, combining the theoretical foundations of SVMs with the flexibility of neural networks.

---

## References

1. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.

2. Hofmann, T., Schölkopf, B., & Smola, A. J. (2008). Kernel methods in machine learning. *The Annals of Statistics*, 36(3), 1171-1220.

3. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932-3937.

4. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS*.

5. Zaheer, M., et al. (2017). Deep sets. *NeurIPS*. (Permutation-invariant architectures)
