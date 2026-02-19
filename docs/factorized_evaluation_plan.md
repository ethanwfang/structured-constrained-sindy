# Factorized Structure Network: Comprehensive Evaluation Plan

## Overview

This document outlines a rigorous evaluation plan for the Factorized Structure Network to validate its effectiveness, understand its limitations, and compare it against baseline approaches.

---

## 1. Evaluation Objectives

| Objective | Question | Metrics |
|-----------|----------|---------|
| **Accuracy** | How well does it predict structure? | F1, Precision, Recall, Accuracy |
| **Generalization** | Does it transfer to unseen systems? | Cross-system F1, Zero-shot accuracy |
| **Dimension Transfer** | Does training on 2D/3D help 4D? | Leave-one-dimension-out F1 |
| **Robustness** | How does noise affect performance? | F1 vs noise level curves |
| **Efficiency** | How fast is training/inference? | Time per sample, GPU memory |
| **Comparison** | Is it better than alternatives? | vs. dimension-specific networks |

---

## 2. Evaluation Experiments

### 2.1 Experiment 1: In-Distribution Test Performance

**Goal:** Measure performance on held-out test systems within the training distribution.

**Protocol:**
```bash
python scripts/train_factorized_full.py --epochs 200 --trajectories 100
python scripts/evaluate_factorized.py --trajectories 50
```

**Metrics:**
- Per-system F1, Precision, Recall
- Per-dimension average F1
- Overall average F1
- Confusion matrix (aggregated across systems)

**Expected Output:**
```
============================================================
TEST SET RESULTS
============================================================
System                    Dim   F1      Prec    Recall
------------------------------------------------------------
DampedHarmonicOscillator  2D    0.XXX   0.XXX   0.XXX
LinearOscillator          2D    0.XXX   0.XXX   0.XXX
...
Lorenz                    3D    0.XXX   0.XXX   0.XXX
...
------------------------------------------------------------
2D Average:               F1 = 0.XXX
3D Average:               F1 = 0.XXX
4D Average:               F1 = 0.XXX
Overall Average:          F1 = 0.XXX
```

---

### 2.2 Experiment 2: Held-Out System Generalization

**Goal:** Evaluate on systems completely excluded from training and testing.

**Protocol:**
```bash
python scripts/evaluate_factorized.py --include-heldout --trajectories 50
```

**Held-Out Systems:**
- 2D: SimplePredatorPrey, QuadraticOscillator
- 3D: ThomasAttractor, SprottD

**Metrics:**
- F1 on held-out systems
- Comparison to test set performance (generalization gap)

---

### 2.3 Experiment 3: Leave-One-Dimension-Out

**Goal:** Test whether training on dimensions {A, B} helps performance on dimension C.

**Protocol:**
```python
# Train on 2D + 3D only, test on 4D
train_2d_3d_only → evaluate_4d

# Train on 2D + 4D only, test on 3D
train_2d_4d_only → evaluate_3d

# Train on 3D + 4D only, test on 2D
train_3d_4d_only → evaluate_2d
```

**Script:**
```python
# scripts/eval_leave_one_dim_out.py
from sc_sindy.evaluation.splits_factorized import (
    TRAIN_SYSTEMS_2D_FACTORIZED,
    TRAIN_SYSTEMS_3D_FACTORIZED,
    TRAIN_SYSTEMS_4D_FACTORIZED,
    TEST_SYSTEMS_2D_FACTORIZED,
    TEST_SYSTEMS_3D_FACTORIZED,
    TEST_SYSTEMS_4D_FACTORIZED,
)

experiments = [
    {
        "name": "Leave-4D-Out",
        "train": {2: TRAIN_SYSTEMS_2D, 3: TRAIN_SYSTEMS_3D},
        "test": {4: TEST_SYSTEMS_4D + TRAIN_SYSTEMS_4D},  # All 4D are "test"
    },
    {
        "name": "Leave-3D-Out",
        "train": {2: TRAIN_SYSTEMS_2D, 4: TRAIN_SYSTEMS_4D},
        "test": {3: TEST_SYSTEMS_3D + TRAIN_SYSTEMS_3D},
    },
    {
        "name": "Leave-2D-Out",
        "train": {3: TRAIN_SYSTEMS_3D, 4: TRAIN_SYSTEMS_4D},
        "test": {2: TEST_SYSTEMS_2D + TRAIN_SYSTEMS_2D},
    },
]
```

**Expected Insights:**
- Can 2D+3D training enable 4D zero-shot prediction?
- Is there cross-dimension transfer learning?

---

### 2.4 Experiment 4: Noise Robustness

**Goal:** Measure performance degradation as measurement noise increases.

**Protocol:**
```python
noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

for noise in noise_levels:
    evaluate_on_test_systems(noise_level=noise)
```

**Output:**
```
Noise Level | 2D F1 | 3D F1 | 4D F1 | Overall F1
------------|-------|-------|-------|------------
0.00        | 0.XXX | 0.XXX | 0.XXX | 0.XXX
0.01        | 0.XXX | 0.XXX | 0.XXX | 0.XXX
0.05        | 0.XXX | 0.XXX | 0.XXX | 0.XXX
0.10        | 0.XXX | 0.XXX | 0.XXX | 0.XXX
0.20        | 0.XXX | 0.XXX | 0.XXX | 0.XXX
```

**Visualization:**
- Line plot: F1 vs noise level (separate lines for 2D, 3D, 4D)
- Identify "noise tolerance threshold" where F1 drops below 0.7

---

### 2.5 Experiment 5: Trajectory Length Sensitivity

**Goal:** Determine minimum trajectory length for reliable predictions.

**Protocol:**
```python
trajectory_lengths = [100, 250, 500, 1000, 2000, 5000]

for length in trajectory_lengths:
    evaluate_on_test_systems(n_points=length)
```

**Expected Insights:**
- Minimum trajectory length for F1 > 0.8
- Trade-off between data quantity and prediction quality

---

### 2.6 Experiment 6: Comparison to Dimension-Specific Networks

**Goal:** Compare factorized network to dedicated 2D/3D networks.

**Protocol:**
1. Train factorized network on all systems
2. Train 2D-specific network on 2D systems only
3. Train 3D-specific network on 3D systems only
4. Compare performance on respective test sets

**Comparison Table:**
```
Model                | 2D Test F1 | 3D Test F1 | 4D Test F1 | Total Params
---------------------|------------|------------|------------|-------------
2D-Specific          | 0.XXX      | N/A        | N/A        | XXX K
3D-Specific          | N/A        | 0.XXX      | N/A        | XXX K
Factorized (all)     | 0.XXX      | 0.XXX      | 0.XXX      | XXX K
```

**Key Questions:**
- Does the factorized network match dimension-specific performance?
- What is the parameter efficiency gain?

---

### 2.7 Experiment 7: Ablation Studies

**Goal:** Understand contribution of each architectural component.

**Ablations:**

| Ablation | Modification | Expected Effect |
|----------|--------------|-----------------|
| No LayerNorm | Remove input normalization | Worse stability |
| No embedding normalization | Remove L2 normalization | Overflow issues |
| Mean → Sum pooling | Change aggregation method | Different results |
| Smaller latent dim (32) | Reduce capacity | Lower F1 |
| Larger latent dim (128) | Increase capacity | Marginal improvement |
| Linear classifier | Remove hidden layer | Lower F1 |
| Additive interaction | Replace bilinear with concat | Different behavior |

**Protocol:**
```python
ablations = [
    ("baseline", {}),
    ("no_layernorm", {"use_layernorm": False}),
    ("no_embed_norm", {"normalize_embeddings": False}),
    ("sum_pooling", {"pooling": "sum"}),
    ("latent_32", {"latent_dim": 32}),
    ("latent_128", {"latent_dim": 128}),
    ("linear_classifier", {"classifier_hidden": 0}),
    ("additive", {"interaction": "additive"}),
]

for name, config in ablations:
    train_and_evaluate(config)
```

---

### 2.8 Experiment 8: Training Data Scaling

**Goal:** Determine how performance scales with training data quantity.

**Protocol:**
```python
trajectories_per_system = [10, 20, 50, 100, 200, 500]

for n_traj in trajectories_per_system:
    train_with_n_trajectories(n_traj)
    evaluate()
```

**Expected Output:**
- Learning curve: F1 vs training data size
- Identify data efficiency (how much data needed for F1 > 0.8)

---

### 2.9 Experiment 9: Per-Term Analysis

**Goal:** Understand which term types are predicted well/poorly.

**Protocol:**
```python
term_categories = {
    "constant": ["1"],
    "linear": ["x", "y", "z", "w"],
    "quadratic_self": ["xx", "yy", "zz", "ww"],
    "quadratic_cross": ["xy", "xz", "xw", "yz", "yw", "zw"],
    "cubic": ["xxx", "xxy", "xyy", "xyz", ...],
}

for category, terms in term_categories.items():
    compute_category_metrics(category, terms)
```

**Output:**
```
Term Category     | Precision | Recall | F1
------------------|-----------|--------|------
Constant (1)      | 0.XXX     | 0.XXX  | 0.XXX
Linear            | 0.XXX     | 0.XXX  | 0.XXX
Quadratic (self)  | 0.XXX     | 0.XXX  | 0.XXX
Quadratic (cross) | 0.XXX     | 0.XXX  | 0.XXX
Cubic             | 0.XXX     | 0.XXX  | 0.XXX
```

**Expected Insights:**
- Are bilinear terms (xy) harder than self-interactions (xx)?
- Are cubic terms learned from limited examples?

---

### 2.10 Experiment 10: Downstream SINDy Performance

**Goal:** Evaluate impact on actual equation discovery.

**Protocol:**
1. Use factorized network to predict structure mask
2. Apply mask to SINDy regression
3. Compare discovered equations to ground truth

**Metrics:**
- Coefficient RMSE: ||ξ_pred - ξ_true||
- Trajectory prediction error (1-step, 10-step, 100-step)
- Stability: Does the discovered system remain bounded?

**Comparison:**
```
Method                          | Coef RMSE | Traj Error (100-step)
--------------------------------|-----------|----------------------
SINDy (no structure guidance)   | X.XXX     | X.XXX
SINDy + Oracle structure        | X.XXX     | X.XXX
SINDy + Factorized structure    | X.XXX     | X.XXX
```

---

## 3. Statistical Rigor

### 3.1 Multiple Runs

All experiments should be run with **5 random seeds** to account for:
- Random weight initialization
- Random trajectory initial conditions
- Random train/val splits

**Reporting:**
- Mean ± std for all metrics
- Statistical significance tests (paired t-test) for comparisons

### 3.2 Confidence Intervals

Report 95% confidence intervals:
```
F1 = 0.85 ± 0.03 (95% CI: [0.82, 0.88])
```

---

## 4. Visualization Plan

### 4.1 Required Plots

1. **Bar chart:** Per-system F1 scores (grouped by dimension)
2. **Line plot:** F1 vs noise level
3. **Line plot:** F1 vs trajectory length
4. **Learning curve:** F1 vs training data size
5. **Confusion matrix:** Aggregated across all systems
6. **Ablation table:** Performance impact of each component
7. **t-SNE:** Trajectory embeddings colored by system type
8. **Attention heatmap:** (if using attention) Which variables/terms matter

### 4.2 Example Visualizations

```python
# Bar chart of F1 scores
import matplotlib.pyplot as plt

systems = ["DampedHarmonic", "Linear", "Lorenz", ...]
f1_scores = [0.83, 0.88, 0.89, ...]
dimensions = ["2D", "2D", "3D", ...]
colors = {"2D": "blue", "3D": "green", "4D": "red"}

plt.bar(systems, f1_scores, color=[colors[d] for d in dimensions])
plt.xlabel("System")
plt.ylabel("F1 Score")
plt.title("Factorized Network Performance by System")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/per_system_f1.png")
```

---

## 5. Implementation Checklist

### 5.1 Scripts to Create

- [x] `scripts/train_factorized_full.py` - Main training script
- [x] `scripts/evaluate_factorized.py` - Main evaluation script
- [ ] `scripts/eval_leave_one_dim_out.py` - Leave-one-dimension-out experiment
- [ ] `scripts/eval_noise_robustness.py` - Noise sensitivity experiment
- [ ] `scripts/eval_trajectory_length.py` - Trajectory length experiment
- [ ] `scripts/eval_ablations.py` - Ablation studies
- [ ] `scripts/eval_data_scaling.py` - Training data scaling
- [ ] `scripts/eval_per_term.py` - Per-term category analysis
- [ ] `scripts/eval_downstream_sindy.py` - Downstream SINDy integration
- [ ] `scripts/compare_to_baseline.py` - Comparison to dimension-specific

### 5.2 Notebooks to Create

- [ ] `notebooks/factorized_analysis.ipynb` - Interactive exploration
- [ ] `notebooks/visualize_embeddings.ipynb` - t-SNE of embeddings
- [ ] `notebooks/error_analysis.ipynb` - Deep dive into failures

---

## 6. Timeline

| Week | Activities |
|------|------------|
| 1 | Run Experiments 1-3 (basic evaluation, held-out, leave-dim-out) |
| 2 | Run Experiments 4-5 (noise robustness, trajectory length) |
| 3 | Run Experiments 6-7 (baseline comparison, ablations) |
| 4 | Run Experiments 8-10 (scaling, per-term, downstream SINDy) |
| 5 | Statistical analysis, visualization, documentation |

---

## 7. Success Criteria

### 7.1 Minimum Viable Performance

| Metric | Target |
|--------|--------|
| Overall test F1 | > 0.70 |
| Lorenz F1 (unseen) | > 0.80 |
| 4D average F1 | > 0.50 |
| Noise tolerance (F1 > 0.7) | noise_level < 0.10 |

### 7.2 Stretch Goals

| Metric | Target |
|--------|--------|
| Overall test F1 | > 0.85 |
| All systems F1 | > 0.60 |
| Zero-shot 5D prediction | F1 > 0.50 |
| Downstream SINDy coefficient RMSE | < 0.1 |

---

## 8. Reporting Template

### 8.1 Experiment Report Structure

```markdown
# Experiment X: [Title]

## Setup
- Training config: epochs, lr, batch_size, ...
- Test systems: list
- Random seeds: 5

## Results

### Quantitative
| System | F1 (mean ± std) | Precision | Recall |
|--------|-----------------|-----------|--------|
| ...    | ...             | ...       | ...    |

### Qualitative
[Observations, patterns, failure modes]

## Discussion
[Key insights, limitations, next steps]

## Artifacts
- Model checkpoint: `models/exp_X/model.pt`
- Results JSON: `results/exp_X/results.json`
- Figures: `figures/exp_X/`
```

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting to train systems | Use held-out set, monitor train/val gap |
| 4D data scarcity | Add synthetic 4D systems, data augmentation |
| Computational cost | Use efficient batching, profile and optimize |
| Numerical instability | Gradient clipping, embedding normalization |
| Unfair comparison to baseline | Match compute budget, use same evaluation protocol |

---

## 10. Conclusion

This evaluation plan provides a comprehensive framework for validating the Factorized Structure Network. By systematically testing generalization, robustness, and efficiency, we can confidently assess whether this architecture advances the state of the art in dimension-agnostic equation discovery.

**Key experiments to prioritize:**
1. **Lorenz benchmark** (canonical test, never trained)
2. **Leave-one-dimension-out** (true zero-shot capability)
3. **Noise robustness** (practical applicability)
4. **Comparison to baseline** (value proposition)
