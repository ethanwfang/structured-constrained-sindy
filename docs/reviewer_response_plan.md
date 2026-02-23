# SC-SINDy Reviewer Response Implementation Plan

**Date:** February 2026
**Status:** Planning Complete
**Estimated Effort:** 15-20 hours

---

## Executive Summary

This plan addresses 8 critical issues raised by conference reviewers regarding the SC-SINDy evaluation methodology. Implementation is organized into 4 phases with clear dependencies.

---

## Phase 1: Statistical Infrastructure (Foundation)
**Priority: HIGHEST - Other issues depend on this**

### Issue 1: Increase Sample Size (n=20 → n=50+)

**Problem:** n=20 trials is inadequate for statistical reliability given high variance in noisy dynamical systems.

**Current State:**
- `scripts/comprehensive_benchmark.py` line 552: `n_trials: int = 20`

**Changes Required:**

1. **File: `scripts/comprehensive_benchmark.py`**
   - Update default n_trials from 20 to 50
   - Add HIGH_VARIANCE_SYSTEMS constant for systems needing n=100
   - Modify `benchmark_noise_sweep()` to use per-system n_trials

```python
# Add near line 104
HIGH_VARIANCE_SYSTEMS = {"Lorenz", "Rossler", "RabinovichFabrikant"}

# Update benchmark_noise_sweep signature
def benchmark_noise_sweep(..., n_trials: int = 50,
                         high_variance_n_trials: int = 100):
    ...
    actual_trials = high_variance_n_trials if system_name in HIGH_VARIANCE_SYSTEMS else n_trials
```

**Complexity:** Low
**Dependencies:** None
**Verification:** Error bars should shrink by ~sqrt(2.5)

---

### Issue 3: Report Statistical Significance

**Problem:** `compute_statistical_significance()` exists but results are never reported in the document.

**Changes Required:**

1. **File: `scripts/comprehensive_benchmark.py`**
   - Add Bonferroni correction (50 comparisons → α=0.001)
   - Add confidence intervals for improvement ratios
   - Ensure significance results are included in JSON output

```python
def compute_statistical_significance(..., n_comparisons: int = 1):
    # ... existing code ...

    alpha_corrected = alpha / n_comparisons

    # Add 95% CI for improvement ratio
    ratios = np.array(method_results) / (np.array(baseline_results) + 1e-10)
    ci_low, ci_high = np.percentile(ratios, [2.5, 97.5])

    return {
        # ... existing fields ...
        "alpha_corrected": alpha_corrected,
        "significant_bonferroni": p_wilcoxon < alpha_corrected,
        "improvement_ci_low": float(ci_low),
        "improvement_ci_high": float(ci_high),
    }
```

2. **File: `docs/factorized_evaluation_report.md`**
   - Add Section 10.9: Statistical Significance Analysis
   - Include p-values, effect sizes, Bonferroni-corrected significance

**Complexity:** Medium
**Dependencies:** Issue 1 (need sufficient samples)

---

## Phase 2: Enhanced Metrics

### Issue 4: Clarify Coefficient Recovery Metrics

**Problem:** Only mean MAE reported; unclear if active-only; no median or success rate.

**Changes Required:**

1. **File: `src/sc_sindy/metrics/coefficient.py`**

```python
def compute_comprehensive_coefficient_metrics(
    xi_pred: np.ndarray,
    xi_true: np.ndarray,
    tol: float = 1e-6
) -> Dict[str, float]:
    """Extended metrics including median and success rate."""
    active_mask = np.abs(xi_true) > tol
    errors = np.abs(xi_pred[active_mask] - xi_true[active_mask])
    true_vals = np.abs(xi_true[active_mask])

    return {
        "mae_active": float(np.mean(errors)),
        "mae_all": float(np.mean(np.abs(xi_pred - xi_true))),
        "median_ae": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "success_rate": float(np.mean(errors < 2 * true_vals)),  # within 2x
        "n_active": int(np.sum(active_mask)),
    }

def compute_lorenz_per_coefficient_error(
    xi_pred: np.ndarray,
    xi_true: np.ndarray,
) -> Dict[str, float]:
    """Per-parameter error for Lorenz (sigma, rho, beta)."""
    # Lorenz: dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y - beta*z
    # Library order 3: [1, x, y, z, x², xy, xz, y², yz, z², ...]

    return {
        "sigma_error": abs(xi_pred[0, 2] - xi_true[0, 2]),  # y term in eq0
        "rho_error": abs(xi_pred[1, 1] - xi_true[1, 1]),    # x term in eq1
        "beta_error": abs(xi_pred[2, 3] - xi_true[2, 3]),   # z term in eq2
    }
```

2. **Update report Section 10.3** with:
   - Table distinguishing active-term vs all-term MAE
   - Median in addition to mean
   - Success rate column
   - Per-coefficient Lorenz breakdown

**Complexity:** Medium
**Dependencies:** Issue 1

---

## Phase 3: Hyperparameter-Tuned Baseline

### Issue 2: Add Hyperparameter-Tuned SINDy Baseline

**Problem:** Fixed threshold=0.1 for SINDy; cannot assess if network value comes from adaptive thresholding.

**Changes Required:**

1. **Create: `src/sc_sindy/core/sindy_tuned.py`**

```python
"""SINDy with cross-validated threshold selection."""

from sklearn.model_selection import KFold
from .sindy import sindy_stls

def cross_validate_threshold(
    Theta: np.ndarray,
    x_dot: np.ndarray,
    thresholds: List[float] = None,
    n_folds: int = 5,
) -> float:
    """Find optimal STLS threshold via cross-validation."""
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]

    best_threshold, best_score = 0.1, -np.inf
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for threshold in thresholds:
        fold_scores = []
        for train_idx, val_idx in kf.split(Theta):
            xi, _ = sindy_stls(Theta[train_idx], x_dot[train_idx], threshold=threshold)
            recon_error = np.mean((Theta[val_idx] @ xi.T - x_dot[val_idx])**2)
            fold_scores.append(-recon_error)

        if np.mean(fold_scores) > best_score:
            best_score = np.mean(fold_scores)
            best_threshold = threshold

    return best_threshold
```

2. **Create: `scripts/tune_thresholds.py`**
   - Run CV on TRAINING systems only
   - Generate per-system-class threshold lookup table
   - Save to JSON for use in benchmarks

3. **Update: `scripts/comprehensive_benchmark.py`**
   - Add `run_sindy_tuned_trial()` function
   - Include "sindy_tuned" in benchmark comparisons

**Complexity:** Medium-High
**Dependencies:** None

---

## Phase 4: Robustness & Decomposition

### Issue 5: Add Robustness Tests (Data Leakage Defense)

**Problem:** Identical trajectory generation for train/test; network may learn generation artifacts.

**Changes Required:**

1. **File: `scripts/comprehensive_benchmark.py`**

```python
def benchmark_robustness(systems, model_path, n_trials=50):
    """Test under varied experimental conditions."""
    results = {}

    # 1. Trajectory length variation
    for t_max in [10, 25, 50, 100]:
        results[f"t_max_{t_max}"] = run_benchmark_with_params(t_span=(0, t_max))

    # 2. Noise model variation
    for noise_type in ["additive", "multiplicative", "outliers"]:
        results[f"noise_{noise_type}"] = run_benchmark_with_noise_model(noise_type)

    # 3. Sampling rate variation
    for dt in [0.005, 0.01, 0.02, 0.05]:
        results[f"dt_{dt}"] = run_benchmark_with_dt(dt)

    return results

def add_noise(X, noise_level, noise_type="additive"):
    """Apply different noise models."""
    if noise_type == "additive":
        return X + np.random.randn(*X.shape) * noise_level * np.std(X)
    elif noise_type == "multiplicative":
        return X * (1 + np.random.randn(*X.shape) * noise_level)
    elif noise_type == "outliers":
        X_noisy = X + np.random.randn(*X.shape) * noise_level * np.std(X)
        mask = np.random.rand(*X.shape) < 0.05
        X_noisy[mask] += np.random.randn(np.sum(mask)) * 10 * noise_level * np.std(X)
        return X_noisy
```

**Complexity:** Medium
**Dependencies:** None

---

### Issue 6: Pipeline Decomposition

**Problem:** Cannot tell where SC-SINDy's value comes from. Resolves Lorenz F1 discrepancy (0.738 vs 1.0).

**Changes Required:**

1. **File: `scripts/comprehensive_benchmark.py`**

```python
def benchmark_pipeline_decomposition(systems, model_path, n_trials=50):
    """Compare: network-only vs network+STLS vs STLS-only vs oracle."""
    results = {}

    for system_name in systems:
        system_results = {
            "network_only": [],      # Raw network structure (P > 0.5)
            "network_stls": [],      # SC-SINDy (network + STLS)
            "stls_only": [],         # Standard SINDy
            "oracle_stls": [],       # STLS with perfect structure
        }

        for trial in range(n_trials):
            # 1. Network only
            probs = model.predict_structure(X, n_vars, poly_order)
            network_structure = (probs > 0.5).astype(float)
            f1_network = compute_f1(network_structure, true_structure)

            # 2. Network + STLS (SC-SINDy)
            xi_sc, _ = sindy_structure_constrained(Theta, X_dot, probs)
            f1_sc = compute_structure_f1(xi_sc, xi_true)

            # 3. STLS only
            xi_stls, _ = sindy_stls(Theta, X_dot)
            f1_stls = compute_structure_f1(xi_stls, xi_true)

            # 4. Oracle (perfect structure)
            xi_oracle, _ = sindy_structure_constrained(Theta, X_dot, true_structure)
            f1_oracle = compute_structure_f1(xi_oracle, xi_true)

            # Store results...

        results[system_name] = system_results

    return results
```

**Expected Output (explains Lorenz discrepancy):**
| Method | Lorenz F1 |
|--------|-----------|
| Network-only | 0.738 |
| Network+STLS | 1.000 |
| STLS-only | 0.86 |
| Oracle | 1.000 |

**Complexity:** Medium
**Dependencies:** None

---

### Issue 7: Learning Curves

**Problem:** No data showing F1 vs training set size.

**Create: `scripts/plot_learning_curves.py`**

```python
def compute_learning_curve(
    training_sizes=[100, 500, 1000, 2500, 5000],
    n_seeds=5,
):
    results = {"sizes": training_sizes, "f1_mean": [], "f1_std": []}

    for size in training_sizes:
        seed_f1s = []
        for seed in range(n_seeds):
            # Subsample training data to `size` samples
            # Train model
            # Evaluate on fixed test set
            seed_f1s.append(test_f1)

        results["f1_mean"].append(np.mean(seed_f1s))
        results["f1_std"].append(np.std(seed_f1s))

    return results
```

**Complexity:** Medium
**Dependencies:** None

---

### Issue 8: Computational Cost Table

**Problem:** No runtime comparison between methods.

**Changes Required:**

1. **File: `scripts/comprehensive_benchmark.py`**

```python
def benchmark_computational_cost(systems, model_path, n_trials=20):
    """Measure runtime and memory."""
    import tracemalloc

    results = {}
    for system_name in systems:
        # Time SINDy
        sindy_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            xi, _ = sindy_stls(Theta, X_dot)
            sindy_times.append(time.perf_counter() - t0)

        # Time SC-SINDy
        scsindy_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            probs = model.predict_structure(X, n_vars, poly_order)
            xi, _ = sindy_structure_constrained(Theta, X_dot, probs)
            scsindy_times.append(time.perf_counter() - t0)

        results[system_name] = {
            "sindy_ms": np.mean(sindy_times) * 1000,
            "scsindy_ms": np.mean(scsindy_times) * 1000,
            "overhead": np.mean(scsindy_times) / np.mean(sindy_times),
        }

    return results
```

**Complexity:** Low
**Dependencies:** None

---

## Implementation Sequence

```
Phase 1 (Week 1): Foundation
├── Issue 1: Increase sample size [2h]
└── Issue 3: Statistical significance [4h]

Phase 2 (Week 2): Metrics & Baseline
├── Issue 4: Coefficient metrics [4h]
└── Issue 2: Tuned baseline [6h]

Phase 3 (Week 3): Robustness
├── Issue 5: Robustness tests [4h]
├── Issue 6: Pipeline decomposition [3h]
├── Issue 7: Learning curves [3h]
└── Issue 8: Computational cost [2h]

Phase 4 (Week 4): Integration
├── Re-run all benchmarks
├── Update report with new tables
└── Generate figures
```

---

## Dependency Graph

```
Issue 1 (Sample Size) ←── FOUNDATION
    ↓
Issue 3 (Significance)
    ↓
Issue 4 (Metrics)

Issue 2 (Tuned Baseline) ←── INDEPENDENT

Issue 5 (Robustness) ←── INDEPENDENT

Issue 6 (Decomposition) ←── EXPLAINS DISCREPANCY

Issue 7 (Learning Curves) ←── INDEPENDENT

Issue 8 (Cost) ←── INDEPENDENT
    ↓
    ↓
Report Update ←── ALL ISSUES FEED INTO THIS
```

---

## Files to Modify/Create

| File | Issues | Action |
|------|--------|--------|
| `scripts/comprehensive_benchmark.py` | 1,3,5,6,8 | Modify |
| `src/sc_sindy/metrics/coefficient.py` | 4 | Modify |
| `src/sc_sindy/core/sindy_tuned.py` | 2 | Create |
| `scripts/tune_thresholds.py` | 2 | Create |
| `scripts/plot_learning_curves.py` | 7 | Create |
| `docs/factorized_evaluation_report.md` | All | Update |

---

## Success Criteria

After implementation, the evaluation should:

1. **Sample Size:** n≥50 for all systems, n=100 for high-variance
2. **Significance:** All comparisons have p-values with Bonferroni correction
3. **Metrics:** Median, success rate, per-coefficient breakdown reported
4. **Baselines:** SINDy-fixed, SINDy-tuned, SC-SINDy compared
5. **Robustness:** Tested under varied trajectory lengths, noise models, sampling rates
6. **Decomposition:** Network-only vs Network+STLS vs STLS-only quantified
7. **Learning Curves:** F1 vs training size plotted
8. **Cost:** Runtime comparison table included

**Target Reviewer Assessment:** Accept (after verification of significance tests)
